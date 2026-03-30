"""Chat message protocol, Qwen3 ChatML formatter, and tool call parser.

Handles the full lifecycle of chat messages:
- Message dataclasses for multi-turn conversations
- Qwen3 ChatML template formatting (with tool definitions)
- Parsing tool calls from model output (<tool_call> tags)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON string

@dataclass
class ToolCall:
    id: str
    function: FunctionCall

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }

@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Qwen3 ChatML formatter
# ---------------------------------------------------------------------------

def _format_tool_schemas(tools: list[dict]) -> str:
    """Format tool schemas for Qwen3's system prompt."""
    lines = []
    for tool in tools:
        func = tool.get("function", tool)
        tool_def = {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            },
        }
        lines.append(json.dumps(tool_def, ensure_ascii=False))
    return "\n".join(lines)


_TOOL_SYSTEM_TEMPLATE = (
    '\n\n# Tools\n\n'
    'You are provided with the following tools. To call a tool, respond with a JSON object wrapped in <tool_call> tags:\n\n'
    '<tool_call>\n'
    '{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}\n'
    '</tool_call>\n\n'
    'You may call one or more tools. You may also respond without calling any tool if appropriate.\n\n'
    'Here are the available tools:\n\n'
    '{tool_schemas}'
)


def format_prompt(
    messages: list[Message],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Format messages into Qwen3 ChatML prompt string.

    This produces a raw text prompt for pegainfer's /v1/completions endpoint.
    When tools are provided, they're injected into the system message.
    """
    parts: list[str] = []

    # Inject tools into system message
    system_injected = False
    for msg in messages:
        if msg.role == "system" and tools and not system_injected:
            tool_schemas = _format_tool_schemas(tools)
            content = msg.content + _TOOL_SYSTEM_TEMPLATE.format(tool_schemas=tool_schemas)
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            system_injected = True
        elif msg.role == "system":
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            text = msg.content
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_json = json.dumps(
                        {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)},
                        ensure_ascii=False,
                    )
                    text += f"\n<tool_call>\n{call_json}\n</tool_call>"
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
        elif msg.role == "tool":
            parts.append(
                f"<|im_start|>tool\n<tool_response>\n{msg.content}\n</tool_response><|im_end|>"
            )

    # If no system message was present but tools exist, prepend one
    if tools and not system_injected:
        tool_schemas = _format_tool_schemas(tools)
        system_content = "You are a helpful assistant." + _TOOL_SYSTEM_TEMPLATE.format(
            tool_schemas=tool_schemas
        )
        parts.insert(0, f"<|im_start|>system\n{system_content}<|im_end|>")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Parse Qwen3-style <tool_call> blocks from model output.

    Returns:
        (content_text, tool_calls): content_text is everything outside
        tool_call tags, tool_calls is a list of parsed ToolCall objects.
    """
    tool_calls: list[ToolCall] = []
    content = _TOOL_CALL_PATTERN.sub("", text).strip()

    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return content, tool_calls
