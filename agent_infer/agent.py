"""Agent loop: generate -> parse tool calls -> execute -> repeat.

Supports two modes:
- Direct mode: Uses pegainfer /v1/completions with ChatML prompt formatting
- Dynamo mode: Uses dynamo frontend /v1/chat/completions with structured messages
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from agent_infer.chat import Message, ToolCall, format_prompt, parse_tool_calls
from agent_infer.client import ClientConfig, LLMClient
from agent_infer.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_turns: int = 10
    max_tokens: int = 2048
    temperature: float = 0.0
    system_prompt: str = "You are a helpful assistant."
    client: ClientConfig = field(default_factory=ClientConfig)


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    status: str = "idle"  # idle, running, complete, error, max_turns

    @property
    def last_response(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return ""


class AgentLoop:
    """Core agent loop tying LLM client, tools, and chat formatting together."""

    def __init__(
        self,
        config: AgentConfig,
        registry: ToolRegistry,
        client: LLMClient | None = None,
    ):
        self.config = config
        self.registry = registry
        self.client = client or LLMClient(config.client)

    async def run(self, user_message: str) -> AgentState:
        """Run the agent loop for a single user message."""
        state = AgentState(status="running")
        state.messages.append(Message(role="system", content=self.config.system_prompt))
        state.messages.append(Message(role="user", content=user_message))

        while state.status == "running":
            if state.turn_count >= self.config.max_turns:
                state.status = "max_turns"
                logger.warning("Agent reached max turns (%d)", self.config.max_turns)
                break

            # Generate response
            try:
                response_text = await self._generate(state)
            except Exception:
                logger.exception("Generation failed")
                state.status = "error"
                break

            # Parse for tool calls
            content, tool_calls = parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls — we're done
                state.messages.append(Message(role="assistant", content=response_text))
                state.status = "complete"
                break

            # Record assistant message with tool calls
            state.messages.append(
                Message(role="assistant", content=content, tool_calls=tool_calls)
            )
            state.turn_count += 1

            # Execute tools
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                logger.info("Executing tool: %s(%s)", tc.function.name, args)
                result = await self.registry.execute(tc.function.name, args)
                state.messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )

        return state

    async def run_interactive(self) -> AsyncIterator[tuple[str, AgentState]]:
        """Interactive loop: yield (response, state) for each turn."""
        state = AgentState(status="running")
        state.messages.append(Message(role="system", content=self.config.system_prompt))

        while True:
            user_input = yield ("", state)
            if user_input is None:
                break

            state.messages.append(Message(role="user", content=user_input))
            state.turn_count = 0
            state.status = "running"

            while state.status == "running":
                if state.turn_count >= self.config.max_turns:
                    state.status = "max_turns"
                    break

                response_text = await self._generate(state)
                content, tool_calls = parse_tool_calls(response_text)

                if not tool_calls:
                    state.messages.append(Message(role="assistant", content=response_text))
                    state.status = "complete"
                    yield (response_text, state)
                    break

                state.messages.append(
                    Message(role="assistant", content=content, tool_calls=tool_calls)
                )
                state.turn_count += 1

                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = await self.registry.execute(tc.function.name, args)
                    state.messages.append(
                        Message(role="tool", content=result, tool_call_id=tc.id, name=tc.function.name)
                    )

    async def _generate(self, state: AgentState) -> str:
        """Generate LLM response based on current state."""
        tools = self.registry.list_schemas() if self.registry.has_tools() else None

        if self.config.client.mode == "chat":
            # Dynamo mode: structured messages
            messages_dicts = [m.to_dict() for m in state.messages]
            full_response = ""
            async for chunk in self.client.chat(
                messages_dicts,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                tools=tools,
            ):
                full_response += chunk
            return full_response
        else:
            # Direct pegainfer mode: raw ChatML prompt
            prompt = format_prompt(state.messages, tools=tools)
            full_response = ""
            async for chunk in self.client.complete(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["<|im_end|>"],
            ):
                full_response += chunk
            return full_response
