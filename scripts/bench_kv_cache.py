"""Benchmark KV cache hit rate with multi-turn agent conversations.

Uses common agent prompt patterns to measure how well prefix caching works
when multiple requests share system prompts and tool definitions.

Metrics:
- Prompt tokens reused vs total (prefix cache hit rate)
- Time-to-first-token per turn
- Throughput
"""

import asyncio
import json
import time
import sys
from dataclasses import dataclass, field

import httpx

# Agent prompt dataset: realistic multi-turn tool-calling conversations
# Each conversation simulates a real agent session with shared system prompt

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. You can use the following tools:

1. python - Execute Python code for calculations, data processing, and analysis
2. shell - Execute shell commands for system operations
3. file - Read, write, and list files

When you need to use a tool, wrap your call in <tool_call> tags:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Always think step by step before answering. Be concise and accurate."""

# Multi-turn conversations that test KV cache reuse
CONVERSATIONS = [
    # Conv 1: Math calculations (3 turns)
    [
        "Calculate the factorial of 20",
        "Now calculate fibonacci(30)",
        "What's the sum of both results?",
    ],
    # Conv 2: File operations (3 turns)
    [
        "List the files in the current directory",
        "Read the contents of pyproject.toml",
        "What Python version does this project require?",
    ],
    # Conv 3: Code analysis (3 turns)
    [
        "Write a Python function to check if a number is prime",
        "Now optimize it using the sieve of Eratosthenes",
        "What's the time complexity of both approaches?",
    ],
    # Conv 4: System info (3 turns)
    [
        "Show me the current system memory usage",
        "What GPU is available?",
        "How much GPU memory is free?",
    ],
    # Conv 5: Data processing (3 turns)
    [
        "Generate a list of 10 random numbers between 1 and 100 using Python",
        "Sort them and find the median",
        "Calculate the standard deviation",
    ],
]


@dataclass
class TurnMetrics:
    conversation_id: int
    turn_id: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # time to first token
    total_time_ms: float


@dataclass
class BenchmarkResult:
    turns: list[TurnMetrics] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def summary(self) -> str:
        if not self.turns:
            return "No results"

        lines = []
        lines.append("=" * 70)
        lines.append("KV Cache Benchmark Results (Dynamo + Pegainfer)")
        lines.append("=" * 70)

        # Per-conversation stats
        conv_ids = sorted(set(t.conversation_id for t in self.turns))
        for cid in conv_ids:
            conv_turns = [t for t in self.turns if t.conversation_id == cid]
            lines.append(f"\nConversation {cid + 1}:")
            for t in conv_turns:
                lines.append(
                    f"  Turn {t.turn_id + 1}: "
                    f"prompt={t.prompt_tokens:4d} tok, "
                    f"completion={t.completion_tokens:3d} tok, "
                    f"TTFT={t.ttft_ms:7.1f}ms, "
                    f"total={t.total_time_ms:7.1f}ms"
                )

            # KV cache analysis: turn 2+ should reuse prefix from turn 1
            if len(conv_turns) > 1:
                t0_prompt = conv_turns[0].prompt_tokens
                shared_prefix = t0_prompt  # system prompt + tools
                later_prompts = sum(t.prompt_tokens for t in conv_turns[1:])
                reusable = shared_prefix * (len(conv_turns) - 1)
                lines.append(
                    f"  KV prefix reuse potential: {reusable} tokens "
                    f"({reusable / (later_prompts + 1) * 100:.1f}% of later prompts)"
                )

        # Overall stats
        lines.append(f"\n{'=' * 70}")
        lines.append("Overall Statistics:")
        total_prompt = sum(t.prompt_tokens for t in self.turns)
        total_completion = sum(t.completion_tokens for t in self.turns)
        avg_ttft = sum(t.ttft_ms for t in self.turns) / len(self.turns)
        avg_total = sum(t.total_time_ms for t in self.turns) / len(self.turns)

        # First turns vs later turns TTFT comparison
        first_turns = [t for t in self.turns if t.turn_id == 0]
        later_turns = [t for t in self.turns if t.turn_id > 0]
        avg_first_ttft = sum(t.ttft_ms for t in first_turns) / max(len(first_turns), 1)
        avg_later_ttft = sum(t.ttft_ms for t in later_turns) / max(len(later_turns), 1)

        lines.append(f"  Total turns:        {len(self.turns)}")
        lines.append(f"  Total prompt tokens: {total_prompt}")
        lines.append(f"  Total completion tokens: {total_completion}")
        lines.append(f"  Avg TTFT (all):     {avg_ttft:.1f}ms")
        lines.append(f"  Avg TTFT (turn 1):  {avg_first_ttft:.1f}ms")
        lines.append(f"  Avg TTFT (turn 2+): {avg_later_ttft:.1f}ms")
        lines.append(f"  Avg total time:     {avg_total:.1f}ms")

        if avg_first_ttft > 0 and avg_later_ttft > 0:
            speedup = avg_first_ttft / avg_later_ttft
            lines.append(f"  TTFT speedup (turn 2+ vs 1): {speedup:.2f}x")
            if speedup > 1.2:
                lines.append("  -> KV cache prefix reuse is effective!")
            else:
                lines.append(
                    "  -> KV cache not reusing prefix "
                    "(expected: pegainfer resets KV between requests)"
                )

        lines.append("=" * 70)
        return "\n".join(lines)


async def run_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    conv_id: int,
    turns: list[str],
    result: BenchmarkResult,
):
    """Run a multi-turn conversation and collect metrics."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn_id, user_msg in enumerate(turns):
        messages.append({"role": "user", "content": user_msg})

        request = {
            "model": "Qwen3-4B",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0,
            "stream": True,
        }

        ttft = None
        start = time.perf_counter()
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=request
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        full_text += content

                        usage = data.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"  [Conv {conv_id+1} Turn {turn_id+1}] Error: {e}")
            continue

        total_time = (time.perf_counter() - start) * 1000
        if ttft is None:
            ttft = total_time

        # If streaming didn't give usage, estimate from response
        if prompt_tokens == 0:
            prompt_tokens = len(json.dumps(messages)) // 4  # rough estimate
        if completion_tokens == 0:
            completion_tokens = len(full_text) // 4

        metrics = TurnMetrics(
            conversation_id=conv_id,
            turn_id=turn_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft,
            total_time_ms=total_time,
        )
        result.turns.append(metrics)
        result.total_prompt_tokens += prompt_tokens
        result.total_completion_tokens += completion_tokens

        print(
            f"  [Conv {conv_id+1} Turn {turn_id+1}] "
            f"TTFT={ttft:.1f}ms total={total_time:.1f}ms "
            f"prompt={prompt_tokens} completion={completion_tokens}"
        )

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": full_text})


async def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8090"
    max_convs = int(sys.argv[2]) if len(sys.argv) > 2 else len(CONVERSATIONS)

    print(f"Benchmarking KV cache with {max_convs} conversations")
    print(f"Target: {base_url}")
    print()

    result = BenchmarkResult()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        # Run conversations sequentially (pegainfer is single-request)
        for i, conv in enumerate(CONVERSATIONS[:max_convs]):
            print(f"Conversation {i + 1}/{max_convs}:")
            await run_conversation(client, base_url, i, conv, result)
            print()

    print(result.summary())


if __name__ == "__main__":
    asyncio.run(main())
