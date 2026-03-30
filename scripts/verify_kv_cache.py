"""Verify KV prefix cache correctness.

Method: Send the same multi-turn conversation twice.
- Run 1: Cold start, each turn is a fresh session (no prefix reuse)
- Run 2: Warm session, each turn reuses prefix from prior turns

If prefix caching is correct, the outputs must be IDENTICAL.
Any divergence means the cached KV entries are stale or corrupted.
"""

import httpx
import json
import time
import sys

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"

TURNS = [
    "What is the capital of Japan?",
    "What language do they speak there?",
    "Name three famous landmarks in that city.",
]


def complete(prompt: str, max_tokens: int = 60) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy — deterministic
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def build_prompt(history: list[tuple[str, str]], next_user: str) -> str:
    """Build ChatML prompt from conversation history + next user turn."""
    prompt = SYSTEM
    for user, assistant in history:
        prompt += f"\n<|im_start|>user\n{user}<|im_end|>"
        prompt += f"\n<|im_start|>assistant\n{assistant}<|im_end|>"
    prompt += f"\n<|im_start|>user\n{next_user}<|im_end|>"
    prompt += "\n<|im_start|>assistant\n"
    return prompt


def run_conversation(label: str) -> list[str]:
    """Run full conversation, return list of assistant responses."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    history: list[tuple[str, str]] = []
    responses: list[str] = []

    for i, user_msg in enumerate(TURNS):
        prompt = build_prompt(history, user_msg)
        prompt_len = len(prompt)

        start = time.perf_counter()
        result = complete(prompt)
        elapsed = (time.perf_counter() - start) * 1000

        text = result["choices"][0]["text"]
        usage = result["usage"]

        print(
            f"  Turn {i+1}: "
            f"prompt={usage['prompt_tokens']:>4} tok, "
            f"completion={usage['completion_tokens']:>3} tok, "
            f"time={elapsed:>7.1f}ms"
        )
        print(f"    Output: {text[:120]}{'...' if len(text)>120 else ''}")

        responses.append(text)
        history.append((user_msg, text))

    return responses


def main():
    print("KV Prefix Cache Correctness Verification")
    print(f"Target: {BASE_URL}")
    print(f"Method: greedy decoding (temperature=0), compare outputs")

    # Run 1: Force cold by sending a garbage prompt to evict cache
    print("\n[Step 1] Evicting cache with unrelated prompt...")
    complete("This is a completely unrelated prompt to evict the KV cache " * 5, max_tokens=5)

    outputs_run1 = run_conversation("Run 1: Cold (no prefix cache)")

    # Run 2: Same conversation — should reuse prefix
    print("\n[Step 2] Evicting cache again...")
    complete("Another unrelated eviction prompt " * 5, max_tokens=5)

    outputs_run2 = run_conversation("Run 2: Cold again (baseline comparison)")

    # Run 3: Same conversation immediately after — this one should hit cache
    outputs_run3 = run_conversation("Run 3: Warm (prefix cache active)")

    # Compare
    print(f"\n{'='*60}")
    print("  VERIFICATION RESULTS")
    print(f"{'='*60}")

    all_pass = True

    # Run 1 vs Run 2: Both cold — should be identical (sanity check)
    print("\n[Sanity] Run 1 vs Run 2 (both cold):")
    for i in range(len(TURNS)):
        match = outputs_run1[i] == outputs_run2[i]
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  Turn {i+1}: {status}")
        if not match:
            print(f"    Run 1: {outputs_run1[i][:80]}")
            print(f"    Run 2: {outputs_run2[i][:80]}")

    # Run 2 vs Run 3: Cold vs warm — must be identical for cache to be correct
    print("\n[Cache Correctness] Run 2 (cold) vs Run 3 (warm/cached):")
    for i in range(len(TURNS)):
        match = outputs_run2[i] == outputs_run3[i]
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  Turn {i+1}: {status}")
        if not match:
            print(f"    Cold: {outputs_run2[i][:100]}")
            print(f"    Warm: {outputs_run3[i][:100]}")

    print(f"\n{'='*60}")
    if all_pass:
        print("  ALL TESTS PASSED — KV prefix cache is correct")
    else:
        print("  SOME TESTS FAILED — KV prefix cache has bugs")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
