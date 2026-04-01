# 2026-04-01 · sglang-parity Steps 1-4: C=4 throughput exceeds sglang

## Context

Benchmarking infer against sglang 0.5.9 on Qwen3-8B / A100-40GB revealed:
- C=1 gap of 8-15% (kernel-level, not scheduling)
- C>1 completely broken (503 errors)
- 2048-token prompts failed

## What Worked

Steps 2-3 (config changes) + Step 4 (scheduler improvements) together fixed all C>1 and long-context failures:

| Change | Impact |
|--------|--------|
| PREFILL_CHUNK_SIZE 512→4096 | Fewer kernel launches for long prefill |
| PREFILL_CHUNK_SIZE_WITH_DECODE 64→512 | Better prefill throughput during active decode |
| max_waiting 16→256 | No more 503 at C>1 |
| DEFAULT_MAX_SEQ 1024→4096 | 2048-token prompts work |
| Process ALL new requests per step | Eliminates multi-step admission delay |
| Process ALL prefill chunks per step | Better prefill throughput |
| Prefix-aware slot assignment | Better KV cache reuse across turns |
| CUDA Graph warmup 1..min(slots,32) | Ready for larger batch sizes |

**Results:**

| Config | Before | After | sglang |
|--------|--------|-------|--------|
| 128in/512out C=1 | 70.2 | 70.2 | 76.3 |
| 512in/256out C=1 | 68.1 | 68.3 | 75.8 |
| 1024in/256out C=1 | 63.5 | 63.9 | 75.5 |
| 2048in/256out C=1 | FAIL | 56.8 | 74.8 |
| 512in/256out C=4 | FAIL | **260.7** | 255.9 |
| 128in/128out C=4 | FAIL | **279.7** | — |
| ITL p50 (128in) | 13.7ms | 13.5ms | 13.0ms |

## Rule

- Scheduling-level changes have outsized impact on concurrent throughput
- C=1 throughput is kernel-bound; improving scheduling doesn't help
- Prefix-aware slot assignment is cheap and effective for agent workloads
- Increasing chunk sizes is a trivial change with measurable impact on long contexts
