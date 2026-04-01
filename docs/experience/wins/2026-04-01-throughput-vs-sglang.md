# 2026-04-01 · Throughput Benchmark: infer vs sglang

## Context

Head-to-head throughput comparison between infer and sglang 0.5.9, both serving Qwen3-8B (bf16) on A100-40GB. Same model, same GPU, same benchmark script.

## Setup

| | infer | sglang |
|---|---|---|
| Version | HEAD (2026-04-01) | 0.5.9 |
| Model | Qwen3-8B bf16 | Qwen3-8B bf16 |
| GPU | A100-SXM4-40GB | A100-SXM4-40GB |
| Slots/Workers | 4 | auto |
| Max seq len | 4096 | 8192 |
| Attention | FlashInfer + CUDA Graph | FlashInfer |
| Paged KV | Token-level pool | RadixAttention |

## Results

### C=1 (Single Request, Sequential)

| Input | Output | infer tok/s | sglang tok/s | Gap |
|-------|--------|-------------|--------------|-----|
| 128 | 128 | **72.3** | 32.5* | +122% |
| 128 | 512 | 70.1 | **76.3** | -8% |
| 512 | 256 | 68.0 | **75.8** | -10% |
| 1024 | 256 | 63.8 | **75.5** | -15% |
| 2048 | 256 | FAIL (503) | **74.8** | — |

\* sglang 128/128 anomaly: p99 TTFT=9007ms (warmup outlier), throughput was depressed by one slow request.

### C=4 (Concurrent)

| Input | Output | infer tok/s | sglang tok/s | Gap |
|-------|--------|-------------|--------------|-----|
| 512 | 256 | FAIL (503) | **255.9** | — |
| 128 | 128 | FAIL (503) | — | — |

### Latency Comparison (C=1)

| Metric | infer | sglang |
|--------|-------|--------|
| **ITL p50** | 13.8-15.5ms | 13.0-13.1ms |
| **TTFT (128 in)** | **19ms** | 49ms |
| **TTFT (1024 in)** | **21-203ms** | 53-76ms |

## Analysis

### Where infer wins
1. **Short-context TTFT**: 19ms vs 49ms — infer's prefill is 2.5x faster for short prompts, likely because sglang has Python overhead in the request dispatch path
2. **CUDA Graph decode**: ITL is comparable (13.8 vs 13.0ms), validating that our FlashInfer + CUDA Graph decode path is competitive

### Where sglang wins
1. **Long-context decode throughput**: 75 vs 64 tok/s at 1024 input — sglang's attention/KV management is more efficient at scale
2. **Concurrency**: sglang handles C=4 effortlessly (256 tok/s). infer returns 503 because:
   - Bounded waiting queue (capacity ~4) fills instantly
   - No queuing with backoff in the scheduler
3. **2048+ input**: infer fails entirely (scheduler rejects), sglang handles fine

### Root Causes of Gap

1. **Scheduler queue capacity** (Critical): The `SchedulerHandle` channel has a bounded buffer. When all slots are busy, new requests get 503 instead of queuing. sglang has an unbounded queue with proper backpressure.

2. **Prefill throughput at long context** (~15% gap): At 1024 input tokens, infer's chunked prefill (512-token chunks) adds overhead. sglang uses FlashAttention-2/3 for full prefill in one pass.

3. **ITL parity**: decode ITL is within 6% (13.8 vs 13.0ms), meaning our CUDA Graph + FlashInfer batched decode is solid. The gap is likely from Python-free overhead in our Rust scheduler (slightly more per-step CPU work for emit_delta, stop sequence checks).

## Optimization Priorities

1. **Fix scheduler queue** — make waiting queue unbounded or much larger (easy, high impact on C>1)
2. **Trace the prefill path** — use `--trace-output-path` to identify where the 15% long-context gap comes from
3. **Tune chunked prefill** — try larger chunks (1024, 2048) or adaptive chunk sizes
4. **Profile emit_delta overhead** — the deferred emit is doing tokenizer.decode() on the CPU between GPU steps

## Rule

For single-request workloads (agent use case), infer is competitive with sglang (~10% gap). For concurrent serving, the scheduler queue must be fixed before meaningful comparison. The decode kernel path is validated — optimization effort should focus on prefill and scheduler, not decode.
