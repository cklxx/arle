# 2026-04-02 · 128 Slots — High Concurrency Fix

## Context

After fixing auto-slots memory sizing and prefix cache crash, agent-infer was limited to MAX_SLOTS=32, causing -25.6% throughput gap at C=32 vs SGLang. All C≥16 tests showed errors due to a waiting counter bug.

## Changes

1. **MAX_SLOTS**: 32 → 128 (auto-computed from GPU memory)
2. **kv_pool_headroom**: 2 GB → 4 GB (leave room for BatchDecodeBuffers + CUDA Graphs)
3. **Prefill rate limiting**: unlimited → 1 per step when decode active, 8 when idle
4. **Shared waiting counter (BUG FIX)**: `SchedulerHandle::with_max_waiting` created its own `Arc<AtomicUsize>`, separate from the scheduler's counter. Handle's counter only incremented (never decremented), so after `max_waiting_requests` (256) total requests, ALL new requests were rejected. Fixed by sharing the same Arc.
5. **CUDA Graph batch sizes**: added 16, 20, 24, 28, 32 (step by 4 from 12-32)

## Raw Data — agent-infer (128 auto-slots, prefix cache disabled)

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99
  128 |   256 |  1 |    113.4 t/s |      21ms |      22ms |    8.8ms |    8.8ms
  512 |   256 |  1 |    108.1 t/s |      50ms |      51ms |    9.1ms |    9.1ms
  128 |   256 |  4 |    399.4 t/s |      79ms |     107ms |    9.3ms |    9.3ms
  128 |   256 |  8 |    742.1 t/s |     139ms |     223ms |   10.0ms |   10.0ms
  512 |   256 |  8 |    685.7 t/s |     286ms |     455ms |    9.9ms |   10.0ms
  128 |   256 | 16 |   1198.7 t/s |     257ms |     462ms |   11.7ms |   11.8ms
  512 |   256 | 16 |   1047.6 t/s |     526ms |     924ms |   11.9ms |   12.0ms
  128 |   256 | 32 |   1818.2 t/s |     493ms |     939ms |   14.2ms |   14.5ms
  512 |   256 | 32 |   1499.7 t/s |    1011ms |    1885ms |   14.4ms |   14.6ms
  128 |   256 | 64 |   2396.8 t/s |    1016ms |    2044ms |   19.9ms |   20.4ms
```

## Comparison vs SGLang 0.5.9

| Config | Before (32 slots) | After (128 slots) | SGLang 0.5.9 | Gap |
|--------|-------------------|-------------------|--------------|-----|
| C=1 128/256 | 113.1 tok/s | 113.4 tok/s | 109.5 tok/s | **+4%** |
| C=8 128/256 | 758.5 tok/s | 742.1 tok/s | 707.4 tok/s | **+5%** |
| C=16 128/256 | 1230.2 tok/s | 1198.7 tok/s | 1260.8 tok/s | -5% |
| C=32 128/256 | 1627.9 (16 err) | **1818.2 (0 err)** | 2189.1 tok/s | **-17%** (was -25.6%) |
| C=64 128/256 | N/A (all errors) | **2396.8 (0 err)** | N/A | new! |

## Remaining Gap at C=32+

**Root cause**: Qwen3.5 batched decode runs in eager mode (no CUDA Graph). The recurrent state pointer arrays (conv1d + GDR) are uploaded from host each step because batch composition changes. This prevents CUDA Graph capture for the recurrent layers.

- ITL at C=32: 14.2ms (ours) vs 12.9ms (SGLang) = +10%
- SGLang uses contiguous mamba cache indexed by request, enabling graph capture

## Environment

```
GPU:          NVIDIA A100-SXM4-80GB
CUDA:         13.0
Model:        Qwen3.5-4B bf16
num_slots:    128 (auto)
prefix_cache: disabled (Qwen3.5)
```
