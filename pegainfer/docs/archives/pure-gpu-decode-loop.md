# Pure GPU Decode Loop

> **TL;DR:** Explored eliminating per-token CPU-GPU round-trips in the decode loop. Phase 1 (argmax-in-graph) shipped in #34. Phase 2 (GPU-side token passing) and Phase 3 (batch launch) prototyped and measured. Conclusion: CPU overhead is only ~0.6% of TPOT — not worth further investment. TPOT is dominated by GPU kernel compute.
>
> **Status:** Concluded. Further TPOT optimization should target kernel compute (GEMV, MLP), not CPU overhead.

## Motivation

Each decode token requires CPU involvement: graph launch, D2H argmax read, H2D decode_meta write, EOS check. At TPOT ~10-12ms, how much does this actually cost?

## Profiling Baseline (2026-03-26, post argmax-in-graph)

### Qwen3-4B (prompt=1, output=128, no graph-trace)

| Metric | Value |
|---|---|
| steady TPOT | **10.49ms** |
| cuGraphLaunch avg | 52.2μs |
| cuMemcpyDtoHAsync avg | 9.6μs |
| cuStreamSynchronize avg | 10.49ms (GPU-bound) |
| cudaLaunchKernel (decode) | 0 (all in graph) |
| **overhead ≈** | **~60μs/token (0.6%)** |

### Qwen3.5-4B (prompt=1, output=128, no graph-trace)

| Metric | Value |
|---|---|
| steady TPOT | **12.53ms** |
| cuGraphLaunch avg | 52.8μs |
| cuMemcpyDtoHAsync avg | 10.9μs |
| cuStreamSynchronize avg | 12.50ms (GPU-bound) |
| cudaLaunchKernel (decode) | 0 (all in graph) |
| **overhead ≈** | **~80μs/token (0.6%)** |

### Per-token CPU-GPU overhead breakdown

```
cuGraphLaunch:          ~52μs   ← graph dispatch to GPU
cuMemcpyHtoDAsync:       ~5μs   ← H2D write decode_meta [token_id, pos, seq_len]
cuMemcpyDtoHAsync:      ~10μs   ← D2H read argmax result
CPU stop check + logic: ~10μs
────────────────────────────────
total overhead:        ~77μs/token
```

## Phases & Results

### Phase 1: Argmax in CUDA Graph (shipped, #34)

Pre-allocate `argmax_out` in `DecodeBuffers`, call `argmax_into()` at end of `decode_kernels()`. Greedy path reads via `read_argmax()` (sync + D2H) instead of alloc → launch → sync → free cycle.

Eliminated per token: 1× `cudaLaunchKernel`, 1× `cuMemAllocAsync`, 1× `cuMemFreeAsync`, 1× `cuMemsetD8Async`.

### Phase 2: GPU-side token passing (prototyped, reverted)

Added `prepare_next_decode` kernel: copies argmax → `decode_meta[0]`, increments pos/seq_len on GPU. Skips H2D write on graph replay. Both e2e tests passed bit-exact.

**nsys confirmed:** `cuMemcpyHtoDAsync` calls dropped by 254 (decode H2D eliminated).

**TPOT impact:** unmeasurable (~0.01ms). H2D of 12 bytes is ~5μs — below noise floor.

### Phase 3: Batch launch / zero-CPU decode (prototyped, reverted)

Launched 127 graph replays back-to-back without per-token sync. Single sync at end. GPU-side `prepare_next_decode` (Phase 2) kept decode_meta current between replays.

**Measured (Qwen3.5-4B, 128 tokens):**

| Approach | e2e | TPOT p50 |
|---|---|---|
| Per-token sync (Phase 1) | 1603ms | 12.53ms |
| Batch launch (Phase 3) | 1602ms | 12.52ms |
| **Savings** | **~1ms total** | **~0.01ms/token** |

128 tokens, zero-CPU loop saves 1ms. The GPU idle gap between tokens is negligible because `cuStreamSynchronize` returns as soon as the GPU finishes, and the next `cuGraphLaunch` dispatches in ~52μs — which the GPU hides while still executing.

## Conclusion

**99.4% of TPOT is GPU kernel compute. CPU overhead is ~0.6% (~77μs/token).** Eliminating all CPU involvement between tokens (Phase 3) saves ~1ms over 128 tokens — not worth the complexity.

### Approaches considered and rejected

| Approach | Complexity | Max savings | Why not |
|---|---|---|---|
| Batch graph launch | Low | ~1ms/128tok | Measured: negligible, breaks streaming + EOS |
| CUDA Graph conditional nodes | High (CUDA 12.4+) | ~6.7ms/128tok theoretical | cudarc doesn't expose API, streaming incompatible |
| Persistent kernel (CDP) | High | Same as conditional | CDP launch overhead > host launch, loses CUDA Graph |
| Mega-kernel fusion | Impossible | N/A | Decode uses ~200 kernels with heterogeneous configs |

### Where to invest instead

TPOT is memory-bandwidth bound at batch=1. The path to lower TPOT is:
1. Roofline analysis: measure bandwidth utilization of GEMV and MLP kernels
2. Close the gap to theoretical memory bandwidth limit
3. Weight quantization (INT8/INT4) to reduce bytes read per token
