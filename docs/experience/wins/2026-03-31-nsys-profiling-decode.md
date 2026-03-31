# 2026-03-31 - nsys Profiling: Decode Pipeline Analysis

## Context
At 8-concurrent decode (Qwen3-4B, A100-40GB), agent-infer achieved 811 tok/s
vs SGLang's 898 tok/s (0.90x). Used nsys to profile both systems side-by-side
to find the exact source of the remaining 10% gap.

## What Worked

### Profiling approach
Used `nsys profile --trace=cuda,nvtx` on both systems running identical
8-concurrent 256-token greedy workloads. Analyzed kernel-level breakdown
with `nsys stats`.

### Key findings

**CUDA Graph body is nearly matched**: 8.56ms (ours) vs 8.18ms (SGLang).
Only 0.38ms gap inside the graph — the GPU compute is efficient.

**The real gap is outside the graph**:
- Our `cuStreamSynchronize` blocks for full graph duration (8.6ms avg)
- SGLang's `cudaEventSynchronize` is more granular (5.1ms avg)
- This allows SGLang to overlap CPU prep with GPU compute

**Kernel-level wins available**:
- FusedAddRMSNorm: SGLang fuses residual add + RMSNorm into one kernel (1.3us).
  We do them separately (add: 2.7us + norm: 4.4us = 7.1us total). Fusing
  saves one global memory read-write pass.
- argmax: our batched kernel (22.6us) is 1.7x slower than SGLang's (13.2us).
  Likely warp reduction strategy difference.

**Our advantage is real**: 5x fewer kernel launches, 7.7x less memcpy time.
Rust control plane eliminates PyTorch overhead entirely.

## Rule
nsys profiling reveals that "overhead" is rarely in one place — it's distributed
across sync strategy (stream vs event), kernel fusion (FusedAddRMSNorm), and
host-device coordination. Optimizing one area shifts the bottleneck to the next.
Always profile both systems under identical conditions for fair comparison.
