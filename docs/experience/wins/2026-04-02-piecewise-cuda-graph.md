# 2026-04-02 · Piecewise CUDA Graph for Qwen3.5

## Context

Qwen3.5 batched decode ran entirely in eager mode — no CUDA Graph for the 24 linear attention layers. Full decode CUDA Graph capture failed because FlashInfer plan metadata changes per step (KV cache grows each token).

## What Worked

**Piecewise CUDA Graph**: capture graphs for GROUPS of consecutive linear layers (3 per group × 8 groups), with full attention layers running eagerly between groups.

Key insight: linear layer groups don't use FlashInfer — they only use batched conv1d + GDR kernels with pre-uploaded pointer arrays. So the graph doesn't include any changing metadata.

Architecture: `[L,L,L,A,L,L,L,A,...]` → 8 groups of 3 linear layers, each captured as a separate graph per batch_size. Full attention layers (8 total) run eagerly.

**Pre-uploaded per-layer pointer arrays** (prerequisite): moved all 48 H2D uploads before the decode body. Each linear layer has its own GPU pointer array. The graph body has zero H2D operations.

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| C=1 ITL p50 (128/256) | 8.6ms | **8.0ms** | **-7%** |
| C=8 ITL p50 | 9.9ms | **9.4ms** | **-5%** |
| C=32 ITL p50 | 14.4ms | **13.6ms** | **-6%** |
| C=1 throughput | 114 tok/s | **123 tok/s** | **+8%** |
| C=8 throughput | 742 tok/s | **790 tok/s** | **+6%** |
| C=32 throughput | 1818 tok/s | **1896 tok/s** | **+4%** |
| C=64 throughput | 2381 tok/s | **2482 tok/s** | **+4%** |

vs SGLang 0.5.9: C=1 to C=8 now **+12-14% ahead**, C=16 parity, C=32 **-13%** (was -17%).

## Rule

For hybrid transformer models with interleaved attention+recurrent layers, piecewise CUDA Graph per contiguous group of recurrent layers avoids the FlashInfer metadata incompatibility while still capturing the majority of kernel launches.

## Environment

```
GPU:          NVIDIA A100-SXM4-80GB
CUDA:         13.0
Model:        Qwen3.5-4B bf16
num_slots:    128 (auto)
```
