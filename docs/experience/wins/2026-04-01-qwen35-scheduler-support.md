# 2026-04-01 · Qwen3.5 Scheduler Support + Batched Decode

## Context

Added full scheduler/HTTP server support for Qwen3.5 (hybrid linear + full attention). Implemented batched decode with FlashInfer HD256 for full attention layers and per-request recurrent ops for linear attention layers.

## What Worked

| Component | Implementation |
|-----------|---------------|
| select_tokens_batch | Launch B sampling kernels, 1 sync, B readbacks — avoids B syncs |
| DecodeBufferPtrs35 | Cached raw pointers for hot-path sampling (same as Qwen3) |
| stop_token_ids | Loaded from generation_config.json with fallback |
| FlashInfer HD256 | New flashinfer_decode_hd256.cu (HEAD_DIM=256) for full attention |
| decode_prep_paged_hd256 | QK-norm (1+w offset) + partial RoPE (64/256) + gate + paged KV write |
| attention_gate_paged_hd256 | Sigmoid gate kernel applied after FlashInfer attention |
| BatchDecodeBuffers35 | Pre-allocated buffers for all 32 layers (full + linear) |
| Hybrid layer dispatch | Batched GEMM for all layers, FlashInfer for 8 full-attn, per-request conv1d/GDR for 24 linear |

## Benchmark Results (Qwen3.5-4B, A100-40GB)

### In-process (bench_serving, single request)

| Config | Throughput | ITL | TTFT |
|--------|-----------|-----|------|
| 128in/128out | 76 tok/s | 13.2ms | 20.9ms |
| 128in/256out | 70 tok/s | 14.3ms | 20.9ms |
| 512in/128out | 49 tok/s | 20.1ms | 51.1ms |
| 512in/256out | 47 tok/s | 21.3ms | 51.2ms |

### HTTP Server (bench_throughput.py)

| Concurrency | infer tok/s | sglang tok/s | ITL ours | ITL sglang | Gap |
|------------|-------------|-------------|----------|-----------|-----|
| C=1 | 100 | **107** | 9.9ms | **8.6ms** | -7% |
| C=4 | 290 | **349** | 13.2ms | **9.4ms** | -17% |
| C=8 | 297 (4 slots) | **680** | 13.2ms | **9.9ms** | -56% |

### TTFT (we win)

| Concurrency | infer TTFT | sglang TTFT |
|------------|-----------|-------------|
| C=1 | **17ms** | 107ms |
| C=4 | **45ms** | 270ms |

## Analysis

### ITL gap root cause: per-request recurrent overhead

C=4 ITL: 13.2ms (ours) vs 9.4ms (sglang) — **40% gap**

The 24 linear attention layers each require per-request:
1. D2D extract: qkv row from batch buffer (16KB)
2. conv1d kernel launch (per-request conv_state)
3. GDR decode kernel launch (per-request recurrent state)
4. D2D insert: result back to batch buffer (8KB)

Per decode step: 24 layers × B requests × 4 ops = 24×4×4 = 384 extra kernel launches + D2D copies.
At ~1.5μs per launch: **~576μs overhead per step** — explains most of the gap.

sglang likely has batched recurrent kernels that process all B requests' recurrent state in one kernel launch per layer.

### Throughput scaling gap: slots

sglang C=8 achieves 680 tok/s (ITL barely increases 8.6→9.9ms).
We saturate at 4 slots (290→297 tok/s from C=4 to C=8).
Increasing slots to 8+ would improve throughput linearly until memory-bound.

## Rule

For hybrid recurrent+attention models, the recurrent per-request overhead dominates at B>1. Batched recurrent kernels (processing all B states in one launch) are critical for competitive batched decode. D2D extract/insert + per-request kernel launches don't scale.
