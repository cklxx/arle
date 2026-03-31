# 2026-03-31 · Batched Decode: 128 -> 690 tok/s (5.4x)

## Context

Goal: match SGLang's concurrent throughput on A100-40GB with Qwen3-4B. Starting point: 128 tok/s at 8 concurrent (SGLang: 886 tok/s, 6.9x gap). Each decode request ran as an independent forward pass — no batching.

## What Worked

### Phase 1: Token-Level KV Pool + FlashInfer Paged Decode (128 -> 434 tok/s)

**Architecture change**: replaced per-slot contiguous KV cache with a shared token-level KV pool (SGLang's `TokenToKVPool` pattern). Pool layout: `[max_tokens, kv_dim]` per layer, page_size=1.

**Prefill flow**: standard contiguous KV cache for Triton FA2, then `kv_cache_to_paged` migration kernel copies processed K/V to the pool after prefill completes.

**Decode flow**: batched GEMM for all linear projections (QKV, O, MLP) + FlashInfer `BatchDecodeWithPagedKVCacheDispatched` for attention across all requests in one kernel launch.

**Three bugs found and fixed** (see `errors/2026-03-31-flashinfer-segfault-debug.md`):
1. Hardcoded `MAX_SEQ=4096` in attention kernels vs runtime `max_seq_len=1024` -> OOB writes
2. FlashInfer `plan_info` allocated on GPU but read/written via CPU `memcpy` -> segfault
3. Double token allocation (scheduler + model both called `alloc_tokens`) -> metadata corruption

### Phase 2: Buffer Pre-allocation (434 -> 681 tok/s, +57%)

**Problem**: `BatchDecodeBuffers::new()` allocated ~10 GPU tensors + 128MB FlashInfer workspace every decode step. Profiling showed 4.5ms/step (32% of total 14ms).

**Fix**: allocate once on first use, reuse across all subsequent steps. Buffers sized for `max_batch_size`; smaller batches adjust `seq_len`.

### Phase 3: FlashInfer Plan Once (681 -> 690 tok/s, +1.3%)

**Problem**: `flashinfer_batch_decode` called both `plan` (CPU scheduling) and `run` (GPU kernel) per layer. Plan was called 36x per step unnecessarily — KV layout is identical across layers.

**Fix**: split into `flashinfer_plan` (once before layer loop) + `flashinfer_run_layer` (per layer).

## Remaining Gap

### Phase 4: Embedding + Logits Pre-allocation (690 -> 700 tok/s)

Moved embedding output into `BatchDecodeBuffers` (eliminating `get_embeddings_batch` allocation) and lazy-allocated logits buffer. Small gain — these allocations were only ~40KB.

### Phase 5: CUDA Graph (attempted, not yet landed)

CUDA Graph capture for the layer loop would eliminate ~360 kernel launches (~1.8ms per step). However, FlashInfer's `batch_decode_run` does CPU `memcpy` of `plan_info` inside the function — this CPU code doesn't run during graph replay, which means kernel parameters from the first capture are baked in. This works only when FlashInfer's plan produces the same scheduling layout across steps (true for fixed batch_size with similar KV lengths, but needs validation).

Implementation prepared (graph_cache HashMap per batch_size, decode_batch_graph_body method) but disabled pending correctness validation.

| Metric | infer | SGLang | Gap |
|--------|-------|--------|-----|
| 8-concurrent throughput | 700 tok/s | 886 tok/s | 1.27x |
| ITL (decode latency) | 9.6ms | 8.2ms | 1.17x |

Remaining ~1.4ms ITL gap: CUDA Graph (~1.0ms kernel launch overhead) + misc overhead.

## Rule

- **Profile before optimizing** — the 4.5ms allocation overhead was invisible without explicit timing. Always measure before assuming.
- **SGLang's architecture is the reference** — token pool, batched GEMM, FlashInfer paged attention, CUDA Graph per batch size. Follow this blueprint.
- **Host/device pointer mismatches are silent killers** — FlashInfer's C++ API uses CPU memcpy for some buffers. Always verify whether a pointer argument is host or device.
- **Allocation in hot loops is expensive** — GPU buffer allocation via cuMemAllocAsync costs ~0.5ms per call. Pre-allocate everything.

### Phase 6: CUDA Graph for Batched Decode (700 -> 756 tok/s, +8%)

Captured CUDA Graphs for the decode layer loop (36 layers x ~14 kernels = ~504 launches). One graph per batch_size cached in HashMap. First call captures, subsequent calls replay.

Key insight (from SGLang source): FlashInfer plan() runs outside graph, only run() is captured. Graph replays kernel launches with same GPU buffer pointers but updated data.

**Final: 756 tok/s at 8-concurrent (SGLang: 886, gap: 1.17x)**
