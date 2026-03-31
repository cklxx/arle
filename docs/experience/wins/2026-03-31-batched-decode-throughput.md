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

| Metric | infer | SGLang | Gap |
|--------|-------|--------|-----|
| 8-concurrent throughput | 690 tok/s | 886 tok/s | 1.28x |
| ITL (decode latency) | 10.1ms | 8.2ms | 1.23x |

Root cause of remaining ~2ms ITL gap: **CUDA Graph**. SGLang pre-records CUDA Graphs for batch sizes [1,2,4,8,12,16,24,32], eliminating ~360 kernel launches per step (~1.8ms overhead). This is the last major optimization needed.

## Rule

- **Profile before optimizing** — the 4.5ms allocation overhead was invisible without explicit timing. Always measure before assuming.
- **SGLang's architecture is the reference** — token pool, batched GEMM, FlashInfer paged attention, CUDA Graph per batch size. Follow this blueprint.
- **Host/device pointer mismatches are silent killers** — FlashInfer's C++ API uses CPU memcpy for some buffers. Always verify whether a pointer argument is host or device.
- **Allocation in hot loops is expensive** — GPU buffer allocation via cuMemAllocAsync costs ~0.5ms per call. Pre-allocate everything.
