# Gap #5 Commit 1 — `PagedKVPool::copy_pages_{to,from}_host` real CUDA impl — 2026-04-20

## Context

- **Backend:** cuda
- **Hardware:** NVIDIA L4, CUDA 13.0, driver `580.82.07`, sm_89
- **Plan:** `docs/plans/gap5-kv-tier-demote-prefetch.md` C1
- **Bench scope:** exempt (no scheduler/runtime change, no new hot path — pure kernel-adjacent wiring + unit test)

## What changed

Replaced the two `todo!("...requires validation on a CUDA host")` stubs in `crates/cuda-kernels/src/paged_kv.rs` with real per-layer per-page `cudaMemcpyAsync` via cudarc's `memcpy_{dtoh,htod}` on a passed-in stream. Host payload layout:

```
page 0: [L0 K bytes][L0 V bytes][L1 K bytes]…[L_{N-1} V bytes]
page 1: same shape
...
```

Size helper `pages_host_byte_len(&[u32]) = pages.len() * num_layers * 2 * num_kv_heads * page_size * head_dim * bpe`. BF16-only in v1 (FP8 / INT8 / TurboQuant have separate scale/norm tensors which this helper does not carry — that's v2).

Signature change (the two helpers now take `&Arc<CudaStream>`):

```rust
pub fn copy_pages_to_host(&self, pages: &[u32], stream: &Arc<CudaStream>) -> Result<Vec<u8>>
pub fn copy_pages_from_host(&mut self, pages: &[u32], payload: &[u8], stream: &Arc<CudaStream>) -> Result<()>
```

Two existing callers (`Scheduler::read_block_payload`, `Scheduler::install_restored_kv` in `infer/src/scheduler/cuda/core.rs`) updated to pass `&self.model.device_context().stream`.

## Validation

New integration test `crates/cuda-kernels/tests/paged_kv_copy_pages_roundtrip.rs` runs on a real GPU:

1. Builds a small live `TokenKVPool` (16 MiB BF16 budget, 4 layers × 2 heads × 32 head_dim).
2. Seeds every page's K and V bytes with `cuMemsetD8` using a known pattern.
3. `copy_pages_to_host` a non-contiguous page selection (`[0, 2, 3]`).
4. Asserts the blob's bytes match the seeded pattern at every `[page][layer][K|V]` offset.
5. Zeros the source pages on device, `copy_pages_from_host` from the blob, reads back via a second D→H, asserts byte-equal with the first blob.
6. Negative: length-mismatch payload and out-of-range page both return `Err` with the expected message.

**Result:** `test cuda_tests::copy_pages_roundtrip_bf16 ... ok` in 0.16 s on L4.

## Why it matters

The `todo!()` in the CUDA branch was the single structural blocker for HiRadixCache-style T1 demote on evict (`docs/plans/gap5-kv-tier-demote-prefetch.md` §"Existing infrastructure audit"). Now C2 (coordinator `Demote` command byte path) and C4 (promote-back in `StagePlanner::stage`) can wire against a validated byte mover. No scheduler change yet — `evict_prefix_cache_if_pressured` still frees pages outright; the wiring lands in C3/C4.

## Key learning

`cudarc 0.18.2`'s `CudaStream::memcpy_{htod,dtoh}` are defined with receiver `self: &Arc<Self>` (see `src/driver/safe/core.rs:1322` and `:1363`), not `&CudaStream`. Passing `&CudaStream` to functions that call these methods compiles with a cryptic "method not found in `&CudaStream`" error — the fix is to type the parameter as `&Arc<CudaStream>`. Noted for future callers.

## Follow-ups

- **Gap #5 C2**: Coordinator `Demote` + `DemoteCompleted` byte path over the dedicated copy stream in `LocalCudaTransport`. Sits under `INFER_T1_DEMOTE_ENABLED=false` gate initially.
- **Gap #5 C3**: Scheduler demote hook in `evict_prefix_cache_if_pressured` + `t1_demote_min_hits` config.
- **Gap #5 C4**: Real promote-back in `StagePlanner::stage` — allocs fresh pool pages via `alloc_detached_pages`, H→D copies from the T1 region, emits `StagingCompleted`.
- **Non-BF16 support (v2)**: FP8/INT8/TurboQuant would also need to round-trip scale/norm tensors (`k_scales`, `v_scales`, `k_norms`, `v_norms`, TurboQuant state). Not blocking current L4 c=16 Qwen3-4B BF16 workload.
