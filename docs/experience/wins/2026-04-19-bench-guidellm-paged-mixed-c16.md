# Paged mixed-batch fusion — c=16 ITL p99 beats sglang, throughput +23%

> **Drift notice (added 2026-04-20):** absolute tok/s numbers cited here
> predate a `guidellm 0.6.0` env drift — today the same commits
> re-measure at ~98 tok/s. See
> [`errors/2026-04-20-bench-drift-environmental-not-code.md`](../errors/2026-04-20-bench-drift-environmental-not-code.md).
> The kernel/scheduler findings below remain accurate.

## Goal

Close the ITL p99 +93% gap and narrow the throughput −27% gap vs sglang
0.5.10 at c=16 × 4096-prompt × 256-output on L4 24 GB Qwen3-4B by
activating paged-KV-aware mixed-batch fusion (sglang's
`--enable-mixed-chunk` equivalent).

Supersedes: [`2026-04-19-bench-guidellm-sglang-parity-c16-postmerge.md`](2026-04-19-bench-guidellm-sglang-parity-c16-postmerge.md).

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8, guidellm 0.6.0, sglang 0.5.10.post1
- **Commit:** unstaged at diff time (builds on `91bc7f2`); commit message to be
  `feat(qwen3): paged mixed-batch — K/V scatter via paged prep`
- **Feature set:** `cargo build --release -p infer` (default features)
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`

## Results — c=16 headline

| metric | sglang 0.5.10 | baseline (91bc7f2) | **pagedmix4** | Δ vs sglang | Δ vs baseline |
|---|---|---|---|---|---|
| TTFT p50 (ms) | 5696 | 5066 | 6358 | +12% | +25% |
| TTFT p99 (ms) | 10727 | 15199 | 24661 | **+130%** | +62% |
| ITL p50 (ms) | 92 | 102 | 98.5 | +7% | −3% |
| **ITL p99 (ms)** | **113** | 221 | **108.9** | **−4% (INFER WINS)** | **−51%** |
| out tok/s | 140 | 101 | **123.7** | −12% | **+23%** |
| successful | 32 | 24 | 26 | −19% | +8% |
| incomplete | 16 | 16 | 16 | = | = |

## Artefacts

- Raw: `bench-output/2026-04-19-cuda-l4-infer-pagedmix4-c16/`
- sglang ref: `bench-output/2026-04-19-cuda-l4-sglang-c16/`
- baseline: `bench-output/2026-04-19-cuda-l4-infer-nograph-c16/`

## What changed

`infer/src/model/qwen3/batch_decode.rs::decode_batch_with_prefill`:

1. **Dropped the dead guard** at lines 364-368 that returned `Ok(false)`
   when `prefill_state.base.kv_cache.k_caches().is_empty()` (always true
   under paged Qwen3 — the contiguous kv_cache is a 16-token stub) or
   when `prefill_start_pos + c > prefill_state.base.kv_cache.max_seq_len()`.
   This guard made the mixed path dead code — it only fired when contiguous
   kv_cache had ≥c + start_pos tokens of headroom, i.e. never under paged.

2. **Swapped `prefill_attention_prep_dual_write_cuda` for
   `prefill_attention_paged_prep_cuda`** inside the per-layer loop. The
   dual-write kernel scattered to both contiguous `k_cache`/`v_cache` AND
   the paged pool; the paged-only variant (already in the codebase,
   originally built for the pure paged prefill path) does QK-norm + RoPE
   + paged page-table scatter. Drops `kc_ptr`/`vc_ptr` from the call.

3. **Uploaded `prefill_page_table` to device** via `ctx.stream.memcpy_stod`
   before the kernel call. This was the root cause of the early
   `gemm_cuda CUDA_ERROR_UNKNOWN` crash in the first swap attempt — the
   paged-prep kernel dereferences `page_table` as a device pointer, but
   the Vec built from `paged_kv_pool.page_indices(slot)` is host-side.
   The dual-write kernel masked this bug because the contiguous write
   was the primary K/V target and the paged scatter's OOB read "happened
   to not corrupt anything fatally" in that path.

4. Dropped the now-unused `prefill_max_seq_len` local (only fed into the
   dual-write signature).

## Problems observed

1. **TTFT p99 regressed +130%** (10.7 s → 24.7 s vs sglang). Cause:
   mixed prefill chunks are now capped at `MIXED_PREFILL_CAP = 64`
   tokens. At 4096-token prompts that's 64 mixed calls per request. The
   scheduler throttles `max_prefills` to 2 when decodes are active, so
   each tick advances prefill by ≤ 128 tokens. Closing TTFT requires
   raising `MIXED_PREFILL_CAP` above 64 — but the prior 2026-04-17
   regression entry flagged that the buffer sizing
   (`max_tokens = max_batch_size + MIXED_PREFILL_CAP`) + CUDA Graph
   shape capture collide when this is bumped. Needs a deliberate follow-up.

2. **ITL p50 slightly worse** (92 → 98.5 ms) — +7%. The mixed-batch
   forward does more work per tick (decode + prefill-chunk prep +
   attention) so per-step latency rises. ITL p99 drops dramatically
   (221 → 109 ms) because the worst-case decode steps no longer get
   stalled behind separate prefill forward passes. Net positive for
   decode variance.

3. **`successful` dropped from sglang 32 → 26** (−19%). The gap is now
   TTFT-dominated, not ITL-dominated. With TTFT p99 at 24.7 s, some
   requests in the 60 s window never reach their first token before
   `--max-seconds` cuts them. This is exactly the mixed-chunk-cap
   follow-up from problem 1.

## Learnings

1. **Latent host-pointer bugs in hybrid kernels surface the moment the
   "other" write is removed.** The dual-write prep kernel accepted a
   host `page_table` pointer because its paged scatter was a
   write-through-to-cold-cache, not the primary target. Under a
   paged-only replacement, the kernel dereferences the page_table on
   every token → a single invalid device read corrupts the CUDA context
   → next sync panics with `CUDA_ERROR_UNKNOWN` at an unrelated gemm.
   The exact line of the panic is a red herring; the root cause is
   always upstream. Always H2D-upload any `*const i32` / `*mut u8` that
   a kernel treats as device memory.

2. **Mixed-batch fusion is the right lever for ITL p99.** With decode
   and prefill-chunk fused into one forward pass, decode never gets
   stalled behind a separate prefill forward. This closes sglang's
   ITL p99 lead completely — infer now edges ahead by 4%.

3. **TTFT and ITL live on the same lever at c=16.** The mixed chunk
   size controls the decode-prefill interleave frequency: larger chunks
   = fewer mixed ticks per prefill = lower TTFT but higher ITL p99;
   smaller chunks = more ticks = higher TTFT but lower ITL p99. sglang
   defaults this to 512 (`--enable-mixed-chunk chunk-size`). We cap at
   64 due to buffer-sizing constraints — see problem 1.

## Rule

For paged-KV kernels that take a `*const i32 page_table` argument,
**always verify the argument is a device pointer at the call site**,
not a host `Vec<T>.as_ptr()`. Search the kernel `.cu` source for the
pattern `page_table[logical_pos / page_size]` — if the index is read
on the GPU, the pointer must be device.

When porting a hybrid-write kernel to a single-sink variant, audit
every pointer arg — the hybrid may have masked host/device confusion.

## Follow-ups

- **Raise `MIXED_PREFILL_CAP` above 64** with a matching bump in
  `batch_decode.rs:119` `max_tokens = max_batch_size + MIXED_PREFILL_CAP`
  and graph-capture shape adjustment. Target: close TTFT p99 from 24.7 s
  to ≤ 12 s (≈ sglang parity + 10%).
- **KV eviction / recall improvements** (sglang-parity page-level
  retract of prefilling slots + cross-tier recall). Needed for the
  remaining 16 incomplete requests on this workload.
- **`/health` route** in `infer/src/http_server/` so guidellm 0.6.0
  validates without the `--backend-kwargs` override.
