# E2E per-request trace + shared FlashInferWorkspace — c=16 TTFT is 92% wait; cap tuning exhausted

## Goal

Instrument the CUDA scheduler so `Σ(stage_ms) ≈ measured E2E_ms` for
every completed request, then probe the 130% TTFT p99 gap vs sglang at
c=16 × 4096-prompt × 256-output on L4 24 GB. Along the way, eliminate
the duplicate `FlashInferDecodeMetadata` the mixed decode+prefill path
was allocating (a ~328 MiB VRAM win regardless of cap value).

## What shipped

### 1) `INFER_TRACE=1` per-request attribution

`infer/src/scheduler/cuda/{request,execution,runtime}.rs`:

- Added 7 fields to `ActiveRequest` (40 bytes/request, zero cost when
  `INFER_TRACE` is unset): `first_step_at`, `finished_at`,
  `t_prefill_us`, `t_decode_us`, `t_emit_us`, `t_new_us`, `step_count`.
- `execution.rs::step()` captures per-stage timings per request inside
  `if trace_on`: each `emit_delta`, `step_new`, and `step_prefill_chunk`
  call is wrapped with `Instant::now()` and charged to the specific
  request whose state it mutated. The batched decode launch+readback
  is charged as `decode_us / decode_count_pre` to every request in
  `Phase::Decoding | Phase::Finished`.
- `runtime.rs::cleanup()` emits `TRACE id=…` log lines when a request
  transitions to `Finished`, with
  `residual_us = e2e_us − (queue + new + prefill + decode + emit)`.
  Residual is the "in-flight but someone else is running" bucket.

### 2) Shared `FlashInferWorkspace` between decode and mixed paths

`infer/src/model/qwen3/batch_decode.rs`:

- Deleted the `metadata: FlashInferDecodeMetadata` field from
  `MixedBatchBuffers` (used to own a separate 256 MiB float workspace
  + 8 MiB int workspace + 8×8 MiB page-locked pool = **~328 MiB
  duplicate VRAM**).
- `decode_batch_with_prefill` now disjoint-borrows
  `BatchDecodeBuffers { mixed, metadata, logits_batch, .. }` and uses
  `bufs.metadata` directly for the mixed path. Safe because the two
  paths never run in the same scheduler tick (single-threaded scheduler
  with mutually-exclusive phases).
- `BatchDecodeBuffers::new` inflates `FlashInferDecodeMetadata` sizing
  from `max_batch_size` to `max_batch_size + MIXED_PREFILL_CAP` so the
  shared `positions`/`kv_indptr`/`qo_indptr`/`kv_last_page_len` buffers
  hold the mixed path's B + C rows without overflow. (The OOB write
  that surfaces without this panics in cudarc `memcpy_htod` with
  `dst.len() >= src.len()` assertion.)

## Results — 4× c=16 sweeps

| metric | sglang 0.5.10 | **baseline mix=64** | mix=128 | mix=256 | **shared-ws mix=64** |
|---|---|---|---|---|---|
| TTFT p50 (ms) | 5696 | 6680 | 6846 | 7154 | **6680** ✓ |
| TTFT p99 (ms) | 10727 | 23884 | 16034 | 15952 | **23817** ✓ |
| ITL p50 (ms) | 92 | 96 | 96 | 93 | **96** ✓ |
| ITL p99 (ms) | 113 | **108** (INFER WINS) | 220 ❌ | 119 | **108** ✓ |
| out tok/s | 140 | 126 | 100 | 90 | **127** ✓ |

**Pareto summary:**
- `mix=64` remains the Pareto-best: beats sglang on ITL p99 (−4%),
  loses TTFT p99 (+130%), throughput −10%.
- `mix=128/256` trade TTFT p99 for ITL p99 and throughput
  dollar-for-dollar. Larger mixed chunks make each fused forward
  slower, directly hitting decode variance and rate.
- **Shared-workspace refactor is Pareto-neutral** at mix=64 (every
  number within noise) while saving 328 MiB VRAM → unblocks any
  future experiment that needs that headroom.

Also ran `--decode-prefill-cap` sweep (512 / 1024 / 2048) — same
pattern: bigger chunks = faster TTFT p99, worse ITL p99, worse tok/s.
No one-knob tune hits 99% parity.

## What the trace revealed (mix=64 baseline, 42 completed reqs, `INFER_TRACE=1`)

| stage | ms/req | % of E2E |
|---|---|---|
| queue (admit→first step) | 57 | 0.4% |
| step_new (CPU admission) | 0 | 0.0% |
| **own prefill chunks** | **336** | **2.1%** |
| **own decode share** (decode_us/B) | **821** | **5.2%** |
| tokenizer emit | 1 | 0.0% |
| **residual (inter-request wait)** | **14499** | **92.3%** |
| **E2E mean** | **15714** | **100%** |

Σ(measured stages + residual) = 100% by construction — 100% coverage
of the link time. Residual is **not a measurement gap**; it is the
real time this request was in `active` while the scheduler was
processing *other* requests.

Scheduler-step level (slow steps >100 ms, N=67):

| field | value |
|---|---|
| avg step total | 356 ms |
| decode launch+readback | 132 ms (37%) |
| regular prefill chunks | 224 ms (63%) |
| avg batch during slow steps | 14.5 |

Each slow tick: 16-way decode (132 ms) + ~1 regular prefill chunk
(224 ms) + 1 mixed chunk (64 tok, negligible). The prefill chunk
**blocks** the next decode tick in the regular path — which is why
ITL p99 is tight only when the regular chunk cap is small.

## Why bigger caps lose

- **`--decode-prefill-cap=512` (baseline):** 4096-tok prompt ÷ 512 =
  8 chunks per req. `max_prefills=2` under decode-active policy,
  so each tick advances ≤2 reqs' prefill. 16 reqs × 8 chunks ÷ 2
  prefills/tick = 64 ticks × 356 ms = **23 s to drain the prefill
  queue** — matches observed TTFT p99 (23.8 s) within 1%.
- **`mix=128`:** each mixed step now does 2× more prefill work (128
  tok vs 64), pushing total step time past 100 ms and collapsing the
  "fast decode" pockets. ITL p99 goes from 108 → 220 ms; throughput
  drops 21%.
- **`mix=256`:** 4× more mixed-prefill work per step. TTFT p99 drops
  (−33%) because mixed chunks accumulate faster, but throughput drops
  29% due to the same "slow step starves decode" effect.

## The root cause

**92% of every request's E2E is time the GPU is processing *other*
requests**. Own work averages 1.15 s out of 15.7 s. Cap knobs can
shift the 8%/92% split by millimetres but cannot shrink the 92%
bucket — that bucket only shrinks when **multiple prefilling requests
advance per tick**.

sglang closes this via `--enable-mixed-chunk`: one FlashInfer
`BatchPrefillWithPagedKVCache` call per step that takes varlen
`qo_indptr` where decode rows have `qo_len=1` and **ALL prefilling
rows** contribute their chunk. Our mixed path currently handles
**one** prefilling req per fused forward (`step_decode_launch_mixed`
picks a single `pending_mixed_prefill_idx`). That's the architectural
delta.

## Rule

Before tuning a scheduler cap, instrument per-request attribution so
`Σ(stage) = E2E` within 5%. If the residual bucket dominates, no cap
value reaches parity — the fix is structural (fuse more work per
forward) not arithmetic (pick a better cap). We burned four bench
runs validating this; the trace would have said so in one.

When sharing workspace buffers between two paths that have different
`max_batch_size` effective widths, size the buffers for the **union**
of needs (here: `max_batch_size + MIXED_PREFILL_CAP`). cudarc's
`memcpy_htod` assertion (`dst.len() >= src.len()`) catches the
overflow, but only at runtime.

## Follow-ups

1. **Multi-request mixed prefill** (the real fix).
   `decode_batch_with_prefill` + `step_decode_launch_mixed` need to
   accept a vector of prefilling request indices, each contributing
   `qo_len = chunk_per_req` rows to the fused FlashInfer plan. Every
   step then advances every in-flight prefill by one chunk, collapsing
   the 16× serial fan-out. With the workspace sharing already in
   place, this is no longer blocked by VRAM.
2. **Admission gate retrospect.** Several reqs hit pool-OOM during
   prefill (3751 pages × 16 = 60 016 tokens vs 16 × 4096 = 65 536
   tokens needed — admission should prevent this fan-out, not
   discover it via `pool alloc failed`).
3. **`/health` route** — guidellm 0.6.0 still probes `GET /health`,
   we work around with `validate_backend=/v1/models`. Small chore.

## Artefacts

- Raw baseline mix=64: `bench-output/2026-04-19-cuda-l4-infer-trace-c16-v2-c16/`
- Raw mix=128: `bench-output/2026-04-19-cuda-l4-infer-mix128-c16-c16/`
- Raw mix=256: `bench-output/2026-04-19-cuda-l4-infer-mix256v2-c16-c16/`
- Raw shared-ws mix=64: `bench-output/2026-04-19-cuda-l4-infer-shared-ws-c16-c16/`
- Raw cap=1024: `bench-output/2026-04-19-cuda-l4-infer-cap1024-c16-c16/`
- Raw cap=2048: `bench-output/2026-04-19-cuda-l4-infer-cap2048-c16-c16/`
- sglang ref: `bench-output/2026-04-19-cuda-l4-sglang-c16/`
- Server logs with TRACE lines at `/tmp/infer-trace/server*.log` (not
  committed; ≥ 100 MB each).

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8, guidellm 0.6.0, sglang 0.5.10.post1
- **Commit:** diff against `71c0f7f`; to commit as
  `feat(scheduler,qwen3): INFER_TRACE per-req attribution + share FlashInferWorkspace`.
- **Feature set:** `cargo build --release -p infer` (default features)
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`, env `INFER_TRACE=1`.
