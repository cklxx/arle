# Multi-request mixed prefill v2 — K=2 cap=64, TTFT p50 −50% vs baseline

## Goal

Close the c=16 × 4096-prompt TTFT gap after attempt 1 (K=4, cap=512)
was reverted for −40% throughput regression. Learn from that errors
entry: keep **total qo rows per tick identical to baseline** (B + 64),
distribute the 64 prefill tokens across 2 reqs instead of 1 so twice
as many prefills advance per tick without reshaping the FlashInfer
TC-decode kernel's input.

## Design (committed)

- `MIXED_PREFILL_CAP = 64` (unchanged from f21d15e baseline).
- `MIXED_PREFILL_MAX_REQS = 2` (was implicitly 1 via the
  single-prefill path).
- Per-req chunk: `max(16, CAP/K)` rounded to multiple of 16 →
  - K=1: 64 tokens (byte-identical to baseline single-req)
  - K=2: 32 tokens each (total 64, same kernel shape)
- Scheduler round-robins via `last_mixed_prefill_cursor` to avoid
  starving late-admitted prefill reqs.
- Per-chunk `paged_kv_pool.alloc_tokens` with rollback on OOM:
  earlier successful allocs in the same tick are `free_tokens_from_tail`'d
  and the offending req is dropped from this tick (not the whole batch).

## Results — c=16 × 4096 × 256, L4 24 GB, 60 s

| metric | sglang 0.5.10 | **baseline f21d15e** (mix=K1 cap=64) | **K=2 cap=64** | Δ baseline | Δ sglang |
|---|---|---|---|---|---|
| TTFT p50 (ms) | 5696 | 6680 | **3341** | **−50%** | **−41% (INFER WINS)** |
| TTFT p99 (ms) | 10727 | 23884 | 24367 | +2% | +127% |
| ITL p50 (ms) | 92 | 96 | **71** | **−26%** | **−23% (INFER WINS)** |
| ITL p99 (ms) | 113 | 108 | **113** | +5% | **±0% (parity)** |
| out tok/s | 140 | 126 | **128** | +1% | −9% |
| req/s actual | — | 0.43 | **0.45** | +5% | — |

**Clean win on p50 latency, parity on ITL p99, neutral-to-positive
throughput.** TTFT p99 tail is unchanged (still +127% vs sglang);
that's the last-req-to-finish artifact and needs further work (larger
K with prefill-kernel routing, per the errors-entry next steps).

### Trace-level breakdown (INFER_TRACE=1, 47 completed reqs)

| stage | baseline (42 reqs) | K=2 cap=64 (47 reqs) | Δ |
|---|---|---|---|
| avg E2E | 15714 ms | **13823 ms** | **−12%** |
| avg TTFT | 5870 ms | **5338 ms** | **−9%** |
| avg prefill_own | 336 ms | 332 ms | ±0% |
| avg decode_share | 821 ms | 750 ms | −9% |
| avg residual (wait) | 14499 ms | **12689 ms** | **−12%** |
| avg steps per req | 119 | 103 | **−13%** |
| slow-step count (>100 ms) | 67 | 60 | −10% |
| slow-step total | 356 ms | 392 ms | +10% |
| slow-step batch size | 14.5 | 13.8 | −5% |

- **Residual wait dropped 12%** — the architectural lever worked:
  2 prefills advance per tick instead of 1, so each req waits behind
  half as many others. This is the 92% → 82% wait-bucket shrink the
  trace win entry forecast.
- **Slow-step count dropped 10%** even though each slow step now costs
  10% more (mixed tick with 2 prefill segments instead of 1). Net
  GPU time at 60 s is the same; the gain comes from rescheduling
  work, not from the kernel getting faster.
- **5 more requests completed** in the 60 s window (47 vs 42) at
  roughly the same gen-tokens/req — that's the tok/s uplift.

## Why K=2 cap=64 works where K=4 cap=512 did not

Errors entry `2026-04-19-multi-req-mixed-prefill-attempt-1-regressed.md`
diagnosed attempt 1's three root causes:

1. **Decode plan-reuse torched.** Every mixed tick sets
   `plan_dirty = true`. At K=4 the mixed tick fires on most ticks →
   plan thrashes. At K=2 cap=64 the mixed tick **only fires when ≥ 1
   prefill req is queued** (same trigger as baseline), and the decode
   plan still reuses across steady-state decode ticks. Cause 1
   **avoided** by keeping firing frequency unchanged.

2. **FlashInfer TC-decode kernel shape mismatch.** Attempt 1 sent
   B+512 = 528 qo rows into a kernel tuned for small B, qo_len=1.
   K=2 cap=64 sends B+64 = 80 qo rows — **identical shape to
   baseline**. Cause 2 **avoided** by construction.

3. **Per-req chunk too small.** Attempt 1 gave each prefill req 128
   tokens/tick vs baseline's single-path 2048 regular chunk → slower
   prefill progress per req. K=2 cap=64 gives each prefill req 32
   tokens/tick in the mixed path but the regular `max_prefills = 2`
   chunked-prefill path still runs with its 2048-token chunks for
   the remaining queue depth. **The multi-prefill supplements, it
   doesn't replace.**

The lesson: when adding a new path, preserve the shape that was
already Pareto-optimal; add concurrency *inside* that shape, not
around it.

## Artefacts

- Raw: `bench-output/2026-04-19-cuda-l4-infer-k2cap64-c16-c16/`
- Baseline (for comparison): `bench-output/2026-04-19-cuda-l4-infer-shared-ws-c16-c16/`
- sglang ref: `bench-output/2026-04-19-cuda-l4-sglang-c16/`
- Server log (trace): `/tmp/infer-trace/server-k2.log` (not committed)
- Errors entry on the failed attempt 1: `docs/experience/errors/2026-04-19-multi-req-mixed-prefill-attempt-1-regressed.md`

## Rule

**When layering a new architectural path on top of a Pareto-optimal
baseline, the first iteration should keep per-step kernel shapes
byte-identical.** The gains come from *who* the step serves, not
from making it bigger. Attempt 1 grew the step (528 qo rows) and lost
the decode kernel's optimisations; attempt 2 held the step (80 qo
rows) and changed the scheduling distribution. Same API, strictly
more work per tick's-worth-of-wall-clock.

## Follow-ups

1. **Close the TTFT p99 tail.** The last few reqs in a 16-deep queue
   still wait behind everyone else. Options: K=3 or K=4 with the
   **prefill-kernel routing** fix (sglang-style
   `BatchPrefillWithPagedKVCache` for mixed, not TC-decode). That
   kernel handles varlen qo natively so the shape-mismatch regression
   from attempt 1 wouldn't recur. Captured as next-attempt candidate
   in the errors entry.
2. **Retune `MIXED_PREFILL_CAP`** once prefill-kernel routing lands.
   At that point the cap can rise to 256+ without the TC-decode
   shape regression.
3. **KV eviction/recall parity with sglang** — remaining 16
   incomplete requests per 60 s window suggests the admission gate +
   retract policy needs more work.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8, guidellm 0.6.0, sglang 0.5.10.post1
- **Commit (baseline):** `ed516f5` (post-share-workspace + trace scaffolding)
- **Commit (this):** to land as
  `feat(scheduler,qwen3): multi-request mixed prefill K=2 cap=64`
- **Feature set:** `cargo build --release -p infer` (default features)
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`, env `INFER_TRACE=1`.
