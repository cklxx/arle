# c=16 paged-pool admission overcommit — scheduler admits more than pool can serve

## Context

- **Date:** 2026-04-18
- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B (bf16)
- **Hardware:** NVIDIA L4 24GB, CUDA 13.0 (driver 580.82.07)
- **Commit:** d91908f (+ local flashinfer include-path build fix)
- **Trigger:** `guidellm --profile concurrent --rate 16 --data prompt_tokens=4096,output_tokens=256`
  against infer launched with `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --cuda-graph=false`.
- **Symptom:** 220 / 241 requests fail during chunked prefill with
  `TokenKVPool: out of pages (requested 512 tokens / 32 new pages, available <30 pages)`.
  Failed requests stream back empty 200-OK responses, so guidellm's p50 TTFT/ITL = 0 ms
  and only p95+ reflect real completions. Headline table entry:
  `| conc16 | 0 | 10691 | 0 | 166.68 | 1004.94 | 3.85 |` — the 0-ms percentiles are
  bogus; the benchmark is contaminated.
- **Raw artefact (contaminated, kept for forensics):** `bench-output/2026-04-18-cuda-l4-infer-c16-run3/`.
- **Related prior finding:** `project_remote_cuda_box.md` — "C≥8 is structurally broken
  on this box"; this entry is the root-cause explanation.

## Root Cause

Two loops conspire to overcommit the paged KV pool:

1. `scheduler/cuda/runtime.rs:258 fn assign_slots()` admits waiting requests
   into any free slot it sees, without checking that the paged pool has enough
   pages for the new request's prefill. The only early-exit is `free_slots().is_empty()`.
2. `scheduler/cuda/prefill.rs:247` (paged prefill chunk path) then calls
   `alloc_pool_tokens_with_retry(slot_idx, chunk_len)` per 512-token chunk
   per request. When 16 concurrent 4096-token prefills all hit
   chunk-`N` simultaneously and the pool has exhausted, the retry-with-
   eviction path at `core.rs:861 alloc_pool_tokens_with_retry` reclaims a
   few pages from the prefix cache and retries — but when the shortfall
   is on the order of hundreds of pages (not ones/tens), reclaim
   cannot cover it. The request is then marked `Phase::Finished` and the
   HTTP streamer emits an empty 200-OK.

Concretely on L4 24 GB:

- After weights load (8 GB) and CUDA Graph + FlashInfer workspace
  (~1–2 GB each), the pool budget at `mem_fraction_static=0.94` with
  `cuda_graph=false` is **8.8 GB = 59 700 tokens = 3 731 pages**
  (16 tok/page, bf16 KV).
- 16 simultaneous 4096-token prefills need **16 × 4096 = 65 536 tokens =
  4 096 pages** just for prefill KV — exceeds the pool even before any
  decode growth.
- `assign_slots()` admits all 16 anyway; chunked prefill then races to
  consume the pool. First ~10 requests complete their early chunks
  successfully; the trailing 6 run out of pages on chunk N>1 and fail.

sglang's scheduler gates admission on `max-running-requests` **combined
with** a pool-capacity check (`--max-total-tokens` = sum of live sequences),
so the same L4 box at c=16 admits only as many as the pool can sustain.
Our scheduler has the slot budget but not the token-budget gate.

Relevant files:

- `infer/src/scheduler/cuda/runtime.rs:258` — admission loop, no pool check.
- `infer/src/scheduler/cuda/prefill.rs:247` — where the failure surfaces.
- `infer/src/scheduler/cuda/core.rs:861 alloc_pool_tokens_with_retry` —
  reclaim-and-retry safety net that cannot close a multi-hundred-page gap.
- `infer/src/scheduler/cuda/core.rs:325` — pool budget sizing (subtracts
  weights + headroom, does not reserve per-slot prefill working set).

## Fix (landed 2026-04-19)

Closed by the admission-time pool-capacity gate in
`scheduler/cuda/runtime.rs::assign_slots` + `scheduler/cuda/core.rs::admission_budget_pages`.
Live result: c=16 at 4096+256 now completes with **zero** pool alloc
failures. See `docs/experience/wins/2026-04-19-sglang-parity-c16-gated.md`
for the paired-bench numbers and the three-iteration convergence story
(only the third formula — `free_count − future_growth − headroom` — is
correct; the earlier two over-admit for different reasons).

Code shape landed:
- `ActiveRequest.reserved_pool_pages: usize` — worst-case reservation
  committed at admission (`(uncached_prompt + max_tokens) / page_size`).
- `Scheduler::admission_budget_pages()` — `free_count()/page_size` minus
  sum of `(reserved − pool.seq_len(slot)/page_size)` across active
  requests, minus `DECODE_HEADROOM_PAGES = 32`.
- `assign_slots`: before admission, checks `needed_pages` against the
  running-per-tick budget; gated requests push-front-back onto `waiting`
  with an `info!` log tagged `held for pool`.
- Regression test: `assign_slots_gates_admission_on_available_pool_pages`
  in `infer/src/scheduler/cuda/runtime.rs`.

Follow-up (open, tracked in the wins entry §Problems): the gate is
intentionally conservative. Steady-state admitted count at c=16 is ~8
instead of 16 — throughput gap vs sglang is now driven by admission
throttling, not the original bug. Three tuning levers are listed in the
wins entry.

## Fix (proposed, superseded by the Fix landed section above)

Add an admission-time token-budget gate, analogous to sglang's:

1. Track `in_flight_prefill_tokens` (sum of prefill-token-budget for each
   admitted, not-yet-decoding request) and `in_flight_decode_tokens` (sum of
   current sequence lengths for decoding requests).
2. Before `assign_slots()` pops a waiting request, estimate its prefill
   token cost = `prompt_len` (minus what prefix cache will serve). If
   `in_flight_prefill_tokens + prefill_cost + in_flight_decode_tokens >
   pool.capacity_tokens − decode_headroom`, break the admission loop and
   wait for a later tick.
3. Release the counter entries as each request transitions prefill→decode
   (prefill cost flips to decode cost) and decode→finished (decode cost
   returns).

This reuses the pool's existing bookkeeping; no new storage. The only
scheduler-policy decision is how much `decode_headroom` to reserve — a
small fraction (e.g. 8 × `prefill_chunk_size`) covers 8 decoding requests
growing by one chunk between admission passes.

Cross-reference: this is the same bug class as the M2b
"retry-with-reclaim" safety net; that code is correct for small transient
shortfalls between cleanup passes but cannot be the primary admission
control.

Out of scope for this entry: redesigning the prefix-cache eviction
priority during admission — current behaviour (evict when pressured) is
fine; the bug is that admission proceeds even when eviction cannot close
the gap.

## Rule

**Slot availability is not pool availability.** Any admission path that
gates solely on `free_slots()` or `max_running_requests` will overcommit
the KV pool under concurrent long-prompt workloads. Every future
scheduler change that touches admission must also verify that pool-token
budget is consulted — add a test that admits `N` requests with a pool
sized for `N-k` and asserts exactly `N-k` enter `Phase::Prefilling`.

For benchmarking: **before trusting a c=N headline, grep server stderr
for `pool alloc for paged prefill failed`**. If it fires even once during
the run, the numbers are contaminated — guidellm reports 0-ms TTFT for
the empty streams and will silently wash out p50 percentiles. The
canonical bench script should probably pipe the server log through a
filter that counts these and aborts the run; filed as a follow-up.
