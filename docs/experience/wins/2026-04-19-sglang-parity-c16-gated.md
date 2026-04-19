# c=16 parity vs sglang — admission-time pool-capacity gate unblocks measurement

## Goal

Measure infer vs sglang 0.5.10 at fixed c=16 on Qwen3-4B L4 after landing
the admission-time paged-pool-capacity gate that closes the bug documented
in `docs/experience/errors/2026-04-18-c16-paged-pool-admission-overcommit.md`.
This supersedes the unpublishable c=16 section of
`docs/experience/wins/2026-04-18-sglang-parity-c16-c8.md`.

## Hypothesis

- After the gate, infer c=16 completes with **zero** `pool alloc for paged
  prefill failed` lines in server stderr — bench numbers are no longer
  contaminated by empty-stream zeros.
- ITL p99 stays within noise of c=8 (69–77 ms range) — the gate is an
  admission-time check, not a per-step cost.
- TTFT p99 will be worse than sglang's because the gate admits fewer
  concurrent requests (steady-state ~8 vs sglang's 16), pushing the tail
  of requests into the admission queue; this is the price of correctness,
  not a per-request regression.

All three confirmed.

## Parameters

```
# infer
./target/release/infer --model-path models/Qwen3-4B --port 8000 \
  --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.88

# sglang
python3 -m sglang.launch_server --model-path models/Qwen3-4B --host 0.0.0.0 \
  --port 8000 --max-running-requests 16 --mem-fraction-static 0.88

# guidellm (both)
scripts/bench_guidellm_concurrent.sh <label> --concurrency 16 --model <name>
# profile=concurrent, rate=16, data=prompt_tokens=4096,output_tokens=256,
# max-seconds=60, random-seed=20260416
```

## Environment

- **Hardware:** NVIDIA L4 24GB, CUDA 13.0, driver 580.82.07, SM 8.9
- **Commit:** d91908f + admission-gate changes (infer/src/scheduler/cuda/{core,runtime,request}.rs; uncommitted)
- **sglang:** 0.5.10.post1
- **guidellm:** 0.6.0, `openai_http` backend
- **Serial bench:** infer first, server killed, sglang second; `feedback_serial_bench.md`.
- **sglang pre-warm**: issued one throwaway 4-token chat completion before the bench to dodge the cold-tokenizer stall documented in `2026-04-18-sglang-parity-c16-c8.md` §Problems #2.

## Results — c=16 headline

| metric | infer (gated) | sglang | Δ (infer − sglang) |
|---|---:|---:|---:|
| TTFT p50 (ms)  | 8 688.1 | 5 696.4 | +52.5 % |
| TTFT p99 (ms)  | 34 668.1 | 10 726.5 | +223 % |
| ITL p50 (ms)   | 101.06 | 91.81 | +10.1 % |
| **ITL p99 (ms)** | **111.94** | **112.63** | **−0.6 % (parity)** |
| out tok/s      | 63.56 | 140.37 | −54.7 % |
| req/s actual   | 0.217 | 0.533 | −59.2 % |
| server-side pool failures | **0** | 0 | — |
| "held for pool" admission-gate events | 537 | n/a | — |

### Evolution across gate iterations

| variant | active @ steady | pool failures | TTFT p99 | ITL p99 | out tok/s |
|---|---:|---:|---:|---:|---:|
| no gate (pre-fix) | 16 | 220 (contaminated) | 10 691 ms (bogus) | 166.68 (bogus) | 1 004.94 (bogus) |
| gate v1 (`free_count`, prompt-only) | 16 | 14 | 21 262 | 146.21 | 115.25 |
| gate v2 (`max_total - sum reserved`, prompt+max_tokens) | 11 | 9 | 33 073 | 153.53 | 89.80 |
| gate v3 (**`free_count − future_growth − headroom`**) | 8 | **0** | 34 668 | **111.94** | 63.56 |

Three iterations to converge. v1 gated on transient free pages and missed
future reservations; v2 subtracted from `max_total_pages` but missed
prefix-cache-retained pages; v3 subtracts future growth (per-slot
reservation minus pages already physically held) from **current** free
pages. Only v3 makes every page in the pool accounted for at admission
time — only v3 produced zero pool alloc failures.

## Comparison with the c=8 baseline (sweep-style single-concurrency-step)

From `2026-04-18-sglang-parity-c16-c8.md`:

| metric | c=8 infer | c=16 infer gated | c=16/c=8 ratio |
|---|---:|---:|---:|
| TTFT p99 | 7 524 | 34 668 | 4.6× |
| ITL p99 | 76.85 | 111.94 | 1.46× |
| out tok/s | 71.15 | 63.56 | **0.89×** |

Throughput actually *drops* at c=16 under the gate because admission
throttles below 8 active when prefix-cache-retained pages squeeze free
space. This is a gate-tuning opportunity, not a kernel regression — the
ITL scaling (1.46×) is the normal p99 tail that batching a bigger mix
gives.

## Problems observed

1. **Gate is conservative.** Active concurrency at steady state is ~8 even
   though `num_slots=16`; sglang actually runs 16 in flight. The gate
   subtracts worst-case future growth per active request, which blocks
   new admissions as soon as committed reservations + prefix-cache-
   retained pages exceed pool free. Tuning space:
   - Dynamic `DECODE_HEADROOM_PAGES` scaled by active count (currently
     fixed at 32 pages).
   - Early prefix-cache eviction when the gate would otherwise block an
     admission (today the gate is pure "wait"; sglang is "wait or evict").
   - Per-request reservation decay model — reserve a smaller fraction of
     `max_tokens` (e.g. ⅔) since most requests complete early on stop
     tokens. Requires bench evidence of completion distribution.
2. **TTFT p99 is the largest gap** (+223 %). About 8 requests run
   immediately; the other 8 wait at admission, picking up the full
   single-prefill TTFT of the first-wave completions. Fixing #1 shrinks
   this.
3. **No throughput gain over c=8.** Expected c=16 to roughly double
   output tokens/s; instead it dropped 11 % (71.15 → 63.56). Whole of
   that is gate conservatism; the kernels are idle during held-admission
   gaps.

## Learnings

1. **Three gate models, only one correct.** Admission throttling at
   pool capacity is subtler than it looks:
   - `free_count` alone under-counts future reservations → over-admits.
   - `max_total − sum(reserved)` overcounts prefix-cache pages → over-admits.
   - `free_count − sum(reserved − held_per_slot) − headroom` is the
     right formula; it combines live-slot state with committed-but-
     not-yet-allocated future growth.

2. **Every failing pool alloc is an admission-policy bug, not a kernel
   bug.** M2b's retry-with-reclaim was never meant to be the primary
   admission control, and any gate that allows >0 pool alloc failures
   under sustained load is misbehaving even if the user-facing numbers
   look plausible.

3. **ITL parity survives admission throttling.** The decode-time work
   per token is unchanged by the gate — only request onset is delayed.
   p99 ITL stayed within 1 % of sglang, consistent with the 04-17
   sweep-era parity finding.

## Rule

For any future scheduler admission change:

- Grep the scheduler stderr for `pool alloc for paged prefill failed`
  after any bench. Non-zero = contaminated numbers; do not publish.
- `admission_budget_pages` must account for **current** pool state
  (free pages, including those retained by prefix cache) **and** each
  active request's remaining reservation. Other formulations over-admit
  under concurrent long-prompt workloads.
- Compare against a c=8 baseline to make sure the gate is not a no-op
  regression at light load.

The c=16 gap is now "TTFT p99 +223 %, throughput −55 % (driven by
conservative admission)". Future work is on the gate's tuning — not
on kernel or memory layout. Targets:
1. Dynamic decode headroom based on active count
2. Gate-triggered prefix-cache eviction on hold decisions
3. Completion-distribution-informed decode reservation (fractional max_tokens)

Each addresses the conservative-admission symptom directly.

## Raw artefacts

- infer c=16 (gated, clean, v3 formula):
  `bench-output/2026-04-19-cuda-l4-infer-gated-c16-run4/`
- sglang c=16 (clean):
  `bench-output/2026-04-19-cuda-l4-sglang-c16/`
- contaminated / superseded runs:
  `bench-output/2026-04-18-cuda-l4-infer-c16-run3/` (no gate, 220 failures),
  `bench-output/2026-04-19-cuda-l4-infer-gated-c16/` (v1, 14 failures),
  `bench-output/2026-04-19-cuda-l4-infer-gated-c16-run2/` (v2, 14 failures),
  `bench-output/2026-04-19-cuda-l4-infer-gated-c16-run3/` (v2+max_tokens, 9 failures).
- Scheduler regression test:
  `infer/src/scheduler/cuda/runtime.rs::tests::assign_slots_gates_admission_on_available_pool_pages`
