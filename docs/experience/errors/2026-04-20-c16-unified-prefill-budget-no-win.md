# C16 unified prefill budget cleanup did not fix TTFT

## Context

- Branch: `claude/c16-admission-gate-v2`
- Post-merge refactor scope:
  - `infer/src/scheduler/cuda/{core.rs,decode.rs,execution.rs,prefill.rs,runtime.rs}`
  - `infer/src/model.rs`
  - `infer/src/model/qwen3/{batch_decode.rs,forward.rs}`
  - `crates/cuda-kernels/src/paged_kv.rs`
- Goal: make CUDA c=16 scheduling coherent after the `origin/main` merge by:
  - removing admission-time prefill-lane choke
  - making pool planning page-aware
  - making execution obey launch-time prefill budgets instead of queueing extra standalone work
  - cleaning `--cuda-graph=false` semantics

## Root Cause

- The merge-era regression had **two distinct problems**, not one:
  - planning used transferable-token math where page availability was the real constraint, causing `alloc_tokens(...)` drift after a "successful" plan
  - execution still had **independent prefill pacing logic**; `decode.rs` planned one shape, then `execution.rs` queued extra standalone prefill work on top
- The page-aware planning fix removed the first failure mode.
- The execution cleanup made the scheduler more internally consistent, but it **did not change the fundamental cold-start shape** enough to close TTFT:
  - large 4k prompts still enter a long prefill staircase before the system reaches a steady decode-heavy regime
  - without true multi-request prefill fusion on the idle path, scheduler cleanup alone cannot match sglang's TTFT behavior

## Fix

- Kept:
  - page-aware pool planning (`required_tokens` + `required_pages`)
  - launch-plan structs consumed directly by decode/mixed/prefill launchers
  - lazy materialization of `Phase::New` into `Phase::Prefilling` only when prefill budget is actually spent
  - unified execution rule: when decode is active, standalone prefill only consumes the residual per-tick budget after mixed prefill
- Reverted:
  - an `idle_prefill_budget_multiplier=2` attempt; it reduced completed-request throughput without improving TTFT

## Evidence

- Diagnostic failure before the page-aware fix:
  - `cuda-l4-c16-main-merge-refactor-r1`
  - repeated `Decode plan allocation drifted after planning ... out of pages`
  - no usable artefacts
- Final retained code bench:
  - label: `cuda-l4-c16-unified-prefill-budget-r4`
  - artefacts: `bench-output/2026-04-20-cuda-l4-c16-unified-prefill-budget-r4/`
  - headline:
    - `TTFT p50 13403.2 ms`
    - `TTFT p99 13750.6 ms`
    - `ITL p50 91.27 ms`
    - `ITL p99 92.62 ms`
    - `out tok/s 108.39`
- Reverted bad tuning probe:
  - label: `cuda-l4-c16-unified-prefill-budget-r3`
  - artefacts: `bench-output/2026-04-20-cuda-l4-c16-unified-prefill-budget-r3/`
  - headline:
    - `TTFT p99 13848.8 ms`
    - `out tok/s 75.81`
- Last known better TTFT reference on this branch family:
  - `docs/experience/wins/2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md`
  - `TTFT p99 4279.8 ms`
  - `ITL p99 83.7 ms`
  - `out tok/s 92.6`

## Rule

- **Do not treat scheduler unification as a proxy for prefill fusion.**
- When c=16 regressions are dominated by long-prompt cold start:
  - first fix planner/execution drift so the system is coherent
  - then measure
  - if TTFT is still multi-second, move to a true architectural lever:
    - multi-request prefill fusion outside decode-only steady state
    - or a fused idle-path prefill batch
- Do not keep tuning-only knobs in tree when the bench says they are neutral or negative.
