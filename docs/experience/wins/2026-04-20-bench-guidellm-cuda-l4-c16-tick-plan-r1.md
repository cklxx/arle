# CUDA L4 C16 Tick Plan R1 Fast Bench

## Context

- Change under test: first scheduler-side planner-first slice for CUDA pool pressure.
- Scope: `infer/src/scheduler/cuda/{core.rs,decode.rs,prefill.rs,runtime.rs}`.
- Goal: move decode/mixed/prefill pool reclaim decisions out of ad hoc launch-time fallback and into explicit plans that launchers consume.
- Benchmark mode: `scripts/bench_guidellm.sh cuda-l4-c16-tick-plan-r1 --fast`.
- Artefacts: `bench-output/2026-04-20-cuda-l4-c16-tick-plan-r1/`.

## What Worked

- The scheduler now builds explicit pool plans before:
  - plain decode launch
  - mixed decode + prefill launch
  - standalone prefill chunk execution
- Launchers consume those plans with direct `alloc_tokens(...)` calls instead of making fresh retry/retract decisions in the hot path.
- The new helper `plan_pool_capacity(...)` successfully retracts non-protected victims up front and is covered by a focused unit test.
- Fast-bench headline improved versus the prior exploration run `cuda-l4-c16-interface-cleanup-defer-retry-r1`:
  - `TTFT p99`: `6089.9 ms -> 4237.9 ms`
  - `ITL p99`: `114.03 ms -> 82.06 ms`
  - `out tok/s`: `44.02 -> 45.72`

## Params

- GPU: NVIDIA L4
- Model: `models/Qwen3-4B`
- Server flags:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--cuda-graph=false`
- GuideLLM profile:
  - `profile=concurrent`
  - `rate=16`
  - `prompt_tokens=4096`
  - `output_tokens=256`
  - `max_seconds=30`

## Results

### Headline

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| conc16 | 3230.9 | 4237.9 | 78.22 | 82.06 | 45.72 | 0.133 |

### Server Throughput

- Requests/sec mean: `0.1`
- Input tok/s mean: `2872.0`
- Output tok/s mean: `63.0`
- Total tok/s mean: `2935.1`

### Completion Volume

- Successful requests: `4`
- Cancelled requests after the 30s window: `46`
- Median request concurrency stayed at `16.0`, so the run remained fully saturated.

## Problems

- This planner-first slice is too conservative at c=16.
- The new `tick prefill plan` gate held admissions repeatedly while only a small number of requests were allowed to advance prefill, so the benchmark spent most of the 30s window with `0` completed requests.
- The tail metrics improved only because fewer requests made forward progress at once; throughput is still far from the sglang target.

## Learnings

- Pre-launch pool planning is directionally correct, but coupling it to a hard per-tick prefill lane budget at admission is too blunt.
- The next cut should keep the launch-time planning structure and remove or relax the admission-side `tick prefill plan` gate.
- The better place to enforce pool safety is the per-launch planner for decode/mixed/prefill, not a global cold-admission choke point that blocks request turnover.

## Rule

- Keep planner-first pool decisions in the scheduler, but do not let a coarse admission-time prefill budget dominate throughput at c=16.
