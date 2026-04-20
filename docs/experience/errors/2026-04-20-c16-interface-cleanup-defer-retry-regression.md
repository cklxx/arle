# Fast c=16 regression after interface cleanup + defer-on-pool-pressure

## Goal

- Type: `regression`
- Capture the first post-merge fast `guidellm` run after the paged-prefill
  interface cleanup and decode/prefill defer-on-pool-pressure changes, and
  compare it against the latest same-env fast baseline.

## Hypothesis

- Removing scheduler-built page-table plumbing should be neutral or slightly
  positive.
- Deferring decode/prefill on transient pool pressure should improve tail
  behavior without collapsing completed-request throughput.

## Command

Server:

```bash
./target/release/infer \
  --model-path /content/workspace/agent-infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --cuda-graph=false
```

Bench:

```bash
./scripts/bench_guidellm.sh cuda-l4-c16-interface-cleanup-defer-retry-r1 --fast
```

## Environment

- GPU: `NVIDIA L4, 23034 MiB`
- Driver: `580.82.07`
- CUDA toolkit: `nvcc` not present on PATH during capture
- Model: `Qwen/Qwen3-4B`
- Weights path: `/content/workspace/agent-infer/models/Qwen3-4B`
- Commit: `88060f4`
- Tree state: `dirty`
- Feature set: `cargo build -p infer --release`
- Output dir:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-interface-cleanup-defer-retry-r1`

## Results

### Raw headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc16 | 5822.7 | 6089.9 | 113.18 | 114.03 | 44.02 | 0 |

### Completed-request latency stats

| metric | p50 | p95 |
|---|---:|---:|
| request latency (s) | 34.7 | 34.7 |
| TTFT (ms) | 5822.7 | 6089.9 |
| ITL (ms) | 113.2 | 114.0 |
| TPOT (ms) | 135.5 | 135.5 |

### Server throughput stats

| concurrency mean | input tok/s | output tok/s | total tok/s |
|---:|---:|---:|---:|
| 16.0 | 4709.9 | 108.0 | 2194.3 |

### Artefacts

- JSON:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-interface-cleanup-defer-retry-r1/benchmarks.json`
  `sha256=79acb352992b6634dc6c1bb0d46416c26f2023b3b2d7af1ebabb27c2fb238a7d`
- CSV:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-interface-cleanup-defer-retry-r1/benchmarks.csv`
  `sha256=81a87060d73951c10c5ef14a9f054f714d7bf91e31e66d9e983cae6e1a89028c`
- HTML:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-interface-cleanup-defer-retry-r1/benchmarks.html`
  `sha256=dfdd299489db0ba334be32c9d96cdf313aa85d4139efeb6514e076f9e91d7b19`

## Problems

- This run missed the prior same-env fast baseline badly:

| metric | baseline (`r4-4608-nograph`) | now | Δ% |
|---|---:|---:|---:|
| TTFT p99 (ms) | 4279.8 | 6089.9 | +42.3% |
| ITL p99 (ms) | 83.7 | 114.03 | +36.2% |
| out tok/s | 92.6 | 44.02 | -52.5% |
| req/s actual | 0.333 | 0 | -100.0% |

- Scheduler logs showed repeated pool holds late in the run:
  `Request 16 held for pool (need=4360 tok, budget=34xx tok, radix_hit=0, reusable_prefix=0)`.
- The same run also emitted multiple immediate completions with zero output:
  `Request 16 done: 0 tokens`, then `Request 17..22 done: 0 tokens`.
- The server still logged CUDA graph capture during warmup even with
  `--cuda-graph=false`; mixed-graph warmup currently appears gated by a
  separate path.
- Because the tree was dirty and contained staged ROI#2 C2 work in addition to
  the local interface/defer changes, this run is diagnostic only, not a clean
  publishable optimization number.

## Learnings

- Deferring pool-pressure failures is not enough if the scheduler still admits
  or keeps requests in shapes that cannot complete inside the active budget.
- The c=16 failure mode is request-completion collapse, not only slower per-step
  kernels; planner-first admission and pre-launch budgeting need to move ahead
  of execution-time retry.
- Fast exploration runs are still valuable for regressions, but when
  `req/s actual` collapses to zero they must be recorded as diagnosis/error,
  not as baseline-quality wins.

## Delta vs baseline

- Baseline:
  [2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md](../wins/2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md)
- Result: hard regression; switched follow-up work from optimization to
  diagnosis + scheduler redesign.

## Rule

- When a c=16 run regresses with `done: 0 tokens` and `held for pool` in the
  same window, treat it as a scheduling/completion-path failure first; do not
  trust kernel-level micro-optimizations to recover the headline numbers.
