# guidellm fast recovery cuda-l4-c16-admission-prefill-plan-r1

## Goal

- Type: `diagnosis`
- Measure whether the first admission-planner cut recovers the c=16 fast-run
  regression after the paged-prefill interface cleanup.

## Hypothesis

- Limiting cold-start admissions by per-tick prefill work, not just pool
  budget, should pull TTFT/ITL back toward the last good fast baseline.
- This cut may trade some completed-request throughput for better tail latency;
  if so, it is only a partial recovery.

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
./scripts/bench_guidellm.sh cuda-l4-c16-admission-prefill-plan-r1 --fast
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
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-admission-prefill-plan-r1`

## Results

### Headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc16 | 3233.8 | 4233.4 | 77.49 | 81.28 | 46.14 | 0.133 |

### Completed-request latency stats

| metric | p50 | p95 |
|---|---:|---:|
| request latency (s) | 23.1 | 24.0 |
| TTFT (ms) | 3233.8 | 4233.4 |
| ITL (ms) | 77.5 | 81.3 |
| TPOT (ms) | 90.2 | 93.7 |

### Server throughput stats

| concurrency mean | input tok/s | output tok/s | total tok/s |
|---:|---:|---:|---:|
| 16.0 | 2899.8 | 63.7 | 2963.5 |

### Artefacts

- JSON:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-admission-prefill-plan-r1/benchmarks.json`
  `sha256=96b55a1f524886c58a120564914849fbfb53db73e4524024dd1f92d104ce4041`
- CSV:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-admission-prefill-plan-r1/benchmarks.csv`
  `sha256=c65e72dce09511244f4f8c1919be14ec3f35aa61f54850d67dab137897667a04`
- HTML:
  `/content/workspace/agent-infer/bench-output/2026-04-20-cuda-l4-c16-admission-prefill-plan-r1/benchmarks.html`
  `sha256=ce48963b57b7a57f18a311f6e332276c91c7a240c11129d4c8e8be22d458079f`

## Problems

- This run recovered latency but not headline completed-request throughput.
- Late in the run the scheduler still emitted `done: 0 tokens` for tail
  requests admitted after the main completed cohort drained.
- During follow-up validation, a separate decode-path bug was confirmed:
  `--cuda-graph=false` still allowed Qwen3 decode graph capture in the model.
  That fix landed after this bench and needs its own rerun.

## Learnings

- Admission pressure needs two budgets: total pool tokens and per-tick prefill
  work. Pool-only admission lets cold-start prefills flood the active set and
  explodes TTFT.
- This first planner cut is enough to recover tail latency quickly, but it does
  not by itself restore c=16 completed-request throughput.
- `--cuda-graph=false` must be enforced inside the model decode path, not only
  at scheduler warmup/CLI wiring.

## Δ vs baseline

### Versus the immediate regression run

- Regression:
  [2026-04-20-c16-interface-cleanup-defer-retry-regression.md](../errors/2026-04-20-c16-interface-cleanup-defer-retry-regression.md)

| metric | regression | now | Δ% |
|---|---:|---:|---:|
| TTFT p99 (ms) | 6089.9 | 4233.4 | -30.5% |
| ITL p99 (ms) | 114.03 | 81.28 | -28.7% |
| out tok/s | 44.02 | 46.14 | +4.8% |

### Versus the last fast baseline

- Baseline:
  [2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md](2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md)

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p99 (ms) | 4279.8 | 4233.4 | -1.1% |
| ITL p99 (ms) | 83.7 | 81.28 | -2.9% |
| out tok/s | 92.6 | 46.14 | -50.2% |

## What Worked

- The admission planner cut prevented the active prefill set from exploding at
  cold start and pulled latency back to the prior envelope.
- The new `assign_slots_limits_cold_prefill_to_tick_plan` runtime test locks in
  the intended admission behavior.

## Rule

- When c=16 regresses, recover latency first by constraining tick-level prefill
  work, then attack throughput separately. Treat those as different scheduler
  problems, not one blended metric.
