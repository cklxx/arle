# CUDA L4 C16 SGLang Align R9 Decode-Active Budget

## Context

- Change under test: replace the admission-time choke with a single per-round
  prefill budget that matches SGLang's `PrefillAdder` shape more closely.
- Scope:
  `infer/src/scheduler/cuda/{core.rs,decode.rs,execution.rs,prefill.rs,runtime.rs}`.
- Goal: ensure decode-active rounds do not run one fused prefill batch and then
  a second oversized standalone prefill batch in the same scheduler tick.
- Benchmark mode:
  `scripts/bench_guidellm.sh cuda-l4-c16-sglang-align-r9-decode-active-budget --fast`.
- Artefacts:
  `bench-output/2026-04-20-cuda-l4-c16-sglang-align-r9-decode-active-budget/`.

## What Worked

- Decode-active standalone prefill now consumes the same token budget as the
  mixed path instead of silently falling back to the configured `4096` token
  standalone chunk.
- Same-tick double spending is gone: if the mixed decode launch already fused
  prefill work, Phase 2c does not enqueue a second standalone prefill batch.
- Scheduler naming is simpler and closer to SGLang's model:
  `PrefillBudget { rem_input_tokens, rem_chunk_tokens, rem_prefill_requests }`.
- Scheduler logs during the run confirmed the intended shape:
  - standalone prefill progressed as `64/4104` token chunks on decode-active
    rounds
  - mixed launches kept running with `Σc=64` instead of a second `4096` token
    standalone batch appearing later in the same tick

## Params

- GPU: NVIDIA L4
- Model: `models/Qwen3-4B`
- Server flags:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--chunked-prefill-size 4096`
  - `--max-prefill-tokens 16384`
  - `--prefill-max-requests 4`
  - `--enable-mixed-chunk=true`
  - `--mem-fraction-static 0.94`
  - `--cuda-graph=false`
- GuideLLM profile:
  - `profile=concurrent`
  - `rate=16`
  - `prompt_tokens=4096`
  - `output_tokens=256`
  - `max_seconds=30`
- Commit under test: `f0993d6`

## Results

### Headline

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| conc16 | 8190.7 | 18727.2 | 53.35 | 54.27 | 57.34 | 0.20 |

### Server Throughput

- Input tok/s mean: `2884.5`
- Output tok/s mean: `69.4`
- Total tok/s mean: `2953.9`

### Completion Volume

- Successful requests: `7`
- Cancelled requests after the 30s window: `45`
- Created requests: `52`

## Problems

- This is a scheduler-semantics cleanup, not a throughput win.
- Compared with the branch-local fast baseline, tail TTFT regressed badly even
  though ITL improved.
- The decode-active budget currently constrains progress too hard at c=16: the
  server remains structurally correct, but request turnover is still much lower
  than the best branch-local run.

## Learnings

- The old admission-time choke was the wrong place to encode safety, but the
  scheduler still needs a higher-throughput decode-active mixed strategy after
  the per-round budget is made correct.
- Matching SGLang's budget shape matters: one scheduler round should have one
  prefill budget, not a fused prefill batch plus a second oversized standalone
  batch.
- This change is worth keeping because it fixes the execution model and removes
  an incorrect project assumption ("decode active implies global chunk=64"),
  but it does not yet align end-to-end throughput with SGLang.

## Δ vs baseline

- Baseline:
  [2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md](2026-04-20-bench-guidellm-cuda-l4-c16-roi2-c2-fast-r4-4608-nograph.md)

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p99 (ms) | 4279.8 | 18727.2 | +337.6% |
| ITL p99 (ms) | 83.7 | 54.27 | -35.2% |
| out tok/s | 92.6 | 57.34 | -38.1% |

## Artefacts

- Raw:
  `bench-output/2026-04-20-cuda-l4-c16-sglang-align-r9-decode-active-budget/benchmarks.json`
- CSV:
  `bench-output/2026-04-20-cuda-l4-c16-sglang-align-r9-decode-active-budget/benchmarks.csv`
- HTML:
  `bench-output/2026-04-20-cuda-l4-c16-sglang-align-r9-decode-active-budget/benchmarks.html`

## Rule

- Keep the single per-round prefill budget model; do not reintroduce an
  admission-time choke or a second standalone prefill batch inside a decode
  round just to make the numbers look better.
