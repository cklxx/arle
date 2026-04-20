# Retract victim ranking — sglang parity (ROI #3 from gap analysis)

> **Drift notice (added 2026-04-20):** absolute tok/s numbers cited here
> predate a `guidellm 0.6.0` env drift — today the same commits
> re-measure at ~98 tok/s. See
> [`errors/2026-04-20-bench-drift-environmental-not-code.md`](../errors/2026-04-20-bench-drift-environmental-not-code.md).
> The Pareto-neutral finding below remains accurate.

## Goal

Swap preemption victim ranking from single-key
`max_by_key(generated_tokens.len())` to the sglang-parity tuple
`max_by_key((generated_tokens.len(), -prompt_tokens.len()))` in both
the mixed-batch (`step_decode_launch_mixed`) and regular-decode
(`step_decode_launch`) preemption loops. Matches
`sglang/srt/managers/schedule_batch.py::retract_decode`.

## Design

Lexicographic tuple: prefer **max output**, then **min input**. Under
mixed-prompt-length workloads, this prefers the cheapest-to-re-prefill
victim. Under equal-prompt workloads (our current canonical
`guidellm` bench setup: all 4096-tok prompts) the tie-breaker is
dormant → behaviour matches previous. Zero-risk structural change.

## Results — c=16 × 4096 × 256, L4 24 GB, equal-length prompts

Three runs in sequence, fresh server per run (weights page-cached so
start-up was <30 s):

| metric | baseline | ROI#3 run 1 | ROI#3 run 2 | within noise |
|---|---|---|---|---|
| TTFT p50 (ms) | 5511 | 5408 | 5621 | ✓ ±3% |
| TTFT p99 (ms) | 16694 | 16437 | 16954 | ✓ ±3% |
| ITL p50 (ms) | 86.66 | 87.02 | 87.20 | ✓ ±1% |
| ITL p99 (ms) | 198.77 | 198.08 | 199.67 | ✓ ±1% |
| out tok/s | 122.86 | 123.55 | 121.48 | ✓ ±2% |

**Pareto-neutral on this workload** — all differences within bench
noise. The tie-breaker is dormant because every bench request has the
same 4096-tok prompt. **Latent improvement** under mixed-prompt
workloads: sglang's change halved preemption re-prefill cost in their
own bench under heavy prompt-length variance.

## Note: baseline drift

The baseline this entry compares to is **fresh** (post fresh
`target/` rebuild + fresh `guidellm 0.6.0` install), not the
`673b9e9` numbers in the multi-req-mixed K=2 cap=64 wins entry. The
`673b9e9` baseline showed TTFT p50=3307, ITL p99=113 under a slightly
different stack (older guidellm or earlier page-cache state). Under
the **current** fresh stack, baseline is TTFT p50=5511, ITL p99=199 —
worse across the board than the historical numbers. Investigating
that drift is out-of-scope for ROI #3; logged as follow-up.

## Rule

Match sglang's preemption heuristic by construction, not by
bench-tuned numbers. The tie-breaker only pays off under mixed-prompt
workloads, but correctness + parity means no regression under any
workload.

## Follow-ups

1. **Investigate post-merge baseline drift.** After fresh `target/`
   rebuild and `guidellm 0.6.0` reinstall, c=16 × 4096 bench numbers
   regressed across the board (TTFT p50 3307 → 5511, ITL p99 113 → 199,
   tok/s 128 → 123). Re-bench
   `673b9e9` exactly (same commit, fresh build, same guidellm version)
   to isolate whether the regression is: (a) guidellm 0.6.0 vs older,
   (b) page-cache warmup differences, (c) subtle side-effect of the
   merge-in of train/autograd work from main.
2. **ROI #2 — mixed CUDA graph + cap raise** (next lever per gap
   analysis). Dependent on this baseline drift being resolved first —
   otherwise the bench delta will be contaminated.

## Artefacts

- Baseline raw: `bench-output/2026-04-19-cuda-l4-infer-baseline-fresh-c16-c16/`
- ROI#3 run 1: `bench-output/2026-04-19-cuda-l4-infer-roi3-clean-c16-c16/`
- ROI#3 run 2: `bench-output/2026-04-19-cuda-l4-infer-roi3-v2-c16-c16/`
- sglang source: `schedule_batch.py::retract_decode:2138-2199`
- Gap analysis: `docs/research/2026-04-19-sglang-gap-analysis.md` gap #4

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8, guidellm 0.6.0
- **Commit:** to land as `feat(scheduler): sglang-parity retract victim ranking`
- **Feature set:** `cargo build --release -p infer`
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`, env `INFER_TRACE=1`.
