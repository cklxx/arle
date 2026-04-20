# Bench drift 128→98 tok/s on L4 c=16 × 4096×256 — environmental, not code

## Context

During ROI#2 Commit 1 validation (hoist mixed-forward allocations,
`b8d1569`), same-day side-by-side bench on L4 / Qwen3-4B BF16 / c=16 ×
4096-prompt × 256-output showed the diff was Pareto-neutral vs HEAD
(`17f58ac`): ~98 tok/s on both with all TTFT/ITL percentiles within ±2 %.

**But the absolute number was ~25 % below the K=2 cap=64 historical
"win" entry (`docs/experience/wins/2026-04-19-multi-req-mixed-prefill-k2-cap64.md`)
which cited 128 tok/s at commit `78e1f8a` on the same workload.** Under
user direction ("解决后再push吧" — resolve before push), bisect commissioned.

## Bisect outcome — drift is NOT from code

Three points measured **today** with fresh `cargo build --release`,
`--num-slots 16`, default flags, `guidellm 0.6.0` sweep profile, same
4096×256 prompt/output canonical params:

| commit | date of commit | throughput tok/s | TTFT p99 | ITL p99 |
|---|---|---|---|---|
| `78e1f8a` — K=2 win entry | 2026-04-19 | **98.31** | 33587 ms | 110 ms |
| `673b9e9` — main merge | 2026-04-19 | **98.94** | 33441 ms | 109 ms |
| `17f58ac` — HEAD pre-ROI#2 | 2026-04-19 | **98.17** | 33688 ms | 110 ms |
| `b8d1569` — ROI#2 C1 hoist | 2026-04-20 | **98.84** | 33306 ms | 110 ms |

All four land at 98–99 tok/s. **No commit in the range shows a drop**;
the K=2 entry's original commit itself now measures 98 — so the 128 →
98 gap is not a code regression.

## Root cause — environmental (guidellm / env drift)

- **Suspect 1 (primary): `guidellm` version change.** The K=2 win entry
  was captured with an older guidellm (exact version not pinned in the
  entry). Current environment has `guidellm 0.6.0`; the research doc
  (`docs/research/2026-04-19-sglang-gap-analysis.md:286-291`) already
  flagged a "fresh guidellm 0.6.0" drift. 0.6.0 shipped new default
  backend (`vllm_python` → required explicit `--backend openai_http`),
  new dataset deserializer routing, and new sweep-rate computation. A
  throughput bench under 0.6.0 measures different numbers from an
  earlier guidellm against identical server code.
- **Suspect 2 (secondary): machine-state drift.** CUDA driver
  `580.82.07`, CUDA 13.0, driver reload since the historical measurement,
  thermal state, PCIe link training — harder to attribute without a
  controlled reproduction.
- **Ruled out:**
  - Code changes in range `78e1f8a..17f58ac` (bisect above).
  - ROI#2 Commit 1 hoist (same-day side-by-side vs HEAD Pareto-neutral).

## Fix / decision

1. **Historical wins remain valid in their original measurement
   environment.** Don't edit past wins entries.
2. **Current baseline is 98 tok/s, not 128.** All ROI#2, Gap #5, and
   downstream optimization wins must cite same-env same-day Δ, not
   historical absolute.
3. **Push ROI#2 Commit 1** — it is Pareto-neutral in the current
   environment and unblocks Commit 2 (graph capture).
4. **Follow-up — pin guidellm version** in `docs/plans/guidellm-integration.md`
   §3 or via `requirements-bench.txt` so future drifts are visible as
   intentional env changes, not mystery regressions. Not blocking this
   commit.

## Rule

- **Absolute perf numbers only compare within one measurement
  environment.** When an apparent regression surfaces, first re-measure
  the historical reference commit in the current environment; if both
  move together, the drift is environmental and the fix is env pinning,
  not code rollback.
- **`--num-slots`, guidellm version, CUDA driver, `mem_fraction_static`,
  server build features** are all part of the measurement environment.
  Any change to these invalidates absolute cross-run comparisons.

## Artefacts

- `bench-output/2026-04-20-bisect-673b9e9/` — 673b9e9 re-measured today.
- `bench-output/2026-04-20-bisect-78e1f8a/` — 78e1f8a re-measured today.
- `bench-output/2026-04-20-head-baseline/` — 17f58ac re-measured today.
- `bench-output/2026-04-20-roi2-c1-hoist-only-run2/` — ROI#2 C1 bench.
