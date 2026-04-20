# ROI#2 Commit 1 — hoist mixed-forward allocations pre-capture — guidellm sweep, cuda-L4, 2026-04-20

## Context

- **Backend:** cuda
- **Model:** Qwen3-4B BF16
- **Hardware:** NVIDIA L4, 24 GB VRAM, CUDA 13.0 (driver `580.82.07`), sm_89
- **Commit:** `b8d1569` (branch `claude/c16-admission-gate-v2`); diff = +66 −17 in `infer/src/model/qwen3/batch_decode.rs`
- **Feature set:** `cargo build --release` (default `cuda` feature)
- **Non-default flags:** `--num-slots 16`
- **Server launch:** `./target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --num-slots 16`
- **Guidellm version:** `0.6.0`

## What changed

First step of the ROI#2 ladder (plan:
[`docs/plans/roi2-mixed-cuda-graph.md`](../../plans/roi2-mixed-cuda-graph.md)).
Hoist the two biggest per-tick allocations on the mixed
decode+prefill forward path so Commit 2 can wrap it in a captured
CUDA graph:

1. `MixedBatchBuffers::prefill_page_table_gpu: Vec<CudaSlice<i32>>` —
   pre-allocated, one slot per prefill req × `max_total_pages`. Replaces
   the per-tick `memcpy_stod(page_table_host)` that allocated a fresh
   `CudaSlice` on every mixed step (hoist audit item #7 — the single
   biggest per-tick alloc and the primary stream-capture blocker).
2. `MixedBatchBuffers::token_ids_scratch: Vec<i32>` — pre-allocated host
   scratch; replaces `Vec::with_capacity(total_tokens)` (hoist audit #3).

Both are semantics-preserving: buffers are reused rather than
re-allocated; byte layout, stream ordering, and kernel inputs are
identical. Mixed path stays eager at `MIXED_PREFILL_CAP = 64`. Remaining
hoist items (#1 eager `logits_batch`, #2 eager `mixed`, scheduler-side
S1–S4) deferred to Commit 1b if needed before Commit 2.

## Canonical params

```
guidellm benchmark --target http://localhost:8000 --model Qwen3-4B \
  --profile sweep --data prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 --random-seed 20260416 --backend openai_http \
  --processor infer/models/Qwen3-4B
```

## Results — sweep headline

### Commit 1 hoist (`b8d1569`)

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 746.1 | 757.8 | 35.39 | 35.46 | 26.5 | 0.1 |
| throughput | — | 33306.1 | — | 109.63 | **98.84** | 0.367 |
| 0.13 r/s | 2785.3 | 2788.9 | 40.03 | 40.47 | 31.29 | 0.117 |
| 0.17 r/s | 2857.9 | 2961.8 | 42.74 | 43.25 | 37.19 | 0.133 |
| 0.20 r/s | 2908.3 | 2949.9 | 45.16 | 45.64 | 43.52 | 0.167 |
| 0.23 r/s | 3011.0 | 9346.7 | 55.08 | 66.71 | 42.64 | 0.150 |
| 0.27 r/s | 4989.2 | 12011.9 | 61.66 | 67.50 | 43.73 | 0.183 |
| 0.30 r/s | 3078.0 | 17568.7 | 60.49 | 80.07 | 44.05 | 0.183 |
| 0.33 r/s | 3176.8 | 13725.9 | 58.58 | 86.24 | 53.07 | 0.233 |
| 0.37 r/s | 3559.4 | 8705.3 | 58.60 | 79.43 | 51.32 | 0.233 |

### Same-day side-by-side — HEAD (`17f58ac`) vs Commit 1 (`b8d1569`)

| metric | HEAD | Commit 1 | Δ% |
|---|---|---|---|
| sync TTFT p50 | 758.0 ms | 746.1 ms | **−1.6 %** |
| sync TTFT p99 | 774.1 ms | 757.8 ms | **−2.1 %** |
| sync ITL p99 | 35.52 ms | 35.46 ms | ~0 % |
| throughput TTFT p99 | 33688.6 ms | 33306.1 ms | **−1.1 %** |
| throughput ITL p99 | 110.29 ms | 109.63 ms | **−0.6 %** |
| throughput out tok/s | 98.17 | 98.84 | **+0.7 %** |
| throughput req/s actual | 0.367 | 0.367 | 0 % |

**Pareto-neutral within noise.** Commit 1 ships clean in the current
measurement environment.

## On the 128 → 98 tok/s perceived regression

Compared against the K=2 cap=64 historical win
([`2026-04-19-multi-req-mixed-prefill-k2-cap64.md`](2026-04-19-multi-req-mixed-prefill-k2-cap64.md)
at commit `78e1f8a`, cited 128 tok/s), the current absolute throughput
is ~25 % lower. Bisect under `errors/2026-04-20-bench-drift-environmental-not-code.md`
establishes this is **not a code regression** — the K=2 win commit itself
re-measured today produces 98 tok/s, identical to HEAD. Drift is
environmental (primary suspect: `guidellm 0.6.0` CLI / metric /
rate-computation changes vs the older guidellm used at the time of the
original win). Historical entries remain valid in their original
environments; current benches measure against a ~98 tok/s baseline until
the env is pinned.

## Artefacts

- `bench-output/2026-04-20-roi2-c1-hoist-only-run2/benchmarks.{json,csv,html}`
- HEAD baseline: `bench-output/2026-04-20-head-baseline/`
- Bisect 673b9e9: `bench-output/2026-04-20-bisect-673b9e9/`
- Bisect 78e1f8a: `bench-output/2026-04-20-bisect-78e1f8a/`

## Follow-ups

- **Env pin for bench stability** — record `guidellm==0.6.0`, CUDA
  driver, and `--num-slots` in `docs/plans/guidellm-integration.md` §3
  so future drift is intentional, not mystery.
- Commit 1b remaining hoist items (#1, #2, S1–S4) before Commit 2 graph
  capture.
- Regenerate `infer/test_data/Qwen3-4B.json` baseline (pre-existing
  e2e drift confirmed independent of Commit 1).
