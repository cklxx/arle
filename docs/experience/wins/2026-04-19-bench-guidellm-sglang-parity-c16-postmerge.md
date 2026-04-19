# c=16 parity snapshot — Qwen3-4B on L4, post-merge + correctness fixes

## Goal

Establish an apples-to-apples c=16 × 4096-prompt × 256-output baseline
after merging origin/main into the c=16 parity branch, quantify the
remaining gap vs sglang 0.5.10, and close two known correctness bugs
surfaced by the investigation.

Supersedes: [`2026-04-18-sglang-parity-c16-c8.md`](2026-04-18-sglang-parity-c16-c8.md)
(c=16 contaminated there; c=8 reference remains valid).

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8, guidellm 0.6.0, sglang 0.5.10.post1
- **Commit:** `e1db13f` (merge of origin/main into `claude/c16-admission-gate-v2`);
  infer code fixes on top are `rotating page-locked pool` +
  `honour --cuda-graph=false` (see §"What changed").
- **Feature set:** `cargo build --release -p infer` (default features)
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`
- **Server launch:** direct target/release/infer binary

## Canonical params (diverges from sweep — intentional)

```
guidellm benchmark run \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --processor /content/workspace/agent-infer/models/Qwen3-4B \
  --profile concurrent --rate 16 \
  --data prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 --random-seed 20260416 \
  --output-dir bench-output/<date>-<label>/ \
  --outputs json,csv,html \
  --backend openai_http \
  --backend-kwargs '{"validate_backend": "/v1/models"}'
```

Non-canonical: fixed-concurrency c=16 instead of `sweep`. Matches
the apples-to-apples comparison the user is tracking. `validate_backend`
override is required because guidellm 0.6.0 defaults to `GET /health`
which infer's HTTP server does not implement (endpoint follow-up below).

## Results — c=16 headline

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | successful | incomplete |
|---|---|---|---|---|---|---|---|
| **sglang 0.5.10** | 5696 | **10727** | **92** | **113** | **140** | **32** | 16 |
| infer e1db13f post-merge | 5066 | 15199 | 102 | 221 | 101 | 24 | 16 |
| infer + rotating page-locked pool | 5371 | 15512 | 102 | 220 | 101 | 24 | 16 |
| infer + cuda-graph=false honoured | 5374 | 15427 | 100 | 218 | 102 | 24 | 16 |

All three infer runs fall within ±1% on out tok/s. **Neither code change
moved the needle** — the sync hack and the graph-capture memory were
not the bottleneck.

## Artefacts

- Raw baseline: `bench-output/2026-04-19-cuda-l4-infer-e1db13f-c16/`
- Rotpool:      `bench-output/2026-04-19-cuda-l4-infer-rotpool-c16/`
- Nograph:      `bench-output/2026-04-19-cuda-l4-infer-nograph-c16/`
- sglang ref:   `bench-output/2026-04-19-cuda-l4-sglang-c16/`

## Delta vs sglang

| metric | sglang | infer (nograph) | Δ% (infer) |
|---|---|---|---|
| TTFT p50 | 5696 | 5374 | **−6%** (infer faster) |
| TTFT p99 | 10727 | 15427 | +44% |
| ITL p50 | 92 | 100 | +9% |
| ITL p99 | 113 | 218 | +93% |
| out tok/s | 140 | 102 | **−27%** |
| successful | 32 | 24 | **−25%** |

99% parity target = `out tok/s ≥ 138.6` and `successful ≥ 31.7`. Gap
on both remains ~27%.

## What changed in the code since baseline

1. **`crates/cuda-kernels/src/flashinfer.rs`** — `FlashInferWorkspace`
   now owns a rotating pool of 8 host-pinned buffers for the
   `page_locked_workspace` slot instead of a single 8 MiB buffer. Each
   plan call (`plan_hd128` / `plan_hd256`) picks the next buffer via a
   per-workspace `Cell<usize>` counter. Rationale: the prior single
   buffer forced a `ctx.stream.synchronize()` before every plan call to
   avoid CPU-side overwrite during the in-flight async memcpy to
   `int_workspace`. Rotating N buffers removes the race without the
   drain. **The drain was not the dominant TTFT bottleneck at c=16.**

2. **`infer/src/ops/attention.rs`** — deletes the 26-line
   `ctx.stream.synchronize()` comment + call in
   `PagedPrefillForward::new_inner`. Replaced with a one-line
   comment pointing at the rotating pool.

3. **`infer/src/model/qwen3/forward.rs`**, **qwen35/forward.rs**,
   **glm4/forward.rs** — `supports_cuda_graph_decode()` now ANDs the
   model's `enable_cuda_graph` flag (previously only Qwen3 checked
   `lora.is_none()`; Qwen35/GLM4 hard-coded `true`). Without this
   fix, `--cuda-graph=false` was silently ignored at warmup and CUDA
   graphs were captured anyway, wasting ~1–2 GB of VRAM that post-pool
   runtime needs. The **flag now works as documented**, but because
   the KV pool is sized from `free_mem - headroom` at startup (before
   graph capture), honouring the flag does NOT retroactively grow the
   pool. It just frees post-pool VRAM for other runtime allocations.

## Problems observed

1. **c=16 × 4096 is physically over-committed on L4.** 16 slots ×
   ~256 pages per fully-prefilled slot = 4096 pages, pool caps at 3751
   (8.8 GB at `mem-fraction-static=0.94`). Tried pushing
   `--mem-fraction-static 0.98` → pool grew to 4152 pages (+10%) but
   every prefill hit `CUDA_ERROR_OUT_OF_MEMORY` on the first chunk alloc
   because headroom dropped below the FlashInfer workspace + per-request
   buffer demand. **Pool cannot grow past ~8.8 GB without breaking the
   runtime.** sglang survives the same workload because its per-tick
   allocation model retracts/preempts more aggressively — both engines
   report 16 incomplete, but sglang keeps the machine moving and
   completes 32 vs our 24 in the 60 s window.

2. **Admission gate holds but retract can't rescue at prefill
   boundaries.** Logs show "Request 16 held for pool
   (need=4360 tok, budget=0 tok)" for ~3 s while running slots
   chunk-allocate to exhaustion, then "pool alloc for paged prefill
   failed: retracting 0 decode tokens" — the retract heuristic only
   considers `Phase::Decoding` candidates, but at c=16 × 4096 the
   window between admit-and-prefill keeps most slots in
   `Phase::Prefilling`, leaving zero valid victims. This matches the
   14-residual-failures note from the prior session.

3. **The paged mixed-batch kernel
   (`decode_batch_with_prefill`) is structurally dead code for
   Qwen3 under paged-prefill.** Verified in the prior session
   (reverted then); confirmed again by this session's research.
   Closing the ITL p99 +93% and the throughput −27% gap almost
   certainly requires a `decode_batch_with_paged_prefill` variant
   that reads K/V from the page pool instead of `k_cache[layer]`.
   This is the L2 work.

4. **guidellm 0.6.0 default validate is `GET /health`**, which
   infer does not implement. Every bench against this version must
   pass `--backend-kwargs '{"validate_backend": "/v1/models"}'`.
   Follow-up: add a tiny `/health` route to
   `infer/src/http_server/` so canonical bench invocation works
   without the override.

## Learnings

1. **Sync-hack-elimination is correctness, not perf.** The
   `stream.synchronize()` in `new_inner` was a real latent-race
   guard, but rotating plan buffers also fully cover that race
   without per-call cost. Tests pass (`paged_prefill_parity` bit-
   identical) and no `gemm_cuda` errors surfaced in the 60 s
   bench. Keep the change for the code-quality win; don't claim
   throughput.

2. **`--cuda-graph=false` was silently broken** for every model
   except Qwen3 (even Qwen3 only partially via the LoRA-gate). The
   flag is advertised but wasn't plumbed into the warmup decision.
   Now it is, on all three models.

3. **The c=16 parity target at 4096-prompt needs a kernel-level
   fix, not a scheduler tweak.** Three scheduler-level attempts
   this session (rotpool, cuda-graph disable, mem-fraction bump)
   produced zero (or negative) throughput movement. The next
   genuine lever is mixed-batch fusion with paged-pool K/V writes.

## Rule

For `c=16 × 4096 × 256` on L4 24 GB, **do not bump
`mem-fraction-static` above 0.94 with graphs disabled** — pool
grows but runtime OOMs. Measure the runtime headroom before
touching that knob.

Any sync-hack elimination that is motivated by perf must be
bench-validated. If the hack was not the bottleneck, keep the
change only if it's correct-on-its-own; don't overlook the
possibility that the hack was cheap.

Every new bench invocation pattern needs the
`--backend-kwargs '{"validate_backend": "/v1/models"}'` override
until the `/health` route lands — otherwise guidellm 0.6.0
validates with `GET /health` → 404 → silent validation failure.

## Follow-ups

- **L2 mixed-batch paged kernel** — `decode_batch_with_paged_prefill`
  variant. Substantial kernel work. Owner: next session.
- **Retract heuristic: prefilling-slot candidates** when no decode
  victims exist — investigate sglang's
  `ScheduleBatch::retract_decode` for whether they retract Prefilling
  slots.
- **`/health` route** in `infer/src/http_server/` so guidellm 0.6.0
  validates without override.
- **Reopen Plan doc `docs/plans/p99-unified-mixed-batch.md`** to
  fold in the paged-KV rewrite requirement.
