# ROI #2 — Mixed CUDA graph replay + raise `MIXED_PREFILL_CAP`

**Branch:** `claude/c16-admission-gate-v2` · **Target:** L4 24 GB · **Workload:** Qwen3-4B BF16 · c=16 × 4096-prompt × 256-output · **Expected delta:** TTFT p99 −2…−4 s, out tok/s +3…+5 %

## Goal

Capture the mixed decode + multi-prefill forward step in CUDA graphs so kernel-launch overhead amortises across replays. Once replay is healthy, raise `MIXED_PREFILL_CAP` from 64 → 256 (bench 128, 256, optional 512) without paying per-tick launch cost proportional to payload size. The K=3 and K=2 cap=256 probes (`docs/research/2026-04-19-sglang-gap-analysis.md:256-299`) confirmed the mixed kernel shape is the binding constraint after ROI #1 + #3 landed; ROI #2 is the sole remaining Pareto lever.

## File blast radius

### Primary (kernel shape + graph cache)

- `infer/src/model/qwen3/batch_decode.rs`
  - `MIXED_PREFILL_CAP` (`:29`), `MIXED_PREFILL_MAX_REQS` (`:34`) — constants feeding `BatchDecodeBuffers::new` sizing at `:222-225`.
  - `MixedBatchBuffers` (`:96-113`) + `new()` (`:117-147`): `max_tokens = max_batch_size + MIXED_PREFILL_CAP` at `:122`. Raising the cap bumps this allocation.
  - `BatchDecodeBuffers::new` (`:171-237`) owns `graph_cache: Vec<Option<CudaGraph>>` indexed `[batch_size-1]` at `:234`. Generalise to `(batch_size, num_tokens)`.
  - `BatchDecodeBuffers::invalidate_graph_cache` (`:322-326`) — key widening.
  - `Qwen3Model::decode_batch_with_prefills` (`:345-805`) — currently eager; whole body is the graph candidate.
  - `Qwen3Model::decode_batch` graph capture path (`:904-940`) — template to mirror for mixed.
- `infer/src/scheduler/cuda/decode.rs`
  - `MIXED_PREFILL_CAP` (`:15`), `MIXED_PREFILL_MAX_REQS` (`:18`) — authoritative policy knob; add `const _: () = assert!(cap_scheduler == cap_model)` coupling.
  - `step_decode_launch_mixed` (`:49-463`) — chunk plan at `:104-156`. After cap raise, per-req chunk `max(16, cap/K)` (`:105`), rounded mod 16 (`:106`). Add padding of `Σc_i` up to nearest captured `num_tokens` bucket.
- `infer/src/model/qwen3/forward.rs`
  - `forward_mixed_batch` (`:454-474`) — passthrough; no change.
  - `supports_mixed_batch` (`:435-442`) — still gates LoRA out; unchanged.
- `infer/src/scheduler/cuda/core.rs`
  - `warmup_cuda_graphs` (`:1113-1239`) + `warmup_graphs_pass` (`:1241-1316`) — add sibling `warmup_graphs_mixed_pass` iterating `(bs, nt)` buckets with dummy prefill sections.
  - `cuda_graph_batch_sizes` (`:1327-1345`) — reused; add companion `cuda_graph_mixed_num_tokens()`.

### Secondary (workspace sizing)

- `crates/cuda-kernels/src/flashinfer.rs`
  - `FlashInferDecodeMetadata::new` (`:751-797`) — already sized at `max_batch + MIXED_PREFILL_CAP + MIXED_PREFILL_MAX_REQS`. Cap raise increases `qo_indptr`/`positions`/`kv_last_page_len`/`kv_indices` at startup (fine).
  - `update_mixed_multi` (`:937-1040`) — `kv_indices` realloc branch at `:998-1006`. **Must be defanged** on mixed-graph path by oversizing `max_total_pages` in `BatchDecodeBuffers::new` (see R4).
  - `tc_plan` (`:1112-1137`) — runs pre-capture, exactly like decode (`batch_decode.rs:905-907` pattern). Already positioned correctly at `:453-460`.

### Ops callees — alloc audit

- `infer/src/ops/linear.rs:gemm_into` (`:385`)
  - Marlin (`:411-578`) — allocs `x_fp16`/`y_fp16`/`workspace`; gated `seq_len>1 && weight.has_marlin()`. Mixed path's `KVFormat::BF16` gate (`batch_decode.rs:71`) means Qwen3 BF16 weights → not Marlin. Safe.
  - QxK (`:614-619`) — dequant workspace alloc; gated `b>8 && is_qxk`. Mixed tick has `total_tokens=B+Σc=80>8` but BF16 KV gate means `is_quantized()=false`. Safe.
  - Bare cuBLAS path (`:564-573`) — no alloc. This is what graph body sees.
- `infer/src/ops/{elementwise,norm,embedding}.rs` — all `_into` variants; no per-call alloc.

### Tests + harness

- `infer/tests/e2e.rs` — bit-identity against `infer/test_data/Qwen3-4B.json`.
- `infer/tests/greedy_consistency.rs` — determinism across restarts.
- `scripts/bench_guidellm.sh` — canonical perf gate.

## Graph capture design

### Ladder

- **batch_size ladder:** reuse `cuda_graph_batch_sizes` (`core.rs:1327-1345`). For max_bs=16: `{1..=16}`; for max_bs=72: `{1..=64, 80}`.
- **num_tokens ladder (new `cuda_graph_mixed_num_tokens()`):** `[16, 32, 64, 128, 256]` for cap=256 target; add `384, 512` for cap=512 stretch.
  - Rationale: K=2 splits `cap/K` mod 16, so per-req chunks are multiples of 16. `Σc_i = K × per_req_chunk` hits natural boundaries `{32, 64, 128, 256}`. Include `16` for K=1 edge.
- **Total graph count:**
  - max_bs=16: 16 bs × 5 nt = **80 graphs**.
  - max_bs=72: 65 bs × 5 nt = **325 graphs** (decode-only already caches 65).

### Per-graph memory

Qwen3-4B forward ≈ 380 recorded launches per replay (36 layers × ~10 kernels + pre/post norm + logits). cudarc `CUgraphExec` with `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` keeps params lazy → **~2-6 MB per graph**.
- max_bs=16: **~160-480 MB**
- max_bs=72: **~650 MB-2 GB**

At ~22/24 GB steady state, max_bs=16 is workable; max_bs=72 needs budget truncation.

**Budget cap:** `MIXED_GRAPH_MAX_TOTAL_MB` env (default 512 MB). Warmup short-circuits past the cap; drops high-bs high-nt combos first.

### Padding to nearest bucket

```rust
fn round_up_to_bucket(nt: usize, buckets: &[usize]) -> Option<usize> {
    buckets.iter().copied().find(|&b| b >= nt)
}
```

Called in scheduler after chunk plan (`decode.rs` ~`:163`, after `prefill_token_total`):

1. `padded_nt = round_up_to_bucket(Σc_i, MIXED_NT_BUCKETS)`.
2. If `padded_nt != Σc_i`: extend the last candidate chunk's token count up to the delta, clipped to remaining prompt tokens for that req. If can't absorb, drop to next smaller bucket.
3. If no bucket fits (Σc_i < 16 — rare, chunks clip mod 16 at `decode.rs:106`): fall through to eager.

**Pad-via-real-tokens, not dummy rows.** Dummy rows need attention masking (kernel-shape change); real-token padding costs only work we'd do next tick anyway. Net zero TTFT impact.

### FlashInfer plan reuse inside capture

Key insight from sglang `cuda_graph_runner.py:1127-1200`: **plan runs outside the captured graph.** Plan mutates `int_workspace` via rotating page-locked pool — stream capture would record a H2D with a shifting host pointer.

Our decode path already does this correctly: `plan_attention` (`batch_decode.rs:293-316`, called pre-`launch`) → graph body reads stable `int_workspace` GPU pointer.

**Mixed symmetry:**
- Pre-capture: `update_mixed_multi` + `tc_plan` + token H2D + page-table H2D + per-prefill logit target init.
- Graph body: embedding_batch → layer×36 → final norm → logits gemm → decode-slice D2D copy + per-prefill D2D row extracts.
- Post-replay: nothing (readback is separate step).

**Replay-time metadata rewrite:** sglang rewrites `qo_indptr`/`kv_indptr` in pre-allocated GPU buffers pre-replay (stable pointers, mutable contents). Our `update_mixed_multi` already H2D-writes into stable `metadata.qo_indptr`/`metadata.kv_indptr` (`flashinfer.rs:1011-1022`). **Replay-compatible modulo the realloc branch** — see R4.

## Allocation hoist audit

Critical path: `Qwen3Model::decode_batch_with_prefills` (`batch_decode.rs:345`). Every alloc must land pre-capture or at `BatchDecodeBuffers::new`.

### Inside the replay window

| # | File:Line | What | Hoist target |
|---|---|---|---|
| 1 | `batch_decode.rs:386-393` | Lazy `logits_batch` alloc | Eager at `BatchDecodeBuffers::new` |
| 2 | `batch_decode.rs:394-400` | Lazy `mixed` (MixedBatchBuffers) | Eager iff `model.supports_mixed_batch()` |
| 3 | `batch_decode.rs:422-426` | `Vec::with_capacity(total_tokens)` for `combined_tokens` host scratch | Struct field `mixed_token_scratch: Vec<i32>` sized `max_tokens`, `clear+extend` per tick |
| 4 | `batch_decode.rs:427-430` | `memcpy_htod` into `mixed.token_ids_gpu` | Pre-capture (already) |
| 5 | `batch_decode.rs:447-452` | `update_mixed_multi` (5× H2D + potential realloc at flashinfer.rs:998-1006) | Pre-capture (already). **Defang realloc via oversizing — see R4.** |
| 6 | `batch_decode.rs:453-460` | `tc_plan` (async H2D into `int_workspace`) | Pre-capture (already) |
| 7 | `batch_decode.rs:469-479` | **Per-prefill `memcpy_stod(page_table_host)` — fresh CudaSlice alloc per call.** The single biggest per-tick alloc. | New `MixedBatchBuffers::prefill_page_table_gpu: Vec<CudaSlice<i32>>` sized `MIXED_PREFILL_MAX_REQS × max_pages_per_req`. `memcpy_htod` into pre-alloc slot. |
| 8 | `batch_decode.rs:788-803` | Per-prefill `DeviceVec::zeros(output_projection.rows)` for `prefill_logits` | Eager at `Qwen3State` construction (`base.prefill_logits`). |
| 9 | `batch_decode.rs:797-803` | Per-prefill `extract_vec_into` into `pstate.base.prefill_logits` | D2D copy into pre-allocated target (safe after #8). Issue `MIXED_PREFILL_MAX_REQS` launches unconditionally, offsets baked from stable CPU-side `qo_indptr_h`; unused slots copy to discard. |

### Outside the replay window (scheduler-side; CPU)

| # | File:Line | What | Hoist target |
|---|---|---|---|
| S1 | `decode.rs:109-156` | `Vec<CandidatePlan>` + per-candidate `effective_tokens[progress..end].to_vec()` | `Scheduler::mixed_candidate_scratch` + `mixed_token_scratch` |
| S2 | `decode.rs:271-276` | `page_table_host: Vec<i32>` per chunk | `Scheduler::mixed_pages_scratch: Vec<Vec<i32>>` |
| S3 | `decode.rs:264-305` | `Vec<PrefillChunk>` | `Scheduler::mixed_chunks_scratch` |
| S4 | `decode.rs:348-356` | `Vec<PrefillSection<'_>>` | Inline builder with borrowed slices |

### Coverage check

After #1–#9 + S1–S4: no allocation remains inside `begin_capture`→`end_capture`. Body = embedding → layer loop → norm → logits → decode-slice D2D + K prefill-row D2Ds, all into pre-alloc targets with stable pointers.

## Risk matrix

### R1 — Shape drift (decode_count changes mid-bench)

Preemption at `decode.rs:162-221` can flip decode_count 16↔15. Decode graph miss → capture once, replay after. Same pattern as decode today. No new risk.

**num_tokens padding edge:** if Σc_i < 16 and no bucket fits (rare — chunks clip mod 16 at `decode.rs:106`), fall to eager. Near-zero TTFT impact.

### R2 — Metadata replay corruption

`tc_plan` runs pre-capture; only `int_workspace` (stable GPU pointer) is read by the replay body. Page-locked rotating pool never enters capture.

**Detection:** debug-only bit-identity check at warmup — capture+replay vs eager once per bucket; assert bit-identity. Production skips.

### R3 — Silent numeric corruption on some `(bs, nt)`

**Mitigation:**
- Env gate `INFER_MIXED_CUDA_GRAPH={auto, always, never}` defaulting `auto`.
- First-capture-per-bucket: compare replay vs eager on capturing tick; if divergent, blacklist that bucket → eager for life.
- Bench plan: commit 1 ships `never` (hoist-only baseline); commit 2 flips to `auto`.

### R4 — Interaction with decode CUDA graph cache

Mixed + decode share `FlashInferDecodeMetadata`. `update_mixed_multi` realloc at `flashinfer.rs:998-1006` would dirty both cached graphs.

**Hoist solution:** size `max_total_pages` in `FlashInferDecodeMetadata::new` to cover worst-case mixed at raised cap: `max_bs × max_seq_pages + MIXED_PREFILL_MAX_REQS × max_prefill_pages ≈ 73k pages` for Qwen3-4B / 4096 ctx / page_size=1. **~300 KB** of GPU i32 at startup. Realloc branch becomes unreachable on mixed path.

**Graph cache layout:** parallel `decode_graphs: Vec<Option<CudaGraph>>` + `mixed_graphs: Vec<Vec<Option<CudaGraph>>>` keeps `invalidate_graph_cache` semantics cleaner than a flat combined cache.

### R5 — L4 memory pressure

c=16 bench already at ~22/24 GB. Adding 80 graphs × 4 MB ≈ **320 MB** — tight but workable. max_bs=72 × 5 nt = 325 × 4 MB ≈ **1.3 GB** — over budget.

**Mitigation:** `MIXED_GRAPH_MAX_TOTAL_MB` (env, 512 MB default). Warmup truncates high-bs low-frequency entries first. Log cache size post-warmup.

### R6 — Baseline drift (un-diagnosed, research doc:286-291)

TTFT p50 drift 3307 → 5511 between historical and fresh-rebuild on same branch. Bench plan: take fresh baseline at HEAD (pre-ROI#2) on work day; cite both (vs fresh, vs historical) to preempt review questions.

### R7 — Cap raise pre-graph (safe partial landing)

**Yes — recommended.** Commit 1 is hoist-only at cap=64; semantics-preserving (eager forward with pre-alloc scratch). Zero-regression gate before adding capture. Clear blame isolation across commits 1→4.

### R8 — LoRA + no-cuda build

- `supports_mixed_batch()` returns false for LoRA → scheduler falls back to plain decode + separate prefill. Untouched.
- `#[cfg(feature = "cuda")]` gates entire mixed-graph code. Mac type-check via `cargo check -p infer --no-default-features --features cuda,no-cuda` passes (stubs).

## Acceptance + bench

### Numerics parity

- `cargo test --release` — CPU tests, ~9 s.
- `cargo test --release --test e2e` — Qwen3-4B BF16 bit-identity against `infer/test_data/Qwen3-4B.json`.
- `cargo test --release --test greedy_consistency` — determinism 100-token seq at c=4.
- **New test** (commit 2): `INFER_FORCE_MIXED_BATCH=1` env forces mixed path in e2e → asserts bit-identity eager vs graph replay.

Any divergence = revert.

### Perf ladder

Per research doc target **TTFT p99 −2…−4 s, out tok/s +3…+5 %**.

1. **Fresh-baseline** at HEAD (pre-ROI#2, cap=64 hoist-less). Drift capture.
2. **Commit 1 — hoist-only**, cap=64, mixed still eager. Expected Pareto-neutral ±5 %.
3. **Commit 2 — graph capture**, cap=64. Expected TTFT p99 **−0.5…−1.5 s** from amortisation.
4. **Commit 3 — cap=128**. TTFT p99 continues down; ITL p99 flat.
5. **Commit 4 — cap=256**. Hit research target: TTFT p99 −2…−4 s, tok/s +3…+5 %, ITL p99 ≤ 115 ms (sglang parity).
6. **Commit 5 (optional) — cap=512 stretch + memory budget guard**. Ship iff ITL holds.

Each commit lands `docs/experience/wins/YYYY-MM-DD-bench-guidellm-roi2-{stepN}.md`. Regressions → `errors/` + replan.

### Canonical params (locked, `docs/plans/guidellm-integration.md` §3)

- Model: Qwen3-4B BF16 · Concurrency: 16 · Prompt: 4096 · Output: 256 · `scripts/bench_guidellm.sh` defaults.

## Commit sequence

### Commit 1 — `feat(qwen3): hoist mixed forward allocations pre-capture` (~200 LoC)

Hoist audit items #1–#9 + S1–S3. Mixed still eager.

**Touched:**
- `infer/src/model/qwen3/batch_decode.rs`: eager-init `logits_batch` + `mixed` in `BatchDecodeBuffers::new`; add `MixedBatchBuffers::prefill_page_table_gpu` + `mixed_token_scratch`; replace `memcpy_stod` (`:475`) with `memcpy_htod` into slot.
- `infer/src/scheduler/cuda/decode.rs`: `Scheduler::{mixed_candidate_scratch, mixed_chunks_scratch, mixed_pages_scratch}`; reuse across ticks.
- `crates/cuda-kernels/src/flashinfer.rs`: raise `max_total_pages` sizing for callers (no type change).

**Acceptance:** bench at cap=64 ΔTTFT p99 within ±5 % of baseline; `e2e.rs` bit-identical.

### Commit 2 — `feat(qwen3,scheduler): mixed-step CUDA graph capture + replay (cap=64)` (~350 LoC)

**Touched:**
- `infer/src/model/qwen3/batch_decode.rs`: replace `graph_cache: Vec<Option<CudaGraph>>` with parallel `decode_graphs` + `mixed_graphs: Vec<Vec<Option<CudaGraph>>>`; add `decode_batch_with_prefills_graph_body`; capture/replay switch mirrors `:904-940`.
- `infer/src/scheduler/cuda/core.rs`: `cuda_graph_mixed_num_tokens()` + `warmup_graphs_mixed_pass`; extend `warmup_cuda_graphs` iff `supports_mixed_batch()`.
- `infer/src/scheduler/cuda/decode.rs`: `round_up_to_bucket` helper + padding call site post-chunk-plan.
- `infer/src/model.rs`: extend `DecodeContextOps` with `invalidate_mixed_graph_cache(bs, nt_bucket)`; decode default no-ops.

**Acceptance:** bench at cap=64 shows TTFT p99 improvement vs commit 1; `e2e.rs` + `greedy_consistency.rs` pass; cache mem < 200 MB logged.

**Rollback:** `INFER_MIXED_CUDA_GRAPH=never`.

### Commit 3 — `feat(scheduler): raise MIXED_PREFILL_CAP to 128` (~20 LoC)

`decode.rs:15` + `batch_decode.rs:29` → 128; confirm buckets `[16,32,64,128]`.

**Acceptance:** TTFT p99 improves vs commit 2; ITL p99 no regression >10 ms.

### Commit 4 — `feat(scheduler): raise MIXED_PREFILL_CAP to 256, full ladder` (~10 LoC + bench)

Same two files; buckets `[16,32,64,128,256]`.

**Acceptance:** hit TTFT p99 −2…−4 s, tok/s +3…+5 %, ITL p99 ≤ 115 ms. Cross-link prior baseline.

### Commit 5 (optional) — `perf(scheduler): cap=512 stretch + memory budget guard` (~30 LoC)

Ships iff commit 4 shows headroom and ITL holds. `MIXED_GRAPH_MAX_TOTAL_MB` warmup truncation.

## Critical files

- `/content/workspace/agent-infer/infer/src/model/qwen3/batch_decode.rs`
- `/content/workspace/agent-infer/infer/src/scheduler/cuda/decode.rs`
- `/content/workspace/agent-infer/infer/src/scheduler/cuda/core.rs`
- `/content/workspace/agent-infer/crates/cuda-kernels/src/flashinfer.rs`
- `/content/workspace/agent-infer/infer/src/model.rs`
