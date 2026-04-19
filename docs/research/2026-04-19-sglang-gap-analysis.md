# sglang 0.5.10 → agent-infer gap analysis (post K=2 cap=64 landing)

**Date:** 2026-04-19 · **Target:** sglang `v0.5.10` on L4 24 GB · **Workload:**
c=16 × 4096-prompt × 256-output · **Remaining deltas to close:** TTFT p99
+127 %, tok/s −9 % (we already lead on TTFT p50 and ITL p50/p99).

> **State note (2026-04-19, post-commit `78e1f8a`).** This doc was authored
> against `main`, which still used the single-prefill mixed path. Branch
> `claude/c16-admission-gate-v2` (commit `78e1f8a`) has since landed **K=2
> cap=64 multi-request mixed prefill**, partially closing gap #1 below.
> Current bench on that branch:
>
> | metric | sglang | this branch | Δ vs sglang |
> |---|---|---|---|
> | TTFT p50 | 5696 ms | **3341 ms** | −41% (INFER WINS) |
> | TTFT p99 | 10727 ms | 24367 ms | +127% |
> | ITL p50 | 92 ms | **71 ms** | −23% (INFER WINS) |
> | ITL p99 | 113 ms | **113 ms** | parity |
> | out tok/s | 140 | 128 | −9% |
>
> **Gap #1 is now only *partially* closed** (K=2, fixed per-tick cap=64).
> sglang's `PrefillAdder` has no fixed K — it packs until the budget
> runs out, so the full gap closure is K=4+ with a dynamic budget. The
> remaining gaps #2–#8 below are unchanged by the K=2 landing.

## Subsystems surveyed

1. Admission / PrefillAdder budget (who admits, what they budget for)
2. Mixed-chunk fusion (`ScheduleBatch.mix_with_running` · `enable_mixed_chunk`)
3. Retract / preemption (`ScheduleBatch.retract_decode`)
4. Chunked-prefill size policy (default 2 k … 16 k by GPU mem)
5. CUDA graph capture coverage (decode-only vs mixed)
6. FlashInfer plan reuse inside graph capture
7. Hierarchical prefix cache T0 → T1 → T2 (HiRadixCache)
8. Speculative decoding (EAGLE-1/3, target verify)

## Gap table

Ranked by expected contribution to the remaining L4 c=16 delta (TTFT p99
+127 %, tok/s −9 %). Citations use `file:line` in sglang (all relative to
`python/sglang/srt/` in the `v0.5.10` tag, fetched via raw GitHub).

| # | Subsystem | sglang does | we do | Expected impact | Effort (LoC · files) | Risk |
|---|-----------|-------------|-------|----------------|----------------------|------|
| **1** | Multi-request prefill fusion | `PrefillAdder.add_one_req` loops over the waiting queue until `rem_chunk_tokens` / `prefill_max_requests` is exhausted, packing N>1 prefill reqs into one batch (`schedule_policy.py:445–545`, `:376–399`) | `step_decode_launch_mixed` fuses exactly ONE prefill req (`prefill_slot_idx: usize`) with the decode batch — hard-coded in `forward_mixed_batch` API (`infer/src/scheduler/cuda/decode.rs:55`, `infer/src/model/qwen3/forward.rs:451`) | **TTFT p99 (primary): halves the long-tail tread when >1 new req queues during a decode-heavy step. Also +throughput.** | ~400 LoC · 4 files (`decode.rs`, `forward.rs`, `batch_decode.rs`, `policy.rs`) | Medium — touches the kernel input layout; needs per-req prefill-start offset arrays |
| **2** | Chunked-prefill default too small | Dynamic: `2048` <20GB, `4096` <60GB, `8192` <90GB (`server_args.py` `_handle_gpu_memory_settings`). L4 24 GB → 4096 | `SchedulerConfig::prefill_chunk_size = 4096` (types.rs:129) which is OK, but `MIXED_PREFILL_CAP = 64` (decode.rs:8) caps the mixed-batch prefill slice to 64 tokens per step. A 4 k prompt walks 64 mixed steps | **TTFT p99 (primary): every prefill that overlaps decode serialises through 64-tok slices; boosting to 256–512 cuts that to 8–16 steps.** | ~50 LoC · 1 file (`decode.rs` cap + tuning sweep) | Low — same knob 2026-04-17 tried at 256 and regressed. Needs decode-graph capture window extended in lockstep (see gap #4) |
| **3** | CUDA graph coverage of mixed batches | Captures DECODE and (for spec) TARGET_VERIFY with `num_tokens_per_bs = K`; padded by `bisect_left` over `capture_bs` (`cuda_graph_runner.py:640–680, 853–910, 1169–1200`) | Captures decode only; mixed prefill step falls off graph replay (`batch_decode.rs:826–859`, mixed path in `forward.rs:451` is eager) | **TTFT p99 + tok/s: mixed step runs ≈8× slower eager vs captured** | ~600 LoC · 3 files — per-chunk-size graph capture, pad-to-nearest replay, metadata replay hook | High — CUDA-graph capture rejects allocations; mixed step currently alloc-heavy |
| **4** | Retract/preempt ordering | Victim ranked by `(len(output_ids), -len(origin_input_ids))` *desc* — prefers most-generated **short-prompt** reqs (`schedule_batch.py:2138–2199`) — so preempted reqs are cheap to recompute | We pick victim by `.generated_tokens.len().max()` only (`decode.rs:315–321` in main path and `:88–95` in mixed path) — ignores prompt length, so we repeatedly preempt the 4 k-prompt reqs and pay 4 k-token re-prefill on every cycle under memory pressure | **TTFT p99 tail (secondary): preemption-thrash at KV-tight steady state; Pareto-neutral** | ~30 LoC · 1 file (`decode.rs` victim pick) | Low — pure heuristic swap; tests unchanged |
| **5** | Hierarchical prefix cache (T1 host-pinned eviction) | `HiRadixCache` demotes GPU blocks to `host_value` (pinned DRAM) on eviction; `match_prefix` returns `last_host_node` so caller can prefetch (`hiradix_cache.py:63–84, 606, 622–676, 1088–1113`). Default `write_through_threshold = 2` | We have the tier taxonomy (`infer/src/kv_tier/tier.rs:14–23`) and a disk store wired, but GPU evictions **free pages outright** — `free_slot` + `clear_slot_prefix_ownership` (`decode.rs:106, 126`) drop bytes instead of demoting to T1 | **TTFT p99 + cold-cache TTFT: under high-concurrency churn, evicted prefixes never come back (always cold re-prefill)** | ~500 LoC · 3 files (coordinator demote, scheduler evict hook, prefix lookup T1 fallback) | Medium — infrastructure largely exists (HostPinnedPool, coordinator); wiring is the work |
| **6** | FlashInfer plan reuse across mixed ticks | Plan built once during `init_forward_metadata_capture_cuda_graph`; replayed via `init_forward_metadata_replay_cuda_graph` which only rewrites varlen metadata (`cuda_graph_runner.py:1127–1137, 1192–1200`) | Our prefill calls `SinglePrefillWithKVCacheDispatched` per chunk with per-call setup — no plan cache (confirmed closed out in `docs/plans/flashinfer-planned-prefill.md`; still relevant to re-evaluate for the mixed path) | TTFT p50 (~40 ms), negligible on tok/s | ~150 LoC · 2 files | Low — but already triaged as low-ROI 2026-04-17 |
| **7** | Speculative decoding | EAGLE-1/3 wired as full replacement of `forward_decode`; `speculative_algorithm` None by default in 0.5.x (`eagle_worker.py`, `server_args.py`) | Pure CPU verify-only framework in `infer/src/speculative.rs` (641 LoC); no draft-model wiring, no target-verify forward. `DraftEngine` trait is a GPU stub | **ITL p50 (marketing upside 1.5–2×) but not the L4 gap we're chasing** | 3–6 kLoC · new (dedicated project) | High — whole new subsystem; defer |
| **8** | PrefillAdder token-budget accounting | `rem_total_tokens` (KV budget) and `rem_chunk_tokens` (per-iter cap) separately throttled, protecting decode (`schedule_policy.py:315–362, 486–492`). Multi-request admission naturally stops at either limit | Our admission is queue-depth only (`policy.rs:74–78 QueueBoundAdmission`) — we don't price KV pressure into admission, so a burst on an almost-full pool triggers preemption instead of backpressure | TTFT p99 tail (shared with #4); tok/s | ~150 LoC · 1 file | Low — pure scheduler CPU code, no kernel touch |

### Subsystems where we match or beat sglang (do not retread)

- **Paged prefill lifting chunk cap.** `Qwen3.prefill_uses_paged_pool() = true`
  (`infer/src/model/qwen3/forward.rs:194`) bypasses the contiguous-KV cap that
  caused the 2026-04-17 TTFT hump — closed. Do not re-open.
- **Decode-only CUDA graph.** `batch_decode.rs:826–859` already captures
  per-batch-size graphs for non-LoRA Qwen3 decode — parity.
- **Paged-pool eviction with refcount pinning.** `RadixCache` + page-ref-count
  bookkeeping matches sglang's `lock_ref > 0` invariant (`infer/src/scheduler/cuda/core.rs:96–114`).
- **Variable-length packed decode.** `batch_decode.rs` handles variable `seq_lens`
  via FlashInfer BatchDecode with HD128 + HD256 kernels — ITL p50 already beats
  sglang at matched rates (`wins/2026-04-17-sglang-p99-parity-qwen3-4b.md`).
- **Short-prompt fixed-overhead (Rust HTTP path).** 37 ms lower than sglang's
  Python-dispatch path — already winning below ~1500-token prompts
  (`wins/2026-04-17-short-prompt-ttft-advantage.md`).
- **Prefix-aware admission *trait*** (`policy.rs:98–130 PrefixAwareAdmission`).
  The policy exists but is unused by the scheduler core — gap #8 is about
  *wiring it in*, not designing it.

---

## Top-3 high-ROI fixes, fully spec'd

### ROI #1 — Multi-request prefill fusion (gap #1)

**Thesis.** Under c=16 with 4 k prompts, ≥2 new requests almost always land on
the waiting queue mid-decode. Today one waits while the other runs — that wait
is the dominant contributor to TTFT p99 tail. sglang's `PrefillAdder` admits
both; we serialise. Closing this halves the p99 tread roughly (one hop is no
longer 2× the mean).

**What sglang does.** `schedule_policy.py:445–545` — `add_one_req` is called
in a loop by the scheduler tick (`scheduler.py:2398–2410`), each call decrements
`rem_chunk_tokens` and `rem_total_tokens`. The `ScheduleBatch` is built with N
request rows; `mix_with_running` (`schedule_batch.py:1951–1982`) concatenates
that multi-request prefill with the decode batch into one `ForwardMode.MIXED`
forward.

**Files to touch.**

1. `infer/src/model.rs` — widen `ModelForward::forward_mixed_batch` signature:
   ```rust
   fn forward_mixed_batch(
       &self,
       decode_tokens: &[u32],
       // was: prefill_slot_idx: usize, prefill_start_pos: usize, prefill_tokens: &[u32]
       prefill_slots:  &[usize],
       prefill_starts: &[usize],
       prefill_tokens: &[&[u32]],
       states:         &mut [Self::State],
       decode_slot_indices: &[usize],
       paged_kv_pool:  Option<&mut PagedKVPool>,
       decode_ctx:     &mut Self::DecodeContext,
   ) -> Result<bool>;
   ```
2. `infer/src/model/qwen3/forward.rs:451` — plumb the multi-prefill vectors
   through `decode_batch_with_prefill`. Kernel-side: the FlashInfer varlen
   prefill already accepts ragged `qo_indptr` + `kv_indptr`; construct them
   from `(prefill_slots, prefill_starts, prefill_tokens.lens())`.
3. `infer/src/model/qwen3/batch_decode.rs` — extend `DecodeContext` with
   per-prefill-request scratch (rope-offset, position-id, indptr). Keep
   the per-request K/V pool write path — it's already per-slot.
4. `infer/src/scheduler/cuda/decode.rs:11–264` — replace the single
   `prefill_idx` with a `Vec<usize>` of prefill candidates. Budget:
   - `rem_chunk_tokens = MIXED_PREFILL_CAP` (bumped from 64 to 256; see #2)
   - `max_prefills = 2..4` (tune)
   - Stop adding when `sum(prefill_token_count) >= rem_chunk_tokens` OR
     `paged_kv_pool.free_count() < decode_indices.len() + Σ prefill_tokens`
5. `infer/src/scheduler/cuda/execution.rs:27–39` — `mixed_prefill_idx` becomes
   `mixed_prefill_idxs: Vec<usize>` built by the same "find prefilling,
   drop those closed, sort by waiting-time desc" rule.
6. `infer/src/scheduler/policy.rs` — extend `ChunkingPolicy` with a
   `multi_request_budget(&self, signals) -> (max_reqs, per_iter_tokens)`
   trait method; default impl matches today's behaviour.

**Approximate LoC.** ~400 new + ~80 refactor. Tests: `scheduler/tests.rs`
already has `test_preemption` and chunk-size fixtures; add 2–3 multi-prefill
scenarios.

**Expected gain.** On the c=16 × 4096 prompt workload:
- TTFT p99 from 24 s → ~12–14 s (approaching sglang's 10.7 s).
- tok/s +5–8 % (more GPU time under the same decode rhythm).

Hypothesis, not fact, until bench: the TTFT p99 improvement should scale with
`min(waiting_queue_depth, max_prefills)`. At K_prefill = 2 it should halve
the wait; at K_prefill = 4 closer to quarter.

**Risk.** Pareto-neutral *if* FlashInfer varlen prefill handles the ragged
layout at our shapes (HD128/HD256, seq_len ≤ 4096). Regression path is the
same as 2026-04-17 mixed256: decode graph capture window (today ≤74 decode
batch) must tolerate the new total `max_tokens` — gate behind the same
decode-batch cutoff as CUDA graph, and disable multi-prefill when
`decode_count > 16` (already our adaptive prefill-count logic at
`execution.rs:90–98`).

---

### ROI #2 — Raise `MIXED_PREFILL_CAP` in lockstep with graph coverage (gap #2, tight coupling with #1)

**Thesis.** `MIXED_PREFILL_CAP = 64` slices every 4 k-prompt into 64 mixed
steps. sglang runs full 4 k in 1 step during idle, and slices ≥512 tokens
during decode overlap. The 2026-04-17 regression at 256 was a symptom of
**decode graph falling off replay**, not of the cap itself. Fix the graph
first, then lift the cap.

**What sglang does.** `cuda_graph_runner.py:655` `bs * num_tokens_per_bs %
mul_base == 0` builds a mixed-friendly `capture_bs` ladder. Replay via
`bisect_left` pads up to the nearest captured shape — so 256-tok mixed
steps replay on the same graph as 64-tok once captured.

**Files to touch.**

1. `infer/src/scheduler/cuda/decode.rs:8` — `const MIXED_PREFILL_CAP: usize =
   256;` (target) or `512` (stretch).
2. `infer/src/model/qwen3/batch_decode.rs:80–859` — extend
   `graph_cache: Vec<Option<CudaGraph>>` from `[batch_size]` to
   `[(batch_size, num_tokens)]` (flatten to a `HashMap<(u16, u16), CudaGraph>`).
3. `infer/src/model/qwen3/batch_decode.rs:826–859` — capture a
   `num_tokens ∈ {1, 64, 128, 256}` ladder at startup; at replay time
   bisect to the nearest `num_tokens` and pad the mixed slice.
4. `infer/src/scheduler/cuda/core.rs:885–914` — `prefill_chunk_size()` returns
   min(config cap, nearest captured `num_tokens`) when decode is active.

**Approximate LoC.** ~200. Mostly `batch_decode.rs`.

**Expected gain.** TTFT p99 −2–4 s on 4 k-prompts (slice count 64 → 16).
tok/s +3–5 % (amortised graph-launch overhead).

**Risk.** Medium. CUDA-stream-capture rejects allocations — the mixed path
today allocates the `prefill_tokens` device buffer inside the forward. The
fix requires hoisting those alloc sites into `DecodeContext::new` (one
scratch per captured `num_tokens`). This is the same hoisting work the
2026-04-18 paged-phase3a hoisted win already did for decode; apply the
pattern to mixed.

---

### ROI #3 — Retract/preempt victim ranking (gap #4)

**Thesis.** Cheap win. Under KV pressure at c=16, we preempt the longest-
generated request — today that's the 4 k-prompt reqs that are deepest into
decode, so recompute cost is 4 k prompt tokens. sglang preempts the one
with most *output* tokens *and* shortest *input* tokens — cheap to
recompute. Our current victim picker actively selects the most expensive
victim to re-prefill.

**What sglang does.** `schedule_batch.py:2138–2199`:
```python
sorted_indices.sort(
    key=lambda i: (len(self.reqs[i].output_ids), -len(self.reqs[i].origin_input_ids)),
    reverse=True,
)
```

**What we do.** `infer/src/scheduler/cuda/decode.rs:315–321` *main path*
and `:88–95` *mixed path*:
```rust
let victim_pos = decode_indices
    .iter().enumerate()
    .max_by_key(|(_, i)| self.active[**i].generated_tokens.len())
    .map(|(pos, _)| pos).unwrap();
```
Ignores `prompt_tokens.len()` entirely.

**Files to touch.** `infer/src/scheduler/cuda/decode.rs` — replace both
`max_by_key` call sites with:
```rust
.max_by_key(|(_, i)| {
    let r = &self.active[**i];
    (
        r.generated_tokens.len() as i64,
        -(r.prompt_tokens.len() as i64),
    )
})
```

**Approximate LoC.** ~20 (two call sites + a helper for readability).
Test: `scheduler/tests.rs::test_preemption` — extend to mix 4 k-prompt and
128-prompt requests, assert the 128-prompt one is preempted.

**Expected gain.** Under the 2026-04-19 trace breakdown's "92 % wait"
observation, a significant fraction of that wait is re-prefill after
preemption. Swap victim → 128-token recompute vs 4 k → ~30× cheaper
recompute per preemption event. **TTFT p99 −1–3 s** on the c=16 workload.

**Risk.** Pareto-neutral. The invariant that eviction always frees
≥1 slot of decode KV is preserved (scoring is strict total order; exactly
one victim picked per loop iter). No kernel change, no layout change.

---

## Suggested sequencing

1. **ROI #3 first** — 20 LoC, zero risk, cheap bench.
2. **ROI #2** — requires hoisting the mixed-step allocations (1–2 day task);
   unlocks the cap raise.
3. **ROI #1** — biggest structural lift; do after #2 so its bench runs on
   the new mixed CUDA graph.

Gap #5 (HiRadixCache T1 demotion) is the next tier. Gap #7 (speculative) is
a whole project, punt until 0.6.x.

## Update (2026-04-19, commit `346355b`) — K=3 probe confirms the tradeoff

ROI #3 landed (`346355b`, Pareto-neutral). Next tried a zero-code-change
probe: flip `MIXED_PREFILL_MAX_REQS = 2 → 3` with `MIXED_PREFILL_CAP = 64`
unchanged. At K=3 the per-req chunk rounds down to 16 tokens → 3 × 16 = 48
total tokens per mixed tick (less than K=2's 64).

| metric | K=2 cap=64 (baseline-fresh) | **K=3 cap=64** | Δ |
|---|---|---|---|
| TTFT p50 (ms) | 5511 | 6558 | **+19%** ❌ |
| TTFT p99 (ms) | 16694 | 21497 | **+29%** ❌ |
| ITL p50 (ms) | 87 | 82 | −6% ✓ |
| ITL p99 (ms) | 199 | **105** | **−47% (INFER BEATS sglang 113)** ✓ |
| out tok/s | 123 | 98 | −20% ❌ |

**What this proves.** The K=2 cap=64 ITL p99 regression (199 vs
sglang 113) isn't a kernel bug — it's the direct cost of the 64 tok
mixed-prefill payload per tick. Shrinking the payload (K=3 cap=64 =
48 tokens) closes the ITL gap immediately, but bleeds TTFT and
throughput dollar-for-dollar. **The ITL–TTFT axis is a strict
tradeoff at the current kernel shape.**

Reverted K=3, staying at K=2 cap=64 which hits the better overall
Pareto corner. The path to *simultaneously* winning ITL p99 AND TTFT
p99 runs through **ROI #2 (mixed CUDA graph + cap raise)** — once
replay eliminates the per-tick kernel-launch overhead, the fixed
cost that makes 64-tok payloads expensive drops, and ITL p99 stops
being a function of mixed-tick size. The research ranking stands.

(Also documented: fresh `guidellm 0.6.0` + fresh `target/` rebuild
showed a baseline drift vs the `673b9e9` numbers cited in the K=2
wins entry — TTFT p50 regressed 3307 → 5511, ITL p99 113 → 199.
Under-the-hood drift cause unknown; all the ROI #3 / K=3 numbers
above were measured against this drifted baseline for apples-to-apples.
If the drift is guidellm 0.6 metric changes, the historical wins still
hold on the old tool; re-baselining or pinning guidellm is a follow-up.)

## Citations

- sglang 0.5.10 source: https://github.com/sgl-project/sglang/tree/v0.5.10/python/sglang/srt
  - `managers/scheduler.py` — `get_new_batch_prefill` :2300, `get_next_batch_to_run` :2150, `init_chunked_prefill` :2050, `_abort_on_running_timeout` :2100, `TEST_RETRACT` :2315
  - `managers/schedule_policy.py` — `PrefillAdder.add_one_req` :445, `add_chunked_req` :376, `rem_total_tokens` :315, `_update_prefill_budget` :346, trunc_len :521
  - `managers/schedule_batch.py` — `retract_decode` :2138, `mix_with_running` :1951, `prepare_for_extend` :1698, `prepare_for_decode` :2059, `merge_batch` :2205
  - `server_args.py` — `_handle_gpu_memory_settings`, `chunked_prefill_size`, `cuda_graph_bs`, `enable_mixed_chunk`, `speculative_*`, `enable_hierarchical_cache`, `schedule_conservativeness` (default 1.0)
  - `model_executor/cuda_graph_runner.py` — `get_batch_sizes_to_capture` :640, `ForwardMode.DECODE` capture :866, `init_forward_metadata_{capture,replay}_cuda_graph` :1127/:1192, `bisect_left` padding :1169
  - `mem_cache/hiradix_cache.py` — tier model :63, `write_through_threshold` :147/:606, evict demote :622/:661/:667, `match_prefix` :1088
  - `speculative/eagle_worker.py` — EAGLE-1/3 split, draft worker hook

- agent-infer source (all paths relative to `/content/workspace/agent-infer/`):
  - Scheduler: `infer/src/scheduler/cuda/core.rs:25` (CONTIGUOUS_KV_TOKENS),
    `:885` (prefill_chunk_size), `:862` (alloc_pool_tokens_with_retry)
  - Mixed path: `infer/src/scheduler/cuda/decode.rs:8` (MIXED_PREFILL_CAP=64),
    `:11–264` (step_decode_launch_mixed, single prefill_idx),
    `:315–321` (main-path preempt victim), `:88–95` (mixed-path preempt victim)
  - Execution: `infer/src/scheduler/cuda/execution.rs:27–39`
    (single `mixed_prefill_idx`), `:90–98` (adaptive prefill-count)
  - Model: `infer/src/model/qwen3/forward.rs:194` (prefill_uses_paged_pool),
    `:435` (supports_mixed_batch), `:451` (forward_mixed_batch — single prefill),
    `infer/src/model/qwen3/batch_decode.rs:80–859` (CUDA graph cache, decode-only)
  - Policy: `infer/src/scheduler/policy.rs:74` (QueueBoundAdmission),
    `:98–130` (PrefixAwareAdmission — trait exists, not wired)
  - Speculative (CPU stub): `infer/src/speculative.rs` (641 LoC, no CUDA draft)
  - Tier: `infer/src/kv_tier/tier.rs:14–23`,
    `infer/src/kv_tier/coordinator.rs:159` (T1 host-pinned region)
  - Prior art context: `docs/experience/wins/2026-04-17-sglang-parity-systematic-analysis.md`,
    `docs/experience/wins/2026-04-17-prefill-ttft-root-cause-contiguous-kv-cap.md`,
    `docs/experience/wins/2026-04-17-bench-guidellm-qwen3-4b-infer-l4-mixed256.md`
    (documents the MIXED_PREFILL_CAP=256 regression — it was decode-graph fall-off,
    not the cap itself).
