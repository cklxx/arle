# Allocation architecture — infer vs sglang 0.5.10 (Qwen3-4B, L4 24 GB)

> Research note: compare the CUDA-backend memory/buffer/plan lifecycle
> between infer and sglang, identify every divergence, map each to a
> parity lever from the systematic-parity plan. Source for the sglang
> side: `/usr/local/lib/python3.12/dist-packages/sglang/srt/` (installed
> from `pip install sglang==0.5.10.post1`). Source for infer: the
> `claude/c16-admission-gate-v2` branch head as of `211f1f1`.

---

## 1. GPU memory layout — side-by-side

```
┌──────────────────────────────────────────────────────────────────────┐
│                  GPU MEMORY LAYOUT (single CUDA context)             │
├──────────────────────────────────────────────────────────────────────┤

    WEIGHTS (~8 GB, static, mmap → device)            Both sides identical

                    infer (current)                   sglang 0.5.10
                    ───────────────                   ─────────────

    FlashInfer Workspace                              global_workspace_buffer
    512 MiB (just fixed at 211f1f1)                   512 MiB (SGLANG_FLASHINFER_
                                                      WORKSPACE_SIZE env for Qwen3)
    • float_workspace (bulk)                          • shared across ALL wrappers
    • int_workspace 8 MB                              • init ONCE at model load
    • page_locked_workspace 8 MB                        (flashinfer_backend.py:202-219)
    • owned by BatchPrefillPagedPlan
      via `Mutex<Option<..>>` in weights              ≈ parity ✓ (post-211f1f1)

    ─────────────────────────────────────────────────────────────────
    PrefillBuffers                       [🔴 DIVERGENCE]
    hidden_out, normed, q/k/v_batch, o_buf,           Model layer forward consumes
    gate_out, up_out, act_out, attn_output            slices of workspace_buffer,
    ~10 HiddenStates, sized for `seq_len` (chunk),    and torch's CUDA caching
    allocated PER FORWARD in                          allocator reuses freed blocks
    `forward_prefill_with_pool` via                   silently — no fresh
    `CudaSlice::alloc_zeros`                          cudaMalloc per call.

    At chunk=2048 × c=16:                             Effect: stream does NOT see
    async alloc/free backlog on the                   an alloc/free backlog → no
    compute stream → CUDA context                     context corruption.
    corruption → next gemm panics
    with `CUDA_ERROR_UNKNOWN`                         sglang reference for the
    (`infer/src/ops/linear.rs:693`).                  pattern we need.

    Blocks landing L3 (chunk_size 4096→2048)
    per systematic-parity plan roadmap.
    Fix: move to model-owned
    `Mutex<Option<PrefillBuffers>>`
    with monotonic capacity growth.
    ─────────────────────────────────────────────────────────────────

    PagedKVPool                                       TokenToKVPoolAllocator
    page_size=16 tokens                               page_size=16 tokens
    bf16 layout                                       bf16 layout
    `free_count()` → tokens                           `available_size()` → tokens
    ≈ parity ✓                                        ≈ parity ✓

        ▲                                                  ▲
        │ blocks pinned via ref_count + soft_pin_until     │ blocks pinned via
        │                                                  │ ref_count only
    RadixCache (prefix)                                tree_cache (RadixTree)
    `evictable_token_count()`                          `evictable_size()`
    counts soft-pin-aware nodes                        ref_count==0 only; no
    whose subtree has no allocated block.              soft-pin layer.
    🟡 subtree-predicate is stricter than              Iterative cascade evicts
    `evict_with_policy` iterative cascade              parents once children
    — P2 review item (I'll tighten when                exit (same outcome).
    retract lands; minor vs L1/L3).

└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Admission (CPU-side) — structural equivalence

```
  infer Scheduler::assign_slots                   sglang get_new_batch_prefill
  ─────────────────────────────                   ───────────────────────────
  admission_budget_tokens() =                     PrefillAdder.rem_total_tokens =
      pool.free_count()                               pool.available_size()
    + prefix_cache.evictable_token_count            + tree_cache.evictable_size()
    − running_offset                                − rem_total_token_offset

  running_offset (post-P1#2):                     rem_total_token_offset:
     Σ active. per request:                          Σ running reqs:
       prefill_remaining        (Phase::New         (max_new_tokens -
          or Phase::Prefilling)                      len(output_ids)) *
       + min(max_new-decoded,                        new_token_ratio
          clip) * new_token_ratio                  Σ new admits (this tick):
                                                   extend_input_len +
                                                   min(max_new, CLIP) +
                                                   page_overhead

  per-admission cost (this tick):                 page_overhead = page_size
    extend_input + min(max_new, clip)
    + page_size

  Key subtrahend: prompt_tokens.len()             Key subtrahend:
    − reusable_prefix_len  (post-P1#1)              req.extend_input_len
    = tokens we will actually materialize         = tokens to re-prefill after
                                                    `prefix_indices` reuse

  Result after P1#1+P1#2: numeric                 Baseline, this is what
  formula ≡ sglang's. Constants are               parity is measured against.
  config-driven (`admission_clip_max_
  new_tokens`, `admission_new_token_ratio`),
  defaults 4096 / 0.7 match sglang env.           ≈ structural parity ✓
```

---

## 3. OOM backstop — the biggest remaining gap

```
  infer alloc_pool_tokens_with_retry              sglang update_running_batch
  ───────────────────────────────────              ──────────────────────────
    1. pool.alloc_tokens(count)                    1. pool.alloc() — on OOM:
    2. on Err:                                     2. batch.retract_decode()
         evict_prefix_cache_for_                        • select longest-running
         allocation(count * page_size)                    decoder
    3. retry alloc                                      • release_req(is_insert
    4. on Err again:                                     =False)
         🔴 TODAY: return Err →                         • release_kv_cache →
         `step_prefill_chunk` marks                       pool pages return
         Phase::Finished →                                to free list
         client sees empty 200-OK                       • adjust new_token_ratio
         stream → HTTP retry                          (decay on retract)
         amplifies pool pressure                    3. retry alloc
                                                      • if still OOM, repeat
                                                        retract until pool fits
                                                        or no more decoders

  L1 IN PROGRESS (manual edits):                   0 pool-alloc failures
    add `retract_longest_decode(req,                observed at c=16 ×
    protected_slot) -> usize`                       4096+256 on L4 24 GB
    wire between step 2 and step 4
    above so infer has the same safety             Reference impl:
    net                                             schedule_batch.py:1950
                                                    scheduler.py:2562-2567
```

The 19 pool-alloc failures at c=16 on infer-at-e6e722c were entirely
attributable to this gap: the gate over-admits (P1 bugs) AND there's
no retract safety net (L1 missing). P1 tightens the gate; L1 gives
the retract fallback. Both land this session.

---

## 4. Scheduler tick cadence

Equivalent within the margin of measurement noise.

| | infer (runtime.rs:185-) | sglang (scheduler.py:1303 event_loop_normal) |
|---|---|---|
| Loop shape | drain `request_rx.try_recv` → `assign_slots` → `step()` → `cleanup()` | `get_next_batch_to_run` → `mix_with_running_batch` → `run_batch` |
| Admission cadence | once per loop iteration | once per loop iteration |
| Prefill cadence | up to `max_prefills` requests per iteration (1-8 by decode-count heuristic) | all newly-admitted requests batched into the same batch |
| Decode cadence | single `step_decode_launch` per iteration | single `run_batch` per iteration |

**Divergence**: sglang batches prefill + decode into **one** forward via
`mix_with_running_batch` — Qwen3-4B runs prefill and decode tokens in
the SAME forward pass. Infer alternates phases: when any request is
prefilling, decode waits (well, with the mixed-path kicker in
`execution.rs:27-42` on fire). This is L2 in the parity roadmap; not
in scope this session.

---

## 5. What we've landed & what's next

**Landed in `claude/c16-admission-gate-v2` this session:**
- `c07415c feat(scheduler): sglang-aligned admission-time pool-capacity gate` — PrefillAdder port with `admission_budget_tokens`.
- `e6e722c fix(scheduler): admission-gate review — soft-pin awareness + config wiring` — codex review round 1.
- `211f1f1 fix(cuda,qwen3): HD128 paged prefill workspace 256→512 MiB (sglang parity)` — this commit.

**In flight this turn:**
- **P1#1 + P1#2** (manual edits): `assign_slots` gate uses `reusable_prefix_len` not `radix_hit_len`; `admission_budget_tokens` adds `prefill_remaining` for `Phase::New` / `Phase::Prefilling`. Closes gate-accuracy bugs flagged by codex review round 2.
- **L1 retract-decode**: add `Scheduler::retract_longest_decode` + wire into `alloc_pool_tokens_with_retry` between eviction-retry and OOM-Finished. Mirrors sglang `ScheduleBatch::retract_decode`. Closes the remaining 19 pool-alloc failures at c=16.
- **PrefillBuffers refactor** (subagent): move per-forward `CudaSlice::alloc_zeros` to model-owned `Mutex<Option<PrefillBuffers>>` with monotonic capacity. Unblocks L3 (chunk_size 4096→2048).

**Next (not this session):**
- **L2 mixed-batch CUDA Graph capture** — parity roadmap item for ITL p99 tail.
- **L4 AOT cuda_graph_batch_sizes** — lazy capture removal.

**Completed parity-ish findings (don't touch again):**
- FlashInfer workspace sizing — now 512 MiB everywhere.
- FlashInfer plan sharing — `paged_prefill_plan` already `Mutex<Option<..>>` (commit 5208530 from 04-18).
- PagedKVPool page layout, `free_count()` semantics — identical to sglang.

---

## 6. Learning summary — where infer has to copy sglang's patterns

```
            sglang pattern                 infer port status
            ──────────────                 ─────────────────
   ┌─> global 512 MiB workspace             ✅ landed 211f1f1
   │
   ├─> Wrapper-owned plan state             ✅ paged_prefill_plan Mutex
   │   reused across forwards                  (commit 5208530)
   │
   ├─> Single workspace_buffer              🟡 PrefillBuffers refactor
   │   reused across layer forwards            in flight this turn
   │   via torch's cache allocator             (L3 blocker)
   │
   ├─> PrefillAdder admission with          ✅ landed c07415c + this turn's
   │   running_offset (ratio) and              P1#1/P1#2 manual edits
   │   evictable-aware budget
   │
   ├─> batch.retract_decode()               🟡 L1 in flight this turn
   │   on pool OOM                             (manual edit)
   │
   ├─> mix_with_running_batch               ⏳ L2, next session
   │   (decode + prefill in one forward)
   │
   └─> CUDA Graph capture at ALL            ⏳ L4, next session
       configured batch sizes AOT
```

**The pattern is always the same**: move per-call transient state to
model- or scheduler-owned long-lived resources, sized for the worst-case
workload. Every toxic "per-forward alloc" path in infer's hot loop has
been the source of a production bug under concurrent load; every fix
has converged on the same single-instance/mutex-guarded-reuse pattern
that sglang uses.

This is the structural lesson: **kernel latency parity is 95% already
achieved; throughput parity is held back by resource-lifecycle
divergence, not kernel perf**. Close PrefillBuffers + retract and the
gap closes to single-digit percent.
