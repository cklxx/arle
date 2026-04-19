# Multi-request mixed prefill attempt 1 — reverted: ITL +30% / tok/s -40%

## Context

Sequel to the c=16 trace-breakdown win at commit `f21d15e`, which
identified that 92% of every request's E2E is inter-request wait and
therefore cap tuning alone cannot reach 99% parity with sglang. The
architectural fix called out: **fuse multiple prefilling requests into
one mixed forward pass**, like sglang's `--enable-mixed-chunk`.

Implemented per the approved Plan output: `K=4` reqs per mixed tick,
`MIXED_PREFILL_CAP=512` total, round-robin selection, per-chunk alloc
with rollback on partial OOM, `update_mixed_multi` metadata builder.
~610 LoC across 8 files. Build clean, K=1 regression check **passed
byte-identical** (functionally correct).

## Result (c=16 × 4096-prompt × 256-output, L4, 60s, INFER_TRACE=1)

| metric | baseline `f21d15e` | attempt 1 (K=4, cap=512) | Δ | gate | verdict |
|---|---|---|---|---|---|
| TTFT p99 (ms) | 19222 | 20879 | +8.6% | ≥-30% | **fail** |
| ITL p99 (ms) | 205 | 266 | **+29.6%** | ≤+5% | **fail** |
| out tok/s | 90.9 | 54.6 | **-40.0%** | ≤-5% | **fail** |

Functional correctness held (K=1 byte-identical to pre-refactor); the
failure is purely a perf design choice.

## Root causes (per trace + step breakdown)

1. **Decode plan-reuse was torched.** `update_mixed_multi` sets
   `plan_dirty = true` on every mixed tick. In baseline the mixed
   path fired only when *one* prefill req was waiting; with K=4 the
   mixed path fires on most ticks, forcing a fresh `tc_plan` call on
   the following decode tick. Plan cost is amortised over *many*
   steady-state decode ticks, so frequent invalidation multiplies the
   per-tick overhead.

2. **FlashInfer kernel shape mismatch.** Each mixed tick now issues
   `B=16` decode rows + up to `ΣC=512` prefill rows = 528 qo entries.
   FlashInfer's TC decode split-KV heuristic is tuned for small-B,
   short-qo shapes; at 528 qo rows with many varlen kv segments, the
   scheduler picks a partitioning that's good for prefill-shaped work
   but starves the 16 decode rows. Result: decode rows cannot keep up
   and throughput collapses 40%.

3. **Per-req chunk too small.** `MIXED_PREFILL_CAP / K = 512/4 = 128`
   tokens/req/tick rounded to multiples of 16. That's worse than
   baseline's regular-path `prefill_chunk_size = 2048` (which fired on
   the `max_prefills = 2` branch of non-mixed ticks). Multi-prefill
   replaced a fast 2048-tok path with a slow 128-tok path — TTFT got
   slightly worse, not better.

4. **Contention for `kv_indices`.** With all K prefill slots' pages
   concatenated in the shared `kv_indices` buffer, the entries count
   spikes per tick, causing occasional reallocation under
   `indices_scratch.len() > self.max_total_pages`. Each realloc
   invalidates cached decode pointers in the CUDA graph (which are
   disabled for this bench but the `invalidate_graph_cache` path still
   fires).

## What's committed vs. reverted

Nothing new committed. The 8-file diff was saved to
`/tmp/multi-prefill-attempt-1.patch` (1254 lines) for reference and
the tree was reset to `f21d15e`.

## Rule

**Before fusing work across multiple request boundaries, check how much
the plan-reuse fast path was contributing to baseline throughput.**
Our decode plan reuse was one of the biggest wins in the baseline (it's
why steady-state decode ticks are <100 ms). Invalidating it per
mixed-tick cost us more than the multi-req fusion saved. Any future
multi-prefill design must either:
- NOT invalidate decode plan on mixed ticks (requires separate
  FlashInferWorkspace for mixed, reintroducing some of the VRAM cost
  the f21d15e refactor removed); or
- Amortise by firing the mixed path far less frequently (e.g. only
  when >=K prefills are queued, with K ≥ 3); or
- Use FlashInfer's *prefill* kernel for the mixed step instead of the
  TC decode kernel, since prefill kernels are designed for many qo
  rows.

## Next step (design candidates, none committed)

1. **Dual-workspace mixed path** — give the mixed path its own
   `FlashInferWorkspace` again but sized for mixed-only shapes
   (256 MiB is overkill; 128 MiB suffices at cap=512). Preserves
   decode plan reuse on the regular decode path at the cost of ~130 MiB
   VRAM.
2. **Threshold-gated mixed** — only fire the K-req mixed path when
   `prefill_count >= 3`. Smaller prefill queues use the existing single-req
   mixed + regular-chunk path that baseline already optimised.
3. **Prefill-kernel mixed** — route the mixed step through
   `flashinfer_batch_prefill_paged_hd128_run` instead of
   `flashinfer_tc_decode_run`. The prefill kernel takes the same varlen
   `qo_indptr` shape and is specifically tuned for large qo counts.
   Would require a new plan path (prefill plan vs. TC decode plan) but
   eliminates root cause #2 directly.

Option 3 is the closest architectural match to sglang (they use
`BatchPrefillWithPagedKVCache` for mixed, not decode). Biggest change
but biggest upside.

## Artefacts

- Patch: `/tmp/multi-prefill-attempt-1.patch`
- Bench raw: `bench-output/2026-04-19-cuda-l4-infer-multi-prefill-attempt1-c16/`
  (if preserved by subagent) — verify via `ls bench-output/` before
  citing.
- Baseline artefact for this comparison:
  `bench-output/2026-04-19-cuda-l4-infer-shared-ws-c16-c16/`.

## Context metadata

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4 24 GB, CUDA 12.8
- **Commit (baseline):** `f21d15e` (post-share-workspace)
- **Commit (this experiment):** uncommitted; tree reset.
- **Feature set:** `cargo build --release -p infer`
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608
  --mem-fraction-static 0.94 --cuda-graph=false`, env `INFER_TRACE=1`.
