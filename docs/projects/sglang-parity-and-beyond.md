# Serving-performance evolution — sglang parity and beyond

**Status:** Active · **Opened:** 2026-04-20 · **Owners:** ckl + Claude

Consolidated roadmap for the CUDA c=16 serving-performance workstream:
what's shipped, what's in flight, and what the architecture enables next
once the sglang 0.5.10 parity gap is closed. Replaces the
ad-hoc stacking of wins/errors/plans that accumulated during the
2026-04-17 → 2026-04-20 parity push with a single forward-looking
tracker.

## Current state (2026-04-20)

| Layer | Status | Representative commit / doc |
|---|---|---|
| **ROI#3 — retract victim ranking** | ✅ shipped | `346355b` + `wins/2026-04-19-retract-victim-sglang-parity.md` |
| **ROI#1 — multi-req mixed prefill K=2 cap=64** | ✅ shipped | `78e1f8a` + `wins/2026-04-19-multi-req-mixed-prefill-k2-cap64.md` |
| **Probe — K=3 cap=64 / K=2 cap=256** | ❌ reverted (ITL/TTFT strict tradeoff confirmed; ROI#2 is the only unconstrained lever) | `wins/...-k3-probe.md` + `wins/...-cap256-probe.md` |
| **ROI#2 C1 — hoist mixed-forward allocs** | ✅ shipped | `b8d1569` + `plans/roi2-mixed-cuda-graph.md` |
| **ROI#2 C2 — mixed CUDA graph capture** | 🚧 planned | `plans/roi2-mixed-cuda-graph.md` §Commit 2 |
| **ROI#2 C3/C4 — raise MIXED_PREFILL_CAP 64 → 128 → 256** | 🚧 planned | `plans/roi2-mixed-cuda-graph.md` §Commit 3-4 |
| **Gap #5 C1 — `PagedKVPool::copy_pages_*` real impl** | ✅ shipped | `dfb16bb` + `wins/2026-04-20-gap5-c1-paged-kv-copy-pages.md` |
| **Gap #5 C2-5 — T1 demote + prefix promote-back** | 🚧 planned | `plans/gap5-kv-tier-demote-prefetch.md` |
| **Drift bisect — 128 → 98 tok/s environmental** | ✅ resolved | `errors/2026-04-20-bench-drift-environmental-not-code.md` |

Bench anchor (L4 / Qwen3-4B BF16 / c=16 × 4096 × 256 /
guidellm 0.6.0): today ~98 tok/s, TTFT p99 ~33 s, ITL p99 ~110 ms.
Target: match sglang's reproducible measurement in the SAME env
(not the historical 128 tok/s which pre-dates the env drift).

## Architecture principles crystallised by this workstream

These are the non-obvious invariants we paid for in bench cycles or
code reviews. Violate them and the next push stalls.

### A1. Hoist before capture

CUDA graph capture rejects every allocation on the recorded stream.
Commit 2 (graph capture) is only safe once Commit 1 has pulled every
per-tick alloc out of the forward body into pre-allocated scratch
owned by `MixedBatchBuffers` / `BatchDecodeBuffers` /
`DecodeContext`. Commit 1's job is specifically to prepare for
Commit 2, not to ship a win on its own — we bench it to prove
Pareto-neutrality, not to celebrate.

### A2. Single source of truth per semantic constant

`MIXED_PREFILL_MAX_REQS` was duplicated across `execution.rs`,
`decode.rs`, and `batch_decode.rs`. The three copies drifted (4 ≠ 2
≠ 2), silently broke the mixed-fusion path for 3–4 candidates, and
codex review caught it (`6aa5372`). The fix: one authoritative
declaration (scheduler-side, `pub(super)`) + a compile-time
`assert!` that the model side matches. Any future bump is a build
break, not a silent runtime fallback.

Rule: if two files hold a constant that MUST stay in sync,
there's exactly one declaration and every other use imports it.
The compile-time assertion replaces the "kept in sync by hand"
comment that's a known lie.

### A3. Drift before blame

When a metric moves, re-measure the historical reference commit in
the current env **before** blaming a commit in the range. At commit
`78e1f8a` with today's `guidellm 0.6.0`, tok/s reads 98 — identical
to HEAD. So the 128 → 98 "regression" was `guidellm` shipping
new backend-resolution / rate-computation in 0.6.0, not any code
commit. Memory record:
`memory/project_bench_env_drift_2026-04-20.md`. Rule encoded in
`errors/2026-04-20-bench-drift-environmental-not-code.md`.

### A4. Observable failures, not silent fallbacks

Three failures during this workstream traced to "function returns
`Err` / `None`, caller drops silently":

- `MIXED_PREFILL_MAX_REQS` mismatch → `decode_batch_with_prefills`
  returns `Ok(false)` → scheduler silently falls back to plain
  decode + prefill. No log, no counter. Found only by code review
  (`6aa5372`).
- `copy_pages_to_host` returns `Err` for non-BF16 → `save_session`'s
  `.ok()` silently drops the block → manifest saves as 200-OK with
  omitted bytes. Found only by code review (`7f0ce50`).
- `copy_pages_from_host` returns `Err` → `install_restored_kv`
  `continue`s without releasing the detached pages → pool leaks
  capacity on every bad payload. Same review (`7f0ce50`).

Rule: a silent fallback path is a latent bug. Every fallback logs at
`warn!` or bumps a counter. Unit tests cover the happy path; the
error path gets observability.

### A5. Tier-aware scheduler, not tier-aware kernels

KV tier (T0 GPU / T1 host-pinned / T2 disk) is a *scheduler*
concept, not a *kernel* concept. Kernels see one contiguous paged
KV pool; the tier machinery lives in the scheduler's cleanup +
admission path (`evict_prefix_cache_if_pressured`, `lookup_or_stage`,
`drain_coordinator_events`). Gap #5 follows this — the coordinator
thread runs D↔H copies on a dedicated stream, the scheduler reacts
to completion events. Kernels are oblivious.

This keeps the kernel surface narrow (one format per path) and the
tier state out of the CUDA graph capture, which is A1-critical.

## Where ROI#2 + Gap #5 converge — the mixed-forward + tier diagram

```
  Admission
    ↓
  lookup_or_stage  — checks T0 hit, T1 hit (post Gap #5 C4 promote-back)
    ↓ match                     ↓ miss
  slot + pages                fresh slot, cold prefill
    ↓
  Scheduler tick
    ├─→ decode batch (B rows)
    └─→ mixed tick = decode (B) + prefill (Σ c_i ≤ CAP)   ← ROI#2 C2 captured as CUDA graph
       └ post-tick ─→ publish_to_prefix_cache
                      ├─→ page_ref_count += 1 on radix pin
                      └─→ hit_count bumped on subsequent lookup
                           ↓ ≥ write_through_threshold
  Cleanup
    ├─→ evict_prefix_cache_if_pressured
    │   ├─ gate: hit_count ≥ threshold → demote to T1 (Gap #5 C3)
    │   │   └ coordinator issues `Demote` → copy stream → DemoteCompleted
    │   │     → scheduler flips tier_location, release_pages
    │   └─ else: release_pages outright (today's behaviour)
    └─→ (Gap #5 C5) T1 watermark → spill to T2 disk via existing
                                   coordinator Spill path
```

Every node in this diagram either ships or has a committed plan.
Nothing speculative. The future-evolution axes below are about the
axes this diagram does NOT cover.

## Future-evolution axes

Open to prioritisation after ROI#2 C4 + Gap #5 C5 land. Listed in
descending expected impact on the c=16 workload; re-rank as the
landscape moves.

### E1. Prefix-aware admission — wire the existing trait

`infer/src/scheduler/policy.rs:97-130` has a `PrefixAwareAdmission`
trait that scores candidates by cached-prefix length (sglang
`PrefillAdder`). It's fully specc'd, but the scheduler core
instantiates `QueueBoundAdmission` by default and nothing wires it
in. Once Gap #5 C4 lands (T1 match affects `prefix_hit_tokens`), the
admission-gate signal is load-bearing for c≥16 throughput — burst of
same-prompt requests should front-load against warm T1 hits, not
cold-prefill in FIFO order. ~150 LoC; low risk; direct TTFT p99 +
tok/s gain.

### E2. Speculative decoding — EAGLE-style draft + target verify

Biggest structural lever beyond this workstream. `infer/src/speculative.rs`
has a 641-LoC CPU verify framework; GPU draft + target-verify is
unwired. sglang ships EAGLE-1/3 as a `forward_decode` replacement.
Estimated 3-6 kLoC new (dedicated subsystem); 1.5-2× ITL p50
potential. Not on the c=16 critical path — targets single-request
latency, not concurrent throughput — so not the next lever, but the
biggest after the mixed-path ladder caps out.

### E3. Metal parity for mixed-path + graph replay

CUDA mixed CUDA graph (ROI#2 C2) is CUDA-only by construction. Metal's
MLX compile graph is a separate mechanism; `crates/mlx-sys/` exposes a
`metal-dflash` GDR-kernel compile hook. Porting the "hoist allocs →
capture graph → bisect-replay" pattern from ROI#2 onto Metal for
Qwen3 mixed decode would close the c=16 Apple-Silicon gap. ~500-800
LoC; requires `mlx_compile_graph` + stable-pointer guarantees. Medium
priority: depends on Mac serving demand, which is lower than CUDA at
our concurrency shapes.

### E4. KV quant format support on the T1 tier

Gap #5 v1 is BF16-only. FP8/INT8/TurboQuant round-trip through T1
requires carrying the per-head scales and (for TurboQuant) rotation
state alongside the raw bytes. Extends `pages_host_byte_len` and the
H↔D helpers to emit / consume scale tensors. ~200 LoC + test matrix
across 4 formats. Unblocks long-session workloads where the live
pool runs quantized for capacity.

### E5. Multi-GPU / NIXL transport

`infer/src/kv_tier/transport/` has NIXL transport infrastructure
(T3 tier) but no consumer. Multi-GPU tensor-parallel serving is out
of scope for c=16 single-L4; NIXL wiring becomes load-bearing when
the model doesn't fit on one GPU (Qwen3-72B, Mixtral). Not this
quarter.

### E6. Admission-time KV budget accounting (gap #8)

`SchedulerConfig` tracks `rem_total_tokens` implicitly via
`admission_budget_tokens`, but the `PrefillAdder`-style per-iter
chunk budget (sglang's `rem_chunk_tokens`) isn't separated out.
Bursts on almost-full pools still trigger preemption instead of
clean backpressure. ~150 LoC, low risk, shares impact with ROI#3
and Gap #5 (pressure-triggered evict paths share victim pick logic).

## Open questions

Blocking decisions that will surface once ROI#2 C2 hits implementation:

- **Q1.** Graph cache memory budget. L4 24 GB headroom is ~2 GB after
  the KV pool; each captured graph costs 2-6 MB. At `max_bs × nt_buckets
  = 16 × 5 = 80 graphs`, cost is ~400 MB — workable. At `max_bs = 72
  × 5 = 360 graphs`, cost is ~1.4 GB — over budget. Gate behind
  `MIXED_GRAPH_MAX_TOTAL_MB` env, truncate high-bs / high-nt first.
  Decision: default 512 MB. Revisit if user reports > 72 slots.

- **Q2.** Padding strategy for mixed-step shape (`Σ c_i`). Three
  options: pad with real tokens (take from next prefill chunk),
  pad with dummy rows + attention mask, or drop to next smaller
  bucket. Plan chose real-token padding per A1 (dummy rows change
  kernel shape). Decision locked in
  `plans/roi2-mixed-cuda-graph.md` §Graph capture design.

- **Q3.** T1 demote ordering vs stream capture. When `Demote` fires
  during a captured mixed tick, the copy stream lives outside the
  compute stream. Per A5 this is safe (decoupled streams), but
  verify at C3 implementation time that no `cudaStreamWaitEvent`
  cross-stream barrier sneaks into the capture.

- **Q4.** Promote-back event → scheduler barrier. Gap #5 plan C4
  specifies the scheduler issues
  `compute_stream.wait_event(copy_event)` before admitting a just-
  promoted slot to prefill. That wait is in the admission hot path
  — worth measuring on real workloads before adding. Likely cheap
  (single cudaEventQuery), but not free.

## Sequencing

Recommended next-commit order, constrained by dependency chain:

1. **ROI#2 C2** — unblocks the real perf win the last 4 days of
   probes forecast. Biggest dollar value. ~350 LoC.
2. **Gap #5 C2** — coordinator Demote byte-path scaffold on top of
   dfb16bb. Not user-visible yet but unblocks C3/C4. ~150 LoC.
3. **ROI#2 C3/C4** — cap raise ladder. Depends on C2. Ship each
   raise as its own commit with bench gate. ~30 LoC + bench.
4. **Gap #5 C3/C4** — scheduler demote hook + promote-back. Ship
   behind `INFER_T1_DEMOTE_ENABLED=false` default until bench
   proves win on the repeated-prefix workload.
5. **E1 prefix-aware admission wire** — quick win once C4 flows
   T1 matches into `prefix_hit_tokens`.

Everything else (E2-E6) is post-parity. The quarter's budget is
finite; hold the post-parity axes until ROI#2 + Gap #5 fully land
and the measured c=16 position against sglang is locked.

## Review cadence

- Codex review on every commit that touches `infer/src/scheduler/`,
  `infer/src/model/qwen3/batch_decode.rs`, or
  `crates/cuda-kernels/src/paged_kv.rs`. It caught 2 P1/P2 issues
  on this ladder already (6aa5372, 7f0ce50).
- Bench gate on every commit per CLAUDE.md §Benchmarks. Drift-aware:
  compare same-day fresh builds, not historical absolute.
- Experience entries (`wins/` Pareto-neutral or positive; `errors/`
  regressions, bisect findings). This doc is the single pointer; it
  replaces future ad-hoc sglang-parity-v3 / v4 / v5 planning docs.

## Relation to other tracker docs

- `projects/tiered-kv-cache.md` — this doc extends it with Gap #5's
  HiRadixCache-style T1 demote/promote path. Not replacing it;
  treat as two views of the same underlying lifecycle.
- `projects/agent-first-architecture.md` — this doc does not
  replace its priority ledger; serving-performance is one lane
  among many (session routing, constrained decoding, spec decode).
- `projects/agent-rl-self-evolving.md` — orthogonal; the Phase 6
  training line runs on the same engine but does not constrain
  serving-performance architecture decisions.
- `research/2026-04-19-sglang-gap-analysis.md` — still the primary
  reference for the 8-subsystem survey. This doc is the forward-
  looking tracker; that one is the post-research snapshot.
