# Gap #5 — HiRadixCache-style KV tier demotion on eviction + prefix prefetch

**Branch:** `claude/c16-admission-gate-v2` · **Target:** L4 24 GB · **Workload:** Qwen3-4B BF16 · c=16 × 4096 × 256 · **Expected delta:** TTFT p99 cold-re-admit → warm (−50 % on re-admit path), overall TTFT p99 modest under churn.

## Goal

Replace the current "free GPU pages on prefix-cache eviction" path with a three-stage lifecycle: **demote** evicted warm radix blocks to the existing T1 host-pinned pool, **match** incoming admissions against T1 in addition to T0, and **promote back** when a match wins. Gate demotion with sglang's `write_through_threshold` so singletons don't thrash T1.

This is a **wiring task**. The T1 pool (`infer/src/kv_tier/host_pool.rs`) already allocates real `cuMemAllocHost_v2` pinned memory, the coordinator (`infer/src/kv_tier/coordinator.rs:339-398`) already runs a command loop with `Stage`/`Spill`/`Rehydrate` variants, `RadixCache::lookup_or_stage` (`infer/src/prefix_cache.rs:385-496`) already returns `StagingFromHost` hits and goes through a `StagePlanner`, and the scheduler's `drain_coordinator_events` (`infer/src/scheduler/cuda/runtime.rs:40-170`) already flips staged blocks back to `BlockLocation::Gpu { slot: u32::MAX }` on `StagingCompleted`. What's missing:

1. A **demote side** — today `evict_prefix_cache_if_pressured` (`infer/src/scheduler/cuda/core.rs:803-870`) drops the block's bytes outright (sentinel ticket `u64::MAX` at `:828`, then `release_pages` at `:856`). It never copies to T1.
2. A **T1 → T0 copy on `StagingFromHost` hits** — the scheduler's stage completion path (`runtime.rs:49-104`) treats staging as a synchronous metadata flip; it doesn't actually move bytes from the host pinned pool into a fresh GPU page.
3. A **hit-count gate** on which evictions produce T1 demotes (sglang `write_through_threshold = 2`). The `hit_count` field already lives on `prefix_cache.rs:Node` (`:100`) and `HitCountLru` (`policy.rs:271-297`) already mirrors the default.

**Effort check vs research doc's ~500 LoC / 3 files estimate.** The wiring crosses `coordinator.rs` (new `Demote` byte-path handler), `host_pool.rs` (one new helper to stamp a region with a kv-format tag), `scheduler/cuda/core.rs` (demote hook + promote-back accept), `runtime.rs` (event drain extension), and `paged_kv_pool`'s `copy_pages_to_host` / `copy_pages_from_host` (currently `todo!()` on CUDA — this is the actual blocker). **Revised estimate: 650–850 LoC across 5 files, Medium risk.** The `todo!()` pair alone is ~150 LoC of `cudaMemcpyAsync` on the coordinator stream; everything else is bookkeeping. This is bigger than the doc claims, and the risk is in that CUDA byte path, not in the scheduler glue.

## Existing infrastructure audit

| Component | File:line | What it already provides |
|---|---|---|
| `Tier` / `BlockLocation` enum | `infer/src/kv_tier/tier.rs:13-38` | T0 (`Gpu{slot}`), T1 (`HostPinned{offset}`), T2 (`Disk{file_id,offset}`), T3 (`Remote`). Ordering `Gpu<HostPinned<Disk<Remote>`. |
| T1 pool | `infer/src/kv_tier/host_pool.rs:83-287` | Real pinned alloc via `cuMemAllocHost_v2` (`:105-130`). Bump allocator + first-fit free list. `reserve(len)`/`release(region)` APIs. `host_ptr(region)` for DMA. MR-stability invariant preserved (single allocation, `Drop` calls `cuMemFreeHost` at `:289-311`). |
| Coordinator command loop | `infer/src/kv_tier/coordinator.rs:339-398` | OS thread, bounded crossbeam channels, `run_once` drain, spawn. Has `Stage`/`Spill`/`Rehydrate`/`Shutdown` variants (`:186-217`) and matching events (`:219-270`). **`Stage` handler today echoes `StagingCompleted` synchronously (`:372-374`) — no byte movement.** `Demote` variant is declared at `:187-191` but has no handler. |
| Radix cache with T1 awareness | `infer/src/prefix_cache.rs:82-119`, `:385-496`, `:1048-1054` | `Node.tier_location: Option<BlockLocation>` (`:103`). `Node.hit_count: u32` (`:100`, incremented on lookup hit at `:353` and `:429`). `lookup_or_stage` returns `StagingFromHost` when tier is HostPinned (`:434-441`). `set_block_location(BlockId, BlockLocation)` (`:1048`). |
| Stage planner wiring | `runtime.rs:281-290` | Scheduler passes `coordinator_handle` as `&dyn StagePlanner` to `lookup_or_stage`. |
| Stage completion event drain | `runtime.rs:49-104` | On `StagingCompleted`, flips `tier_location` to `Gpu{slot: u32::MAX}` (`:66-76`), releases stage-era refs but keeps `soft_pin_until`, requeues the admission. **No bytes are copied today — the bytes must already be at T0, which is only true because we never actually demoted them.** |
| Page lifecycle | `coordinator.rs:22-153`, `core.rs:835-854` | `Free`/`Resident`/`Demoting{ticket,target}` state machine. Sentinel ticket `StageTicket(u64::MAX)` (`core.rs:828`) represents "instant demote, no byte move" — this is the exact extension point. |
| Hit-count gate | `policy.rs:271-297`, `prefix_cache.rs:99-100, :353, :429` | `HitCountLru { hit_threshold: 2 }` default — mirrors sglang `write_through_threshold`. The threshold can be consulted at eviction time as "do I bother demoting?". |
| `write_through_threshold` analogue | Not yet a `SchedulerConfig` field. | Needs to be added (field + validation + default `2`). |
| `PagedKVPool::copy_pages_to_host` / `_from_host` | `crates/cuda-kernels/src/paged_kv.rs:514-545` | **`todo!("PagedKVPool::copy_pages_to_host requires validation on a CUDA host")`** under `feature="cuda"`. This is the real blocker. Must be implemented for the demote bytes path. |
| `LocalCudaTransport` | `infer/src/kv_tier/transport/local_cuda.rs:1-80`+ | Skeleton only. `submit()`, `poll()`, event pool, dedicated stream — all TBD per `tiered-kv-cache-coordinator-real-byte-path.md`. Demote needs `cudaMemcpyDeviceToHost` on the copy stream; promote-back needs `cudaMemcpyHostToDevice`. |
| `SchedulerConfig` T1 watermarks | `scheduler/types.rs:82-91`, defaults at `:121-123` | `t1_host_pinned_{high,low}_water` (0.85 / 0.70), `t1_host_pinned_keepalive_ticks` (128) — already wired, nobody reads them yet. |
| `PrefixAwareAdmission` trait | `scheduler/policy.rs:97-130` | Cold-throttle policy; **not wired** into the scheduler core (the scheduler still uses `QueueBoundAdmission` — see `types.rs:9`). |

## Demote-on-evict path

### Where the drop happens today

`infer/src/scheduler/cuda/core.rs:803-870` (`evict_prefix_cache_if_pressured`) and `:876-922` (`evict_prefix_cache_for_allocation`). For each evicted `BlockId`, the bytes go directly to free list via `paged_kv_pool.release_pages(&pages)` at `:856` and `:904`. Between picking the victim (`:822` `evict_with_policy`) and calling `release_pages`, the block's contiguous page span lives in `self.block_to_pages[bid]`. The block's tier_location node entry is already `Some(BlockLocation::Gpu{..})` (from `publish_to_prefix_cache` at `:745-747`).

### Block shape and copy size

The radix block is `PREFIX_CACHE_BLOCK_SIZE = 16` tokens (`core.rs:20`). One radix block spans `pages_per_block = block_size.div_ceil(pool.page_size)` pool pages (`core.rs:814-816`, typically 1 when page_size=16). Each pool page holds `page_size` tokens' worth of K/V across all layers. The byte length is already computed in `publish_to_prefix_cache` at `core.rs:718-722`: `block_byte_len = model.kv_cache_bytes_per_token() × block_size` (for Qwen3-4B with 36 layers × 2 K/V × 8 KV heads × 128 head_dim × 2 bytes/bf16 ≈ 147 KiB per 16-token block). That's already stored on the radix node (`set_block_byte_len` at `prefix_cache.rs:1057`, flowed from `publish_to_prefix_cache` at `core.rs:718-748`).

### New sequence

Replace the bytes-drop sequence with a copy-then-drop. In `evict_prefix_cache_if_pressured` around `core.rs:827-858`:

1. **Gate by hit_count (sglang `write_through_threshold`).** Read `prefix_cache.hit_count_of(bid)` (new accessor — `Node::hit_count` is private today). If `hit_count < config.t1_demote_min_hits`, go straight to `release_pages` (today's path). Otherwise:
2. **Reserve a T1 region.** Call `host_pool.reserve(block_byte_len)`. If `None` (T1 exhausted), fall back to today's free-outright path and bump a `t1_pool_exhausted_total` counter.
3. **Enter `Demoting{ticket, target: HostPinned{offset: region.offset}}`.** Use `page_lifecycle.begin_demote` (`coordinator.rs:97-119`) with a real minted ticket (not `u64::MAX`).
4. **Submit `CoordinatorCommand::Demote { block, from: Gpu{slot:..}, to: HostPinned{offset} }`** via `coordinator_handle.send`. New payload extension: the command must carry the page list so the coordinator can call `paged_kv_pool.copy_pages_to_host_async(pages, host_region, stream)`. Either (a) extend `Demote` with `pages: Vec<u32>` and `host_region: HostPinnedRegion`, or (b) wire demote through the existing `Stage` shape but with explicit direction — option (a) is cleaner; `Stage` is already overloaded for T1→T0 rehydrate in `drain_coordinator_events`.
5. **Flip `tier_location` to `HostPinned{offset}` in the radix** via `prefix_cache.set_block_location(bid, BlockLocation::HostPinned{offset})`. This is done **after** the copy completes, on the `DemoteCompleted` event (new). Until then, the node keeps the `Gpu{slot}` location **but with slot = u32::MAX sentinel** so no scheduler path tries to reuse it (mirrors the stage-in-flight flip at `runtime.rs:66-76`).
6. **Release pool pages.** `paged_kv_pool.release_pages(&pages)` runs **after** the D→H copy is in flight. Safe because the copy is reading from the physical page pointers, not the pool's free-list state; the pool's page-ref-count bookkeeping owns whether the page is "in limbo" (`paged_kv.rs:570-577`). We must ensure `release_pages` does not reuse the page for a new alloc until the D→H `cudaEventQuery` returns Ready. Two ways to enforce:
   - (a) **Easy:** keep the radix ref pinned (refcount stays > 0) until DemoteCompleted, at which point we call `release_pages`.
   - (b) **Hard:** release immediately, rely on the pool's free-list FIFO ordering + copy stream ordering. Fragile because an unrelated admission can realloc the page before the D→H copy drains.

   **Pick (a).** Radix ref stays pinned across the demote window; `finish_demote` in the event handler triggers `release_pages`.

### Async vs sync relative to scheduler tick

**Async.** sglang's HiRadixCache uses a background CUDA copy stream (`hiradix_cache.py:606`). Our `LocalCudaTransport` (per the coordinator-real-byte-path plan `docs/plans/tiered-kv-cache-coordinator-real-byte-path.md:104-127`) is specced to own a dedicated `cudaStreamNonBlocking` stream. The scheduler tick enqueues the demote command and returns; the coordinator drains it, issues `cudaMemcpyAsync(DtoH, copy_stream)`, records an event, and polls in `run_once`. On event completion, it emits `DemoteCompleted{ticket, block_id, host_region}` — scheduler drains that in `drain_coordinator_events` and finalizes the radix flip + page release.

This matches the existing `Spill`/`Rehydrate` shape (`coordinator.rs:203-215`, events at `:232-270`). **Do not open a second stream.** The same copy stream services both demote (D→H) and promote-back (H→D); per-ticket event serialization keeps ordering correct. The real byte-path plan §4 open question #1 also lands on "one stream" as the v1 answer.

### Radix-cache node granularity

**Per-block** (= per 16-token unit). The radix already stores `tier_location: Option<BlockLocation>` **on each Node that carries a `block_id`** (`prefix_cache.rs:100-103`). No new state needed — we're just writing `HostPinned{offset}` instead of `Gpu{slot}`. The `byte_len` field is already populated at publish time (`core.rs:718-722`, `:748`) and equals exactly `block_byte_len` — it's what we reserve in T1.

## Promote-back path (T1 → T0)

### Current `match_prefix` call site

Single call: `infer/src/scheduler/cuda/runtime.rs:286-290` inside `assign_slots`. Receives a `LookupOutcome { matched_len, blocks, staging_ticket, recompute_advised }` (`kv_tier/lookup.rs:44-53`). Today, any `StagingFromHost` block is routed to the `StagePlanner::stage` path (`prefix_cache.rs:489-493`), which returns a `StageTicket`. The scheduler stashes the admission in `stage_waiting` (`runtime.rs:320-328`) until `StagingCompleted` fires.

**The completion path does not copy bytes.** `runtime.rs:66-76` simply re-flips `tier_location` to `Gpu{slot: u32::MAX}` on the assumption that the bytes are already at T0. That assumption holds only because today we never demote, so `StagingFromHost` is unreachable in practice. This is the hidden contract break that has to be closed before shipping.

### Minimal new return signature

Our `LookupOutcome` already carries what we need — it has a per-block `LookupBlock { block_id, hit_kind }` array (`lookup.rs:25-29`). No signature change required. sglang's `(matched_tokens, last_device_node, last_host_node)` triple maps to:
- `matched_tokens` = `LookupOutcome.matched_len`
- `last_device_node` = index of the last `HitKind::ReadyOnGpu` block in `blocks`
- `last_host_node` = index of the last `HitKind::StagingFromHost` block in `blocks`

Those derived views can be computed by a small inline helper (`last_host_node_index(&blocks)`) in `runtime.rs`; no type widening.

### Real promote-back wiring

The demote side only changed **what the radix records**. The promote-back wire must actually copy bytes. Replace the `StagePlanner::stage` implementation on `CoordinatorHandle` (`coordinator.rs:323-337`) so the `Stage` command, when processed (`coordinator.rs:365-375`), does:

1. For each `StageRequest { block_id, from: HostPinned{offset}, byte_len }`:
   - Allocate `pages_per_block` pool pages via `paged_kv_pool.alloc_detached_pages(pages_per_block)` — already exists (`crates/cuda-kernels/src/paged_kv.rs:490-511`). The pool detaches pages from the free list without binding them to a slot.
   - Submit `cudaMemcpyAsync(HtoD, dst=page_physical_ptr, src=host_region_ptr, byte_len, copy_stream)`.
   - Record an event, stash an in-flight record `{ ticket, block_id, new_pages, host_region }`.
2. On event completion, emit `StagingCompleted { ticket }`. The scheduler's existing drain (`runtime.rs:49-104`) flips `tier_location` to `Gpu{slot: u32::MAX}` — the slot is overwritten to a real slot index by `publish_to_prefix_cache` when admission picks up the request.
3. On completion, also **release the T1 region** via `host_pool.release(host_region)` (T1 region recycled; bytes still valid until overwritten).

### Stream, timing, scheduler interaction

- **Stream:** same dedicated copy stream as demote. Copy stream is owned by `LocalCudaTransport`, independent from the model compute stream (`model.device_context().stream`). This matches the project doc §4.4 "dedicated copy stream" requirement.
- **When launched:** at admission time, as today. `assign_slots` calls `lookup_or_stage`, which internally calls `StagePlanner::stage`. The coordinator picks up the `Stage` command from its channel on the next `run_once` tick (≤1 ms timeout at `coordinator.rs:362`).
- **Compute-stream barrier:** no barrier on compute until the admission gets picked back up. The actual compute (first prefill step on those pages) runs much later (next tick after completion event), by which point the H→D copy event has fired — if not, we'd `cudaStreamWaitEvent(compute_stream, copy_event)` before launching prefill. Safest to add that wait at the model forward entry for any slot whose pages were newly promoted. One simple implementation: the scheduler, after taking a slot off `stage_waiting`, calls `model.device_context().stream.wait_event(copy_event_for_ticket)` before admission. This adds one event-wait to the admission hot path — cheap.

### What if T0 is full when promote-back needs pages

`paged_kv_pool.alloc_detached_pages(pages_per_block)` returns `Err` when the pool is out of free pages. Handling options:

1. **Refuse to promote, fall through to cold prefill.** Simple. The existing `LookupHeuristics::advise_recompute` already does this bandwidth-vs-recompute math at lookup time (`lookup.rs:92-115`). If `recompute_advised` is set, the scheduler already releases stage refs and cold-admits (`runtime.rs:297-298`). We extend the same check: if alloc fails mid-stage, the coordinator emits a new `StagingFailed { ticket, reason }` event and the scheduler cold-requeues.
2. **Evict something first.** Call `evict_prefix_cache_if_pressured` / `evict_prefix_cache_for_allocation` from inside the coordinator's stage handler — but this is a **scheduler-owned** operation (RadixCache is scheduler-thread-only per `kv_tier/AGENTS.md` invariant #5). Must not happen.

**Pick option 1** for v1. Admission-time budget check is already done by `admission_budget_tokens` (`core.rs:204-232`) — the T1 pages reserved by the prefix hit are already accounted for (they're "evictable tokens" included in `free_count + evictable_token_count`). If the budget says admit, we try the alloc; if it fails, we cold-fall-through.

## Prefetch policy

### Threshold policy

- **`t1_demote_min_hits: u32` on `SchedulerConfig`, default 2** (sglang parity). Validation: `≥ 1`; at 1 every eviction demotes (parity with "always write-through"), at 2 only blocks reused at least once demote.
- **Read access:** `evict_prefix_cache_if_pressured` / `evict_prefix_cache_for_allocation` both need to know a block's hit_count at eviction time. Add `RadixCache::hit_count_of(bid: BlockId) -> Option<u32>` — O(1) via the existing `block_index` (`prefix_cache.rs:185-186`).
- **No per-request override** in v1. The threshold is global; sglang doesn't expose it per-request either.

### T1 capacity and eviction down-stack

- **T1 sizing:** `HostPinnedPool::new(capacity_bytes)` — needs to be allocated at engine init. No constructor call in the scheduler today; must be added alongside the existing `TokenKVPool` alloc in `Scheduler::with_config` (`core.rs:390-422`). Default: `min(4 GiB, 2 × T0 pool budget / 8)` — about 1.5 GB at L4 with the default `mem_fraction_static = 0.88`. Expose `SchedulerConfig::t1_host_pinned_bytes: usize` (new field, default 2 GiB, validate `> 0`).
- **T1 → T2 spill:** the `t1_host_pinned_{high,low}_water` watermarks (`types.rs:82-86`) already exist. A new `check_t1_pressure_and_spill()` runs in `cleanup()` alongside `evict_prefix_cache_if_pressured` (`core.rs:803`), issues `CoordinatorCommand::Spill` for LRU T1 blocks, and awaits `SpillCompleted` to flip `tier_location` to `Disk{file_id, offset}`. **The coordinator-real-byte-path plan already sizes and specs this** (`docs/plans/tiered-kv-cache-coordinator-real-byte-path.md:185-214`). This gap-5 plan depends on that plan shipping first, OR inlines the T1→T2 half as a v1.1 follow-up if the coordinator plan slips.

### When prefetching hurts

- **Burst of unique new requests → T1 thrash.** A burst of never-seen 4 k prompts under the c=16 workload will fill T0, evict the long-running session blocks to T1, then when those sessions come back they've already been spilled from T1 to T2 (or dropped). `write_through_threshold ≥ 2` mitigates: first-time blocks don't even enter T1. But a burst of hits on the SAME prompt from 16 clients could still thrash. Mitigation: `HitCountLru` protects hot blocks from T0 eviction in the first place (already landed in `policy.rs:271-297`), but **it isn't wired as the default eviction policy**: `core.rs:822` uses `SessionBiasedLru::default()`. Follow-up (not in this plan's scope): flip the default to a composite policy or pass `HitCountLru` as the secondary tier.
- **Demote cost during peak TTFT.** Demote runs on the copy stream, not compute, so it doesn't steal compute FLOPs. But D→H over PCIe is ≤20 GB/s on L4 (per `lookup.rs:82-84` heuristic). A 147 KiB block demotes in ≤10 µs — cheap. Worst-case: evict 100 blocks at once on a watermark hit → 1 ms of copy queue. Batched into one `Demote` command with multiple blocks, the per-call overhead amortises. **Batch every watermark-triggered demote pass into a single `Demote{blocks: Vec<_>}` command.**

## Interaction with existing invariants

### MR stability (`kv_tier/AGENTS.md:48-54`)

- **T1 stays stable.** `HostPinnedPool` is allocated once (`host_pool.rs:91-142`), never reallocated. Regions are handed out as offsets into the same base pointer. Promote-back returns the region to the free list (`host_pool.rs:196-204`); demote reserves a new region. No MR re-registration required — the pool itself is the MR.
- **Promote-back does NOT introduce an "evicted MR comes back with new data" issue.** The MR is the host-pinned pool base, not a per-block region. A per-block region being reused for different content across its lifetime is fine — MR registration applies to the base pointer, not to what lives at offset `X`.
- **No escape hatch needed.** The original concern (new MR identity on T1→T0 copy) applies to NIXL/remote transports, not to local cudaMemcpy.

### RadixCache `lock_ref > 0` invariant (`core.rs:96-114`)

- **Today:** `paged_kv_pool.page_ref_count` tracks external refs. `publish_to_prefix_cache` bumps it via `retain_pages` (`core.rs:716`); `release_pages` drops it. While refcount > 0, `free_slot` leaves the page in limbo (`paged_kv.rs:570-577`).
- **After this plan:** the radix node still owns one ref across the Demoting window. When the T1 copy completes, we call `release_pages` (ref drops), but the radix node now has `tier_location = HostPinned{offset}` with no GPU-page backing. The reverse map `block_to_pages` must lose the entry for this bid (it's invalidated — page is free again, could be allocated to a new slot). New invariant: **when `tier_location == HostPinned`, the bid has no `block_to_pages` entry.** Consistency: `block_to_pages.remove(bid)` happens in the `DemoteCompleted` handler, just before `release_pages`.
- **Promote-back** allocates fresh pages via `alloc_detached_pages`, populates `block_to_pages[bid] = new_pages`, and the existing `retain_pages` path increments refcount. Same invariant holds.
- **No dangling ref risk** as long as the demote handler serialises `block_to_pages.remove` → `release_pages` on the scheduler thread (single-writer by construction).

### `PrefixAwareAdmission` trait (`policy.rs:97-130`)

- **Unwired today.** Scheduler always uses `QueueBoundAdmission` (`types.rs:9` + wherever the default is constructed).
- **T1 match should affect admission scoring.** sglang's `PrefillAdder` uses cached-prefix length as a speedup bonus when ordering the waiting queue. We already pass `SchedulerSignals.prefix_hit_tokens` (`policy.rs:28-31`) into `PrefixAwareAdmission::allow`. Gap 5 makes T1 matches count toward `prefix_hit_tokens` — previously only T0 hits counted (`runtime.rs:334-338`). The wire is: in `assign_slots`, if `lookup.matched_len > 0` and any blocks have `StagingFromHost`, still report `prefix_hit_tokens = lookup.matched_len` (not 0). This is a one-line tweak but it's a behaviour change — **call it out as a separate commit** so regression bisect is clean.
- **No extra admission-gate semantic needed for gap 5.** The scheduler can continue to use `QueueBoundAdmission`; if we want the T1-aware warm preference, that's gap #8 territory.

## Concurrency + correctness risks

### Demote stream vs compute stream

- **Problem:** the radix-cache must not advertise a T1 location (`StagingFromHost` hit) until the D→H copy is complete. If it does, a promote-back could race, reading stale data.
- **Solution:** stamp `tier_location = HostPinned{offset}` only on `DemoteCompleted`. Until then, the node keeps `tier_location = Gpu{slot: u32::MAX}` (or clears to `None`, forcing a cold miss on any interim lookup). **Pick: keep `Gpu{slot: u32::MAX}`** — matches the existing sentinel pattern at `runtime.rs:69-72`. `lookup_or_stage`'s match on `Gpu{..}` routes that to `HitKind::ReadyOnGpu` but the actual pool pages are in limbo, so a sibling request with an interim hit gets `ReadyOnGpu` bytes that... actually, this is a subtle bug. Let me re-check: during the demoting window, the pool pages are still alive (refcount kept > 0 by radix), so `ReadyOnGpu` is actually correct for a mid-demote hit. The copy to T1 doesn't invalidate the source. Once `DemoteCompleted` fires, we release_pages, which drops refcount and frees the page — at that point `Gpu{slot}` would go stale. **So the flip to `HostPinned{offset}` must happen atomically with `release_pages`** in the `DemoteCompleted` handler. Both run on the scheduler thread, both are single-writer operations on radix and pool — the atomicity is trivial.
- **CUDA event sync:** cudaEventRecord on copy stream after the last memcpy. `cudaEventQuery` returns Ready when the stream is drained past the event. This is what the coordinator's `run_once` polls. No additional fence needed.
- **No flag on the radix node.** The `tier_location` enum itself is the flag — `Gpu{slot:u32::MAX}` vs `HostPinned{..}` encodes the lifecycle.

### Promote-back while a sibling mid-prefill has partial overlap

- **Possible?** Request A is mid-prefill on slot S1 with its first 48 tokens already written to pool pages `P1, P2, P3`. Request B admits with the same 16-token prefix; radix already has a T1 entry for `P0` (demoted from a prior session). B gets a `StagingFromHost` hit on block 0 only; B's slot S2 gets a fresh page for that block. **Not the same pages as S1.** No aliasing. Safe.
- **Excluded by design:** cross-slot page aliasing is already unsupported (`core.rs:139-142` comment: "cross-slot page aliasing is intentionally unsupported"). Promote-back always allocates fresh pages; it never binds to a live slot's pages.

### T1 evict to T2 while promote-back holds a T1 region

- **Refcount on T1 nodes.** T1 isn't the issue — the T1 pool region doesn't have a refcount today (`host_pool.rs:66-71`: just capacity + free list + backing). But a T1 region is "in use" as long as the radix node's `tier_location == HostPinned{offset}`. T1→T2 spill targets radix nodes with a `HostPinned` tier; it flips them to `Disk` and releases the region.
- **Race:** scheduler picks a radix node for spill (T1→T2), coordinator starts D→?→Disk I/O (actually it's just host→disk, `DiskStore::put_block`), meanwhile another admission does `lookup_or_stage` on that block and gets `StagingFromHost`. The radix node's tier_location says HostPinned, but it's on its way to becoming Disk.
- **Mitigation:** same refcount on T1 as on T0. Extend `Node.ref_count` to span T1 too — when a `StagingFromHost` admission pins the node, refcount is incremented (already happens at `prefix_cache.rs:428`). `evict_with_policy`'s `pinned` check at `prefix_cache.rs:914` already excludes `ref_count > 0` nodes. Spill must go through the same eviction policy path, so a pinned-T1 node cannot be picked for spill. **This means T1 spill cannot use a separate code path — it must call `prefix_cache.evict_with_policy` with a "T1 residents only" filter.** Add that as a follow-up (tiered-kv-cache-coordinator-real-byte-path.md #4-#6 already specs this).
- **v1 scope:** T1→T2 spill is out of this plan's scope. T1 pool becoming exhausted just means new demotes are refused (the watermark plan in the coordinator doc handles the downstream). This plan only needs to avoid writing corrupt bytes on promote-back; the spill interaction is isolated in the coordinator plan.

### Scheduler tick ordering — latency in `free_slot`

- **Today:** `free_slot` (`paged_kv.rs:565-580`) is pure bookkeeping — O(pages_in_slot), no copies. The eviction-heavy function is `evict_prefix_cache_if_pressured`, called **only at end of `cleanup()`** (`core.rs:800-801`), not on the hot admission path.
- **After:** the demote hook issues `coordinator_handle.send(Demote{..})`. That's a crossbeam channel send — non-blocking, O(1). No added latency to the scheduler tick. The real work (D→H copy) runs on the coordinator thread + copy stream, asynchronously.
- **`evict_prefix_cache_for_allocation` (`core.rs:876-922`) is called on the hot path** (from `alloc_pool_tokens_with_retry`, on OOM). We should **skip demote here** — OOM reclaim is latency-critical, copying to T1 adds PCIe bandwidth pressure. Demote only from the watermark-triggered evict path. Add a `demote_on_evict: bool` parameter to a shared helper, `true` for watermark, `false` for OOM-retry.
- **No regression expected on TTFT at eviction time.** Same scheduler-thread cost; just one extra channel send per evicted warm block.

## Observability

New counters on `ServerMetrics` (`infer/src/metrics.rs:163-191`):

- `t1_demotes_total: AtomicU64` — number of D→H copies completed.
- `t1_promotes_total: AtomicU64` — number of H→D copies completed (= `StagingCompleted` with real bytes).
- `t1_bytes_demoted_total: AtomicU64` — cumulative bytes copied D→H.
- `t1_bytes_promoted_total: AtomicU64` — cumulative bytes copied H→D.
- `t1_evictions_gated_total: AtomicU64` — evictions skipped because `hit_count < threshold`.
- `t1_pool_exhausted_total: AtomicU64` — demotes that fell through because `host_pool.reserve` returned None.
- `t1_bytes_resident: AtomicU64` (gauge) — `host_pool.reserved_bytes()`, refreshed each cleanup tick.
- `t1_hit_rate` — derived: `t1_promotes_total / (t1_promotes_total + prefix_misses_that_had_t1_candidate)`. Approximate; for v1 expose raw counters and let callers compute.

Expose via `/v1/stats` render path (`metrics.rs:590+` — `render_summary`).

DEBUG logs:
- Demote: `t1_demote: bid={} size={}B host_offset={} ticket={}` at submit; `t1_demote_done: bid={} ticket={} elapsed_us={}` at event completion.
- Promote: `t1_promote: bid={} host_offset={} → pages={:?} ticket={}` at submit; `t1_promote_done: bid={} ticket={} elapsed_us={}` on completion.
- Gated: `t1_demote_gated: bid={} hit_count={} threshold={}` on skip.

## Acceptance + bench

### Unit tests

1. **`coordinator_demote_routes_bytes`** — `Coordinator::new` with a `LocalCudaTransport` stub that records calls. Emit `Demote{block, from: Gpu{slot:0}, to: HostPinned{offset:0}, pages: vec![0], host_region}`. Assert the stub records one D→H op and emits `DemoteCompleted` after `poll` reports Ready.
2. **`radix_cache_demote_then_match_returns_t1`** — insert a block, `set_block_location(bid, HostPinned{offset:4096})`, `set_block_byte_len(bid, 8192)`, then `lookup_or_stage`. Assert matching block has `HitKind::StagingFromHost` and the returned `StageTicket` is `Some`.
3. **`t1_demote_min_hits_gates_demote`** — configure threshold=2, simulate eviction of a block with `hit_count=1`. Assert no `Demote` command fired (check the `t1_evictions_gated_total` counter bumps, coordinator's command channel stays empty).
4. **`promote_back_round_trip_bytes`** — `feature="cuda"` test: alloc a pool page, write a pattern, `copy_pages_to_host` into a host region, free the page, `alloc_detached_pages`, `copy_pages_from_host` from the same region, assert byte-equal round trip. Blocked on the `todo!()` pair at `paged_kv.rs:514` — landing this test is the same commit that implements the copy helpers.

### Integration tests

1. **`c=2 retract-then-readmit`** (`infer/tests/e2e.rs` sibling): submit request A (prompt P, 256 decode). Artificially force retract mid-decode (`retract_longest_decode` via test hook, currently `pub(super)` — may need a test-only accessor). Assert A's prompt-prefix blocks are demoted to T1 (counter bumps). Re-admit request B with the same prompt P. Assert B's first token TTFT < 50 % of A's first token TTFT (warm promote vs cold prefill).
2. **`repeated_prefix_c16`**: 16 clients submit the same 4 k-prompt prefix (varied suffixes). First 16 pay cold prefill, next 16 (after first wave finishes) should hit T1 for the shared prefix. Assert `t1_promotes_total > 0` and TTFT p99 on the second wave is lower than on the first. Needs a new bench-scale test harness — can slot under `infer/tests/` as an integration test gated on CUDA.

### Bench ladder (`scripts/bench_guidellm.sh`)

- **Baseline:** c=16 × 4096 × 256 against current main (post-ROI#3). Snapshot to `wins/2026-XX-XX-bench-guidellm-gap5-before.md`.
- **After each commit (per CLAUDE.md "no half-states" rule):** re-run the same sweep. Expected deltas:
  - C1 (demote side only, promote still flips metadata): **null delta** — nothing reads T1 yet. Safety bench.
  - C2 (promote-back real bytes): modest TTFT p99 gain on repeat workloads; flat on guidellm's random-prompt mix (no repeat prefixes).
  - C3 (hit_count gate): flat on random-prompt mix (gate suppresses all demotes, degenerates to today's behaviour); gain on repeat mix.
- **Propose a "repeated-prefix" bench variant.** Current guidellm mix has too few prefix collisions to stress T1. Add `scripts/bench_guidellm_repeated_prefix.sh` with a prompt-mix that repeats a 2 k-token system-prompt header across 80 % of requests (mirrors sglang's own HiRadix benchmark fixture `sglang/benchmark/hicache`). This is essential — without it, gap 5's win doesn't show up in canonical numbers.

### Non-CUDA gate

The no-cuda lane (`HostPinnedPool` falls back to `Backing::InMemory` at `host_pool.rs:131-134`) should fully exercise the coordinator command plumbing and the radix state machine. CI `cargo test --release --no-default-features --features no-cuda` must stay green. The CUDA byte-path tests are gated on a real GPU; stub them under `#[cfg(not(feature = "cuda"))]`.

## Commit sequence (5 commits, each compiles + tests + benches independently)

Each commit gated behind a feature flag or `SchedulerConfig` default so a regression can be reverted without rolling the whole batch. The gating env var is `INFER_T1_DEMOTE_ENABLED`, default `false` until C5.

1. **C1 — `PagedKVPool::copy_pages_{to,from}_host` real CUDA impl.** Unblocks the byte path; pure kernel-adjacent code.
   - Files: `crates/cuda-kernels/src/paged_kv.rs:514-545`.
   - Replace `todo!()` with real `cudaMemcpyAsync` against a passed stream. Return `Result<Vec<u8>>` / `Result<()>`.
   - Bench: unit test round-trip only; no scheduler change, no guidellm run (doc-exempt commit per CLAUDE.md §Benchmarks).

2. **C2 — Coordinator `Demote` + `DemoteCompleted` byte path.** Wire `LocalCudaTransport` (per tiered-kv-cache-coordinator-real-byte-path.md Phase A) and make `Demote` actually copy. Nothing in the scheduler calls this yet.
   - Files: `infer/src/kv_tier/transport/local_cuda.rs`, `infer/src/kv_tier/coordinator.rs`, new `DemoteCompleted`/`DemoteFailed` events.
   - Feature-flagged behind `INFER_T1_DEMOTE_ENABLED=false`; coordinator path exists but scheduler never sends the command.
   - Unit test: `coordinator_demote_routes_bytes`.
   - Bench: guidellm c=16 baseline; expected flat. Commit the snapshot as regression-check per CLAUDE.md.

3. **C3 — Scheduler demote hook + T1 reserve + radix tier_location flip on evict.** `evict_prefix_cache_if_pressured` consults `t1_demote_min_hits` and issues `Demote` for eligible blocks. `drain_coordinator_events` handles `DemoteCompleted` by flipping `tier_location` to `HostPinned` and calling `release_pages`.
   - Files: `infer/src/scheduler/cuda/core.rs:803-870`, `infer/src/scheduler/cuda/runtime.rs:40-170`, `infer/src/scheduler/types.rs` (new `t1_demote_min_hits: u32` field, default 2; `t1_host_pinned_bytes: usize`, default 2 GiB).
   - Also: `HostPinnedPool::new(config.t1_host_pinned_bytes)` alloc at engine init (`core.rs:~390`).
   - Still gated: `INFER_T1_DEMOTE_ENABLED=false` default skips the `Demote` submit and goes through today's `release_pages`-outright path.
   - Unit tests: `t1_demote_min_hits_gates_demote`, `radix_cache_demote_then_match_returns_t1`.
   - Bench: guidellm with flag on and off; expect parity on both (demote happens but nothing reads T1 yet).

4. **C4 — Promote-back design + accessor only.** Initial attempt at
   scheduler-owns-copy promote-back at the `StagingCompleted` event drain
   leaked HBM (codex caught: pages inserted into `block_to_pages` without
   a `block_owner_slots` entry → next cold-prefill rewrites radix node →
   pages orphaned with refcount > 0). **Reverted byte path; shipped
   accessor + design-gap analysis only.** See
   `docs/plans/gap5-c2-byte-path-architecture.md` §"C4 design gap".
   - Files actually shipped: `infer/src/prefix_cache.rs` (`tier_location_of`
     accessor), `docs/plans/gap5-c2-byte-path-architecture.md` (gap doc),
     this plan (status update).
   - Bench: exempt — no runtime behaviour change (StagingCompleted arm
     unchanged from pre-attempt state; `t1_demote_min_hits=0` default
     keeps C3's demote hook dormant so the metadata-flip path is sound).

4.5. **C4.5 — Slot-bound promote-back (the real C4).** Move the H→D copy
   into `assign_slots` so promoted pages graft onto the request's slot
   page list before `step_new`'s alloc — pages have an owner before
   `publish_to_prefix_cache` rewrites the radix node, eliminating the
   orphan path.
   - New `PagedKVPool::attach_detached_pages_to_slot(slot_idx, pages)`
     primitive — splices pages into the slot's page_indices Vec at offset 0,
     advances `seq_len`, marks pool refcount as already-bumped.
   - `Scheduler::assign_slots` reorder: pick slot → lookup_or_stage →
     promote+graft if any HostPinned matched → call `step_new` with a
     "skip-prefix-alloc" hint covering the just-grafted prefix.
   - Files: `crates/cuda-kernels/src/paged_kv.rs` (new primitive),
     `infer/src/scheduler/cuda/core.rs` (assign_slots reorder + graft),
     `infer/src/scheduler/cuda/runtime.rs` (StagePlanner::stage shape may
     change to return promoted-pages map), `infer/src/scheduler/cuda/decode.rs`
     (step_new prefix-skip hint).
   - Unit tests: `attach_detached_pages_to_slot_splices`, `promote_back_round_trip_bytes_with_slot_owner`.
   - Integration: `c=2 retract-then-readmit` end-to-end.
   - Bench: c=16 with `--t1-demote-min-hits 2`. Expected modest TTFT p99 gain on repeat-prefix.
   - Estimated scope: ~300-400 LoC across 4 files.

5. **C5 — Flip default, add `/v1/stats` counters, repeated-prefix bench.** Default `INFER_T1_DEMOTE_ENABLED=true`. Expose the 7 new counters in `ServerMetrics`. Land `scripts/bench_guidellm_repeated_prefix.sh` and the `wins/` entry comparing pre/post.
   - Files: `infer/src/metrics.rs`, `infer/src/http_server.rs` (stats render), `scripts/bench_guidellm_repeated_prefix.sh`.
   - Bench (**the one that matters**): repeated-prefix sweep, pre/post. Commit `docs/experience/wins/YYYY-MM-DD-bench-guidellm-gap5-t1-demote-promote.md`.

Gating knob for fast revert: `INFER_T1_DEMOTE_ENABLED=false` restores today's behaviour across all 5 commits (C1/C2 are side-effect-free; C3–C5 are flag-gated at the demote-issue call site in `evict_prefix_cache_if_pressured`).

## Collision points with ROI#2 (mixed CUDA graph)

Both plans touch scheduler state around the same ticks but in orthogonal regions:

- **ROI#2 touches:** `scheduler/cuda/decode.rs` (mixed tick chunk plan), `scheduler/cuda/core.rs:1113-1316` (warmup graphs), `model/qwen3/batch_decode.rs` (graph cache widening), `model/qwen3/forward.rs` mixed entry.
- **Gap 5 touches:** `scheduler/cuda/core.rs:803-922` (eviction paths), `scheduler/cuda/runtime.rs:40-170` (event drain), `scheduler/cuda/core.rs:~390-440` (init alloc), `kv_tier/coordinator.rs`, `kv_tier/transport/local_cuda.rs`, `crates/cuda-kernels/src/paged_kv.rs:514-545`.
- **Overlap:** both touch `scheduler/cuda/core.rs`, but ROI#2 is around `warmup_cuda_graphs` + `prefill_chunk_size` (`:1063`-`:1317`) and gap 5 is around eviction (`:803-:922`) and init (`:390-:440`). **Zero line-number collisions.** The `drain_coordinator_events` extension and the mixed-path tick changes are in different files.
- **Merge risk:** low. Possible conflict in `SchedulerConfig` struct field list (`types.rs:27-94`) if both batches add fields simultaneously — resolve by alphabetising field additions. Also: `pending_mixed_prefill_idxs` (`core.rs:181`) is set by mixed-path code ROI#2 generalises; gap 5 doesn't read it.
- **Stream ownership:** ROI#2 uses the model compute stream (graph capture/replay bound to it). Gap 5 uses the coordinator's copy stream (new). Stream independence is load-bearing for both plans.

**Recommended sequencing:** land ROI#2 first if practical (it's pure perf, no persistent state change), then gap 5. But they can land in parallel if the `SchedulerConfig` field alphabetisation is enforced in both branches.

## Risk summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| `PagedKVPool::copy_pages_{to,from}_host` ships buggy (wrong layout, wrong stride, byte-swap) | Medium | Unit test round-trip with known pattern. Compare against `publish_to_prefix_cache`'s existing page layout (`core.rs:610-616`). |
| D→H copy races H→D promote for the same bid | Low | Serialise by StageTicket; page ref held by radix across the window. |
| T1 pool sizing misconfigured on small-GPU hosts | Medium | Make `t1_host_pinned_bytes` default a fraction of `free_ram` rather than an absolute 2 GiB; fall through to "no T1" if alloc fails. |
| Hit-count threshold too conservative → no demotes in practice | Low | Threshold=1 toggle for debug; bench the default=2 vs =1. |
| Repeated-prefix bench not representative of real workloads | Medium | Treat win as workload-specific; don't gate shipping on the canonical guidellm sweep showing a gain there. |
| Coordinator thread lag turns "async" demote into synchronous bottleneck | Low | `run_once` already polls the bounded command channel with 1 ms timeout (`coordinator.rs:362`); batch demote commands to amortise the poll. |

## Critical files for implementation

- `/content/workspace/agent-infer/infer/src/kv_tier/coordinator.rs`
- `/content/workspace/agent-infer/infer/src/kv_tier/transport/local_cuda.rs`
- `/content/workspace/agent-infer/infer/src/scheduler/cuda/core.rs`
- `/content/workspace/agent-infer/infer/src/scheduler/cuda/runtime.rs`
- `/content/workspace/agent-infer/crates/cuda-kernels/src/paged_kv.rs`

Also loaded for context (non-critical):
- `/content/workspace/agent-infer/infer/src/prefix_cache.rs` (radix lookup + tier_location; read-mostly, small touch)
- `/content/workspace/agent-infer/infer/src/kv_tier/host_pool.rs` (no structural change; one accessor)
- `/content/workspace/agent-infer/infer/src/scheduler/types.rs` (2 new config fields)
- `/content/workspace/agent-infer/infer/src/metrics.rs` (7 new counters)
