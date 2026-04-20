# Gap #5 C2 — byte-path architecture decision

**Parent plan:** `docs/plans/gap5-kv-tier-demote-prefetch.md` · **Date:** 2026-04-20 · **Decision:** scheduler-owns-copy (Option A)

## The fork

Gap #5 C1 (`dfb16bb`) landed `PagedKVPool::copy_pages_{to,from}_host` — validated round-trip in `crates/cuda-kernels/tests/paged_kv_copy_pages_roundtrip.rs`. Gap #5 C2 wires the actual demote byte movement. Two structural options:

### Option A — scheduler owns the D→H copy

```
scheduler.cleanup()
  └─ evict_prefix_cache_if_pressured
     └─ for victim block:
        ├─ PagedKVPool::copy_pages_to_host(pages, compute_stream) → Vec<u8>
        ├─ HostPinnedPool::reserve(len) → HostPinnedRegion
        ├─ memcpy bytes into region (CPU, ~50 μs / 147 KiB)
        ├─ RadixCache::set_block_location(bid, HostPinned{offset})
        ├─ block_to_pages.remove(bid)
        ├─ release_pages(&pages)
        └─ CoordinatorHandle::submit_demote(block, offset) — pure telemetry
```

Coordinator's role: **observability only**. It receives `Demote{block, from, to}`, emits `DemoteCompleted` for metrics / debug logs. No byte movement, no pool mutation.

### Option B — coordinator owns the D→H copy

```
scheduler.cleanup() → submit Demote{pages, host_region} → return

coordinator thread
  ├─ pool_handle.lock() — Arc<Mutex<PagedKVPool>>
  ├─ cudaMemcpyAsync(DtoH, copy_stream)
  ├─ cudaEventRecord, poll in run_once
  └─ on event complete: emit DemoteCompleted
scheduler drain_coordinator_events
  ├─ RadixCache::set_block_location(bid, HostPinned{offset})
  ├─ block_to_pages.remove(bid)
  └─ release_pages(&pages)
```

Coordinator owns a mutable pool handle (Arc<Mutex<PagedKVPool>>) and a dedicated `cudaStreamNonBlocking` copy stream. Byte movement is truly async vs compute.

## Cost analysis

| Axis | Option A (scheduler) | Option B (coordinator) |
|---|---|---|
| Scheduler tick blocking | ~7 μs/block × N blocks/tick in cleanup | ~0 (channel send only) |
| Stream independence | D→H queues on compute stream (FIFO after decode); typical ~7 μs/block serialises against the next decode kernel | Dedicated copy stream; D→H overlaps next decode |
| Pool ownership | single writer (scheduler) | shared (Arc<Mutex>); lock contention on every `alloc_tokens` and every `free_slot` |
| MR stability | preserved (HostPinnedPool base doesn't move) | preserved (same) |
| Implementation cost | ~100 LoC (scheduler demote hook + pool byte move + telemetry event) | ~300–500 LoC (copy stream + pool handle + event polling + stage mapper + Mutex rewire across scheduler accesses) |
| Correctness surface | small — single-writer invariant preserved | large — lock ordering, ref-count races during copy, stream-event ordering |

### Measured steady-state budget at c=16 × 4096 × 256

- Per-block D→H: 147 KiB × PCIe @ ~20 GB/s = ~7 μs
- Typical watermark evict pass: ≤ 100 blocks = ~700 μs total
- Tick time at c=16: ~60 ms
- Overhead Option A adds to tick: **~1.2 % worst case**, ~0 % when no demote fires (most ticks)

The "async" value of Option B over Option A is at most ~1 ms of cleanup work per pressure event. That is not a lever that moves TTFT p99 or tok/s at c=16.

## Decision — Option A

**Ship Option A for v1.** Scheduler owns the D→H copy via the existing
`PagedKVPool::copy_pages_to_host` (validated by C1 round-trip test).
Coordinator becomes a pure observability sink for demote/promote events.

Rationale:
1. **3-5× lower implementation cost** with the same on-paper TTFT/tok/s impact.
2. **Preserves A5 principle** from `projects/sglang-parity-and-beyond.md` — tier state lives in the scheduler, not spread across scheduler + coordinator.
3. **Correctness surface stays small** — no Arc<Mutex<PagedKVPool>>, no cross-stream ref-count reasoning, no event/stream ordering in the hot path.
4. **Future-evolvable** — if a later workload (multi-GPU, RDMA via NIXL) makes the async D→H meaningfully better, migration to Option B is local: the `CoordinatorCommand::Demote` shape we freeze now is already enough, and the scheduler-side hook in `evict_prefix_cache_if_pressured` is one function to rewrite. No API break anywhere.

Rejected: Option B — revisit when (a) L4 c=16 cleanup tick profiling shows demote is the dominant cost in a real workload, OR (b) an async byte path is forced by a non-local transport (NIXL / multi-GPU).

## Frozen shapes — `CoordinatorEvent::Demote{Queued,Completed,Failed}`

The event shape stays minimal in v1 — enough for telemetry, not byte transport:

```rust
pub enum CoordinatorEvent {
    // ... existing ...
    /// Demote enqueued. Informational — the scheduler already moved
    /// the bytes before sending this; the event exists so /v1/stats
    /// counters increment uniformly with the Spill/Rehydrate paths.
    DemoteQueued {
        ticket: StageTicket,
        block: BlockId,
        to: BlockLocation,  // always HostPinned{offset} in v1
    },
    /// Demote acknowledged. Scheduler can increment
    /// `t1_demotes_total` + `t1_bytes_demoted_total` on receipt.
    DemoteCompleted {
        ticket: StageTicket,
        block: BlockId,
        bytes: usize,
    },
    /// Reserved for a future Option-B migration where the coordinator
    /// actually runs the copy. In Option A this variant is unused.
    DemoteFailed {
        ticket: StageTicket,
        block: BlockId,
        reason: String,
    },
}
```

Corresponding `CoordinatorCommand::Demote` gets a `ticket` field + optional `byte_count` (for telemetry):

```rust
Demote {
    ticket: StageTicket,
    block: BlockId,
    from: BlockLocation,
    to: BlockLocation,
    /// For v1/stats bookkeeping — scheduler already moved the bytes.
    byte_count: usize,
},
```

`CoordinatorHandle::submit_demote(block, from, to, byte_count) -> Option<StageTicket>` mirrors the existing `submit_spill` / `submit_rehydrate` pattern.

Handler in `Coordinator::run_once`:

```rust
CoordinatorCommand::Demote { ticket, block, to, byte_count, .. } => {
    self.events.send(CoordinatorEvent::DemoteQueued { ticket, block, to: to.clone() })?;
    // Option A: scheduler already moved bytes; ack immediately.
    self.events.send(CoordinatorEvent::DemoteCompleted { ticket, block, bytes: byte_count })?;
}
```

Migration note for Option B: swap the synchronous ack for an async
poll + event record. Command shape and event shape don't change.

## C2 implementation scope (this commit, ~80 LoC + test)

1. `CoordinatorCommand::Demote` — add `ticket: StageTicket, byte_count: usize`. Existing callers are none (command never handled before today).
2. `CoordinatorEvent::{DemoteQueued, DemoteCompleted, DemoteFailed}` — new variants.
3. `CoordinatorHandle::submit_demote` — new helper.
4. `Coordinator::run_once` — match arm that emits the Queued + Completed events.
5. Unit test in `infer/src/kv_tier/coordinator.rs` tests module (pattern matches the existing Spill / Rehydrate tests).

**Not in C2:**
- Scheduler-side demote hook (that's C3; depends on this event shape).
- `HostPinnedPool::reserve` wiring for demote reservations (C3).
- `t1_write_through_threshold` config field (C3).
- `/v1/stats` counter wiring (C3–C5, per Gap #5 plan).

## Verify gate

- `cargo check --release` + Mac typecheck green.
- `cargo test --release -p infer` unit tests green (event ordering assertion).
- No bench change — coordinator path still doesn't fire in production until C3 wires `submit_demote` into `evict_prefix_cache_if_pressured`.

Bench exempt (per CLAUDE.md §Benchmarks): structural plumbing only,
no runtime behaviour change at c=16.
