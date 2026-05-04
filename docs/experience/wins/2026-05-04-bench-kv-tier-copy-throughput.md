# 2026-05-04 · T1↔T2 copy-throughput micro-benchmark + CPU↔GPU overlap analysis

## Context

User asked: 测一下拷贝的速度,以及考虑 CPU 和 GPU 的 overlap. The new
HiCache-borrowed roadmap (`docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md`)
needs concrete numbers for layer-wise compute-transfer overlap (HiCache
optimization 6.2) viability — without measured bandwidth, we can't tell
whether ARLE's per-layer KV transfer fits inside a layer's compute budget.

This entry pins the actual T1 (host pinned DRAM) and T2 (disk) bandwidth
numbers measured **through the real ARLE types** (`HostPinnedPool`,
`DiskStore`, `Coordinator`), then derives the overlap conclusion.

## What worked

Added `infer/src/kv_tier/coordinator/bench.rs` — a `#[ignore]`-tagged
bench that uses the actual coordinator + Zig-backed pinned arena +
DiskStore to time T1 self-copy, T2 disk round-trip, and the full
T1→T2→release→T1 coordinator cycle at 5 sizes from 4 KiB to 256 MiB.
Run: `cargo test --release --no-default-features --features no-cuda
-p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture`.

### Measured throughput

Hardware: Linux x86_64 sandbox, Zig 0.16 (kv-native-sys), tempfs-backed
DiskStore via `tempdir()`. CPU-only, no GPU. Each row counts both
directions of the round-trip (write+read for T1, put+get for T2,
store-leg+fetch-leg for coordinator).

| size    | iters | T1 self-copy   | T2 put+get   | Coord T1→T2→T1 | (coord ops/s) |
|---------|-------|----------------|--------------|----------------|---------------|
| 4 KiB   | 5000  | 24929.3 MiB/s  | 18.2 MiB/s   | 17.0 MiB/s     | 2176.94 ops/s |
| 64 KiB  | 2000  | 33211.7 MiB/s  | 142.5 MiB/s  | 125.6 MiB/s    | 1005.19 ops/s |
| 1 MiB   | 500   | 19862.0 MiB/s  | 976.0 MiB/s  | 408.0 MiB/s    | 203.98 ops/s  |
| 16 MiB  | 50    | 11914.9 MiB/s  | 1541.9 MiB/s | 441.3 MiB/s    | 13.79 ops/s   |
| 256 MiB | 10    | 3172.5 MiB/s   | 519.0 MiB/s  | 245.7 MiB/s    | 0.48 ops/s    |

### Three readable signals

1. **T1 self-copy is memcpy-bandwidth-bound.** 12–33 GiB/s for medium
   sizes; drops to 3.2 GiB/s at 256 MiB as cache misses dominate.
   Approximates the host-side ceiling for any cudaMemcpyAsync /
   GPU-SM-kernel target (HiCache 6.1 / ARLE A8).

2. **T2 disk round-trip ramps with size, then plateaus.** 18 MiB/s at
   4 KiB (small-file overhead) → 1.5 GiB/s at 16 MiB (peak, tempfs
   helps) → 519 MiB/s at 256 MiB (likely page-cache thrashing). This
   bounds anything T1↔T2 can do.

3. **Coordinator is ~half of raw T2 at large sizes.** 245 MiB/s vs
   519 MiB/s at 256 MiB. The ~50% gap is one extra host memcpy in the
   read-after-fetch path (Coordinator `stage_into_host_pool` allocates
   a fresh region and copies bytes in via `as_mut_slice`, then test
   reads back via `read_region` for verification). For real
   readmission, the test's final `read_region` step doesn't happen,
   so production overhead should be smaller.

## CPU↔GPU layer-wise overlap viability

Apply the measured T1 self-copy ceiling (≈ 12 GiB/s usable for medium
blocks) as a lower bound on what cudaMemcpyAsync can achieve over
PCIe Gen4 x16 (32 GiB/s nominal). For Qwen3-32B with TP=8:

- 64 layers, ~8 KV heads per rank, head_dim=128, fp16 (2 B/elem)
- Per-token-per-layer KV (per rank) = 2 (K+V) × 8 × 128 × 2 = 4 KiB
- For 4096-token context window: per-layer KV = 16 MiB per rank
- Per-layer T1→T0 transfer at 12 GiB/s ≈ **1.3 ms**
- Per-layer compute on a saturated GPU at 32B class: ≈ **5–15 ms**

**Conclusion:** the per-layer transfer fits inside a single layer's
compute budget by a margin of 4–10×. **Layer-wise compute-transfer
overlap (P1.1 in the borrowed roadmap, HiCache 6.2) is feasible and
will hide essentially all T1↔T0 transfer time on this scale class.**
The remaining serialization cost is the first layer (no compute to
hide it) and pipeline drain at the last.

For T1↔T2 (disk) the same arithmetic gives 16 MiB / 441 MiB/s = 36 ms
per layer, **far larger than per-layer compute**. T2-readmitted blocks
cannot be hidden by per-layer overlap; they need either (a) batched
prefetch with `PrefetchPolicy::Timeout` so the wait happens once before
prefill rather than per-layer, or (b) faster T2 (NVMe / RDMA / Mooncake
L3). This matches HiCache's design choice to give L3 its own
`prefetch_threshold + Timeout policy` rather than rely on layer-wise
overlap for the slowest tier.

## Rule

Before designing a per-layer overlap optimization for any tier, measure
that tier's bandwidth at the realistic per-layer block size and compare
to per-layer compute time on the model class of interest. The two-order-
of-magnitude gap between T1↔T0 (hideable) and T1↔T2 (not hideable)
shows the same optimization shape doesn't apply uniformly across tiers
— this is why HiCache layers compute-transfer overlap on T1↔T0 but
prefetch+timeout on L2↔L3, and ARLE's roadmap mirrors that split.

## Pointers

- Bench source: `infer/src/kv_tier/coordinator/bench.rs`
- Roadmap that consumes these numbers: `docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md` §P1.1, §P0.2
- HiCache reference: `docs/research/2026-05-04-sglang-hicache-guide.md` Part VI §6.2, §6.5
- Re-run the bench: `cargo test --release --no-default-features --features no-cuda -p infer --lib bench_kv_tier_copy_throughput -- --ignored --nocapture`

## Status

`pending-remote-gpu` for the **T0↔T1 PCIe leg** — this run is CPU-only,
so the cudaMemcpyAsync number is extrapolated from host-memcpy. The
actual GPU-PCIe number must be measured on a CUDA box before the
P1.1 layer-wise overlap implementation lands. Open a follow-up bench
on the next CUDA-capable run to validate the 12 GiB/s lower-bound
assumption against real `cudaMemcpyAsync` and (after A8 lands) the
SM-assisted I/O kernel.
