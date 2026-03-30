//! Tensor Parallel configuration and sharding utilities.
//!
//! This module provides the **CPU-side** configuration and sharding math for
//! tensor parallelism.  GPU communication primitives (NCCL all-reduce /
//! all-gather) are declared as stubs and live behind `#[cfg(feature = "cuda")]`.
//!
//! # Tensor Parallel overview
//!
//! Tensor parallelism (TP) splits model weight matrices across multiple GPUs:
//!
//! ```text
//! ColumnParallelLinear:   output dim split → each GPU holds W[:, offset..offset+size]
//!                         Requires all-reduce on output to sum partial results.
//!
//! RowParallelLinear:      input dim split  → each GPU holds W[offset..offset+size, :]
//!                         Input is pre-sharded; all-reduce needed at output.
//! ```
//!
//! # CPU-verifiable
//!
//! - [`TpConfig`] validation
//! - [`ShardingSpec`] computation via [`column_shard`] / [`row_shard`]
//! - Head assignment via [`head_shard`]
//! - [`TpLinearConfig`] builder for both parallel linear types

use anyhow::{Result, bail};

// ============================================================================
// TpConfig
// ============================================================================

/// Tensor parallel configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpConfig {
    /// Total number of TP ranks (GPUs in the tensor-parallel group).
    pub world_size: usize,
    /// This rank's index within the TP group (0 ≤ rank < world_size).
    pub rank: usize,
}

impl TpConfig {
    /// Single-GPU configuration (no parallelism).
    pub fn single() -> Self {
        Self { world_size: 1, rank: 0 }
    }

    /// Multi-GPU configuration.
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("world_size must be ≥ 1");
        }
        if rank >= world_size {
            bail!("rank ({rank}) must be < world_size ({world_size})");
        }
        Ok(Self { world_size, rank })
    }

    /// True when running on a single GPU (no all-reduce needed).
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            bail!("world_size must be ≥ 1");
        }
        if self.rank >= self.world_size {
            bail!("rank {} ≥ world_size {}", self.rank, self.world_size);
        }
        Ok(())
    }
}

impl Default for TpConfig {
    fn default() -> Self {
        Self::single()
    }
}

// ============================================================================
// ShardingSpec
// ============================================================================

/// Describes a rank's slice of a dimension of size `total`.
///
/// The rank owns `self.size` elements starting at `self.offset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardingSpec {
    /// Starting index of this rank's shard.
    pub offset: usize,
    /// Number of elements owned by this rank.
    pub size: usize,
    /// Total size of the dimension (sum of all ranks' sizes).
    pub total: usize,
}

impl ShardingSpec {
    /// Exclusive end index: `offset + size`.
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Return the range as a `std::ops::Range`.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }

    /// True if this rank owns the entire dimension (single-GPU case).
    pub fn is_full(&self) -> bool {
        self.offset == 0 && self.size == self.total
    }
}

// ============================================================================
// Sharding functions
// ============================================================================

/// Compute the shard for a **column-parallel** dimension (output features split
/// across TP ranks).
///
/// The last rank absorbs any remainder so that `sum(all sizes) == total`.
///
/// # Panics
/// Panics if `total < world_size` (cannot give each rank at least 1 element).
pub fn column_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    assert!(
        total >= tp.world_size,
        "total ({total}) < world_size ({}): cannot shard",
        tp.world_size
    );
    let base = total / tp.world_size;
    let remainder = total % tp.world_size;
    // Distribute remainder to the last rank.
    let offset = tp.rank * base;
    let size = if tp.rank == tp.world_size - 1 { base + remainder } else { base };
    ShardingSpec { offset, size, total }
}

/// Compute the shard for a **row-parallel** dimension (input features split
/// across TP ranks).
///
/// Identical formula to column_shard — differs only in semantic interpretation.
pub fn row_shard(total: usize, tp: &TpConfig) -> ShardingSpec {
    column_shard(total, tp)
}

/// Compute the assignment of attention heads for this TP rank.
///
/// Returns `(num_q_heads_local, num_kv_heads_local)`.
///
/// # Errors
/// Returns an error if `num_kv_heads` is not divisible by `world_size`
/// (GQA head assignment must be uniform across TP ranks).
pub fn head_shard(
    num_q_heads: usize,
    num_kv_heads: usize,
    tp: &TpConfig,
) -> Result<(usize, usize)> {
    if num_q_heads % tp.world_size != 0 {
        bail!(
            "num_q_heads ({num_q_heads}) not divisible by world_size ({})",
            tp.world_size
        );
    }
    if num_kv_heads % tp.world_size != 0 {
        bail!(
            "num_kv_heads ({num_kv_heads}) not divisible by world_size ({})",
            tp.world_size
        );
    }
    Ok((num_q_heads / tp.world_size, num_kv_heads / tp.world_size))
}

// ============================================================================
// TpLinearConfig
// ============================================================================

/// Type of parallel linear layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelLinearKind {
    /// Split output dimension across TP ranks; all-reduce result.
    Column,
    /// Split input dimension across TP ranks; all-reduce result.
    Row,
}

/// Configuration for a tensor-parallel linear layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TpLinearConfig {
    pub kind: ParallelLinearKind,
    pub shard: ShardingSpec,
    /// Whether an all-reduce is needed after this layer (always true for both kinds,
    /// unless this is an intermediate result that will be combined in the next layer).
    pub needs_all_reduce: bool,
}

impl TpLinearConfig {
    /// Build config for a column-parallel linear layer.
    pub fn column(out_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Column,
            shard: column_shard(out_features, tp),
            needs_all_reduce: true,
        }
    }

    /// Build config for a row-parallel linear layer.
    pub fn row(in_features: usize, tp: &TpConfig) -> Self {
        Self {
            kind: ParallelLinearKind::Row,
            shard: row_shard(in_features, tp),
            needs_all_reduce: true,
        }
    }
}

// ============================================================================
// NcclComm (GPU required)
// ============================================================================

/// NCCL communicator handle. GPU required.
///
/// On CPU builds this struct exists but all methods panic with
/// `todo!("GPU required: ...")`.
pub struct NcclComm {
    tp: TpConfig,
}

impl NcclComm {
    /// Create a new NCCL communicator for the given TP config.
    ///
    /// **GPU required** — panics on CPU builds.
    #[allow(unused_variables)]
    pub fn new(tp: TpConfig) -> Result<Self> {
        // GPU required: NCCL communicator initialization
        // In production: ncclCommInitRank / ncclGetUniqueId exchange via shared store
        #[cfg(not(feature = "cuda"))]
        todo!("GPU required: NCCL communicator initialization");
        #[cfg(feature = "cuda")]
        Ok(Self { tp })
    }

    /// All-reduce (sum) across all TP ranks.
    ///
    /// **GPU required**.
    #[allow(unused_variables)]
    pub fn all_reduce_sum_f16(&self, data: *mut u16, numel: usize) -> Result<()> {
        // GPU required: ncclAllReduce(data, data, numel, ncclFloat16, ncclSum, comm, stream)
        todo!("GPU required: NCCL all-reduce f16")
    }

    /// All-gather across all TP ranks, concatenating along dim 0.
    ///
    /// **GPU required**.
    #[allow(unused_variables)]
    pub fn all_gather_f16(
        &self,
        send: *const u16,
        recv: *mut u16,
        numel_per_rank: usize,
    ) -> Result<()> {
        // GPU required: ncclAllGather(send, recv, numel_per_rank, ncclFloat16, comm, stream)
        todo!("GPU required: NCCL all-gather f16")
    }

    pub fn tp(&self) -> &TpConfig {
        &self.tp
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- TpConfig

    #[test]
    fn tp_config_single() {
        let tp = TpConfig::single();
        assert!(tp.is_single());
        tp.validate().unwrap();
    }

    #[test]
    fn tp_config_valid_multi() {
        let tp = TpConfig::new(4, 2).unwrap();
        assert!(!tp.is_single());
        assert_eq!(tp.world_size, 4);
        assert_eq!(tp.rank, 2);
    }

    #[test]
    fn tp_config_invalid_rank() {
        assert!(TpConfig::new(4, 4).is_err());
        assert!(TpConfig::new(0, 0).is_err());
    }

    // ---------------------------------------------------------------- column_shard

    #[test]
    fn column_shard_even_division() {
        let tp = TpConfig::new(4, 0).unwrap();
        let s = column_shard(16, &tp);
        assert_eq!(s.offset, 0);
        assert_eq!(s.size, 4);
        assert_eq!(s.total, 16);
        assert_eq!(s.end(), 4);

        let tp3 = TpConfig::new(4, 3).unwrap();
        let s3 = column_shard(16, &tp3);
        assert_eq!(s3.offset, 12);
        assert_eq!(s3.size, 4);
    }

    #[test]
    fn column_shard_with_remainder() {
        // 10 / 4: base=2, remainder=2; last rank gets 2+2=4
        let tp0 = TpConfig::new(4, 0).unwrap();
        let tp3 = TpConfig::new(4, 3).unwrap();
        let s0 = column_shard(10, &tp0);
        let s3 = column_shard(10, &tp3);
        assert_eq!(s0.size, 2);
        assert_eq!(s3.size, 4); // absorbs remainder
        // All shards together cover the full dimension
        let total_covered: usize = (0..4)
            .map(|r| column_shard(10, &TpConfig::new(4, r).unwrap()).size)
            .sum();
        assert_eq!(total_covered, 10);
    }

    #[test]
    fn column_shard_single_gpu() {
        let tp = TpConfig::single();
        let s = column_shard(1024, &tp);
        assert!(s.is_full());
        assert_eq!(s.offset, 0);
        assert_eq!(s.size, 1024);
    }

    // ---------------------------------------------------------------- row_shard (same formula)

    #[test]
    fn row_shard_matches_column_shard() {
        let tp = TpConfig::new(8, 3).unwrap();
        assert_eq!(row_shard(128, &tp), column_shard(128, &tp));
    }

    // ---------------------------------------------------------------- head_shard

    #[test]
    fn head_shard_gqa() {
        // Llama-70B: 64 Q heads, 8 KV heads, TP=8
        let tp = TpConfig::new(8, 0).unwrap();
        let (q, kv) = head_shard(64, 8, &tp).unwrap();
        assert_eq!(q, 8);
        assert_eq!(kv, 1);
    }

    #[test]
    fn head_shard_mha() {
        // Standard MHA: 32 Q == 32 KV, TP=4
        let tp = TpConfig::new(4, 2).unwrap();
        let (q, kv) = head_shard(32, 32, &tp).unwrap();
        assert_eq!(q, 8);
        assert_eq!(kv, 8);
    }

    #[test]
    fn head_shard_indivisible_kv() {
        // 7 KV heads not divisible by 4
        let tp = TpConfig::new(4, 0).unwrap();
        assert!(head_shard(32, 7, &tp).is_err());
    }

    // ---------------------------------------------------------------- TpLinearConfig

    #[test]
    fn tp_linear_config_column() {
        let tp = TpConfig::new(4, 1).unwrap();
        let cfg = TpLinearConfig::column(512, &tp);
        assert_eq!(cfg.kind, ParallelLinearKind::Column);
        assert_eq!(cfg.shard.offset, 128);
        assert_eq!(cfg.shard.size, 128);
        assert!(cfg.needs_all_reduce);
    }

    #[test]
    fn tp_linear_config_row() {
        let tp = TpConfig::new(2, 0).unwrap();
        let cfg = TpLinearConfig::row(4096, &tp);
        assert_eq!(cfg.kind, ParallelLinearKind::Row);
        assert_eq!(cfg.shard.offset, 0);
        assert_eq!(cfg.shard.size, 2048);
    }

    // ---------------------------------------------------------------- ShardingSpec helpers

    #[test]
    fn sharding_spec_range() {
        let s = ShardingSpec { offset: 8, size: 4, total: 16 };
        assert_eq!(s.end(), 12);
        assert_eq!(s.range(), 8..12);
        assert!(!s.is_full());
    }

    #[test]
    fn sharding_spec_full() {
        let s = ShardingSpec { offset: 0, size: 1024, total: 1024 };
        assert!(s.is_full());
    }
}
