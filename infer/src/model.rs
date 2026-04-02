//! Model implementations: Qwen3 and Qwen3.5.

use anyhow::Result;
use rand::rngs::StdRng;

use crate::paged_kv::{PagedKVPool, TokenKVPool};
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

pub(crate) mod common;
pub(crate) mod cuda_graph;
pub(crate) mod generation_state;
mod kv_cache;

pub mod qwen3;
pub mod qwen35;

pub use qwen3::{ModelRuntimeConfig, Qwen3Model, Qwen3State};
pub use qwen35::{Qwen35Model, Qwen35State};

// ============================================================================
// ModelForward trait — shared by Qwen3 and Qwen3.5
// ============================================================================

/// Per-request mutable state. Separate from model weights for bs > 1 future.
pub trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
    /// Truncate KV cache to `len` tokens, keeping the first `len` tokens.
    fn truncate_to(&mut self, len: usize) -> Result<()>;
    /// Set max KV tokens on GPU. Excess offloads to CPU.
    fn set_max_gpu_kv(&mut self, max_tokens: usize);
    /// Set the maximum sequence length (total, GPU + CPU) for the KV cache.
    /// Must be called before the KV cache is first initialized.
    fn set_max_seq_len(&mut self, max_seq: usize);
    /// Offload excess KV to CPU if over GPU budget. Called between requests.
    fn offload_kv_if_needed(&mut self) -> Result<()>;

    /// Migrate KV data from contiguous cache to paged pool.
    /// Called after prefill completes, before first decode step.
    fn migrate_kv_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &crate::paged_kv::PagedKVPool,
        slot: usize,
    ) -> Result<()>;
}

/// Deep module interface: explicit prefill/decode phases with typed decode context.
///
/// Phase semantics:
/// - `forward_prefill`: process multiple tokens, populate KV cache
/// - `forward_decode`: process exactly one token, use existing KV cache
/// - `forward_decode_batch`: process B tokens from B requests in one pass
pub trait ModelForward: Send {
    type State: GenerationState + Send;

    /// Pre-allocated buffers for batched decode, owned by the scheduler.
    /// Replaces `Box<dyn Any + Send>` with compile-time type safety.
    type DecodeContext: Send;

    fn create_state(&self) -> Result<Self::State>;

    /// Create decode context for batched decode (lazy-init by scheduler).
    fn create_decode_context(
        &self,
        max_batch_size: usize,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext>;

    /// KV cache memory cost per token in bytes (across all layers, K+V, bf16).
    fn kv_cache_bytes_per_token(&self) -> usize;

    /// Number of KV cache layers (for paged KV pool sizing).
    /// For hybrid models, this is the number of full-attention layers only.
    fn num_kv_layers(&self) -> usize;

    /// Number of KV heads per layer (for paged KV pool sizing).
    fn num_kv_heads(&self) -> usize;

    /// Head dimension (for paged KV pool sizing).
    fn head_dim(&self) -> usize;

    /// Prefill: process multiple tokens, populate KV cache and produce logits.
    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;

    /// Decode: process exactly one token using existing KV cache.
    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()>;

    /// Convenience: dispatch to prefill or decode based on token count.
    /// Callers that know the phase should use `forward_prefill`/`forward_decode` directly.
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.forward_decode(tokens[0], state)
        } else {
            self.forward_prefill(tokens, state)
        }
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32>;
    fn is_stop_token(&self, token_id: u32) -> bool;
    fn device_context(&self) -> &DeviceContext;

    /// Batched sampling: launch all sampling kernels, sync once, readback all.
    /// Returns one token per request. Default falls back to sequential select_token.
    fn select_tokens_batch(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        params: &[&SamplingParams],
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(slot_indices.len());
        for (i, &si) in slot_indices.iter().enumerate() {
            tokens.push(self.select_token(&mut states[si], params[i], rng)?);
        }
        Ok(tokens)
    }

    /// Prefill forward pass that also scatter-writes K/V to the token pool.
    ///
    /// Called by the scheduler instead of `forward_prefill()` when a paged KV pool is
    /// active. The default implementation just calls `forward_prefill()` (no pool write).
    /// Override in model implementations that support scatter-writing K/V to the
    /// pool during prefill (e.g., Qwen3).
    ///
    /// `new_token_indices` are the physical pool indices (on GPU) allocated for
    /// this chunk's tokens. The slice has length `tokens.len()`.
    fn forward_prefill_with_pool(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
        _pool: &TokenKVPool,
        _slot: usize,
        _new_token_indices: &cudarc::driver::CudaSlice<i32>,
    ) -> Result<()> {
        // Default: just call forward_prefill() (no pool write)
        self.forward_prefill(tokens, state)
    }

    /// Fast-path batched greedy sampling on internal contiguous logits.
    /// Returns None if fast path unavailable (non-greedy, or model doesn't support it).
    fn sample_batch_greedy(
        &self,
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Batched decode: process B tokens from B requests in one forward pass.
    ///
    /// `tokens[b]` is decoded using `states[slot_indices[b]]`. Uses GEMM for
    /// linear projections (batched) and per-request attention.
    ///
    /// `paged_kv_pool` is provided when the scheduler owns a paged KV pool.
    /// Implementations may use it for paged attention in batched decode.
    ///
    /// Default implementation falls back to sequential `forward_decode()` calls.
    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
        _skip_logit_scatter: bool,
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_decode(token, &mut states[slot_indices[i]])?;
        }
        Ok(())
    }
}
