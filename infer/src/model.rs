//! Model implementations: Qwen3 and Qwen3.5.

use anyhow::Result;
use rand::rngs::StdRng;

use crate::paged_kv::PagedKVPool;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

pub(crate) mod cuda_graph;
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
}

/// Deep module interface: one `forward` method hides prefill/decode strategy,
/// layer types, CUDA Graph, buffer management, KV cache, and recurrent state.
pub trait ModelForward: Send {
    type State: GenerationState + Send;

    fn create_state(&self) -> Result<Self::State>;

    /// KV cache memory cost per token in bytes (across all layers, K+V, bf16).
    /// Used by the scheduler to compute the dynamic max_seq_len based on
    /// available GPU memory.
    /// Formula: 2 (K+V) * num_kv_layers * num_kv_heads * head_dim * 2 (bf16 bytes)
    fn kv_cache_bytes_per_token(&self) -> usize;

    /// Number of KV cache layers (for paged KV pool sizing).
    /// For hybrid models, this is the number of full-attention layers only.
    fn num_kv_layers(&self) -> usize;

    /// Number of KV heads per layer (for paged KV pool sizing).
    fn num_kv_heads(&self) -> usize;

    /// Head dimension (for paged KV pool sizing).
    fn head_dim(&self) -> usize;

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;
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

    /// Batched decode: process B tokens from B requests in one forward pass.
    ///
    /// `tokens[b]` is decoded using `states[slot_indices[b]]`. Uses GEMM for
    /// linear projections (batched) and per-request attention.
    ///
    /// `paged_kv_pool` is provided when the scheduler owns a paged KV pool.
    /// Implementations may use it for paged attention in batched decode.
    ///
    /// Default implementation falls back to sequential `forward()` calls.
    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(&[token], &mut states[slot_indices[i]])?;
        }
        Ok(())
    }
}
