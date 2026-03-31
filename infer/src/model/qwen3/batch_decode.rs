//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM (matrix multiply) for all linear projections (QKV, O, MLP),
//! batching B requests together. Attention is also batched: per-request KV cache
//! pointers are gathered into GPU arrays and a single batched CUDA kernel
//! (split-KV + reduce) handles all requests in 2 launches instead of 2*B.

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtrMut};

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::ops;
use crate::tensor::{DeviceContext, HiddenStates};

/// Temporary buffers for batched decode, analogous to PrefillBuffers.
struct BatchDecodeBuffers {
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,

    // Batched attention buffers
    /// Per-request positions on GPU: [B] i32
    positions: CudaSlice<i32>,
    /// Per-request seq_lens on GPU: [B] i32
    seq_lens: CudaSlice<i32>,
    /// Per-request K cache device pointers on GPU: [B] u64
    k_cache_ptrs: CudaSlice<u64>,
    /// Per-request V cache device pointers on GPU: [B] u64
    v_cache_ptrs: CudaSlice<u64>,
    /// Split-KV partial output: [B * num_qheads * NUM_KV_SPLITS * HEAD_DIM] f32
    partial_out: CudaSlice<f32>,
    /// Split-KV partial max: [B * num_qheads * NUM_KV_SPLITS] f32
    partial_m: CudaSlice<f32>,
    /// Split-KV partial sum: [B * num_qheads * NUM_KV_SPLITS] f32
    partial_l: CudaSlice<f32>,
}

impl BatchDecodeBuffers {
    /// NUM_KV_SPLITS must match the CUDA kernel constant.
    const NUM_KV_SPLITS: usize = 4;

    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        batch_size: usize,
        num_qheads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let partial_size = batch_size * num_qheads * Self::NUM_KV_SPLITS;
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, batch_size)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, batch_size)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, batch_size)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, batch_size)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, batch_size)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, batch_size)?,

            positions: ctx
                .stream
                .alloc_zeros(batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc positions failed: {e}"))?,
            seq_lens: ctx
                .stream
                .alloc_zeros(batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc seq_lens failed: {e}"))?,
            k_cache_ptrs: ctx
                .stream
                .alloc_zeros(batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc k_cache_ptrs failed: {e}"))?,
            v_cache_ptrs: ctx
                .stream
                .alloc_zeros(batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc v_cache_ptrs failed: {e}"))?,
            partial_out: ctx
                .stream
                .alloc_zeros(partial_size * head_dim)
                .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {e}"))?,
            partial_m: ctx
                .stream
                .alloc_zeros(partial_size)
                .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {e}"))?,
            partial_l: ctx
                .stream
                .alloc_zeros(partial_size)
                .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {e}"))?,
        })
    }
}

impl Qwen3Model {
    /// Batched decode: process B tokens from B different requests in one pass.
    ///
    /// `tokens[b]` is the next token for request `b`, whose state is
    /// `states[slot_indices[b]]`. All linear projections are batched via GEMM;
    /// attention is batched via a single CUDA kernel launch with per-request
    /// KV cache pointer indirection.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Qwen3State],
        slot_indices: &[usize],
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        debug_assert!(batch_size > 1);

        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter_dim = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;

        // Initialize KV caches (lazy init on first use).
        for &si in slot_indices {
            states[si].kv_cache.init_if_needed(&self.ctx, head_dim)?;
        }

        // Allocate batch buffers (reused across layers).
        let mut bufs = BatchDecodeBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            batch_size,
            num_heads,
            head_dim,
        )?;

        // 1. Batched embedding: [B, hidden_dim]
        let mut hidden = self.get_embeddings_batch(tokens)?;

        // 2. Process all layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_batch_layer(
                layer_idx,
                layer,
                &mut hidden,
                &mut bufs,
                states,
                slot_indices,
            )?;
        }

        // 3. Increment KV cache seq_len for each request
        for &si in slot_indices {
            states[si].kv_cache.increment_seq_len();
        }

        // 4. Compute logits for all requests in one batched norm + one GEMM
        ops::rms_norm_batch_into(&self.ctx, &hidden, &self.norm, eps, &mut bufs.normed);
        let logits_batch = ops::gemm(&self.ctx, self.output_projection(), &bufs.normed)?;
        for (b, &si) in slot_indices.iter().enumerate() {
            let logits = ops::extract_vec(&self.ctx, &logits_batch, b)?;
            states[si].prefill_logits = Some(logits);
        }

        Ok(())
    }

    fn decode_batch_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers,
        states: &mut [Qwen3State],
        slot_indices: &[usize],
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let batch_size = slot_indices.len();

        // 1. Batched RMSNorm → bufs.normed [B, hidden_dim]
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 2. Batched QKV projections (GEMM) → [B, q/kv_dim]
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        );

        // 3. Batched attention — gather metadata, launch 2 kernels for all requests
        {
            let num_heads = self.config.num_attention_heads;
            let num_kv_heads = self.config.num_key_value_heads;
            let head_dim = self.config.head_dim;
            let max_seq_len = 32768; // matches KVCache::max_seq_len default

            // Gather per-request positions, seq_lens, and KV cache pointers (CPU side)
            let mut positions_host = vec![0i32; batch_size];
            let mut seq_lens_host = vec![0i32; batch_size];
            let mut k_cache_ptrs_host = vec![0u64; batch_size];
            let mut v_cache_ptrs_host = vec![0u64; batch_size];

            for b in 0..batch_size {
                let si = slot_indices[b];
                let state = &mut states[si];
                let pos = state.kv_cache.len();
                positions_host[b] = pos as i32;
                seq_lens_host[b] = (pos + 1) as i32;

                let (k_cache, v_cache) = state.kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
                let (k_ptr, _gk) = k_cache.data.device_ptr_mut(&self.ctx.stream);
                let (v_ptr, _gv) = v_cache.data.device_ptr_mut(&self.ctx.stream);
                k_cache_ptrs_host[b] = k_ptr;
                v_cache_ptrs_host[b] = v_ptr;
            }

            // Upload gathered arrays to GPU (one H2D copy each)
            self.ctx
                .stream
                .memcpy_htod(&positions_host, &mut bufs.positions)
                .map_err(|e| anyhow::anyhow!("H2D positions failed: {e}"))?;
            self.ctx
                .stream
                .memcpy_htod(&seq_lens_host, &mut bufs.seq_lens)
                .map_err(|e| anyhow::anyhow!("H2D seq_lens failed: {e}"))?;
            self.ctx
                .stream
                .memcpy_htod(&k_cache_ptrs_host, &mut bufs.k_cache_ptrs)
                .map_err(|e| anyhow::anyhow!("H2D k_cache_ptrs failed: {e}"))?;
            self.ctx
                .stream
                .memcpy_htod(&v_cache_ptrs_host, &mut bufs.v_cache_ptrs)
                .map_err(|e| anyhow::anyhow!("H2D v_cache_ptrs failed: {e}"))?;

            // Launch batched attention: 2 kernel launches for all B requests
            ops::fused_attention_decode_batched_into(
                &self.ctx,
                &bufs.q_batch,
                &bufs.k_batch,
                &bufs.v_batch,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                &bufs.positions,
                &bufs.seq_lens,
                &bufs.k_cache_ptrs,
                &bufs.v_cache_ptrs,
                &mut bufs.attn_output,
                &mut bufs.partial_out,
                &mut bufs.partial_m,
                &mut bufs.partial_l,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                eps,
            )?;
        }

        // 4. Batched O projection → bufs.o_buf [B, hidden_dim]
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        // 5. Batched residual add: hidden + o_buf → hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        // 6. Batched MLP RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 7. Batched MLP: gate + up → silu_mul → down
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );

        // 8. Batched residual add
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
