//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM (matrix multiply) for all linear projections (QKV, O, MLP),
//! batching B requests together. Attention runs per-request (each has its own
//! KV cache). Since projections dominate compute (~98%), this gives near-linear
//! throughput scaling with batch size.

use anyhow::Result;

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::ops;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

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
}

impl BatchDecodeBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        batch_size: usize,
    ) -> Result<Self> {
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
        })
    }
}

/// Copy row `row` from a HiddenStates [B, dim] into a DeviceVec [dim].
fn copy_row_to_vec(
    ctx: &DeviceContext,
    src: &HiddenStates,
    row: usize,
    dst: &mut DeviceVec,
) -> Result<()> {
    let dim = src.hidden_dim;
    let src_view = src.data.slice(row * dim..(row + 1) * dim);
    ctx.stream
        .memcpy_dtod(&src_view, &mut dst.data)
        .map_err(|e| anyhow::anyhow!("D2D row→vec failed: {e}"))?;
    Ok(())
}

/// Copy a DeviceVec [dim] into row `row` of a HiddenStates [B, dim].
fn copy_vec_to_row(
    ctx: &DeviceContext,
    src: &DeviceVec,
    dst: &mut HiddenStates,
    row: usize,
) -> Result<()> {
    let dim = dst.hidden_dim;
    let mut dst_view = dst.data.slice_mut(row * dim..(row + 1) * dim);
    ctx.stream
        .memcpy_dtod(&src.data, &mut dst_view)
        .map_err(|e| anyhow::anyhow!("D2D vec→row failed: {e}"))?;
    Ok(())
}

impl Qwen3Model {
    /// Batched decode: process B tokens from B different requests in one pass.
    ///
    /// `tokens[b]` is the next token for request `b`, whose state is
    /// `states[slot_indices[b]]`. All linear projections are batched via GEMM;
    /// attention runs per-request (each has its own KV cache).
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

        // 4. Compute logits for each request
        for (b, &si) in slot_indices.iter().enumerate() {
            let row_vec = ops::extract_vec(&self.ctx, &hidden, b)?;
            let normed = ops::rms_norm(&self.ctx, &row_vec, &self.norm, eps)?;
            let logits = ops::linear(&self.ctx, &normed, self.output_projection())?;
            // Store logits in the state's prefill_logits field (select_token reads from here)
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

        // 3. Per-request attention (each has its own KV cache)
        for b in 0..batch_size {
            let si = slot_indices[b];
            let state = &mut states[si];
            let pos = state.kv_cache.len();
            let seq_len_with_new = pos + 1;

            // Copy batch row b → per-request decode buffers
            copy_row_to_vec(&self.ctx, &bufs.q_batch, b, &mut state.decode_bufs.q)?;
            copy_row_to_vec(&self.ctx, &bufs.k_batch, b, &mut state.decode_bufs.k)?;
            copy_row_to_vec(&self.ctx, &bufs.v_batch, b, &mut state.decode_bufs.v)?;

            // Set decode metadata: [token_id (unused here), pos, seq_len]
            self.ctx
                .stream
                .memcpy_htod(
                    &[0i32, pos as i32, seq_len_with_new as i32],
                    &mut state.decode_bufs.decode_meta,
                )
                .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {e}"))?;

            // Run fused decode attention (QK-norm + RoPE + KV write + attention)
            let (k_cache, v_cache) =
                state.kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
            ops::fused_attention_decode_into(
                &self.ctx,
                &state.decode_bufs.q,
                &state.decode_bufs.k,
                &state.decode_bufs.v,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                &state.decode_bufs.decode_meta,
                k_cache,
                v_cache,
                &mut state.decode_bufs.attn_out,
                &mut state.decode_bufs.partial_out,
                &mut state.decode_bufs.partial_m,
                &mut state.decode_bufs.partial_l,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
            )?;

            // Copy attention output back to batch row
            copy_vec_to_row(&self.ctx, &state.decode_bufs.attn_out, &mut bufs.attn_output, b)?;
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
