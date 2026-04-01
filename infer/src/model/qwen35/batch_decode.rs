//! Batched decode for Qwen3.5: process multiple requests in one forward pass.
//!
//! Hybrid architecture: 8 full attention layers use FlashInfer HD256 paged decode,
//! 24 linear attention layers use batched GEMMs + per-request recurrent ops (conv1d + GDR).
//! No CUDA Graph (per-request recurrent state pointers vary with batch composition).

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::forward::Qwen35State;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::flashinfer_metadata::FlashInferDecodeMetadata;
use crate::model::ModelForward;
use crate::ops;
use crate::paged_kv::PagedKVPool;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated buffers for batched decode, reused across steps.
pub(crate) struct BatchDecodeBuffers35 {
    // ── Common buffers ──
    hidden_out: HiddenStates,
    normed: HiddenStates,
    embedding_out: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,

    // ── Full attention (8 layers) ──
    /// Q+gate projection output [B, q_proj_dim] where q_proj_dim = num_q_heads * 256 * 2
    q_full_batch: HiddenStates,
    /// Q output after norm+RoPE (no gate) [B, q_dim] where q_dim = num_q_heads * 256
    q_batch: HiddenStates,
    k_batch: HiddenStates,
    v_batch: HiddenStates,
    attn_output: HiddenStates,

    // ── Linear attention (24 layers) — batched projections ──
    qkv_batch: HiddenStates,
    z_batch: HiddenStates,
    b_batch: HiddenStates,
    a_batch: HiddenStates,

    // ── Linear attention — per-request single-token scratch ──
    qkv_single: HiddenStates,
    qkv_conv_single: HiddenStates,
    b_single: HiddenStates,
    a_single: HiddenStates,
    gdr_out_single: HiddenStates,

    // ── Linear attention — batched output ──
    gdr_out_batch: HiddenStates,
    normed_gated: HiddenStates,

    // ── Attention/recurrent output → O/out proj input [B, hidden] ──
    attn_results: HiddenStates,

    // ── Residual intermediate [B, hidden] ──
    hidden_mid: HiddenStates,

    // ── Logits + sampling ──
    pub(super) logits_batch: Option<HiddenStates>,
    pub(super) argmax_out: CudaSlice<i32>,
    pub(super) argmax_host: Vec<i32>,

    // ── Token IDs ──
    token_ids_gpu: CudaSlice<i32>,
    token_ids_scratch: Vec<i32>,

    // ── FlashInfer metadata (for full attention layers) ──
    pub(crate) metadata: FlashInferDecodeMetadata,

    max_batch_size: usize,
}

// SAFETY: Exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers35 {}

impl BatchDecodeBuffers35 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_proj_dim: usize, // num_q_heads * 256 * 2 (includes gate)
        q_dim: usize,      // num_q_heads * 256
        kv_dim: usize,     // num_kv_heads * 256
        inter_dim: usize,
        qkv_dim: usize, // linear attention QKV dim
        z_dim: usize,   // linear attention Z dim
        b_dim: usize,   // linear attention B dim (num_value_heads)
        max_batch_size: usize,
        num_qheads: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,

            q_full_batch: HiddenStates::zeros(ctx, q_proj_dim, max_batch_size)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,

            qkv_batch: HiddenStates::zeros(ctx, qkv_dim, max_batch_size)?,
            z_batch: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,
            b_batch: HiddenStates::zeros(ctx, b_dim, max_batch_size)?,
            a_batch: HiddenStates::zeros(ctx, b_dim, max_batch_size)?,

            // Per-request scratch (seq_len=1)
            qkv_single: HiddenStates::zeros(ctx, qkv_dim, 1)?,
            qkv_conv_single: HiddenStates::zeros(ctx, qkv_dim, 1)?,
            b_single: HiddenStates::zeros(ctx, b_dim, 1)?,
            a_single: HiddenStates::zeros(ctx, b_dim, 1)?,
            gdr_out_single: HiddenStates::zeros(ctx, z_dim, 1)?,

            gdr_out_batch: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, max_batch_size)?,

            attn_results: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            hidden_mid: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,

            logits_batch: None,
            argmax_out: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc argmax_out: {e}"))?,
            argmax_host: vec![0i32; max_batch_size],

            token_ids_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc token_ids_gpu: {e}"))?,
            token_ids_scratch: Vec::with_capacity(max_batch_size),

            metadata: FlashInferDecodeMetadata::new(
                ctx,
                max_batch_size,
                max_total_pages,
                num_qheads,
            )?,

            max_batch_size,
        })
    }

    fn set_batch_size(&mut self, bs: usize) {
        debug_assert!(bs <= self.max_batch_size);
        self.hidden_out.seq_len = bs;
        self.normed.seq_len = bs;
        self.q_full_batch.seq_len = bs;
        self.q_batch.seq_len = bs;
        self.k_batch.seq_len = bs;
        self.v_batch.seq_len = bs;
        self.attn_output.seq_len = bs;
        self.o_buf.seq_len = bs;
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.act_out.seq_len = bs;
        self.qkv_batch.seq_len = bs;
        self.z_batch.seq_len = bs;
        self.b_batch.seq_len = bs;
        self.a_batch.seq_len = bs;
        self.gdr_out_batch.seq_len = bs;
        self.normed_gated.seq_len = bs;
        self.attn_results.seq_len = bs;
        self.hidden_mid.seq_len = bs;
    }
}

impl Qwen35Model {
    /// Batched decode: process B tokens from B different requests in one pass.
    /// Falls back to sequential forward() for non-paged path.
    pub fn decode_batch_contiguous(
        &self,
        tokens: &[u32],
        states: &mut [Qwen35State],
        slot_indices: &[usize],
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(&[token], &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    /// Batched decode with paged KV for full attention, per-request recurrent for linear.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Qwen35State],
        slot_indices: &[usize],
        skip_logit_scatter: bool,
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        debug_assert!(batch_size >= 1);
        debug_assert!(batch_size <= bufs.max_batch_size);

        let c = &self.config;
        let num_heads = c.num_attention_heads;
        let num_kv_heads = c.num_key_value_heads;
        let head_dim = c.head_dim; // 256
        let page_size = 1;

        bufs.set_batch_size(batch_size);

        // ── Pre-step: metadata H2D + FlashInfer plan + embedding ──

        bufs.token_ids_scratch.clear();
        bufs.token_ids_scratch
            .extend(tokens.iter().map(|&x| x as i32));
        self.ctx
            .stream
            .memcpy_htod(&bufs.token_ids_scratch, &mut bufs.token_ids_gpu)
            .map_err(|e| anyhow::anyhow!("H2D token_ids: {e}"))?;

        let reallocated = bufs
            .metadata
            .update(&self.ctx, paged_kv_pool, slot_indices)?;
        let _ = reallocated; // No CUDA graph cache to invalidate

        bufs.metadata.plan_hd256(
            &self.ctx,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;

        bufs.embedding_out.seq_len = batch_size;

        // Lazy-init logits buffer
        if bufs.logits_batch.is_none() {
            let vocab_size = self.embed_tokens.rows;
            bufs.logits_batch = Some(HiddenStates::zeros(
                &self.ctx,
                vocab_size,
                bufs.max_batch_size,
            )?);
        }

        // ── Forward pass: no CUDA graph (per-request recurrent state varies) ──
        self.decode_batch_body(bufs, states, slot_indices, paged_kv_pool, batch_size)?;

        // Scatter per-slot logits when needed (non-greedy fallback)
        if !skip_logit_scatter {
            let logits = bufs.logits_batch.as_ref().unwrap();
            for (b, &si) in slot_indices.iter().enumerate() {
                ops::extract_vec_into(
                    &self.ctx,
                    logits,
                    b,
                    &mut states[si].decode_bufs.logits_scratch,
                )?;
                states[si].base.prefill_logits = None;
            }
        }

        Ok(())
    }

    fn decode_batch_body(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        states: &mut [Qwen35State],
        slot_indices: &[usize],
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        // Embedding
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_gpu,
            &mut bufs.embedding_out,
        )?;

        let hidden_ptr = &mut bufs.embedding_out as *mut HiddenStates;

        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;

        for layer in &self.layers {
            let hidden = unsafe { &mut *hidden_ptr };
            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.decode_batch_full_attn_layer(
                        layer, attn, hidden, bufs, kv_pool, full_idx, batch_size,
                    )?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.decode_batch_linear_attn_layer(
                        layer,
                        attn,
                        hidden,
                        bufs,
                        states,
                        slot_indices,
                        linear_idx,
                        batch_size,
                    )?;
                    linear_idx += 1;
                }
            }
        }

        // Final norm (offset variant) + logits GEMM
        let hidden = unsafe { &*hidden_ptr };
        ops::rms_norm_batch_offset_into(&self.ctx, hidden, &self.norm, eps, &mut bufs.normed)?;
        let logits_buf = bufs.logits_batch.as_mut().unwrap();
        logits_buf.seq_len = batch_size;
        ops::gemm_into(&self.ctx, &self.embed_tokens, &bufs.normed, logits_buf);

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_batch_full_attn_layer(
        &self,
        layer: &TransformerBlock35,
        attn: &FullAttentionLayer,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        kv_pool: &PagedKVPool,
        full_idx: usize,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let num_heads = c.num_attention_heads;
        let num_kv_heads = c.num_key_value_heads;
        let head_dim = c.head_dim;
        let page_size = 1;

        // 1. Input RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // 2. QKV projections (batched GEMM)
        ops::gemm_into(
            &self.ctx,
            &attn.q_proj,
            &bufs.normed,
            &mut bufs.q_full_batch,
        );
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_batch);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_batch);

        // 3. Decode prep: QK-norm (1+w) + partial RoPE + paged KV write
        ops::decode_prep_paged_hd256(
            &self.ctx,
            &bufs.q_full_batch,
            &mut bufs.q_batch,
            &bufs.k_batch,
            &bufs.v_batch,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.metadata.positions,
            kv_pool,
            full_idx,
            &bufs.metadata.kv_indices,
            &bufs.metadata.kv_indptr,
            &bufs.metadata.kv_last_page_len,
            num_heads,
            num_kv_heads,
            page_size,
            c.rotary_dim,
            eps,
        )?;

        // 4. FlashInfer HD256 attention
        ops::flashinfer_run_layer_hd256(
            &self.ctx,
            &bufs.q_batch,
            kv_pool,
            full_idx,
            &bufs.metadata.kv_indptr,
            &bufs.metadata.kv_indices,
            &bufs.metadata.kv_last_page_len,
            &mut bufs.attn_output,
            &mut bufs.metadata.flashinfer_ws,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;

        // 5. Apply sigmoid gate
        ops::attention_gate_paged_hd256(
            &self.ctx,
            &bufs.q_full_batch,
            &mut bufs.attn_output,
            num_heads,
        );

        // 6. O projection
        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_output,
            &mut bufs.attn_results,
        );

        // 7. Residual + post-attention norm + MLP
        self.decode_batch_mlp(layer, hidden, bufs, batch_size)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_batch_linear_attn_layer(
        &self,
        layer: &TransformerBlock35,
        attn: &LinearAttentionLayer,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        states: &mut [Qwen35State],
        slot_indices: &[usize],
        linear_idx: usize,
        batch_size: usize,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        // 1. Input RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // 2. Batched projections (GEMM)
        ops::gemm_into(
            &self.ctx,
            &attn.in_proj_qkv,
            &bufs.normed,
            &mut bufs.qkv_batch,
        );
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z_batch);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_batch);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_batch);

        // 3. Per-request: conv1d + GDR (modifies per-request recurrent state)
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let b_dim = c.linear_num_value_heads;

        for (b, &si) in slot_indices.iter().enumerate() {
            let recurrent = &mut states[si].recurrent_state;
            let layer_state = &mut recurrent.layers[linear_idx];

            // Extract row b from batched projections
            let qkv_src = bufs.qkv_batch.data.slice(b * qkv_dim..(b + 1) * qkv_dim);
            self.ctx
                .stream
                .memcpy_dtod(&qkv_src, &mut bufs.qkv_single.data)
                .map_err(|e| anyhow::anyhow!("D2D qkv extract: {e}"))?;

            let b_src = bufs.b_batch.data.slice(b * b_dim..(b + 1) * b_dim);
            self.ctx
                .stream
                .memcpy_dtod(&b_src, &mut bufs.b_single.data)
                .map_err(|e| anyhow::anyhow!("D2D b extract: {e}"))?;

            let a_src = bufs.a_batch.data.slice(b * b_dim..(b + 1) * b_dim);
            self.ctx
                .stream
                .memcpy_dtod(&a_src, &mut bufs.a_single.data)
                .map_err(|e| anyhow::anyhow!("D2D a extract: {e}"))?;

            // Conv1d
            ops::conv1d_prefill_batch_into(
                &self.ctx,
                &bufs.qkv_single,
                &attn.conv1d_weight,
                &mut layer_state.conv_state,
                &mut bufs.qkv_conv_single,
                c.linear_conv_kernel_dim,
            );

            // GDR decode (single-step)
            ops::gated_delta_rule_decode_into(
                &self.ctx,
                &bufs.qkv_conv_single,
                &bufs.b_single,
                &bufs.a_single,
                &attn.dt_bias,
                &attn.a_log,
                &mut layer_state.state,
                &mut bufs.gdr_out_single,
                c.linear_num_key_heads,
                c.linear_num_value_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
            )?;

            // Write GDR output to batch buffer row b
            let mut gdr_dst = bufs
                .gdr_out_batch
                .data
                .slice_mut(b * z_dim..(b + 1) * z_dim);
            self.ctx
                .stream
                .memcpy_dtod(&bufs.gdr_out_single.data, &mut gdr_dst)
                .map_err(|e| anyhow::anyhow!("D2D gdr insert: {e}"))?;
        }

        // 4. Batched gated RMSNorm
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out_batch,
            &attn.norm_weight,
            &bufs.z_batch,
            &mut bufs.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            eps,
        );

        // 5. Batched out projection
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );

        // 6. Residual + post-attention norm + MLP
        self.decode_batch_mlp(layer, hidden, bufs, batch_size)?;

        Ok(())
    }

    /// Shared: residual add + post-attention norm + MLP + residual add.
    fn decode_batch_mlp(
        &self,
        layer: &TransformerBlock35,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        _batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Residual 1: hidden_mid = hidden + attn_results
        ops::add_batch_into(&self.ctx, hidden, &bufs.attn_results, &mut bufs.hidden_mid)?;

        // Post-attention RMSNorm (offset variant)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // MLP: gate + up → silu_mul → down
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

        // Residual 2: hidden = hidden_mid + mlp_out
        ops::add_batch_into(
            &self.ctx,
            &bufs.hidden_mid,
            &bufs.o_buf,
            &mut bufs.hidden_out,
        )?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
