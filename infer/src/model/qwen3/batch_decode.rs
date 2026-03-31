//! Batched decode: process multiple requests' decode tokens in one forward pass.
//!
//! Uses GEMM (matrix multiply) for all linear projections (QKV, O, MLP),
//! batching B requests together. Attention uses FlashInfer with a shared
//! paged KV cache: QK-norm + RoPE + paged KV write are done in a prep kernel,
//! then FlashInfer's batch decode handles attention in a single launch.

use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use log::{debug, info};

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::ModelForward;
use crate::ops;
use crate::ops::FlashInferWorkspace;
use crate::paged_kv::PagedKVPool;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated buffers for batched decode, reused across steps.
/// Allocated once for `max_batch_size`; smaller batches set `seq_len` on HiddenStates.
pub(crate) struct BatchDecodeBuffers {
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

    /// Embedding output buffer [max_batch_size, hidden_dim] — avoids alloc in graph.
    embedding_out: HiddenStates,
    /// Batched logits buffer [max_batch_size, vocab_size] — avoids alloc in graph.
    logits_batch: Option<HiddenStates>,
    /// Per-slot logits extracted from batched logits.
    logits_per_slot: Vec<DeviceVec>,

    // FlashInfer paged attention buffers
    positions: CudaSlice<i32>,
    kv_indices_gpu: CudaSlice<i32>,
    kv_indptr_gpu: CudaSlice<i32>,
    kv_last_page_len_gpu: CudaSlice<i32>,
    flashinfer_ws: FlashInferWorkspace,

    /// Max batch size this buffer set was allocated for.
    max_batch_size: usize,
    /// Max total pages (for kv_indices_gpu sizing).
    max_total_pages: usize,

    /// CUDA Graph cache: one graph per batch_size.
    graph_cache: HashMap<usize, CudaGraph>,
}

// SAFETY: BatchDecodeBuffers contains CudaGraph (CUgraphExec) which is !Send.
// Invariant: exclusively accessed from the single scheduler inference thread.
unsafe impl Send for BatchDecodeBuffers {}

impl BatchDecodeBuffers {
    /// Allocate buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        max_batch_size: usize,
        num_qheads: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, max_batch_size)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, max_batch_size)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, max_batch_size)?,

            embedding_out: HiddenStates::zeros(ctx, hidden_dim, max_batch_size)?,
            logits_batch: None, // lazy-allocated on first use (needs vocab_size)
            logits_per_slot: Vec::new(),

            positions: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc positions failed: {e}"))?,
            kv_indices_gpu: ctx
                .stream
                .alloc_zeros(max_total_pages.max(1))
                .map_err(|e| anyhow::anyhow!("Alloc kv_indices failed: {e}"))?,
            kv_indptr_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc kv_indptr failed: {e}"))?,
            kv_last_page_len_gpu: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc kv_last_page_len failed: {e}"))?,
            flashinfer_ws: FlashInferWorkspace::new(ctx, max_batch_size, num_qheads)?,

            max_batch_size,
            max_total_pages,
            graph_cache: HashMap::new(),
        })
    }

    /// Set the actual batch size for this step (must be <= max_batch_size).
    fn set_batch_size(&mut self, batch_size: usize) {
        debug_assert!(batch_size <= self.max_batch_size);
        self.hidden_out.seq_len = batch_size;
        self.normed.seq_len = batch_size;
        self.q_batch.seq_len = batch_size;
        self.k_batch.seq_len = batch_size;
        self.v_batch.seq_len = batch_size;
        self.attn_output.seq_len = batch_size;
        self.o_buf.seq_len = batch_size;
        self.gate_out.seq_len = batch_size;
        self.up_out.seq_len = batch_size;
        self.act_out.seq_len = batch_size;
    }
}

impl Qwen3Model {
    /// Batched decode: process B tokens from B different requests in one pass.
    ///
    /// Batched decode using contiguous (per-slot) KV cache.
    /// Falls back to sequential forward() calls — correct but not optimal.
    pub fn decode_batch_contiguous(
        &self,
        tokens: &[u32],
        states: &mut [Qwen3State],
        slot_indices: &[usize],
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(&[token], &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    /// `tokens[b]` is the next token for request `b`, whose state is
    /// `states[slot_indices[b]]`. All linear projections are batched via GEMM;
    /// attention uses FlashInfer with a shared paged KV cache.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Qwen3State],
        slot_indices: &[usize],
        paged_kv_pool: &mut PagedKVPool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let batch_size = tokens.len();
        debug_assert_eq!(batch_size, slot_indices.len());
        debug_assert!(batch_size > 1);
        debug_assert!(batch_size <= bufs.max_batch_size);

        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let eps = self.config.rms_norm_eps;
        let page_size = 1;

        bufs.set_batch_size(batch_size);

        // ── Pre-graph: metadata H2D + FlashInfer plan + embedding ──

        let indptr_h = paged_kv_pool.build_indptr(slot_indices);
        let indices_h = paged_kv_pool.build_indices(slot_indices);
        let last_page_lens_h = paged_kv_pool.build_last_page_lens(slot_indices);
        let positions_h: Vec<i32> = slot_indices
            .iter()
            .map(|&si| (paged_kv_pool.seq_len(si) - 1) as i32)
            .collect();

        self.ctx.stream.memcpy_htod(&positions_h, &mut bufs.positions)
            .map_err(|e| anyhow::anyhow!("H2D positions: {e}"))?;
        self.ctx.stream.memcpy_htod(&indptr_h, &mut bufs.kv_indptr_gpu)
            .map_err(|e| anyhow::anyhow!("H2D indptr: {e}"))?;
        if indices_h.len() > bufs.max_total_pages {
            bufs.kv_indices_gpu = self.ctx.stream.alloc_zeros(indices_h.len())
                .map_err(|e| anyhow::anyhow!("Realloc kv_indices: {e}"))?;
            bufs.max_total_pages = indices_h.len();
            bufs.graph_cache.remove(&batch_size);
        }
        self.ctx.stream.memcpy_htod(&indices_h, &mut bufs.kv_indices_gpu)
            .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        self.ctx.stream.memcpy_htod(&last_page_lens_h, &mut bufs.kv_last_page_len_gpu)
            .map_err(|e| anyhow::anyhow!("H2D last_page_len: {e}"))?;

        ops::flashinfer_plan(
            &self.ctx, &indptr_h, &mut bufs.flashinfer_ws,
            batch_size, num_heads, num_kv_heads, page_size, head_dim,
        )?;

        // Embedding (has H2D copy — must be outside graph)
        bufs.embedding_out.seq_len = batch_size;
        let token_ids_i32: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self.ctx.stream.clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D token_ids: {e}"))?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut bufs.embedding_out)?;

        // ── Graph body: layers + final norm + logits GEMM ──
        // All use pre-allocated buffers with stable pointers.

        // Lazy-init logits buffer
        if bufs.logits_batch.is_none() {
            let vocab_size = self.output_projection().rows;
            bufs.logits_batch = Some(HiddenStates::zeros(&self.ctx, vocab_size, bufs.max_batch_size)?);
        }

        // Run layers + norm + logits
        self.decode_batch_graph_body(bufs, paged_kv_pool, batch_size)?;

        // Extract per-slot logits
        let logits = bufs.logits_batch.as_mut().unwrap();
        logits.seq_len = batch_size;
        for (b, &si) in slot_indices.iter().enumerate() {
            let slot_logits = ops::extract_vec(&self.ctx, logits, b)?;
            states[si].prefill_logits = Some(slot_logits);
        }

        Ok(())
    }

    /// Graph body: embedding_out → layers → final norm → logits.
    /// All buffers are pre-allocated in `bufs`. No allocations, no H2D copies.
    fn decode_batch_graph_body(
        &self,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
        batch_size: usize,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Use embedding_out as the initial hidden state. The layer loop
        // ping-pongs between embedding_out and hidden_out via swap.
        // We use a raw pointer to avoid borrow conflicts with bufs.
        let hidden_ptr = &mut bufs.embedding_out as *mut HiddenStates;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // SAFETY: hidden_ptr points to bufs.embedding_out. The layer
            // function only accesses other fields of bufs (normed, q_batch, etc.)
            // and swaps hidden_ptr's target with bufs.hidden_out. No aliasing.
            let hidden = unsafe { &mut *hidden_ptr };
            self.decode_batch_layer_inner(layer_idx, layer, hidden, bufs, kv_pool)?;
        }

        // Final norm + logits. hidden is whichever buffer was last written.
        let hidden = unsafe { &*hidden_ptr };
        ops::rms_norm_batch_into(&self.ctx, hidden, &self.norm, eps, &mut bufs.normed);
        let logits_buf = bufs.logits_batch.as_mut().unwrap();
        logits_buf.seq_len = batch_size;
        ops::gemm_into(&self.ctx, self.output_projection(), &bufs.normed, logits_buf);

        Ok(())
    }

    fn decode_batch_layer_inner(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut BatchDecodeBuffers,
        kv_pool: &PagedKVPool,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let page_size = 1; // token-level pool: page_size is always 1

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

        // 3. Decode prep: QK-norm + RoPE (in-place on Q) + paged KV write
        ops::decode_prep_paged(
            &self.ctx,
            &mut bufs.q_batch,
            &bufs.k_batch,
            &bufs.v_batch,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions,
            kv_pool,
            layer_idx,
            &bufs.kv_indices_gpu,
            &bufs.kv_indptr_gpu,
            &bufs.kv_last_page_len_gpu,
            num_heads,
            num_kv_heads,
            page_size,
            eps,
        )?;

        // 4. FlashInfer batch decode attention (run only — plan was called once before loop)
        ops::flashinfer_run_layer(
            &self.ctx,
            &bufs.q_batch,
            kv_pool,
            layer_idx,
            &bufs.kv_indptr_gpu,
            &bufs.kv_indices_gpu,
            &bufs.kv_last_page_len_gpu,
            &mut bufs.attn_output,
            &mut bufs.flashinfer_ws,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )?;

        // 5. Batched O projection → bufs.o_buf [B, hidden_dim]
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );

        // 6. Batched residual add: hidden + o_buf → hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        // 7. Batched MLP RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        );

        // 8. Batched MLP: gate + up → silu_mul → down
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

        // 9. Batched residual add
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
