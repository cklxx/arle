use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::kv_cache::KVCache;
use crate::ops;
use cuda_kernels::TokenKVPool;
use cuda_kernels::flashinfer::BatchPrefillPagedPlan;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for one prefill forward pass.
///
/// Model-owned, lazily-initialized, mutex-guarded shared buffer pool sized
/// for the scheduler's `prefill_chunk_size`. Ports sglang's
/// `global_workspace_buffer` pattern (`flashinfer_backend.py:202-219`): one
/// allocation per model, reused across all subsequent prefills. Eliminates
/// ~11k `cuMemAllocAsync` calls per forward AND eliminates the async
/// alloc/free backlog that poisons the CUDA stream under c=16 concurrency
/// at chunk_size=2048 (manifests as `gemm_cuda CUDA_ERROR_UNKNOWN`).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
///
/// ## Capacity semantics
/// - `capacity_tokens` records the guaranteed minimum token-count that all
///   non-`hidden_out` buffers can hold without reallocation. Callers must
///   invoke `ensure_capacity` before using the buffers for a chunk of
///   length N; a larger N triggers monotonic regrowth of every buffer.
/// - `hidden_out` is special: the forward-layer code `std::mem::swap`s it
///   with the incoming `hidden` on every residual add, so by the time a
///   subsequent forward starts, `hidden_out` may hold a smaller (embedding-
///   sized) buffer than `capacity_tokens`. `ensure_capacity` therefore
///   checks `hidden_out.data.len()` independently of `capacity_tokens`.
pub(super) struct PrefillBuffers {
    /// Last guaranteed capacity (tokens) for non-`hidden_out` buffers.
    capacity_tokens: usize,
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    /// fp32 shadow of the residual stream. Maintained across layers so that
    /// per-layer bf16 outputs accumulate into fp32 without compounding
    /// ~0.4% bf16 rounding noise at each residual add. Norm reads from here
    /// directly to avoid a further bf16 round-trip on the hidden state.
    /// `None` unless `INFER_QWEN3_FP32_RESIDUAL=1` is set.
    residual_f32: Option<CudaSlice<f32>>,
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    pub(super) fn new_with_capacity(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        // `residual_f32` stays `None` here: `rms_norm_batch_f32_in_into` asserts
        // strict equality between `x_f32.len()` and `hidden_dim * seq_len`, so
        // the f32 shadow must be sized per-chunk rather than to the pool
        // capacity. `ensure_capacity` sizes it to the active seq_len below.
        Ok(Self {
            capacity_tokens,
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, capacity_tokens)?,
            residual_f32: None,
            normed: HiddenStates::zeros(ctx, hidden_dim, capacity_tokens)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, capacity_tokens)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, capacity_tokens)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, capacity_tokens)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, capacity_tokens)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, capacity_tokens)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, capacity_tokens)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, capacity_tokens)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, capacity_tokens)?,
        })
    }

    /// Grow buffers if `required` exceeds the last guaranteed capacity.
    /// Also repairs `hidden_out` if a prior forward's `std::mem::swap` left
    /// it holding a smaller (embedding-sized) buffer, and (re)allocates
    /// `residual_f32` to match the active chunk length when the fp32
    /// residual shadow is enabled (opt-in diagnostic, not on the hot path).
    pub(super) fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        required: usize,
    ) -> Result<()> {
        if required > self.capacity_tokens {
            // Monotonic regrowth: reallocate every bf16 buffer at the new max.
            *self = Self::new_with_capacity(ctx, hidden_dim, q_dim, kv_dim, inter_dim, required)?;
        } else if self.hidden_out.data.len() < hidden_dim * required {
            // Even when `capacity_tokens` is sufficient, `hidden_out` may have
            // been swapped with a smaller incoming `hidden` on the previous
            // forward — check its actual storage independently.
            self.hidden_out = HiddenStates::zeros(ctx, hidden_dim, self.capacity_tokens)?;
        }
        // `residual_f32` must match `hidden_dim * seq_len` exactly
        // (`rms_norm_batch_f32_in_into` strict-equality assert). Allocate
        // only when the env flag is set; size to the active chunk.
        if std::env::var("INFER_QWEN3_FP32_RESIDUAL").is_ok() {
            let needed = hidden_dim * required;
            let realloc = match &self.residual_f32 {
                Some(r) => r.len() != needed,
                None => true,
            };
            if realloc {
                self.residual_f32 = Some(
                    ctx.stream
                        .alloc_zeros::<f32>(needed)
                        .map_err(|e| anyhow::anyhow!("alloc residual_f32: {e}"))?,
                );
            }
        } else {
            self.residual_f32 = None;
        }
        Ok(())
    }

    /// Set the `seq_len` field on every scratch buffer to the active chunk
    /// length. The underlying `data` capacity is already >= `seq_len` thanks
    /// to `ensure_capacity`; updating `seq_len` makes the `_into` ops'
    /// `assert_eq!(out.seq_len, x.seq_len)` shape checks pass.
    fn set_active_seq_len(&mut self, seq_len: usize) {
        self.hidden_out.seq_len = seq_len;
        self.normed.seq_len = seq_len;
        self.q_batch.seq_len = seq_len;
        self.k_batch.seq_len = seq_len;
        self.v_batch.seq_len = seq_len;
        self.o_buf.seq_len = seq_len;
        self.gate_out.seq_len = seq_len;
        self.up_out.seq_len = seq_len;
        self.act_out.seq_len = seq_len;
        self.attn_output.seq_len = seq_len;
    }
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        crate::model::common::get_embeddings_batch(
            &self.ctx,
            &self.embed_tokens,
            token_ids,
            self.config.hidden_size,
        )
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    pub(super) fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Model-owned scratch pool: eliminates ~11k cuMemAllocAsync calls per
        // forward and keeps the CUDA stream's async alloc/free backlog
        // bounded. See `PrefillBuffers` doc comment.
        let mut bufs_guard = self
            .prefill_buffers
            .lock()
            .map_err(|_| anyhow::anyhow!("prefill_buffers mutex poisoned"))?;
        let bufs = match &mut *bufs_guard {
            Some(existing) => {
                existing.ensure_capacity(
                    &self.ctx,
                    self.config.hidden_size,
                    q_dim,
                    kv_dim,
                    inter_dim,
                    seq_len,
                )?;
                existing
            }
            slot @ None => {
                *slot = Some(PrefillBuffers::new_with_capacity(
                    &self.ctx,
                    self.config.hidden_size,
                    q_dim,
                    kv_dim,
                    inter_dim,
                    seq_len,
                )?);
                slot.as_mut().expect("just populated")
            }
        };
        bufs.set_active_seq_len(seq_len);

        // If fp32 residual shadow is enabled, seed it from the bf16 embedding.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::cast_bf16_to_f32(&self.ctx, &hidden, r)?;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(layer_idx, layer, &mut hidden, start_pos, kv_cache, bufs)?;
        }

        // If fp32 residual shadow was active, convert back to bf16 for the
        // final norm + LM head which still consume bf16.
        if let Some(ref r) = bufs.residual_f32 {
            ops::cast_f32_to_bf16(&self.ctx, r, &mut hidden)?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    /// Paged-KV prefill. Writes K/V directly to the paged pool via
    /// page-table indirection and runs FlashInfer `BatchPrefillWithPagedKVCache`
    /// for attention. No contiguous KV cache is touched; the scheduler must
    /// skip `migrate_kv_range_to_paged` for this forward.
    ///
    /// Callable only when the scheduler has already pre-allocated pool pages
    /// for the chunk (`pool.page_indices(slot)` covers `[0, start_pos+seq_len)`).
    #[fastrace::trace(name = "process_all_layers_batch_paged")]
    pub(super) fn process_all_layers_batch_paged(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        pool: &TokenKVPool,
        slot: usize,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Model-owned scratch pool (see `PrefillBuffers` doc). Guard is held
        // for the full forward; `paged_prefill_plan` is acquired afterwards
        // so the lock-acquisition order is stable across all prefill paths.
        let mut bufs_guard = self
            .prefill_buffers
            .lock()
            .map_err(|_| anyhow::anyhow!("prefill_buffers mutex poisoned"))?;
        let bufs = match &mut *bufs_guard {
            Some(existing) => {
                existing.ensure_capacity(
                    &self.ctx,
                    self.config.hidden_size,
                    q_dim,
                    kv_dim,
                    inter_dim,
                    seq_len,
                )?;
                existing
            }
            slot @ None => {
                *slot = Some(PrefillBuffers::new_with_capacity(
                    &self.ctx,
                    self.config.hidden_size,
                    q_dim,
                    kv_dim,
                    inter_dim,
                    seq_len,
                )?);
                slot.as_mut().expect("just populated")
            }
        };
        bufs.set_active_seq_len(seq_len);

        // Per-forward GPU-resident page table (1 i32 per logical page in the
        // slot). Uploaded once and reused across all 36 layers.
        let all_pages = pool.page_indices(slot);
        let num_pages_needed = (start_pos + seq_len).div_ceil(pool.page_size);
        anyhow::ensure!(
            all_pages.len() >= num_pages_needed,
            "paged prefill: slot {slot} has {} pages, expected at least {num_pages_needed}",
            all_pages.len()
        );
        let pages_u32 = &all_pages[..num_pages_needed];
        let pages_i32: Vec<i32> = pages_u32.iter().map(|&p| p as i32).collect();
        let slot_page_indices: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&pages_i32)
            .map_err(|e| anyhow::anyhow!("slot_page_indices H2D failed: {e}"))?;

        // Lazy-init the shared paged-prefill plan on first call and reuse
        // across all subsequent prefills. Allocating a fresh 264MB
        // FlashInferWorkspace per forward caused async-free backlog →
        // foreign C++ exception under concurrent load. Matches sglang's
        // single `workspace_buffer` pattern (`flashinfer_backend.py:260-292`).
        let mut plan_guard = self
            .paged_prefill_plan
            .lock()
            .map_err(|_| anyhow::anyhow!("paged_prefill_plan mutex poisoned"))?;
        if plan_guard.is_none() {
            // Size the plan for the maximum chunk we'll ever see (one
            // prefill_chunk_size worth of tokens).
            *plan_guard = Some(BatchPrefillPagedPlan::new(
                &self.ctx,
                seq_len.max(4096),
                num_heads,
            )?);
        }
        let plan = plan_guard.as_mut().expect("just initialized");

        // Structural invariant: plan ONCE per forward, not per layer.
        // All 36 layers share the same (batch_size, qo_len, kv_len,
        // num_heads, page_size) shape, so one plan call covers them all.
        // Calling plan per layer overwrites `page_locked_workspace`
        // (host pinned) while a prior layer's `cudaMemcpyAsync` is still
        // queued on the compute stream — classic async-memcpy data race
        // that corrupts FlashInfer's `int_workspace` and poisons the
        // CUDA context under bench concurrency.
        let mut fwd = crate::ops::PagedPrefillForward::new_hd128(
            &self.ctx,
            plan,
            seq_len,
            start_pos,
            num_heads,
            num_kv_heads,
            pool.page_size,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_paged(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                bufs,
                pool,
                &slot_page_indices,
                &mut fwd,
            )?;
        }

        Ok(hidden)
    }

    /// Paged-KV variant of `forward_layer_batch`. Differences vs the contiguous
    /// path:
    ///  - No `kv_cache.init_if_needed` / `prepare_layer` / `commit_layer`.
    ///  - Attention call writes K/V directly into the paged pool through the
    ///    page-table indirection kernel + FlashInfer paged-prefill.
    ///  - No `scatter_write_kv` dual-write step.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        bufs: &mut PrefillBuffers,
        pool: &TokenKVPool,
        slot_page_indices: &CudaSlice<i32>,
        fwd: &mut crate::ops::PagedPrefillForward,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 2. QKV projections
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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }

        // 3. Paged-KV attention: QK norm + RoPE + paged K/V write (page-table
        //    indirection) + FlashInfer BatchPrefillWithPagedKVCache.
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        let meta = ops::PagedPrefillMeta {
            pool,
            layer_idx,
            slot_page_indices,
            start_pos,
            page_size: pool.page_size,
        };
        ops::prefill_attention_paged_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            &meta,
            fwd,
            &mut bufs.attn_output,
            &heads,
        )?;

        // 4-8: Same as forward_layer_batch (O proj, residual, MLP)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.gate_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.normed,
                    &mut bufs.gate_out,
                )?;
            }
            if let Some(ad) = ll.up_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.up_out)?;
            }
        }
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.act_out, &mut bufs.o_buf)?;
            }
        }

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }

    pub(super) fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        crate::model::common::compute_logits_batch(
            &self.ctx,
            hidden,
            &self.norm,
            self.output_projection(),
            self.config.rms_norm_eps,
            false, // standard RMSNorm (not offset)
        )
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm → bufs.normed. When the fp32 residual shadow is active,
        //    read from it directly — skipping the bf16 rounding in `hidden`.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.input_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            );
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.normed,
            &format!("L{layer_idx} after-input-norm"),
            self.config.hidden_size,
        );

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q_proj_out (pre-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.k_batch,
            &format!("L{layer_idx} k_proj_out (pre-norm-rope)"),
            bufs.k_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.v_batch,
            &format!("L{layer_idx} v_proj_out"),
            bufs.v_batch.hidden_dim,
        );

        // 3. FlashAttention-2 (Triton) → bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            &heads,
            start_pos,
        )?;
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q (post-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.attn_output,
            &format!("L{layer_idx} attn_output (pre-o-proj)"),
            bufs.attn_output.hidden_dim,
        );
        // Quantize newly written KV tokens → INT8 storage (no-op for BF16)
        kv_cache.commit_layer(&self.ctx, layer_idx, start_pos, hidden.seq_len)?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.o_buf,
            &format!("L{layer_idx} o_proj_out"),
            bufs.o_buf.hidden_dim,
        );

        // 5. Residual add: hidden + o_buf.
        //    With fp32 shadow: accumulate into residual_f32 (fp32 precision),
        //    then sync hidden for downstream bf16 consumers / debug dumps.
        //    Without shadow: use the classic bf16 add + swap path.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        // 6. MLP RMSNorm → bufs.normed.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.post_attention_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            );
        }

        // 7. MLP: gate + up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
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
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.gate_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.normed,
                    &mut bufs.gate_out,
                )?;
            }
            if let Some(ad) = ll.up_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.up_out)?;
            }
        }
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.act_out, &mut bufs.o_buf)?;
            }
        }

        // 8. Residual add: attn_residual + mlp_out.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }
}
