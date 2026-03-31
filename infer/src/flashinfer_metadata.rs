//! FlashInfer decode metadata management.
//!
//! Owns the GPU-side paged KV metadata buffers (positions, indptr, indices,
//! last_page_len) and the FlashInfer workspace. Extracted from
//! `model::qwen3::batch_decode` so multiple model implementations can share it.

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::ops;
use crate::ops::FlashInferWorkspace;
use crate::paged_kv::PagedKVPool;
use crate::tensor::DeviceContext;

/// GPU-side metadata buffers for FlashInfer batched decode attention.
///
/// Manages positions, KV indptr/indices/last_page_len arrays, the FlashInfer
/// workspace, and host-side scratch buffers for incremental index updates.
pub(crate) struct FlashInferDecodeMetadata {
    pub positions: CudaSlice<i32>,
    pub kv_indices: CudaSlice<i32>,
    pub kv_indptr: CudaSlice<i32>,
    pub kv_last_page_len: CudaSlice<i32>,
    pub flashinfer_ws: FlashInferWorkspace,
    pub max_total_pages: usize,
    /// Total tokens in kv_indices from previous step (for incremental update).
    prev_total_tokens: usize,
    /// Previous batch slot indices (for detecting batch composition changes).
    prev_slot_indices: Vec<usize>,
    /// Scratch buffer for building positions on the host side.
    positions_scratch: Vec<i32>,
    /// Scratch buffer for incremental index H2D (avoids alloc).
    indices_scratch: Vec<i32>,
    /// Cached host-side indptr from last `update()`, reused by `plan()`.
    indptr_h: Vec<i32>,
}

impl FlashInferDecodeMetadata {
    /// Allocate metadata buffers for up to `max_batch_size` requests.
    /// `max_total_pages` should be large enough for the worst-case total KV pages.
    pub(crate) fn new(
        ctx: &DeviceContext,
        max_batch_size: usize,
        max_total_pages: usize,
        num_qheads: usize,
    ) -> Result<Self> {
        Ok(Self {
            positions: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc positions failed: {e}"))?,
            kv_indices: ctx
                .stream
                .alloc_zeros(max_total_pages.max(1))
                .map_err(|e| anyhow::anyhow!("Alloc kv_indices failed: {e}"))?,
            kv_indptr: ctx
                .stream
                .alloc_zeros(max_batch_size + 1)
                .map_err(|e| anyhow::anyhow!("Alloc kv_indptr failed: {e}"))?,
            kv_last_page_len: ctx
                .stream
                .alloc_zeros(max_batch_size)
                .map_err(|e| anyhow::anyhow!("Alloc kv_last_page_len failed: {e}"))?,
            flashinfer_ws: FlashInferWorkspace::new(ctx, max_batch_size, num_qheads)?,
            max_total_pages,
            prev_total_tokens: 0,
            prev_slot_indices: Vec::new(),
            positions_scratch: Vec::with_capacity(max_batch_size),
            indices_scratch: Vec::with_capacity(max_batch_size),
            indptr_h: Vec::with_capacity(max_batch_size + 1),
        })
    }

    /// Upload metadata (positions, indptr, last_page_len, KV indices) to GPU.
    ///
    /// Uses incremental index updates when the batch composition is unchanged
    /// (same slot indices as previous step), falling back to a full rebuild
    /// when the batch changes or on the first call.
    ///
    /// Returns `true` if the kv_indices GPU buffer was reallocated (caller
    /// should invalidate any CUDA graph that captured the old pointer).
    pub(crate) fn update(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool> {
        self.indptr_h = pool.build_indptr(slot_indices);
        let last_page_lens_h = pool.build_last_page_lens(slot_indices);

        // Build positions (each request's current sequence position).
        self.positions_scratch.clear();
        self.positions_scratch
            .extend(slot_indices.iter().map(|&si| (pool.seq_len(si) - 1) as i32));

        // H2D copies -- positions, indptr, last_page_len (small, fixed size).
        ctx.stream
            .memcpy_htod(&self.positions_scratch, &mut self.positions)
            .map_err(|e| anyhow::anyhow!("H2D positions: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.indptr_h, &mut self.kv_indptr)
            .map_err(|e| anyhow::anyhow!("H2D indptr: {e}"))?;
        ctx.stream
            .memcpy_htod(&last_page_lens_h, &mut self.kv_last_page_len)
            .map_err(|e| anyhow::anyhow!("H2D last_page_len: {e}"))?;

        // KV indices update: if same batch composition as last step, only do
        // an incremental update (H2D of full index array using scratch buffer).
        // Otherwise, full rebuild of the O(total_tokens) indices array.
        let same_batch = self.prev_slot_indices == slot_indices;
        let new_total = *self.indptr_h.last().unwrap() as usize;
        let mut reallocated = false;

        if same_batch && self.prev_total_tokens > 0 && new_total <= self.max_total_pages {
            // Incremental: rebuild into pre-allocated scratch buffer (avoids heap alloc).
            self.indices_scratch.clear();
            for &slot in slot_indices.iter() {
                for &idx in pool.token_indices(slot) {
                    self.indices_scratch.push(idx as i32);
                }
            }
            ctx.stream
                .memcpy_htod(&self.indices_scratch, &mut self.kv_indices)
                .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        } else {
            // Full rebuild (first call, or batch composition changed).
            let indices_h = pool.build_indices(slot_indices);
            if indices_h.len() > self.max_total_pages {
                self.kv_indices = ctx
                    .stream
                    .alloc_zeros(indices_h.len())
                    .map_err(|e| anyhow::anyhow!("Realloc kv_indices: {e}"))?;
                self.max_total_pages = indices_h.len();
                reallocated = true;
            }
            ctx.stream
                .memcpy_htod(&indices_h, &mut self.kv_indices)
                .map_err(|e| anyhow::anyhow!("H2D indices: {e}"))?;
        }
        self.prev_total_tokens = new_total;
        self.prev_slot_indices.clear();
        self.prev_slot_indices.extend_from_slice(slot_indices);

        Ok(reallocated)
    }

    /// Call FlashInfer's plan step (CPU-side scheduling).
    ///
    /// Must be called once per decode step after `update()`, before any layer
    /// runs. The plan result works for all layers since KV layout is the same.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        ops::flashinfer_plan(
            ctx,
            &self.indptr_h,
            &mut self.flashinfer_ws,
            batch_size,
            num_heads,
            num_kv_heads,
            page_size,
            head_dim,
        )
    }
}
