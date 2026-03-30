//! KV Cache — contiguous buffers for fused attention, with CPU offload support.
//!
//! When the sequence length exceeds `max_gpu_seq_len`, the oldest KV blocks are
//! offloaded to CPU (host) memory. Before attention kernels run, `ensure_on_gpu()`
//! restores the full sequence to GPU so the kernels see a contiguous `0..seq_len` range.
//! After attention, `offload_to_host()` moves the prefix back to CPU to free GPU memory.

use anyhow::Result;
use half::bf16;
use log::info;

use crate::tensor::{DeviceContext, DeviceVec};

/// Block size for offloading (in tokens). Offload happens in multiples of this.
const OFFLOAD_BLOCK_SIZE: usize = 64;

/// KV Cache — contiguous buffers for fused attention.
pub(crate) struct KVCache {
    // [layer] -> contiguous buffer (num_kv_heads * max_seq * head_dim)
    k_cache: Vec<DeviceVec>,
    v_cache: Vec<DeviceVec>,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,
    num_kv_heads: usize,
    max_seq_len: usize,

    // --- CPU offload fields ---
    /// Maximum tokens to keep on GPU. When seq_len exceeds this, oldest tokens
    /// are offloaded to CPU. Defaults to max_seq_len (no offload).
    max_gpu_seq_len: usize,
    /// Number of tokens currently offloaded to CPU host memory.
    offloaded_len: usize,
    /// CPU shadow buffers for offloaded K data. [layer] -> Vec<bf16>.
    /// Each stores `offloaded_len * num_kv_heads * head_dim` elements.
    k_host: Vec<Vec<bf16>>,
    /// CPU shadow buffers for offloaded V data. [layer] -> Vec<bf16>.
    v_host: Vec<Vec<bf16>>,
    /// Whether the GPU buffers currently contain the full sequence (including
    /// data restored from CPU). This is set by `ensure_on_gpu()` and cleared
    /// by `offload_to_host()`.
    gpu_has_full_seq: bool,
}

impl KVCache {
    pub(crate) fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            seq_len: 0,
            head_dim: 0,
            num_layers,
            num_kv_heads,
            max_seq_len: 32768,
            max_gpu_seq_len: 32768,
            offloaded_len: 0,
            k_host: Vec::new(),
            v_host: Vec::new(),
            gpu_has_full_seq: true,
        }
    }

    /// Set the maximum number of tokens to keep on GPU.
    /// Tokens beyond this limit will be offloaded to CPU.
    /// Must be called before `init_if_needed()`. The value is rounded down
    /// to the nearest `OFFLOAD_BLOCK_SIZE` boundary.
    pub(crate) fn set_max_gpu_seq_len(&mut self, max_gpu: usize) {
        // Round down to block boundary so offloads are block-aligned.
        let aligned = (max_gpu / OFFLOAD_BLOCK_SIZE) * OFFLOAD_BLOCK_SIZE;
        // Ensure at least one block stays on GPU.
        self.max_gpu_seq_len = aligned.max(OFFLOAD_BLOCK_SIZE);
        info!(
            "KV cache: max_gpu_seq_len set to {} tokens ({} blocks of {})",
            self.max_gpu_seq_len,
            self.max_gpu_seq_len / OFFLOAD_BLOCK_SIZE,
            OFFLOAD_BLOCK_SIZE,
        );
    }

    /// Set the maximum sequence length (total, GPU + CPU).
    /// Must be called before `init_if_needed()`.
    pub(crate) fn set_max_seq_len(&mut self, max_seq: usize) {
        self.max_seq_len = max_seq;
        // If max_gpu_seq_len hasn't been explicitly set (still at old default),
        // update it to match.
        if self.max_gpu_seq_len == 32768 && max_seq != 32768 {
            self.max_gpu_seq_len = max_seq;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.seq_len
    }

    /// Returns true if offloading is active (some tokens are on CPU).
    pub(crate) fn has_offloaded(&self) -> bool {
        self.offloaded_len > 0
    }

    /// Number of tokens currently offloaded to CPU.
    pub(crate) fn offloaded_len(&self) -> usize {
        self.offloaded_len
    }

    /// Number of tokens currently on GPU.
    pub(crate) fn gpu_seq_len(&self) -> usize {
        self.seq_len - self.offloaded_len
    }

    /// Elements per token per layer in the KV cache (num_kv_heads * head_dim).
    fn elems_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Get mutable references to K/V cache for a layer
    pub(crate) fn get_cache_mut(
        &mut self,
        ctx: &DeviceContext,
        layer: usize,
    ) -> Result<(&mut DeviceVec, &mut DeviceVec)> {
        // Initialize on first access
        if self.k_cache.is_empty() {
            for _ in 0..self.num_layers {
                // Allocate max size upfront
                let cache_size = self.num_kv_heads * self.max_seq_len * self.head_dim;
                self.k_cache.push(DeviceVec::zeros(ctx, cache_size)?);
                self.v_cache.push(DeviceVec::zeros(ctx, cache_size)?);
            }
        }
        Ok((&mut self.k_cache[layer], &mut self.v_cache[layer]))
    }

    pub(crate) fn init_if_needed(&mut self, ctx: &DeviceContext, head_dim: usize) -> Result<()> {
        if self.head_dim == 0 {
            self.head_dim = head_dim;
            for _ in 0..self.num_layers {
                let cache_size = self.num_kv_heads * self.max_seq_len * head_dim;
                self.k_cache.push(DeviceVec::zeros(ctx, cache_size)?);
                self.v_cache.push(DeviceVec::zeros(ctx, cache_size)?);
            }
            // Initialize host buffers (empty initially).
            self.k_host = vec![Vec::new(); self.num_layers];
            self.v_host = vec![Vec::new(); self.num_layers];
        }
        Ok(())
    }

    pub(crate) fn increment_seq_len(&mut self) {
        self.seq_len += 1;
    }

    pub(crate) fn advance_seq_len(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Truncate KV cache to `new_len` tokens, discarding everything after.
    /// Used for partial prefix reuse: keep the common prefix, discard divergent suffix.
    /// If data was offloaded to CPU, keeps offloaded data up to `new_len`.
    pub(crate) fn truncate_to(&mut self, new_len: usize) {
        if new_len >= self.seq_len {
            return; // Nothing to truncate
        }
        if new_len == 0 {
            self.reset();
            return;
        }
        if new_len <= self.offloaded_len {
            // Truncation point is within the offloaded region.
            // Keep only the CPU data up to new_len.
            let ept = self.elems_per_token();
            let keep_elems = new_len * ept;
            for buf in &mut self.k_host {
                buf.truncate(keep_elems);
            }
            for buf in &mut self.v_host {
                buf.truncate(keep_elems);
            }
            self.offloaded_len = new_len;
            // GPU portion is now empty (all remaining data was past new_len).
            self.seq_len = new_len;
            self.gpu_has_full_seq = false;
        } else {
            // Truncation point is within the GPU region.
            // Just update seq_len; GPU data past new_len is stale but won't be read.
            self.seq_len = new_len;
        }
        info!(
            "KV cache truncated to {} tokens (offloaded: {}, gpu: {})",
            self.seq_len, self.offloaded_len, self.gpu_seq_len()
        );
    }

    /// Prefetch offloaded KV data from CPU back to GPU.
    /// Call this before prefill to ensure the full prefix is on GPU.
    /// This is a no-op if nothing is offloaded.
    pub(crate) fn prefetch_to_gpu(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || self.gpu_has_full_seq {
            return Ok(());
        }
        info!(
            "KV cache prefetch: restoring {} tokens from CPU to GPU",
            self.offloaded_len
        );
        self.ensure_on_gpu(ctx)?;
        Ok(())
    }

    /// Reset sequence length to 0 for reuse across requests.
    /// Keeps allocated GPU buffers (stable GPU pointers for CUDA Graph replay).
    /// Clears CPU offload state.
    pub(crate) fn reset(&mut self) {
        self.seq_len = 0;
        self.offloaded_len = 0;
        self.gpu_has_full_seq = true;
        // Clear host buffers but keep the Vec allocations.
        for buf in &mut self.k_host {
            buf.clear();
        }
        for buf in &mut self.v_host {
            buf.clear();
        }
    }

    /// Offload oldest KV blocks to CPU if GPU seq_len exceeds the budget.
    ///
    /// Call this after `advance_seq_len()` or `increment_seq_len()` when the
    /// GPU buffer might be over capacity. This copies the oldest blocks to CPU
    /// and shifts the remaining GPU data to the start of the buffer.
    ///
    /// The GPU buffer layout after offload:
    /// - Positions `0..gpu_seq_len` contain the most recent tokens.
    /// - The CUDA kernels will be told `seq_len = gpu_seq_len` until
    ///   `ensure_on_gpu()` is called.
    pub(crate) fn offload_if_needed(&mut self, ctx: &DeviceContext) -> Result<()> {
        // If the full sequence was restored to GPU by ensure_on_gpu(), we need to
        // first move the prefix back to its CPU-only state before computing what
        // else needs offloading. Otherwise we'd re-offload already-offloaded data.
        if self.gpu_has_full_seq && self.offloaded_len > 0 {
            // GPU has [0..seq_len] with [0..offloaded_len] being the restored prefix.
            // Shift the new portion left so GPU has [0..gpu_tokens] = only the new data.
            let ept = self.elems_per_token();
            let offloaded_elems = self.offloaded_len * ept;
            let gpu_tokens = self.seq_len - self.offloaded_len;
            let gpu_elems = gpu_tokens * ept;

            if gpu_elems > 0 {
                for layer in 0..self.num_layers {
                    let mut temp = vec![bf16::ZERO; gpu_elems];

                    self.k_cache[layer].copy_region_to_host(
                        ctx, offloaded_elems, gpu_elems, &mut temp,
                    )?;
                    self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                    self.v_cache[layer].copy_region_to_host(
                        ctx, offloaded_elems, gpu_elems, &mut temp,
                    )?;
                    self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
                }
                ctx.sync()?;
            }
            self.gpu_has_full_seq = false;
        }

        let gpu_tokens = self.seq_len - self.offloaded_len;
        if gpu_tokens <= self.max_gpu_seq_len {
            return Ok(());
        }

        // Calculate how many tokens to offload (in whole blocks).
        let excess = gpu_tokens - self.max_gpu_seq_len;
        let blocks_to_offload =
            (excess + OFFLOAD_BLOCK_SIZE - 1) / OFFLOAD_BLOCK_SIZE;
        let tokens_to_offload = blocks_to_offload * OFFLOAD_BLOCK_SIZE;
        // Don't offload more than what's on GPU.
        let tokens_to_offload = tokens_to_offload.min(gpu_tokens.saturating_sub(OFFLOAD_BLOCK_SIZE));

        if tokens_to_offload == 0 {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offload_elems = tokens_to_offload * ept;

        info!(
            "KV cache offload: moving {} tokens ({} blocks) to CPU (total offloaded: {})",
            tokens_to_offload,
            tokens_to_offload / OFFLOAD_BLOCK_SIZE,
            self.offloaded_len + tokens_to_offload,
        );

        // GPU buffer now has [0..gpu_tokens] = only the non-offloaded portion.
        // Copy oldest `tokens_to_offload` tokens to CPU, then shift remaining left.
        for layer in 0..self.num_layers {
            let mut host_buf = vec![bf16::ZERO; offload_elems];

            self.k_cache[layer].copy_region_to_host(ctx, 0, offload_elems, &mut host_buf)?;
            self.k_host[layer].extend_from_slice(&host_buf);

            self.v_cache[layer].copy_region_to_host(ctx, 0, offload_elems, &mut host_buf)?;
            self.v_host[layer].extend_from_slice(&host_buf);

            let remaining_tokens = gpu_tokens - tokens_to_offload;
            let remaining_elems = remaining_tokens * ept;
            if remaining_elems > 0 {
                let src_offset = offload_elems;
                let mut temp = vec![bf16::ZERO; remaining_elems];

                self.k_cache[layer].copy_region_to_host(
                    ctx, src_offset, remaining_elems, &mut temp,
                )?;
                self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                self.v_cache[layer].copy_region_to_host(
                    ctx, src_offset, remaining_elems, &mut temp,
                )?;
                self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
            }
        }

        ctx.sync()?;

        self.offloaded_len += tokens_to_offload;
        self.gpu_has_full_seq = false;

        Ok(())
    }

    /// Ensure the full sequence (including offloaded tokens) is on GPU.
    ///
    /// Call this before attention kernels that need to scan the full KV range.
    /// This copies CPU-offloaded data back to the GPU buffer, shifting current
    /// GPU data to make room for the restored prefix.
    ///
    /// After this call, the GPU buffer contains `0..seq_len` contiguously,
    /// and `gpu_has_full_seq` is true.
    pub(crate) fn ensure_on_gpu(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || self.gpu_has_full_seq {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offloaded_elems = self.offloaded_len * ept;
        let gpu_tokens = self.seq_len - self.offloaded_len;
        let gpu_elems = gpu_tokens * ept;

        info!(
            "KV cache restore: copying {} offloaded tokens back to GPU",
            self.offloaded_len,
        );

        for layer in 0..self.num_layers {
            // 1. Shift current GPU data right to make room for the prefix.
            // Move [0..gpu_elems] to [offloaded_elems..offloaded_elems+gpu_elems].
            // Bounce through host to avoid borrow conflicts on CudaSlice.
            if gpu_elems > 0 {
                let mut temp = vec![bf16::ZERO; gpu_elems];

                self.k_cache[layer].copy_region_to_host(ctx, 0, gpu_elems, &mut temp)?;
                self.k_cache[layer].copy_region_from_host(ctx, offloaded_elems, &temp)?;

                self.v_cache[layer].copy_region_to_host(ctx, 0, gpu_elems, &mut temp)?;
                self.v_cache[layer].copy_region_from_host(ctx, offloaded_elems, &temp)?;
            }

            // 2. Copy offloaded prefix from CPU to GPU [0..offloaded_elems].
            self.k_cache[layer].copy_region_from_host(
                ctx,
                0,
                &self.k_host[layer],
            )?;
            self.v_cache[layer].copy_region_from_host(
                ctx,
                0,
                &self.v_host[layer],
            )?;
        }

        ctx.sync()?;
        self.gpu_has_full_seq = true;

        Ok(())
    }

    /// Move restored prefix data back to CPU after attention is done.
    ///
    /// Call this after attention kernels have finished reading the full sequence.
    /// Shifts GPU data back to position 0 so new tokens can be appended at
    /// the correct offset within the GPU-resident portion.
    pub(crate) fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.offloaded_len == 0 || !self.gpu_has_full_seq {
            return Ok(());
        }

        let ept = self.elems_per_token();
        let offloaded_elems = self.offloaded_len * ept;
        let gpu_tokens = self.seq_len - self.offloaded_len;
        let gpu_elems = gpu_tokens * ept;

        for layer in 0..self.num_layers {
            // Shift GPU data left: move [offloaded_elems..offloaded_elems+gpu_elems]
            // to [0..gpu_elems]. Bounce through host to avoid borrow conflicts.
            if gpu_elems > 0 {
                let mut temp = vec![bf16::ZERO; gpu_elems];

                self.k_cache[layer].copy_region_to_host(
                    ctx, offloaded_elems, gpu_elems, &mut temp,
                )?;
                self.k_cache[layer].copy_region_from_host(ctx, 0, &temp)?;

                self.v_cache[layer].copy_region_to_host(
                    ctx, offloaded_elems, gpu_elems, &mut temp,
                )?;
                self.v_cache[layer].copy_region_from_host(ctx, 0, &temp)?;
            }
        }

        ctx.sync()?;
        self.gpu_has_full_seq = false;

        Ok(())
    }
}
