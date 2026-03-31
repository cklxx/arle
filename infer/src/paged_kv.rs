//! Paged KV Cache — FlashInfer-compatible block-based KV storage.
//!
//! Instead of pre-allocating max_seq_len per slot, we maintain a pool of fixed-size
//! pages and allocate them on demand. This allows 32+ concurrent requests on a single GPU.
//!
//! Memory layout (HND): `[max_pages, num_kv_heads, page_size, head_dim]`
//! This matches FlashInfer's expected layout with kv_layout_code=1 (HND).

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use log::info;

use crate::tensor::DeviceContext;

/// Default page size in tokens. Each page stores this many tokens of K and V.
pub const DEFAULT_PAGE_SIZE: usize = 16;

/// Paged KV cache pool — shared across all request slots.
pub struct PagedKVPool {
    /// K pages for all layers: `[num_layers]` each `[max_pages * num_kv_heads * page_size * head_dim]` bf16
    k_pools: Vec<CudaSlice<u16>>,
    /// V pages for all layers
    v_pools: Vec<CudaSlice<u16>>,

    /// Free page indices (stack-based allocator)
    free_pages: Vec<u32>,

    /// Per-slot page tables: `slot -> Vec<page_id>`
    /// page_tables[slot][i] = physical page index for logical page i
    page_tables: Vec<Vec<u32>>,
    /// Per-slot sequence lengths
    seq_lens: Vec<usize>,

    // Config
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub page_size: usize,
    pub head_dim: usize,
    pub max_pages: usize,
    pub num_slots: usize,

    /// Stride for one page: num_kv_heads * page_size * head_dim
    pub stride_page: usize,
}

impl PagedKVPool {
    /// Create a new paged KV pool.
    ///
    /// `budget_bytes` controls how much GPU memory to use for the KV cache pool.
    /// Pages are allocated from this budget.
    pub fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_slots: usize,
        budget_bytes: usize,
    ) -> Result<Self> {
        let stride_page = num_kv_heads * page_size * head_dim;
        let bytes_per_page_per_layer = stride_page * 2; // bf16 = 2 bytes
        let bytes_per_page = bytes_per_page_per_layer * num_layers * 2; // K + V
        let max_pages = if bytes_per_page > 0 {
            budget_bytes / bytes_per_page
        } else {
            0
        };
        let max_pages = max_pages.max(num_slots); // at least 1 page per slot (may be 0-sized)

        info!(
            "PagedKVPool: {} pages × {} tokens/page = {} max tokens, \
             {:.1} GB for {} layers ({} kv_heads × {} head_dim)",
            max_pages,
            page_size,
            max_pages * page_size,
            (max_pages * bytes_per_page) as f64 / 1e9,
            num_layers,
            num_kv_heads,
            head_dim,
        );

        // Allocate K and V pools per layer (skip if budget is 0 — stub mode)
        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        let pool_elements = max_pages * stride_page;

        if pool_elements > 0 {
            for _ in 0..num_layers {
                let k: CudaSlice<u16> = ctx
                    .stream
                    .alloc_zeros(pool_elements)
                    .map_err(|e| anyhow!("KV pool alloc failed: {e}"))?;
                let v: CudaSlice<u16> = ctx
                    .stream
                    .alloc_zeros(pool_elements)
                    .map_err(|e| anyhow!("KV pool alloc failed: {e}"))?;
                k_pools.push(k);
                v_pools.push(v);
            }
        }

        // Initialize free list (all pages free)
        let free_pages: Vec<u32> = (0..max_pages as u32).rev().collect();

        // Initialize per-slot state
        let page_tables = vec![Vec::new(); num_slots];
        let seq_lens = vec![0; num_slots];

        Ok(Self {
            k_pools,
            v_pools,
            free_pages,
            page_tables,
            seq_lens,
            num_layers,
            num_kv_heads,
            page_size,
            head_dim,
            max_pages,
            num_slots,
            stride_page,
        })
    }

    /// Allocate pages for a slot to hold `new_tokens` additional tokens.
    /// Returns the number of new pages allocated.
    pub fn grow_slot(&mut self, slot: usize, new_tokens: usize) -> Result<usize> {
        let current_len = self.seq_lens[slot];
        let new_len = current_len + new_tokens;
        let pages_needed = (new_len + self.page_size - 1) / self.page_size;
        let pages_have = self.page_tables[slot].len();

        let mut allocated = 0;
        for _ in pages_have..pages_needed {
            let page_id = self
                .free_pages
                .pop()
                .ok_or_else(|| anyhow!("PagedKVPool: out of pages"))?;
            self.page_tables[slot].push(page_id);
            allocated += 1;
        }

        self.seq_lens[slot] = new_len;
        Ok(allocated)
    }

    /// Free all pages for a slot, returning them to the pool.
    pub fn free_slot(&mut self, slot: usize) {
        for &page_id in &self.page_tables[slot] {
            self.free_pages.push(page_id);
        }
        self.page_tables[slot].clear();
        self.seq_lens[slot] = 0;
    }

    /// Reset a slot to prepare for a new request (keeps no pages).
    pub fn reset_slot(&mut self, slot: usize) {
        self.free_slot(slot);
    }

    /// Get the sequence length for a slot.
    pub fn seq_len(&self, slot: usize) -> usize {
        self.seq_lens[slot]
    }

    /// Set sequence length (used when restoring from prefix cache).
    pub fn set_seq_len(&mut self, slot: usize, len: usize) {
        self.seq_lens[slot] = len;
    }

    /// Increment sequence length by 1 (after decode step).
    pub fn increment_seq_len(&mut self, slot: usize) {
        self.seq_lens[slot] += 1;
    }

    /// Number of free pages remaining.
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Get the page table for a slot (physical page indices).
    pub fn page_table(&self, slot: usize) -> &[u32] {
        &self.page_tables[slot]
    }

    /// Get the last page length for a slot (tokens in the last page).
    pub fn last_page_len(&self, slot: usize) -> usize {
        let len = self.seq_lens[slot];
        if len == 0 {
            return 0;
        }
        let remainder = len % self.page_size;
        if remainder == 0 {
            self.page_size
        } else {
            remainder
        }
    }

    /// Get raw K pool pointer for a layer (for FFI).
    pub fn k_pool_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.k_pools[layer].device_ptr(stream);
        ptr as u64
    }

    /// Get raw V pool pointer for a layer (for FFI).
    pub fn v_pool_ptr(&self, layer: usize, stream: &cudarc::driver::CudaStream) -> u64 {
        let (ptr, _guard) = self.v_pools[layer].device_ptr(stream);
        ptr as u64
    }

    /// Build FlashInfer-compatible indptr array for a batch of slots.
    /// Returns `[batch_size+1]` array where `indptr[i+1] - indptr[i]` = num pages for slot i.
    pub fn build_indptr(&self, slots: &[usize]) -> Vec<i32> {
        let mut indptr = Vec::with_capacity(slots.len() + 1);
        indptr.push(0);
        for &slot in slots {
            let last = *indptr.last().unwrap();
            indptr.push(last + self.page_tables[slot].len() as i32);
        }
        indptr
    }

    /// Build FlashInfer-compatible page indices array for a batch of slots.
    /// Concatenates all page tables.
    pub fn build_indices(&self, slots: &[usize]) -> Vec<i32> {
        let mut indices = Vec::new();
        for &slot in slots {
            for &page_id in &self.page_tables[slot] {
                indices.push(page_id as i32);
            }
        }
        indices
    }

    /// Build FlashInfer-compatible last_page_len array for a batch of slots.
    pub fn build_last_page_lens(&self, slots: &[usize]) -> Vec<i32> {
        slots
            .iter()
            .map(|&slot| self.last_page_len(slot) as i32)
            .collect()
    }
}
