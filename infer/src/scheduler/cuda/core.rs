use std::collections::HashMap;

use super::{
    ActiveRequest, Arc, AtomicUsize, GenerationState, IncomingRequest, ModelForward, PagedKVPool,
    Phase, RequestPriority, Result, SchedulerConfig, SchedulerHandle, SeedableRng, StdRng,
    Tokenizer, VecDeque, error, info, mpsc, warn,
};
use crate::kv_tier::BlockLocation;
use crate::kv_tier::transport::DiskStore;
use crate::prefix_cache::{BlockId, BlockMetadataUpdate, RadixCache};
use crate::scheduler::policy::{
    ChunkingPolicy, DecodeAwareChunking, SchedulerSignals, SessionBiasedLru,
};
use crate::types::{BlockFingerprint, InferenceMode, KvContentContext};

/// Block size (in tokens) for the global `RadixCache` prefix observer.
/// Chosen to match the M0.3 target paged-pool page size so that when
/// M2 dual residency wires the radix directly onto the pool, the block
/// boundaries already agree.
pub(super) const PREFIX_CACHE_BLOCK_SIZE: usize = 16;

/// Contiguous KV working buffer per slot (tokens). Only prefill uses it;
/// decode writes directly to the paged pool via `decode_prep_paged`.
/// Prefill chunk size is capped to this value to prevent buffer overflow.
pub(super) const CONTIGUOUS_KV_TOKENS: usize = 512;

// Prefix-cache watermark / keepalive tunables moved to
// `crate::scheduler::types::SchedulerConfig` in Tier C. See the doc
// comments on `prefix_cache_high_water` / `prefix_cache_low_water` /
// `prefix_cache_retain_hard_cap` / `prefix_cache_keepalive_ticks` /
// `stage_wait_keepalive_ticks` there for the defaults and validation
// semantics. Per the project env-var policy (`docs/environment.md` §0),
// these are **not** env-driven — callers assign directly to
// `SchedulerConfig` fields. Env vars are reserved for debug-only knobs.

fn prefix_cache_retain_hard_cap_pages(total_pages: usize, cap_fraction: f64) -> usize {
    (total_pages as f64 * cap_fraction) as usize
}

fn can_publish_prefix_pages(
    retained_pages: usize,
    total_pages: usize,
    new_pages: usize,
    cap_fraction: f64,
) -> bool {
    retained_pages.saturating_add(new_pages)
        <= prefix_cache_retain_hard_cap_pages(total_pages, cap_fraction)
}

pub(super) struct StagedAdmission {
    pub(super) request: IncomingRequest,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) block_ids: Vec<BlockId>,
    pub(super) enqueued_at_clock: u64,
}

/// One prefill request fused into a mixed decode+prefill tick, preserved
/// between the launch and readback halves of `step_decode_*`.
///
/// `req_idx` indexes `Scheduler.active` at the time the mixed step was
/// launched; since `step_decode_readback` runs before any new admission,
/// these indices stay valid.
///
/// `logit_row` is the row index into the mixed batch's logits at which
/// this section's last-token logits land: for section `i`,
/// `logit_row = B + Σ_{j≤i} c_j - 1`. Kept as a documentation hook and
/// for a future batched-logit-scatter path; today the model extracts
/// each row straight into `state.base.prefill_logits` so the readback
/// loop samples via `select_token` without referencing `logit_row`.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingPrefillChunk {
    pub req_idx: usize,
    pub completes: bool,
    #[allow(dead_code)]
    pub logit_row: usize,
}

/// State preserved between decode launch and readback for GPU/CPU overlap.
pub(super) struct PendingDecode {
    pub decode_indices: Vec<usize>,
    pub slot_indices: Vec<usize>,
    /// True only when `sample_batch_greedy_launch` actually fired the argmax kernel.
    pub greedy_launched: bool,
    pub sampling_params_greedy: Vec<bool>,
    /// Prefill chunks fused into the current tick. Empty when no mixed
    /// prefill ran (plain decode path).
    pub mixed_prefill_chunks: Vec<PendingPrefillChunk>,
}

/// CUDA-backed scheduler state and initialization.
pub struct Scheduler<M: ModelForward> {
    pub(super) config: SchedulerConfig,
    pub(super) metrics: crate::metrics::ServerMetrics,
    pub(super) model: M,
    pub(super) tokenizer: Tokenizer,
    /// Stable within one engine instance; real weight checksum upgrade is M5-era work.
    pub(super) model_fingerprint: Vec<u8>,
    /// Per-slot states (KV caches, decode buffers). Stored separately from
    /// slot metadata so we can pass `&mut [M::State]` to batched decode.
    pub(super) states: Vec<M::State>,
    /// Number of prompt tokens still materialized in each slot's contiguous
    /// state. This is the scheduler's only slot-local prefix-reuse metadata
    /// after M2b removes `cached_prompts: Vec<Vec<u32>>`.
    ///
    /// A non-zero value means: if the slot is free and the global radix says
    /// an incoming request matches a prefix owned by this slot, `step_new()`
    /// may reuse the first `matched_len` tokens already present in the slot's
    /// contiguous state instead of restarting cold.
    pub(super) slot_materialized_prompt_lens: Vec<usize>,
    /// Global cross-slot prefix observer and (as of M2a) the authority
    /// on which pool pages must survive a slot's `free_slot` call.
    /// Owned by the single-writer scheduler thread, no lock needed.
    ///
    /// The `BlockId`s stored under each node are **real physical pool
    /// page indices** pulled from `paged_kv_pool.token_indices(slot_idx)`
    /// at publish time. The pool's `page_ref_count` tracks how many radix
    /// nodes reference each page, and `free_slot` leaves pinned pages in
    /// limbo instead of returning them to the free list. CUDA admission now
    /// consults this radix-owned state to pick a reusable free slot, but
    /// reuse is still limited to slots whose contiguous state remains
    /// materialized; cross-slot page aliasing is intentionally unsupported.
    pub(super) prefix_cache: RadixCache,
    pub(super) disk_store: Arc<DiskStore>,
    /// Side map from `BlockId` → full contiguous page span for that
    /// block. The radix stores just the first page of each block
    /// (block id = `slot_pages[i * block_size]`), but the actual
    /// `block_size` pages belonging to that block can be arbitrary
    /// pool indices because the LIFO `free_slots` allocator produces
    /// non-contiguous ranges after a few alloc/free cycles. This map
    /// keeps the full span so eviction can release the right pages.
    ///
    /// Invariant: every `BlockId` inserted into `prefix_cache` has
    /// an entry here with exactly `prefix_cache.block_size()` pages,
    /// and every page in the value appears in `page_ref_count > 0`.
    /// Entries are removed when eviction releases the block.
    pub(super) block_to_pages: HashMap<BlockId, Vec<u32>>,
    /// Best-effort mapping from a radix block to the free slot whose
    /// contiguous state still materializes that prefix. This is intentionally
    /// separate from `prefix_cache`: the radix owns reusable bytes / page pins,
    /// while this map only tracks which free slot can safely reuse those bytes
    /// without cross-slot page aliasing.
    pub(super) block_owner_slots: HashMap<BlockId, usize>,
    /// Reverse index for `block_owner_slots`, keyed by slot.
    pub(super) slot_owned_blocks: Vec<Vec<BlockId>>,
    pub(super) coordinator_handle: crate::kv_tier::CoordinatorHandle,
    pub(super) coordinator_events: crossbeam_channel::Receiver<crate::kv_tier::CoordinatorEvent>,
    pub(super) coordinator_thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
    pub(super) stage_waiting: HashMap<crate::kv_tier::StageTicket, StagedAdmission>,
    pub(super) coordinator_unavailable: bool,
    pub(super) page_lifecycle: crate::kv_tier::coordinator::PageLifecycle,
    pub(super) request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    /// Shared waiting count with the handle (for backpressure decrement).
    pub(super) waiting_count: Arc<AtomicUsize>,
    pub(super) waiting: VecDeque<IncomingRequest>,
    pub(super) active: Vec<ActiveRequest>,
    pub(super) next_id: u64,
    pub(super) rng: StdRng,
    /// Paged KV cache pool shared across all slots (for batched decode).
    pub(super) paged_kv_pool: PagedKVPool,
    /// Pre-allocated buffers for batched decode (reused across steps).
    /// Typed via `M::DecodeContext` — no downcasting needed.
    pub(super) decode_bufs: Option<M::DecodeContext>,
    /// Round-robin index for fair decode scheduling.
    pub(super) last_served: usize,
    /// Lifetime stats.
    pub(super) total_completed: u64,
    pub(super) total_generated_tokens: u64,
    /// EMA step timing (microseconds) for /v1/stats profiling.
    pub(super) step_timing_decode_us: f64,
    pub(super) step_timing_emit_us: f64,
    pub(super) step_timing_prefill_us: f64,
    pub(super) step_timing_total_us: f64,
    /// Throttled GPU memory query — last poll time and peak high-water mark.
    pub(super) last_mem_query: std::time::Instant,
    pub(super) peak_mem_bytes: u64,
    /// Pending decode state for GPU/CPU overlap.
    pub(super) pending_decode: Option<PendingDecode>,
    /// Prefill requests consumed by the mixed decode launch in the current
    /// step. Populated by `step_decode_launch_mixed`, read by the regular
    /// prefill section in `execution.rs` so we skip reqs already fused.
    pub(super) pending_mixed_prefill_idxs: Vec<usize>,
    /// Round-robin cursor into `active` for fair mixed-prefill selection.
    /// Advances by the number of prefills actually fused each tick.
    pub(super) last_mixed_prefill_cursor: usize,
    /// Requests retracted by `retract_longest_decode` to free pool pages for
    /// another slot's admission. Populated when we mark a victim
    /// `Phase::Finished`, consumed by `cleanup()` so the re-queued
    /// `IncomingRequest` in `waiting` isn't counted as a completion and its
    /// stale cached-prompt isn't republished to the prefix cache.
    pub(super) retracted_request_ids: std::collections::HashSet<u64>,
}

impl<M: ModelForward> Scheduler<M> {
    fn eviction_signals(&self) -> SchedulerSignals {
        SchedulerSignals::queue_state(
            self.waiting.len(),
            self.active
                .iter()
                .filter(|req| matches!(req.phase, Phase::Decoding))
                .count(),
        )
    }

    pub(super) fn admission_budget_tokens(&self) -> usize {
        let available = self.paged_kv_pool.free_count();
        let evictable = self.prefix_cache.evictable_token_count();
        let clip = self.config.admission_clip_max_new_tokens;
        let ratio = self.config.admission_new_token_ratio;
        // Decode-only running offset (sglang PrefillAdder
        // `_get_running_request_total_token_offset` parity —
        // `schedule_policy.py:447-454`). The prior commit (ff8228e)
        // also charged `prefill_remaining` for Phase::New/Prefilling,
        // which double-counted for one tick (the incoming-admit cost
        // already burned `extend_input + clipped_max_new + page_overhead`
        // from `tick_budget_tokens` in the same `assign_slots` loop)
        // and dropped c=16 throughput from 114 → 74 out tok/s. The
        // pool's own `free_count()` shrinks monotonically as chunks
        // alloc, so the next tick's budget naturally reflects pages
        // already claimed. Mid-tick reservation gaps are absorbed by
        // the `retract_longest_decode` backstop in
        // `alloc_pool_tokens_with_retry`.
        let running_offset: f64 = self
            .active
            .iter()
            .map(|r| {
                let decode_remaining = r.max_tokens.saturating_sub(r.generated_tokens.len());
                let clipped_decode = decode_remaining.min(clip);
                (clipped_decode as f64) * ratio
            })
            .sum();
        (available + evictable).saturating_sub(running_offset.ceil() as usize)
    }

    fn prefix_cache_watermarks_pages(&self) -> (usize, usize) {
        let total = self.paged_kv_pool.max_total_pages;
        let high = (total as f64 * self.config.prefix_cache_high_water) as usize;
        let low = (total as f64 * self.config.prefix_cache_low_water) as usize;
        (high, low)
    }

    pub fn session_fingerprints(&self, session_id: &str) -> Vec<BlockFingerprint> {
        self.prefix_cache.fingerprints_for_session(session_id)
    }

    pub fn read_block_payload(&self, fingerprint: BlockFingerprint) -> Option<Vec<u8>> {
        let block_id = self.prefix_cache.block_id_for_fingerprint(fingerprint)?;
        let pages = self.block_to_pages.get(&block_id)?;
        let stream = &self.model.device_context().stream;
        // Don't silently drop copy failures: the caller
        // (`save_session`) serialises the radix separately and would
        // produce a half-consistent manifest if we returned `None` for
        // a block that actually exists. Today this triggers for any
        // non-BF16 pool (FP8 / INT8 / TurboQuant) because Gap #5 C1's
        // `copy_pages_to_host` errors on those formats. Warn loud so
        // a 200-OK `/v1/sessions/{id}/save` with dropped blocks is
        // observable instead of silent.
        match self.paged_kv_pool.copy_pages_to_host(pages, stream) {
            Ok(bytes) => Some(bytes),
            Err(err) => {
                warn!(
                    "read_block_payload: fingerprint={fingerprint:?} block_id={block_id:?} \
                     copy_pages_to_host failed: {err}"
                );
                None
            }
        }
    }

    pub fn install_restored_kv(
        &mut self,
        payloads: &HashMap<BlockFingerprint, Vec<u8>>,
    ) -> Box<dyn FnMut(BlockFingerprint) -> Option<BlockId> + Send> {
        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size)
            .max(1);
        let mut prepared = HashMap::with_capacity(payloads.len());

        let stream = self.model.device_context().stream.clone();
        for (&fingerprint, payload) in payloads {
            let Ok(pages) = self.paged_kv_pool.alloc_detached_pages(pages_per_block) else {
                break;
            };
            if let Err(err) = self
                .paged_kv_pool
                .copy_pages_from_host(&pages, payload, &stream)
            {
                // Copy failed (payload/geometry mismatch, unsupported
                // format, PCIe error). The detached pages are still on
                // our books — release them back to the free list before
                // moving on, otherwise one bad block permanently shrinks
                // the pool and the caller sees a misleading
                // `PoolExhausted` on a later valid restore.
                warn!(
                    "install_restored_kv: fingerprint={fingerprint:?} \
                     copy_pages_from_host failed: {err}; returning {} detached pages to free list",
                    pages.len(),
                );
                // Detached pages have refcount 0 by construction — use
                // `free_detached_pages`, not `release_pages`, which would
                // panic the `debug_assert!(cur > 0)` invariant.
                self.paged_kv_pool.free_detached_pages(&pages);
                continue;
            }

            let block_id = BlockId(
                *pages
                    .first()
                    .expect("detached restored block must allocate at least one page"),
            );
            self.block_to_pages.insert(block_id, pages);
            prepared.insert(fingerprint, block_id);
        }

        Box::new(move |fingerprint| prepared.remove(&fingerprint))
    }

    pub fn kv_format_tag(&self) -> u8 {
        self.paged_kv_pool.format.stable_tag().unwrap_or(0)
    }

    pub fn session_disk_store(&self) -> &DiskStore {
        self.disk_store.as_ref()
    }

    pub fn session_radix_cache(&self) -> &RadixCache {
        &self.prefix_cache
    }

    /// Create a new scheduler and its handle.
    ///
    /// `num_slots` controls how many concurrent requests can be active (each gets
    /// its own KV cache). More slots = more GPU memory usage.
    pub fn new(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
            metrics,
            SchedulerConfig::runtime_defaults(num_slots),
            None,
            crate::model::kv_cache::KVCacheDtype::BF16,
            crate::model::kv_cache::KVFormat::BF16,
        )
    }

    /// Create a new scheduler with an explicit max sequence length override.
    pub fn with_max_seq_len(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
        max_seq_len_override: Option<usize>,
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_config(
            model,
            tokenizer,
            model_id,
            seed,
            metrics,
            SchedulerConfig::runtime_defaults(num_slots),
            max_seq_len_override,
            crate::model::kv_cache::KVCacheDtype::BF16,
            crate::model::kv_cache::KVFormat::BF16,
        )
    }

    /// Create a scheduler from an explicit runtime configuration.
    pub fn with_config(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        seed: u64,
        metrics: crate::metrics::ServerMetrics,
        config: SchedulerConfig,
        max_seq_len_override: Option<usize>,
        kv_cache_dtype: crate::model::kv_cache::KVCacheDtype,
        kv_pool_format: crate::model::kv_cache::KVFormat,
    ) -> Result<(Self, SchedulerHandle)> {
        config.validate()?;

        let (tx, rx) = mpsc::unbounded_channel();
        let effective_max_seq_len =
            Self::compute_max_seq_len(&model, &config, max_seq_len_override);

        // When the model writes prefill K/V directly to the paged pool, the
        // per-slot contiguous scratch buffer is unused by prefill. Shrink it
        // to the minimum that single-token decode / INT8 working buffers
        // still require, and reclaim the freed bytes into the pool budget.
        let model_uses_paged_prefill = model.prefill_uses_paged_pool();
        let contiguous_tokens = if model_uses_paged_prefill {
            // Single-token decode path still allocates per-slot contiguous
            // K/V of this size; 1 page's worth is enough.
            PREFIX_CACHE_BLOCK_SIZE
        } else {
            CONTIGUOUS_KV_TOKENS
        };

        let mut states = Vec::with_capacity(config.max_slots);
        let mut slot_materialized_prompt_lens = Vec::with_capacity(config.max_slots);
        let mut slot_owned_blocks = Vec::with_capacity(config.max_slots);
        for i in 0..config.max_slots {
            let mut state = model.create_state()?;
            state.set_max_seq_len(contiguous_tokens);
            state.set_kv_dtype(kv_cache_dtype);
            states.push(state);
            slot_materialized_prompt_lens.push(0);
            slot_owned_blocks.push(Vec::new());
            info!("Initialized state slot {}/{}", i + 1, config.max_slots);
        }

        let paged_kv_pool = {
            let bytes_per_token = model.kv_cache_bytes_per_token();
            let contiguous_cost = config.max_slots * contiguous_tokens * bytes_per_token;
            // SGLang-compatible: pool budget = free − contiguous − headroom.
            // Headroom is derived from mem_fraction_static via the auto-sizer;
            // here we just subtract contiguous cost from post-weight-load free memory.
            let budget_bytes = match crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
            {
                Ok((free, total)) => {
                    let headroom = ((total as f64) * (1.0 - config.mem_fraction_static)) as usize;
                    free.saturating_sub(contiguous_cost.saturating_add(headroom))
                }
                Err(_) => config.kv_pool_fallback_bytes,
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous={:.1} GB, fraction={:.0}%)",
                budget_bytes as f64 / 1e9,
                contiguous_cost as f64 / 1e9,
                config.mem_fraction_static * 100.0,
            );

            let ctx = model.device_context();
            PagedKVPool::with_format(
                ctx,
                model.num_kv_layers(),
                model.num_kv_heads(),
                model.head_dim(),
                config.max_slots,
                budget_bytes,
                kv_pool_format,
            )?
        };

        info!(
            "Scheduler ready: model={}, slots={}, seed={}, max_seq_len={}, max_waiting={}, prefill_chunk_size={}",
            model_id,
            config.max_slots,
            seed,
            effective_max_seq_len.map_or_else(|| "32768 (default)".to_string(), |n| n.to_string()),
            config.max_waiting_requests,
            config.prefill_chunk_size,
        );

        let waiting_count = Arc::new(AtomicUsize::new(0));
        let page_lifecycle =
            crate::kv_tier::coordinator::PageLifecycle::new(paged_kv_pool.max_total_pages);
        let (coordinator, coordinator_handle, coordinator_events) =
            crate::kv_tier::Coordinator::new(config.max_slots.max(16));
        let coordinator_thread = Some(coordinator.spawn("infer-tiered-kv-coord"));
        let scheduler = Self {
            config: config.clone(),
            metrics,
            model,
            tokenizer,
            model_fingerprint: blake3::hash(model_id.as_bytes()).as_bytes().to_vec(),
            states,
            slot_materialized_prompt_lens,
            prefix_cache: RadixCache::with_soft_pin_keepalive(
                PREFIX_CACHE_BLOCK_SIZE,
                config.prefix_cache_keepalive_ticks,
            ),
            disk_store: Arc::new(DiskStore::new(config.disk_store_root.clone())),
            block_to_pages: HashMap::new(),
            block_owner_slots: HashMap::new(),
            slot_owned_blocks,
            coordinator_handle,
            coordinator_events,
            coordinator_thread,
            stage_waiting: HashMap::new(),
            coordinator_unavailable: false,
            page_lifecycle,
            request_rx: rx,
            waiting_count: Arc::clone(&waiting_count),
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            paged_kv_pool,
            decode_bufs: None,
            last_served: 0,
            total_completed: 0,
            total_generated_tokens: 0,
            step_timing_decode_us: 0.0,
            step_timing_emit_us: 0.0,
            step_timing_prefill_us: 0.0,
            step_timing_total_us: 0.0,
            last_mem_query: std::time::Instant::now(),
            peak_mem_bytes: 0,
            pending_decode: None,
            pending_mixed_prefill_idxs: Vec::new(),
            last_mixed_prefill_cursor: 0,
            retracted_request_ids: std::collections::HashSet::new(),
        };

        let handle = SchedulerHandle::with_shared_waiting_count(
            tx,
            model_id,
            config.max_waiting_requests,
            Arc::clone(&waiting_count),
        );
        debug_assert_eq!(handle.waiting_count(), 0);

        Ok((scheduler, handle))
    }

    /// Compute the effective max_seq_len per slot based on available GPU memory.
    fn compute_max_seq_len(
        model: &M,
        config: &SchedulerConfig,
        override_val: Option<usize>,
    ) -> Option<usize> {
        use crate::backend::cuda::tensor::DeviceContext;

        const DEFAULT_MAX_SEQ: usize = 4096;

        if let Some(val) = override_val {
            info!("KV cache: using explicit --max-seq-len={}", val);
            return Some(val);
        }

        let (free_bytes, total_bytes) = match DeviceContext::gpu_memory_info() {
            Ok(info) => info,
            Err(e) => {
                info!(
                    "KV cache: could not query GPU memory ({}), using default max_seq_len={}",
                    e, DEFAULT_MAX_SEQ
                );
                return None;
            }
        };

        let headroom = ((total_bytes as f64) * (1.0 - config.mem_fraction_static)) as usize;
        let min_seq = config.min_seq_len;
        let available = free_bytes.saturating_sub(headroom);
        let bytes_per_token = model.kv_cache_bytes_per_token();
        let total_kv_budget = available;
        let per_slot_budget = total_kv_budget / config.max_slots.max(1);
        let affordable_seq_len = per_slot_budget / bytes_per_token.max(1);
        let effective = affordable_seq_len.clamp(min_seq, DEFAULT_MAX_SEQ);

        info!(
            "KV cache auto-sizing: gpu_free={:.1} GB, gpu_total={:.1} GB, \
             headroom={:.1} GB, bytes_per_token={}, num_slots={}, \
             affordable_seq_len={}, effective_max_seq_len={}",
            free_bytes as f64 / 1e9,
            total_bytes as f64 / 1e9,
            headroom as f64 / 1e9,
            bytes_per_token,
            config.max_slots,
            affordable_seq_len,
            effective,
        );

        if affordable_seq_len < min_seq {
            error!(
                "KV cache: only {} tokens affordable per slot (need at least {}). \
                 Reduce --num-slots or free GPU memory.",
                affordable_seq_len, min_seq,
            );
        }

        Some(effective)
    }

    /// Fold a completed request's prompt into the global
    /// [`RadixCache`] prefix observer and pin the corresponding pool
    /// pages so they survive the subsequent `free_slot` call.
    ///
    /// The [`BlockId`]s stored in the radix are **real physical pool page
    /// indices** pulled from `paged_kv_pool.token_indices(slot_idx)`. For a prompt of
    /// `L` tokens and the radix's `block_size = B`, the first
    /// `num_blocks = L / B` full blocks are inserted under the page
    /// ids covering positions `[0, num_blocks * B)` — i.e. exactly the
    /// contiguous pool pages that hold those tokens' KV state. The
    /// trailing `L % B` tokens are dropped per `RadixCache::insert`
    /// semantics (partial blocks are never cached).
    ///
    /// After inserting, `paged_kv_pool.retain_pages` bumps the
    /// refcount on each page. Because the scheduler's `cleanup()`
    /// calls this method **before** `free_slot`, the pool's
    /// refcount-aware `free_slot` will leave these pages in limbo
    /// (out of any slot, out of the free list, still physically
    /// alive in HBM) instead of recycling them. This is the T0-only
    /// dual-residency data model that the current safe same-slot
    /// resurrection path consumes.
    ///
    /// Caller contract: `slot_idx` must currently own the pages in
    /// `paged_kv_pool.token_indices(slot_idx)` and the slot must not
    /// have been `free_slot`ed yet. `prompt_tokens.len()` must equal
    /// the number of tokens currently allocated to the slot (i.e.
    /// `paged_kv_pool.seq_len(slot_idx)`). Both are true at the
    /// `cleanup()` call site where this is invoked.
    pub(super) fn publish_to_prefix_cache(
        &mut self,
        slot_idx: usize,
        prompt_tokens: &[u32],
        session_id: Option<crate::types::SessionId>,
    ) {
        let block_size = self.prefix_cache.block_size();
        // The slot's `seq_len` is the actual ground truth for how many tokens
        // are currently allocated in the paged pool. `prompt_tokens.len()`
        // is the snapshot the caller saved at request submission time and
        // can be LARGER than the slot's current footprint after a
        // recompute-style preemption: in that case the slot was rolled back
        // to a shorter prefix (or zero) but cleanup() still calls us with
        // the original prompt. Capping `num_blocks` to the slot's actual
        // page coverage prevents the
        // `paged_kv.rs:595 page_indices_for_token_range` index-OOB panic
        // that fires when we ask for pages past the slot's allocation.
        // See docs/experience/wins/2026-04-15-bench-longseq-int8-splits32.md
        // and 2026-04-15-bench-hbm-peak-throughput.md for the trigger
        // sequences.
        let slot_tokens_now = self.paged_kv_pool.seq_len(slot_idx);
        let publishable_tokens = prompt_tokens.len().min(slot_tokens_now);
        let num_blocks = publishable_tokens / block_size;
        if num_blocks == 0 {
            return;
        }
        let required_tokens = num_blocks * block_size;
        let block_pages: Vec<Vec<u32>> = (0..num_blocks)
            .map(|block_i| {
                self.paged_kv_pool
                    .page_indices_for_token_range(slot_idx, block_i * block_size, block_size)
                    .to_vec()
            })
            .collect();
        let required_pages = block_pages.iter().map(std::vec::Vec::len).sum::<usize>();
        let retained_pages = self.paged_kv_pool.retained_count();
        let total_pages = self.paged_kv_pool.max_total_pages;
        let retain_cap_fraction = self.config.prefix_cache_retain_hard_cap;
        if !can_publish_prefix_pages(
            retained_pages,
            total_pages,
            required_pages,
            retain_cap_fraction,
        ) {
            info!(
                "prefix cache publish skipped for slot {}: retain hard cap hit \
                 (retained={}, new_pages={}, cap={}, total={})",
                slot_idx,
                retained_pages,
                required_pages,
                prefix_cache_retain_hard_cap_pages(total_pages, retain_cap_fraction),
                total_pages,
            );
            return;
        }

        let blocks: Vec<BlockId> = block_pages
            .iter()
            .map(|pages| {
                BlockId(
                    *pages
                        .first()
                        .expect("full radix block must map to at least one physical page"),
                )
            })
            .collect();
        // M4 review A4: `stable_tag()` is now `Option<u8>`. If the
        // live pool format has no assigned tag (a future
        // TurboQuant bit-pair combination that shipped to the pool
        // but not to the disk format), publish silently with no
        // fingerprints — persistence is not available for that
        // format yet. Warn once per publish so operators notice.
        let kv_format_tag = if let Some(tag) = self.paged_kv_pool.format.stable_tag() {
            tag
        } else {
            warn!(
                "prefix_cache publish: live KV format has no stable_tag assignment; \
                 fingerprints skipped for slot {} (format = {:?})",
                slot_idx, self.paged_kv_pool.format,
            );
            // Zero = "unset"; persistence code refuses format 0
            // at load time, so this can never drive a cross-format
            // reload. Still stamp fingerprints because Tier C's
            // O(1) block_index and M4c's reconcile both want a
            // non-zero fingerprint on each published node.
            0
        };
        let mut parent_fingerprint: Option<BlockFingerprint> = None;
        let mut block_fingerprints: Vec<BlockFingerprint> = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let fp = BlockFingerprint::compute(
                KvContentContext {
                    model_fingerprint: &self.model_fingerprint,
                    kv_format_tag,
                    parent: parent_fingerprint,
                },
                &prompt_tokens[i * block_size..(i + 1) * block_size],
            );
            block_fingerprints.push(fp);
            parent_fingerprint = Some(fp);
        }

        // Slice prompt_tokens to the actually-publishable prefix so the
        // radix insert walks only the blocks we have fingerprints for.
        // Without this, when `slot_tokens_now < prompt_tokens.len()` the
        // insert path would try to traverse blocks past `num_blocks` and
        // mismatch its own internal block count.
        let publishable_prompt = &prompt_tokens[..required_tokens];
        let inserted = self.prefix_cache.insert_with_fingerprints(
            publishable_prompt,
            &blocks,
            &block_fingerprints,
        );
        if inserted != required_tokens {
            warn!(
                "prefix_cache.insert: expected {} tokens, got {} (slot={}, num_blocks={}, prompt={})",
                required_tokens,
                inserted,
                slot_idx,
                num_blocks,
                prompt_tokens.len(),
            );
            return;
        }

        // Pin every physical page that backs the inserted blocks.
        // The radix refs a "block" as a unit, and the *entire*
        // `block_size`-wide span must survive `free_slot` so the
        // reuse path can read the full KV state back out.
        let slot_pages: Vec<u32> = block_pages
            .iter()
            .flat_map(|pages| pages.iter().copied())
            .collect();
        self.paged_kv_pool.retain_pages(&slot_pages);
        self.slot_owned_blocks[slot_idx].clear();
        let block_byte_len = self
            .model
            .kv_cache_bytes_per_token()
            .saturating_mul(block_size)
            .min(u32::MAX as usize) as u32;
        let keepalive_deadline = session_id.as_ref().map(|_| {
            self.prefix_cache
                .logical_clock()
                .saturating_add(self.config.prefix_cache_keepalive_ticks)
        });
        for (block_i, &bid) in blocks.iter().enumerate() {
            let pages_for_block = block_pages[block_i].clone();
            // If a prior publish already registered this BlockId
            // (same pool page as first-of-block), the existing entry
            // is authoritative — don't clobber it. The `retain_pages`
            // call above bumped the refcount a second time, so the
            // first `release_pages` will leave the entry with
            // refcount > 0 and the eviction path will correctly
            // short-circuit on the second release.
            self.block_to_pages
                .entry(bid)
                .or_insert_with(|| pages_for_block);
            self.block_owner_slots.insert(bid, slot_idx);
            self.slot_owned_blocks[slot_idx].push(bid);
            let _ = self.prefix_cache.update_block_metadata(
                bid,
                BlockMetadataUpdate {
                    location: Some(BlockLocation::Gpu {
                        slot: slot_idx as u32,
                    }),
                    byte_len: Some(block_byte_len),
                    session_id: Some(session_id.clone()),
                    soft_pin_until: Some(keepalive_deadline),
                },
            );
            for &page in &block_pages[block_i] {
                if let Err(err) = self.page_lifecycle.mark_resident(page as usize) {
                    match err {
                        crate::kv_tier::coordinator::PageLifecycleError::InvalidTransition {
                            ..
                        } => log::debug!(
                            "page lifecycle mark_resident ignored for page {}: {}",
                            page,
                            err
                        ),
                        crate::kv_tier::coordinator::PageLifecycleError::UnknownPage { .. } => {
                            warn!(
                                "page lifecycle unknown page {} during publish: {}",
                                page, err
                            );
                        }
                    }
                }
            }
        }
    }

    /// Remove the transient "this free slot still owns a materialized prompt
    /// state" mapping for `slot_idx`.
    pub(super) fn clear_slot_prefix_ownership(&mut self, slot_idx: usize) {
        for bid in self.slot_owned_blocks[slot_idx].drain(..) {
            self.block_owner_slots.remove(&bid);
        }
    }

    /// Release radix-held pool pages back to the free list once the
    /// pool's retained fraction exceeds the high-water mark.
    ///
    /// Policy: when `retained / total > PREFIX_CACHE_HIGH_WATER`,
    /// evict LRU radix blocks until `retained / total ≤
    /// PREFIX_CACHE_LOW_WATER`. The gap between high and low marks
    /// prevents thrash where every completed request immediately
    /// evicts the block it just inserted.
    ///
    /// Each evicted `BlockId` is looked up in `block_to_pages` and
    /// the full per-block page span is released via
    /// `paged_kv_pool.release_pages`. If the refcount hits zero the
    /// pages rejoin the pool's primary free list immediately; if
    /// another radix block also references them the refcount just
    /// decrements and the pages stay in limbo.
    ///
    /// Returns the number of pool pages actually reclaimed (0 when
    /// not under pressure). Called at the end of `cleanup()` so the
    /// eviction cost is amortised over request completions, not the
    /// admission hot path.
    pub(super) fn evict_prefix_cache_if_pressured(&mut self) -> usize {
        let total = self.paged_kv_pool.max_total_pages;
        if total == 0 {
            return 0;
        }
        let retained = self.paged_kv_pool.retained_count();
        let (high, target) = self.prefix_cache_watermarks_pages();
        if retained <= high {
            return 0;
        }
        let want_free = retained.saturating_sub(target);
        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size);
        let blocks_to_evict = want_free.div_ceil(pages_per_block.max(1));
        if blocks_to_evict == 0 {
            return 0;
        }
        let evicted = self.prefix_cache.evict_with_policy(
            &SessionBiasedLru::default(),
            self.eviction_signals(),
            blocks_to_evict,
        );
        let mut reclaimed_pages: usize = 0;
        let sentinel_ticket = crate::kv_tier::StageTicket(u64::MAX);
        for bid in evicted {
            if let Some(pages) = self.block_to_pages.remove(&bid) {
                self.block_owner_slots.remove(&bid);
                for &page in &pages {
                    // A5 stub: sentinel ticket represents "instant demote"
                    // until coordinator transport lane is wired.
                    if let Err(err) = self.page_lifecycle.begin_demote(
                        page as usize,
                        sentinel_ticket,
                        BlockLocation::HostPinned { offset: 0 },
                    ) {
                        log::debug!(
                            "page lifecycle begin_demote ignored for page {}: {}",
                            page,
                            err
                        );
                    } else if let Err(err) = self
                        .page_lifecycle
                        .finish_demote(page as usize, sentinel_ticket)
                    {
                        log::debug!(
                            "page lifecycle finish_demote ignored for page {}: {}",
                            page,
                            err
                        );
                    }
                }
                let freed_now = self.paged_kv_pool.release_pages(&pages);
                reclaimed_pages += freed_now.len();
            }
        }
        if reclaimed_pages > 0 {
            info!(
                "prefix cache eviction: released {} pool pages back to free list \
                 ({} evicted blocks; retained now {})",
                reclaimed_pages,
                blocks_to_evict,
                self.paged_kv_pool.retained_count(),
            );
        }
        reclaimed_pages
    }

    /// Best-effort synchronous reclamation used on the hot path when pool
    /// allocation fails. Unlike the watermark-based cleanup path, this may run
    /// below the usual high-water mark: the immediate goal is to recover enough
    /// prefix-cache pages to satisfy one allocation.
    pub(super) fn evict_prefix_cache_for_allocation(&mut self, required_tokens: usize) -> usize {
        let shortage = required_tokens.saturating_sub(self.paged_kv_pool.free_count());
        if shortage == 0 {
            return 0;
        }

        let pages_per_block = self
            .prefix_cache
            .block_size()
            .div_ceil(self.paged_kv_pool.page_size);
        let mut reclaimed_pages = 0usize;
        let mut reclaimed_tokens = 0usize;
        let mut remaining = shortage;
        while remaining > 0 {
            let blocks_to_evict = remaining
                .div_ceil((pages_per_block * self.paged_kv_pool.page_size).max(1))
                .max(1);
            let evicted = self.prefix_cache.evict_with_policy(
                &SessionBiasedLru::default(),
                self.eviction_signals(),
                blocks_to_evict,
            );
            if evicted.is_empty() {
                break;
            }
            for bid in evicted {
                if let Some(pages) = self.block_to_pages.remove(&bid) {
                    self.block_owner_slots.remove(&bid);
                    let freed_now = self.paged_kv_pool.release_pages(&pages);
                    reclaimed_pages += freed_now.len();
                    reclaimed_tokens += freed_now.len() * self.paged_kv_pool.page_size;
                }
            }
            remaining = shortage.saturating_sub(reclaimed_tokens);
        }

        if reclaimed_pages > 0 {
            info!(
                "prefix cache emergency eviction: reclaimed {} pool pages for allocation \
                 (required={}, free_now={})",
                reclaimed_pages,
                required_tokens,
                self.paged_kv_pool.free_count(),
            );
        }
        reclaimed_pages
    }

    /// Allocate pool pages with a three-tier safety net:
    /// 1. Direct alloc.
    /// 2. On OOM: evict unlocked prefix-cache blocks, retry.
    /// 3. On still-OOM: **retract the longest-running decoder** (sglang
    ///    `ScheduleBatch::retract_decode` parity — `schedule_batch.py:1950` +
    ///    `scheduler.py:2562-2567`), free its pool pages, re-queue, retry.
    /// 4. Final Err only when no decoder can be retracted either (true OOM).
    ///
    /// Tier 3 is the difference between "returns 200-OK with empty stream"
    /// (prior behavior — `Phase::Finished` on OOM → client retry storm) and
    /// "admitted request completes after upstream transient". At c=16 ×
    /// 4096-prompt on L4 this closes the residual 19 pool-alloc failures.
    pub(super) fn alloc_pool_tokens_with_retry(
        &mut self,
        slot: usize,
        count: usize,
    ) -> Result<Vec<u32>> {
        match self.paged_kv_pool.alloc_tokens(slot, count) {
            Ok(indices) => Ok(indices),
            Err(first_err) => {
                let reclaimed_evict = self.evict_prefix_cache_for_allocation(count);
                if reclaimed_evict > 0 {
                    if let Ok(indices) = self.paged_kv_pool.alloc_tokens(slot, count) {
                        return Ok(indices);
                    }
                }
                // Tier 3: retract the longest-running decoder.
                let retract_target_tokens = count.saturating_sub(self.paged_kv_pool.free_count());
                let retracted_tokens = self.retract_longest_decode(retract_target_tokens, slot);
                if retracted_tokens == 0 && reclaimed_evict == 0 {
                    return Err(first_err);
                }
                self.paged_kv_pool
                    .alloc_tokens(slot, count)
                    .map_err(|retry_err| {
                        anyhow::anyhow!(
                            "TokenKVPool alloc retry failed after reclaiming {reclaimed_evict} \
                         prefix pages and retracting {retracted_tokens} decode tokens: \
                         first error: {first_err}; retry error: {retry_err}"
                        )
                    })
            }
        }
    }

    /// Retract decoders to free at least `required_tokens` worth of pool pages.
    /// Picks the longest-running (most generated tokens) decoders first, mirroring
    /// sglang's `retract_longest_running_req` selection. The protected slot
    /// (the caller's slot) is never retracted — that would invalidate the caller's
    /// in-flight forward.
    ///
    /// For each victim:
    /// - Clone the request's incoming fields (prompt, max_tokens, sampling,
    ///   stop, session_id, delta_tx) into a fresh `IncomingRequest`,
    ///   preserving the client's HTTP stream.
    /// - `pool.free_slot(victim_slot)` — releases all pages.
    /// - `states[victim_slot].reset()` — drops the slot's per-state KV scratch.
    /// - `clear_slot_prefix_ownership(victim_slot)` — releases radix refs.
    /// - Mark active[pos] as `Phase::Finished` and record its id in
    ///   `retracted_request_ids` so `cleanup()` skips the prefix-cache
    ///   publish (the prompt hasn't been fully materialized from this slot's
    ///   POV) and skips the completion counter increment.
    /// - `waiting.push_front(requeue)` — FIFO semantics push the retracted
    ///   request back to the FRONT so it gets preference next tick.
    ///
    /// Returns the total tokens freed back to the pool.
    pub(super) fn retract_longest_decode(
        &mut self,
        required_tokens: usize,
        protected_slot: usize,
    ) -> usize {
        if required_tokens == 0 {
            return 0;
        }
        let mut candidates: Vec<(u64, usize)> = self
            .active
            .iter()
            .filter(|req| matches!(req.phase, Phase::Decoding) && req.slot_idx != protected_slot)
            .map(|req| (req.id, req.generated_tokens.len()))
            .collect();
        if candidates.is_empty() {
            return 0;
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        let mut freed_total = 0usize;
        for (victim_id, decoded_len) in candidates {
            if freed_total >= required_tokens {
                break;
            }
            let Some(victim_pos) = self.active.iter().position(|r| r.id == victim_id) else {
                continue;
            };
            let victim_slot = self.active[victim_pos].slot_idx;
            let free_before = self.paged_kv_pool.free_count();

            warn!(
                "Request {}: retracting decode for pool OOM (decoded={} tok, \
                 free_before={}, target={} tok)",
                victim_id, decoded_len, free_before, required_tokens
            );

            let requeue = {
                let v = &mut self.active[victim_pos];
                IncomingRequest {
                    prompt: std::mem::take(&mut v.prompt),
                    max_tokens: v.max_tokens,
                    sampling: v.sampling.clone(),
                    stop: v.stop.take(),
                    priority: RequestPriority::Normal,
                    session_id: v.session_id.clone(),
                    delta_tx: v.delta_tx.clone(),
                }
            };

            self.paged_kv_pool.free_slot(victim_slot);
            if let Err(err) = self.states[victim_slot].reset() {
                error!(
                    "Request {}: slot state reset after retract failed: {}",
                    victim_id, err
                );
            }
            self.slot_materialized_prompt_lens[victim_slot] = 0;
            self.clear_slot_prefix_ownership(victim_slot);
            self.active[victim_pos].phase = Phase::Finished;
            self.retracted_request_ids.insert(victim_id);
            self.waiting.push_front(requeue);

            let freed_now = self.paged_kv_pool.free_count().saturating_sub(free_before);
            freed_total = freed_total.saturating_add(freed_now);
            info!(
                "Request {}: re-queued after retract (freed={} tok, \
                 running_total_freed={}, target={})",
                victim_id, freed_now, freed_total, required_tokens
            );
        }
        freed_total
    }

    pub(super) fn prefill_chunk_size(&self) -> usize {
        let signals = SchedulerSignals::queue_state(
            self.waiting.len(),
            self.active
                .iter()
                .filter(|req| matches!(req.phase, Phase::Decoding))
                .count(),
        );
        // When the model writes prefill K/V directly to the paged pool, there
        // is no per-slot contiguous scratch to size the chunk against, so the
        // `CONTIGUOUS_KV_TOKENS` cap does not apply and the configured
        // `prefill_chunk_size` (default 4096) is the only upper bound.
        let contiguous_cap = if self.model.prefill_uses_paged_pool() {
            usize::MAX
        } else {
            CONTIGUOUS_KV_TOKENS
        };
        let policy_chunk = DecodeAwareChunking {
            decode_active_chunk: self.config.decode_active_prefill_cap,
            idle_chunk: self.config.prefill_chunk_size,
        }
        .next_chunk_size(InferenceMode::Prefill, signals);
        let out = policy_chunk
            .max(1)
            .min(self.config.prefill_chunk_size)
            .min(contiguous_cap);
        log::debug!(
            "prefill_chunk_size: policy={policy_chunk} cap={contiguous_cap} cfg={} paged={} active_decodes={} => {out}",
            self.config.prefill_chunk_size,
            self.model.prefill_uses_paged_pool(),
            signals.active_decodes,
        );
        out
    }

    /// Pre-capture CUDA Graphs for batched decode at common batch sizes.
    ///
    /// Uses SGLang-style batch size schedule: 1, 2, 4, 8, 12, 16, 24, 32, 40, ...
    /// up to min(num_slots, 256). This covers the most common concurrent request
    /// counts without capturing every single size.
    ///
    /// Two-pass warmup:
    /// 1. Pass 1 drives forward_decode_batch per batch size, which populates the
    ///    cublasLt heuristic algo cache for every shape. In graph-capture mode
    ///    it also records a graph per batch size.
    /// 2. `autotune_all_cached_gemms_cuda` benchmarks all heuristic candidates and
    ///    replaces each shape's algo with the measured-fastest one.
    /// 3. Pass 2 (graph-capture mode only) re-captures graphs with the autotuned
    ///    algorithms. Eager decode (e.g. LoRA) skips pass 2 since no graphs
    ///    were cached.
    pub(super) fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if !self.paged_kv_pool.is_active() {
            return;
        }

        let graph_capture_enabled = self.model.supports_cuda_graph_decode();
        let max_bs = num_slots.min(256);
        let warmup_sizes = Self::cuda_graph_batch_sizes(max_bs);

        if graph_capture_enabled {
            info!(
                "Warming up CUDA Graphs for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        } else {
            info!(
                "Graph capture disabled (eager decode, e.g. LoRA); running \
                 eager warmup + cublasLt autotune for {} batch sizes (max {})...",
                warmup_sizes.len(),
                max_bs,
            );
        }
        let t0 = std::time::Instant::now();

        // Track how many slots we actually allocated so any early exit below
        // still frees them in the cleanup loop. Previously, a failing
        // `alloc_tokens` or `create_decode_context` would `return` with slots
        // still holding warmup tokens — `free_slots()` would then consider
        // them free while the pool still had dirty state, and the first real
        // request could inherit stale paged-KV entries.
        let mut allocated: usize = 0;
        let mut warmed: usize = 0;

        'warmup: {
            for slot in 0..max_bs {
                if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                    error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                    break 'warmup;
                }
                allocated = slot + 1;
            }

            // Lazy-init decode context before warmup.
            if self.decode_bufs.is_none() {
                match self
                    .model
                    .create_decode_context(self.states.len(), &self.paged_kv_pool)
                {
                    Ok(ctx) => self.decode_bufs = Some(ctx),
                    Err(e) => {
                        error!("Warmup: failed to create decode context: {}", e);
                        break 'warmup;
                    }
                }
            }

            let dummy_tokens: Vec<u32> = vec![0; max_bs];
            let slot_indices: Vec<usize> = (0..max_bs).collect();

            // Pass 1: drive forward for each warmup batch size. Populates the
            // cublasLt heuristic algo cache for all GEMM shapes used by decode.
            // In graph-capture mode, also captures a graph per batch size.
            warmed = self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);

            // Autotune: benchmark all heuristic candidates, replace with measured best.
            // Runs regardless of graph mode so eager LoRA decode lands on the same
            // tuned algorithms as graph-mode decode.
            if warmed > 0 {
                info!("Autotuning cublasLt GEMM algorithms ({} shapes)...", warmed);
                let t_at = std::time::Instant::now();
                unsafe {
                    cuda_kernels::ffi::autotune_all_cached_gemms_cuda(
                        self.model.device_context().stream.cu_stream(),
                    );
                }
                info!(
                    "cublasLt autotune done in {:.0}ms",
                    t_at.elapsed().as_secs_f64() * 1e3,
                );

                if graph_capture_enabled {
                    // Invalidate graphs captured with heuristic algos.
                    {
                        use crate::model::DecodeContextOps;
                        let decode_ctx = self
                            .decode_bufs
                            .as_mut()
                            .expect("invariant: decode_bufs initialized above");
                        for &bs in &warmup_sizes[..warmed] {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }

                    // Pass 2: re-capture with autotuned algorithms.
                    let recaptured =
                        self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);
                    info!(
                        "Re-captured {} graphs with autotuned GEMM algorithms",
                        recaptured,
                    );
                }
            }
        }

        // Always reached: frees any slots the warmup body allocated, whether
        // the body ran to completion or bailed on an error above.
        for slot in 0..allocated {
            self.paged_kv_pool.free_slot(slot);
            let _ = self.states[slot].reset();
        }

        let mode = if graph_capture_enabled {
            "CUDA Graph warmup"
        } else {
            "Eager warmup + cublasLt autotune"
        };
        info!(
            "{} done in {:.0}ms ({} batch sizes, max {})",
            mode,
            t0.elapsed().as_secs_f64() * 1e3,
            warmed,
            warmup_sizes.last().copied().unwrap_or(0),
        );
    }

    /// Single pass of graph warmup: set up metadata and forward for each batch size.
    fn warmup_graphs_pass(
        &mut self,
        warmup_sizes: &[usize],
        dummy_tokens: &[u32],
        slot_indices: &[usize],
    ) -> usize {
        let mut captured = 0;
        for &bs in warmup_sizes {
            let tokens = &dummy_tokens[..bs];
            let si = &slot_indices[..bs];
            let decode_ctx = self
                .decode_bufs
                .as_mut()
                .expect("invariant: decode_bufs initialized in warmup block above");

            {
                use crate::model::DecodeContextOps;
                let ctx = self.model.device_context();
                decode_ctx.set_batch_size(bs);
                if let Err(e) = decode_ctx.upload_token_ids(ctx, tokens) {
                    info!(
                        "Warmup: upload_token_ids for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
                match decode_ctx.update_metadata(ctx, &self.paged_kv_pool, si) {
                    Ok(reallocated) => {
                        if reallocated {
                            decode_ctx.invalidate_graph_cache(bs);
                        }
                    }
                    Err(e) => {
                        info!(
                            "Warmup: update_metadata for B={} failed ({}), skipping",
                            bs, e
                        );
                        break;
                    }
                }
                if let Err(e) = decode_ctx.plan_attention(
                    ctx,
                    bs,
                    self.model.num_q_heads(),
                    self.model.num_kv_heads(),
                    1,
                    self.model.head_dim(),
                    self.paged_kv_pool.format,
                ) {
                    info!(
                        "Warmup: plan_attention for B={} failed ({}), skipping",
                        bs, e
                    );
                    break;
                }
            }

            if let Err(e) = self.model.forward_decode_batch(
                tokens,
                &mut self.states,
                si,
                Some(&mut self.paged_kv_pool),
                decode_ctx,
                false,
            ) {
                info!(
                    "Warmup: graph capture for B={} failed ({}), skipping larger sizes",
                    bs, e
                );
                break;
            }
            let _ = self.model.device_context().sync();
            captured += 1;
        }
        captured
    }

    /// Generate batch size schedule for CUDA Graph warmup.
    ///
    /// Warm up EVERY batch size from 1..=min(max_bs, 64). This eliminates
    /// graph-miss eager fallbacks when the batch composition changes during
    /// request transitions, which was the primary source of p99 ITL spikes
    /// (100-150ms outliers at B=16).
    ///
    /// Beyond 64 we use a sparse schedule (step by 16) since the marginal
    /// difference between B=65 and B=64 graphs is negligible.
    fn cuda_graph_batch_sizes(max_bs: usize) -> Vec<usize> {
        let mut sizes = Vec::new();
        // Dense: every size from 1 to min(64, max_bs)
        let dense_limit = 64.min(max_bs);
        for bs in 1..=dense_limit {
            sizes.push(bs);
        }
        // Sparse: from 80 onward, step by 16
        let mut bs = 80;
        while bs <= max_bs {
            sizes.push(bs);
            bs += 16;
        }
        // Ensure max_bs itself is included
        if sizes.last() != Some(&max_bs) && max_bs > 8 {
            sizes.push(max_bs);
        }
        sizes
    }
}

impl<M: ModelForward> Drop for Scheduler<M> {
    fn drop(&mut self) {
        // Non-blocking shutdown hint. If the command channel is full we
        // still force-disconnect below; the coordinator's `rx.recv_timeout`
        // will then observe Disconnected and return from `run_once`.
        self.coordinator_handle.try_send_shutdown();
        // Force-disconnect both sides of the coordinator by swapping our
        // handle and events receiver for dummy channels we immediately
        // drop. Without this, `thread.join()` can deadlock: a blocking
        // `send(Shutdown)` on a full command channel, or a coordinator
        // that is itself stuck on `self.events.send(...)` because the
        // scheduler stopped draining events before reaching Drop.
        //
        // Dropping `_old_handle` here is the last `CoordinatorCommand`
        // sender (the scheduler was the only owner), so the command
        // channel disconnects. Dropping `_old_events` kills the
        // coordinator's event path on its next send. Either one is
        // sufficient to unwedge `run_once`; we do both for safety.
        let (dummy_coord, dummy_handle, dummy_events) = crate::kv_tier::Coordinator::new(1);
        drop(dummy_coord);
        let old_handle = std::mem::replace(&mut self.coordinator_handle, dummy_handle);
        let old_events = std::mem::replace(&mut self.coordinator_events, dummy_events);
        drop(old_handle);
        drop(old_events);
        if let Some(thread) = self.coordinator_thread.take() {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => warn!("Coordinator thread shutdown failed: {}", err),
                Err(_) => warn!("Coordinator thread panicked during shutdown"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{can_publish_prefix_pages, prefix_cache_retain_hard_cap_pages};

    const HARD_CAP: f64 = 0.90;

    #[test]
    fn retain_hard_cap_is_ninety_percent_of_pool() {
        assert_eq!(prefix_cache_retain_hard_cap_pages(100, HARD_CAP), 90);
        assert_eq!(prefix_cache_retain_hard_cap_pages(16, HARD_CAP), 14);
    }

    #[test]
    fn publish_is_denied_once_new_pages_cross_hard_cap() {
        assert!(can_publish_prefix_pages(70, 100, 20, HARD_CAP));
        assert!(can_publish_prefix_pages(80, 100, 10, HARD_CAP));
        assert!(!can_publish_prefix_pages(81, 100, 10, HARD_CAP));
        assert!(!can_publish_prefix_pages(90, 100, 1, HARD_CAP));
    }
}
