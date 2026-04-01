use super::*;

/// CUDA-backed scheduler state and initialization.
pub struct Scheduler<M: ModelForward> {
    pub(super) model: M,
    pub(super) tokenizer: Tokenizer,
    /// Per-slot states (KV caches, decode buffers). Stored separately from
    /// slot metadata so we can pass `&mut [M::State]` to batched decode.
    pub(super) states: Vec<M::State>,
    /// Per-slot cached prompts for prefix reuse.
    pub(super) cached_prompts: Vec<Vec<u32>>,
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
    pub(super) decode_bufs: Option<Box<dyn std::any::Any + Send>>,
    /// Round-robin index for fair decode scheduling.
    pub(super) last_served: usize,
    /// Lifetime stats.
    pub(super) total_completed: u64,
    pub(super) total_generated_tokens: u64,
}

impl<M: ModelForward> Scheduler<M> {
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
    ) -> Result<(Self, SchedulerHandle)> {
        Self::with_max_seq_len(model, tokenizer, model_id, num_slots, seed, None)
    }

    /// Create a new scheduler with an explicit max sequence length override.
    pub fn with_max_seq_len(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
        max_seq_len_override: Option<usize>,
    ) -> Result<(Self, SchedulerHandle)> {
        let (tx, rx) = mpsc::unbounded_channel();
        let effective_max_seq_len =
            Self::compute_max_seq_len(&model, num_slots, max_seq_len_override);

        let mut states = Vec::with_capacity(num_slots);
        let mut cached_prompts = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            let mut state = model.create_state()?;
            if let Some(max_seq) = effective_max_seq_len {
                state.set_max_seq_len(max_seq);
            }
            states.push(state);
            cached_prompts.push(Vec::new());
            info!("Initialized state slot {}/{}", i + 1, num_slots);
        }

        let paged_kv_pool = {
            let contiguous_max = effective_max_seq_len.unwrap_or(1024);
            let bytes_per_token = model.kv_cache_bytes_per_token();
            let contiguous_cost = num_slots * contiguous_max * bytes_per_token;
            let headroom: usize = 2 * 1024 * 1024 * 1024;
            let budget_bytes = match crate::tensor::DeviceContext::gpu_memory_info() {
                Ok((free, _)) => free.saturating_sub(contiguous_cost + headroom),
                Err(_) => 4 * 1024 * 1024 * 1024,
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous KV={:.1} GB, headroom=2 GB)",
                budget_bytes as f64 / 1e9,
                contiguous_cost as f64 / 1e9,
            );

            let ctx = model.device_context();
            PagedKVPool::new(
                ctx,
                model.num_kv_layers(),
                model.num_kv_heads(),
                model.head_dim(),
                num_slots,
                budget_bytes,
            )?
        };

        info!(
            "Scheduler ready: model={}, slots={}, seed={}, max_seq_len={}",
            model_id,
            num_slots,
            seed,
            effective_max_seq_len.map_or_else(|| "32768 (default)".to_string(), |n| n.to_string()),
        );

        let waiting_count = Arc::new(AtomicUsize::new(0));
        let scheduler = Self {
            model,
            tokenizer,
            states,
            cached_prompts,
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
        };

        let handle = SchedulerHandle::with_max_waiting(tx, model_id, num_slots * 4);
        debug_assert_eq!(handle.waiting_count(), 0);

        Ok((scheduler, handle))
    }

    /// Compute the effective max_seq_len per slot based on available GPU memory.
    fn compute_max_seq_len(
        model: &M,
        num_slots: usize,
        override_val: Option<usize>,
    ) -> Option<usize> {
        use crate::tensor::DeviceContext;

        const DEFAULT_MAX_SEQ: usize = 1024;
        const RESERVED_BYTES: usize = 512 * 1024 * 1024;
        const MIN_SEQ_LEN: usize = 256;

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

        let available = free_bytes.saturating_sub(RESERVED_BYTES);
        let bytes_per_token = model.kv_cache_bytes_per_token();
        let total_kv_budget = available;
        let per_slot_budget = total_kv_budget / num_slots.max(1);
        let affordable_seq_len = per_slot_budget / bytes_per_token.max(1);
        let effective = affordable_seq_len.clamp(MIN_SEQ_LEN, DEFAULT_MAX_SEQ);

        info!(
            "KV cache auto-sizing: gpu_free={:.1} GB, gpu_total={:.1} GB, \
             reserved={:.1} GB, bytes_per_token={}, num_slots={}, \
             affordable_seq_len={}, effective_max_seq_len={}",
            free_bytes as f64 / 1e9,
            total_bytes as f64 / 1e9,
            RESERVED_BYTES as f64 / 1e9,
            bytes_per_token,
            num_slots,
            affordable_seq_len,
            effective,
        );

        if affordable_seq_len < MIN_SEQ_LEN {
            error!(
                "KV cache: only {} tokens affordable per slot (need at least {}). \
                 Reduce --num-slots or free GPU memory.",
                affordable_seq_len, MIN_SEQ_LEN,
            );
        }

        Some(effective)
    }

    /// Pre-capture CUDA Graphs for batched decode at common batch sizes.
    pub(super) fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if num_slots < 2 || self.paged_kv_pool.k_buffers.is_empty() {
            return;
        }

        info!("Warming up CUDA Graphs for batch sizes 1..{}...", num_slots);
        let t0 = std::time::Instant::now();

        for slot in 0..num_slots {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                return;
            }
        }

        let dummy_tokens: Vec<u32> = vec![0; num_slots];
        let slot_indices: Vec<usize> = (0..num_slots).collect();
        for bs in 1..=num_slots {
            let tokens = &dummy_tokens[..bs];
            let si = &slot_indices[..bs];
            if let Err(e) = self.model.forward_decode_batch(
                tokens,
                &mut self.states,
                si,
                Some(&mut self.paged_kv_pool),
                &mut self.decode_bufs,
                false,
            ) {
                info!(
                    "Warmup: graph capture for B={} failed ({}), skipping larger sizes",
                    bs, e
                );
                break;
            }
            let _ = self.model.device_context().sync();
        }

        for slot in 0..num_slots {
            self.paged_kv_pool.free_slot(slot);
            let _ = self.states[slot].reset();
        }

        info!(
            "CUDA Graph warmup done in {:.0}ms (batch sizes 1..{})",
            t0.elapsed().as_secs_f64() * 1e3,
            num_slots,
        );
    }
}
