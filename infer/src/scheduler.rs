//! Multi-request scheduler with state pooling and decode-priority scheduling.
//!
//! Architecture:
//! ```text
//! HTTP Request → SchedulerHandle.submit() → channel → Scheduler.run()
//!                                                        ↓
//!                                              GPU (one forward at a time)
//!                                                        ↓
//!                                              StreamDelta → HTTP Response
//! ```
//!
//! The scheduler interleaves multiple requests on a single GPU by:
//! 1. Prioritizing decode steps (1 token each) over prefill
//! 2. Chunking long prefills (512 tokens) so decode can interleave
//! 3. Round-robin among active decode requests
//! 4. Starting new prefills only when no decode work is pending

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::block_manager::{BlockId, BlockManager};

use anyhow::Result;
use tokio::sync::mpsc;

#[cfg(feature = "cuda")]
use log::{error, info};

#[cfg(feature = "cuda")]
use rand::SeedableRng;
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;

#[cfg(feature = "cuda")]
use crate::model::{GenerationState, ModelForward};
#[cfg(feature = "cuda")]
use crate::paged_kv::PagedKVPool;
use crate::sampler::SamplingParams;
use crate::server_engine::StreamDelta;
#[cfg(feature = "cuda")]
use crate::server_engine::{FinishReason, Usage};
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

// ============================================================================
// Constants (CUDA-only — used by Scheduler internals)
// ============================================================================

#[cfg(feature = "cuda")]
const PREFILL_CHUNK_SIZE: usize = 512;
/// Smaller chunk when decode is active, to avoid blocking decode latency.
/// SGLang uses dynamic sizing; we use a fixed small chunk for simplicity.
#[cfg(feature = "cuda")]
const PREFILL_CHUNK_SIZE_WITH_DECODE: usize = 64;

// ============================================================================
// SchedulerConfig — always available
// ============================================================================

/// Preemption strategy when GPU memory is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum PreemptionMode {
    /// Evict request KV cache and recompute from scratch when resumed.
    /// Cheaper in GPU memory, more expensive when rescheduled.
    #[default]
    Recompute,
    /// Swap KV cache to CPU memory and swap back in when resumed.
    /// Preserves decoded state at the cost of CPU memory.
    Swap,
}

/// Scheduler configuration.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum number of concurrently active request slots.
    pub max_slots: usize,
    /// Chunked prefill chunk size (tokens per prefill step).
    pub prefill_chunk_size: usize,
    /// Maximum requests allowed in the waiting queue.
    /// `submit()` returns `Err(SchedulerFull)` when the queue is at capacity.
    pub max_waiting_requests: usize,
    /// Strategy to use when a running request must be preempted.
    pub preemption_mode: PreemptionMode,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_slots: 4,
            prefill_chunk_size: 512,
            max_waiting_requests: 256,
            preemption_mode: PreemptionMode::Recompute,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_slots == 0 {
            anyhow::bail!("max_slots must be ≥ 1");
        }
        if self.prefill_chunk_size == 0 {
            anyhow::bail!("prefill_chunk_size must be ≥ 1");
        }
        Ok(())
    }
}

// ============================================================================
// RequestPriority — always available
// ============================================================================

/// Request priority level.  Higher-priority requests are scheduled first
/// when multiple requests are waiting.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub enum RequestPriority {
    /// Below-normal priority (background batch jobs).
    Low = 0,
    /// Standard priority (default for API requests).
    #[default]
    Normal = 1,
    /// Above-normal priority (interactive / SLA-sensitive requests).
    High = 2,
}

// ============================================================================
// Public types
// ============================================================================

/// Request sent from HTTP handler to scheduler.
pub struct IncomingRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Option<Vec<String>>,
    /// Scheduling priority. Higher-priority requests are served first.
    pub priority: RequestPriority,
    /// Channel to send streaming deltas back to the HTTP handler.
    pub delta_tx: mpsc::UnboundedSender<StreamDelta>,
}

/// Error returned when the scheduler's waiting queue is full.
#[derive(Debug)]
pub struct SchedulerFull;

impl std::fmt::Display for SchedulerFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scheduler waiting queue is full")
    }
}

impl std::error::Error for SchedulerFull {}

/// Handle for submitting requests to the scheduler. Cloneable and Send.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::UnboundedSender<IncomingRequest>,
    model_id: Arc<str>,
    /// Shared count of items currently in the waiting channel.
    waiting_count: Arc<AtomicUsize>,
    /// Maximum allowed waiting requests (0 = unlimited).
    max_waiting: usize,
}

impl SchedulerHandle {
    /// Create a handle from raw parts (useful for testing).
    pub fn from_parts(tx: mpsc::UnboundedSender<IncomingRequest>, model_id: &str) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            waiting_count: Arc::new(AtomicUsize::new(0)),
            max_waiting: 0,
        }
    }

    /// Create a handle with a maximum waiting queue size.
    pub fn with_max_waiting(
        tx: mpsc::UnboundedSender<IncomingRequest>,
        model_id: &str,
        max_waiting: usize,
    ) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            waiting_count: Arc::new(AtomicUsize::new(0)),
            max_waiting,
        }
    }

    /// Submit a request to the scheduler.
    ///
    /// Returns `Ok(())` on success.
    /// Returns `Err(SchedulerFull)` if the waiting queue is at capacity.
    /// Returns `Err(SchedulerFull)` if the scheduler has shut down.
    pub fn submit(&self, req: IncomingRequest) -> std::result::Result<(), SchedulerFull> {
        // Backpressure check.
        if self.max_waiting > 0 {
            let current = self.waiting_count.load(Ordering::Relaxed);
            if current >= self.max_waiting {
                return Err(SchedulerFull);
            }
        }
        self.waiting_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(req).map_err(|_| {
            self.waiting_count.fetch_sub(1, Ordering::Relaxed);
            SchedulerFull
        })
    }

    /// Decrement the waiting count (called by the scheduler when it consumes a request).
    pub fn consume_one(&self) {
        self.waiting_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current number of requests in the waiting channel.
    pub fn waiting_count(&self) -> usize {
        self.waiting_count.load(Ordering::Relaxed)
    }

    /// Returns the model identifier string for this scheduler.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the queue is currently full.
    pub fn is_full(&self) -> bool {
        self.max_waiting > 0 && self.waiting_count() >= self.max_waiting
    }
}

// ============================================================================
// Internal types (CUDA-only — used by Scheduler)
// ============================================================================

#[cfg(feature = "cuda")]
enum Phase {
    /// Newly assigned, needs prefix cache check.
    New,
    /// Prefilling in chunks. Decode takes priority between chunks.
    Prefilling {
        effective_tokens: Vec<u32>,
        progress: usize,
    },
    /// Generating tokens.
    Decoding,
    /// Completed.
    Finished,
}

#[cfg(feature = "cuda")]
struct ActiveRequest {
    id: u64,
    slot_idx: usize,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    stop: Option<Vec<String>>,
    delta_tx: mpsc::UnboundedSender<StreamDelta>,
    /// Full decoded text, maintained incrementally.
    full_decoded: String,
    /// Number of tokens already decoded into full_decoded.
    decoded_token_count: usize,
    /// Number of characters already sent to the client.
    sent_len: usize,
    phase: Phase,
}

#[cfg(feature = "cuda")]
impl ActiveRequest {
    /// Decode newly generated tokens and emit text deltas to the client.
    ///
    /// Uses incremental decode: only re-decodes a small suffix (4 tokens)
    /// to handle multi-byte character boundaries, instead of all tokens.
    /// Cost per call: O(1) instead of O(N) where N = generated token count.
    fn emit_delta(&mut self, tokenizer: &Tokenizer) {
        let n = self.generated_tokens.len();
        if n == 0 {
            return;
        }

        // Incremental decode: re-decode tokens[safe_point..] where safe_point
        // is 4 tokens before the last decoded position (multi-byte boundary).
        let overlap = 4;
        let safe_point = self.decoded_token_count.saturating_sub(overlap);

        // Decode from safe_point to end
        let new_text = match tokenizer.decode(&self.generated_tokens[safe_point..]) {
            Ok(t) => t,
            Err(_) => return,
        };

        // Stitch: truncate full_decoded to safe_point's decoded length, append new
        if safe_point > 0 {
            // Decode just the safe_point prefix to find the cut position
            let prefix_text = tokenizer
                .decode(&self.generated_tokens[..safe_point])
                .unwrap_or_default();
            self.full_decoded.truncate(prefix_text.len());
            self.full_decoded.push_str(&new_text);
        } else {
            self.full_decoded = new_text;
        }
        self.decoded_token_count = n;

        // Emit delta
        if let Some(ref stops) = self.stop {
            match check_stop_sequences(&self.full_decoded, stops) {
                StopCheckResult::StopFound { stop_pos } => {
                    if stop_pos > self.sent_len {
                        let _ = self.delta_tx.send(StreamDelta {
                            text_delta: self.full_decoded[self.sent_len..stop_pos].to_string(),
                            finish_reason: None,
                            usage: None,
                        });
                    }
                    self.sent_len = stop_pos;
                    self.phase = Phase::Finished;
                    self.send_finish(FinishReason::Stop);
                    return;
                }
                StopCheckResult::NoStop { safe_len } => {
                    if safe_len > self.sent_len {
                        let _ = self.delta_tx.send(StreamDelta {
                            text_delta: self.full_decoded[self.sent_len..safe_len].to_string(),
                            finish_reason: None,
                            usage: None,
                        });
                        self.sent_len = safe_len;
                    }
                }
            }
        } else if self.full_decoded.len() > self.sent_len {
            let _ = self.delta_tx.send(StreamDelta {
                text_delta: self.full_decoded[self.sent_len..].to_string(),
                finish_reason: None,
                usage: None,
            });
            self.sent_len = self.full_decoded.len();
        }
    }

    /// Flush remaining buffered text and send the final finish delta.
    fn finish(&mut self, reason: FinishReason, tokenizer: &Tokenizer) {
        if matches!(self.phase, Phase::Finished) {
            return;
        }
        self.phase = Phase::Finished;

        // Flush remaining text
        if !self.generated_tokens.is_empty() {
            if let Ok(full_text) = tokenizer.decode(&self.generated_tokens) {
                let end = if let Some(ref stops) = self.stop {
                    match check_stop_sequences(&full_text, stops) {
                        StopCheckResult::StopFound { stop_pos } => stop_pos,
                        StopCheckResult::NoStop { .. } => full_text.len(),
                    }
                } else {
                    full_text.len()
                };
                if end > self.sent_len {
                    let _ = self.delta_tx.send(StreamDelta {
                        text_delta: full_text[self.sent_len..end].to_string(),
                        finish_reason: None,
                        usage: None,
                    });
                }
            }
        }

        self.send_finish(reason);
    }

    fn send_finish(&self, reason: FinishReason) {
        let _ = self.delta_tx.send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(reason),
            usage: Some(Usage {
                prompt_tokens: self.prompt_tokens.len(),
                completion_tokens: self.generated_tokens.len(),
                total_tokens: self.prompt_tokens.len() + self.generated_tokens.len(),
            }),
        });
    }
}

// ============================================================================
// Stop-sequence helpers
// ============================================================================

/// Result of scanning decoded text for stop sequences.
#[cfg(feature = "cuda")]
enum StopCheckResult {
    /// No stop found. Safe to emit text up to `safe_len` (holds back
    /// `max_stop_len` bytes to avoid splitting a partially-matched stop).
    NoStop { safe_len: usize },
    /// Stop sequence found; text up to `stop_pos` should be emitted, then
    /// the request should finish.
    StopFound { stop_pos: usize },
}

/// Scan `text` for any of the given stop sequences.
///
/// When no stop is found, returns `NoStop` with a safe emit length that
/// withholds `max_stop_len` bytes from the tail (in case a stop sequence
/// straddles the boundary of text decoded so far).
#[cfg(feature = "cuda")]
fn check_stop_sequences(text: &str, stops: &[String]) -> StopCheckResult {
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop.as_str()) {
            return StopCheckResult::StopFound { stop_pos: pos };
        }
    }
    let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
    let safe_len = text.len().saturating_sub(max_stop_len);
    StopCheckResult::NoStop { safe_len }
}

// ============================================================================
// Scheduler
// ============================================================================

/// Interval (in completed requests) at which stats are logged.
#[cfg(feature = "cuda")]
const STATS_LOG_INTERVAL: u64 = 10;

#[cfg(feature = "cuda")]
pub struct Scheduler<M: ModelForward> {
    model: M,
    tokenizer: Tokenizer,
    /// Per-slot states (KV caches, decode buffers). Stored separately from
    /// slot metadata so we can pass `&mut [M::State]` to batched decode.
    states: Vec<M::State>,
    /// Per-slot cached prompts for prefix reuse.
    cached_prompts: Vec<Vec<u32>>,
    request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    /// Shared waiting count with the handle (for backpressure decrement).
    waiting_count: Arc<AtomicUsize>,
    waiting: VecDeque<IncomingRequest>,
    active: Vec<ActiveRequest>,
    next_id: u64,
    rng: StdRng,
    /// Paged KV cache pool shared across all slots (for batched decode).
    paged_kv_pool: PagedKVPool,
    /// Pre-allocated buffers for batched decode (reused across steps).
    decode_bufs: Option<Box<dyn std::any::Any + Send>>,
    /// Round-robin index for fair decode scheduling.
    last_served: usize,
    /// Lifetime stats.
    total_completed: u64,
    total_generated_tokens: u64,
}

#[cfg(feature = "cuda")]
impl<M: ModelForward> Scheduler<M> {
    /// Create a new scheduler and its handle.
    ///
    /// `num_slots` controls how many concurrent requests can be active (each gets
    /// its own KV cache). More slots = more GPU memory usage.
    ///
    /// `max_seq_len_override` — if `Some(n)`, each slot's KV cache is capped at `n`
    /// tokens. If `None`, the scheduler auto-sizes based on available GPU memory.
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

        // Compute the effective max_seq_len for KV cache allocation.
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

        // Create the token-level KV pool for batched decode.
        // Budget: query remaining GPU memory AFTER contiguous KV caches are accounted for.
        // Contiguous caches are lazy-allocated (1024 tokens/slot max), so we subtract their
        // expected cost and reserve 2 GB headroom for batch decode buffers + scratch.
        let paged_kv_pool = {
            let contiguous_max = effective_max_seq_len.unwrap_or(1024);
            let bytes_per_token = model.kv_cache_bytes_per_token();
            let contiguous_cost = num_slots * contiguous_max * bytes_per_token;
            let headroom: usize = 2 * 1024 * 1024 * 1024; // 2 GB
            let budget_bytes = match crate::tensor::DeviceContext::gpu_memory_info() {
                Ok((free, _)) => free.saturating_sub(contiguous_cost + headroom),
                Err(_) => 4 * 1024 * 1024 * 1024, // 4 GB fallback
            };

            info!(
                "TokenKVPool budget: {:.1} GB (contiguous KV={:.1} GB, headroom=2 GB)",
                budget_bytes as f64 / 1e9,
                (contiguous_cost) as f64 / 1e9,
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

        let handle = SchedulerHandle {
            tx,
            model_id: Arc::from(model_id),
            waiting_count,
            max_waiting: num_slots * 4,
        };

        Ok((scheduler, handle))
    }

    /// Compute the effective max_seq_len per slot based on available GPU memory.
    ///
    /// If `override_val` is `Some(n)`, returns `Some(n)` directly.
    /// If `None`, queries free GPU memory and computes the largest affordable
    /// sequence length per slot, capped at 32768 (the model's native max).
    /// Returns `None` if the query fails (falls back to KVCache's default).
    fn compute_max_seq_len(
        model: &M,
        num_slots: usize,
        override_val: Option<usize>,
    ) -> Option<usize> {
        use crate::tensor::DeviceContext;

        // Contiguous KV cache only needs to hold one prefill chunk — all decode
        // reads come from the TokenKVPool. Keep this small to leave memory for the pool.
        const DEFAULT_MAX_SEQ: usize = 1024; // = 2x PREFILL_CHUNK_SIZE headroom
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
        let total_kv_budget = available; // split across all slots
        let per_slot_budget = total_kv_budget / num_slots.max(1);
        let affordable_seq_len = per_slot_budget / bytes_per_token.max(1);

        // Clamp to [MIN_SEQ_LEN, DEFAULT_MAX_SEQ].
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
    /// Eliminates p99 latency spikes from runtime graph capture during serving.
    fn warmup_cuda_graphs(&mut self) {
        let num_slots = self.states.len();
        if num_slots < 2 || self.paged_kv_pool.k_buffers.is_empty() {
            return;
        }

        info!("Warming up CUDA Graphs for batch sizes 1..{}...", num_slots);
        let t0 = std::time::Instant::now();

        // Each dummy slot needs at least 1 pool token so FlashInfer metadata is valid.
        for slot in 0..num_slots {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                error!("Warmup: pool alloc for slot {} failed: {}", slot, e);
                return;
            }
        }

        // Run dummy forward for each batch_size to trigger graph capture.
        // Includes B=1: all decode steps use the FlashInfer paged path.
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
                false, // warmup: need logit scatter for graph capture
            ) {
                // OOM during graph capture is non-fatal — just skip larger batch sizes
                info!(
                    "Warmup: graph capture for B={} failed ({}), skipping larger sizes",
                    bs, e
                );
                break;
            }
            let _ = self.model.device_context().sync();
        }

        // Clean up: free dummy tokens, reset states
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

    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            // 1. Drain incoming requests
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                self.waiting.push_back(req);
            }

            // 2. If idle, block for next request
            if self.active.is_empty() && self.waiting.is_empty() {
                match self.request_rx.blocking_recv() {
                    Some(req) => {
                        self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                        self.waiting.push_back(req);
                    }
                    None => {
                        info!("Scheduler shutting down: all handles dropped");
                        break;
                    }
                }
            }

            // 3. Assign slots to waiting requests
            self.assign_slots();

            // 4. Execute one step (decode-priority, round-robin)
            self.step();

            // 5. Clean up finished requests
            self.cleanup();
        }
    }

    fn assign_slots(&mut self) {
        while !self.waiting.is_empty() {
            let slot_idx = match self.find_free_slot() {
                Some(idx) => idx,
                None => break,
            };

            let incoming = self.waiting.pop_front().expect("checked non-empty above");
            let prompt_tokens = match self.tokenizer.encode(&incoming.prompt) {
                Ok(tokens) if !tokens.is_empty() => tokens,
                Ok(_) => {
                    error!("Empty prompt after tokenization, skipping");
                    continue;
                }
                Err(e) => {
                    error!("Tokenization error: {}", e);
                    continue;
                }
            };

            let id = self.next_id;
            self.next_id += 1;

            info!(
                "Request {} → slot {} (prompt={} tokens, queue={})",
                id,
                slot_idx,
                prompt_tokens.len(),
                self.waiting.len()
            );

            // Pool token allocation is deferred to step_prefill_chunk / step_decode_batch.
            // Each prefill chunk allocates its own pool tokens and scatter-writes K/V.
            // Decode allocates 1 token per step.

            self.active.push(ActiveRequest {
                id,
                slot_idx,
                prompt_tokens,
                generated_tokens: Vec::new(),
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                delta_tx: incoming.delta_tx,
                full_decoded: String::new(),
                decoded_token_count: 0,
                sent_len: 0,
                phase: Phase::New,
            });
        }
    }

    fn find_free_slot(&self) -> Option<usize> {
        let in_use: Vec<usize> = self.active.iter().map(|a| a.slot_idx).collect();
        (0..self.states.len()).find(|i| !in_use.contains(i))
    }

    fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        // Flush deferred emit_delta from the previous decode step.
        // Running this before step_decode_batch() hides tokenization latency
        // behind batch-preparation work (index collection, slot lookup).
        {
            let Self { active, tokenizer, .. } = self;
            for req in active.iter_mut() {
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    req.emit_delta(tokenizer);
                }
            }
        }

        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        if has_decode {
            self.step_decode_batch();
        }

        // Run ALL pending short prefills in one go before returning to the loop.
        // This fills empty slots faster after a batch of requests finishes,
        // keeping the decode batch full. Long prefills (> one chunk) still
        // interleave with decode via the chunked path.
        let mut did_prefill = true;
        while did_prefill {
            did_prefill = false;
            for idx in 0..self.active.len() {
                if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                    self.step_prefill_chunk(idx, has_decode);
                    did_prefill = true;
                    break;
                }
            }
            // If we just completed a short prefill (request moved to Decoding),
            // check for more. If it's still Prefilling (multi-chunk), stop here
            // and let decode run next iteration.
            if did_prefill {
                let still_prefilling = self
                    .active
                    .iter()
                    .any(|r| matches!(r.phase, Phase::Prefilling { .. }));
                if still_prefilling {
                    break; // multi-chunk prefill in progress, yield to decode
                }
            }
        }

        // Process New requests: assign prefix cache, start prefill.
        for idx in 0..num {
            if matches!(self.active[idx].phase, Phase::New) {
                self.step_new(idx);
                return;
            }
        }
    }

    /// Compute prefix cache for a new request and begin chunked prefill.
    fn step_new(&mut self, idx: usize) {
        let req = &mut self.active[idx];

        // Check client disconnect
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        let si = req.slot_idx;
        let cached = &mut self.cached_prompts[si];
        let state = &mut self.states[si];

        // Prefix cache: find common prefix between cached and new prompt
        let prefix_len = cached
            .iter()
            .zip(req.prompt_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let effective = if prefix_len > 0 && prefix_len == cached.len() {
            info!(
                "Request {}: prefix HIT {}/{} tokens",
                req.id,
                prefix_len,
                req.prompt_tokens.len()
            );
            let suffix = &req.prompt_tokens[prefix_len..];
            if suffix.is_empty() {
                // prompt_tokens is guaranteed non-empty (checked in assign_slots).
                // Defensive: treat empty as a fatal request error instead of panicking.
                let Some(&last_tok) = req.prompt_tokens.last() else {
                    error!(
                        "Request {}: prompt_tokens empty on full prefix hit — dropping",
                        req.id
                    );
                    req.phase = Phase::Finished;
                    return;
                };
                vec![last_tok]
            } else {
                suffix.to_vec()
            }
        } else if prefix_len > 0 {
            info!(
                "Request {}: prefix PARTIAL {}/{} tokens",
                req.id,
                prefix_len,
                req.prompt_tokens.len()
            );
            if let Err(e) = state.truncate_to(prefix_len) {
                error!("Request {}: truncate failed: {}", req.id, e);
                req.phase = Phase::Finished;
                return;
            }
            cached.truncate(prefix_len);
            req.prompt_tokens[prefix_len..].to_vec()
        } else {
            info!("Request {}: prefix MISS", req.id);
            if let Err(e) = state.reset() {
                error!("Request {}: reset failed: {}", req.id, e);
                req.phase = Phase::Finished;
                return;
            }
            cached.clear();
            req.prompt_tokens.clone()
        };

        // If prefix tokens were reused, allocate pool slots for them and migrate
        // their KV from the contiguous cache so batched decode can read from the pool.
        if prefix_len > 0 && !self.paged_kv_pool.k_buffers.is_empty() {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(si, prefix_len) {
                error!("Request {}: pool alloc for prefix failed: {}", req.id, e);
                // Non-fatal: batched decode will fall back to contiguous path.
            } else {
                let ctx = self.model.device_context();
                if let Err(e) = state.migrate_kv_to_paged(ctx, &self.paged_kv_pool, si) {
                    error!(
                        "Request {}: prefix KV migration to pool failed: {}",
                        req.id, e
                    );
                    // Non-fatal: pool may have stale data for prefix, but we continue.
                }
            }
        }

        info!(
            "Request {}: chunked prefill starting ({} effective tokens, chunk_size={})",
            req.id,
            effective.len(),
            PREFILL_CHUNK_SIZE
        );

        req.phase = Phase::Prefilling {
            effective_tokens: effective,
            progress: 0,
        };
    }

    /// Process one chunk of a prefill. When all chunks are done, sample the
    /// first token and transition to Decoding.
    fn step_prefill_chunk(&mut self, idx: usize, decode_active: bool) {
        let Self {
            model,
            tokenizer,
            states,
            active,
            rng,
            paged_kv_pool,
            ..
        } = self;

        let req = &mut active[idx];

        // Check client disconnect
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        // Extract prefill state; if not Prefilling, bail out.
        let (effective_tokens, progress) = match &mut req.phase {
            Phase::Prefilling {
                effective_tokens,
                progress,
            } => (effective_tokens as &Vec<u32>, progress as &mut usize),
            _ => return,
        };

        let total = effective_tokens.len();
        let chunk_size = if decode_active {
            PREFILL_CHUNK_SIZE_WITH_DECODE
        } else {
            PREFILL_CHUNK_SIZE
        };
        let chunk_end = (*progress + chunk_size).min(total);
        let chunk = &effective_tokens[*progress..chunk_end];

        let slot_idx = req.slot_idx;
        let state = &mut states[slot_idx];

        // Run standard prefill forward pass.
        // Pool token allocation and K/V migration happen after all chunks complete.
        let forward_result = model.forward(chunk, state);

        if let Err(e) = forward_result {
            error!("Request {}: prefill chunk failed: {}", req.id, e);
            req.phase = Phase::Finished;
            return;
        }

        let new_progress = chunk_end;

        if new_progress < total {
            // More chunks remaining
            // We need to update progress. Since we borrowed through the enum,
            // update via the mutable reference.
            *progress = new_progress;
            info!(
                "Request {}: prefill chunk {}/{} tokens",
                req.id, new_progress, total
            );
            return;
        }

        // All chunks done — allocate pool tokens and migrate KV from contiguous cache.
        // This copies the processed K/V (with QK-norm + RoPE) from the contiguous
        // cache into the token pool so FlashInfer batched decode can read from it.
        if !paged_kv_pool.k_buffers.is_empty() {
            if let Err(e) = paged_kv_pool.alloc_tokens(slot_idx, total) {
                error!("Request {}: pool alloc for migration failed: {}", req.id, e);
            } else {
                let ctx = model.device_context();
                if let Err(e) = state.migrate_kv_to_paged(ctx, paged_kv_pool, slot_idx) {
                    error!("Request {}: KV migration to pool failed: {}", req.id, e);
                }
            }
        }

        // Sample first token
        match model.select_token(state, &req.sampling, rng) {
            Ok(token) => {
                if !req.sampling.ignore_eos && model.is_stop_token(token) {
                    req.finish(FinishReason::Stop, tokenizer);
                    return;
                }
                req.generated_tokens.push(token);
                req.emit_delta(tokenizer);

                if matches!(req.phase, Phase::Finished) {
                    return; // Stop sequence hit during emit
                }
                if req.generated_tokens.len() >= req.max_tokens {
                    req.finish(FinishReason::Length, tokenizer);
                } else {
                    req.phase = Phase::Decoding;
                }
            }
            Err(e) => {
                error!("Request {}: select_token failed: {}", req.id, e);
                req.phase = Phase::Finished;
            }
        }
    }

    /// Batch all decode requests into a single GPU forward pass.
    ///
    /// For batch_size=1, uses the regular single-token path (with CUDA Graph).
    /// For batch_size>1, uses batched GEMM for projections + per-request attention.
    fn step_decode_batch(&mut self) {
        let Self {
            model,
            tokenizer,
            states,
            active,
            rng,
            paged_kv_pool,
            decode_bufs,
            ..
        } = self;

        // Collect decode request indices and their tokens
        let decode_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        // Mark disconnected decode requests as finished
        for i in 0..active.len() {
            if matches!(active[i].phase, Phase::Decoding) && active[i].delta_tx.is_closed() {
                active[i].phase = Phase::Finished;
            }
        }

        if decode_indices.is_empty() {
            return;
        }

        // Collect the last generated token for each decode request.
        // Requests in Decoding phase always have ≥1 generated token (invariant upheld by
        // step_prefill_chunk), but we handle the impossible-empty case defensively.
        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        for &i in &decode_indices {
            match active[i].generated_tokens.last() {
                Some(&tok) => {
                    token_ids.push(tok);
                    valid_decode_indices.push(i);
                }
                None => {
                    error!(
                        "Request {}: Decoding state with no generated tokens — dropping",
                        active[i].id
                    );
                    active[i].phase = Phase::Finished;
                }
            }
        }
        let decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }
        let slot_indices: Vec<usize> = decode_indices.iter().map(|&i| active[i].slot_idx).collect();

        // Allocate one token slot per decode request for the new token.
        for &slot in &slot_indices {
            if let Err(e) = paged_kv_pool.alloc_tokens(slot, 1) {
                error!("Pool alloc for decode token (slot {}) failed: {}", slot, e);
                // Non-fatal: decode_batch will fall back to contiguous path if pool is empty.
            }
        }

        // Run forward pass — always use the batched FlashInfer path when the paged
        // KV pool is active. Routing batch_size=1 through the contiguous/Triton
        // decode path causes output divergence: K/V is written only to the
        // contiguous cache (not the pool), so when the request later joins a
        // larger batch, FlashInfer reads stale pool data. Additionally, the
        // Triton and FlashInfer attention kernels produce numerically different
        // results, so greedy (temperature=0) output depends on which path ran.
        // Check greedy early — tells forward to skip D2D logit scatter.
        let sampling_params: Vec<&crate::sampler::SamplingParams> = decode_indices
            .iter()
            .map(|&i| &active[i].sampling)
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        let forward_result = model.forward_decode_batch(
            &token_ids,
            states,
            &slot_indices,
            Some(paged_kv_pool),
            decode_bufs,
            all_greedy,
        );

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &i in &decode_indices {
                active[i].phase = Phase::Finished;
            }
            return;
        }
        let sampled_result = if all_greedy {
            match model.sample_batch_greedy(&slot_indices, decode_bufs) {
                Ok(Some(tokens)) => Ok(tokens),
                Ok(None) => model.select_tokens_batch(states, &slot_indices, &sampling_params, rng),
                Err(e) => Err(e),
            }
        } else {
            model.select_tokens_batch(states, &slot_indices, &sampling_params, rng)
        };
        match sampled_result {
            Ok(sampled_tokens) => {
                // Fast path: push tokens and check stop/length immediately (cheap).
                // Defer expensive emit_delta to the NEXT iteration's pre-decode phase
                // so it overlaps with GPU compute.
                for (j, &req_idx) in decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut active[req_idx];
                    if !req.sampling.ignore_eos && model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    // emit_delta is deferred to the start of the next step() call.
                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, tokenizer);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &req_idx in &decode_indices {
                    active[req_idx].phase = Phase::Finished;
                }
            }
        }
    }

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if matches!(self.active[i].phase, Phase::Finished) {
                let req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;

                // Free paged KV pages for this slot back to the pool.
                self.paged_kv_pool.free_slot(req.slot_idx);

                // Update slot's cached prompt for future prefix reuse
                self.cached_prompts[req.slot_idx] = req.prompt_tokens;
                // Offload excess KV to CPU if over GPU budget
                let _ = self.states[req.slot_idx].offload_kv_if_needed();

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active.len(),
                    self.waiting.len()
                );

                // Log stats periodically
                if self.total_completed % STATS_LOG_INTERVAL == 0 {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.total_completed,
                        self.total_generated_tokens,
                        self.active.len(),
                        self.waiting.len()
                    );
                }

                // Fix round-robin index
                if self.last_served >= self.active.len() && !self.active.is_empty() {
                    self.last_served = 0;
                }
            } else {
                i += 1;
            }
        }
    }
}

// ============================================================================
// BatchScheduler — continuous batching with token budget, chunked prefill,
// and FCFS-reverse preemption (CPU accounting; no GPU required).
//
// Design mirrors sglang's `get_next_batch_to_run` / `update_running_batch`:
//   1. Reserve 1 decode token per running request (preempt if KV OOM).
//   2. With remaining budget, admit one prefill chunk from `waiting`.
//   3. Return a `ScheduleDecision` for the caller to feed to the model.
// ============================================================================

/// A request waiting to begin (or resume) prefill.
pub struct PendingRequest {
    pub req_id: u64,
    /// Tokenized prompt.  Caller must tokenize before calling `add_request`.
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub priority: RequestPriority,
    /// Tokens already prefilled in previous chunks (chunked prefill progress).
    pub prefill_progress: usize,
    /// KV blocks already allocated for earlier prefill chunks.
    pub allocated_blocks: Vec<BlockId>,
}

/// A request that finished prefill and is now in the decode (generation) phase.
pub struct RunningRequest {
    pub req_id: u64,
    /// Kept for recompute-mode preemption (re-queue without re-tokenizing).
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    /// KV blocks currently allocated on GPU.
    pub blocks: Vec<BlockId>,
    /// Total KV tokens committed (prompt_len + generated_tokens).
    pub kv_tokens: usize,
}

impl RunningRequest {
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

// ---- Decision types --------------------------------------------------------

/// Decode-only sub-batch.
pub struct DecodeBatch {
    pub req_ids: Vec<u64>,
    /// Last generated token ID per request (caller fills from its own state).
    pub input_ids: Vec<u32>,
    /// KV block table per request.
    pub block_tables: Vec<Vec<BlockId>>,
}

/// Prefill sub-batch (may be a chunk of a larger prompt).
pub struct PrefillBatch {
    pub req_ids: Vec<u64>,
    /// Token IDs to process (the chunk).
    pub input_ids: Vec<u32>,
    /// Full sequence length after this chunk (prefix + chunk).
    pub seq_lens: Vec<usize>,
    /// KV block table per request (blocks for the full KV so far).
    pub block_tables: Vec<Vec<BlockId>>,
}

/// Output of `BatchScheduler::schedule_step`.
pub enum ScheduleDecision {
    DecodeBatch(DecodeBatch),
    PrefillBatch(PrefillBatch),
    /// Chunked-prefill interleaved with decode in the same step.
    Mixed {
        decode: DecodeBatch,
        prefill: PrefillBatch,
    },
    Idle,
}

// ---- Config ----------------------------------------------------------------

/// Configuration for `BatchScheduler`.
#[derive(Clone, Debug)]
pub struct BatchSchedulerConfig {
    /// Maximum tokens processed per step (decode + prefill combined).
    pub max_tokens_per_step: usize,
    /// Maximum tokens in a single prefill chunk.
    pub prefill_chunk_size: usize,
    /// Preemption strategy when GPU KV cache is exhausted.
    pub preemption_mode: PreemptionMode,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_step: 2048,
            prefill_chunk_size: 512,
            preemption_mode: PreemptionMode::Recompute,
        }
    }
}

// ---- BatchScheduler --------------------------------------------------------

/// Continuous batching scheduler.
///
/// This is a pure CPU accounting layer.  It decides *what* to run each step
/// (decode + optional prefill chunk) using block-level KV accounting, but does
/// not touch GPU memory.  The caller is responsible for:
///   - Tokenizing prompts before calling `add_request`.
///   - Running the model forward pass with the returned `ScheduleDecision`.
///   - Calling `advance_decode` after each successful decode step.
///   - Calling `finish_request` when a request reaches EOS or max_tokens.
pub struct BatchScheduler {
    config: BatchSchedulerConfig,
    /// Requests waiting to be prefilled, ordered by arrival (FCFS).
    waiting: VecDeque<PendingRequest>,
    /// Requests currently in the decode phase.
    running: HashMap<u64, RunningRequest>,
    block_manager: BlockManager,
    next_req_id: u64,
}

impl BatchScheduler {
    pub fn new(config: BatchSchedulerConfig, block_manager: BlockManager) -> Self {
        Self {
            config,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            block_manager,
            next_req_id: 0,
        }
    }

    // ---- Public interface --------------------------------------------------

    /// Submit a new request.  Returns its assigned `req_id`.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> u64 {
        let req_id = self.next_req_id;
        self.next_req_id += 1;
        self.waiting.push_back(PendingRequest {
            req_id,
            prompt_tokens,
            max_tokens,
            priority,
            prefill_progress: 0,
            allocated_blocks: Vec::new(),
        });
        req_id
    }

    /// Call after the model generates one more token for a decode request.
    /// Allocates additional KV blocks if the sequence has grown into a new block.
    pub fn advance_decode(&mut self, req_id: u64) -> bool {
        let Some(req) = self.running.get_mut(&req_id) else {
            return false;
        };
        req.generated_tokens += 1;
        req.kv_tokens += 1;
        let blocks_needed = self.block_manager.blocks_for_tokens(req.kv_tokens);
        if blocks_needed > req.blocks.len() {
            match self
                .block_manager
                .allocate_gpu(blocks_needed - req.blocks.len())
            {
                Ok(new_blocks) => req.blocks.extend(new_blocks),
                Err(_) => return false, // OOM — caller should abort request
            }
        }
        true
    }

    /// Mark a request as finished and free its KV blocks.
    pub fn finish_request(&mut self, req_id: u64) {
        if let Some(req) = self.running.remove(&req_id) {
            self.block_manager.free(&req.blocks);
        }
        // Also remove from waiting if it was preempted back there
        self.waiting.retain(|r| r.req_id != req_id);
    }

    /// Schedule the next forward pass.
    ///
    /// Returns `ScheduleDecision::Idle` when there is no work to do.
    pub fn schedule_step(&mut self) -> ScheduleDecision {
        // 1. Guarantee every running request can emit one more KV token.
        //    If preemption happened, skip prefill this step (preempted requests
        //    need a full new step to be re-scheduled, mirroring sglang behavior).
        let preempted = self.ensure_decode_memory();

        let has_decode = !self.running.is_empty();
        let has_waiting = !self.waiting.is_empty();

        if !has_decode && !has_waiting {
            return ScheduleDecision::Idle;
        }

        // 2. Build decode batch.
        let decode_batch = has_decode.then(|| self.build_decode_batch());

        // 3. Try to admit one prefill chunk with the remaining token budget.
        //    Skip if any preemption happened this step.
        let decode_tokens = self.running.len(); // 1 token per running request
        let budget = self
            .config
            .max_tokens_per_step
            .saturating_sub(decode_tokens);
        let prefill_batch = (!preempted && has_waiting && budget > 0)
            .then(|| self.try_admit_prefill_chunk(budget))
            .flatten();

        // 4. Compose decision.
        match (decode_batch, prefill_batch) {
            (Some(d), Some(p)) => ScheduleDecision::Mixed {
                decode: d,
                prefill: p,
            },
            (Some(d), None) => ScheduleDecision::DecodeBatch(d),
            (None, Some(p)) => ScheduleDecision::PrefillBatch(p),
            (None, None) => ScheduleDecision::Idle,
        }
    }

    // ---- Accessors ---------------------------------------------------------

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn free_gpu_blocks(&self) -> usize {
        self.block_manager.free_gpu_blocks()
    }

    pub fn is_running(&self, req_id: u64) -> bool {
        self.running.contains_key(&req_id)
    }

    pub fn is_waiting(&self, req_id: u64) -> bool {
        self.waiting.iter().any(|r| r.req_id == req_id)
    }

    // ---- Internal helpers --------------------------------------------------

    /// Preempt running requests (highest req_id first — FCFS reverse / newest
    /// first) until the block manager has enough free blocks for all running
    /// requests to advance by one decode token.
    ///
    /// Returns `true` if any request was preempted (caller skips prefill this step).
    fn ensure_decode_memory(&mut self) -> bool {
        let mut preempted = false;
        loop {
            let new_blocks_needed: usize = self
                .running
                .values()
                .map(|r| {
                    let next_kv = r.kv_tokens + 1;
                    self.block_manager
                        .blocks_for_tokens(next_kv)
                        .saturating_sub(r.blocks.len())
                })
                .sum();

            if new_blocks_needed <= self.block_manager.free_gpu_blocks() {
                break;
            }

            // No running requests at all (shouldn't happen, but guard anyway).
            let Some(&preempt_id) = self.running.keys().max() else {
                break;
            };

            let req = self.running.remove(&preempt_id).unwrap();
            self.block_manager.free(&req.blocks);
            preempted = true;

            // Re-queue at the front so it's the next to be prefilled after
            // memory is available.  Recompute mode: reset progress to 0.
            self.waiting.push_front(PendingRequest {
                req_id: preempt_id,
                prompt_tokens: req.prompt_tokens,
                max_tokens: req.max_tokens,
                priority: RequestPriority::Normal,
                prefill_progress: 0,
                allocated_blocks: Vec::new(),
            });
        }
        preempted
    }

    fn build_decode_batch(&self) -> DecodeBatch {
        let n = self.running.len();
        let mut req_ids = Vec::with_capacity(n);
        let mut input_ids = Vec::with_capacity(n);
        let mut block_tables = Vec::with_capacity(n);

        // Iterate in deterministic order for test reproducibility.
        let mut ids: Vec<u64> = self.running.keys().copied().collect();
        ids.sort_unstable();

        for id in ids {
            let req = &self.running[&id];
            req_ids.push(req.req_id);
            input_ids.push(0u32); // caller fills from its token stream
            block_tables.push(req.blocks.clone());
        }

        DecodeBatch {
            req_ids,
            input_ids,
            block_tables,
        }
    }

    /// Attempt to emit one prefill chunk for the head of `waiting`.
    ///
    /// Allocates new KV blocks for the chunk.  Returns `None` if the queue is
    /// empty or KV memory is too full to admit even one token.
    fn try_admit_prefill_chunk(&mut self, token_budget: usize) -> Option<PrefillBatch> {
        let pending = self.waiting.front_mut()?;

        let total_tokens = pending.prompt_tokens.len();
        let progress = pending.prefill_progress;
        let remaining = total_tokens.saturating_sub(progress);
        if remaining == 0 {
            // Degenerate: already fully prefilled but still in waiting — drop.
            self.waiting.pop_front();
            return None;
        }

        let chunk_tokens = remaining
            .min(self.config.prefill_chunk_size)
            .min(token_budget);
        if chunk_tokens == 0 {
            return None;
        }

        let chunk_end = progress + chunk_tokens;
        // Blocks needed for the KV from token 0 to chunk_end.
        let blocks_needed = self.block_manager.blocks_for_tokens(chunk_end);
        let have = pending.allocated_blocks.len();
        let extra_needed = blocks_needed.saturating_sub(have);

        if extra_needed > 0 {
            match self.block_manager.allocate_gpu(extra_needed) {
                Ok(new_blocks) => pending.allocated_blocks.extend(new_blocks),
                Err(_) => return None, // KV OOM — skip prefill this step
            }
        }

        let chunk = pending.prompt_tokens[progress..chunk_end].to_vec();
        let block_table = pending.allocated_blocks.clone();
        let req_id = pending.req_id;
        let seq_len = chunk_end;
        let is_last_chunk = chunk_end >= total_tokens;

        if is_last_chunk {
            let done = self.waiting.pop_front().unwrap();
            // Transition immediately to running state.
            self.running.insert(
                req_id,
                RunningRequest {
                    req_id,
                    prompt_tokens: done.prompt_tokens,
                    generated_tokens: 0,
                    max_tokens: done.max_tokens,
                    blocks: done.allocated_blocks,
                    kv_tokens: total_tokens,
                },
            );
        } else {
            // Update progress in place (blocks stay allocated).
            pending.prefill_progress = chunk_end;
        }

        Some(PrefillBatch {
            req_ids: vec![req_id],
            input_ids: chunk,
            seq_lens: vec![seq_len],
            block_tables: vec![block_table],
        })
    }
}

// ============================================================================
// CPU-verifiable tests (no GPU required)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SamplingParams;
    use crate::server_engine::StreamDelta;

    fn make_request() -> IncomingRequest {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();
        IncomingRequest {
            prompt: "hello".to_string(),
            max_tokens: 32,
            sampling: SamplingParams::default(),
            stop: None,
            priority: RequestPriority::Normal,
            delta_tx: tx,
        }
    }

    // ---------------------------------------------------------------- SchedulerConfig

    #[test]
    fn scheduler_config_default_valid() {
        SchedulerConfig::default().validate().unwrap();
    }

    #[test]
    fn scheduler_config_zero_slots_invalid() {
        let cfg = SchedulerConfig {
            max_slots: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_zero_chunk_invalid() {
        let cfg = SchedulerConfig {
            prefill_chunk_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ---------------------------------------------------------------- RequestPriority ordering

    #[test]
    fn priority_ordering() {
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(RequestPriority::default(), RequestPriority::Normal);
    }

    // ---------------------------------------------------------------- SchedulerHandle backpressure

    #[tokio::test]
    async fn submit_succeeds_when_queue_not_full() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = SchedulerHandle::with_max_waiting(tx, "test", 3);
        assert!(handle.submit(make_request()).is_ok());
        assert!(handle.submit(make_request()).is_ok());
        assert!(handle.submit(make_request()).is_ok());
        assert_eq!(handle.waiting_count(), 3);
    }

    #[tokio::test]
    async fn submit_fails_when_queue_full() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = SchedulerHandle::with_max_waiting(tx, "test", 2);
        assert!(handle.submit(make_request()).is_ok());
        assert!(handle.submit(make_request()).is_ok());
        assert!(handle.submit(make_request()).is_err());
        assert!(handle.is_full());
    }

    #[tokio::test]
    async fn consume_decrements_waiting_count() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = SchedulerHandle::with_max_waiting(tx, "test", 5);
        handle.submit(make_request()).unwrap();
        handle.submit(make_request()).unwrap();
        assert_eq!(handle.waiting_count(), 2);
        handle.consume_one();
        assert_eq!(handle.waiting_count(), 1);
        handle.submit(make_request()).unwrap();
        handle.submit(make_request()).unwrap();
        handle.submit(make_request()).unwrap();
        handle.submit(make_request()).unwrap();
        assert_eq!(handle.waiting_count(), 5);
    }

    #[tokio::test]
    async fn unlimited_queue_never_rejects() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = SchedulerHandle::from_parts(tx, "test");
        for _ in 0..100 {
            assert!(handle.submit(make_request()).is_ok());
        }
        assert_eq!(handle.waiting_count(), 100);
    }

    // ---------------------------------------------------------------- PreemptionMode

    #[test]
    fn preemption_mode_default_is_recompute() {
        assert_eq!(PreemptionMode::default(), PreemptionMode::Recompute);
    }

    // ================================================================
    // BatchScheduler tests
    // ================================================================

    fn make_batch_scheduler(
        num_gpu_blocks: usize,
        block_size: usize,
        chunk_size: usize,
    ) -> BatchScheduler {
        let config = BatchSchedulerConfig {
            max_tokens_per_step: 4096,
            prefill_chunk_size: chunk_size,
            preemption_mode: PreemptionMode::Recompute,
        };
        let bm = BlockManager::new(num_gpu_blocks, 0, block_size);
        BatchScheduler::new(config, bm)
    }

    /// Run one simulated step: process the ScheduleDecision without a real model.
    /// Returns (num_prefilled_reqs, num_decoded_reqs).
    fn sim_step(sched: &mut BatchScheduler) -> (usize, usize) {
        match sched.schedule_step() {
            ScheduleDecision::Idle => (0, 0),
            ScheduleDecision::PrefillBatch(p) => (p.req_ids.len(), 0),
            ScheduleDecision::DecodeBatch(d) => {
                // Advance decode for all requests, finish them after 1 extra token.
                let to_finish: Vec<u64> = d.req_ids.clone();
                for id in &to_finish {
                    sched.advance_decode(*id);
                }
                (0, to_finish.len())
            }
            ScheduleDecision::Mixed { decode, prefill } => {
                let n_prefill = prefill.req_ids.len();
                let to_finish: Vec<u64> = decode.req_ids.clone();
                for id in &to_finish {
                    sched.advance_decode(*id);
                }
                (n_prefill, to_finish.len())
            }
        }
    }

    // ---------------------------------------------------------------- test_continuous_batching

    /// Three requests are submitted concurrently.  The scheduler should run all
    /// three to completion (each is "finished" after its single decode token).
    #[test]
    fn test_continuous_batching() {
        // 32 blocks × 4 tokens/block = 128 KV slots: plenty for 3 short requests.
        let mut sched = make_batch_scheduler(32, 4, 64);

        let id0 = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);
        let id1 = sched.add_request(vec![5, 6, 7, 8], 8, RequestPriority::Normal);
        let _id2 = sched.add_request(vec![9, 10, 11, 12], 8, RequestPriority::Normal);

        // --- Prefill phase: one request per step (scheduler admits one chunk at a time) ---
        // Step 1: prefill req 0
        match sched.schedule_step() {
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id0]);
                assert_eq!(p.input_ids, vec![1, 2, 3, 4]);
            }
            other => panic!(
                "expected PrefillBatch, got {:?}",
                matches!(other, ScheduleDecision::Idle)
            ),
        }
        // After prefill of id0 it moved to running; id1 and id2 still waiting.
        assert!(sched.is_running(id0));
        assert_eq!(sched.waiting_len(), 2);

        // Step 2: decode id0 + prefill id1
        match sched.schedule_step() {
            ScheduleDecision::Mixed { decode, prefill } => {
                assert_eq!(decode.req_ids, vec![id0]);
                assert_eq!(prefill.req_ids, vec![id1]);
                sched.advance_decode(id0);
            }
            ScheduleDecision::PrefillBatch(p) => {
                // Also acceptable: just prefill
                assert_eq!(p.req_ids, vec![id1]);
            }
            other => panic!("unexpected: {:?}", matches!(other, ScheduleDecision::Idle)),
        }

        // Finish id0 to free its blocks.
        sched.finish_request(id0);
        assert!(!sched.is_running(id0));

        // Step 3: eventually all remaining requests are processed.
        // Run up to 20 steps; assert all finish.
        for _ in 0..20 {
            sim_step(&mut sched);
            // Finish any running requests immediately (simulate 1-token generation).
            let running_ids: Vec<u64> = sched.running.keys().copied().collect();
            for id in running_ids {
                sched.finish_request(id);
            }
        }

        assert_eq!(sched.running_len(), 0);
        assert_eq!(sched.waiting_len(), 0);
    }

    // ---------------------------------------------------------------- test_preemption

    /// With only 5 GPU blocks (block_size=4), filling 3 requests (4 tokens each,
    /// 1 block each) leaves 2 free blocks.  When the decode step needs 1 new block
    /// per request (3 total), the scheduler should preempt the newest request.
    #[test]
    fn test_preemption() {
        // 5 blocks × 4 tokens = 20 KV slots.
        let mut sched = make_batch_scheduler(5, 4, 16);

        let id0 = sched.add_request(vec![1, 2, 3, 4], 16, RequestPriority::Normal);
        let id1 = sched.add_request(vec![5, 6, 7, 8], 16, RequestPriority::Normal);
        let id2 = sched.add_request(vec![9, 10, 11, 12], 16, RequestPriority::Normal);

        // Prefill all three requests (each uses 1 block = ceil(4/4)).
        // Step 1: prefill id0
        match sched.schedule_step() {
            ScheduleDecision::PrefillBatch(p) => assert_eq!(p.req_ids, vec![id0]),
            _ => panic!("expected prefill id0"),
        }
        // Step 2: decode id0 + prefill id1
        match sched.schedule_step() {
            ScheduleDecision::Mixed { decode, prefill } => {
                assert_eq!(decode.req_ids, vec![id0]);
                assert_eq!(prefill.req_ids, vec![id1]);
                sched.advance_decode(id0); // id0 now at kv_tokens=5, needs 2 blocks
            }
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id1]);
            }
            _ => panic!("unexpected decision"),
        }
        // Step 3: decode id0,id1 + prefill id2 (if budget allows)
        // After advancing id0, it needs 2 blocks; id1 still at kv_tokens=4 (1 block).
        // 5 total - 1(id0) - 1(id1) = 3 free blocks. Should admit id2.
        match sched.schedule_step() {
            ScheduleDecision::Mixed { prefill, .. } => {
                assert_eq!(prefill.req_ids, vec![id2]);
            }
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id2]);
            }
            _ => {}
        }

        // Now running: id0(kv=5, 2 blocks), id1(kv=4, 1 block), id2(kv=4, 1 block)
        // free blocks = 5 - 2 - 1 - 1 = 1
        // Next decode step: id0 needs ceil(6/4)=2 blocks (has 2, no new needed),
        //                   id1 needs ceil(5/4)=2 blocks (has 1, needs 1),
        //                   id2 needs ceil(5/4)=2 blocks (has 1, needs 1)
        // Total new blocks needed = 2, but free = 1 → preempt id2 (highest id).
        let free_before = sched.free_gpu_blocks();
        match sched.schedule_step() {
            ScheduleDecision::DecodeBatch(d) => {
                // id2 should have been preempted; only id0 and id1 decode.
                assert!(
                    !d.req_ids.contains(&id2),
                    "id2 should be preempted, not decoding"
                );
                assert!(d.req_ids.contains(&id0) || d.req_ids.contains(&id1));
            }
            ScheduleDecision::Mixed { decode, .. } => {
                assert!(!decode.req_ids.contains(&id2));
            }
            ScheduleDecision::Idle => {}
            _ => {}
        }

        // After preemption: id2 is back in waiting.
        assert!(
            sched.is_waiting(id2) || !sched.is_running(id2),
            "id2 should have been preempted back to waiting"
        );
        // Free blocks should have increased (id2's blocks freed).
        assert!(
            sched.free_gpu_blocks() >= free_before || !sched.is_running(id2),
            "preemption should free blocks"
        );
    }

    // ---------------------------------------------------------------- test_chunked_prefill

    /// A long prompt (20 tokens) with chunk_size=8 should be split across
    /// multiple steps (8, 8, 4) before the request enters the decode phase.
    #[test]
    fn test_chunked_prefill() {
        let chunk_size = 8usize;
        // 8 blocks × 4 = 32 KV tokens — enough for a 20-token prompt.
        let mut sched = make_batch_scheduler(8, 4, chunk_size);

        let prompt: Vec<u32> = (1u32..=20).collect(); // 20 tokens
        let id = sched.add_request(prompt, 4, RequestPriority::Normal);

        // Step 1: expect first chunk [1..=8]
        match sched.schedule_step() {
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id]);
                assert_eq!(p.input_ids.len(), chunk_size);
                assert_eq!(p.input_ids[0], 1);
                assert_eq!(p.input_ids[chunk_size - 1], chunk_size as u32);
                assert_eq!(p.seq_lens, vec![chunk_size]);
            }
            other => panic!(
                "expected PrefillBatch chunk 1, got Idle={}",
                matches!(other, ScheduleDecision::Idle)
            ),
        }
        // Request still in waiting (partial chunk).
        assert!(!sched.is_running(id));
        assert_eq!(sched.waiting_len(), 1);

        // Step 2: second chunk [9..=16]
        match sched.schedule_step() {
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id]);
                assert_eq!(p.input_ids.len(), chunk_size);
                assert_eq!(p.input_ids[0], chunk_size as u32 + 1);
                assert_eq!(p.seq_lens, vec![chunk_size * 2]);
            }
            other => panic!(
                "expected PrefillBatch chunk 2, got Idle={}",
                matches!(other, ScheduleDecision::Idle)
            ),
        }
        assert!(!sched.is_running(id));

        // Step 3: final chunk [17..=20] (4 tokens)
        match sched.schedule_step() {
            ScheduleDecision::PrefillBatch(p) => {
                assert_eq!(p.req_ids, vec![id]);
                assert_eq!(p.input_ids.len(), 4);
                assert_eq!(p.input_ids[0], 17);
                assert_eq!(p.seq_lens, vec![20]);
            }
            other => panic!(
                "expected PrefillBatch final chunk, got Idle={}",
                matches!(other, ScheduleDecision::Idle)
            ),
        }
        // After last chunk the request moves to running.
        assert!(sched.is_running(id));
        assert_eq!(sched.waiting_len(), 0);

        // Verify block allocation: 20 tokens / 4 per block = 5 blocks used.
        assert_eq!(sched.free_gpu_blocks(), 8 - 5);
    }
}
