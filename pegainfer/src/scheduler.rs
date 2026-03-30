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

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use log::{error, info};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

#[cfg(feature = "cuda")]
use crate::model::{GenerationState, ModelForward};
use crate::sampler::SamplingParams;
use crate::server_engine::{FinishReason, StreamDelta, Usage};
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

// ============================================================================
// Constants (CUDA-only — used by Scheduler internals)
// ============================================================================

#[cfg(feature = "cuda")]
const PREFILL_CHUNK_SIZE: usize = 512;

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

/// Stats snapshot from the scheduler.
pub struct SchedulerStats {
    pub active_requests: usize,
    pub waiting_requests: usize,
    pub total_completed: u64,
    pub total_generated_tokens: u64,
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
    /// Full decoded text of generated tokens so far.
    decoded_text: String,
    /// Number of characters already sent to the client.
    sent_len: usize,
    phase: Phase,
}

#[cfg(feature = "cuda")]
impl ActiveRequest {
    /// Decode newly generated tokens and emit text deltas to the client.
    fn emit_delta(&mut self, tokenizer: &Tokenizer) {
        if self.generated_tokens.is_empty() {
            return;
        }

        let full_text = match tokenizer.decode(&self.generated_tokens) {
            Ok(t) => t,
            Err(_) => return,
        };
        self.decoded_text = full_text.clone();

        // Check stop sequences in full text
        if let Some(ref stops) = self.stop {
            for stop in stops {
                if stop.is_empty() {
                    continue;
                }
                if let Some(pos) = full_text.find(stop.as_str()) {
                    // Stop found — emit text up to stop, then finish
                    if pos > self.sent_len {
                        let _ = self.delta_tx.send(StreamDelta {
                            text_delta: full_text[self.sent_len..pos].to_string(),
                            finish_reason: None,
                            usage: None,
                        });
                    }
                    self.sent_len = pos;
                    self.phase = Phase::Finished;
                    self.send_finish(FinishReason::Stop);
                    return;
                }
            }

            // Hold back max_stop_len characters to avoid sending partial stop matches
            let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
            let safe_len = full_text.len().saturating_sub(max_stop_len);
            if safe_len > self.sent_len {
                let _ = self.delta_tx.send(StreamDelta {
                    text_delta: full_text[self.sent_len..safe_len].to_string(),
                    finish_reason: None,
                    usage: None,
                });
                self.sent_len = safe_len;
            }
        } else if full_text.len() > self.sent_len {
            let _ = self.delta_tx.send(StreamDelta {
                text_delta: full_text[self.sent_len..].to_string(),
                finish_reason: None,
                usage: None,
            });
            self.sent_len = full_text.len();
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
                let mut end = full_text.len();
                // Truncate at stop sequence if present
                if let Some(ref stops) = self.stop {
                    for stop in stops {
                        if let Some(pos) = full_text.find(stop.as_str()) {
                            end = end.min(pos);
                        }
                    }
                }
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
    waiting: VecDeque<IncomingRequest>,
    active: Vec<ActiveRequest>,
    next_id: u64,
    rng: StdRng,
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
    pub fn new(
        model: M,
        tokenizer: Tokenizer,
        model_id: &str,
        num_slots: usize,
        seed: u64,
    ) -> Result<(Self, SchedulerHandle)> {
        let (tx, rx) = mpsc::unbounded_channel();

        let mut states = Vec::with_capacity(num_slots);
        let mut cached_prompts = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            states.push(model.create_state()?);
            cached_prompts.push(Vec::new());
            info!("Initialized state slot {}/{}", i + 1, num_slots);
        }

        info!(
            "Scheduler ready: model={}, slots={}, seed={}",
            model_id, num_slots, seed
        );

        let scheduler = Self {
            model,
            tokenizer,
            states,
            cached_prompts,
            request_rx: rx,
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            last_served: 0,
            total_completed: 0,
            total_generated_tokens: 0,
        };

        let handle = SchedulerHandle {
            tx,
            model_id: Arc::from(model_id),
        };

        Ok((scheduler, handle))
    }

    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        info!("Scheduler run loop started");
        loop {
            // 1. Drain incoming requests
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting.push_back(req);
            }

            // 2. If idle, block for next request
            if self.active.is_empty() && self.waiting.is_empty() {
                match self.request_rx.blocking_recv() {
                    Some(req) => self.waiting.push_back(req),
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

            let incoming = self.waiting.pop_front().unwrap();
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

            self.active.push(ActiveRequest {
                id,
                slot_idx,
                prompt_tokens,
                generated_tokens: Vec::new(),
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                delta_tx: incoming.delta_tx,
                decoded_text: String::new(),
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

        // Each step does TWO things:
        // 1. Batch decode ALL active decode requests (1 token each)
        // 2. Run ONE prefill chunk (so new requests don't starve)
        //
        // This interleaves decode and prefill properly. Without step 2,
        // decode-priority would completely starve prefills, making requests
        // process sequentially instead of concurrently.

        let has_decode = self.active.iter().any(|r| matches!(r.phase, Phase::Decoding));
        if has_decode {
            self.step_decode_batch();
        }

        // Run one prefill operation (ongoing chunk or new request setup)
        for idx in 0..num {
            if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                self.step_prefill_chunk(idx);
                return;
            }
        }
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
                req.id, prefix_len, req.prompt_tokens.len()
            );
            let suffix = &req.prompt_tokens[prefix_len..];
            if suffix.is_empty() {
                vec![*req.prompt_tokens.last().unwrap()]
            } else {
                suffix.to_vec()
            }
        } else if prefix_len > 0 {
            info!(
                "Request {}: prefix PARTIAL {}/{} tokens",
                req.id, prefix_len, req.prompt_tokens.len()
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
    fn step_prefill_chunk(&mut self, idx: usize) {
        let Self {
            model,
            tokenizer,
            states,
            active,
            rng,
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
        let chunk_end = (*progress + PREFILL_CHUNK_SIZE).min(total);
        let chunk = &effective_tokens[*progress..chunk_end];

        let state = &mut states[req.slot_idx];

        // Run forward pass for this chunk
        if let Err(e) = model.forward(chunk, state) {
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

        // All chunks done — sample first token
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
            ..
        } = self;

        // Collect decode request indices and their tokens
        let decode_indices: Vec<usize> = active.iter()
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

        let token_ids: Vec<u32> = decode_indices.iter()
            .map(|&i| *active[i].generated_tokens.last().unwrap())
            .collect();
        let slot_indices: Vec<usize> = decode_indices.iter()
            .map(|&i| active[i].slot_idx)
            .collect();

        // Run forward pass
        let forward_result = if decode_indices.len() == 1 {
            // Single request: use regular path (benefits from CUDA Graph)
            model.forward(&[token_ids[0]], &mut states[slot_indices[0]])
        } else {
            // Multiple requests: batched decode (GEMM for projections)
            model.forward_decode_batch(&token_ids, states, &slot_indices)
        };

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &i in &decode_indices {
                active[i].phase = Phase::Finished;
            }
            return;
        }

        // Sample and emit for each request
        for &req_idx in &decode_indices {
            let req = &mut active[req_idx];
            let state = &mut states[req.slot_idx];

            match model.select_token(state, &req.sampling, rng) {
                Ok(token) => {
                    if !req.sampling.ignore_eos && model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    req.emit_delta(tokenizer);

                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, tokenizer);
                    }
                }
                Err(e) => {
                    error!("Request {}: select_token failed: {}", req.id, e);
                    req.phase = Phase::Finished;
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
        let cfg = SchedulerConfig { max_slots: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn scheduler_config_zero_chunk_invalid() {
        let cfg = SchedulerConfig { prefill_chunk_size: 0, ..Default::default() };
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
}
