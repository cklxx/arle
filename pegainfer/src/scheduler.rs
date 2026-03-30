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
//! 2. Round-robin among active decode requests
//! 3. Starting new prefills only when no decode work is pending

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use log::{error, info};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::model::{GenerationState, ModelForward};
use crate::sampler::SamplingParams;
use crate::server_engine::{FinishReason, StreamDelta, Usage};
use crate::tokenizer::Tokenizer;

// ============================================================================
// Public types
// ============================================================================

/// Request sent from HTTP handler to scheduler.
pub struct IncomingRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Option<Vec<String>>,
    /// Channel to send streaming deltas back to the HTTP handler.
    pub delta_tx: mpsc::UnboundedSender<StreamDelta>,
}

/// Handle for submitting requests to the scheduler. Cloneable and Send.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::UnboundedSender<IncomingRequest>,
    model_id: Arc<str>,
}

impl SchedulerHandle {
    /// Create a handle from raw parts (useful for testing).
    pub fn from_parts(tx: mpsc::UnboundedSender<IncomingRequest>, model_id: &str) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
        }
    }

    /// Submit a request to the scheduler. Returns false if scheduler is shut down.
    pub fn submit(&self, req: IncomingRequest) -> bool {
        self.tx.send(req).is_ok()
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ============================================================================
// Internal types
// ============================================================================

struct Slot<S> {
    state: S,
    /// Prompt tokens from the last request processed in this slot (for prefix reuse).
    cached_prompt: Vec<u32>,
}

#[derive(PartialEq)]
enum Phase {
    NeedsPrefill,
    Decoding,
    Finished,
}

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
        if self.phase == Phase::Finished {
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

pub struct Scheduler<M: ModelForward> {
    model: M,
    tokenizer: Tokenizer,
    slots: Vec<Slot<M::State>>,
    request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    waiting: VecDeque<IncomingRequest>,
    active: Vec<ActiveRequest>,
    next_id: u64,
    rng: StdRng,
    /// Round-robin index for fair decode scheduling.
    last_served: usize,
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
        let (tx, rx) = mpsc::unbounded_channel();

        let mut slots = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            let state = model.create_state()?;
            slots.push(Slot {
                state,
                cached_prompt: Vec::new(),
            });
            info!("Initialized state slot {}/{}", i + 1, num_slots);
        }

        info!(
            "Scheduler ready: model={}, slots={}, seed={}",
            model_id, num_slots, seed
        );

        let scheduler = Self {
            model,
            tokenizer,
            slots,
            request_rx: rx,
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            last_served: 0,
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
                phase: Phase::NeedsPrefill,
            });
        }
    }

    fn find_free_slot(&self) -> Option<usize> {
        let in_use: Vec<usize> = self.active.iter().map(|a| a.slot_idx).collect();
        (0..self.slots.len()).find(|i| !in_use.contains(i))
    }

    fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        // Priority 1: serve one decode request (round-robin)
        for offset in 0..num {
            let idx = (self.last_served.wrapping_add(1).wrapping_add(offset)) % num;
            if self.active[idx].phase == Phase::Decoding {
                self.step_decode(idx);
                self.last_served = idx;
                return;
            }
        }

        // Priority 2: run one prefill
        for idx in 0..num {
            if self.active[idx].phase == Phase::NeedsPrefill {
                self.step_prefill(idx);
                return;
            }
        }
    }

    fn step_prefill(&mut self, idx: usize) {
        // Destructure to satisfy borrow checker (separate field borrows).
        let Self {
            model,
            tokenizer,
            slots,
            active,
            rng,
            ..
        } = self;

        let req = &mut active[idx];

        // Skip if client disconnected
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        let slot = &mut slots[req.slot_idx];

        // Prefix cache: find common prefix between cached and new prompt
        let prefix_len = slot
            .cached_prompt
            .iter()
            .zip(req.prompt_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let effective = if prefix_len > 0 && prefix_len == slot.cached_prompt.len() {
            // Full prefix hit — reuse all cached KV
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
            // Partial prefix hit — truncate KV and reuse common part
            info!(
                "Request {}: prefix PARTIAL {}/{} tokens",
                req.id, prefix_len, req.prompt_tokens.len()
            );
            if let Err(e) = slot.state.truncate_to(prefix_len) {
                error!("Request {}: truncate failed: {}", req.id, e);
                req.phase = Phase::Finished;
                return;
            }
            slot.cached_prompt.truncate(prefix_len);
            req.prompt_tokens[prefix_len..].to_vec()
        } else {
            // No match — full reset
            info!("Request {}: prefix MISS", req.id);
            if let Err(e) = slot.state.reset() {
                error!("Request {}: reset failed: {}", req.id, e);
                req.phase = Phase::Finished;
                return;
            }
            slot.cached_prompt.clear();
            req.prompt_tokens.clone()
        };

        // Run prefill forward pass
        if let Err(e) = model.forward(&effective, &mut slot.state) {
            error!("Request {}: prefill failed: {}", req.id, e);
            req.phase = Phase::Finished;
            return;
        }

        // Sample first token
        match model.select_token(&mut slot.state, &req.sampling, rng) {
            Ok(token) => {
                if !req.sampling.ignore_eos && model.is_stop_token(token) {
                    req.finish(FinishReason::Stop, tokenizer);
                    return;
                }
                req.generated_tokens.push(token);
                req.emit_delta(tokenizer);

                if req.phase == Phase::Finished {
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

    fn step_decode(&mut self, idx: usize) {
        let Self {
            model,
            tokenizer,
            slots,
            active,
            rng,
            ..
        } = self;

        let req = &mut active[idx];

        // Skip if client disconnected
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        let slot = &mut slots[req.slot_idx];
        let last_token = *req.generated_tokens.last().unwrap();

        // One decode step
        if let Err(e) = model.forward(&[last_token], &mut slot.state) {
            error!("Request {}: decode failed: {}", req.id, e);
            req.phase = Phase::Finished;
            return;
        }

        match model.select_token(&mut slot.state, &req.sampling, rng) {
            Ok(token) => {
                if !req.sampling.ignore_eos && model.is_stop_token(token) {
                    req.finish(FinishReason::Stop, tokenizer);
                    return;
                }
                req.generated_tokens.push(token);
                req.emit_delta(tokenizer);

                if req.phase == Phase::Finished {
                    return; // Stop sequence hit
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

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].phase == Phase::Finished {
                let req = self.active.remove(i);
                // Update slot's cached prompt for future prefix reuse
                self.slots[req.slot_idx].cached_prompt = req.prompt_tokens;
                // Offload excess KV to CPU if over GPU budget
                let _ = self.slots[req.slot_idx].state.offload_kv_if_needed();
                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    req.generated_tokens.len(),
                    self.active.len(),
                    self.waiting.len()
                );
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
