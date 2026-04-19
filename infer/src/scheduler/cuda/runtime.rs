use super::*;
use crate::prefix_cache::BlockMetadataUpdate;

fn best_reusable_slot_for_radix_hit(
    matched_blocks: &[crate::prefix_cache::BlockId],
    free_slots: &[usize],
    block_owner_slots: &std::collections::HashMap<crate::prefix_cache::BlockId, usize>,
    slot_materialized_prompt_lens: &[usize],
    block_size: usize,
) -> Option<(usize, usize, usize)> {
    for (idx, &bid) in matched_blocks.iter().enumerate().rev() {
        let Some(&slot_idx) = block_owner_slots.get(&bid) else {
            continue;
        };
        if !free_slots.contains(&slot_idx) {
            continue;
        }
        let reusable_prefix_len = (idx + 1) * block_size;
        let cached_prompt_len = slot_materialized_prompt_lens
            .get(slot_idx)
            .copied()
            .unwrap_or_default();
        if cached_prompt_len >= reusable_prefix_len && reusable_prefix_len > 0 {
            return Some((slot_idx, reusable_prefix_len, cached_prompt_len));
        }
    }
    None
}

fn lookup_blocks_ready_on_gpu(blocks: &[crate::kv_tier::LookupBlock]) -> bool {
    blocks
        .iter()
        .filter(|block| !matches!(block.hit_kind, crate::kv_tier::HitKind::Miss))
        .all(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
}

impl<M: ModelForward> Scheduler<M> {
    fn drain_coordinator_events(&mut self) {
        loop {
            match self.coordinator_events.try_recv() {
                Ok(crate::kv_tier::CoordinatorEvent::CommandQueued(_))
                | Ok(crate::kv_tier::CoordinatorEvent::StagingQueued { .. })
                | Ok(crate::kv_tier::CoordinatorEvent::SpillQueued { .. })
                | Ok(crate::kv_tier::CoordinatorEvent::RehydrateQueued { .. }) => {}
                Ok(crate::kv_tier::CoordinatorEvent::StagingCompleted { ticket }) => {
                    let Some(staged) = self.stage_waiting.remove(&ticket) else {
                        log::debug!(
                            "Dropping staging completion for unknown ticket {}",
                            ticket.0
                        );
                        continue;
                    };
                    let waited_ticks = self
                        .prefix_cache
                        .logical_clock()
                        .saturating_sub(staged.enqueued_at_clock);
                    // Flip the staged blocks to ReadyOnGpu so the next
                    // lookup_or_stage sees them as T0 hits. `u32::MAX` is
                    // a "ready but not yet owned by a slot" sentinel;
                    // `publish_to_prefix_cache` will overwrite it with a
                    // real slot once decode picks up the request.
                    for &block_id in &staged.block_ids {
                        let _ = self.prefix_cache.update_block_metadata(
                            block_id,
                            BlockMetadataUpdate {
                                location: Some(crate::kv_tier::BlockLocation::Gpu {
                                    slot: u32::MAX,
                                }),
                                ..BlockMetadataUpdate::default()
                            },
                        );
                    }
                    // Release the stage-era refs but **deliberately leave
                    // `soft_pin_until` at its stage-wait deadline**. The
                    // soft pin keeps eviction off these blocks until the
                    // re-admission's next `lookup_or_stage` refreshes it
                    // down to the normal keepalive_ticks. Without this,
                    // the window between `release` and the second lookup
                    // is unprotected — the next `assign_slots` call runs
                    // `evict_prefix_cache_if_pressured` before picking
                    // the parked request back up and could reclaim the
                    // just-staged blocks.
                    self.prefix_cache.release(&staged.block_ids);
                    log::debug!(
                        "Staging completed for ticket={} (prompt={} tokens, staged_blocks={}, waited_ticks={})",
                        ticket.0,
                        staged.prompt_tokens.len(),
                        staged.block_ids.len(),
                        waited_ticks
                    );
                    // `waiting_count` tracks the `request_rx` channel
                    // depth (decremented at intake, incremented only at
                    // `SchedulerHandle::submit`). `self.waiting` is the
                    // scheduler-internal VecDeque; moving a request
                    // into its front here is NOT a new submission, so
                    // we MUST NOT `fetch_add(1)` — doing so would
                    // permanently inflate the counter, break the idle
                    // exit guard, and saturate backpressure.
                    self.waiting.push_front(staged.request);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Coordinator thread gone — we cannot complete any
                    // future StageTicket. Cold-requeue every parked
                    // admission so the idle-exit guard in `run()` can
                    // fire and waiting requests are not leaked. Release
                    // stage-era refs but leave soft_pin_until untouched
                    // (same reasoning as the completion path above —
                    // the re-admission might still hit the prefix).
                    let pending = std::mem::take(&mut self.stage_waiting);
                    if !pending.is_empty() {
                        error!(
                            "Coordinator event channel disconnected; cold-requeuing {} staged admissions",
                            pending.len()
                        );
                    } else {
                        error!("Coordinator event channel disconnected");
                    }
                    // Same `waiting_count` semantics as the completion
                    // path: re-queuing into `self.waiting.push_front`
                    // is not a new submission, so the counter stays
                    // untouched here.
                    for (ticket, staged) in pending {
                        self.prefix_cache.release(&staged.block_ids);
                        warn!(
                            "Cold-requeuing ticket {} ({} prompt tokens, {} staged blocks dropped)",
                            ticket.0,
                            staged.prompt_tokens.len(),
                            staged.block_ids.len()
                        );
                        self.waiting.push_front(staged.request);
                    }
                    self.coordinator_unavailable = true;
                    break;
                }
                Ok(crate::kv_tier::CoordinatorEvent::SpillCompleted { ticket, locations }) => {
                    log::debug!(
                        "Spill completed for ticket={} ({} persisted blocks)",
                        ticket.0,
                        locations.len()
                    );
                }
                Ok(crate::kv_tier::CoordinatorEvent::SpillFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Spill failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                }
                Ok(crate::kv_tier::CoordinatorEvent::RehydrateCompleted {
                    ticket,
                    rehydrated_blocks,
                }) => {
                    log::debug!(
                        "Rehydrate completed for ticket={} ({} restored blocks)",
                        ticket.0,
                        rehydrated_blocks.len()
                    );
                }
                Ok(crate::kv_tier::CoordinatorEvent::RehydrateFailed {
                    ticket,
                    failed_block,
                    reason,
                }) => {
                    warn!(
                        "Rehydrate failed for ticket {} on block {:?}: {}",
                        ticket.0, failed_block, reason
                    );
                }
            }
        }
    }

    /// Run the scheduler loop. Blocks until all handles are dropped.
    pub fn run(mut self) {
        self.warmup_cuda_graphs();
        info!("Scheduler run loop started");
        loop {
            while let Ok(req) = self.request_rx.try_recv() {
                self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                self.waiting.push_back(req);
            }

            self.drain_coordinator_events();

            if self.active.is_empty() && self.waiting.is_empty() && self.stage_waiting.is_empty() {
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

            let step_start = std::time::Instant::now();
            self.assign_slots();
            let assign_us = step_start.elapsed().as_micros();

            let step_t = std::time::Instant::now();
            // FUTURE WORK (GPU/CPU overlap): `self.step()` already overlaps
            // decode with `emit_delta`, but batched decode itself is still
            // serial because `step_decode_batch()` runs
            // `forward_decode_batch(...)` and then immediately
            // `sample_batch_greedy(...)`, whose fast path launches argmax,
            // `ctx.sync()`s, and reads tokens/logprobs back. Real overlap
            // needs a `step_launch()` / `step_readback()` split at that
            // boundary; loop reordering alone does not create it.
            self.step();
            let step_us = step_t.elapsed().as_micros();

            let clean_t = std::time::Instant::now();
            self.cleanup();
            let clean_us = clean_t.elapsed().as_micros();
            self.metrics.set_active(self.active.len() as u64);
            self.metrics.set_waiting(self.waiting.len() as u64);
            if self.paged_kv_pool.is_active() {
                // Both in token units so kv_util = (total-free)/total is correct.
                let total =
                    (self.paged_kv_pool.max_total_pages * self.paged_kv_pool.page_size) as u64;
                let free = self.paged_kv_pool.free_count() as u64;
                self.metrics.set_kv_gpu_blocks(free, total);
            }
            // Throttled GPU memory query — at most once per second.
            if self.last_mem_query.elapsed().as_secs() >= 1 {
                self.last_mem_query = std::time::Instant::now();
                if let Ok((free, total)) =
                    crate::backend::cuda::tensor::DeviceContext::gpu_memory_info()
                {
                    let active = (total - free) as u64;
                    self.peak_mem_bytes = self.peak_mem_bytes.max(active);
                    self.metrics
                        .set_memory_bytes(active, self.peak_mem_bytes, 0);
                }
            }

            let total_us = step_start.elapsed().as_micros();
            if total_us > 50_000 {
                // Log slow iterations (>50ms)
                info!(
                    "Scheduler step: assign={}us step={}us cleanup={}us total={}us active={}",
                    assign_us,
                    step_us,
                    clean_us,
                    total_us,
                    self.active.len()
                );
            }
        }
    }

    fn assign_slots(&mut self) {
        let mut admission_budget_remaining_this_tick = self.admission_budget_pages();
        while !self.waiting.is_empty() {
            let _ = self.evict_prefix_cache_if_pressured();
            let free_slots = self.free_slots();
            if free_slots.is_empty() {
                break;
            }

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

            let planner = if self.coordinator_unavailable {
                None
            } else {
                Some(&self.coordinator_handle as &dyn crate::kv_tier::StagePlanner)
            };
            let lookup = self.prefix_cache.lookup_or_stage(
                &prompt_tokens,
                crate::kv_tier::LookupHeuristics::default(),
                planner,
            );
            let radix_blocks: Vec<_> = lookup
                .blocks
                .iter()
                .filter_map(|block| block.block_id)
                .collect();
            if let Some(ticket) = lookup.staging_ticket {
                if lookup.recompute_advised {
                    self.prefix_cache.release(&radix_blocks);
                } else {
                    let stage_deadline = self
                        .prefix_cache
                        .logical_clock()
                        .saturating_add(self.config.stage_wait_keepalive_ticks);
                    for &block_id in &radix_blocks {
                        let _ = self.prefix_cache.update_block_metadata(
                            block_id,
                            BlockMetadataUpdate {
                                soft_pin_until: Some(Some(stage_deadline)),
                                ..BlockMetadataUpdate::default()
                            },
                        );
                    }
                    log::info!(
                        "Request {} staged on ticket={} (prompt={} tokens, staged_blocks={})",
                        self.next_id,
                        ticket.0,
                        prompt_tokens.len(),
                        radix_blocks.len()
                    );
                    self.stage_waiting.insert(
                        ticket,
                        super::core::StagedAdmission {
                            request: incoming,
                            prompt_tokens,
                            block_ids: radix_blocks,
                            enqueued_at_clock: self.prefix_cache.logical_clock(),
                        },
                    );
                    continue;
                }
            }

            let ready_on_gpu = lookup_blocks_ready_on_gpu(&lookup.blocks);
            let radix_hit_len = if ready_on_gpu && !lookup.recompute_advised {
                lookup.matched_len
            } else {
                0
            };
            // Reserve pages for BOTH prefill and the request's expected decode
            // growth. Prefill pages stick around for decode, and each output
            // token eventually crosses a page boundary. Under-counting here
            // was the cause of the 14 residual decode-path pool allocations
            // at c=16 (see 2026-04-18-c16-paged-pool-admission-overcommit.md).
            let uncached_tokens = prompt_tokens.len().saturating_sub(radix_hit_len);
            let decode_budget_tokens = incoming.max_tokens;
            let needed_pages = uncached_tokens
                .saturating_add(decode_budget_tokens)
                .div_ceil(self.paged_kv_pool.page_size);
            if needed_pages > admission_budget_remaining_this_tick {
                self.prefix_cache.release(&radix_blocks);
                info!(
                    "Request {} held for pool (need={} pages, budget={})",
                    self.next_id, needed_pages, admission_budget_remaining_this_tick
                );
                self.waiting.push_front(incoming);
                break;
            }
            admission_budget_remaining_this_tick =
                admission_budget_remaining_this_tick.saturating_sub(needed_pages);
            let gpu_ready_prefix_blocks: Vec<_> = lookup
                .blocks
                .iter()
                .take_while(|block| matches!(block.hit_kind, crate::kv_tier::HitKind::ReadyOnGpu))
                .filter_map(|block| block.block_id)
                .collect();
            let reusable_gpu_prefix = best_reusable_slot_for_radix_hit(
                &gpu_ready_prefix_blocks,
                &free_slots,
                &self.block_owner_slots,
                &self.slot_materialized_prompt_lens,
                self.prefix_cache.block_size(),
            );
            let reusable = if ready_on_gpu && !lookup.recompute_advised {
                reusable_gpu_prefix
            } else {
                None
            };
            let (slot_idx, reusable_prefix_len, reusable_cached_prompt_len) =
                reusable.unwrap_or((free_slots[0], 0, 0));
            self.prefix_cache.release(&radix_blocks);

            let id = self.next_id;
            self.next_id += 1;

            if reusable_prefix_len > 0 {
                info!(
                    "Request {} → slot {} (prompt={} tokens, radix_hit={}, reusable_prefix={}, cached_len={}, queue={})",
                    id,
                    slot_idx,
                    prompt_tokens.len(),
                    radix_hit_len,
                    reusable_prefix_len,
                    reusable_cached_prompt_len,
                    self.waiting.len()
                );
            } else {
                let bytes_not_on_gpu =
                    lookup.matched_len > 0 && (!ready_on_gpu || lookup.recompute_advised);
                let no_reusable_free_slot = lookup.matched_len > 0
                    && !gpu_ready_prefix_blocks.is_empty()
                    && reusable_gpu_prefix.is_none();
                // A radix match is only reusable when the bytes already live in
                // T0 and a free slot still materializes that prefix. When
                // either precondition fails we degrade to cold prefill, but we
                // keep both blockers in the log so admission debugging does not
                // lose the "no reusable free slot" signal behind a staging miss.
                if bytes_not_on_gpu || no_reusable_free_slot {
                    info!(
                        "Request {} → slot {} (prompt={} tokens, radix_hit={} not reusable: bytes_not_on_gpu={}, no_free_slot={}, queue={})",
                        id,
                        slot_idx,
                        prompt_tokens.len(),
                        lookup.matched_len,
                        bytes_not_on_gpu,
                        no_reusable_free_slot,
                        self.waiting.len()
                    );
                } else {
                    info!(
                        "Request {} → slot {} (prompt={} tokens, queue={})",
                        id,
                        slot_idx,
                        prompt_tokens.len(),
                        self.waiting.len()
                    );
                }
            }

            self.active.push(ActiveRequest {
                id,
                slot_idx,
                admitted_at: std::time::Instant::now(),
                first_token_at: None,
                prompt: incoming.prompt,
                prompt_tokens,
                generated_tokens: Vec::new(),
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                session_id: incoming.session_id,
                delta_tx: incoming.delta_tx,
                full_decoded: String::new(),
                decoded_token_count: 0,
                sent_len: 0,
                phase: Phase::New,
                cacheable_prompt_len: 0,
                prefix_byte_len: 0,
                latest_logprob: None,
                reusable_prefix_len,
                reusable_cached_prompt_len,
                reserved_pool_pages: needed_pages,
            });
        }
    }

    /// Find all free slot indices.
    fn free_slots(&self) -> Vec<usize> {
        let in_use: Vec<usize> = self.active.iter().map(|a| a.slot_idx).collect();
        (0..self.states.len())
            .filter(|i| !in_use.contains(i))
            .collect()
    }

    fn cleanup(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if matches!(self.active[i].phase, Phase::Finished) {
                let req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;
                self.clear_slot_prefix_ownership(req.slot_idx);

                if let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    self.slot_materialized_prompt_lens[req.slot_idx] = prompt_vec.len();
                    self.publish_to_prefix_cache(req.slot_idx, &prompt_vec, req.session_id.clone());
                } else {
                    self.slot_materialized_prompt_lens[req.slot_idx] = 0;
                }
                self.paged_kv_pool.free_slot(req.slot_idx);

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;
                let e2e_s = req.admitted_at.elapsed().as_secs_f64();
                let ttft_s = req
                    .first_token_at
                    .map(|t| t.duration_since(req.admitted_at).as_secs_f64())
                    .unwrap_or(e2e_s);
                let tpot_s = if gen_tokens > 1 {
                    (e2e_s - ttft_s).max(0.0) / (gen_tokens - 1) as f64
                } else {
                    0.0
                };
                self.metrics.record_request_completed(
                    req.prompt_tokens.len() as u64,
                    gen_tokens,
                    ttft_s,
                    tpot_s,
                    e2e_s,
                );

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active.len(),
                    self.waiting.len()
                );

                if self.total_completed % STATS_LOG_INTERVAL == 0 {
                    info!(
                        "Scheduler stats: completed={}, generated_tokens={}, active={}, waiting={}",
                        self.total_completed,
                        self.total_generated_tokens,
                        self.active.len(),
                        self.waiting.len()
                    );
                }

                if self.last_served >= self.active.len() && !self.active.is_empty() {
                    self.last_served = 0;
                }
            } else {
                i += 1;
            }
        }

        // M2a: amortised LRU eviction for the prefix cache.
        // Runs after the per-request free_slot loop so the pool's
        // retained fraction is fresh. No-op unless retained pages
        // crossed `PREFIX_CACHE_HIGH_WATER`; then evicts down to
        // `PREFIX_CACHE_LOW_WATER`. See
        // `core::Scheduler::evict_prefix_cache_if_pressured`.
        let _reclaimed = self.evict_prefix_cache_if_pressured();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::paged_kv::PagedKVPool;
    use crate::backend::cuda::tensor::DeviceContext;
    use crate::metrics::ServerMetrics;
    use crate::model::{
        DecodeContextOps, GenerationState, KVCacheDtype, ModelForward, kv_cache::KVFormat,
    };
    use crate::prefix_cache::BlockId;
    use crate::sampler::SamplingParams;
    use crate::scheduler::cuda::core::PREFIX_CACHE_BLOCK_SIZE;
    use crate::server_engine::CompletionStreamDelta;
    use crate::tokenizer::Tokenizer;
    use crate::types::SessionId;
    use anyhow::Result;
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex, Once, OnceLock};
    use tempfile::tempdir;
    use tokenizers::Tokenizer as HfTokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    #[derive(Default)]
    struct CapturingLogger {
        records: Mutex<Vec<String>>,
    }

    impl CapturingLogger {
        fn clear(&self) {
            self.records.lock().expect("logger mutex poisoned").clear();
        }

        fn snapshot(&self) -> Vec<String> {
            self.records.lock().expect("logger mutex poisoned").clone()
        }
    }

    impl log::Log for CapturingLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Info
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                self.records
                    .lock()
                    .expect("logger mutex poisoned")
                    .push(format!("{}", record.args()));
            }
        }

        fn flush(&self) {}
    }

    fn test_logger() -> &'static CapturingLogger {
        static LOGGER: OnceLock<CapturingLogger> = OnceLock::new();
        static INIT: Once = Once::new();

        let logger = LOGGER.get_or_init(CapturingLogger::default);
        INIT.call_once(|| {
            let _ = log::set_logger(logger);
            log::set_max_level(log::LevelFilter::Info);
        });
        logger
    }

    struct FakeDecodeContext;

    impl DecodeContextOps for FakeDecodeContext {
        fn upload_token_ids(&mut self, _ctx: &DeviceContext, _tokens: &[u32]) -> Result<()> {
            Ok(())
        }

        fn update_metadata(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot_indices: &[usize],
        ) -> Result<bool> {
            Ok(false)
        }

        fn plan_attention(
            &mut self,
            _ctx: &DeviceContext,
            _batch_size: usize,
            _num_q_heads: usize,
            _num_kv_heads: usize,
            _page_size: usize,
            _head_dim: usize,
            _kv_format: KVFormat,
        ) -> Result<()> {
            Ok(())
        }

        fn set_batch_size(&mut self, _bs: usize) {}

        fn invalidate_graph_cache(&mut self, _batch_size: usize) {}
    }

    struct FakeState {
        logits: crate::backend::cuda::tensor::DeviceVec,
    }

    impl FakeState {
        fn new(ctx: &DeviceContext) -> Result<Self> {
            Ok(Self {
                logits: crate::backend::cuda::tensor::DeviceVec::zeros(ctx, 1)?,
            })
        }
    }

    impl GenerationState for FakeState {
        fn logits(&self) -> &crate::backend::cuda::tensor::DeviceVec {
            &self.logits
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }

        fn truncate_to(&mut self, _len: usize) -> Result<()> {
            Ok(())
        }

        fn set_max_seq_len(&mut self, _max_seq: usize) {}

        fn set_kv_dtype(&mut self, _dtype: KVCacheDtype) {}

        fn migrate_kv_to_paged(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn migrate_kv_range_to_paged(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot: usize,
            _start_pos: usize,
            _token_count: usize,
        ) -> Result<()> {
            Ok(())
        }
    }

    struct FakeModel {
        ctx: DeviceContext,
    }

    impl ModelForward for FakeModel {
        type State = FakeState;
        type DecodeContext = FakeDecodeContext;

        fn create_state(&self) -> Result<Self::State> {
            FakeState::new(&self.ctx)
        }

        fn create_decode_context(
            &self,
            _max_batch_size: usize,
            _pool: &PagedKVPool,
        ) -> Result<Self::DecodeContext> {
            Ok(FakeDecodeContext)
        }

        fn kv_cache_bytes_per_token(&self) -> usize {
            4
        }

        fn num_kv_layers(&self) -> usize {
            1
        }

        fn num_kv_heads(&self) -> usize {
            1
        }

        fn head_dim(&self) -> usize {
            1
        }

        fn num_q_heads(&self) -> usize {
            1
        }

        fn forward_prefill(&self, _tokens: &[u32], _state: &mut Self::State) -> Result<()> {
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &mut Self::State) -> Result<()> {
            Ok(())
        }

        fn select_token(
            &self,
            _state: &mut Self::State,
            _params: &SamplingParams,
            _rng: &mut rand::rngs::StdRng,
        ) -> Result<u32> {
            Ok(0)
        }

        fn is_stop_token(&self, _token_id: u32) -> bool {
            false
        }

        fn device_context(&self) -> &DeviceContext {
            &self.ctx
        }
    }

    fn build_test_tokenizer() -> Tokenizer {
        let dir = tempdir().expect("tempdir");
        let vocab_path = dir.path().join("vocab.json");
        fs::write(&vocab_path, r#"{"[UNK]":0,"a":1}"#).expect("write vocab");

        let model = WordLevel::builder()
            .files(vocab_path.to_string_lossy().into_owned())
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel tokenizer");
        let mut raw = HfTokenizer::new(model);
        raw.with_pre_tokenizer(Some(Whitespace::default()));

        let tokenizer_path = dir.path().join("tokenizer.json");
        let tokenizer_json = serde_json::to_string(&raw).expect("serialize tokenizer");
        fs::write(&tokenizer_path, tokenizer_json).expect("write tokenizer");

        Tokenizer::from_file(dir.path().to_str().expect("tokenizer dir str"))
            .expect("load test tokenizer")
    }

    fn repeated_prompt(token_count: usize) -> String {
        std::iter::repeat_n("a", token_count)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn test_request(idx: usize, prompt: &str) -> IncomingRequest {
        let (delta_tx, _delta_rx) = tokio::sync::mpsc::unbounded_channel::<CompletionStreamDelta>();
        IncomingRequest {
            prompt: prompt.to_string(),
            max_tokens: 1,
            sampling: SamplingParams::default(),
            stop: None,
            priority: RequestPriority::Normal,
            session_id: Some(SessionId::new(format!("req-{idx}"))),
            delta_tx,
        }
    }

    fn make_test_scheduler() -> Option<Scheduler<FakeModel>> {
        let ctx = match DeviceContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Skipping CUDA admission gate test: {err}");
                return None;
            }
        };

        let model = FakeModel { ctx: ctx.clone() };
        let states = (0..16)
            .map(|_| FakeState::new(&ctx))
            .collect::<Result<Vec<_>>>()
            .expect("fake states");
        let paged_kv_pool = PagedKVPool::with_format(&ctx, 1, 1, 1, 16, 4_096, KVFormat::BF16)
            .expect("small paged pool");
        let page_lifecycle =
            crate::kv_tier::coordinator::PageLifecycle::new(paged_kv_pool.max_total_pages);
        let (_request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_coordinator, coordinator_handle, coordinator_events) =
            crate::kv_tier::Coordinator::new(16);
        let config = SchedulerConfig::runtime_defaults(16);

        Some(Scheduler {
            config: config.clone(),
            metrics: ServerMetrics::new("test"),
            model,
            tokenizer: build_test_tokenizer(),
            model_fingerprint: blake3::hash(b"test-model").as_bytes().to_vec(),
            states,
            slot_materialized_prompt_lens: vec![0; config.max_slots],
            prefix_cache: crate::prefix_cache::RadixCache::with_soft_pin_keepalive(
                PREFIX_CACHE_BLOCK_SIZE,
                config.prefix_cache_keepalive_ticks,
            ),
            disk_store: Arc::new(crate::kv_tier::transport::DiskStore::new(
                std::env::temp_dir().join("infer-kv-admission-test"),
            )),
            block_to_pages: HashMap::new(),
            block_owner_slots: HashMap::new(),
            slot_owned_blocks: vec![Vec::new(); config.max_slots],
            coordinator_handle,
            coordinator_events,
            coordinator_thread: None,
            stage_waiting: HashMap::new(),
            coordinator_unavailable: true,
            page_lifecycle,
            request_rx,
            waiting_count: Arc::new(AtomicUsize::new(0)),
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(42),
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
            pending_mixed_prefill_idx: None,
        })
    }

    #[test]
    fn best_reusable_slot_prefers_deepest_free_owned_block() {
        let matched_blocks = vec![BlockId(10), BlockId(20), BlockId(30)];
        let free_slots = vec![1, 2];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 0);
        owners.insert(BlockId(20), 1);
        owners.insert(BlockId(30), 2);

        let reusable = best_reusable_slot_for_radix_hit(
            &matched_blocks,
            &free_slots,
            &owners,
            &[0, 32, 48],
            16,
        );
        assert_eq!(reusable, Some((2, 48, 48)));
    }

    #[test]
    fn best_reusable_slot_skips_busy_or_stale_slots() {
        let matched_blocks = vec![BlockId(10), BlockId(20)];
        let free_slots = vec![1];
        let mut owners = HashMap::new();
        owners.insert(BlockId(10), 1);
        owners.insert(BlockId(20), 0);

        let reusable =
            best_reusable_slot_for_radix_hit(&matched_blocks, &free_slots, &owners, &[0, 8], 16);
        assert_eq!(reusable, None);
    }

    #[test]
    fn assign_slots_gates_admission_on_available_pool_pages() {
        let logger = test_logger();
        logger.clear();

        let Some(mut scheduler) = make_test_scheduler() else {
            return;
        };

        assert_eq!(scheduler.paged_kv_pool.max_total_pages, 64);
        assert_eq!(scheduler.paged_kv_pool.page_size, 16);
        assert_eq!(scheduler.admission_budget_pages(), 32);

        let prompt = repeated_prompt(128);
        // Admission cost = uncached prompt tokens + max_tokens budget
        // (128 prompt + 1 max_tokens = 129 tokens = 9 pages @ page_size=16).
        let per_request_pages = (128_usize + 1).div_ceil(scheduler.paged_kv_pool.page_size);
        let expected_prefilling = scheduler.admission_budget_pages() / per_request_pages;

        for idx in 0..16 {
            scheduler.waiting.push_back(test_request(idx, &prompt));
        }

        scheduler.assign_slots();

        // Gate is an admission-time check. Admitted requests land in `active`
        // with `Phase::New`; step_new later flips them to `Phase::Prefilling`.
        // So assert on `active.len()` and `waiting.len()`, not on phase.
        assert_eq!(scheduler.active.len(), expected_prefilling);
        assert_eq!(scheduler.waiting.len(), 16 - expected_prefilling);
        for req in &scheduler.active {
            assert!(
                matches!(req.phase, Phase::New),
                "admitted req {} should be Phase::New before step_new",
                req.id
            );
        }
        // First request that didn't fit is named req-{expected_prefilling}
        // (0-indexed). We infer it from `expected_prefilling` so the test stays
        // valid if the gate's per-page budget tuning changes again.
        let first_held = format!("req-{expected_prefilling}");
        assert_eq!(
            scheduler
                .waiting
                .front()
                .and_then(|req| req.session_id.as_ref())
                .map(SessionId::as_str),
            Some(first_held.as_str())
        );

        let logs = logger.snapshot();
        assert!(
            logs.iter().any(|line| line.contains("held for pool")),
            "expected a pool-hold admission log, got {logs:?}"
        );
        assert!(
            logs.iter()
                .all(|line| !line.contains("pool alloc for paged prefill failed")),
            "unexpected paged prefill alloc failure log: {logs:?}"
        );
    }
}
