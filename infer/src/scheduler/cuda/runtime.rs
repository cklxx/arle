use super::{
    ActiveRequest, ModelForward, Ordering, Phase, STATS_LOG_INTERVAL, Scheduler, error, info, warn,
};
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
                Ok(
                    crate::kv_tier::CoordinatorEvent::CommandQueued(_)
                    | crate::kv_tier::CoordinatorEvent::StagingQueued { .. }
                    | crate::kv_tier::CoordinatorEvent::SpillQueued { .. }
                    | crate::kv_tier::CoordinatorEvent::RehydrateQueued { .. }
                    | crate::kv_tier::CoordinatorEvent::DemoteQueued { .. },
                ) => {}
                Ok(crate::kv_tier::CoordinatorEvent::DemoteCompleted {
                    ticket,
                    block,
                    bytes,
                }) => {
                    // Gap #5 C2 shape freeze — v1 coordinator is a pure
                    // telemetry sink (scheduler-owns-copy per
                    // `docs/plans/gap5-c2-byte-path-architecture.md`).
                    // Scheduler-side demote hook that wires this into
                    // `t1_demotes_total` + `t1_bytes_demoted_total`
                    // counters lands in C3.
                    log::debug!(
                        "demote acknowledged: ticket={} block={block:?} bytes={bytes}",
                        ticket.0
                    );
                }
                Ok(crate::kv_tier::CoordinatorEvent::DemoteFailed {
                    ticket,
                    block,
                    reason,
                }) => {
                    log::warn!(
                        "demote failed: ticket={} block={block:?} reason={reason}",
                        ticket.0
                    );
                }
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
                    if pending.is_empty() {
                        error!("Coordinator event channel disconnected");
                    } else {
                        error!(
                            "Coordinator event channel disconnected; cold-requeuing {} staged admissions",
                            pending.len()
                        );
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
                if let Some(req) = self.request_rx.blocking_recv() {
                    self.waiting_count.fetch_sub(1, Ordering::Relaxed);
                    self.waiting.push_back(req);
                } else {
                    info!("Scheduler shutting down: all handles dropped");
                    break;
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
        let mut tick_budget_tokens = self.admission_budget_tokens();
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
            // Admission gate (P1#1): subtract what the scheduler will
            // *actually* reuse. When `best_reusable_slot_for_radix_hit`
            // returns None, reuse is 0 even if `radix_hit_len` matched —
            // step_new re-prefills the whole prompt onto a cold slot.
            // Gating on `radix_hit_len` (the looser global match) under-
            // reserves pool tokens and OOMs on chunked prefill.
            let extend_input_tokens = prompt_tokens.len().saturating_sub(reusable_prefix_len);
            let clipped_max_new = incoming
                .max_tokens
                .min(self.config.admission_clip_max_new_tokens);
            let total_tokens_needed = extend_input_tokens + clipped_max_new;
            if total_tokens_needed >= tick_budget_tokens {
                self.prefix_cache.release(&radix_blocks);
                info!(
                    "Request {} held for pool (need={} tok, budget={} tok, radix_hit={}, reusable_prefix={})",
                    self.next_id,
                    total_tokens_needed,
                    tick_budget_tokens,
                    radix_hit_len,
                    reusable_prefix_len
                );
                self.waiting.push_front(incoming);
                break;
            }
            tick_budget_tokens = tick_budget_tokens.saturating_sub(
                extend_input_tokens + clipped_max_new + self.paged_kv_pool.page_size,
            );
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
                first_step_at: None,
                finished_at: None,
                t_prefill_us: 0,
                t_decode_us: 0,
                t_emit_us: 0,
                t_new_us: 0,
                step_count: 0,
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
                let mut req = self.active.remove(i);
                let gen_tokens = req.generated_tokens.len() as u64;
                // L1: retracted victims are already re-queued onto
                // `waiting` by `retract_longest_decode`; their pool pages
                // are already released; their cached prompt is incomplete
                // (partial prefill) so we must NOT publish it to the
                // prefix cache, and this is NOT a completion for metrics.
                let was_retracted = self.retracted_request_ids.remove(&req.id);
                self.clear_slot_prefix_ownership(req.slot_idx);

                if !was_retracted && let Some(prompt_tokens) = req.cached_prompt_to_publish() {
                    let prompt_vec = prompt_tokens.to_vec();
                    self.slot_materialized_prompt_lens[req.slot_idx] = prompt_vec.len();
                    self.publish_to_prefix_cache(req.slot_idx, &prompt_vec, req.session_id.clone());
                } else {
                    self.slot_materialized_prompt_lens[req.slot_idx] = 0;
                }
                // Pool pages already freed inside retract; `free_slot`
                // here is idempotent but we keep it so the non-retract
                // path (normal completion) still releases.
                self.paged_kv_pool.free_slot(req.slot_idx);

                if was_retracted {
                    if self.last_served >= self.active.len() && !self.active.is_empty() {
                        self.last_served = 0;
                    }
                    continue;
                }

                self.total_completed += 1;
                self.total_generated_tokens += gen_tokens;
                req.finished_at = Some(std::time::Instant::now());
                let e2e_s = req.admitted_at.elapsed().as_secs_f64();
                let ttft_s = req
                    .first_token_at
                    .map_or(e2e_s, |t| t.duration_since(req.admitted_at).as_secs_f64());
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

                if std::env::var("INFER_TRACE").ok().as_deref() == Some("1") {
                    let e2e_us = req.admitted_at.elapsed().as_micros() as u64;
                    let ttft_us = req.first_token_at.map_or(e2e_us, |t| {
                        t.duration_since(req.admitted_at).as_micros() as u64
                    });
                    let queue_us = req
                        .first_step_at
                        .map_or(0, |t| t.duration_since(req.admitted_at).as_micros() as u64);
                    let sum_stages = queue_us
                        + req.t_new_us
                        + req.t_prefill_us
                        + req.t_decode_us
                        + req.t_emit_us;
                    let residual_us = e2e_us.saturating_sub(sum_stages);
                    let gen_count = req.generated_tokens.len().max(1) as u64;
                    let tpot_us = e2e_us.saturating_sub(ttft_us) / gen_count;
                    info!(
                        "TRACE id={} e2e_us={} ttft_us={} tpot_us={} steps={} prompt={} gen={} queue_us={} new_us={} prefill_us={} decode_us={} emit_us={} residual_us={}",
                        req.id,
                        e2e_us,
                        ttft_us,
                        tpot_us,
                        req.step_count,
                        req.prompt_tokens.len(),
                        req.generated_tokens.len(),
                        queue_us,
                        req.t_new_us,
                        req.t_prefill_us,
                        req.t_decode_us,
                        req.t_emit_us,
                        residual_us,
                    );
                }

                info!(
                    "Request {} done: {} tokens (active={}, waiting={})",
                    req.id,
                    gen_tokens,
                    self.active.len(),
                    self.waiting.len()
                );

                if self.total_completed.is_multiple_of(STATS_LOG_INTERVAL) {
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
    use super::{Scheduler, best_reusable_slot_for_radix_hit};
    use crate::backend::cuda::paged_kv::PagedKVPool;
    use crate::kv_tier::coordinator::PageLifecycle;
    use crate::metrics::ServerMetrics;
    use crate::model::kv_cache::{KVCacheDtype, KVFormat};
    use crate::model::{DecodeContextOps, GenerationState, ModelForward};
    use crate::prefix_cache::BlockId;
    use crate::sampler::SamplingParams;
    use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerConfig};
    use crate::tokenizer::Tokenizer;
    use crate::types::SessionId;
    use cuda_kernels::prelude::{DeviceContext, DeviceVec};
    use log::{Level, LevelFilter, Log, Metadata, Record};
    use std::collections::HashMap;
    use std::sync::{Mutex, Once};
    use tokio::sync::mpsc;

    struct TestLogger {
        entries: Mutex<Vec<String>>,
    }

    impl TestLogger {
        const fn new() -> Self {
            Self {
                entries: Mutex::new(Vec::new()),
            }
        }

        fn reset(&self) {
            self.entries.lock().expect("logger mutex poisoned").clear();
        }

        fn snapshot(&self) -> String {
            self.entries
                .lock()
                .expect("logger mutex poisoned")
                .join("\n")
        }
    }

    impl Log for TestLogger {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.level() <= Level::Info
        }

        fn log(&self, record: &Record<'_>) {
            if self.enabled(record.metadata()) {
                self.entries
                    .lock()
                    .expect("logger mutex poisoned")
                    .push(format!("{} {}", record.level(), record.args()));
            }
        }

        fn flush(&self) {}
    }

    static TEST_LOGGER: TestLogger = TestLogger::new();
    static TEST_LOGGER_INIT: Once = Once::new();

    fn init_test_logger() {
        TEST_LOGGER_INIT.call_once(|| {
            log::set_logger(&TEST_LOGGER).expect("test logger should initialize once");
            log::set_max_level(LevelFilter::Info);
        });
        TEST_LOGGER.reset();
    }

    struct FakeDecodeContext;

    impl DecodeContextOps for FakeDecodeContext {
        fn upload_token_ids(
            &mut self,
            _ctx: &DeviceContext,
            _tokens: &[u32],
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn update_metadata(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot_indices: &[usize],
        ) -> anyhow::Result<bool> {
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
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn set_batch_size(&mut self, _bs: usize) {}

        fn invalidate_graph_cache(&mut self, _batch_size: usize) {}
    }

    struct FakeState {
        logits: DeviceVec,
    }

    impl GenerationState for FakeState {
        fn logits(&self) -> &DeviceVec {
            &self.logits
        }

        fn reset(&mut self) -> anyhow::Result<()> {
            Ok(())
        }

        fn truncate_to(&mut self, _len: usize) -> anyhow::Result<()> {
            Ok(())
        }

        fn set_max_seq_len(&mut self, _max_seq: usize) {}

        fn set_kv_dtype(&mut self, _dtype: KVCacheDtype) {}

        fn migrate_kv_to_paged(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot: usize,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn migrate_kv_range_to_paged(
            &mut self,
            _ctx: &DeviceContext,
            _pool: &PagedKVPool,
            _slot: usize,
            _start_pos: usize,
            _token_count: usize,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct FakeModel {
        ctx: DeviceContext,
    }

    impl FakeModel {
        fn new() -> anyhow::Result<Self> {
            Ok(Self {
                ctx: DeviceContext::new()?,
            })
        }
    }

    impl ModelForward for FakeModel {
        type State = FakeState;
        type DecodeContext = FakeDecodeContext;

        fn create_state(&self) -> anyhow::Result<Self::State> {
            Ok(FakeState {
                logits: DeviceVec::zeros(&self.ctx, 1)?,
            })
        }

        fn create_decode_context(
            &self,
            _max_batch_size: usize,
            _pool: &PagedKVPool,
        ) -> anyhow::Result<Self::DecodeContext> {
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

        fn forward_prefill(&self, _tokens: &[u32], _state: &mut Self::State) -> anyhow::Result<()> {
            Ok(())
        }

        fn forward_decode(&self, _token: u32, _state: &mut Self::State) -> anyhow::Result<()> {
            Ok(())
        }

        fn select_token(
            &self,
            _state: &mut Self::State,
            _params: &SamplingParams,
            _rng: &mut rand::rngs::StdRng,
        ) -> anyhow::Result<u32> {
            Ok(0)
        }

        fn is_stop_token(&self, _token_id: u32) -> bool {
            false
        }

        fn device_context(&self) -> &DeviceContext {
            &self.ctx
        }
    }

    fn test_tokenizer() -> Tokenizer {
        let dir = tempfile::tempdir().expect("tempdir should create");
        std::fs::write(
            dir.path().join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"tok":1},"unk_token":"<unk>"}}"#,
        )
        .expect("tokenizer.json should write");
        Tokenizer::from_file(
            dir.path()
                .to_str()
                .expect("temp tokenizer path should be valid UTF-8"),
        )
        .expect("test tokenizer should load")
    }

    fn test_scheduler() -> Scheduler<FakeModel> {
        let config = SchedulerConfig {
            max_slots: 16,
            admission_clip_max_new_tokens: 4096,
            admission_new_token_ratio: 0.7,
            ..SchedulerConfig::runtime_defaults(16)
        };
        let (mut scheduler, _handle) = Scheduler::with_config(
            FakeModel::new().expect("CUDA context should initialize"),
            test_tokenizer(),
            "fake-cuda-test",
            0,
            ServerMetrics::new("fake-cuda-test"),
            config,
            Some(256),
            KVCacheDtype::BF16,
            KVFormat::BF16,
        )
        .expect("scheduler should initialize");

        let replacement_pool = PagedKVPool::with_format(
            scheduler.model.device_context(),
            1,
            1,
            1,
            16,
            64 * super::super::core::PREFIX_CACHE_BLOCK_SIZE * 4,
            KVFormat::BF16,
        )
        .expect("replacement pool should initialize");
        assert_eq!(
            replacement_pool.page_size,
            super::super::core::PREFIX_CACHE_BLOCK_SIZE
        );
        assert_eq!(replacement_pool.max_total_pages, 64);

        scheduler.paged_kv_pool = replacement_pool;
        scheduler.page_lifecycle = PageLifecycle::new(scheduler.paged_kv_pool.max_total_pages);
        scheduler
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
    fn assign_slots_admission_gate_matches_sglang_model() {
        init_test_logger();
        let mut scheduler = test_scheduler();
        TEST_LOGGER.reset();
        let prompt = std::iter::repeat_n("tok", 128)
            .collect::<Vec<_>>()
            .join(" ");

        assert_eq!(scheduler.paged_kv_pool.free_count(), 1024);
        assert_eq!(scheduler.prefix_cache.evictable_token_count(), 0);
        assert_eq!(scheduler.admission_budget_tokens(), 1024);
        assert_eq!(128usize + 32 + scheduler.paged_kv_pool.page_size, 176);

        for i in 0..16 {
            let (delta_tx, _delta_rx) = mpsc::unbounded_channel();
            scheduler.waiting.push_back(IncomingRequest {
                prompt: prompt.clone(),
                max_tokens: 32,
                sampling: SamplingParams::default(),
                stop: None,
                priority: RequestPriority::default(),
                session_id: Some(SessionId::from(format!("req-{i}"))),
                delta_tx,
            });
        }

        scheduler.assign_slots();

        assert_eq!(scheduler.active.len(), 5);
        assert_eq!(scheduler.waiting.len(), 11);
        assert!(
            scheduler
                .waiting
                .front()
                .and_then(|req| req.session_id.as_ref())
                .is_some_and(|session_id| session_id.as_str().ends_with("-5"))
        );

        let logs = TEST_LOGGER.snapshot();
        assert_eq!(
            logs.match_indices("held for pool").count(),
            1,
            "unexpected hold log count: {logs}"
        );
        assert!(
            logs.contains("Request 5 held for pool (need=160 tok, budget=144 tok)"),
            "missing expected hold log: {logs}"
        );
        assert!(
            !logs.contains("pool alloc for paged prefill failed"),
            "unexpected alloc failure log: {logs}"
        );
    }
}
