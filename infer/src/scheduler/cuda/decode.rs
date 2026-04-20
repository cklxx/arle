use super::{
    FinishReason, GenerationState, IncomingRequest, ModelForward, Phase, RequestPriority,
    Scheduler, error, info, warn,
};
use crate::model::PrefillSection;
use crate::model::kv_cache::KVFormat;
use crate::model::qwen3::{
    MIXED_PREFILL_CAP as MODEL_MIXED_PREFILL_CAP, cuda_graph_mixed_num_tokens,
};
use crate::scheduler::cuda::core::{
    PendingDecode, PendingPrefillChunk, PlannedDecodeBatch, PlannedDecodeLaunch,
    PlannedMixedDecodeBatch, PlannedMixedPrefillChunk,
};

/// Compile-time ceiling for mixed-path prefill tokens.
///
/// The runtime scheduler budget comes from `SchedulerConfig.max_prefill_tokens`;
/// this constant only guards the model/kernel allocation limit.
const MIXED_PREFILL_TOKENS_ALLOC_CAP: usize = 64;
const _: () = assert!(MIXED_PREFILL_TOKENS_ALLOC_CAP == MODEL_MIXED_PREFILL_CAP);
const MIXED_PREFILL_TOKEN_ALIGN: usize = 16;

/// Per-req chunk plan built before allocation commits. Local to
/// `step_decode_launch_mixed`; lifted above the function body so the
/// struct doesn't trigger clippy::items_after_statements.
struct CandidatePlan {
    req_idx: usize,
    slot_idx: usize,
    start_pos: usize,
    total: usize,
    tokens: Vec<u32>,
}

fn round_up_to_bucket(nt: usize) -> Option<usize> {
    cuda_graph_mixed_num_tokens()
        .iter()
        .copied()
        .find(|&bucket| bucket >= nt)
}

fn align_mixed_prefill_chunk_len(requested: usize, available: usize) -> usize {
    if available == 0 {
        return 0;
    }
    if available < MIXED_PREFILL_TOKEN_ALIGN {
        return available;
    }
    let target = requested.max(MIXED_PREFILL_TOKEN_ALIGN).min(available);
    let aligned = target - (target % MIXED_PREFILL_TOKEN_ALIGN);
    if aligned == 0 {
        MIXED_PREFILL_TOKEN_ALIGN.min(available)
    } else {
        aligned
    }
}

impl<M: ModelForward> Scheduler<M> {
    fn finish_closed_decode_requests(&mut self) {
        for i in 0..self.active.len() {
            if matches!(self.active[i].phase, Phase::Decoding)
                && self.active[i].delta_tx.is_closed()
            {
                self.active[i].phase = Phase::Finished;
            }
        }
    }

    fn collect_decode_batch_inputs(&mut self) -> Option<(Vec<usize>, Vec<u32>)> {
        let decode_indices: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();
        if decode_indices.is_empty() {
            return None;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        for &i in &decode_indices {
            if let Some(&tok) = self.active[i].generated_tokens.last() {
                token_ids.push(tok);
                valid_decode_indices.push(i);
            } else {
                error!(
                    "Request {}: Decoding state with no generated tokens - dropping",
                    self.active[i].id
                );
                self.active[i].phase = Phase::Finished;
            }
        }
        if valid_decode_indices.is_empty() {
            None
        } else {
            Some((valid_decode_indices, token_ids))
        }
    }

    fn preempt_decode_victim(
        &mut self,
        decode_indices: &mut Vec<usize>,
        token_ids: &mut Vec<u32>,
        reason: &str,
    ) -> bool {
        let Some(victim_pos) = decode_indices
            .iter()
            .enumerate()
            .max_by_key(|(_, i)| {
                let r = &self.active[**i];
                (
                    r.generated_tokens.len() as i64,
                    -(r.prompt_tokens.len() as i64),
                )
            })
            .map(|(pos, _)| pos)
        else {
            return false;
        };
        let victim_idx = decode_indices[victim_pos];
        let victim = &mut self.active[victim_idx];
        let victim_slot = victim.slot_idx;
        warn!(
            "Request {}: preempting (recompute) {} — {} generated tokens, pool free={}",
            victim.id,
            reason,
            victim.generated_tokens.len(),
            self.paged_kv_pool.free_count()
        );

        self.paged_kv_pool.free_slot(victim_slot);
        if let Err(e) = self.states[victim_slot].reset() {
            error!(
                "Request {}: slot reset after preempt failed: {}",
                victim.id, e
            );
        }
        self.slot_materialized_prompt_lens[victim_slot] = 0;

        let requeue = IncomingRequest {
            prompt: std::mem::take(&mut victim.prompt),
            max_tokens: victim.max_tokens,
            sampling: victim.sampling.clone(),
            stop: victim.stop.take(),
            priority: RequestPriority::Normal,
            session_id: victim.session_id.clone(),
            delta_tx: victim.delta_tx.clone(),
        };
        victim.phase = Phase::Finished;
        let _ = victim;
        self.clear_slot_prefix_ownership(victim_slot);

        decode_indices.remove(victim_pos);
        token_ids.remove(victim_pos);
        self.waiting.push_front(requeue);
        true
    }

    fn build_plain_decode_plan(&mut self) -> Option<PlannedDecodeBatch> {
        self.finish_closed_decode_requests();
        let (mut decode_indices, mut token_ids) = self.collect_decode_batch_inputs()?;

        while !decode_indices.is_empty() {
            let decode_slot_indices: Vec<usize> = decode_indices
                .iter()
                .map(|&i| self.active[i].slot_idx)
                .collect();
            let decode_appends: Vec<(usize, usize)> = decode_slot_indices
                .iter()
                .map(|&slot| (slot, 1usize))
                .collect();
            let (required_tokens, required_pages) = self.pool_append_requirements(&decode_appends);
            while self.paged_kv_pool.is_active()
                && (self.paged_kv_pool.free_count() < required_tokens
                    || self.paged_kv_pool.free_page_count() < required_pages)
                && decode_indices.len() > 1
            {
                if !self.preempt_decode_victim(
                    &mut decode_indices,
                    &mut token_ids,
                    "for decode batch",
                ) {
                    break;
                }
            }
            if decode_indices.is_empty() {
                return None;
            }

            let slot_indices = decode_slot_indices;
            match self.plan_pool_capacity(required_tokens, required_pages, &slot_indices) {
                Ok(pool_plan) => {
                    let sampling_params_greedy = decode_indices
                        .iter()
                        .map(|&i| {
                            let p = &self.active[i].sampling;
                            p.is_greedy() && !p.has_penalties()
                        })
                        .collect();
                    return Some(PlannedDecodeBatch {
                        decode_indices,
                        token_ids,
                        slot_indices,
                        sampling_params_greedy,
                        pool_plan,
                    });
                }
                Err(e) if decode_indices.len() > 1 => {
                    warn!(
                        "Decode plan: trimming one decode after pool planning failed: {}",
                        e
                    );
                    self.preempt_decode_victim(
                        &mut decode_indices,
                        &mut token_ids,
                        "for decode plan",
                    );
                }
                Err(e) => {
                    warn!("Decode plan deferred this tick: {}", e);
                    return None;
                }
            }
        }
        None
    }

    fn pad_mixed_candidates_to_bucket(&self, candidates: &mut [CandidatePlan]) {
        let current_total: usize = candidates.iter().map(|cand| cand.tokens.len()).sum();
        let Some(target_total) = round_up_to_bucket(current_total) else {
            return;
        };
        if target_total <= current_total {
            return;
        }

        let mut remaining = target_total - current_total;
        for cand in candidates.iter_mut().rev() {
            let Phase::Prefilling {
                effective_tokens, ..
            } = &self.active[cand.req_idx].phase
            else {
                continue;
            };
            let current_end = cand.start_pos + cand.tokens.len();
            if current_end >= effective_tokens.len() {
                continue;
            }
            let extend_by = remaining.min(effective_tokens.len() - current_end);
            cand.tokens
                .extend_from_slice(&effective_tokens[current_end..current_end + extend_by]);
            remaining -= extend_by;
            if remaining == 0 {
                break;
            }
        }
    }

    fn mixed_prefill_budget_tokens(&self) -> usize {
        self.config
            .max_prefill_tokens
            .min(MIXED_PREFILL_TOKENS_ALLOC_CAP)
    }

    fn build_mixed_prefill_candidates(&self, prefill_indices: &[usize]) -> Vec<CandidatePlan> {
        let mut candidates = Vec::with_capacity(prefill_indices.len());
        let mut remaining_budget = self.mixed_prefill_budget_tokens();
        let eligible: Vec<(usize, usize, usize, usize)> = prefill_indices
            .iter()
            .copied()
            .filter_map(|req_idx| {
                let slot_idx = self.active[req_idx].slot_idx;
                match &self.active[req_idx].phase {
                    Phase::Prefilling {
                        effective_tokens,
                        progress,
                    } if !self.active[req_idx].delta_tx.is_closed() => {
                        let pool_seq = self.paged_kv_pool.seq_len(slot_idx);
                        if *progress >= effective_tokens.len() || pool_seq != *progress {
                            return None;
                        }
                        Some((req_idx, slot_idx, *progress, effective_tokens.len()))
                    }
                    _ => None,
                }
            })
            .take(self.config.prefill_max_requests)
            .collect();

        for (ordinal, &(req_idx, slot_idx, progress, total)) in eligible.iter().enumerate() {
            if remaining_budget == 0 {
                break;
            }
            let remaining = total.saturating_sub(progress);
            if remaining == 0 {
                continue;
            }

            let candidates_left = eligible.len().saturating_sub(ordinal).max(1);
            let fair_share = remaining_budget / candidates_left;
            let chunk_len =
                align_mixed_prefill_chunk_len(fair_share.max(1), remaining.min(remaining_budget));
            if chunk_len == 0 {
                continue;
            }

            let tokens = match &self.active[req_idx].phase {
                Phase::Prefilling {
                    effective_tokens, ..
                } => effective_tokens[progress..progress + chunk_len].to_vec(),
                _ => continue,
            };
            if tokens.is_empty() {
                continue;
            }

            remaining_budget = remaining_budget.saturating_sub(tokens.len());
            candidates.push(CandidatePlan {
                req_idx,
                slot_idx,
                start_pos: progress,
                total,
                tokens,
            });
        }
        candidates
    }

    pub(super) fn step_decode_launch_mixed(&mut self, prefill_indices: &[usize]) {
        let plan = self.build_mixed_decode_plan(prefill_indices);
        match plan {
            Some(PlannedDecodeLaunch::Plain(plan)) => self.execute_plain_decode_launch(plan),
            Some(PlannedDecodeLaunch::Mixed(plan)) => self.execute_mixed_decode_launch(plan),
            None => {}
        }
    }

    fn build_mixed_decode_plan(
        &mut self,
        prefill_indices: &[usize],
    ) -> Option<PlannedDecodeLaunch> {
        self.finish_closed_decode_requests();
        let Some((decode_indices, token_ids)) = self.collect_decode_batch_inputs() else {
            return None;
        };

        // Zero-decode edge (spec: "mixed path always requires B ≥ 1"): fall
        // through to the regular decode/prefill path. Same for non-supporting
        // backends.
        if !self.config.enable_mixed_chunk
            || !self.model.supports_mixed_batch()
            || self.paged_kv_pool.format != KVFormat::BF16
            || prefill_indices.is_empty()
        {
            return self
                .build_plain_decode_plan()
                .map(PlannedDecodeLaunch::Plain);
        }

        // Decode-first mixed scheduling: decode rows are always planned first,
        // and prefills only consume the residual mixed-token budget for this tick.
        let mut candidates = self.build_mixed_prefill_candidates(prefill_indices);
        if candidates.is_empty() {
            return self
                .build_plain_decode_plan()
                .map(PlannedDecodeLaunch::Plain);
        }
        // Pad toward the canonical mixed graph buckets using real prompt
        // tokens. If a request cannot absorb the extra rows this tick we keep
        // the exact shape and let the model cache it separately.
        self.pad_mixed_candidates_to_bucket(&mut candidates);

        loop {
            let decode_slot_indices: Vec<usize> = decode_indices
                .iter()
                .map(|&i| self.active[i].slot_idx)
                .collect();
            let mut mixed_appends: Vec<(usize, usize)> = decode_slot_indices
                .iter()
                .map(|&slot| (slot, 1usize))
                .collect();
            mixed_appends.extend(
                candidates
                    .iter()
                    .map(|cand| (cand.slot_idx, cand.tokens.len())),
            );
            let (required_tokens, required_pages) = self.pool_append_requirements(&mixed_appends);
            if candidates.is_empty() {
                return self
                    .build_plain_decode_plan()
                    .map(PlannedDecodeLaunch::Plain);
            }

            let mut protected_slots = decode_slot_indices.clone();
            protected_slots.extend(candidates.iter().map(|cand| cand.slot_idx));
            match self.plan_pool_capacity_without_retract(
                required_tokens,
                required_pages,
                &protected_slots,
            ) {
                Ok(pool_plan) => {
                    let sampling_params_greedy = decode_indices
                        .iter()
                        .map(|&i| {
                            let p = &self.active[i].sampling;
                            p.is_greedy() && !p.has_penalties()
                        })
                        .collect();
                    let prefills = candidates
                        .into_iter()
                        .map(|cand| PlannedMixedPrefillChunk {
                            req_idx: cand.req_idx,
                            slot_idx: cand.slot_idx,
                            start_pos: cand.start_pos,
                            completes: cand.start_pos + cand.tokens.len() >= cand.total,
                            tokens: cand.tokens,
                        })
                        .collect();
                    return Some(PlannedDecodeLaunch::Mixed(PlannedMixedDecodeBatch {
                        decode: PlannedDecodeBatch {
                            decode_indices,
                            token_ids,
                            slot_indices: decode_slot_indices,
                            sampling_params_greedy,
                            pool_plan,
                        },
                        prefill_chunks: prefills,
                    }));
                }
                Err(e) if candidates.len() > 1 => {
                    let dropped = candidates.pop().expect("checked len > 1");
                    warn!(
                        "Mixed decode plan: dropping req {} from this tick after non-retract pool planning failed: {}",
                        self.active[dropped.req_idx].id, e
                    );
                }
                Err(e) => {
                    warn!(
                        "Mixed decode plan falling back to plain decode after non-retract pool planning failed: {}",
                        e
                    );
                    return self
                        .build_plain_decode_plan()
                        .map(PlannedDecodeLaunch::Plain);
                }
            }
        }
    }

    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_launch(&mut self) {
        let Some(plan) = self.build_plain_decode_plan() else {
            return;
        };
        self.execute_plain_decode_launch(plan);
    }

    fn execute_plain_decode_launch(&mut self, plan: PlannedDecodeBatch) {
        let PlannedDecodeBatch {
            decode_indices,
            token_ids,
            slot_indices,
            sampling_params_greedy,
            pool_plan,
        } = plan;
        let mut allocated_decode_slots = Vec::with_capacity(slot_indices.len());
        for &slot in &slot_indices {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                for &allocated_slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(allocated_slot, 1);
                }
                error!(
                    "Decode plan allocation drifted after planning (slot {}, need_tokens={}, need_pages={}): {}",
                    slot, pool_plan.required_tokens, pool_plan.required_pages, e
                );
                return;
            }
            allocated_decode_slots.push(slot);
        }
        let all_greedy = sampling_params_greedy.iter().all(|&g| g);

        // Lazy-init decode context on first batched decode.
        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &slot in &allocated_decode_slots {
                        self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                    }
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
        }
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        // Pre-decode: scheduler handles H2D, metadata, and FlashInfer plan
        // via DecodeContextOps. This decouples scheduler-level work from
        // model-level neural computation.
        {
            use crate::model::DecodeContextOps;
            let ctx = self.model.device_context();
            decode_ctx.set_batch_size(token_ids.len());
            if let Err(e) = decode_ctx.upload_token_ids(ctx, &token_ids) {
                error!("Pre-decode upload_token_ids failed: {}", e);
                for &slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                }
                for &i in &decode_indices {
                    self.active[i].phase = Phase::Finished;
                }
                return;
            }
            match decode_ctx.update_metadata(ctx, &self.paged_kv_pool, &slot_indices) {
                Ok(reallocated) => {
                    if reallocated {
                        decode_ctx.invalidate_graph_cache(token_ids.len());
                    }
                }
                Err(e) => {
                    error!("Pre-decode update_metadata failed: {}", e);
                    for &slot in &allocated_decode_slots {
                        self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                    }
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
            if let Err(e) = decode_ctx.plan_attention(
                ctx,
                token_ids.len(),
                self.model.num_q_heads(),
                self.model.num_kv_heads(),
                self.paged_kv_pool.page_size,
                self.model.head_dim(),
                self.paged_kv_pool.format,
            ) {
                error!("Pre-decode plan_attention failed: {}", e);
                for &slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                }
                for &i in &decode_indices {
                    self.active[i].phase = Phase::Finished;
                }
                return;
            }
        }

        // Plain decode-only launch. Mixed decode+prefill has its own
        // planning/execution path in `execute_mixed_decode_launch()`.
        let forward_result = self.model.forward_decode_batch(
            &token_ids,
            &mut self.states,
            &slot_indices,
            Some(&mut self.paged_kv_pool),
            decode_ctx,
            all_greedy,
        );

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &i in &decode_indices {
                self.active[i].phase = Phase::Finished;
            }
            return;
        }

        let greedy_launched = if all_greedy {
            match self
                .model
                .sample_batch_greedy_launch(&slot_indices, decode_ctx)
            {
                Ok(true) => true,
                Ok(false) => {
                    if let Err(e) = self.model.prepare_batch_sampling_fallback(
                        &mut self.states,
                        &slot_indices,
                        decode_ctx,
                    ) {
                        error!("Preparing batched sampling fallback failed: {}", e);
                        for &req_idx in &decode_indices {
                            self.active[req_idx].phase = Phase::Finished;
                        }
                        return;
                    }
                    false
                }
                Err(e) => {
                    error!("Batched greedy sampling launch failed: {}", e);
                    for &req_idx in &decode_indices {
                        self.active[req_idx].phase = Phase::Finished;
                    }
                    return;
                }
            }
        } else {
            false
        };

        self.pending_decode = Some(PendingDecode {
            decode_indices,
            slot_indices,
            greedy_launched,
            sampling_params_greedy,
            mixed_prefill_chunks: Vec::new(),
        });
    }

    fn execute_mixed_decode_launch(&mut self, plan: PlannedMixedDecodeBatch) {
        let PlannedMixedDecodeBatch {
            decode,
            prefill_chunks,
        } = plan;
        let PlannedDecodeBatch {
            decode_indices,
            token_ids,
            slot_indices,
            sampling_params_greedy,
            pool_plan,
        } = decode;
        let all_greedy = sampling_params_greedy.iter().all(|&g| g);

        let mut allocated_decode_slots = Vec::with_capacity(slot_indices.len());
        for &slot in &slot_indices {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(slot, 1) {
                for &allocated_slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(allocated_slot, 1);
                }
                error!(
                    "Mixed decode plan allocation drifted after planning (decode slot {}, need_tokens={}, need_pages={}): {}",
                    slot, pool_plan.required_tokens, pool_plan.required_pages, e
                );
                return;
            }
            allocated_decode_slots.push(slot);
        }

        let mut allocated_prefills: Vec<(usize, usize)> = Vec::with_capacity(prefill_chunks.len());
        for chunk in &prefill_chunks {
            if let Err(e) = self
                .paged_kv_pool
                .alloc_tokens(chunk.slot_idx, chunk.tokens.len())
            {
                for &(slot, count) in allocated_prefills.iter().rev() {
                    self.paged_kv_pool.free_tokens_from_tail(slot, count);
                }
                for &slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                }
                error!(
                    "Mixed decode prefill allocation drifted after planning (slot {}, len={}, need_tokens={}, need_pages={}): {}",
                    chunk.slot_idx,
                    chunk.tokens.len(),
                    pool_plan.required_tokens,
                    pool_plan.required_pages,
                    e
                );
                return;
            }
            allocated_prefills.push((chunk.slot_idx, chunk.tokens.len()));
        }

        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &(slot, count) in allocated_prefills.iter().rev() {
                        self.paged_kv_pool.free_tokens_from_tail(slot, count);
                    }
                    for &slot in &allocated_decode_slots {
                        self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                    }
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
                    }
                    return;
                }
            }
        }
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        let prefill_sections: Vec<PrefillSection<'_>> = prefill_chunks
            .iter()
            .map(|chunk| PrefillSection {
                slot_idx: chunk.slot_idx,
                start_pos: chunk.start_pos,
                tokens: chunk.tokens.as_slice(),
            })
            .collect();

        let forward_result = self.model.forward_mixed_batch(
            &token_ids,
            &prefill_sections,
            &mut self.states,
            &slot_indices,
            Some(&mut self.paged_kv_pool),
            decode_ctx,
        );
        drop(prefill_sections);

        let mixed_ok = match forward_result {
            Ok(true) => {
                info!(
                    "Mixed batch: B={} decode + N={} prefill (Σc={})",
                    token_ids.len(),
                    prefill_chunks.len(),
                    prefill_chunks
                        .iter()
                        .map(|chunk| chunk.tokens.len())
                        .sum::<usize>(),
                );
                true
            }
            Ok(false) => {
                info!("Mixed batch: fallback (Ok(false))");
                for &(slot, count) in allocated_prefills.iter().rev() {
                    self.paged_kv_pool.free_tokens_from_tail(slot, count);
                }
                for &slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                }
                self.execute_plain_decode_launch(PlannedDecodeBatch {
                    decode_indices,
                    token_ids,
                    slot_indices,
                    sampling_params_greedy,
                    pool_plan,
                });
                return;
            }
            Err(e) => {
                error!("Mixed batched decode failed, falling back: {}", e);
                for &(slot, count) in allocated_prefills.iter().rev() {
                    self.paged_kv_pool.free_tokens_from_tail(slot, count);
                }
                for &slot in &allocated_decode_slots {
                    self.paged_kv_pool.free_tokens_from_tail(slot, 1);
                }
                self.execute_plain_decode_launch(PlannedDecodeBatch {
                    decode_indices,
                    token_ids,
                    slot_indices,
                    sampling_params_greedy,
                    pool_plan,
                });
                return;
            }
        };
        debug_assert!(mixed_ok);

        let greedy_launched = if all_greedy {
            match self
                .model
                .sample_batch_greedy_launch(&slot_indices, decode_ctx)
            {
                Ok(true) => true,
                Ok(false) => {
                    if let Err(e) = self.model.prepare_batch_sampling_fallback(
                        &mut self.states,
                        &slot_indices,
                        decode_ctx,
                    ) {
                        error!("Preparing batched sampling fallback failed: {}", e);
                        for &req_idx in &decode_indices {
                            self.active[req_idx].phase = Phase::Finished;
                        }
                        return;
                    }
                    false
                }
                Err(e) => {
                    error!("Batched greedy sampling launch failed: {}", e);
                    for &req_idx in &decode_indices {
                        self.active[req_idx].phase = Phase::Finished;
                    }
                    return;
                }
            }
        } else {
            false
        };

        let b = decode_indices.len();
        let mut running = b;
        let mut mixed_prefill_chunks: Vec<PendingPrefillChunk> =
            Vec::with_capacity(prefill_chunks.len());
        let mut fused_req_idxs: Vec<usize> = Vec::with_capacity(prefill_chunks.len());
        for chunk in &prefill_chunks {
            let c_i = chunk.tokens.len();
            running += c_i;
            let logit_row = running - 1;
            let new_progress = chunk.start_pos + c_i;
            if let Phase::Prefilling { progress, .. } = &mut self.active[chunk.req_idx].phase {
                *progress = new_progress;
            }
            mixed_prefill_chunks.push(PendingPrefillChunk {
                req_idx: chunk.req_idx,
                token_count: c_i,
                completes: chunk.completes,
                logit_row,
            });
            fused_req_idxs.push(chunk.req_idx);
        }

        self.pending_mixed_prefill_idxs = fused_req_idxs;
        self.pending_decode = Some(PendingDecode {
            decode_indices,
            slot_indices,
            greedy_launched,
            sampling_params_greedy,
            mixed_prefill_chunks,
        });
    }

    pub(super) fn step_decode_readback(&mut self) {
        let Some(pending) = self.pending_decode.take() else {
            return;
        };
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        let sampled_result = if pending.greedy_launched {
            // Argmax kernel was launched — sync + readback
            match self
                .model
                .sample_batch_greedy_readback(&pending.slot_indices, decode_ctx)
            {
                Ok(Some(tokens)) => Ok(tokens),
                Ok(None) => {
                    let params: Vec<&crate::sampler::SamplingParams> = pending
                        .decode_indices
                        .iter()
                        .zip(&pending.sampling_params_greedy)
                        .map(|(&i, _)| &self.active[i].sampling)
                        .collect();
                    self.model.select_tokens_batch(
                        &mut self.states,
                        &pending.slot_indices,
                        &params,
                        &mut self.rng,
                    )
                }
                Err(e) => Err(e),
            }
        } else {
            let params: Vec<&crate::sampler::SamplingParams> = pending
                .decode_indices
                .iter()
                .zip(&pending.sampling_params_greedy)
                .map(|(&i, _)| &self.active[i].sampling)
                .collect();
            self.model.select_tokens_batch(
                &mut self.states,
                &pending.slot_indices,
                &params,
                &mut self.rng,
            )
        };

        match sampled_result {
            Ok(sampled_tokens) => {
                let logprobs_host: Option<&[f32]> = if pending.greedy_launched {
                    Some(crate::model::DecodeContextOps::logprobs_host(&*decode_ctx))
                } else {
                    None
                };
                for (j, &req_idx) in pending.decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut self.active[req_idx];
                    req.latest_logprob = logprobs_host.and_then(|lps| lps.get(j).copied());
                    if !req.sampling.ignore_eos && self.model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, &self.tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, &self.tokenizer);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &req_idx in &pending.decode_indices {
                    self.active[req_idx].phase = Phase::Finished;
                }
            }
        }

        // Per-prefill first-token sampling for every chunk that completed.
        // Each completing chunk's logits landed in its own state's
        // `prefill_logits` DeviceVec (the model extracted
        // `mixed.logits[chunk.logit_row]` into it), so the usual
        // `select_token` path samples the right logits by request.
        //
        // `logit_row` is stored on `PendingPrefillChunk` for future use
        // (e.g. batched logit scatter); the per-slot DeviceVec already
        // carries the data the sampler needs for this tick.
        for chunk in pending.mixed_prefill_chunks {
            if !chunk.completes {
                continue;
            }
            let prefill_idx = chunk.req_idx;
            // Guard: if the request was finished mid-tick (e.g. stream
            // closed during the forward), skip sampling.
            if matches!(self.active[prefill_idx].phase, Phase::Finished) {
                continue;
            }
            let slot_idx = self.active[prefill_idx].slot_idx;
            if let Err(e) = self.states[slot_idx].save_prefix_snapshot() {
                warn!(
                    "Request {}: save prefix snapshot failed: {} (prefix cache disabled for this slot)",
                    self.active[prefill_idx].id, e
                );
            } else {
                self.active[prefill_idx].mark_prompt_cacheable();
            }

            let sampling = self.active[prefill_idx].sampling.clone();
            match self
                .model
                .select_token(&mut self.states[slot_idx], &sampling, &mut self.rng)
            {
                Ok(token) => {
                    if !self.active[prefill_idx].sampling.ignore_eos
                        && self.model.is_stop_token(token)
                    {
                        self.active[prefill_idx].finish(FinishReason::Stop, &self.tokenizer);
                        continue;
                    }
                    self.active[prefill_idx].generated_tokens.push(token);
                    self.active[prefill_idx].emit_delta(&self.tokenizer);

                    if matches!(self.active[prefill_idx].phase, Phase::Finished) {
                        continue;
                    }
                    if self.active[prefill_idx].generated_tokens.len()
                        >= self.active[prefill_idx].max_tokens
                    {
                        self.active[prefill_idx].finish(FinishReason::Length, &self.tokenizer);
                    } else {
                        if self.active[prefill_idx].first_token_at.is_none() {
                            self.active[prefill_idx].first_token_at =
                                Some(std::time::Instant::now());
                        }
                        self.active[prefill_idx].phase = Phase::Decoding;
                    }
                }
                Err(e) => {
                    error!(
                        "Request {}: select_token failed after mixed prefill: {}",
                        self.active[prefill_idx].id, e
                    );
                    self.active[prefill_idx].phase = Phase::Finished;
                }
            }
        }
    }
}
