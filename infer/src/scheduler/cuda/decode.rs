use super::{
    FinishReason, GenerationState, IncomingRequest, ModelForward, Phase, RequestPriority,
    Scheduler, error, info, warn,
};
use crate::model::PrefillSection;
use crate::model::kv_cache::KVFormat;
use crate::scheduler::cuda::core::{PendingDecode, PendingPrefillChunk};

/// Total mixed-prefill tokens per tick, summed across fused prefill reqs.
/// Total mixed-prefill tokens per tick, summed across fused prefill reqs.
/// Held at 64 (= f21d15e baseline's single-req cap) so total qo rows of the
/// fused kernel don't grow; what changes is how those 64 tokens are
/// distributed across the queue. K=2 splits 32 tokens per req → 2 reqs
/// advance per tick instead of 1.
const MIXED_PREFILL_CAP: usize = 64;
/// Maximum number of prefill requests fused per tick. Mirrored in
/// `scheduler/cuda/execution.rs` and `model/qwen3/batch_decode.rs`.
const MIXED_PREFILL_MAX_REQS: usize = 2;

/// Per-req chunk fused into the current mixed tick. Owns the host-side
/// token slice + page-table scratch so the model-facing
/// `PrefillSection<'a>` can borrow from it for the duration of the
/// forward call.
struct PrefillChunk {
    req_idx: usize,
    slot_idx: usize,
    start_pos: usize,
    tokens: Vec<u32>,
    page_table_host: Vec<i32>,
    completes: bool,
    /// Number of logical pool tokens this chunk allocated — used to roll
    /// back via `free_tokens_from_tail` if a later chunk in the same tick
    /// fails to allocate.
    alloc_count: usize,
}

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

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn step_decode_launch_mixed(&mut self, prefill_indices: &[usize]) {
        let decode_indices: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        for i in 0..self.active.len() {
            if matches!(self.active[i].phase, Phase::Decoding)
                && self.active[i].delta_tx.is_closed()
            {
                self.active[i].phase = Phase::Finished;
            }
        }

        // Zero-decode edge (spec: "mixed path always requires B ≥ 1"): fall
        // through to the regular decode/prefill path. Same for non-supporting
        // backends.
        if decode_indices.is_empty()
            || !self.model.supports_mixed_batch()
            || self.paged_kv_pool.format != KVFormat::BF16
            || prefill_indices.is_empty()
        {
            self.step_decode_launch();
            return;
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
        let mut decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }

        // --- Build PrefillChunks ---
        //
        // Per-req budget: max(16, (MIXED_PREFILL_CAP / K)) rounded down to a
        // multiple of 16, clipped to remaining. Only reqs that are actually in
        // `Phase::Prefilling` with remaining work are folded in. If none of
        // the candidate prefill reqs have remaining work, fall through to the
        // plain decode path.
        let k = prefill_indices.len().clamp(1, MIXED_PREFILL_MAX_REQS);
        let base_per_req = (MIXED_PREFILL_CAP / k).max(16);
        let base_per_req = base_per_req - (base_per_req % 16).min(base_per_req); // round down to 16
        let base_per_req = if base_per_req == 0 { 16 } else { base_per_req };

        let mut candidates: Vec<CandidatePlan> = Vec::with_capacity(prefill_indices.len());
        let mut chunks_budget_used: usize = 0;
        for &pi in prefill_indices {
            if chunks_budget_used >= MIXED_PREFILL_CAP {
                break;
            }
            let slot_idx = self.active[pi].slot_idx;
            let (total, progress, effective_len) = match &self.active[pi].phase {
                Phase::Prefilling {
                    effective_tokens,
                    progress,
                } if !self.active[pi].delta_tx.is_closed() => {
                    let pool_seq = self.paged_kv_pool.seq_len(slot_idx);
                    if *progress >= effective_tokens.len() || pool_seq != *progress {
                        continue;
                    }
                    (effective_tokens.len(), *progress, effective_tokens.len())
                }
                _ => continue,
            };
            let remaining = total.saturating_sub(progress);
            if remaining == 0 {
                continue;
            }
            let per_req_cap = base_per_req.min(MIXED_PREFILL_CAP - chunks_budget_used);
            let chunk_count = per_req_cap.min(remaining).max(1);
            let end = progress + chunk_count;
            let tokens = match &self.active[pi].phase {
                Phase::Prefilling {
                    effective_tokens, ..
                } => effective_tokens[progress..end.min(effective_len)].to_vec(),
                _ => continue,
            };
            if tokens.is_empty() {
                continue;
            }
            chunks_budget_used += tokens.len();
            candidates.push(CandidatePlan {
                req_idx: pi,
                slot_idx,
                start_pos: progress,
                total,
                tokens,
            });
            if candidates.len() >= MIXED_PREFILL_MAX_REQS {
                break;
            }
        }
        if candidates.is_empty() {
            self.step_decode_launch();
            return;
        }

        // Preempt long-running decoders if pool can't fit the mixed batch.
        let prefill_token_total: usize = candidates.iter().map(|c| c.tokens.len()).sum();
        while self.paged_kv_pool.is_active()
            && self.paged_kv_pool.free_count() < decode_indices.len() + prefill_token_total
            && decode_indices.len() > 1
        {
            // sglang-parity victim ranking (`ScheduleBatch.retract_decode`):
            // prefer the request with the most OUTPUT tokens AND the shortest
            // INPUT tokens — i.e. the cheapest to re-prefill. Previous
            // single-key ranking only considered output length, so we kept
            // preempting the 4 k-prompt reqs that cost 4 k tokens of re-prefill
            // each cycle under memory pressure.
            let victim_pos = decode_indices
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
                .unwrap();
            let victim_idx = decode_indices[victim_pos];
            let victim = &mut self.active[victim_idx];
            let victim_slot = victim.slot_idx;
            warn!(
                "Request {}: preempting (recompute) for mixed batch — {} generated tokens, pool free={}",
                victim.id,
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
        }

        // Allocate one pool token per remaining decode request.
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &i) in decode_indices.iter().enumerate() {
            // L1 retract regression guard (codex review 2026-04-19): if
            // `alloc_pool_tokens_with_retry` retracted this slot (or a
            // sibling we were about to decode) to free pool pages, its
            // phase has flipped to Finished. Skip it so the decode
            // dispatch doesn't sample a token for a just-retracted slot;
            // cleanup() will reap it next tick.
            if matches!(self.active[i].phase, Phase::Finished) {
                continue;
            }
            let slot = self.active[i].slot_idx;
            if let Err(e) = self.alloc_pool_tokens_with_retry(slot, 1) {
                error!(
                    "Request {}: KV pool exhausted after preemption (slot {}): {} — finishing",
                    self.active[i].id, slot, e
                );
                self.active[i].finish(FinishReason::Length, &self.tokenizer);
                continue;
            }
            if matches!(self.active[i].phase, Phase::Finished) {
                continue;
            }
            alloc_ok_indices.push(i);
            alloc_ok_tokens.push(token_ids[j]);
        }
        let decode_indices = alloc_ok_indices;
        let token_ids = alloc_ok_tokens;
        if decode_indices.is_empty() {
            // Roll back any in-flight prefill allocations (none yet at this
            // point — we haven't alloc'd for prefill — just bail).
            return;
        }

        // --- Allocate each prefill chunk's pool tokens, rolling back on OOM ---
        //
        // Spec: "Per-prefill `alloc_tokens` may OOM after earlier ones succeed
        // → ROLLBACK: `free_tokens_from_tail` on each already-allocated, drop
        // just that req from this tick, log warn, continue."
        let mut chunks: Vec<PrefillChunk> = Vec::with_capacity(candidates.len());
        for cand in candidates {
            match self
                .paged_kv_pool
                .alloc_tokens(cand.slot_idx, cand.tokens.len())
            {
                Ok(_pages) => {
                    let page_table_host: Vec<i32> = self
                        .paged_kv_pool
                        .page_indices(cand.slot_idx)
                        .iter()
                        .map(|&idx| idx as i32)
                        .collect();
                    let completes = cand.start_pos + cand.tokens.len() >= cand.total;
                    chunks.push(PrefillChunk {
                        req_idx: cand.req_idx,
                        slot_idx: cand.slot_idx,
                        start_pos: cand.start_pos,
                        tokens: cand.tokens,
                        page_table_host,
                        completes,
                        alloc_count: 0, // set below once committed
                    });
                    // Stamp alloc_count now that the push succeeded.
                    let last = chunks.last_mut().unwrap();
                    last.alloc_count = last.tokens.len();
                }
                Err(e) => {
                    warn!(
                        "Mixed prefill: dropping req {} this tick (slot {}, alloc_tokens({}) failed: {})",
                        self.active[cand.req_idx].id,
                        cand.slot_idx,
                        cand.tokens.len(),
                        e
                    );
                    // The failing req is simply skipped — its start_pos
                    // advances 0 this tick, scheduler retries next tick.
                    // Already-allocated earlier chunks stay allocated
                    // since they will participate in this tick's forward.
                }
            }
        }
        if chunks.is_empty() {
            // Every prefill candidate failed alloc; fall back to plain decode.
            self.step_decode_launch();
            return;
        }

        let slot_indices: Vec<usize> = decode_indices
            .iter()
            .map(|&i| self.active[i].slot_idx)
            .collect();
        let sampling_params_greedy: Vec<bool> = decode_indices
            .iter()
            .map(|&i| {
                let p = &self.active[i].sampling;
                p.is_greedy() && !p.has_penalties()
            })
            .collect();
        let all_greedy = sampling_params_greedy.iter().all(|&g| g);

        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
                    for &i in &decode_indices {
                        self.active[i].phase = Phase::Finished;
                    }
                    // Roll back prefill allocations made above.
                    for chunk in &chunks {
                        self.paged_kv_pool
                            .free_tokens_from_tail(chunk.slot_idx, chunk.alloc_count);
                    }
                    return;
                }
            }
        }
        let decode_ctx = self.decode_bufs.as_mut().unwrap();

        // Build borrowed PrefillSections for the model API.
        let prefill_sections: Vec<PrefillSection<'_>> = chunks
            .iter()
            .map(|c| PrefillSection {
                slot_idx: c.slot_idx,
                start_pos: c.start_pos,
                tokens: c.tokens.as_slice(),
                page_table_host: c.page_table_host.as_slice(),
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
                    chunks.len(),
                    chunks.iter().map(|c| c.tokens.len()).sum::<usize>(),
                );
                true
            }
            Ok(false) => {
                info!("Mixed batch: fallback (Ok(false))");
                // Roll back prefill allocations so next tick sees a clean pool.
                for chunk in &chunks {
                    self.paged_kv_pool
                        .free_tokens_from_tail(chunk.slot_idx, chunk.alloc_count);
                }
                self.step_decode_launch();
                return;
            }
            Err(e) => {
                error!("Mixed batched decode failed, falling back: {}", e);
                for chunk in &chunks {
                    self.paged_kv_pool
                        .free_tokens_from_tail(chunk.slot_idx, chunk.alloc_count);
                }
                self.step_decode_launch();
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

        // Advance scheduler-side prefill progress and build the PendingPrefillChunk
        // list. `logit_row` matches the model's per-prefill extraction:
        // row = b + Σ_{j≤i} c_j - 1.
        let b = decode_indices.len();
        let mut running = b;
        let mut mixed_prefill_chunks: Vec<PendingPrefillChunk> = Vec::with_capacity(chunks.len());
        let mut fused_req_idxs: Vec<usize> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let c_i = chunk.tokens.len();
            running += c_i;
            let logit_row = running - 1;
            let new_progress = chunk.start_pos + c_i;
            if let Phase::Prefilling { progress, .. } = &mut self.active[chunk.req_idx].phase {
                *progress = new_progress;
            }
            mixed_prefill_chunks.push(PendingPrefillChunk {
                req_idx: chunk.req_idx,
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

    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_launch(&mut self) {
        let decode_indices: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        for i in 0..self.active.len() {
            if matches!(self.active[i].phase, Phase::Decoding)
                && self.active[i].delta_tx.is_closed()
            {
                self.active[i].phase = Phase::Finished;
            }
        }

        if decode_indices.is_empty() {
            return;
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
        let mut decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }
        // Preemption: if pool can't fit all decode requests, preempt the
        // request with the most generated tokens (highest KV cost, preserves
        // shorter conversations that are closer to completion).
        // Recompute mode: preempted request is re-queued and re-prefilled
        // when GPU memory frees up.
        while self.paged_kv_pool.is_active()
            && self.paged_kv_pool.free_count() < decode_indices.len()
            && decode_indices.len() > 1
        {
            // Pick victim: sglang-parity `retract_decode` ranking. Prefer the
            // request with the most output tokens AND the shortest input
            // tokens — cheapest to re-prefill when it resumes. Previous
            // single-key ranking only considered output length, repeatedly
            // preempting the 4 k-prompt reqs and paying 4 k of re-prefill on
            // each cycle under pool pressure.
            let victim_pos = decode_indices
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
                .unwrap();
            let victim_idx = decode_indices[victim_pos];
            let victim = &mut self.active[victim_idx];
            let victim_slot = victim.slot_idx;
            warn!(
                "Request {}: preempting (recompute) — {} generated tokens, pool free={}",
                victim.id,
                victim.generated_tokens.len(),
                self.paged_kv_pool.free_count()
            );

            // Free victim's paged pool tokens and reset slot state
            self.paged_kv_pool.free_slot(victim_slot);
            if let Err(e) = self.states[victim_slot].reset() {
                error!(
                    "Request {}: slot reset after preempt failed: {}",
                    victim.id, e
                );
            }
            self.slot_materialized_prompt_lens[victim_slot] = 0;

            // Re-queue as waiting (will be re-prefilled when re-admitted).
            // NOTE: preempted request loses generated tokens (recompute mode).
            // The prompt is re-tokenized on re-admission via assign_slots().
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

            // Remove from decode batch
            decode_indices.remove(victim_pos);
            token_ids.remove(victim_pos);

            // Push to front of waiting queue (priority re-admission)
            self.waiting.push_front(requeue);
        }

        // Allocate one pool token per remaining decode request.
        let mut alloc_ok_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        let mut alloc_ok_tokens: Vec<u32> = Vec::with_capacity(decode_indices.len());
        for (j, &i) in decode_indices.iter().enumerate() {
            // L1 retract regression guard (codex review 2026-04-19): if
            // `alloc_pool_tokens_with_retry` retracted this slot (or a
            // sibling we were about to decode) to free pool pages, its
            // phase has flipped to Finished. Skip it so the decode
            // dispatch doesn't sample a token for a just-retracted slot;
            // cleanup() will reap it next tick.
            if matches!(self.active[i].phase, Phase::Finished) {
                continue;
            }
            let slot = self.active[i].slot_idx;
            if let Err(e) = self.alloc_pool_tokens_with_retry(slot, 1) {
                error!(
                    "Request {}: KV pool exhausted after preemption (slot {}): {} — finishing",
                    self.active[i].id, slot, e
                );
                self.active[i].finish(FinishReason::Length, &self.tokenizer);
                continue;
            }
            if matches!(self.active[i].phase, Phase::Finished) {
                // Retry succeeded but this index was retracted as the
                // victim to satisfy its own request's shortfall.
                continue;
            }
            alloc_ok_indices.push(i);
            alloc_ok_tokens.push(token_ids[j]);
        }
        let decode_indices = alloc_ok_indices;
        let token_ids = alloc_ok_tokens;
        if decode_indices.is_empty() {
            return;
        }
        let slot_indices: Vec<usize> = decode_indices
            .iter()
            .map(|&i| self.active[i].slot_idx)
            .collect();

        let sampling_params: Vec<&crate::sampler::SamplingParams> = decode_indices
            .iter()
            .map(|&i| &self.active[i].sampling)
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        // Lazy-init decode context on first batched decode.
        if self.decode_bufs.is_none() {
            match self
                .model
                .create_decode_context(self.states.len(), &self.paged_kv_pool)
            {
                Ok(ctx) => self.decode_bufs = Some(ctx),
                Err(e) => {
                    error!("Failed to create decode context: {}", e);
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
                for &i in &decode_indices {
                    self.active[i].phase = Phase::Finished;
                }
                return;
            }
        }

        // TODO(mixed-batch): probe `forward_mixed_batch(...)` here once a model
        // implementation overrides the default `Ok(false)` path with a validated
        // eager mixed decode+prefill forward. Keep the existing decode-only path
        // unchanged until the model-side paged prefill prep is wired safely.
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
            sampling_params_greedy: sampling_params
                .iter()
                .map(|p| p.is_greedy() && !p.has_penalties())
                .collect(),
            mixed_prefill_chunks: Vec::new(),
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
