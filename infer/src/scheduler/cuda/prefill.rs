use super::*;

impl<M: ModelForward> Scheduler<M> {
    /// Compute prefix cache for a new request and begin chunked prefill.
    pub(super) fn step_new(&mut self, idx: usize) {
        let default_chunk_size = self.prefill_chunk_size(false);
        let req = &mut self.active[idx];
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        let si = req.slot_idx;
        let cached = &mut self.cached_prompts[si];
        let state = &mut self.states[si];

        let raw_prefix_len = cached
            .iter()
            .zip(req.prompt_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // Track effective prefix length (may be zeroed on fallback to miss).
        let mut prefix_len = raw_prefix_len;

        let effective = if raw_prefix_len > 0 && raw_prefix_len == cached.len() {
            // Full prefix hit: cached prompt is entirely contained in new prompt.
            // Restore recurrent state snapshot if available (Qwen3.5 hybrid model).
            let restore_ok = match state.restore_recurrent_snapshot() {
                Ok(true) => {
                    info!(
                        "Request {}: prefix HIT {}/{} tokens (recurrent restored)",
                        req.id, prefix_len, req.prompt_tokens.len()
                    );
                    true
                }
                Ok(false) => {
                    info!(
                        "Request {}: prefix HIT {}/{} tokens",
                        req.id, prefix_len, req.prompt_tokens.len()
                    );
                    true
                }
                Err(e) => {
                    error!(
                        "Request {}: recurrent restore failed ({}), falling back to miss",
                        req.id, e
                    );
                    false
                }
            };

            if !restore_ok {
                // Restore failed, fall back to full miss.
                prefix_len = 0;
                if let Err(e) = state.reset() {
                    error!("Request {}: reset failed: {}", req.id, e);
                    req.phase = Phase::Finished;
                    return;
                }
                cached.clear();
                req.prompt_tokens.clone()
            } else {
                let suffix = &req.prompt_tokens[prefix_len..];
                if suffix.is_empty() {
                    let Some(&last_tok) = req.prompt_tokens.last() else {
                        error!(
                            "Request {}: prompt_tokens empty on full prefix hit - dropping",
                            req.id
                        );
                        req.phase = Phase::Finished;
                        return;
                    };
                    vec![last_tok]
                } else {
                    suffix.to_vec()
                }
            }
        } else if prefix_len > 0 && !state.has_recurrent_state() {
            // Partial prefix hit — only safe for models without recurrent state.
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
            // Miss (or partial hit on recurrent model — unsafe to reuse partial state).
            if raw_prefix_len > 0 {
                info!(
                    "Request {}: prefix PARTIAL {}/{} tokens (recurrent model, treating as MISS)",
                    req.id, raw_prefix_len, req.prompt_tokens.len()
                );
            } else {
                info!("Request {}: prefix MISS", req.id);
            }
            prefix_len = 0;
            if let Err(e) = state.reset() {
                error!("Request {}: reset failed: {}", req.id, e);
                req.phase = Phase::Finished;
                return;
            }
            cached.clear();
            req.prompt_tokens.clone()
        };

        if prefix_len > 0 && !self.paged_kv_pool.k_buffers.is_empty() {
            if let Err(e) = self.paged_kv_pool.alloc_tokens(si, prefix_len) {
                error!("Request {}: pool alloc for prefix failed: {}", req.id, e);
            } else {
                let ctx = self.model.device_context();
                if let Err(e) = state.migrate_kv_to_paged(ctx, &self.paged_kv_pool, si) {
                    error!(
                        "Request {}: prefix KV migration to pool failed: {}",
                        req.id, e
                    );
                }
            }
        }

        info!(
            "Request {}: chunked prefill starting ({} effective tokens, chunk_size={})",
            req.id,
            effective.len(),
            default_chunk_size
        );

        req.phase = Phase::Prefilling {
            effective_tokens: effective,
            progress: 0,
        };
    }

    /// Process one chunk of a prefill. When all chunks are done, sample the
    /// first token and transition to Decoding.
    pub(super) fn step_prefill_chunk(&mut self, idx: usize, decode_active: bool) {
        let chunk_size = self.prefill_chunk_size(decode_active);

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
        if req.delta_tx.is_closed() {
            req.phase = Phase::Finished;
            return;
        }

        let (effective_tokens, progress) = match &mut req.phase {
            Phase::Prefilling {
                effective_tokens,
                progress,
            } => (effective_tokens as &Vec<u32>, progress as &mut usize),
            _ => return,
        };

        let total = effective_tokens.len();
        let chunk_end = (*progress + chunk_size).min(total);
        let chunk = &effective_tokens[*progress..chunk_end];

        let slot_idx = req.slot_idx;
        let state = &mut states[slot_idx];
        let forward_result = model.forward_prefill(chunk, state);

        if let Err(e) = forward_result {
            error!("Request {}: prefill chunk failed: {}", req.id, e);
            req.phase = Phase::Finished;
            return;
        }

        let new_progress = chunk_end;
        if new_progress < total {
            *progress = new_progress;
            info!(
                "Request {}: prefill chunk {}/{} tokens",
                req.id, new_progress, total
            );
            return;
        }

        // Save recurrent state snapshot for future prefix cache reuse.
        // Must happen after all prefill chunks complete but before decode modifies state.
        if let Err(e) = state.save_recurrent_snapshot() {
            error!(
                "Request {}: save recurrent snapshot failed: {}",
                req.id, e
            );
            // Non-fatal: prefix cache just won't work for this slot's next request.
        }

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

        match model.select_token(state, &req.sampling, rng) {
            Ok(token) => {
                if !req.sampling.ignore_eos && model.is_stop_token(token) {
                    req.finish(FinishReason::Stop, tokenizer);
                    return;
                }
                req.generated_tokens.push(token);
                req.emit_delta(tokenizer);

                if matches!(req.phase, Phase::Finished) {
                    return;
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
}
