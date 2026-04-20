use super::{ModelForward, Phase, Scheduler, info};

impl<M: ModelForward> Scheduler<M> {
    fn pending_mixed_prefill_tokens(&self) -> usize {
        self.pending_decode
            .as_ref()
            .map(|pending| {
                pending
                    .mixed_prefill_chunks
                    .iter()
                    .map(|chunk| chunk.token_count)
                    .sum()
            })
            .unwrap_or(0)
    }

    fn standalone_prefill_candidates(&self, already_mixed: &[usize]) -> Vec<usize> {
        let n_active = self.active.len();
        if n_active == 0 {
            return Vec::new();
        }
        let start = self.last_prefill_cursor % n_active;
        (0..n_active)
            .map(|k| (start + k) % n_active)
            .filter(|&i| matches!(self.active[i].phase, Phase::New | Phase::Prefilling { .. }))
            .filter(|&i| !already_mixed.contains(&i))
            .collect()
    }

    pub(super) fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        let step_t0 = std::time::Instant::now();
        let trace_on = std::env::var("INFER_TRACE").ok().as_deref() == Some("1");
        let decode_count_pre = self
            .active
            .iter()
            .filter(|r| matches!(r.phase, Phase::Decoding))
            .count();

        // Phase 1: Batched decode (GPU forward pass + sampling)
        //
        // Launched BEFORE emit_delta so that GPU compute overlaps with
        // CPU tokenizer work in Phase 2. emit_delta only needs tokens
        // from the PREVIOUS step (already in generated_tokens), so
        // there's no data dependency with the current forward pass.
        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        assert!(
            self.pending_decode.is_none(),
            "pending decode must be cleared before scheduler step"
        );
        self.pending_mixed_prefill_idxs.clear();
        let decode_launch_us = if has_decode {
            let t = std::time::Instant::now();
            // Round-robin selection of up to config.prefill_max_requests prefill
            // requests to fuse with the decode batch. Starts at
            // `last_mixed_prefill_cursor` so a busy request doesn't starve
            // a late-admitted one; cursor advances by the number actually
            // fused (not the number attempted).
            let n_active = self.active.len();
            let mixed_prefill_idxs: Vec<usize> = if !self.config.enable_mixed_chunk || n_active == 0 {
                Vec::new()
            } else {
                let start = self.last_mixed_prefill_cursor % n_active;
                (0..n_active)
                    .map(|k| (start + k) % n_active)
                    .filter(|&i| {
                        matches!(self.active[i].phase, Phase::Prefilling { .. })
                            && !self.active[i].delta_tx.is_closed()
                    })
                    .take(self.config.prefill_max_requests)
                    .collect()
            };
            if mixed_prefill_idxs.is_empty() {
                self.step_decode_launch();
            } else {
                // Advance cursor past the last selected req so next tick
                // picks up where this one left off.
                if let Some(&last) = mixed_prefill_idxs.last() {
                    self.last_mixed_prefill_cursor = (last + 1) % n_active.max(1);
                }
                self.step_decode_launch_mixed(&mixed_prefill_idxs);
            }
            t.elapsed().as_micros()
        } else {
            0
        };

        // Phase 2: Overlap window — GPU decode forward is in flight.
        //
        // Queue CPU work + prefill on the same stream while decode runs:
        //   a) emit_delta (CPU tokenizer, no GPU dependency)
        //   b) admit new requests (CPU)
        //   c) prefill chunks — GPU kernels queued AFTER decode on same stream,
        //      eliminating the dead time that existed when prefill ran after readback.
        //      When step_prefill_chunk internally syncs (for sampling), that sync
        //      catches both decode argmax AND prefill GPU work. The subsequent
        //      decode readback sync becomes a no-op.
        let emit_t = std::time::Instant::now();
        {
            let Self {
                active, tokenizer, ..
            } = self;
            for req in active.iter_mut() {
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    if trace_on {
                        let t = std::time::Instant::now();
                        req.emit_delta(tokenizer);
                        req.t_emit_us =
                            req.t_emit_us.saturating_add(t.elapsed().as_micros() as u64);
                    } else {
                        req.emit_delta(tokenizer);
                    }
                }
            }
        }
        let emit_us = emit_t.elapsed().as_micros();

        // Phase 2b: New requests are materialized lazily in Phase 2c when
        // they actually receive prefill budget. This keeps admission,
        // transition-to-prefill, and chunk launch on one path.
        let new_us = 0;

        // Phase 2c: Prefill chunks — queued on the same GPU stream.
        //
        // Decode-first rule:
        // - when decode is active, mixed launch gets first claim on the
        //   per-tick prefill budget;
        // - standalone prefill can only consume the residual token budget.
        // This keeps execution aligned with the launch-time planner instead
        // of letting Phase 2c independently inject extra prefill work.
        let prefill_t = std::time::Instant::now();
        let already_mixed: Vec<usize> = std::mem::take(&mut self.pending_mixed_prefill_idxs);
        let per_req_cap = self.prefill_chunk_size();
        let prefill_indices = self.standalone_prefill_candidates(&already_mixed);
        let mut remaining_prefill_budget = if has_decode && self.config.enable_mixed_chunk {
            self.config
                .max_prefill_tokens
                .saturating_sub(self.pending_mixed_prefill_tokens())
        } else {
            self.config.max_prefill_tokens
        };
        let mut remaining_prefill_requests = if has_decode && self.config.enable_mixed_chunk {
            self.config
                .prefill_max_requests
                .saturating_sub(already_mixed.len())
        } else {
            self.config.prefill_max_requests
        };
        for idx in prefill_indices {
            if remaining_prefill_budget == 0 || remaining_prefill_requests == 0 {
                break;
            }
            if matches!(self.active[idx].phase, Phase::New) {
                if trace_on {
                    let t = std::time::Instant::now();
                    self.step_new(idx);
                    let elapsed_us = t.elapsed().as_micros() as u64;
                    if let Some(req) = self.active.get_mut(idx) {
                        req.t_new_us = req.t_new_us.saturating_add(elapsed_us);
                    }
                } else {
                    self.step_new(idx);
                }
            }
            if !matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                self.last_prefill_cursor = (idx + 1) % self.active.len().max(1);
                continue;
            }
            let chunk_cap = remaining_prefill_budget.min(per_req_cap);
            let advanced = if trace_on {
                let t = std::time::Instant::now();
                let advanced = self.step_prefill_chunk_with_limit(idx, chunk_cap);
                let elapsed_us = t.elapsed().as_micros() as u64;
                if let Some(req) = self.active.get_mut(idx) {
                    req.t_prefill_us = req.t_prefill_us.saturating_add(elapsed_us);
                }
                advanced
            } else {
                self.step_prefill_chunk_with_limit(idx, chunk_cap)
            };
            if advanced > 0 {
                remaining_prefill_budget = remaining_prefill_budget.saturating_sub(advanced);
                remaining_prefill_requests = remaining_prefill_requests.saturating_sub(1);
                self.last_prefill_cursor = (idx + 1) % self.active.len().max(1);
            }
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Phase 3: Readback decode results (sync + D2H + token mutation).
        // If prefill already synced the stream, this sync is a no-op.
        let readback_us = if has_decode && self.pending_decode.is_some() {
            let t = std::time::Instant::now();
            self.step_decode_readback();
            t.elapsed().as_micros()
        } else {
            0
        };
        let decode_us = decode_launch_us + readback_us;

        // Step timing — always tracked for /v1/stats and profiling.
        // EMA (exponential moving average) with α=0.1 smooths noise
        // while responding quickly to sustained changes.
        let total_us = decode_us + emit_us + new_us + prefill_us;
        let update_ema = |ema: &mut f64, val: u128| {
            const ALPHA: f64 = 0.1;
            let v = val as f64;
            if *ema == 0.0 {
                *ema = v;
            } else {
                *ema = ALPHA * v + (1.0 - ALPHA) * *ema;
            }
        };
        update_ema(&mut self.step_timing_decode_us, decode_us);
        update_ema(&mut self.step_timing_emit_us, emit_us);
        update_ema(&mut self.step_timing_prefill_us, prefill_us);
        update_ema(&mut self.step_timing_total_us, total_us);

        if trace_on {
            let per_decode_us = if decode_count_pre > 0 {
                (decode_us as u64) / (decode_count_pre as u64)
            } else {
                0
            };
            for req in &mut self.active {
                if per_decode_us > 0 && matches!(req.phase, Phase::Decoding | Phase::Finished) {
                    req.t_decode_us = req.t_decode_us.saturating_add(per_decode_us);
                }
                req.step_count = req.step_count.saturating_add(1);
                if req.first_step_at.is_none() {
                    req.first_step_at = Some(step_t0);
                }
            }
        }

        if total_us > 100_000 {
            info!(
                "step breakdown: decode={}us emit={}us new={}us prefill={}us total={}us batch={}",
                decode_us, emit_us, new_us, prefill_us, total_us, num
            );
        }
    }
}
