use super::{ModelForward, Phase, Scheduler, info};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefillBudget {
    rem_input_tokens: usize,
    rem_chunk_tokens: usize,
    rem_prefill_requests: usize,
}

impl PrefillBudget {
    fn exhausted(self) -> bool {
        self.rem_input_tokens == 0 || self.rem_chunk_tokens == 0 || self.rem_prefill_requests == 0
    }
}

fn plan_prefill_budget(
    has_decode: bool,
    enable_mixed_chunk: bool,
    already_mixed: &[usize],
    max_prefill_tokens: usize,
    chunked_prefill_size: usize,
    prefill_max_requests: usize,
    decode_active_chunk_budget: usize,
) -> PrefillBudget {
    // Match SGLang's PrefillAdder shape: one per-round budget with a total
    // input-token budget plus a chunk budget. Once the decode launch already
    // fused prefill work for this round, the standalone path gets a zeroed
    // budget instead of constructing a second prefill batch.
    if has_decode && enable_mixed_chunk && !already_mixed.is_empty() {
        PrefillBudget {
            rem_input_tokens: 0,
            rem_chunk_tokens: 0,
            rem_prefill_requests: 0,
        }
    } else {
        let rem_chunk_tokens = if has_decode && enable_mixed_chunk {
            decode_active_chunk_budget
        } else {
            chunked_prefill_size
        };
        PrefillBudget {
            rem_input_tokens: max_prefill_tokens,
            rem_chunk_tokens,
            rem_prefill_requests: prefill_max_requests,
        }
    }
}

impl<M: ModelForward> Scheduler<M> {
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
        // Decode-active ticks now have a single prefill owner:
        // - if the mixed decode launch fused any prefill sections, this tick's
        //   prefill work is complete and Phase 2c does nothing;
        // - otherwise Phase 2c may run standalone prefill with the normal
        //   configured budgets.
        //
        // This mirrors SGLang's one-`PrefillAdder` semantics instead of mixing
        // a tiny fused prefill batch with a second large standalone prefill path
        // in the same scheduler round.
        let prefill_t = std::time::Instant::now();
        let already_mixed: Vec<usize> = std::mem::take(&mut self.pending_mixed_prefill_idxs);
        let mut prefill_budget = plan_prefill_budget(
            has_decode,
            self.config.enable_mixed_chunk,
            &already_mixed,
            self.config.max_prefill_tokens,
            self.config.chunked_prefill_size,
            self.config.prefill_max_requests,
            self.mixed_prefill_budget_tokens(),
        );
        if !prefill_budget.exhausted() {
            let per_req_cap = self.prefill_chunk_size();
            let prefill_indices = self.standalone_prefill_candidates(&already_mixed);
            for idx in prefill_indices {
                if prefill_budget.exhausted() {
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
                let chunk_cap = per_req_cap
                    .min(prefill_budget.rem_input_tokens)
                    .min(prefill_budget.rem_chunk_tokens);
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
                    prefill_budget.rem_input_tokens =
                        prefill_budget.rem_input_tokens.saturating_sub(advanced);
                    prefill_budget.rem_chunk_tokens =
                        prefill_budget.rem_chunk_tokens.saturating_sub(advanced);
                    prefill_budget.rem_prefill_requests =
                        prefill_budget.rem_prefill_requests.saturating_sub(1);
                    self.last_prefill_cursor = (idx + 1) % self.active.len().max(1);
                }
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

#[cfg(test)]
mod tests {
    use super::{PrefillBudget, plan_prefill_budget};

    #[test]
    fn mixed_decode_tick_exhausts_standalone_prefill_budget() {
        assert_eq!(
            plan_prefill_budget(true, true, &[3], 16384, 4096, 4, 64),
            PrefillBudget {
                rem_input_tokens: 0,
                rem_chunk_tokens: 0,
                rem_prefill_requests: 0,
            }
        );
    }

    #[test]
    fn decode_tick_without_fused_prefill_uses_decode_active_chunk_budget() {
        assert_eq!(
            plan_prefill_budget(true, true, &[], 16384, 4096, 4, 64),
            PrefillBudget {
                rem_input_tokens: 16384,
                rem_chunk_tokens: 64,
                rem_prefill_requests: 4,
            }
        );
    }

    #[test]
    fn non_decode_tick_keeps_configured_budget() {
        assert_eq!(
            plan_prefill_budget(false, true, &[3], 16384, 4096, 4, 64),
            PrefillBudget {
                rem_input_tokens: 16384,
                rem_chunk_tokens: 4096,
                rem_prefill_requests: 4,
            }
        );
    }
}
