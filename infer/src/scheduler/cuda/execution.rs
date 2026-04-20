use super::decode::MIXED_PREFILL_MAX_REQS;
use super::{ModelForward, Phase, Scheduler, info};

impl<M: ModelForward> Scheduler<M> {
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
            // Round-robin selection of up to MIXED_PREFILL_MAX_REQS prefill
            // requests to fuse with the decode batch. Starts at
            // `last_mixed_prefill_cursor` so a busy request doesn't starve
            // a late-admitted one; cursor advances by the number actually
            // fused (not the number attempted).
            let n_active = self.active.len();
            let mixed_prefill_idxs: Vec<usize> = if n_active == 0 {
                Vec::new()
            } else {
                let start = self.last_mixed_prefill_cursor % n_active;
                (0..n_active)
                    .map(|k| (start + k) % n_active)
                    .filter(|&i| {
                        matches!(self.active[i].phase, Phase::Prefilling { .. })
                            && !self.active[i].delta_tx.is_closed()
                    })
                    .take(MIXED_PREFILL_MAX_REQS)
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

        // Phase 2b: Accept new requests (CPU-only, no GPU)
        let new_t = std::time::Instant::now();
        loop {
            let new_idx =
                (0..self.active.len()).find(|&i| matches!(self.active[i].phase, Phase::New));
            match new_idx {
                Some(idx) => {
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
                None => break,
            }
        }
        let new_us = new_t.elapsed().as_micros();

        // Phase 2c: Prefill chunks — queued on GPU stream while decode is in flight.
        // Adaptive rate based on decode batch size.
        let prefill_t = std::time::Instant::now();
        let decode_count = self
            .active
            .iter()
            .filter(|r| matches!(r.phase, Phase::Decoding))
            .count();
        let max_prefills = if !has_decode {
            8 // No decode: drain prefill queue fast
        } else if decode_count <= 4 {
            4 // Ramp-up: GPU has headroom, reduce TTFT
        } else if decode_count <= 16 {
            2 // Moderate load: balance TTFT vs ITL
        } else {
            1 // High concurrency: protect decode latency
        };
        let already_mixed: Vec<usize> = std::mem::take(&mut self.pending_mixed_prefill_idxs);
        let prefill_indices: Vec<usize> = (0..self.active.len())
            .filter(|&i| matches!(self.active[i].phase, Phase::Prefilling { .. }))
            .filter(|&i| !already_mixed.contains(&i))
            .take(max_prefills)
            .collect();
        for idx in prefill_indices {
            if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                if trace_on {
                    let t = std::time::Instant::now();
                    self.step_prefill_chunk(idx);
                    let elapsed_us = t.elapsed().as_micros() as u64;
                    if let Some(req) = self.active.get_mut(idx) {
                        req.t_prefill_us = req.t_prefill_us.saturating_add(elapsed_us);
                    }
                } else {
                    self.step_prefill_chunk(idx);
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
