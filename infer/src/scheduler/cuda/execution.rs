use super::*;

impl<M: ModelForward> Scheduler<M> {
    pub(super) fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

        // Phase 1: Flush deferred deltas (CPU tokenizer decode)
        let emit_t = std::time::Instant::now();
        {
            let Self {
                active, tokenizer, ..
            } = self;
            for req in active.iter_mut() {
                if matches!(req.phase, Phase::Decoding)
                    && req.decoded_token_count < req.generated_tokens.len()
                {
                    req.emit_delta(tokenizer);
                }
            }
        }
        let emit_us = emit_t.elapsed().as_micros();

        // Phase 2: Batched decode (GPU forward pass)
        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        let decode_us = if has_decode {
            let t = std::time::Instant::now();
            self.step_decode_batch();
            t.elapsed().as_micros()
        } else {
            0
        };

        // Phase 3: Accept ALL new requests (prefix cache + start prefill)
        // Process all new requests in one step to avoid multi-iteration admission delay.
        let new_t = std::time::Instant::now();
        loop {
            let new_idx =
                (0..self.active.len()).find(|&i| matches!(self.active[i].phase, Phase::New));
            match new_idx {
                Some(idx) => self.step_new(idx),
                None => break,
            }
        }
        let new_us = new_t.elapsed().as_micros();

        // Phase 4: Prefill chunks — process all pending prefill requests,
        // one chunk each, to maximize prefill throughput while still yielding
        // to the next decode step.
        let prefill_t = std::time::Instant::now();
        let prefill_indices: Vec<usize> = (0..self.active.len())
            .filter(|&i| matches!(self.active[i].phase, Phase::Prefilling { .. }))
            .collect();
        for idx in prefill_indices {
            // Re-check phase since step_prefill_chunk may transition to Decoding/Finished
            if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                self.step_prefill_chunk(idx, has_decode);
            }
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Log slow steps for profiling
        let total_us = emit_us + decode_us + new_us + prefill_us;
        if total_us > 100_000 {
            info!(
                "step breakdown: emit={}us decode={}us new={}us prefill={}us total={}us batch={}",
                emit_us, decode_us, new_us, prefill_us, total_us, num
            );
        }
    }
}
