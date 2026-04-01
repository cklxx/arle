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

        // Phase 3: Prefill chunks
        let prefill_t = std::time::Instant::now();
        let mut did_prefill = true;
        while did_prefill {
            did_prefill = false;
            for idx in 0..self.active.len() {
                if matches!(self.active[idx].phase, Phase::Prefilling { .. }) {
                    self.step_prefill_chunk(idx, has_decode);
                    did_prefill = true;
                    break;
                }
            }
            if did_prefill {
                let still_prefilling = self
                    .active
                    .iter()
                    .any(|r| matches!(r.phase, Phase::Prefilling { .. }));
                if still_prefilling {
                    break;
                }
            }
        }
        let prefill_us = prefill_t.elapsed().as_micros();

        // Phase 4: Accept new requests
        for idx in 0..num {
            if matches!(self.active[idx].phase, Phase::New) {
                self.step_new(idx);
                return;
            }
        }

        // Log slow steps for profiling
        let total_us = emit_us + decode_us + prefill_us;
        if total_us > 100_000 {
            info!(
                "step breakdown: emit={}us decode={}us prefill={}us total={}us batch={}",
                emit_us, decode_us, prefill_us, total_us, num
            );
        }
    }
}
