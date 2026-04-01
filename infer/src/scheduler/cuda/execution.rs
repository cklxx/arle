use super::*;

impl<M: ModelForward> Scheduler<M> {
    fn step(&mut self) {
        let num = self.active.len();
        if num == 0 {
            return;
        }

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

        let has_decode = self
            .active
            .iter()
            .any(|r| matches!(r.phase, Phase::Decoding));
        if has_decode {
            self.step_decode_batch();
        }

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

        for idx in 0..num {
            if matches!(self.active[idx].phase, Phase::New) {
                self.step_new(idx);
                return;
            }
        }
    }
}
