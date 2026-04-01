use super::*;

impl<M: ModelForward> Scheduler<M> {
    /// Batch all decode requests into a single GPU forward pass.
    pub(super) fn step_decode_batch(&mut self) {
        let Self {
            model,
            tokenizer,
            states,
            active,
            rng,
            paged_kv_pool,
            decode_bufs,
            ..
        } = self;

        let decode_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, r)| matches!(r.phase, Phase::Decoding) && !r.delta_tx.is_closed())
            .map(|(i, _)| i)
            .collect();

        for i in 0..active.len() {
            if matches!(active[i].phase, Phase::Decoding) && active[i].delta_tx.is_closed() {
                active[i].phase = Phase::Finished;
            }
        }

        if decode_indices.is_empty() {
            return;
        }

        let mut token_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
        let mut valid_decode_indices: Vec<usize> = Vec::with_capacity(decode_indices.len());
        for &i in &decode_indices {
            match active[i].generated_tokens.last() {
                Some(&tok) => {
                    token_ids.push(tok);
                    valid_decode_indices.push(i);
                }
                None => {
                    error!(
                        "Request {}: Decoding state with no generated tokens - dropping",
                        active[i].id
                    );
                    active[i].phase = Phase::Finished;
                }
            }
        }
        let decode_indices = valid_decode_indices;
        if decode_indices.is_empty() {
            return;
        }
        let slot_indices: Vec<usize> = decode_indices.iter().map(|&i| active[i].slot_idx).collect();

        for &slot in &slot_indices {
            if let Err(e) = paged_kv_pool.alloc_tokens(slot, 1) {
                error!("Pool alloc for decode token (slot {}) failed: {}", slot, e);
            }
        }

        let sampling_params: Vec<&crate::sampler::SamplingParams> = decode_indices
            .iter()
            .map(|&i| &active[i].sampling)
            .collect();
        let all_greedy = sampling_params
            .iter()
            .all(|p| p.is_greedy() && !p.has_penalties());

        let forward_result = model.forward_decode_batch(
            &token_ids,
            states,
            &slot_indices,
            Some(paged_kv_pool),
            decode_bufs,
            all_greedy,
        );

        if let Err(e) = forward_result {
            error!("Batched decode failed: {}", e);
            for &i in &decode_indices {
                active[i].phase = Phase::Finished;
            }
            return;
        }
        let sampled_result = if all_greedy {
            match model.sample_batch_greedy(&slot_indices, decode_bufs) {
                Ok(Some(tokens)) => Ok(tokens),
                Ok(None) => model.select_tokens_batch(states, &slot_indices, &sampling_params, rng),
                Err(e) => Err(e),
            }
        } else {
            model.select_tokens_batch(states, &slot_indices, &sampling_params, rng)
        };
        match sampled_result {
            Ok(sampled_tokens) => {
                for (j, &req_idx) in decode_indices.iter().enumerate() {
                    let token = sampled_tokens[j];
                    let req = &mut active[req_idx];
                    if !req.sampling.ignore_eos && model.is_stop_token(token) {
                        req.finish(FinishReason::Stop, tokenizer);
                        continue;
                    }
                    req.generated_tokens.push(token);
                    if matches!(req.phase, Phase::Finished) {
                        continue;
                    }
                    if req.generated_tokens.len() >= req.max_tokens {
                        req.finish(FinishReason::Length, tokenizer);
                    }
                }
            }
            Err(e) => {
                error!("Batched sampling failed: {}", e);
                for &req_idx in &decode_indices {
                    active[req_idx].phase = Phase::Finished;
                }
            }
        }
    }
}
