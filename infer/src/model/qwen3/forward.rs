use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers;
use super::weights::Qwen3Model;
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::KVCache;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::DeviceVec;

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    pub(super) ctx: crate::tensor::DeviceContext,
    pub(crate) decode_bufs: DecodeBuffers,
    pub(super) kv_cache: KVCache,
    pub(super) graph_state: CudaGraphState,
    /// Logits from multi-token prefill (None after decode path — logits are in decode_bufs).
    pub(crate) prefill_logits: Option<DeviceVec>,
}

// SAFETY: `Qwen3State` contains CUDA resources (`DeviceContext`, `CudaSlice` inside
// `DecodeBuffers`, `KVCache`, `CudaGraphState`, `DeviceVec`) that hold raw CUDA
// device pointers.  These types are `!Send` by default because CUDA contexts and
// allocations must be accessed from the thread that created them.
//
// Invariant upheld: every `Qwen3State` instance is exclusively owned by its
// scheduler slot and only ever accessed from the single blocking inference
// thread that runs `Scheduler::run()`.  No other thread holds a reference to
// or borrows from this state while the inference thread is running.
//
// Violation would mean: concurrent access from multiple threads could cause
// data races on GPU memory or corrupt the CUDA driver state.
unsafe impl Send for Qwen3State {}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits
            .as_ref()
            .unwrap_or(&self.decode_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_cache.reset();
        self.prefill_logits = None;
        self.graph_state = CudaGraphState::new(); // Invalidate CUDA graph
        Ok(())
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.kv_cache.truncate_to(len);
        self.prefill_logits = None;
        self.graph_state = CudaGraphState::new();
        Ok(())
    }

    fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.kv_cache.set_max_gpu_seq_len(max_tokens);
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.kv_cache.set_max_seq_len(max_seq);
    }

    fn offload_kv_if_needed(&mut self) -> Result<()> {
        self.kv_cache.offload_if_needed(&self.ctx)
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &crate::tensor::DeviceContext,
        pool: &crate::paged_kv::PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        pool.migrate_from_contiguous(
            ctx,
            slot,
            &self.kv_cache.k_caches(),
            &self.kv_cache.v_caches(),
            self.kv_cache.max_seq_len(),
        )
    }
}

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
            ctx: self.ctx.clone(),
            decode_bufs: DecodeBuffers::new(&self.ctx, &self.config)?,
            kv_cache: KVCache::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
            graph_state: CudaGraphState::new(),
            prefill_logits: None,
        })
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (bf16 = 2 bytes)
        2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }

    fn num_kv_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        // Prefetch offloaded KV before PREFILL only (not every decode step).
        // During decode of a single request, all KV stays on GPU.
        if tokens.len() > 1 && state.kv_cache.has_offloaded() {
            state.kv_cache.prefetch_to_gpu(&self.ctx)?;
        }

        if tokens.len() == 1 {
            self.decode_one_token(
                tokens[0],
                &mut state.kv_cache,
                &mut state.decode_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            let start_pos = state.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);
        }

        // NOTE: offload_if_needed is NOT called here. It is called between
        // requests in GenericServerEngine, after complete()/complete_stream()
        // finishes the entire generation. During a single request's decode
        // loop, all KV stays on GPU for correct attention.

        Ok(())
    }

    fn forward_prefill_with_pool(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
        pool: &crate::paged_kv::TokenKVPool,
        _slot: usize,
        new_token_indices: &cudarc::driver::CudaSlice<i32>,
    ) -> Result<()> {
        // Prefetch offloaded KV before PREFILL only.
        if tokens.len() > 1 && state.kv_cache.has_offloaded() {
            state.kv_cache.prefetch_to_gpu(&self.ctx)?;
        }

        if tokens.len() == 1 {
            // Single-token: use standard decode path (CUDA Graph).
            // No pool scatter-write needed — decode path writes to pool via
            // decode_prep_paged kernel.
            self.decode_one_token(
                tokens[0],
                &mut state.kv_cache,
                &mut state.decode_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            let start_pos = state.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            // Dual-write: writes K/V to both contiguous cache AND token pool.
            let hidden = self.process_all_layers_batch_with_pool(
                hidden,
                start_pos,
                &mut state.kv_cache,
                pool,
                new_token_indices,
            )?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);
        }

        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state
            .prefill_logits
            .as_ref()
            .unwrap_or(&state.decode_bufs.logits);
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            &mut state.decode_bufs.sample_probs,
            &mut state.decode_bufs.sample_out,
            params,
            random_val,
        )
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }

    fn device_context(&self) -> &crate::tensor::DeviceContext {
        &self.ctx
    }

    fn select_tokens_batch(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        params: &[&crate::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<Vec<u32>> {
        let b = slot_indices.len();

        // Phase 1: Launch all sampling kernels using cached pointers
        for i in 0..b {
            let si = slot_indices[i];
            let random_val: f32 = rng.random();
            // When prefill_logits is set, fall back to the non-cached path
            if states[si].prefill_logits.is_some() {
                let logits = states[si].prefill_logits.as_ref().unwrap();
                crate::ops::gpu_sample_launch(
                    &self.ctx,
                    logits,
                    &mut states[si].decode_bufs.sample_probs,
                    &mut states[si].decode_bufs.sample_out,
                    params[i],
                    random_val,
                )?;
            } else {
                let ptrs = &states[si].decode_bufs.ptrs;
                crate::ops::gpu_sample_launch_raw(
                    &self.ctx,
                    ptrs.logits_ptr,
                    ptrs.logits_len,
                    ptrs.sample_probs_ptr,
                    ptrs.sample_out_ptr,
                    params[i],
                    random_val,
                )?;
            }
        }

        // Phase 2: Single sync
        self.ctx.sync()?;

        // Phase 3: Readback all results
        let mut tokens = Vec::with_capacity(b);
        for i in 0..b {
            let si = slot_indices[i];
            tokens.push(crate::ops::gpu_sample_readback(
                &self.ctx,
                &states[si].decode_bufs.sample_out,
            )?);
        }
        Ok(tokens)
    }

    fn sample_batch_greedy(
        &self,
        slot_indices: &[usize],
        decode_bufs_cache: &mut Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<Option<Vec<u32>>> {
        use super::batch_decode::BatchDecodeBuffers;
        let bufs = match decode_bufs_cache {
            Some(cache) => match cache.downcast_mut::<BatchDecodeBuffers>() {
                Some(b) => b,
                None => return Ok(None),
            },
            None => return Ok(None),
        };
        let logits = match bufs.logits_batch.as_ref() {
            Some(l) if l.seq_len > 0 => l,
            _ => return Ok(None),
        };
        let batch_size = slot_indices.len();
        crate::ops::argmax_batch_launch(&self.ctx, logits, &mut bufs.argmax_out, batch_size)?;
        self.ctx.sync()?;
        crate::ops::argmax_batch_readback_into(
            &self.ctx,
            &bufs.argmax_out,
            &mut bufs.argmax_host,
            batch_size,
        )?;
        Ok(Some(
            bufs.argmax_host[..batch_size]
                .iter()
                .map(|&x| x as u32)
                .collect(),
        ))
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        paged_kv_pool: Option<&mut crate::paged_kv::PagedKVPool>,
        decode_bufs_cache: &mut Option<Box<dyn std::any::Any + Send>>,
        skip_logit_scatter: bool,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        // Always use the FlashInfer paged path when the pool is active, even for
        // batch_size=1. Routing B=1 through the contiguous/Triton decode path
        // causes greedy output divergence: (1) K/V is written only to the
        // contiguous cache, not the pool, so later batches read stale pool data;
        // (2) Triton and FlashInfer attention produce numerically different bf16
        // results, making greedy (argmax) output depend on batch composition.
        match paged_kv_pool {
            Some(pool) if !pool.k_buffers.is_empty() => {
                use super::batch_decode::BatchDecodeBuffers;

                // Lazy-init or reuse pre-allocated buffers
                let bufs = match decode_bufs_cache {
                    Some(cache) => cache
                        .downcast_mut::<BatchDecodeBuffers>()
                        .expect("decode_bufs_cache type mismatch"),
                    None => {
                        let num_heads = self.config.num_attention_heads;
                        let num_kv_heads = self.config.num_key_value_heads;
                        let head_dim = self.config.head_dim;
                        let q_dim = num_heads * head_dim;
                        let kv_dim = num_kv_heads * head_dim;
                        let inter_dim = self.config.intermediate_size;
                        let max_bs = states.len(); // max possible batch size = num_slots
                        // max_total_pages: worst case = max_bs * max_seq_len
                        let max_pages = pool.max_total_tokens;
                        let b: Box<dyn std::any::Any + Send> = Box::new(BatchDecodeBuffers::new(
                            &self.ctx,
                            self.config.hidden_size,
                            q_dim,
                            kv_dim,
                            inter_dim,
                            max_bs,
                            num_heads,
                            max_pages,
                        )?);
                        *decode_bufs_cache = Some(b);
                        decode_bufs_cache
                            .as_mut()
                            .unwrap()
                            .downcast_mut::<BatchDecodeBuffers>()
                            .unwrap()
                    }
                };
                self.decode_batch(tokens, states, slot_indices, skip_logit_scatter, pool, bufs)
            }
            _ => self.decode_batch_contiguous(tokens, states, slot_indices),
        }
    }
}
