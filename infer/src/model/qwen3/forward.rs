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
        use rand::Rng;
        let b = slot_indices.len();

        // Phase 1: Launch all sampling kernels (no sync between them)
        for i in 0..b {
            let si = slot_indices[i];
            let random_val: f32 = rng.random();
            let logits = states[si]
                .prefill_logits
                .as_ref()
                .unwrap_or(&states[si].decode_bufs.logits);
            crate::ops::gpu_sample_launch(
                &self.ctx,
                logits,
                &mut states[si].decode_bufs.sample_probs,
                &mut states[si].decode_bufs.sample_out,
                params[i],
                random_val,
            )?;
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

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        paged_kv_pool: Option<&mut crate::paged_kv::PagedKVPool>,
    ) -> Result<()> {
        if tokens.len() <= 1 {
            // Fall back to single-token path for bs=1 (benefits from CUDA Graph)
            if tokens.len() == 1 {
                return self.forward(&[tokens[0]], &mut states[slot_indices[0]]);
            }
            return Ok(());
        }
        let pool = paged_kv_pool
            .expect("FlashInfer batched decode requires a PagedKVPool");
        self.decode_batch(tokens, states, slot_indices, pool)
    }
}
