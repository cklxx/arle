use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::Qwen35Model;
use crate::model::generation_state::GenerationStateBase;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

pub struct Qwen35State {
    pub(super) ctx: DeviceContext,
    pub(super) decode_bufs: DecodeBuffers35,
    pub(super) single_token_bufs: SingleTokenBuffers,
    pub(crate) base: GenerationStateBase,
    pub(super) recurrent_state: RecurrentState,
}

// SAFETY: `Qwen35State` contains CUDA resources (`DeviceContext`, `CudaSlice` inside
// `DecodeBuffers35`, `SingleTokenBuffers`, `GenerationStateBase` wrapping `KVCache`,
// `CudaGraphState`, `DeviceVec`, plus `RecurrentState`) that hold raw CUDA device
// pointers.  These types are `!Send` by default because CUDA contexts and
// allocations must be accessed from the thread that created them.
//
// Invariant upheld: every `Qwen35State` instance is exclusively owned by its
// scheduler slot and only ever accessed from the single blocking inference
// thread that runs `Scheduler::run()`.  No other thread holds a reference to
// or borrows from this state while the inference thread is running.
//
// Violation would mean: concurrent access from multiple threads could cause
// data races on GPU memory or corrupt the CUDA driver state.
unsafe impl Send for Qwen35State {}

impl GenerationState for Qwen35State {
    fn logits(&self) -> &DeviceVec {
        self.base.logits_or(&self.single_token_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.base.reset()?;
        self.recurrent_state.reset(&self.ctx)?;
        Ok(())
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.base.truncate_to(len)?;
        self.recurrent_state.reset(&self.ctx)?;
        Ok(())
    }

    fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.base.set_max_gpu_kv(max_tokens);
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.base.set_max_seq_len(max_seq);
    }

    fn offload_kv_if_needed(&mut self) -> Result<()> {
        self.base.offload_kv_if_needed(&self.ctx)
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &crate::tensor::DeviceContext,
        pool: &crate::paged_kv::PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        self.base.migrate_kv_to_paged(ctx, pool, slot)
    }
}

impl ModelForward for Qwen35Model {
    type State = Qwen35State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen35State {
            ctx: self.ctx.clone(),
            decode_bufs: DecodeBuffers35::new(&self.ctx, &self.config)?,
            single_token_bufs: SingleTokenBuffers::new(&self.ctx, &self.config)?,
            base: GenerationStateBase::new(
                self.config.num_full_attention_layers(),
                self.config.num_key_value_heads,
            ),
            recurrent_state: RecurrentState::new(&self.ctx, &self.config)?,
        })
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // Only full-attention layers have KV cache (linear layers use recurrent state).
        // 2 (K+V) * num_full_attn_layers * num_kv_heads * head_dim * 2 (bf16 = 2 bytes)
        2 * self.config.num_full_attention_layers()
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }

    fn num_kv_layers(&self) -> usize {
        self.config.num_full_attention_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        // Prefetch offloaded KV before PREFILL only.
        if tokens.len() > 1 && state.base.kv_cache.has_offloaded() {
            state.base.kv_cache.prefetch_to_gpu(&self.ctx)?;
        }

        if tokens.len() == 1 {
            self.prefill_forward_single_token(
                tokens[0],
                &mut state.base.kv_cache,
                &mut state.recurrent_state,
                &mut state.single_token_bufs,
                &mut state.base.graph_state,
            )?;
            state.base.prefill_logits = None;
        } else {
            let logits =
                self.prefill_forward(tokens, &mut state.base.kv_cache, &mut state.recurrent_state)?;
            state.base.prefill_logits = Some(logits);
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
        let logits = state.base.logits_or(&state.single_token_bufs.logits);
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
        token_id == self.config.eos_token_id
    }

    fn device_context(&self) -> &crate::tensor::DeviceContext {
        &self.ctx
    }
}
