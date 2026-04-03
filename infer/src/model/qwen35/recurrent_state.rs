//! Recurrent state for Qwen3.5 linear attention layers.
//!
//! Each linear attention layer maintains:
//! - Recurrent state: [num_value_heads, key_head_dim, value_head_dim] f32, V contiguous ([H,K,V])
//! - Conv state: [qkv_dim × (conv_kernel_dim - 1)] bf16

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::config::Config35;
use crate::tensor::{DeviceContext, DeviceVec};

/// CPU-backed snapshot of all recurrent state layers, for prefix cache reuse.
/// Saved after prefill completes, restored on prefix cache hit to avoid
/// recomputing recurrent state from scratch.
pub(crate) struct RecurrentSnapshot {
    /// Per-layer GDR recurrent state (f32, on CPU).
    pub(crate) gdr_states: Vec<Vec<f32>>,
    /// Per-layer conv1d state (bf16, on CPU).
    pub(crate) conv_states: Vec<Vec<half::bf16>>,
    /// Number of tokens processed when snapshot was taken.
    pub(crate) seq_len: usize,
}

/// Per-layer recurrent state for a single linear attention layer.
pub(crate) struct LayerRecurrentState {
    /// Recurrent state matrix: [num_value_heads * key_head_dim * value_head_dim] f32
    /// Stored as f32 per mamba_ssm_dtype="float32" in config.
    pub(crate) state: CudaSlice<f32>,
    /// Conv1d state buffer: [qkv_dim * (conv_kernel_dim - 1)] bf16
    /// Stores the last (kernel_dim - 1) inputs for causal conv1d.
    pub(crate) conv_state: DeviceVec,
}

/// Recurrent state for all linear attention layers.
pub(crate) struct RecurrentState {
    pub(crate) layers: Vec<LayerRecurrentState>,
    /// Number of tokens processed so far (for prefill/decode tracking).
    pub(crate) seq_len: usize,
}

impl RecurrentState {
    /// Allocate zeroed recurrent state for all linear attention layers.
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        let num_linear_layers = config.num_hidden_layers - config.num_full_attention_layers();

        let state_size = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let qkv_dim = config.linear_attn_qkv_dim();
        let conv_state_size = qkv_dim * (config.linear_conv_kernel_dim - 1);

        let mut layers = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            let state: CudaSlice<f32> = ctx
                .stream
                .alloc_zeros(state_size)
                .map_err(|e| anyhow::anyhow!("Alloc recurrent state failed: {}", e))?;
            layers.push(LayerRecurrentState {
                state,
                conv_state: DeviceVec::zeros(ctx, conv_state_size)?,
            });
        }

        Ok(Self { layers, seq_len: 0 })
    }

    /// Save all recurrent state to CPU memory for prefix cache reuse.
    /// Returns a snapshot that can be restored later via `restore_snapshot`.
    pub(crate) fn save_snapshot(&self, ctx: &DeviceContext) -> Result<RecurrentSnapshot> {
        let mut gdr_states = Vec::with_capacity(self.layers.len());
        let mut conv_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut gdr_host = vec![0f32; layer.state.len()];
            ctx.stream
                .memcpy_dtoh(&layer.state, &mut gdr_host)
                .map_err(|e| anyhow::anyhow!("D2H recurrent state failed: {}", e))?;
            gdr_states.push(gdr_host);

            let mut conv_host = vec![half::bf16::ZERO; layer.conv_state.data.len()];
            ctx.stream
                .memcpy_dtoh(&layer.conv_state.data, &mut conv_host)
                .map_err(|e| anyhow::anyhow!("D2H conv state failed: {}", e))?;
            conv_states.push(conv_host);
        }
        Ok(RecurrentSnapshot {
            gdr_states,
            conv_states,
            seq_len: self.seq_len,
        })
    }

    /// Restore recurrent state from a CPU snapshot.
    pub(crate) fn restore_snapshot(
        &mut self,
        ctx: &DeviceContext,
        snap: &RecurrentSnapshot,
    ) -> Result<()> {
        self.seq_len = snap.seq_len;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            ctx.stream
                .memcpy_htod(&snap.gdr_states[i], &mut layer.state)
                .map_err(|e| anyhow::anyhow!("H2D recurrent state failed: {}", e))?;
            ctx.stream
                .memcpy_htod(&snap.conv_states[i], &mut layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("H2D conv state failed: {}", e))?;
        }
        Ok(())
    }

    /// Reset all state to zeros for a new generation.
    pub(crate) fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.seq_len = 0;
        for layer in &mut self.layers {
            ctx.stream
                .memset_zeros(&mut layer.state)
                .map_err(|e| anyhow::anyhow!("memset recurrent state failed: {}", e))?;
            ctx.stream
                .memset_zeros(&mut layer.conv_state.data)
                .map_err(|e| anyhow::anyhow!("memset conv state failed: {}", e))?;
        }
        Ok(())
    }
}
