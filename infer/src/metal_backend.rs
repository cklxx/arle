//! Metal inference backend for Apple Silicon.
//!
//! Uses [`mlx-rs`](https://crates.io/crates/mlx-rs) (Rust bindings to Apple's
//! MLX framework) for Metal-accelerated tensor operations.
//!
//! # Architecture
//!
//! Implements a full Qwen2.5 transformer forward pass:
//! - safetensors weight loading into MLX unified memory
//! - RMSNorm, RoPE, GQA attention, SwiGLU MLP
//! - Simple append-based KV cache (one buffer per generate call)
//! - Greedy / temperature + top-p sampling
//!
//! # Feature flag
//!
//! Compile with `--no-default-features --features metal,no-cuda`.
//!
//! # Example
//! ```no_run
//! use infer::metal_backend::MetalBackend;
//! use infer::backend::InferenceBackend;
//! use std::path::Path;
//!
//! let mut backend = MetalBackend::new();
//! backend.load(Path::new("Qwen/Qwen2.5-0.5B-Instruct")).unwrap();
//! ```

use std::path::{Path, PathBuf};
#[cfg(feature = "metal")]
use std::time::Instant;

use anyhow::{Context, Result};

use crate::{
    backend::{GenerateResult, InferenceBackend},
    hf_hub,
    sampler::SamplingParams,
    tokenizer::Tokenizer,
};

// ── mlx-rs types (Metal GPU required) ────────────────────────────────────────
#[cfg(feature = "metal")]
use mlx_rs::Array;

/// Apple Silicon Metal inference backend.
///
/// One instance per process; weights are loaded once and kept resident in
/// unified memory (Metal).
pub struct MetalBackend {
    /// Local directory containing config.json, tokenizer.json, and weight shards.
    model_dir: Option<PathBuf>,
    /// Tokenizer loaded from tokenizer.json.
    tokenizer: Option<Tokenizer>,
    /// Loaded model configuration.
    config: Option<MetalModelConfig>,
    /// Weight tensors — resident in Metal unified memory.
    // GPU required: populated in `load()` via mlx-rs.
    #[cfg(feature = "metal")]
    weights: Option<MetalWeights>,
    #[cfg(not(feature = "metal"))]
    _weights: (),
}

/// MLX affine quantization parameters (from `config.json → "quantization"`).
#[derive(Debug, Clone, Copy)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Parsed fields from config.json that the Metal forward pass needs.
#[derive(Debug, Clone)]
pub struct MetalModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Per-head dimension. Qwen3 models set this explicitly in config.json; for
    /// others it falls back to `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    /// Token ID that signals end-of-generation. Defaults to 151645 (Qwen2.5 `<|im_end|>`).
    pub eos_token_id: u32,
    /// MLX 4-bit quantization config, or `None` for full-precision (BF16/F16).
    pub quantization: Option<QuantConfig>,
}

/// A weight matrix that is either full-precision BF16 or MLX affine 4-bit quantized.
///
/// MLX quantization format (affine per-group):
/// - `w`: packed uint32 — shape `[out, in / (32/bits)]`
/// - `scales`: f16/bf16 — shape `[out, in / group_size]`
/// - `biases`: f16/bf16 — shape `[out, in / group_size]`
// GPU required: Array is backed by Metal buffers.
#[cfg(feature = "metal")]
pub enum WeightTensor {
    Dense(Array),
    Quantized {
        w: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

/// Weight tensors loaded from safetensors shards into Metal unified memory.
// GPU required: all fields are mlx-rs Arrays backed by Metal buffers.
#[cfg(feature = "metal")]
pub struct MetalWeights {
    /// Token embedding table — shape [vocab_size, hidden_size] (dequantized to float at load).
    pub embed_tokens: Array,
    /// Per-layer attention + MLP weights.
    pub layers: Vec<MetalLayerWeights>,
    /// Final layer-norm scale — shape [hidden_size].
    pub norm: Array,
    /// Output projection (lm_head) — dense or quantized.
    pub lm_head: WeightTensor,
}

/// Weights for a single transformer layer.
// GPU required: all fields are mlx-rs Arrays or WeightTensors.
#[cfg(feature = "metal")]
pub struct MetalLayerWeights {
    // Self-attention projections (possibly quantized)
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    // MLP projections (possibly quantized)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
    // Per-head QK norms (always float)
    pub q_norm: Array,
    pub k_norm: Array,
    // Layer norms (always float)
    pub input_layernorm: Array,
    pub post_attention_layernorm: Array,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            model_dir: None,
            tokenizer: None,
            config: None,
            #[cfg(feature = "metal")]
            weights: None,
            #[cfg(not(feature = "metal"))]
            _weights: (),
        }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

// mlx_rs::Array wraps a *mut c_void (Metal buffer handle). MLX manages its own
// internal locking, so it is safe to send the backend across threads.
// SAFETY: MetalBackend is used from a single inference thread at a time (the
// scheduler ensures exclusive access). No concurrent mutation occurs.
#[cfg(feature = "metal")]
unsafe impl Send for MetalBackend {}
#[cfg(feature = "metal")]
unsafe impl Sync for MetalBackend {}

impl InferenceBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    /// Load model from `model_path`.
    ///
    /// `model_path` may be:
    /// - An existing local directory (e.g. `/path/to/Qwen2.5-0.5B-Instruct`)
    /// - A HuggingFace model ID (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`)
    fn load(&mut self, model_path: &Path) -> Result<()> {
        // ── 1. Resolve model path ────────────────────────────────────────────
        let path_str = model_path.to_string_lossy();
        let local_dir = hf_hub::resolve_model_path(&path_str)
            .with_context(|| format!("failed to resolve model '{path_str}'"))?;

        log::info!("MetalBackend: loading model from {}", local_dir.display());

        // ── 2. Load tokenizer ────────────────────────────────────────────────
        let tokenizer = Tokenizer::from_file(local_dir.to_str().unwrap_or("."))
            .with_context(|| format!("failed to load tokenizer from {}", local_dir.display()))?;

        // ── 3. Parse config.json ─────────────────────────────────────────────
        let config = load_metal_config(&local_dir)?;

        log::info!(
            "  arch: {} layers, hidden={}, heads={}/{}(kv), vocab={}, eos={}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
            config.eos_token_id,
        );

        // ── 4. Load weights into Metal memory ───────────────────────────────
        #[cfg(feature = "metal")]
        {
            let weights = load_metal_weights(&local_dir, &config)
                .with_context(|| "failed to load weights into Metal memory")?;
            self.weights = Some(weights);
            log::info!("  weights loaded into Metal unified memory");
        }
        #[cfg(not(feature = "metal"))]
        {
            log::warn!(
                "MetalBackend: compiled without 'metal' feature — \
                 weights not loaded into GPU. Rebuild with --features metal,no-cuda."
            );
        }

        self.tokenizer = Some(tokenizer);
        self.config = Some(config);
        self.model_dir = Some(local_dir);
        Ok(())
    }

    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerateResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("model not loaded — call load() first")?;
        let config = self.config.as_ref().unwrap();

        // ── Tokenise ──────────────────────────────────────────────────────────
        let input_ids = tokenizer.encode(prompt)?;
        let prompt_tokens = input_ids.len();

        // ── Forward pass + sampling (Metal GPU required) ─────────────────────
        #[cfg(feature = "metal")]
        {
            let weights = self.weights.as_ref().context("weights not loaded")?;

            let max_new_tokens = 512usize;
            let t0 = Instant::now();

            let generated =
                metal_generate(&input_ids, weights, config, params, max_new_tokens, t0)?;

            let elapsed = t0.elapsed().as_secs_f64();
            let gen_tps = generated.len() as f64 / elapsed.max(1e-9);

            let text = tokenizer.decode(&generated)?;
            return Ok(GenerateResult {
                text,
                prompt_tokens,
                completion_tokens: generated.len(),
                finish_reason: "stop".to_string(),
                prompt_tps: 0.0,
                generation_tps: gen_tps,
            });
        }

        // ── Fallback when compiled without metal feature ──────────────────────
        #[cfg(not(feature = "metal"))]
        {
            let _ = (tokenizer, config, params, prompt_tokens);
            todo!(
                "Metal GPU required: rebuild with --no-default-features \
                 --features metal,no-cuda to enable Metal inference"
            )
        }
    }
}

// ── Metal forward pass (GPU required) ────────────────────────────────────────

/// Autoregressive generation using MLX Metal kernels.
///
/// Forward pass per step:
/// ```text
/// input_ids → embed → [transformer layer × N] → norm → lm_head → logits → sample
/// ```
///
/// Transformer layer (Qwen2.5):
/// 1. `residual = x`
/// 2. `x = rms_norm(x, input_layernorm)`
/// 3. `q = rms_norm(x @ q_proj.T, q_norm)`, `k = rms_norm(x @ k_proj.T, k_norm)`
/// 4. `v = x @ v_proj.T`
/// 5. Reshape, apply RoPE to q/k, append to KV cache
/// 6. GQA via `fast::scaled_dot_product_attention`
/// 7. `x = residual + attn_out @ o_proj.T`
/// 8. `x = x + silu(x_norm @ gate.T) * (x_norm @ up.T) @ down.T`
// GPU required: all tensor operations use mlx-rs Arrays on Metal unified memory.
#[cfg(feature = "metal")]
fn metal_generate(
    input_ids: &[u32],
    weights: &MetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
) -> Result<Vec<u32>> {
    use mlx_rs::{
        fast,
        nn::silu,
        ops::indexing::take_axis,
        ops::{self, concatenate_axis, reshape, transpose_axes},
        transforms::eval,
    };

    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let eps = config.rms_norm_eps as f32;
    let rope_base = config.rope_theta as f32;
    let eos_id = config.eos_token_id;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Per-layer KV cache: None until first use, then [1, n_kv_heads, kv_len, head_dim]
    let mut kv_cache: Vec<Option<(Array, Array)>> = vec![None; n_layers];

    let mut generated: Vec<u32> = Vec::new();
    let mut first_token_logged = false;
    let mut cache_len: i32 = 0;

    // Prefill on first step, then single-token decode
    let mut current_ids: Vec<u32> = input_ids.to_vec();

    loop {
        let seq = current_ids.len() as i32;

        // ── Embedding lookup ─────────────────────────────────────────────────
        // Wrap token IDs into a 1-D int32 Array and do a row-gather on the table.
        let idx_i32: Vec<i32> = current_ids.iter().map(|&t| t as i32).collect();
        let idx_arr = Array::from_slice(&idx_i32, &[seq]);
        // embed: [seq, hidden]
        let mut x = take_axis(&weights.embed_tokens, &idx_arr, 0).context("embedding take_axis")?;

        // ── Transformer layers ────────────────────────────────────────────────
        for (li, layer) in weights.layers.iter().enumerate() {
            // 1. Input norm + residual
            let residual = x.clone();
            x = fast::rms_norm(&x, &layer.input_layernorm, eps).context("input_layernorm")?;

            // 2. QKV projections  [seq, n_heads*head_dim], [seq, n_kv_heads*head_dim]
            let q_raw = linear(&x, &layer.q_proj)?;
            let k_raw = linear(&x, &layer.k_proj)?;
            let v_raw = linear(&x, &layer.v_proj)?;

            // 3. Per-head QK norm (Qwen2.5): reshape → rms_norm → reshape
            let q_raw = apply_head_norm(&q_raw, seq, n_heads, head_dim, &layer.q_norm, eps)?;
            let k_raw = apply_head_norm(&k_raw, seq, n_kv_heads, head_dim, &layer.k_norm, eps)?;

            // 4. Reshape to [1, seq, n_heads, head_dim] for RoPE
            let q = reshape(&q_raw, &[1, seq, n_heads, head_dim]).context("reshape q")?;
            let k = reshape(&k_raw, &[1, seq, n_kv_heads, head_dim]).context("reshape k")?;
            let v = reshape(&v_raw, &[1, seq, n_kv_heads, head_dim]).context("reshape v")?;

            // 5. RoPE (input: [1, seq, n_heads, head_dim], traditional=false)
            let q = fast::rope(&q, head_dim, false, rope_base, 1.0f32, cache_len, None)
                .context("rope q")?;
            let k = fast::rope(&k, head_dim, false, rope_base, 1.0f32, cache_len, None)
                .context("rope k")?;

            // 6. Transpose to [1, n_heads, seq, head_dim] for attention
            let q = transpose_axes(&q, &[0, 2, 1, 3]).context("transpose q")?;
            let k = transpose_axes(&k, &[0, 2, 1, 3]).context("transpose k")?;
            let v = transpose_axes(&v, &[0, 2, 1, 3]).context("transpose v")?;

            // 7. KV cache: concatenate along seq axis (axis=2)
            let (k_full, v_full) = match kv_cache[li].take() {
                None => (k, v),
                Some((k_prev, v_prev)) => {
                    let kk = concatenate_axis(&[&k_prev, &k], 2).context("concat k_cache")?;
                    let vv = concatenate_axis(&[&v_prev, &v], 2).context("concat v_cache")?;
                    (kk, vv)
                }
            };

            // 8. Grouped-query attention (Metal-optimised for seq_len=1 decode)
            // q: [1, n_heads, seq, head_dim]
            // k/v: [1, n_kv_heads, total_kv_len, head_dim]
            let use_causal = cache_len == 0 && seq > 1;
            let mask_arg = if use_causal {
                Some(fast::ScaledDotProductAttentionMask::Causal)
            } else {
                None
            };
            let attn_out =
                fast::scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, mask_arg)
                    .context("scaled_dot_product_attention")?;
            // attn_out: [1, n_heads, seq, head_dim]

            kv_cache[li] = Some((k_full, v_full));

            // 9. Reshape back to [seq, hidden] and project
            let attn_out =
                transpose_axes(&attn_out, &[0, 2, 1, 3]).context("transpose attn_out")?;
            let attn_out =
                reshape(&attn_out, &[seq, n_heads * head_dim]).context("reshape attn_out")?;
            let attn_out = linear(&attn_out, &layer.o_proj)?;
            x = ops::add(&residual, &attn_out).context("residual + attn")?;

            // 10. Post-attention norm + SwiGLU MLP
            let residual2 = x.clone();
            let xn = fast::rms_norm(&x, &layer.post_attention_layernorm, eps)
                .context("post_attn_layernorm")?;

            let gate = silu(&linear(&xn, &layer.gate_proj)?).context("silu gate")?;
            let up = linear(&xn, &layer.up_proj)?;
            let mlp = linear(
                &ops::multiply(&gate, &up).context("gate*up")?,
                &layer.down_proj,
            )?;
            x = ops::add(&residual2, &mlp).context("residual + mlp")?;
        }

        // ── Final norm + lm_head ─────────────────────────────────────────────
        // Take the last token's hidden state for the logit projection.
        let last_idx = Array::from_slice(&[seq - 1], &[1]);
        let last_x = take_axis(&x, &last_idx, 0).context("take last hidden")?;
        let last_x = fast::rms_norm(&last_x, &weights.norm, eps).context("final norm")?;
        let logits = linear(&last_x, &weights.lm_head)?; // [1, vocab]

        // Materialise logits on CPU for sampling
        eval([&logits]).context("eval logits")?;

        // ── Sampling ─────────────────────────────────────────────────────────
        let next_token = sample_token(&logits, params)?;

        if !first_token_logged {
            let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "  TTFT: {ttft_ms:.1}ms (including prefill of {} tokens)",
                input_ids.len()
            );
            first_token_logged = true;
        }

        let stop = next_token == eos_id || params.stop_token_ids.contains(&next_token);
        generated.push(next_token);
        cache_len += seq;

        if stop || generated.len() >= max_new_tokens {
            break;
        }

        // Next step: single new token
        current_ids = vec![next_token];
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let tps = generated.len() as f64 / elapsed.max(1e-9);
    log::info!("  generated {} tokens  ({tps:.1} tok/s)", generated.len());

    Ok(generated)
}

/// `x @ weight.T` — no bias, dispatches to dense matmul or quantized matmul.
#[cfg(feature = "metal")]
#[inline]
fn linear(x: &Array, weight: &WeightTensor) -> Result<Array> {
    match weight {
        WeightTensor::Dense(w) => {
            use mlx_rs::ops::transpose;
            let w_t = transpose(w).context("weight transpose")?;
            mlx_rs::ops::matmul(x, &w_t).context("matmul")
        }
        WeightTensor::Quantized { w, scales, biases, group_size, bits } => {
            // w stored as [out, in] packed uint32; transpose=true → x @ w.T
            mlx_rs::ops::quantized_matmul(
                x,
                w,
                scales,
                biases,
                Some(true),
                Some(*group_size),
                Some(*bits),
            )
            .context("quantized_matmul")
        }
    }
}

/// Reshape `x` to [seq, n_heads, head_dim], apply RMS-norm, reshape back.
#[cfg(feature = "metal")]
fn apply_head_norm(
    x: &Array,
    seq: i32,
    n_heads: i32,
    head_dim: i32,
    norm_weight: &Array,
    eps: f32,
) -> Result<Array> {
    use mlx_rs::{fast, ops::reshape};
    let x3 = reshape(x, &[seq, n_heads, head_dim]).context("reshape for head_norm")?;
    let x3 = fast::rms_norm(&x3, norm_weight, eps).context("head rms_norm")?;
    reshape(&x3, &[seq, n_heads * head_dim]).context("reshape after head_norm")
}

/// Sample the next token from `logits` with shape [1, vocab].
#[cfg(feature = "metal")]
fn sample_token(logits: &Array, params: &SamplingParams) -> Result<u32> {
    use mlx_rs::{
        ops::indexing::argmax,
        ops::{multiply, softmax_axis},
        transforms::eval,
    };

    let temp = params.temperature;

    // Greedy
    if temp <= 1e-6 {
        let best = argmax(logits, None).context("argmax")?;
        eval([&best])?;
        return Ok(best.item::<i32>() as u32);
    }

    // Temperature scaling → softmax → float32 probs
    let inv_t: Array = (1.0f32 / temp).into();
    let scaled = multiply(logits, &inv_t).context("temp scale")?;
    let probs = softmax_axis(&scaled, -1, false).context("softmax")?;
    // Cast to f32 for CPU-side sampling (weights may be bf16)
    let probs_f32 = probs.as_type::<f32>().context("probs cast to f32")?;
    eval([&probs_f32])?;
    let probs_slice: &[f32] = probs_f32.as_slice();

    if params.top_p >= 1.0 - 1e-6 || params.top_p <= 0.0 {
        return sample_multinomial(probs_slice);
    }

    // Top-p (nucleus) sampling
    let mut indexed: Vec<(f32, usize)> = probs_slice
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    indexed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top_p = params.top_p;
    let mut cum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, (p, _)) in indexed.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let nucleus = &indexed[..cutoff];
    let sum: f32 = nucleus.iter().map(|(p, _)| p).sum();
    if sum <= 0.0 {
        return Ok(indexed[0].1 as u32);
    }

    let r: f32 = rand::random::<f32>() * sum;
    let mut acc = 0.0f32;
    for (p, idx) in nucleus {
        acc += p;
        if acc >= r {
            return Ok(*idx as u32);
        }
    }
    Ok(nucleus.last().map_or(0, |(_, i)| *i) as u32)
}

/// Multinomial sampling from a flat probability distribution.
#[cfg(feature = "metal")]
fn sample_multinomial(probs: &[f32]) -> Result<u32> {
    let r: f32 = rand::random::<f32>();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return Ok(i as u32);
        }
    }
    Ok((probs.len().saturating_sub(1)) as u32)
}

// ── Config loading ─────────────────────────────────────────────────────────────

fn load_metal_config(model_dir: &Path) -> Result<MetalModelConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;

    let get_usize = |key: &str, default: usize| -> usize {
        v.get(key)
            .and_then(serde_json::Value::as_u64)
            .map_or(default, |x| x as usize)
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        v.get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default)
    };

    // eos_token_id may be a scalar or a list; take the first value.
    let eos_token_id: u32 = v
        .get("eos_token_id")
        .and_then(|e| {
            e.as_u64().map(|n| n as u32).or_else(|| {
                e.as_array()
                    .and_then(|a| a.first())
                    .and_then(serde_json::Value::as_u64)
                    .map(|n| n as u32)
            })
        })
        .unwrap_or(151_645); // Qwen2.5 <|im_end|>

    // MLX-LM 4-bit quantization config lives under "quantization" key.
    let quantization = v.get("quantization").map(|q| QuantConfig {
        group_size: q
            .get("group_size")
            .and_then(serde_json::Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(64),
        bits: q
            .get("bits")
            .and_then(serde_json::Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(4),
    });

    if let Some(qc) = quantization {
        log::info!("  quantization: {} bits, group_size={}", qc.bits, qc.group_size);
    }

    let hidden_size = get_usize("hidden_size", 2048);
    let num_attention_heads = get_usize("num_attention_heads", 16);
    // Use explicit `head_dim` from config when present (Qwen3 sets it independently
    // of hidden_size); fall back to hidden_size / n_heads for Qwen2.5 style models.
    let head_dim = get_usize("head_dim", hidden_size / num_attention_heads);

    Ok(MetalModelConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads: get_usize("num_key_value_heads", 8),
        num_hidden_layers: get_usize("num_hidden_layers", 24),
        intermediate_size: get_usize("intermediate_size", 11_008),
        vocab_size: get_usize("vocab_size", 151_936),
        max_position_embeddings: get_usize("max_position_embeddings", 32_768),
        rms_norm_eps: get_f64("rms_norm_eps", 1e-6),
        rope_theta: get_f64("rope_theta", 1_000_000.0),
        head_dim,
        eos_token_id,
        quantization,
    })
}

// ── Weight loading (Metal GPU required) ───────────────────────────────────────

/// Load safetensors shards into Metal unified memory via mlx-rs.
// GPU required: Array is backed by Metal buffers.
#[cfg(feature = "metal")]
fn load_metal_weights(model_dir: &Path, config: &MetalModelConfig) -> Result<MetalWeights> {
    use mlx_rs::Dtype;
    use std::collections::HashMap;

    let shards = collect_safetensors_shards(model_dir)?;
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files in {}", model_dir.display());
    }
    log::info!("  loading {} shard(s) …", shards.len());

    let mut tensors: HashMap<String, Array> = HashMap::new();

    for shard_path in &shards {
        let bytes = std::fs::read(shard_path)
            .with_context(|| format!("read shard {}", shard_path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse safetensors {}", shard_path.display()))?;

        for (name, view) in st.tensors() {
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => Dtype::Float32,
                safetensors::Dtype::F16 => Dtype::Float16,
                safetensors::Dtype::BF16 => Dtype::Bfloat16,
                safetensors::Dtype::I32 => Dtype::Int32,
                // Packed 4-bit weights stored as uint32 in MLX-LM format
                safetensors::Dtype::U32 => Dtype::Uint32,
                other => anyhow::bail!("unsupported dtype {:?} for '{}'", other, name),
            };
            // SAFETY: `bytes` is alive for this scope; from_raw_data copies into Metal memory.
            let arr = unsafe { Array::from_raw_data(view.data().as_ptr().cast(), &shape, dtype) };
            tensors.insert(name.to_string(), arr);
        }
    }

    log::info!("  parsed {} tensors", tensors.len());

    let get = |name: &str| -> Result<Array> {
        tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing weight '{name}'"))
    };

    // Load a possibly-quantized projection weight.
    // Tries `<base>.scales` to detect MLX-LM 4-bit format.
    let load_proj = |base: &str| -> Result<WeightTensor> {
        if let Some(qc) = config.quantization {
            let scales_key = format!("{base}.scales");
            if let Some(scales) = tensors.get(&scales_key).cloned() {
                let w = tensors
                    .get(&format!("{base}.weight"))
                    .cloned()
                    .with_context(|| format!("missing quantized weight '{base}.weight'"))?;
                let biases = tensors
                    .get(&format!("{base}.biases"))
                    .cloned()
                    .with_context(|| format!("missing quantized biases '{base}.biases'"))?;
                return Ok(WeightTensor::Quantized {
                    w,
                    scales,
                    biases,
                    group_size: qc.group_size,
                    bits: qc.bits,
                });
            }
        }
        Ok(WeightTensor::Dense(get(&format!("{base}.weight"))?))
    };

    // embed_tokens: dequantize at load time if quantized (needed for embedding lookup via take_axis).
    let embed_tokens = {
        let w = get("model.embed_tokens.weight")?;
        if let Some(qc) = config.quantization {
            if let Some(scales) = tensors.get("model.embed_tokens.scales").cloned() {
                let biases = tensors
                    .get("model.embed_tokens.biases")
                    .cloned()
                    .context("missing model.embed_tokens.biases")?;
                log::info!("  dequantizing embed_tokens at load time");
                mlx_rs::ops::dequantize(&w, &scales, &biases, Some(qc.group_size), Some(qc.bits))
                    .context("dequantize embed_tokens")?
            } else {
                w
            }
        } else {
            w
        }
    };

    let norm = get("model.norm.weight")?;

    // lm_head may be weight-tied to embed_tokens; handle both dense and quantized.
    let lm_head = if tensors.contains_key("lm_head.weight") || tensors.contains_key("lm_head.scales") {
        load_proj("lm_head")?
    } else {
        // Weight-tied: share the (already-dequantized) embed_tokens matrix
        WeightTensor::Dense(embed_tokens.clone())
    };

    let n = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        layers.push(MetalLayerWeights {
            q_proj: load_proj(&p("self_attn.q_proj"))?,
            k_proj: load_proj(&p("self_attn.k_proj"))?,
            v_proj: load_proj(&p("self_attn.v_proj"))?,
            o_proj: load_proj(&p("self_attn.o_proj"))?,
            gate_proj: load_proj(&p("mlp.gate_proj"))?,
            up_proj: load_proj(&p("mlp.up_proj"))?,
            down_proj: load_proj(&p("mlp.down_proj"))?,
            q_norm: get(&p("self_attn.q_norm.weight"))?,
            k_norm: get(&p("self_attn.k_norm.weight"))?,
            input_layernorm: get(&p("input_layernorm.weight"))?,
            post_attention_layernorm: get(&p("post_attention_layernorm.weight"))?,
        });
    }

    Ok(MetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
    })
}

/// Sorted list of `.safetensors` shards in `model_dir`.
#[cfg(feature = "metal")]
fn collect_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .with_context(|| format!("read_dir {}", model_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    shards.sort();
    Ok(shards)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use crate::sampler::SamplingParams;

    fn run_bench(model_dir: &std::path::Path, label: &str) {
        if !model_dir.exists() {
            eprintln!("SKIP: model not found at {}", model_dir.display());
            return;
        }

        let mut backend = MetalBackend::new();
        backend.load(model_dir).expect("load model");

        let prompt = "<|im_start|>user\nWrite a short poem about systems programming.\
                      <|im_end|>\n<|im_start|>assistant\n";
        let params = SamplingParams { temperature: 0.0, ..Default::default() };

        // Warm-up pass (fills KV cache, JIT compiles Metal kernels)
        let _ = backend.generate(prompt, &params).expect("warmup");

        // Timed pass
        let t0 = std::time::Instant::now();
        let result = backend.generate(prompt, &params).expect("generate");
        let elapsed = t0.elapsed().as_secs_f64();

        println!("\n=== Metal Benchmark: {label} ===");
        println!("Model:      {}", model_dir.display());
        println!("Tokens out: {}", result.completion_tokens);
        println!("Elapsed:    {elapsed:.2}s");
        println!("Throughput: {:.1} tok/s", result.generation_tps);
        println!("Output:     {:?}", &result.text[..result.text.len().min(200)]);
        println!("=================={}===\n", "=".repeat(label.len()));

        assert!(result.completion_tokens > 0, "should generate at least one token");
    }

    /// Throughput benchmark for a 4-bit quantized MLX model.
    ///
    /// ```bash
    /// cargo test --no-default-features --features metal,no-cuda --release \
    ///     --lib -- bench_4bit --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires Metal GPU + model weights at models/Qwen3-0.6B-4bit"]
    fn bench_4bit() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/Qwen3-0.6B-4bit");
        run_bench(&model_dir, "Qwen3-0.6B 4-bit");
    }

    /// Throughput benchmark for a BF16 MLX model (comparison baseline).
    ///
    /// ```bash
    /// cargo test --no-default-features --features metal,no-cuda --release \
    ///     --lib -- bench_bf16 --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires Metal GPU + model weights at models/Qwen3-0.6B-bf16"]
    fn bench_bf16() {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/Qwen3-0.6B-bf16");
        run_bench(&model_dir, "Qwen3-0.6B BF16");
    }
}
