//! Metal inference backend for Apple Silicon.
//!
//! Uses [`mlx-rs`](https://crates.io/crates/mlx-rs) (Rust bindings to Apple's
//! MLX framework) for Metal-accelerated tensor operations.
//!
//! # Status
//!
//! The load path (model ID → local cache → config + tokenizer) is fully
//! implemented.  The actual forward pass is stubbed with `todo!()` — MLX
//! tensor construction is laid out but the full transformer implementation
//! requires `mlx-rs` to stabilise its higher-level ops API.
//!
//! # Feature flag
//!
//! Compile with `--no-default-features --features metal,no-cuda`.
//!
//! # Example
//! ```no_run
//! use pegainfer::metal_backend::MetalBackend;
//! use pegainfer::backend::InferenceBackend;
//! use std::path::Path;
//!
//! let mut backend = MetalBackend::new();
//! backend.load(Path::new("Qwen/Qwen2.5-0.5B-Instruct")).unwrap();
//! ```

use std::path::{Path, PathBuf};

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
    /// GPU required: populated in `load()` via mlx-rs.
    #[cfg(feature = "metal")]
    weights: Option<MetalWeights>,
    #[cfg(not(feature = "metal"))]
    _weights: (),
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
}

/// Weight tensors loaded from safetensors shards into Metal unified memory.
// GPU required: all fields are mlx-rs Arrays backed by Metal buffers.
#[cfg(feature = "metal")]
pub struct MetalWeights {
    /// Token embedding table — shape [vocab_size, hidden_size].
    pub embed_tokens: Array,
    /// Per-layer attention + MLP weights.
    pub layers: Vec<MetalLayerWeights>,
    /// Final layer-norm scale — shape [hidden_size].
    pub norm: Array,
    /// Output projection (lm_head) — shape [vocab_size, hidden_size].
    pub lm_head: Array,
}

/// Weights for a single transformer layer.
// GPU required: all fields are mlx-rs Arrays.
#[cfg(feature = "metal")]
pub struct MetalLayerWeights {
    // Self-attention
    pub q_proj: Array,
    pub k_proj: Array,
    pub v_proj: Array,
    pub o_proj: Array,
    pub q_norm: Array,
    pub k_norm: Array,
    // MLP
    pub gate_proj: Array,
    pub up_proj: Array,
    pub down_proj: Array,
    // Layer norms
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

impl InferenceBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    /// Load model from `model_path`.
    ///
    /// `model_path` may be:
    /// - An existing local directory (e.g. `/path/to/Qwen2.5-0.5B-Instruct`)
    /// - A HuggingFace model ID (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`)
    ///
    /// If the model ID is not found locally it is automatically downloaded to
    /// `~/.cache/huggingface/hub/` via [`hf_hub::resolve_model_path`].
    fn load(&mut self, model_path: &Path) -> Result<()> {
        // ── 1. Resolve model path (local or HF Hub download) ────────────────
        let path_str = model_path.to_string_lossy();
        let local_dir = hf_hub::resolve_model_path(&path_str)
            .with_context(|| format!("failed to resolve model '{path_str}'"))?;

        log::info!("MetalBackend: loading model from {}", local_dir.display());

        // ── 2. Load tokenizer ────────────────────────────────────────────────
        // Tokenizer::from_file expects the model directory (it appends /tokenizer.json).
        let tokenizer = Tokenizer::from_file(local_dir.to_str().unwrap_or("."))
            .with_context(|| format!("failed to load tokenizer from {}", local_dir.display()))?;

        // ── 3. Parse config.json ─────────────────────────────────────────────
        let config = load_metal_config(&local_dir)
            .with_context(|| "failed to parse config.json")?;

        log::info!(
            "  arch: {} layers, hidden={}, heads={}/{}(kv), vocab={}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
        );

        // ── 4. Load weights into Metal memory ───────────────────────────────
        #[cfg(feature = "metal")]
        {
            let weights = load_metal_weights(&local_dir, &config)
                .with_context(|| "failed to load weights into Metal memory")?;
            self.weights = Some(weights);
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
            let weights = self
                .weights
                .as_ref()
                .context("weights not loaded")?;

            let max_new_tokens = 256usize; // TODO: surface through SamplingParams or a wrapper
            let generated = metal_generate(
                &input_ids,
                weights,
                config,
                params,
                max_new_tokens,
            )?;

            let text = tokenizer.decode(&generated)?;
            return Ok(GenerateResult {
                text,
                prompt_tokens,
                completion_tokens: generated.len(),
                finish_reason: "stop".to_string(),
                prompt_tps: 0.0,
                generation_tps: 0.0,
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

/// Autoregressive generation loop using MLX Metal kernels.
///
/// ## Skeleton (to be completed)
///
/// ```text
/// input_ids → embed → [transformer layer × N] → lm_head → logits
///                                                              ↓
///                                                    sample next token
///                                                              ↓
///                                              append to kv_cache, repeat
/// ```
///
/// Each transformer layer:
/// 1. `x = input_layernorm(x)`
/// 2. `q,k,v = x @ q_proj, k_proj, v_proj`
/// 3. Apply RoPE to q, k
/// 4. Grouped-query attention → `attn_out`
/// 5. `x = x + o_proj(attn_out)`
/// 6. `x = post_attention_layernorm(x)`
/// 7. `gate = silu(x @ gate_proj) * (x @ up_proj)`
/// 8. `x = x + gate @ down_proj`
///
// GPU required: all tensor operations use mlx-rs Arrays on Metal unified memory.
#[cfg(feature = "metal")]
fn metal_generate(
    _input_ids: &[u32],
    _weights: &MetalWeights,
    _config: &MetalModelConfig,
    _params: &SamplingParams,
    _max_new_tokens: usize,
) -> Result<Vec<u32>> {
    // GPU required: implement transformer forward pass + sampling with mlx-rs ops.
    // Key ops needed:
    //   mlx_rs::ops::take()         — token embedding lookup
    //   mlx_rs::ops::matmul()       — Q/K/V projections, MLP
    //   mlx_rs::ops::fast::rope()   — rotary position embeddings
    //   mlx_rs::ops::fast::rms_norm() — layer normalisation
    //   mlx_rs::ops::softmax()      — attention weights
    //   mlx_rs::ops::silu()         — SwiGLU activation
    todo!(
        "GPU required: Metal transformer forward pass not yet implemented. \
         Implement using mlx_rs::ops — see function doc for layer structure."
    )
}

// ── Config loading ─────────────────────────────────────────────────────────────

fn load_metal_config(model_dir: &Path) -> Result<MetalModelConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).context("config.json is not valid JSON")?;

    fn get_usize(v: &serde_json::Value, key: &str, default: usize) -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default)
    }
    fn get_f64(v: &serde_json::Value, key: &str, default: f64) -> f64 {
        v.get(key)
            .and_then(|x| x.as_f64())
            .unwrap_or(default)
    }

    Ok(MetalModelConfig {
        hidden_size:              get_usize(&v, "hidden_size",              2048),
        num_attention_heads:      get_usize(&v, "num_attention_heads",      16),
        num_key_value_heads:      get_usize(&v, "num_key_value_heads",      8),
        num_hidden_layers:        get_usize(&v, "num_hidden_layers",        24),
        intermediate_size:        get_usize(&v, "intermediate_size",        11008),
        vocab_size:               get_usize(&v, "vocab_size",               151936),
        max_position_embeddings:  get_usize(&v, "max_position_embeddings",  32768),
        rms_norm_eps:             get_f64(&v,   "rms_norm_eps",             1e-6),
        rope_theta:               get_f64(&v,   "rope_theta",               10000.0),
    })
}

// ── Weight loading (Metal GPU required) ───────────────────────────────────────

/// Load all safetensors shards from `model_dir` into Metal unified memory.
// GPU required: mlx-rs Array backed by Metal buffers.
#[cfg(feature = "metal")]
fn load_metal_weights(model_dir: &Path, config: &MetalModelConfig) -> Result<MetalWeights> {
    use mlx_rs::Array;

    // Collect shard files (model.safetensors or model-00001-of-00002.safetensors …)
    let shards = collect_safetensors_shards(model_dir)?;
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files found in {}", model_dir.display());
    }

    log::info!("  loading {} weight shard(s) into Metal memory", shards.len());

    // TODO(metal): use mlx_rs::load_safetensors() once that API stabilises.
    // For now, fall back to the safetensors crate to mmap the files, then copy
    // each tensor's raw bytes into an mlx_rs::Array via from_slice().
    //
    // GPU required: the Array constructor transfers data to Metal unified memory.
    todo!(
        "GPU required: safetensors → mlx-rs Array loading not yet implemented. \
         Tracking: https://github.com/oxideai/mlx-rs/issues"
    )
}

/// Return all `.safetensors` shard paths sorted by filename.
#[cfg(feature = "metal")]
fn collect_safetensors_shards(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut shards: Vec<_> = std::fs::read_dir(model_dir)
        .with_context(|| format!("cannot read dir {}", model_dir.display()))?
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
