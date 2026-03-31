//! HuggingFace Hub automatic model download.
//!
//! Mirrors sglang's `hf_utils.py` approach:
//! 1. If the string looks like an existing local path, use it directly.
//! 2. Otherwise treat it as a HF model ID (`org/repo` or `repo`), download
//!    all relevant files to `~/.cache/huggingface/hub/` and return the cache dir.
//!
//! Relevant files downloaded:
//! - `config.json` — model architecture / hyper-params
//! - `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json`
//! - `*.safetensors` (preferred over pickle)
//! - `model.safetensors.index.json` (sharded weight index, if present)
//! - `generation_config.json` (optional)
//!
//! Authentication: reads `HF_TOKEN` env var, then `~/.cache/huggingface/token`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Repo, RepoType,
};

/// Resolve a model source string to a local directory containing model files.
///
/// # Arguments
/// * `model_id_or_path` — Either an existing local path or a HuggingFace model
///   ID (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`, `"mlx-community/Qwen2.5-0.5B-4bit"`).
///
/// Returns the path to the directory that contains `config.json` and weight files.
pub fn resolve_model_path(model_id_or_path: &str) -> Result<PathBuf> {
    let local = Path::new(model_id_or_path);
    if local.exists() {
        log::info!("Using local model path: {}", local.display());
        return Ok(local.to_path_buf());
    }

    log::info!("Model not found locally — downloading from HuggingFace: {model_id_or_path}");
    download_from_hub(model_id_or_path)
}

/// Download a model from HuggingFace Hub and return the local cache directory.
///
/// Files are stored under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/`.
/// Subsequent calls with the same model ID are served from cache (no re-download).
pub fn download_from_hub(model_id: &str) -> Result<PathBuf> {
    let api = build_api().context("failed to initialise HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    // Fetch repo metadata to discover which files exist.
    let info = repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{model_id}'"))?;

    let filenames: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .collect();

    log::info!(
        "Hub repo '{}' has {} files — selecting required ones",
        model_id,
        filenames.len()
    );

    // ── mandatory files ────────────────────────────────────────────────────
    let mandatory = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    for name in &mandatory {
        if filenames.iter().any(|f| f == name) {
            fetch_file(&repo, name, model_id)?;
        } else {
            log::warn!("'{model_id}': expected file '{name}' not found in repo");
        }
    }

    // ── optional config files ─────────────────────────────────────────────
    let optional = [
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ];
    for name in &optional {
        if filenames.iter().any(|f| f == name) {
            fetch_file(&repo, name, model_id)?;
        }
    }

    // ── weight shards (safetensors preferred, no pickle) ──────────────────
    let weight_files: Vec<&str> = filenames
        .iter()
        .filter(|f| {
            (f.ends_with(".safetensors") || f.ends_with(".bin"))
                // Skip adapter / lora weight files
                && !f.contains("adapter")
                // Prefer safetensors; skip .bin when a .safetensors twin exists
                && !(f.ends_with(".bin") && has_safetensors_twin(&filenames, f))
        })
        .map(|f| f.as_str())
        .collect();

    if weight_files.is_empty() {
        anyhow::bail!(
            "no weight files (.safetensors or .bin) found in HF repo '{model_id}'"
        );
    }

    for name in &weight_files {
        fetch_file(&repo, name, model_id)?;
    }

    // ── derive local cache dir from the first downloaded file ─────────────
    let first = mandatory
        .iter()
        .find(|name| filenames.iter().any(|f| f == *name))
        .unwrap_or(&"config.json");

    // `repo.get()` returns the full path to the cached file; its parent is
    // the snapshot directory that contains all downloaded files.
    let file_path = repo
        .get(first)
        .with_context(|| format!("failed to resolve cache path for '{first}'"))?;

    let cache_dir = file_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| file_path.clone());

    log::info!(
        "Model '{}' ready at: {}",
        model_id,
        cache_dir.display()
    );
    Ok(cache_dir)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn build_api() -> Result<Api> {
    let mut builder = ApiBuilder::new();

    // Honour HF_TOKEN for private / gated models.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            log::debug!("HF_TOKEN found — using for authentication");
            builder = builder.with_token(Some(token));
        }
    }

    builder.build().context("ApiBuilder::build failed")
}

fn fetch_file(repo: &ApiRepo, filename: &str, model_id: &str) -> Result<PathBuf> {
    log::info!("  ↓ {filename}");
    repo.get(filename)
        .with_context(|| format!("failed to download '{filename}' from '{model_id}'"))
}

/// Returns true when there is a `.safetensors` counterpart for a `.bin` file.
/// e.g. `model.bin` → looks for `model.safetensors` in the file list.
fn has_safetensors_twin(all: &[String], bin_file: &str) -> bool {
    let stem = bin_file.strip_suffix(".bin").unwrap_or(bin_file);
    let twin = format!("{stem}.safetensors");
    all.iter().any(|f| f == &twin)
}

#[cfg(test)]
mod tests {
    use super::has_safetensors_twin;

    #[test]
    fn twin_detection() {
        let files = vec![
            "model.bin".to_string(),
            "model.safetensors".to_string(),
            "pytorch_model.bin".to_string(),
        ];
        assert!(has_safetensors_twin(&files, "model.bin"));
        assert!(!has_safetensors_twin(&files, "pytorch_model.bin"));
    }
}
