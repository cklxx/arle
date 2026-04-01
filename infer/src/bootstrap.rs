//! Shared model bootstrap and runtime factory helpers.
//!
//! This module owns model discovery plus the common "load tokenizer + weights +
//! runtime options" path used by both the server entrypoint and direct engine
//! constructors.

#[cfg(feature = "cuda")]
use std::fmt;
#[cfg(feature = "cuda")]
use std::path::Path;

#[cfg(feature = "cuda")]
use anyhow::{Result, bail};

#[cfg(feature = "cuda")]
use crate::model::{ModelForward, ModelRuntimeConfig, Qwen3Model, Qwen35Model};
#[cfg(feature = "cuda")]
use crate::model_registry::{ModelArch, detect_arch};
#[cfg(feature = "cuda")]
use crate::scheduler::{Scheduler, SchedulerHandle};
#[cfg(feature = "cuda")]
use crate::tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
}

#[cfg(feature = "cuda")]
impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    pub enable_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

#[cfg(feature = "cuda")]
pub fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(feature = "cuda")]
pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    match detect_arch(model_path)? {
        ModelArch::Qwen3 => Ok(ModelType::Qwen3),
        ModelArch::Qwen35 => Ok(ModelType::Qwen35),
        arch => bail!("model architecture {arch:?} is not supported by the runtime yet"),
    }
}

#[cfg(feature = "cuda")]
pub struct ModelComponents<M> {
    pub model_id: String,
    pub tokenizer: Tokenizer,
    pub model: M,
}

#[cfg(feature = "cuda")]
pub enum LoadedModelComponents {
    Qwen3(ModelComponents<Qwen3Model>),
    Qwen35(ModelComponents<Qwen35Model>),
}

#[cfg(feature = "cuda")]
fn load_model_with<M>(
    model_path: &str,
    options: EngineOptions,
    load_model: impl FnOnce(&str, EngineOptions) -> Result<M>,
) -> Result<ModelComponents<M>> {
    let tokenizer = Tokenizer::from_file(model_path)?;
    let model = load_model(model_path, options)?;
    Ok(ModelComponents {
        model_id: model_id_from_path(model_path),
        tokenizer,
        model,
    })
}

#[cfg(feature = "cuda")]
pub fn load_qwen3_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<ModelComponents<Qwen3Model>> {
    load_model_with(model_path, options, |model_path, options| {
        Qwen3Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.enable_cuda_graph,
            },
        )
    })
}

#[cfg(feature = "cuda")]
pub fn load_qwen35_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<ModelComponents<Qwen35Model>> {
    load_model_with(model_path, options, |model_path, options| {
        Qwen35Model::from_safetensors_with_options(model_path, options.enable_cuda_graph)
    })
}

#[cfg(feature = "cuda")]
pub fn load_model_components(
    model_path: &str,
    options: EngineOptions,
) -> Result<LoadedModelComponents> {
    match detect_model_type(model_path)? {
        ModelType::Qwen3 => Ok(LoadedModelComponents::Qwen3(load_qwen3_components(
            model_path, options,
        )?)),
        ModelType::Qwen35 => Ok(LoadedModelComponents::Qwen35(load_qwen35_components(
            model_path, options,
        )?)),
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle(
    components: LoadedModelComponents,
    num_slots: usize,
    seed: u64,
    max_seq_len: Option<usize>,
) -> Result<SchedulerHandle> {
    match components {
        LoadedModelComponents::Qwen3(components) => {
            spawn_scheduler_for_model(components, num_slots, seed, max_seq_len)
        }
        LoadedModelComponents::Qwen35(components) => {
            spawn_scheduler_for_model(components, num_slots, seed, max_seq_len)
        }
    }
}

#[cfg(feature = "cuda")]
pub fn spawn_scheduler_handle_from_path(
    model_path: &str,
    num_slots: usize,
    seed: u64,
    options: EngineOptions,
    max_seq_len: Option<usize>,
) -> Result<SchedulerHandle> {
    let components = load_model_components(model_path, options)?;
    spawn_scheduler_handle(components, num_slots, seed, max_seq_len)
}

#[cfg(feature = "cuda")]
fn spawn_scheduler_for_model<M: ModelForward + 'static>(
    components: ModelComponents<M>,
    num_slots: usize,
    seed: u64,
    max_seq_len: Option<usize>,
) -> Result<SchedulerHandle> {
    let ModelComponents {
        model_id,
        tokenizer,
        model,
    } = components;

    let (scheduler, handle) =
        Scheduler::with_max_seq_len(model, tokenizer, &model_id, num_slots, seed, max_seq_len)?;
    std::thread::spawn(move || scheduler.run());
    Ok(handle)
}
