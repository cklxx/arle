//! Qwen3 model: full attention transformer.

#[cfg(feature = "cuda")]
#[path = "qwen3/batch_decode.rs"]
mod batch_decode;
#[path = "qwen3/config.rs"]
mod config;
#[path = "qwen3/decode.rs"]
mod decode;
#[path = "qwen3/decode_buffers.rs"]
mod decode_buffers;
#[path = "qwen3/forward.rs"]
mod forward;
#[cfg(feature = "cuda")]
#[path = "qwen3/lora.rs"]
pub mod lora;
#[path = "qwen3/prefill.rs"]
mod prefill;
#[path = "qwen3/weights.rs"]
mod weights;

pub use config::Config;
pub use forward::Qwen3State;
pub use weights::{ModelRuntimeConfig, Qwen3Model};

/// Compile-time upper bound on prefill-request fusion per mixed tick —
/// drives static allocation sizes in `MixedBatchBuffers` and
/// `FlashInferDecodeMetadata`. The runtime value lives on
/// `SchedulerConfig::mixed_prefill_max_reqs` (CLI flag
/// `--mixed-prefill-max-reqs`); this is the hard cap it validates
/// against.
#[cfg(feature = "cuda")]
pub(crate) use batch_decode::MIXED_PREFILL_MAX_REQS_ALLOC_CAP;
