// CUDA-only modules — excluded when `no-cuda` feature is active.
#[cfg(feature = "cuda")]
pub mod bootstrap;
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod flashinfer_metadata;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod weight_loader;

// Always-available modules (pure Rust, no GPU dependency).
pub mod backend;
pub mod block_manager;
pub mod chat;
pub mod chat_protocol;
pub mod cuda_graph_pool;
pub mod hf_hub;
pub mod http_server;
pub mod logging;
pub mod metal_backend;
pub mod metal_kv_pool;
pub mod metrics;
pub mod model_registry;
pub mod prefix_cache;
pub mod quant;
pub mod sampler;
pub mod scheduler;
pub mod server_engine;
pub mod speculative;
pub mod tensor_parallel;
pub mod tokenizer;
pub mod trace_reporter;
