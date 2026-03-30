// CUDA-only modules — excluded when `no-cuda` feature is active.
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod weight_loader;

// Always-available modules (pure Rust, no GPU dependency).
pub mod chat;
pub mod http_server;
pub mod logging;
pub mod block_manager;
pub mod metrics;
pub mod model_registry;
pub mod prefix_cache;
pub mod sampler;
pub mod scheduler;
pub mod server_engine;
pub mod tokenizer;
pub mod trace_reporter;
