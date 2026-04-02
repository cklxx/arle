use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use infer::bootstrap::{
    EngineOptions, ServerRuntimeConfig, detect_model_type, spawn_scheduler_handle_from_path,
};
use infer::http_server::build_app;
use infer::logging;
use infer::scheduler::SchedulerConfig;
use infer::trace_reporter::FileReporter;
use log::info;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "infer", about = "Qwen3/3.5 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,

    /// Number of concurrent request slots (each gets its own KV cache)
    #[arg(long, default_value_t = 4)]
    num_slots: usize,

    /// Maximum sequence length (tokens) per KV cache slot. If unset, auto-computed
    /// from available GPU memory to fit all slots without OOM.
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// Prefill chunk cap (tokens) when decode requests are active.
    /// Lower values reduce decode latency at the cost of prefill throughput.
    #[arg(long, default_value_t = 512)]
    decode_prefill_cap: usize,

    /// GPU memory (MB) reserved as headroom when auto-sizing KV cache.
    #[arg(long, default_value_t = 512)]
    gpu_reserved_mb: usize,

    /// KV pool headroom (MB) reserved during pool initialization.
    #[arg(long, default_value_t = 2048)]
    kv_pool_headroom_mb: usize,
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();

    if let Some(ref trace_path) = args.trace_output_path {
        std::fs::create_dir_all(trace_path).expect("Failed to create trace output directory");
        fastrace::set_reporter(
            FileReporter::new(trace_path.clone()),
            fastrace::collector::Config::default(),
        );
        info!("Tracing enabled: output_dir={}", trace_path.display());
    }

    let model_path = args
        .model_path
        .to_str()
        .expect("Model path must be valid UTF-8");
    let model_type = detect_model_type(model_path).expect("Failed to detect model type");
    info!("=== Infer Server - {} (GPU) ===", model_type);
    info!("Loading model...");
    let start = Instant::now();
    info!(
        "Config: model_path={}, cuda_graph={}, num_slots={}",
        args.model_path.display(),
        args.cuda_graph,
        args.num_slots,
    );

    let runtime = ServerRuntimeConfig {
        engine: EngineOptions {
            enable_cuda_graph: args.cuda_graph,
        },
        scheduler: SchedulerConfig {
            decode_active_prefill_cap: args.decode_prefill_cap,
            gpu_reserved_bytes: args.gpu_reserved_mb.saturating_mul(1024 * 1024),
            kv_pool_headroom_bytes: args.kv_pool_headroom_mb.saturating_mul(1024 * 1024),
            ..SchedulerConfig::runtime_defaults(args.num_slots)
        },
        seed: 42,
        max_seq_len: args.max_seq_len,
    };

    let handle =
        spawn_scheduler_handle_from_path(model_path, runtime).expect("Failed to create scheduler");

    info!(
        "Model loaded: elapsed_ms={}, model_id={}",
        start.elapsed().as_millis(),
        handle.model_id()
    );

    let app = build_app(handle);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {addr}: {e}"));
    axum::serve(
        axum::serve::ListenerExt::tap_io(listener, |tcp_stream| {
            let _ = tcp_stream.set_nodelay(true);
        }),
        app,
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .expect("Server error");

    if args.trace_output_path.is_some() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
}
