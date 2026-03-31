use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use infer::http_server::build_app;
use infer::logging;
use infer::model::{ModelRuntimeConfig, Qwen3Model, Qwen35Model};
use infer::scheduler::Scheduler;
use infer::server_engine::{EngineOptions, ModelType, detect_model_type, model_id_from_path};
use infer::tokenizer::Tokenizer;
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
    let model_id = model_id_from_path(model_path);

    info!("=== Infer Server - {} (GPU) ===", model_type);
    info!("Loading model...");
    let start = Instant::now();
    info!(
        "Config: model_path={}, cuda_graph={}, num_slots={}",
        args.model_path.display(),
        args.cuda_graph,
        args.num_slots,
    );

    let options = EngineOptions {
        enable_cuda_graph: args.cuda_graph,
    };

    // Load model, create scheduler, and spawn scheduler thread.
    // The scheduler owns the model and state pool; the handle is Send+Clone.
    let handle = match model_type {
        ModelType::Qwen35 => {
            let model =
                Qwen35Model::from_safetensors_with_options(model_path, options.enable_cuda_graph)
                    .expect("Failed to load Qwen3.5 model");
            let tokenizer = Tokenizer::from_file(model_path).expect("Failed to load tokenizer");
            let (scheduler, handle) = Scheduler::with_max_seq_len(
                model,
                tokenizer,
                &model_id,
                args.num_slots,
                42,
                args.max_seq_len,
            )
            .expect("Failed to create scheduler");
            std::thread::spawn(move || scheduler.run());
            handle
        }
        ModelType::Qwen3 => {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: options.enable_cuda_graph,
                },
            )
            .expect("Failed to load Qwen3 model");
            let tokenizer = Tokenizer::from_file(model_path).expect("Failed to load tokenizer");
            let (scheduler, handle) = Scheduler::with_max_seq_len(
                model,
                tokenizer,
                &model_id,
                args.num_slots,
                42,
                args.max_seq_len,
            )
            .expect("Failed to create scheduler");
            std::thread::spawn(move || scheduler.run());
            handle
        }
    };

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
