mod agent;
mod chat;
mod tools;

#[cfg(feature = "dynamo")]
mod dynamo_integration;

use std::io::{self, BufRead, Write};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use log::info;

use infer::server_engine::{
    EngineOptions, ModelType, Qwen35ServerEngine, RealServerEngine, ServerEngine, detect_model_type,
};

use crate::tools::builtin_tools;

#[derive(Parser)]
#[command(name = "agent-infer", about = "Local LLM agent with tool use")]
struct Args {
    /// Path to model directory (config.json, tokenizer, safetensors)
    #[arg(long)]
    model_path: String,

    /// Maximum agent turns (generate-execute cycles) per query
    #[arg(long, default_value_t = 10)]
    max_turns: usize,

    /// Maximum tokens to generate per turn
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    no_cuda_graph: bool,

    /// Register with Dynamo distributed runtime for service discovery
    /// and KV-aware routing. Requires the `dynamo` feature.
    #[arg(long, default_value_t = false)]
    dynamo: bool,

    /// Max KV cache tokens on GPU. Excess offloads to CPU.
    /// Use a small value (e.g. 512) to test KV offload behavior.
    #[arg(long)]
    max_gpu_kv: Option<usize>,
}

fn run_repl(
    engine: &mut dyn ServerEngine,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!();
    println!("=== agent-infer REPL ===");
    println!("Model: {}", engine.model_id());
    println!(
        "Tools: {}",
        tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Max turns: {}, Max tokens: {}, Temperature: {}",
        max_turns, max_tokens, temperature
    );
    println!("Type your query and press Enter. Type 'quit' or 'exit' to stop.");
    println!();

    loop {
        print!("\x1b[1;32m> \x1b[0m");
        stdout.flush()?;

        let mut input = String::new();
        let bytes = stdin.lock().read_line(&mut input)?;
        if bytes == 0 {
            // EOF
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        let start = Instant::now();
        match agent::run_agent(engine, input, &tools, max_turns, max_tokens, temperature) {
            Ok(response) => {
                println!();
                println!("\x1b[1;34m{}\x1b[0m", response);
                println!();
                let elapsed = start.elapsed();
                println!("\x1b[2m({:.1}s)\x1b[0m", elapsed.as_secs_f64());
                println!();
            }
            Err(e) => {
                eprintln!("\x1b[1;31mError: {e:#}\x1b[0m");
                println!();
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    infer::logging::init_default();

    let args = Args::parse();

    // If --dynamo is passed, load the model, create a scheduler, and register
    // with the Dynamo distributed runtime.
    if args.dynamo {
        #[cfg(feature = "dynamo")]
        {
            use infer::model::{ModelRuntimeConfig, Qwen3Model, Qwen35Model};
            use infer::scheduler::Scheduler;
            use infer::server_engine::{detect_model_type as detect, model_id_from_path};
            use infer::tokenizer::Tokenizer;

            info!("Starting Dynamo distributed runtime registration...");

            let options = EngineOptions {
                enable_cuda_graph: !args.no_cuda_graph,
            };
            let model_type = detect(&args.model_path)?;
            let model_id = model_id_from_path(&args.model_path);
            info!("Detected model type: {}", model_type);
            info!("Loading model from: {}", args.model_path);

            let load_start = Instant::now();

            let handle = match model_type {
                infer::server_engine::ModelType::Qwen3 => {
                    let model = Qwen3Model::from_safetensors_with_runtime(
                        &args.model_path,
                        ModelRuntimeConfig {
                            enable_cuda_graph: options.enable_cuda_graph,
                        },
                    )?;
                    let tokenizer = Tokenizer::from_file(&args.model_path)?;
                    let (scheduler, handle) = Scheduler::new(model, tokenizer, &model_id, 4, 42)?;
                    std::thread::spawn(move || scheduler.run());
                    handle
                }
                infer::server_engine::ModelType::Qwen35 => {
                    let model = Qwen35Model::from_safetensors_with_options(
                        &args.model_path,
                        options.enable_cuda_graph,
                    )?;
                    let tokenizer = Tokenizer::from_file(&args.model_path)?;
                    let (scheduler, handle) = Scheduler::new(model, tokenizer, &model_id, 4, 42)?;
                    std::thread::spawn(move || scheduler.run());
                    handle
                }
            };

            info!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());
            return dynamo_integration::run_dynamo_worker(handle);
        }

        #[cfg(not(feature = "dynamo"))]
        {
            anyhow::bail!(
                "--dynamo flag requires the `dynamo` feature. \
                 Rebuild with: cargo build --features dynamo"
            );
        }
    }

    let options = EngineOptions {
        enable_cuda_graph: !args.no_cuda_graph,
    };

    let model_type = detect_model_type(&args.model_path)?;
    info!("Detected model type: {}", model_type);
    info!("Loading model from: {}", args.model_path);

    let load_start = Instant::now();

    match model_type {
        ModelType::Qwen3 => {
            let mut engine = RealServerEngine::load_with_options(&args.model_path, 42, options)?;
            if let Some(max_kv) = args.max_gpu_kv {
                info!(
                    "Setting max GPU KV to {} tokens (offload test mode)",
                    max_kv
                );
                engine.set_max_gpu_kv(max_kv);
            }
            info!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());
            run_repl(
                &mut engine,
                args.max_turns,
                args.max_tokens,
                args.temperature,
            )?;
        }
        ModelType::Qwen35 => {
            let mut engine = Qwen35ServerEngine::load_with_options(&args.model_path, 42, options)?;
            if let Some(max_kv) = args.max_gpu_kv {
                info!(
                    "Setting max GPU KV to {} tokens (offload test mode)",
                    max_kv
                );
                engine.set_max_gpu_kv(max_kv);
            }
            info!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());
            run_repl(
                &mut engine,
                args.max_turns,
                args.max_tokens,
                args.temperature,
            )?;
        }
    }

    Ok(())
}
