//! Metal inference benchmark — prefill and decode throughput on Apple Silicon.
//!
//! Requires the `metal` feature. Build with:
//!   cargo build --release --no-default-features --features metal,no-cuda -p infer --lib --bin metal_bench
//!
//! Run:
//!   ./target/release/metal_bench --model Qwen/Qwen2.5-0.5B-Instruct
//!   ./target/release/metal_bench --model models/Qwen3-4B --prompt-len 256 --warmup 2 --repetitions 5

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use infer::backend::InferenceBackend;
use infer::metal_backend::MetalBackend;
use infer::sampler::SamplingParams;

#[derive(Parser)]
#[command(name = "metal_bench", about = "Metal/MLX inference throughput benchmark (Apple Silicon)")]
struct Cli {
    /// Model path or HuggingFace repo ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)
    #[arg(long, short)]
    model: String,

    /// Approximate prompt length: repeats a filler phrase this many times
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,

    /// Number of warmup runs (not included in measurements)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of timed measurement runs
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Build a synthetic prompt by repeating a neutral sentence.
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(cli.prompt_len.max(1));

    // Greedy decoding with ignore_eos for fully deterministic, fixed-length runs.
    let params = SamplingParams {
        temperature: 0.0,
        ignore_eos: true,
        ..SamplingParams::default()
    };

    let mut backend = MetalBackend::new();
    eprint!("loading {}… ", cli.model);
    let t_load = Instant::now();
    backend.load(std::path::Path::new(&cli.model))?;
    eprintln!("done ({:.1}s)", t_load.elapsed().as_secs_f64());

    // Warmup
    if cli.warmup > 0 {
        eprint!("warming up ({} run{})… ", cli.warmup, if cli.warmup == 1 { "" } else { "s" });
        for _ in 0..cli.warmup {
            backend.generate(&prompt, &params)?;
        }
        eprintln!("done");
    }

    // Timed measurement runs
    eprintln!("measuring ({} run{}):", cli.repetitions, if cli.repetitions == 1 { "" } else { "s" });
    let mut prefill_tps_samples = Vec::with_capacity(cli.repetitions);
    let mut decode_tps_samples = Vec::with_capacity(cli.repetitions);
    let mut completion_tokens_samples = Vec::with_capacity(cli.repetitions);

    for i in 0..cli.repetitions {
        let t = Instant::now();
        let result = backend.generate(&prompt, &params)?;
        let elapsed = t.elapsed().as_secs_f64();
        eprintln!(
            "  [{}/{}] prefill {:.1} tok/s  decode {:.1} tok/s  ({} tokens, {:.2}s)",
            i + 1,
            cli.repetitions,
            result.prompt_tps,
            result.generation_tps,
            result.completion_tokens,
            elapsed,
        );
        prefill_tps_samples.push(result.prompt_tps);
        decode_tps_samples.push(result.generation_tps);
        completion_tokens_samples.push(result.completion_tokens);
    }

    let n = cli.repetitions as f64;
    let avg_prefill = prefill_tps_samples.iter().sum::<f64>() / n;
    let avg_decode = decode_tps_samples.iter().sum::<f64>() / n;
    let avg_tokens = completion_tokens_samples.iter().sum::<usize>() as f64 / n;

    // Print final summary to stdout so it can be captured/redirected easily.
    println!();
    println!("── Metal Benchmark Summary ──────────────────────────────");
    println!("model:          {}", cli.model);
    println!("prompt length:  ~{} repetitions", cli.prompt_len);
    println!("output tokens:  {:.0} (avg over {} runs)", avg_tokens, cli.repetitions);
    println!("prefill:        {:.1} tok/s (avg)", avg_prefill);
    println!("decode:         {:.1} tok/s (avg)", avg_decode);

    Ok(())
}
