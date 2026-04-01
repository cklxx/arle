//! Metal inference benchmark — prompt speed, TTFT, decode throughput, peak RSS.
//!
//! Requires the `metal` feature. Build with:
//!   cargo build --release --no-default-features --features metal,no-cuda --bin metal_bench
//!
//! # Examples
//! ```bash
//! # Basic
//! ./target/release/metal_bench --model models/Qwen3-0.6B-4bit
//!
//! # Full options
//! ./target/release/metal_bench \
//!   --model models/Qwen3-0.6B-4bit \
//!   --max-tokens 256 --warmup 3 --runs 5 --json
//! ```

use std::time::Instant;

use anyhow::Result;
use clap::Parser;

/// Metal backend benchmark: prompt speed, TTFT, decode throughput, peak RSS.
#[derive(Parser)]
#[command(
    name = "metal_bench",
    about = "Metal/MLX inference benchmark (Apple Silicon)"
)]
struct Cli {
    /// Model path (local directory) or HuggingFace repo ID.
    #[arg(long, short)]
    model: String,

    /// Maximum new tokens to generate per run.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Number of warmup runs (excluded from statistics).
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Number of timed runs for statistics.
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Print per-run timing detail to stderr.
    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Emit results as JSON on stdout (suppresses human-readable table).
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() -> Result<()> {
    #[cfg(feature = "metal")]
    return run_bench();

    #[cfg(not(feature = "metal"))]
    {
        eprintln!(
            "metal_bench requires the `metal` feature.\n\
             Rebuild: cargo build --no-default-features --features metal,no-cuda --bin metal_bench --release"
        );
        std::process::exit(1);
    }
}

#[cfg(feature = "metal")]
fn run_bench() -> Result<()> {
    use infer::backend::InferenceBackend;
    use infer::metal_backend::MetalBackend;
    use infer::sampler::SamplingParams;

    let cli = Cli::parse();

    // Prompt exercises both prefill and decode.
    let prompt = "<|im_start|>user\n\
                  Write a detailed explanation of how attention mechanisms work in transformers.\
                  <|im_end|>\n<|im_start|>assistant\n";

    let params = SamplingParams {
        temperature: 0.0, // greedy — deterministic runs
        max_new_tokens: Some(cli.max_tokens),
        ..Default::default()
    };

    // ── Load ─────────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    let mut backend = MetalBackend::new();
    backend.load(std::path::Path::new(&cli.model))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let quant_hint = if cli.model.contains("4bit") || cli.model.contains("4-bit") {
        "4-bit"
    } else if cli.model.contains("8bit") || cli.model.contains("8-bit") {
        "8-bit"
    } else {
        "bf16"
    };

    if !cli.json {
        eprintln!("Loaded {load_ms:.0}ms  [{quant_hint}]");
    }

    // ── Warmup ───────────────────────────────────────────────────────────────
    if !cli.json {
        eprintln!("Warmup ({} runs)…", cli.warmup);
    }
    for _ in 0..cli.warmup {
        backend.generate(prompt, &params)?;
    }

    // ── Timed runs ───────────────────────────────────────────────────────────
    struct Run {
        total_time_ms: f64,
        prompt_tokens: usize,
        tokens: usize,
        prompt_tps: f64,
        generation_tps: f64,
        ttft_ms: f64,
        e2e_tps: f64,
    }

    let mut runs: Vec<Run> = Vec::with_capacity(cli.runs);

    for i in 0..cli.runs {
        let result = backend.generate(prompt, &params)?;
        let e2e_tps = result.completion_tokens as f64 / (result.total_time_ms / 1000.0).max(1e-9);

        if cli.profile || !cli.json {
            eprintln!(
                "  run {:2}: prompt {:3} tok @ {:6.1} tok/s  gen {:4} tok @ {:6.1} tok/s  e2e {:6.1} tok/s  ttft {:6.1}ms  total {:6.1}ms",
                i + 1,
                result.prompt_tokens,
                result.prompt_tps,
                result.completion_tokens,
                result.generation_tps,
                e2e_tps,
                result.ttft_ms,
                result.total_time_ms,
            );
        }

        runs.push(Run {
            total_time_ms: result.total_time_ms,
            prompt_tokens: result.prompt_tokens,
            tokens: result.completion_tokens,
            prompt_tps: result.prompt_tps,
            generation_tps: result.generation_tps,
            ttft_ms: result.ttft_ms,
            e2e_tps,
        });
    }

    // ── Statistics ───────────────────────────────────────────────────────────
    let mut prompt_tps_sorted: Vec<f64> = runs.iter().map(|r| r.prompt_tps).collect();
    prompt_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_prompt_tps = prompt_tps_sorted.iter().sum::<f64>() / prompt_tps_sorted.len() as f64;
    let p50_prompt_tps = percentile(&prompt_tps_sorted, 50.0);
    let p99_prompt_tps = percentile(&prompt_tps_sorted, 99.0);

    let mut generation_tps_sorted: Vec<f64> = runs.iter().map(|r| r.generation_tps).collect();
    generation_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_generation_tps =
        generation_tps_sorted.iter().sum::<f64>() / generation_tps_sorted.len() as f64;
    let p50_generation_tps = percentile(&generation_tps_sorted, 50.0);
    let p99_generation_tps = percentile(&generation_tps_sorted, 99.0);

    let mut e2e_tps_sorted: Vec<f64> = runs.iter().map(|r| r.e2e_tps).collect();
    e2e_tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_e2e_tps = e2e_tps_sorted.iter().sum::<f64>() / e2e_tps_sorted.len() as f64;
    let p50_e2e_tps = percentile(&e2e_tps_sorted, 50.0);
    let p99_e2e_tps = percentile(&e2e_tps_sorted, 99.0);

    let mut ttft_sorted: Vec<f64> = runs.iter().map(|r| r.ttft_ms).collect();
    ttft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ttft_ms = ttft_sorted.iter().sum::<f64>() / ttft_sorted.len() as f64;
    let p50_ttft_ms = percentile(&ttft_sorted, 50.0);
    let p99_ttft_ms = percentile(&ttft_sorted, 99.0);

    let avg_tokens = runs.iter().map(|r| r.tokens).sum::<usize>() / runs.len().max(1);
    let prompt_tokens = runs.first().map_or(0, |r| r.prompt_tokens);
    let mut total_time_ms_sorted: Vec<f64> = runs.iter().map(|r| r.total_time_ms).collect();
    total_time_ms_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_total_time_ms =
        total_time_ms_sorted.iter().sum::<f64>() / total_time_ms_sorted.len() as f64;
    let p50_total_time_ms = percentile(&total_time_ms_sorted, 50.0);
    let p99_total_time_ms = percentile(&total_time_ms_sorted, 99.0);

    let peak_mb = peak_rss_kb() as f64 / 1024.0;

    // ── Output ───────────────────────────────────────────────────────────────
    if cli.json {
        println!(
            "{}",
            serde_json::json!({
                "model": cli.model,
                "quantization": quant_hint,
                "max_tokens": cli.max_tokens,
                "warmup_runs": cli.warmup,
                "timed_runs": cli.runs,
                "load_ms": load_ms,
                "prompt_tokens": prompt_tokens,
                "avg_tokens": avg_tokens,
                "prompt_tps": { "mean": mean_prompt_tps, "p50": p50_prompt_tps, "p99": p99_prompt_tps },
                "generation_tps": { "mean": mean_generation_tps, "p50": p50_generation_tps, "p99": p99_generation_tps },
                "e2e_tps": { "mean": mean_e2e_tps, "p50": p50_e2e_tps, "p99": p99_e2e_tps },
                "ttft_ms": { "mean": mean_ttft_ms, "p50": p50_ttft_ms, "p99": p99_ttft_ms },
                "total_time_ms": { "mean": mean_total_time_ms, "p50": p50_total_time_ms, "p99": p99_total_time_ms },
                "peak_rss_mb": peak_mb,
            })
        );
    } else {
        println!();
        println!("=== Metal Benchmark: {} [{}] ===", cli.model, quant_hint);
        println!("  Warmup / timed  : {} / {}", cli.warmup, cli.runs);
        println!("  Prompt tokens   : {prompt_tokens}");
        println!("  Avg tokens out  : {avg_tokens}");
        println!(
            "  Prompt speed    : {mean_prompt_tps:.1} tok/s mean  |  {p50_prompt_tps:.1} p50  |  {p99_prompt_tps:.1} p99"
        );
        println!(
            "  Generation      : {mean_generation_tps:.1} tok/s mean  |  {p50_generation_tps:.1} p50  |  {p99_generation_tps:.1} p99"
        );
        println!(
            "  End-to-end      : {mean_e2e_tps:.1} tok/s mean  |  {p50_e2e_tps:.1} p50  |  {p99_e2e_tps:.1} p99"
        );
        println!(
            "  TTFT            : {mean_ttft_ms:.0}ms mean  |  {p50_ttft_ms:.0}ms p50  |  {p99_ttft_ms:.0}ms p99"
        );
        println!(
            "  Total wall      : {mean_total_time_ms:.0}ms mean  |  {p50_total_time_ms:.0}ms p50  |  {p99_total_time_ms:.0}ms p99"
        );
        println!("  Peak RSS        : {peak_mb:.0}MB");
        println!(
            "==={}===",
            "=".repeat(cli.model.len() + quant_hint.len() + 4)
        );
    }

    Ok(())
}

/// Linear-interpolation percentile over a *sorted* slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Peak resident-set size in KB (macOS: `ru_maxrss` is in bytes).
#[allow(unused)]
fn peak_rss_kb() -> u64 {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        ru.ru_maxrss as u64 / 1024
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        0
    }
}
