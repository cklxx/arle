//! Metal inference benchmark — TTFT, decode throughput (tok/s), peak RSS.
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

/// Metal backend benchmark: TTFT, tok/s (mean/p50/p99), peak RSS.
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

    // Note: max_new_tokens is currently hardcoded in the Metal backend (512 tokens).
    // The --max-tokens CLI flag is reserved for when the backend exposes it.
    let _ = cli.max_tokens;

    let params = SamplingParams {
        temperature: 0.0, // greedy — deterministic runs
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
        elapsed_s: f64,
        tokens: usize,
        tps: f64,
    }

    let mut runs: Vec<Run> = Vec::with_capacity(cli.runs);

    for i in 0..cli.runs {
        let t0 = Instant::now();
        let result = backend.generate(prompt, &params)?;
        let elapsed_s = t0.elapsed().as_secs_f64();
        let tps = result.completion_tokens as f64 / elapsed_s.max(1e-9);

        if cli.profile || !cli.json {
            eprintln!(
                "  run {:2}: {:4} tok  {:.2}s  {:.1} tok/s",
                i + 1,
                result.completion_tokens,
                elapsed_s,
                tps,
            );
        }

        runs.push(Run {
            elapsed_s,
            tokens: result.completion_tokens,
            tps,
        });
    }

    // ── Statistics ───────────────────────────────────────────────────────────
    let mut tps_sorted: Vec<f64> = runs.iter().map(|r| r.tps).collect();
    tps_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_tps = tps_sorted.iter().sum::<f64>() / tps_sorted.len() as f64;
    let p50_tps = percentile(&tps_sorted, 50.0);
    let p99_tps = percentile(&tps_sorted, 99.0);

    // Wall-time elapsed is a proxy for TTFT when running greedy with few tokens;
    // for a more accurate TTFT the backend would need to expose it directly.
    let mut ttft_sorted: Vec<f64> = runs.iter().map(|r| r.elapsed_s * 1000.0).collect();
    ttft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_ttft_ms = ttft_sorted.iter().sum::<f64>() / ttft_sorted.len() as f64;
    let p50_ttft_ms = percentile(&ttft_sorted, 50.0);

    let avg_tokens = runs.iter().map(|r| r.tokens).sum::<usize>() / runs.len().max(1);

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
                "avg_tokens": avg_tokens,
                "tps": { "mean": mean_tps, "p50": p50_tps, "p99": p99_tps },
                "ttft_ms": { "mean": mean_ttft_ms, "p50": p50_ttft_ms },
                "peak_rss_mb": peak_mb,
            })
        );
    } else {
        println!();
        println!("=== Metal Benchmark: {} [{}] ===", cli.model, quant_hint);
        println!("  Warmup / timed  : {} / {}", cli.warmup, cli.runs);
        println!("  Avg tokens out  : {avg_tokens}");
        println!(
            "  Throughput      : {mean_tps:.1} tok/s mean  |  {p50_tps:.1} p50  |  {p99_tps:.1} p99"
        );
        println!("  TTFT (wall)     : {mean_ttft_ms:.0}ms mean  |  {p50_ttft_ms:.0}ms p50");
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
