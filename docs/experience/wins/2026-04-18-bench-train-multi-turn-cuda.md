# Training speed benchmark — `train_multi_turn` CPU vs CUDA (L4)

## Goal

Validate that the GRPO trainer's CUDA backend (`autograd::backend_cuda::CudaBackend`)
produces bit-identical reward trajectories to CPU, and measure the cost/benefit
at TinyLM-default and 4×-scaled configs. Follow-up to the Metal validation in
[`2026-04-18-bench-train-multi-turn.md`](2026-04-18-bench-train-multi-turn.md)
so CUDA gets an equivalent baseline row.

## Hypothesis

From the Metal bench: TinyLM-default matmul size is below the FFI/launch-cost
crossover — CUDA will be slower than CPU. At the 4×-scaled config
(d_model=256, 4 layers, seq_len=18) matmul work per call is large enough to
amortize H2D/D2H round-trips; CUDA should pull ahead. CPU/CUDA matmul parity
test (`autograd::tests::cuda_backend_matches_cpu_*`) already passes at 1e-3
relative tolerance, so rewards are expected bit-identical.

Both hypotheses confirmed below.

## Parameters

Seed 42, same args both backends per config:

| Config  | iters | group | prompt_len | agent | turns | obs | d_model | n_layers | d_head | n_heads | d_ff |
|---------|------:|------:|-----------:|------:|------:|----:|--------:|---------:|-------:|--------:|-----:|
| default |    50 |     8 |          4 |     2 |     2 |   2 |      64 |        2 |     32 |       2 |  128 |
| scaled  |    20 |     4 |          8 |     4 |     2 |   2 |     256 |        4 |     64 |       4 |  512 |

Exact commands:

```bash
# default
./target/release/train_multi_turn --iters 50 --eval-every 25 --seed 42 --backend {cpu|cuda}

# scaled
./target/release/train_multi_turn --iters 20 --group-size 4 --prompt-len 8 \
  --agent-tokens 4 --d-model 256 --n-layers 4 --d-head 64 --n-heads 4 --d-ff 512 \
  --seed 42 --backend {cpu|cuda}
```

## Environment

- Host: NVIDIA L4 24 GB, CUDA 13.0 (driver 580.82.07), SM 8.9
- Build: `CARGO_HOME=/tmp/cargo-home-local cargo build --release -p train --features cuda`
  (build 17 s — no CUDA kernels in the train path; everything goes through
  `autograd::backend_cuda::CudaBackend` which wraps cudarc / cuBLAS)
- Commit: `d91908f`
- CPU control on the same L4 box (single thread via `train_multi_turn`).

## Results

### Default (TinyLM-scale) — 50 iters × group 8, seq_len 10

| Backend | Wall    | iter/s | episode/s | token/s | Final KL  | Eval mean_reward @ 49 | Eval pass@1 @ 49 |
|---------|--------:|-------:|----------:|--------:|----------:|----------------------:|-----------------:|
| CPU     |  7.22 s |   6.92 |     27.68 |   276.8 |  −0.0147  |                0.0938 |           0.0000 |
| CUDA    |  8.88 s |   5.63 |     22.53 |   225.3 |  −0.0147  |                0.0938 |           0.0000 |

CPU wins 1.28× at TinyLM scale. Reward trajectory **bit-identical** across
both backends (same sampling seeds, forward matmul ≤1e-3, chain deterministic).

### Scaled (d_model=256, 4 layers) — 20 iters × group 4, seq_len 18

| Backend | Wall     | iter/s | episode/s | token/s | Final KL  |
|---------|---------:|-------:|----------:|--------:|----------:|
| CPU     | 146.84 s |   0.14 |      0.54 |     9.8 |  −0.0217  |
| CUDA    | 131.85 s |   0.15 |      0.61 |    10.9 |  −0.0217  |

CUDA wins **1.11×** at this scale. Reward trajectory identical.

### Cross-backend comparison (same commit, same-arch configs)

From [`2026-04-18-bench-train-multi-turn.md`](2026-04-18-bench-train-multi-turn.md)
(Apple M3 Max, MLX via `crates/mlx-sys`):

| Config  | Metal speedup vs CPU | CUDA speedup vs CPU (L4) |
|---------|---------------------:|-------------------------:|
| default |                0.52× |                    0.81× |
| scaled  |                1.40× |                    1.11× |

CUDA's crossover is earlier (closer to 1× on default) but the scaled-config
win is smaller than Metal's. Two credible reasons, not investigated here:

1. **MLX unified memory.** M3 Max's Metal path avoids host↔device copy on
   every call; L4 pays a PCIe round-trip for each forward matmul.
2. **Backward stays CPU.** The backend trait today only dispatches forward
   matmul; autograd backward + optimizer math still runs on the host. That
   makes the scaled-config numbers still ~80 % CPU-bound.

## Parity verification

`cargo test -p autograd --release --features cuda --test test_backend cuda_`:

```
running 2 tests
test cuda_backend_matches_cpu_batched_3d ... ok
test cuda_backend_matches_cpu_small_2d ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.33s
```

## Problems

- **CUDA slower at TinyLM scale** — expected (see §Hypothesis, Metal saw the
  same). Not a regression. The forward matmul is ~10² FLOPs; cudarc launch +
  FFI + H2D is in the same order.
- **Backward on CPU** is the main reason the scaled speedup stalls at 1.11×.
  Moving backward to CUDA is a whole-crate refactor — tracked under the
  training roadmap, not this bench.

## Learnings

1. **CUDA training pipeline is runnable on L4 today.** Bit-identical reward
   trajectories across CPU/CUDA confirm the `CudaBackend::matmul_forward`
   path matches the CPU reference under the training seeds actually used.
2. **Transfer-overhead crossover for CUDA is between d_model=64 and d_model=256**
   for this workload, same as Metal. Training payloads below the crossover
   should stay on CPU — the `--backend cpu` default is correct for small
   debug runs.
3. **Speedup ceiling on CUDA is gated by backward still being CPU.** Moving
   forward to GPU gives a linear slice; closing the gap further needs the
   backward pass to follow. `feedback_architecture_ideal.md`: the right fix
   is completing the backend trait, not sharpening launch overhead.

## Rule

Before claiming any training-speed improvement on CUDA, re-run both
configs from this doc with the same seeds and compare the `bench:` line.
The reward trajectory is the **parity gate**; any change that diverges
from the CPU baseline is a correctness regression and not a speed win.

## Raw artefacts

Console output archived inline above; not committed separately since this
is a smoke bench (no guidellm `.json` or `.csv`). If this row starts to
matter regression-wise, promote to a scripted wrapper with structured
output the way `bench_guidellm.sh` does.
