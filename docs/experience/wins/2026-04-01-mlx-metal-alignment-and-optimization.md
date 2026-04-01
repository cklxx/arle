# 2026-04-01 · MLX Metal Alignment And Optimization Log

Goal: align agent-infer's Metal benchmark semantics with `mlx_lm`, then optimize the 4-bit decode hot path with one small, reviewable step per commit.

## Baseline Before Alignment

Model: `mlx-community/Qwen3-0.6B-4bit`

Same machine, same prompt family, `max_tokens=512`, greedy decode:

| Runner | Load | Prompt speed | Generation speed | E2E speed | TTFT | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlx_lm 0.30.4` | `~5.56s` | `~1114 tok/s` | `~316.4 tok/s` | not reported | prompt-only timing | `~0.48 GB` |
| `agent-infer` (before metric fix) | `~2.00s` cached | not reported | **incorrectly reported as E2E** | `~204.0 tok/s` | `~2509ms` | `~802 MB` |

Diagnosis:

- The old `metal_bench` mixed prompt prefill and decode into one `tok/s` number.
- `mlx_lm` reports prompt speed and generation speed separately.
- `metal_bench --max-tokens` was exposed in CLI but ignored by the backend; Metal generation still used a hardcoded `512`.

## Step 1: Metric And CLI Alignment

Changes:

- `GenerateResult` now carries `ttft_ms`.
- `GenerateResult` now also carries end-to-end `total_time_ms`.
- Metal backend now computes:
  - `prompt_tps = prompt_tokens / ttft`
  - `generation_tps = generated_tokens / (total_time - ttft)`
- `SamplingParams` now carries an optional `max_new_tokens` override.
- `metal_bench` now forwards `--max-tokens` to the backend.
- `metal_bench` now reports:
  - `prompt_tps`
  - `generation_tps`
  - `e2e_tps`
  - `ttft_ms`
  - `total_time_ms`

Expected outcome:

- apples-to-apples comparison with `mlx_lm`
- no more confusing `generation_tps` vs total-wall-time mismatch
- benchmark knobs match actual runtime behavior

Measured after alignment (`warmup=1`, `runs=3`, `max_tokens=512`):

| Runner | Prompt speed | Generation speed | E2E speed | TTFT |
| --- | ---: | ---: | ---: | ---: |
| `mlx_lm 0.30.4` | `~1114 tok/s` | `~316.4 tok/s` | not reported | prompt-only timing |
| `agent-infer` (aligned) | `~971.8 tok/s` | `~146.5 tok/s` | `~145.7 tok/s` | `~21.4ms` |

Measured end-to-end wall time on the aligned benchmark:

| Runner | Total wall |
| --- | ---: |
| `agent-infer` (aligned) | `~3515ms` mean |

What this clarified:

- prompt/prefill is not the main issue on this workload
- the big gap is decode throughput on the 4-bit path
- optimization work should focus on the quantized per-layer decode hot path, not benchmark plumbing

## Next Optimization Step

Priority candidate for 4-bit Qwen3 Metal path:

- merge quantized `q_proj + k_proj + v_proj` into one `qkv_proj`
- merge quantized `gate_proj + up_proj` into one `gate_up_proj`

Why this is next:

- the 4-bit path never uses the dense fused C++ block
- current fallback issues 5 separate quantized projection matmuls per layer
- collapsing that to 2 matmuls is the smallest likely throughput win without changing model semantics

## Step 2: Merge Quantized Projections In The Metal Fallback

Changes:

- `metal_backend.rs` now models layer input projections explicitly:
  - attention input projections are either split (`q/k/v`) or merged (`qkv`)
  - MLP input projections are either split (`gate/up`) or merged (`gate_up`)
- quantized layer loading now merges compatible rows at load time:
  - `q_proj + k_proj + v_proj -> qkv_proj`
  - `gate_proj + up_proj -> gate_up_proj`
- the quantized Rust fallback now projects once and splits outputs with `split_sections`
- dense weights and the fused C++ path are unchanged

Benchmark (`mlx-community/Qwen3-0.6B-4bit`, `warmup=1`, `runs=3`, `max_tokens=512`):

| Metric | Before step 2 | After step 2 | Delta |
| --- | ---: | ---: | ---: |
| Prompt speed | `~1031.2 tok/s` | `~1426.1 tok/s` | `+38.3%` |
| Generation speed | `~151.6 tok/s` | `~161.3 tok/s` | `+6.4%` |
| E2E speed | `~150.5 tok/s` | `~160.6 tok/s` | `+6.7%` |
| TTFT | `~25.8ms` | `~14.5ms` | `-43.8%` |
| Total wall | `~3439ms` | `~3355ms` | `-2.5%` |

What this improved:

- the 4-bit decode path now launches fewer quantized matmuls per layer
- the optimization is isolated to the quantized Metal fallback
- code structure now matches the intended architecture more closely: merged projections are explicit types, not an implicit future idea

Remaining gap:

- this still trailed the earlier local prose-prompt `mlx_lm` reading (`~316 tok/s`)
- benchmark parity still needed exact token-count prompts and `ignore_eos`, matching `mlx_lm` more closely

## Step 3: Token-Count Benchmark Parity With `mlx_lm`

Changes:

- `MetalBackend` now exposes `generate_from_token_ids(...)` so benchmarking does not have to round-trip through text prompts
- `metal_bench` now benchmarks:
  - exact `--prompt-tokens`
  - exact `--generation-tokens`
  - `ignore_eos: true`
- benchmark runs now fail fast unless they finish with `finish_reason == "length"` and produce the requested token count
- `prompt_tps` and `generation_tps` are now the primary metrics
- repo-specific `e2e` throughput is still reported, but demoted and clearly labeled as `repo_e2e_tps`

Measured on this machine, exact token-count benchmark (`prompt_tokens=20`, `generation_tokens=512`, `warmup=1`, `runs=3`):

| Runner | Prompt speed | Generation speed | Peak memory |
| --- | ---: | ---: | ---: |
| `mlx_lm 0.30.4` benchmark | `~823.4 tok/s` | `~200.1 tok/s` | `~0.486 GB` |
| `agent-infer` token benchmark | `~1443.6 tok/s` | `~174.7 tok/s` | `~800 MB RSS` |

What this clarified:

- the apples-to-apples gap is now much smaller than the earlier prose-prompt comparison suggested
- with the merged quantized projection path, `agent-infer` reaches about `87.3%` of local `mlx_lm` decode throughput on this workload
- remaining gap is real decode efficiency, not mostly benchmark semantics anymore
