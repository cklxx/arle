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

Expected outcome:

- apples-to-apples comparison with `mlx_lm`
- no more confusing `generation_tps` vs total-wall-time mismatch
- benchmark knobs match actual runtime behavior

Measured after alignment (`warmup=1`, `runs=3`, `max_tokens=512`):

| Runner | Prompt speed | Generation speed | E2E speed | TTFT |
| --- | ---: | ---: | ---: | ---: |
| `mlx_lm 0.30.4` | `~1114 tok/s` | `~316.4 tok/s` | not reported | prompt-only timing |
| `agent-infer` (aligned) | `~971.8 tok/s` | `~146.5 tok/s` | `~145.7 tok/s` | `~21.4ms` |

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
