# agent-infer

Pure Rust LLM inference engine with multi-turn agent tool-calling. Built on **Pegainfer** (Rust+CUDA inference) + **Dynamo** (distributed orchestration).

**No Python glue** — GPU inference calls go directly from Rust to CUDA kernels.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/cklxx/agent-infer && cd agent-infer

# 2. Build (CPU-only, no CUDA)
cargo build --release --no-default-features --features no-cuda

# 3. Build with GPU support
CUDA_HOME=/usr/local/cuda cargo build --release

# Run interactive agent REPL
./target/release/agent-infer --model-path /path/to/Qwen3-8B

# Run OpenAI-compatible HTTP server
./target/release/pegainfer --model-path /path/to/Qwen3-8B --port 8000
```

---

## Architecture

```
User
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│  agent-infer binary  (src/)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ ChatML fmt   │  │ tool_call    │  │ Agent loop    │ │
│  │ (chat.rs)    │  │ parser       │  │ gen→parse→exec│ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │  (linked library)
                           ▼
┌─────────────────────────────────────────────────────────┐
│  pegainfer  (pegainfer/src/)                            │
│                                                         │
│  HTTP layer          Scheduler          Sampler         │
│  /v1/completions  ──▶ continuous  ──▶  top-k/p/temp    │
│  /v1/chat/...       batching          min-p/penalties   │
│  /metrics                                               │
│                      ┌──────────┐                       │
│  ModelForward ──────▶│ Qwen3    │ ← GQA                 │
│  (trait)             │ Qwen3.5  │ ← Hybrid recurrent+attn│
│                      └──────────┘                       │
│  KV Cache  ◀──────── GPU ─────────────────────────────  │
│  (+ CPU offload)                                        │
│  Prefix cache   Paged blocks   CUDA graph pool          │
│  (radix tree)   (accounting)   (batch decode)           │
└──────────────────────────┬──────────────────────────────┘
                           │  CUDA kernels
                           ▼
         FlashAttention-2 · RMSNorm · GEMM/GEMV · Sampling
                    (Triton + CUDA C, pegainfer/csrc/)
```

---

## Supported Models

| Model | Attention | Status |
|-------|-----------|--------|
| Qwen3-0.5B / 1.8B / 4B / 8B / 14B / 32B / 72B | GQA | ✅ Full |
| Qwen3.5-4B (hybrid linear + full attention) | HybridGQA | ✅ Full |
| Llama 3 / 4 | GQA | 🔜 Planned (Phase 1) |
| DeepSeek-V3 / R1 | MLA | 🔜 Planned (Phase 1) |
| Mistral / Mixtral | GQA | 🔜 Planned (Phase 1) |
| Gemma 2 / 3 | MHA | 🔜 Planned (Phase 1) |
| Phi-4 | GQA | 🔜 Planned (Phase 1) |

---

## API

Pegainfer exposes an OpenAI-compatible REST API.

### POST /v1/completions

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B",
    "prompt": "The quick brown fox",
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }'
```

**Streaming (SSE):**

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":128,"stream":true}'
```

### POST /v1/chat/completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "What is 2+2?"}
    ],
    "max_tokens": 64
  }'
```

### GET /metrics

Prometheus text format. Compatible with any Prometheus scraper.

```
pegainfer_requests_total{model="Qwen3-8B"} 42
pegainfer_ttft_seconds_bucket{le="0.100"} 38
pegainfer_kv_gpu_utilization{model="Qwen3-8B"} 0.7200
...
```

### GET /v1/stats

Human-readable summary:

```
requests=42 active=2 waiting=0 tokens_out=3891 kv_util=72.0% ttft_p50=85ms ttft_p99=210ms tpot_p50=12ms
```

### Request parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | int | 16 | Maximum output tokens |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | -1 | Top-K truncation (-1 = disabled) |
| `min_p` | float | 0.0 | Min-P filter (0 = disabled) |
| `repetition_penalty` | float | 1.0 | Repetition penalty (>1 discourages repeats) |
| `frequency_penalty` | float | 0.0 | OpenAI-style frequency penalty |
| `presence_penalty` | float | 0.0 | OpenAI-style presence penalty |
| `stop` | list[str] | null | Stop strings |
| `stop_token_ids` | list[int] | null | Stop token IDs |
| `seed` | int | null | RNG seed for deterministic output |
| `stream` | bool | false | Enable SSE streaming |
| `stream_options.include_usage` | bool | false | Include usage stats in stream |

---

## Built-in Agent Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python 3 snippets via `python3 -c` |
| `shell` | Execute shell commands via `bash -c` |

---

## Server Options

```bash
./target/release/pegainfer \
  --model-path /path/to/model  \  # Required
  --port 8000                  \  # Default: 8000
  --num-slots 4                \  # Concurrent request slots (each gets own KV cache)
  --cuda-graph true            \  # CUDA graph decode (default: true)
  --trace-output-path ./traces    # Optional: write Chrome trace JSON files
```

---

## Agent Options

```bash
./target/release/agent-infer \
  --model-path /path/to/model  \  # Required
  --max-turns 10               \  # Agent loop iterations
  --max-tokens 4096            \  # Tokens per turn
  --temperature 0.0            \  # 0.0 = greedy
  --no-cuda-graph              \  # Disable CUDA graph (for debugging)
  --max-gpu-kv 512             \  # Limit GPU KV tokens (tests CPU offload)
  --dynamo                        # Register with Dynamo runtime
```

---

## Features

### KV Prefix Cache
Reuses KV cache across multi-turn conversations. When a new prompt shares a prefix with the previous one, only the new suffix is computed. Saves 12–38% of prefill computation on agent workloads.

### KV Offload (GPU → CPU)
When GPU HBM is full, older KV blocks are migrated to CPU RAM and prefetched back before attention. Enables contexts beyond GPU VRAM capacity.

### Continuous Batching + Chunked Prefill
Scheduler interleaves multiple requests on a single GPU. Long prefills are chunked (512 tokens) so decode steps can run between chunks, keeping decode latency low for concurrent requests.

### CUDA Graph Decode
First decode token captures a CUDA graph; subsequent tokens replay it, eliminating CPU→GPU dispatch overhead per step.

### Dynamo Integration
```bash
cargo build --release --features dynamo
./target/release/agent-infer --model-path ... --dynamo
```
Registers with Dynamo's ETCD service registry and NATS event bus for service discovery and KV-aware routing.

---

## Benchmarks

### Running

```bash
# Agent benchmark (5 prompts, 10 turns each)
python3 scripts/bench_agent.py /path/to/Qwen3-8B

# HTTP server throughput
python3 scripts/bench_throughput.py --num-prompts 1000 --concurrency 32

# KV cache correctness
python3 scripts/verify_kv_cache.py http://localhost:8000
```

### Results (NVIDIA A100-SXM4-40GB)

| Model | Prompts | Turns | Tool Calls | KV Hit Rate | Avg Time |
|-------|---------|-------|-----------|-------------|----------|
| Qwen3-4B | 5 | 10 | 8 | 100% | 31.9s |
| Qwen3-8B | 5 | 10 | 11 | 100% | 88.5s |

Test environment: A100-SXM4-40GB · Intel Xeon @ 2.20GHz · 83GB RAM · CUDA 13.0

---

## Development

### Project Layout

```
agent-infer/
├── src/                         # Rust agent binary
│   ├── main.rs                  # CLI + REPL + Dynamo path
│   ├── agent.rs                 # Agent loop: generate → parse → execute
│   ├── chat.rs                  # ChatML formatter + <tool_call> parser
│   ├── tools.rs                 # shell / python tool execution
│   └── dynamo_integration.rs    # Dynamo runtime bridge
├── pegainfer/                   # Inference engine (Rust library)
│   ├── src/
│   │   ├── model/               # Qwen3, Qwen3.5 implementations
│   │   ├── ops/                 # GPU ops: attention, linear, norm, sampling
│   │   ├── scheduler.rs         # Multi-request continuous batching
│   │   ├── sampler.rs           # Sampling parameters + penalty logic
│   │   ├── http_server.rs       # Axum HTTP server + SSE streaming
│   │   ├── block_manager.rs     # Paged KV block accounting
│   │   ├── prefix_cache.rs      # Radix-tree prefix cache
│   │   ├── metrics.rs           # Prometheus metrics
│   │   ├── model_registry.rs    # Architecture detection
│   │   ├── quant.rs             # Quantization format detection
│   │   ├── speculative.rs       # Speculative decoding framework
│   │   └── tensor_parallel.rs   # TP config + sharding math
│   ├── csrc/                    # CUDA C kernels
│   └── tools/triton/            # Triton Python kernels (AOT compiled)
├── dynamo/                      # Dynamo runtime submodule
├── scripts/                     # Benchmark + utility scripts
└── Cargo.toml
```

### Building

```bash
# CPU-only (CI / macOS / no CUDA)
cargo build --no-default-features --features no-cuda

# GPU (requires CUDA toolkit + nvidia driver)
cargo build --release

# With Dynamo distributed runtime
cargo build --release --features dynamo
```

### Testing

```bash
# All unit tests (CPU-only, fast)
cargo test --no-default-features --features no-cuda --workspace

# GPU integration tests (requires model weights)
PEGAINFER_TEST_MODEL_PATH=pegainfer/models/Qwen3-4B \
  cargo test --release --test e2e

# Lint
cargo clippy --no-default-features --features no-cuda --workspace -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Adding a New Model

1. Create `pegainfer/src/model/<name>/` with `config.rs`, `weights.rs`, `forward.rs`
2. Implement the `ModelForward` trait (see `pegainfer/src/model.rs`)
3. Register the architecture string in `pegainfer/src/model_registry.rs`
4. Add `ModelType` variant in `pegainfer/src/server_engine.rs`

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full phased plan.

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Foundation (CPU-verifiable) | 🔄 In progress |
| 1 | Core GPU features (PagedAttn, more models) | 🔜 Planned |
| 2 | Quantization (GPTQ/AWQ/FP8/INT8) | 🔜 Planned |
| 3 | Tensor/Pipeline Parallel | 🔜 Planned |
| 4 | Advanced decoding (beam search, speculative) | 🔜 Planned |
| 5 | Performance optimization | 🔜 Planned |
