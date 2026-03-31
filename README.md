# agent-infer

Pure Rust LLM inference engine with multi-turn agent tool-calling. Built on **Infer** (Rust+CUDA inference) + **Dynamo** (distributed orchestration).

**No Python glue** — GPU inference calls go directly from Rust to CUDA kernels.

---

## Performance vs SGLang (Qwen3-4B, A100-SXM4-40GB)

| Metric | agent-infer | SGLang v0.5.9 | Ratio |
|--------|-------------|---------------|-------|
| TTFT (C=1) | **17.9ms** | 40.5ms | **2.3x faster** |
| Throughput C=1 | 119.5 tok/s | 121.0 tok/s | **0.99x** |
| Throughput C=4 | 414.8 tok/s | 419.4 tok/s | **0.99x** |
| Throughput C=8 | 417.2 tok/s | 814.8 tok/s | **0.51x** |
| ITL (decode step) | 8.3ms | 8.0ms | 1.04x |

**Benchmark parameters**: 50 synthetic prompts, max_tokens=128, temperature=0 (greedy), same machine/GPU.

**TTFT lead**: Rust runtime eliminates Python dispatch overhead; CUDA Graph decode removes per-step CPU→GPU launches.

**Single-request parity**: At C=1, agent-infer matches SGLang throughput (119.5 vs 121.0 tok/s) with 2.3x faster first-token latency.

**Concurrent gap**: At C=8, agent-infer's 4-slot scheduler saturates at C=4 throughput. SGLang scales to higher concurrency with its mature batching system. This is the primary optimization target.

---

## Optimization Journey (8-concurrent, A100-40GB)

| Phase | Optimization | Throughput | Delta |
|-------|-------------|-----------|-------|
| Baseline | Per-request decode loop | 128 tok/s | — |
| 1 | Token-level KV pool + FlashInfer paged decode | 434 tok/s | +239% |
| 2 | Buffer pre-allocation (eliminate per-step GPU alloc) | 681 tok/s | +57% |
| 3 | FlashInfer plan once per step (not per layer) | 690 tok/s | +1% |
| 4 | Embedding + logits buffer pre-allocation | 700 tok/s | +1% |
| 5 | CUDA Graph investigation (CPU memcpy constraint) | 700 tok/s | — |
| 6 | CUDA Graph batched decode (one graph per batch_size) | **756 tok/s** | +8% |

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
│  infer  (infer/src/)                                    │
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
                    (Triton + CUDA C, infer/csrc/)
```

---

## Quick Start

```bash
# 1. Clone (with submodules for Dynamo support)
git clone --recurse-submodules https://github.com/cklxx/agent-infer && cd agent-infer

# 2. Build inference server (requires CUDA toolkit)
cd infer && cargo build --release

# 3. Start HTTP server
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  ./target/release/infer \
    --model-path /path/to/Qwen3-4B \
    --port 8000 \
    --num-slots 4 \
    --cuda-graph true

# 4. Test with curl
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"temperature":0}'
```

### Startup Flow

```
main()
  1. Parse args (model-path, port, num-slots, cuda-graph, max-seq-len)
  2. Detect model type (Qwen3 / Qwen3.5) from config.json
  3. Load model weights from safetensors (~2s for 4B on A100)
  4. Create Scheduler with N state slots + TokenKVPool
  5. Warm up CUDA Graphs for batch sizes 1..N (~0.4s)
  6. Bind HTTP server on 0.0.0.0:port
  7. Scheduler run loop on dedicated thread:
       ─ Decode priority: all active decodes run before new prefills
       ─ Chunked prefill: 512 tokens/chunk (64 when decode active)
       ─ Batched decode: FlashInfer paged attention + CUDA Graph replay
```

### Prerequisites

- CUDA Toolkit 12.x
- NVIDIA driver with compute capability >= 8.0 (A100, H100, 4090, ...)
- Rust 1.85+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.10+ with `flashinfer-python` installed (for Triton AOT compilation at build time)
- Model weights in safetensors format (HuggingFace Qwen3 or Qwen3.5)

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

## Features

### KV Prefix Cache
Reuses KV cache across multi-turn conversations. When a new prompt shares a prefix with the previous one, only the new suffix is computed. Saves 12–38% of prefill computation on agent workloads.

### KV Offload (GPU → CPU)
When GPU HBM is full, older KV blocks are migrated to CPU RAM and prefetched back before attention. Enables contexts beyond GPU VRAM capacity.

### Continuous Batching + Chunked Prefill
Scheduler interleaves multiple requests on a single GPU. Long prefills are chunked (512 tokens) so decode steps can run between chunks, keeping decode latency low for concurrent requests.

### CUDA Graph Decode
Decode layer loop (36 layers × ~14 kernels = ~504 launches) is captured into CUDA Graphs — one graph per batch_size, cached in a HashMap. First call captures; subsequent calls replay. Eliminates CPU→GPU dispatch overhead per step.

### FlashInfer Batched Decode
Token-level KV pool (SGLang's `TokenToKVPool` pattern, page_size=1) enables `BatchDecodeWithPagedKVCacheDispatched` across all requests in a single kernel launch. FlashInfer plan runs once per step (not per layer) with buffers pre-allocated at startup.

### Dynamo Integration
```bash
cargo build --release --features dynamo
./target/release/agent-infer --model-path ... --dynamo
```
Registers with Dynamo's ETCD service registry and NATS event bus for service discovery and KV-aware routing.

---

## API

Infer exposes an OpenAI-compatible REST API.

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
infer_requests_total{model="Qwen3-8B"} 42
infer_ttft_seconds_bucket{le="0.100"} 38
infer_kv_gpu_utilization{model="Qwen3-8B"} 0.7200
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

## Benchmarks

### Running

```bash
# Start server
./target/release/infer --model-path models/Qwen3-4B --port 8000 --num-slots 4 --cuda-graph true

# Throughput benchmark (synthetic, aligned with SGLang comparison)
python3 scripts/bench_throughput.py --url http://localhost:8000 \
  --dataset synthetic --num-prompts 50 --concurrency 4 --max-tokens 128 --temperature 0

# Agent benchmark (multi-turn tool calling)
python3 scripts/bench_agent.py /path/to/Qwen3-8B

# Multi-request stress test
python3 scripts/bench_multi_request.py --url http://localhost:8000
```

### Throughput Results (Qwen3-4B, A100-SXM4-40GB)

| Concurrency | agent-infer tok/s | SGLang tok/s | TTFT (ours) | TTFT (SGLang) |
|-------------|-------------------|--------------|-------------|---------------|
| 1 | 119.5 | 121.0 | **17.9ms** | 40.5ms |
| 4 | 414.8 | 419.4 | **53.1ms** | 137.0ms |
| 8 | 417.2 | 814.8 | 1123ms | 78.1ms |

### Agent Benchmark Results

| Model | Prompts | Turns | Tool Calls | KV Hit Rate | Avg Time |
|-------|---------|-------|-----------|-------------|----------|
| Qwen3-4B | 5 | 10 | 8 | 100% | 31.9s |
| Qwen3-8B | 5 | 10 | 11 | 100% | 88.5s |

### Test Environment

- GPU: NVIDIA A100-SXM4-40GB
- Driver: 580.82.07 / CUDA 13.0
- SGLang: v0.5.9 (FlashInfer backend, `--disable-radix-cache`)
- agent-infer: `--num-slots 4 --cuda-graph true`
- Benchmark: `bench_throughput.py --dataset synthetic --num-prompts 50 --max-tokens 128 --temperature 0 --seed 42`

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
├── infer/                       # Inference engine (Rust library)
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

### Server Options

```bash
./target/release/infer \
  --model-path /path/to/model  \  # Required
  --port 8000                  \  # Default: 8000
  --num-slots 4                \  # Concurrent request slots (each gets own KV cache)
  --cuda-graph true            \  # CUDA graph decode (default: true)
  --trace-output-path ./traces    # Optional: write Chrome trace JSON files
```

### Agent Options

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

### Built-in Agent Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python 3 snippets via `python3 -c` |
| `shell` | Execute shell commands via `bash -c` |

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
PEGAINFER_TEST_MODEL_PATH=infer/models/Qwen3-4B \
  cargo test --release --test e2e

# Lint
cargo clippy --no-default-features --features no-cuda --workspace -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Adding a New Model

1. Create `infer/src/model/<name>/` with `config.rs`, `weights.rs`, `forward.rs`
2. Implement the `ModelForward` trait (see `infer/src/model.rs`)
3. Register the architecture string in `infer/src/model_registry.rs`
4. Add `ModelType` variant in `infer/src/server_engine.rs`

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

---

## License

MIT
