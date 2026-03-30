# agent-infer

Pure Rust agent system with multi-turn tool calling, built on [Pegainfer](https://github.com/cklxx/pegainfer) (Rust+CUDA inference) + [Dynamo](https://github.com/cklxx/dynamo) (inference orchestration).

**No Python glue** — direct `ModelForward::forward()` calls from Rust.

## Architecture

```
User Input
     ↓
Rust Agent Binary (src/)
  ├─ ChatML formatter (src/chat.rs)
  ├─ <tool_call> parser (src/chat.rs)
  ├─ Tool executor: shell, python (src/tools.rs)
  └─ Agent loop (src/agent.rs)
     ↓
Pegainfer (Rust library, linked directly)
  ├─ ModelForward::forward()    ← GPU inference
  ├─ KV prefix cache            ← cross-request reuse
  ├─ KV offload (GPU → CPU)     ← long context support
  └─ CUDA Graph                 ← decode acceleration
```

## Quick Start

```bash
# Build
cargo build --release

# Run (interactive REPL)
./target/release/agent-infer --model-path pegainfer/models/Qwen3-8B

# With options
./target/release/agent-infer \
  --model-path pegainfer/models/Qwen3-8B \
  --max-tokens 4096 \
  --max-turns 10 \
  --temperature 0.0
```

## Features

### KV Prefix Cache
Reuses KV cache across multi-turn requests. When a new prompt shares a prefix with the previous one, only the new suffix is computed.

### KV Offload (GPU → CPU)
When GPU memory is full, older KV blocks are offloaded to CPU RAM. Before attention, they're automatically prefetched back. Enables contexts beyond GPU VRAM.

### Partial Prefix Reuse
When conversations diverge, `truncate_to()` keeps the common prefix instead of resetting entirely.

### Dynamo Integration (optional)
```bash
cargo build --release --features dynamo
./target/release/agent-infer --model-path ... --dynamo
```
Registers with Dynamo's distributed runtime for service discovery and KV-aware routing.

## Built-in Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python code via `python3 -c` |
| `shell` | Execute shell commands via `bash -c` |

## Benchmark

```bash
python3 scripts/bench_agent.py pegainfer/models/Qwen3-8B
```

### Test Environment

| Spec | Value |
|------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| CPU | Intel Xeon @ 2.20GHz, 12 cores |
| RAM | 83GB |
| CUDA | 13.0 (Driver 580.82) |
| OS | Linux 6.6.113+ |

### Results

| Model | Prompts | Turns | Tool Calls | KV Hit Rate | Avg Time |
|-------|---------|-------|-----------|-------------|----------|
| Qwen3-4B | 5 | 10 | 8 | 100% | 31.9s |
| Qwen3-8B | 5 | 10 | 11 | 100% | 88.5s |

KV prefix cache saves 12-38% of prefill computation on multi-turn agent conversations.

### Verify KV Cache Correctness

```bash
# Starts pegainfer HTTP server, compares cold vs warm outputs (greedy decoding)
python3 scripts/verify_kv_cache.py http://localhost:8000
```

## Project Structure

```
agent-infer/
├── src/                         # Rust agent binary
│   ├── main.rs                  # CLI + REPL, model loading
│   ├── agent.rs                 # Agent loop: generate → parse → execute → repeat
│   ├── chat.rs                  # ChatML formatter + <tool_call> parser
│   ├── tools.rs                 # Built-in shell/python tools
│   └── dynamo_integration.rs    # Optional Dynamo runtime integration
├── pegainfer/                   # Inference engine (submodule)
│   └── src/model/kv_cache.rs    # KV cache with CPU offload
├── dynamo/                      # Inference orchestration (submodule)
│   └── components/src/dynamo/pegainfer/  # Dynamo backend module
├── scripts/
│   ├── bench_agent.py           # Agent prompt benchmark
│   ├── verify_kv_cache.py       # KV cache correctness test
│   ├── start_pegainfer.sh       # Start inference server
│   └── start_dynamo.sh          # Start Dynamo stack
└── Cargo.toml
```
