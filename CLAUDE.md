# agent-infer — Claude Code Config

Assisting **ckl**. Agent contract below.

---

## Execution Workflow

Non-trivial tasks follow phases. **Each phase has a clear exit condition.**

### Phase 1: Explore

**Posture**: Cartographer.

- See a relevant file → trace its callers and dependents before reading it
- Want to write new code → Grep for existing implementations first
- Involves a trait change → list all implementors, mark blast radius
- Ready to conclude → stop. List unread related files. Read them first
- Uncertain → surface as open question, don't guess

### Phase 2: Plan

**Posture**: Architect. Output decisions, not options.

- Start planning → write "how would this fail?" before "how should this work?"
- Plan touches >5 files → question if there's a simpler path
- Hit an irreversible decision (public API, serialization format, trait redesign) → stop, flag it, wait for user confirmation
- Before coding → identify the 3–5 most consequential decisions for THIS task, show choices, wait for approval
- Plan done → align with user before proceeding

### Phase 3: Implement

**Posture**: Contractor. Build to spec. Spec wrong → fix spec first.

- Before writing code → check prior art in `infer/src/` and `docs/` first
- Want to change something outside the plan → stop. Update plan or note it for later
- Completed a logical unit → `cargo check` immediately
- Writing new code → match adjacent code style (error handling, logging, naming)
- Need a new file → find a similar file for structural reference
- All changes done → `cargo clippy --workspace -- -D warnings`

### Phase 4: Verify

**Posture**: Adversary. Assume bugs exist until proven otherwise.

- Code changed → `cargo test --workspace`. Fix before proceeding
- Each diff line → does this serve the goal? No → remove
- New `unwrap()` → can this panic? Under what input?
- GPU-touching code → does it handle CUDA errors, OOM, stream sync?
- Async code → cancel-safe? Race conditions?

### Phase 5: Reflect

**Posture**: Retrospector. Extract rules, not narratives.

- Bug took >1 attempt → write `docs/experience/errors/YYYY-MM-DD-slug.md`
- Approach worked well → write `docs/experience/wins/YYYY-MM-DD-slug.md`
- User corrected you → write feedback memory before resuming

### Skip rules

- **Trivial** (typo, one-liner): Implement + Verify only.
- **Exploration** ("how does X work?"): Phase 1 only.
- **"Just do it"**: Implement + Verify. Note skipped exploration.

---

## Memory

**Always-load**: auto memory + latest 3 from `docs/experience/errors/` and `docs/experience/wins/`.

**On-demand**: `docs/plans/`, full experience entries, `ROADMAP.md`.

---

## Editing Rules

- **Preserve by default**: When modifying existing content, NEVER delete content not explicitly asked to change.
- **Diff before delete**: Before any deletion, show what's being removed and get confirmation.
- **Approach-first for complex features**: Changes touching >3 files or architectural decisions → outline approach (files to modify, key decision, tradeoffs) and wait for approval before coding.

---

## Behavior Rules

- **Self-correction**: On ANY user correction → codify a preventive feedback memory before resuming.
- **GPU/CPU separation**: Always distinguish between CPU-only logic and GPU-required code. Mark GPU-only code with `// GPU required` comments in placeholder implementations.
- **Kernel placeholders**: When a CUDA/Triton kernel is needed but not yet implemented, write a clear stub with the expected signature and a `todo!("GPU required: ...")` body.
- **Opportunistic cleanup**: Reading code and spot something inelegant → fix it in a separate commit, report inline.

---

## Experience Entries

**Error** (`docs/experience/errors/YYYY-MM-DD-slug.md`):
```
# YYYY-MM-DD · Title
## Context
## Root Cause
## Fix
## Rule
```

**Win** (`docs/experience/wins/YYYY-MM-DD-slug.md`):
```
# YYYY-MM-DD · Title
## Context
## What Worked
## Rule
```

---

## Build & Run

```bash
# Build (CPU-only, no CUDA)
cargo build --release

# Build with CUDA
CUDA_HOME=/usr/local/cuda cargo build --release

# Build with Dynamo integration
cargo build --release --features dynamo

# All tests
cargo test --workspace

# Single test
cargo test -p infer -- <test_name>

# Lint
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all -- --check
```

```bash
# Run agent REPL (requires model)
./target/release/agent-infer --model-path /path/to/model

# Run infer HTTP server
./target/release/infer --model-path /path/to/model --port 8000

# Python agent (HTTP mode, points at running infer)
python -m agent_infer --url http://localhost:8000

# Benchmark
python3 scripts/bench_agent.py /path/to/model
python3 kv_cache_benchmark.py
```

### Python tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Architecture

Primary language: **Rust** (inference engine + agent runtime). Secondary: **Python** (tooling, benchmarks, test scripts) + **CUDA C / Triton** (kernels).

### Workspace layout

```
agent-infer/          ← top-level Cargo workspace
├── src/              ← Rust agent binary (ChatML, tool calling, REPL)
├── agent_infer/      ← Python agent package (async HTTP client mode)
├── infer/        ← Inference engine (Rust library + CUDA kernels)
│   ├── src/
│   │   ├── model/       ← Model implementations (Qwen3, Qwen35, ...)
│   │   ├── ops/         ← CUDA-backed tensor ops (attention, linear, norm...)
│   │   ├── scheduler.rs ← Multi-request continuous batching scheduler
│   │   ├── sampler.rs   ← Sampling strategies
│   │   ├── http_server/ ← OpenAI-compatible HTTP API
│   │   └── server_engine.rs ← Single-request inference façade
│   ├── csrc/            ← CUDA C kernels
│   └── tools/triton/    ← Triton Python kernels (AOT compiled)
├── dynamo/           ← Dynamo distributed runtime (submodule)
└── scripts/          ← Benchmark + utility scripts
```

### Key abstractions

- **`ModelForward` trait** (`infer/src/model.rs`) — single `forward()` entry point per model. Implement for each new architecture.
- **`GenerationState` trait** — per-request mutable state (KV cache, recurrent state). Separate from weights.
- **`ServerEngine` trait** — single-request synchronous inference (used by agent binary + tests).
- **`Scheduler`** (`infer/src/scheduler.rs`) — multi-request continuous batching over any `ModelForward`.
- **`SamplingParams`** (`infer/src/sampler.rs`) — sampling config (temperature, top-k, top-p, ...).
- **`SchedulerHandle`** — `Clone + Send` handle for submitting requests from HTTP handlers.

### Model implementation pattern

Every model lives in `infer/src/model/<name>/`:
```
config.rs       ← JSON config parsing
weights.rs      ← safetensors loading
decode_buffers.rs   ← GPU buffers for decode step
prefill_buffers.rs  ← GPU buffers for prefill step (if separate)
forward.rs      ← ModelForward + GenerationState impl
```

### CUDA kernel integration

Kernels live in `infer/csrc/` (CUDA C) and `infer/tools/triton/` (Triton).
FFI bindings are declared in `infer/src/ffi.rs`.
`build.rs` compiles CUDA C and links against pre-compiled Triton binaries.

---

## Key References

[ModelForward trait](infer/src/model.rs) · [Scheduler](infer/src/scheduler.rs) · [KV cache](infer/src/model/kv_cache.rs) · [Attention ops](infer/src/ops/attention.rs) · [HTTP server](infer/src/http_server.rs) · [Roadmap](ROADMAP.md)

---

## What's Implemented (as of 2026-03-31)

| Component | Status |
|-----------|--------|
| Qwen3 model (GQA) | ✅ |
| Qwen3.5 model (hybrid recurrent+attn) | ✅ |
| FlashAttention-2 (Triton, prefill) | ✅ |
| Decode attention kernel (Triton) | ✅ |
| KV cache with CPU offload | ✅ |
| CUDA graph for decode | ✅ |
| CUDA graph batch pool (CPU tracking + GPU stub) | ✅ |
| Continuous batching scheduler | ✅ |
| Chunked prefill (512-token chunks) | ✅ |
| Decode-priority scheduling | ✅ |
| Request priority + backpressure | ✅ |
| top-k / top-p / temperature / min-p sampling | ✅ |
| Repetition / frequency / presence penalties | ✅ |
| OpenAI `/v1/completions` API | ✅ |
| OpenAI `/v1/chat/completions` API | ✅ |
| SSE streaming | ✅ |
| Prometheus `/metrics` endpoint | ✅ |
| Stats `/v1/stats` endpoint | ✅ |
| Model architecture registry (9 architectures) | ✅ |
| Quantization format detection (GPTQ/AWQ/FP8/INT8/GGUF) | ✅ (detection only) |
| Radix tree prefix cache (data structure) | ✅ (CPU, not yet GPU-wired) |
| Paged KV block manager (accounting) | ✅ (CPU, not yet GPU-wired) |
| Speculative decoding framework | ✅ (CPU stubs, GPU pending) |
| Tensor parallel config + sharding math | ✅ (CPU, NCCL stubs) |
| Rust agent binary (tool calling) | ✅ |
| Python agent (async HTTP) | ✅ |
| Dynamo distributed runtime integration | ✅ (optional feature) |
| PagedAttention CUDA kernel | ❌ |
| Llama / DeepSeek / Mistral / Gemma / Phi models | ❌ |
| FlashAttention-3 | ❌ |
| MLA attention (DeepSeek) | ❌ |
| Beam search | ❌ |
| Quantization GPU kernels (GPTQ/AWQ/FP8/INT8) | ❌ |
| NCCL all-reduce / all-gather | ❌ |
| Benchmark suite (TTFT/TBT) | partial |
