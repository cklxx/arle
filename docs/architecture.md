# agent-infer Architecture

> Generated 2026-03-31. Read-only analysis of the full Rust + CUDA codebase.

---

## A. High-Level Architecture — HTTP Request to GPU Execution

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           agent-infer REPL (src/)                            │
│   stdin → agent::run_agent() → ServerEngine::complete_stream()               │
│           tools.rs (builtin tools)  chat.rs (ChatML)                         │
│           [optional] dynamo_integration.rs → Dynamo distributed runtime      │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ ServerEngine trait (single-request)
                                 │
┌──────────────────────────────────────────────────────────────────────────────┐
│                       infer HTTP Server (infer/src/main.rs)          │
│                                                                              │
│   HTTP POST /v1/completions                                                  │
│   HTTP POST /v1/chat/completions ──► axum handler (http_server.rs)          │
│   GET  /metrics (Prometheus)         │                                       │
│   GET  /v1/stats                     │                                       │
│                                      ▼                                       │
│                           AppState { SchedulerHandle, ServerMetrics }        │
│                                      │                                       │
│                           SchedulerHandle::submit(IncomingRequest)           │
│                                      │  mpsc::unbounded_channel             │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                    Scheduler<M: ModelForward> (scheduler.rs)                 │
│                           (dedicated blocking thread)                        │
│                                      │                                       │
│   ┌──────────────────────────────────▼──────────────────────────────────┐   │
│   │                        Scheduler::run() loop                        │   │
│   │                                                                     │   │
│   │  1. drain request_rx → waiting: VecDeque<IncomingRequest>          │   │
│   │  2. assign_slots() → active: Vec<ActiveRequest>                    │   │
│   │  3. step():                                                         │   │
│   │     ├─ step_decode_batch()  ← ALL Decoding requests in one pass    │   │
│   │     └─ step_prefill_chunk() or step_new()  ← ONE prefill at a time │   │
│   │  4. cleanup() → send StreamDelta to HTTP handler via mpsc          │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
│                                      │                                       │
│   State Pool [N slots] (N = num_slots, default 4)                           │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│   │ Slot 0   │ │ Slot 1   │ │ Slot 2   │ │ Slot 3   │                      │
│   │ M::State │ │ M::State │ │ M::State │ │ M::State │  ← KV caches,        │
│   │ KVCache  │ │ KVCache  │ │ KVCache  │ │ KVCache  │    decode buffers,   │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘    CUDA graphs       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ModelForward::forward(tokens, state)
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                          Model Layer (model/)                                │
│                                      │                                       │
│         tokens.len() > 1 → Prefill path (batched GEMM, FlashAttention-2)   │
│         tokens.len() == 1 → Decode path (GEMV, CUDA Graph replay)           │
│                                      │                                       │
│   ┌──────────────────┐   ┌───────────────────────────────────────────────┐  │
│   │   Qwen3Model     │   │             Qwen35Model                        │  │
│   │ 32 full-attn     │   │ 32 layers total:                               │  │
│   │ layers (GQA)     │   │  ├─ 24 linear-attn (gated delta rule)          │  │
│   └──────────────────┘   │  └─  8 full-attn (GQA)                        │  │
│                           │ + RecurrentState (persists across decode steps)│  │
│                           └───────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ops::* calls
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                           ops/ GPU Operations                                │
│                                      │                                       │
│  attention: fused_attention_decode_into, prefill_attention_batch             │
│  linear:    gemv (decode), gemm (prefill), fused_mlp_into                   │
│  norm:      rms_norm, fused_add_rms_norm, rms_norm_gated                    │
│  embedding: embedding_batch (prefill), embedding_decode_into                 │
│  recurrent: gated_delta_rule_prefill_chunkwise, conv1d_prefill               │
│  sampling:  argmax, gpu_sample (top-k/p/temperature)                         │
│  element:   silu_mul_batch, add_batch                                        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ ffi.rs unsafe extern "C"
                                       │
┌──────────────────────────────────────┼───────────────────────────────────────┐
│                     CUDA / Triton Kernels                                    │
│                                      │                                       │
│  csrc/ (CUDA C, compiled by build.rs via cc crate + nvcc)                   │
│  ├─ common.cuh          shared utilities                                     │
│  ├─ fused_attention.cu  decode attention (split-KV)                          │
│  ├─ prefill_attention.cu / prefill_attention_hd256.cu                        │
│  ├─ fused_mlp.cu        SwiGLU MLP fusion                                   │
│  ├─ gemv.cu             GEMV for decode linear projections                   │
│  ├─ norm.cu             RMSNorm + fused variants                             │
│  ├─ pos_enc.cu          RoPE positional encoding                             │
│  ├─ conv1d.cu           1D convolution (Qwen3.5 recurrent)                  │
│  ├─ gated_delta_rule.cu gated delta rule (Qwen3.5 linear attn)              │
│  └─ sampling.cu         top-k / top-p GPU sampling                           │
│                                                                              │
│  tools/triton/ (Triton Python, AOT compiled to .cubin / .h)                 │
│  ├─ flash_attention_prefill_kernel.py   FlashAttention-2 prefill             │
│  ├─ flash_attention_prefill_hd256_kernel.py  hd=256 variant                 │
│  ├─ attention_decode_kernel.py          split-KV decode attention            │
│  ├─ attention_reduce_kernel.py          decode attention reduction           │
│  ├─ silu_mul_kernel.py                  fused SiLU·Mul                      │
│  ├─ basic_kernels.py                    add, embedding variants              │
│  └─ gated_delta_rule_chunkwise_kernels.py  Qwen3.5 recurrent prefill        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## B. Module Dependency Graph

```
agent-infer (src/)
├── agent.rs         ← uses ServerEngine trait from infer
├── chat.rs          ← uses infer::chat (ChatML formatting)
├── tools.rs         ← pure Rust, no infer dep
└── dynamo_integration.rs  ← uses SchedulerHandle, Scheduler, model types

infer/src/
├── lib.rs           ← re-exports all public modules
│
├── [CUDA-gated — only compiled with feature = "cuda"]
│   ├── ffi.rs           ← unsafe extern "C" bindings to csrc/ kernels
│   ├── tensor.rs        ← DeviceContext, DeviceVec, DeviceMatrix (cudarc)
│   ├── weight_loader.rs ← safetensors loading + mmap into GPU
│   ├── ops.rs           ← re-exports ops submodules
│   │   ├── ops/attention.rs   ← fused_attention_decode, prefill_attention
│   │   ├── ops/linear.rs      ← gemv, gemm, fused_mlp
│   │   ├── ops/norm.rs        ← rms_norm variants
│   │   ├── ops/embedding.rs   ← token embedding ops
│   │   ├── ops/elementwise.rs ← silu_mul, add
│   │   ├── ops/recurrent.rs   ← gated delta rule, conv1d
│   │   └── ops/sampling.rs    ← gpu_sample, argmax
│   └── model/
│       ├── model.rs           ← ModelForward + GenerationState traits
│       ├── model/kv_cache.rs  ← KVCache (contiguous buffers + CPU offload)
│       ├── model/cuda_graph.rs ← CudaGraphState (capture/replay)
│       ├── model/qwen3.rs     ← pub mod re-export
│       │   model/qwen3/
│       │   ├── config.rs      ← JSON config parsing
│       │   ├── weights.rs     ← safetensors loading → Qwen3Model
│       │   ├── decode_buffers.rs ← GPU buffers for decode step
│       │   ├── prefill.rs     ← prefill forward pass
│       │   ├── decode.rs      ← single-request decode
│       │   ├── batch_decode.rs ← batched decode (GEMM path)
│       │   └── forward.rs     ← ModelForward impl + Qwen3State
│       └── model/qwen35.rs    ← pub mod re-export
│           model/qwen35/
│           ├── config.rs
│           ├── weights.rs     ← Qwen35Model
│           ├── decode_buffers.rs
│           ├── prefill_buffers.rs
│           ├── single_token_buffers.rs ← decode buffers for Qwen35
│           ├── recurrent_state.rs ← RecurrentState (linear attn state)
│           ├── prefill.rs
│           └── forward.rs     ← ModelForward impl + Qwen35State
│
├── [Always compiled — pure Rust, no GPU dependency]
│   ├── backend.rs         ← InferenceBackend trait (load/generate)
│   ├── metal_backend.rs   ← MetalBackend impl (InferenceBackend)
│   │                          depends on: backend, tokenizer, hf_hub, sampler
│   │                          [feature=metal]: mlx-rs
│   ├── sampler.rs         ← SamplingParams struct (pure Rust)
│   ├── tokenizer.rs       ← Tokenizer wrapper (tokenizers crate)
│   ├── chat.rs            ← ChatML message formatting
│   ├── hf_hub.rs          ← HuggingFace Hub auto-download
│   ├── model_registry.rs  ← ModelArch enum, config.json parsing
│   ├── quant.rs           ← Quantization format detection
│   ├── logging.rs         ← logforth setup
│   ├── metrics.rs         ← ServerMetrics (Prometheus counters)
│   ├── trace_reporter.rs  ← fastrace FileReporter
│   │
│   ├── block_manager.rs   ← BlockManager (paged KV accounting, CPU-only)
│   │                          no deps on GPU modules
│   ├── prefix_cache.rs    ← RadixCache (content-addressable tree, CPU-only)
│   │                          no deps on GPU modules
│   ├── cuda_graph_pool.rs ← GraphPool + batch size padding (CPU accounting)
│   │                          GPU stubs behind #[cfg(feature = "cuda")]
│   ├── speculative.rs     ← SpecConfig, verify_tokens (CPU); DraftModel stub
│   ├── tensor_parallel.rs ← TpConfig, sharding math (CPU); NCCL stubs
│   │
│   ├── server_engine.rs   ← ServerEngine trait, GenericServerEngine<M>
│   │                          depends on: model (cuda), sampler, tokenizer
│   ├── scheduler.rs       ← Scheduler<M>, SchedulerHandle, IncomingRequest
│   │                          depends on: model (cuda), server_engine, tokenizer,
│   │                                      sampler, block_manager
│   └── http_server.rs     ← axum Router, SSE streaming
│       http_server/openai_v1.rs ← OpenAI API request/response types
│                          depends on: scheduler, server_engine, metrics, sampler
```

---

## C. Data Flow Diagram — Single Request Lifecycle

```
Client                 HTTP Server             Scheduler              GPU
  │                       │                      │                     │
  │── POST /v1/chat ──────►│                      │                     │
  │                       │                      │                     │
  │                       │ parse ChatCompletionRequest               │
  │                       │ sampling_params_from_request()            │
  │                       │                      │                     │
  │                       │ mpsc::unbounded_channel() → (tx, rx)     │
  │                       │ IncomingRequest{prompt,params,tx}        │
  │                       │──── handle.submit() ──►│                  │
  │                       │                      │                     │
  │ SSE stream open       │                      │ waiting.push_back()│
  │◄── 200 OK ────────────│                      │                     │
  │                       │                      │                     │
  │                       │         ┌────────────┘                     │
  │                       │         │ assign_slots(): find free slot   │
  │                       │         │ tokenizer.encode(prompt)         │
  │                       │         │ → ActiveRequest{slot=N,          │
  │                       │         │   phase=Phase::New}              │
  │                       │         │                                  │
  │                       │         │ step_new(idx):                   │
  │                       │         │  check prefix cache hit/miss    │
  │                       │         │  phase = Prefilling{tokens, 0}  │
  │                       │         │                                  │
  │                       │         │ step_prefill_chunk():            │
  │                       │         │  chunk = tokens[0..512]         │
  │                       │         │  model.forward(chunk, state) ───►│
  │                       │         │  (repeat until all tokens done) │◄─── FlashAttn-2
  │                       │         │  → phase = Decoding              │     (Triton)
  │                       │         │  sample first token              │
  │                       │         │                                  │
  │                       │         │ step_decode_batch():             │
  │                       │         │  model.forward_decode_batch()───►│
  │                       │         │  (all Decoding requests batched) │◄─── GEMV + CUDA
  │                       │         │  sample tokens                   │     Graph replay
  │                       │         │  emit_delta() → tokenizer.decode │
  │◄── SSE data ──────────│◄─ tx.send(StreamDelta{text_delta}) ──────┘
  │                       │         │
  │                       │         │ ... repeat decode steps ...
  │                       │         │
  │                       │         │ EOS or max_tokens reached:
  │                       │         │  finish() → send_finish(reason)
  │◄── SSE [DONE] ────────│◄─ tx.send(StreamDelta{finish_reason})
  │                       │         │
  │                       │         │ slot freed, state.offload_kv_if_needed()
```

**Prefix Cache Reuse** (within Scheduler or GenericServerEngine):
```
New request prompt tokens: [A B C D E F G H]
Cached prompt in slot:     [A B C D X Y]

Common prefix = [A B C D] (len=4)
→ Partial hit: state.truncate_to(4) (keep KV for first 4 tokens)
→ Only prefill [E F G H] (suffix after divergence)
```

---

## D. Dual Backend Architecture

```
                         InferenceBackend trait (backend.rs)
                         ├── load(model_path) → Result<()>
                         ├── generate(prompt, params) → GenerateResult
                         └── name() → &'static str
                                 │
              ┌──────────────────┴────────────────────┐
              │                                        │
    ┌─────────▼──────────┐              ┌─────────────▼─────────────┐
    │   CUDA Backend     │              │   Metal Backend            │
    │  (implicit, via    │              │  metal_backend.rs          │
    │  ModelForward +    │              │  [feature = "metal"]       │
    │  GenericServerEngine│             │                            │
    │  / Scheduler)      │              │  MetalBackend struct:      │
    │                    │              │  - model_dir: PathBuf      │
    │  Feature: "cuda"   │              │  - tokenizer: Tokenizer    │
    │  Deps: cudarc,     │              │  - config: MetalModelConfig│
    │         memmap2    │              │  - weights: MetalWeights   │
    │                    │              │    (mlx-rs Arrays in       │
    │  Supports:         │              │     unified memory)        │
    │  - Multi-request   │              │                            │
    │    Scheduler       │              │  Implements:               │
    │  - Continuous      │              │  - Full Qwen2.5 forward    │
    │    batching        │              │  - RMSNorm, RoPE, GQA attn │
    │  - CUDA Graphs     │              │  - SwiGLU MLP              │
    │  - KV CPU offload  │              │  - Append-based KV cache   │
    │  - Qwen3 + Qwen35  │              │  - Single-request only     │
    └────────────────────┘              └────────────────────────────┘

Build flag selection:
  Linux/GPU: cargo build --release                     → CUDA backend
  macOS/CI:  cargo build --release --no-default-features --features no-cuda
  Apple M*:  cargo build --release --no-default-features --features metal,no-cuda
```

**Code Separation Pattern:**

```rust
// In lib.rs:
#[cfg(feature = "cuda")]          // ← CUDA-only modules
pub mod model;
pub mod ops;
pub mod tensor;
pub mod weight_loader;

pub mod metal_backend;            // ← Always compiled
pub mod backend;                  // ← Trait definition, always compiled

// In metal_backend.rs:
#[cfg(feature = "metal")]
use mlx_rs::Array;                // ← mlx-rs only when feature active

pub struct MetalBackend {
    #[cfg(feature = "metal")]
    weights: Option<MetalWeights>, // ← Metal weights only if feature
    #[cfg(not(feature = "metal"))]
    _weights: (),                  // ← Zero-sized placeholder otherwise
}
```

---

## E. Scheduler State Machine

```
                    ┌─────────────────────┐
                    │     WAITING queue   │
                    │  VecDeque<Incoming> │
                    └──────────┬──────────┘
                               │ assign_slots() — free slot found
                               ▼
                    ┌─────────────────────┐
                    │      Phase::New     │ ◄── ActiveRequest created
                    │  (just assigned)    │
                    └──────────┬──────────┘
                               │ step_new() — compute prefix cache
                               │  - Full hit → slice suffix, skip to Prefilling
                               │  - Partial hit → truncate_to(), slice suffix
                               │  - Miss → state.reset()
                               ▼
                    ┌─────────────────────┐
                    │  Phase::Prefilling  │
                    │  { tokens, progress }│
                    └──────────┬──────────┘
                               │ step_prefill_chunk() per step:
                               │   chunk = tokens[progress..progress+512]
                               │   model.forward(chunk, state)
                               │   progress += 512
                               │
                    ┌──────────┤ progress < total → stay in Prefilling
                    │          │ progress == total → sample first token
                    │          ▼
                    │ ┌─────────────────────┐
                    │ │   Phase::Decoding   │ ← emit first token delta
                    │ └──────────┬──────────┘
                    │            │ step_decode_batch() — EVERY step:
                    │            │   ALL Decoding requests batched together
                    │            │   model.forward_decode_batch(tokens, states)
                    │            │   sample tokens
                    │            │   emit_delta() → StreamDelta via mpsc
                    │            │
                    │ ┌──────────┤ EOS token || max_tokens reached
                    │ │          │ stop sequence found in decoded text
                    │ │          │ client disconnected (tx.is_closed())
                    │ │          ▼
                    │ │ ┌─────────────────────┐
                    │ └─►  Phase::Finished    │ → cleanup(): remove from active
                    │   └──────────┬──────────┘   slot freed for next request
                    │              │ send_finish(FinishReason::Stop|Length)
                    │              ▼
                    │   ┌─────────────────────┐
                    └──►│  Slot returned to   │
                        │  pool (state reused │
                        │  with prefix cache) │
                        └─────────────────────┘

Scheduling priority:
  Each Scheduler::step() does BOTH:
  1. step_decode_batch()  — batch ALL active Decoding requests (priority)
  2. step_prefill_chunk() — exactly ONE prefill chunk (fairness)

  Without the prefill step, decode-priority would completely starve new
  requests. One chunk per step means new requests always make progress.

Preemption (PreemptionMode):
  Recompute (default): evict KV cache, re-prefill on resume (saves GPU mem)
  Swap: KV cache to CPU, swap back on resume (preserves state, costs DRAM)
```

---

## F. KV Cache Management Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Three-Layer KV Cache System                              │
│                                                                              │
│  Layer 1: KVCache (model/kv_cache.rs) — Contiguous GPU Buffers              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Per-slot, per-layer contiguous K/V buffers                          │  │
│  │  k_cache: Vec<DeviceVec>  [num_layers × (num_kv_heads × max_seq × head_dim)]│
│  │  v_cache: Vec<DeviceVec>                                              │  │
│  │                                                                       │  │
│  │  CPU Offload (spill):                                                 │  │
│  │  max_gpu_seq_len: usize  (default = max_seq_len = 32768)             │  │
│  │  When seq_len > max_gpu_seq_len:                                      │  │
│  │    oldest 64-token blocks → k_host/v_host: Vec<Vec<bf16>> (CPU RAM)  │  │
│  │    offloaded_len: usize   tracks tokens on CPU                       │  │
│  │                                                                       │  │
│  │  Before attention:  ensure_on_gpu() — prefetch CPU → GPU            │  │
│  │  After attention:   offload_to_host() — move prefix GPU → CPU       │  │
│  │                                                                       │  │
│  │  Operations:                                                          │  │
│  │    get_cache_mut(layer) → (&mut DeviceVec, &mut DeviceVec)          │  │
│  │    truncate_to(len)     — prefix reuse, no re-prefill needed         │  │
│  │    reset()              — full clear for new request                 │  │
│  │    offload_if_needed()  — after request ends, reclaim GPU memory     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Layer 2: BlockManager (block_manager.rs) — Paged Block Accounting          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  CPU-only accounting — no actual CUDA memory copies                  │  │
│  │  Fixed-size blocks; requests hold a block table (logical → physical) │  │
│  │                                                                       │  │
│  │  BlockId(u32): physical block identifier                             │  │
│  │  BlockLocation: { Gpu, Cpu }                                          │  │
│  │  Block: { id, location, ref_count }                                   │  │
│  │  SwapPlan: { gpu_to_cpu: Vec<(BlockId,BlockId)>,                     │  │
│  │              cpu_to_gpu: Vec<(BlockId,BlockId)> }                    │  │
│  │                                                                       │  │
│  │  Operations:                                                          │  │
│  │    allocate_gpu(n) → Vec<BlockId>                                    │  │
│  │    swap_out(blocks) → SwapPlan   (GPU → CPU, before preemption)      │  │
│  │    swap_in(blocks)  → SwapPlan   (CPU → GPU, on resume)              │  │
│  │    free_gpu/free_cpu(blocks)                                          │  │
│  │    cow_clone(block) → BlockId    (copy-on-write for shared prefix)   │  │
│  │                                                                       │  │
│  │  Status: CPU accounting complete; CUDA swap kernels pending (stub)   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Layer 3: RadixCache (prefix_cache.rs) — Content-Addressable Prefix Tree    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Maps token sequences → BlockId (cross-request prefix sharing)       │  │
│  │                                                                       │  │
│  │  Data model:                                                          │  │
│  │    Root                                                               │  │
│  │    ├── [tok0,tok1,tok2] → BlockId(0)   "common system prompt"       │  │
│  │    │   ├── [tok3,tok4]  → BlockId(1)   "request A continuation"     │  │
│  │    │   └── [tok3,tok5]  → BlockId(2)   "request B continuation"     │  │
│  │    └── [tok6]           → BlockId(3)                                 │  │
│  │                                                                       │  │
│  │  Node: { tokens, block_id, ref_count, last_access, children }       │  │
│  │  Granularity: block_size tokens per node (fractional tail not cached)│  │
│  │  Eviction: LRU leaves with ref_count == 0                            │  │
│  │                                                                       │  │
│  │  Operations:                                                          │  │
│  │    insert(tokens, block_id)                                           │  │
│  │    lookup(tokens) → Vec<BlockId>   (longest prefix match)           │  │
│  │    evict(n) → Vec<BlockId>         (LRU eviction)                   │  │
│  │    pin/unpin(node)                 (ref counting)                    │  │
│  │                                                                       │  │
│  │  Status: CPU data structure complete; GPU-wiring to KVCache pending  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Simple Prefix Cache (in GenericServerEngine / Scheduler per-slot):         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  cached_prompt: Vec<u32>  — tokens of last request in this slot      │  │
│  │  Linear common-prefix comparison on each new request                  │  │
│  │  Full hit → reuse all KV; Partial hit → truncate_to(prefix_len)     │  │
│  │  This is the ACTIVE prefix cache; RadixCache is not yet GPU-wired    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## G. Module Responsibilities (One-Liner Each)

### infer/src/ — Core Inference Engine

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | Module registration; feature-gates CUDA modules |
| `model.rs` | `ModelForward` + `GenerationState` traits — the core abstraction |
| `model/kv_cache.rs` | Contiguous K/V GPU buffers with CPU offload at 64-token block granularity |
| `model/cuda_graph.rs` | CUDA Graph capture on first decode token, replay on subsequent tokens |
| `model/qwen3/config.rs` | Parse Qwen3 `config.json` → typed `Qwen3Config` struct |
| `model/qwen3/weights.rs` | Load safetensors shards into GPU memory → `Qwen3Model` |
| `model/qwen3/decode_buffers.rs` | Pre-allocated GPU scratch buffers for decode step |
| `model/qwen3/prefill.rs` | Prefill forward pass (batched GEMM + FlashAttention-2) |
| `model/qwen3/decode.rs` | Single-request decode (GEMV path) |
| `model/qwen3/batch_decode.rs` | Batched decode for multiple concurrent requests (GEMM path) |
| `model/qwen3/forward.rs` | `ModelForward` impl for `Qwen3Model`; `Qwen3State` definition |
| `model/qwen35/recurrent_state.rs` | Per-request recurrent state for 24 linear-attention layers |
| `model/qwen35/single_token_buffers.rs` | GPU scratch buffers for Qwen3.5 single-token (decode) step |
| `model/qwen35/prefill_buffers.rs` | GPU scratch buffers for Qwen3.5 prefill step |
| `model/qwen35/forward.rs` | `ModelForward` impl for `Qwen35Model`; `Qwen35State` definition |
| `ops.rs` | Re-exports all GPU operation submodules |
| `ops/attention.rs` | Fused decode attention + FlashAttention-2 prefill (calls Triton/CUDA) |
| `ops/linear.rs` | GEMV (decode), GEMM (prefill), fused MLP |
| `ops/norm.rs` | RMSNorm variants including fused add+norm |
| `ops/embedding.rs` | Token embedding lookup (prefill + decode) |
| `ops/elementwise.rs` | SiLU·Mul fused, element-wise add |
| `ops/recurrent.rs` | Gated delta rule (Qwen3.5 linear attn), conv1d |
| `ops/sampling.rs` | GPU top-k/top-p sampling and argmax |
| `tensor.rs` | `DeviceContext` (CUDA context+stream), `DeviceVec`, `DeviceMatrix` |
| `ffi.rs` | Unsafe FFI declarations for all CUDA C kernel entry points |
| `weight_loader.rs` | Memory-mapped safetensors → GPU tensor loading |
| `scheduler.rs` | Multi-request continuous batching scheduler (decode-priority, chunked prefill) |
| `server_engine.rs` | `ServerEngine` trait; `GenericServerEngine<M>` (single-request, for REPL/agent) |
| `http_server.rs` | Axum router: OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, SSE |
| `http_server/openai_v1.rs` | OpenAI API request/response JSON types |
| `backend.rs` | `InferenceBackend` trait — backend-agnostic single-request interface |
| `metal_backend.rs` | Apple Silicon Metal backend via mlx-rs; implements `InferenceBackend` |
| `sampler.rs` | `SamplingParams` struct (temperature, top-k/p, min-p, penalties, EOS) |
| `tokenizer.rs` | Tokenizer wrapper (HuggingFace `tokenizers` crate) with incremental decoding |
| `chat.rs` | ChatML message formatting and tool definition injection |
| `block_manager.rs` | Paged KV block allocation accounting (CPU-only; GPU wiring pending) |
| `prefix_cache.rs` | Radix tree content-addressable prefix cache (CPU-only; GPU wiring pending) |
| `cuda_graph_pool.rs` | Batch-size padding arithmetic + CUDA Graph pool state tracking |
| `speculative.rs` | Speculative decoding framework: `SpecConfig`, `verify_tokens`, `DraftModel` stub |
| `tensor_parallel.rs` | Tensor parallel config + column/row sharding math; NCCL stubs |
| `model_registry.rs` | Architecture detection from `config.json` → `ModelArch` enum (9 architectures) |
| `quant.rs` | Quantization format detection (GPTQ/AWQ/FP8/INT8/GGUF) |
| `metrics.rs` | Prometheus metrics (`ServerMetrics`) |
| `trace_reporter.rs` | fastrace `FileReporter` for request tracing |
| `hf_hub.rs` | HuggingFace Hub auto-download of model weights + tokenizer |
| `logging.rs` | logforth logging setup |

### src/ — Agent Binary

| Module | Responsibility |
|--------|----------------|
| `main.rs` | CLI entry point: detect model type, load engine, run REPL or Dynamo worker |
| `agent.rs` | Agent loop: generate → parse tool calls → execute tools → feed results back |
| `chat.rs` | ChatML message builder for agent conversations |
| `tools.rs` | Builtin tool definitions (functions the agent can call) |
| `dynamo_integration.rs` | Register with Dynamo distributed runtime for service discovery + KV routing |

### CUDA Kernels (infer/csrc/)

| File | Responsibility |
|------|----------------|
| `common.cuh` | Shared CUDA utilities, warp-level primitives |
| `fused_attention.cu` | Decode attention kernel (split-KV, one token per request) |
| `prefill_attention.cu` | Prefill attention for general head dims |
| `prefill_attention_hd256.cu` | Optimized prefill attention for head_dim=256 |
| `fused_mlp.cu` | Fused gate+up projection → SiLU → down projection |
| `gemv.cu` | GEMV for decode linear projections |
| `norm.cu` | RMSNorm + fused add+norm variants |
| `pos_enc.cu` | RoPE positional encoding |
| `conv1d.cu` | 1D depthwise convolution for Qwen3.5 recurrent layers |
| `gated_delta_rule.cu` | Gated delta rule (Qwen3.5 linear attention mechanism) |
| `sampling.cu` | Top-k, top-p, temperature GPU sampling |

### Triton Kernels (infer/tools/triton/)

| File | Responsibility |
|------|----------------|
| `flash_attention_prefill_kernel.py` | FlashAttention-2 prefill (general head dim) |
| `flash_attention_prefill_hd256_kernel.py` | FlashAttention-2 prefill optimized for head_dim=256 |
| `attention_decode_kernel.py` | Split-KV decode attention (per-head parallelism) |
| `attention_reduce_kernel.py` | Reduction across split-KV decode results |
| `silu_mul_kernel.py` | Fused SiLU·Mul element-wise kernel |
| `basic_kernels.py` | Add, embedding lookup variants |
| `gated_delta_rule_chunkwise_kernels.py` | Qwen3.5 chunkwise gated delta rule for prefill |
| `gen_triton_aot.py` | AOT compilation driver: runs all Triton kernels through `triton.compile()` |

---

## Key Design Decisions

1. **No PyTorch** — direct cudarc (Rust CUDA bindings) + custom kernels. Eliminates Python GIL, framework overhead, and enables CUDA Graph capture.

2. **Single compute stream** — all GPU work on one CUDA stream. Enables CUDA Graph capture without cross-stream synchronization. `ctx.disable_event_tracking()` called at init.

3. **Weights `&self`, state `&mut State`** — `ModelForward` separates immutable weights from mutable per-request state. Enables N concurrent requests sharing one model copy.

4. **Decode before prefill** — scheduler prioritizes batch-decoding all active requests before any prefill chunk. Prevents decode latency spikes from prefill interference.

5. **Chunked prefill (512 tokens)** — long prompts are prefilled in 512-token chunks, allowing decode steps to interleave. Prevents prefill from starving active decoders.

6. **CUDA Graph per slot** — each slot captures its decode graph on the first decode token, then replays it. CUDA Graph eliminates CPU-GPU kernel dispatch overhead at decode time.

7. **bf16 throughout** — all GPU tensors are `CudaSlice<bf16>`. No FP32 copies needed at inference time.

8. **CPU-complete stubs** — BlockManager, RadixCache, CudaGraphPool, SpecConfig, TpConfig are fully implemented in pure Rust with no GPU dependency. GPU wiring is the only remaining step.
