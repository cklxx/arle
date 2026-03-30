# agent-infer Roadmap

Target: **production-grade LLM inference engine** on par with sglang (excluding KV-affinity routing).

---

## Current State (2026-03-31)

Working: Qwen3/Qwen3.5 inference, FlashAttention-2 (Triton), KV cache + CPU offload, continuous batching with chunked prefill, top-k/p/temp/min-p/penalty sampling, OpenAI `/v1/completions` + `/v1/chat/completions` + SSE, CUDA graph decode, Rust agent binary, Python async agent, Dynamo integration, Prometheus `/metrics` + `/v1/stats` endpoints, model architecture registry, radix-tree prefix cache (data structure), paged KV block manager (accounting), speculative decoding framework (CPU stubs), tensor parallel config/sharding math (CPU stubs), CUDA graph batch pool (CPU tracking), quantization format detection (GPTQ/AWQ/FP8/INT8/GGUF parser).

Missing: multi-architecture GPU inference (Llama/DeepSeek/Mistral/Gemma/Phi), PagedAttention CUDA kernel, MLA attention, quantization GPU kernels, tensor parallel communication (NCCL), speculative decoding GPU integration, comprehensive benchmark suite.

---

## Implementation Philosophy

**CPU-first**: implement all control-plane logic (schedulers, data structures, APIs, config) without GPU. Mark GPU-only code with `// GPU required` stubs. This lets development and testing proceed without a GPU for the majority of the codebase.

**Incremental validation**: each module has a local-verifiable test path. GPU integration tests run on the A100 machine.

---

## Phase 0 — Foundation (CPU-verifiable, no GPU needed)

These tasks can be built and tested entirely on CPU.

### 0.1 Chat Completions API
**Files**: `pegainfer/src/http_server/openai_v1.rs`
**Goal**: Add `/v1/chat/completions` endpoint with message → ChatML conversion, streaming SSE, and `finish_reason` handling. Essential for LLM clients (OpenAI SDK, litellm, etc.).

```
ChatCompletionRequest { messages, model, max_tokens, temperature, stream, tools, ... }
ChatCompletionResponse / stream chunks
Tool call delta streaming (OpenAI format)
```

**Local test**: `curl -X POST http://localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}]}'`

---

### 0.2 Sampler Expansion
**Files**: `pegainfer/src/sampler.rs`, `pegainfer/src/ops/sampling.rs`
**Goal**: Add missing sampling strategies so the engine is compatible with all common LLM configs.

```
repetition_penalty: f32        ← penalize already-generated tokens
frequency_penalty: f32         ← OpenAI-style frequency penalty
presence_penalty: f32          ← OpenAI-style presence penalty
min_p: f32                     ← min-p sampling (popular in 2024–2025)
seed: Option<u64>              ← deterministic sampling
stop_token_ids: Vec<u32>       ← additional stop tokens beyond model EOS
```

**Local test**: unit tests for penalty application on toy logit vectors (no GPU).

---

### 0.3 Radix Tree Prefix Cache
**Files**: `pegainfer/src/model/prefix_cache.rs` (new)
**Goal**: Content-addressable KV cache reuse across requests. Requests sharing a common prefix (system prompt, few-shot examples) skip prefill for the shared portion.

```
RadixTree<BlockId>
  insert(tokens: &[u32]) → (hit_len, new_blocks)
  lookup(tokens: &[u32]) → (matched_prefix_len, matched_block_ids)
  evict_lru(n_blocks: usize)
  ref_count per node (requests in-flight hold refs)
```

**Integrates with**: `GenerationState::truncate_to()` already exists — radix cache provides the `hit_len` input.
**Local test**: pure data structure tests (no GPU). Fuzz with random token sequences.

---

### 0.4 Paged KV Cache Block Manager
**Files**: `pegainfer/src/model/block_manager.rs` (new), extend `kv_cache.rs`
**Goal**: Fixed-size block allocator for KV cache. Required for PagedAttention and for radix cache to work across requests without copying.

```
BlockManager {
  block_size: usize         ← tokens per block (e.g. 16)
  num_gpu_blocks: usize
  num_cpu_blocks: usize
  free_gpu_blocks: VecDeque<BlockId>
  free_cpu_blocks: VecDeque<BlockId>
  allocate_gpu(n: usize) → Vec<BlockId>
  free(blocks: &[BlockId])
  swap_out(blocks: &[BlockId])   // GPU → CPU
  swap_in(blocks: &[BlockId])    // CPU → GPU
}
```

**Local test**: allocation/free/swap accounting logic — no GPU needed for the manager itself.

---

### 0.5 Scheduler Improvements
**Files**: `pegainfer/src/scheduler.rs`
**Goal**: Upgrade scheduler for PagedAttention-era requirements.

- **Preemption with swap**: when GPU OOM, preempt lowest-priority request, swap its KV blocks to CPU, resume later
- **Priority queue**: add request priority field, FCFS + optional priority override
- **Waiting queue backpressure**: expose `max_waiting_requests` config, return 503 when full
- **Chunked prefill config**: make `PREFILL_CHUNK_SIZE` configurable per-request
- **Stats endpoint**: `/metrics` with Prometheus-compatible counters (requests/s, TTFT p50/p95/p99, TBT p50/p95, KV cache utilization)

**Local test**: mock `ModelForward` for scheduler unit tests (no GPU).

---

### 0.6 Model Architecture Registry
**Files**: `pegainfer/src/model/registry.rs` (new), `pegainfer/src/server_engine.rs`
**Goal**: Replace hard-coded `ModelType` enum with a dynamic registry. Enables adding new architectures without changing the dispatch code.

```rust
pub trait ModelConfig: for<'de> Deserialize<'de> {
    fn model_type() -> &'static str;
}

pub struct ModelRegistry {
    loaders: HashMap<&'static str, Box<dyn ModelLoader>>,
}

impl ModelRegistry {
    pub fn detect_and_load(path: &Path) -> Result<Box<dyn ErasedModelForward>>;
}
```

Register `Qwen3`, `Qwen35`, and stubs for `Llama`, `DeepSeek`, `Mistral`, `Gemma`, `Phi` upfront.

**Local test**: load config.json, assert correct architecture detection.

---

### 0.7 Benchmark Suite
**Files**: `scripts/bench_throughput.py`, `scripts/bench_latency.py`, `pegainfer/src/bin/bench_serving.rs`
**Goal**: Comprehensive benchmark framework measuring all production metrics.

```
Metrics:
  TTFT   — time to first token (prefill latency)
  TPOT   — time per output token (decode latency, aka TBT)
  E2E    — end-to-end request latency
  Throughput — requests/s and tokens/s

Scripts:
  bench_throughput.py --num-prompts 1000 --concurrency 32
  bench_latency.py --input-len 512 --output-len 128 --batch-sizes 1,4,8,16
  bench_serving.rs (Rust binary, lower overhead)
```

ShareGPT dataset loader for realistic prompt distributions.

**Local test**: scripts run against mock HTTP server (no GPU).

---

## Phase 1 — Core GPU Features (GPU required)

### 1.1 PagedAttention Kernel
**Files**: `pegainfer/csrc/paged_attention.cu`, `pegainfer/src/ops/attention.rs`
**Goal**: Replace contiguous KV cache with paged blocks. Eliminates memory fragmentation, enables sharing blocks across requests (prefix cache), enables swap.

```c
// CUDA kernel signature
void paged_attention_decode(
    float* out,                    // [num_heads, head_dim]
    const float* q,                // [num_heads, head_dim]
    const float* k_cache,          // [num_gpu_blocks, block_size, num_kv_heads, head_dim]
    const float* v_cache,          // [num_gpu_blocks, block_size, num_kv_heads, head_dim]
    const int* block_table,        // [max_context_len / block_size]
    int context_len,
    int block_size,
    int num_heads, int num_kv_heads, int head_dim
);
```

Placeholder stub in `ops/attention.rs` with `todo!("GPU required: paged attention decode")`.

---

### 1.2 FlashAttention-3 (prefill)
**Files**: `pegainfer/tools/triton/flash_attention_v3_kernel.py`, `pegainfer/src/ops/attention.rs`
**Goal**: Upgrade prefill kernel from FA2 → FA3 for better A100/H100 utilization. FA3 uses warp specialization + async pipeline.

Key improvements over FA2:
- Ping-pong SMEM pipeline (2× producer/consumer overlap)
- FP8 accumulation path for H100
- Better causal mask handling for long sequences

Keep FA2 as fallback for older GPUs (< SM90).

---

### 1.3 Multi-Head Latent Attention (MLA) — DeepSeek
**Files**: `pegainfer/src/ops/attention.rs`, `pegainfer/src/model/deepseek/`
**Goal**: Implement MLA as used in DeepSeek-V2/V3/R1. MLA compresses KV cache to a latent vector (rank << d_model), drastically reducing KV memory.

```
MLA:
  KV compression: C_KV = W_DKV · x   [d_c << d_head * num_kv_heads]
  K decompression: K = W_UK · C_KV   [num_kv_heads, d_head]
  V decompression: V = W_UV · C_KV
  Q compression: C_Q = W_DQ · x (optional)
  RoPE decoupled: only rope-part of K/Q goes through position encoding
```

Cache only `C_KV` (compressed), not `K, V` — huge memory saving.

---

### 1.4 Llama 3/4 Model
**Files**: `pegainfer/src/model/llama/`
**Goal**: Support Meta's Llama 3 (8B, 70B, 405B) and Llama 4 Scout/Maverick.

Llama 3 differences from Qwen3:
- RoPE with `rope_scaling` (YaRN / dynamic)
- Tied embeddings
- Group query attention (8 KV heads for 70B/405B)
- No QK norm
- Different tokenizer (tiktoken-based)

Llama 4 additions:
- Mixture of Experts (MoE) layers
- Interleaved attention + MoE blocks
- 128-expert architecture

---

### 1.5 DeepSeek-V3 / R1 Model
**Files**: `pegainfer/src/model/deepseek/`
**Goal**: Support DeepSeek-V3 (671B MoE) and R1 (671B MoE + chain-of-thought).

Key components:
- MLA (see 1.3)
- DeepSeekMoE: 256 routed experts + 2 shared experts, top-8 routing
- Multi-Token Prediction (MTP) head for speculative decoding
- FP8 mixed precision (E4M3 weights, BF16 compute)

---

### 1.6 Mistral / Mixtral Model
**Files**: `pegainfer/src/model/mistral/`
**Goal**: Support Mistral 7B v0.3, Mixtral 8x7B, Mixtral 8x22B, Mistral Large 2.

- Sliding window attention (Mistral 7B)
- MoE (Mixtral) — top-2 routing, 8 experts
- No bias in linear layers

---

### 1.7 Gemma 2 / 3 Model
**Files**: `pegainfer/src/model/gemma/`
**Goal**: Support Google Gemma 2 (2B, 9B, 27B) and Gemma 3.

Gemma 2 differences:
- Alternating sliding-window + global attention
- Logit soft-capping (tanh gate on logits)
- Pre + post norm (two RMSNorm per layer)
- Group query attention

Gemma 3 additions:
- Much longer context (128K default)
- Interleaved local/global attention with ratio 5:1

---

### 1.8 Phi-4 Model
**Files**: `pegainfer/src/model/phi/`
**Goal**: Support Microsoft Phi-4 (14B).

- Dense transformer, standard GQA
- Custom tokenizer
- Flash attention compatible

---

## Phase 2 — Quantization (GPU required for kernels, CPU for config)

### 2.1 Weight Loader Abstraction
**Files**: `pegainfer/src/weight_loader.rs`
**Goal**: Extend weight loader to detect and dispatch quantized formats.

```rust
pub enum QuantFormat { None, GPTQ, AWQ, FP8, INT8 }

impl WeightLoader {
    pub fn detect_quant_format(model_path: &Path) -> QuantFormat;
    pub fn load_quantized(path: &Path, fmt: QuantFormat) -> Result<QuantizedWeights>;
}
```

**CPU-verifiable**: config detection, metadata parsing, no kernel execution.

---

### 2.2 GPTQ / AWQ INT4 Kernels
**Files**: `pegainfer/csrc/gemm_int4.cu`
**Goal**: Fused dequantize-GEMM kernels for INT4 weights. Required for serving quantized open-source models (most popular community models are GPTQ/AWQ quantized).

Implement W4A16 (INT4 weights, FP16 activations) via:
- Exllama v2 kernel (GPTQ) as reference
- Marlin kernel (AWQ/GPTQ) for A100+ — tile-level dequant fused with GEMM

---

### 2.3 FP8 (E4M3) Kernels
**Files**: `pegainfer/csrc/gemm_fp8.cu`
**Goal**: FP8 GEMM for H100 (SM90). Required for DeepSeek-V3 native precision.

- `__nv_fp8_e4m3` weight storage, BF16 accumulation
- Static per-tensor scaling (DeepSeek-V3 style)
- Dynamic per-token activation scaling

---

### 2.4 INT8 / SmoothQuant
**Files**: `pegainfer/csrc/gemm_int8.cu`
**Goal**: INT8 W8A8 GEMM using cuBLAS `cublasLtMatmul` with INT8 I/O.

SmoothQuant migration: apply channel-wise scale to activations before quantizing.

---

### 2.5 GGUF Loading
**Files**: `pegainfer/src/weight_loader.rs`, add gguf reader
**Goal**: Load GGUF format files (llama.cpp ecosystem). Read GGUF header, tensor layout, Q4_K_M / Q8_0 block formats. Dequantize on load to BF16.

---

## Phase 3 — Tensor Parallelism (GPU required, multi-GPU)

### 3.1 TP Communication Primitives
**Files**: `pegainfer/src/ops/comm.rs` (new)
**Goal**: All-reduce and all-gather over NCCL for tensor parallel.

```rust
pub struct NcclComm { ... }
impl NcclComm {
    pub fn all_reduce_sum(tensor: &mut DeviceVec) -> Result<()>;
    pub fn all_gather(tensor: &DeviceVec, world_size: usize) -> Result<DeviceVec>;
}
```

---

### 3.2 Column / Row Parallel Linear
**Files**: `pegainfer/src/ops/linear.rs`
**Goal**: Sharded linear layers for tensor parallel.

```
ColumnParallelLinear:  Y_i = X · W_i^T / TP_SIZE  (all-reduce after)
RowParallelLinear:     Y = X_i · W_i^T             (scatter input, gather output)
```

---

### 3.3 TP-aware Model Wrappers
**Files**: `pegainfer/src/model/<name>/tp.rs` per model
**Goal**: Wrap each model's attention (split Q/K/V heads) and MLP (split intermediate dim) across TP ranks.

---

### 3.4 Pipeline Parallel (PP)
**Files**: `pegainfer/src/model/pp.rs` (new)
**Goal**: Partition layers across devices. Send hidden states between stages via P2P (cuMemcpyPeer / NCCL send/recv).

---

### 3.5 Expert Parallel (EP) for MoE
**Files**: `pegainfer/src/ops/moe.rs` (new)
**Goal**: Distribute MoE experts across EP ranks. All-to-all routing for expert dispatch/combine.

---

## Phase 4 — Advanced Decoding

### 4.1 Beam Search
**Files**: `pegainfer/src/sampler.rs`
**Goal**: Beam search with configurable beam width. Uses block manager to share KV prefixes across beams.

---

### 4.2 Speculative Decoding
**Files**: `pegainfer/src/speculative.rs` (new)
**Goal**: Draft-verify loop: small draft model generates K candidate tokens, large target model verifies in one forward pass.

```
DraftEngine (small model, e.g. Qwen3-0.5B)
TargetEngine (large model, e.g. Qwen3-32B)
SpeculativeScheduler: draft K tokens, batch-verify, accept/reject
```

For DeepSeek-V3: use MTP heads (built into model) as draft — zero additional model cost.

---

### 4.3 Structured Output (JSON schema / regex)
**Files**: `pegainfer/src/structured_output.rs` (new)
**Goal**: Constrained decoding via logit masking. At each step, mask invalid token IDs per grammar state.

- JSON schema → LALR(1) grammar → token mask table
- Regex → NFA → per-step allowed token set
- Outlines / llguidance integration as optional backend

---

## Phase 5 — Performance Optimization

### 5.1 CUDA Graph for Multi-Request Decode
**Files**: `pegainfer/src/model/cuda_graph.rs`
**Goal**: Extend existing single-request CUDA graph to capture multi-request batched decode. Requires fixed batch size → maintain a pool of graphs (batch sizes 1, 2, 4, 8, 16, 32).

---

### 5.2 Overlap Compute and Communication (TP)
**Files**: `pegainfer/src/ops/comm.rs`
**Goal**: Async NCCL all-reduce overlapped with next layer's compute using two CUDA streams.

---

### 5.3 Torch.compile / `torch.export` Integration
**Goal**: For Python-based model paths (if added), use `torch.compile` for additional kernel fusion.

---

### 5.4 Prefill/Decode Disaggregation
**Goal**: Split prefill (compute-bound) and decode (memory-bandwidth-bound) onto separate GPU instances (à la DistServe / Mooncake). Prefill machines send KV cache to decode machines via RDMA.

---

## Dependency Graph

```
Phase 0 (CPU) → can start immediately, no GPU needed
  0.1 Chat API          → enables downstream integration
  0.2 Sampler           → no deps
  0.3 Radix Cache       → data structure only
  0.4 Block Manager     → depends on 0.3 design
  0.5 Scheduler++       → depends on 0.4 design
  0.6 Model Registry    → no deps
  0.7 Benchmark Suite   → depends on 0.1

Phase 1 (GPU)
  1.1 PagedAttention    → depends on 0.4 (block manager)
  1.2 FA3               → depends on existing FA2
  1.3 MLA               → needed by 1.5
  1.4 Llama             → depends on 0.6 (registry), 1.1
  1.5 DeepSeek          → depends on 0.6, 1.1, 1.3
  1.6 Mistral           → depends on 0.6, 1.1
  1.7 Gemma             → depends on 0.6, 1.1
  1.8 Phi               → depends on 0.6, 1.1

Phase 2 (Quantization)
  2.1 Loader            → depends on 0.6
  2.2 GPTQ/AWQ          → depends on 2.1
  2.3 FP8               → depends on 2.1
  2.4 INT8              → depends on 2.1
  2.5 GGUF              → depends on 2.1

Phase 3 (TP/PP/EP)
  3.1 NCCL primitives   → Phase 1 complete
  3.2 Parallel linear   → depends on 3.1
  3.3 TP model wrappers → depends on 3.2
  3.4 PP                → depends on 3.3
  3.5 EP (MoE)          → depends on 3.3, 1.5/1.6 (MoE models)

Phase 4 (Decoding)
  4.1 Beam search       → depends on 0.4 (block sharing)
  4.2 Speculative       → depends on 1.x (two models loaded)
  4.3 Structured output → depends on 0.2 (sampler logit hooks)

Phase 5 (Optimization)
  5.1 CUDA graph batch  → depends on 1.1 (paged attn)
  5.2 Compute/comm overlap → depends on 3.1
  5.4 Disaggregation    → depends on all Phase 3
```

---

## Immediate Next Steps (Start Here)

1. **0.1 Chat Completions API** — unblocks all OpenAI-compatible clients
2. **0.2 Sampler Expansion** — repetition penalty is needed for all production use cases
3. **0.6 Model Registry** — clean up `ModelType` enum before adding more models
4. **0.3 + 0.4 Radix Cache + Block Manager** — foundation for all KV cache work
5. **0.5 Scheduler improvements** — preemption, stats endpoint
6. **0.7 Benchmark Suite** — need baseline numbers before GPU kernel work

First GPU task (when on the A100): **1.1 PagedAttention** — unlocks all model-architecture work, enables large contexts without OOM.

---

## Local Verifiability Checklist

| Task | Verifiable without GPU? |
|------|------------------------|
| 0.1 Chat API | ✅ Mock scheduler |
| 0.2 Sampler | ✅ Unit tests on f32 arrays |
| 0.3 Radix Cache | ✅ Pure data structure |
| 0.4 Block Manager | ✅ Accounting only |
| 0.5 Scheduler++ | ✅ Mock ModelForward |
| 0.6 Model Registry | ✅ Config parsing |
| 0.7 Benchmarks | ✅ Mock HTTP server |
| 1.x–5.x | ❌ GPU required |
