# 2026-04-02 · SGLang 0.5.9 vs agent-infer — Qwen3.5-4B on A100-80GB

## Context

Head-to-head comparison to establish SGLang as the optimization target. Both servers running on the same machine, same model, same benchmark script.

## SGLang 0.5.9 Default Config

```
model_type:           qwen3_5
attention_backend:    flashinfer
sampling_backend:     flashinfer
radix_cache:          ON (LRU eviction)
cuda_graph:           ON (36 batch sizes: 1,2,4,8,12,16,...,256)
cuda_graph_max_bs:    256
chunked_prefill_size: 8192
max_prefill_tokens:   16384
mem_fraction_static:  0.794
torch_compile:        OFF
overlap_schedule:     OFF (disabled — mamba no_buffer incompatible)
schedule_policy:      fcfs
page_size:            1
max_running_requests: auto (limited by memory, ~177 at bs peak)
```

Key: SGLang uses FLA batched recurrent kernels for linear attention, and auto-sizes concurrency from GPU memory.

## Raw Data — SGLang 0.5.9

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99
  128 |    64 |  1 |     90.8 t/s |      72ms |     608ms |    9.0ms |    9.2ms
  128 |   128 |  1 |    104.6 t/s |      72ms |     221ms |    8.9ms |    9.0ms
  128 |   256 |  1 |    108.3 t/s |      73ms |     194ms |    8.9ms |    9.0ms
  128 |   512 |  1 |    110.5 t/s |      72ms |     192ms |    8.9ms |    8.9ms
  512 |   128 |  1 |    105.0 t/s |      73ms |     206ms |    8.9ms |    8.9ms
  512 |   256 |  1 |    108.1 t/s |      74ms |     204ms |    8.9ms |    9.0ms
  512 |   512 |  1 |    110.8 t/s |      73ms |     197ms |    8.8ms |    8.9ms
 1024 |   128 |  1 |    104.7 t/s |      76ms |     213ms |    8.9ms |    8.9ms
 1024 |   256 |  1 |    109.0 t/s |      76ms |      86ms |    8.9ms |    9.0ms
 1024 |   512 |  1 |    109.8 t/s |      77ms |     206ms |    8.9ms |    9.1ms
 2048 |   256 |  1 |    106.0 t/s |      98ms |     242ms |    9.0ms |    9.3ms
  512 |   256 |  2 |    191.6 t/s |     140ms |     259ms |    9.5ms |    9.5ms
  512 |   256 |  4 |    368.5 t/s |     205ms |     396ms |    9.8ms |    9.8ms
  128 |   128 |  2 |    186.2 t/s |     132ms |     265ms |    9.5ms |    9.5ms
  128 |   128 |  4 |    349.9 t/s |     143ms |     266ms |    9.9ms |    9.9ms
```

High concurrency:
```
  C |    In |   Out |  N | Throughput | ITL p50
  8 |   128 |   256 | 16 |  1229.5 t/s |  11.0ms
  8 |   512 |   256 | 16 |  1162.4 t/s |  11.0ms
  8 |   128 |   512 | 16 |  1383.9 t/s |  11.1ms
 16 |   512 |   256 | 32 |  1823.2 t/s |  13.1ms
```

## Raw Data — agent-infer (16 slots)

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99
  128 |   128 |  1 |    111.2 t/s |      23ms |      28ms |    8.8ms |    8.8ms
  128 |   512 |  1 |    110.2 t/s |      23ms |      24ms |    9.0ms |    9.1ms
  512 |   256 |  1 |    107.9 t/s |      22ms |      53ms |    9.2ms |    9.2ms
 1024 |   256 |  1 |    103.7 t/s |      25ms |      30ms |    9.6ms |    9.6ms
 2048 |   256 |  1 |     95.3 t/s |      19ms |      29ms |   10.4ms |   10.4ms
  512 |   256 |  4 |    318.7 t/s |     114ms |     217ms |   12.0ms |   12.0ms
```

High concurrency:
```
  C |    In |   Out |  N | Throughput | ITL p50
  8 |   128 |   256 | 16 |    639.7 t/s |  23.7ms
  8 |   512 |   256 | 16 |    592.2 t/s |  23.8ms
  8 |   128 |   512 | 16 |    648.1 t/s |  24.3ms
```

## Comparison

| Metric | agent-infer | SGLang | Gap |
|--------|------------|--------|-----|
| C=1 throughput (512/256) | 107.9 tok/s | 108.1 tok/s | **~0% (parity)** |
| C=1 ITL p50 | 8.8-9.2ms | 8.8-9.0ms | **~0% (parity)** |
| C=1 TTFT p50 | **22ms** | 74ms | **+236% (we win)** |
| C=4 throughput | 318.7 tok/s | 368.5 tok/s | **-14%** |
| C=4 ITL p50 | 12.0ms | 9.8ms | **-22%** |
| C=8 throughput | 639.7 tok/s | 1229.5 tok/s | **-48%** |
| C=8 ITL p50 | 23.7ms | 11.0ms | **-115%** |
| C=16 throughput | N/A (16 slot limit) | 1823.2 tok/s | — |

## Root Cause Analysis

### Why C=1 is at parity
Both engines are bottlenecked by the same thing: single-request decode latency (~9ms/token). Model weights, attention kernels, and memory bandwidth are the same. Our TTFT advantage comes from lighter scheduler overhead.

### Why C>1 diverges — per-request recurrent serial execution
At C=8, our ITL is 23.7ms vs SGLang's 11.0ms. The gap is **12.7ms**, explained by:

- 24 linear attention layers × per-request serial conv1d + GDR
- At C=8: 24 layers × 8 requests × (~15μs conv1d + ~35μs GDR + ~5μs D2D×3) = **~10ms overhead**
- SGLang uses batched FLA kernels: one kernel launch per layer regardless of batch size

The batched conv1d + GDR kernels we wrote (`conv1d_decode_batch.cu`, `gdr_decode_batch.cu`) should eliminate this: from 24×8=192 kernel launches to 24×2=48 per decode step.

### Remaining gap sources (after batched kernels)
1. SGLang CUDA Graph covers batch sizes up to 256 (36 sizes); we only go to num_slots
2. SGLang radix cache avoids redundant prefill on repeated system prompts
3. SGLang auto-sizes concurrency from GPU memory; we need manual --num-slots

## Environment

```
GPU:          NVIDIA A100-SXM4-80GB
CUDA:         12.8
Model:        Qwen3.5-4B bf16
SGLang:       0.5.9
agent-infer:  feat/batched-recurrent-kernels (b5bf86c)
```
