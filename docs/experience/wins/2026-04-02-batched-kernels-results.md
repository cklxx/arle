# 2026-04-02 · Batched Recurrent Kernels — Results

## Context

After integrating batched conv1d + GDR decode kernels, expanded CUDA Graph warmup, and auto num_slots. Comparing against baseline (per-request serial kernels) and SGLang 0.5.9.

## Raw Data — agent-infer (batched kernels, 8 auto-slots)

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99
  128 |    64 |  1 |    115.4 t/s |      14ms |      31ms |    8.5ms |    8.5ms
  128 |   128 |  1 |    115.3 t/s |      19ms |      22ms |    8.6ms |    8.6ms
  128 |   256 |  1 |     93.7 t/s |      22ms |      23ms |    8.7ms |   17.3ms
  128 |   512 |  1 |    103.8 t/s |      22ms |      22ms |    8.8ms |   17.6ms
  512 |   128 |  1 |     89.0 t/s |      29ms |      53ms |   17.5ms |   17.6ms
  512 |   256 |  1 |    104.4 t/s |      29ms |      53ms |    8.9ms |   17.5ms
  512 |   512 |  1 |    108.7 t/s |      29ms |      53ms |    9.2ms |    9.2ms
 1024 |   128 |  1 |    104.5 t/s |      29ms |      88ms |    9.3ms |    9.3ms
 1024 |   256 |  1 |    105.2 t/s |      35ms |      88ms |    9.4ms |    9.4ms
 1024 |   512 |  1 |    104.4 t/s |      38ms |      79ms |    9.5ms |    9.5ms
 2048 |   256 |  1 |     95.4 t/s |      56ms |     189ms |   10.1ms |   10.2ms
  512 |   256 |  2 |    204.0 t/s |      48ms |     111ms |    8.8ms |    8.8ms
  512 |   256 |  4 |    405.3 t/s |     110ms |     208ms |    9.2ms |    9.2ms
  128 |   128 |  2 |    215.3 t/s |      34ms |      51ms |    8.8ms |    8.8ms
  128 |   128 |  4 |    408.9 t/s |      65ms |      93ms |    9.2ms |    9.2ms
```

High concurrency:
```
  C=8 in=128 out=256: 772.3 tok/s, TTFT=22ms, ITL=9.8ms
  C=8 in=512 out=256: 720.2 tok/s, TTFT=22ms, ITL=9.8ms
  C=8 in=128 out=512: 781.5 tok/s, TTFT=21ms, ITL=10.0ms
```

## Comparison: Before → After → SGLang

| Metric | Before (serial) | After (batched) | SGLang 0.5.9 | Improvement |
|--------|-----------------|-----------------|--------------|-------------|
| C=1 ITL p50 (128/128) | 8.8ms | **8.6ms** | 8.9ms | -2%, **beats SGLang** |
| C=4 throughput (512/256) | 318.7 tok/s | **405.3 tok/s** | 368.5 tok/s | **+27%, beats SGLang +10%** |
| C=4 ITL p50 | 12.0ms | **9.2ms** | 9.8ms | **-23%, beats SGLang** |
| C=4 throughput (128/128) | — | **408.9 tok/s** | 349.9 tok/s | **beats SGLang +17%** |
| C=8 throughput (128/256) | 639.7 tok/s | **772.3 tok/s** | 1229.5 tok/s | +21%, SGLang still 59% ahead |
| C=8 ITL p50 | 23.7ms | **9.8ms** | 11.0ms | **-59%, beats SGLang** |
| C=1 TTFT p50 | 22ms | **14-22ms** | 72ms | **still 3x faster** |

## Key Findings

1. **C=4 now exceeds SGLang** — 405 vs 369 tok/s (+10%), ITL 9.2 vs 9.8ms
2. **C=8 ITL dramatically improved** — 23.7ms → 9.8ms (59% reduction), now beats SGLang's 11.0ms
3. **C=8 throughput still behind SGLang** — 772 vs 1230 tok/s. Gap is from 8 vs ~177 slots (SGLang auto-sizes to much higher concurrency). Need more slots or dynamic batching.
4. **C=1 slightly improved** — 115 vs 111 tok/s, ITL 8.5 vs 8.8ms
5. **Some ITL p99 spikes** — 17ms at medium configs, likely prefill interference

## What Worked

- Batched recurrent kernels eliminated ~13ms serial overhead at C=8
- C=4 ITL went from 12ms to 9.2ms — now better than SGLang
- TTFT advantage maintained (14-56ms vs SGLang's 72-98ms)

## Remaining Gap

C=8 throughput is 772 vs SGLang's 1230 tok/s. Root cause: we're limited to 8 slots → only 8 concurrent decode requests. SGLang handles ~177 concurrent requests. Fix: increase slots or implement dynamic batching.

## Environment

```
GPU:          NVIDIA A100-SXM4-80GB
CUDA:         12.8
Model:        Qwen3.5-4B bf16
num_slots:    8 (auto)
Commit:       4ac4b06 (main, merged feat/batched-recurrent-kernels)
```
