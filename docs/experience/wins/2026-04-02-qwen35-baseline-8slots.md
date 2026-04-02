# 2026-04-02 · Qwen3.5-4B Baseline Benchmark (8 slots, A100-80GB)

## Context

Baseline benchmark before batched recurrent kernel integration. Server running with `--num-slots 8`.

## Raw Data

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err
  128 |    64 |  1 |    111.5 t/s |      14ms |      24ms |    8.8ms |    8.9ms |   0
  128 |   128 |  1 |    109.5 t/s |      23ms |      23ms |    8.8ms |    8.9ms |   0
  128 |   256 |  1 |    110.6 t/s |      23ms |      23ms |    8.9ms |    8.9ms |   0
  128 |   512 |  1 |    110.0 t/s |      14ms |      23ms |    9.1ms |    9.1ms |   0
  512 |   128 |  1 |     94.3 t/s |      27ms |      53ms |    9.2ms |   18.3ms |   0
  512 |   256 |  1 |    107.6 t/s |      21ms |      54ms |    9.2ms |    9.2ms |   0
  512 |   512 |  1 |    106.1 t/s |      21ms |      53ms |    9.4ms |    9.4ms |   0
 1024 |   128 |  1 |     78.6 t/s |      39ms |      88ms |   19.0ms |   19.0ms |   0
 1024 |   256 |  1 |     89.7 t/s |      35ms |      88ms |    9.6ms |   19.2ms |   0
 1024 |   512 |  1 |     82.5 t/s |      35ms |      88ms |    9.8ms |   19.5ms |   0
 2048 |   256 |  1 |     93.5 t/s |      46ms |     190ms |   10.4ms |   10.4ms |   0
  512 |   256 |  2 |    157.5 t/s |      55ms |     114ms |    9.9ms |   10.0ms |   0
  512 |   256 |  4 |    315.2 t/s |     122ms |     214ms |   12.1ms |   12.1ms |   0
```

Peak: C=1 111.5 tok/s, C=4 315.2 tok/s, ITL p50 8.8-12.1ms

## Observations

1. C=1 throughput ~110 tok/s — significant improvement over prior A100-40GB baseline (100 tok/s)
2. C=4 512/256 at 315 tok/s — close to SGLang's 349 tok/s target
3. ITL p50 C=1 at 8.8ms — already below SGLang's 8.6ms on A100-40GB
4. 1024-input configs show ITL p99 spikes to ~19ms — likely prefill chunk interference

## Environment

```
GPU:        NVIDIA A100-SXM4-80GB
CUDA:       12.8
Model:      Qwen3.5-4B bf16
num_slots:  8
```
