# 2026-04-01 · Benchmark Raw Data

All benchmarks run on 2026-04-01. Environment: A100-SXM4-40GB, CUDA 12.8, Qwen3-8B bf16.

## 1. Throughput Sweep — infer (C=1, 4 slots, max_seq=4096)

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err
  128 |   128 |  1 |  72.3 t/s  |      19ms |      40ms | 13.7ms   | 13.8ms   |   0
  128 |   512 |  1 |  70.2 t/s  |      17ms |      32ms | 14.2ms   | 14.2ms   |   0
  512 |   256 |  1 |  68.1 t/s  |      19ms |      97ms | 14.6ms   | 14.6ms   |   0
 1024 |   256 |  1 |  63.5 t/s  |      21ms |     228ms | 15.5ms   | 15.6ms   |   0
 2048 |   256 |  1 |    FAIL    |   (503 — scheduler queue full)                |   4
  512 |   256 |  4 |    FAIL    |   (503 — scheduler queue full)                |   4
```

## 2. Throughput Sweep — sglang 0.5.9 (auto slots, max_total=8192)

```
   In |   Out |  C | Throughput |  TTFT p50 |  TTFT p99 |  ITL p50 |  ITL p99 | Err
  128 |   128 |  1 |  32.5 t/s  |      49ms |    9007ms | 13.0ms   | 13.0ms   |   0
  128 |   512 |  1 |  76.3 t/s  |      49ms |      52ms | 13.0ms   | 13.0ms   |   0
  512 |   256 |  1 |  75.8 t/s  |      50ms |      53ms | 13.0ms   | 13.0ms   |   0
 1024 |   256 |  1 |  75.5 t/s  |      53ms |      76ms | 13.1ms   | 13.1ms   |   0
 2048 |   256 |  1 |  74.8 t/s  |      52ms |     121ms | 13.1ms   | 13.1ms   |   0
  512 |   256 |  4 | 255.9 t/s  |     665ms |     665ms | 13.1ms   | 13.1ms   |   0
```

Note: sglang 128/128 C=1 anomaly (32.5 t/s, p99 TTFT=9s) is likely a JIT warmup outlier on first request.

## 3. Long-Sequence Agent — infer (2 slots, max_seq=8192, 512 gen/turn)

```
Turn | TTFT     | ITL     | Gen | tok/s | Turn  | E2E   | ~Prompt
T1   |   37ms   | 14.1ms  | 512 | 70.8  |  7.2s |  7.3s |    ~39
T2   |  171ms   | 15.4ms  | 512 | 63.8  |  8.0s | 15.4s |   ~527
T3   |  282ms   | 16.7ms  | 512 | 58.1  |  8.8s | 24.2s |   ~991
T4   |  386ms   | 17.9ms  | 512 | 53.6  |  9.5s | 33.7s | ~1396
T5   |  496ms   | 19.2ms  | 512 | 49.6  | 10.3s | 44.0s | ~1779
T6   |  610ms   | 20.5ms  | 512 | 46.1  | 11.1s | 55.1s | ~2293
T7   |  718ms   | 21.8ms  | 512 | 43.2  | 11.9s | 67.0s | ~2647
T8   |  834ms   | 23.1ms  | 512 | 40.5  | 12.7s | 79.7s | ~2971
```

Summary: 8/15 turns, 4096 tokens, 79.7s E2E. TTFT degradation 6.9x, ITL degradation 1.5x.

## 4. nsjail Sandbox Stress Test

```
32/32 direct sandbox tests: PASS (1.2s total)
5/5 E2E agent tests with nsjail: PASS
0 zombie processes
```

## 5. Unit Tests

```
infer crate: 160 passed, 7 failed (missing model files), 10 ignored
agent binary: 2 passed, 0 failed
```

## 6. Environment

```
Rust:       1.94.1 (2026-03-25)
CUDA:       12.8 (V12.8.93)
GPU:        NVIDIA A100-SXM4-40GB
FlashInfer: 0.6.3 (headers)
Triton:     3.5.1
sglang:     0.5.9
Python:     3.12.13
Model:      Qwen3-8B bf16 (15.5GB, 36 layers, 4096 hidden, 32 heads, 8 kv heads)
```
