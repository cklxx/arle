# GPU Benchmark Report — agent-infer (Qwen3-4B on A100-40GB)

## 环境
- GPU: NVIDIA A100-SXM4-40GB
- CUDA: 12.8 / Driver 580.82.07 / SM 80
- 模型: Qwen3-4B (BF16, 8.0GB)
- 模型加载: ~3.0s
- CUDA Graph: enabled

## End-to-End Serving (单请求)
| 指标 | avg | p50 | p95 | p99 |
|------|-----|-----|-----|-----|
| TTFT | 16.38ms | 16.33ms | 16.66ms | 17.00ms |
| 稳态 TPOT | 7.93ms | 7.94ms | 7.96ms | 7.97ms |
| Decode 吞吐 | 125.81 tok/s |

## Prefill
| seq_len | TTFT | 吞吐 |
|---------|------|------|
| 256 | 40.75ms | 6,283 tok/s |
| 1024 | 210.17ms | 4,872 tok/s |

## Kernel Microbenchmarks
| Kernel | 配置 | 耗时 | 吞吐 |
|--------|------|------|------|
| GEMV (lm_head) | 248320×2560 | 941µs | 675 Gelem/s |
| GEMM (batch=64) | 1024×1024 | 27.6µs | 2,432 Gelem/s |
| Fused MLP | 2560×9216 | 125µs | 188 Gelem/s |
| Prefill Attention | batch=64 | 159µs | 413 Melem/s |
| Decode Attention | seq=64 | 21.8µs | 47 Melem/s |

Triton AOT ≈ 手写 CUDA（<1% 差距）。
