| Path | TL;DR |
| --- | --- |
| [experience/errors/2026-03-31-flashinfer-segfault-debug.md](experience/errors/2026-03-31-flashinfer-segfault-debug.md) | 3 bugs causing FlashInfer batch decode crash: hardcoded MAX_SEQ, GPU plan_info, double alloc |
| [experience/wins/2026-03-31-batched-decode-throughput.md](experience/wins/2026-03-31-batched-decode-throughput.md) | 128 -> 811 tok/s (6.3x) via token pool + FlashInfer + buffer reuse + plan-once + CUDA Graph + argmax/scatter |
