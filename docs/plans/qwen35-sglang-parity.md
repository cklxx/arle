# Plan: Qwen3.5 SGLang 0.5.9 Parity (Qwen3.5-4B, A100-80GB)

> Status: **C=1–C=8 ahead, C=16 parity, C=32 -17%, C=64 works**
> Created: 2026-04-01
> Updated: 2026-04-02
> Goal: Match SGLang 0.5.9 throughput on Qwen3.5-4B

---

## Current Results (2026-04-02, 128 auto-slots, prefix cache disabled)

| Config | agent-infer | SGLang 0.5.9 | Gap |
|--------|------------|--------------|-----|
| C=1 throughput | 113 tok/s | 110 tok/s | **+3%** |
| C=4 throughput | 399 tok/s | 376 tok/s | **+6%** |
| C=8 throughput | 742 tok/s | 707 tok/s | **+5%** |
| C=16 throughput | 1199 tok/s | 1261 tok/s | -5% (parity) |
| C=32 throughput | 1818 tok/s | 2189 tok/s | -17% |
| C=64 throughput | 2397 tok/s | N/A | new! |
| C=8 ITL p50 | **10.0ms** | 10.3ms | **+3%** |
| C=32 ITL p50 | 14.2ms | 12.9ms | -10% |
| C=1 TTFT p50 | **21ms** | 71ms | **3.4x faster** |

## Completed Steps

### Step 1: Auto slots + batched kernels
- Auto-computed slots from GPU memory (32 on A100-80GB)
- Batched conv1d + GDR decode kernels
- SGLang-style CUDA Graph warmup (12 batch sizes)

### Step 2: Fix prefix cache crash (2026-04-02)
- **Root cause**: Full prefix reuse on Qwen3.5 kept contaminated recurrent state from previous request's decode tokens. GDR kernel hit illegal memory during batched decode.
- **Fix**: Disabled prefix cache for Qwen3.5 (temporary)

### Step 3: Fix auto-slots OOM (2026-04-02)
- **Root cause**: MAX_SLOTS=64 used ~80 GB (entire GPU). No headroom for workspace.
- **Fix**: Increased RESERVED_BYTES to 6 GB, capped MAX_SLOTS to 32

### Step 4: 128 slots + high concurrency fixes (2026-04-02)
- **MAX_SLOTS**: 32 → 128, auto-computed from GPU memory
- **kv_pool_headroom**: 2 GB → 4 GB for batch buffers + CUDA graphs
- **Prefill rate limiting**: 1 per step when decode active (prevents scheduler blocking)
- **BUG FIX**: Shared waiting counter — handle and scheduler had separate AtomicUsize counters, causing all requests rejected after 256 total
- **CUDA Graph sizes**: Added 16, 20, 24, 28, 32 to warmup schedule
- C=32 improved from 1628 → 1818 tok/s, C=64 now works at 2397 tok/s

## Remaining Work

| Step | Priority | Impact | Description |
|------|----------|--------|-------------|
| CUDA Graph for Qwen3.5 decode | High | C=32 -10% ITL | Recurrent layers run eager; need contiguous cache like SGLang |
| Fix prefix cache for Qwen3.5 | Medium | TTFT | Reset recurrent state on prefix hit |
| Async scheduling (overlap) | Low | ~5% | Prepare batch N+1 while GPU runs batch N |
