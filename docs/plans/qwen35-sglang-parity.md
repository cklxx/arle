# Plan: Qwen3.5 SGLang 0.5.9 Parity (Qwen3.5-4B, A100-80GB)

> Status: **Parity achieved C=1 through C=16** (C=32 slot-limited)
> Created: 2026-04-01
> Updated: 2026-04-02
> Goal: Match SGLang 0.5.9 throughput on Qwen3.5-4B

---

## Current Results (2026-04-02, 32 auto-slots, prefix cache disabled)

| Config | agent-infer | SGLang 0.5.9 | Gap |
|--------|------------|--------------|-----|
| C=1 throughput | 113 tok/s | 110 tok/s | **+3%** |
| C=4 throughput | 413 tok/s | 376 tok/s | **+10%** |
| C=8 throughput | 759 tok/s | 707 tok/s | **+7%** |
| C=16 throughput | 1230 tok/s | 1261 tok/s | -2.4% (parity) |
| C=32 throughput | 1628 tok/s | 2189 tok/s | -25.6% (slot limit) |
| C=8 ITL p50 | **9.9ms** | 10.3ms | **+4%** |
| C=1 TTFT p50 | **21ms** | 71ms | **3.4x faster** |

## Completed Steps

### Step 1: Auto slots + batched kernels
- Auto-computed slots from GPU memory (32 on A100-80GB)
- Batched conv1d + GDR decode kernels
- SGLang-style CUDA Graph warmup (12 batch sizes)

### Step 2: Fix prefix cache crash (2026-04-02)
- **Root cause**: Full prefix reuse on Qwen3.5 kept contaminated recurrent state from previous request's decode tokens. GDR kernel hit illegal memory during batched decode.
- **Fix**: Disabled prefix cache for Qwen3.5 (temporary)
- **Proper fix needed**: Reset recurrent state on prefix hit, preserving only KV cache

### Step 3: Fix auto-slots OOM (2026-04-02)
- **Root cause**: MAX_SLOTS=64 used ~80 GB (entire GPU). No headroom for workspace.
- **Fix**: Increased RESERVED_BYTES to 6 GB, capped MAX_SLOTS to 32

## Remaining Work

| Step | Priority | Impact | Description |
|------|----------|--------|-------------|
| Fix prefix cache for Qwen3.5 | Medium | Correctness | Reset recurrent state on prefix hit |
| More slots (48-64) with better memory sizing | Low | C=32+ | Need per-slot memory accounting like SGLang |
| Async scheduling (overlap) | Low | ~5% | Prepare batch N+1 while GPU runs batch N |

## Previous Results (before fixes)

Earlier SGLang comparison data in docs had C=8 = 1230 tok/s for SGLang, which was incorrect (mathematically impossible at C=8 with ITL 11ms). Corrected SGLang C=8 = 707 tok/s using the same benchmark tool.
