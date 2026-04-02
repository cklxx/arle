# Plan: Qwen3.5 追平 sglang 0.5.9 (Qwen3.5-4B · A100-80GB)

> Status: **Steps 1-3 Done** — C=4 exceeds SGLang, C=8 ITL beats SGLang
> Created: 2026-04-01
> Updated: 2026-04-02
> Goal: 在 Qwen3.5-4B 上匹配 sglang 0.5.9 吞吐量

---

## Current Results (2026-04-02, A100-80GB)

| 配置 | infer (before) | infer (after) | sglang 0.5.9 | Status |
|------|---------------|---------------|-------------|--------|
| C=1 throughput | 110 tok/s | 115 tok/s | 111 tok/s | **✅ +4%** |
| C=1 ITL p50 | 8.8ms | 8.5ms | 8.9ms | **✅ beats sglang** |
| C=4 throughput (512/256) | 319 tok/s | 405 tok/s | 369 tok/s | **✅ +10%** |
| C=4 ITL p50 | 12.0ms | 9.2ms | 9.8ms | **✅ beats sglang** |
| C=8 throughput (128/256) | 640 tok/s | 772 tok/s | 1230 tok/s | ❌ -37% |
| C=8 ITL p50 | 23.7ms | 9.8ms | 11.0ms | **✅ beats sglang** |
| C=1 TTFT p50 | 22ms | 14ms | 72ms | **✅ 5x faster** |

## Completed Steps

### Step 1: Auto slots + increased concurrency ✅
- `--num-slots` now optional, auto-computed from GPU memory (default 8)
- Tested with 8 and 16 slots

### Step 2: Batched conv1d kernel ✅
- `infer/csrc/cuda/conv1d_decode_batch.cu`: K=4 specialized, register-cached weights
- Grid: (channels/256, B), pointer array for per-request conv states
- Rust ops wrapper with kernel_size ∈ [2,4] assert

### Step 3: Batched GDR decode kernel ✅
- `infer/csrc/cuda/gdr_decode_batch.cu`: 2D grid (num_value_heads, B)
- 512 threads/block, same j-slice parallelism as single-request kernel
- Pointer array for per-request recurrent states
- Integrated into `batch_decode.rs`, replacing per-request loop

### CUDA Graph warmup expansion ✅
- SGLang-style batch sizes: 1,2,4,8,12,16,24,...,min(num_slots,256)
- Matches SGLang's 36-size warmup schedule

## Remaining Gap Analysis

### C=8 throughput: 772 vs 1230 tok/s (-37%)

**Root cause: slot count, not ITL**. Our ITL (9.8ms) already beats SGLang (11.0ms). The throughput gap is purely from concurrency:
- agent-infer: 8 slots → max 8 concurrent decode requests
- SGLang: auto-sized to ~177 concurrent requests

At 8 concurrent with ITL 9.8ms: theoretical max = 8 / 0.0098 = 816 tok/s. We achieve 772 (94% efficiency).
SGLang at 16 concurrent with ITL 13.1ms: 16 / 0.0131 = 1221 tok/s. They achieve 1823 (unclear, likely higher effective concurrency).

**Fix**: Increase slots to 32+ or implement dynamic batching (sglang-parity.md Step 4).

### ITL p99 spikes to ~17ms at some C=1 configs
- Seen at 512/128 C=1 (ITL p50=17.5ms) and 128/256 C=1 (p99=17.3ms)
- Likely prefill chunk interference — prefill chunk size 4096 is large
- SGLang uses 8192 but has better overlap scheduling

## Next Steps

| Step | Status | Impact |
|------|--------|--------|
| Step 4: Fuse conv1d + GDR | Pending | Low — already 2 launches/layer vs 2B+6B D2D |
| Step 5: CUDA Graph for Qwen3.5 batched decode | Pending | Medium — eliminates remaining launch overhead |
| More slots / dynamic batching | Pending | High — C=8+ throughput |
| Investigate ITL p99 spikes | Pending | Medium — prefill interference |

## Verification Benchmarks

See `docs/experience/wins/2026-04-02-*.md` for full raw data.
