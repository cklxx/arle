# Plan: Qwen3.5 追平 sglang 0.5.9 (Qwen3.5-4B · A100-40GB)

> Status: **In Progress** — scheduler support done, ITL gap identified
> Created: 2026-04-01
> Goal: 在 Qwen3.5-4B 上匹配 sglang 0.5.9 吞吐量

---

## Baseline 数据

| 配置 | infer | sglang 0.5.9 | Gap |
|------|-------|-------------|-----|
| C=1 throughput | 100 tok/s | 107 tok/s | **-7%** |
| C=1 ITL | 9.9ms | 8.6ms | **-15%** |
| C=4 throughput | 290 tok/s | 349 tok/s | **-17%** |
| C=4 ITL | 13.2ms | 9.4ms | **-40%** |
| C=8 throughput | 297 tok/s (4 slots) | 680 tok/s | **-56%** |
| C=1 TTFT | **17ms** | 107ms | +530% (we win) |
| C=4 TTFT | **45ms** | 270ms | +500% (we win) |

## 差距根因分析

### 1. C=1 ITL 差距 (9.9ms vs 8.6ms, -15%)

**根因**: Qwen3.5 recurrent ops 开销
- 24 个 linear attention 层的 conv1d + GDR decode kernel
- 每层 ~50μs (conv1d ~15μs + GDR ~35μs)
- 24 层 × 50μs = **1.2ms overhead** — 解释了大部分 1.3ms gap
- sglang 可能有更高效的 recurrent kernel 或 fused conv1d+GDR

**文件**: `infer/csrc/cuda/conv1d_decode.cu`, `infer/csrc/cuda/gated_delta_rule.cu`

### 2. C=4 ITL 差距 (13.2ms vs 9.4ms, -40%)

**根因**: per-request recurrent D2D + kernel launch overhead
- 24 层 × 4 请求 × (3 D2D copies + conv1d + GDR) = 480 extra kernel launches
- 每个 ~1.5μs → **~720μs launch overhead**
- 加上 D2D 数据搬运: 24 × 4 × 24KB = 2.3MB → **~115μs bandwidth**
- 总额外开销: ~835μs per decode step

**需要**: batched recurrent kernel (一个 kernel 处理 B 个请求的 recurrent state)

### 3. 吞吐上限 (297 vs 680 tok/s)

**根因**: slot 数量限制
- infer 默认 4 slots，sglang 默认更多
- Qwen3.5-4B 只有 8 KV 层 (vs Qwen3-8B 的 36 层)，KV cache 很小
- 可以轻松开 8-16 slots

## 优化步骤

### Step 1: 增加 slots (Easy, High Impact on throughput)

**目标**: C=8 throughput 提升到 ~580+ tok/s
**做法**: 默认 slots 从 4 增到 8 (Qwen3.5 KV cache 很小)
**预期**: throughput 线性扩展到 slot 数，ITL 不变
**文件**: `infer/src/scheduler/cuda/core.rs` (num_slots default), `infer/src/bin/bench_serving.rs`

### Step 2: Batched conv1d kernel (Medium, Critical for C>1 ITL)

**目标**: C=4 ITL 从 13.2ms 降到 ~10ms
**做法**: 写新 CUDA kernel `conv1d_decode_batch`
- 输入: QKV [B, qkv_dim], conv_states [B × num_linear_layers, qkv_dim × (kernel_dim-1)]
- 一个 kernel launch 处理 B 个请求的 conv state 更新
- Grid: (qkv_dim, B), Threads: kernel_dim
- 需要 gather/scatter conv_state 到连续 buffer，或传 per-request pointers

**文件**: 新建 `infer/csrc/cuda/conv1d_decode_batch.cu`

### Step 3: Batched GDR decode kernel (Medium, Critical for C>1 ITL)

**目标**: 配合 Step 2，消除 per-request GDR overhead
**做法**: 写新 CUDA kernel `gated_delta_rule_decode_batch`
- 输入: QKV_conv [B, qkv_dim], B/A projections [B, num_v_heads], per-request states [B × ...]
- 一个 kernel launch 处理 B 个请求的 GDR state update
- 关键: state 是 f32 [num_v_heads × key_dim × value_dim] per request，需要 pointer array

**文件**: 新建 `infer/csrc/cuda/gdr_decode_batch.cu`

### Step 4: Fuse conv1d + GDR (Hard, diminishing returns)

**目标**: 进一步减少 kernel launch
**做法**: 合并 conv1d output → GDR input 为一个 fused kernel
**预期**: 每 linear layer 从 2 launches → 1 launch

### Step 5: CUDA Graph for batched decode (Hard, depends on Step 2-3)

**目标**: 消除剩余 kernel launch overhead
**做法**: 如果 batched recurrent kernels 的参数都在 pre-allocated buffers 中
- 需要将 per-request recurrent state 映射到连续的 batch buffers
- 在 graph capture 前做 gather，graph 后做 scatter
- Gather/scatter 在 graph 外，graph 内全是 batch kernels

**风险**: recurrent state 很大 (24 layers × 2MB per request = 48MB per request)，gather/scatter 开销可能抵消 graph 收益

## 验收标准

| 指标 | 当前 | 目标 | sglang |
|------|------|------|--------|
| C=1 ITL | 9.9ms | ≤9ms | 8.6ms |
| C=4 ITL | 13.2ms | ≤10ms | 9.4ms |
| C=4 throughput | 290 tok/s | ≥340 tok/s | 349 tok/s |
| C=8 throughput | 297 tok/s | ≥600 tok/s | 680 tok/s |
