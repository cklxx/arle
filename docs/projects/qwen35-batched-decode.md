# Qwen3.5 Batched Decode 实现计划

> **状态**: 调研完成，待实现
> **当前瓶颈**: Qwen3.5 无 `forward_decode_batch`，高并发回退到串行 forward()，C=4 仅 90 tok/s

---

## SGLang 的做法

SGLang 完整支持 Qwen3.5，核心设计是 **Layer-Aware Routing + Indexed State Pooling**。

### 架构

```
ForwardBatch (B requests)
        ↓
HybridLinearAttnBackend
    ├─ layer_id % 4 == 3 → full_attn_backend (FlashInfer GQA)     ← 8 层
    └─ else              → linear_attn_backend (FLA delta rule)    ← 24 层
```

### 3 个关键设计

**1. Global State Pool（非 per-slot）**

```python
conv_pool:  [pool_size, conv_dim, kernel_size]                    # Conv1d state
ssm_pool:   [num_layers, pool_size, num_heads, state_size]        # Delta rule state
# 每个请求用 cache_indices[batch_idx] 索引自己的 state
```

**2. Batched FLA Kernels**

- `fused_recurrent_kda_fwd()` — 一个 kernel 同时更新 B 个请求的 delta rule state
- `causal_conv1d_update()` — 批量更新 Conv1d sliding window
- `chunk_gated_delta_rule()` — prefill 时按 chunk 处理

**3. 全注意力层复用标准 FlashInfer**

8 层全注意力和 Qwen3 用完全相同的 FlashInfer paged decode。

### SGLang 代码位置

| 文件 | 内容 |
|------|------|
| `sglang/srt/models/qwen3_5.py` | 模型定义 |
| `sglang/srt/configs/qwen3_5.py` | 配置类 |
| `sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | 混合 backend 路由 |
| `sglang/srt/layers/radix_linear_attention.py` | 线性注意力包装 |
| `sglang/srt/layers/attention/fla/kda.py` | KDA kernel |

---

## 我们需要做的

### 可复用（来自 Qwen3）

- FlashInfer paged decode（8 层全注意力 ← 直接复用）
- BatchDecodeBuffers 的 GEMM/MLP buffer 结构
- CUDA Graph capture/replay 框架
- Batched greedy argmax + cached ptr
- FlashInfer metadata 管理

### 需要新建

| 组件 | 工作量 | 说明 |
|------|--------|------|
| `qwen35/batch_decode.rs` | ~400 行 | BatchDecodeBuffers35 + decode_batch() |
| Batched GDR kernel | ~200 行 CUDA | 当前 gated_delta_rule_decode_into 只支持单 token |
| Batched Conv1d update | ~100 行 CUDA | 当前只有 prefill batch 版本 |
| RecurrentState → State Pool | ~100 行 Rust | 改造 per-slot state 为 indexed pool |
| forward.rs 集成 | ~50 行 | 实现 forward_decode_batch |
| 测试 | ~100 行 | E2E consistency |

### 实现步骤

1. **Phase 1: 全注意力层 batched decode**（复用 Qwen3 代码）
   - 只 batch 8 层全注意力，24 层线性注意力仍然串行
   - 预期收益：C=4 从 90 → ~200 tok/s

2. **Phase 2: 线性注意力层 batched decode**
   - 写 batched GDR + Conv1d CUDA kernel
   - State pool 改造
   - 预期收益：C=4 从 ~200 → ~400 tok/s

3. **Phase 3: CUDA Graph 捕获**
   - 整个 hybrid layer loop 放入 graph
   - 预期收益：接近 Qwen3 的吞吐水平

---

## Qwen3.5 层结构参考

```
32 layers total:
  Layer 0-2:   Linear (Conv1d + GDR)     key_heads=16, val_heads=32, key_dim=128, val_dim=128
  Layer 3:     Full Attention (GQA)       q_heads=16, kv_heads=4, head_dim=256
  Layer 4-6:   Linear
  Layer 7:     Full Attention
  ...
  Layer 28-30: Linear
  Layer 31:    Full Attention
```

Config dimensions:
- `hidden_size`: 2560
- `intermediate_size`: 6912
- `linear_key_head_dim`: 128, `linear_value_head_dim`: 128
- `linear_num_key_heads`: 16, `linear_num_value_heads`: 32
- `num_attention_heads`: 16, `num_key_value_heads`: 4, `head_dim`: 256
- `linear_conv_kernel_dim`: 4
