# XMA / accelerated-model-architectures 调研笔记

## Context

调研对象：

- GitHub: `open-lm-engine/accelerated-model-architectures`
- 文档站: XMA documentation

这个项目的定位不是 serving runtime，而是一个面向训练和实验型模型架构的 kernel 仓库。它公开强调支持多种 accelerator，包括 NVIDIA、AMD、TPU 和 Trainium。

对 `agent-infer` 来说，它的价值不在于“直接引入”，而在于作为未来研究雷达，帮助判断接下来值得提前设计哪些抽象和优化方向。

## 观察

### 1. 关注点不是标准 Transformer，而是实验型架构

XMA 文档把这些 layer 直接放进 capability matrix：

- `LinearAttention`
- `M2RNN`
- `GRU`
- `RNN`
- `MoE`

这说明它的核心诉求不是把标准 dense attention 再抠出一点性能，而是让“非标准架构”也能拥有一组可复用的高性能 kernel。

这和我们当前的经验是对得上的：Qwen3.5 一旦从标准 dense attention 走向 hybrid recurrent + full attention，decode 热路径立刻出现了新的 recurrent state、batching、graph capture 问题。以后如果继续支持 Llama 之外的架构，runtime 不能再假设“模型状态 = KV cache”。

### 2. kernel 融合重点不只在 attention

XMA 的 functional 列表里，有几类内容和我们下一阶段方向高度相关：

- `fused_residual_add_rmsnorm`
- `swiglu`
- `swiglu_packed`
- `fused_linear_cross_entropy`
- `pack_sequence` / `unpack_sequence`

这和我们项目现在的瓶颈判断非常一致：后续值得投的，不只是 attention kernel。

真正会持续复用的热点还包括：

- residual + norm 融合
- MLP activation / gate 路径
- packed sequence
- output head / loss 侧融合

也就是说，未来优化路线应该继续从“attention-centric”走向“whole-block hot path”。

### 3. 编译器驱动的 pattern replacement 值得单独研究

XMA 仓库里有 `xma.inductor`，示例 `examples/inductor.py` 展示了一个很明确的思路：

- 在高层图里先匹配常见子图模式
- 再用自定义 kernel 做 replacement
- 通过 `torch.compile` / Inductor 让替换自动发生

这和我们当前纯 Rust + FFI 直接调 kernel 的路线不同，但不冲突。

如果以后项目需要下面任何一种能力，这条线都值得研究：

- Python-based reference / fallback path
- 训练侧工具链
- 自动 kernel 选择
- 高层图模式到自定义 kernel 的替换机制

结论不是“要切回 Python”，而是编译器和图替换本身是一个独立能力层，未来不一定只能靠手写 runtime dispatch。

### 4. 多后端 capability matrix 值得尽早抽象

XMA 文档用一个显式矩阵列 capability：

- CUDA
- MPS
- Pallas
- NKI
- ROCm
- Triton

从当前公开矩阵看，很多高级 op 还是 Triton-first，CUDA 覆盖并不完整，因此它并不是我们可以直接对标的 serving 基座。

但它给了一个很有用的启发：

- 如果项目未来继续同时维护 CUDA 和 Metal
- 或者进一步考虑 ROCm

那么算子抽象最好尽早从简单的 `cfg(feature)` 分叉，演进到更明确的 capability / registration 层。

否则模型一多、后端一多，判断“哪个 op 在哪个后端可用、是否有 fused 版本、是否支持 graph capture”会越来越散。

## 对 agent-infer 的启发

XMA 最值得吸收的不是代码，而是问题意识：

1. **模型结构会继续分化**：`LinearAttention`、`RNN`、`MoE` 这类结构需要 runtime 把 state 管理当成一等问题，而不是 attention 的附属品。
2. **融合面要扩大**：后续优化不该只盯着 attention，还要继续看 residual/norm、MLP、packed sequence、head/loss。
3. **后端能力需要矩阵化**：未来 CUDA、Metal、可能的 ROCm 不该只是散落在 `if cfg` 里的分支。
4. **编译器路线值得保留观察**：即使主线仍是 Rust runtime，图替换和自动 kernel 选择也可能成为未来工具链的一部分。

## 结论

这条研究线的核心结论是：

**未来不只是把现有 attention 跑得更快，还要为“新架构 + 新后端 + 新编译路径”预留演化空间。**

XMA 当前更像一个训练和实验架构仓库，不适合作为 serving 直接依赖；但它很适合作为外部雷达，持续跟踪哪些模型结构和 kernel 组合开始变成主流。

## Sources

- https://github.com/open-lm-engine/accelerated-model-architectures
- https://open-lm-engine.github.io/accelerated-model-architectures/
- https://raw.githubusercontent.com/open-lm-engine/accelerated-model-architectures/main/examples/inductor.py
