# 从零到一：我是怎么徒手写出一个 LLM 推理引擎的

> 本文记录 agent-infer 项目的构建历程——一个纯 Rust + CUDA 的 LLM 推理引擎，从第一行代码到能跑 354 tok/s、142 个单元测试全绿，中间走了哪些弯路，做了哪些选择。

---

## 为什么要自己写

市面上已经有 vLLM、SGLang、TensorRT-LLM，为什么还要再造一个轮子？

原因说起来有点讽刺：**我想真正理解推理引擎**。读论文和读代码是两回事。KV cache 的论文我看过，但直到自己写了一遍 CPU offload 的 block-aligned 换出逻辑，才理解为什么要按 64-token 对齐、为什么 swap-back 要在 attention 之前完成。读别人的代码总隔着一层——你能看懂，但不知道那个设计是第一次就想到的，还是踩坑之后改出来的。

另一个原因是技术栈。主流推理框架深度绑定 PyTorch，控制平面（scheduler、API 层）是 Python，GPU 调度的延迟隐藏得很深。Rust 在这里有优势：内存安全、无 GC 停顿、极低的 CPU→GPU dispatch 开销。我想验证：纯 Rust 控制平面 + 直接 CUDA/Triton 内核，能不能在延迟上打过 Python 栈？

最后，我需要一个能跑 agent 工作负载的平台。Agent 的特点是多轮对话、工具调用、反复提交相同的 system prompt——这对 prefix cache 和连续批处理的要求远比单轮补全严苛。自己做才能按需优化。

---

## 架构设计

整个项目分两层：

**控制平面（Rust）**：请求接收、调度、采样、KV 管理、HTTP API。这部分全部不依赖 GPU，可以在普通 MacBook 上测试。

**计算内核（CUDA C + Triton）**：attention、GEMV、RMSNorm、SiLU、采样——所有需要 GPU 的地方。通过 Rust FFI 调用。

```
HTTP 请求
    ↓
SchedulerHandle.submit()  →  channel  →  Scheduler.run()（独占线程）
                                               ↓
                                     State Pool（每个 slot 独立 KV cache）
                                               ↓
                                    model.forward(tokens, &mut state)
                                               ↓
                                    StreamDelta  →  SSE / JSON 响应
```

核心抽象只有三个：

- **`ModelForward` trait**：`forward(&self, tokens, state)` 一个方法。`tokens.len() > 1` 是 prefill，`== 1` 是 decode。权重是 `&self`（不可变，多 slot 共享），per-request 状态在 `state`（可变）。
- **`Scheduler`**：多请求连续批处理。通过 channel 收请求，在单 GPU 上交织执行。
- **`SamplingParams`**：采样配置（temperature、top-k、top-p、min-p、repetition penalty 等）。

一个设计原则贯穿始终：**CPU 优先**。每个模块先写 CPU 可验证的逻辑，GPU 相关的地方留 stub 标 `// GPU required`。这样开发期不需要 GPU，142 个单元测试跑起来只要 9 秒。

---

## 核心模块

### Scheduler：连续批处理的关键

最早的版本是串行的：一个请求跑完再跑下一个。吞吐量极差，原因显而易见——GPU 在等 CPU 处理 HTTP、采样、token decode，而 CPU 也在等 GPU 算完。

连续批处理解决这个问题：把多个请求的 decode 步骤打成一个 batch，一次 forward 处理所有人。难点在调度策略：

**Decode 优先**：Decode 步骤（每次 1 个 token）永远优先于新请求的 prefill。原因是 decode 的 KV cache 已经在 GPU 上了，中断很贵；而 prefill 是新请求，等一轮没关系。

**Chunked prefill**：长 prompt 切成 512-token 的块，每块之间给 decode 请求插队。这避免了一个 32K prompt 的新请求把其他人的响应延迟拉高几秒。

**Backpressure**：队列满了直接返回 503，不无限堆积。这个看起来简单，但对生产可用性很重要。

### KV Cache：省内存比拼速度更难

KV cache 是推理内存的大头。一个 7B 模型跑 4K 上下文，KV cache 大约 2GB——每个请求都要这么多，GPU 很快就不够用了。

我实现了两层策略：

**CPU offload**：生成过程中，把已经算完的 prefix KV blocks 换出到 CPU 内存。换出粒度是 64-token 一个 block，换回来时在 attention 之前做。这让 GPU 专注持有当前正在 decode 的部分，理论上可以处理远超 GPU 显存的上下文长度。

**Radix tree prefix cache**：多轮对话的 system prompt 是完全相同的前缀。如果每次请求都重新算一遍，纯属浪费。用 radix tree 存已算的 KV blocks，命中时直接复用——实测多轮 agent 工作负载下 KV hit rate 可以达到 100%，相当于完全跳过 prefill。

### Attention：Triton 的生产力

Attention 是推理最耗时的算子，尤其是 prefill 阶段的全序列 attention（O(n²) 复杂度）。

我用 Triton 写了 FlashAttention-2 的 prefill 和 decode 内核。Triton 比手写 CUDA 快很多，原因不是性能更好，而是开发效率：调 tile size、handle bank conflict、处理边界条件，这些在 CUDA 里要写几百行，Triton 里几十行搞定，编译出来的机器码质量也不差。

Decode 阶段用了另一个技巧：**CUDA Graph**。第一次 decode 时把整个 forward pass 录制成一个 CUDA graph，之后每个 token 只需要 replay 这个 graph，CPU→GPU 的 kernel launch overhead 降到接近零。实测单 token 延迟下降约 15-20%。

Qwen3.5 模型的 24 层 linear attention + 8 层全 attention 的混合架构特别有趣。Linear attention 层是 O(1) KV cache（recurrent state，每 token 固定大小），全 attention 层才需要 KV cache 增长。这使得长上下文的内存占用大幅降低，是个值得关注的方向。

### Sampling：细节比想象中多

采样看起来简单，但 OpenAI 兼容的采样参数有很多坑。应用顺序很关键：

1. Repetition / frequency / presence penalty（先修改 logits）
2. Temperature scaling
3. Top-K 截断
4. Top-P（nucleus）过滤
5. Min-P 过滤
6. Argmax（greedy）或 categorical sample

顺序错了结果就不对。另外 stop string 的处理也有坑——token boundary 和 string boundary 不一致，需要先 decode 整个序列、在 string 层面匹配 stop，而不是在 token 层面。这个 bug 我修了两次才彻底修对。

---

## MLX Backend 的意外发现

项目主要目标是 NVIDIA GPU，但有天我想在 MacBook 上测试 API 层——不是要跑推理，只是验证 HTTP 接口格式。

结果发现 `mlx-lm` 已经把 Apple Silicon 的 Metal GPU 封装得很好，接上 Qwen2.5-0.5B-4bit 量化模型，M3 Pro 直接跑到 **354 tok/s**。这比我预期的高很多——4bit 量化 + Metal 的内存带宽利用率相当不错。

更有意思的是，加上各种 pre-flight 检查（Apple Silicon 检测、内存检查、磁盘检查）之后，整个启动流程的开销只有约 **0.4%** 的额外 overhead——几乎可以忽略不计。这说明 Python 层的这些检查没有成为瓶颈。

于是我把 MLX backend 正式加进了项目，作为"不需要 NVIDIA GPU 也能跑"的替代路径。架构是一样的——OpenAI 兼容 API、SSE streaming、相同的采样参数——只是底层换成了 MLX。

这个意外收获让我意识到：**推理引擎的价值有一部分在架构和 API 层**，不完全在内核。一个好的调度器、一个干净的 HTTP API、一个稳定的 sampling 实现，这些放在任何后端上都有价值。

---

## 对比结果

在 NVIDIA A100-SXM4-40GB 上，跑了一个 agent 基准测试（5 个并发请求，每个 10 轮对话，包含工具调用）：

| 模型 | Tool Calls | KV Hit Rate | 平均完成时间 |
|------|-----------|-------------|------------|
| Qwen3-4B | 8 | 100% | 31.9s |
| Qwen3-8B | 11 | 100% | 88.5s |

KV hit rate 100% 意味着多轮对话的 system prompt 完全被 prefix cache 命中，后续轮次几乎不做 prefill。这在 agent 工作负载下非常关键——system prompt + tool definitions 通常占整个 context 的 30-50%。

测试方面，142 个单元测试覆盖了所有 CPU 可验证的模块：scheduler 的 backpressure 逻辑、prefix cache 的 radix tree 操作、sampler 的各种 penalty、block manager 的分配/回收、CUDA graph pool 的管理……这些测试不依赖 GPU，在 CI 里 9 秒跑完。GPU 相关的端到端测试另外维护，需要实际 model weights。

---

## 踩过的坑

**Stop string 匹配**：如前所述，不能在 token 级别匹配，必须先 decode 再在字符串级别处理。

**Prefix cache 的 block 对齐**：radix tree 的边以 token 为粒度，但 GPU block 以 16/64 token 为粒度。如果不对齐，换出/换入的时候会出现 partial block，处理起来很麻烦。解决方案是 prefix match 之后向下取整到 block boundary。

**CUDA Graph capture 和 dynamic shape**：CUDA Graph 捕获时假设 tensor shape 固定。Decode 是单 token，shape 不变，适合 Graph。但如果 decode batch size 变了（有新请求加入），Graph 就失效了——要么为每个 batch size 各捕获一个 Graph，要么回退到非 Graph 路径。我选了前者，维护一个 batch pool（支持 1/2/4/8/16/32 等固定 batch size）。

**Async 和 GPU 的交互**：Tokio 是多线程异步运行时，GPU state 含有 raw CUDA pointer，默认不是 `Send`。解决方案是 `unsafe impl Send`，并确保 GPU state 只从单一推理线程访问。这个正确性要靠架构保证，不是编译器保证的——是目前设计里少数几个"相信自己"的地方。

---

## 未来方向

当前的状态可以概括为：**控制平面完整，GPU 内核基础扎实，但还有几个关键缺口**。

**PagedAttention**：目前 KV cache 是连续内存，每个 slot 独享一块。真正的 paged KV 可以让多个请求共享同一段物理 KV block（prefix sharing），内存利用率更高。这是下一个要做的大功能。

**更多模型架构**：目前只支持 Qwen3 和 Qwen3.5。Llama、DeepSeek、Mistral、Gemma 这些主流架构还没有——每个新模型都要实现 `ModelForward` trait，工作量相对可控。

**量化内核**：INT4/FP8/INT8 的检测逻辑已经有了，但 GPU 内核还没有。量化在推理吞吐上的收益很大，这个优先级很高。

**多 GPU（Tensor Parallel）**：TP config 和 sharding math（column_shard/row_shard/head_shard）已经实现，但 NCCL all-reduce/all-gather 还是 stub。多 GPU 推理是 70B+ 模型的刚需，但依赖 NCCL，门槛比单 GPU 高一个台阶。

**Speculative decoding**：草图已经有了，GPU 部分待实现。理论上对小模型做 draft、大模型做 verify，可以把 TBT（time between tokens）压低 2-3 倍。

---

## 结语

写推理引擎这件事，让我对 LLM serving 的理解完全换了一个层次。很多"显而易见"的设计，背后都有不显而易见的 tradeoff：

- Decode 优先还是公平轮转？（decode 优先降低 TBT，但新请求 TTFT 会变高）
- Chunked prefill 的 chunk size 选多大？（太小 GPU 利用率低，太大 decode 请求等太久）
- Prefix cache 用多少 GPU 内存？（cache 命中率高，但占用了可以跑更多请求的显存）

这些都没有唯一正确答案，取决于工作负载。自己实现过一遍之后，才能在读论文或看别人的系统时，真正理解他们在优化什么、放弃了什么。

项目还在持续迭代中。如果你对推理引擎感兴趣，欢迎一起搞。
