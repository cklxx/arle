# 2026-03-31 - Throughput Profiling: Decode ITL is the Bottleneck

## Context
At C=8 with Qwen3-4B on A100-40GB, agent-infer achieved 786 tok/s vs SGLang's 898 tok/s (0.88x).
Profiled the scheduler loop to find the gap.

## What Worked

### Measurement approach
Added timing instrumentation at three levels:
1. **Scheduler loop**: step_avg vs overhead_avg per iteration
2. **decode_batch**: pre_graph / graph_replay / logit_extract breakdown
3. **step_prefill_chunk**: forward / KV migration breakdown

### Key findings

**Scheduler overhead is negligible** (~14us/loop). Not the bottleneck.

**Prefill cost per request**: ~3.5ms forward + 0.1ms migration for 6-14 token prompts.
At 50 requests total this is only ~2% of wall time, but at 200 prompts the ramp-up
amortizes and throughput reaches 786 tok/s.

**The real bottleneck is decode ITL: 10.4ms at B=8 vs SGLang's 8.2ms**.
This 2.2ms gap (21% slower per-step) directly explains the throughput difference:
- 8 tokens / 10.4ms = 769 tok/s theoretical
- 8 tokens / 8.2ms = 976 tok/s theoretical

**Pre-graph overhead inside decode_batch**: ~60-100us total (metadata H2D + plan + embedding).
This is <1% of step time, not the issue.

### Optimization: batch short prefills
Changed scheduler to complete all pending short prefills in one loop iteration
before returning to decode. Keeps batch slots full after request turnover.
Impact: small improvement in TTFT p99 (247ms -> 186ms), throughput unchanged.

## Next Steps
The 2.2ms/step decode gap is in GPU kernel efficiency:
- FlashInfer plan runs outside CUDA graph (can it be moved inside?)
- Embedding H2D + lookup runs outside graph (can it be pre-staged?)
- cuBLAS GEMM vs SGLang's kernel choices
- Profile with nsys to get kernel-level breakdown

## Rule
At high concurrency, per-step decode latency (ITL) is the throughput ceiling.
Scheduler and prefill overhead matter only during ramp-up/ramp-down.
Always measure with enough prompts (200+) to reach steady state.
