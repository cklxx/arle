# 2026-03-31 · FlashInfer Batch Decode Segfault

## Context

Enabling FlashInfer's `BatchDecodeWithPagedKVCacheDispatched` for concurrent batched decode caused immediate segfaults. Single-request decode (contiguous KV cache + CUDA Graph) worked fine. The crash appeared only with 2+ concurrent requests in the decode phase.

## Root Cause

**Three bugs**, found through systematic bisection and code review:

### Bug 1: Hardcoded MAX_SEQ=4096 in attention kernels (OOB writes)

All Triton AOT and CUDA C attention kernels used a hardcoded `MAX_SEQ = 4096` constant for KV cache head stride:
- `tools/triton/flash_attention_prefill_kernel.py` line 42: `MAX_SEQ: tl.constexpr = 4096`
- `tools/triton/attention_decode_kernel.py` line 55: `cache_base = kv_head_idx * 4096 * HEAD_DIM`
- `csrc/prefill_attention.cu` line 141: `int max_seq_len = 4096;`

When the scheduler reduced `DEFAULT_MAX_SEQ` from 32768 to 1024, the contiguous KV cache was allocated as `[8 heads * 1024 * 128]` but kernels indexed at `kv_head * 4096 * 128`. Heads 2-7 wrote out of bounds, silently corrupting GPU memory (model weights, buffers). This produced garbage output for ALL requests, not just concurrent ones.

**Fix**: Pass `max_seq_len` as a runtime parameter to all kernels. Updated build.rs AOT signatures, FFI declarations, and Rust callers.

### Bug 2: plan_info allocated on GPU, used by CPU memcpy (segfault)

FlashInfer's `DecodePlan()` writes a small `DecodePlanInfo` struct (~48 bytes) via **CPU `memcpy`**:
```c++
std::memcpy(plan_info_out, &plan_info, sizeof(DecodePlanInfo));
```

The Rust code allocated `plan_info` as a GPU `CudaSlice<u8>` and passed a device pointer. CPU memcpy to a GPU pointer = immediate segfault.

**Fix**: Allocate `plan_info` as pinned host memory via `cuMemAllocHost`, matching how `page_locked_workspace` was already allocated.

### Bug 3: Double token allocation per decode step (metadata corruption)

Both the scheduler's `step_decode_batch()` AND the model's `decode_batch()` called `paged_kv_pool.alloc_tokens(slot, 1)`. Each decode step allocated 2 tokens instead of 1, corrupting FlashInfer's indptr/indices metadata.

**Fix**: Remove allocation from `decode_batch()`, let the scheduler handle it exclusively.

## Debug Process

1. **Bisect**: Checked each commit from known-good (a3d0fd3) to HEAD (f7a60c6). All commits before f7a60c6 were correct. f7a60c6 broke correctness.
2. **Isolate**: Disabled pool path → still garbage. Disabled scatter_write → still garbage. Forced budget=0 → still garbage. This ruled out pool/scatter_write.
3. **Diff analysis**: Compared 400d683 vs f7a60c6 diff. Key change: `DEFAULT_MAX_SEQ 32768 → 1024`. Triton kernels use `MAX_SEQ=4096` as stride.
4. **Kernel audit**: Searched for hardcoded 4096 in all .py/.cu files. Found in 3 kernels + 1 hd256 variant.
5. **Re-enable FlashInfer**: After max_seq_len fix, single request worked. Concurrent still segfaulted.
6. **Code review**: Read `flashinfer_decode.cu` carefully. `memcpy(plan_info_out, ...)` writes to the pointer passed from Rust. Checked Rust side: `CudaSlice<u8>` = GPU allocation. CPU memcpy to GPU pointer = segfault.
7. **Double alloc**: Traced token allocation flow through scheduler → decode_batch. Found both allocating.

## Rule

- **Never hardcode buffer strides as kernel constants** — always pass as runtime parameters or derive from buffer metadata.
- **FlashInfer's plan/run API uses CPU memcpy for plan_info** — allocate as host pinned memory, not GPU.
- **Token/page allocation must happen in exactly one place** — either scheduler or model, never both.
- **When debugging GPU crashes, check HOST↔DEVICE pointer mismatches first** — they cause immediate segfaults that look like GPU memory corruption.
