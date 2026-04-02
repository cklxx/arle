// Copy KV cache from contiguous layout to FlashInfer paged layout.
//
// Contiguous layout: [num_kv_heads, max_seq_len, head_dim] (per layer)
//   offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim
//
// Paged layout (HND): [max_pages, num_kv_heads, page_size, head_dim]
//   logical_page = pos / page_size
//   offset_in_page = pos % page_size
//   physical_page = page_indices[indptr[batch_idx] + logical_page]
//   offset = physical_page * stride_page + kv_head * page_size * head_dim + offset_in_page * head_dim + dim
//
// Grid: (num_kv_heads, seq_len)  Threads: head_dim

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void kv_cache_to_paged_kernel(
    const __nv_bfloat16* __restrict__ k_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ k_paged,             // paged pool
    __nv_bfloat16* __restrict__ v_paged,             // paged pool
    const int32_t* __restrict__ page_indices,        // page table for this slot
    int max_seq_len,         // contiguous max_seq_len
    int seq_len,             // actual tokens to copy
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page)         // num_kv_heads * page_size * head_dim
{
    int kv_head = blockIdx.x;
    int pos = blockIdx.y;
    int dim = threadIdx.x;

    if (pos >= seq_len || dim >= head_dim) return;

    // Source: contiguous offset
    int src_offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim;

    // Destination: paged offset
    int logical_page = pos / page_size;
    int offset_in_page = pos % page_size;
    int physical_page = page_indices[logical_page];
    int dst_offset = physical_page * stride_page
                   + kv_head * page_size * head_dim
                   + offset_in_page * head_dim
                   + dim;

    k_paged[dst_offset] = k_contiguous[src_offset];
    v_paged[dst_offset] = v_contiguous[src_offset];
}

extern "C" cudaError_t kv_cache_to_paged_cuda(
    const __nv_bfloat16* k_contiguous,
    const __nv_bfloat16* v_contiguous,
    __nv_bfloat16* k_paged,
    __nv_bfloat16* v_paged,
    const int32_t* page_indices,
    int max_seq_len,
    int seq_len,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page,
    cudaStream_t stream)
{
    dim3 grid(num_kv_heads, seq_len);
    dim3 block(head_dim);
    kv_cache_to_paged_kernel<<<grid, block, 0, stream>>>(
        k_contiguous, v_contiguous, k_paged, v_paged, page_indices,
        max_seq_len, seq_len, num_kv_heads, page_size, head_dim, stride_page);
    return cudaGetLastError();
}
