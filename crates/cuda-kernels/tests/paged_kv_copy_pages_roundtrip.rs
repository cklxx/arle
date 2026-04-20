//! Round-trip test for `PagedKVPool::copy_pages_to_host` +
//! `copy_pages_from_host` (Gap #5 Commit 1, `docs/plans/gap5-kv-tier-demote-prefetch.md`).
//!
//! Writes a known byte pattern into a live pool's K/V buffers for a set
//! of physical pages via `cuMemsetD8`, D→H copies those pages, zeros the
//! source buffers, H→D copies back, and asserts byte-equal recovery.
//! Gated on the `cuda` feature + a real GPU.

#[cfg(feature = "cuda")]
mod cuda_tests {
    use std::sync::{Mutex, OnceLock};

    use cuda_kernels::kv_types::KVCacheDtype;
    use cuda_kernels::paged_kv::TokenKVPool;
    use cuda_kernels::tensor::DeviceContext;

    fn cuda_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("cuda test mutex poisoned")
    }

    /// Fill `len_bytes` starting at `dst_ptr + offset_bytes` with `value`
    /// via `cuMemsetD8_v2` (synchronous on the default stream context).
    fn memset_region(dst_ptr: u64, offset_bytes: usize, len_bytes: usize, value: u8) {
        use cudarc::driver::sys;
        let addr = dst_ptr + offset_bytes as u64;
        unsafe {
            sys::cuMemsetD8_v2(addr, value, len_bytes)
                .result()
                .expect("cuMemsetD8_v2 failed");
        }
    }

    #[test]
    fn copy_pages_roundtrip_bf16() {
        let _guard = cuda_test_guard();

        let num_layers = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 32usize;
        let num_slots = 4usize;
        // 16 MiB budget is comfortably above the minimum the pool allocator
        // needs to produce a few pages with this shape — it derives
        // max_total_tokens from the budget internally.
        let budget_bytes = 16 * 1024 * 1024;

        let ctx = DeviceContext::new().expect("ctx");
        let mut pool = TokenKVPool::new(
            &ctx,
            num_layers,
            num_kv_heads,
            head_dim,
            num_slots,
            budget_bytes,
            KVCacheDtype::BF16,
        )
        .expect("pool new");

        let page_size = pool.page_size;
        let stride = num_kv_heads * page_size * head_dim * 2; // bf16
        assert_eq!(stride, pool.pages_host_byte_len(&[0]) / (num_layers * 2));
        let max_pages = pool.max_total_pages;
        assert!(max_pages >= 4, "need ≥4 pages for test; got {max_pages}");

        // Seed: page p, layer l, K → byte value (p*17 + l*5 + 1) & 0xFF,
        //                  V → byte value (p*17 + l*5 + 128) & 0xFF.
        for page in 0..max_pages {
            for layer in 0..num_layers {
                let k_val: u8 = ((page as u32)
                    .wrapping_mul(17)
                    .wrapping_add(layer as u32 * 5)
                    .wrapping_add(1)
                    & 0xFF) as u8;
                let v_val: u8 = ((page as u32)
                    .wrapping_mul(17)
                    .wrapping_add(layer as u32 * 5)
                    .wrapping_add(128)
                    & 0xFF) as u8;
                memset_region(
                    pool.k_data_ptr(layer, &ctx.stream),
                    page * stride,
                    stride,
                    k_val,
                );
                memset_region(
                    pool.v_data_ptr(layer, &ctx.stream),
                    page * stride,
                    stride,
                    v_val,
                );
            }
        }
        ctx.stream.synchronize().expect("sync post-memset");

        let pages: Vec<u32> = vec![0, 2, 3];

        let expected_len = pool.pages_host_byte_len(&pages);
        let blob = pool
            .copy_pages_to_host(&pages, &ctx.stream)
            .expect("copy_pages_to_host");
        assert_eq!(blob.len(), expected_len);

        for (page_slot, &page) in pages.iter().enumerate() {
            let page_base = page_slot * num_layers * 2 * stride;
            for layer in 0..num_layers {
                let k_off = page_base + layer * 2 * stride;
                let v_off = k_off + stride;
                let k_val: u8 = ((page as u32)
                    .wrapping_mul(17)
                    .wrapping_add(layer as u32 * 5)
                    .wrapping_add(1)
                    & 0xFF) as u8;
                let v_val: u8 = ((page as u32)
                    .wrapping_mul(17)
                    .wrapping_add(layer as u32 * 5)
                    .wrapping_add(128)
                    & 0xFF) as u8;
                assert!(
                    blob[k_off..k_off + stride].iter().all(|&b| b == k_val),
                    "K bytes page {} layer {} not uniform {} (found {:?}…)",
                    page,
                    layer,
                    k_val,
                    &blob[k_off..k_off + 8.min(stride)],
                );
                assert!(
                    blob[v_off..v_off + stride].iter().all(|&b| b == v_val),
                    "V bytes page {} layer {} not uniform {}",
                    page,
                    layer,
                    v_val,
                );
            }
        }

        // Zero the pages on device, then H→D copy back from `blob`.
        for &page in &pages {
            for layer in 0..num_layers {
                memset_region(
                    pool.k_data_ptr(layer, &ctx.stream),
                    page as usize * stride,
                    stride,
                    0,
                );
                memset_region(
                    pool.v_data_ptr(layer, &ctx.stream),
                    page as usize * stride,
                    stride,
                    0,
                );
            }
        }
        ctx.stream.synchronize().expect("sync post-zero");

        pool.copy_pages_from_host(&pages, &blob, &ctx.stream)
            .expect("copy_pages_from_host");

        let blob2 = pool
            .copy_pages_to_host(&pages, &ctx.stream)
            .expect("copy_pages_to_host 2");
        assert_eq!(blob, blob2, "round-trip mismatch");

        // Negative: length-mismatch payload rejected.
        let err = pool
            .copy_pages_from_host(&pages, &blob[..blob.len() - 1], &ctx.stream)
            .expect_err("undersized payload should fail");
        assert!(
            format!("{err}").contains("payload len"),
            "unexpected err: {err}",
        );

        // Negative: out-of-range page rejected.
        let bad_pages: Vec<u32> = vec![max_pages as u32];
        let err = pool
            .copy_pages_to_host(&bad_pages, &ctx.stream)
            .expect_err("oor page should fail");
        assert!(
            format!("{err}").contains("out of range"),
            "unexpected err: {err}",
        );
    }
}
