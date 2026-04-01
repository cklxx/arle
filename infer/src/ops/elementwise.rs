use anyhow::{Result, anyhow};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Batched element-wise add: out = a + b (same shape HiddenStates)
pub fn add_batch(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;
    add_batch_into(ctx, a, b, &mut out)?;
    Ok(out)
}

/// Batched element-wise add into pre-allocated output buffer (zero allocation).
pub(crate) fn add_batch_into(
    ctx: &DeviceContext,
    a: &HiddenStates,
    b: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.seq_len, b.seq_len);
    assert_eq!(out.hidden_dim, a.hidden_dim);
    assert_eq!(out.seq_len, a.seq_len);

    let n = a.hidden_dim * a.seq_len;
    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched SiLU+mul: out[i] = silu(gate[i]) * up[i]
pub fn silu_mul_batch(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;
    silu_mul_batch_into(ctx, gate, up, &mut out)?;
    Ok(out)
}

/// Batched SiLU+mul into pre-allocated output buffer (zero allocation).
pub(crate) fn silu_mul_batch_into(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    assert_eq!(out.hidden_dim, gate.hidden_dim);
    assert_eq!(out.seq_len, gate.seq_len);

    let n = gate.hidden_dim * gate.seq_len;
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::silu_mul_triton_aot_cuda(
            g_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Extract a single token's vector from a HiddenStates batch (GPU copy)
pub(crate) fn extract_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<DeviceVec> {
    let offset = token_idx * batch.hidden_dim;
    let len = batch.hidden_dim;
    let mut out = DeviceVec::zeros(ctx, len)?;

    let src_view = batch.data.slice(offset..offset + len);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;

    Ok(out)
}

/// Extract into a pre-allocated DeviceVec (zero-alloc D2D copy).
pub(crate) fn extract_vec_into(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
    out: &mut DeviceVec,
) -> Result<()> {
    let offset = token_idx * batch.hidden_dim;
    let src_view = batch.data.slice(offset..offset + batch.hidden_dim);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;
    Ok(())
}

/// Split merged QKV buffer [B, q_dim + 2*kv_dim] into separate Q, K, V buffers.
/// Single kernel launch — no per-token D2D copies.
pub(crate) fn split_qkv_batch(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    v: &mut HiddenStates,
) -> Result<()> {
    let batch_size = qkv.seq_len;
    let q_dim = q.hidden_dim;
    let kv_dim = k.hidden_dim;
    debug_assert_eq!(qkv.hidden_dim, q_dim + 2 * kv_dim);

    let (qkv_ptr, _g0) = qkv.data.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _g2) = k.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _g3) = v.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::split_qkv_cuda(
            qkv_ptr as *const ffi::Half,
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *mut ffi::Half,
            batch_size as i32,
            q_dim as i32,
            kv_dim as i32,
            ctx.stream.cu_stream(),
        );
    }
    Ok(())
}

/// Fused SiLU-mul from merged gate+up buffer [B, 2*inter_dim].
/// out = silu(gate_up[..inter_dim]) * gate_up[inter_dim..]
pub(crate) fn silu_mul_fused_batch_into(
    ctx: &DeviceContext,
    gate_up: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let batch_size = gate_up.seq_len;
    let inter_dim = out.hidden_dim;
    debug_assert_eq!(gate_up.hidden_dim, 2 * inter_dim);

    let (gu_ptr, _g0) = gate_up.data.device_ptr(&ctx.stream);
    let (out_ptr, _g1) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::silu_mul_fused_cuda(
            gu_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            batch_size as i32,
            inter_dim as i32,
            ctx.stream.cu_stream(),
        );
    }
    Ok(())
}
