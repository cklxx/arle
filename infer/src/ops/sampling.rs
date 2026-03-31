use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec};

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Used by benchmarks; model code uses
/// `gpu_sample_into` for both greedy and non-greedy paths.
pub fn argmax(ctx: &DeviceContext, x: &DeviceVec) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                x.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Batched argmax — returns the index of the maximum element for each of B rows.
///
/// Input: `logits` is a contiguous [B, vocab_size] bf16 tensor.
/// Output: `out` is a pre-allocated [B] i32 buffer on device.
/// Launches B blocks of 1024 threads in a single kernel.
pub fn argmax_batch(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    out: &mut CudaSlice<i32>,
    batch_size: usize,
    vocab_size: usize,
) -> Result<Vec<u32>> {
    assert_eq!(
        logits.len,
        batch_size * vocab_size,
        "logits length must equal batch_size * vocab_size"
    );

    {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_batch_cuda(
                l_ptr as *const ffi::Half,
                o_ptr as *mut i32,
                batch_size as i32,
                vocab_size as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H batch argmax read failed: {}", e))?;

    Ok(result.iter().map(|&x| x as u32).collect())
}

/// Launch batched argmax on a HiddenStates [B, vocab] buffer. No sync, no readback.
pub(crate) fn argmax_batch_launch(
    ctx: &DeviceContext,
    logits: &crate::tensor::HiddenStates,
    out: &mut CudaSlice<i32>,
    batch_size: usize,
) -> Result<()> {
    let vocab_size = logits.hidden_dim;
    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_cuda(
            l_ptr as *const ffi::Half,
            o_ptr as *mut i32,
            batch_size as i32,
            vocab_size as i32,
            ctx.stream.cu_stream(),
        );
    }
    Ok(())
}

/// Read back B argmax results into a pre-allocated host slice. Call after sync.
pub(crate) fn argmax_batch_readback_into(
    ctx: &DeviceContext,
    out: &CudaSlice<i32>,
    dst: &mut [i32],
    batch_size: usize,
) -> Result<()> {
    let tmp = ctx
        .stream
        .clone_dtoh(&out.slice(0..batch_size))
        .map_err(|e| anyhow!("D2H batch argmax readback: {e}"))?;
    dst[..batch_size].copy_from_slice(&tmp);
    Ok(())
}

/// GPU sampling: temperature → softmax → top-k → top-p → multinomial.
/// Allocates a temporary output buffer — use `gpu_sample_into` for the decode loop.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    gpu_sample_core(ctx, logits, probs_scratch, &mut out_gpu, params, random_val)
}

/// GPU sampling into pre-allocated buffers — zero allocation, suitable for decode loop.
///
/// Greedy dispatch: argmax kernel. Non-greedy: full sampling kernel.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_core(ctx, logits, probs_scratch, out, params, random_val)
}

fn gpu_sample_core(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_launch(ctx, logits, probs_scratch, out, params, random_val)?;

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Launch the sampling kernel without syncing. Call `ctx.sync()` separately.
pub fn gpu_sample_launch(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<()> {
    if params.is_greedy() {
        let (x_ptr, _gx) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                o_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    } else {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::gpu_sample_cuda(
                l_ptr as *const ffi::Half,
                p_ptr as *mut f32,
                o_ptr as *mut i32,
                logits.len as i32,
                1.0 / params.temperature,
                params.top_k,
                params.top_p,
                random_val,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(())
}

/// Launch sampling kernel using pre-cached raw device pointers.
/// Eliminates device_ptr() overhead on every call.
pub(crate) fn gpu_sample_launch_raw(
    ctx: &DeviceContext,
    logits_ptr: crate::tensor::RawDevicePtr<half::bf16>,
    logits_len: usize,
    probs_ptr: crate::tensor::RawDevicePtr<f32>,
    out_ptr: crate::tensor::RawDevicePtr<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<()> {
    if params.is_greedy() {
        unsafe {
            ffi::argmax_cuda(
                logits_ptr.as_ptr() as *const ffi::Half,
                out_ptr.as_mut_ptr(),
                logits_len as i32,
                ctx.stream.cu_stream(),
            );
        }
    } else {
        unsafe {
            ffi::gpu_sample_cuda(
                logits_ptr.as_ptr() as *const ffi::Half,
                probs_ptr.as_mut_ptr(),
                out_ptr.as_mut_ptr(),
                logits_len as i32,
                1.0 / params.temperature,
                params.top_k,
                params.top_p,
                random_val,
                ctx.stream.cu_stream(),
            );
        }
    }
    Ok(())
}

/// Read back the sampling result after a prior `gpu_sample_launch` + `ctx.sync()`.
pub fn gpu_sample_readback(ctx: &DeviceContext, out: &CudaSlice<i32>) -> Result<u32> {
    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;
    Ok(result[0] as u32)
}
