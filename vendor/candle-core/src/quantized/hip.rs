//! Quantized storage for the HIP backend (Phase 45e / ADR 073).
//!
//! Fills the `quantized::cuda` slot when the `hip` feature is on and `cuda` is not (see
//! `quantized/mod.rs`). `CudaDevice` / `CudaStorage` here resolve to the HIP types
//! (`HipDevice` / `HipStorage`, re-exported); `QCudaStorage` is the HIP `QStorage` variant.
//!
//! **GGUF coverage:** the GPU dequant kernel implements **Q4_K** (the dominant K-quant for
//! 7B–27B GGUFs). Q4_K weights live as raw 144-byte/256-weight superblocks resident on the
//! GPU; `dequantize` expands them into an f32 device buffer via a HIP kernel that mirrors
//! candle's CPU `BlockQ4K::to_float` bit-for-bit, and `fwd` runs that f32 weight matrix
//! through the existing rocBLAS GEMM path (`HipStorage::matmul`). Other K-quants / legacy
//! quants dequantize on the CPU (slow path, used by `dequantize` for any dtype) but the fast
//! GPU dequant kernel + `fwd` GEMM are Q4_K-only here — every other dtype returns a clear
//! "not yet implemented (Phase 45e)" error from `fwd`.
//!
//! Construction mirrors `cuda::load_quantized`: the GGUF block bytes are uploaded H2D into a
//! device buffer (`QCudaStorage { data, dtype, device }`). `quantize*` go via candle's CPU
//! quantizer then H2D, matching `cuda.rs`.
#![allow(unused)]
use super::{GgmlDType, GgmlType, QStorage};
use crate::backend::BackendDevice;
use crate::hip_backend::runtime;
use crate::{CpuStorage, CudaDevice, CudaStorage, DType, Device, Error, Result, Shape, Storage};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ---- GEMV profiling (Phase 45h, AKH_HIP_PROFILE=1) -------------------------
// Wall-time spent inside the fused decode GEMV (with a sync around it, so it is
// the kernel's *intrinsic* time, not the async-overlapped time) + a call count.
// Lets the bench split decode time into "quantized matmul" vs "everything else"
// without rocprof. Off by default (the sync would otherwise serialize decode).
pub static GEMV_NANOS: AtomicU64 = AtomicU64::new(0);
pub static GEMV_CALLS: AtomicU64 = AtomicU64::new(0);

fn gemv_profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("AKH_HIP_PROFILE").as_deref() == Ok("1"))
}

/// Largest activation-row count `m` the fused *batched* GEMV (Phase 47d) handles; must match
/// `AKH_GEMV_BATCH_MAX` in `quant.hip` (one template instantiation per `m` up to this). Above
/// it the dequant+GEMM path amortizes the dequant over enough rows to win, so `fwd` routes
/// larger `m` there. `m == 1` keeps the proven 45h single-row GEMV.
const AKH_GEMV_BATCH_MAX: usize = 16;

/// Whether the fused batched GEMV (Phase 47d) is enabled for `2 ≤ m ≤ AKH_GEMV_BATCH_MAX`.
///
/// Defaults **off**: this kernel was authored off-GPU (no ROCm to compile/measure against), so
/// it must be parity- and perf-validated on hardware before becoming the default. Set
/// `AKH_GEMV_BATCH=1` to route small-`m` quantized matmuls (the spec-decode verify forward,
/// short prefills) through it; when off, those keep the byte-identical dequant+GEMM path.
/// Read fresh (not cached) — only consulted on the `m > 1` branch, never the m=1 decode hot
/// path, so the env lookup never touches per-token decode latency.
fn gemv_batch_enabled() -> bool {
    std::env::var("AKH_GEMV_BATCH").as_deref() == Ok("1")
}

/// `(total_nanos, calls)` spent in the fused GEMV since process start. Flushes any
/// pending event-profiler samples first (event mode) so the read is complete.
pub fn gemv_profile() -> (u64, u64) {
    crate::hip_backend::flush_events();
    (GEMV_NANOS.load(Ordering::Relaxed), GEMV_CALLS.load(Ordering::Relaxed))
}

// ---- Q4_K dequant kernel library (built by build.rs → libcandle_hip_quant.so) ----

type DequantQ4K =
    unsafe extern "C" fn(*const u8, *mut f32, usize, usize) -> c_int;
type DequantQ6K =
    unsafe extern "C" fn(*const u8, *mut f32, usize, usize) -> c_int;
// Fused GEMV: out[n_rows] = W[n_rows,in_dim] · x[in_dim], dequant inline.
type GemvQK = unsafe extern "C" fn(*const u8, *const f32, *mut f32, usize, usize) -> c_int;
// Fused *batched* GEMV (Phase 47d): out[m,n_rows] = X[m,in_dim] · W[n_rows,in_dim]ᵀ, dequant
// inline and reused across the m rows. Trailing `m` selects the template instantiation;
// returns -1 if `m` is outside `1..=AKH_GEMV_BATCH_MAX` (caller routes larger m to GEMM).
type GemvBatchedQK =
    unsafe extern "C" fn(*const u8, *const f32, *mut f32, usize, usize, usize) -> c_int;
// get_rows: out[n_rows*in_dim] = dequant(table[indices[i]]) — embedding lookup off the
// quantized table (Phase 46). `indices` is a device u32 array of length `n_rows`.
type GetRowsQK =
    unsafe extern "C" fn(*const u8, *const u32, *mut f32, usize, usize) -> c_int;
// Phase 46c: set the launch stream for this kernel library (null = default stream).
type SetStream = unsafe extern "C" fn(*mut c_void);

struct QuantKernels {
    _lib: libloading::Library,
    dequant_q4k: DequantQ4K,
    dequant_q6k: DequantQ6K,
    gemv_q4k: GemvQK,
    gemv_q6k: GemvQK,
    gemv_batched_q4k: GemvBatchedQK,
    gemv_batched_q6k: GemvBatchedQK,
    get_rows_q4k: GetRowsQK,
    get_rows_q6k: GetRowsQK,
    set_stream: SetStream,
}
// The fn pointers live in a process-lifetime library; calls are externally synchronized
// to the synchronous compute thread (same contract as `hip_backend::runtime::Kernels`).
unsafe impl Send for QuantKernels {}
unsafe impl Sync for QuantKernels {}

impl QuantKernels {
    fn load() -> std::result::Result<Self, String> {
        let path = concat!(env!("OUT_DIR"), "/libcandle_hip_quant.so");
        let lib = unsafe { libloading::Library::new(path) }.map_err(|e| {
            format!("could not dlopen {path}: {e} (was the HIP toolchain present at build time?)")
        })?;
        unsafe {
            let dequant_q4k: DequantQ4K = *lib
                .get(b"akh_dequant_q4k\0")
                .map_err(|e| format!("missing akh_dequant_q4k: {e}"))?;
            let dequant_q6k: DequantQ6K = *lib
                .get(b"akh_dequant_q6k\0")
                .map_err(|e| format!("missing akh_dequant_q6k: {e}"))?;
            let gemv_q4k: GemvQK = *lib
                .get(b"akh_gemv_q4k_f32\0")
                .map_err(|e| format!("missing akh_gemv_q4k_f32: {e}"))?;
            let gemv_q6k: GemvQK = *lib
                .get(b"akh_gemv_q6k_f32\0")
                .map_err(|e| format!("missing akh_gemv_q6k_f32: {e}"))?;
            let gemv_batched_q4k: GemvBatchedQK = *lib
                .get(b"akh_gemv_batched_q4k_f32\0")
                .map_err(|e| format!("missing akh_gemv_batched_q4k_f32: {e}"))?;
            let gemv_batched_q6k: GemvBatchedQK = *lib
                .get(b"akh_gemv_batched_q6k_f32\0")
                .map_err(|e| format!("missing akh_gemv_batched_q6k_f32: {e}"))?;
            let get_rows_q4k: GetRowsQK = *lib
                .get(b"akh_get_rows_q4k_f32\0")
                .map_err(|e| format!("missing akh_get_rows_q4k_f32: {e}"))?;
            let get_rows_q6k: GetRowsQK = *lib
                .get(b"akh_get_rows_q6k_f32\0")
                .map_err(|e| format!("missing akh_get_rows_q6k_f32: {e}"))?;
            let set_stream: SetStream = *lib
                .get(b"akh_set_stream\0")
                .map_err(|e| format!("missing akh_set_stream: {e}"))?;
            Ok(QuantKernels {
                _lib: lib,
                dequant_q4k,
                dequant_q6k,
                gemv_q4k,
                gemv_q6k,
                gemv_batched_q4k,
                gemv_batched_q6k,
                get_rows_q4k,
                get_rows_q6k,
                set_stream,
            })
        }
    }

    /// Row-major Q4_K dequant: `rows * in_dim` f32 into `out` (device ptr). `blocks` is the
    /// raw GGUF Q4_K bytes (device ptr) for `rows` rows of `in_dim` weights (in_dim % 256).
    fn dequant_q4k(
        &self,
        blocks: *const u8,
        out: *mut f32,
        rows: usize,
        in_dim: usize,
    ) -> std::result::Result<(), String> {
        let rc = unsafe { (self.dequant_q4k)(blocks, out, rows, in_dim) };
        if rc != 0 {
            return Err(format!("akh_dequant_q4k kernel failed (hipError={rc})"));
        }
        Ok(())
    }

    /// Row-major Q6_K dequant (Phase 45f): same calling convention as [`dequant_q4k`].
    fn dequant_q6k(
        &self,
        blocks: *const u8,
        out: *mut f32,
        rows: usize,
        in_dim: usize,
    ) -> std::result::Result<(), String> {
        let rc = unsafe { (self.dequant_q6k)(blocks, out, rows, in_dim) };
        if rc != 0 {
            return Err(format!("akh_dequant_q6k kernel failed (hipError={rc})"));
        }
        Ok(())
    }

    /// Fused GEMV (Phase 45g): `out[n_rows] = W[n_rows,in_dim] · x[in_dim]`, weights
    /// dequantized inline. `dtype` selects the Q4_K / Q6_K kernel.
    fn gemv(
        &self,
        dtype: GgmlDType,
        w: *const u8,
        x: *const f32,
        out: *mut f32,
        n_rows: usize,
        in_dim: usize,
    ) -> std::result::Result<(), String> {
        let f = match dtype {
            GgmlDType::Q4K => self.gemv_q4k,
            GgmlDType::Q6K => self.gemv_q6k,
            other => return Err(format!("HIP GEMV: dtype {other:?} not supported")),
        };
        let rc = unsafe { f(w, x, out, n_rows, in_dim) };
        if rc != 0 {
            return Err(format!("akh_gemv ({dtype:?}) kernel failed (hipError={rc})"));
        }
        Ok(())
    }

    /// Fused *batched* GEMV (Phase 47d): `out[m, n_rows] = X[m, in_dim] · W[n_rows, in_dim]ᵀ`,
    /// weights dequantized inline and reused across the `m` rows (no f32 weight temp). `x` is a
    /// contiguous row-major `[m, in_dim]` f32 activation; `out` a `[m, n_rows]` f32 buffer.
    /// `m` must be in `1..=AKH_GEMV_BATCH_MAX` (the kernel returns -1 otherwise — surfaced here
    /// as an error so the caller can fall back to the GEMM path).
    fn gemv_batched(
        &self,
        dtype: GgmlDType,
        w: *const u8,
        x: *const f32,
        out: *mut f32,
        n_rows: usize,
        in_dim: usize,
        m: usize,
    ) -> std::result::Result<(), String> {
        let f = match dtype {
            GgmlDType::Q4K => self.gemv_batched_q4k,
            GgmlDType::Q6K => self.gemv_batched_q6k,
            other => return Err(format!("HIP batched GEMV: dtype {other:?} not supported")),
        };
        let rc = unsafe { f(w, x, out, n_rows, in_dim, m) };
        if rc != 0 {
            return Err(format!(
                "akh_gemv_batched ({dtype:?}) kernel failed for m={m} (hipError/code={rc}); \
                 m must be in 1..={AKH_GEMV_BATCH_MAX}"
            ));
        }
        Ok(())
    }

    /// Quantized embedding lookup (Phase 46): `out[n_rows*in_dim]` f32, row `i` =
    /// dequant(`table[indices[i]]`). `blocks` is the quantized table, `indices` a device
    /// u32 array of length `n_rows`. `dtype` selects the Q4_K / Q6_K kernel.
    fn get_rows(
        &self,
        dtype: GgmlDType,
        blocks: *const u8,
        indices: *const u32,
        out: *mut f32,
        n_rows: usize,
        in_dim: usize,
    ) -> std::result::Result<(), String> {
        let f = match dtype {
            GgmlDType::Q4K => self.get_rows_q4k,
            GgmlDType::Q6K => self.get_rows_q6k,
            other => return Err(format!("HIP get_rows: dtype {other:?} not supported")),
        };
        let rc = unsafe { f(blocks, indices, out, n_rows, in_dim) };
        if rc != 0 {
            return Err(format!("akh_get_rows ({dtype:?}) kernel failed (hipError={rc})"));
        }
        Ok(())
    }
}

static QUANT_KERNELS: OnceLock<std::result::Result<QuantKernels, String>> = OnceLock::new();

/// The process-wide Q4_K dequant kernel library, dlopen'd on first use.
fn quant_kernels() -> Result<&'static QuantKernels> {
    QUANT_KERNELS
        .get_or_init(QuantKernels::load)
        .as_ref()
        .map_err(|e| Error::Msg(e.clone()))
}

/// Point the quantized kernel library's launches at `s` (Phase 46c graph capture; null =
/// default stream). No-op if the library hasn't loaded yet (nothing to launch).
pub fn set_stream(s: *mut c_void) {
    if let Ok(k) = quant_kernels() {
        unsafe { (k.set_stream)(s) }
    }
}

/// Quantized weights resident on a [`CudaDevice`] (HIP GPU). `data` is the raw GGUF block
/// bytes (no row padding — the dequant kernel reads exactly `ceil(elem/256)` blocks per row);
/// `data_len` is its byte length. Re-exported as the `cuda::QCudaStorage` slot.
pub struct QCudaStorage {
    /// Device pointer to the raw quantized block bytes. Null iff `data_len == 0`.
    data: *mut c_void,
    /// Byte length of the quantized block data.
    data_len: usize,
    dtype: GgmlDType,
    device: CudaDevice,
}

unsafe impl Send for QCudaStorage {}
unsafe impl Sync for QCudaStorage {}

impl Drop for QCudaStorage {
    fn drop(&mut self) {
        if let Ok(h) = runtime::hip() {
            h.free(self.data);
        }
    }
}

/// Upload raw quantized bytes H2D into a fresh device allocation. The buffer owns the
/// allocation (freed when the returned `QCudaStorage` drops).
fn upload_bytes(device: &CudaDevice, bytes: &[u8], dtype: GgmlDType) -> Result<QCudaStorage> {
    let h = runtime::hip().map_err(Error::Msg)?;
    let ptr = h.malloc(bytes.len()).map_err(Error::Msg)?;
    if !bytes.is_empty() {
        if let Err(e) = h.memcpy(
            ptr,
            bytes.as_ptr() as *const c_void,
            bytes.len(),
            runtime::MEMCPY_H2D,
        ) {
            h.free(ptr);
            return Err(Error::Msg(e));
        }
    }
    Ok(QCudaStorage {
        data: ptr,
        data_len: bytes.len(),
        dtype,
        device: device.clone(),
    })
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = el_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let h = runtime::hip().map_err(Error::Msg)?;
        let ptr = h.malloc(size_in_bytes).map_err(Error::Msg)?;
        if size_in_bytes != 0 {
            if let Err(e) = h.memset(ptr, 0, size_in_bytes) {
                h.free(ptr);
                return Err(Error::Msg(e));
            }
        }
        Ok(QCudaStorage {
            data: ptr,
            data_len: size_in_bytes,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Number of quantized blocks currently resident.
    fn n_blocks(&self) -> usize {
        self.data_len / self.dtype.type_size()
    }

    /// Dequantize `elem_count` weights into a fresh f32 device buffer, returned as a
    /// `CudaStorage`. Q4_K uses the on-GPU dequant kernel (row-major, bit-for-bit parity with
    /// the CPU `to_float`); every other dtype falls back to a CPU dequant + H2D upload so the
    /// op still works (just without GPU acceleration of the dequant itself).
    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        match self.dtype {
            GgmlDType::Q4K => self.dequantize_q4k(elem_count),
            GgmlDType::Q6K => self.dequantize_q6k(elem_count),
            _ => self.dequantize_cpu(elem_count),
        }
    }

    fn dequantize_q4k(&self, elem_count: usize) -> Result<CudaStorage> {
        if !elem_count.is_multiple_of(256) {
            return Err(Error::Msg(format!(
                "HIP Q4_K dequantize: elem_count {elem_count} not a multiple of the 256-weight \
                 block size"
            )));
        }
        let h = runtime::hip().map_err(Error::Msg)?;
        let k = quant_kernels()?;
        let out = h
            .malloc(elem_count * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        // The kernel writes row-major; one "row" of `elem_count` covers the whole tensor here.
        if let Err(e) = k.dequant_q4k(self.data as *const u8, out as *mut f32, 1, elem_count) {
            h.free(out);
            return Err(Error::Msg(e));
        }
        // No sync (Phase 45h): ordered on the null stream; consumed by the GEMM that
        // follows on the same stream, read back to host only at the logits D2H.
        Ok(CudaStorage::from_raw(
            self.device.clone(),
            out,
            DType::F32,
            elem_count,
        ))
    }

    /// On-GPU Q6_K dequant (Phase 45f). Same shape contract as [`dequantize_q4k`]: writes a
    /// fresh row-major f32 device buffer, bit-for-bit parity with CPU `BlockQ6K::to_float`.
    fn dequantize_q6k(&self, elem_count: usize) -> Result<CudaStorage> {
        if !elem_count.is_multiple_of(256) {
            return Err(Error::Msg(format!(
                "HIP Q6_K dequantize: elem_count {elem_count} not a multiple of the 256-weight \
                 block size"
            )));
        }
        let h = runtime::hip().map_err(Error::Msg)?;
        let k = quant_kernels()?;
        let out = h
            .malloc(elem_count * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        if let Err(e) = k.dequant_q6k(self.data as *const u8, out as *mut f32, 1, elem_count) {
            h.free(out);
            return Err(Error::Msg(e));
        }
        // No sync (Phase 45h): ordered on the null stream; consumed by the GEMM that
        // follows on the same stream, read back to host only at the logits D2H.
        Ok(CudaStorage::from_raw(
            self.device.clone(),
            out,
            DType::F32,
            elem_count,
        ))
    }

    /// Quantized embedding lookup (Phase 46 VRAM): dequantize only the rows named in
    /// `indices` (host u32), returning a fresh row-major f32 `[n_rows, in_dim]` device
    /// buffer — so the token_embd table stays Q4_K/Q6_K resident (~0.83 GB) instead of a
    /// 4 GB f32 expansion. Q4_K/Q6_K only (the embedding dtypes); other dtypes return an
    /// error so the caller can fall back to dequantize+index_select.
    pub fn get_rows(&self, indices: &[u32], in_dim: usize) -> Result<CudaStorage> {
        if !in_dim.is_multiple_of(self.dtype.block_size()) {
            return Err(Error::Msg(format!(
                "HIP get_rows: row width {in_dim} not a multiple of the {:?} block size",
                self.dtype
            )));
        }
        let h = runtime::hip().map_err(Error::Msg)?;
        let k = quant_kernels()?;
        let n = indices.len();
        let out = h
            .malloc(n * in_dim * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        // Upload the (small) index list H2D for the kernel.
        let idx_bytes = n * std::mem::size_of::<u32>();
        let idx_dev = match h.malloc(idx_bytes) {
            Ok(p) => p,
            Err(e) => {
                h.free(out);
                return Err(Error::Msg(e));
            }
        };
        if let Err(e) = h.memcpy(
            idx_dev,
            indices.as_ptr() as *const c_void,
            idx_bytes,
            runtime::MEMCPY_H2D,
        ) {
            h.free(out);
            h.free(idx_dev);
            return Err(Error::Msg(e));
        }
        let res = k.get_rows(
            self.dtype,
            self.data as *const u8,
            idx_dev as *const u32,
            out as *mut f32,
            n,
            in_dim,
        );
        // Index upload is a synchronous hipMemcpy and the kernel is queued after it on the
        // null stream, so recycling idx_dev now is safe (same contract as other transients).
        h.free(idx_dev);
        if let Err(e) = res {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(CudaStorage::from_raw(
            self.device.clone(),
            out,
            DType::F32,
            n * in_dim,
        ))
    }

    /// CPU fallback dequant for non-Q4_K dtypes: D2H the blocks, dequantize on the CPU via the
    /// `GgmlType::to_float` implementations, then upload the f32 result H2D.
    fn dequantize_cpu(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }
        let buffer = self.data()?;
        let mut out = vec![0.0f32; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, elem_count, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, elem_count, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, elem_count, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }
        // Upload the dequantized f32 to the device.
        self.device.storage_from_cpu_storage(&CpuStorage::F32(out))
    }

    /// f16 dequant. The HIP backend has no f16 dequant kernel yet, so dequantize to f32 on the
    /// device and cast to f16 (the cast kernel is part of the elementwise library).
    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        use crate::backend::BackendStorage;
        let f32_storage = self.dequantize(elem_count)?;
        let layout = crate::Layout::contiguous((elem_count,));
        f32_storage.to_dtype(&layout, DType::F16)
    }

    /// Quantize a device-resident f32 tensor: D2H → CPU quantize → H2D (mirrors `cuda.rs`).
    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        use crate::backend::BackendStorage;
        if src.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP quantize: only f32 can be quantized, got {:?}",
                src.dtype()
            )));
        }
        let host = match src.to_cpu_storage()? {
            CpuStorage::F32(v) => v,
            other => {
                return Err(Error::Msg(format!(
                    "HIP quantize: expected f32 host storage, got {:?}",
                    crate::backend::BackendStorage::dtype(&other)
                )))
            }
        };
        self.quantize_from_f32(&host)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &CudaStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::Msg(
            "HIP quantized backend: `quantize_imatrix` not yet implemented (Phase 45e)."
                .to_string(),
        ))
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::Msg(
            "HIP quantized backend: `quantize_imatrix_onto` not yet implemented (Phase 45e)."
                .to_string(),
        ))
    }

    /// Quantize a CPU-resident f32 tensor onto this device: CPU quantize → H2D.
    pub fn quantize_onto(&mut self, src: &CpuStorage) -> Result<()> {
        let host = src.as_slice::<f32>()?.to_vec();
        self.quantize_from_f32(&host)
    }

    /// Shared CPU-quantize + H2D-upload path for `quantize` / `quantize_onto`.
    fn quantize_from_f32(&mut self, host: &[f32]) -> Result<()> {
        let src_len = host.len();
        let src = Storage::Cpu(CpuStorage::F32(host.to_vec()));
        let mut qcpu = Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu.quantize(&src)?;
        let data = qcpu.data()?;
        let uploaded = upload_bytes(&self.device, &data, self.dtype)?;
        // Replace our buffer (the old allocation is freed by the temporary's Drop).
        let old = std::mem::replace(self, uploaded);
        // `old` (a QCudaStorage holding the previous device ptr) drops here and frees it.
        // We must not also free `self.data` (now the new ptr); `old.data` is the stale one.
        drop(old);
        Ok(())
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data as *const u8)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data_len
    }

    /// Quantized matmul: `out = x · Wᵀ` where `W` is `self` with shape `(n, k)` (transposed,
    /// per candle's `QMatMul`) and `x` is the activation with shape `(.., k)`. Dequantizes the
    /// Q4_K weights to f32 on-device, then runs candle's f32 GEMM (`HipStorage::matmul`, the
    /// rocBLAS path) with the same `(b, m, n, k)` contract as `cuda.rs`'s `dequantize_matmul`
    /// DMMV fallback.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        // Q4_K (45e) + Q6_K (45f) dequant on the GPU; both feed the same rocBLAS f32 GEMM.
        // gemma3-12b (Q4_K_M) carries both — Q6_K backs ffn_down/attn_v/output, touched every
        // token, so resident decode requires it. Other dtypes still error (CPU-fallback dequant
        // would round-trip every token, defeating residency — add their GPU kernels as needed).
        if !matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q6K) {
            return Err(Error::Msg(format!(
                "HIP quantized matmul (`fwd`): only Q4_K/Q6_K are implemented (Phase 45e/45f); \
                 got {:?}. Other GGUF dtypes land in a later step.",
                self.dtype
            )));
        }
        if storage.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP quantized matmul: activation must be f32 (Phase 45e), got {:?}",
                storage.dtype()
            )));
        }
        // self is transposed: (n, k) = (out_features, in_features).
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => {
                return Err(Error::Msg(format!(
                    "HIP quantized matmul: unexpected input shape {s:?}"
                )))
            }
        };
        if k2 != k {
            return Err(Error::Msg(format!(
                "HIP quantized matmul: dim mismatch {self_shape:?} vs {:?}",
                layout.shape()
            )));
        }

        // Decode fast path (Phase 45g): a single-row activation (b·m == 1) is a GEMV.
        // The fused kernel dequantizes each weight inline and accumulates the dot
        // product — streaming only the quantized weights, never materializing the 4×
        // f32 weight matrix (that temp was the per-token VRAM + bandwidth hog). Needs a
        // contiguous f32 activation and in_dim % 256 == 0 (the K-quant block); else fall
        // through to the dequant + GEMM path (prefill / batched).
        if b * m == 1 && layout.is_contiguous() && k % 256 == 0 {
            let h = runtime::hip().map_err(Error::Msg)?;
            let kern = quant_kernels()?;
            let x_ptr =
                unsafe { (storage.device_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = h.malloc(n * std::mem::size_of::<f32>()).map_err(Error::Msg)?;
            let run = || kern.gemv(self.dtype, self.data as *const u8, x_ptr, out_ptr as *mut f32, n, k);
            // Three profile modes (mirrors hip_backend::profiled!): event markers (pure
            // GPU time, no drain), legacy sync-bracket, or off.
            let res = if crate::hip_backend::event_profiling() {
                GEMV_CALLS.fetch_add(1, Ordering::Relaxed);
                crate::hip_backend::event_time(&GEMV_NANOS, run)
            } else if gemv_profiling() {
                let _ = h.synchronize();
                let t0 = std::time::Instant::now();
                let r = run();
                let _ = h.synchronize();
                GEMV_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                GEMV_CALLS.fetch_add(1, Ordering::Relaxed);
                r
            } else {
                run()
            };
            if let Err(e) = res {
                h.free(out_ptr);
                return Err(Error::Msg(e));
            }
            // No sync (Phase 45h): the GEMV is ordered on the null stream; the result
            // is read to host only at the per-token logits D2H.
            let out = CudaStorage::from_raw(self.device.clone(), out_ptr, DType::F32, n);
            let mut out_shape = layout.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(n);
            return Ok((out, out_shape.into()));
        }

        // Verify / small-batch fast path (Phase 47d, opt-in `AKH_GEMV_BATCH=1`): a few
        // activation rows (`2 ≤ b·m ≤ AKH_GEMV_BATCH_MAX`) — the spec-decode verify forward,
        // short prefills — over a contiguous f32 activation with `k % 256 == 0`. The fused
        // *batched* GEMV dequantizes each weight superblock once and reuses it across all
        // `b·m` rows in registers, so it never materializes the 4× f32 weight temp the GEMM
        // path below writes+reads. For larger `m` the GEMM amortizes that dequant over enough
        // rows to win, so we cap and fall through. Output is row-major `[b·m, n]`, matching the
        // GEMM path's `[b, m, n]` flattened.
        let rows_x = b * m;
        if (2..=AKH_GEMV_BATCH_MAX).contains(&rows_x)
            && layout.is_contiguous()
            && k % 256 == 0
            && gemv_batch_enabled()
        {
            let h = runtime::hip().map_err(Error::Msg)?;
            let kern = quant_kernels()?;
            let x_ptr =
                unsafe { (storage.device_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = h
                .malloc(rows_x * n * std::mem::size_of::<f32>())
                .map_err(Error::Msg)?;
            if let Err(e) = kern.gemv_batched(
                self.dtype,
                self.data as *const u8,
                x_ptr,
                out_ptr as *mut f32,
                n,
                k,
                rows_x,
            ) {
                h.free(out_ptr);
                return Err(Error::Msg(e));
            }
            // No sync (Phase 45h): ordered on the null stream; consumed downstream, read to
            // host only at the per-token logits D2H.
            let out = CudaStorage::from_raw(self.device.clone(), out_ptr, DType::F32, rows_x * n);
            let mut out_shape = layout.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(n);
            return Ok((out, out_shape.into()));
        }

        // Dequantize the weights to an f32 [n, k] device matrix (row-major).
        let data_f32 = self.dequantize(n * k)?;
        // Mirror `cuda.rs` dequantize_matmul: rhs = Wᵀ as a (b, k, n) broadcast view of the
        // row-major (k, n)-strided [n, k] buffer; candle's matmul reads strides to pick the
        // transpose, so this matches the row-major weight without a physical transpose.
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
        let out = storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?;

        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    /// D2H the raw quantized block bytes.
    pub fn data(&self) -> Result<Vec<u8>> {
        let h = runtime::hip().map_err(Error::Msg)?;
        let mut out = vec![0u8; self.data_len];
        if self.data_len != 0 {
            h.memcpy(
                out.as_mut_ptr() as *mut c_void,
                self.data,
                self.data_len,
                runtime::MEMCPY_D2H,
            )
            .map_err(Error::Msg)?;
        }
        Ok(out)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &CudaStorage,
        _: &crate::Layout,
        _: &CudaStorage,
        _: &crate::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(Error::Msg(
            "HIP quantized backend: `indexed_moe_forward` not yet implemented (Phase 45e)."
                .to_string(),
        ))
    }
}

/// Construct a `QStorage::Cuda` (HIP) from GGUF block data, uploading it H2D. Mirrors
/// `cuda::load_quantized`: the typed block slice is reinterpreted as raw bytes and uploaded.
pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let storage = upload_bytes(device, bytes, T::DTYPE)?;
    Ok(QStorage::Cuda(storage))
}

#[cfg(all(test, feature = "hip"))]
mod test {
    use crate::quantized::{GgmlDType, QMatMul, QTensor};
    use crate::{Device, Module, Result, Tensor};

    fn gpu() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Q4_K dequant on the GPU matches the CPU `to_float` within tolerance, and a
    /// `QMatMul::forward` on the GPU matches the CPU within tolerance. Skips (does not fail)
    /// when no ROCm GPU is visible.
    #[test]
    fn hip_q4k_dequant_and_matmul_parity() -> Result<()> {
        let Some(dev) = gpu() else {
            eprintln!("no HIP GPU visible; skipping hip_q4k_dequant_and_matmul_parity");
            return Ok(());
        };
        let cpu = Device::Cpu;

        // Weight matrix W: [n, k] = [out_features, in_features]. k must be a multiple of 256.
        let (n, k) = (8usize, 512usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.7 + 0.05)
            .collect();

        // CPU reference QTensor (quantize on CPU).
        let w_cpu = Tensor::from_vec(w.clone(), (n, k), &cpu)?;
        let qt_cpu = QTensor::quantize(&w_cpu, GgmlDType::Q4K)?;

        // Same QTensor built on the GPU via quantize_onto (CPU src → GPU storage).
        let qt_gpu = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4K, &dev)?;

        // (a) dequantize parity.
        let deq_cpu = qt_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let deq_gpu = qt_gpu
            .dequantize(&dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let d = max_abs_diff(&deq_cpu, &deq_gpu);
        eprintln!("Q4_K dequantize max|gpu-cpu| = {d}");
        assert!(d < 1e-2, "dequantize parity: max abs diff {d} >= 1e-2");

        // (b) QMatMul::forward parity. x: [m, k]; out: [m, n].
        let m = 3usize;
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.007).cos() * 0.5).collect();
        let x_cpu = Tensor::from_vec(x.clone(), (m, k), &cpu)?;
        let x_gpu = Tensor::from_vec(x.clone(), (m, k), &dev)?;

        // Reference: dequantize on CPU then a plain f32 matmul (x · Wᵀ). Both the GPU
        // quantized `fwd` and the CPU quantized `forward` approximate this; their small
        // mutual difference is f32 GEMM accumulation-order noise (the dequant itself is
        // bit-exact — see part (a)), not a correctness gap. We therefore assert relative
        // closeness, scaled by the output magnitude, as is standard for GEMM parity.
        // (Compute this before moving `qt_gpu` into the QMatMul below.)
        let w_deq = qt_gpu.dequantize(&cpu)?; // [n, k]

        let mm_cpu = QMatMul::from_qtensor(qt_cpu)?;
        let mm_gpu = QMatMul::from_qtensor(qt_gpu)?;

        let y_cpu = mm_cpu.forward(&x_cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let y_gpu = mm_gpu
            .forward(&x_gpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let y_ref = x_cpu
            .matmul(&w_deq.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let scale = y_ref
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);

        let dm_cpu = max_abs_diff(&y_cpu, &y_gpu);
        let dm_ref = max_abs_diff(&y_gpu, &y_ref);
        eprintln!(
            "Q4_K QMatMul::forward: max|gpu-cpu| = {dm_cpu}, max|gpu-ref| = {dm_ref}, \
             out scale = {scale}, rel(gpu-ref) = {}",
            dm_ref / scale
        );
        // GPU quantized matmul must match the true dequantized f32 matmul to f32 GEMM
        // precision (relative), proving the `fwd` path is numerically correct.
        assert!(
            dm_ref / scale < 1e-2,
            "matmul vs dequant-ref: rel diff {} >= 1e-2 (abs {dm_ref}, scale {scale})",
            dm_ref / scale
        );
        // And it must agree with the CPU quantized path to within ~1e-2 relative.
        assert!(
            dm_cpu / scale < 1e-2,
            "matmul gpu-vs-cpu parity: rel diff {} >= 1e-2 (abs {dm_cpu}, scale {scale})",
            dm_cpu / scale
        );

        Ok(())
    }

    /// Q6_K (Phase 45f) dequant on the GPU matches CPU `BlockQ6K::to_float`, and a
    /// `QMatMul::forward` over a Q6_K weight matches the dequantized-f32 reference.
    /// gemma3-12b backs ffn_down/attn_v/output with Q6_K, so this gates resident decode.
    #[test]
    fn hip_q6k_dequant_and_matmul_parity() -> Result<()> {
        let Some(dev) = gpu() else {
            eprintln!("no HIP GPU visible; skipping hip_q6k_dequant_and_matmul_parity");
            return Ok(());
        };
        let cpu = Device::Cpu;

        let (n, k) = (8usize, 512usize); // k % 256 == 0
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.011).cos() * 0.6 - 0.03)
            .collect();
        let w_cpu = Tensor::from_vec(w.clone(), (n, k), &cpu)?;
        let qt_cpu = QTensor::quantize(&w_cpu, GgmlDType::Q6K)?;
        let qt_gpu = QTensor::quantize_onto(&w_cpu, GgmlDType::Q6K, &dev)?;

        // (a) dequantize parity — must be bit-for-bit (the GPU kernel mirrors to_float).
        let deq_cpu = qt_cpu.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let deq_gpu = qt_gpu.dequantize(&dev)?.flatten_all()?.to_vec1::<f32>()?;
        let d = max_abs_diff(&deq_cpu, &deq_gpu);
        eprintln!("Q6_K dequantize max|gpu-cpu| = {d}");
        assert!(d == 0.0, "Q6_K dequant must be bit-exact, got max abs diff {d}");

        // (b) QMatMul::forward parity vs the dequantized-f32 reference matmul.
        let m = 3usize;
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.009).sin() * 0.5).collect();
        let x_cpu = Tensor::from_vec(x.clone(), (m, k), &cpu)?;
        let x_gpu = Tensor::from_vec(x.clone(), (m, k), &dev)?;
        let w_deq = qt_gpu.dequantize(&cpu)?;
        let mm_gpu = QMatMul::from_qtensor(qt_gpu)?;
        let y_gpu = mm_gpu.forward(&x_gpu)?.flatten_all()?.to_vec1::<f32>()?;
        let y_ref = x_cpu.matmul(&w_deq.t()?)?.flatten_all()?.to_vec1::<f32>()?;
        let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let dm_ref = max_abs_diff(&y_gpu, &y_ref);
        eprintln!("Q6_K QMatMul::forward rel(gpu-ref) = {}", dm_ref / scale);
        assert!(
            dm_ref / scale < 1e-2,
            "Q6_K matmul vs dequant-ref: rel diff {} >= 1e-2",
            dm_ref / scale
        );
        Ok(())
    }

    /// The fused decode GEMV (m=1) path matches the dequant-f32 reference matmul for
    /// both Q4_K and Q6_K. Exercises `fwd`'s `b·m == 1` branch (Phase 45g).
    #[test]
    fn hip_gemv_decode_parity() -> Result<()> {
        let Some(dev) = gpu() else {
            eprintln!("no HIP GPU visible; skipping hip_gemv_decode_parity");
            return Ok(());
        };
        let cpu = Device::Cpu;
        let (n, k) = (16usize, 512usize); // k % 256 == 0
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32) * 0.017).sin() * 0.5 + 0.02)
                .collect();
            let w_cpu = Tensor::from_vec(w.clone(), (n, k), &cpu)?;
            let qt_gpu = QTensor::quantize_onto(&w_cpu, dtype, &dev)?;
            let w_deq = qt_gpu.dequantize(&cpu)?; // [n, k]

            // x is a single row -> triggers the GEMV path.
            let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos() * 0.4).collect();
            let x_gpu = Tensor::from_vec(x.clone(), (1, k), &dev)?;
            let x_cpu = Tensor::from_vec(x, (1, k), &cpu)?;

            let mm = QMatMul::from_qtensor(qt_gpu)?;
            let y_gpu = mm.forward(&x_gpu)?.flatten_all()?.to_vec1::<f32>()?;
            let y_ref = x_cpu.matmul(&w_deq.t()?)?.flatten_all()?.to_vec1::<f32>()?;
            let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            let d = max_abs_diff(&y_gpu, &y_ref);
            eprintln!("{dtype:?} GEMV rel(gpu-ref) = {}", d / scale);
            assert!(
                d / scale < 1e-2,
                "{dtype:?} GEMV vs dequant-ref: rel diff {} >= 1e-2",
                d / scale
            );
        }
        Ok(())
    }

    /// The fused **batched** GEMV (Phase 47d, `m>1`) matches the dequant-f32 reference matmul
    /// for both Q4_K and Q6_K across several small `m` — the spec-decode verify forward's path.
    /// Forces the opt-in `AKH_GEMV_BATCH` route (default-off; this kernel is on-GPU-unvalidated
    /// until this test passes) and uses an `n` whose last block is not warp-aligned to exercise
    /// the `row >= n_rows` tail. Run the hip test suite with `--test-threads=1`: the flag is set
    /// process-wide here.
    #[test]
    fn hip_gemv_batched_parity() -> Result<()> {
        let Some(dev) = gpu() else {
            eprintln!("no HIP GPU visible; skipping hip_gemv_batched_parity");
            return Ok(());
        };
        // SAFETY: hardware-gated test; intended to run single-threaded (--test-threads=1).
        unsafe { std::env::set_var("AKH_GEMV_BATCH", "1") };
        let cpu = Device::Cpu;
        let (n, k) = (20usize, 512usize); // n=20 → grid tail (20 = 2*8 + 4 active warps); k%256==0
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32) * 0.017).sin() * 0.5 + 0.02)
                .collect();
            let w_cpu = Tensor::from_vec(w.clone(), (n, k), &cpu)?;
            // Dequant reference (build a throwaway QTensor; quantize_onto consumes it).
            let w_deq = QTensor::quantize_onto(&w_cpu, dtype, &dev)?.dequantize(&cpu)?; // [n, k]
            for m in [2usize, 4, 7, 9] {
                let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos() * 0.4).collect();
                let x_gpu = Tensor::from_vec(x.clone(), (m, k), &dev)?;
                let x_cpu = Tensor::from_vec(x, (m, k), &cpu)?;
                // Fresh QTensor per m (from_qtensor consumes it). Triggers the m>1 batched path.
                let qt_gpu = QTensor::quantize_onto(&w_cpu, dtype, &dev)?;
                let mm = QMatMul::from_qtensor(qt_gpu)?;
                let y_gpu = mm.forward(&x_gpu)?.flatten_all()?.to_vec1::<f32>()?;
                let y_ref = x_cpu.matmul(&w_deq.t()?)?.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(y_gpu.len(), m * n, "batched GEMV output shape [m,n] = [{m},{n}]");
                let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
                let d = max_abs_diff(&y_gpu, &y_ref);
                eprintln!("{dtype:?} batched GEMV m={m} rel(gpu-ref) = {}", d / scale);
                assert!(
                    d / scale < 1e-2,
                    "{dtype:?} batched GEMV m={m} vs dequant-ref: rel diff {} >= 1e-2",
                    d / scale
                );
            }
        }
        Ok(())
    }
}
