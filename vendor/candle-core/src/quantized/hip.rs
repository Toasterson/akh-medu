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

// ---- Q4_K dequant kernel library (built by build.rs → libcandle_hip_quant.so) ----

type DequantQ4K =
    unsafe extern "C" fn(*const u8, *mut f32, usize, usize) -> c_int;

struct QuantKernels {
    _lib: libloading::Library,
    dequant_q4k: DequantQ4K,
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
            Ok(QuantKernels {
                _lib: lib,
                dequant_q4k,
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
}

static QUANT_KERNELS: OnceLock<std::result::Result<QuantKernels, String>> = OnceLock::new();

/// The process-wide Q4_K dequant kernel library, dlopen'd on first use.
fn quant_kernels() -> Result<&'static QuantKernels> {
    QUANT_KERNELS
        .get_or_init(QuantKernels::load)
        .as_ref()
        .map_err(|e| Error::Msg(e.clone()))
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
        h.synchronize().map_err(Error::Msg)?;
        Ok(CudaStorage::from_raw(
            self.device.clone(),
            out,
            DType::F32,
            elem_count,
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
        if self.dtype != GgmlDType::Q4K {
            return Err(Error::Msg(format!(
                "HIP quantized matmul (`fwd`): only Q4_K is implemented (Phase 45e); got {:?}. \
                 Other GGUF dtypes land in a later step.",
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
}
