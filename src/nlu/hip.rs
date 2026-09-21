//! ROCm/HIP GPU compute reachable from the Rust/Candle engine (feature `rocm`, ADR 068).
//!
//! This is the **first kernel** of the AMD GPU backend: a single-precision matrix multiply
//! that executes on the gfx1100 via **rocBLAS**, exposed to Candle as a [`CustomOp2`] so an
//! `akh-medu` Tensor graph can offload a matmul to the GPU while the rest of the graph — and
//! hidden-state extraction — stays in Candle. It proves the engine (not a standalone C++
//! binary) drives the AMD GPU.
//!
//! ## Honesty / scope
//! This op keeps tensors in **CPU storage** and round-trips H2D → sgemm → D2H per call. That
//! is correctness-first and PCIe-bound — it is the building block, **not** a fast full-model
//! path. Running quantized Gemma end-to-end on the GPU needs resident device storage and the
//! quantized kernels (the multi-day fork in ADR 068). What is real here: the matmul math runs
//! on the GPU and matches Candle's CPU result.
//!
//! The HIP/rocBLAS libraries are loaded at runtime via `dlopen` (libloading), so a binary
//! built with `--features rocm` still starts on a box without ROCm — [`RocmBlas::load`]
//! returns an error that callers handle (falling back to CPU).

use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::os::raw::c_int;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CpuStorage, CustomOp2, Layout, Result as CandleResult, Shape, Tensor};

use super::error::{NluError, NluResult};

// --- HIP / rocBLAS C ABI constants ----------------------------------------------------------
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const ROCBLAS_OPERATION_NONE: c_int = 111;
const ROCBLAS_STATUS_SUCCESS: c_int = 0;

/// Host↔device transfer counters (Phase 45a / ADR 073). Incremented on every
/// `hipMemcpy`. The bench [`reset_pcie_counts`] after model load, then divides by
/// generated tokens to report **PCIe crossings/token** — the metric the op-offload
/// path is bottlenecked on (~216/token for a 27-layer 12B). Cheap relaxed atomics,
/// always live (the cost is negligible vs a PCIe copy); only *read* when asked.
pub static PCIE_H2D: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PCIE_D2H: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(host→device, device→host)` `hipMemcpy` counts since the last [`reset_pcie_counts`].
pub fn pcie_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (PCIE_H2D.load(Relaxed), PCIE_D2H.load(Relaxed))
}

/// Zero the PCIe transfer counters (call after model load, before a timed generation).
pub fn reset_pcie_counts() {
    use std::sync::atomic::Ordering::Relaxed;
    PCIE_H2D.store(0, Relaxed);
    PCIE_D2H.store(0, Relaxed);
}

// --- FFI function signatures (HIP runtime API + rocBLAS are plain C ABI) ---------------------
type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type HipDeviceSynchronize = unsafe extern "C" fn() -> c_int;
type RocblasCreateHandle = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type RocblasDestroyHandle = unsafe extern "C" fn(*mut c_void) -> c_int;
#[allow(clippy::too_many_arguments)]
type RocblasSgemm = unsafe extern "C" fn(
    *mut c_void, // handle
    c_int,       // transA
    c_int,       // transB
    c_int,       // m
    c_int,       // n
    c_int,       // k
    *const f32,  // alpha
    *const f32,  // A
    c_int,       // lda
    *const f32,  // B
    c_int,       // ldb
    *const f32,  // beta
    *mut f32,    // C
    c_int,       // ldc
) -> c_int;

const HIP_LIB_CANDIDATES: &[&str] = &[
    "libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.6",
];
const ROCBLAS_LIB_CANDIDATES: &[&str] = &[
    "librocblas.so",
    "/opt/rocm/lib/librocblas.so",
    "/opt/rocm/lib/librocblas.so.4",
];

fn open_first(candidates: &[&str], what: &str) -> NluResult<Library> {
    for cand in candidates {
        // SAFETY: dlopen of a system shared library by a fixed allow-list of names.
        if let Ok(lib) = unsafe { Library::new(cand) } {
            return Ok(lib);
        }
    }
    Err(NluError::ModelLoadFailed {
        reason: format!(
            "could not dlopen {what} (tried {candidates:?}); install the ROCm userspace \
             (see ADR 068 / scripts/rocm_smoke.sh)"
        ),
    })
}

/// C ABI of the dequant kernel launcher (see `src/nlu/hip_kernels/dequant.hip`).
type AkhDequantQ4 =
    unsafe extern "C" fn(*const u8, *const f32, *mut f32, c_int, c_int, c_int) -> c_int;
/// C ABI of the SwiGLU activation kernel: `out[i] = silu(g[i]) * u[i]`, length `n` (i64).
type AkhSiluMul = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64) -> c_int;
/// C ABI of the native GGUF Q4_K dequant kernel: `(blocks, out, rows, in_dim)`.
type AkhDequantQ4K = unsafe extern "C" fn(*const u8, *mut f32, c_int, c_int) -> c_int;

/// A runtime-loaded handle to HIP + rocBLAS for single-precision GEMM on the AMD GPU.
#[derive(Debug)]
pub struct RocmBlas {
    hip: Library,
    rocblas: Library,
    /// The akh-medu HIP dequant kernel (built by build.rs). `None` if it was not compiled
    /// (no HIP toolchain at build time) — the quantized path is then unavailable.
    kernels: Option<Library>,
    /// A rocBLAS handle created **once** at load and reused across every GEMM. Creating a
    /// handle allocates device workspace and queries the device — doing it per call (as the
    /// first cut did) dominates single-token decode, where the activation itself is only a
    /// few KB. A 27B forward issues ~420 GEMMs per token; one persistent handle removes ~420
    /// create/destroy cycles. Destroyed in [`Drop`]. See ADR 068 (on-device-activations).
    handle: *mut c_void,
}

// SAFETY: `handle` is a rocBLAS handle used only on the single synchronous generation thread
// (the OODA loop has no async runtime; `DeviceBuffer` carries the same invariant). rocBLAS
// handles are not safe to *share* across concurrent threads, which we never do.
unsafe impl Send for RocmBlas {}
unsafe impl Sync for RocmBlas {}

impl Drop for RocmBlas {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy) =
            Self::sym::<RocblasDestroyHandle>(&self.rocblas, b"rocblas_destroy_handle\0")
        {
            // SAFETY: handle came from rocblas_create_handle in `load` and is destroyed once.
            unsafe {
                destroy(self.handle);
            }
        }
    }
}

impl RocmBlas {
    /// Load HIP + rocBLAS at runtime. Errors (handled by the caller → CPU fallback) when the
    /// ROCm userspace is not installed.
    pub fn load() -> NluResult<Self> {
        let hip = open_first(HIP_LIB_CANDIDATES, "libamdhip64")?;
        let rocblas = open_first(ROCBLAS_LIB_CANDIDATES, "librocblas")?;
        // Optional for the older CPU-storage adapter, but an explicitly selected
        // bundle must fail closed rather than silently lose quantized GPU support.
        let kernel_path = candle_core::hip_kernel_path::resolve(
            "libakh_hip_kernels.so",
            concat!(env!("OUT_DIR"), "/libakh_hip_kernels.so"),
        );
        let kernels = kernel_path
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|path| unsafe { Library::new(path) }.map_err(|error| error.to_string()));
        let kernels = match kernels {
            Ok(library) => Some(library),
            Err(reason) if std::env::var_os("AKH_HIP_KERNEL_DIR").is_some() => {
                return Err(NluError::ModelLoadFailed { reason });
            }
            Err(_) => None,
        };
        if kernels.is_none() {
            tracing::warn!(
                path = ?kernel_path,
                "HIP dequant kernel not found — quantized GPU path unavailable"
            );
        }
        // Create the persistent rocBLAS handle once.
        let rb_create = Self::sym::<RocblasCreateHandle>(&rocblas, b"rocblas_create_handle\0")?;
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-param sized for one pointer; rc checked below.
        let rc = unsafe { rb_create(&mut handle) };
        if rc != ROCBLAS_STATUS_SUCCESS || handle.is_null() {
            return Err(NluError::ModelLoadFailed {
                reason: format!("rocblas_create_handle failed (status={rc})"),
            });
        }
        Ok(Self {
            hip,
            rocblas,
            kernels,
            handle,
        })
    }

    /// True if the quantized GPU path (dequant kernel) is available.
    pub fn has_dequant_kernel(&self) -> bool {
        self.kernels.is_some()
    }

    /// Dequantize `rows` int4 weight rows into the device scratch `out` (transposed
    /// `[in_dim, rows]`), on the GPU. `q`/`scales` are device pointers already offset to the
    /// chunk's first row.
    fn dequant_q4(
        &self,
        q: *const u8,
        scales: *const f32,
        out: *mut f32,
        rows: usize,
        in_dim: usize,
        group: usize,
    ) -> NluResult<()> {
        let lib = self
            .kernels
            .as_ref()
            .ok_or_else(|| NluError::ModelLoadFailed {
                reason: "HIP dequant kernel not built (no rocm toolchain at build time)".into(),
            })?;
        let f = Self::sym::<AkhDequantQ4>(lib, b"akh_dequant_q4\0")?;
        // SAFETY: device pointers valid & sized for rows*in_dim; group divides in_dim.
        let rc = unsafe {
            f(
                q,
                scales,
                out,
                rows as c_int,
                in_dim as c_int,
                group as c_int,
            )
        };
        if rc != 0 {
            return Err(NluError::ModelLoadFailed {
                reason: format!("akh_dequant_q4 kernel failed (hipError={rc})"),
            });
        }
        Ok(())
    }

    /// `out[i] = silu(g[i]) * u[i]` on the GPU (the SwiGLU activation), all device pointers of
    /// length `n`. Lets the MLP keep its hidden state resident on the GPU instead of bringing
    /// gate/up back to the host to multiply.
    fn silu_mul(&self, g: *const f32, u: *const f32, out: *mut f32, n: usize) -> NluResult<()> {
        let lib = self
            .kernels
            .as_ref()
            .ok_or_else(|| NluError::ModelLoadFailed {
                reason: "HIP silu_mul kernel not built (no rocm toolchain at build time)".into(),
            })?;
        let f = Self::sym::<AkhSiluMul>(lib, b"akh_silu_mul\0")?;
        // SAFETY: g/u/out are device pointers each holding `n` floats.
        let rc = unsafe { f(g, u, out, n as i64) };
        if rc != 0 {
            return Err(NluError::ModelLoadFailed {
                reason: format!("akh_silu_mul kernel failed (hipError={rc})"),
            });
        }
        Ok(())
    }

    /// True if the fused on-device activation kernels are available (silu_mul ships in the same
    /// kernel lib as the dequant kernel).
    pub fn has_fused_kernels(&self) -> bool {
        self.kernels.is_some()
    }

    /// Pure on-device GEMM: row-major `C += A·B` semantics with `C[m,n] = A[m,k] · B[k,n]`,
    /// where **A, B and C are all device pointers** — nothing crosses PCIe. `ldc` is the row
    /// stride of the destination in floats (pass `n` for a contiguous `[m,n]` output, or the
    /// full output width to scatter a column-chunk into a wider `[m, out_dim]` buffer).
    ///
    /// Same column-major operand-swap as [`sgemm`](Self::sgemm): we pass (B, A) so rocBLAS
    /// computes `Cᵀ = Bᵀ·Aᵀ`, which lands as the row-major product. Runs on the default stream
    /// (serialized with the dequant / silu_mul kernels) and does **not** synchronize — the
    /// caller syncs once before reading `C` on the host (the final D2H copy is synchronizing).
    fn sgemm_into(
        &self,
        a_ptr: *const f32,
        m: usize,
        k: usize,
        n: usize,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        ldc: usize,
    ) -> NluResult<()> {
        let rb_sgemm = Self::sym::<RocblasSgemm>(&self.rocblas, b"rocblas_sgemm\0")?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: pointers are device pointers sized for the operands; handle is valid.
        let status = unsafe {
            rb_sgemm(
                self.handle,
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                b_ptr,
                n as c_int,
                a_ptr,
                k as c_int,
                &beta,
                c_ptr,
                ldc as c_int,
            )
        };
        if status != ROCBLAS_STATUS_SUCCESS {
            return Err(NluError::ModelLoadFailed {
                reason: format!("rocblas_sgemm (device) failed (status={status})"),
            });
        }
        Ok(())
    }

    /// Download `n` floats from a device buffer to host (synchronizes).
    fn download_f32(&self, src: *const c_void, n: usize) -> NluResult<Vec<f32>> {
        let hip_memcpy = Self::sym::<HipMemcpy>(&self.hip, b"hipMemcpy\0")?;
        let hip_sync = Self::sym::<HipDeviceSynchronize>(&self.hip, b"hipDeviceSynchronize\0")?;
        let mut out = vec![0f32; n];
        // SAFETY: src holds at least `n` floats; out sized to n; rc checked.
        unsafe {
            if hip_sync() != 0 {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipDeviceSynchronize before D2H failed".into(),
                });
            }
            if hip_memcpy(
                out.as_mut_ptr() as *mut c_void,
                src,
                n * std::mem::size_of::<f32>(),
                HIP_MEMCPY_DEVICE_TO_HOST,
            ) != 0
            {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMemcpy D2H failed".into(),
                });
            }
        }
        PCIE_D2H.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }

    /// Native GGUF **Q4_K** dequant: `rows` rows of `in_dim` weights (raw Q4_K blocks at `blocks`)
    /// into the transposed device scratch `out` (`[in_dim, rows]`), matching candle's CPU
    /// `BlockQ4K::to_float` bit-for-bit. `in_dim` must be a multiple of 256.
    fn dequant_q4k(
        &self,
        blocks: *const u8,
        out: *mut f32,
        rows: usize,
        in_dim: usize,
    ) -> NluResult<()> {
        let lib = self
            .kernels
            .as_ref()
            .ok_or_else(|| NluError::ModelLoadFailed {
                reason: "HIP Q4_K kernel not built (no rocm toolchain at build time)".into(),
            })?;
        let f = Self::sym::<AkhDequantQ4K>(lib, b"akh_dequant_q4k\0")?;
        // SAFETY: `blocks` holds rows*(in_dim/256) Q4_K blocks; `out` holds rows*in_dim floats.
        let rc = unsafe { f(blocks, out, rows as c_int, in_dim as c_int) };
        if rc != 0 {
            return Err(NluError::ModelLoadFailed {
                reason: format!("akh_dequant_q4k kernel failed (hipError={rc})"),
            });
        }
        Ok(())
    }

    fn sym<'a, T>(lib: &'a Library, name: &[u8]) -> NluResult<Symbol<'a, T>> {
        // SAFETY: name is a NUL-terminated C symbol that exists in the loaded ROCm library;
        // the caller binds it to the matching C ABI signature.
        unsafe { lib.get(name) }.map_err(|e| NluError::ModelLoadFailed {
            reason: format!("ROCm symbol {} missing: {e}", String::from_utf8_lossy(name)),
        })
    }

    /// Row-major C[m,n] = A[m,k] · B[k,n], computed on the GPU via rocBLAS sgemm.
    ///
    /// rocBLAS is column-major, so we compute Cᵀ = Bᵀ·Aᵀ by passing B then A — the standard
    /// operand-swap that yields the row-major product without any host-side transpose.
    pub fn sgemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> NluResult<Vec<f32>> {
        assert_eq!(a.len(), m * k, "A must be m*k");
        assert_eq!(b.len(), k * n, "B must be k*n");

        let hip_malloc = Self::sym::<HipMalloc>(&self.hip, b"hipMalloc\0")?;
        let hip_free = Self::sym::<HipFree>(&self.hip, b"hipFree\0")?;
        let hip_memcpy = Self::sym::<HipMemcpy>(&self.hip, b"hipMemcpy\0")?;
        let hip_sync = Self::sym::<HipDeviceSynchronize>(&self.hip, b"hipDeviceSynchronize\0")?;
        let rb_sgemm = Self::sym::<RocblasSgemm>(&self.rocblas, b"rocblas_sgemm\0")?;

        let bytes_a = std::mem::size_of_val(a);
        let bytes_b = std::mem::size_of_val(b);
        let bytes_c = m * n * std::mem::size_of::<f32>();
        let mut out = vec![0f32; m * n];

        // SAFETY: each pointer is sized to its allocation; all return codes are checked and we
        // free every device allocation before returning on every path.
        unsafe {
            let mut da: *mut c_void = std::ptr::null_mut();
            let mut db: *mut c_void = std::ptr::null_mut();
            let mut dc: *mut c_void = std::ptr::null_mut();
            let check = |rc: c_int, what: &str| -> NluResult<()> {
                if rc != 0 {
                    return Err(NluError::ModelLoadFailed {
                        reason: format!("HIP/rocBLAS call {what} failed (rc={rc})"),
                    });
                }
                Ok(())
            };
            check(hip_malloc(&mut da, bytes_a), "hipMalloc A")?;
            check(hip_malloc(&mut db, bytes_b), "hipMalloc B")?;
            check(hip_malloc(&mut dc, bytes_c), "hipMalloc C")?;

            let free_all = |da: *mut c_void, db: *mut c_void, dc: *mut c_void| {
                hip_free(da);
                hip_free(db);
                hip_free(dc);
            };

            if let Err(e) = check(
                hip_memcpy(
                    da,
                    a.as_ptr() as *const c_void,
                    bytes_a,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy H2D A",
            )
            .and_then(|_| {
                check(
                    hip_memcpy(
                        db,
                        b.as_ptr() as *const c_void,
                        bytes_b,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "memcpy H2D B",
                )
            }) {
                free_all(da, db, dc);
                return Err(e);
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;
            // Cᵀ(n×m) = Bᵀ(n×k) · Aᵀ(k×m): pass (B, A) with leading dims (n, k), output ld = n.
            let status = rb_sgemm(
                self.handle,
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                db as *const f32,
                n as c_int,
                da as *const f32,
                k as c_int,
                &beta,
                dc as *mut f32,
                n as c_int,
            );
            let sync = hip_sync();
            let copy = hip_memcpy(
                out.as_mut_ptr() as *mut c_void,
                dc,
                bytes_c,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            free_all(da, db, dc);

            if status != ROCBLAS_STATUS_SUCCESS {
                return Err(NluError::ModelLoadFailed {
                    reason: format!("rocblas_sgemm failed (status={status})"),
                });
            }
            check(sync, "hipDeviceSynchronize")?;
            check(copy, "memcpy D2H C")?;
        }
        Ok(out)
    }
}

// =============================================================================================
// Resident weights: upload a weight to the GPU once, then per call only ship the (small)
// activation. This is what makes routing a model's linear layers through the GPU viable — the
// per-call `RocmBlas::sgemm` above re-uploads *both* operands every time, which is fatal when
// the weight is large and reused across every token. Still f32 (no quantized kernels yet) and
// still downloads the result each call, but the big buffer stays on-device.
// =============================================================================================

use std::sync::Arc;

/// An owned GPU allocation, freed on drop. Tracks its size in bytes so it can hold either f32
/// activations/weights or packed int4 quantized data.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
    blas: Arc<RocmBlas>,
}

impl DeviceBuffer {
    fn len_f32(&self) -> usize {
        self.bytes / std::mem::size_of::<f32>()
    }
}

// The raw pointer is a HIP device handle; access is mediated through `&self` rocBLAS calls and
// the buffer is only ever used on the thread that owns it during synchronous generation.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(hip_free) = RocmBlas::sym::<HipFree>(&self.blas.hip, b"hipFree\0") {
            // SAFETY: ptr came from hipMalloc in alloc_upload and is freed exactly once here.
            unsafe {
                hip_free(self.ptr);
            }
        }
    }
}

impl RocmBlas {
    /// Allocate device memory and upload `data` (host → device). The returned [`DeviceBuffer`]
    /// stays resident on the GPU until dropped.
    pub fn alloc_upload(self: &Arc<Self>, data: &[f32]) -> NluResult<DeviceBuffer> {
        let hip_malloc = Self::sym::<HipMalloc>(&self.hip, b"hipMalloc\0")?;
        let hip_memcpy = Self::sym::<HipMemcpy>(&self.hip, b"hipMemcpy\0")?;
        let bytes = std::mem::size_of_val(data);
        // SAFETY: ptr is sized to `bytes`; the copy length matches; rc checked.
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            if hip_malloc(&mut ptr, bytes) != 0 {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMalloc (resident weight) failed".into(),
                });
            }
            if hip_memcpy(
                ptr,
                data.as_ptr() as *const c_void,
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                let hip_free = Self::sym::<HipFree>(&self.hip, b"hipFree\0")?;
                hip_free(ptr);
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMemcpy H2D (resident weight) failed".into(),
                });
            }
            PCIE_H2D.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(DeviceBuffer {
                ptr,
                bytes,
                blas: Arc::clone(self),
            })
        }
    }

    /// Allocate device memory and upload raw `bytes` (host → device). Used for packed int4
    /// quantized weights.
    pub fn alloc_upload_u8(self: &Arc<Self>, data: &[u8]) -> NluResult<DeviceBuffer> {
        let hip_malloc = Self::sym::<HipMalloc>(&self.hip, b"hipMalloc\0")?;
        let hip_memcpy = Self::sym::<HipMemcpy>(&self.hip, b"hipMemcpy\0")?;
        let bytes = data.len();
        // SAFETY: ptr sized to `bytes`; copy length matches; rc checked.
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            if hip_malloc(&mut ptr, bytes) != 0 {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMalloc (u8) failed".into(),
                });
            }
            if hip_memcpy(
                ptr,
                data.as_ptr() as *const c_void,
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                let hip_free = Self::sym::<HipFree>(&self.hip, b"hipFree\0")?;
                hip_free(ptr);
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMemcpy H2D (u8) failed".into(),
                });
            }
            PCIE_H2D.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(DeviceBuffer {
                ptr,
                bytes,
                blas: Arc::clone(self),
            })
        }
    }

    /// Allocate `n_f32` floats of device scratch (uninitialized).
    pub fn alloc_f32(self: &Arc<Self>, n_f32: usize) -> NluResult<DeviceBuffer> {
        let hip_malloc = Self::sym::<HipMalloc>(&self.hip, b"hipMalloc\0")?;
        let bytes = n_f32 * std::mem::size_of::<f32>();
        // SAFETY: ptr sized to `bytes`; rc checked; left uninitialized (written before read).
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            if hip_malloc(&mut ptr, bytes) != 0 {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMalloc (scratch) failed".into(),
                });
            }
            Ok(DeviceBuffer {
                ptr,
                bytes,
                blas: Arc::clone(self),
            })
        }
    }

    /// Row-major C[m,n] = A[m,k] · B[k,n] where **B is already resident** on the GPU (`b_dev`).
    /// Only A (the activation) is uploaded and C is downloaded — B never re-crosses PCIe.
    pub fn sgemm_resident(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
        b_dev: &DeviceBuffer,
    ) -> NluResult<Vec<f32>> {
        assert_eq!(b_dev.len_f32(), k * n, "resident B must be k*n");
        self.sgemm_resident_ptr(a, m, k, n, b_dev.ptr)
    }

    /// Like [`sgemm_resident`](Self::sgemm_resident) but B is a raw device pointer (`[k,n]`
    /// row-major) — used by the quantized path where B is a reused dequant scratch buffer.
    pub fn sgemm_resident_ptr(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
        b_ptr: *mut c_void,
    ) -> NluResult<Vec<f32>> {
        assert_eq!(a.len(), m * k, "A must be m*k");
        let hip_malloc = Self::sym::<HipMalloc>(&self.hip, b"hipMalloc\0")?;
        let hip_free = Self::sym::<HipFree>(&self.hip, b"hipFree\0")?;
        let hip_memcpy = Self::sym::<HipMemcpy>(&self.hip, b"hipMemcpy\0")?;
        let hip_sync = Self::sym::<HipDeviceSynchronize>(&self.hip, b"hipDeviceSynchronize\0")?;
        let rb_sgemm = Self::sym::<RocblasSgemm>(&self.rocblas, b"rocblas_sgemm\0")?;

        let bytes_a = std::mem::size_of_val(a);
        let bytes_c = m * n * std::mem::size_of::<f32>();
        let mut out = vec![0f32; m * n];
        // SAFETY: device allocations sized to operands; all rcs checked; da/dc freed on exit.
        unsafe {
            let mut da: *mut c_void = std::ptr::null_mut();
            let mut dc: *mut c_void = std::ptr::null_mut();
            if hip_malloc(&mut da, bytes_a) != 0 || hip_malloc(&mut dc, bytes_c) != 0 {
                hip_free(da);
                hip_free(dc);
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMalloc (activation) failed".into(),
                });
            }
            if hip_memcpy(
                da,
                a.as_ptr() as *const c_void,
                bytes_a,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ) != 0
            {
                hip_free(da);
                hip_free(dc);
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMemcpy H2D (activation) failed".into(),
                });
            }
            PCIE_H2D.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = rb_sgemm(
                self.handle,
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                b_ptr as *const f32,
                n as c_int,
                da as *const f32,
                k as c_int,
                &beta,
                dc as *mut f32,
                n as c_int,
            );
            hip_sync();
            let copy = hip_memcpy(
                out.as_mut_ptr() as *mut c_void,
                dc,
                bytes_c,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            hip_free(da);
            hip_free(dc);
            if status != ROCBLAS_STATUS_SUCCESS {
                return Err(NluError::ModelLoadFailed {
                    reason: format!("rocblas_sgemm (resident) failed (status={status})"),
                });
            }
            if copy != 0 {
                return Err(NluError::ModelLoadFailed {
                    reason: "hipMemcpy D2H (resident result) failed".into(),
                });
            }
        }
        PCIE_D2H.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }
}

/// A linear layer weight resident on the GPU: `y[.., out] = x[.., in] · Wᵀ`.
///
/// Built once from a Candle weight tensor `W[out,in]` (e.g. a dequantized GGUF weight); the
/// transposed f32 copy `Wᵀ[in,out]` lives on the GPU. `forward` ships only the activation.
#[derive(Debug, Clone)]
pub struct ResidentLinear {
    blas: Arc<RocmBlas>,
    wt: Arc<DeviceBuffer>, // Wᵀ, row-major [in, out]
    in_dim: usize,
    out_dim: usize,
}

impl ResidentLinear {
    /// Upload `weight` (Candle tensor shaped `[out, in]`) to the GPU as `Wᵀ[in,out]`.
    pub fn from_weight(blas: &Arc<RocmBlas>, weight: &Tensor) -> NluResult<Self> {
        let (out_dim, in_dim) = weight.dims2().map_err(|e| NluError::ModelLoadFailed {
            reason: format!("weight not 2-D: {e}"),
        })?;
        let wt = weight
            .t()
            .and_then(|t| t.contiguous())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("weight transpose: {e}"),
            })?;
        let wt = Arc::new(blas.alloc_upload(&wt)?);
        Ok(Self {
            blas: Arc::clone(blas),
            wt,
            in_dim,
            out_dim,
        })
    }

    /// `y = x · Wᵀ`, with the GEMM on the GPU. `x` is `[.., in]` (any leading dims); the result
    /// keeps those leading dims with the last replaced by `out`.
    pub fn forward(&self, x: &Tensor) -> NluResult<Tensor> {
        let dims = x.dims().to_vec();
        let last = *dims.last().expect("x has rank >= 1");
        if last != self.in_dim {
            return Err(NluError::ModelLoadFailed {
                reason: format!("ResidentLinear: x last dim {last} != in {}", self.in_dim),
            });
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x
            .reshape((m, self.in_dim))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("activation: {e}"),
            })?;
        let y = self
            .blas
            .sgemm_resident(&x2, m, self.in_dim, self.out_dim, &self.wt)?;
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_dim;
        Tensor::from_vec(y, out_dims, x.device()).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("rebuild tensor: {e}"),
        })
    }

    /// `y = x · Wᵀ` writing the result into a **device** buffer (`[m, out]`), reading `x` from a
    /// **device** pointer (`[m, in]`). Nothing crosses PCIe — used by the fused MLP to chain
    /// gate → silu·up → down on-GPU. The caller owns `m` (product of `x`'s leading dims).
    fn forward_dev(&self, x_ptr: *const f32, m: usize) -> NluResult<DeviceBuffer> {
        let c = self.blas.alloc_f32(m * self.out_dim)?;
        self.blas.sgemm_into(
            x_ptr,
            m,
            self.in_dim,
            self.out_dim,
            self.wt.ptr as *const f32,
            c.ptr as *mut f32,
            self.out_dim,
        )?;
        Ok(c)
    }
}

/// A GPU-resident linear layer, either full-precision (f32) or int4-quantized. The quantized
/// variant is required for 12B/27B (f32 resident is ~4× the size); f32 is fine for small
/// models and is exact. Both compute `y = x · Wᵀ` on the GPU.
#[derive(Debug, Clone)]
pub enum GpuLinear {
    F32(ResidentLinear),
    Q4(ResidentQuantLinear),
    /// Native GGUF Q4_K — uses the shipped weights as-is (no dequant→requant loss).
    Q4K(ResidentQ4KLinear),
}

impl GpuLinear {
    /// Build from a Candle weight `[out, in]`. Uses int4 when `quant` is set and the shape is
    /// quantizable (falls back to f32 otherwise — e.g. a dim not divisible by the group).
    pub fn from_weight(blas: &Arc<RocmBlas>, weight: &Tensor, quant: bool) -> NluResult<Self> {
        if quant && ResidentQuantLinear::supports(weight) && blas.has_dequant_kernel() {
            Ok(Self::Q4(ResidentQuantLinear::from_weight(blas, weight)?))
        } else {
            Ok(Self::F32(ResidentLinear::from_weight(blas, weight)?))
        }
    }

    /// Build from a GGUF `QTensor`, preferring the **native Q4_K** path (weights used as shipped)
    /// when `quant` is set and the tensor is Q4_K-shaped. Otherwise dequantizes to f32 and uses
    /// the akh-medu int4 (`quant`) or f32 path via [`from_weight`](Self::from_weight) — so a
    /// Q6_K / Q5_K / other tensor still loads correctly, just without the native fast path.
    pub fn from_qtensor(
        blas: &Arc<RocmBlas>,
        qt: &QTensor,
        device: &candle_core::Device,
        quant: bool,
    ) -> NluResult<Self> {
        if quant && blas.has_dequant_kernel() && ResidentQ4KLinear::supports(qt) {
            return Ok(Self::Q4K(ResidentQ4KLinear::from_qtensor(blas, qt)?));
        }
        let w = qt
            .dequantize(device)
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("dequantize: {e}"),
            })?;
        Self::from_weight(blas, &w, quant)
    }

    pub fn forward(&self, x: &Tensor) -> NluResult<Tensor> {
        match self {
            Self::F32(l) => l.forward(x),
            Self::Q4(l) => l.forward(x),
            Self::Q4K(l) => l.forward(x),
        }
    }

    /// `y = x · Wᵀ` into a device buffer, reading `x` from a device pointer (`[m, in]`). The
    /// fused-MLP building block; keeps the activation on the GPU.
    fn forward_dev(&self, x_ptr: *const f32, m: usize) -> NluResult<DeviceBuffer> {
        match self {
            Self::F32(l) => l.forward_dev(x_ptr, m),
            Self::Q4(l) => l.forward_dev(x_ptr, m),
            Self::Q4K(l) => l.forward_dev(x_ptr, m),
        }
    }

    fn in_dim(&self) -> usize {
        match self {
            Self::F32(l) => l.in_dim,
            Self::Q4(l) => l.in_dim,
            Self::Q4K(l) => l.in_dim,
        }
    }

    fn out_dim(&self) -> usize {
        match self {
            Self::F32(l) => l.out_dim,
            Self::Q4(l) => l.out_dim,
            Self::Q4K(l) => l.out_dim,
        }
    }

    fn blas(&self) -> &Arc<RocmBlas> {
        match self {
            Self::F32(l) => &l.blas,
            Self::Q4(l) => &l.blas,
            Self::Q4K(l) => &l.blas,
        }
    }
}

/// A SwiGLU feed-forward block with all three projections resident on the GPU.
///
/// `down(silu(gate(x)) * up(x))` — the three big matmuls run on the gfx1100 via rocBLAS while
/// `silu`/elementwise stay in Candle on the CPU. This is what lets a real Qwen/Gemma forward
/// pass offload its MLP compute to the GPU (weights uploaded once at load).
#[derive(Debug, Clone)]
pub struct GpuMlp {
    gate: GpuLinear, // ffn_gate / w1
    up: GpuLinear,   // ffn_up   / w3
    down: GpuLinear, // ffn_down / w2
}

impl GpuMlp {
    /// Build from dequantized f32 weight tensors (each `[out, in]`). `quant` selects int4.
    pub fn from_weights(
        blas: &Arc<RocmBlas>,
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        quant: bool,
    ) -> NluResult<Self> {
        Ok(Self {
            gate: GpuLinear::from_weight(blas, gate_w, quant)?,
            up: GpuLinear::from_weight(blas, up_w, quant)?,
            down: GpuLinear::from_weight(blas, down_w, quant)?,
        })
    }

    /// Build from GGUF `QTensor`s, preferring the native Q4_K path per projection (see
    /// [`GpuLinear::from_qtensor`]). `quant` selects the quantized path; otherwise f32.
    pub fn from_qtensors(
        blas: &Arc<RocmBlas>,
        gate: &QTensor,
        up: &QTensor,
        down: &QTensor,
        device: &candle_core::Device,
        quant: bool,
    ) -> NluResult<Self> {
        Ok(Self {
            gate: GpuLinear::from_qtensor(blas, gate, device, quant)?,
            up: GpuLinear::from_qtensor(blas, up, device, quant)?,
            down: GpuLinear::from_qtensor(blas, down, device, quant)?,
        })
    }

    /// `down(silu(gate(x)) * up(x))`, matmuls on the GPU.
    ///
    /// When the fused activation kernel is available this keeps the hidden state **resident on
    /// the GPU through the whole block**: `xs` is uploaded once, gate/up/silu·mul/down all run
    /// on-device, and only the final result is downloaded — one H2D + one D2H instead of the
    /// unfused path's two uploads, three downloads, and a host-side `silu`. Falls back to the
    /// unfused per-linear path when the kernel was not built (no HIP toolchain at build time).
    pub fn forward(&self, xs: &Tensor) -> NluResult<Tensor> {
        if self.gate.blas().has_fused_kernels() {
            match self.forward_fused(xs) {
                Ok(t) => return Ok(t),
                // A device-path failure shouldn't kill generation — fall back to the unfused
                // path (which itself falls back to CPU if the GPU is unavailable).
                Err(e) => {
                    tracing::warn!(error = %e, "fused GPU MLP failed; using unfused path");
                }
            }
        }
        self.forward_unfused(xs)
    }

    /// Fused on-device SwiGLU. See [`forward`](Self::forward).
    fn forward_fused(&self, xs: &Tensor) -> NluResult<Tensor> {
        let dims = xs.dims().to_vec();
        let in_dim = *dims.last().expect("xs rank >= 1");
        if in_dim != self.gate.in_dim() {
            return Err(NluError::ModelLoadFailed {
                reason: format!(
                    "GpuMlp: xs last dim {in_dim} != gate in {}",
                    self.gate.in_dim()
                ),
            });
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let ffn = self.gate.out_dim();
        let hidden = self.down.out_dim();
        let blas = self.gate.blas();

        // Upload the activation once; it then stays on the GPU for the whole block.
        let x = xs
            .reshape((m, in_dim))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("activation: {e}"),
            })?;
        let x_dev = blas.alloc_upload(&x)?;

        let g_dev = self.gate.forward_dev(x_dev.ptr as *const f32, m)?; // [m, ffn]
        let u_dev = self.up.forward_dev(x_dev.ptr as *const f32, m)?; // [m, ffn]
        // act = silu(g) * u, written in place into g_dev (each thread touches one index).
        blas.silu_mul(
            g_dev.ptr as *const f32,
            u_dev.ptr as *const f32,
            g_dev.ptr as *mut f32,
            m * ffn,
        )?;
        let y_dev = self.down.forward_dev(g_dev.ptr as *const f32, m)?; // [m, hidden]
        let y = blas.download_f32(y_dev.ptr, m * hidden)?;

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = hidden;
        Tensor::from_vec(y, out_dims, xs.device()).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("rebuild tensor: {e}"),
        })
    }

    /// Unfused reference path: each projection round-trips host↔GPU and `silu` runs on the host.
    fn forward_unfused(&self, xs: &Tensor) -> NluResult<Tensor> {
        let g = self.gate.forward(xs)?;
        let u = self.up.forward(xs)?;
        let act = candle_nn::ops::silu(&g).and_then(|s| s * u).map_err(|e| {
            NluError::ModelLoadFailed {
                reason: format!("swiglu act: {e}"),
            }
        })?;
        self.down.forward(&act)
    }
}

/// Symmetric 4-bit group quantization of a row-major `[out_dim, in_dim]` weight.
///
/// Returns `(packed, scales)`: `packed` is `out_dim * in_dim/2` bytes (2 nibbles/byte, nibble
/// `n` encodes `q = n-8 ∈ [-7,7]`), `scales` is `out_dim * in_dim/group` f32 (one absmax/7 per
/// group along `in`). Requires `in_dim % group == 0` and `in_dim` even.
fn quantize_q4(weight: &[f32], out_dim: usize, in_dim: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let groups_per_row = in_dim / group;
    let mut packed = vec![0u8; out_dim * in_dim / 2];
    let mut scales = vec![0f32; out_dim * groups_per_row];
    for o in 0..out_dim {
        for g in 0..groups_per_row {
            let base = o * in_dim + g * group;
            let mut absmax = 0f32;
            for j in 0..group {
                absmax = absmax.max(weight[base + j].abs());
            }
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
            scales[o * groups_per_row + g] = scale;
            for j in 0..group {
                let q = (weight[base + j] / scale).round().clamp(-7.0, 7.0) as i32;
                let nib = ((q + 8) as u8) & 0x0F;
                let elem = base + j;
                let byte_idx = elem >> 1;
                if elem & 1 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xF0) | nib;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0F) | (nib << 4);
                }
            }
        }
    }
    (packed, scales)
}

/// A linear layer whose weight is **int4 group-quantized and resident on the GPU** —
/// `y[.., out] = x[.., in] · Wᵀ`. This is the path that makes 12B/27B fit in 24 GB: the
/// weight stays ~4-bit on-device; each matmul dequantizes chunks of output columns into a
/// bounded f32 scratch (HIP kernel) and runs rocBLAS sgemm against the activation (ADR 068).
#[derive(Debug, Clone)]
pub struct ResidentQuantLinear {
    blas: Arc<RocmBlas>,
    qweight: Arc<DeviceBuffer>, // packed int4, [out, in/2] bytes
    scales: Arc<DeviceBuffer>,  // f32, [out, in/group]
    in_dim: usize,
    out_dim: usize,
    group: usize,
}

/// f32 scratch budget for on-the-fly dequant (≈256 MB); caps the per-matmul output chunk.
const DEQUANT_SCRATCH_FLOATS: usize = 64 * 1024 * 1024;

impl ResidentQuantLinear {
    /// Group size for quantization (along `in`). 32 divides every transformer hidden/intermediate/vocab dim in practice.
    pub const GROUP: usize = 32;

    /// True if `weight`'s `[out, in]` shape is quantizable with [`Self::GROUP`].
    pub fn supports(weight: &Tensor) -> bool {
        match weight.dims2() {
            Ok((_, in_dim)) => in_dim % Self::GROUP == 0 && in_dim % 2 == 0,
            Err(_) => false,
        }
    }

    /// Quantize a Candle weight `[out, in]` to int4 and upload it (packed + scales) to the GPU.
    pub fn from_weight(blas: &Arc<RocmBlas>, weight: &Tensor) -> NluResult<Self> {
        if !blas.has_dequant_kernel() {
            return Err(NluError::ModelLoadFailed {
                reason: "quantized GPU path requires the HIP dequant kernel (not built)".into(),
            });
        }
        let (out_dim, in_dim) = weight.dims2().map_err(|e| NluError::ModelLoadFailed {
            reason: format!("weight not 2-D: {e}"),
        })?;
        let group = Self::GROUP;
        if in_dim % group != 0 || in_dim % 2 != 0 {
            return Err(NluError::ModelLoadFailed {
                reason: format!("in_dim {in_dim} not divisible by group {group}/2"),
            });
        }
        let w = weight
            .contiguous()
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("weight read: {e}"),
            })?;
        let (packed, scales) = quantize_q4(&w, out_dim, in_dim, group);
        let qweight = Arc::new(blas.alloc_upload_u8(&packed)?);
        let scales = Arc::new(blas.alloc_upload(&scales)?);
        Ok(Self {
            blas: Arc::clone(blas),
            qweight,
            scales,
            in_dim,
            out_dim,
            group,
        })
    }

    /// `y = x · Wᵀ` with the quantized weight resident on the GPU.
    pub fn forward(&self, x: &Tensor) -> NluResult<Tensor> {
        let dims = x.dims().to_vec();
        let last = *dims.last().expect("x rank >= 1");
        if last != self.in_dim {
            return Err(NluError::ModelLoadFailed {
                reason: format!("ResidentQuantLinear: x last {last} != in {}", self.in_dim),
            });
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x
            .reshape((m, self.in_dim))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("activation: {e}"),
            })?;

        let groups_per_row = self.in_dim / self.group;
        // Chunk output columns so the dequant scratch stays bounded.
        let chunk = (DEQUANT_SCRATCH_FLOATS / self.in_dim).clamp(1, self.out_dim);
        let mut out = vec![0f32; m * self.out_dim];
        let mut scratch = self.blas.alloc_f32(self.in_dim * chunk)?;

        let mut o0 = 0usize;
        while o0 < self.out_dim {
            let co = chunk.min(self.out_dim - o0);
            // Pointers into the resident quantized weight for this chunk of output rows.
            // SAFETY: device-pointer arithmetic only (no deref); offsets within the buffers.
            let (q_ptr, sc_ptr) = unsafe {
                (
                    (self.qweight.ptr as *const u8).add(o0 * (self.in_dim / 2)),
                    (self.scales.ptr as *const f32).add(o0 * groups_per_row),
                )
            };
            self.blas.dequant_q4(
                q_ptr,
                sc_ptr,
                scratch.ptr as *mut f32,
                co,
                self.in_dim,
                self.group,
            )?;
            // scratch now holds Wᵀ_chunk[in, co]; y_chunk[m,co] = x[m,in]·scratch[in,co].
            // sgemm_resident asserts b length == in*co, so size the scratch view via a temp.
            let y_chunk = self
                .blas
                .sgemm_resident_ptr(&x2, m, self.in_dim, co, scratch.ptr)?;
            for r in 0..m {
                out[r * self.out_dim + o0..r * self.out_dim + o0 + co]
                    .copy_from_slice(&y_chunk[r * co..r * co + co]);
            }
            o0 += co;
        }
        // `scratch` is borrowed (via `scratch.ptr`) inside the loop above; it stays alive to the
        // end of this scope, which is all the dequant chunks need.
        drop(scratch);

        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_dim;
        Tensor::from_vec(out, out_dims, x.device()).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("rebuild tensor: {e}"),
        })
    }

    /// `y = x · Wᵀ` into a **device** buffer (`[m, out]`), reading `x` from a device pointer.
    /// Each output-column chunk is dequantized into scratch and GEMM'd straight into the right
    /// columns of the wide output (`ldc = out_dim`, base offset `o0`) — no host round-trip.
    fn forward_dev(&self, x_ptr: *const f32, m: usize) -> NluResult<DeviceBuffer> {
        let c = self.blas.alloc_f32(m * self.out_dim)?;
        let groups_per_row = self.in_dim / self.group;
        let chunk = (DEQUANT_SCRATCH_FLOATS / self.in_dim).clamp(1, self.out_dim);
        let scratch = self.blas.alloc_f32(self.in_dim * chunk)?;

        let mut o0 = 0usize;
        while o0 < self.out_dim {
            let co = chunk.min(self.out_dim - o0);
            // SAFETY: device-pointer arithmetic only; offsets stay within the buffers.
            let (q_ptr, sc_ptr) = unsafe {
                (
                    (self.qweight.ptr as *const u8).add(o0 * (self.in_dim / 2)),
                    (self.scales.ptr as *const f32).add(o0 * groups_per_row),
                )
            };
            self.blas.dequant_q4(
                q_ptr,
                sc_ptr,
                scratch.ptr as *mut f32,
                co,
                self.in_dim,
                self.group,
            )?;
            // Write this chunk's [m, co] block into columns [o0, o0+co) of the [m, out_dim]
            // output: base = c + o0, row stride ldc = out_dim.
            // SAFETY: c holds m*out_dim floats; the block written stays within it.
            let c_chunk = unsafe { (c.ptr as *mut f32).add(o0) };
            self.blas.sgemm_into(
                x_ptr,
                m,
                self.in_dim,
                co,
                scratch.ptr as *const f32,
                c_chunk,
                self.out_dim,
            )?;
            o0 += co;
        }
        drop(scratch);
        Ok(c)
    }
}

/// QK_K super-block size and on-disk byte size of one GGUF Q4_K block (256 weights → 144 bytes).
const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;

/// A linear layer whose weight stays in **native GGUF Q4_K** format, resident on the GPU and
/// dequantized on demand by [`RocmBlas::dequant_q4k`] (matching candle's CPU dequant exactly).
///
/// Unlike [`ResidentQuantLinear`] (akh-medu's own symmetric int4, which dequantizes the GGUF
/// weight to f32 and *re-quantizes* — losing precision), this uses the GGUF bytes **as shipped**:
/// no dequant→requant step, so GPU output tracks CPU output as tightly as the f32 path while
/// keeping the ~4.5-bit Q4_K footprint. Q4_K is asymmetric (per-sub-block scale *and* min) and
/// uses 6-bit sub-scales, so it is also more accurate than the symmetric int4.
#[derive(Debug, Clone)]
pub struct ResidentQ4KLinear {
    blas: Arc<RocmBlas>,
    blocks: Arc<DeviceBuffer>, // raw Q4_K bytes, [out * in/256 * 144]
    in_dim: usize,
    out_dim: usize,
}

impl ResidentQ4KLinear {
    /// True if `qt` is a 2-D Q4_K tensor with `in` divisible by 256 (the kernel's requirement).
    pub fn supports(qt: &QTensor) -> bool {
        qt.dtype() == GgmlDType::Q4K
            && match qt.shape().dims() {
                [_, in_dim] => in_dim % QK_K == 0,
                _ => false,
            }
    }

    /// Upload a Q4_K `QTensor`'s raw blocks (`[out, in]`) to the GPU, unchanged.
    pub fn from_qtensor(blas: &Arc<RocmBlas>, qt: &QTensor) -> NluResult<Self> {
        if !blas.has_dequant_kernel() {
            return Err(NluError::ModelLoadFailed {
                reason: "native Q4_K path requires the HIP kernel (not built)".into(),
            });
        }
        let (out_dim, in_dim) = match qt.shape().dims() {
            [o, i] => (*o, *i),
            other => {
                return Err(NluError::ModelLoadFailed {
                    reason: format!("Q4_K weight not 2-D: {other:?}"),
                });
            }
        };
        if in_dim % QK_K != 0 {
            return Err(NluError::ModelLoadFailed {
                reason: format!("Q4_K in_dim {in_dim} not a multiple of {QK_K}"),
            });
        }
        let data = qt.data().map_err(|e| NluError::ModelLoadFailed {
            reason: format!("Q4_K raw data: {e}"),
        })?;
        let expect = out_dim * (in_dim / QK_K) * Q4K_BLOCK_BYTES;
        if data.len() != expect {
            return Err(NluError::ModelLoadFailed {
                reason: format!("Q4_K byte count {} != expected {expect}", data.len()),
            });
        }
        let blocks = Arc::new(blas.alloc_upload_u8(&data)?);
        Ok(Self {
            blas: Arc::clone(blas),
            blocks,
            in_dim,
            out_dim,
        })
    }

    /// Per-output-chunk dequant budget (reuse the symmetric path's scratch cap).
    fn chunk(&self) -> usize {
        (DEQUANT_SCRATCH_FLOATS / self.in_dim).clamp(1, self.out_dim)
    }

    /// Device pointer to the first Q4_K block of output row `o0`.
    ///
    /// SAFETY: caller ensures `o0 < out_dim`; the offset stays within the resident buffer.
    unsafe fn row_blocks_ptr(&self, o0: usize) -> *const u8 {
        let blocks_per_row = self.in_dim / QK_K;
        unsafe { (self.blocks.ptr as *const u8).add(o0 * blocks_per_row * Q4K_BLOCK_BYTES) }
    }

    /// `y = x · Wᵀ` (host in/out) using the native Q4_K weight on the GPU.
    pub fn forward(&self, x: &Tensor) -> NluResult<Tensor> {
        let dims = x.dims().to_vec();
        let last = *dims.last().expect("x rank >= 1");
        if last != self.in_dim {
            return Err(NluError::ModelLoadFailed {
                reason: format!("ResidentQ4KLinear: x last {last} != in {}", self.in_dim),
            });
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x
            .reshape((m, self.in_dim))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("activation: {e}"),
            })?;

        let chunk = self.chunk();
        let mut out = vec![0f32; m * self.out_dim];
        let scratch = self.blas.alloc_f32(self.in_dim * chunk)?;
        let mut o0 = 0usize;
        while o0 < self.out_dim {
            let co = chunk.min(self.out_dim - o0);
            // SAFETY: o0 < out_dim.
            let q_ptr = unsafe { self.row_blocks_ptr(o0) };
            self.blas
                .dequant_q4k(q_ptr, scratch.ptr as *mut f32, co, self.in_dim)?;
            let y_chunk = self
                .blas
                .sgemm_resident_ptr(&x2, m, self.in_dim, co, scratch.ptr)?;
            for r in 0..m {
                out[r * self.out_dim + o0..r * self.out_dim + o0 + co]
                    .copy_from_slice(&y_chunk[r * co..r * co + co]);
            }
            o0 += co;
        }
        drop(scratch);
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = self.out_dim;
        Tensor::from_vec(out, out_dims, x.device()).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("rebuild tensor: {e}"),
        })
    }

    /// `y = x · Wᵀ` into a device buffer, reading `x` from a device pointer (`[m, in]`). Each
    /// chunk dequantizes into scratch and GEMMs straight into the wide output (`ldc=out_dim`).
    fn forward_dev(&self, x_ptr: *const f32, m: usize) -> NluResult<DeviceBuffer> {
        let c = self.blas.alloc_f32(m * self.out_dim)?;
        let chunk = self.chunk();
        let scratch = self.blas.alloc_f32(self.in_dim * chunk)?;
        let mut o0 = 0usize;
        while o0 < self.out_dim {
            let co = chunk.min(self.out_dim - o0);
            // SAFETY: o0 < out_dim.
            let q_ptr = unsafe { self.row_blocks_ptr(o0) };
            self.blas
                .dequant_q4k(q_ptr, scratch.ptr as *mut f32, co, self.in_dim)?;
            // SAFETY: c holds m*out_dim floats; the [m,co] block at column o0 stays within it.
            let c_chunk = unsafe { (c.ptr as *mut f32).add(o0) };
            self.blas.sgemm_into(
                x_ptr,
                m,
                self.in_dim,
                co,
                scratch.ptr as *const f32,
                c_chunk,
                self.out_dim,
            )?;
            o0 += co;
        }
        drop(scratch);
        Ok(c)
    }
}

/// The four attention projections (q, k, v, output) resident on the GPU. The model applies
/// biases / RoPE / attention math in Candle; only the projection GEMMs run on the gfx1100.
#[derive(Debug, Clone)]
pub struct GpuAttnProj {
    pub wq: GpuLinear,
    pub wk: GpuLinear,
    pub wv: GpuLinear,
    pub wo: GpuLinear,
}

impl GpuAttnProj {
    /// Build from dequantized f32 weights, each `[out, in]`. `quant` selects int4.
    pub fn from_weights(
        blas: &Arc<RocmBlas>,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wo: &Tensor,
        quant: bool,
    ) -> NluResult<Self> {
        Ok(Self {
            wq: GpuLinear::from_weight(blas, wq, quant)?,
            wk: GpuLinear::from_weight(blas, wk, quant)?,
            wv: GpuLinear::from_weight(blas, wv, quant)?,
            wo: GpuLinear::from_weight(blas, wo, quant)?,
        })
    }

    /// Build from GGUF `QTensor`s, preferring the native Q4_K path per projection.
    pub fn from_qtensors(
        blas: &Arc<RocmBlas>,
        wq: &QTensor,
        wk: &QTensor,
        wv: &QTensor,
        wo: &QTensor,
        device: &candle_core::Device,
        quant: bool,
    ) -> NluResult<Self> {
        Ok(Self {
            wq: GpuLinear::from_qtensor(blas, wq, device, quant)?,
            wk: GpuLinear::from_qtensor(blas, wk, device, quant)?,
            wv: GpuLinear::from_qtensor(blas, wv, device, quant)?,
            wo: GpuLinear::from_qtensor(blas, wo, device, quant)?,
        })
    }

    /// Project the hidden state into `(q, k, v)` with the **activation uploaded once** and
    /// shared across all three GEMMs (vs `wq.forward`/`wk.forward`/`wv.forward` each uploading
    /// `x` separately). Each projection runs on the GPU; the results are downloaded because
    /// RoPE / softmax / the KV cache run in Candle between qkv and `wo`. `q`/`k`/`v` may have
    /// different output widths (grouped-query attention).
    ///
    /// Returns `None` when the device path is unavailable (kernel not built) so the caller can
    /// use the per-projection fallback. Errors propagate so the caller can fall back to CPU.
    pub fn project_qkv(&self, x: &Tensor) -> NluResult<(Tensor, Tensor, Tensor)> {
        let dims = x.dims().to_vec();
        let in_dim = *dims.last().expect("x rank >= 1");
        if in_dim != self.wq.in_dim() {
            return Err(NluError::ModelLoadFailed {
                reason: format!(
                    "GpuAttnProj: x last dim {in_dim} != wq in {}",
                    self.wq.in_dim()
                ),
            });
        }
        let m: usize = dims[..dims.len() - 1].iter().product();
        let blas = self.wq.blas();
        let x_host = x
            .reshape((m, in_dim))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("activation: {e}"),
            })?;
        let x_dev = blas.alloc_upload(&x_host)?;

        let one = |proj: &GpuLinear| -> NluResult<Tensor> {
            let out = proj.out_dim();
            let y_dev = proj.forward_dev(x_dev.ptr as *const f32, m)?;
            let y = blas.download_f32(y_dev.ptr, m * out)?;
            let mut out_dims = dims.clone();
            *out_dims.last_mut().unwrap() = out;
            Tensor::from_vec(y, out_dims, x.device()).map_err(|e| NluError::ModelLoadFailed {
                reason: format!("rebuild q/k/v: {e}"),
            })
        };
        Ok((one(&self.wq)?, one(&self.wk)?, one(&self.wv)?))
    }

    /// True if the on-device `project_qkv` path can run (the dequant kernel backs the quant
    /// projections; f32 projections only need rocBLAS, but we gate on the same flag for a
    /// single, predictable decision point).
    pub fn supports_fused(&self) -> bool {
        self.wq.blas().has_fused_kernels()
    }
}

/// A Candle [`CustomOp2`] whose forward pass runs the matmul on the AMD GPU via rocBLAS.
///
/// Operates on contiguous 2-D f32 tensors: `A[m,k] · B[k,n] → C[m,n]`. Use via
/// [`gpu_matmul`]. The default `cuda_fwd`/`metal_fwd` are irrelevant here since storage stays
/// on the CPU device; only `cpu_fwd` is used, and it dispatches to the GPU.
pub struct RocmMatmul {
    blas: RocmBlas,
}

impl RocmMatmul {
    pub fn new() -> NluResult<Self> {
        Ok(Self {
            blas: RocmBlas::load()?,
        })
    }
}

impl CustomOp2 for RocmMatmul {
    fn name(&self) -> &'static str {
        "rocm-sgemm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let (m, k) = l1.shape().dims2()?;
        let (k2, n) = l2.shape().dims2()?;
        if k != k2 {
            candle_core::bail!("rocm-sgemm: inner dims disagree ({k} vs {k2})");
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("rocm-sgemm: inputs must be contiguous");
        }
        let a = &s1.as_slice::<f32>()?[l1.start_offset()..l1.start_offset() + m * k];
        let b = &s2.as_slice::<f32>()?[l2.start_offset()..l2.start_offset() + k * n];
        let c = self
            .blas
            .sgemm(a, b, m, k, n)
            .map_err(|e| candle_core::Error::Msg(format!("rocm-sgemm: {e}")))?;
        Ok((CpuStorage::F32(c), Shape::from((m, n))))
    }
}

/// Multiply two contiguous 2-D f32 tensors on the AMD GPU, returning a Candle tensor.
///
/// Mirrors `a.matmul(b)` but the GEMM executes on the gfx1100 via rocBLAS. Errors (handled by
/// the caller → CPU fallback) when ROCm is unavailable.
pub fn gpu_matmul(a: &Tensor, b: &Tensor) -> NluResult<Tensor> {
    let op = RocmMatmul::new()?;
    a.apply_op2_no_bwd(b, &op)
        .map_err(|e| NluError::ModelLoadFailed {
            reason: format!("gpu_matmul: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlu::gpu::{RocmStatus, detect_rocm};

    /// The GPU matmul must equal Candle's CPU matmul (correctness of the GPU kernel + the
    /// row-major↔column-major operand swap). Skips cleanly when ROCm is not installed.
    #[test]
    fn gpu_matmul_matches_cpu() {
        if !matches!(detect_rocm(), RocmStatus::Ready { .. }) {
            eprintln!("skipping: ROCm not ready on this host");
            return;
        }
        let dev = candle_core::Device::Cpu;
        // Non-square, non-symmetric → catches any transpose/leading-dim bug.
        let a = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6.], // 2x3
            (2, 3),
            &dev,
        )
        .unwrap();
        let b = Tensor::from_vec(
            vec![7f32, 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.], // 3x4
            (3, 4),
            &dev,
        )
        .unwrap();

        let cpu = a
            .matmul(&b)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let gpu = gpu_matmul(&a, &b)
            .expect("gpu matmul")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!((c - g).abs() < 1e-3, "mismatch at {i}: cpu={c} gpu={g}");
        }
    }

    /// A GPU-resident linear must equal Candle's `x · Wᵀ` (matches `candle_nn::Linear` math),
    /// including the leading-dim reshape. This is the path the model's layers use.
    #[test]
    fn resident_linear_matches_cpu_linear() {
        if !matches!(detect_rocm(), RocmStatus::Ready { .. }) {
            eprintln!("skipping: ROCm not ready on this host");
            return;
        }
        let dev = candle_core::Device::Cpu;
        let in_dim = 5;
        let out_dim = 3;
        // W[out,in], x[2, in] → y[2, out] = x · Wᵀ
        let w = Tensor::from_vec(
            (0..(out_dim * in_dim))
                .map(|v| (v as f32) * 0.1 - 0.5)
                .collect::<Vec<_>>(),
            (out_dim, in_dim),
            &dev,
        )
        .unwrap();
        let x = Tensor::from_vec(
            (0..(2 * in_dim))
                .map(|v| (v as f32) * 0.2)
                .collect::<Vec<_>>(),
            (2, in_dim),
            &dev,
        )
        .unwrap();

        let cpu = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let blas = Arc::new(RocmBlas::load().unwrap());
        let lin = ResidentLinear::from_weight(&blas, &w).unwrap();
        let gpu = lin
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!((c - g).abs() < 1e-3, "mismatch at {i}: cpu={c} gpu={g}");
        }
    }

    /// The int4 quantized GPU linear must equal the CPU f32 linear **exactly** when the
    /// weights are chosen so 4-bit dequant is lossless (each group has absmax 3.5 and all
    /// values are multiples of 0.5, |q| ≤ 7). This validates the HIP dequant kernel, the
    /// nibble packing, the transposed scratch layout, and the chunked GEMM together.
    #[test]
    fn quant_linear_matches_cpu_when_lossless() {
        if !matches!(detect_rocm(), RocmStatus::Ready { .. }) {
            eprintln!("skipping: ROCm not ready on this host");
            return;
        }
        let blas = Arc::new(RocmBlas::load().unwrap());
        if !blas.has_dequant_kernel() {
            eprintln!("skipping: HIP dequant kernel not built");
            return;
        }
        let dev = candle_core::Device::Cpu;
        let in_dim = 64; // 2 groups of 32
        let out_dim = 3;
        // Each row: one 3.5 (q=7 sets the group scale to 0.5) + multiples of 0.5 with |q|<=3.
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                wv[o * in_dim + i] = if i % 32 == 0 {
                    3.5
                } else {
                    (((i + o) % 7) as f32 - 3.0) * 0.5
                };
            }
        }
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &dev).unwrap();
        let x = Tensor::from_vec(
            (0..(2 * in_dim))
                .map(|v| ((v % 5) as f32) - 2.0)
                .collect::<Vec<_>>(),
            (2, in_dim),
            &dev,
        )
        .unwrap();

        let cpu = x
            .matmul(&w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let lin = ResidentQuantLinear::from_weight(&blas, &w).unwrap();
        let gpu = lin
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!((c - g).abs() < 1e-3, "mismatch at {i}: cpu={c} gpu={g}");
        }
    }

    /// The fused on-device SwiGLU MLP (`down(silu(gate(x)) * up(x))`, kept resident on the GPU
    /// with one upload + one download) must equal the CPU reference. Exercises the silu_mul
    /// kernel, `forward_dev`, the strided device output, and the gate/up/down chaining together.
    /// `m=2` (>1) also checks the row-stride (`ldc`) handling.
    #[test]
    fn fused_mlp_matches_cpu() {
        if !matches!(detect_rocm(), RocmStatus::Ready { .. }) {
            eprintln!("skipping: ROCm not ready on this host");
            return;
        }
        let blas = Arc::new(RocmBlas::load().unwrap());
        if !blas.has_fused_kernels() {
            eprintln!("skipping: HIP fused kernels not built");
            return;
        }
        let dev = candle_core::Device::Cpu;
        let in_dim = 4;
        let ffn = 6;
        let out_dim = 4;
        let m = 2;
        let mk = |rows: usize, cols: usize, off: f32| {
            Tensor::from_vec(
                (0..(rows * cols))
                    .map(|v| (v as f32) * 0.05 - off)
                    .collect::<Vec<_>>(),
                (rows, cols),
                &dev,
            )
            .unwrap()
        };
        let gate_w = mk(ffn, in_dim, 0.3); // [ffn, in]
        let up_w = mk(ffn, in_dim, 0.2); // [ffn, in]
        let down_w = mk(out_dim, ffn, 0.25); // [out, ffn]
        let x = mk(m, in_dim, 0.1); // [m, in]

        // CPU reference: down(silu(gate(x)) * up(x)).
        let g = x.matmul(&gate_w.t().unwrap()).unwrap();
        let u = x.matmul(&up_w.t().unwrap()).unwrap();
        let act = (candle_nn::ops::silu(&g).unwrap() * u).unwrap();
        let cpu = act
            .matmul(&down_w.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // f32 fused GPU path.
        let mlp = GpuMlp::from_weights(&blas, &gate_w, &up_w, &down_w, false).unwrap();
        let gpu = mlp
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-3,
                "f32 fused mismatch at {i}: cpu={c} gpu={g}"
            );
        }
    }

    /// The native GGUF Q4_K path must match candle's own Q4_K dequant: quantize an f32 weight to
    /// Q4_K with candle, dequantize it (candle CPU), and check `x · dequant(W)ᵀ` equals the GPU
    /// `ResidentQ4KLinear::forward` (which dequantizes the *same raw bytes* on the GPU and GEMMs).
    /// This validates the HIP Q4_K kernel (f16 scales, 6-bit sub-scale unpack, nibble layout,
    /// transposed write) against the reference, end to end through rocBLAS.
    #[test]
    fn q4k_native_matches_candle() {
        if !matches!(detect_rocm(), RocmStatus::Ready { .. }) {
            eprintln!("skipping: ROCm not ready on this host");
            return;
        }
        let blas = Arc::new(RocmBlas::load().unwrap());
        if !blas.has_dequant_kernel() {
            eprintln!("skipping: HIP kernel not built");
            return;
        }
        let dev = candle_core::Device::Cpu;
        let in_dim = 256; // one Q4_K super-block per row
        let out_dim = 5;
        let m = 3;
        // A weight with some structure so quantization is non-trivial.
        let wv: Vec<f32> = (0..(out_dim * in_dim))
            .map(|v| ((v % 17) as f32 - 8.0) * 0.03 + ((v / 17) as f32).sin() * 0.1)
            .collect();
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &dev).unwrap();
        let qt = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
        // candle's CPU dequant of the same blocks → the reference weight.
        let w_dq = qt.dequantize(&dev).unwrap();
        let x = Tensor::from_vec(
            (0..(m * in_dim))
                .map(|v| ((v % 11) as f32 - 5.0) * 0.1)
                .collect::<Vec<_>>(),
            (m, in_dim),
            &dev,
        )
        .unwrap();
        let cpu = x
            .matmul(&w_dq.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(
            ResidentQ4KLinear::supports(&qt),
            "Q4_K tensor should be supported"
        );
        let lin = ResidentQ4KLinear::from_qtensor(&blas, &qt).unwrap();
        let gpu = lin
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            // Same dequant formula on both sides; only matmul accumulation order differs.
            assert!(
                (c - g).abs() < 1e-3,
                "Q4_K native mismatch at {i}: cpu={c} gpu={g}"
            );
        }
    }
}
