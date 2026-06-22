//! AMD **ROCm/HIP** device backend for Candle (Phase 45 / ADR 073).
//!
//! HIP is source-compatible with the CUDA runtime API, so rather than add a parallel
//! `Device::Hip` enum arm across ~130 match sites, this backend **fills the `cuda`
//! module slot** when the `hip` feature is enabled and `cuda` is not (see `lib.rs` /
//! `quantized/mod.rs`). Every `Storage::Cuda` / `Device::Cuda` dispatch in candle-core
//! then routes to HIP unchanged — the "hipify the CUDA backend" approach (ADR 068
//! step 3). Public type names stay `CudaDevice` / `CudaStorage` (re-exported from the
//! honest `HipDevice` / `HipStorage`); `DeviceLocation::Cuda` labels the AMD GPU.
//!
//! **Phase 45c (this revision):** the device is real — `HipDevice::new` dlopen's
//! `libamdhip64`, validates the ordinal and `hipSetDevice`s it; `HipStorage` owns a
//! real device allocation with working host↔device round-trips (`storage_from_*`,
//! `to_cpu_storage`, `try_clone`, `zeros`). The *compute* ops (matmul, elementwise,
//! reduce, …) and the quantized path land in Phase 45d/45e, built on the rocBLAS +
//! dequant kernels already validated in `akh_medu::nlu::hip`.
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use std::ffi::c_void;
use std::os::raw::c_int;

/// Minimal dlopen'd HIP runtime (device mgmt + memory). Kept inside candle-core
/// because candle cannot depend on the akh-medu crate; Phase 45d migrates the
/// rocBLAS / kernel primitives from `akh_medu::nlu::hip` alongside this.
///
/// `pub(crate)` so the quantized backend (`quantized/hip.rs`) can reuse `hip()`
/// (malloc/free/memcpy/synchronize) and `rocblas()` for the Q4_K dequant + GEMM
/// path without re-dlopen'ing the runtime (Phase 45e).
pub(crate) mod runtime {
    use libloading::Library;
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::sync::OnceLock;

    pub const MEMCPY_H2D: c_int = 1;
    pub const MEMCPY_D2H: c_int = 2;
    pub const MEMCPY_D2D: c_int = 3;

    const HIP_LIB_CANDIDATES: &[&str] = &[
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.6",
        "libamdhip64.so",
        "libamdhip64.so.6",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
    ];

    type GetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type SetDevice = unsafe extern "C" fn(c_int) -> c_int;
    type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
    type Free = unsafe extern "C" fn(*mut c_void) -> c_int;
    type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
    type Memset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
    type Synchronize = unsafe extern "C" fn() -> c_int;

    /// Loaded HIP runtime entry points. The symbols are plain C fn pointers into a
    /// library kept alive for the process lifetime (the `OnceLock` below); calls are
    /// externally synchronized to the synchronous compute thread.
    pub struct Hip {
        _lib: Library,
        get_device_count: GetDeviceCount,
        set_device: SetDevice,
        malloc: Malloc,
        free: Free,
        memcpy: Memcpy,
        memset: Memset,
        synchronize: Synchronize,
    }
    unsafe impl Send for Hip {}
    unsafe impl Sync for Hip {}

    impl Hip {
        fn load() -> std::result::Result<Self, String> {
            let lib = HIP_LIB_CANDIDATES
                .iter()
                .find_map(|p| unsafe { Library::new(p).ok() })
                .ok_or_else(|| {
                    format!(
                        "could not dlopen libamdhip64 (tried {HIP_LIB_CANDIDATES:?}); \
                         is the ROCm userspace installed?"
                    )
                })?;
            unsafe {
                macro_rules! sym {
                    ($n:literal, $t:ty) => {
                        *lib.get::<$t>($n).map_err(|e| {
                            format!("missing HIP symbol {}: {e}", stringify!($n))
                        })?
                    };
                }
                // Bind each symbol to a local so the borrowing `Symbol` temporary is
                // dropped (`;`) before `lib` is moved into the struct below.
                let get_device_count: GetDeviceCount = sym!(b"hipGetDeviceCount\0", GetDeviceCount);
                let set_device: SetDevice = sym!(b"hipSetDevice\0", SetDevice);
                let malloc: Malloc = sym!(b"hipMalloc\0", Malloc);
                let free: Free = sym!(b"hipFree\0", Free);
                let memcpy: Memcpy = sym!(b"hipMemcpy\0", Memcpy);
                let memset: Memset = sym!(b"hipMemset\0", Memset);
                let synchronize: Synchronize = sym!(b"hipDeviceSynchronize\0", Synchronize);
                Ok(Hip {
                    _lib: lib,
                    get_device_count,
                    set_device,
                    malloc,
                    free,
                    memcpy,
                    memset,
                    synchronize,
                })
            }
        }

        pub fn device_count(&self) -> std::result::Result<i32, String> {
            let mut n: c_int = 0;
            let rc = unsafe { (self.get_device_count)(&mut n) };
            if rc != 0 {
                return Err(format!("hipGetDeviceCount failed (rc={rc})"));
            }
            Ok(n)
        }

        pub fn set_device(&self, ordinal: usize) -> std::result::Result<(), String> {
            let rc = unsafe { (self.set_device)(ordinal as c_int) };
            if rc != 0 {
                return Err(format!("hipSetDevice({ordinal}) failed (rc={rc})"));
            }
            Ok(())
        }

        pub fn malloc(&self, bytes: usize) -> std::result::Result<*mut c_void, String> {
            if bytes == 0 {
                return Ok(std::ptr::null_mut());
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { (self.malloc)(&mut ptr, bytes) };
            if rc != 0 || ptr.is_null() {
                return Err(format!("hipMalloc({bytes}) failed (rc={rc})"));
            }
            Ok(ptr)
        }

        pub fn free(&self, ptr: *mut c_void) {
            if !ptr.is_null() {
                unsafe {
                    (self.free)(ptr);
                }
            }
        }

        pub fn memcpy(
            &self,
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: c_int,
        ) -> std::result::Result<(), String> {
            if bytes == 0 {
                return Ok(());
            }
            let rc = unsafe { (self.memcpy)(dst, src, bytes, kind) };
            if rc != 0 {
                return Err(format!("hipMemcpy(kind={kind}, {bytes}B) failed (rc={rc})"));
            }
            Ok(())
        }

        pub fn memset(
            &self,
            ptr: *mut c_void,
            value: i32,
            bytes: usize,
        ) -> std::result::Result<(), String> {
            if bytes == 0 {
                return Ok(());
            }
            let rc = unsafe { (self.memset)(ptr, value as c_int, bytes) };
            if rc != 0 {
                return Err(format!("hipMemset({bytes}B) failed (rc={rc})"));
            }
            Ok(())
        }

        pub fn synchronize(&self) -> std::result::Result<(), String> {
            let rc = unsafe { (self.synchronize)() };
            if rc != 0 {
                return Err(format!("hipDeviceSynchronize failed (rc={rc})"));
            }
            Ok(())
        }
    }

    static HIP: OnceLock<std::result::Result<Hip, String>> = OnceLock::new();

    /// The process-wide HIP runtime, loaded on first use. `Err` (with an actionable
    /// message) when the ROCm userspace is absent — the caller maps it to a candle error.
    pub fn hip() -> std::result::Result<&'static Hip, String> {
        HIP.get_or_init(Hip::load).as_ref().map_err(|e| e.clone())
    }

    // ---- rocBLAS (matmul; Phase 45d) ----

    /// `rocblas_operation`: no transpose / transpose.
    pub const OP_N: c_int = 111;
    pub const OP_T: c_int = 112;

    const ROCBLAS_LIB_CANDIDATES: &[&str] = &[
        "/opt/rocm/lib/librocblas.so",
        "/opt/rocm/lib/librocblas.so.4",
        "librocblas.so",
        "librocblas.so.4",
    ];

    type CreateHandle = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
    #[allow(clippy::type_complexity)]
    type SgemmStridedBatched = unsafe extern "C" fn(
        *mut c_void, // handle
        c_int,
        c_int, // transa, transb
        c_int,
        c_int,
        c_int, // m, n, k
        *const f32,
        *const f32,
        c_int,
        i64, // alpha, A, lda, stride_a
        *const f32,
        c_int,
        i64, // B, ldb, stride_b
        *const f32,
        *mut f32,
        c_int,
        i64,   // beta, C, ldc, stride_c
        c_int, // batch_count
    ) -> c_int;

    // `rocblas_hgemm_strided_batched` (f16). Identical shape to sgemm but the matrix
    // and alpha/beta pointers are `rocblas_half*` — a 2-byte value (see rocblas-types.h:
    // `typedef struct rocblas_half { uint16_t data; }`). We pass them as `*const u16` /
    // `*mut u16`, matching the bit layout of `half::f16`.
    #[allow(clippy::type_complexity)]
    type HgemmStridedBatched = unsafe extern "C" fn(
        *mut c_void, // handle
        c_int,
        c_int, // transa, transb
        c_int,
        c_int,
        c_int, // m, n, k
        *const u16,
        *const u16,
        c_int,
        i64, // alpha, A, lda, stride_a
        *const u16,
        c_int,
        i64, // B, ldb, stride_b
        *const u16,
        *mut u16,
        c_int,
        i64,   // beta, C, ldc, stride_c
        c_int, // batch_count
    ) -> c_int;

    // `rocblas_gemm_strided_batched_ex` — the type-erased GEMM (we use it for bf16).
    // Matrices are `void*`; alpha/beta are `void*` (f32 here, per `compute_type`);
    // a/b/c/d dtypes + compute_type are `rocblas_datatype` enum ints; the last three
    // args are algo + solution_index + flags. NB: the exported symbol takes exactly
    // these 30 args — the header `#define` only strips obsolete workspace params, which
    // never reach the real entry point we dlopen.
    #[allow(clippy::type_complexity)]
    type GemmStridedBatchedEx = unsafe extern "C" fn(
        *mut c_void, // handle
        c_int,
        c_int, // transa, transb
        c_int,
        c_int,
        c_int,       // m, n, k
        *const c_void, // alpha (f32)
        *const c_void, // A
        c_int,         // a_type
        c_int,
        i64, // lda, stride_a
        *const c_void, // B
        c_int,         // b_type
        c_int,
        i64,           // ldb, stride_b
        *const c_void, // beta (f32)
        *const c_void, // C
        c_int,         // c_type
        c_int,
        i64,         // ldc, stride_c
        *mut c_void, // D
        c_int,       // d_type
        c_int,
        i64,   // ldd, stride_d
        c_int, // batch_count
        c_int, // compute_type
        c_int, // algo
        i32,   // solution_index
        u32,   // flags
    ) -> c_int;

    // rocBLAS `rocblas_datatype` enum values (verified against
    // /opt/rocm/include/rocblas/internal/rocblas-types.h on this host):
    //   rocblas_datatype_f16_r  = 150
    //   rocblas_datatype_f32_r  = 151
    //   rocblas_datatype_bf16_r = 168   (NB: NOT 151 — the 152/16x range is f64/int)
    //   rocblas_gemm_algo_standard = 0
    pub const ROCBLAS_DATATYPE_F32_R: c_int = 151;
    pub const ROCBLAS_DATATYPE_BF16_R: c_int = 168;
    pub const ROCBLAS_GEMM_ALGO_STANDARD: c_int = 0;

    /// rocBLAS handle + the one GEMM entry point we need. The handle is created once
    /// (binds to the device current at first use) and reused, mirroring the persistent
    /// handle in `akh_medu::nlu::hip`.
    pub struct Rocblas {
        _lib: Library,
        handle: *mut c_void,
        sgemm_strided_batched: SgemmStridedBatched,
        hgemm_strided_batched: HgemmStridedBatched,
        gemm_strided_batched_ex: GemmStridedBatchedEx,
    }
    unsafe impl Send for Rocblas {}
    unsafe impl Sync for Rocblas {}

    impl Rocblas {
        fn load() -> std::result::Result<Self, String> {
            let lib = ROCBLAS_LIB_CANDIDATES
                .iter()
                .find_map(|p| unsafe { Library::new(p).ok() })
                .ok_or_else(|| {
                    format!("could not dlopen librocblas (tried {ROCBLAS_LIB_CANDIDATES:?})")
                })?;
            unsafe {
                let create: CreateHandle = *lib
                    .get(b"rocblas_create_handle\0")
                    .map_err(|e| format!("missing rocblas_create_handle: {e}"))?;
                let sgemm: SgemmStridedBatched = *lib
                    .get(b"rocblas_sgemm_strided_batched\0")
                    .map_err(|e| format!("missing rocblas_sgemm_strided_batched: {e}"))?;
                let hgemm: HgemmStridedBatched = *lib
                    .get(b"rocblas_hgemm_strided_batched\0")
                    .map_err(|e| format!("missing rocblas_hgemm_strided_batched: {e}"))?;
                let gemm_ex: GemmStridedBatchedEx = *lib
                    .get(b"rocblas_gemm_strided_batched_ex\0")
                    .map_err(|e| format!("missing rocblas_gemm_strided_batched_ex: {e}"))?;
                let mut handle: *mut c_void = std::ptr::null_mut();
                let rc = create(&mut handle);
                if rc != 0 || handle.is_null() {
                    return Err(format!("rocblas_create_handle failed (rc={rc})"));
                }
                Ok(Rocblas {
                    _lib: lib,
                    handle,
                    sgemm_strided_batched: sgemm,
                    hgemm_strided_batched: hgemm,
                    gemm_strided_batched_ex: gemm_ex,
                })
            }
        }

        /// `C = A·B` (column-major) batched, with `alpha=1, beta=0`. Pointers are
        /// device pointers; the caller maps candle's row-major (b,m,n,k) onto this.
        #[allow(clippy::too_many_arguments)]
        pub fn sgemm_strided_batched(
            &self,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const f32,
            lda: c_int,
            stride_a: i64,
            b: *const f32,
            ldb: c_int,
            stride_b: i64,
            c: *mut f32,
            ldc: c_int,
            stride_c: i64,
            batch: c_int,
        ) -> std::result::Result<(), String> {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let rc = unsafe {
                (self.sgemm_strided_batched)(
                    self.handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    &alpha,
                    a,
                    lda,
                    stride_a,
                    b,
                    ldb,
                    stride_b,
                    &beta,
                    c,
                    ldc,
                    stride_c,
                    batch,
                )
            };
            if rc != 0 {
                return Err(format!("rocblas_sgemm_strided_batched failed (rc={rc})"));
            }
            Ok(())
        }

        /// f16 `C = A·B` (column-major) batched, `alpha=1, beta=0`. Pointers are device
        /// pointers to `half::f16` data (passed as `*const u16` — same 2-byte layout as
        /// `rocblas_half`). Mirrors [`Rocblas::sgemm_strided_batched`].
        #[allow(clippy::too_many_arguments)]
        pub fn hgemm_strided_batched(
            &self,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const u16,
            lda: c_int,
            stride_a: i64,
            b: *const u16,
            ldb: c_int,
            stride_b: i64,
            c: *mut u16,
            ldc: c_int,
            stride_c: i64,
            batch: c_int,
        ) -> std::result::Result<(), String> {
            // f16 bit patterns of 1.0 and 0.0 (half::f16::{ONE,ZERO}.to_bits()).
            let alpha: u16 = 0x3C00; // 1.0
            let beta: u16 = 0x0000; // 0.0
            let rc = unsafe {
                (self.hgemm_strided_batched)(
                    self.handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    &alpha,
                    a,
                    lda,
                    stride_a,
                    b,
                    ldb,
                    stride_b,
                    &beta,
                    c,
                    ldc,
                    stride_c,
                    batch,
                )
            };
            if rc != 0 {
                return Err(format!("rocblas_hgemm_strided_batched failed (rc={rc})"));
            }
            Ok(())
        }

        /// bf16 `C = A·B` (column-major) batched via `rocblas_gemm_strided_batched_ex`
        /// with `a/b/c/d_type = bf16_r`, `compute_type = f32_r`, `algo = standard`.
        /// alpha/beta are f32 (per `compute_type`). Data pointers are device pointers
        /// to `half::bf16` data, passed as `*const c_void`.
        #[allow(clippy::too_many_arguments)]
        pub fn bf16gemm_strided_batched_ex(
            &self,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            a: *const c_void,
            lda: c_int,
            stride_a: i64,
            b: *const c_void,
            ldb: c_int,
            stride_b: i64,
            c: *mut c_void,
            ldc: c_int,
            stride_c: i64,
            batch: c_int,
        ) -> std::result::Result<(), String> {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let rc = unsafe {
                (self.gemm_strided_batched_ex)(
                    self.handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    &alpha as *const f32 as *const c_void,
                    a,
                    ROCBLAS_DATATYPE_BF16_R,
                    lda,
                    stride_a,
                    b,
                    ROCBLAS_DATATYPE_BF16_R,
                    ldb,
                    stride_b,
                    &beta as *const f32 as *const c_void,
                    // C and D alias (in-place): standard candle/cublas pattern.
                    c as *const c_void,
                    ROCBLAS_DATATYPE_BF16_R,
                    ldc,
                    stride_c,
                    c,
                    ROCBLAS_DATATYPE_BF16_R,
                    ldc,
                    stride_c,
                    batch,
                    ROCBLAS_DATATYPE_F32_R,
                    ROCBLAS_GEMM_ALGO_STANDARD,
                    0, // solution_index
                    0, // flags
                )
            };
            if rc != 0 {
                return Err(format!("rocblas_gemm_strided_batched_ex (bf16) failed (rc={rc})"));
            }
            Ok(())
        }
    }

    static ROCBLAS: OnceLock<std::result::Result<Rocblas, String>> = OnceLock::new();

    /// The process-wide rocBLAS handle, created on first matmul (after a device is
    /// selected). Separate from [`hip`] so device + memory (45c) work without rocBLAS.
    pub fn rocblas() -> std::result::Result<&'static Rocblas, String> {
        ROCBLAS.get_or_init(Rocblas::load).as_ref().map_err(|e| e.clone())
    }

    // ---- elementwise kernels (build.rs → libcandle_hip_kernels.so; Phase 45d) ----

    type AffineF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, f32, f32) -> c_int;
    type UnaryF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, c_int) -> c_int;
    type BinaryF32 = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, c_int) -> c_int;
    // Phase 45d (subagent ops): reduce / cast / index_select / where_cond / strided-copy.
    #[allow(clippy::type_complexity)]
    type ReduceF32 = unsafe extern "C" fn(
        *const f32,
        *mut f32,
        usize,
        c_int,
        *const c_int,
        *const i64,
        *const c_int,
        usize,
        c_int,
    ) -> c_int;
    type CastFn = unsafe extern "C" fn(*const c_void, *mut c_void, usize) -> c_int;
    type IndexSelectF32 = unsafe extern "C" fn(
        *const f32,
        *const c_void,
        *mut f32,
        usize,
        usize,
        usize,
        usize,
    ) -> c_int;
    type WhereCondU8F32 =
        unsafe extern "C" fn(*const u8, *const f32, *const f32, *mut f32, usize) -> c_int;
    type WhereCondU32F32 =
        unsafe extern "C" fn(*const u32, *const f32, *const f32, *mut f32, usize) -> c_int;
    type UcopyF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, usize, *const usize) -> c_int;
    // Phase 45f: strided elementwise compare -> u8 (powers `mask.eq(0)` in prefill).
    #[allow(clippy::type_complexity)]
    type CmpF32 = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut u8,
        usize,
        usize,
        *const i64,
        *const i64,
        *const i64,
        c_int,
    ) -> c_int;

    /// The AOT-compiled elementwise kernel library (built by `build.rs`).
    pub struct Kernels {
        _lib: Library,
        affine_f32: AffineF32,
        unary_f32: UnaryF32,
        binary_f32: BinaryF32,
        reduce_f32: ReduceF32,
        cast_f32_f16: CastFn,
        cast_f16_f32: CastFn,
        cast_f32_bf16: CastFn,
        cast_bf16_f32: CastFn,
        cast_f16_bf16: CastFn,
        cast_bf16_f16: CastFn,
        cast_f32_u32: CastFn,
        cast_u32_f32: CastFn,
        cast_f32_i64: CastFn,
        cast_i64_f32: CastFn,
        index_select_u32_f32: IndexSelectF32,
        index_select_i64_f32: IndexSelectF32,
        where_cond_u8_f32: WhereCondU8F32,
        where_cond_u32_f32: WhereCondU32F32,
        ucopy_f32: UcopyF32,
        cmp_f32: CmpF32,
    }
    unsafe impl Send for Kernels {}
    unsafe impl Sync for Kernels {}

    impl Kernels {
        fn load() -> std::result::Result<Self, String> {
            let path = concat!(env!("OUT_DIR"), "/libcandle_hip_kernels.so");
            let lib = unsafe { Library::new(path) }.map_err(|e| {
                format!("could not dlopen {path}: {e} (was the HIP toolchain present at build time?)")
            })?;
            unsafe {
                let affine_f32: AffineF32 = *lib
                    .get(b"akh_affine_f32\0")
                    .map_err(|e| format!("missing akh_affine_f32: {e}"))?;
                let unary_f32: UnaryF32 = *lib
                    .get(b"akh_unary_f32\0")
                    .map_err(|e| format!("missing akh_unary_f32: {e}"))?;
                let binary_f32: BinaryF32 = *lib
                    .get(b"akh_binary_f32\0")
                    .map_err(|e| format!("missing akh_binary_f32: {e}"))?;
                let reduce_f32: ReduceF32 = *lib
                    .get(b"akh_reduce_f32\0")
                    .map_err(|e| format!("missing akh_reduce_f32: {e}"))?;
                macro_rules! cast_sym {
                    ($n:literal) => {
                        *lib.get::<CastFn>($n)
                            .map_err(|e| format!("missing {}: {e}", stringify!($n)))?
                    };
                }
                let cast_f32_f16: CastFn = cast_sym!(b"akh_cast_f32_f16\0");
                let cast_f16_f32: CastFn = cast_sym!(b"akh_cast_f16_f32\0");
                let cast_f32_bf16: CastFn = cast_sym!(b"akh_cast_f32_bf16\0");
                let cast_bf16_f32: CastFn = cast_sym!(b"akh_cast_bf16_f32\0");
                let cast_f16_bf16: CastFn = cast_sym!(b"akh_cast_f16_bf16\0");
                let cast_bf16_f16: CastFn = cast_sym!(b"akh_cast_bf16_f16\0");
                let cast_f32_u32: CastFn = cast_sym!(b"akh_cast_f32_u32\0");
                let cast_u32_f32: CastFn = cast_sym!(b"akh_cast_u32_f32\0");
                let cast_f32_i64: CastFn = cast_sym!(b"akh_cast_f32_i64\0");
                let cast_i64_f32: CastFn = cast_sym!(b"akh_cast_i64_f32\0");
                let index_select_u32_f32: IndexSelectF32 = *lib
                    .get(b"akh_index_select_u32_f32\0")
                    .map_err(|e| format!("missing akh_index_select_u32_f32: {e}"))?;
                let index_select_i64_f32: IndexSelectF32 = *lib
                    .get(b"akh_index_select_i64_f32\0")
                    .map_err(|e| format!("missing akh_index_select_i64_f32: {e}"))?;
                let where_cond_u8_f32: WhereCondU8F32 = *lib
                    .get(b"akh_where_cond_u8_f32\0")
                    .map_err(|e| format!("missing akh_where_cond_u8_f32: {e}"))?;
                let where_cond_u32_f32: WhereCondU32F32 = *lib
                    .get(b"akh_where_cond_u32_f32\0")
                    .map_err(|e| format!("missing akh_where_cond_u32_f32: {e}"))?;
                let ucopy_f32: UcopyF32 = *lib
                    .get(b"akh_ucopy_f32\0")
                    .map_err(|e| format!("missing akh_ucopy_f32: {e}"))?;
                let cmp_f32: CmpF32 = *lib
                    .get(b"akh_cmp_f32\0")
                    .map_err(|e| format!("missing akh_cmp_f32: {e}"))?;
                Ok(Kernels {
                    _lib: lib,
                    affine_f32,
                    unary_f32,
                    binary_f32,
                    reduce_f32,
                    cast_f32_f16,
                    cast_f16_f32,
                    cast_f32_bf16,
                    cast_bf16_f32,
                    cast_f16_bf16,
                    cast_bf16_f16,
                    cast_f32_u32,
                    cast_u32_f32,
                    cast_f32_i64,
                    cast_i64_f32,
                    index_select_u32_f32,
                    index_select_i64_f32,
                    where_cond_u8_f32,
                    where_cond_u32_f32,
                    ucopy_f32,
                    cmp_f32,
                })
            }
        }

        pub fn affine_f32(
            &self,
            x: *const f32,
            y: *mut f32,
            n: usize,
            mul: f32,
            add: f32,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.affine_f32)(x, y, n, mul, add) };
            if rc != 0 {
                return Err(format!("akh_affine_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        pub fn unary_f32(
            &self,
            x: *const f32,
            y: *mut f32,
            n: usize,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.unary_f32)(x, y, n, op) };
            if rc != 0 {
                return Err(format!("akh_unary_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        pub fn binary_f32(
            &self,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            n: usize,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.binary_f32)(a, b, c, n, op) };
            if rc != 0 {
                return Err(format!("akh_binary_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Contiguous reduce (op: 0 sum, 1 min, 2 max). `dims`/`strides`/`reduce_mask`
        /// are DEVICE pointers of length `num_dims`; the caller owns them. `out_numel`
        /// = product of kept dims, `reduce_numel` = product of reduced dims.
        #[allow(clippy::too_many_arguments)]
        pub fn reduce_f32(
            &self,
            x: *const f32,
            y: *mut f32,
            out_numel: usize,
            num_dims: c_int,
            dims: *const c_int,
            strides: *const i64,
            reduce_mask: *const c_int,
            reduce_numel: usize,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.reduce_f32)(
                    x,
                    y,
                    out_numel,
                    num_dims,
                    dims,
                    strides,
                    reduce_mask,
                    reduce_numel,
                    op,
                )
            };
            if rc != 0 {
                return Err(format!("akh_reduce_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Run the dtype cast keyed by `(src, dst)` over `n` elements of device memory.
        /// `dst` must hold `n * dst.size_in_bytes()` bytes. Unsupported pairs error.
        pub fn cast(
            &self,
            src: crate::DType,
            dst: crate::DType,
            src_ptr: *const c_void,
            dst_ptr: *mut c_void,
            n: usize,
        ) -> std::result::Result<(), String> {
            use crate::DType::*;
            let f: CastFn = match (src, dst) {
                (F32, F16) => self.cast_f32_f16,
                (F16, F32) => self.cast_f16_f32,
                (F32, BF16) => self.cast_f32_bf16,
                (BF16, F32) => self.cast_bf16_f32,
                (F16, BF16) => self.cast_f16_bf16,
                (BF16, F16) => self.cast_bf16_f16,
                (F32, U32) => self.cast_f32_u32,
                (U32, F32) => self.cast_u32_f32,
                (F32, I64) => self.cast_f32_i64,
                (I64, F32) => self.cast_i64_f32,
                _ => {
                    return Err(format!(
                        "HIP to_dtype: cast {src:?} -> {dst:?} not yet implemented (Phase 45d). \
                         Supported: F32<->F16, F32<->BF16, F16<->BF16, F32<->U32, F32<->I64."
                    ));
                }
            };
            let rc = unsafe { f(src_ptr, dst_ptr, n) };
            if rc != 0 {
                return Err(format!("akh_cast {src:?}->{dst:?} failed (rc={rc})"));
            }
            Ok(())
        }

        /// `index_select` gather (u32 indices). `left = ∏dims[..dim]`,
        /// `dim_size = dims[dim]`, `n_idx = ids.len()`, `right = ∏dims[dim+1..]`.
        #[allow(clippy::too_many_arguments)]
        pub fn index_select_u32_f32(
            &self,
            src: *const f32,
            ids: *const u32,
            out: *mut f32,
            left: usize,
            dim_size: usize,
            n_idx: usize,
            right: usize,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.index_select_u32_f32)(src, ids as *const c_void, out, left, dim_size, n_idx, right)
            };
            if rc != 0 {
                return Err(format!("akh_index_select_u32_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// `index_select` gather (i64 indices); same layout/semantics as the u32 variant.
        #[allow(clippy::too_many_arguments)]
        pub fn index_select_i64_f32(
            &self,
            src: *const f32,
            ids: *const i64,
            out: *mut f32,
            left: usize,
            dim_size: usize,
            n_idx: usize,
            right: usize,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.index_select_i64_f32)(src, ids as *const c_void, out, left, dim_size, n_idx, right)
            };
            if rc != 0 {
                return Err(format!("akh_index_select_i64_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// `out[i] = cond[i] ? t[i] : f[i]` with a u8 condition.
        pub fn where_cond_u8_f32(
            &self,
            cond: *const u8,
            t: *const f32,
            f: *const f32,
            out: *mut f32,
            n: usize,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.where_cond_u8_f32)(cond, t, f, out, n) };
            if rc != 0 {
                return Err(format!("akh_where_cond_u8_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// `out[i] = cond[i] ? t[i] : f[i]` with a u32 condition.
        pub fn where_cond_u32_f32(
            &self,
            cond: *const u32,
            t: *const f32,
            f: *const f32,
            out: *mut f32,
            n: usize,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.where_cond_u32_f32)(cond, t, f, out, n) };
            if rc != 0 {
                return Err(format!("akh_where_cond_u32_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided gather-copy: `dst[o] = src[strided_index(o)]` for `o` in `[0, n)`.
        /// `info` is a device buffer of `2*rank` `usize` (`[dims…, strides…]`);
        /// `src`/`dst` are device pointers already advanced to their start offsets.
        pub fn ucopy_f32(
            &self,
            src: *const f32,
            dst: *mut f32,
            n: usize,
            rank: usize,
            info: *const usize,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.ucopy_f32)(src, dst, n, rank, info) };
            if rc != 0 {
                return Err(format!("akh_ucopy_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided elementwise compare -> u8. `dims`/`a_str`/`b_str` are DEVICE
        /// arrays of length `rank`; `a`/`b` already advanced to start offsets;
        /// `out` is a contiguous u8 buffer of `n` elements. op: candle `CmpOp`
        /// (0 eq 1 ne 2 le 3 ge 4 lt 5 gt).
        #[allow(clippy::too_many_arguments)]
        pub fn cmp_f32(
            &self,
            a: *const f32,
            b: *const f32,
            out: *mut u8,
            n: usize,
            rank: usize,
            dims: *const i64,
            a_str: *const i64,
            b_str: *const i64,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.cmp_f32)(a, b, out, n, rank, dims, a_str, b_str, op) };
            if rc != 0 {
                return Err(format!("akh_cmp_f32 failed (rc={rc})"));
            }
            Ok(())
        }
    }

    static KERNELS: OnceLock<std::result::Result<Kernels, String>> = OnceLock::new();

    /// The process-wide elementwise kernel library, dlopen'd on first use.
    pub fn kernels() -> std::result::Result<&'static Kernels, String> {
        KERNELS.get_or_init(Kernels::load).as_ref().map_err(|e| e.clone())
    }
}

/// Fetch the HIP runtime, mapping a load failure to a candle error.
fn hip() -> Result<&'static runtime::Hip> {
    runtime::hip().map_err(Error::Msg)
}

/// Port of the batch-stride computation in candle's `cuda_backend::gemm_config`:
/// the element stride between consecutive matrices in a batched GEMM.
fn batch_stride(stride: &[usize], dims: &[usize], default: usize) -> Result<usize> {
    match stride[..stride.len() - 2] {
        [s1, s] if s1 == s * dims[1] => Ok(s),
        [_, s] if dims[0] == 1 => Ok(s),
        [s, _] if dims[1] == 1 => Ok(s),
        [s] => Ok(s),
        [] => Ok(default),
        _ => Err(Error::Msg(format!(
            "HIP matmul: unsupported batched stride {stride:?}"
        ))),
    }
}

/// "Implemented in a later Phase 45 step" error for compute ops the HIP backend
/// routes through but has not yet built a kernel for (Phase 45d/45e).
macro_rules! hip_todo {
    ($op:literal) => {
        Err(Error::Msg(format!(
            "HIP backend: `{}` not yet implemented (Phase 45d/45e). Device memory and \
             host<->device transfers work (45c); compute kernels follow.",
            $op
        )))
    };
}

/// An AMD GPU exposed through the HIP runtime. Re-exported as `CudaDevice` to fill
/// the candle `cuda` module slot.
#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

/// Storage resident on a [`HipDevice`]: an owned device allocation of `len` elements
/// of `dtype`. Freed on drop. Re-exported as `CudaStorage`.
pub struct HipStorage {
    device: HipDevice,
    /// Device pointer (HIP global memory). Null iff `len == 0`.
    ptr: *mut c_void,
    dtype: DType,
    /// Element count (bytes = `len * dtype.size_in_bytes()`).
    len: usize,
}

pub use HipDevice as CudaDevice;
pub use HipStorage as CudaStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl std::fmt::Debug for HipStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipStorage")
            .field("dtype", &self.dtype)
            .field("len", &self.len)
            .field("ordinal", &self.device.ordinal)
            .finish()
    }
}

impl Drop for HipStorage {
    fn drop(&mut self) {
        if let Ok(h) = runtime::hip() {
            h.free(self.ptr);
        }
    }
}

// The device pointer is only dereferenced on the synchronous compute thread; the
// HIP runtime is internally thread-safe for these mgmt/memcpy calls.
unsafe impl Send for HipStorage {}
unsafe impl Sync for HipStorage {}

impl HipStorage {
    fn byte_len(&self) -> usize {
        self.len * self.dtype.size_in_bytes()
    }

    pub fn transfer_to_device(&self, _dst: &HipDevice) -> Result<Self> {
        // Single-GPU for now; a real P2P/host-staged copy lands with multi-device.
        self.try_clone_full()
    }

    /// Wrap an already-allocated device pointer as a `HipStorage` that owns it (freed
    /// on drop). Used by the quantized backend (`quantized/hip.rs`) to return
    /// dequantized f32 buffers as `CudaStorage` (Phase 45e).
    pub(crate) fn from_raw(device: HipDevice, ptr: *mut c_void, dtype: DType, len: usize) -> Self {
        HipStorage {
            device,
            ptr,
            dtype,
            len,
        }
    }

    /// The raw device pointer backing this storage (HIP global memory). Exposed for the
    /// quantized backend's rocBLAS calls (Phase 45e).
    pub(crate) fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Element count of this storage (bytes = `len * dtype.size_in_bytes()`).
    pub(crate) fn elem_count(&self) -> usize {
        self.len
    }

    /// Duplicate the full device allocation (device-to-device copy).
    fn try_clone_full(&self) -> Result<Self> {
        let h = hip()?;
        let bytes = self.byte_len();
        let ptr = h.malloc(bytes).map_err(Error::Msg)?;
        h.memcpy(ptr, self.ptr, bytes, runtime::MEMCPY_D2D)
            .map_err(Error::Msg)?;
        Ok(HipStorage {
            device: self.device.clone(),
            ptr,
            dtype: self.dtype,
            len: self.len,
        })
    }

    /// Common setup for a contiguous-f32 elementwise op producing a new buffer of the
    /// same element count. Returns `(src_ptr_at_offset, out_storage, n)`.
    fn elementwise_setup(&self, layout: &Layout, op: &str) -> Result<(*const f32, HipStorage, usize)> {
        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP {op}: only f32 supported (Phase 45d), got {:?}",
                self.dtype
            )));
        }
        if !layout.is_contiguous() {
            return Err(Error::Msg(format!(
                "HIP {op}: only contiguous layouts supported (Phase 45d)"
            )));
        }
        let h = hip()?;
        let n = layout.shape().elem_count();
        let src = unsafe { (self.ptr as *const f32).add(layout.start_offset()) };
        let out = h
            .malloc(n * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        Ok((
            src,
            HipStorage {
                device: self.device.clone(),
                ptr: out,
                dtype: DType::F32,
                len: n,
            },
            n,
        ))
    }
}

impl HipDevice {
    /// The GPU ordinal this handle targets.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        <Self as crate::backend::BackendDevice>::new(ordinal)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.ordinal)
    }

    /// Upload a typed host slice to a fresh device allocation.
    fn upload_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<HipStorage> {
        let h = hip()?;
        let bytes = std::mem::size_of_val(data);
        let ptr = h.malloc(bytes).map_err(Error::Msg)?;
        h.memcpy(
            ptr,
            data.as_ptr() as *const c_void,
            bytes,
            runtime::MEMCPY_H2D,
        )
        .map_err(Error::Msg)?;
        Ok(HipStorage {
            device: self.clone(),
            ptr,
            dtype: T::DTYPE,
            len: data.len(),
        })
    }
}

impl crate::backend::BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        self.try_clone_full()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        hip_todo!("const_set")
    }

    /// Download the full device allocation to a host `CpuStorage` of the matching dtype.
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let h = hip()?;
        let bytes = self.byte_len();
        macro_rules! download {
            ($ty:ty, $variant:ident) => {{
                let mut v = vec![<$ty>::default(); self.len];
                h.memcpy(
                    v.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    runtime::MEMCPY_D2H,
                )
                .map_err(Error::Msg)?;
                Ok(CpuStorage::$variant(v))
            }};
        }
        match self.dtype {
            DType::U8 => download!(u8, U8),
            DType::U32 => download!(u32, U32),
            DType::I64 => download!(i64, I64),
            DType::BF16 => download!(half::bf16, BF16),
            DType::F16 => download!(half::f16, F16),
            DType::F32 => download!(f32, F32),
            DType::F64 => download!(f64, F64),
            other => Err(Error::Msg(format!(
                "HIP to_cpu_storage: dtype {other:?} not yet supported (Phase 45c)"
            ))),
        }
    }

    /// `y = x * mul + add`, contiguous f32 (Phase 45d).
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let k = runtime::kernels().map_err(Error::Msg)?;
        let (src, out, n) = self.elementwise_setup(layout, "affine")?;
        k.affine_f32(src, out.ptr as *mut f32, n, mul as f32, add as f32)
            .map_err(Error::Msg)?;
        Ok(out)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        hip_todo!("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        hip_todo!("elu")
    }

    /// Contiguous-f32 reduction (`Sum`/`Max`/`Min`) over `reduce_dims` (Phase 45d).
    /// Output shape matches candle: each reduced dim collapses to size 1, in place.
    /// One GPU thread per output element loops over the reduced extent. `Sum` allows
    /// multiple dims; `Max`/`Min` are single-dim (parity with the CPU backend);
    /// `ArgMin`/`ArgMax` (u32 index outputs) are deferred.
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP reduce_op: only f32 supported (Phase 45d), got {:?}",
                self.dtype
            )));
        }
        if !layout.is_contiguous() {
            return Err(Error::Msg(
                "HIP reduce_op: only contiguous layouts supported (Phase 45d)".to_string(),
            ));
        }
        let op_code: c_int = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Min => {
                if reduce_dims.len() != 1 {
                    return Err(Error::Msg(format!(
                        "HIP reduce_op: `min` over multiple dims {reduce_dims:?} not yet \
                         implemented (Phase 45d); reduce one dim at a time"
                    )));
                }
                1
            }
            ReduceOp::Max => {
                if reduce_dims.len() != 1 {
                    return Err(Error::Msg(format!(
                        "HIP reduce_op: `max` over multiple dims {reduce_dims:?} not yet \
                         implemented (Phase 45d); reduce one dim at a time"
                    )));
                }
                2
            }
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                return Err(Error::Msg(
                    "HIP reduce_op: `argmin`/`argmax` (u32 index outputs) not yet implemented \
                     (Phase 45d)"
                        .to_string(),
                ));
            }
        };

        let src_dims = layout.dims();
        let num_dims = src_dims.len();
        let mut strides = vec![0i64; num_dims];
        let mut acc = 1i64;
        for d in (0..num_dims).rev() {
            strides[d] = acc;
            acc *= src_dims[d] as i64;
        }
        let mut reduce_mask = vec![0i32; num_dims];
        let mut out_dims = src_dims.to_vec();
        let mut reduce_numel: usize = 1;
        for &d in reduce_dims {
            reduce_mask[d] = 1;
            reduce_numel *= src_dims[d];
            out_dims[d] = 1;
        }
        let out_shape = Shape::from(out_dims);
        let out_numel = out_shape.elem_count();
        let dims_i32: Vec<c_int> = src_dims.iter().map(|&d| d as c_int).collect();

        let h = hip()?;
        let k = runtime::kernels().map_err(Error::Msg)?;

        let dims_dev = h
            .malloc(num_dims * std::mem::size_of::<c_int>())
            .map_err(Error::Msg)?;
        let strides_dev = h
            .malloc(num_dims * std::mem::size_of::<i64>())
            .map_err(Error::Msg)?;
        let mask_dev = h
            .malloc(num_dims * std::mem::size_of::<c_int>())
            .map_err(Error::Msg)?;
        let free_meta = || {
            h.free(dims_dev);
            h.free(strides_dev);
            h.free(mask_dev);
        };
        let upload = |dst: *mut c_void, bytes: usize, src: *const c_void| -> Result<()> {
            h.memcpy(dst, src, bytes, runtime::MEMCPY_H2D)
                .map_err(Error::Msg)
        };
        if let Err(e) = upload(
            dims_dev,
            num_dims * std::mem::size_of::<c_int>(),
            dims_i32.as_ptr() as *const c_void,
        )
        .and_then(|_| {
            upload(
                strides_dev,
                num_dims * std::mem::size_of::<i64>(),
                strides.as_ptr() as *const c_void,
            )
        })
        .and_then(|_| {
            upload(
                mask_dev,
                num_dims * std::mem::size_of::<c_int>(),
                reduce_mask.as_ptr() as *const c_void,
            )
        }) {
            free_meta();
            return Err(e);
        }

        let src = unsafe { (self.ptr as *const f32).add(layout.start_offset()) };
        let out = match h.malloc(out_numel * std::mem::size_of::<f32>()) {
            Ok(p) => p,
            Err(e) => {
                free_meta();
                return Err(Error::Msg(e));
            }
        };
        let launch = k.reduce_f32(
            src,
            out as *mut f32,
            out_numel,
            num_dims as c_int,
            dims_dev as *const c_int,
            strides_dev as *const i64,
            mask_dev as *const c_int,
            reduce_numel,
            op_code,
        );
        free_meta();
        if let Err(e) = launch {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: out_numel,
        })
    }

    /// Strided elementwise compare -> U8 (Phase 45f). Both operands may be
    /// broadcast/non-contiguous (e.g. prefill's `mask.eq(0)`), so we pass each
    /// side's own strides and reconstruct offsets per output element. f32 only.
    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP cmp: only f32 supported (Phase 45f), got lhs={:?} rhs={:?}",
                self.dtype, rhs.dtype
            )));
        }
        let op_code: c_int = match op {
            CmpOp::Eq => 0,
            CmpOp::Ne => 1,
            CmpOp::Le => 2,
            CmpOp::Ge => 3,
            CmpOp::Lt => 4,
            CmpOp::Gt => 5,
        };
        let dims = lhs_l.dims();
        let rank = dims.len();
        let n = lhs_l.shape().elem_count();
        let h = hip()?;
        if n == 0 {
            let ptr = h.malloc(0).map_err(Error::Msg)?;
            return Ok(HipStorage {
                device: self.device.clone(),
                ptr,
                dtype: DType::U8,
                len: 0,
            });
        }
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let a_str: Vec<i64> = lhs_l.stride().iter().map(|&s| s as i64).collect();
        let b_str: Vec<i64> = rhs_l.stride().iter().map(|&s| s as i64).collect();
        let k = runtime::kernels().map_err(Error::Msg)?;

        let bytes = rank.max(1) * std::mem::size_of::<i64>();
        let dims_dev = h.malloc(bytes).map_err(Error::Msg)?;
        let a_dev = h.malloc(bytes).map_err(Error::Msg)?;
        let b_dev = h.malloc(bytes).map_err(Error::Msg)?;
        let free_meta = || {
            h.free(dims_dev);
            h.free(a_dev);
            h.free(b_dev);
        };
        let upload = |dst: *mut c_void, src: *const c_void| -> Result<()> {
            h.memcpy(dst, src, rank * std::mem::size_of::<i64>(), runtime::MEMCPY_H2D)
                .map_err(Error::Msg)
        };
        if rank > 0 {
            if let Err(e) = upload(dims_dev, dims_i64.as_ptr() as *const c_void)
                .and_then(|_| upload(a_dev, a_str.as_ptr() as *const c_void))
                .and_then(|_| upload(b_dev, b_str.as_ptr() as *const c_void))
            {
                free_meta();
                return Err(e);
            }
        }

        let a = unsafe { (self.ptr as *const f32).add(lhs_l.start_offset()) };
        let b = unsafe { (rhs.ptr as *const f32).add(rhs_l.start_offset()) };
        let out = match h.malloc(n * std::mem::size_of::<u8>()) {
            Ok(p) => p,
            Err(e) => {
                free_meta();
                return Err(Error::Msg(e));
            }
        };
        let launch = k.cmp_f32(
            a,
            b,
            out as *mut u8,
            n,
            rank,
            dims_dev as *const i64,
            a_dev as *const i64,
            b_dev as *const i64,
            op_code,
        );
        free_meta();
        if let Err(e) = launch {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::U8,
            len: n,
        })
    }

    /// Contiguous dtype cast (Phase 45d). A no-op cast is a device clone; otherwise
    /// allocate a target-dtype buffer and run the keyed (src,dst) cast kernel.
    /// Supported pairs: F32<->F16, F32<->BF16, F16<->BF16, F32<->U32, F32<->I64.
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return self.try_clone(layout);
        }
        if !layout.is_contiguous() {
            return Err(Error::Msg(format!(
                "HIP to_dtype ({:?}->{dtype:?}): only contiguous layouts supported (Phase 45d)",
                self.dtype
            )));
        }
        let h = hip()?;
        let k = runtime::kernels().map_err(Error::Msg)?;
        let n = layout.shape().elem_count();
        let src_ptr = unsafe {
            (self.ptr as *const u8).add(layout.start_offset() * self.dtype.size_in_bytes())
                as *const c_void
        };
        let out_ptr = h.malloc(n * dtype.size_in_bytes()).map_err(Error::Msg)?;
        if let Err(e) = k.cast(self.dtype, dtype, src_ptr, out_ptr, n) {
            h.free(out_ptr);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out_ptr,
            dtype,
            len: n,
        })
    }

    /// Contiguous-f32 unary op, dispatched on `B::NAME` (Phase 45d).
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op: c_int = match B::NAME {
            "neg" => 0,
            "exp" => 1,
            "silu" => 2,
            "gelu" => 3,
            "sqr" => 4,
            "sqrt" => 5,
            "recip" => 6,
            "relu" => 7,
            "tanh" => 8,
            "abs" => 9,
            other => {
                return Err(Error::Msg(format!(
                    "HIP unary `{other}` not yet implemented (Phase 45d)"
                )));
            }
        };
        let k = runtime::kernels().map_err(Error::Msg)?;
        let (src, out, n) = self.elementwise_setup(layout, B::NAME)?;
        k.unary_f32(src, out.ptr as *mut f32, n, op).map_err(Error::Msg)?;
        Ok(out)
    }

    /// Contiguous-f32 same-shape binary op, dispatched on `B::NAME` (Phase 45d).
    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP binary `{}`: only f32 supported (Phase 45d)",
                B::NAME
            )));
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            return Err(Error::Msg(format!(
                "HIP binary `{}`: only contiguous layouts supported (Phase 45d)",
                B::NAME
            )));
        }
        let op: c_int = match B::NAME {
            "add" => 0,
            "sub" => 1,
            "mul" => 2,
            "div" => 3,
            other => {
                return Err(Error::Msg(format!(
                    "HIP binary `{other}` not yet implemented (Phase 45d)"
                )));
            }
        };
        let k = runtime::kernels().map_err(Error::Msg)?;
        let h = hip()?;
        let n = l1.shape().elem_count();
        let a = unsafe { (self.ptr as *const f32).add(l1.start_offset()) };
        let b = unsafe { (rhs.ptr as *const f32).add(l2.start_offset()) };
        let out = h
            .malloc(n * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        if let Err(e) = k.binary_f32(a, b, out as *mut f32, n, op) {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: n,
        })
    }

    /// `out[i] = cond[i] != 0 ? t[i] : f[i]`, contiguous, f32 output (Phase 45d).
    /// `self` is the condition (U8 or U32); `t`/`f` are F32. Start-offsets applied.
    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if !matches!(self.dtype, DType::U8 | DType::U32) {
            return Err(Error::Msg(format!(
                "HIP where_cond: condition dtype must be U8 or U32 (Phase 45d), got {:?}",
                self.dtype
            )));
        }
        if t.dtype != DType::F32 || f.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP where_cond: only f32 value tensors supported (Phase 45d); got t={:?} f={:?}",
                t.dtype, f.dtype
            )));
        }
        if !layout.is_contiguous() || !t_l.is_contiguous() || !f_l.is_contiguous() {
            return Err(Error::Msg(
                "HIP where_cond: only contiguous layouts supported (Phase 45d)".to_string(),
            ));
        }
        let k = runtime::kernels().map_err(Error::Msg)?;
        let h = hip()?;
        let n = layout.shape().elem_count();
        let t_ptr = unsafe { (t.ptr as *const f32).add(t_l.start_offset()) };
        let f_ptr = unsafe { (f.ptr as *const f32).add(f_l.start_offset()) };
        let out = h
            .malloc(n * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;
        let launch = match self.dtype {
            DType::U8 => {
                let cond = unsafe { (self.ptr as *const u8).add(layout.start_offset()) };
                k.where_cond_u8_f32(cond, t_ptr, f_ptr, out as *mut f32, n)
            }
            DType::U32 => {
                let cond = unsafe { (self.ptr as *const u32).add(layout.start_offset()) };
                k.where_cond_u32_f32(cond, t_ptr, f_ptr, out as *mut f32, n)
            }
            _ => unreachable!("dtype gated above"),
        };
        if let Err(e) = launch {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: n,
        })
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        hip_todo!("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        hip_todo!("conv_transpose1d")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        hip_todo!("conv2d")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        hip_todo!("conv_transpose2d")
    }

    /// Gather along `dim` over a contiguous f32 source with a 1-D integer (U32/I64)
    /// index tensor (Phase 45d). Output replaces `dims[dim]` with `indices.len()`.
    /// Dominant case: embedding gather (`src [vocab, hidden]`, `dim = 0`).
    fn index_select(
        &self,
        indices: &Self,
        layout: &Layout,
        indices_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP index_select: only f32 source supported (Phase 45d), got {:?}",
                self.dtype
            )));
        }
        if !layout.is_contiguous() {
            return Err(Error::Msg(
                "HIP index_select: only contiguous f32 source supported (Phase 45d)".to_string(),
            ));
        }
        let dims = layout.dims();
        if dim >= dims.len() {
            return Err(Error::Msg(format!(
                "HIP index_select: dim {dim} out of range for source rank {}",
                dims.len()
            )));
        }
        if !indices_l.is_contiguous() {
            return Err(Error::Msg(
                "HIP index_select: only contiguous indices supported (Phase 45d)".to_string(),
            ));
        }
        let n_idx = indices_l.shape().elem_count();
        let left: usize = dims[..dim].iter().product();
        let dim_size = dims[dim];
        let right: usize = dims[dim + 1..].iter().product();
        let out_len = left * n_idx * right;

        let k = runtime::kernels().map_err(Error::Msg)?;
        let h = hip()?;
        let src = unsafe { (self.ptr as *const f32).add(layout.start_offset()) };
        let out = h
            .malloc(out_len * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;

        let res = match indices.dtype {
            DType::U32 => {
                let ids = unsafe { (indices.ptr as *const u32).add(indices_l.start_offset()) };
                k.index_select_u32_f32(src, ids, out as *mut f32, left, dim_size, n_idx, right)
            }
            DType::I64 => {
                let ids = unsafe { (indices.ptr as *const i64).add(indices_l.start_offset()) };
                k.index_select_i64_f32(src, ids, out as *mut f32, left, dim_size, n_idx, right)
            }
            other => {
                h.free(out);
                return Err(Error::Msg(format!(
                    "HIP index_select: indices must be U32 or I64, got {other:?}"
                )));
            }
        };
        if let Err(e) = res {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: out_len,
        })
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        hip_todo!("gather")
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        hip_todo!("scatter_set")
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        hip_todo!("scatter_add_set")
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        hip_todo!("index_add")
    }

    /// Batched f32 matmul via rocBLAS (Phase 45d). Maps candle's row-major
    /// `(b, m, n, k)` + operand layouts onto a column-major strided-batched GEMM
    /// exactly as candle's `cuda_backend::gemm_config` does (rhs = "A", lhs = "B",
    /// `m`/`n` swapped, transposes derived from strides). f16/bf16 follow later.
    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != rhs.dtype {
            return Err(Error::Msg(format!(
                "HIP matmul: dtype mismatch lhs={:?} rhs={:?}",
                self.dtype, rhs.dtype
            )));
        }
        if !matches!(self.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::Msg(format!(
                "HIP matmul: only f32/f16/bf16 supported (Phase 45d/45e); got {:?}",
                self.dtype
            )));
        }
        let rb = runtime::rocblas().map_err(Error::Msg)?;
        let h = hip()?;

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
            (n as c_int, runtime::OP_N)
        } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
            (k as c_int, runtime::OP_T)
        } else {
            return Err(Error::Msg(format!(
                "HIP matmul: rhs layout not row/col-contiguous (strides {rhs_stride:?}, mnk=({m},{n},{k}))"
            )));
        };
        let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
            (k as c_int, runtime::OP_N)
        } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
            (m as c_int, runtime::OP_T)
        } else {
            return Err(Error::Msg(format!(
                "HIP matmul: lhs layout not row/col-contiguous (strides {lhs_stride:?}, mnk=({m},{n},{k}))"
            )));
        };
        let stride_b = batch_stride(lhs_stride, lhs_l.dims(), m * k)?;
        let stride_a = batch_stride(rhs_stride, rhs_l.dims(), n * k)?;

        // Device byte pointers, offset to each operand's start. The gemm-config above
        // (lda/transa/ldb/transb/stride_*) is dtype-agnostic; only the element type and
        // rocBLAS entry point change. C is column-major (ldc = n, stride_c = m*n).
        let esz = self.dtype.size_in_bytes();
        let a_ptr = unsafe { (rhs.ptr as *const u8).add(rhs_l.start_offset() * esz) };
        let b_ptr = unsafe { (self.ptr as *const u8).add(lhs_l.start_offset() * esz) };

        let out_len = b * m * n;
        let out = h.malloc(out_len * esz).map_err(Error::Msg)?;
        let ldc = n as c_int;
        let stride_c = (m * n) as i64;

        let res = match self.dtype {
            DType::F32 => rb.sgemm_strided_batched(
                transa,
                transb,
                n as c_int,
                m as c_int,
                k as c_int,
                a_ptr as *const f32,
                lda,
                stride_a as i64,
                b_ptr as *const f32,
                ldb,
                stride_b as i64,
                out as *mut f32,
                ldc,
                stride_c,
                b as c_int,
            ),
            DType::F16 => rb.hgemm_strided_batched(
                transa,
                transb,
                n as c_int,
                m as c_int,
                k as c_int,
                a_ptr as *const u16,
                lda,
                stride_a as i64,
                b_ptr as *const u16,
                ldb,
                stride_b as i64,
                out as *mut u16,
                ldc,
                stride_c,
                b as c_int,
            ),
            DType::BF16 => rb.bf16gemm_strided_batched_ex(
                transa,
                transb,
                n as c_int,
                m as c_int,
                k as c_int,
                a_ptr as *const c_void,
                lda,
                stride_a as i64,
                b_ptr as *const c_void,
                ldb,
                stride_b as i64,
                out,
                ldc,
                stride_c,
                b as c_int,
            ),
            // Guarded above; unreachable.
            other => {
                h.free(out);
                return Err(Error::Msg(format!("HIP matmul: unsupported dtype {other:?}")));
            }
        };
        if let Err(e) = res {
            h.free(out);
            return Err(Error::Msg(e));
        }
        h.synchronize().map_err(Error::Msg)?;
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: self.dtype,
            len: out_len,
        })
    }

    /// Copy `self` (read through the strided/offset `src_l`) into `dst` at element
    /// `dst_offset`, row-major over `src_l.shape()` — powers `.contiguous()` (Phase 45d).
    /// Contiguous source → one D2D memcpy (any dtype); strided → f32 unravel kernel.
    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if self.dtype != dst.dtype {
            return Err(Error::Msg(format!(
                "HIP copy_strided_src: dtype mismatch (src {:?} != dst {:?})",
                self.dtype, dst.dtype
            )));
        }
        let dims = src_l.dims();
        let n = src_l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        let h = hip()?;
        let elt = self.dtype.size_in_bytes();
        let src = unsafe { (self.ptr as *const u8).add(src_l.start_offset() * elt) };
        let dst_ptr = unsafe { (dst.ptr as *mut u8).add(dst_offset * elt) };

        if src_l.is_contiguous() {
            return h
                .memcpy(
                    dst_ptr as *mut c_void,
                    src as *const c_void,
                    n * elt,
                    runtime::MEMCPY_D2D,
                )
                .map_err(Error::Msg);
        }

        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP copy_strided_src: strided (non-contiguous) copy only supports f32 \
                 (Phase 45d), got {:?}; a contiguous copy of this dtype works",
                self.dtype
            )));
        }
        let rank = dims.len();
        let mut info: Vec<usize> = Vec::with_capacity(2 * rank);
        info.extend_from_slice(dims);
        info.extend_from_slice(src_l.stride());
        let info_bytes = std::mem::size_of_val(info.as_slice());
        let info_ptr = h.malloc(info_bytes).map_err(Error::Msg)?;
        if let Err(e) = h.memcpy(
            info_ptr,
            info.as_ptr() as *const c_void,
            info_bytes,
            runtime::MEMCPY_H2D,
        ) {
            h.free(info_ptr);
            return Err(Error::Msg(e));
        }
        let k = runtime::kernels().map_err(Error::Msg)?;
        let res = k.ucopy_f32(
            src as *const f32,
            dst_ptr as *mut f32,
            n,
            rank,
            info_ptr as *const usize,
        );
        h.free(info_ptr);
        res.map_err(Error::Msg)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        hip_todo!("copy2d")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        hip_todo!("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        hip_todo!("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        hip_todo!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        hip_todo!("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        hip_todo!("upsample_bilinear2d")
    }
}

impl crate::backend::BackendDevice for HipDevice {
    type Storage = HipStorage;

    /// Construct a handle for GPU `ordinal`: load the HIP runtime, validate the
    /// ordinal against the device count, and select it. Errors (with an actionable
    /// message) when ROCm is absent or the ordinal is out of range.
    fn new(ordinal: usize) -> Result<Self> {
        let h = hip()?;
        let count = h.device_count().map_err(Error::Msg)?;
        if ordinal as i32 >= count {
            return Err(Error::Msg(format!(
                "HIP device {ordinal} out of range (only {count} GPU(s) visible)"
            )));
        }
        h.set_device(ordinal).map_err(Error::Msg)?;
        Ok(Self { ordinal })
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        // No on-device RNG yet (rand_* are 45d); accept seeds as a no-op so callers
        // that set a seed before CPU-side sampling do not fail.
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    fn location(&self) -> crate::DeviceLocation {
        // HIP fills the cuda slot; label it with the GPU ordinal.
        crate::DeviceLocation::Cuda {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let h = hip()?;
        let len = shape.elem_count();
        let bytes = len * dtype.size_in_bytes();
        let ptr = h.malloc(bytes).map_err(Error::Msg)?;
        h.memset(ptr, 0, bytes).map_err(Error::Msg)?;
        Ok(HipStorage {
            device: self.clone(),
            ptr,
            dtype,
            len,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let h = hip()?;
        let len = shape.elem_count();
        let ptr = h.malloc(len * dtype.size_in_bytes()).map_err(Error::Msg)?;
        Ok(HipStorage {
            device: self.clone(),
            ptr,
            dtype,
            len,
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        self.upload_slice(s)
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        match s {
            CpuStorage::U8(v) => self.upload_slice(v),
            CpuStorage::U32(v) => self.upload_slice(v),
            CpuStorage::I16(v) => self.upload_slice(v),
            CpuStorage::I32(v) => self.upload_slice(v),
            CpuStorage::I64(v) => self.upload_slice(v),
            CpuStorage::BF16(v) => self.upload_slice(v),
            CpuStorage::F16(v) => self.upload_slice(v),
            CpuStorage::F32(v) => self.upload_slice(v),
            CpuStorage::F64(v) => self.upload_slice(v),
            other => Err(Error::Msg(format!(
                "HIP storage_from_cpu_storage: dtype {:?} not yet supported (Phase 45c)",
                crate::backend::BackendStorage::dtype(other)
            ))),
        }
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&s)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        hip_todo!("rand_uniform")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        hip_todo!("rand_normal")
    }

    fn synchronize(&self) -> Result<()> {
        hip()?.synchronize().map_err(Error::Msg)
    }
}

/// Whether reduced-precision f16 GEMM accumulation is allowed (parity with the cuda slot).
pub fn gemm_reduced_precision_f16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f16(_: bool) {}
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_bf16(_: bool) {}
pub fn gemm_reduced_precision_f32() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f32(_b: bool) {}

/// Phase 45d (subagent ops) — GPU-vs-CPU parity on the gfx1100. Each test builds the
/// same tensor on `Device::new_cuda(0)` (HIP) and `Device::Cpu`, runs the op, and
/// asserts agreement; skips (does not fail) when no ROCm GPU is visible.
#[cfg(all(test, feature = "hip"))]
mod op_tests {
    use crate::{DType, Device, Result, Tensor, D};

    fn gpu() -> Option<Device> {
        Device::new_cuda(0).ok()
    }
    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what} len");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{what}: gpu={x} cpu={y}");
        }
    }

    #[test]
    fn reduce_sum_max_min() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let d: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.75).collect();
        let g = Tensor::from_vec(d.clone(), (3, 4), &dev)?;
        let c = Tensor::from_vec(d, (3, 4), &Device::Cpu)?;
        for (gr, cr, name) in [
            (g.sum_keepdim(D::Minus1)?, c.sum_keepdim(D::Minus1)?, "sum"),
            (g.max_keepdim(D::Minus1)?, c.max_keepdim(D::Minus1)?, "max"),
            (g.min_keepdim(0)?, c.min_keepdim(0)?, "min"),
        ] {
            close(
                &gr.flatten_all()?.to_vec1::<f32>()?,
                &cr.flatten_all()?.to_vec1::<f32>()?,
                1e-3,
                name,
            );
        }
        Ok(())
    }

    #[test]
    fn to_dtype_roundtrips() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let d: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let t = Tensor::from_vec(d.clone(), (64,), &dev)?;
        close(
            &d,
            &t.to_dtype(DType::F16)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            1e-2,
            "f16-roundtrip",
        );
        close(
            &d,
            &t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            1e-1,
            "bf16-roundtrip",
        );
        Ok(())
    }

    #[test]
    fn index_select_embedding() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let idx: Vec<u32> = vec![2, 0, 3];
        let g = Tensor::from_vec(src.clone(), (4, 3), &dev)?
            .index_select(&Tensor::from_vec(idx.clone(), 3, &dev)?, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let c = Tensor::from_vec(src, (4, 3), &Device::Cpu)?
            .index_select(&Tensor::from_vec(idx, 3, &Device::Cpu)?, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(&g, &c, 0.0, "index_select");
        Ok(())
    }

    #[test]
    fn where_cond_mask() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let cond = [1u8, 0, 1, 0, 1, 0];
        let tv: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let fv: [f32; 6] = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        let g = Tensor::new(&cond, &dev)?
            .where_cond(&Tensor::new(&tv, &dev)?, &Tensor::new(&fv, &dev)?)?
            .to_vec1::<f32>()?;
        let c = Tensor::new(&cond, &Device::Cpu)?
            .where_cond(&Tensor::new(&tv, &Device::Cpu)?, &Tensor::new(&fv, &Device::Cpu)?)?
            .to_vec1::<f32>()?;
        close(&g, &c, 0.0, "where_cond");
        Ok(())
    }

    #[test]
    fn cmp_eq_and_broadcast_mask() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // Scalar eq (the `mask.eq(0)` shape): contiguous lhs, broadcast scalar rhs.
        let d: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0];
        let g = Tensor::from_vec(d.clone(), (2, 3), &dev)?
            .eq(0f32)?
            .to_vec2::<u8>()?;
        let c = Tensor::from_vec(d, (2, 3), &Device::Cpu)?
            .eq(0f32)?
            .to_vec2::<u8>()?;
        assert_eq!(g, c, "scalar eq");

        // Broadcast lhs (a causal mask broadcast over batch/head) eq 0 — exercises a
        // non-contiguous (stride-0) lhs layout, exactly the prefill attention path.
        let mask: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0]; // (2,2)
        let gm = Tensor::from_vec(mask.clone(), (2, 2), &dev)?
            .broadcast_as((3, 1, 2, 2))?
            .eq(0f32)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let cm = Tensor::from_vec(mask, (2, 2), &Device::Cpu)?
            .broadcast_as((3, 1, 2, 2))?
            .eq(0f32)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        assert_eq!(gm, cm, "broadcast mask eq");
        Ok(())
    }

    #[test]
    fn copy_strided_contiguous_after_transpose() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let d: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let g = Tensor::from_vec(d.clone(), (3, 4), &dev)?
            .t()?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let c = Tensor::from_vec(d, (3, 4), &Device::Cpu)?
            .t()?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(&g, &c, 0.0, "copy_strided");
        Ok(())
    }

    /// Non-square f16 matmul (2,3)@(3,4) on GPU (rocBLAS hgemm) vs CPU reference.
    /// f16 has ~10 bits mantissa, so ~1e-2 tolerance after the f32->f16 cast + accumulate.
    #[test]
    fn matmul_f16_nonsquare() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let a: Vec<f32> = (0..6).map(|i| (i as f32 - 3.0) * 0.5).collect(); // (2,3)
        let b: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.25).collect(); // (3,4)
        let ga = Tensor::from_vec(a.clone(), (2, 3), &dev)?.to_dtype(DType::F16)?;
        let gb = Tensor::from_vec(b.clone(), (3, 4), &dev)?.to_dtype(DType::F16)?;
        let g = ga
            .matmul(&gb)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let ca = Tensor::from_vec(a, (2, 3), &Device::Cpu)?;
        let cb = Tensor::from_vec(b, (3, 4), &Device::Cpu)?;
        let c = ca.matmul(&cb)?.flatten_all()?.to_vec1::<f32>()?;
        close(&g, &c, 5e-2, "matmul_f16");
        Ok(())
    }

    /// Batched f16 matmul (2,2,3)@(2,3,4) — exercises the strided-batched path.
    #[test]
    fn matmul_f16_batched() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let a: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.2).collect(); // (2,2,3)
        let b: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect(); // (2,3,4)
        let ga = Tensor::from_vec(a.clone(), (2, 2, 3), &dev)?.to_dtype(DType::F16)?;
        let gb = Tensor::from_vec(b.clone(), (2, 3, 4), &dev)?.to_dtype(DType::F16)?;
        let g = ga
            .matmul(&gb)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let ca = Tensor::from_vec(a, (2, 2, 3), &Device::Cpu)?;
        let cb = Tensor::from_vec(b, (2, 3, 4), &Device::Cpu)?;
        let c = ca.matmul(&cb)?.flatten_all()?.to_vec1::<f32>()?;
        close(&g, &c, 5e-2, "matmul_f16_batched");
        Ok(())
    }

    /// Non-square bf16 matmul (2,3)@(3,4) via rocblas_gemm_strided_batched_ex.
    /// bf16 has only 7 mantissa bits, so a looser ~1e-1 tolerance.
    #[test]
    fn matmul_bf16_nonsquare() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let a: Vec<f32> = (0..6).map(|i| (i as f32 - 3.0) * 0.5).collect();
        let b: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.25).collect();
        let ga = Tensor::from_vec(a.clone(), (2, 3), &dev)?.to_dtype(DType::BF16)?;
        let gb = Tensor::from_vec(b.clone(), (3, 4), &dev)?.to_dtype(DType::BF16)?;
        let g = ga
            .matmul(&gb)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let ca = Tensor::from_vec(a, (2, 3), &Device::Cpu)?;
        let cb = Tensor::from_vec(b, (3, 4), &Device::Cpu)?;
        let c = ca.matmul(&cb)?.flatten_all()?.to_vec1::<f32>()?;
        close(&g, &c, 2e-1, "matmul_bf16");
        Ok(())
    }
}
