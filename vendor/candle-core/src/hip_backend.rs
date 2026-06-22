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
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, Tensor};
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
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::sync::{Mutex, OnceLock};

    pub const MEMCPY_H2D: c_int = 1;
    pub const MEMCPY_D2H: c_int = 2;
    pub const MEMCPY_D2D: c_int = 3;

    /// Upper bound on bytes the caching allocator keeps cached across `free`s
    /// (Phase 45g). `hipFree` *synchronizes the whole device*, so freeing a temp
    /// after every op serializes the ~1900 ops/token. The pool recycles freed
    /// blocks by exact byte size instead (no `hipFree`, no sync, no re-`hipMalloc`
    /// of the recurring per-token f32 dequant temps). Over the cap, `free` really
    /// `hipFree`s. **VRAM tension:** the resident 12B already sits at ~23.5 GB on a
    /// 24 GB card, so a large pool spills to GTT — measured (gemma4-12b): 2 GB pool
    /// = 1.25 tok/s @ 25.7 GB (GTT spill) vs 512 MB = 0.96 tok/s @ 24.0 GB. The
    /// default stays conservative; bump `AKH_HIP_POOL_MB` if you have headroom (or,
    /// better, once the fused quantized GEMV removes the f32 temps that force this).
    const DEFAULT_POOL_CAP_BYTES: usize = 512 * 1024 * 1024;

    /// Caching device allocator: a free-list of recycled blocks keyed by exact
    /// byte size + a `ptr -> size` map so size-less `free(ptr)` can still recycle.
    #[derive(Default)]
    struct Pool {
        free: HashMap<usize, Vec<usize>>,
        live: HashMap<usize, usize>,
        cached_bytes: usize,
        cap_bytes: usize,
    }

    // PCIe-crossing counters (Phase 45f): every non-empty H2D/D2H `memcpy` bumps one.
    // The bench resets them after model load, then divides by tokens decoded to prove
    // the resident forward crosses the bus ~twice/token (input id up, logits down) vs
    // the op-offload baseline's hundreds. D2D (on-device) copies are not counted.
    pub static PCIE_H2D: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static PCIE_D2H: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    // hipMalloc instrumentation (Phase 45h): the caching allocator recycles by EXACT
    // size, so growing per-token buffers (e.g. attention scores `[B,1,seq_kv]`) miss and
    // fall through to a real `hipMalloc` — which SYNCHRONIZES the device, stalling the
    // null stream. `MALLOC_CALLS` = total malloc() calls, `MALLOC_MISS` = cache misses
    // (real hipMalloc), `MALLOC_MISS_NANOS` = wall time in the hipMalloc call (incl. its
    // device sync). The bench divides by tokens to expose malloc-stall as a wall fraction.
    pub static MALLOC_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static MALLOC_MISS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    pub static MALLOC_MISS_NANOS: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

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
    // hipMemcpy2D(dst, dpitch, src, spitch, width_bytes, height, kind) — Phase 45f copy2d.
    type Memcpy2D = unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const c_void,
        usize,
        usize,
        usize,
        c_int,
    ) -> c_int;
    type Memset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
    type Synchronize = unsafe extern "C" fn() -> c_int;
    // hipEvent (Phase 45h event profiler): `hipEvent_t`/`hipStream_t` are opaque
    // pointers (void*). Stream markers timestamp the GPU clock when the (ordered null)
    // stream reaches them, so `elapsed(start,stop)` is the *pure GPU* time of the
    // work between them — no device-drain mis-attribution like the sync-bracket mode.
    type EventCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
    type EventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
    type EventSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;
    type EventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> c_int;
    type EventDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;

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
        memcpy2d: Memcpy2D,
        memset: Memset,
        synchronize: Synchronize,
        event_create: EventCreate,
        event_record: EventRecord,
        event_synchronize: EventSynchronize,
        event_elapsed_time: EventElapsedTime,
        event_destroy: EventDestroy,
        pool: Mutex<Pool>,
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
                let memcpy2d: Memcpy2D = sym!(b"hipMemcpy2D\0", Memcpy2D);
                let memset: Memset = sym!(b"hipMemset\0", Memset);
                let synchronize: Synchronize = sym!(b"hipDeviceSynchronize\0", Synchronize);
                let event_create: EventCreate = sym!(b"hipEventCreate\0", EventCreate);
                let event_record: EventRecord = sym!(b"hipEventRecord\0", EventRecord);
                let event_synchronize: EventSynchronize =
                    sym!(b"hipEventSynchronize\0", EventSynchronize);
                let event_elapsed_time: EventElapsedTime =
                    sym!(b"hipEventElapsedTime\0", EventElapsedTime);
                let event_destroy: EventDestroy = sym!(b"hipEventDestroy\0", EventDestroy);
                let cap_bytes = std::env::var("AKH_HIP_POOL_MB")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .map(|mb| mb * 1024 * 1024)
                    .unwrap_or(DEFAULT_POOL_CAP_BYTES);
                Ok(Hip {
                    _lib: lib,
                    get_device_count,
                    set_device,
                    malloc,
                    free,
                    memcpy,
                    memcpy2d,
                    memset,
                    synchronize,
                    event_create,
                    event_record,
                    event_synchronize,
                    event_elapsed_time,
                    event_destroy,
                    pool: Mutex::new(Pool {
                        cap_bytes,
                        ..Pool::default()
                    }),
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
            MALLOC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Reuse a recycled block of this exact size if one is cached.
            {
                let mut pool = self.pool.lock().unwrap();
                if let Some(list) = pool.free.get_mut(&bytes) {
                    if let Some(p) = list.pop() {
                        pool.cached_bytes = pool.cached_bytes.saturating_sub(bytes);
                        pool.live.insert(p, bytes);
                        return Ok(p as *mut c_void);
                    }
                }
            }
            // Cache miss: a real hipMalloc, which synchronizes the device. Time it.
            let t0 = std::time::Instant::now();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { (self.malloc)(&mut ptr, bytes) };
            MALLOC_MISS_NANOS.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            MALLOC_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if rc != 0 || ptr.is_null() {
                return Err(format!("hipMalloc({bytes}) failed (rc={rc})"));
            }
            self.pool.lock().unwrap().live.insert(ptr as usize, bytes);
            Ok(ptr)
        }

        /// Recycle `ptr` into the pool instead of `hipFree`-ing it (which would
        /// synchronize the device). Untracked or over-cap blocks are really freed.
        pub fn free(&self, ptr: *mut c_void) {
            if ptr.is_null() {
                return;
            }
            let key = ptr as usize;
            let mut pool = self.pool.lock().unwrap();
            match pool.live.remove(&key) {
                Some(bytes) if pool.cached_bytes + bytes <= pool.cap_bytes => {
                    pool.cached_bytes += bytes;
                    pool.free.entry(bytes).or_default().push(key);
                }
                _ => {
                    // Untracked, or pooling it would exceed the cap: free for real.
                    drop(pool);
                    unsafe {
                        (self.free)(ptr);
                    }
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
            use std::sync::atomic::Ordering::Relaxed;
            match kind {
                MEMCPY_H2D => {
                    PCIE_H2D.fetch_add(1, Relaxed);
                }
                MEMCPY_D2H => {
                    PCIE_D2H.fetch_add(1, Relaxed);
                }
                _ => {}
            }
            Ok(())
        }

        /// 2D strided copy: `height` rows of `width` bytes, src/dst row pitches in bytes.
        /// Used by candle's `cat` fast path (`copy2d`). Counts as a PCIe crossing only for
        /// H2D/D2H kinds (cat is D2D — on-device — so it does not).
        #[allow(clippy::too_many_arguments)]
        pub fn memcpy2d(
            &self,
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: c_int,
        ) -> std::result::Result<(), String> {
            if width == 0 || height == 0 {
                return Ok(());
            }
            let rc = unsafe { (self.memcpy2d)(dst, dpitch, src, spitch, width, height, kind) };
            if rc != 0 {
                return Err(format!(
                    "hipMemcpy2D(kind={kind}, {width}x{height}B) failed (rc={rc})"
                ));
            }
            use std::sync::atomic::Ordering::Relaxed;
            match kind {
                MEMCPY_H2D => {
                    PCIE_H2D.fetch_add(1, Relaxed);
                }
                MEMCPY_D2H => {
                    PCIE_D2H.fetch_add(1, Relaxed);
                }
                _ => {}
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

        // ---- hipEvent (event profiler; Phase 45h) -----------------------------
        /// Create a timing event (`hipEvent_t`). `None` on failure (profiler is
        /// best-effort — a failed create just drops that sample, never the run).
        pub fn event_create(&self) -> Option<*mut c_void> {
            let mut ev: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { (self.event_create)(&mut ev) };
            (rc == 0 && !ev.is_null()).then_some(ev)
        }

        /// Record `ev` into the default (null) stream — timestamps the GPU clock when
        /// the ordered stream reaches this marker.
        pub fn event_record(&self, ev: *mut c_void) {
            unsafe {
                let _ = (self.event_record)(ev, std::ptr::null_mut());
            }
        }

        /// Block until `ev` has completed (its GPU timestamp is final).
        pub fn event_synchronize(&self, ev: *mut c_void) {
            unsafe {
                let _ = (self.event_synchronize)(ev);
            }
        }

        /// Elapsed GPU time `stop - start` in milliseconds (both must be complete).
        pub fn event_elapsed_ms(&self, start: *mut c_void, stop: *mut c_void) -> Option<f32> {
            let mut ms: f32 = 0.0;
            let rc = unsafe { (self.event_elapsed_time)(&mut ms, start, stop) };
            (rc == 0).then_some(ms)
        }

        pub fn event_destroy(&self, ev: *mut c_void) {
            unsafe {
                let _ = (self.event_destroy)(ev);
            }
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

    /// Fixed-rank metadata passed BY VALUE to the strided kernels (Phase 45g) —
    /// matches the C `struct AkhI64x8 { long v[8]; }`. Rides in the kernel launch
    /// params, so there is no per-op device dims/strides alloc + synchronous H2D +
    /// free (those stalled the stream ~4000×/token). Rank ≤ 8 (decode peaks at 5).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct I64x8 {
        pub(crate) v: [i64; 8],
    }
    impl I64x8 {
        pub(crate) fn pack(s: &[i64]) -> Self {
            let mut v = [0i64; 8];
            let n = s.len().min(8);
            v[..n].copy_from_slice(&s[..n]);
            I64x8 { v }
        }
    }

    type AffineF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, f32, f32) -> c_int;
    // Phase 45h: fused RMSNorm (x, weight) -> out; one kernel replaces ~7.
    type RmsNormF32 =
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, usize, f32) -> c_int;
    type UnaryF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, c_int) -> c_int;
    type BinaryF32 = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, c_int) -> c_int;
    // Phase 45d (subagent ops): reduce / cast / index_select / where_cond / strided-copy.
    #[allow(clippy::type_complexity)]
    type ReduceF32 = unsafe extern "C" fn(
        *const f32,
        *mut f32,
        usize,
        c_int,
        I64x8,
        I64x8,
        I64x8,
        usize,
        c_int,
    ) -> c_int;
    // Phase 45f: argmin/argmax reduce -> u32 index (same metadata as ReduceF32).
    #[allow(clippy::type_complexity)]
    type ReduceArgF32 = unsafe extern "C" fn(
        *const f32,
        *mut u32,
        usize,
        c_int,
        I64x8,
        I64x8,
        I64x8,
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
    // Phase 45f: strided where_cond (cond/t/f own strides) — `t`=neg_inf is broadcast.
    #[allow(clippy::type_complexity)]
    type WhereCondStridedU8F32 = unsafe extern "C" fn(
        *const u8,
        *const f32,
        *const f32,
        *mut f32,
        usize,
        usize,
        *const i64,
        *const i64,
        *const i64,
        *const i64,
    ) -> c_int;
    #[allow(clippy::type_complexity)]
    type WhereCondStridedU32F32 = unsafe extern "C" fn(
        *const u32,
        *const f32,
        *const f32,
        *mut f32,
        usize,
        usize,
        *const i64,
        *const i64,
        *const i64,
        *const i64,
    ) -> c_int;
    type UcopyF32 = unsafe extern "C" fn(*const f32, *mut f32, usize, usize, I64x8, I64x8) -> c_int;
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
    // Phase 45f: U32 variant of the strided compare (the attention mask is u32).
    #[allow(clippy::type_complexity)]
    type CmpU32 = unsafe extern "C" fn(
        *const u32,
        *const u32,
        *mut u8,
        usize,
        usize,
        *const i64,
        *const i64,
        *const i64,
        c_int,
    ) -> c_int;
    // Phase 45f: strided broadcasting binary (add/sub/mul/div) -> f32. Powers the
    // broadcast_div/mul/sub inside rms_norm/softmax/rope on non-contiguous layouts.
    // (45g) dims/strides passed by value.
    type BinaryStridedF32 = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut f32,
        usize,
        usize,
        I64x8,
        I64x8,
        I64x8,
        c_int,
    ) -> c_int;
    // Phase 45f: strided unary -> f32. Powers `neg` (and friends) on non-contiguous
    // narrow views, e.g. rotate_half inside rope_slow. (45g) dims/stride by value.
    type UnaryStridedF32 = unsafe extern "C" fn(
        *const f32,
        *mut f32,
        usize,
        usize,
        I64x8,
        I64x8,
        c_int,
    ) -> c_int;

    /// The AOT-compiled elementwise kernel library (built by `build.rs`).
    pub struct Kernels {
        _lib: Library,
        affine_f32: AffineF32,
        rms_norm_f32: RmsNormF32,
        unary_f32: UnaryF32,
        binary_f32: BinaryF32,
        reduce_f32: ReduceF32,
        reduce_arg_f32: ReduceArgF32,
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
        cmp_u32: CmpU32,
        binary_strided_f32: BinaryStridedF32,
        unary_strided_f32: UnaryStridedF32,
        where_cond_strided_u8_f32: WhereCondStridedU8F32,
        where_cond_strided_u32_f32: WhereCondStridedU32F32,
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
                let rms_norm_f32: RmsNormF32 = *lib
                    .get(b"akh_rms_norm_f32\0")
                    .map_err(|e| format!("missing akh_rms_norm_f32: {e}"))?;
                let unary_f32: UnaryF32 = *lib
                    .get(b"akh_unary_f32\0")
                    .map_err(|e| format!("missing akh_unary_f32: {e}"))?;
                let binary_f32: BinaryF32 = *lib
                    .get(b"akh_binary_f32\0")
                    .map_err(|e| format!("missing akh_binary_f32: {e}"))?;
                let reduce_f32: ReduceF32 = *lib
                    .get(b"akh_reduce_f32\0")
                    .map_err(|e| format!("missing akh_reduce_f32: {e}"))?;
                let reduce_arg_f32: ReduceArgF32 = *lib
                    .get(b"akh_reduce_arg_f32\0")
                    .map_err(|e| format!("missing akh_reduce_arg_f32: {e}"))?;
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
                let cmp_u32: CmpU32 = *lib
                    .get(b"akh_cmp_u32\0")
                    .map_err(|e| format!("missing akh_cmp_u32: {e}"))?;
                let binary_strided_f32: BinaryStridedF32 = *lib
                    .get(b"akh_binary_strided_f32\0")
                    .map_err(|e| format!("missing akh_binary_strided_f32: {e}"))?;
                let unary_strided_f32: UnaryStridedF32 = *lib
                    .get(b"akh_unary_strided_f32\0")
                    .map_err(|e| format!("missing akh_unary_strided_f32: {e}"))?;
                let where_cond_strided_u8_f32: WhereCondStridedU8F32 = *lib
                    .get(b"akh_where_cond_strided_u8_f32\0")
                    .map_err(|e| format!("missing akh_where_cond_strided_u8_f32: {e}"))?;
                let where_cond_strided_u32_f32: WhereCondStridedU32F32 = *lib
                    .get(b"akh_where_cond_strided_u32_f32\0")
                    .map_err(|e| format!("missing akh_where_cond_strided_u32_f32: {e}"))?;
                Ok(Kernels {
                    _lib: lib,
                    affine_f32,
                    rms_norm_f32,
                    unary_f32,
                    binary_f32,
                    reduce_f32,
                    reduce_arg_f32,
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
                    cmp_u32,
                    binary_strided_f32,
                    unary_strided_f32,
                    where_cond_strided_u8_f32,
                    where_cond_strided_u32_f32,
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

        /// Fused RMSNorm: one block per row over `hidden`. `x`/`out` are
        /// `n_rows * hidden` f32 (row-major); `w` is `hidden` f32.
        pub fn rms_norm_f32(
            &self,
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            n_rows: usize,
            hidden: usize,
            eps: f32,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.rms_norm_f32)(x, w, out, n_rows, hidden, eps) };
            if rc != 0 {
                return Err(format!("akh_rms_norm_f32 failed (rc={rc})"));
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
            dims: I64x8,
            strides: I64x8,
            reduce_mask: I64x8,
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

        /// argmin/argmax reduce -> u32 index. Same metadata contract as [`reduce_f32`];
        /// op 0 argmin, 1 argmax. `y` holds `out_numel` u32 indices.
        #[allow(clippy::too_many_arguments)]
        pub fn reduce_arg_f32(
            &self,
            x: *const f32,
            y: *mut u32,
            out_numel: usize,
            num_dims: c_int,
            dims: I64x8,
            strides: I64x8,
            reduce_mask: I64x8,
            reduce_numel: usize,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.reduce_arg_f32)(
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
                return Err(format!("akh_reduce_arg_f32 failed (rc={rc})"));
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
            dims: I64x8,
            strides: I64x8,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.ucopy_f32)(src, dst, n, rank, dims, strides) };
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

        /// U32 variant of [`cmp_f32`] (attention mask is u32).
        #[allow(clippy::too_many_arguments)]
        pub fn cmp_u32(
            &self,
            a: *const u32,
            b: *const u32,
            out: *mut u8,
            n: usize,
            rank: usize,
            dims: *const i64,
            a_str: *const i64,
            b_str: *const i64,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.cmp_u32)(a, b, out, n, rank, dims, a_str, b_str, op) };
            if rc != 0 {
                return Err(format!("akh_cmp_u32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided broadcasting binary -> f32. Same metadata contract as [`cmp_f32`];
        /// op 0 add / 1 sub / 2 mul / 3 div.
        #[allow(clippy::too_many_arguments)]
        pub fn binary_strided_f32(
            &self,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            n: usize,
            rank: usize,
            dims: I64x8,
            a_str: I64x8,
            b_str: I64x8,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc =
                unsafe { (self.binary_strided_f32)(a, b, c, n, rank, dims, a_str, b_str, op) };
            if rc != 0 {
                return Err(format!("akh_binary_strided_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided unary -> f32. `dims`/`x_str` are by-value rank-≤8 metadata; `x`
        /// already advanced to its start offset; `out` contiguous. Op codes match
        /// [`unary_f32`].
        pub fn unary_strided_f32(
            &self,
            x: *const f32,
            y: *mut f32,
            n: usize,
            rank: usize,
            dims: I64x8,
            x_str: I64x8,
            op: c_int,
        ) -> std::result::Result<(), String> {
            let rc = unsafe { (self.unary_strided_f32)(x, y, n, rank, dims, x_str, op) };
            if rc != 0 {
                return Err(format!("akh_unary_strided_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided where_cond (u8 cond) -> f32. `dims`/`c_str`/`t_str`/`f_str` are
        /// DEVICE long arrays of length `rank`; cond/t/f advanced to start offsets.
        #[allow(clippy::too_many_arguments)]
        pub fn where_cond_strided_u8_f32(
            &self,
            cond: *const u8,
            t: *const f32,
            f: *const f32,
            out: *mut f32,
            n: usize,
            rank: usize,
            dims: *const i64,
            c_str: *const i64,
            t_str: *const i64,
            f_str: *const i64,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.where_cond_strided_u8_f32)(cond, t, f, out, n, rank, dims, c_str, t_str, f_str)
            };
            if rc != 0 {
                return Err(format!("akh_where_cond_strided_u8_f32 failed (rc={rc})"));
            }
            Ok(())
        }

        /// Strided where_cond (u32 cond) -> f32. See [`where_cond_strided_u8_f32`].
        #[allow(clippy::too_many_arguments)]
        pub fn where_cond_strided_u32_f32(
            &self,
            cond: *const u32,
            t: *const f32,
            f: *const f32,
            out: *mut f32,
            n: usize,
            rank: usize,
            dims: *const i64,
            c_str: *const i64,
            t_str: *const i64,
            f_str: *const i64,
        ) -> std::result::Result<(), String> {
            let rc = unsafe {
                (self.where_cond_strided_u32_f32)(cond, t, f, out, n, rank, dims, c_str, t_str, f_str)
            };
            if rc != 0 {
                return Err(format!("akh_where_cond_strided_u32_f32 failed (rc={rc})"));
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

/// `(host→device, device→host)` `hipMemcpy` counts since the last [`reset_pcie_counts`]
/// (Phase 45f). Lets a harness prove the resident forward crosses PCIe ~twice/token
/// (input id up, logits down) instead of the op-offload path's hundreds. Accessible as
/// `candle_core::cuda::pcie_counts()` (the HIP backend fills the cuda slot).
pub fn pcie_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        runtime::PCIE_H2D.load(Relaxed),
        runtime::PCIE_D2H.load(Relaxed),
    )
}

/// Reset the PCIe-crossing counters (call after model load, before decode).
pub fn reset_pcie_counts() {
    use std::sync::atomic::Ordering::Relaxed;
    runtime::PCIE_H2D.store(0, Relaxed);
    runtime::PCIE_D2H.store(0, Relaxed);
}

/// `(calls, misses, miss_nanos)` for the caching allocator since the last reset:
/// total `malloc()` calls, cache misses (a real device-synchronizing `hipMalloc`), and
/// wall time spent inside those `hipMalloc` calls. Quantifies malloc-stall in decode.
pub fn malloc_profile() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        runtime::MALLOC_CALLS.load(Relaxed),
        runtime::MALLOC_MISS.load(Relaxed),
        runtime::MALLOC_MISS_NANOS.load(Relaxed),
    )
}

/// Reset the allocator counters (call after model load, before decode).
pub fn reset_malloc_counts() {
    use std::sync::atomic::Ordering::Relaxed;
    runtime::MALLOC_CALLS.store(0, Relaxed);
    runtime::MALLOC_MISS.store(0, Relaxed);
    runtime::MALLOC_MISS_NANOS.store(0, Relaxed);
}

/// Fused RMSNorm as a single HIP kernel (Phase 45h): `x / sqrt(mean(x²)+eps) · w`,
/// matching `candle_nn::ops::rms_norm_slow` but in one launch instead of ~7. Profiling
/// showed rms_norm was the single biggest source of decode kernel launches
/// (~2000/token). `x` is `[…, hidden]`, `w` is `[hidden]`, both contiguous f32 on the
/// HIP device. Accessible as `candle_core::cuda::rms_norm_fused`.
pub fn rms_norm_fused(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &FusedRmsNorm { eps })
}

struct FusedRmsNorm {
    eps: f32,
}

impl crate::CustomOp2 for FusedRmsNorm {
    fn name(&self) -> &'static str {
        "fused-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // GPU-only: the model routes here solely for cuda tensors (CPU keeps the
        // candle-nn fast op). Never reached in practice.
        Err(Error::Msg(
            "fused-rms-norm is GPU-only; use candle_nn::ops::rms_norm on CPU".to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        s1: &HipStorage,
        l1: &Layout,
        s2: &HipStorage,
        l2: &Layout,
    ) -> Result<(HipStorage, Shape)> {
        if s1.dtype != DType::F32 || s2.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP fused rms_norm: f32 only, got x={:?} w={:?}",
                s1.dtype, s2.dtype
            )));
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            return Err(Error::Msg(
                "HIP fused rms_norm: contiguous x/weight only".to_string(),
            ));
        }
        let dims = l1.dims();
        let hidden = *dims.last().ok_or_else(|| Error::Msg("rms_norm: scalar x".into()))?;
        let n = l1.shape().elem_count();
        if hidden == 0 || n % hidden != 0 {
            return Err(Error::Msg(format!(
                "HIP fused rms_norm: bad shape {dims:?} (hidden {hidden})"
            )));
        }
        let n_rows = n / hidden;
        let h = hip()?;
        let k = runtime::kernels().map_err(Error::Msg)?;
        let x = unsafe { (s1.ptr as *const f32).add(l1.start_offset()) };
        let w = unsafe { (s2.ptr as *const f32).add(l2.start_offset()) };
        let out = h.malloc(n * std::mem::size_of::<f32>()).map_err(Error::Msg)?;
        let res = event_time(&NORM_NANOS, || {
            k.rms_norm_f32(x, w, out as *mut f32, n_rows, hidden, self.eps)
        });
        if let Err(e) = res {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok((
            HipStorage {
                device: s1.device.clone(),
                ptr: out,
                dtype: DType::F32,
                len: n,
            },
            l1.shape().clone(),
        ))
    }
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

// ---- per-category op profiling (Phase 45h, AKH_HIP_PROFILE) -----------------
// Per-category GPU time, so the bench can split decode into matmul/eltwise/copy/GEMV
// and see whether the next bottleneck is real compute or host/launch overhead. Two
// modes (env `AKH_HIP_PROFILE`):
//   =1            sync-bracket: wall of (sync, op, sync). Simple, but on a single
//                 ordered stream the LEADING sync drains all prior queued work, so it
//                 OVER-attributes time to whichever op it wraps — unreliable for an
//                 async pipeline (it billed attention 37.5% when removing it changed
//                 nothing).
//   =event (=2)   hipEvent stream markers around each op; `elapsed(start,stop)` is the
//                 PURE GPU time of that op — no device drain, no mis-attribution.
//                 Accumulated at flush (one sync per EVENT_FLUSH_CAP pairs). The honest
//                 instrument: Σ(categories) vs decode wall reveals how much is GPU
//                 compute vs host-side stalls (malloc/sync/launch latency).
pub static MATMUL_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ELTWISE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COPY_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// 45h cont: split the former "other" bucket — NORM = fused RMSNorm (~288 launches/token),
// REDUCE = reduce_op (softmax's max/sum, index_select, argmax).
pub static NORM_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REDUCE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfMode {
    Off,
    Sync,
    Event,
}

pub(crate) fn profiling_mode() -> ProfMode {
    use std::sync::OnceLock;
    static M: OnceLock<ProfMode> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("AKH_HIP_PROFILE").as_deref() {
        Ok("1") => ProfMode::Sync,
        Ok("event") | Ok("2") => ProfMode::Event,
        _ => ProfMode::Off,
    })
}

/// True in the legacy sync-bracket mode (`=1`).
pub(crate) fn profiling() -> bool {
    profiling_mode() == ProfMode::Sync
}
/// True in the event-marker mode (`=event`).
pub(crate) fn event_profiling() -> bool {
    profiling_mode() == ProfMode::Event
}

/// A `hipEvent_t` handle. `Send` because it is only ever touched on the single HIP
/// compute thread, additionally guarded by the [`event_prof`] mutex.
#[derive(Clone, Copy)]
struct Ev(*mut c_void);
unsafe impl Send for Ev {}

/// Event-profiler state: a recycled pool of event objects + the not-yet-drained
/// `(counter, start, stop)` triples. Drained (one device sync) every
/// `EVENT_FLUSH_CAP` pairs and at read time.
struct EventProf {
    free: Vec<Ev>,
    pending: Vec<(&'static std::sync::atomic::AtomicU64, Ev, Ev)>,
}
const EVENT_FLUSH_CAP: usize = 4096;

fn event_prof() -> &'static std::sync::Mutex<EventProf> {
    static P: std::sync::OnceLock<std::sync::Mutex<EventProf>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        std::sync::Mutex::new(EventProf {
            free: Vec::new(),
            pending: Vec::new(),
        })
    })
}

/// Acquire a `(start, stop)` event pair (recycled or freshly created). `None` if event
/// creation fails — the caller then runs the op untimed.
fn event_prof_acquire() -> Option<(Ev, Ev)> {
    let h = hip().ok()?;
    let mut p = event_prof().lock().unwrap();
    let a = p.free.pop().or_else(|| h.event_create().map(Ev))?;
    match p.free.pop().or_else(|| h.event_create().map(Ev)) {
        Some(b) => Some((a, b)),
        None => {
            p.free.push(a); // return the half-pair to the pool, give up this sample
            None
        }
    }
}

/// Register a recorded pair against a category counter; flush if pending grew past cap.
fn event_prof_register(counter: &'static std::sync::atomic::AtomicU64, a: Ev, b: Ev) {
    let mut p = event_prof().lock().unwrap();
    p.pending.push((counter, a, b));
    if p.pending.len() >= EVENT_FLUSH_CAP {
        drain_locked(&mut p);
    }
}

/// Drain pending pairs: one device sync, read each elapsed time into its counter,
/// recycle the events. Caller holds the lock.
fn drain_locked(p: &mut EventProf) {
    if p.pending.is_empty() {
        return;
    }
    let Ok(h) = hip() else {
        p.pending.clear();
        return;
    };
    let _ = h.synchronize(); // every recorded event is now complete
    let pairs: Vec<_> = p.pending.drain(..).collect();
    for (counter, a, b) in pairs {
        if let Some(ms) = h.event_elapsed_ms(a.0, b.0) {
            counter.fetch_add(
                (ms as f64 * 1.0e6) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        p.free.push(a);
        p.free.push(b);
    }
}

/// Flush pending event samples into their counters (call before reading a profile).
pub(crate) fn flush_events() {
    if profiling_mode() != ProfMode::Event {
        return;
    }
    let mut p = event_prof().lock().unwrap();
    drain_locked(&mut p);
}

/// Time `f`'s GPU work into `counter` via event markers (event mode). The closure does
/// the launch; the events bracket it on the stream. Returns `f`'s result unchanged.
/// Used by both the `profiled!` macro and the GEMV path in `quantized::hip`.
pub(crate) fn event_time<R>(
    counter: &'static std::sync::atomic::AtomicU64,
    f: impl FnOnce() -> R,
) -> R {
    if let (Ok(h), Some((a, b))) = (hip(), event_prof_acquire()) {
        h.event_record(a.0);
        let r = f();
        h.event_record(b.0);
        event_prof_register(counter, a, b);
        r
    } else {
        f()
    }
}

/// `(matmul, eltwise, copy, norm, reduce)` intrinsic GPU ns since process start.
/// Flushes any pending event samples first so the read is complete.
pub fn op_profile() -> (u64, u64, u64, u64, u64) {
    flush_events();
    use std::sync::atomic::Ordering::Relaxed;
    (
        MATMUL_NANOS.load(Relaxed),
        ELTWISE_NANOS.load(Relaxed),
        COPY_NANOS.load(Relaxed),
        NORM_NANOS.load(Relaxed),
        REDUCE_NANOS.load(Relaxed),
    )
}

/// Run an op body, accumulating its GPU time into `$counter` per the active profile mode.
macro_rules! profiled {
    ($counter:expr, $body:block) => {{
        match profiling_mode() {
            ProfMode::Off => (|| $body)(),
            ProfMode::Sync => {
                if let Ok(h) = hip() {
                    let _ = h.synchronize();
                }
                let __t0 = std::time::Instant::now();
                let __r = (|| $body)();
                if let Ok(h) = hip() {
                    let _ = h.synchronize();
                }
                $counter.fetch_add(
                    __t0.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                __r
            }
            ProfMode::Event => event_time(&$counter, || $body),
        }
    }};
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
        // (op_code, is_arg): value reductions emit f32; arg reductions emit a u32 index
        // and require a single reduced dim (parity with the CPU backend).
        let (op_code, is_arg): (c_int, bool) = match op {
            ReduceOp::Sum => (0, false),
            ReduceOp::Min => {
                if reduce_dims.len() != 1 {
                    return Err(Error::Msg(format!(
                        "HIP reduce_op: `min` over multiple dims {reduce_dims:?} not yet \
                         implemented (Phase 45d); reduce one dim at a time"
                    )));
                }
                (1, false)
            }
            ReduceOp::Max => {
                if reduce_dims.len() != 1 {
                    return Err(Error::Msg(format!(
                        "HIP reduce_op: `max` over multiple dims {reduce_dims:?} not yet \
                         implemented (Phase 45d); reduce one dim at a time"
                    )));
                }
                (2, false)
            }
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                if reduce_dims.len() != 1 {
                    return Err(Error::Msg(format!(
                        "HIP reduce_op: `argmin`/`argmax` over multiple dims {reduce_dims:?} not \
                         supported; reduce one dim at a time"
                    )));
                }
                (if matches!(op, ReduceOp::ArgMax) { 1 } else { 0 }, true)
            }
        };

        let src_dims = layout.dims();
        let num_dims = src_dims.len();
        if num_dims > 8 {
            return Err(Error::Msg(format!(
                "HIP reduce_op: rank {num_dims} > 8 unsupported (by-value metadata, Phase 45g)"
            )));
        }
        let mut strides = vec![0i64; num_dims];
        let mut acc = 1i64;
        for d in (0..num_dims).rev() {
            strides[d] = acc;
            acc *= src_dims[d] as i64;
        }
        let mut reduce_mask = vec![0i64; num_dims];
        let mut out_dims = src_dims.to_vec();
        let mut reduce_numel: usize = 1;
        for &d in reduce_dims {
            reduce_mask[d] = 1;
            reduce_numel *= src_dims[d];
            out_dims[d] = 1;
        }
        let out_shape = Shape::from(out_dims);
        let out_numel = out_shape.elem_count();
        let dims_i64: Vec<i64> = src_dims.iter().map(|&d| d as i64).collect();

        let h = hip()?;
        let k = runtime::kernels().map_err(Error::Msg)?;

        // Metadata rides in the kernel launch params (by value) — no device alloc /
        // synchronous H2D / free per reduce (Phase 45g).
        let dims_m = runtime::I64x8::pack(&dims_i64);
        let strides_m = runtime::I64x8::pack(&strides);
        let mask_m = runtime::I64x8::pack(&reduce_mask);

        let src = unsafe { (self.ptr as *const f32).add(layout.start_offset()) };
        let elem = if is_arg {
            std::mem::size_of::<u32>()
        } else {
            std::mem::size_of::<f32>()
        };
        let out = h.malloc(out_numel * elem).map_err(Error::Msg)?;
        let launch = event_time(&REDUCE_NANOS, || {
            if is_arg {
                k.reduce_arg_f32(
                    src,
                    out as *mut u32,
                    out_numel,
                    num_dims as c_int,
                    dims_m,
                    strides_m,
                    mask_m,
                    reduce_numel,
                    op_code,
                )
            } else {
                k.reduce_f32(
                    src,
                    out as *mut f32,
                    out_numel,
                    num_dims as c_int,
                    dims_m,
                    strides_m,
                    mask_m,
                    reduce_numel,
                    op_code,
                )
            }
        });
        if let Err(e) = launch {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: if is_arg { DType::U32 } else { DType::F32 },
            len: out_numel,
        })
    }

    /// Strided elementwise compare -> U8 (Phase 45f). Both operands may be
    /// broadcast/non-contiguous (e.g. prefill's `mask.eq(0)`), so we pass each
    /// side's own strides and reconstruct offsets per output element. f32 only.
    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype != rhs.dtype || !matches!(self.dtype, DType::F32 | DType::U32) {
            return Err(Error::Msg(format!(
                "HIP cmp: only matching f32/u32 supported (Phase 45f), got lhs={:?} rhs={:?}",
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

        let out = match h.malloc(n * std::mem::size_of::<u8>()) {
            Ok(p) => p,
            Err(e) => {
                free_meta();
                return Err(Error::Msg(e));
            }
        };
        let launch = match self.dtype {
            DType::F32 => {
                let a = unsafe { (self.ptr as *const f32).add(lhs_l.start_offset()) };
                let b = unsafe { (rhs.ptr as *const f32).add(rhs_l.start_offset()) };
                k.cmp_f32(
                    a,
                    b,
                    out as *mut u8,
                    n,
                    rank,
                    dims_dev as *const i64,
                    a_dev as *const i64,
                    b_dev as *const i64,
                    op_code,
                )
            }
            DType::U32 => {
                let a = unsafe { (self.ptr as *const u32).add(lhs_l.start_offset()) };
                let b = unsafe { (rhs.ptr as *const u32).add(rhs_l.start_offset()) };
                k.cmp_u32(
                    a,
                    b,
                    out as *mut u8,
                    n,
                    rank,
                    dims_dev as *const i64,
                    a_dev as *const i64,
                    b_dev as *const i64,
                    op_code,
                )
            }
            _ => unreachable!("dtype gated above"),
        };
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
        profiled!(ELTWISE_NANOS, {
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
            "cos" => 10,
            "sin" => 11,
            "log" => 12,
            other => {
                return Err(Error::Msg(format!(
                    "HIP unary `{other}` not yet implemented (Phase 45d)"
                )));
            }
        };
        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP unary `{}`: only f32 supported (Phase 45d), got {:?}",
                B::NAME,
                self.dtype
            )));
        }
        let k = runtime::kernels().map_err(Error::Msg)?;
        // Fast path: contiguous. Otherwise a strided kernel reads the input through its
        // own strides (e.g. `neg` on a narrow view inside rotate_half / rope_slow).
        if layout.is_contiguous() {
            let (src, out, n) = self.elementwise_setup(layout, B::NAME)?;
            k.unary_f32(src, out.ptr as *mut f32, n, op).map_err(Error::Msg)?;
            return Ok(out);
        }
        let h = hip()?;
        let dims = layout.dims();
        let rank = dims.len();
        if rank > 8 {
            return Err(Error::Msg(format!(
                "HIP unary `{}`: rank {rank} > 8 unsupported (by-value metadata, Phase 45g)",
                B::NAME
            )));
        }
        let n = layout.shape().elem_count();
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let strv: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
        // Metadata by value — no device alloc / synchronous H2D / free (Phase 45g).
        let dims_m = runtime::I64x8::pack(&dims_i64);
        let str_m = runtime::I64x8::pack(&strv);
        let out = h.malloc(n * std::mem::size_of::<f32>()).map_err(Error::Msg)?;
        let src = unsafe { (self.ptr as *const f32).add(layout.start_offset()) };
        if let Err(e) = k.unary_strided_f32(src, out as *mut f32, n, rank, dims_m, str_m, op) {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: n,
        })
        })
    }

    /// Contiguous-f32 same-shape binary op, dispatched on `B::NAME` (Phase 45d).
    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        profiled!(ELTWISE_NANOS, {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "HIP binary `{}`: only f32 supported (Phase 45d)",
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
        let out = h
            .malloc(n * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;

        // Fast path: both contiguous (and same shape) -> dense kernel. Otherwise the
        // strided kernel handles broadcast/non-contiguous layouts (the broadcast_mul/
        // broadcast_div/broadcast_sub inside rms_norm_slow / softmax / rope_slow).
        if l1.is_contiguous() && l2.is_contiguous() {
            let a = unsafe { (self.ptr as *const f32).add(l1.start_offset()) };
            let b = unsafe { (rhs.ptr as *const f32).add(l2.start_offset()) };
            if let Err(e) = k.binary_f32(a, b, out as *mut f32, n, op) {
                h.free(out);
                return Err(Error::Msg(e));
            }
            return Ok(HipStorage {
                device: self.device.clone(),
                ptr: out,
                dtype: DType::F32,
                len: n,
            });
        }

        // Strided path: candle broadcasts both operands to the common shape before
        // calling binary_impl, so l1.dims() == l2.dims(); each side keeps its own
        // (possibly stride-0) strides — passed by value in the launch (Phase 45g).
        let dims = l1.dims();
        let rank = dims.len();
        if rank > 8 {
            h.free(out);
            return Err(Error::Msg(format!(
                "HIP binary `{}`: rank {rank} > 8 unsupported (by-value metadata, Phase 45g)",
                B::NAME
            )));
        }
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let a_strv: Vec<i64> = l1.stride().iter().map(|&s| s as i64).collect();
        let b_strv: Vec<i64> = l2.stride().iter().map(|&s| s as i64).collect();
        let dims_m = runtime::I64x8::pack(&dims_i64);
        let a_m = runtime::I64x8::pack(&a_strv);
        let b_m = runtime::I64x8::pack(&b_strv);
        let a = unsafe { (self.ptr as *const f32).add(l1.start_offset()) };
        let b = unsafe { (rhs.ptr as *const f32).add(l2.start_offset()) };
        if let Err(e) = k.binary_strided_f32(a, b, out as *mut f32, n, rank, dims_m, a_m, b_m, op)
        {
            h.free(out);
            return Err(Error::Msg(e));
        }
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: DType::F32,
            len: n,
        })
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
        let k = runtime::kernels().map_err(Error::Msg)?;
        let h = hip()?;
        let n = layout.shape().elem_count();
        let t_ptr = unsafe { (t.ptr as *const f32).add(t_l.start_offset()) };
        let f_ptr = unsafe { (f.ptr as *const f32).add(f_l.start_offset()) };
        let out = h
            .malloc(n * std::mem::size_of::<f32>())
            .map_err(Error::Msg)?;

        // Fast path: all three contiguous. Otherwise a strided kernel reads each
        // operand through its own strides (the masked-attention `neg_inf` `t` operand
        // is a broadcast view).
        if layout.is_contiguous() && t_l.is_contiguous() && f_l.is_contiguous() {
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
            return Ok(HipStorage {
                device: self.device.clone(),
                ptr: out,
                dtype: DType::F32,
                len: n,
            });
        }

        // Strided path. candle broadcasts cond/t/f to the common shape, so all share
        // `layout.dims()`; each keeps its own (possibly stride-0) strides.
        let dims = layout.dims();
        let rank = dims.len();
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let c_strv: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
        let t_strv: Vec<i64> = t_l.stride().iter().map(|&s| s as i64).collect();
        let f_strv: Vec<i64> = f_l.stride().iter().map(|&s| s as i64).collect();
        let bytes = rank.max(1) * std::mem::size_of::<i64>();
        // Allocate the four metadata buffers; on any failure free what we have + out.
        let mut metas: [*mut c_void; 4] = [std::ptr::null_mut(); 4];
        for slot in metas.iter_mut() {
            match h.malloc(bytes) {
                Ok(p) => *slot = p,
                Err(e) => {
                    for p in metas.iter() {
                        h.free(*p);
                    }
                    h.free(out);
                    return Err(Error::Msg(e));
                }
            }
        }
        let [dims_dev, c_dev, t_dev, f_dev] = metas;
        let free_metas = || {
            for p in metas.iter() {
                h.free(*p);
            }
        };
        if rank > 0 {
            let up = |dst: *mut c_void, src: &[i64]| {
                h.memcpy(dst, src.as_ptr() as *const c_void, rank * 8, runtime::MEMCPY_H2D)
                    .map_err(Error::Msg)
            };
            if let Err(e) = up(dims_dev, &dims_i64)
                .and_then(|_| up(c_dev, &c_strv))
                .and_then(|_| up(t_dev, &t_strv))
                .and_then(|_| up(f_dev, &f_strv))
            {
                free_metas();
                h.free(out);
                return Err(e);
            }
        }
        let cond8 = unsafe { (self.ptr as *const u8).add(layout.start_offset()) };
        let cond32 = unsafe { (self.ptr as *const u32).add(layout.start_offset()) };
        let launch = match self.dtype {
            DType::U8 => k.where_cond_strided_u8_f32(
                cond8,
                t_ptr,
                f_ptr,
                out as *mut f32,
                n,
                rank,
                dims_dev as *const i64,
                c_dev as *const i64,
                t_dev as *const i64,
                f_dev as *const i64,
            ),
            DType::U32 => k.where_cond_strided_u32_f32(
                cond32,
                t_ptr,
                f_ptr,
                out as *mut f32,
                n,
                rank,
                dims_dev as *const i64,
                c_dev as *const i64,
                t_dev as *const i64,
                f_dev as *const i64,
            ),
            _ => unreachable!("dtype gated above"),
        };
        free_metas();
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
        profiled!(MATMUL_NANOS, {
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
        // No sync (Phase 45h): ordered on the null stream; the final D2H readback syncs.
        Ok(HipStorage {
            device: self.device.clone(),
            ptr: out,
            dtype: self.dtype,
            len: out_len,
        })
        })
    }

    /// Copy `self` (read through the strided/offset `src_l`) into `dst` at element
    /// `dst_offset`, row-major over `src_l.shape()` — powers `.contiguous()` (Phase 45d).
    /// Contiguous source → one D2D memcpy (any dtype); strided → f32 unravel kernel.
    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        profiled!(COPY_NANOS, {
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
        if rank > 8 {
            return Err(Error::Msg(format!(
                "HIP copy_strided_src: rank {rank} > 8 unsupported (by-value metadata, Phase 45g)"
            )));
        }
        // dims/strides by value — no per-call info alloc / synchronous H2D / free.
        let dims_m = runtime::I64x8::pack(&dims.iter().map(|&d| d as i64).collect::<Vec<_>>());
        let str_m =
            runtime::I64x8::pack(&src_l.stride().iter().map(|&s| s as i64).collect::<Vec<_>>());
        let k = runtime::kernels().map_err(Error::Msg)?;
        k.ucopy_f32(src as *const f32, dst_ptr as *mut f32, n, rank, dims_m, str_m)
            .map_err(Error::Msg)
        })
    }

    /// 2D block copy (Phase 45f): `d1` rows of `d2` elements, src/dst row strides
    /// `src_s`/`dst_s` (in elements), into `dst` at element offsets `src_o`/`dst_o`.
    /// Powers candle's `cat` fast path (KV-cache concat). On-device (D2D), one
    /// `hipMemcpy2D`; dtype-agnostic via the element byte size.
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if self.dtype != dst.dtype {
            return Err(Error::Msg(format!(
                "HIP copy2d: dtype mismatch src={:?} dst={:?}",
                self.dtype, dst.dtype
            )));
        }
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let elem = self.dtype.size_in_bytes();
        let h = hip()?;
        let src = unsafe { (self.ptr as *const u8).add(src_o * elem) as *const c_void };
        let dstp = unsafe { (dst.ptr as *mut u8).add(dst_o * elem) as *mut c_void };
        // KV-cache concat in decode — count under COPY (profiler).
        event_time(&COPY_NANOS, || {
            h.memcpy2d(
                dstp,
                dst_s * elem, // dpitch
                src,
                src_s * elem, // spitch
                d2 * elem,    // width (bytes per row)
                d1,           // height (rows)
                runtime::MEMCPY_D2D,
            )
        })
        .map_err(Error::Msg)
        // No sync (Phase 45h): the D2D copy is ordered on the null stream.
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
    fn unary_on_noncontiguous() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // `neg` on a narrowed last-dim view — exactly rotate_half inside rope_slow.
        let d: Vec<f32> = (0..2 * 6).map(|i| i as f32 - 3.0).collect();
        let g = Tensor::from_vec(d.clone(), (2, 6), &dev)?
            .narrow(1, 3, 3)?
            .neg()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let c = Tensor::from_vec(d, (2, 6), &Device::Cpu)?
            .narrow(1, 3, 3)?
            .neg()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(&g, &c, 0.0, "neg narrow");
        Ok(())
    }

    #[test]
    fn cat_on_seq_dim() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // KV-cache shape concat on dim 2: (b,h,t,d) ++ (b,h,1,d) -> exercises copy2d.
        let a: Vec<f32> = (0..2 * 2 * 3 * 4).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..2 * 2 * 1 * 4).map(|i| (i as f32) + 100.0).collect();
        let ga = Tensor::from_vec(a.clone(), (2, 2, 3, 4), &dev)?;
        let gb = Tensor::from_vec(b.clone(), (2, 2, 1, 4), &dev)?;
        let gc = Tensor::cat(&[&ga, &gb], 2)?.flatten_all()?.to_vec1::<f32>()?;
        let ca = Tensor::from_vec(a, (2, 2, 3, 4), &Device::Cpu)?;
        let cb = Tensor::from_vec(b, (2, 2, 1, 4), &Device::Cpu)?;
        let cc = Tensor::cat(&[&ca, &cb], 2)?.flatten_all()?.to_vec1::<f32>()?;
        close(&gc, &cc, 0.0, "cat dim2");
        Ok(())
    }

    #[test]
    fn broadcast_binary_ops() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // (2,3) op (3,) broadcast, and (2,3) op (2,1) broadcast — the rms_norm/softmax
        // shapes (normalize a row by a per-row scalar). Exercises stride-0 dims on rhs.
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let row: Vec<f32> = vec![10.0, 20.0, 30.0];
        let colv: Vec<f32> = vec![2.0, 4.0];
        for (label, run) in [
            ("bcast_row_div", 0usize),
            ("bcast_col_mul", 1usize),
            ("bcast_row_sub", 2usize),
        ] {
            let ga = Tensor::from_vec(a.clone(), (2, 3), &dev)?;
            let ca = Tensor::from_vec(a.clone(), (2, 3), &Device::Cpu)?;
            let (g, c) = match run {
                0 => {
                    let gr = Tensor::from_vec(row.clone(), (3,), &dev)?;
                    let cr = Tensor::from_vec(row.clone(), (3,), &Device::Cpu)?;
                    (ga.broadcast_div(&gr)?, ca.broadcast_div(&cr)?)
                }
                1 => {
                    let gc = Tensor::from_vec(colv.clone(), (2, 1), &dev)?;
                    let cc = Tensor::from_vec(colv.clone(), (2, 1), &Device::Cpu)?;
                    (ga.broadcast_mul(&gc)?, ca.broadcast_mul(&cc)?)
                }
                _ => {
                    let gr = Tensor::from_vec(row.clone(), (3,), &dev)?;
                    let cr = Tensor::from_vec(row.clone(), (3,), &Device::Cpu)?;
                    (ga.broadcast_sub(&gr)?, ca.broadcast_sub(&cr)?)
                }
            };
            close(
                &g.flatten_all()?.to_vec1::<f32>()?,
                &c.flatten_all()?.to_vec1::<f32>()?,
                1e-5,
                label,
            );
        }
        Ok(())
    }

    #[test]
    fn argmax_last_dim() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // Greedy sampling: argmax over the vocab (last) dim of a logits row.
        let d: Vec<f32> = vec![0.1, 0.9, 0.3, 0.2, 0.7, 0.4, 1.5, 0.0];
        let g = Tensor::from_vec(d.clone(), (2, 4), &dev)?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        let c = Tensor::from_vec(d, (2, 4), &Device::Cpu)?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        assert_eq!(g, c, "argmax");
        Ok(())
    }

    #[test]
    fn where_cond_broadcast_t() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        // Masked attention: where(mask.eq(0), neg_inf_scalar_broadcast, scores).
        let cond = [1u8, 0, 0, 1, 1, 0];
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let run = |d: &Device| -> Result<Vec<f32>> {
            let c = Tensor::from_vec(cond.to_vec(), (2, 3), d)?;
            let neg = Tensor::new(-1e30f32, d)?.broadcast_as((2, 3))?;
            let s = Tensor::from_vec(scores.clone(), (2, 3), d)?;
            c.where_cond(&neg, &s)?.flatten_all()?.to_vec1::<f32>()
        };
        close(&run(&dev)?, &run(&Device::Cpu)?, 0.0, "where_cond bcast t");
        Ok(())
    }

    #[test]
    fn fused_rms_norm_matches() -> Result<()> {
        let Some(dev) = gpu() else {
            return Ok(());
        };
        let (rows, hidden) = (3usize, 384usize);
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
        let w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let eps = 1e-6f32;
        let gx = Tensor::from_vec(x.clone(), (rows, hidden), &dev)?;
        let gw = Tensor::from_vec(w.clone(), hidden, &dev)?;
        let g = super::rms_norm_fused(&gx, &gw, eps)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Reference: candle-core ops mirroring rms_norm_slow (candle-nn isn't a dep here).
        let cx = Tensor::from_vec(x, (rows, hidden), &Device::Cpu)?;
        let cw = Tensor::from_vec(w, hidden, &Device::Cpu)?;
        let var = cx.sqr()?.mean_keepdim(D::Minus1)?;
        let c = cx
            .broadcast_div(&(var + eps as f64)?.sqrt()?)?
            .broadcast_mul(&cw)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(&g, &c, 1e-4, "fused rms_norm");
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

        // U32 mask path (the attention mask is built as u32): broadcast eq 0.
        let um: Vec<u32> = vec![1, 0, 1, 1];
        let gu = Tensor::from_vec(um.clone(), (2, 2), &dev)?
            .broadcast_as((3, 1, 2, 2))?
            .eq(0u32)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let cu = Tensor::from_vec(um, (2, 2), &Device::Cpu)?
            .broadcast_as((3, 1, 2, 2))?
            .eq(0u32)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        assert_eq!(gu, cu, "broadcast mask eq u32");
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
