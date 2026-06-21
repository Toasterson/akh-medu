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

/// Minimal dlopen'd HIP runtime (device mgmt + memory). Kept inside candle-core
/// because candle cannot depend on the akh-medu crate; Phase 45d migrates the
/// rocBLAS / kernel primitives from `akh_medu::nlu::hip` alongside this.
mod runtime {
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
}

/// Fetch the HIP runtime, mapping a load failure to a candle error.
fn hip() -> Result<&'static runtime::Hip> {
    runtime::hip().map_err(Error::Msg)
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

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        hip_todo!("affine")
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        hip_todo!("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        hip_todo!("elu")
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        hip_todo!("reduce_op")
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        hip_todo!("cmp")
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        hip_todo!("to_dtype")
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        hip_todo!("unary_impl")
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        hip_todo!("binary_impl")
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        hip_todo!("where_cond")
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

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        hip_todo!("index_select")
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

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        hip_todo!("matmul")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        hip_todo!("copy_strided_src")
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
