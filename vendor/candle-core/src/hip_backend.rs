//! AMD **ROCm/HIP** device backend for Candle (Phase 45 / ADR 073).
//!
//! HIP is source-compatible with the CUDA runtime API, so rather than add a parallel
//! `Device::Hip` enum arm across ~130 match sites, this backend **fills the `cuda`
//! module slot** when the `hip` feature is enabled and `cuda` is not (see `lib.rs` /
//! `quantized/mod.rs`). Every `Storage::Cuda` / `Device::Cuda` dispatch in candle-core
//! then routes to HIP unchanged — the standard "hipify the CUDA backend" approach
//! (ADR 068 step 3). The public type names therefore stay `CudaDevice` / `CudaStorage`
//! (re-exported from the honest `HipDevice` / `HipStorage`), and `DeviceLocation::Cuda`
//! labels the AMD GPU.
//!
//! **Phase 45b scope (this file):** the device *constructs* and the routing compiles
//! both with and without the `hip` feature; tensor/storage ops return a clear
//! "not yet implemented" error. Real on-device memory + the op kernels (rocBLAS GEMM,
//! dequant, elementwise, RMSNorm/RoPE/attention) land in Phase 45c/45d, built on the
//! dlopen'd `libamdhip64` / `librocblas` runtime.
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

/// Return a clear "implemented in a later Phase 45 step" error for an op that the
/// HIP backend routes through but has not yet built a kernel for.
macro_rules! hip_todo {
    ($op:literal) => {
        Err(Error::Msg(format!(
            "HIP backend: `{}` not yet implemented (Phase 45c/45d). The device \
             constructs and routing is wired (45b); op kernels follow.",
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

/// Storage resident on a [`HipDevice`]. Placeholder for 45b — 45c gives it a real
/// device allocation (pointer + dtype + len). Re-exported as `CudaStorage`.
#[derive(Debug)]
pub struct HipStorage;

pub use HipDevice as CudaDevice;
pub use HipStorage as CudaStorage;

macro_rules! fail {
    () => {
        unimplemented!("HIP storage value op reached on a placeholder (Phase 45b); kernels land in 45c/45d.")
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl HipStorage {
    pub fn transfer_to_device(&self, _dst: &HipDevice) -> Result<Self> {
        hip_todo!("transfer_to_device")
    }
}

impl HipDevice {
    /// The GPU ordinal this handle targets.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        Ok(Self { ordinal })
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.ordinal)
    }
}

impl crate::backend::BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        hip_todo!("try_clone")
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        hip_todo!("const_set")
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        hip_todo!("to_cpu_storage")
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

    /// Phase 45b: construct a handle for GPU `ordinal`. Real HIP runtime init
    /// (dlopen `libamdhip64`, `hipGetDeviceCount`, `hipSetDevice`) lands in 45c.
    fn new(ordinal: usize) -> Result<Self> {
        Ok(Self { ordinal })
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        hip_todo!("set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        hip_todo!("get_current_seed")
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

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        hip_todo!("zeros_impl")
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        hip_todo!("alloc_uninit")
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        hip_todo!("storage_from_slice")
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        hip_todo!("storage_from_cpu_storage")
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        hip_todo!("storage_from_cpu_storage_owned")
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        hip_todo!("rand_uniform")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        hip_todo!("rand_normal")
    }

    fn synchronize(&self) -> Result<()> {
        // No outstanding async work in the placeholder; real hipDeviceSynchronize in 45c.
        Ok(())
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
