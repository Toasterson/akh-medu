//! Quantized storage for the HIP backend (Phase 45 / ADR 073).
//!
//! Fills the `quantized::cuda` slot when the `hip` feature is on (see
//! `quantized/mod.rs`). `CudaDevice` / `CudaStorage` here resolve to the HIP types
//! (the `cuda` module is aliased to `hip_backend`). Phase 45b is a compiling
//! placeholder; the native Q4_K dequant + rocBLAS GEMM path (already validated in
//! `akh_medu::nlu::hip`) is wired in during Phase 45e.
#![allow(unused)]
use super::GgmlDType;
use crate::{CudaDevice, CudaStorage, Error, Result};

macro_rules! hip_todo {
    ($op:literal) => {
        Err(Error::Msg(format!(
            "HIP quantized backend: `{}` not yet implemented (Phase 45e).",
            $op
        )))
    };
}

pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        hip_todo!("zeros")
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        hip_todo!("dequantize")
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        hip_todo!("dequantize_f16")
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        hip_todo!("quantize")
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &CudaStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        hip_todo!("quantize_imatrix")
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        hip_todo!("quantize_imatrix_onto")
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        hip_todo!("quantize_onto")
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        hip_todo!("device_ptr")
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        hip_todo!("fwd")
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        hip_todo!("data")
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &CudaStorage,
        _: &crate::Layout,
        _: &CudaStorage,
        _: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        hip_todo!("indexed_moe_forward")
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::Msg(
        "HIP quantized backend: `load_quantized` not yet implemented (Phase 45e).".to_string(),
    ))
}
