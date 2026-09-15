//! Exact GGUF packed-weight Vulkan kernels; no CPU compute fallback.
use super::GgmlDType;
use crate::backend::BackendStorage;
use crate::{DType, Layout, Result, Shape, VulkanDevice, VulkanStorage};

pub struct QVulkanStorage {
    buffer: wgpu::Buffer,
    device: VulkanDevice,
    dtype: GgmlDType,
    bytes: usize,
}

impl QVulkanStorage {
    pub fn from_data(device: &VulkanDevice, data: &[u8], dtype: GgmlDType) -> Result<Self> {
        if !matches!(
            dtype,
            GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            crate::bail!("Vulkan quant format {dtype:?} unsupported");
        }
        if data.len() % dtype.type_size() != 0 {
            crate::bail!("Vulkan invalid packed byte length");
        }
        Ok(Self {
            buffer: device.upload_bytes(data)?,
            device: device.clone(),
            dtype,
            bytes: data.len(),
        })
    }
    pub fn zeros(device: &VulkanDevice, count: usize, dtype: GgmlDType) -> Result<Self> {
        if count % dtype.block_size() != 0 {
            crate::bail!("Vulkan quant block alignment");
        }
        Self::from_data(
            device,
            &vec![0; count / dtype.block_size() * dtype.type_size()],
            dtype,
        )
    }
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
    pub fn storage_size_in_bytes(&self) -> usize {
        self.bytes
    }
    pub fn data(&self) -> Result<Vec<u8>> {
        self.device.read_bytes(&self.buffer, self.bytes)
    }

    fn weight_fn(&self) -> String {
        let expression = match self.dtype {
            GgmlDType::Q4K => "weight_q4k(i / 256u, i % 256u)",
            GgmlDType::Q6K => "weight_q6k(i / 256u, i % 256u)",
            GgmlDType::F32 => "bitcast<f32>(packed[i])",
            GgmlDType::F16 => "unpack2x16float(packed[i / 2u])[i % 2u]",
            GgmlDType::BF16 => "bitcast<f32>((byte_at(i*2u) | (byte_at(i*2u+1u)<<8u))<<16u)",
            _ => unreachable!(),
        };
        format!(
            "{}\nfn weight(i:u32)->f32{{return {expression};}}",
            include_str!("vulkan_quant.wgsl")
        )
    }

    pub fn dequantize(&self, count: usize) -> Result<VulkanStorage> {
        if count != self.bytes / self.dtype.type_size() * self.dtype.block_size() {
            crate::bail!("Vulkan dequantize size mismatch");
        }
        let output = self.device.alloc(count, DType::F32)?;
        let code=format!("@group(0) @binding(0) var<storage,read> packed:array<u32>; @group(0) @binding(1) var<storage,read_write> out:array<f32>; {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{if dispatch_index(id)<{count}u{{out[dispatch_index(id)]=weight(dispatch_index(id));}}}}",self.weight_fn());
        self.device.dispatch(
            code,
            &[&self.buffer, &output.buffer],
            count.div_ceil(64) as u32,
        )?;
        Ok(output)
    }

    pub fn fwd(
        &self,
        shape: &Shape,
        input: &VulkanStorage,
        layout: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        let (rows, cols) = shape.dims2()?;
        if input.dtype() != DType::F32 || layout.dims().last() != Some(&cols) {
            crate::bail!("Vulkan quant matmul requires matching f32 input");
        }
        let batch = layout.shape().elem_count() / cols;
        let outputs = batch * rows;
        let mut dims = layout.dims().to_vec();
        *dims.last_mut().unwrap() = rows;
        let output = self.device.alloc(batch * rows, DType::F32)?;
        if batch == 1 && cols % 256 == 0
            && matches!(self.dtype, GgmlDType::Q4K | GgmlDType::Q6K)
            && std::env::var("AKH_VULKAN_GEMV_BLOCKS").as_deref() != Ok("0")
        {
            let kernel = include_str!("vulkan_gemv_blocks.wgsl")
                .replace("__ROWS__", &rows.to_string())
                .replace("__BLOCKS__", &(cols / 256).to_string())
                .replace("__DOT__", if self.dtype == GgmlDType::Q4K { "dot_q4" } else { "dot_q6" });
            let code = format!("{} {} {kernel}", self.weight_fn(),
                crate::vulkan_backend::index_fn("ii", layout));
            self.device.dispatch(code, &[&self.buffer, &input.buffer, &output.buffer], rows as u32)?;
            return Ok((output, Shape::from(dims)));
        }
        if batch>1 && std::env::var_os("AKH_VULKAN_PREFILL_REFERENCE").is_none() {
            let groups=rows*batch.div_ceil(4);
            let kernel=include_str!("vulkan_prefill.wgsl")
                .replace("__GROUPS__",&groups.to_string()).replace("__ROWS__",&rows.to_string())
                .replace("__COLS__",&cols.to_string()).replace("__BATCH__",&batch.to_string());
            let code=format!("{} {} {kernel}",self.weight_fn(),crate::vulkan_backend::index_fn("ii",layout));
            self.device.dispatch(code,&[&self.buffer,&input.buffer,&output.buffer],groups as u32)?;
            return Ok((output,Shape::from(dims)));
        }
        let code=format!("@group(0) @binding(0) var<storage,read> packed:array<u32>; @group(0) @binding(1) var<storage,read> input:array<f32>; @group(0) @binding(2) var<storage,read_write> out:array<f32>; var<workgroup> partial:array<f32,64>; {} {} @compute @workgroup_size(64) fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32){{if dispatch_group(wid)>={outputs}u{{return;}}let row=dispatch_group(wid)%{rows}u;let batch=dispatch_group(wid)/{rows}u;var sum=0.0;for(var col=lane;col<{cols}u;col+=64u){{sum+=weight(row*{cols}u+col)*input[ii(batch*{cols}u+col)];}}partial[lane]=sum;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){{if lane<stride{{partial[lane]+=partial[lane+stride];}}workgroupBarrier();}}if lane==0u{{out[dispatch_group(wid)]=partial[0];}}}}",self.weight_fn(),crate::vulkan_backend::index_fn("ii",layout));
        self.device.dispatch(
            code,
            &[&self.buffer, &input.buffer, &output.buffer],
            (batch * rows) as u32,
        )?;
        Ok((output, Shape::from(dims)))
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        let values = src.as_slice::<f32>()?;
        let mut quant = self.dtype.cpu_zeros(values.len());
        quant.from_float(values);
        // CPU source quantization is explicit; no Vulkan tensor is read back.
        let bytes =
            unsafe { std::slice::from_raw_parts(quant.as_ptr(), quant.storage_size_in_bytes()) };
        *self = Self::from_data(&self.device, bytes, self.dtype)?;
        Ok(())
    }
    pub fn get_rows(
        &self,
        shape: &Shape,
        ids: &VulkanStorage,
        layout: &Layout,
    ) -> Result<VulkanStorage> {
        let (rows, cols) = shape.dims2()?;
        if ids.dtype() != DType::U32 {
            crate::bail!("Vulkan quantized row indices must be u32");
        }
        let n = layout.shape().elem_count() * cols;
        let output = self.device.alloc(n, DType::F32)?;
        let error = self.device.alloc(1, DType::U32)?;
        let code=format!("@group(0) @binding(0) var<storage,read> packed:array<u32>; @group(0) @binding(1) var<storage,read> ids:array<u32>; @group(0) @binding(2) var<storage,read_write> out:array<f32>; @group(0) @binding(3) var<storage,read_write> error:atomic<u32>; {} {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i>={n}u{{return;}}let row=ids[ii(i/{cols}u)];if row>={rows}u{{atomicStore(&error,1u);return;}}out[i]=weight(row*{cols}u+i%{cols}u);}}",self.weight_fn(),crate::vulkan_backend::index_fn("ii",layout));
        self.device.dispatch(
            code,
            &[&self.buffer, &ids.buffer, &output.buffer, &error.buffer],
            n.div_ceil(64) as u32,
        )?;
        if self.device.read_bytes(&error.buffer, 4)? != [0, 0, 0, 0] {
            crate::bail!("Vulkan quantized row index out of bounds");
        }
        Ok(output)
    }
    pub fn quantize(&mut self, _: &VulkanStorage) -> Result<()> {
        crate::bail!("Vulkan on-device quantization unsupported")
    }
    pub fn quantize_imatrix(&mut self, _: &VulkanStorage, _: &[f32], _: usize) -> Result<()> {
        crate::bail!("Vulkan imatrix quantization unsupported")
    }
    pub fn quantize_imatrix_onto(
        &mut self,
        _: &crate::CpuStorage,
        _: &[f32],
        _: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan imatrix quantization unsupported")
    }
}
