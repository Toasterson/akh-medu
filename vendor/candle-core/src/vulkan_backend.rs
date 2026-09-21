//! Native Vulkan storage. Tensor computation never falls back to CPU.
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
#[path = "vulkan_pipeline_cache.rs"]
mod pipeline_cache;
pub use pipeline_cache::PipelineCacheStats;
use pipeline_cache::PipelineCache;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use wgpu::util::DeviceExt;

fn vk_error(error: impl std::fmt::Display) -> crate::Error {
    crate::Error::Msg(format!("Vulkan: {error}"))
}

/// Inference-only causal depthwise convolution and immutable history update.
pub fn causal_conv_fused(
    input: &crate::Tensor,
    kernel: &crate::Tensor,
    history: Option<&crate::Tensor>,
) -> Result<(crate::Tensor, crate::Tensor)> {
    let (batch, length, channels) = input.dims3()?;
    let (kernel_channels, width) = kernel.dims2()?;
    if batch == 0 || length == 0 || channels == 0 || width == 0 || kernel_channels != channels {
        crate::bail!("Vulkan causal convolution: nonempty input and matching kernel channels required");
    }
    let history_width = width - 1;
    if input.dtype() != DType::F32 || kernel.dtype() != DType::F32
        || !input.device().same_device(kernel.device())
        || history.is_some_and(|h| h.dtype() != DType::F32
            || h.dims() != [batch, history_width, channels]
            || !h.device().same_device(input.device()))
    {
        crate::bail!("Vulkan causal convolution: matching f32 input/kernel/history on one device required");
    }
    let (input_guard, input_layout) = input.storage_and_layout();
    let (kernel_guard, kernel_layout) = kernel.storage_and_layout();
    let history_guard = history.map(crate::Tensor::storage_and_layout);
    let (crate::Storage::Vulkan(x), crate::Storage::Vulkan(weight)) = (&*input_guard, &*kernel_guard) else {
        crate::bail!("Vulkan causal convolution: GPU inputs required");
    };
    let (history_buffer, history_index) = if let Some((guard, layout)) = &history_guard {
        let crate::Storage::Vulkan(h) = &**guard else {
            crate::bail!("Vulkan causal convolution: GPU history required");
        };
        (&h.buffer, index_fn("hi", layout))
    } else {
        // No allocation or host transfer for zero initial history. This binding
        // is read-only and never accessed by the no-history specialization.
        (&x.buffer, String::from("fn hi(i:u32)->u32{return i;}"))
    };
    let history_count = batch.checked_mul(history_width).and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| vk_error("causal convolution history size overflow"))?;
    let count = input.elem_count();
    let groups = u32::try_from(count.max(history_count).div_ceil(64)).map_err(vk_error)?;
    let device = &x.device;
    let output = device.alloc(count, DType::F32)?;
    let next = device.alloc(history_count, DType::F32)?;
    let history_write = if history_width == 0 {
        // Zero-length storage still has one padded word. Referencing it keeps
        // the fixed five-binding layout valid for this specialization.
        String::from("if i==0u{next[0]=0.0;}")
    } else {
        format!("if i<{history_count}u{{let c=i%C;let h=(i/C)%H;let b=i/(H*C);next[i]=extended(b,L+h,c);}}")
    };
    let source = include_str!("vulkan_causal_conv.wgsl")
        .replace("__HISTORY_WRITE__", &history_write);
    let code = format!("const L:u32={length}u;const C:u32={channels}u;const K:u32={width}u;const H:u32={history_width}u;const N:u32={count}u;const HAS_HISTORY:bool={};\n{} {} {history_index} {source}",
        history.is_some(), index_fn("xi", input_layout), index_fn("wi", kernel_layout));
    device.dispatch(code, &[&x.buffer, &weight.buffer, history_buffer, &output.buffer, &next.buffer], groups)?;
    Ok((
        crate::Tensor::from_storage(crate::Storage::Vulkan(output), input.shape(), crate::op::BackpropOp::none(), false),
        crate::Tensor::from_storage(crate::Storage::Vulkan(next), (batch, history_width, channels), crate::op::BackpropOp::none(), false),
    ))
}

/// Inference-only weighted RMS normalization; no host tensor computation.
pub fn rms_norm_fused(x: &crate::Tensor, weight: &crate::Tensor, eps: f32) -> Result<crate::Tensor> {
    let width = x.dim(crate::D::Minus1)?;
    if width == 0 || weight.dims() != [width] || !eps.is_finite() || eps < 0.0 {
        crate::bail!("Vulkan RMS: nonempty rows, matching weight vector and finite nonnegative epsilon required");
    }
    if x.dtype() != DType::F32 || weight.dtype() != DType::F32
        || !x.device().same_device(weight.device())
    {
        crate::bail!("Vulkan RMS: f32 inputs on the same Vulkan device required");
    }
    let (x_guard, x_layout) = x.storage_and_layout();
    let (w_guard, w_layout) = weight.storage_and_layout();
    let (crate::Storage::Vulkan(input), crate::Storage::Vulkan(weight)) = (&*x_guard, &*w_guard) else {
        crate::bail!("Vulkan RMS: GPU inputs required");
    };
    let device = &input.device;
    let rows = x.elem_count() / width;
    let groups = u32::try_from(rows).map_err(vk_error)?;
    let lanes = if device.0.device.limits().max_compute_invocations_per_workgroup >= 256
        && device.0.device.limits().max_compute_workgroup_size_x >= 256 { 256 } else { 64 };
    let output = device.alloc(x.elem_count(), DType::F32)?;
    let kernel = include_str!("vulkan_rms_norm.wgsl")
        .replace("__WIDTH__", &width.to_string())
        .replace("__ROWS__", &rows.to_string())
        .replace("__LANES__", &lanes.to_string())
        .replace("__INV_WIDTH__", &((1.0 / width as f64) as f32).to_bits().to_string())
        .replace("__EPS__", &eps.to_bits().to_string());
    let code = format!("{} {} {kernel}", index_fn("xi", x_layout), index_fn("wi", w_layout));
    device.dispatch(code, &[&input.buffer, &weight.buffer, &output.buffer], groups)?;
    Ok(crate::Tensor::from_storage(crate::Storage::Vulkan(output), x.shape(),
        crate::op::BackpropOp::none(), false))
}

#[derive(Debug)]
struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ordinal: usize,
    info: wgpu::AdapterInfo,
    pipelines: Mutex<PipelineCache<wgpu::ComputePipeline>>,
    pending: Mutex<Vec<wgpu::CommandBuffer>>,
    submission_batch: usize,
    encoder_batching: bool,
    pass_batching: bool,
    encoder: Mutex<EncoderBatch>,
}

#[derive(Debug, Default)]
struct EncoderBatch {
    // A retained compute pass must be dropped before its encoder.
    pass: Option<wgpu::ComputePass<'static>>,
    encoder: Option<wgpu::CommandEncoder>,
    commands: usize,
    // A failed encoder may contain earlier queued work. Never let callers read
    // uncomputed tensors after that failure; a fresh device is required.
    failed: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VulkanDevice(Arc<Context>);

#[derive(Debug)]
pub struct VulkanStorage {
    pub(crate) buffer: wgpu::Buffer,
    device: VulkanDevice,
    dtype: DType,
    len: usize,
}

// Layout metadata is evaluated on CPU; all element values stay on the GPU.
pub(crate) fn index_fn(name: &str, layout: &Layout) -> String {
    let mut code = format!(
        "fn {name}(flat: u32) -> u32 {{ var i=flat; var out={}u;",
        layout.start_offset()
    );
    for (&dim, &stride) in layout.dims().iter().zip(layout.stride()).rev() {
        code.push_str(&format!(
            "out += (i % {}u) * {stride}u; i /= {}u;",
            dim.max(1),
            dim.max(1)
        ));
    }
    code.push_str("return out; }\n");
    code
}

impl VulkanDevice {
    fn fail_shared(&self, batch: &mut EncoderBatch, error: impl std::fmt::Display) -> Result<()> {
        batch.failed = Some(error.to_string());
        // Ending an already-invalid pass can itself report validation errors.
        // Preserve the first failure and keep cleanup inside an error scope.
        let scope = self.0.device.push_error_scope(wgpu::ErrorFilter::Validation);
        batch.pass.take();
        batch.encoder.take();
        batch.commands = 0;
        let _ = pollster::block_on(scope.pop());
        Err(vk_error(error))
    }

    fn record_compute(&self, pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup,
        groups_x: u32, groups_y: u32) -> Result<()> {
        if !self.0.pass_batching {
            return self.record_shared(false, |encoder| {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            });
        }
        let mut batch = self.0.encoder.lock().map_err(vk_error)?;
        if let Some(error) = &batch.failed { return Err(vk_error(error)); }
        let scope = self.0.device.push_error_scope(wgpu::ErrorFilter::Validation);
        if batch.encoder.is_none() {
            batch.encoder = Some(self.0.device.create_command_encoder(&Default::default()));
        }
        if batch.pass.is_none() {
            // Safe wgpu API: the encoder is runtime-locked until this pass ends.
            // All encoder mutation is serialized here and closes the pass first.
            batch.pass = Some(batch.encoder.as_mut().unwrap()
                .begin_compute_pass(&Default::default()).forget_lifetime());
        }
        let pass = batch.pass.as_mut().unwrap();
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return self.fail_shared(&mut batch, error);
        }
        batch.commands += 1;
        if batch.commands >= self.0.submission_batch { self.finish_shared(&mut batch)?; }
        Ok(())
    }

    fn record_shared(&self, force: bool, encode: impl FnOnce(&mut wgpu::CommandEncoder)) -> Result<()> {
        let mut batch = self.0.encoder.lock().map_err(vk_error)?;
        if let Some(error) = &batch.failed { return Err(vk_error(error)); }
        let scope = self.0.device.push_error_scope(wgpu::ErrorFilter::Validation);
        batch.pass.take();
        let encoder = batch.encoder.get_or_insert_with(||
            self.0.device.create_command_encoder(&Default::default()));
        encode(encoder);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return self.fail_shared(&mut batch, error);
        }
        batch.commands += 1;
        if force || batch.commands >= self.0.submission_batch {
            self.finish_shared(&mut batch)?;
        }
        Ok(())
    }

    fn finish_shared(&self, batch: &mut EncoderBatch) -> Result<()> {
        if let Some(error) = &batch.failed { return Err(vk_error(error)); }
        if let Some(encoder) = batch.encoder.take() {
            let scope = self.0.device.push_error_scope(wgpu::ErrorFilter::Validation);
            batch.pass.take();
            let command = encoder.finish();
            batch.commands = 0;
            if let Some(error) = pollster::block_on(scope.pop()) {
                batch.failed = Some(error.to_string());
                return Err(vk_error(error));
            }
            // One already-batched encoder, not 64 batches of 64 operations.
            if let Err(error) = self.submit(command, true) {
                batch.failed = Some(error.to_string());
                return Err(error);
            }
        }
        Ok(())
    }

    fn copy_bytes(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, bytes: u64, force: bool) -> Result<()> {
        if self.0.encoder_batching {
            self.record_shared(force, |encoder| encoder.copy_buffer_to_buffer(src, 0, dst, 0, bytes))
        } else {
            let mut encoder = self.0.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, bytes);
            self.submit(encoder.finish(), force)
        }
    }

    // Keep command-buffer order identical to the immediate route. Batching
    // changes queue submission only, not kernels, buffer lifetimes or fences.
    // Readback and synchronize always flush the incomplete tail.
    fn submit(&self, command: wgpu::CommandBuffer, force: bool) -> Result<()> {
        let mut pending = self.0.pending.lock().map_err(vk_error)?;
        pending.push(command);
        if force || pending.len() >= self.0.submission_batch {
            self.submit_pending(&mut pending)?;
        }
        Ok(())
    }

    fn submit_pending(&self, pending: &mut Vec<wgpu::CommandBuffer>) -> Result<()> {
        if !pending.is_empty() {
            let scope = self.0.device.push_error_scope(wgpu::ErrorFilter::Validation);
            self.0.queue.submit(pending.drain(..));
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(vk_error(error));
            }
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        if self.0.encoder_batching {
            let mut batch = self.0.encoder.lock().map_err(vk_error)?;
            return self.finish_shared(&mut batch);
        }
        let mut pending = self.0.pending.lock().map_err(vk_error)?;
        self.submit_pending(&mut pending)
    }

    /// Shader specialization cache counters for this device. Application-held
    /// residency only: it does not observe driver-internal pipeline caches.
    pub fn pipeline_cache_stats(&self) -> Result<PipelineCacheStats> {
        Ok(self.0.pipelines.lock().map_err(vk_error)?.stats())
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.0.info
    }

    pub fn storage_binding_limit(&self) -> u64 {
        self.0.device.limits().max_storage_buffer_binding_size
    }

    pub(crate) fn upload_bytes(&self, bytes: &[u8]) -> Result<wgpu::Buffer> {
        let size = bytes.len().max(4).next_multiple_of(4);
        self.check_size(size)?;
        let mut padded = bytes.to_vec();
        padded.resize(size, 0);
        Ok(self
            .0
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Candle Vulkan upload"),
                contents: &padded,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }))
    }

    fn check_size(&self, size: usize) -> Result<()> {
        let limits = self.0.device.limits();
        if size as u64 > limits.max_buffer_size
            || size as u64 > limits.max_storage_buffer_binding_size as u64
        {
            crate::bail!(
                "Vulkan buffer size {size} exceeds negotiated storage limit {}",
                limits.max_storage_buffer_binding_size
            );
        }
        Ok(())
    }

    pub(crate) fn alloc(&self, len: usize, dtype: DType) -> Result<VulkanStorage> {
        if !matches!(dtype, DType::F32 | DType::U32 | DType::U8) {
            crate::bail!("Vulkan storage dtype {dtype:?} unsupported");
        }
        let bytes = len
            .checked_mul(4)
            .ok_or_else(|| vk_error("allocation overflow"))?
            .max(4);
        self.check_size(bytes)?;
        let buffer = self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Vulkan tensor"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(VulkanStorage {
            buffer,
            device: self.clone(),
            dtype,
            len,
        })
    }

    pub(crate) fn read_bytes(&self, buffer: &wgpu::Buffer, bytes: usize) -> Result<Vec<u8>> {
        let size = bytes.max(4).next_multiple_of(4) as u64;
        let dst = self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.copy_bytes(buffer, &dst, size, true)?;
        let (tx, rx) = std::sync::mpsc::channel();
        dst.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.synchronize()?;
        rx.recv_timeout(Duration::from_secs(30))
            .map_err(vk_error)?
            .map_err(vk_error)?;
        let result = dst.slice(..).get_mapped_range()[..bytes].to_vec();
        dst.unmap();
        Ok(result)
    }

    pub(crate) fn dispatch(
        &self,
        code: String,
        buffers: &[&wgpu::Buffer],
        groups: u32,
    ) -> Result<()> {
        if groups == 0 {
            return Ok(());
        }
        let limit = self.0.device.limits().max_compute_workgroups_per_dimension;
        let groups_x = groups.min(limit);
        let groups_y = groups.div_ceil(groups_x);
        if groups_y > limit {
            crate::bail!("Vulkan dispatch too large: {groups}");
        }
        // Elementwise kernels use 64 lanes with dispatch_index; row kernels
        // use dispatch_group with their own lane counts. Both span a 2D grid.
        let code = format!("fn dispatch_group(v:vec3<u32>)->u32{{return v.x+v.y*{groups_x}u;}} fn dispatch_index(v:vec3<u32>)->u32{{return v.x+v.y*{groups_x}u*64u;}}\n{code}");
        let mut cache = self.0.pipelines.lock().map_err(vk_error)?;
        let pipeline = cache.get_or_try_insert_with(code, |code| {
            let scope = self
                .0
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let shader = self
                .0
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Candle Vulkan compute"),
                    source: wgpu::ShaderSource::Wgsl(code.into()),
                });
            let pipeline =
                self.0
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: None,
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(vk_error(error));
            }
            Ok(pipeline)
        })?;
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let scope = self
            .0
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let group = self.0.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        if self.0.encoder_batching {
            // Reject bad bindings before they can invalidate queued operations.
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(vk_error(error));
            }
            return self.record_compute(&pipeline, &group, groups_x, groups_y);
        }
        let mut encoder = self.0.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        let command = encoder.finish();
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(vk_error(error));
        }
        self.submit(command, false)?;
        Ok(())
    }
}

impl VulkanStorage {
    fn require_f32(&self) -> Result<()> {
        if self.dtype != DType::F32 {
            crate::bail!("Vulkan operation requires f32, got {:?}", self.dtype);
        }
        Ok(())
    }
    fn elementwise(
        &self,
        layout: &Layout,
        rhs: Option<(&Self, &Layout)>,
        expression: &str,
        dtype: DType,
    ) -> Result<Self> {
        let n = layout.shape().elem_count();
        let output = self.device.alloc(n, dtype)?;
        let mut code = String::from("@group(0) @binding(0) var<storage,read> a: array<u32>;\n");
        let mut buffers = vec![&self.buffer];
        code.push_str(&index_fn("ai", layout));
        if let Some((r, l)) = rhs {
            code.push_str("@group(0) @binding(1) var<storage,read> b: array<u32>;\n");
            code.push_str(&index_fn("bi", l));
            buffers.push(&r.buffer);
        }
        code.push_str(&format!(
            "@group(0) @binding({}) var<storage,read_write> out: array<u32>;\n",
            buffers.len()
        ));
        buffers.push(&output.buffer);
        code.push_str(&format!("@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{ let i=dispatch_index(id); if i >= {n}u {{ return; }} let x=bitcast<f32>(a[ai(i)]);"));
        if rhs.is_some() {
            code.push_str("let y=bitcast<f32>(b[bi(i)]);");
        }
        code.push_str(&format!("out[i] = {expression}; }}"));
        self.device
            .dispatch(code, &buffers, n.div_ceil(64) as u32)?;
        Ok(output)
    }
}

impl crate::backend::BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Tensor::copy preserves its original layout, including offsets/strides.
        let output = self.device.alloc(self.len, self.dtype)?;
        self.device.copy_bytes(&self.buffer, &output.buffer, (self.len * 4).max(4) as u64, false)?;
        Ok(output)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let word = match self.dtype {
            DType::F32 => (v.to_f64() as f32).to_bits(),
            DType::U32 | DType::U8 => v.to_f64() as u32,
            _ => crate::bail!("Vulkan const dtype unsupported"),
        };
        let n = l.shape().elem_count();
        let code=format!("@group(0) @binding(0) var<storage,read_write> out:array<u32>; {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i<{n}u{{out[oi(i)]={word}u;}}}}",index_fn("oi",l));
        self.device
            .dispatch(code, &[&self.buffer], n.div_ceil(64) as u32)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self.device.read_bytes(&self.buffer, self.len * 4)?;
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        match self.dtype {
            DType::F32 => Ok(CpuStorage::F32(
                words.into_iter().map(f32::from_bits).collect(),
            )),
            DType::U32 => Ok(CpuStorage::U32(words)),
            DType::U8 => Ok(CpuStorage::U8(words.into_iter().map(|v| v as u8).collect())),
            _ => crate::bail!("Vulkan dtype unsupported"),
        }
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.require_f32()?;
        self.elementwise(
            l,
            None,
            &format!(
                "bitcast<u32>(x * bitcast<f32>({}u) + bitcast<f32>({}u))",
                (mul as f32).to_bits(),
                (add as f32).to_bits()
            ),
            DType::F32,
        )
    }

    fn powf(&self, l: &Layout, p: f64) -> Result<Self> {
        self.require_f32()?;
        self.elementwise(
            l,
            None,
            &format!(
                "bitcast<u32>(pow(x,bitcast<f32>({}u)))",
                (p as f32).to_bits()
            ),
            DType::F32,
        )
    }

    fn elu(&self, l: &Layout, p: f64) -> Result<Self> {
        self.require_f32()?;
        self.elementwise(
            l,
            None,
            &format!(
                "bitcast<u32>(select(bitcast<f32>({}u)*(exp(x)-1.0),x,x>=0.0))",
                (p as f32).to_bits()
            ),
            DType::F32,
        )
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        self.require_f32()?;
        let kept: Vec<_> = (0..l.dims().len()).filter(|i| !dims.contains(i)).collect();
        let layout_for = |axes: &[usize], offset| {
            Layout::new(
                Shape::from(axes.iter().map(|&i| l.dims()[i]).collect::<Vec<_>>()),
                axes.iter().map(|&i| l.stride()[i]).collect(),
                offset,
            )
        };
        let outer = layout_for(&kept, l.start_offset());
        let inner = layout_for(dims, 0);
        let n = outer.shape().elem_count();
        let reduce = inner.shape().elem_count();
        if reduce == 0 {
            crate::bail!("Vulkan empty reduction unsupported");
        }
        let is_arg = matches!(op, ReduceOp::ArgMin | ReduceOp::ArgMax);
        if is_arg && dims.len() != 1 {
            crate::bail!("Vulkan arg reduction requires one axis");
        }
        let output = self
            .device
            .alloc(n, if is_arg { DType::U32 } else { DType::F32 })?;
        if (is_arg || op == ReduceOp::Sum)
            && dims.len() == 1
            && reduce >= 256
            && l.is_contiguous()
            && dims[0] + 1 == l.dims().len()
        {
            let limits = self.device.0.device.limits();
            let lanes = if limits.max_compute_invocations_per_workgroup >= 256
                && limits.max_compute_workgroup_size_x >= 256
            {
                256
            } else {
                64
            };
            let source = if is_arg {
                include_str!("vulkan_arg_reduce.wgsl")
            } else {
                include_str!("vulkan_sum_reduce.wgsl")
            };
            let code = source
                .replace("__HALF_LANES__", &(lanes / 2).to_string())
                .replace("__LANES__", &lanes.to_string())
                .replace("__ROWS__", &n.to_string())
                .replace("__WIDTH__", &reduce.to_string())
                .replace("__OFFSET__", &l.start_offset().to_string())
                .replace(
                    "__COMPARE__",
                    if op == ReduceOp::ArgMax { ">" } else { "<" },
                );
            self.device
                .dispatch(code, &[&self.buffer, &output.buffer], n as u32)?;
            return Ok(output);
        }
        let (init, update) = match op {
            ReduceOp::Sum => ("0.0", "sum+=v;"),
            ReduceOp::Min => ("bitcast<f32>(0x7f800000u)", "sum=min(sum,v);"),
            ReduceOp::Max => ("bitcast<f32>(0xff800000u)", "sum=max(sum,v);"),
            ReduceOp::ArgMin => ("a[oi(i)+ri(0u)]", "if v<sum{sum=v;arg=j;}"),
            ReduceOp::ArgMax => ("a[oi(i)+ri(0u)]", "if v>sum{sum=v;arg=j;}"),
        };
        let value = if is_arg { "arg" } else { "bitcast<u32>(sum)" };
        let start = if is_arg { 1 } else { 0 };
        let nan_guard = if is_arg {
            "if (bitcast<u32>(sum)&0x7fffffffu)>0x7f800000u{out[i]=0u;return;}"
        } else {
            ""
        };
        let value_guard = if is_arg {
            "if (bitcast<u32>(v)&0x7fffffffu)>0x7f800000u{continue;}"
        } else {
            ""
        };
        let code=format!("@group(0) @binding(0) var<storage,read> a:array<f32>; @group(0) @binding(1) var<storage,read_write> out:array<u32>; {} {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i>={n}u{{return;}}var sum={init};var arg=0u;{nan_guard}for(var j={start}u;j<{reduce}u;j++){{let v=a[oi(i)+ri(j)];{value_guard}{update}}}out[i]={value};}}",index_fn("oi",&outer),index_fn("ri",&inner));
        self.device
            .dispatch(code, &[&self.buffer, &output.buffer], n.div_ceil(64) as u32)?;
        Ok(output)
    }

    fn cmp(&self, op: CmpOp, r: &Self, l: &Layout, rl: &Layout) -> Result<Self> {
        let cmp = match op {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Le => "<=",
            CmpOp::Ge => ">=",
            CmpOp::Lt => "<",
            CmpOp::Gt => ">",
        };
        let expr = if self.dtype == DType::F32 {
            format!("select(0u,1u,x {cmp} y)")
        } else {
            format!("select(0u,1u,a[ai(i)] {cmp} b[bi(i)])")
        };
        self.elementwise(l, Some((r, rl)), &expr, DType::U8)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let expr = match (self.dtype, dtype) {
            (a, b) if a == b => "a[ai(i)]",
            (DType::F32, DType::U32) | (DType::F32, DType::U8) => "u32(x)",
            (DType::U32, DType::F32) | (DType::U8, DType::F32) => "bitcast<u32>(f32(a[ai(i)]))",
            // Both integer dtypes occupy a whole word here, so widening is a
            // copy; narrowing has to truncate to keep `as u8` semantics.
            (DType::U8, DType::U32) => "a[ai(i)]",
            (DType::U32, DType::U8) => "a[ai(i)] & 255u",
            _ => crate::bail!("Vulkan cast {:?} to {dtype:?} unsupported", self.dtype),
        };
        self.elementwise(l, None, expr, dtype)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        self.require_f32()?;
        let expression = match B::NAME {
            "exp" => "exp(x)",
            "log" => "log(x)",
            "sin" => "sin(x)",
            "cos" => "cos(x)",
            "abs" => "abs(x)",
            "neg" => "-x",
            "recip" => "1.0/x",
            "sqr" => "x*x",
            "sqrt" => "sqrt(x)",
            "relu" => "max(x,0.0)",
            "silu" => "x/(1.0+exp(-x))",
            "tanh" => "tanh(x)",
            "ceil" => "ceil(x)",
            "floor" => "floor(x)",
            "round" => "round(x)",
            "sign" => "sign(x)",
            _ => crate::bail!("Vulkan unary {} unsupported", B::NAME),
        };
        self.elementwise(l, None, &format!("bitcast<u32>({expression})"), DType::F32)
    }

    fn binary_impl<B: BinaryOpT>(&self, r: &Self, l: &Layout, rl: &Layout) -> Result<Self> {
        self.require_f32()?;
        r.require_f32()?;
        let expression = match B::NAME {
            "add" => "x+y",
            "sub" => "x-y",
            "mul" => "x*y",
            "div" => "x/y",
            "maximum" => "max(x,y)",
            "minimum" => "min(x,y)",
            _ => crate::bail!("Vulkan binary {} unsupported", B::NAME),
        };
        self.elementwise(
            l,
            Some((r, rl)),
            &format!("bitcast<u32>({expression})"),
            DType::F32,
        )
    }

    fn where_cond(&self, l: &Layout, t: &Self, tl: &Layout, f: &Self, fl: &Layout) -> Result<Self> {
        if !matches!(self.dtype, DType::U8 | DType::U32) {
            crate::bail!("Vulkan where condition must be integer");
        }
        let n = l.shape().elem_count();
        let output = self.device.alloc(n, t.dtype)?;
        let code=format!("@group(0) @binding(0) var<storage,read> c:array<u32>; @group(0) @binding(1) var<storage,read> t:array<u32>; @group(0) @binding(2) var<storage,read> f:array<u32>; @group(0) @binding(3) var<storage,read_write> out:array<u32>; {} {} {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i<{n}u{{out[i]=select(f[fi(i)],t[ti(i)],c[ci(i)]!=0u);}}}}",index_fn("ci",l),index_fn("ti",tl),index_fn("fi",fl));
        self.device.dispatch(
            code,
            &[&self.buffer, &t.buffer, &f.buffer, &output.buffer],
            n.div_ceil(64) as u32,
        )?;
        Ok(output)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv1d not implemented")
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv_transpose1d not implemented")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv2d not implemented")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv_transpose2d not implemented")
    }

    fn index_select(&self, ids: &Self, l: &Layout, il: &Layout, dim: usize) -> Result<Self> {
        if ids.dtype != DType::U32 {
            crate::bail!("Vulkan indices must be u32");
        }
        let count = il.shape().elem_count();
        let width = l.dims()[dim];
        let trailing: usize = l.dims()[dim + 1..].iter().product();
        let leading: usize = l.dims()[..dim].iter().product();
        let n = leading * count * trailing;
        let output = self.device.alloc(n, self.dtype)?;
        let error = self.device.alloc(1, DType::U32)?;
        let code=format!("@group(0) @binding(0) var<storage,read> a:array<u32>; @group(0) @binding(1) var<storage,read> ids:array<u32>; @group(0) @binding(2) var<storage,read_write> out:array<u32>; @group(0) @binding(3) var<storage,read_write> error:atomic<u32>; {} {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i>={n}u{{return;}}let selected=ids[ii((i/{trailing}u)%{count}u)];if selected>={width}u{{atomicStore(&error,1u);return;}}let source=(i/({count}u*{trailing}u))*{width}u*{trailing}u+selected*{trailing}u+i%{trailing}u;out[i]=a[ai(source)];}}",index_fn("ai",l),index_fn("ii",il));
        self.device.dispatch(
            code,
            &[&self.buffer, &ids.buffer, &output.buffer, &error.buffer],
            n.div_ceil(64) as u32,
        )?;
        if self.device.read_bytes(&error.buffer, 4)? != [0, 0, 0, 0] {
            crate::bail!("Vulkan index_select index out of bounds");
        }
        Ok(output)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("Vulkan gather not implemented")
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
        crate::bail!("Vulkan scatter_set not implemented")
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
        crate::bail!("Vulkan scatter_add_set not implemented")
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
        crate::bail!("Vulkan index_add not implemented")
    }

    fn matmul(
        &self,
        r: &Self,
        (batch, m, n, k): (usize, usize, usize, usize),
        l: &Layout,
        rl: &Layout,
    ) -> Result<Self> {
        self.require_f32()?;
        r.require_f32()?;
        let count = batch * m * n;
        let output = self.device.alloc(count, DType::F32)?;
        let code=format!("@group(0) @binding(0) var<storage,read> a:array<f32>; @group(0) @binding(1) var<storage,read> b:array<f32>; @group(0) @binding(2) var<storage,read_write> out:array<f32>; {} {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i>={count}u{{return;}} let batch=i/({m}u*{n}u);let row=(i/{n}u)%{m}u;let col=i%{n}u;var sum=0.0;for(var j=0u;j<{k}u;j++){{sum+=a[ai((batch*{m}u+row)*{k}u+j)]*b[bi((batch*{k}u+j)*{n}u+col)];}}out[i]=sum;}}",index_fn("ai",l),index_fn("bi",rl));
        self.device.dispatch(
            code,
            &[&self.buffer, &r.buffer, &output.buffer],
            count.div_ceil(64) as u32,
        )?;
        Ok(output)
    }

    fn copy_strided_src(&self, dst: &mut Self, offset: usize, l: &Layout) -> Result<()> {
        let n = l.shape().elem_count();
        if offset + n > dst.len {
            crate::bail!("Vulkan copy destination out of bounds");
        }
        let code=format!("@group(0) @binding(0) var<storage,read> a:array<u32>; @group(0) @binding(1) var<storage,read_write> out:array<u32>; {} @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i<{n}u{{out[i+{offset}u]=a[ai(i)];}}}}",index_fn("ai",l));
        self.device
            .dispatch(code, &[&self.buffer, &dst.buffer], n.div_ceil(64) as u32)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        ss: usize,
        ds: usize,
        so: usize,
        do_: usize,
    ) -> Result<()> {
        let n = d1 * d2;
        if d1 > 0
            && d2 > 0
            && (so + (d1 - 1) * ss + d2 > self.len || do_ + (d1 - 1) * ds + d2 > dst.len)
        {
            crate::bail!("Vulkan copy2d bounds");
        }
        let code=format!("@group(0) @binding(0) var<storage,read> a:array<u32>; @group(0) @binding(1) var<storage,read_write> out:array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){{let i=dispatch_index(id);if i<{n}u{{out[{do_}u+(i/{d2}u)*{ds}u+i%{d2}u]=a[{so}u+(i/{d2}u)*{ss}u+i%{d2}u];}}}}");
        self.device
            .dispatch(code, &[&self.buffer, &dst.buffer], n.div_ceil(64) as u32)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("Vulkan avg_pool2d not implemented")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("Vulkan max_pool2d not implemented")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("Vulkan upsample_nearest1d not implemented")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        crate::bail!("Vulkan upsample_nearest2d not implemented")
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
        crate::bail!("Vulkan upsample_bilinear2d not implemented")
    }
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;
    fn new(ordinal: usize) -> Result<Self> {
        let encoder_batching = std::env::var("AKH_VULKAN_BATCH_ENCODER").as_deref() != Ok("0");
        let pass_batching = match std::env::var("AKH_VULKAN_BATCH_PASS").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            // Disabling encoders alone remains a valid reference escape hatch.
            _ => encoder_batching,
        };
        if pass_batching && !encoder_batching {
            crate::bail!("Vulkan compute-pass batching requires shared encoders");
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .nth(ordinal)
            .ok_or_else(|| vk_error(format!("hardware adapter {ordinal} unavailable")))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(vk_error)?;
        Ok(Self(Arc::new(Context {
            device,
            queue,
            ordinal,
            info,
            // Bound application-held specializations, not opaque driver allocations.
            pipelines: Mutex::new(PipelineCache::new(2048, 16 * 1024 * 1024)),
            pending: Mutex::new(Vec::new()),
            submission_batch: if std::env::var("AKH_VULKAN_BATCH_SUBMIT").as_deref() != Ok("0") {
                64
            } else {
                1
            },
            encoder_batching,
            pass_batching,
            encoder: Mutex::new(EncoderBatch::default()),
        })))
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        crate::bail!("Vulkan set_seed not implemented")
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("Vulkan get_current_seed not implemented")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.0.ordinal,
        }
    }

    fn same_device(&self, r: &Self) -> bool {
        Arc::ptr_eq(&self.0, &r.0)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(shape.elem_count(), dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(shape.elem_count(), dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(data))
    }

    fn storage_from_cpu_storage(&self, data: &CpuStorage) -> Result<Self::Storage> {
        let words: Vec<u32> = match data {
            CpuStorage::F32(v) => v.iter().map(|f| f.to_bits()).collect(),
            CpuStorage::U32(v) => v.clone(),
            CpuStorage::U8(v) => v.iter().map(|&x| x as u32).collect(),
            _ => crate::bail!("Vulkan upload dtype {:?} unsupported", data.dtype()),
        };
        let buffer = self.upload_bytes(bytemuck::cast_slice(&words))?;
        Ok(VulkanStorage {
            buffer,
            device: self.clone(),
            dtype: data.dtype(),
            len: words.len(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, data: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&data)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("Vulkan rand_uniform not implemented")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("Vulkan rand_normal not implemented")
    }

    fn synchronize(&self) -> Result<()> {
        self.flush()?;
        self.0
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(vk_error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "exclusive Vulkan device"]
    fn cast_u8_u32_roundtrip() -> Result<()> {
        let device = crate::Device::new_vulkan(0)?;
        // Spans the byte range including both endpoints. Both integer dtypes
        // occupy a full word in Vulkan storage, so widening must be a copy and
        // narrowing must truncate the way `as u8` does.
        let bytes: Vec<u8> = vec![0, 1, 7, 15, 128, 254, 255, 3];
        let as_u8 = crate::Tensor::from_vec(bytes.clone(), (8,), &device)?;
        assert_eq!(as_u8.dtype(), DType::U8);

        let widened = as_u8.to_dtype(DType::U32)?;
        assert_eq!(widened.dtype(), DType::U32);
        assert_eq!(
            widened.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            bytes.iter().map(|b| f32::from(*b)).collect::<Vec<f32>>()
        );

        assert_eq!(widened.to_dtype(DType::U8)?.to_vec1::<u8>()?, bytes);

        // Truncation, not saturation or a silent widening, above the byte range.
        let wide = crate::Tensor::from_vec(vec![256u32, 511, 4097], (3,), &device)?;
        assert_eq!(wide.to_dtype(DType::U8)?.to_vec1::<u8>()?, vec![0u8, 255, 1]);
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Vulkan device"]
    fn pipeline_eviction_preserves_pending_commands() -> Result<()> {
        let device = crate::Device::new_vulkan(0)?;
        let crate::Device::Vulkan(vulkan) = &device else { unreachable!() };
        *vulkan.0.pipelines.lock().unwrap() = PipelineCache::new(2, 1024 * 1024);
        let mut value = crate::Tensor::new(&[0.0f32], &device)?;
        // Different affine constants generate different pipelines. Keep work
        // pending across eviction, including across submission boundaries.
        for increment in 1..=130 {
            value = value.affine(1.0, increment as f64)?;
            assert!(vulkan.0.pipelines.lock().unwrap().len() <= 2);
        }
        assert_eq!(value.to_vec1::<f32>()?, vec![8515.0]);
        // Recompile an evicted specialization and verify it still executes.
        assert_eq!(value.affine(1.0, 1.0)?.to_vec1::<f32>()?, vec![8516.0]);
        vulkan.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Vulkan device; set AKH_VULKAN_BATCH_PASS=1"]
    fn shared_compute_pass_retains_commands_until_copy_or_fence() -> Result<()> {
        let device = crate::Device::new_vulkan(0)?;
        let crate::Device::Vulkan(vulkan) = &device else { unreachable!() };
        assert!(vulkan.0.encoder_batching);
        let input = crate::Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
        let mut value = input.affine(2.0, 0.0)?;
        assert!(vulkan.0.encoder.lock().unwrap().pass.is_some(),
            "candidate must keep a pass open between dispatches");
        let snapshot = value.clone();
        value = value.copy()?;
        assert!(vulkan.0.encoder.lock().unwrap().pass.is_none(),
            "buffer copies must end the active compute pass first");
        for _ in 0..65 { value = value.affine(1.0, 1.0)?; }
        assert_eq!(value.to_vec1::<f32>()?, vec![67.0,69.0,71.0,73.0]);
        assert_eq!(snapshot.to_vec1::<f32>()?, vec![2.0,4.0,6.0,8.0]);
        vulkan.synchronize()?;
        assert!(vulkan.0.encoder.lock().unwrap().pass.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Vulkan device; set AKH_VULKAN_BATCH_PASS=1"]
    fn shared_compute_pass_deferred_failure_never_exposes_uncomputed_tensors() -> Result<()> {
        let device = crate::Device::new_vulkan(0)?;
        let crate::Device::Vulkan(vulkan) = &device else { unreachable!() };
        assert!(vulkan.0.pass_batching);
        let pending = crate::Tensor::new(&[1.0f32], &device)?.affine(2.0, 0.0)?;
        // wgpu records dispatch commands and validates their limits at pass end.
        // Exercise that deferred failure, not just an immediately-invalid copy.
        let invalid_groups = vulkan.0.device.limits().max_compute_workgroups_per_dimension + 1;
        vulkan.0.encoder.lock().unwrap().pass.as_mut().unwrap()
            .dispatch_workgroups(invalid_groups, 1, 1);
        assert!(vulkan.synchronize().is_err());
        assert!(pending.to_vec1::<f32>().is_err());
        let fresh = crate::Device::new_vulkan(0)?;
        assert_eq!(crate::Tensor::new(&[3.0f32], &fresh)?.affine(2.0,0.0)?.to_vec1::<f32>()?,vec![6.0]);
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Vulkan device; set AKH_VULKAN_BATCH_ENCODER=1"]
    fn shared_encoder_failure_never_exposes_uncomputed_tensors() -> Result<()> {
        let device = crate::Device::new_vulkan(0)?;
        let crate::Device::Vulkan(vulkan) = &device else { unreachable!() };
        assert!(vulkan.0.encoder_batching);
        assert!(vulkan.0.submission_batch > 1);
        let input = crate::Tensor::new(&[1.0f32], &device)?;
        let pending = input.affine(2.0, 0.0)?;
        let scratch = vulkan.alloc(1, DType::F32)?;
        // Deliberately invalidate the shared encoder after valid queued work.
        // The normal public operations validate their bindings before encoding.
        let invalid = vulkan.record_shared(true, |encoder| {
            encoder.copy_buffer_to_buffer(&scratch.buffer, 4, &scratch.buffer, 0, 4);
        });
        assert!(invalid.is_err());
        assert!(vulkan.synchronize().is_err());
        assert!(pending.to_vec1::<f32>().is_err(), "must not expose uncomputed output as zeros");
        // Failure is confined to this device instance, not a global switch.
        let fresh = crate::Device::new_vulkan(0)?;
        assert_eq!(crate::Tensor::new(&[1.0f32], &fresh)?.affine(2.0,0.0)?.to_vec1::<f32>()?, vec![2.0]);
        Ok(())
    }
}
