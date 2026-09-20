use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, GgmlType};
use std::{error::Error, time::Duration};
use wgpu::util::DeviceExt;

fn main() -> Result<(), Box<dyn Error>> {
    // This mode parses and validates WGSL without creating any GPU instance.
    if std::env::args().any(|arg| arg == "--validate-shader") {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("q4k.wgsl"))?;
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)?;
        println!("Q4_K/Q6_K WGSL parsed and validated; no GPU initialized");
        return Ok(());
    }
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    let info = adapter.get_info();
    if info.backend != wgpu::Backend::Vulkan || info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("Refusing non-hardware Vulkan adapter: {info:?}").into());
    }
    println!("adapter={info:?}");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let shader = device.create_shader_module(wgpu::include_wgsl!("q4k.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("exact Q4_K/Q6_K GEMV"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    for (dtype, rows, cols) in [4u32, 6].into_iter().flat_map(|dtype| {
        [(1usize, 256usize), (7, 512), (64, 2048), (127, 4096)]
            .map(|(rows, cols)| (dtype, rows, cols))
    }) {
        let source: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 991) as f32 - 495.) / 173. + (i / 32 % 8) as f32 * 0.13)
            .collect();
        let mut decoded = vec![0f32; rows * cols];
        macro_rules! quantize {
            ($block:ty, $bytes:expr) => {{
                let mut quant = vec![<$block>::zeros(); rows * cols / 256];
                <$block>::from_float(&source, &mut quant);
                <$block>::to_float(&quant, &mut decoded);
                assert_eq!(std::mem::size_of::<$block>(), $bytes);
                // SAFETY: Both concrete Candle blocks are repr(C), initialized,
                // padding-free. Their byte sizes are statically asserted upstream.
                unsafe {
                    std::slice::from_raw_parts(quant.as_ptr().cast::<u8>(), quant.len() * $bytes)
                }
                .to_vec()
            }};
        }
        let mut packed = if dtype == 4 {
            quantize!(BlockQ4K, 144)
        } else {
            quantize!(BlockQ6K, 210)
        };
        let weight_bytes = packed.len();
        // Storage is a u32 array; pad only the final buffer, never between blocks.
        packed.resize(packed.len().next_multiple_of(4), 0);
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 19 % 251) as f32 - 125.) / 127.)
            .collect();
        let expected: Vec<f64> = decoded
            .chunks_exact(cols)
            .map(|row| row.iter().zip(&x).map(|(&w, &v)| w as f64 * v as f64).sum())
            .collect();
        let make_buffer = |label, bytes: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            })
        };
        let weights = make_buffer("packed quant weights", &packed, wgpu::BufferUsages::STORAGE);
        let input = make_buffer(
            "input",
            bytemuck::cast_slice(&x),
            wgpu::BufferUsages::STORAGE,
        );
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shape = make_buffer(
            "shape",
            bytemuck::cast_slice(&[rows as u32, cols as u32, dtype, 0]),
            wgpu::BufferUsages::UNIFORM,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bindings: Vec<_> = [&weights, &input, &output, &shape]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (rows * 4) as u64);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })?;
        rx.recv_timeout(Duration::from_secs(5))??;
        let mapped = readback.slice(..).get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&mapped);
        let mut max_abs = 0f64;
        for (row, (&actual, &reference)) in got.iter().zip(&expected).enumerate() {
            let error = (actual as f64 - reference).abs();
            max_abs = max_abs.max(error);
            let tolerance = 0.002 + 0.00002 * reference.abs();
            if !actual.is_finite() || error > tolerance {
                return Err(format!("{rows}x{cols} row {row}: {actual} != {reference}; error {error}, tolerance {tolerance}").into());
            }
        }
        println!(
            "PASS Q{dtype}_K {rows}x{cols}: max_abs={max_abs:.8}, packed_bytes={weight_bytes}"
        );
        drop(mapped);
        readback.unmap();
    }
    Ok(())
}
