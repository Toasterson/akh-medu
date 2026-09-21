use candle_core::{Device, Module, Result, Tensor};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};

fn main() -> Result<()> {
    assert_eq!(std::env::var("AKH_VULKAN_GEMV_BLOCKS").as_deref(), Ok("1"),
        "explicitly enable the candidate kernel for this acceptance probe");
    let gpu = Device::new_vulkan(0)?;
    for format in [GgmlDType::Q4K, GgmlDType::Q6K] {
        for (rows, cols) in [(1,256), (7,512), (7,768), (13,1280), (7,5120), (7,17408)] {
            let weights = Tensor::from_vec((0..rows*cols)
                .map(|i| ((i*37%991) as f32-495.)/173.).collect(), (rows,cols), &Device::Cpu)?;
            let exact = QTensor::quantize(&weights, format)?.dequantize(&Device::Cpu)?.to_vec2::<f32>()?;
            let op = QMatMul::from_qtensor(QTensor::quantize_onto(&weights,format,&gpu)?)?;
            let input: Vec<_> = (0..2*cols).map(|i| ((i*19%251) as f32-125.)/127.).collect();
            let storage = Tensor::from_vec(input.clone(), (cols,2), &gpu)?;
            // Construct the strided, nonzero-offset view on GPU so a device
            // transfer cannot accidentally materialize away the tested layout.
            let view = storage.t()?.narrow(0,1,1)?;
            for x in [view.clone(), view.contiguous()?] {
                let result = op.forward(&x)?.to_vec2::<f32>()?;
                let mut max_error = 0.0f64;
                for (row, result) in result[0].iter().enumerate() {
                    let expected: f64 = exact[row].iter().enumerate()
                        .map(|(col,w)| *w as f64*input[2*col+1] as f64).sum();
                    let error = (*result as f64-expected).abs();
                    assert!(error <= 0.002+0.00002*expected.abs(),
                        "{format:?} {rows}x{cols} row={row}: {result} != {expected}");
                    max_error = max_error.max(error);
                }
                println!("PASS {format:?} {rows}x{cols} contiguous={} offset={} max_abs={max_error:.8}",
                    x.is_contiguous(), x.layout().start_offset());
            }
        }
    }
    Ok(())
}
