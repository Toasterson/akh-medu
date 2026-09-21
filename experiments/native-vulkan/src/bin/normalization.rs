//! Numerical and latency gate for the real Vulkan weighted RMS interface.
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn reference(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let width = x.dim(candle_core::D::Minus1)?;
    let mean_square = (x.sqr()?.sum_keepdim(candle_core::D::Minus1)? / width as f64)?;
    x.broadcast_div(&(mean_square + eps as f64)?.sqrt()?)?.broadcast_mul(weight)
}

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let candidate = std::env::args().any(|arg| arg == "--candidate");
    let norm = |x: &Tensor, weight: &Tensor, eps: f32| {
        if candidate { candle_core::vulkan_backend::rms_norm_fused(x, weight, eps) }
        else { reference(x, weight, eps) }
    };
    for (rows, width) in [(1,1), (3,127), (3,128), (3,255), (3,256), (3,257), (1,5120), (19,5120), (3,17408), (65537,1)] {
        let values: Vec<f32> = (0..rows*width*2+5)
            .map(|i| ((i*37%997) as f32-498.0)/251.0).collect();
        let weights: Vec<f32> = (0..width+7)
            .map(|i| ((i*19%251) as f32-125.0)/97.0).collect();
        let input = Tensor::new(values.as_slice(), &device)?
            .narrow(0,5,rows*width*2)?.reshape((rows,width,2))?
            .narrow(2,1,1)?.squeeze(2)?;
        let weight = Tensor::new(weights.as_slice(), &device)?.narrow(0,7,width)?;
        for input in [input.clone(), input.contiguous()?] {
            let actual = norm(&input,&weight,1e-6)?.to_vec2::<f32>()?;
            assert_eq!(input.flatten_all()?.to_vec1::<f32>()?,
                (0..rows*width).map(|i| values[5+2*i+1]).collect::<Vec<_>>(),
                "normalization must not mutate its input or checkpoint aliases");
            let mut max_error = 0.0f64;
            for row in 0..rows {
                let mean: f64 = (0..width).map(|col| {
                    let x = values[5+(row*width+col)*2+1] as f64;
                    x*x
                }).sum::<f64>() / width as f64;
                for col in 0..width {
                    let expected = values[5+(row*width+col)*2+1] as f64
                        / (mean+1e-6f32 as f64).sqrt() * weights[col+7] as f64;
                    let error = (actual[row][col] as f64-expected).abs();
                    assert!(actual[row][col].is_finite() && error <= 2e-5+expected.abs()*2e-5,
                        "RMS {rows}x{width} [{row},{col}]: {} != {expected}", actual[row][col]);
                    max_error = max_error.max(error);
                }
            }
            println!("PASS RMS candidate={candidate} {rows}x{width} contiguous={} max_abs={max_error:.8}", input.is_contiguous());
        }
    }
    let x = Tensor::from_vec(vec![0.5f32;5120], (1,5120), &device)?;
    let w = Tensor::from_vec(vec![1.5f32;5120], 5120, &device)?;
    if candidate {
        assert!(norm(&x, &w.narrow(0,0,5119)?, 1e-6).is_err());
        assert!(norm(&x.to_dtype(candle_core::DType::U32)?, &w, 1e-6).is_err());
        assert!(norm(&x, &w.to_device(&Device::Cpu)?, 1e-6).is_err());
        assert!(norm(&x, &w, f32::NAN).is_err());
        assert!(norm(&x, &w, -1.0).is_err());
    }
    let mut times = Vec::new();
    for round in 0..18 {
        device.synchronize()?;
        let start = Instant::now();
        let result = norm(&x,&w,1e-6)?;
        device.synchronize()?;
        std::hint::black_box(result);
        if round >= 3 { times.push(start.elapsed().as_secs_f64()*1e6); }
    }
    times.sort_by(f64::total_cmp);
    println!("RMS candidate={candidate} 1x5120 median_us={:.3}", times[7]);
    Ok(())
}
