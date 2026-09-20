//! Public causal-convolution gate: independent f64 oracle, chunk history and aliases.
use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn reference(
    input: &Tensor,
    kernel: &Tensor,
    history: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, length, channels) = input.dims3()?;
    let width = kernel.dim(1)?;
    let zeros = Tensor::zeros((batch, width - 1, channels), DType::F32, input.device())?;
    let extended = Tensor::cat(&[history.unwrap_or(&zeros), input], 1)?;
    let mut output = None;
    for tap in 0..width {
        let term = extended
            .narrow(1, tap, length)?
            .broadcast_mul(&kernel.narrow(1, tap, 1)?.squeeze(1)?)?;
        output = Some(match output {
            None => term,
            Some(value) => (value + term)?,
        });
    }
    Ok((output.unwrap(), extended.narrow(1, length, width - 1)?))
}

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let candidate = candle_core::vulkan_backend::causal_conv_fused;
    let mut maximum = 0f64;
    for (batch, length, channels, width) in [
        (1, 1, 1, 1),
        (2, 3, 7, 4),
        (1, 19, 129, 4),
        (1, 1, 10240, 4),
        (1, 128, 10240, 4),
        (1, 2, 65, 7),
        (1, 1, 4_194_305, 1),
    ] {
        for with_history in [false, true] {
            // Slice every channel/tap axis to exercise real non-contiguous views.
            let raw: Vec<_> = (0..batch * length * channels * 2)
                .map(|i| ((i * 13 % 79) as f32 - 39.0) / 31.0)
                .collect();
            let input = Tensor::from_vec(raw, (batch, length, channels * 2), &device)?
                .narrow(2, 1, channels)?;
            let raw: Vec<_> = (0..channels * width * 2)
                .map(|i| ((i * 7 % 31) as f32 - 15.0) / 23.0)
                .collect();
            let kernel =
                Tensor::from_vec(raw, (channels, width * 2), &device)?.narrow(1, 1, width)?;
            let raw: Vec<_> = (0..batch * (width - 1) * channels * 2)
                .map(|i| ((i * 3 % 29) as f32 - 14.0) / 17.0)
                .collect();
            // Empty history has no elements to offset; use its canonical empty
            // layout rather than a nonzero-offset empty readback view.
            let history = if width == 1 {
                Tensor::zeros((batch, 0, channels), DType::F32, &device)?
            } else {
                Tensor::from_vec(raw, (batch, width - 1, channels * 2), &device)?
                    .narrow(2, 1, channels)?
            };
            let input_values = input.flatten_all()?.to_vec1::<f32>()?;
            let kernel_values = kernel.flatten_all()?.to_vec1::<f32>()?;
            let history_values = history.flatten_all()?.to_vec1::<f32>()?;
            let prior = with_history.then_some(&history);
            let (expected, expected_history) = reference(&input, &kernel, prior)?;
            let (actual, next) = candidate(&input, &kernel, prior)?;
            let values = actual.flatten_all()?.to_vec1::<f32>()?;
            for b in 0..batch {
                for token in 0..length {
                    for channel in 0..channels {
                        let mut oracle = 0f64;
                        for tap in 0..width {
                            let position = token + tap;
                            let value = if position < width - 1 {
                                if with_history {
                                    history_values
                                        [(b * (width - 1) + position) * channels + channel]
                                } else {
                                    0.0
                                }
                            } else {
                                input_values
                                    [(b * length + position - (width - 1)) * channels + channel]
                            };
                            oracle += value as f64 * kernel_values[channel * width + tap] as f64;
                        }
                        let error = (values[(b * length + token) * channels + channel] as f64
                            - oracle)
                            .abs();
                        maximum = maximum.max(error);
                        assert!(error < 2e-6, "oracle error {error}");
                    }
                }
            }
            for (a, b) in values.iter().zip(expected.flatten_all()?.to_vec1::<f32>()?) {
                assert!((*a - b).abs() < 2e-6);
            }
            assert_eq!(
                next.flatten_all()?.to_vec1::<f32>()?,
                expected_history.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(input.flatten_all()?.to_vec1::<f32>()?, input_values);
            assert_eq!(history.flatten_all()?.to_vec1::<f32>()?, history_values);
        }
    }
    // Entire prefill versus sequential appends, with retained old snapshots.
    let input = Tensor::arange(0u32, 3 * 7 * 5, &device)?
        .to_dtype(DType::F32)?
        .reshape((3, 7, 5))?;
    let kernel = Tensor::ones((5, 4), DType::F32, &device)?;
    let (whole, whole_history) = candidate(&input, &kernel, None)?;
    let mut history = None;
    let mut outputs = Vec::new();
    for token in 0..7 {
        let snapshot = history.clone();
        let before = snapshot
            .as_ref()
            .map(|t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let (output, next) = candidate(&input.narrow(1, token, 1)?, &kernel, history.as_ref())?;
        if let (Some(snapshot), Some(before)) = (snapshot, before) {
            assert_eq!(snapshot.flatten_all()?.to_vec1::<f32>()?, before);
        }
        outputs.push(output);
        history = Some(next);
    }
    assert_eq!(
        Tensor::cat(&outputs, 1)?.flatten_all()?.to_vec1::<f32>()?,
        whole.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        history.unwrap().flatten_all()?.to_vec1::<f32>()?,
        whole_history.flatten_all()?.to_vec1::<f32>()?
    );
    let x = Tensor::ones((1, 1, 10240), DType::F32, &device)?;
    let k = Tensor::ones((10240, 4), DType::F32, &device)?;
    assert!(candidate(&x, &Tensor::ones((1, 4), DType::F32, &device)?, None).is_err());
    assert!(
        candidate(
            &x,
            &k,
            Some(&Tensor::zeros((1, 2, 10240), DType::F32, &device)?)
        )
        .is_err()
    );
    assert!(candidate(&x.to_dtype(DType::U32)?, &k, None).is_err());
    assert!(candidate(&x, &k.to_device(&Device::Cpu)?, None).is_err());
    assert!(
        candidate(
            &Tensor::zeros((1, 0, 10240), DType::F32, &device)?,
            &k,
            None
        )
        .is_err()
    );
    assert!(candidate(&x, &Tensor::zeros((10240, 0), DType::F32, &device)?, None).is_err());
    assert!(
        candidate(
            &x,
            &k,
            Some(&Tensor::zeros((1, 3, 10240), DType::U32, &device)?)
        )
        .is_err()
    );
    for fused in [false, true, false, true] {
        let mut samples = Vec::new();
        for round in 0..22 {
            device.synchronize()?;
            let start = Instant::now();
            let pair = if fused {
                candidate(&x, &k, None)?
            } else {
                reference(&x, &k, None)?
            };
            device.synchronize()?;
            std::hint::black_box(pair);
            if round >= 2 {
                samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "convolution fused={fused} median_us={:.3}",
            (samples[9] + samples[10]) / 2.0
        );
    }
    println!("PASS causal convolution, history, snapshots and errors; max_abs={maximum:.9}");
    Ok(())
}
