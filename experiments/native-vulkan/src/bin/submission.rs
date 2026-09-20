//! Public-tensor ordering and submission-cost probe. No model or CPU fallback.
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn chain(device: &Device, steps: usize, copies: bool) -> Result<Tensor> {
    let mut value = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0], device)?;
    for step in 0..steps {
        value = value.affine(1.0, 1.0)?;
        if copies && step % 7 == 0 {
            value = value.copy()?;
        }
        if copies && step % 19 == 0 {
            device.synchronize()?;
        }
    }
    Ok(value)
}

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    // Span proposed batching boundaries, leave incomplete batches, and mix
    // buffer copies with compute. The independent oracle uses exact integers.
    for steps in [1, 2, 63, 64, 65, 127, 129] {
        for copies in [false, true] {
            let result = chain(&device, steps, copies)?.to_vec1::<f32>()?;
            assert_eq!(result, (0..4).map(|i| (i + steps) as f32).collect::<Vec<_>>());
        }
    }
    // Two callers sharing one device must retain independent values even if
    // their command streams interleave.
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2).map(|_| {
            let device = &device;
            scope.spawn(move || {
                assert_eq!(chain(device, 129, true).unwrap().to_vec1::<f32>().unwrap(),
                    vec![129.0, 130.0, 131.0, 132.0]);
            })
        }).collect();
        for handle in handles { handle.join().unwrap(); }
    });
    let mut times = Vec::new();
    for round in 0..12 {
        device.synchronize()?;
        let start = Instant::now();
        let result = chain(&device, 256, false)?.to_vec1::<f32>()?;
        assert_eq!(result, vec![256.0, 257.0, 258.0, 259.0]);
        if round >= 2 { times.push(start.elapsed().as_secs_f64() * 1e3); }
    }
    times.sort_by(f64::total_cmp);
    println!("PASS submission ordering; 256 dependent operations median_ms={:.3}",
        (times[4] + times[5]) / 2.0);
    Ok(())
}
