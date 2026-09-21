//! Separate resource-creation cost from queued arithmetic and its final fence.
use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    for width in [4, 5120, 48*128*128] {
        for arithmetic in [false, true] {
            let mut samples = Vec::new();
            for round in 0..12 {
                let mut value = Tensor::ones(width, DType::F32, &device)?;
                device.synchronize()?;
                let start = Instant::now();
                for _ in 0..128 {
                    if arithmetic { value = value.affine(1.0,1.0)?; }
                    else { std::hint::black_box(Tensor::zeros(width,DType::F32,&device)?); }
                }
                let submitted = start.elapsed().as_secs_f64()*1e3;
                device.synchronize()?;
                let finished = start.elapsed().as_secs_f64()*1e3;
                if arithmetic {
                    let values = value.to_vec1::<f32>()?;
                    assert!(values.iter().all(|&v| v == 129.0));
                }
                if round >= 2 { samples.push((submitted,finished)); }
            }
            samples.sort_by(|a,b| a.1.total_cmp(&b.1));
            println!("allocation width={width} arithmetic={arithmetic} host_ms={:.3} fenced_ms={:.3}",
                (samples[4].0+samples[5].0)/2.0,(samples[4].1+samples[5].1)/2.0);
        }
    }
    // A completed or still-queued producer must never alter retained aliases.
    let mut value = Tensor::new(&[1.0f32,2.0,3.0,4.0],&device)?;
    let snapshot = value.clone();
    for _ in 0..129 { value = value.affine(1.0,1.0)?; }
    assert_eq!(value.to_vec1::<f32>()?,vec![130.0,131.0,132.0,133.0]);
    assert_eq!(snapshot.to_vec1::<f32>()?,vec![1.0,2.0,3.0,4.0]);
    drop(value);
    assert_eq!(Tensor::zeros(4,DType::F32,&device)?.to_vec1::<f32>()?,vec![0.0;4]);
    // Error flags are initialized state, not uninitialized output scratch.
    let input = Tensor::new(&[10.0f32,20.0],&device)?;
    for _ in 0..16 {
        let invalid = Tensor::new(&[2u32],&device)?;
        assert!(input.index_select(&invalid,0).is_err());
        let valid = Tensor::new(&[1u32],&device)?;
        assert_eq!(input.index_select(&valid,0)?.to_vec1::<f32>()?,vec![20.0]);
    }
    println!("PASS retained aliases, zero initialization and error recovery");
    Ok(())
}
