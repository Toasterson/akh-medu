//! Exclusive-device numerical and warm-timing check for decode-shaped matmuls.
#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{Device, Tensor};
    use std::time::Instant;
    let gpu = Device::new_vulkan(0)?;
    for &(batch, n, k, transposed) in &[
        (2, 37, 19, false),
        (16, 773, 512, true),
        (16, 512, 773, false),
        (16, 256, 1025, false),
    ] {
        let av: Vec<f32> = (0..batch * k)
            .map(|i| ((i * 13 % 97) as f32 - 48.) / 97.)
            .collect();
        let bv: Vec<f32> = (0..batch * n * k)
            .map(|i| ((i * 17 % 113) as f32 - 56.) / 113.)
            .collect();
        let a = Tensor::from_vec(av, (batch, 1, k), &Device::Cpu)?;
        let b = if transposed {
            Tensor::from_vec(bv, (batch, n, k), &Device::Cpu)?.transpose(1, 2)?
        } else {
            Tensor::from_vec(bv, (batch, k, n), &Device::Cpu)?
        };
        let expected = a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
        let ga = a.to_device(&gpu)?;
        // Preserve the right-hand operand's transpose layout after upload.
        let gb = if transposed {
            b.transpose(1, 2)?.to_device(&gpu)?.transpose(1, 2)?
        } else {
            b.to_device(&gpu)?
        };
        let observed = ga.matmul(&gb)?.flatten_all()?.to_vec1::<f32>()?;
        let max_error = expected
            .iter()
            .zip(&observed)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if observed.iter().any(|x| !x.is_finite()) || max_error > 0.0002 {
            return Err(format!("parity failure {batch}/{n}/{k}: {max_error}").into());
        }
        let start = Instant::now();
        for _ in 0..20 {
            let _ = ga.matmul(&gb)?.flatten_all()?.to_vec1::<f32>()?;
        }
        println!(
            "{}",
            serde_json::json!({"batch":batch,"n":n,"k":k,"transposed":transposed,"max_abs_error":max_error,"mean_seconds":start.elapsed().as_secs_f64()/20.})
        );
    }

    for &(kv, d) in &[(8usize, 256usize), (1, 512)] {
        let values = |n| {
            (0..n)
                .map(|i| ((i * 13 % 97) as f32 - 48.) / 97.)
                .collect::<Vec<_>>()
        };
        let old = Tensor::from_vec(values(kv * 772 * d), (1, kv, 772, d), &gpu)?;
        let new = Tensor::from_vec(values(kv * d), (1, kv, 1, d), &gpu)?;
        let q = Tensor::from_vec(values(16 * d), (1, 16, 1, d), &gpu)?;
        let sync = |name: &str,
                    f: &mut dyn FnMut() -> candle_core::Result<Tensor>|
         -> candle_core::Result<Tensor> {
            let mut result = f()?;
            gpu.synchronize()?;
            let start = Instant::now();
            for _ in 0..10 {
                result = f()?;
                gpu.synchronize()?;
            }
            println!(
                "{}",
                serde_json::json!({"stage":name,"kv_heads":kv,"head_dim":d,"mean_seconds":start.elapsed().as_secs_f64()/10.})
            );
            Ok(result)
        };
        let k = sync("kv_append", &mut || Tensor::cat(&[&old, &new], 2))?;
        let k = sync("repeat_kv", &mut || {
            Tensor::cat(&vec![&k; 16 / kv], 2)?.reshape((1, 16, 773, d))
        })?;
        let w = sync("qk", &mut || q.matmul(&k.transpose(2, 3)?))?;
        let w = sync("softmax", &mut || {
            let max = w.max_keepdim(candle_core::D::Minus1)?;
            let exp = w.broadcast_sub(&max)?.exp()?;
            exp.broadcast_div(&exp.sum_keepdim(candle_core::D::Minus1)?)
        })?;
        let _ = sync("wv", &mut || w.matmul(&k))?;
    }
    Ok(())
}
#[cfg(not(feature = "vulkan"))]
fn main() {
    panic!("requires vulkan")
}
