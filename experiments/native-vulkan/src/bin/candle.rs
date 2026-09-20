use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

fn close(label: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert!(actual.device().is_vulkan());
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.shape(), expected.shape());
    let error = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        a.iter().all(|x| x.is_finite()) && error <= tolerance,
        "{label}: {error} > {tolerance}"
    );
    println!("PASS {label}: max_abs={error}");
    Ok(())
}

fn main() -> Result<()> {
    let vk = Device::new_vulkan(0)?;
    if let Device::Vulkan(device) = &vk {
        println!(
            "adapter={:?}, storage_binding_limit={}",
            device.adapter_info(),
            device.storage_binding_limit()
        );
    }
    let cpu = Tensor::from_vec(
        (0..24).map(|i| i as f32 * 0.1 - 0.7).collect(),
        (2, 3, 4),
        &Device::Cpu,
    )?;
    let gpu = cpu.to_device(&vk)?;
    close("upload", &gpu, &cpu, 0.)?;
    close(
        "strided",
        &gpu.transpose(1, 2)?.contiguous()?,
        &cpu.transpose(1, 2)?.contiguous()?,
        0.,
    )?;
    close(
        "strided copy",
        &gpu.transpose(1, 2)?.narrow(1, 1, 2)?.copy()?,
        &cpu.transpose(1, 2)?.narrow(1, 1, 2)?.copy()?,
        0.,
    )?;
    close(
        "narrow affine",
        &gpu.narrow(2, 1, 2)?.affine(1.7, 0.2)?,
        &cpu.narrow(2, 1, 2)?.affine(1.7, 0.2)?,
        1e-6,
    )?;
    close("silu", &gpu.silu()?, &cpu.silu()?, 1e-6)?;
    let bias = Tensor::new(&[0.2f32, 0.3, 0.4, 0.5], &Device::Cpu)?;
    close(
        "broadcast",
        &gpu.broadcast_add(&bias.to_device(&vk)?)?,
        &cpu.broadcast_add(&bias)?,
        1e-6,
    )?;
    close("sum", &gpu.sum_keepdim(2)?, &cpu.sum_keepdim(2)?, 1e-6)?;
    close("max", &gpu.max_keepdim(1)?, &cpu.max_keepdim(1)?, 1e-6)?;
    assert_eq!(
        gpu.argmax(2)?.to_vec2::<u32>()?,
        cpu.argmax(2)?.to_vec2::<u32>()?
    );
    println!("PASS argmax");
    close(
        "dense matmul",
        &gpu.matmul(&gpu.transpose(1, 2)?)?,
        &cpu.matmul(&cpu.transpose(1, 2)?)?,
        1e-5,
    )?;
    let ids = Tensor::new(&[2u32, 0], &Device::Cpu)?;
    close(
        "index_select",
        &gpu.index_select(&ids.to_device(&vk)?, 1)?,
        &cpu.index_select(&ids, 1)?,
        0.,
    )?;
    assert!(gpu.index_select(&Tensor::new(&[3u32], &vk)?, 1).is_err());
    close(
        "where",
        &gpu.ge(0.)?.where_cond(&gpu, &gpu.neg()?)?,
        &cpu.ge(0.)?.where_cond(&cpu, &cpu.neg()?)?,
        1e-6,
    )?;
    close(
        "zeros",
        &Tensor::zeros((3, 4), DType::F32, &vk)?,
        &Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
        0.,
    )?;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        let source = Tensor::from_vec(
            (0..7 * 512)
                .map(|i| ((i * 37 % 991) as f32 - 495.) / 173.)
                .collect(),
            (7, 512),
            &Device::Cpu,
        )?;
        let quant = QTensor::quantize_onto(&source, dtype, &vk)?;
        let reference = QTensor::quantize(&source, dtype)?.dequantize(&Device::Cpu)?;
        close("quant dequant", &quant.dequantize(&vk)?, &reference, 1e-6)?;
        let ids = Tensor::new(&[6u32, 0, 3], &Device::Cpu)?;
        close(
            "quant get_rows",
            &quant.get_rows(&ids.to_device(&vk)?)?,
            &reference.index_select(&ids, 0)?,
            1e-6,
        )?;
        assert!(quant.get_rows(&Tensor::new(&[7u32], &vk)?).is_err());
        let x = Tensor::from_vec(
            (0..3 * 512)
                .map(|i| ((i * 19 % 251) as f32 - 125.) / 127.)
                .collect(),
            (3, 512),
            &Device::Cpu,
        )?;
        close(
            &format!("{dtype:?} batched forward"),
            &QMatMul::from_qtensor(quant)?.forward(&x.to_device(&vk)?)?,
            &x.matmul(&reference.t()?)?,
            0.002,
        )?;
    }
    for dtype in [GgmlDType::Q4K,GgmlDType::Q6K] {
        let source=Tensor::from_vec((0..9*512).map(|i|((i*23%331)as f32-165.)/165.).collect(),(9,512),&Device::Cpu)?;
        let quant=QMatMul::from_qtensor(QTensor::quantize_onto(&source,dtype,&vk)?)?;
        let reference=QTensor::quantize(&source,dtype)?.dequantize(&Device::Cpu)?;
        for batch in [1,2,3,4,5,19] {
            let x=Tensor::from_vec((0..batch*512).map(|i|((i*13%137)as f32-68.)/68.).collect(),(512,batch),&Device::Cpu)?.t()?;
            close(&format!("{dtype:?} prefill batch={batch} strided"),&quant.forward(&x.to_device(&vk)?)?,&x.matmul(&reference.t()?)?,0.002)?;
        }
    }
    // Exercise GGUF's actual public shared-loader path, not only quantize_onto.
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        let source = Tensor::from_vec(
            (0..512).map(|i| i as f32 / 31. - 7.).collect(),
            (2, 256),
            &Device::Cpu,
        )?;
        let cpu_quant = QTensor::quantize(&source, dtype)?;
        let mut cursor = std::io::Cursor::new(Vec::new());
        candle_core::quantized::gguf_file::write(
            &mut cursor,
            &[],
            &[("fixture.weight", &cpu_quant)],
        )?;
        cursor.set_position(0);
        let content = candle_core::quantized::gguf_file::Content::read(&mut cursor)?;
        let loaded = content.tensor(&mut cursor, "fixture.weight", &vk)?;
        assert_eq!(loaded.data()?.as_ref(), cpu_quant.data()?.as_ref());
        close(
            &format!("GGUF {dtype:?} loader"),
            &loaded.dequantize(&vk)?,
            &cpu_quant.dequantize(&Device::Cpu)?,
            1e-6,
        )?;
        let bytes = cpu_quant.data()?;
        assert!(
            candle_core::quantized::ggml_file::qtensor_from_ggml(
                dtype,
                &bytes[..bytes.len() - 1],
                vec![2, 256],
                &vk
            )
            .is_err()
        );
    }
    // Cross the Vulkan x-workgroup limit and exercise padded final dispatch rows.
    let big = Tensor::ones(4_194_305, DType::F32, &vk)?.affine(2.0, 1.0)?;
    assert_eq!(
        big.narrow(0, 4_194_240, 65)?.to_vec1::<f32>()?,
        vec![3.; 65]
    );
    println!("PASS 2D elementwise dispatch tail");
    let block = QTensor::quantize(
        &Tensor::ones((1, 256), DType::F32, &Device::Cpu)?,
        GgmlDType::Q4K,
    )?;
    let packed = block.data()?.repeat(65_537);
    let quant = QTensor::new(
        QStorage::from_data(packed.into(), &vk, GgmlDType::Q4K)?,
        (65_537, 256),
    )?;
    let out = QMatMul::from_qtensor(quant)?.forward(&Tensor::ones((1, 256), DType::F32, &vk)?)?;
    let expected = block
        .dequantize(&Device::Cpu)?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        out.flatten_all()?
            .narrow(0, 65_534, 3)?
            .to_vec1::<f32>()?
            .iter()
            .all(|&v| (v - expected).abs() < 0.001)
    );
    println!("PASS 2D quantized dispatch tail");
    println!("PASS first-class Candle Vulkan; HIP and Vulkan compiled together");
    Ok(())
}
