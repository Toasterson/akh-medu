use candle_core::{Device,Result,Tensor};
fn main()->Result<()> {
    let vk=Device::new_vulkan(0)?;
    for width in [255,256,257,4097,5120,17408] {
        let values:Vec<_>=(0..7*width).map(|i|((i*17%997)as f32-498.)/997.).collect();
        let cpu=Tensor::from_vec(values.clone(),(7,width),&Device::Cpu)?;
        let gpu=cpu.to_device(&vk)?;
        for squared in [false,true] {
            let cpu=if squared{cpu.sqr()?}else{cpu.clone()};let gpu=if squared{gpu.sqr()?}else{gpu.clone()};
            let expected=cpu.sum(1)?.to_vec1::<f32>()?;
            for tensor in [gpu.clone(),gpu.transpose(0,1)?.contiguous()?.transpose(0,1)?] {
                let got=tensor.sum(1)?.to_vec1::<f32>()?;
                for (a,b) in got.iter().zip(&expected) {assert!((a-b).abs()<0.003+0.00002*b.abs(),"width={width} squared={squared}: {a} != {b}");}
                let offset=tensor.narrow(0,2,3)?.sum(1)?.to_vec1::<f32>()?;
                for (a,b) in offset.iter().zip(&expected[2..5]) {assert!((a-b).abs()<0.003+0.00002*b.abs());}
            }
        }
        println!("PASS sum width={width}: signed, squared, offset, strided");
    }
    Ok(())
}
