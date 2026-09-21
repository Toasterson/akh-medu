use candle_core::{Device,DType,Tensor,Result,Module};
use candle_core::quantized::{gguf_file,QMatMul};
use std::time::Instant;

fn bench(label:&str,device:&Device,mut op:impl FnMut()->Result<Tensor>)->Result<()> {
    for _ in 0..3{let _=op()?;device.synchronize()?;}
    let mut times=Vec::new();let mut last=None;
    for _ in 0..15{device.synchronize()?;let start=Instant::now();last=Some(op()?);device.synchronize()?;times.push(start.elapsed().as_secs_f64()*1e6);}
    times.sort_by(f64::total_cmp);
    let out=last.unwrap().flatten_all()?;let sample=out.narrow(0,0,out.elem_count().min(4))?.to_vec1::<f32>()?;
    device.synchronize()?;let start=Instant::now();let mut keep=Vec::new();
    for _ in 0..10{keep.push(op()?);}
    let host=start.elapsed().as_secs_f64()*1e6/10.;device.synchronize()?;
    let queued=start.elapsed().as_secs_f64()*1e6/10.;
    println!("{label}: median_us={:.1} p95_us={:.1} host_submit_us={host:.1} queued_us={queued:.1} sample={sample:?}",times[7],times[13]);Ok(())
}

fn main()->Result<()> {
    let device=Device::new_vulkan(0)?;
    for (batch,m,n,k) in [(48,128,1,128),(48,128,128,1),(32,1,20,128),(32,19,19,128)] {
        let a=Tensor::ones((batch,m,k),DType::F32,&device)?.affine(0.1,0.0)?;
        let b=Tensor::ones((batch,k,n),DType::F32,&device)?.affine(0.2,0.0)?;
        bench(&format!("dense {batch}x{m}x{n}x{k}"),&device,||a.matmul(&b))?;
    }
    for (rows,width) in [(1,5120),(19,5120),(48,128),(32,19)]{
        let input=Tensor::ones((rows,width),DType::F32,&device)?;
        bench(&format!("sum {rows}x{width}"),&device,||input.sum_keepdim(1))?;
        bench(&format!("unary {rows}x{width}"),&device,||input.silu())?;
    }
    if let Some(path)=std::env::args().nth(1) {
        let mut file=std::fs::File::open(path)?;let content=gguf_file::Content::read(&mut file)?;
        for name in ["blk.0.ffn_up.weight","blk.0.ffn_down.weight","output.weight"] {
            let info=&content.tensor_infos[name];let cols=*info.shape.dims().last().unwrap();
            let q=QMatMul::from_qtensor(content.tensor(&mut file,name,&device)?)?;
            for batch in [1,19] {
                if name=="output.weight"&&batch>1{continue;}
                let x=Tensor::from_vec((0..batch*cols).map(|i|((i*17%997)as f32-498.)/997.).collect(),(batch,cols),&device)?;
                bench(&format!("packed {name} {:?} {:?} batch={batch}",info.ggml_dtype,info.shape),&device,||q.forward(&x))?;
            }
        }
    }
    Ok(())
}
