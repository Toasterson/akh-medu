#[path="quantized_gemma4_profile.rs"] mod gemma;
use candle_core::{Device,Tensor,quantized::gguf_file};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=std::env::args().collect();let model=&args[1];let tok=&args[2];let prompt=std::fs::read_to_string(&args[3])?;
 let dev=Device::new_vulkan(0)?;let mut file=std::fs::File::open(model)?;let content=gguf_file::Content::read(&mut file)?;
 let mut model=gemma::ModelWeights::from_gguf(content,&mut file,&dev)?;
 let tokenizer=tokenizers::Tokenizer::from_file(tok).map_err(|e|e.to_string())?;let ids=tokenizer.encode(prompt.as_str(),true).map_err(|e|e.to_string())?.get_ids().to_vec();
 let start=std::time::Instant::now();let mut logits=model.forward(&Tensor::new(ids.as_slice(),&dev)?.unsqueeze(0)?,0)?;dev.synchronize()?;
 println!("{{\"prefill_seconds\":{},\"tokens\":{}}}",start.elapsed().as_secs_f64(),ids.len());
 for i in 0..3 {let values=logits.flatten_all()?.to_vec1::<f32>()?;let token=values.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0 as u32;let start=std::time::Instant::now();logits=model.forward(&Tensor::new(&[token],&dev)?.unsqueeze(0)?,ids.len()+i)?;dev.synchronize()?;println!("{{\"step\":{},\"token\":{},\"seconds\":{}}}",i,token,start.elapsed().as_secs_f64());}
 Ok(())
}
