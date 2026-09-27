//! Exclusive-device numerical and warm-timing check for decode-shaped matmuls.
#[cfg(feature="vulkan")]
fn main()->Result<(),Box<dyn std::error::Error>> {
 use candle_core::{Device,Tensor};
 use std::time::Instant;
 let gpu=Device::new_vulkan(0)?;
 for &(batch,n,k,transposed) in &[(2,37,19,false),(16,773,512,true),(16,512,773,false),(16,256,1025,false)] {
  let av:Vec<f32>=(0..batch*k).map(|i| ((i*13%97) as f32-48.)/97.).collect();
  let bv:Vec<f32>=(0..batch*n*k).map(|i| ((i*17%113) as f32-56.)/113.).collect();
  let a=Tensor::from_vec(av,(batch,1,k),&Device::Cpu)?;
  let b=if transposed {Tensor::from_vec(bv,(batch,n,k),&Device::Cpu)?.transpose(1,2)?}else{Tensor::from_vec(bv,(batch,k,n),&Device::Cpu)?};
  let expected=a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
  let ga=a.to_device(&gpu)?;
  // Preserve the right-hand operand's transpose layout after upload.
  let gb=if transposed {b.transpose(1,2)?.to_device(&gpu)?.transpose(1,2)?}else{b.to_device(&gpu)?};
  let observed=ga.matmul(&gb)?.flatten_all()?.to_vec1::<f32>()?;
  let max_error=expected.iter().zip(&observed).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
  if observed.iter().any(|x| !x.is_finite()) || max_error>0.0002 {return Err(format!("parity failure {batch}/{n}/{k}: {max_error}").into())}
  let start=Instant::now();
  for _ in 0..20 {let _=ga.matmul(&gb)?.flatten_all()?.to_vec1::<f32>()?;}
  println!("{}",serde_json::json!({"batch":batch,"n":n,"k":k,"transposed":transposed,"max_abs_error":max_error,"mean_seconds":start.elapsed().as_secs_f64()/20.}));
 }
 Ok(())
}
#[cfg(not(feature="vulkan"))]
fn main(){panic!("requires vulkan")}
