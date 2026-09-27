//! Direct real-model decode control. Run variants in separate exclusive processes.
#[cfg(feature = "vulkan")]
#[path = "../src/nlu/vendored/quantized_gemma4.rs"]
mod gemma;
#[cfg(feature = "vulkan")]
use candle_core::{Device, Tensor, quantized::gguf_file};
#[cfg(feature = "vulkan")]
use sha2::{Digest, Sha256};
#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("model tokenizer prompt tokens".into());
    }
    let prompt = std::fs::read_to_string(&args[3])?;
    let count: usize = args[4].parse()?;
    let loaded = std::time::Instant::now();
    let dev = Device::new_vulkan(0)?;
    let mut file = std::fs::File::open(&args[1])?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = gemma::ModelWeights::from_gguf(content, &mut file, &dev)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&args[2]).map_err(|e| e.to_string())?;
    let mut ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if ids.first() != Some(&2) {
        ids.insert(0, 2);
    }
    println!(
        "{}",
        serde_json::json!({"phase":"loaded","seconds":loaded.elapsed().as_secs_f64(),"prompt_tokens":ids.len()})
    );
    for round in 0..2 {
        model.reset_kv_cache();
        let start = std::time::Instant::now();
        let mut logits = model.forward(&Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?, 0)?;
        dev.synchronize()?;
        let prefill = start.elapsed().as_secs_f64();
        let decode = std::time::Instant::now();
        let mut tokens = vec![];
        let mut hashes = vec![];
        for i in 0..count {
            let values = logits.flatten_all()?.to_vec1::<f32>()?;
            if values.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite logits".into());
            }
            let token = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            let mut hash = Sha256::new();
            for value in &values {
                hash.update(value.to_le_bytes());
            }
            hashes.push(format!("{:x}", hash.finalize()));
            tokens.push(token);
            if i + 1 < count {
                logits =
                    model.forward(&Tensor::new(&[token], &dev)?.unsqueeze(0)?, ids.len() + i)?;
            }
        }
        dev.synchronize()?;
        let seconds = decode.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({"phase":"generation","round":round,"prefill_seconds":prefill,"decode_seconds":seconds,"decode_tps":(count-1) as f64/seconds,"tokens":tokens,"logits_sha256":hashes})
        );
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    panic!("requires vulkan")
}
