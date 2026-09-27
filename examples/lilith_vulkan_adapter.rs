//! Explicit scratch-adapter application/reload/rollback probe on the real GGUF.
#[cfg(feature = "vulkan")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use akh_medu::nlu::vendored::quantized_gemma4::ModelWeights;
    use candle_core::{Device, Tensor, quantized::gguf_file};
    use clap::Parser;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        io::Read,
        path::{Path, PathBuf},
        time::Instant,
    };
    const BASE_SHA256: &str = "43fec98c5102b1c446b4ddd0a9439f1db3a2e1f2e0b8cd143ce1ea619a9403d6";
    const TRAINING_PARENT_SHA256: &str =
        "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d";
    const TOKENIZER_SHA256: &str =
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
    const TARGET: &str = "model.language_model.layers.47.self_attn.q_proj";
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
    }
    fn hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        let mut f = fs::File::open(path)?;
        let mut h = Sha256::new();
        let mut b = vec![0u8; 8 * 1024 * 1024];
        loop {
            let n = f.read(&mut b)?;
            if n == 0 {
                break;
            }
            h.update(&b[..n]);
        }
        Ok(format!("{:x}", h.finalize()))
    }
    fn values(t: Tensor) -> candle_core::Result<Vec<f32>> {
        t.flatten_all()?.to_vec1::<f32>()
    }
    fn difference(a: &[f32], b: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
            return Err("invalid logits".into());
        }
        Ok(a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max))
    }
    fn logits_sha256(values: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err("non-finite logits".into());
        }
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    fn generate(
        model: &mut ModelWeights,
        input: &[u32],
        dev: &Device,
        count: usize,
    ) -> candle_core::Result<(Vec<u32>, f64)> {
        model.reset_kv_cache();
        let mut logits = model.forward(&Tensor::new(input, dev)?.unsqueeze(0)?, 0)?;
        let start = Instant::now();
        let mut out = vec![];
        for i in 0..count {
            let v = values(logits)?;
            let token = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            out.push(token);
            logits = model.forward(&Tensor::new(&[token], dev)?.unsqueeze(0)?, input.len() + i)?;
        }
        Ok((out, start.elapsed().as_secs_f64()))
    }
    let args = Args::parse();
    if !(1..=256).contains(&args.tokens) {
        return Err("tokens must be 1..256".into());
    }
    let manifest_path = args.adapter.join("manifest.json");
    let manifest_sha256 = hash(&manifest_path)?;
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest["schema"] != "lilith/vulkan-query-adapter@1"
        || manifest["candidate_only"] != true
        || manifest["quality_certified"] != false
        || manifest["base_sha256"] != BASE_SHA256
        || manifest["training_parent_sha256"] != TRAINING_PARENT_SHA256
        || manifest["target"] != TARGET
        || manifest["scale"] != 1.0
        || manifest["export_dtype"] != "F32"
        || manifest["exact_tensor_roundtrip"] != true
    {
        return Err("unsupported adapter manifest".into());
    }
    let adapter_path = args.adapter.join("query-lora.safetensors");
    let model_sha256 = hash(&args.model)?;
    let tokenizer_sha256 = hash(&args.tokenizer)?;
    if model_sha256 != BASE_SHA256
        || tokenizer_sha256 != TOKENIZER_SHA256
        || manifest["adapter_sha256"] != hash(&adapter_path)?
    {
        return Err("model/tokenizer/adapter hash mismatch".into());
    }
    let dev = Device::new_vulkan(0)?;
    let start = Instant::now();
    let mut file = fs::File::open(&args.model)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &dev)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(|e| e.to_string())?;
    let prompt = "<bos><start_of_turn>user\nExplain why integer bounds need checking.<end_of_turn>\n<start_of_turn>model\n";
    let ids = tokenizer
        .encode(prompt, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
    let base = values(model.forward(&input, 0)?)?;
    let weights = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;
    if weights.len() != 2 {
        return Err("unexpected exported tensors".into());
    }
    let a = weights.get("lora_a").ok_or("missing A")?;
    let b = weights.get("lora_b").ok_or("missing B")?;
    model.set_last_query_lora(a, &b.zeros_like()?, 1.0)?;
    let zero = values(model.forward(&input, 0)?)?;
    model.set_last_query_lora(a, b, 1.0)?;
    let adapted = values(model.forward(&input, 0)?)?;
    let (adapted_ids, adapted_seconds) = generate(&mut model, &ids, &dev, args.tokens)?;
    model.clear_query_lora();
    let rollback = values(model.forward(&input, 0)?)?;
    let (base_ids, base_seconds) = generate(&mut model, &ids, &dev, args.tokens)?;
    let fresh = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;
    model.set_last_query_lora(&fresh["lora_a"], &fresh["lora_b"], 1.0)?;
    let reload = values(model.forward(&input, 0)?)?;
    let zero_error = difference(&base, &zero)?;
    let rollback_error = difference(&base, &rollback)?;
    let reload_error = difference(&adapted, &reload)?;
    let effect = difference(&base, &adapted)?;
    let base_logits_sha256 = logits_sha256(&base)?;
    let adapted_logits_sha256 = logits_sha256(&adapted)?;
    let passed =
        zero_error <= 1e-6 && rollback_error <= 1e-6 && reload_error <= 1e-6 && effect > 0.0;
    let result = serde_json::json!({"schema":"lilith/vulkan-adapter-probe@1", "passed":passed,
        "manifest":manifest, "manifest_sha256":manifest_sha256,
        "model_sha256":model_sha256,"tokenizer_sha256":tokenizer_sha256,"load_seconds":load_seconds,
        "zero_adapter_max_error":zero_error,"rollback_max_error":rollback_error,"reload_max_error":reload_error,
        "adapter_effect_max_abs":effect,"logits_hash_encoding":"f32-little-endian-full-vocabulary",
        "base_logits_sha256":base_logits_sha256,"adapted_logits_sha256":adapted_logits_sha256,
        "base_tokens":base_ids,"adapter_tokens":adapted_ids,
        "base_decode_seconds":base_seconds,"adapter_decode_seconds":adapted_seconds,
        "base_decode_tps":args.tokens as f64/base_seconds,"adapter_decode_tps":args.tokens as f64/adapted_seconds,
        "tokens":args.tokens,"prompt":prompt,"quality_claim":false,"deployed_weights_changed":false});
    fs::write(args.output, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", result);
    if !passed {
        return Err("Vulkan adapter controls failed".into());
    }
    Ok(())
}
#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("Requires --features vulkan");
    std::process::exit(2);
}
