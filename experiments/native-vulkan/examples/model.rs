//! Fast core-model benchmark with optional resident-idle sampling. Imports
//! production source instead of maintaining another model. This does not replace
//! the native completion wrapper and four-daemon integration gates.
#[path = "../../../src/nlu/vendored/quantized_qwen35.rs"]
mod model;
#[path = "../../../examples/support/gpu_idle.rs"]
mod gpu_idle;

use candle_core::{Device, IndexOp, Tensor, quantized::gguf_file};
use sha2::{Digest, Sha256};
use std::{fs::File, io::{BufReader, Read}, path::Path, time::Instant};

fn verify(path: &Path, expected: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BufReader::with_capacity(8*1024*1024, File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8;8*1024*1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 { break; }
        hash.update(&buffer[..count]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!("artifact fingerprint mismatch: {}", path.display()).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: model MODEL TOKENIZER hip|vulkan [ROUNDS=2]".into());
    }
    let rounds: usize = args.get(3).map(|value| value.parse()).transpose()?.unwrap_or(2);
    if !(1..=32).contains(&rounds) { return Err("rounds must be 1..32".into()); }
    let idle_path = std::env::var_os("AKH_PROBE_GPU_SYSFS").map(std::path::PathBuf::from);
    let baseline = if let Some(path) = &idle_path {
        std::thread::sleep(std::time::Duration::from_secs(10));
        Some(gpu_idle::sample_idle(path, 3)?)
    } else { None };
    verify(Path::new(&args[0]), "3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec")?;
    verify(Path::new(&args[1]), "e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&args[1]).map_err(|e| e.to_string())?;
    let prompt = "<|im_start|>user\nWrite a numbered list of integers from 1 through 100, one line per number. Do not summarize.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
    let prompt_ids = encoding.get_ids();
    assert_eq!(prompt_ids.len(), 36, "pinned counting fixture tokenization changed");
    let device = match args[2].as_str() {
        "hip" => Device::new_cuda(0)?,
        "vulkan" => Device::new_vulkan(0)?,
        _ => return Err("explicit device must be hip or vulkan; no CPU fallback".into()),
    };
    let load_started = Instant::now();
    let mut file = File::open(&args[0])?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = model::ModelWeights::from_gguf(content, &mut file, &device)?;
    println!("{}", serde_json::json!({"phase":"core_loaded", "seconds":load_started.elapsed().as_secs_f64(), "device":args[2]}));
    let expected = (1..=46).map(|n| n.to_string()).collect::<Vec<_>>().join("\n");
    for round in 0..rounds {
        let started = Instant::now();
        let input = Tensor::new(prompt_ids, &device)?.unsqueeze(0)?;
        let mut logits = model.prefill_controlled(&input, 128, &mut || Ok(()))?;
        let mut ids = Vec::with_capacity(128);
        let mut first_sample = None;
        for index in 0..128 {
            let token = logits.i((0,..))?.argmax(0)?.to_scalar::<u32>()?;
            first_sample.get_or_insert_with(Instant::now);
            ids.push(token);
            if index < 127 {
                let token = Tensor::new(&[token], &device)?.unsqueeze(0)?;
                logits = model.decode_step(&token, prompt_ids.len()+index)?;
            }
        }
        let finished = Instant::now();
        let text = tokenizer.decode(&ids, false).map_err(|e| e.to_string())?;
        assert_eq!(text, expected, "core benchmark changed the pinned greedy fixture");
        let first_sample = first_sample.unwrap();
        let decode_seconds = finished.duration_since(first_sample).as_secs_f64();
        println!("{}", serde_json::json!({"phase":"core_generation", "round":round,
            "prompt_tokens":prompt_ids.len(), "tokens":ids.len(), "decode_steps":127,
            "model_ttft_seconds":first_sample.duration_since(started).as_secs_f64(),
            "decode_seconds":decode_seconds, "decode_steps_per_second":127.0/decode_seconds,
            "expected_text_matched":true}));
    }
    if let (Some(path), Some((_, baseline_watts))) = (idle_path.as_ref(), baseline) {
        std::thread::sleep(std::time::Duration::from_secs(30));
        let (busy, watts) = gpu_idle::sample_idle(path, 60)?;
        let pass = busy <= 5.0 && watts <= baseline_watts+20.0;
        println!("{}", serde_json::json!({"phase":"core_resident_idle", "samples":60,
            "mean_busy_percent":busy, "mean_watts":watts, "baseline_watts":baseline_watts,
            "pass":pass}));
        std::hint::black_box(&model);
        if !pass { return Err("core resident GPU idle power regression".into()); }
    }
    Ok(())
}
