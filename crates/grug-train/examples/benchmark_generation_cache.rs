//! Compare compatibility and KV-cached greedy decoding on one verified
//! foundation checkpoint. Both paths must emit identical tokens before the
//! timing result is accepted.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::Backend;
use clap::Parser;
use grug_train::{
    BpeTokenizer, GenerateConfig, generate_from_prompt_ids, generate_from_prompt_ids_cached,
    verify_foundation_bundle,
};

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
use grug_train::CpuBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
use grug_train::GpuBackend as ActiveBackend;

#[derive(Debug, Parser)]
#[command(name = "benchmark_generation_cache")]
struct Args {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 3)]
    repeats: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.prompt.trim().is_empty() || args.max_new_tokens == 0 || args.repeats == 0 {
        return Err("prompt, max-new-tokens and repeats must be non-empty/positive".into());
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let device = Default::default();
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let device = grug_train::gpu_device();
    run::<ActiveBackend>(args, device)
}

fn run<B: Backend>(args: Args, device: B::Device) -> Result<(), Box<dyn Error>> {
    let verified = verify_foundation_bundle(&args.bundle)?;
    let tokenizer = BpeTokenizer::load(verified.tokenizer_path())?;
    let model = verified.load_model_only::<B>(&device)?;
    let prompt_ids = tokenizer.encode(&args.prompt)?;
    let config = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len: verified.artifact().shape.max_seq_len,
    };
    if prompt_ids.len() + config.max_new_tokens + 2 > config.max_seq_len {
        return Err("prompt does not leave the requested cached completion budget".into());
    }

    // Compile both backend paths outside the timed loop.
    let reference = generate_from_prompt_ids(&model, &tokenizer, &prompt_ids, config, &device);
    let cached = generate_from_prompt_ids_cached(&model, &tokenizer, &prompt_ids, config, &device);
    if cached != reference {
        return Err("cached and compatibility decoding emitted different output".into());
    }

    let started = Instant::now();
    for _ in 0..args.repeats {
        let output = generate_from_prompt_ids(&model, &tokenizer, &prompt_ids, config, &device);
        if output != reference {
            return Err("compatibility decoding was not deterministic".into());
        }
    }
    let uncached_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repeats as f64;

    let started = Instant::now();
    for _ in 0..args.repeats {
        let output =
            generate_from_prompt_ids_cached(&model, &tokenizer, &prompt_ids, config, &device);
        if output != reference {
            return Err("cached decoding was not deterministic".into());
        }
    }
    let cached_ms = started.elapsed().as_secs_f64() * 1_000.0 / args.repeats as f64;

    println!(
        "checkpoint_step={} prompt_tokens={} completion_tokens={} repeats={} uncached_ms={uncached_ms:.3} cached_ms={cached_ms:.3} speedup={:.2}",
        verified.artifact().checkpoint_step,
        prompt_ids.len() + 2,
        tokenizer.encode(&reference)?.len(),
        args.repeats,
        uncached_ms / cached_ms.max(f64::EPSILON),
    );
    Ok(())
}
