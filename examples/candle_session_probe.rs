//! Resident native generation + idle-power acceptance probe, not a throughput oracle.
//! Run with `--features hip` and explicit model/tokenizer and AMD sysfs device paths.
//! Keeps the model alive throughout idle measurement; exits nonzero on regression.
#[cfg(feature = "candle-backend")]
#[path = "support/gpu_idle.rs"]
mod gpu_idle;
#[cfg(feature = "candle-backend")]
#[path = "support/continuation.rs"]
mod continuation;

#[cfg(feature = "candle-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use akh_medu::nlu::candle_backend::{CandleBackend, CandleConfig, CandleDevice};
    use clap::Parser;
    use gpu_idle::sample_idle;
    use std::{path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "hip")]
        device: String,
        /// PCI device directory containing gpu_busy_percent and hwmon.
        #[arg(long)]
        gpu_sysfs: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        tokens: u32,
        /// Test keyed continuation, eviction, cancellation and context limits.
        #[arg(long)]
        continuation: bool,
        #[arg(long, default_value_t = 10)]
        idle_samples: usize,
        #[arg(long, default_value_t = 10)]
        settle_seconds: u64,
        /// Let power telemetry settle after any previous benchmark process.
        #[arg(long, default_value_t = 10)]
        baseline_settle_seconds: u64,
        #[arg(long, default_value_t = 5.0)]
        max_idle_busy: f64,
        #[arg(long, default_value_t = 20.0)]
        max_idle_extra_watts: f64,
        #[arg(
            long,
            default_value = "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
        )]
        prompt: String,
    }
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    if args.idle_samples == 0 || args.idle_samples > 300 {
        return Err("idle samples must be 1..300".into());
    }
    let placement = match args.device.as_str() {
        "hip" => CandleDevice::Hip { ordinal: 0 },
        "cpu" => CandleDevice::Cpu,
        "vulkan" => CandleDevice::Vulkan { ordinal: 0 },
        _ => return Err("device must be hip, cpu or vulkan".into()),
    };
    let baseline = args
        .gpu_sysfs
        .as_deref()
        .map(|path| {
            std::thread::sleep(std::time::Duration::from_secs(args.baseline_settle_seconds));
            sample_idle(path, 3)
        })
        .transpose()?;
    let start = Instant::now();
    let mut backend = CandleBackend::load_on_device(
        &args.model,
        &args.tokenizer,
        CandleConfig::default(),
        placement,
    )?;
    println!(
        "{}",
        serde_json::json!({"phase":"loaded","seconds":start.elapsed().as_secs_f64(),"family":format!("{:?}",backend.family()),"context_length":backend.context_length()})
    );
    let mut first = None;
    for round in 0..2 {
        let started = Instant::now();
        let reply = backend.generate_completion(&args.prompt, args.tokens, &mut || Ok(()))?;
        let elapsed = started.elapsed().as_secs_f64();
        if let Some(previous) = first.as_ref() {
            if previous != &reply.text {
                return Err("greedy repeated prompt changed across resident calls".into());
            }
        } else {
            first = Some(reply.text.clone());
        }
        println!(
            "{}",
            serde_json::json!({"phase":"generation","round":round,"seconds_including_prefill":elapsed,"tokens":reply.tokens_generated,"prompt_tokens":reply.prompt_tokens,"finish":format!("{:?}",reply.finish_reason),"text":reply.text,
                "time_to_first_token_seconds":reply.time_to_first_token.as_secs_f64(),
                "decode_seconds":reply.decode_time.as_secs_f64(),
                "decode_steps_including_eos":reply.decode_steps,
                "decode_steps_per_second":if reply.decode_steps > 0 {Some(reply.decode_steps as f64 / reply.decode_time.as_secs_f64())} else {None}})
        );
    }
    if args.continuation {
        continuation::check(&mut backend)?;
    }
    if let (Some(path), Some((_, baseline_watts))) = (&args.gpu_sysfs, baseline) {
        std::thread::sleep(std::time::Duration::from_secs(args.settle_seconds));
        let (busy, watts) = sample_idle(path, args.idle_samples)?;
        let pass =
            busy <= args.max_idle_busy && watts <= baseline_watts + args.max_idle_extra_watts;
        println!(
            "{}",
            serde_json::json!({"phase":"resident_idle","samples":args.idle_samples,"mean_busy_percent":busy,"mean_watts":watts,"baseline_watts":baseline_watts,"pass":pass})
        );
        // Keep ownership visibly alive through measurement, not just until last use.
        std::hint::black_box(&backend);
        if !pass {
            return Err("resident GPU idle power regression".into());
        }
    }
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("candle_session_probe requires candle-backend (or hip)");
    std::process::exit(1);
}
