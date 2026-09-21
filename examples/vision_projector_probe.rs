//! Run the exact Ptah vision projector on a device and report its output.
//!
//! Loads only the projector, not the language model, so a device run costs a
//! GGUF read rather than 16.8 GB of weights. Prints per-device statistics so
//! HIP and Vulkan results can be compared against CPU and each other.
#[cfg(feature = "candle-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use akh_medu::nlu::vision_tower::VisionTower;
    use candle_core::{quantized::gguf_file, Device, Tensor};
    use clap::Parser;
    use std::{path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        /// Language model blob; the projector is found through its manifest.
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Square input side in pixels. Must be a multiple of patch x merge.
        #[arg(long, default_value_t = 64)]
        side: usize,
        /// Write the full projection as little-endian f32 for cross-device
        /// comparison. Summary statistics can agree while the vectors do not.
        #[arg(long)]
        dump: Option<PathBuf>,
    }
    let args = Args::parse();
    // Each device arm needs its backend compiled in; naming one that is not
    // says so, rather than failing to build the whole example.
    let placement = match args.device.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "hip")]
        "hip" => Device::new_cuda(0)?,
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::new_vulkan(0)?,
        other => {
            return Err(format!(
                "device {other} is unavailable in this build; rebuild with the matching feature"
            )
            .into())
        }
    };

    let projector = akh_medu::nlu::projector_artifact::locate(&args.model)
        .ok_or("no projector alongside this model")?;
    let mut file = std::fs::File::open(&projector)?;
    let content = gguf_file::Content::read(&mut file)?;

    let started = Instant::now();
    let tower = VisionTower::load(&content, &mut file, &placement)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let config = tower.config().clone();

    // Deterministic pseudo-image, so every device sees identical input.
    let pixels: Vec<f32> = (0..3 * args.side * args.side)
        .map(|i| (((i * 37) % 255) as f32 / 127.5) - 1.0)
        .collect();
    let input = Tensor::from_vec(pixels, (3, args.side, args.side), &placement)?;

    let started = Instant::now();
    let out = tower.forward(&input)?;
    let forward_seconds = started.elapsed().as_secs_f64();
    let values: Vec<f32> = out.flatten_all()?.to_vec1()?;

    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let norm = values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    println!(
        "{}",
        serde_json::json!({
            "phase": "vision_projection",
            "device": args.device,
            "projector": projector.file_name().and_then(|n| n.to_str()),
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "input_side": args.side,
            "rows": out.dims()[0],
            "width": out.dims()[1],
            "blocks": config.block_count,
            "all_finite": values.iter().all(|v| v.is_finite()),
            "mean": mean,
            "std_dev": variance.sqrt(),
            "l2_norm": norm,
            // A few exact values so two devices can be compared directly.
            "head": values.iter().take(8).collect::<Vec<_>>(),
        })
    );
    if let Some(path) = args.dump.as_ref() {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
        eprintln!("wrote {} values to {}", values.len(), path.display());
    }
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("vision_projector_probe requires the candle-backend feature");
}
