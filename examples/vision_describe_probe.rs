//! Run one image through the full native vision path and print what the
//! model says about it. A smoke signal, not a gate: if the pinned model
//! describes a synthetic image correctly, every stage from decode to rope
//! cursor is at least not broken; it says nothing numerical.
#[cfg(feature = "candle-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use akh_medu::nlu::candle_backend::{CandleBackend, CandleConfig, CandleDevice};
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "hip")]
        device: String,
        /// An encoded PNG/JPEG. Absent, a synthetic image is used: a red
        /// square on white, with a blue bar along the bottom.
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        tokens: u32,
        #[arg(
            long,
            default_value = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n"
        )]
        prompt: String,
    }
    let args = Args::parse();
    let placement = match args.device.as_str() {
        "hip" => CandleDevice::Hip { ordinal: 0 },
        "cpu" => CandleDevice::Cpu,
        "vulkan" => CandleDevice::Vulkan { ordinal: 0 },
        _ => return Err("device must be hip, cpu or vulkan".into()),
    };
    let bytes = match &args.image {
        Some(path) => std::fs::read(path)?,
        None => {
            let img = image::RgbImage::from_fn(512, 512, |x, y| {
                if y >= 448 {
                    image::Rgb([0, 0, 255])
                } else if (128..384).contains(&x) && (96..352).contains(&y) {
                    image::Rgb([220, 20, 20])
                } else {
                    image::Rgb([255, 255, 255])
                }
            });
            let mut out = Vec::new();
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
            out
        }
    };
    let mut backend = CandleBackend::load_on_device(
        &args.model,
        &args.tokenizer,
        CandleConfig::default(),
        placement,
    )?;
    let reply = backend.generate_with_image(&args.prompt, &bytes, args.tokens, &mut || Ok(()))?;
    println!(
        "{}",
        serde_json::json!({
            "phase": "vision_describe",
            "prompt_tokens": reply.prompt_tokens,
            "tokens": reply.tokens_generated,
            "finish": format!("{:?}", reply.finish_reason),
            "time_to_first_token_seconds": reply.time_to_first_token.as_secs_f64(),
            "decode_seconds": reply.decode_time.as_secs_f64(),
            "text": reply.text,
        })
    );
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("build with --features hip or --features vulkan");
}
