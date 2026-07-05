//! Native audio input for Gemma 4 (ADR 054): fetch/decode an audio clip and
//! ask Gemma 4 about it.
//!
//! ```text
//! AKH_GPU_MATMUL=1 AKH_GPU_QUANT=1 cargo run --release \
//!   --features rocm,candle-backend --example gemma4_audio -- \
//!   <audio-url-or-path> ["question"]
//! ```
//!
//! The model + tokenizer are discovered under
//! `~/.local/share/akh-medu/models/llm/` (gemma4-12b.gguf +
//! gemma4-12b.tokenizer.json), with the projector sidecar
//! gemma4-12b.mmproj.gguf alongside.
#[cfg(feature = "candle-backend")]
fn main() {
    use akh_medu::nlu::audio::{decode_wav_bytes_to_16k_mono, decode_wav_to_16k_mono, fetch_media_bytes};
    use akh_medu::nlu::candle_backend::CandleBackend;
    use std::path::Path;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,akh_medu=info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gemma4_audio <audio-url-or-path> [question]");
        std::process::exit(2);
    }
    let source = &args[1];
    let question = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "What is being said in this audio? Describe it.".to_string());

    // Model paths.
    let dir = dirs_data().join("models/llm");
    let model = dir.join("gemma4-12b.gguf");
    let tok = dir.join("gemma4-12b.tokenizer.json");

    // 1. Obtain WAV bytes (URL or local path) and decode to 16 kHz mono.
    eprintln!("[1/3] loading audio from {source}");
    let pcm = if source.starts_with("http://") || source.starts_with("https://") {
        let bytes = fetch_media_bytes(source, 60).unwrap_or_else(|e| {
            eprintln!("fetch failed: {e:?}");
            std::process::exit(1);
        });
        eprintln!("  fetched {} bytes", bytes.len());
        decode_wav_bytes_to_16k_mono(&bytes)
    } else {
        decode_wav_to_16k_mono(Path::new(source))
    }
    .unwrap_or_else(|e| {
        eprintln!("decode failed: {e:?}");
        std::process::exit(1);
    });

    // Optionally trim to the first N seconds (each 40 ms = 1 audio frame, so a
    // long clip becomes a long prompt). Default 10s; set AKH_AUDIO_MAX_SECS=0
    // to disable.
    let max_secs: f32 = std::env::var("AKH_AUDIO_MAX_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let mut pcm = pcm;
    if max_secs > 0.0 {
        let cap = (max_secs * 16_000.0) as usize;
        if pcm.len() > cap {
            eprintln!("  trimming to first {max_secs}s ({cap} samples)");
            pcm.truncate(cap);
        }
    }

    let secs = pcm.len() as f32 / 16_000.0;
    eprintln!(
        "  decoded {} samples ({secs:.2}s @ 16kHz, {} frames)",
        pcm.len(),
        pcm.len().div_ceil(640)
    );

    // 2. Load Gemma 4.
    eprintln!("[2/3] loading Gemma 4 from {}", model.display());
    let mut backend = CandleBackend::load(&model, &tok).unwrap_or_else(|e| {
        eprintln!("model load failed: {e:?}");
        std::process::exit(1);
    });
    eprintln!("  loaded, family = {:?}", backend.family());

    // 3. Understand the audio.
    eprintln!("[3/3] understand_audio — question: {question:?}");
    match backend.understand_audio(&pcm, &question) {
        Ok(answer) => {
            println!("\n=== Gemma 4 answer ===\n{answer}\n");
        }
        Err(e) => {
            eprintln!("understand_audio failed: {e:?}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "candle-backend")]
fn dirs_data() -> std::path::PathBuf {
    // Mirror the engine's default data dir (~/.local/share/akh-medu).
    if let Ok(d) = std::env::var("AKH_DATA_DIR") {
        return std::path::PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".local/share/akh-medu")
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("build with --features candle-backend");
}
