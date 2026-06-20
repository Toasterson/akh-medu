//! Diagnostic: load a specific GGUF via CandleBackend::load and print the error verbatim.
//! `cargo run --release --features rocm,candle-backend --example g4_load -- <model.gguf> <tok.json>`
#[cfg(feature = "candle-backend")]
fn main() {
    use akh_medu::nlu::candle_backend::CandleBackend;
    use std::path::Path;
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let tok = &args[2];
    eprintln!("loading {model}");
    match CandleBackend::load(Path::new(model), Path::new(tok)) {
        Ok(mut b) => {
            eprintln!("OK loaded, family = {:?}", b.family());
            // Short greedy generation to judge coherence quickly. The backend
            // prepends the leading BOS for Gemma; Gemma 4 uses the `<|turn>` chat
            // format. Override with PROMPT / NTOK for ad-hoc testing.
            let prompt = std::env::var("PROMPT")
                .unwrap_or_else(|_| "<|turn>user\nName three colors.<turn|>\n<|turn>model\n".to_string());
            let ntok: u32 = std::env::var("NTOK").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
            eprintln!("--- generate ({ntok} tok) ---");
            match b.generate(&prompt, ntok) {
                Ok(text) => eprintln!("OUT: {text:?}"),
                Err(e) => eprintln!("GEN ERROR: {e:?}"),
            }
        }
        Err(e) => eprintln!("LOAD ERROR: {e:?}"),
    }
}
#[cfg(not(feature = "candle-backend"))]
fn main() {}
