//! Verify the vendored Candle models expose TRUE hidden states (not logits).
//!
//! `AKH_LLM_MODEL=gemma3 cargo run --release --example candle_hidden --features candle-backend`
#[cfg(feature = "candle-backend")]
fn main() -> miette::Result<()> {
    use akh_medu::nlu::candle_backend::CandleBackend;
    use akh_medu::paths::AkhPaths;
    let models = AkhPaths::resolve()?.models_dir();
    let mut b = CandleBackend::try_load(&models).expect("load model");
    eprintln!("family: {:?}", b.family());

    let h = b
        .extract_hidden_state("The kingdom of Eldoria is vast.")
        .expect("hidden");
    eprintln!("hidden_state dim = {}", h.len());
    eprintln!("first 6 = {:?}", &h[..6.min(h.len())]);
    // Qwen2.5-1.5B hidden = 1536; Gemma-3-4B = 2560. Logits would be ~151936/262144.
    let looks_like_logits = h.len() > 50_000;
    eprintln!(
        "=> {}",
        if looks_like_logits {
            "LOOKS LIKE LOGITS (wrong)"
        } else {
            "TRUE HIDDEN STATE (ok)"
        }
    );

    // Sanity: verbalize still works through the vendored forward().
    let v = b.verbalize(
        "What is Eldoria?",
        "Eldoria is a kingdom.",
        "Akh",
        &["precise".into()],
    );
    eprintln!("verbalize -> {:?}", v);
    Ok(())
}
#[cfg(not(feature = "candle-backend"))]
fn main() {}
