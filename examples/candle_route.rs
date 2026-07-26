//! Exercise the native Candle LLM route (translate / dialogue / verbalize).
//!
//! Loads a GGUF model from the shared akh-medu models dir
//! (`$XDG_DATA_HOME/akh-medu/models/llm/`). Pick the model with `AKH_LLM_MODEL`
//! (e.g. `gemma3`), otherwise the canonical Qwen model is used.
//!
//! ```sh
//! AKH_LLM_MODEL=gemma3 cargo run --example candle_route --features candle-backend
//! ```

#[cfg(feature = "candle-backend")]
fn main() -> miette::Result<()> {
    use akh_medu::nlu::candle_backend::CandleBackend;
    use akh_medu::paths::AkhPaths;

    let paths = AkhPaths::resolve()?;
    let models = paths.models_dir();
    println!("models dir: {}", models.display());

    let mut backend = match CandleBackend::try_load(&models) {
        Some(b) => b,
        None => {
            eprintln!("No Candle model found under {}/llm/. Place a .gguf + tokenizer.json.", models.display());
            std::process::exit(1);
        }
    };
    println!("loaded family: {:?}\n", backend.family());

    println!("== Route 1: translate (NL → AbsTree) ==");
    match backend.translate("dogs are mammals") {
        Ok(t) => println!("  ok: {}\n", t.json),
        Err(e) => println!("  ERR: {e}\n"),
    }

    println!("== Route 2: generate_dialogue (greeting) ==");
    match backend.generate_dialogue("greeting", "Akh", &["warm".into(), "curious".into()], None) {
        Ok(s) => println!("  ok: {s}\n"),
        Err(e) => println!("  ERR: {e}\n"),
    }

    println!("== Route 3: verbalize (grounded answer) ==");
    let facts = "Paris is the capital of France. France is in Europe.";
    match backend.verbalize("What is the capital of France?", facts, "Akh", &["precise".into()]) {
        Ok(s) => println!("  ok: {s}\n"),
        Err(e) => println!("  ERR: {e}\n"),
    }

    println!("All Candle routes exercised.");
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("This example requires the `candle-backend` feature.");
    std::process::exit(1);
}
