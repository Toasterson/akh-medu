//! Hidden-state parity: GPU `forward_with_hidden()` ≈ CPU (ADR 053).
//!
//! The Neural→VSA bridge (Phase 26) consumes the model's post-norm hidden state. This example
//! proves the GPU linear path produces the **same hidden state** as the pure-Candle CPU path,
//! so routing the model through the AMD GPU does not corrupt the representation the bridge
//! depends on.
//!
//! It loads the same model three ways and compares the extracted hidden vector for one prompt:
//!   * CPU            — pure Candle (`AKH_GPU_MATMUL` unset).
//!   * GPU f32        — resident f32 linears (`AKH_GPU_MATMUL=1`, no quant). Both start from the
//!                      same q4_K GGUF weights, so this should match CPU to a **tight** tolerance
//!                      (cosine ≈ 1.0) — the headline parity check.
//!   * GPU int4       — `AKH_GPU_MATMUL=1 AKH_GPU_QUANT=1`. Adds akh-medu's own int4 quant on top,
//!                      so it diverges more; the printed cosine quantifies that noise for the bridge.
//!
//! Run (model auto-detected from `AKH_LLM_MODEL`, default in models dir):
//!   `cargo run --release --features rocm --example candle_hidden_parity -- "your prompt"`
//!
//! Exit code is non-zero if the GPU-f32 cosine similarity falls below the threshold, so this
//! doubles as a CI-style gate when ROCm is present.

#[cfg(feature = "candle-backend")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(feature = "candle-backend")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "candle-backend")]
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

#[cfg(feature = "candle-backend")]
fn extract(models: &std::path::Path, prompt: &str) -> Option<Vec<f32>> {
    use akh_medu::nlu::candle_backend::CandleBackend;
    let mut b = CandleBackend::try_load(models)?;
    b.extract_hidden_state(prompt).ok()
}

#[cfg(feature = "candle-backend")]
fn main() -> miette::Result<()> {
    use akh_medu::paths::AkhPaths;
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The kingdom of Eldoria is vast.".into());
    let models = AkhPaths::resolve()?.models_dir();

    // Env is read at model load. Toggle around each load. (set_var/remove_var are `unsafe` in
    // edition 2024; this is a single-threaded example so it is sound.)
    let set = |k: &str, v: Option<&str>| unsafe {
        match v {
            Some(val) => std::env::set_var(k, val),
            None => std::env::remove_var(k),
        }
    };

    // CPU reference.
    set("AKH_GPU_MATMUL", None);
    set("AKH_GPU_QUANT", None);
    let h_cpu = extract(&models, &prompt).expect("CPU hidden");
    eprintln!("hidden dim = {}", h_cpu.len());

    // GPU f32.
    set("AKH_GPU_MATMUL", Some("1"));
    set("AKH_GPU_QUANT", None);
    let h_gpu_f32 = extract(&models, &prompt);

    // GPU int4.
    set("AKH_GPU_MATMUL", Some("1"));
    set("AKH_GPU_QUANT", Some("1"));
    let h_gpu_q4 = extract(&models, &prompt);

    let mut gpu_f32_cos = 1.0f32;
    if let Some(h) = &h_gpu_f32 {
        let cos = cosine(&h_cpu, h);
        gpu_f32_cos = cos;
        eprintln!(
            "GPU f32  vs CPU: cosine={cos:.6}  max|Δ|={:.4e}  relL2={:.4e}",
            max_abs_diff(&h_cpu, h),
            rel_l2(&h_cpu, h)
        );
    } else {
        eprintln!("GPU f32  vs CPU: (GPU unavailable — built without rocm or no device)");
    }
    if let Some(h) = &h_gpu_q4 {
        eprintln!(
            "GPU int4 vs CPU: cosine={:.6}  max|Δ|={:.4e}  relL2={:.4e}",
            cosine(&h_cpu, h),
            max_abs_diff(&h_cpu, h),
            rel_l2(&h_cpu, h)
        );
    }

    // Gate: GPU-f32 must track CPU tightly (only matmul accumulation order differs).
    const THRESHOLD: f32 = 0.999;
    if h_gpu_f32.is_some() && gpu_f32_cos < THRESHOLD {
        eprintln!("FAIL: GPU f32 hidden cosine {gpu_f32_cos:.6} < {THRESHOLD}");
        std::process::exit(1);
    }
    eprintln!("PARITY OK (GPU f32 cosine ≥ {THRESHOLD} or GPU unavailable)");
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {}
