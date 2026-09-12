//! Phase 44n.1 context-rot curve bench (WI-386).
//!
//! Sweeps needle-depth × context-length over a seeded nonce-needle haystack,
//! scores per-cell hits, and emits schema-versioned CSVs plus a summary table.
//! Backends:
//!
//! - `--dry-run` — MockProbe; no model attached, validates plumbing and CSV
//!   emission deterministically (mirrors the `rot` unit-test mock).
//! - *(default)* — Ollama probe against a running server; sets `num_ctx` per
//!   request so the sweep never silently truncates.
//! - `--candle` — local GGUF via the `candle-backend` feature.
//!
//! Run with:
//!
//! ```sh
//! # plumbing-only validation (no model needed):
//! cargo run --release --example bench_context_rot -- --dry-run
//!
//! # real Ollama sweep (server running, model pulled):
//! AKH_ROT_MODEL=gemma4:e4b cargo run --release --example bench_context_rot
//!
//! # quick smoke grid:
//! AKH_BENCH_QUICK=1 cargo run --release --example bench_context_rot -- --dry-run
//!
//! # local GGUF:
//! cargo run --release --example bench_context_rot --features candle-backend -- --candle
//! ```
//!
//! The grid is further tunable via `AKH_ROT_LENGTHS` / `AKH_ROT_DEPTHS` /
//! `AKH_ROT_REPEATS` / `AKH_ROT_FILLER` / `AKH_ROT_THRESHOLD` — see
//! [`akh_medu::bench_phase44::rot::GridConfig::from_env`].
//!
//! All logic lives in `src/bench_phase44/rot.rs`; this file is only the entry
//! point the `bench_thinking` template prescribes.

use std::path::PathBuf;

use akh_medu::bench_phase44::rot::{
    GridConfig, ProbeResponse, RotProbe, emit_runs_csv, emit_summary_csv, format_table, run_grid,
    summarize,
    OUTPUT_DIR,
};
use akh_medu::agent::context::{HeuristicTokenCounter, TokenCounter};

// ---------------------------------------------------------------------------
// Dry-run mock probe
// ---------------------------------------------------------------------------

/// Mock probe: echoes the vault code when it can see it, so `hit` is always
/// true and every metric has data — this exercises the harness plumbing and
/// CSV emission, not LLM recall. Identical in spirit to the mock in
/// `rot::tests::dry_run_emits_schema_versioned_csv`.
struct MockProbe;

impl RotProbe for MockProbe {
    fn label(&self) -> &str {
        "mock"
    }

    fn backend(&self) -> &str {
        "mock"
    }

    fn context_window(&self) -> Option<usize> {
        // Unknown window: the grid is not capped in dry-run mode.
        None
    }

    fn complete(
        &mut self,
        _system: &str,
        user: &str,
        _max_tokens: u32,
    ) -> miette::Result<ProbeResponse> {
        // Recover the needle's "is XXXXXX." tail from the prompt.
        let text = user
            .lines()
            .find_map(|l| {
                l.strip_prefix("The access code for ")
                    .and_then(|l| l.split(" is ").nth(1))
            })
            .map(|tail| tail.trim_end_matches('.').to_string())
            .unwrap_or_default();
        Ok(ProbeResponse {
            text,
            prompt_tokens: Some(HeuristicTokenCounter.count(user) as u64),
        })
    }
}

// ---------------------------------------------------------------------------
// Probe selection
// ---------------------------------------------------------------------------

/// Build the Ollama probe from env (`AKH_ROT_MODEL`, else the daemon default).
fn ollama_probe() -> impl RotProbe {
    use akh_medu::agent::ollama::{OllamaClient, OllamaConfig};

    let mut cfg = OllamaConfig::default();
    if let Ok(model) = std::env::var("AKH_ROT_MODEL") {
        cfg.model = model;
    }
    if let Ok(url) = std::env::var("AKH_ROT_OLLAMA_URL") {
        cfg.base_url = url;
    }
    akh_medu::bench_phase44::rot::OllamaProbe::new(OllamaClient::new(cfg))
}

/// Build the Candle probe (`candle-backend` feature only).
#[cfg(feature = "candle-backend")]
fn candle_probe() -> miette::Result<impl RotProbe> {
    use akh_medu::nlu::candle_backend::CandleBackend;
    use akh_medu::paths::AkhPaths;

    let paths = AkhPaths::resolve()?;
    let model_dir = paths.models_dir();
    eprintln!("bench_context_rot: loading Candle GGUF from {}", model_dir.display());
    let backend = CandleBackend::try_load(&model_dir)
        .ok_or_else(|| miette::miette!("no .gguf model found under {}", model_dir.display()))?;
    Ok(akh_medu::bench_phase44::rot::CandleProbe::new(backend))
}

#[cfg(not(feature = "candle-backend"))]
fn candle_probe() -> miette::Result<MockProbe> {
    Err(miette::miette!(
        "--candle requires the `candle-backend` feature (cargo run --example bench_context_rot --features candle-backend -- --candle)"
    ))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> miette::Result<()> {
    miette::set_panic_hook();

    let dry_run = std::env::args().any(|a| a == "--dry-run");
    let use_candle = std::env::args().any(|a| a == "--candle");

    let mut cfg = GridConfig::from_env();

    if dry_run {
        eprintln!(
            "bench_context_rot: --dry-run with MockProbe ({} probes)",
            cfg.total_probes()
        );
        let mut probe = MockProbe;
        cfg.cap_to_window(probe.context_window());
        run_with(&mut probe, &cfg)
    } else if use_candle {
        let mut probe = candle_probe()?;
        eprintln!("bench_context_rot: candle probe ({} probes)", cfg.total_probes());
        cfg.cap_to_window(probe.context_window());
        run_with(&mut probe, &cfg)
    } else {
        let mut probe = ollama_probe();
        eprintln!(
            "bench_context_rot: ollama model={} ({} probes)",
            probe.label(),
            cfg.total_probes()
        );
        cfg.cap_to_window(probe.context_window());
        run_with(&mut probe, &cfg)
    }
}

/// Run the grid against one probe, emit CSVs, and print the table.
fn run_with(probe: &mut dyn RotProbe, cfg: &GridConfig) -> miette::Result<()> {
    let started = std::time::Instant::now();
    let rows = run_grid(probe, cfg)?;
    eprintln!(
        "bench_context_rot: grid done — {} probes in {:.1}s",
        rows.len(),
        started.elapsed().as_secs_f64()
    );

    let out_dir = PathBuf::from(OUTPUT_DIR);
    emit_runs_csv(&rows, &out_dir).map_err(|e| miette::miette!("emit runs.csv: {e}"))?;
    emit_summary_csv(&rows, cfg, &out_dir)
        .map_err(|e| miette::miette!("emit summary.csv: {e}"))?;

    let summaries = summarize(&rows, cfg);
    println!("\n== Phase 44n.1: context-rot curve ==\n");
    print!("{}", format_table(&summaries));
    println!("\nCSV artefacts: {}", out_dir.display());

    Ok(())
}
