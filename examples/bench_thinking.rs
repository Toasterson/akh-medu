//! Phase 41f scaling-law benchmark runner.
//!
//! Sweeps the Thinking Loop over step budgets {100, 300, 1000, 3000, 10000}
//! on three deterministic task families (multi-hop KG queries at depths
//! 2/4/8/16, VSA analogy, factual evidence queries), then emits CSVs to
//! `target/bench-results/thinking/` and prints a summary table.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example bench_thinking
//! # quick smoke sweep:
//! AKH_BENCH_QUICK=1 cargo run --release --example bench_thinking
//! ```
//!
//! Plan: docs/ai/plans/2026-04-27-phase41-thinking-engine.md (41f)
//! Findings: docs/ai/decisions/085-thinking-engine-findings.md

use std::path::PathBuf;
use std::time::Instant;

use akh_medu::thinking::bench::{
    BudgetMode, OUTPUT_DIR, STEP_BUDGETS, TaskSetConfig, build_engine, build_task_set,
    emit_runs_csv, emit_summary_csv, format_table, run_sweep, summarize,
};

fn main() -> miette::Result<()> {
    miette::set_panic_hook();

    let quick = std::env::var("AKH_BENCH_QUICK").is_ok_and(|v| v != "0");
    let cfg = if quick {
        TaskSetConfig::smoke()
    } else {
        TaskSetConfig::default()
    };
    let budgets: Vec<u32> = if quick {
        vec![100, 300]
    } else {
        STEP_BUDGETS.to_vec()
    };

    eprintln!(
        "phase41f: building engine + task set (seed {})...",
        cfg.seed
    );
    let engine = build_engine().map_err(akh_medu::error::AkhError::from)?;
    let tasks = build_task_set(&engine, &cfg).map_err(akh_medu::error::AkhError::from)?;
    eprintln!(
        "phase41f: {} tasks, budgets {:?}, KG triples: {}",
        tasks.len(),
        budgets,
        engine.knowledge_graph().triple_count()
    );

    // Primary sweep: fixed budgets (the pure scaling-law measurement).
    let started = Instant::now();
    let mut runs = run_sweep(&engine, &tasks, &budgets, BudgetMode::Fixed)
        .map_err(akh_medu::error::AkhError::from)?;
    eprintln!(
        "phase41f: fixed sweep done in {:.1}s ({} runs)",
        started.elapsed().as_secs_f64(),
        runs.len()
    );

    // Secondary sweep: VOC-adaptive budget (Phase 41e economic halting) at
    // the largest budget only — shows what adaptive halting saves/costs.
    let max_budget = *budgets.iter().max().expect("non-empty budgets");
    let adaptive = run_sweep(&engine, &tasks, &[max_budget], BudgetMode::Adaptive)
        .map_err(akh_medu::error::AkhError::from)?;
    eprintln!(
        "phase41f: adaptive sweep done ({} runs, total {:.1}s)",
        adaptive.len(),
        started.elapsed().as_secs_f64()
    );
    runs.extend(adaptive);

    let out_dir = PathBuf::from(OUTPUT_DIR);
    emit_runs_csv(&runs, &out_dir).map_err(|e| miette::miette!("emit runs.csv: {e}"))?;
    let summaries = summarize(&runs);
    emit_summary_csv(&summaries, &out_dir).map_err(|e| miette::miette!("emit summary.csv: {e}"))?;

    println!("\n== Phase 41f: accuracy vs step budget ==\n");
    print!("{}", format_table(&summaries));
    println!("\nCSV artefacts: {}", out_dir.display());

    Ok(())
}
