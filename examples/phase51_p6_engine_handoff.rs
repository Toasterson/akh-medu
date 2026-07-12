//! Phase 51 P6 slice 2 -- the dual-process handoff's *economics* measurement.
//!
//! ADR 093 slice 1 (`grug_data::reason::handoff`) is a lab stand-in: it
//! measures the *mechanics* of splicing a neural trace prefix onto a
//! symbolic continuation, but its "System-2" is a deterministic chain
//! follower over the example's own prompt facts, so it always solves --
//! the lab cannot show what deliberation *costs*. This example replaces
//! that lab continuation with the real Phase-41 `ThinkingLoop`
//! (`Engine::deliberate`) over real KG triples, so the answer to "how much
//! System-2 budget does the neural prefix save" can actually be measured.
//!
//! Topology is file-based, matching the organ-offline / engine-online split
//! the ADR calls out: `grug-train`'s `eval_handoff --dump-traces` writes one
//! JSON line per scored example (gold chain + the neural/symbolic splice
//! point `handoff` found); this example reads that file, rebuilds each
//! chain as a real isolated sub-graph of triples in one shared [`Engine`],
//! and deliberates twice per record -- once from the chain start
//! (`goal_full`, what a pure-System-2 run would have to do) and once from
//! the splice entity (`goal_splice`, what System-2 has to do *after* the
//! neural prefix already resolved `neural_hops` of the chain) -- across a
//! budget sweep. The gap between the two solve-rate curves *is* the
//! neural prefix's economic value.
//!
//! See `docs/ai/decisions/093-dual-process-trace-handoff.md` for the design
//! and the lab<->production mapping table this example fills in the right
//! column of.
//!
//! # Run it
//!
//! ```sh
//! # 1. Dump traces from a trained checkpoint (grug-train, GPU):
//! cargo run -p grug-train --example eval_handoff --features gpu --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace/checkpoints/step_00019999 \
//!     --train-depths 2,3,4,5,6 --axis-a-depths 7,8 \
//!     --dump-traces /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace/handoff_traces.jsonl
//!
//! # 2. Measure System-2 budget economics over those traces (CPU only):
//! cargo run --example phase51_p6_engine_handoff -- \
//!     --traces /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace/handoff_traces.jsonl
//! ```

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use akh_medu::engine::Engine;
use akh_medu::error::AkhError;
use akh_medu::graph::Triple;
use akh_medu::symbol::SymbolId;
use akh_medu::thinking::bench::build_engine;
use akh_medu::thinking::{ThinkingBudget, ThinkingPredicates};

use clap::Parser;
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "phase51-p6-engine-handoff",
    about = "ADR 093 slice 2: measure System-2 deliberation budget saved by a neural trace prefix, over real KG triples"
)]
struct Cli {
    /// JSONL trace dump from `eval_handoff --dump-traces` (grug-train).
    #[arg(long)]
    traces: PathBuf,

    /// Comma-separated `ThinkingBudget` step-count sweep points.
    #[arg(long, default_value = "60,120,240")]
    budgets: String,

    /// Dead-end chaff branches per chain hop (mirrors
    /// `TaskSetConfig::branching`'s default).
    #[arg(long, default_value_t = 3)]
    branching: usize,

    /// Children per chaff dead-end node (mirrors
    /// `TaskSetConfig::sub_branching`'s default).
    #[arg(long, default_value_t = 2)]
    sub_branching: usize,

    /// Report JSON path. Defaults to `<traces>` with the file name
    /// `<stem>.p6_engine_handoff.json` alongside it.
    #[arg(long)]
    out: Option<PathBuf>,
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
enum HandoffBenchError {
    #[error("failed to read trace file {path}: {source}")]
    #[diagnostic(
        code(phase51_p6::io),
        help("Check --traces points at a file produced by `eval_handoff --dump-traces`.")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} line {line} is not a valid trace record: {source}")]
    #[diagnostic(
        code(phase51_p6::bad_json),
        help(
            "Each line must be one JSON object matching eval_handoff's TraceDumpRecord shape. Regenerate the dump if it was hand-edited."
        )
    )]
    BadJson {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("--budgets must be a comma-separated list of positive integers, got {value:?}")]
    #[diagnostic(code(phase51_p6::bad_budgets), help("Example: --budgets 60,120,240"))]
    BadBudgets { value: String },

    #[error(
        "record {id:?}: chain has {chain_len} entities but depth is {depth} (expected {expected})"
    )]
    #[diagnostic(
        code(phase51_p6::depth_chain_mismatch),
        help(
            "chain.len() must equal depth + 1. Regenerate the dump -- this file looks corrupted or hand-edited."
        )
    )]
    DepthChainMismatch {
        id: String,
        depth: usize,
        chain_len: usize,
        expected: usize,
    },

    #[error(
        "record {id:?}: splice_entity {splice:?} does not equal chain[neural_hops={neural_hops}] ({chain:?})"
    )]
    #[diagnostic(
        code(phase51_p6::bad_splice),
        help(
            "splice_entity must be exactly chain[neural_hops] per ADR 093's handoff() contract. Regenerate the dump."
        )
    )]
    BadSplice {
        id: String,
        splice: String,
        neural_hops: usize,
        chain: Vec<String>,
    },

    #[error("abstention record {id:?} has no broken_hop")]
    #[diagnostic(
        code(phase51_p6::missing_broken_hop),
        help(
            "Abstention records must carry Some(broken_hop) -- check is_abstention wasn't set without it upstream."
        )
    )]
    MissingBrokenHop { id: String },

    #[error("failed to write report to {path}: {source}")]
    #[diagnostic(code(phase51_p6::out_io))]
    OutIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] AkhError),
}

type BenchResult<T> = Result<T, HandoffBenchError>;

// ---------------------------------------------------------------------
// Trace records (mirrors grug-train's `eval_handoff::TraceDumpRecord`)
// ---------------------------------------------------------------------

/// One line of an `eval_handoff --dump-traces` JSONL file.
#[derive(Debug, Clone, Deserialize)]
struct TraceRecord {
    id: String,
    #[allow(dead_code)] // carried through for --out provenance, not filtered on
    axis: String,
    depth: usize,
    chain: Vec<String>,
    neural_hops: usize,
    splice_entity: String,
    #[allow(dead_code)]
    gold_answer: String,
    is_abstention: bool,
    broken_hop: Option<usize>,
    organ_answer_correct: bool,
}

// ---------------------------------------------------------------------
// CLI plumbing
// ---------------------------------------------------------------------

fn parse_budgets(s: &str) -> BenchResult<Vec<u32>> {
    let budgets: Option<Vec<u32>> = s
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().ok().filter(|&n| n > 0))
        .collect();
    match budgets {
        Some(b) if !b.is_empty() => Ok(b),
        _ => Err(HandoffBenchError::BadBudgets {
            value: s.to_string(),
        }),
    }
}

fn load_traces(path: &Path) -> BenchResult<Vec<TraceRecord>> {
    let file = File::open(path).map_err(|source| HandoffBenchError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| HandoffBenchError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: TraceRecord =
            serde_json::from_str(trimmed).map_err(|source| HandoffBenchError::BadJson {
                path: path.to_path_buf(),
                line: idx + 1,
                source,
            })?;
        records.push(record);
    }
    Ok(records)
}

fn default_out_path(traces: &Path) -> PathBuf {
    let stem = traces
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("traces");
    traces.with_file_name(format!("{stem}.p6_engine_handoff.json"))
}

// ---------------------------------------------------------------------
// Sub-graph construction
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ChaffConfig {
    branching: usize,
    sub_branching: usize,
}

/// The entities + goals built for one trace record's isolated sub-graph.
struct RecordGraph {
    final_entity: SymbolId,
    goal_full: SymbolId,
    goal_splice: SymbolId,
    /// `true` when the splice point is already the final entity (organ
    /// resolved the whole chain, or its own answer was already correct) --
    /// the splice condition needs no deliberation at all.
    fully_neural: bool,
}

/// Confirm `record.splice_entity == record.chain[record.neural_hops]`
/// (the contract `grug_data::reason::handoff` guarantees) and return that
/// index -- cheap integrity check against a hand-edited or stale dump.
fn splice_chain_index(record: &TraceRecord) -> BenchResult<usize> {
    match record.chain.get(record.neural_hops) {
        Some(label) if *label == record.splice_entity => Ok(record.neural_hops),
        _ => Err(HandoffBenchError::BadSplice {
            id: record.id.clone(),
            splice: record.splice_entity.clone(),
            neural_hops: record.neural_hops,
            chain: record.chain.clone(),
        }),
    }
}

fn build_goal(
    engine: &Engine,
    preds: &ThinkingPredicates,
    label: &str,
    seeks: SymbolId,
    relation: SymbolId,
    answer_type: SymbolId,
) -> BenchResult<SymbolId> {
    let goal = engine.resolve_or_create_entity(label)?;
    engine.add_triple(&Triple::new(goal, preds.seeks, seeks))?;
    engine.add_triple(&Triple::new(goal, preds.relation, relation))?;
    engine.add_triple(&Triple::new(goal, preds.answer_type, answer_type))?;
    Ok(goal)
}

/// Build one record's isolated sub-graph: chain entities + edges (bench-style
/// dead-end chaff per hop, matching `thinking::bench::build_multi_hop`),
/// final `is_a` typing, and both goals (`goal_full`/`goal_splice`).
///
/// `skip_hop`, when set (1-indexed), omits the real chain edge
/// `chain_nodes[skip_hop - 1] -> chain_nodes[skip_hop]` -- used for
/// abstention records, where that hop's fact was never established, so the
/// chain is genuinely broken past that point.
fn build_record_graph(
    engine: &Engine,
    preds: &ThinkingPredicates,
    record: &TraceRecord,
    chaff: ChaffConfig,
    skip_hop: Option<usize>,
) -> BenchResult<RecordGraph> {
    let expected_len = record.depth + 1;
    if record.chain.len() != expected_len {
        return Err(HandoffBenchError::DepthChainMismatch {
            id: record.id.clone(),
            depth: record.depth,
            chain_len: record.chain.len(),
            expected: expected_len,
        });
    }
    let splice_index = splice_chain_index(record)?;

    let relation = engine.resolve_or_create_relation(&format!("p6:{}:path", record.id))?;
    let target_type = engine.resolve_or_create_entity(&format!("p6:{}:target", record.id))?;

    let mut chain_nodes = Vec::with_capacity(record.chain.len());
    for label in &record.chain {
        chain_nodes.push(engine.resolve_or_create_entity(&format!("p6:{}:{label}", record.id))?);
    }

    for hop in 1..=record.depth {
        for j in 0..chaff.branching {
            let dead =
                engine.resolve_or_create_entity(&format!("p6:{}:dead{hop}-{j}", record.id))?;
            engine.add_triple(
                &Triple::new(chain_nodes[hop - 1], relation, dead).with_confidence(0.95),
            )?;
            for k in 0..chaff.sub_branching {
                let sub = engine
                    .resolve_or_create_entity(&format!("p6:{}:sub{hop}-{j}-{k}", record.id))?;
                engine.add_triple(&Triple::new(dead, relation, sub).with_confidence(0.95))?;
            }
        }
        if skip_hop != Some(hop) {
            engine.add_triple(
                &Triple::new(chain_nodes[hop - 1], relation, chain_nodes[hop])
                    .with_confidence(0.95),
            )?;
        }
    }

    let final_entity = *chain_nodes
        .last()
        .expect("chain.len() == depth + 1 >= 1, checked above");
    engine.add_triple(&Triple::new(final_entity, preds.is_a, target_type))?;

    let fully_neural = record.neural_hops >= record.depth || record.organ_answer_correct;

    let goal_full = build_goal(
        engine,
        preds,
        &format!("p6:{}:goal_full", record.id),
        chain_nodes[0],
        relation,
        target_type,
    )?;
    let goal_splice = build_goal(
        engine,
        preds,
        &format!("p6:{}:goal_splice", record.id),
        chain_nodes[splice_index],
        relation,
        target_type,
    )?;

    Ok(RecordGraph {
        final_entity,
        goal_full,
        goal_splice,
        fully_neural,
    })
}

// ---------------------------------------------------------------------
// Deliberation
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize)]
struct ConditionResult {
    solved: bool,
    steps_used: u32,
}

fn run_condition(
    engine: &Engine,
    goal: SymbolId,
    expected: SymbolId,
    budget_steps: u32,
) -> BenchResult<ConditionResult> {
    let conclusion = engine.deliberate(goal, ThinkingBudget::steps(budget_steps))?;
    Ok(ConditionResult {
        solved: conclusion.answer_symbol == Some(expected),
        steps_used: conclusion.steps_used,
    })
}

/// One scored (record, budget) row for the main full-vs-splice comparison.
#[derive(Debug, Clone)]
struct RecordBudgetResult {
    depth: usize,
    neural_hops: usize,
    budget: u32,
    full: ConditionResult,
    splice: ConditionResult,
}

// ---------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct DepthBudgetStats {
    depth: usize,
    budget: u32,
    n: usize,
    full_solve_rate: f64,
    splice_solve_rate: f64,
    full_mean_steps: f64,
    splice_mean_steps: f64,
    mean_steps_saved: f64,
    /// Mean of `(full.steps_used - splice.steps_used) / neural_hops`, over
    /// records with `neural_hops > 0` only (division by neural hops is
    /// undefined at zero) -- `n_with_neural_hops` reports how many that was.
    mean_steps_saved_per_neural_hop: f64,
    n_with_neural_hops: usize,
}

fn aggregate_by_depth_budget(scored: &[RecordBudgetResult]) -> Vec<DepthBudgetStats> {
    let depths: BTreeSet<usize> = scored.iter().map(|r| r.depth).collect();
    let budgets: BTreeSet<u32> = scored.iter().map(|r| r.budget).collect();
    let mut out = Vec::new();
    for depth in depths {
        for &budget in &budgets {
            let rows: Vec<&RecordBudgetResult> = scored
                .iter()
                .filter(|r| r.depth == depth && r.budget == budget)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let n = rows.len();
            let nf = n as f64;
            let full_solve_rate = rows.iter().filter(|r| r.full.solved).count() as f64 / nf;
            let splice_solve_rate = rows.iter().filter(|r| r.splice.solved).count() as f64 / nf;
            let full_mean_steps = rows.iter().map(|r| r.full.steps_used as f64).sum::<f64>() / nf;
            let splice_mean_steps =
                rows.iter().map(|r| r.splice.steps_used as f64).sum::<f64>() / nf;
            let mean_steps_saved = rows
                .iter()
                .map(|r| r.full.steps_used as f64 - r.splice.steps_used as f64)
                .sum::<f64>()
                / nf;

            let with_hops: Vec<&&RecordBudgetResult> =
                rows.iter().filter(|r| r.neural_hops > 0).collect();
            let n_with_neural_hops = with_hops.len();
            let mean_steps_saved_per_neural_hop = if n_with_neural_hops == 0 {
                0.0
            } else {
                with_hops
                    .iter()
                    .map(|r| {
                        (r.full.steps_used as f64 - r.splice.steps_used as f64)
                            / r.neural_hops as f64
                    })
                    .sum::<f64>()
                    / n_with_neural_hops as f64
            };

            out.push(DepthBudgetStats {
                depth,
                budget,
                n,
                full_solve_rate,
                splice_solve_rate,
                full_mean_steps,
                splice_mean_steps,
                mean_steps_saved,
                mean_steps_saved_per_neural_hop,
                n_with_neural_hops,
            });
        }
    }
    out
}

/// The headline: per depth, the smallest swept budget at which each
/// condition's solve rate reaches 0.95, and the gap between them (positive
/// = the neural prefix buys back that much budget).
#[derive(Debug, Clone, Serialize)]
struct DepthHeadline {
    depth: usize,
    full_crossover_budget: Option<u32>,
    splice_crossover_budget: Option<u32>,
    budget_bought_back: Option<i64>,
}

const CROSSOVER_THRESHOLD: f64 = 0.95;

fn headline(per_depth_budget: &[DepthBudgetStats]) -> Vec<DepthHeadline> {
    let depths: BTreeSet<usize> = per_depth_budget.iter().map(|s| s.depth).collect();
    depths
        .into_iter()
        .map(|depth| {
            let mut rows: Vec<&DepthBudgetStats> = per_depth_budget
                .iter()
                .filter(|s| s.depth == depth)
                .collect();
            rows.sort_by_key(|s| s.budget);
            let full_crossover_budget = rows
                .iter()
                .find(|s| s.full_solve_rate >= CROSSOVER_THRESHOLD)
                .map(|s| s.budget);
            let splice_crossover_budget = rows
                .iter()
                .find(|s| s.splice_solve_rate >= CROSSOVER_THRESHOLD)
                .map(|s| s.budget);
            let budget_bought_back = match (full_crossover_budget, splice_crossover_budget) {
                (Some(f), Some(s)) => Some(f as i64 - s as i64),
                _ => None,
            };
            DepthHeadline {
                depth,
                full_crossover_budget,
                splice_crossover_budget,
                budget_bought_back,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct AbstentionSafety {
    n: usize,
    budget: u32,
    false_answer_count: usize,
    false_answer_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    traces_path: PathBuf,
    branching: usize,
    sub_branching: usize,
    budgets: Vec<u32>,
    n_records_scored: usize,
    n_records_abstention: usize,
    per_depth_budget: Vec<DepthBudgetStats>,
    headline: Vec<DepthHeadline>,
    abstention: AbstentionSafety,
}

// ---------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------

fn print_report(report: &Report) {
    println!(
        "\n=== Phase 51 P6 slice 2 -- engine handoff economics ({} scored, {} abstention) ===",
        report.n_records_scored, report.n_records_abstention
    );
    println!(
        "{:<6} {:>8} {:>5} {:>10} {:>12} {:>10} {:>12} {:>12} {:>14}",
        "depth",
        "budget",
        "n",
        "full_slv",
        "splice_slv",
        "full_stp",
        "splice_stp",
        "saved_stp",
        "saved/hop"
    );
    for s in &report.per_depth_budget {
        println!(
            "{:<6} {:>8} {:>5} {:>9.0}% {:>11.0}% {:>10.1} {:>12.1} {:>12.1} {:>14.2}",
            s.depth,
            s.budget,
            s.n,
            s.full_solve_rate * 100.0,
            s.splice_solve_rate * 100.0,
            s.full_mean_steps,
            s.splice_mean_steps,
            s.mean_steps_saved,
            s.mean_steps_saved_per_neural_hop,
        );
    }

    println!(
        "\n-- headline: smallest budget reaching {:.0}% solve rate --",
        CROSSOVER_THRESHOLD * 100.0
    );
    for h in &report.headline {
        println!(
            "  depth {}: full-start={} splice-start={} budget bought back={}",
            h.depth,
            h.full_crossover_budget
                .map(|b| b.to_string())
                .unwrap_or_else(|| "never".to_string()),
            h.splice_crossover_budget
                .map(|b| b.to_string())
                .unwrap_or_else(|| "never".to_string()),
            h.budget_bought_back
                .map(|b| b.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }

    println!(
        "\n-- abstention safety (budget={}) --\n  n={} false_answer_count={} false_answer_rate={:.3}",
        report.abstention.budget,
        report.abstention.n,
        report.abstention.false_answer_count,
        report.abstention.false_answer_rate,
    );
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

fn run(args: Cli) -> BenchResult<Report> {
    let budgets = parse_budgets(&args.budgets)?;
    let max_budget = *budgets
        .iter()
        .max()
        .expect("parse_budgets guarantees a non-empty list");
    let chaff = ChaffConfig {
        branching: args.branching,
        sub_branching: args.sub_branching,
    };

    let records = load_traces(&args.traces)?;
    eprintln!(
        "phase51_p6_engine_handoff: loaded {} trace records from {}",
        records.len(),
        args.traces.display()
    );

    let engine = build_engine().map_err(AkhError::from)?;
    let preds = ThinkingPredicates::resolve(&engine).map_err(AkhError::from)?;

    let mut scored: Vec<RecordBudgetResult> = Vec::new();
    let mut abstention_false_answers = 0usize;
    let mut abstention_n = 0usize;

    for record in &records {
        if record.is_abstention {
            let Some(broken_hop) = record.broken_hop else {
                return Err(HandoffBenchError::MissingBrokenHop {
                    id: record.id.clone(),
                });
            };
            let graph = build_record_graph(&engine, &preds, record, chaff, Some(broken_hop))?;
            let conclusion =
                engine.deliberate(graph.goal_full, ThinkingBudget::steps(max_budget))?;
            abstention_n += 1;
            if conclusion.answer_symbol == Some(graph.final_entity) {
                abstention_false_answers += 1;
            }
            continue;
        }

        let graph = build_record_graph(&engine, &preds, record, chaff, None)?;
        for &budget in &budgets {
            let full = run_condition(&engine, graph.goal_full, graph.final_entity, budget)?;
            let splice = if graph.fully_neural {
                ConditionResult {
                    solved: true,
                    steps_used: 0,
                }
            } else {
                run_condition(&engine, graph.goal_splice, graph.final_entity, budget)?
            };
            scored.push(RecordBudgetResult {
                depth: record.depth,
                neural_hops: record.neural_hops,
                budget,
                full,
                splice,
            });
        }
    }

    let per_depth_budget = aggregate_by_depth_budget(&scored);
    let headline = headline(&per_depth_budget);
    let n_records_scored = scored.len() / budgets.len().max(1);

    Ok(Report {
        traces_path: args.traces.clone(),
        branching: args.branching,
        sub_branching: args.sub_branching,
        budgets,
        n_records_scored,
        n_records_abstention: abstention_n,
        per_depth_budget,
        headline,
        abstention: AbstentionSafety {
            n: abstention_n,
            budget: max_budget,
            false_answer_count: abstention_false_answers,
            false_answer_rate: if abstention_n == 0 {
                0.0
            } else {
                abstention_false_answers as f64 / abstention_n as f64
            },
        },
    })
}

fn main() -> miette::Result<()> {
    miette::set_panic_hook();
    let args = Cli::parse();
    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_out_path(&args.traces));

    let report = run(args)?;
    print_report(&report);

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| miette::miette!("serializing report: {e}"))?;
    std::fs::write(&out_path, json).map_err(|source| HandoffBenchError::OutIo {
        path: out_path.clone(),
        source,
    })?;
    println!("\nphase51_p6_engine_handoff: wrote {}", out_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_budgets_accepts_a_clean_list() {
        assert_eq!(parse_budgets("60,120,240").unwrap(), vec![60, 120, 240]);
        assert_eq!(parse_budgets(" 10 , 20 ").unwrap(), vec![10, 20]);
    }

    #[test]
    fn parse_budgets_rejects_garbage_and_empty() {
        assert!(parse_budgets("").is_err());
        assert!(parse_budgets("abc").is_err());
        assert!(parse_budgets("0,10").is_err());
    }

    fn sample_record() -> TraceRecord {
        TraceRecord {
            id: "axis_a-0".to_string(),
            axis: "axis_a".to_string(),
            depth: 3,
            chain: vec![
                "p000".to_string(),
                "c001".to_string(),
                "r002".to_string(),
                "k003".to_string(),
            ],
            neural_hops: 2,
            splice_entity: "r002".to_string(),
            gold_answer: "k003".to_string(),
            is_abstention: false,
            broken_hop: None,
            organ_answer_correct: false,
        }
    }

    #[test]
    fn splice_chain_index_matches_neural_hops() {
        let record = sample_record();
        assert_eq!(splice_chain_index(&record).unwrap(), 2);
    }

    #[test]
    fn splice_chain_index_rejects_mismatch() {
        let mut record = sample_record();
        record.splice_entity = "k003".to_string();
        assert!(splice_chain_index(&record).is_err());
    }

    #[test]
    fn build_record_graph_and_deliberate_full_vs_splice() {
        let engine = build_engine().unwrap();
        let preds = ThinkingPredicates::resolve(&engine).unwrap();
        let chaff = ChaffConfig {
            branching: 2,
            sub_branching: 1,
        };
        let record = sample_record();
        let graph = build_record_graph(&engine, &preds, &record, chaff, None).unwrap();
        assert!(!graph.fully_neural);

        let full = run_condition(&engine, graph.goal_full, graph.final_entity, 200).unwrap();
        assert!(
            full.solved,
            "full-start should solve a depth-3 chain within budget 200"
        );

        let splice = run_condition(&engine, graph.goal_splice, graph.final_entity, 200).unwrap();
        assert!(
            splice.solved,
            "splice-start should solve the remaining 1 hop within budget 200"
        );
        assert!(
            splice.steps_used <= full.steps_used,
            "splice-start should never need more steps than full-start: splice={} full={}",
            splice.steps_used,
            full.steps_used
        );
    }

    #[test]
    fn fully_neural_record_skips_splice_deliberation() {
        let mut record = sample_record();
        record.neural_hops = record.depth;
        record.splice_entity = record.chain.last().unwrap().clone();
        assert!(record.neural_hops >= record.depth || record.organ_answer_correct);
    }

    #[test]
    fn abstention_record_breaks_the_chain_at_the_right_hop() {
        let engine = build_engine().unwrap();
        let preds = ThinkingPredicates::resolve(&engine).unwrap();
        let chaff = ChaffConfig {
            branching: 2,
            sub_branching: 1,
        };
        let mut record = sample_record();
        record.is_abstention = true;
        record.broken_hop = Some(2); // breaks chain[1] -> chain[2]

        let graph = build_record_graph(&engine, &preds, &record, chaff, record.broken_hop).unwrap();
        let conclusion = engine
            .deliberate(graph.goal_full, ThinkingBudget::steps(300))
            .unwrap();
        assert_ne!(
            conclusion.answer_symbol,
            Some(graph.final_entity),
            "the engine must not find the final entity when its path is broken"
        );
    }
}
