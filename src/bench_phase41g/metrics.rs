//! Metrics aggregation and CSV/plot-script emission.
//!
//! Writes three artefacts to `target/bench-results/prompt-to-plan/`:
//!   - `pipeline_results.csv`   — per-(prompt, budget) raw measurements
//!   - `agreement_matrix.csv`   — per-prompt symbolic vs. Pattern B comparison
//!   - `plot.py`                — matplotlib script that renders the four plots

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;

use super::{BenchResult, PatternBResult};

// ---------------------------------------------------------------------------
// Agreement matrix
// ---------------------------------------------------------------------------

/// One row of the agreement matrix.
#[derive(Debug, Clone, Serialize)]
pub struct AgreementRecord {
    pub prompt_id: String,
    pub category: String,
    pub symbolic_success: bool,
    pub pattern_b_success: bool,
    /// Both agree (both pass or both fail).
    pub agrees: bool,
    /// Symbolic passed, Pattern B failed.
    pub symbolic_wins: bool,
    /// Pattern B passed, symbolic failed.
    pub pattern_b_wins: bool,
}

impl AgreementRecord {
    pub fn from_pair(symbolic: &BenchResult, pb: &PatternBResult) -> Self {
        let agrees = symbolic.success == pb.success;
        Self {
            prompt_id: symbolic.prompt_id.clone(),
            category: symbolic.category.clone(),
            symbolic_success: symbolic.success,
            pattern_b_success: pb.success,
            agrees,
            symbolic_wins: symbolic.success && !pb.success,
            pattern_b_wins: !symbolic.success && pb.success,
        }
    }
}

// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

/// Aggregated statistics for a step-budget bucket.
#[derive(Debug, Clone, Serialize)]
pub struct BudgetSummary {
    pub step_budget: u64,
    pub total: usize,
    pub successes: usize,
    pub accuracy: f64,
    pub mean_wall_ms: f64,
    pub mean_voc: f64,
}

/// Top-level summary over all results.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub by_budget: Vec<BudgetSummary>,
    pub by_category: HashMap<String, BudgetSummary>,
    pub total_runs: usize,
    pub overall_accuracy: f64,
}

impl MetricsSummary {
    pub fn from_results(results: &[BenchResult]) -> Self {
        let total_runs = results.len();
        let overall_accuracy = if total_runs == 0 {
            0.0
        } else {
            results.iter().filter(|r| r.success).count() as f64 / total_runs as f64
        };

        // Group by step budget.
        let mut by_budget_map: HashMap<u64, Vec<&BenchResult>> = HashMap::new();
        for r in results {
            by_budget_map.entry(r.step_budget).or_default().push(r);
        }
        let mut by_budget: Vec<BudgetSummary> = by_budget_map
            .into_iter()
            .map(|(budget, rs)| budget_summary(budget, &rs))
            .collect();
        by_budget.sort_by_key(|b| b.step_budget);

        // Group by category (using max-budget results to avoid double-counting).
        let max_budget = results.iter().map(|r| r.step_budget).max().unwrap_or(0);
        let mut by_cat: HashMap<String, Vec<&BenchResult>> = HashMap::new();
        for r in results.iter().filter(|r| r.step_budget == max_budget) {
            by_cat.entry(r.category.clone()).or_default().push(r);
        }
        let by_category = by_cat
            .into_iter()
            .map(|(cat, rs)| (cat.clone(), budget_summary(max_budget, &rs)))
            .collect();

        Self {
            by_budget,
            by_category,
            total_runs,
            overall_accuracy,
        }
    }
}

fn budget_summary(step_budget: u64, rs: &[&BenchResult]) -> BudgetSummary {
    let total = rs.len();
    let successes = rs.iter().filter(|r| r.success).count();
    let accuracy = if total == 0 { 0.0 } else { successes as f64 / total as f64 };
    let mean_wall_ms = rs.iter().map(|r| r.wall_ms as f64).sum::<f64>() / total.max(1) as f64;
    let mean_voc = rs.iter().map(|r| r.voc_consumed as f64).sum::<f64>() / total.max(1) as f64;
    BudgetSummary { step_budget, total, successes, accuracy, mean_wall_ms, mean_voc }
}

// ---------------------------------------------------------------------------
// CSV emission
// ---------------------------------------------------------------------------

/// Write raw pipeline results to `output_dir/pipeline_results.csv`.
pub fn emit_csv(results: &[BenchResult], output_dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join("pipeline_results.csv");
    let mut f = std::fs::File::create(&path)?;
    writeln!(
        f,
        "prompt_id,category,step_budget,success,wall_ms,voc_consumed,halt_reason,steps_taken"
    )?;
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            r.prompt_id,
            r.category,
            r.step_budget,
            r.success,
            r.wall_ms,
            r.voc_consumed,
            r.halt_reason.as_str(),
            r.steps_taken,
        )?;
    }
    Ok(())
}

/// Write agreement matrix to `output_dir/agreement_matrix.csv`.
pub fn emit_agreement_csv(records: &[AgreementRecord], output_dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join("agreement_matrix.csv");
    let mut f = std::fs::File::create(&path)?;
    writeln!(
        f,
        "prompt_id,category,symbolic_success,pattern_b_success,agrees,symbolic_wins,pattern_b_wins"
    )?;
    for r in records {
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            r.prompt_id,
            r.category,
            r.symbolic_success,
            r.pattern_b_success,
            r.agrees,
            r.symbolic_wins,
            r.pattern_b_wins,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Plot script emission
// ---------------------------------------------------------------------------

/// Write `output_dir/plot.py` — a self-contained matplotlib script that reads
/// the two CSVs and renders the four Phase 41g plots as PNGs.
pub fn emit_plot_script(output_dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let script = r#"#!/usr/bin/env python3
"""Phase 41g benchmark plots — run after the bench harness emits CSVs."""
import csv, os, sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("matplotlib and numpy required: pip install matplotlib numpy")

HERE = Path(__file__).parent

# ── load pipeline_results.csv ──────────────────────────────────────────────
with open(HERE / "pipeline_results.csv") as f:
    rows = list(csv.DictReader(f))

budgets = sorted({int(r["step_budget"]) for r in rows})
cats = sorted({r["category"] for r in rows})


def accuracy(subset):
    if not subset:
        return 0.0
    return sum(1 for r in subset if r["success"] == "true") / len(subset)


# ── Plot 1: accuracy vs step budget ───────────────────────────────────────
fig, ax = plt.subplots()
for cat in cats:
    xs = budgets
    ys = [accuracy([r for r in rows if r["category"] == cat and int(r["step_budget"]) == b]) for b in xs]
    ax.semilogx(xs, ys, marker="o", label=cat)
ax.set_xlabel("Step budget")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs step budget (Phase 41g)")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_budget.png", dpi=150)
plt.close(fig)

# ── Plot 2: accuracy vs wall-clock time ───────────────────────────────────
fig, ax = plt.subplots()
for cat in cats:
    pts = [(float(r["wall_ms"]), r["success"] == "true")
           for r in rows if r["category"] == cat]
    pts.sort()
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [float(p[1]) for p in pts]
    # Rolling mean over sorted wall_ms.
    window = max(1, len(ys) // 5)
    smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
    ax.plot(xs[:len(smoothed)], smoothed, label=cat)
ax.set_xlabel("Wall time (ms)")
ax.set_ylabel("Accuracy (smoothed)")
ax.set_title("Accuracy vs wall-clock time (Phase 41g)")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_wall.png", dpi=150)
plt.close(fig)

# ── Plot 3: accuracy vs VOC ───────────────────────────────────────────────
fig, ax = plt.subplots()
for cat in cats:
    pts = [(int(r["voc_consumed"]), r["success"] == "true")
           for r in rows if r["category"] == cat]
    pts.sort()
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [float(p[1]) for p in pts]
    window = max(1, len(ys) // 5)
    smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
    ax.plot(xs[:len(smoothed)], smoothed, label=cat)
ax.set_xlabel("VOC (computation units)")
ax.set_ylabel("Accuracy (smoothed)")
ax.set_title("Accuracy vs VOC consumed (Phase 41g)")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "accuracy_vs_voc.png", dpi=150)
plt.close(fig)

# ── Plot 4: agreement matrix heatmap ─────────────────────────────────────
agree_path = HERE / "agreement_matrix.csv"
if agree_path.exists():
    with open(agree_path) as f:
        agree_rows = list(csv.DictReader(f))
    prompt_ids = [r["prompt_id"] for r in agree_rows]
    sym_ok = [int(r["symbolic_success"] == "true") for r in agree_rows]
    pb_ok = [int(r["pattern_b_success"] == "true") for r in agree_rows]
    matrix = np.array([sym_ok, pb_ok])
    fig, ax = plt.subplots(figsize=(max(8, len(prompt_ids) * 0.4), 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Symbolic", "Pattern B"])
    ax.set_xticks(range(len(prompt_ids)))
    ax.set_xticklabels(prompt_ids, rotation=90, fontsize=7)
    ax.set_title("Agreement matrix: symbolic vs Pattern B (Phase 41g)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(HERE / "agreement_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Agreement matrix saved to {HERE / 'agreement_matrix.png'}")
else:
    print("agreement_matrix.csv not found — skipping plot 4")

print("Plots written to", HERE)
"#;
    let path = output_dir.join("plot.py");
    let mut f = std::fs::File::create(&path)?;
    f.write_all(script.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_phase41g::{BenchResult, HaltReason};

    fn make_result(id: &str, budget: u64, success: bool, wall_ms: u64, voc: u64) -> BenchResult {
        BenchResult {
            prompt_id: id.into(),
            category: "code-add".into(),
            step_budget: budget,
            success,
            wall_ms,
            voc_consumed: voc,
            halt_reason: HaltReason::PlanCompleted,
            steps_taken: 3,
            output_text: String::new(),
        }
    }

    #[test]
    fn summary_accuracy_correct() {
        let results = vec![
            make_result("a", 100, true, 5, 30),
            make_result("b", 100, false, 4, 20),
            make_result("c", 100, true, 6, 25),
        ];
        let summary = MetricsSummary::from_results(&results);
        assert!((summary.overall_accuracy - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn emit_csv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let results = vec![make_result("x", 300, true, 10, 50)];
        emit_csv(&results, dir.path()).unwrap();
        let content = std::fs::read_to_string(dir.path().join("pipeline_results.csv")).unwrap();
        assert!(content.contains("prompt_id"));
        assert!(content.contains("x,code-add,300,true"));
    }

    #[test]
    fn emit_plot_script_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        emit_plot_script(dir.path()).unwrap();
        assert!(dir.path().join("plot.py").exists());
    }
}
