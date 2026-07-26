//! Cell specifications and results for the Phase 40 wake-path harness.
//!
//! A *cell* is one point in the measurement grid. The parent process spawns
//! one subprocess per cell (so peak-RSS attribution is clean) and the cell
//! prints exactly one JSON [`CellResult`] line on stdout.

use serde::{Deserialize, Serialize};

/// Which rule pack to load. Sizes reflect the real compiled packs in
/// `akh_medu::reason` — there is no synthetic rule padding, because the
/// decision is about *our* rule set, not an imaginary one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum RuleSet {
    /// `builtin_rules()` — 7 rewrites (VSA algebra + logic).
    Small,
    /// builtin + pim + calendar + causal — 15 rewrites.
    Medium,
    /// medium + epistemic — 18 rewrites. The epistemic pack contains
    /// `positive-introspection`, which is deliberately non-terminating
    /// (K(a,p) → K(a,K(a,p))) and always drives the runner into a limit.
    Large,
}

impl RuleSet {
    pub fn label(self) -> &'static str {
        match self {
            RuleSet::Small => "small",
            RuleSet::Medium => "medium",
            RuleSet::Large => "large",
        }
    }
}

/// Runner limit profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Limits {
    /// `egg::Runner::default()` — iter 30, nodes 10 000, time 5 s.
    /// This is byte-for-byte what `Engine::simplify_expression()` uses in
    /// production, so it measures the *actual* wake-path exposure.
    Prod,
    /// iter 1 000, nodes 1 000 000, time 60 s. Answers "what would full
    /// saturation cost if we removed the engine's own guard rails?" —
    /// the number the persistence decision thresholds compare against.
    Generous,
}

impl Limits {
    pub fn label(self) -> &'static str {
        match self {
            Limits::Prod => "prod",
            Limits::Generous => "generous",
        }
    }
}

/// One cell of the measurement grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "arm", rename_all = "snake_case")]
pub enum CellSpec {
    /// E-graph re-saturation over a synthetic theory of `n_facts` facts.
    Egraph {
        n_facts: usize,
        rule_set: RuleSet,
        branching: usize,
        limits: Limits,
        seed: u64,
        reps: usize,
    },
    /// HNSW(VSA) index rebuild over `n_items` 10 000-bit hypervectors.
    Hnsw { n_items: usize, reps: usize },
    /// Q2 workload: fork a saturated e-graph `forks` times with a small
    /// perturbation each (MCTS-style exploration pattern).
    Fork {
        n_facts: usize,
        forks: usize,
        seed: u64,
    },
}

/// Per-repetition e-graph measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgraphRep {
    pub saturate_ms: f64,
    pub extract_ms: f64,
    pub stop_reason: String,
    pub iterations: usize,
    pub e_classes: usize,
    pub e_nodes: usize,
}

/// Result payload for one cell (mirrors [`CellSpec`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "arm", rename_all = "snake_case")]
pub enum CellResult {
    Egraph {
        spec: Box<CellSpec>,
        /// Time to construct the theory `RecExpr` (wake would pay this too).
        build_expr_ms: f64,
        initial_nodes: usize,
        reps: Vec<EgraphRep>,
        peak_rss_mb: f64,
    },
    Hnsw {
        spec: Box<CellSpec>,
        /// Cold build: random-vector generation + first HNSW insertion.
        cold_build_ms: f64,
        /// `ItemMemory::rebuild_hnsw()` — the wake-path rebuild from
        /// already-loaded vectors. One entry per repetition.
        rebuild_ms: Vec<f64>,
        peak_rss_mb: f64,
    },
    Fork {
        spec: Box<CellSpec>,
        base_saturate_ms: f64,
        /// Per-fork cost: clone + perturb + re-run.
        fork_ms: Vec<f64>,
        peak_rss_mb: f64,
    },
}

/// Median of a slice (not-NaN f64s). Returns 0.0 on empty input.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in measurements"));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Peak resident set size of *this* process in MiB, from `/proc/self/status`
/// (`VmHWM`). Linux-only; returns 0.0 elsewhere or on parse failure.
pub fn peak_rss_mb() -> f64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}
