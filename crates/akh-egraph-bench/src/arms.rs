//! The three measurement arms: e-graph re-saturation, HNSW(VSA) rebuild,
//! and the Q2 fork workload. Each runs inside a dedicated subprocess so
//! `VmHWM` peak-RSS attribution is per-cell.

use std::time::{Duration, Instant};

use akh_medu::reason::{
    AkhLang, builtin_rules, calendar_rules, causal_rules, epistemic_rules, pim_rules,
};
use akh_medu::simd;
use akh_medu::symbol::SymbolId;
use akh_medu::vsa::item_memory::ItemMemory;
use akh_medu::vsa::ops::VsaOps;
use akh_medu::vsa::{Dimension, Encoding};
use egg::{AstSize, Extractor, RecExpr, Rewrite, Runner};

use crate::spec::{CellResult, CellSpec, EgraphRep, Limits, RuleSet, peak_rss_mb};
use crate::theory::build_theory;

/// Materialize a rule pack. Mirrors what `Engine::all_rules()` can load —
/// no synthetic rule padding.
pub fn rule_pack(set: RuleSet) -> Vec<Rewrite<AkhLang, ()>> {
    let mut rules = builtin_rules();
    if matches!(set, RuleSet::Medium | RuleSet::Large) {
        rules.extend(pim_rules());
        rules.extend(calendar_rules());
        rules.extend(causal_rules());
    }
    if matches!(set, RuleSet::Large) {
        rules.extend(epistemic_rules());
    }
    rules
}

fn make_runner(limits: Limits) -> Runner<AkhLang, ()> {
    match limits {
        // Byte-for-byte the production configuration used by
        // `Engine::simplify_expression()` (src/engine.rs).
        Limits::Prod => Runner::default(),
        Limits::Generous => Runner::default()
            .with_iter_limit(1_000)
            .with_node_limit(1_000_000)
            .with_time_limit(Duration::from_secs(60)),
    }
}

fn short_stop_reason(r: &egg::StopReason) -> String {
    match r {
        egg::StopReason::Saturated => "Saturated".into(),
        egg::StopReason::IterationLimit(n) => format!("IterationLimit({n})"),
        egg::StopReason::NodeLimit(n) => format!("NodeLimit({n})"),
        egg::StopReason::TimeLimit(s) => format!("TimeLimit({s:.1}s)"),
        egg::StopReason::Other(s) => format!("Other({s})"),
    }
}

/// Run one cell and return its result.
pub fn run_cell(spec: &CellSpec) -> CellResult {
    match spec {
        CellSpec::Egraph {
            n_facts,
            rule_set,
            branching,
            limits,
            seed,
            reps,
        } => run_egraph(spec, *n_facts, *rule_set, *branching, *limits, *seed, *reps),
        CellSpec::Hnsw { n_items, reps } => run_hnsw(spec, *n_items, *reps),
        CellSpec::Fork {
            n_facts,
            forks,
            seed,
        } => run_fork(spec, *n_facts, *forks, *seed),
    }
}

fn run_egraph(
    spec: &CellSpec,
    n_facts: usize,
    rule_set: RuleSet,
    branching: usize,
    limits: Limits,
    seed: u64,
    reps: usize,
) -> CellResult {
    let rules = rule_pack(rule_set);

    let t0 = Instant::now();
    let expr = build_theory(n_facts, branching, seed);
    let build_expr_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    let initial_nodes = expr.len();

    let mut rep_results = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let runner = make_runner(limits).with_expr(&expr).run(&rules);
        let saturate_ms = t.elapsed().as_secs_f64() * 1_000.0;

        let t = Instant::now();
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let _best = extractor.find_best(runner.roots[0]);
        let extract_ms = t.elapsed().as_secs_f64() * 1_000.0;

        rep_results.push(EgraphRep {
            saturate_ms,
            extract_ms,
            stop_reason: runner
                .stop_reason
                .as_ref()
                .map(short_stop_reason)
                .unwrap_or_else(|| "None".into()),
            iterations: runner.iterations.len(),
            e_classes: runner.egraph.number_of_classes(),
            e_nodes: runner.egraph.total_number_of_nodes(),
        });
    }

    CellResult::Egraph {
        spec: Box::new(spec.clone()),
        build_expr_ms,
        initial_nodes,
        reps: rep_results,
        peak_rss_mb: peak_rss_mb(),
    }
}

fn run_hnsw(spec: &CellSpec, n_items: usize, reps: usize) -> CellResult {
    let ops = VsaOps::new(simd::best_kernel(), Dimension::DEFAULT, Encoding::Bipolar);
    let mem = ItemMemory::new(Dimension::DEFAULT, Encoding::Bipolar, n_items);
    let syms: Vec<SymbolId> = (1..=n_items as u64)
        .map(|i| SymbolId::new(i).expect("non-zero id"))
        .collect();

    // Cold build: vector generation + first HNSW insertion. This is the
    // "fresh akh" path; the wake path is the rebuild below (vectors already
    // in memory, HNSW re-derived).
    let t = Instant::now();
    mem.insert_batch(&ops, &syms);
    let cold_build_ms = t.elapsed().as_secs_f64() * 1_000.0;

    let mut rebuild_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        mem.rebuild_hnsw();
        rebuild_ms.push(t.elapsed().as_secs_f64() * 1_000.0);
    }

    CellResult::Hnsw {
        spec: Box::new(spec.clone()),
        cold_build_ms,
        rebuild_ms,
        peak_rss_mb: peak_rss_mb(),
    }
}

/// Q2: MCTS-style "fork the saturated e-graph N times with a small
/// perturbation". Measures clone + add + incremental re-run per fork.
fn run_fork(spec: &CellSpec, n_facts: usize, forks: usize, seed: u64) -> CellResult {
    let rules = rule_pack(RuleSet::Medium);
    let expr = build_theory(n_facts, 3, seed);

    let t = Instant::now();
    let base = Runner::default().with_expr(&expr).run(&rules);
    let base_saturate_ms = t.elapsed().as_secs_f64() * 1_000.0;
    let base_egraph = base.egraph;

    let mut fork_ms = Vec::with_capacity(forks);
    for i in 0..forks as i64 {
        let t = Instant::now();
        let mut g = base_egraph.clone();
        let pert: RecExpr<AkhLang> = format!("(causes {} {})", 9_000_000 + i, 9_000_001 + i)
            .parse()
            .expect("perturbation parses");
        g.add_expr(&pert);
        let _rerun = Runner::default().with_egraph(g).run(&rules);
        fork_ms.push(t.elapsed().as_secs_f64() * 1_000.0);
    }

    CellResult::Fork {
        spec: Box::new(spec.clone()),
        base_saturate_ms,
        fork_ms,
        peak_rss_mb: peak_rss_mb(),
    }
}
