//! Phase 41f — scaling-law benchmark harness.
//!
//! Measures accuracy vs. step budget for the [`super::ThinkingLoop`] on
//! three deterministic task families:
//!
//! 1. **Multi-hop KG queries** (depths 2/4/8/16): find the unique
//!    answer-typed node at the end of a relation chain buried in branching
//!    dead-end distractors. Discovery requires frontier expansion, so deeper
//!    chains need more spreading-activation steps — the family where
//!    test-time compute should buy accuracy.
//! 2. **Analogical retrieval** (`A : B :: C : ?` over structured bindings,
//!    ADR 002 encoding): exact VSA arithmetic resolves in one retrieval —
//!    the family where the curve should be flat (no benefit from budget).
//! 3. **Factual evidence queries** (`(S, R, ?)` with one well-supported and
//!    several weakly-supported candidates): argumentation + Dempster-Shafer
//!    fusion saturate within one rotation — flat curve expected early.
//!
//! Raven's Progressive Matrices (plan benchmark 3) needs external data files
//! and is out of scope for this harness run.
//!
//! Everything is seeded and deterministic. Results are CSV rows suitable for
//! `target/bench-results/thinking/`.

use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::engine::{Engine, EngineConfig};
use crate::graph::Triple;
use crate::symbol::SymbolId;
use crate::vsa::Dimension;

use super::budget::ThinkingBudget;
use super::error::ThinkingResult;
use super::loop_runner::ThinkingLoop;
use super::state::HaltReason;
use super::ThinkingPredicates;

/// Step budgets swept by the harness (plan 41f sweep points).
pub const STEP_BUDGETS: &[u32] = &[100, 300, 1_000, 3_000, 10_000];

/// Default output directory for CSV artefacts.
pub const OUTPUT_DIR: &str = "target/bench-results/thinking";

// ---------------------------------------------------------------------------
// Task set
// ---------------------------------------------------------------------------

/// One benchmark task: a deliberation goal with a known correct answer.
#[derive(Debug, Clone)]
pub struct BenchTask {
    pub id: String,
    /// Family label, e.g. `multi-hop-d8`, `analogy`, `factual`.
    pub family: String,
    /// Chain depth for multi-hop tasks (0 otherwise; CSV convenience).
    pub depth: usize,
    pub goal: SymbolId,
    pub expected: SymbolId,
}

/// Task-set shape. Defaults match the Phase 41f experiment.
#[derive(Debug, Clone)]
pub struct TaskSetConfig {
    /// Multi-hop chain depths (plan: 2, 4, 8, 16).
    pub multi_hop_depths: Vec<usize>,
    /// Instances per depth.
    pub instances_per_depth: usize,
    /// Dead-end branches per chain node.
    pub branching: usize,
    /// Children per dead-end node (frontier chaff).
    pub sub_branching: usize,
    /// Analogy task count.
    pub analogy_tasks: usize,
    /// Distractor symbols in item memory per analogy task.
    pub analogy_distractors: usize,
    /// Factual task count.
    pub factual_tasks: usize,
    /// Weak competing candidates per factual task.
    pub factual_distractors: usize,
    /// RNG seed (analogy binding vectors, confidence jitter).
    pub seed: u64,
}

impl Default for TaskSetConfig {
    fn default() -> Self {
        Self {
            multi_hop_depths: vec![2, 4, 8, 16],
            instances_per_depth: 5,
            branching: 6,
            sub_branching: 3,
            analogy_tasks: 10,
            analogy_distractors: 8,
            factual_tasks: 10,
            factual_distractors: 4,
            seed: 41,
        }
    }
}

impl TaskSetConfig {
    /// A small task set for smoke tests.
    pub fn smoke() -> Self {
        Self {
            multi_hop_depths: vec![2, 4],
            instances_per_depth: 1,
            branching: 3,
            sub_branching: 1,
            analogy_tasks: 2,
            analogy_distractors: 4,
            factual_tasks: 2,
            factual_distractors: 2,
            seed: 41,
        }
    }
}

/// Build the shared benchmark engine (in-memory, test dimension for speed).
pub fn build_engine() -> ThinkingResult<Engine> {
    Ok(Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })?)
}

/// Populate `engine` with the full deterministic task set.
pub fn build_task_set(engine: &Engine, cfg: &TaskSetConfig) -> ThinkingResult<Vec<BenchTask>> {
    let preds = ThinkingPredicates::resolve(engine)?;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut tasks = Vec::new();

    // -- Family 1: multi-hop chains ---------------------------------------
    for &depth in &cfg.multi_hop_depths {
        for inst in 0..cfg.instances_per_depth {
            let tag = format!("mh-d{depth}-i{inst}");
            let task = build_multi_hop(engine, &preds, cfg, &tag, depth)?;
            tasks.push(task);
        }
    }

    // -- Family 2: analogy -------------------------------------------------
    for inst in 0..cfg.analogy_tasks {
        let tag = format!("an-i{inst}");
        let task = build_analogy(engine, &preds, cfg, &tag, &mut rng)?;
        tasks.push(task);
    }
    // One HNSW rebuild after all custom vector inserts.
    engine.item_memory().rebuild_hnsw();

    // -- Family 3: factual evidence ----------------------------------------
    for inst in 0..cfg.factual_tasks {
        let tag = format!("fa-i{inst}");
        let task = build_factual(engine, &preds, cfg, &tag, &mut rng)?;
        tasks.push(task);
    }

    Ok(tasks)
}

/// Chain `c0 -r-> c1 -r-> ... -r-> cd` where only `cd` is answer-typed.
/// Every chain node also has `branching` dead-end children (created *before*
/// the next chain node so ties in activation resolve toward the chaff —
/// the loop has to do real search work per ring).
fn build_multi_hop(
    engine: &Engine,
    preds: &ThinkingPredicates,
    cfg: &TaskSetConfig,
    tag: &str,
    depth: usize,
) -> ThinkingResult<BenchTask> {
    let relation = engine.resolve_or_create_relation(&format!("{tag}:path"))?;
    let answer_type = engine.resolve_or_create_entity(&format!("{tag}:target"))?;

    let mut prev = engine.resolve_or_create_entity(&format!("{tag}:hop0"))?;
    let start = prev;
    for i in 1..=depth {
        // Dead-end chaff first (lower symbol ids → expanded first on ties).
        for j in 0..cfg.branching {
            let dead = engine.resolve_or_create_entity(&format!("{tag}:dead{i}-{j}"))?;
            engine.add_triple(&Triple::new(prev, relation, dead).with_confidence(0.95))?;
            for k in 0..cfg.sub_branching {
                let sub = engine.resolve_or_create_entity(&format!("{tag}:sub{i}-{j}-{k}"))?;
                engine.add_triple(&Triple::new(dead, relation, sub).with_confidence(0.95))?;
            }
        }
        let next = engine.resolve_or_create_entity(&format!("{tag}:hop{i}"))?;
        engine.add_triple(&Triple::new(prev, relation, next).with_confidence(0.95))?;
        prev = next;
    }
    engine.add_triple(&Triple::new(prev, preds.is_a, answer_type))?;

    let goal = engine.resolve_or_create_entity(&format!("query:{tag}"))?;
    engine.add_triple(&Triple::new(goal, preds.seeks, start))?;
    engine.add_triple(&Triple::new(goal, preds.relation, relation))?;
    engine.add_triple(&Triple::new(goal, preds.answer_type, answer_type))?;

    Ok(BenchTask {
        id: tag.to_string(),
        family: format!("multi-hop-d{depth}"),
        depth,
        goal,
        expected: prev,
    })
}

/// `A : B :: C : ?` with structured bindings (`B = A ⊗ R`, `D = C ⊗ R`) —
/// the ADR 002 analogical-arithmetic encoding.
fn build_analogy(
    engine: &Engine,
    preds: &ThinkingPredicates,
    cfg: &TaskSetConfig,
    tag: &str,
    rng: &mut StdRng,
) -> ThinkingResult<BenchTask> {
    let ops = engine.ops();
    let mem = engine.item_memory();

    let a = engine.resolve_or_create_entity(&format!("{tag}:a"))?;
    let b = engine.resolve_or_create_entity(&format!("{tag}:b"))?;
    let c = engine.resolve_or_create_entity(&format!("{tag}:c"))?;
    let d = engine.resolve_or_create_entity(&format!("{tag}:d"))?;

    let va = mem.get_or_create(ops, a);
    let vc = mem.get_or_create(ops, c);
    let r = ops.random(rng);
    mem.insert(b, ops.bind(&va, &r).map_err(crate::error::AkhError::from)?);
    mem.insert(d, ops.bind(&vc, &r).map_err(crate::error::AkhError::from)?);
    for i in 0..cfg.analogy_distractors {
        let noise = engine.resolve_or_create_entity(&format!("{tag}:noise{i}"))?;
        mem.get_or_create(ops, noise);
    }

    let goal = engine.resolve_or_create_entity(&format!("query:{tag}"))?;
    engine.add_triple(&Triple::new(goal, preds.analogy_a, a))?;
    engine.add_triple(&Triple::new(goal, preds.analogy_b, b))?;
    engine.add_triple(&Triple::new(goal, preds.analogy_c, c))?;

    Ok(BenchTask {
        id: tag.to_string(),
        family: "analogy".to_string(),
        depth: 0,
        goal,
        expected: d,
    })
}

/// `(S, R, ?)` with one well-supported candidate and several weak ones.
fn build_factual(
    engine: &Engine,
    preds: &ThinkingPredicates,
    cfg: &TaskSetConfig,
    tag: &str,
    rng: &mut StdRng,
) -> ThinkingResult<BenchTask> {
    let subject = engine.resolve_or_create_entity(&format!("{tag}:subject"))?;
    let relation = engine.resolve_or_create_relation(&format!("{tag}:rel"))?;
    let answer_type = engine.resolve_or_create_entity(&format!("{tag}:type"))?;

    let good = engine.resolve_or_create_entity(&format!("{tag}:good"))?;
    engine.add_triple(
        &Triple::new(subject, relation, good).with_confidence(0.85 + rng.gen_range(0.0..0.1)),
    )?;
    engine.add_triple(&Triple::new(good, preds.is_a, answer_type))?;
    for i in 0..cfg.factual_distractors {
        let weak = engine.resolve_or_create_entity(&format!("{tag}:weak{i}"))?;
        engine.add_triple(
            &Triple::new(subject, relation, weak)
                .with_confidence(0.15 + rng.gen_range(0.0..0.25)),
        )?;
        engine.add_triple(&Triple::new(weak, preds.is_a, answer_type))?;
    }

    let goal = engine.resolve_or_create_entity(&format!("query:{tag}"))?;
    engine.add_triple(&Triple::new(goal, preds.seeks, subject))?;
    engine.add_triple(&Triple::new(goal, preds.relation, relation))?;
    engine.add_triple(&Triple::new(goal, preds.answer_type, answer_type))?;

    Ok(BenchTask {
        id: tag.to_string(),
        family: "factual".to_string(),
        depth: 0,
        goal,
        expected: good,
    })
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Budget-adaptation mode of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMode {
    /// Fixed step budget (pure scaling-law measurement).
    Fixed,
    /// Phase 41e VOC controller enabled (economic halting).
    Adaptive,
}

impl BudgetMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }
}

/// One measured (task, budget, mode) run.
#[derive(Debug, Clone)]
pub struct BenchRun {
    pub task_id: String,
    pub family: String,
    pub depth: usize,
    pub budget: u32,
    pub mode: BudgetMode,
    pub success: bool,
    pub steps_used: u32,
    pub wall_ms: u64,
    pub voc_used: f32,
    pub halt_reason: String,
    pub belief: f32,
}

/// Run one task under one budget.
pub fn run_task(
    engine: &Engine,
    task: &BenchTask,
    budget_steps: u32,
    mode: BudgetMode,
) -> ThinkingResult<BenchRun> {
    let mut config = super::loop_runner::ThinkingConfig::default();
    config.adaptive_budget = matches!(mode, BudgetMode::Adaptive);
    let mut tl = ThinkingLoop::new(engine)?.with_config(config);

    let start = Instant::now();
    let conclusion = tl.deliberate(task.goal, ThinkingBudget::steps(budget_steps))?;
    let wall_ms = start.elapsed().as_millis() as u64;

    Ok(BenchRun {
        task_id: task.id.clone(),
        family: task.family.clone(),
        depth: task.depth,
        budget: budget_steps,
        mode,
        success: conclusion.answer_symbol == Some(task.expected),
        steps_used: conclusion.steps_used,
        wall_ms,
        voc_used: conclusion.voc_used,
        halt_reason: match &conclusion.halt_reason {
            HaltReason::Converged { .. } => "converged".to_string(),
            other => other.as_str().to_string(),
        },
        belief: conclusion.confidence.belief,
    })
}

/// Sweep every task over every budget in `budgets` under `mode`.
pub fn run_sweep(
    engine: &Engine,
    tasks: &[BenchTask],
    budgets: &[u32],
    mode: BudgetMode,
) -> ThinkingResult<Vec<BenchRun>> {
    let mut runs = Vec::with_capacity(tasks.len() * budgets.len());
    for &budget in budgets {
        for task in tasks {
            runs.push(run_task(engine, task, budget, mode)?);
        }
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// Aggregation + CSV
// ---------------------------------------------------------------------------

/// Accuracy/latency aggregate for one (family, budget, mode) bucket.
#[derive(Debug, Clone)]
pub struct BucketSummary {
    pub family: String,
    pub budget: u32,
    pub mode: BudgetMode,
    pub total: usize,
    pub successes: usize,
    pub accuracy: f64,
    pub mean_steps: f64,
    pub mean_wall_ms: f64,
    pub mean_voc: f64,
}

/// Aggregate runs into per-(family, budget, mode) buckets, sorted.
pub fn summarize(runs: &[BenchRun]) -> Vec<BucketSummary> {
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<(String, u32, &'static str), Vec<&BenchRun>> = BTreeMap::new();
    for r in runs {
        buckets
            .entry((r.family.clone(), r.budget, r.mode.as_str()))
            .or_default()
            .push(r);
    }
    buckets
        .into_iter()
        .map(|((family, budget, _), rs)| {
            let total = rs.len();
            let successes = rs.iter().filter(|r| r.success).count();
            BucketSummary {
                family,
                budget,
                mode: rs[0].mode,
                total,
                successes,
                accuracy: successes as f64 / total.max(1) as f64,
                mean_steps: rs.iter().map(|r| r.steps_used as f64).sum::<f64>()
                    / total.max(1) as f64,
                mean_wall_ms: rs.iter().map(|r| r.wall_ms as f64).sum::<f64>()
                    / total.max(1) as f64,
                mean_voc: rs.iter().map(|r| r.voc_used as f64).sum::<f64>()
                    / total.max(1) as f64,
            }
        })
        .collect()
}

/// Write per-run rows to `dir/runs.csv`.
pub fn emit_runs_csv(runs: &[BenchRun], dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join("runs.csv"))?;
    writeln!(
        f,
        "task_id,family,depth,budget,mode,success,steps_used,wall_ms,voc_used,halt_reason,belief"
    )?;
    for r in runs {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{:.3},{},{:.4}",
            r.task_id,
            r.family,
            r.depth,
            r.budget,
            r.mode.as_str(),
            r.success,
            r.steps_used,
            r.wall_ms,
            r.voc_used,
            r.halt_reason,
            r.belief,
        )?;
    }
    Ok(())
}

/// Write aggregates to `dir/summary.csv`.
pub fn emit_summary_csv(summaries: &[BucketSummary], dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join("summary.csv"))?;
    writeln!(
        f,
        "family,budget,mode,total,successes,accuracy,mean_steps,mean_wall_ms,mean_voc"
    )?;
    for s in summaries {
        writeln!(
            f,
            "{},{},{},{},{},{:.4},{:.1},{:.1},{:.1}",
            s.family,
            s.budget,
            s.mode.as_str(),
            s.total,
            s.successes,
            s.accuracy,
            s.mean_steps,
            s.mean_wall_ms,
            s.mean_voc,
        )?;
    }
    Ok(())
}

/// Render the accuracy-vs-budget table as printable text.
pub fn format_table(summaries: &[BucketSummary]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<16} {:>8} {:>9} {:>10} {:>10} {:>12}\n",
        "family", "budget", "mode", "accuracy", "avg steps", "avg wall ms"
    ));
    for s in summaries {
        out.push_str(&format!(
            "{:<16} {:>8} {:>9} {:>9.0}% {:>10.1} {:>12.1}\n",
            s.family,
            s.budget,
            s.mode.as_str(),
            s.accuracy * 100.0,
            s.mean_steps,
            s.mean_wall_ms
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_task_set_builds_and_runs() {
        let engine = build_engine().unwrap();
        let tasks = build_task_set(&engine, &TaskSetConfig::smoke()).unwrap();
        assert!(!tasks.is_empty());

        let runs = run_sweep(&engine, &tasks, &[60], BudgetMode::Fixed).unwrap();
        assert_eq!(runs.len(), tasks.len());
        // Shallow multi-hop, analogy, and factual tasks should all pass
        // at a comfortable budget.
        for r in &runs {
            assert!(
                r.success,
                "task {} ({}) failed at budget 60: halt={} belief={}",
                r.task_id, r.family, r.halt_reason, r.belief
            );
        }

        let summaries = summarize(&runs);
        assert!(!summaries.is_empty());
        let table = format_table(&summaries);
        assert!(table.contains("factual"));
    }

    #[test]
    fn csv_emission_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let runs = vec![BenchRun {
            task_id: "t".into(),
            family: "factual".into(),
            depth: 0,
            budget: 100,
            mode: BudgetMode::Fixed,
            success: true,
            steps_used: 12,
            wall_ms: 3,
            voc_used: 14.0,
            halt_reason: "verifier-sufficient".into(),
            belief: 0.8,
        }];
        emit_runs_csv(&runs, dir.path()).unwrap();
        emit_summary_csv(&summarize(&runs), dir.path()).unwrap();
        let text = std::fs::read_to_string(dir.path().join("runs.csv")).unwrap();
        assert!(text.contains("t,factual,0,100,fixed,true"));
        let text = std::fs::read_to_string(dir.path().join("summary.csv")).unwrap();
        assert!(text.contains("factual,100,fixed,1,1,1.0000"));
    }
}
