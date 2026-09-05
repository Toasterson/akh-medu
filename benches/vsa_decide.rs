//! Benchmarks for VSA-native vs LLM-driven OODA Decide phase.
//!
//! Measures latency of the pure VSA similarity ranking path vs the
//! heuristic-enhanced legacy path.  Both paths share the same
//! Observe and Orient stages; only the Decide step differs.
//!
//! Run with:
//!   cargo bench --bench vsa_decide

use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use akh_medu::agent::goal::{Goal, GoalStatus};
use akh_medu::agent::memory::WorkingMemory;
use akh_medu::agent::ooda::{Observation, Orientation};
use akh_medu::agent::tool::ToolRegistry;
use akh_medu::agent::tools;
use akh_medu::agent::vsa_decide::{DecideMode, VsaDecideConfig, vsa_native_decide};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::vsa::Dimension;

// ── Setup helpers ──────────────────────────────────────────────────────────

fn make_engine() -> Arc<Engine> {
    Arc::new(
        Engine::new(EngineConfig {
            dimension: Dimension::DEFAULT,
            ..Default::default()
        })
        .expect("engine"),
    )
}

fn make_registry(engine: &Engine) -> ToolRegistry {
    let _ = engine; // engine is used indirectly via tool execution
    let mut r = ToolRegistry::new();
    r.register(Box::new(tools::KgQueryTool));
    r.register(Box::new(tools::KgMutateTool));
    r.register(Box::new(tools::ReasonTool));
    r.register(Box::new(tools::SimilaritySearchTool));
    r.register(Box::new(tools::GapAnalysisTool));
    r.register(Box::new(tools::InferRulesTool));
    r
}

fn make_goal(engine: &Engine, description: &str) -> Goal {
    let sym = engine
        .resolve_or_create_entity(description)
        .expect("create entity");
    Goal {
        symbol_id: sym,
        description: description.to_string(),
        status: GoalStatus::Active,
        priority: 128,
        success_criteria: "task completed".to_string(),
        parent: None,
        children: vec![],
        created_at: 0,
        cycles_worked: 0,
        last_progress_cycle: 0,
        source: None,
        blocked_by: vec![],
        justification: None,
        priority_rationale: None,
        reformulated_from: None,
        estimated_effort: None,
    }
}

fn empty_context() -> (Observation, Orientation) {
    let obs = Observation {
        active_goals: vec![],
        working_memory_size: 0,
        recent_entries: vec![],
        recalled_episodes: vec![],
    };
    let orient = Orientation {
        relevant_knowledge: vec![],
        inferences: vec![],
        memory_pressure: 0.0,
    };
    (obs, orient)
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

/// Benchmark the VSA-native decide path (pure similarity + optional MCTS).
fn bench_vsa_native_decide(c: &mut Criterion) {
    let engine = make_engine();
    let registry = make_registry(&engine);
    let wm = WorkingMemory::new(64);
    let goal = make_goal(&engine, "search for knowledge about the knowledge graph");
    let (obs, orient) = empty_context();
    let config = VsaDecideConfig::default();

    c.bench_function("vsa_native_decide", |b| {
        b.iter(|| {
            black_box(vsa_native_decide(
                &goal, &obs, &orient, &engine, &wm, None, &registry, &config,
            ))
        })
    });
}

/// Benchmark the VSA-native decide with MCTS always triggered.
fn bench_vsa_native_decide_with_mcts(c: &mut Criterion) {
    let engine = make_engine();
    let registry = make_registry(&engine);
    let wm = WorkingMemory::new(64);
    let goal = make_goal(&engine, "search for knowledge about the knowledge graph");
    let (obs, orient) = empty_context();
    // Force MCTS by setting threshold to 1.0 (all similarities are below 1.0).
    let config = VsaDecideConfig {
        mcts_fallback_threshold: 1.0,
        mcts_iterations: 64,
        ..Default::default()
    };

    c.bench_function("vsa_native_decide_with_mcts", |b| {
        b.iter(|| {
            black_box(vsa_native_decide(
                &goal, &obs, &orient, &engine, &wm, None, &registry, &config,
            ))
        })
    });
}

/// Benchmark the VSA-native decide with a larger MCTS budget (256 iterations).
fn bench_vsa_native_decide_mcts_256(c: &mut Criterion) {
    let engine = make_engine();
    let registry = make_registry(&engine);
    let wm = WorkingMemory::new(64);
    let goal = make_goal(&engine, "search for knowledge about the knowledge graph");
    let (obs, orient) = empty_context();
    let config = VsaDecideConfig {
        mcts_fallback_threshold: 1.0,
        mcts_iterations: 256,
        ..Default::default()
    };

    c.bench_function("vsa_native_decide_mcts_256", |b| {
        b.iter(|| {
            black_box(vsa_native_decide(
                &goal, &obs, &orient, &engine, &wm, None, &registry, &config,
            ))
        })
    });
}

/// Benchmark the decide path with multiple different goal descriptions to
/// stress-test the encoding pipeline.
fn bench_vsa_native_decide_varied_goals(c: &mut Criterion) {
    let engine = make_engine();
    let registry = make_registry(&engine);
    let wm = WorkingMemory::new(64);
    let (obs, orient) = empty_context();
    let config = VsaDecideConfig::default();

    let goals = vec![
        make_goal(
            &engine,
            "query the knowledge graph for entity relationships",
        ),
        make_goal(&engine, "apply symbolic reasoning to derive new facts"),
        make_goal(&engine, "find gaps in current knowledge coverage"),
        make_goal(&engine, "recall past experiences from episodic memory"),
        make_goal(&engine, "run inference rules over known triples"),
    ];

    let mut idx = 0usize;
    c.bench_function("vsa_native_decide_varied", |b| {
        b.iter(|| {
            let goal = &goals[idx % goals.len()];
            idx += 1;
            black_box(vsa_native_decide(
                goal, &obs, &orient, &engine, &wm, None, &registry, &config,
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_vsa_native_decide,
    bench_vsa_native_decide_with_mcts,
    bench_vsa_native_decide_mcts_256,
    bench_vsa_native_decide_varied_goals,
);
criterion_main!(benches);
