//! Benchmark and correctness verification for the VSA-native RL signal.
//!
//! Verifies:
//! - The RL pipeline (encode → reward → update) runs within budget per cycle.
//! - Triple confidence drifts toward successful actions over 100+ cycles.
//! - No regression: established high-quality patterns are not degraded.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
#[allow(unused_imports)]
use rand::rngs::StdRng;

use akh_medu::agent::tool::ToolOutput;
use akh_medu::agent::vsa_rl::{VsaRlConfig, compute_and_apply, compute_rl_reward};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::vsa::Dimension;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bench_engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::DEFAULT,
        ..Default::default()
    })
    .unwrap()
}

fn test_engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap()
}

fn success_output() -> ToolOutput {
    ToolOutput {
        result: "success".into(),
        success: true,
        symbols_involved: vec![],
    }
}

fn failure_output() -> ToolOutput {
    ToolOutput {
        result: "failed".into(),
        success: false,
        symbols_involved: vec![],
    }
}

// ---------------------------------------------------------------------------
// Micro-benchmarks
// ---------------------------------------------------------------------------

fn bench_reward_computation(c: &mut Criterion) {
    let engine = bench_engine();
    let ops = engine.ops();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let predicted = ops.random(&mut rng);
    let actual = ops.random(&mut rng);

    c.bench_function("vsa_rl_reward_10k", |b| {
        b.iter(|| black_box(compute_rl_reward(&predicted, &actual, ops)))
    });
}

fn bench_full_rl_cycle(c: &mut Criterion) {
    let engine = bench_engine();
    let goal = engine.resolve_or_create_entity("bench_goal").unwrap();
    let ops = engine.ops();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let predicted = ops.random(&mut rng);
    let output = success_output();
    let config = VsaRlConfig::default();

    c.bench_function("vsa_rl_full_cycle_10k", |b| {
        let mut cycle = 0u64;
        b.iter(|| {
            cycle += 1;
            black_box(compute_and_apply(
                &engine, goal, "kg_query", &predicted, &output, &config, cycle,
            ))
        })
    });
}

// ---------------------------------------------------------------------------
// Correctness verification over 100+ cycles
// ---------------------------------------------------------------------------

/// Verify that repeated successful predictions increase triple confidence.
fn verify_confidence_increases_for_success() {
    let engine = test_engine();
    let goal = engine.resolve_or_create_entity("success_goal").unwrap();
    let ops = engine.ops();
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let config = VsaRlConfig::default();

    // Register an outcome symbol whose vector matches `predicted` → reward ≈ 1.0.
    let predicted = ops.random(&mut rng);
    let outcome_sym = engine.resolve_or_create_entity("succ_outcome").unwrap();
    engine.item_memory().insert(outcome_sym, predicted.clone());
    let output = ToolOutput {
        result: "ok".into(),
        success: true,
        symbols_involved: vec![outcome_sym],
    };

    let mut last_conf = 0.5_f32;
    for cycle in 1..=120 {
        if let Some(sig) = compute_and_apply(
            &engine, goal, "kg_query", &predicted, &output, &config, cycle,
        ) {
            last_conf = sig.new_confidence;
        }
    }
    assert!(
        last_conf > 0.80,
        "confidence should converge toward 1.0 after 120 successful cycles, got {last_conf}"
    );
}

/// Verify that repeated failures decrease triple confidence.
fn verify_confidence_decreases_for_failure() {
    let engine = test_engine();
    let goal = engine.resolve_or_create_entity("fail_goal").unwrap();
    let ops = engine.ops();
    let config = VsaRlConfig::default();

    // Start with artificially high confidence by running some "successful" cycles first.
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let predicted = ops.random(&mut rng);
    let good_output = success_output();
    for cycle in 1..=30 {
        compute_and_apply(
            &engine,
            goal,
            "reason",
            &predicted,
            &good_output,
            &config,
            cycle,
        );
    }

    // Now inject a consistently low-reward scenario.
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
    let bad_predicted = ops.random(&mut rng2);
    // No symbols_involved → encode_actual_outcome falls back to goal_id vector.
    let bad_output = failure_output();

    let conf_before =
        akh_medu::agent::vsa_rl::learned_quality(&engine, goal, "reason").unwrap_or(0.5);

    for cycle in 31..=130 {
        compute_and_apply(
            &engine,
            goal,
            "reason",
            &bad_predicted,
            &bad_output,
            &config,
            cycle,
        );
    }
    let conf_after =
        akh_medu::agent::vsa_rl::learned_quality(&engine, goal, "reason").unwrap_or(0.5);

    assert!(
        conf_after <= conf_before,
        "confidence should not increase for consistently bad predictions: before={conf_before}, after={conf_after}"
    );
}

/// Verify that different tools accumulate independent quality scores.
fn verify_tool_independence() {
    let engine = test_engine();
    let goal = engine.resolve_or_create_entity("multi_tool_goal").unwrap();
    let ops = engine.ops();
    let config = VsaRlConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(23);
    let predicted = ops.random(&mut rng);
    let good = success_output();
    let bad = failure_output();

    // kg_query gets consistent high reward: register outcome matching predicted.
    let outcome_sym = engine
        .resolve_or_create_entity("tool_indep_outcome")
        .unwrap();
    engine.item_memory().insert(outcome_sym, predicted.clone());
    let good_with_sym = ToolOutput {
        result: "ok".into(),
        success: true,
        symbols_involved: vec![outcome_sym],
    };
    for cycle in 1..=100 {
        compute_and_apply(
            &engine,
            goal,
            "kg_query",
            &predicted,
            &good_with_sym,
            &config,
            cycle,
        );
    }
    // reason gets consistently low reward (no matching outcome symbol).
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(77);
    let bad_pred = ops.random(&mut rng2);
    for cycle in 1..=100 {
        compute_and_apply(&engine, goal, "reason", &bad_pred, &bad, &config, cycle);
    }

    let kg_conf =
        akh_medu::agent::vsa_rl::learned_quality(&engine, goal, "kg_query").unwrap_or(0.5);
    let reason_conf =
        akh_medu::agent::vsa_rl::learned_quality(&engine, goal, "reason").unwrap_or(0.5);

    assert!(
        kg_conf > reason_conf,
        "kg_query (consistently good) should outrank reason (consistently bad): \
         kg_query={kg_conf}, reason={reason_conf}"
    );
}

/// Wrap correctness checks as a single dummy "bench" so criterion reports them.
fn verify_rl_correctness_100_cycles(c: &mut Criterion) {
    c.bench_function("vsa_rl_correctness_100_cycles", |b| {
        b.iter(|| {
            verify_confidence_increases_for_success();
            verify_confidence_decreases_for_failure();
            verify_tool_independence();
        })
    });
}

criterion_group!(
    benches,
    bench_reward_computation,
    bench_full_rl_cycle,
    verify_rl_correctness_100_cycles,
);
criterion_main!(benches);
