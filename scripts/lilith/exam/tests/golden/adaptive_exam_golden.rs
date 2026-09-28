//! Golden-value generator for scripts/lilith/exam/tests/golden/adaptive_exam_rust.json.
//!
//! Not built by default. To regenerate the golden file after a change to
//! crates/grug-data/src/adaptive_exam.rs:
//!
//!   cp scripts/lilith/exam/tests/golden/adaptive_exam_golden.rs crates/grug-data/examples/
//!   cargo run -q -p grug-data --example adaptive_exam_golden > scripts/lilith/exam/tests/golden/adaptive_exam_rust.json
//!   rm crates/grug-data/examples/adaptive_exam_golden.rs
//!
//! then rerun the Python controller tests; a mismatch means the port drifted.
use grug_data::adaptive_exam::*;
use serde_json::json;

fn ck(step: u64) -> CheckpointRef {
    CheckpointRef { step, artifact_hash: format!("sha256:{step:064x}") }
}
fn obs(step: u64, tier: ExamTier, fam: &str, n: u64, pass: u64, pre: u64, cat: u64) -> ExamObservation {
    ExamObservation {
        checkpoint: ck(step),
        tier,
        family: fam.into(),
        difficulty: 2,
        sample_count: n,
        pass_count: pass,
        precursor_count: pre,
        catastrophic_count: cat,
    }
}
fn ctl(cfg: ControllerConfig) -> AdaptiveExamController {
    AdaptiveExamController::new(
        "test-generator-v1",
        "run-42",
        42,
        cfg,
        [
            CapabilitySpec::new("reason.chain.depth", 0, 5).with_initial_difficulty(2),
            CapabilitySpec::new("vision.text.line", 0, 5).with_initial_difficulty(2),
        ],
    )
    .unwrap()
}
fn main() {
    let c = ctl(ControllerConfig::default());
    let mut plans = vec![];
    for tier in [ExamTier::Sentinel, ExamTier::RollingProbe, ExamTier::Frontier, ExamTier::Audit] {
        for step in [100u64, 200] {
            for fam in ["reason.chain.depth", "vision.text.line"] {
                let p = c.plan_for_family(tier, ck(step), fam, 200).unwrap();
                plans.push(json!({"plan": p, "fingerprint": p.fingerprint()}));
            }
        }
    }
    let mut w = vec![];
    for (s, n) in [(0u64, 0u64), (5, 20), (20, 200), (199, 200), (200, 200), (123, 400), (1, 3)] {
        w.push(json!({"s": s, "n": n, "i": wilson_interval(s, n, 1.959_963_984_540_054)}));
    }
    let mut scen = serde_json::Map::new();
    let seqs: Vec<(&str, ControllerConfig, Vec<ExamObservation>)> = vec![
        (
            "sentinel_regression_stop",
            ControllerConfig::default(),
            vec![
                obs(0, ExamTier::Sentinel, "reason.chain.depth", 400, 380, 0, 0),
                obs(10, ExamTier::Sentinel, "reason.chain.depth", 400, 300, 0, 0),
                obs(20, ExamTier::Sentinel, "reason.chain.depth", 400, 295, 0, 0),
                obs(30, ExamTier::Sentinel, "reason.chain.depth", 400, 379, 0, 0),
            ],
        ),
        (
            "catastrophic_stop",
            ControllerConfig::default(),
            vec![
                obs(10, ExamTier::Frontier, "vision.text.line", 400, 300, 40, 40),
                obs(20, ExamTier::Frontier, "vision.text.line", 400, 300, 40, 40),
            ],
        ),
        (
            "precursor_steer",
            ControllerConfig { precursor_steering_enabled: true, ..ControllerConfig::default() },
            vec![
                obs(10, ExamTier::Frontier, "vision.text.line", 400, 360, 100, 0),
                obs(20, ExamTier::Frontier, "vision.text.line", 400, 350, 105, 0),
            ],
        ),
        (
            "precursor_shadow",
            ControllerConfig::default(),
            vec![
                obs(10, ExamTier::Frontier, "vision.text.line", 400, 360, 120, 0),
                obs(20, ExamTier::Frontier, "vision.text.line", 400, 360, 120, 0),
            ],
        ),
        (
            "regression_steer",
            ControllerConfig::default(),
            vec![
                obs(10, ExamTier::Frontier, "reason.chain.depth", 400, 300, 0, 0),
                obs(20, ExamTier::Frontier, "reason.chain.depth", 400, 200, 0, 0),
                obs(30, ExamTier::Frontier, "reason.chain.depth", 400, 100, 0, 0),
            ],
        ),
        (
            "difficulty_and_undersampled",
            ControllerConfig::default(),
            vec![
                obs(10, ExamTier::Frontier, "vision.text.line", 400, 390, 0, 0),
                obs(15, ExamTier::Frontier, "vision.text.line", 20, 0, 20, 0),
                obs(20, ExamTier::Frontier, "vision.text.line", 400, 20, 0, 0),
                obs(30, ExamTier::RollingProbe, "reason.chain.depth", 400, 220, 10, 0),
                obs(40, ExamTier::RollingProbe, "reason.chain.depth", 400, 60, 10, 0),
            ],
        ),
        (
            "release_recovery",
            ControllerConfig {
                release_recovery: true,
                min_samples_per_window: 24,
                steer_windows: 1,
                ..ControllerConfig::default()
            },
            vec![
                obs(1, ExamTier::Sentinel, "reason.chain.depth", 24, 23, 0, 1),
                obs(1, ExamTier::Sentinel, "vision.text.line", 400, 200, 200, 0),
                obs(1, ExamTier::Frontier, "reason.chain.depth", 400, 400, 0, 0),
                obs(2, ExamTier::Sentinel, "reason.chain.depth", 400, 400, 0, 0),
            ],
        ),
    ];
    for (name, cfg, seq) in seqs {
        let mut c = ctl(cfg.clone());
        let mut out = vec![];
        for o in seq {
            let d = c.observe(o.clone()).unwrap();
            let next = c.next_plan(ExamTier::Frontier, o.checkpoint.clone(), 24).unwrap();
            out.push(json!({"observation": o, "decision": d, "next_frontier_plan": next}));
        }
        scen.insert(name.into(), json!({"config": cfg, "steps": out, "final_state": c}));
    }
    let bh = json!([
        {"p": [0.04, 0.001, 0.02, 0.8], "alpha": 0.05, "accepted": benjamini_hochberg(&[0.04, 0.001, 0.02, 0.8], 0.05)},
        {"p": [0.01, 0.02, 0.03, 0.04, 0.05], "alpha": 0.05, "accepted": benjamini_hochberg(&[0.01, 0.02, 0.03, 0.04, 0.05], 0.05)},
        {"p": [0.5, 0.2, 0.9], "alpha": 0.05, "accepted": benjamini_hochberg(&[0.5, 0.2, 0.9], 0.05)},
    ]);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source": "crates/grug-data/src/adaptive_exam.rs",
            "generated_by": "a throwaway grug-data example (not committed); regenerate by porting the scenario list",
            "default_config": ControllerConfig::default(),
            "plans": plans, "wilson": w, "benjamini_hochberg": bh, "scenarios": scen
        }))
        .unwrap()
    );
}
