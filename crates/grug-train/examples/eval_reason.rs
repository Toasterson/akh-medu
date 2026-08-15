//! Phase 51 P3-R inference-only eval: load a `train_reason` checkpoint and
//! report the full [`grug_data::reason::score_reason_outputs`] summary on all
//! three frozen axes (axis-B in-distribution, axis-A length generalization,
//! abstention) -- the headline "supervision differential" number is
//! axis-A's answer_accuracy compared across a `p3r-trace` vs `p3r-answer`
//! checkpoint pair.
//!
//! ```text
//! cargo run -p grug-train --example eval_reason --features gpu --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3r-trace \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p3r-trace/checkpoints/step_00019999
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::reason::{ReasonConfig, ReasonDataset, ReasonEvalSummary, score_reason_outputs};
use grug_train::generate::GenerateConfig;
use grug_train::reason_train::generate_reason_outputs;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "eval_reason")]
struct Args {
    #[arg(long)]
    artifacts_dir: PathBuf,

    #[arg(long)]
    resume: PathBuf,

    /// Which training arm this checkpoint was trained under -- echoed into
    /// the printed report and the output JSON; does not affect eval (eval
    /// always scores whatever the model emits via replay).
    #[arg(long, default_value = "trace")]
    arm: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 300)]
    entities_per_type: usize,

    #[arg(long, default_value_t = 2)]
    distractor_factor: usize,

    #[arg(long, default_value = "2,3,4")]
    train_depths: String,

    #[arg(long, default_value_t = 0.15)]
    holdout_subject_frac: f64,

    #[arg(long, default_value = "5,6")]
    axis_a_depths: String,

    #[arg(long, default_value_t = 256)]
    axis_b_n: usize,

    #[arg(long, default_value_t = 256)]
    axis_a_n: usize,

    #[arg(long, default_value_t = 128)]
    abstention_n: usize,

    /// Must cover a depth-6 trace (recall/derive/call-answer lines).
    #[arg(long, default_value_t = 160)]
    eval_max_new_tokens: usize,
}

fn parse_depths(s: &str, flag: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag}: not an integer depth: {t:?}"))
        })
        .collect()
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("eval_reason: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("eval_reason: CPU backend");
        Default::default()
    };

    let train_depths = parse_depths(&args.train_depths, "--train-depths");
    let axis_a_depths = parse_depths(&args.axis_a_depths, "--axis-a-depths");

    let dataset = ReasonDataset::new(ReasonConfig {
        seed: args.seed,
        entities_per_type: args.entities_per_type,
        distractor_factor: args.distractor_factor,
        train_depths,
        holdout_subject_frac: args.holdout_subject_frac,
    })
    .expect("ReasonConfig builds a valid dataset -- must match the training run's config");

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) =
        checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint loads");
    let model = to_eval_model(&model);
    println!("eval_reason: ARM={} checkpoint step {step}", args.arm);

    let axis_b = dataset.axis_b_examples(args.axis_b_n).expect("axis-B draw");
    let axis_a = dataset
        .axis_a_examples(args.axis_a_n, &axis_a_depths)
        .expect("axis-A draw");
    let abstention = dataset
        .abstention_examples(args.abstention_n)
        .expect("abstention draw");

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };

    let axis_b_outputs = generate_reason_outputs(&model, &tokenizer, &axis_b, gen_config, &device);
    let axis_a_outputs = generate_reason_outputs(&model, &tokenizer, &axis_a, gen_config, &device);
    let abstention_outputs =
        generate_reason_outputs(&model, &tokenizer, &abstention, gen_config, &device);

    let axis_b_summary = score_reason_outputs(&axis_b, &axis_b_outputs);
    let axis_a_summary = score_reason_outputs(&axis_a, &axis_a_outputs);
    let abstention_summary = score_reason_outputs(&abstention, &abstention_outputs);

    println!(
        "\n=== Phase 51 P3-R -- ARM={} checkpoint step {step} ===",
        args.arm
    );
    print_summary_table("axis-B (in-distribution)", &axis_b_summary);
    print_summary_table("axis-A (length generalization)", &axis_a_summary);
    print_summary_table("abstention", &abstention_summary);
    println!(
        "abstention confusion : recall={:.3} precision={:.3} (n_expected={}, n_emitted={})",
        abstention_summary.abstention.recall,
        abstention_summary.abstention.precision,
        abstention_summary.abstention.n_expected,
        abstention_summary.abstention.n_emitted,
    );

    let summary_json = serde_json::json!({
        "checkpoint_step": step,
        "arm": args.arm,
        "axis_b": summary_to_json(&axis_b_summary),
        "axis_a": summary_to_json(&axis_a_summary),
        "abstention": summary_to_json(&abstention_summary),
    });
    let out = args.artifacts_dir.join("eval_reason.json");
    match std::fs::write(&out, serde_json::to_string_pretty(&summary_json).unwrap()) {
        Ok(()) => println!("\neval_reason: wrote {}", out.display()),
        Err(e) => eprintln!("eval_reason: write failed: {e}"),
    }
}

fn print_summary_table(label: &str, s: &ReasonEvalSummary) {
    println!(
        "{label:<32}: n={:<4} answer_acc={:.3} exec={:.3} trace_valid={:.3} distinct_answer={:.3} entropy={:.3}",
        s.n,
        s.answer_accuracy,
        s.executability_rate,
        s.trace_valid_rate,
        s.diversity.distinct_answer_ratio,
        s.diversity.answer_entropy,
    );
    // P3-RL A1: the reward-hacking surfaces replay cannot see. `looping` is
    // directly comparable to SRaR's 48.1% (arXiv 2605.17291); `wasted` is the
    // headroom a ratio-based executability reward would hand to a farming
    // policy.
    let q = &s.quality;
    println!(
        "{:<32}: looping={:.3} redundancy={:.3} drift={:.3} hop_cov={:.3} wasted={:.2}",
        format!("{label} [quality]"),
        q.looping_fraction,
        q.mean_redundancy_rate,
        q.mean_drift_rate,
        q.mean_hop_coverage,
        q.mean_wasted_valid_steps,
    );
    let mut depths: Vec<&usize> = s.per_depth.keys().collect();
    depths.sort();
    for depth in depths {
        let d = &s.per_depth[depth];
        println!(
            "    depth {depth}: n={:<4} answer_acc={:.3} exec={:.3}",
            d.n, d.answer_accuracy, d.executability_rate
        );
    }
}

fn summary_to_json(s: &ReasonEvalSummary) -> serde_json::Value {
    let per_depth: serde_json::Map<String, serde_json::Value> = s
        .per_depth
        .iter()
        .map(|(depth, stats)| {
            (
                depth.to_string(),
                serde_json::json!({
                    "n": stats.n,
                    "answer_accuracy": stats.answer_accuracy,
                    "executability_rate": stats.executability_rate,
                }),
            )
        })
        .collect();
    serde_json::json!({
        "n": s.n,
        "answer_accuracy": s.answer_accuracy,
        "executability_rate": s.executability_rate,
        "trace_valid_rate": s.trace_valid_rate,
        "distinct_answer_ratio": s.diversity.distinct_answer_ratio,
        "answer_entropy": s.diversity.answer_entropy,
        "quality": {
            "looping_fraction": s.quality.looping_fraction,
            "mean_redundancy_rate": s.quality.mean_redundancy_rate,
            "mean_drift_rate": s.quality.mean_drift_rate,
            "mean_hop_coverage": s.quality.mean_hop_coverage,
            "mean_wasted_valid_steps": s.quality.mean_wasted_valid_steps,
        },
        "abstention": {
            "n_expected": s.abstention.n_expected,
            "n_emitted": s.abstention.n_emitted,
            "recall": s.abstention.recall,
            "precision": s.abstention.precision,
        },
        "per_depth": per_depth,
    })
}
