//! Phase 51 P6 -- dual-process reasoning handoff eval: load a `train_reason`
//! checkpoint and, on each of the three frozen axes (axis-B in-distribution,
//! axis-A length generalization, abstention), score three conditions against
//! the *same* generated organ outputs:
//!
//! - `organ_alone`  -- the raw System-1 trace, scored by
//!   [`grug_data::reason::score_reason_outputs`] exactly as `eval_reason`
//!   does.
//! - `handoff`       -- [`grug_data::reason::handoff`] spliced onto the raw
//!   organ output (longest valid on-path prefix + deterministic symbolic
//!   continuation).
//! - `symbolic_alone` -- [`grug_data::reason::handoff`] with an *empty*
//!   organ output, i.e. the System-2 continuation with zero neural
//!   contribution -- reuses the exact same code path as `handoff` (an empty
//!   string trivially fails the trivial-accept check and yields a
//!   longest-valid-prefix of 0), so this is a true ablation rather than a
//!   separately-implemented baseline.
//!
//! Expected pattern (from the P6 measurement that motivated this eval):
//! axis-A (trained depths 2-6, eval depths 7-8) organ_alone answer accuracy
//! ~0.00 while handoff tracks close to symbolic_alone's ~1.0, with
//! `mean_neural_hops > 0` -- the ~42% of individual steps that still replay
//! correctly past the trained depth becomes load-bearing progress rather
//! than being thrown away. axis-B (in-distribution) handoff should
//! approximately equal organ_alone (rule 1's trivial-accept keeps the
//! handoff a no-op when the organ is already right). Abstention behavior
//! (gap recall/precision) should be preserved end to end.
//!
//! ```text
//! cargo run -p grug-train --example eval_handoff --features gpu --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p3r-mixed-trace/checkpoints/step_00019999 \
//!     --train-depths 2,3,4,5,6 --axis-a-depths 7,8
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use grug_data::reason::{
    HandoffOutcome, ReasonConfig, ReasonDataset, ReasonEvalSummary, ReasonExample, handoff,
    score_reason_outputs,
};
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
#[command(name = "eval_handoff")]
struct Args {
    #[arg(long)]
    artifacts_dir: PathBuf,

    #[arg(long)]
    resume: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 300)]
    entities_per_type: usize,

    #[arg(long, default_value_t = 2)]
    distractor_factor: usize,

    /// Comma-separated training depths this checkpoint was trained on --
    /// must match the training run's `--train-depths`.
    #[arg(long, default_value = "2,3,4,5,6")]
    train_depths: String,

    #[arg(long, default_value_t = 0.15)]
    holdout_subject_frac: f64,

    /// Length-generalization depths, never in `--train-depths` -- this is
    /// exactly where `organ_alone` is expected to collapse and the handoff
    /// is expected to recover.
    #[arg(long, default_value = "7,8")]
    axis_a_depths: String,

    #[arg(long, default_value_t = 256)]
    axis_b_n: usize,

    #[arg(long, default_value_t = 256)]
    axis_a_n: usize,

    #[arg(long, default_value_t = 128)]
    abstention_n: usize,

    /// Must cover a depth-8 trace (recall/derive/call-answer lines).
    #[arg(long, default_value_t = 160)]
    max_new_tokens: usize,

    /// Defaults to `<artifacts-dir>/eval_handoff.json`.
    #[arg(long)]
    out: Option<PathBuf>,
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

/// Aggregate stats for one (axis, condition) cell -- shared shape for
/// `handoff` and `symbolic_alone` (whose neural-hop fields are trivially
/// zero, which is itself a useful sanity check that the empty-organ-output
/// ablation is really contributing nothing neurally).
#[derive(Debug, Clone)]
struct HandoffAxisStats {
    n: usize,
    answer_accuracy: f64,
    mean_neural_hops: f64,
    mean_symbolic_hops: f64,
    organ_answer_used_rate: f64,
    composed_executability_mean: f64,
    per_depth_accuracy: HashMap<usize, f64>,
}

fn score_handoff_axis(examples: &[ReasonExample], outputs: &[String]) -> HandoffAxisStats {
    let n = examples.len().min(outputs.len());
    let outcomes: Vec<(&ReasonExample, HandoffOutcome)> = (0..n)
        .map(|i| (&examples[i], handoff(&examples[i], &outputs[i])))
        .collect();
    summarize_handoff_outcomes(n, &outcomes)
}

/// The `symbolic_alone` ablation: identical scoring path, but every organ
/// output is replaced with the empty string before splicing -- so
/// `mean_neural_hops` should land at (or very near) `0.0`.
fn score_symbolic_alone_axis(examples: &[ReasonExample]) -> HandoffAxisStats {
    let n = examples.len();
    let outcomes: Vec<(&ReasonExample, HandoffOutcome)> =
        examples.iter().map(|ex| (ex, handoff(ex, ""))).collect();
    summarize_handoff_outcomes(n, &outcomes)
}

fn summarize_handoff_outcomes(
    n: usize,
    outcomes: &[(&ReasonExample, HandoffOutcome)],
) -> HandoffAxisStats {
    if n == 0 {
        return HandoffAxisStats {
            n: 0,
            answer_accuracy: 0.0,
            mean_neural_hops: 0.0,
            mean_symbolic_hops: 0.0,
            organ_answer_used_rate: 0.0,
            composed_executability_mean: 0.0,
            per_depth_accuracy: HashMap::new(),
        };
    }

    let nf = n as f64;
    let answer_accuracy = outcomes.iter().filter(|(_, o)| o.answer_correct).count() as f64 / nf;
    let mean_neural_hops = outcomes
        .iter()
        .map(|(_, o)| o.neural_hops as f64)
        .sum::<f64>()
        / nf;
    let mean_symbolic_hops = outcomes
        .iter()
        .map(|(_, o)| o.symbolic_hops as f64)
        .sum::<f64>()
        / nf;
    let organ_answer_used_rate =
        outcomes.iter().filter(|(_, o)| o.organ_answer_used).count() as f64 / nf;
    let composed_executability_mean = outcomes
        .iter()
        .map(|(_, o)| o.composed_executability)
        .sum::<f64>()
        / nf;

    let mut per_depth_counts: HashMap<usize, (usize, usize)> = HashMap::new();
    for (ex, outcome) in outcomes {
        let entry = per_depth_counts.entry(ex.depth).or_insert((0, 0));
        entry.0 += 1;
        if outcome.answer_correct {
            entry.1 += 1;
        }
    }
    let per_depth_accuracy = per_depth_counts
        .into_iter()
        .map(|(depth, (count, correct))| (depth, correct as f64 / count.max(1) as f64))
        .collect();

    HandoffAxisStats {
        n,
        answer_accuracy,
        mean_neural_hops,
        mean_symbolic_hops,
        organ_answer_used_rate,
        composed_executability_mean,
        per_depth_accuracy,
    }
}

fn organ_alone_per_depth_accuracy(summary: &ReasonEvalSummary) -> HashMap<usize, f64> {
    summary
        .per_depth
        .iter()
        .map(|(depth, stats)| (*depth, stats.answer_accuracy))
        .collect()
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("eval_handoff: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("eval_handoff: CPU backend");
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
    println!("eval_handoff: checkpoint step {step}");

    let axis_b = dataset.axis_b_examples(args.axis_b_n).expect("axis-B draw");
    let axis_a = dataset
        .axis_a_examples(args.axis_a_n, &axis_a_depths)
        .expect("axis-A draw");
    let abstention = dataset
        .abstention_examples(args.abstention_n)
        .expect("abstention draw");

    let gen_config = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len,
    };

    let axis_b_outputs = generate_reason_outputs(&model, &tokenizer, &axis_b, gen_config, &device);
    let axis_a_outputs = generate_reason_outputs(&model, &tokenizer, &axis_a, gen_config, &device);
    let abstention_outputs =
        generate_reason_outputs(&model, &tokenizer, &abstention, gen_config, &device);

    let axes: [(&str, &[ReasonExample], &[String]); 3] = [
        ("axis-B (in-distribution)", &axis_b, &axis_b_outputs),
        ("axis-A (length generalization)", &axis_a, &axis_a_outputs),
        ("abstention", &abstention, &abstention_outputs),
    ];

    println!("\n=== Phase 51 P6 -- dual-process handoff eval, checkpoint step {step} ===");

    let mut axes_json = serde_json::Map::new();
    for (label, examples, outputs) in axes {
        let organ_summary = score_reason_outputs(examples, outputs);
        let handoff_stats = score_handoff_axis(examples, outputs);
        let symbolic_stats = score_symbolic_alone_axis(examples);

        print_axis_table(label, &organ_summary, &handoff_stats, &symbolic_stats);

        axes_json.insert(
            axis_key(label),
            axis_to_json(&organ_summary, &handoff_stats, &symbolic_stats),
        );
    }

    let summary_json = serde_json::json!({
        "checkpoint_step": step,
        "axes": serde_json::Value::Object(axes_json),
    });

    let out_path = args
        .out
        .unwrap_or_else(|| args.artifacts_dir.join("eval_handoff.json"));
    match std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&summary_json).unwrap(),
    ) {
        Ok(()) => println!("\neval_handoff: wrote {}", out_path.display()),
        Err(e) => eprintln!("eval_handoff: write failed: {e}"),
    }
}

fn axis_key(label: &str) -> String {
    match label {
        l if l.starts_with("axis-B") => "axis_b".to_string(),
        l if l.starts_with("axis-A") => "axis_a".to_string(),
        _ => "abstention".to_string(),
    }
}

fn print_axis_table(
    label: &str,
    organ: &ReasonEvalSummary,
    handoff_stats: &HandoffAxisStats,
    symbolic: &HandoffAxisStats,
) {
    println!("\n-- {label} --");
    println!(
        "  organ_alone    : n={:<4} answer_acc={:.3}",
        organ.n, organ.answer_accuracy,
    );
    println!(
        "  handoff        : n={:<4} answer_acc={:.3} mean_neural_hops={:.2} mean_symbolic_hops={:.2} organ_answer_used_rate={:.3} composed_exec={:.3}",
        handoff_stats.n,
        handoff_stats.answer_accuracy,
        handoff_stats.mean_neural_hops,
        handoff_stats.mean_symbolic_hops,
        handoff_stats.organ_answer_used_rate,
        handoff_stats.composed_executability_mean,
    );
    println!(
        "  symbolic_alone : n={:<4} answer_acc={:.3} mean_neural_hops={:.2} (expect ~0.00) composed_exec={:.3}",
        symbolic.n,
        symbolic.answer_accuracy,
        symbolic.mean_neural_hops,
        symbolic.composed_executability_mean,
    );

    let organ_per_depth = organ_alone_per_depth_accuracy(organ);
    let mut depths: Vec<usize> = organ_per_depth
        .keys()
        .chain(handoff_stats.per_depth_accuracy.keys())
        .chain(symbolic.per_depth_accuracy.keys())
        .copied()
        .collect();
    depths.sort_unstable();
    depths.dedup();
    for depth in depths {
        println!(
            "    depth {depth}: organ_alone={:.3} handoff={:.3} symbolic_alone={:.3}",
            organ_per_depth.get(&depth).copied().unwrap_or(0.0),
            handoff_stats
                .per_depth_accuracy
                .get(&depth)
                .copied()
                .unwrap_or(0.0),
            symbolic
                .per_depth_accuracy
                .get(&depth)
                .copied()
                .unwrap_or(0.0),
        );
    }
}

fn per_depth_to_json(per_depth: &HashMap<usize, f64>) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = per_depth
        .iter()
        .map(|(depth, acc)| (depth.to_string(), serde_json::json!(acc)))
        .collect();
    serde_json::Value::Object(map)
}

fn axis_to_json(
    organ: &ReasonEvalSummary,
    handoff_stats: &HandoffAxisStats,
    symbolic: &HandoffAxisStats,
) -> serde_json::Value {
    serde_json::json!({
        "organ_alone": {
            "n": organ.n,
            "answer_accuracy": organ.answer_accuracy,
            "per_depth_accuracy": per_depth_to_json(&organ_alone_per_depth_accuracy(organ)),
        },
        "handoff": {
            "n": handoff_stats.n,
            "answer_accuracy": handoff_stats.answer_accuracy,
            "mean_neural_hops": handoff_stats.mean_neural_hops,
            "mean_symbolic_hops": handoff_stats.mean_symbolic_hops,
            "organ_answer_used_rate": handoff_stats.organ_answer_used_rate,
            "composed_executability_mean": handoff_stats.composed_executability_mean,
            "per_depth_accuracy": per_depth_to_json(&handoff_stats.per_depth_accuracy),
        },
        "symbolic_alone": {
            "n": symbolic.n,
            "answer_accuracy": symbolic.answer_accuracy,
            "mean_neural_hops": symbolic.mean_neural_hops,
            "mean_symbolic_hops": symbolic.mean_symbolic_hops,
            "composed_executability_mean": symbolic.composed_executability_mean,
            "per_depth_accuracy": per_depth_to_json(&symbolic.per_depth_accuracy),
        },
    })
}
