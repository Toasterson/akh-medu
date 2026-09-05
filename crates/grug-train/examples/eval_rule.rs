//! Phase 51 P4 inference-only eval: load a `train_rule` checkpoint and report
//! the full `select/bind/fire/chain` breakdown (Q5) + per-depth answer/exec on
//! the held-out streams. The capacity-ladder readout compares `chain_rate`
//! across a p1a / p1a_50m / p1a_100m checkpoint set — a positive slope says
//! "scale it", flat-at-floor says "broken corpus, not capacity".
//!
//! The dataset config MUST match the training run's (same seed + world knobs)
//! so the eval streams are the same held-out individuals.
//!
//! ```text
//! cargo run -p grug-train --example eval_rule --features gpu --release -- \
//!     --logic rule --typed-slots true --model p1a \
//!     --artifacts-dir runs/p4-rule-typed \
//!     --resume runs/p4-rule-typed/checkpoints/step_00019999
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{
    RuleCorpusConfig, RuleDataset, RuleEvalSummary, RuleExample, RuleWorldConfig, TraceArm,
    score_rule_outputs,
};
use grug_train::generate::GenerateConfig;
use grug_train::rule_train::generate_rule_outputs;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "eval_rule")]
struct Args {
    #[arg(long)]
    artifacts_dir: PathBuf,
    #[arg(long)]
    resume: PathBuf,

    #[arg(long, default_value = "rule")]
    logic: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    typed_slots: bool,
    #[arg(long, default_value = "p1a")]
    model: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 6)]
    preds_per_cell: u32,
    #[arg(long, default_value_t = 8)]
    max_level: u32,
    #[arg(long, default_value_t = 0.5)]
    guard_prob: f64,
    #[arg(long, default_value_t = 4)]
    distractor_facts: usize,
    #[arg(long, default_value_t = 6)]
    distractor_rules: usize,
    #[arg(long, default_value_t = 0.5)]
    yes_frac: f64,
    #[arg(long, default_value = "2,4")]
    prop_literals: String,

    #[arg(long, default_value = "2,3,4")]
    train_depths: String,
    #[arg(long, default_value = "5,6")]
    eval_depths: String,

    #[arg(long, default_value_t = 512)]
    n: usize,
    #[arg(long, default_value_t = 220)]
    eval_max_new_tokens: usize,
    /// Override the model context window — MUST match the training run's
    /// `--max-seq-len` (the checkpoint's position-embedding table is that size).
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,
}

fn parse_depths(s: &str, flag: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag}: not an integer: {t:?}"))
        })
        .collect()
}

fn model_config(name: &str, vocab: usize) -> TinyTransformerConfig {
    match name {
        "spike" => TinyTransformerConfig::spike(vocab),
        "p1a" => TinyTransformerConfig::p1a(vocab),
        "p1a_large" => TinyTransformerConfig::p1a_large(vocab),
        "p1a_50m" => TinyTransformerConfig::p1a_50m(vocab),
        "p1a_100m" => TinyTransformerConfig::p1a_100m(vocab),
        "p1a_350m" => TinyTransformerConfig::p1a_350m(vocab),
        other => {
            panic!("--model must be spike|p1a|p1a_large|p1a_50m|p1a_100m|p1a_350m, got {other:?}")
        }
    }
}

fn main() {
    let args = Args::parse();
    let is_prop = matches!(args.logic.as_str(), "prop" | "propositional");
    let train_depths = parse_depths(&args.train_depths, "--train-depths");
    let eval_depths = parse_depths(&args.eval_depths, "--eval-depths");
    let plits = parse_depths(&args.prop_literals, "--prop-literals");
    assert_eq!(plits.len(), 2, "--prop-literals must be lo,hi");

    // Vendor-agnostic GPU device (ROCm under `--features gpu`, CUDA under
    // `--features cuda`) so this example builds unchanged on the local 7900 XTX
    // and on an H100 cloud box.
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let device = {
        let d = grug_train::gpu_device();
        println!("eval_rule: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let device = {
        println!("eval_rule: CPU backend");
        Default::default()
    };

    let dataset = RuleDataset::new(RuleCorpusConfig {
        world: RuleWorldConfig {
            seed: args.seed,
            preds_per_cell: args.preds_per_cell,
            max_level: args.max_level,
            guard_prob: args.guard_prob,
        },
        distractor_facts: args.distractor_facts,
        distractor_rules: args.distractor_rules,
        yes_frac: args.yes_frac,
        prop_literals: (plits[0], plits[1]),
    })
    .expect("RuleCorpusConfig must match the training run's config");

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let mut cfg = model_config(&args.model, tokenizer.vocab_size());
    if args.max_seq_len > 0 {
        cfg.max_seq_len = args.max_seq_len;
    }
    let max_seq_len = cfg.max_seq_len;

    let (model, optim) = init_training::<ActiveBackend>(&cfg, &device);
    let (model, _optim, step) =
        checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint loads");
    let model = to_eval_model(&model);

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };

    println!(
        "\n=== Phase 51 P4 -- LOGIC={} TYPED={} MODEL={} step {step} ===",
        args.logic, args.typed_slots, args.model
    );

    let in_dist: Vec<RuleExample> = if is_prop {
        dataset.propositional_examples_eval(args.n, TraceArm::Trace, args.typed_slots)
    } else {
        dataset.rule_examples_eval(args.n, &train_depths, TraceArm::Trace, args.typed_slots)
    };
    let in_out = generate_rule_outputs(&model, &tokenizer, &in_dist, gen_config, &device);
    let in_summary = score_rule_outputs(&in_dist, &in_out);
    print_summary_table("in-distribution", &in_summary);

    let len_summary = if is_prop {
        None
    } else {
        let lg =
            dataset.rule_examples_eval(args.n, &eval_depths, TraceArm::Trace, args.typed_slots);
        let lg_out = generate_rule_outputs(&model, &tokenizer, &lg, gen_config, &device);
        let s = score_rule_outputs(&lg, &lg_out);
        print_summary_table("length-generalization", &s);
        Some(s)
    };

    let summary_json = serde_json::json!({
        "checkpoint_step": step,
        "logic": args.logic,
        "typed_slots": args.typed_slots,
        "model": args.model,
        "in_distribution": summary_to_json(&in_summary),
        "length_generalization": len_summary.as_ref().map(summary_to_json),
    });
    let out = args.artifacts_dir.join("eval_rule.json");
    match std::fs::write(&out, serde_json::to_string_pretty(&summary_json).unwrap()) {
        Ok(()) => println!("\neval_rule: wrote {}", out.display()),
        Err(e) => eprintln!("eval_rule: write failed: {e}"),
    }
}

fn print_summary_table(label: &str, s: &RuleEvalSummary) {
    let c = &s.components;
    println!(
        "\n{label:<24}: n={} ans={:.3} exec={:.3} valid={:.3}",
        s.n, s.answer_accuracy, s.executability_rate, s.trace_valid_rate
    );
    println!(
        "  components           : select={:.3} bind={:.3} fire={:.3} chain={:.3} (fires={}, chain_expected={})",
        c.select_rate, c.bind_rate, c.fire_rate, c.chain_rate, c.fire_total, c.chain_expected_total
    );
    let mut depths: Vec<&usize> = s.per_depth.keys().collect();
    depths.sort();
    for depth in depths {
        let d = &s.per_depth[depth];
        println!(
            "    depth {depth}: n={:<4} ans={:.3} exec={:.3}",
            d.n, d.answer_accuracy, d.executability_rate
        );
    }
}

fn summary_to_json(s: &RuleEvalSummary) -> serde_json::Value {
    let per_depth: serde_json::Map<String, serde_json::Value> = s
        .per_depth
        .iter()
        .map(|(depth, st)| {
            (
                depth.to_string(),
                serde_json::json!({
                    "n": st.n,
                    "answer_accuracy": st.answer_accuracy,
                    "executability_rate": st.executability_rate,
                }),
            )
        })
        .collect();
    serde_json::json!({
        "n": s.n,
        "answer_accuracy": s.answer_accuracy,
        "executability_rate": s.executability_rate,
        "trace_valid_rate": s.trace_valid_rate,
        "components": {
            "select_rate": s.components.select_rate,
            "bind_rate": s.components.bind_rate,
            "fire_rate": s.components.fire_rate,
            "chain_rate": s.components.chain_rate,
            "fire_total": s.components.fire_total,
            "chain_expected_total": s.components.chain_expected_total,
        },
        "per_depth": per_depth,
    })
}
