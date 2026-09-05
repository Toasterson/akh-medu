//! Phase 51 P3b: measure the KG-backed repair pass (ADR 089, plan
//! `docs/ai/plans/2026-07-07-phase51-p2-format-first-corpus.md`).
//!
//! Inference only. Loads a trained base [`grug_train::TinyTransformer`]
//! checkpoint, greedy-generates on both holdout axes once, and scores each
//! **raw** and **KG-repaired** (`grug_data::repair` grounds/repairs the call
//! against the offered schemas). The raw→repaired delta is what the symbolic
//! System-2 layer recovers on top of the neural System-1 proposal — the claim
//! of ADR 089 that in-distribution correctness is a repair problem, not a
//! bigger-model problem.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example eval_repair --features gpu --release -- \
//!     --format grug --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p2-grug \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p2-grug/checkpoints/step_00019999 \
//!     --n-tools 1000 --holdout-frac 0.15 --holdout-examples 128
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::eval::{FrozenTwoAxisHoldout, run_two_axis_eval_raw_and_repaired_fmt};
use grug_train::generate::GenerateConfig;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "eval_repair")]
struct Args {
    #[arg(long, default_value = "grug")]
    format: String,
    #[arg(long)]
    artifacts_dir: PathBuf,
    #[arg(long)]
    resume: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    n_tools: Option<usize>,
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,
    #[arg(long, default_value_t = 128)]
    holdout_examples: usize,
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,
    #[arg(long, default_value_t = 2)]
    distractor_count: usize,
}

fn main() {
    let args = Args::parse();
    let fmt = TargetFormat::parse_label(&args.format).expect("--format json|grug");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("eval_repair: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("eval_repair: CPU backend");
        Default::default()
    };

    let distractors = DistractorConfig::easy_with_count(args.distractor_count);
    let dataset = match args.n_tools {
        Some(n) => {
            Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors)
        }
        None => Dataset::new(args.seed),
    }
    .expect("dataset splits cleanly");

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) =
        checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint loads");
    let model = to_eval_model(&model);
    println!("eval_repair: FORMAT={fmt:?} checkpoint step {step}");

    let holdout =
        FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("holdout draws");
    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };

    let ((raw_b, raw_a), (rep_b, rep_a)) = run_two_axis_eval_raw_and_repaired_fmt(
        &model, &tokenizer, &holdout, gen_config, &device, fmt,
    );

    let line = |tag: &str, r: &grug_data::AxisReport| {
        println!(
            "{tag:>34}: schema={:.3} arg_correct={:.3} (n={})",
            r.eval.schema_adherence_rate, r.eval.arg_correct_rate, r.eval.n
        );
    };

    println!("\n=== Phase 51 P3b — KG-backed repair pass (raw → repaired) ===");
    println!("--- axis B (held_out_phrasing, in-distribution) ---");
    line("raw (neural only)", &raw_b);
    line("repaired (KG-grounded)", &rep_b);
    println!(
        "{:>34}: schema {:+.3}   arg_correct {:+.3}",
        "Δ lift",
        rep_b.eval.schema_adherence_rate - raw_b.eval.schema_adherence_rate,
        rep_b.eval.arg_correct_rate - raw_b.eval.arg_correct_rate,
    );
    println!("--- axis A (whole_tool_holdout, zero-shot) ---");
    line("raw (neural only)", &raw_a);
    line("repaired (KG-grounded)", &rep_a);
    println!(
        "{:>34}: schema {:+.3}   arg_correct {:+.3}",
        "Δ lift",
        rep_a.eval.schema_adherence_rate - raw_a.eval.schema_adherence_rate,
        rep_a.eval.arg_correct_rate - raw_a.eval.arg_correct_rate,
    );

    let summary = serde_json::json!({
        "checkpoint_step": step,
        "format": args.format,
        "axis_b_raw": { "schema": raw_b.eval.schema_adherence_rate, "arg": raw_b.eval.arg_correct_rate },
        "axis_b_repaired": { "schema": rep_b.eval.schema_adherence_rate, "arg": rep_b.eval.arg_correct_rate },
        "axis_a_raw": { "schema": raw_a.eval.schema_adherence_rate, "arg": raw_a.eval.arg_correct_rate },
        "axis_a_repaired": { "schema": rep_a.eval.schema_adherence_rate, "arg": rep_a.eval.arg_correct_rate },
    });
    let out = args.artifacts_dir.join("eval_repair.json");
    match std::fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()) {
        Ok(()) => println!("\neval_repair: wrote {}", out.display()),
        Err(e) => eprintln!("eval_repair: write failed: {e}"),
    }
}
