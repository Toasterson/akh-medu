//! Phase 51 P1a-4 Step A: schema-constrained-decoding eval of an existing
//! checkpoint (ADR 087, plan `docs/ai/plans/2026-07-05-phase51-p1a4-zeroshot-copy-mechanism.md`).
//!
//! Inference only — no training. Loads a trained tokenizer + a saved
//! [`grug_train::TinyTransformer`] checkpoint, reconstructs the *same* frozen
//! two-axis holdout the training run evaluated on, and runs the holdout twice:
//! once with ordinary greedy decoding (the baseline — this should reproduce the
//! checkpoint's training-run eval numbers, confirming the load is faithful) and
//! once with **schema-constrained decoding** (the treatment). It prints both
//! side by side so the delta answers Step A's question: does forcing valid,
//! schema-consistent parameter names lift schema-adherence — especially the
//! zero-shot whole-tool axis — and how far does argument-correctness follow?
//!
//! ```text
//! # Reproduce + constrain the P1a-2 (8-epoch, 13M, 1000-tool) checkpoint:
//! source scripts/train-env.sh
//! cargo run -p grug-train --example eval_constrained --features gpu --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p1a2-ep8 \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p1a2-ep8/checkpoints/step_00019999 \
//!     --n-tools 1000 --holdout-frac 0.15 --holdout-examples 128
//! ```
//!
//! Distractor defaults match the *original* P1a-2 run (2 uniform-random
//! distractors — see `DistractorConfig::easy`), which is what the `p1a2-ep8`
//! checkpoint trained and evaluated under; pass `--hard-distractors true` /
//! `--distractor-count N` only to match a run that used those.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig};
use grug_train::eval::{FrozenTwoAxisHoldout, run_two_axis_eval, run_two_axis_eval_constrained};
use grug_train::generate::GenerateConfig;
use grug_train::tokenizer::Vocab;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "eval_constrained")]
struct Args {
    /// Run artifacts dir holding `tokenizer.json` (and, by default,
    /// `checkpoints/`). Must be the dir the checkpoint was trained under so
    /// the tokenizer matches.
    #[arg(long)]
    artifacts_dir: PathBuf,

    /// Checkpoint directory to load (e.g. `<artifacts>/checkpoints/step_00019999`).
    #[arg(long)]
    resume: PathBuf,

    /// Base RNG seed — must match the training run to reproduce its frozen
    /// holdout.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of procedural tools (must match the training run). Absent =
    /// hand-authored built-in catalog.
    #[arg(long)]
    n_tools: Option<usize>,

    /// Whole-tool holdout fraction (must match the training run).
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,

    /// Examples per holdout axis (128 matches the P1a-2 run).
    #[arg(long, default_value_t = 128)]
    holdout_examples: usize,

    /// Max tokens greedily generated per eval example (96 matches the P1a-2 run).
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,

    /// Number of distractor tools offered alongside the target (must match the
    /// training run; P1a-2 used 2).
    #[arg(long, default_value_t = 2)]
    distractor_count: usize,

    /// Use hard (similarity-biased) distractors. Default false = uniform-random
    /// (the original P1a-2 behavior the `p1a2-ep8` checkpoint used).
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    hard_distractors: bool,

    /// Load the ~30M `p1a_large` config instead of ~13M `p1a` (must match the
    /// training run).
    #[arg(long, default_value_t = false)]
    large_model: bool,
}

fn build_dataset(args: &Args) -> grug_data::GrugResult<Dataset> {
    let distractors = if args.hard_distractors {
        DistractorConfig::hard_with_count(args.distractor_count)
    } else {
        DistractorConfig::easy_with_count(args.distractor_count)
    };
    match args.n_tools {
        Some(n) => {
            Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors)
        }
        None => Dataset::new(args.seed),
    }
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("eval_constrained: GPU backend (Burn ROCm/HIP), device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("eval_constrained: CPU backend (burn-ndarray)");
        Default::default()
    };

    let dataset = build_dataset(&args).expect("dataset splits cleanly");
    println!(
        "eval_constrained: catalog train_tools={} holdout_tools={} (distractors: {} {})",
        dataset.train_tools().len(),
        dataset.holdout_tools().len(),
        args.distractor_count,
        if args.hard_distractors {
            "hard"
        } else {
            "easy"
        },
    );

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::load(&tokenizer_path).expect("tokenizer.json should load");
    println!(
        "eval_constrained: tokenizer vocab_size={}",
        tokenizer.vocab_size()
    );

    let model_config = if args.large_model {
        TinyTransformerConfig::p1a_large(tokenizer.vocab_size())
    } else {
        TinyTransformerConfig::p1a(tokenizer.vocab_size())
    };

    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) = checkpoint::load_checkpoint(&args.resume, model, optim, &device)
        .expect("checkpoint should load");
    println!("eval_constrained: loaded checkpoint at step {step}");

    let eval_model = to_eval_model(&model);
    let pieces = Vocab::token_pieces(&tokenizer);

    let holdout =
        FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("holdout draws cleanly");
    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len: model_config.max_seq_len,
    };

    // --- Baseline (unconstrained) — should reproduce the training-run eval. ---
    let t0 = Instant::now();
    let (base_b, base_a) =
        run_two_axis_eval(&eval_model, &tokenizer, &holdout, gen_config, &device);
    println!(
        "eval_constrained: baseline eval in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // --- Treatment (schema-constrained). ---
    let t1 = Instant::now();
    let (con_b, con_a) = run_two_axis_eval_constrained(
        &eval_model,
        &tokenizer,
        &pieces,
        &holdout,
        gen_config,
        &device,
    );
    println!(
        "eval_constrained: constrained eval in {:.1}s",
        t1.elapsed().as_secs_f64()
    );

    let line = |tag: &str, r: &grug_data::AxisReport| {
        format!(
            "{tag:>28}: valid_json={:.3} schema={:.3} arg_correct={:.3} distinct={:.3} entropy={:.3} (n={})",
            r.eval.valid_json_rate,
            r.eval.schema_adherence_rate,
            r.eval.arg_correct_rate,
            r.diversity.distinct_call_ratio,
            r.diversity.tool_name_entropy,
            r.eval.n,
        )
    };

    println!("\n=== Phase 51 P1a-4 Step A — schema-constrained decoding ===");
    println!("--- axis B (held_out_phrasing, in-distribution) ---");
    println!("{}", line("baseline (unconstrained)", &base_b));
    println!("{}", line("constrained", &con_b));
    println!("--- axis A (whole_tool_holdout, zero-shot) ---");
    println!("{}", line("baseline (unconstrained)", &base_a));
    println!("{}", line("constrained", &con_a));

    let summary = serde_json::json!({
        "checkpoint_step": step,
        "resume": args.resume.display().to_string(),
        "holdout_examples": args.holdout_examples,
        "eval_max_new_tokens": args.eval_max_new_tokens,
        "distractor_count": args.distractor_count,
        "hard_distractors": args.hard_distractors,
        "axis_b_baseline": axis_json(&base_b),
        "axis_b_constrained": axis_json(&con_b),
        "axis_a_baseline": axis_json(&base_a),
        "axis_a_constrained": axis_json(&con_a),
    });
    let out_path = args.artifacts_dir.join("eval_constrained.json");
    match std::fs::write(&out_path, serde_json::to_string_pretty(&summary).unwrap()) {
        Ok(()) => println!("\neval_constrained: wrote {}", out_path.display()),
        Err(e) => eprintln!(
            "eval_constrained: could not write {}: {e}",
            out_path.display()
        ),
    }
}

fn axis_json(r: &grug_data::AxisReport) -> serde_json::Value {
    serde_json::json!({
        "valid_json_rate": r.eval.valid_json_rate,
        "schema_adherence_rate": r.eval.schema_adherence_rate,
        "arg_correct_rate": r.eval.arg_correct_rate,
        "distinct_call_ratio": r.diversity.distinct_call_ratio,
        "tool_name_entropy": r.diversity.tool_name_entropy,
        "n": r.eval.n,
    })
}
