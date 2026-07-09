//! Phase 51 P3a full-stack: the M2 KG-repair pass measured under M1's
//! retrieval-scoped offered set (ADR 089). Inference only — loads a trained
//! `train_registry` checkpoint and reports registry-wide call reliability
//! **raw and M2-repaired**, alongside retrieval recall@k (M1 premise) and the
//! abstention confusion matrix (M3), so the full stack
//! M1-retrieve → propose → M2-repair → M3-gap is one number.
//!
//! The difference from P3b's `eval_repair`: there the offered set was easy-2
//! random distractors (3 tools); here it is the retriever's top-8
//! request-keyed *confusables*, so this measures how much symbolic grounding
//! recovers on the realistic, harder offered set the organ actually presents.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example eval_registry_repair --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --retrieval-k 8 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3a-registry \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p3a-registry/checkpoints/step_00019999
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::eval::{run_retrieval_eval_fmt, FrozenRetrievalHoldout};
use grug_train::generate::GenerateConfig;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{checkpoint, BpeTokenizer};

#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "eval_registry_repair")]
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
    #[arg(long, default_value_t = 8)]
    retrieval_k: usize,
    #[arg(long, default_value_t = 256)]
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
        println!("eval_registry_repair: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("eval_registry_repair: CPU backend");
        Default::default()
    };

    let distractors = DistractorConfig::easy_with_count(args.distractor_count);
    let dataset = match args.n_tools {
        Some(n) => Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors),
        None => Dataset::new(args.seed),
    }
    .expect("dataset splits cleanly");

    let tokenizer = BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) =
        checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint loads");
    let model = to_eval_model(&model);
    println!("eval_registry_repair: FORMAT={fmt:?} checkpoint step {step} | registry={} out_of_registry={}",
        dataset.registry().len(), dataset.holdout_tools().len());

    let holdout = FrozenRetrievalHoldout::new(
        &dataset,
        args.retrieval_k,
        args.holdout_examples,
        args.holdout_examples,
        args.holdout_examples,
    )
    .expect("retrieval holdout draws");
    let gen_config = GenerateConfig { max_new_tokens: args.eval_max_new_tokens, max_seq_len };

    let r = run_retrieval_eval_fmt(&model, &tokenizer, &holdout, gen_config, &device, fmt);

    println!("\n=== Phase 51 P3a full-stack — M1 retrieve → propose → M2 repair → M3 gap ===");
    println!("M1  retrieval recall@{}     : {:.3}  (target present in offered set)", args.retrieval_k, r.recall_at_k);
    println!("    registry call (n={})", r.n_call);
    println!(
        "      raw (neural)          : schema={:.3} arg(whole-call)={:.3} arg(per-value)={:.3}",
        r.call.schema_adherence_rate, r.call.arg_correct_rate, r.call.arg_value_accuracy
    );
    println!(
        "M2  repaired (name-grounded): schema={:.3} arg(whole-call)={:.3} arg(per-value)={:.3}",
        r.call_repaired.schema_adherence_rate, r.call_repaired.arg_correct_rate, r.call_repaired.arg_value_accuracy
    );
    println!(
        "P3a grounded (value-grounded): schema={:.3} arg(whole-call)={:.3} arg(per-value)={:.3}",
        r.call_grounded.schema_adherence_rate, r.call_grounded.arg_correct_rate, r.call_grounded.arg_value_accuracy
    );
    println!(
        "    Δ (raw→grounded)          : whole-call {:+.3}   per-value {:+.3}",
        r.call_grounded.arg_correct_rate - r.call.arg_correct_rate,
        r.call_grounded.arg_value_accuracy - r.call.arg_value_accuracy,
    );
    println!(
        "M3  abstention (n_abstain={})  : recall={:.3} false_abstain={:.3} precision={:.3}",
        r.n_abstain, r.abstention.abstain_recall, r.abstention.false_abstention_rate, r.abstention.abstain_precision,
    );

    let summary = serde_json::json!({
        "checkpoint_step": step,
        "format": args.format,
        "retrieval_k": args.retrieval_k,
        "recall_at_k": r.recall_at_k,
        "n_call": r.n_call,
        "n_abstain": r.n_abstain,
        "call_raw": { "schema": r.call.schema_adherence_rate, "arg_whole_call": r.call.arg_correct_rate, "arg_per_value": r.call.arg_value_accuracy },
        "call_repaired": { "schema": r.call_repaired.schema_adherence_rate, "arg_whole_call": r.call_repaired.arg_correct_rate, "arg_per_value": r.call_repaired.arg_value_accuracy },
        "call_grounded": { "schema": r.call_grounded.schema_adherence_rate, "arg_whole_call": r.call_grounded.arg_correct_rate, "arg_per_value": r.call_grounded.arg_value_accuracy },
        "abstain_recall": r.abstention.abstain_recall,
        "false_abstention_rate": r.abstention.false_abstention_rate,
        "abstain_precision": r.abstention.abstain_precision,
    });
    let out = args.artifacts_dir.join("eval_registry_repair.json");
    match std::fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()) {
        Ok(()) => println!("\neval_registry_repair: wrote {}", out.display()),
        Err(e) => eprintln!("eval_registry_repair: write failed: {e}"),
    }
}
