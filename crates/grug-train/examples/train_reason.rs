//! Phase 51 P3-R -- High-Grug v1 reasoning-register training run (ADR 092,
//! `docs/ai/plans/2026-07-11-phase51-p3r-reasoning-corpus.md`).
//!
//! Mirrors `train_registry.rs`'s recipe exactly: per-run BPE tokenizer
//! trained over the corpus, `TinyTransformerConfig::p1a` (~13M params),
//! batch 16, lr 3e-4, Adam grad-clip Norm 1.0 (via `init_training`),
//! PROMPT-LEN sanity check before the long run, `--ckpt-every` checkpointing,
//! an eval JSONL time series, and the same CPU/GPU backend-swap-by-feature
//! pattern. The corpus/scoring side is `grug_data::reason` instead of
//! `grug_data::{Dataset, schema}`; the batch/encode plumbing is
//! `grug_train::reason_train` instead of `grug_train::train`'s
//! `format_prompt_fmt`/`encode_example_fmt`/`build_batch_fmt` (which are
//! tool-schema-specific).
//!
//! `--arm trace|answer` selects [`TraceArm`] for the **training stream**
//! only -- the frozen axis-B/axis-A/abstention holdouts are always drawn as
//! full [`TraceArm::Trace`] targets (`ReasonDataset::axis_b_examples` /
//! `axis_a_examples` / `abstention_examples` don't take an arm), and eval
//! always scores whatever the model actually emits via
//! `grug_data::reason::score_reason_outputs`, never by comparing against
//! those target strings.
//!
//! ```text
//! # CPU smoke test:
//! cargo run -p grug-train --example train_reason -- \
//!     --train-examples 64 --epochs 1 --batch-size 4 --eval-every 10 --ckpt-every 20 \
//!     --artifacts-dir /tmp/p3r-smoke
//!
//! # Real GPU runs (source scripts/train-env.sh first) -- see the task's two
//! # invocations (trace arm, answer arm) at the bottom of this file's summary.
//! cargo run -p grug-train --example train_reason --features gpu --release -- \
//!     --arm trace --seed 42 --train-examples 40000 --epochs 8 --batch-size 16 \
//!     --eval-every 500 --ckpt-every 2000 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3r-trace
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::reason::{
    ReasonConfig, ReasonDataset, ReasonExample, TraceArm, score_reason_outputs,
};
use grug_train::generate::GenerateConfig;
use grug_train::reason_train::{build_reason_batch, format_reason_prompt, generate_reason_outputs};
use grug_train::train::{init_training, to_eval_model, train_step};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_reason")]
struct Args {
    /// Which target format the TRAINING stream is rendered in: `trace` (full
    /// High-Grug v1 trace, intermediate-supervision arm) or `answer`
    /// (answer-only control). The frozen eval holdouts are always scored via
    /// replay against whatever the model emits, regardless of this flag.
    #[arg(long, default_value = "trace")]
    arm: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 300)]
    entities_per_type: usize,

    #[arg(long, default_value_t = 2)]
    distractor_factor: usize,

    /// Comma-separated training depths, e.g. `2,3,4` (the plan default).
    #[arg(long, default_value = "2,3,4")]
    train_depths: String,

    #[arg(long, default_value_t = 0.15)]
    holdout_subject_frac: f64,

    /// Axis-A (length-generalization) depths, never in `--train-depths`.
    #[arg(long, default_value = "5,6")]
    axis_a_depths: String,

    #[arg(long, default_value_t = 40_000)]
    train_examples: usize,

    #[arg(long, default_value_t = 8)]
    epochs: usize,

    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,

    #[arg(long, default_value_t = 3_000)]
    bpe_vocab_size: usize,

    /// Frozen holdout size per axis (axis-B, axis-A, abstention each draw
    /// this many, once, before training).
    #[arg(long, default_value_t = 128)]
    holdout_examples: usize,

    #[arg(long, default_value_t = 500)]
    eval_every: usize,

    #[arg(long, default_value_t = 2_000)]
    ckpt_every: usize,

    /// Max tokens to greedily generate per eval example -- must cover a
    /// depth-6 trace (recall/derive/call-answer lines), hence the generous
    /// default versus the tool-calling runs' ~96.
    #[arg(long, default_value_t = 160)]
    eval_max_new_tokens: usize,

    #[arg(long)]
    artifacts_dir: PathBuf,

    #[arg(long)]
    resume: Option<PathBuf>,
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
    std::fs::create_dir_all(&args.artifacts_dir).expect("can create artifacts dir");

    let arm = TraceArm::parse_label(&args.arm).expect("--arm must be trace|answer");
    let train_depths = parse_depths(&args.train_depths, "--train-depths");
    let axis_a_depths = parse_depths(&args.axis_a_depths, "--axis-a-depths");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("train_reason: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_reason: CPU backend");
        Default::default()
    };

    let dataset = ReasonDataset::new(ReasonConfig {
        seed: args.seed,
        entities_per_type: args.entities_per_type,
        distractor_factor: args.distractor_factor,
        train_depths: train_depths.clone(),
        holdout_subject_frac: args.holdout_subject_frac,
    })
    .expect("ReasonConfig builds a valid dataset");
    println!(
        "train_reason: ARM={arm:?} entities_per_type={} distractor_factor={} train_depths={:?} axis_a_depths={:?}",
        args.entities_per_type, args.distractor_factor, train_depths, axis_a_depths,
    );

    println!(
        "train_reason: generating {} training examples (seed {})...",
        args.train_examples, args.seed
    );
    let train_examples = dataset
        .train_examples(args.train_examples, arm)
        .expect("generator produces train examples");

    // Frozen holdouts, drawn once before training (per the plan): axis-B
    // n=128, axis-A n=128 (depths 5,6), abstention n=128 by default.
    println!(
        "train_reason: drawing frozen holdouts (axis-B, axis-A, abstention) n={} each...",
        args.holdout_examples
    );
    let axis_b_holdout = dataset
        .axis_b_examples(args.holdout_examples)
        .expect("axis-B holdout draws");
    let axis_a_holdout = dataset
        .axis_a_examples(args.holdout_examples, &axis_a_depths)
        .expect("axis-A holdout draws");
    let abstention_holdout = dataset
        .abstention_examples(args.holdout_examples)
        .expect("abstention holdout draws");

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(&train_examples, &tokenizer_path, args.bpe_vocab_size);
    println!(
        "train_reason: BPE tokenizer vocab_size={} ({})",
        tokenizer.vocab_size(),
        if tokenizer_path.exists() {
            "loaded"
        } else {
            "freshly trained"
        }
    );

    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    // PROMPT-LEN sanity check (mirrors train_p1a.rs/train_registry.rs): if
    // prompt+target exceeds max_seq_len, build_reason_batch silently
    // truncates the TARGET off the end and corrupts the loss. A depth-6
    // trace with distractors is the longest sequence this corpus produces --
    // measure it before a long run.
    {
        let mut probe: Vec<ReasonExample> = dataset
            .train_examples(128.min(args.train_examples), arm)
            .unwrap_or_default();
        probe.extend(
            dataset
                .axis_a_examples(64, &axis_a_depths)
                .unwrap_or_default(),
        );
        let mut lens: Vec<usize> = probe
            .iter()
            .filter_map(|ex| {
                grug_train::reason_train::encode_reason_example(&tokenizer, ex)
                    .ok()
                    .map(|e| e.ids.len())
            })
            .collect();
        lens.sort_unstable();
        let max = lens.last().copied().unwrap_or(0);
        let p95 = lens
            .get(lens.len().saturating_mul(95) / 100)
            .copied()
            .unwrap_or(0);
        let over = lens.iter().filter(|&&l| l > max_seq_len).count();
        println!(
            "train_reason: PROMPT-LEN check (n={}, incl. depth-6 axis-A probe) max={max} p95={p95} ctx={max_seq_len} over_ctx={over}",
            lens.len(),
        );
    }

    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);
    println!(
        "train_reason: TinyTransformer ~{} params",
        burn::module::Module::num_params(&model)
    );

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        let (m, o, step) = checkpoint::load_checkpoint(resume_dir, model, optim, &device)
            .expect("checkpoint loads");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_reason: resumed at step {start_step}");
    }

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };
    let eval_log_path = args.artifacts_dir.join("eval_reason.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!(
        "train_reason: {total_steps} steps ({} epochs), batch={}, lr={}, eval JSONL -> {}",
        args.epochs,
        args.batch_size,
        args.learning_rate,
        eval_log_path.display(),
    );

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<ReasonExample> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch = build_reason_batch::<ActiveBackend, _>(
            &tokenizer,
            &batch_examples,
            max_seq_len,
            &device,
        )
        .expect("encoding should not fail on generator output");

        let step_start = Instant::now();
        let (next_model, loss) = train_step(model, &mut optim, &batch, args.learning_rate);
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;
        last_loss = loss;

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} ({step_ms:.1} ms/step)");
            if loss.is_nan() {
                eprintln!("train_reason: NaN at step {step} -- aborting.");
                break;
            }
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model(&model);
            let axis_b_outputs = generate_reason_outputs(
                &eval_model,
                &tokenizer,
                &axis_b_holdout,
                gen_config,
                &device,
            );
            let axis_a_outputs = generate_reason_outputs(
                &eval_model,
                &tokenizer,
                &axis_a_holdout,
                gen_config,
                &device,
            );
            let abstention_outputs = generate_reason_outputs(
                &eval_model,
                &tokenizer,
                &abstention_holdout,
                gen_config,
                &device,
            );

            let axis_b_summary = score_reason_outputs(&axis_b_holdout, &axis_b_outputs);
            let axis_a_summary = score_reason_outputs(&axis_a_holdout, &axis_a_outputs);
            let abstention_summary = score_reason_outputs(&abstention_holdout, &abstention_outputs);

            println!(
                "  eval@{step} [axis-B in-dist]  answer_acc={:.3} exec={:.3} trace_valid={:.3} (n={})",
                axis_b_summary.answer_accuracy,
                axis_b_summary.executability_rate,
                axis_b_summary.trace_valid_rate,
                axis_b_summary.n,
            );
            println!(
                "  eval@{step} [axis-A len-gen ]  answer_acc={:.3} exec={:.3} trace_valid={:.3} (n={})",
                axis_a_summary.answer_accuracy,
                axis_a_summary.executability_rate,
                axis_a_summary.trace_valid_rate,
                axis_a_summary.n,
            );
            println!(
                "  eval@{step} [abstention     ]  recall={:.3} precision={:.3} (n_expected={})",
                abstention_summary.abstention.recall,
                abstention_summary.abstention.precision,
                abstention_summary.abstention.n_expected,
            );

            log_eval_row(
                &eval_log_path,
                step,
                args.arm.as_str(),
                &axis_b_summary,
                &axis_a_summary,
                &abstention_summary,
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save");
            println!("  checkpoint saved -> {}", ckpt_dir.display());
        }
    }

    println!(
        "train_reason: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display(),
    );
}

fn load_or_train_tokenizer(
    train_examples: &[ReasonExample],
    path: &Path,
    vocab_size: usize,
) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json loads");
    }
    let corpus: Vec<String> = train_examples
        .iter()
        .take(60_000)
        .flat_map(|ex| vec![format_reason_prompt(ex).to_string(), ex.target.clone()])
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, vocab_size).expect("BPE training");
    tokenizer.save(path).expect("tokenizer save");
    tokenizer
}

fn log_eval_row(
    path: &Path,
    step: usize,
    arm_label: &str,
    axis_b: &grug_data::reason::ReasonEvalSummary,
    axis_a: &grug_data::reason::ReasonEvalSummary,
    abstention: &grug_data::reason::ReasonEvalSummary,
) {
    use std::io::Write as _;
    let row = serde_json::json!({
        "step": step,
        "arm": arm_label,
        "axis_b": {
            "n": axis_b.n,
            "answer_accuracy": axis_b.answer_accuracy,
            "executability_rate": axis_b.executability_rate,
            "trace_valid_rate": axis_b.trace_valid_rate,
            "distinct_answer_ratio": axis_b.diversity.distinct_answer_ratio,
            "answer_entropy": axis_b.diversity.answer_entropy,
        },
        "axis_a": {
            "n": axis_a.n,
            "answer_accuracy": axis_a.answer_accuracy,
            "executability_rate": axis_a.executability_rate,
            "trace_valid_rate": axis_a.trace_valid_rate,
            "distinct_answer_ratio": axis_a.diversity.distinct_answer_ratio,
            "answer_entropy": axis_a.diversity.answer_entropy,
        },
        "abstention": {
            "n_expected": abstention.abstention.n_expected,
            "n_emitted": abstention.abstention.n_emitted,
            "recall": abstention.abstention.recall,
            "precision": abstention.abstention.precision,
        },
    })
    .to_string();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{row}");
    }
}
