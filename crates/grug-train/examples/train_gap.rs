//! Phase 51 P3c — gap-token abstention (ADR 089 mechanism 3, plan
//! `docs/ai/plans/2026-07-07-phase51-p2-format-first-corpus.md`).
//!
//! Everything is held constant from the P2 `train_format_ab` recipe (seed 42,
//! 1000 procedural tools, 13M base [`TinyTransformer`], High-Grug format, easy
//! 2-distractors); the ONE change is the corpus: a fraction `--abstain-frac`
//! of training examples are **abstention** examples — a request phrased for a
//! tool that is *not* in the offered bundle, whose target is the gap token
//! (`gap`), rather than a concrete call.
//!
//! The claim under test (ADR 089): unlike open-world zero-shot (four measured
//! negatives), abstention — "none of these offered tools fit" — is an
//! in-distribution classification the weights *can* learn. So we watch two
//! rates alongside the usual call accuracy:
//!   - **abstain_recall** should climb high (the model learns to emit gap), and
//!   - **false_abstention_rate** should stay low (it does not start wrongly
//!     abstaining on requests a tool *does* cover),
//!     while axis-B call arg-correct holds (no regression from the mixed corpus).
//!
//! Sparse checkpointing by default (`--ckpt-every 2000`) — the P2 run filled
//! the disk writing every 500 across many runs; do not repeat that.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example train_gap --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --train-examples 40000 --abstain-frac 0.2 \
//!     --epochs 8 --batch-size 16 --eval-every 500 --ckpt-every 2000 --seed 42 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3c-gap
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::eval::{
    FrozenAbstentionHoldout, FrozenTwoAxisHoldout, run_abstention_eval_fmt, run_two_axis_eval_fmt,
};
use grug_train::generate::GenerateConfig;
use grug_train::train::{build_batch_fmt, init_training, to_eval_model, train_step};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_gap")]
struct Args {
    /// Boundary format. Grug (the locked-in default, ADR 088); json still works.
    #[arg(long, default_value = "grug")]
    format: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    n_tools: Option<usize>,
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,
    #[arg(long, default_value_t = 40_000)]
    train_examples: usize,
    /// Fraction of training examples that are abstention (gap-target) examples.
    #[arg(long, default_value_t = 0.2)]
    abstain_frac: f64,
    #[arg(long, default_value_t = 8)]
    epochs: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,
    #[arg(long, default_value_t = 3_000)]
    bpe_vocab_size: usize,
    #[arg(long, default_value_t = 128)]
    holdout_examples: usize,
    #[arg(long, default_value_t = 500)]
    eval_every: usize,
    #[arg(long, default_value_t = 2_000)]
    ckpt_every: usize,
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,
    #[arg(long, default_value_t = 2)]
    distractor_count: usize,
    #[arg(long)]
    artifacts_dir: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let fmt = TargetFormat::parse_label(&args.format).expect("--format must be json or grug");
    std::fs::create_dir_all(&args.artifacts_dir).expect("can create artifacts dir");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("train_gap: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_gap: CPU backend");
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
    println!(
        "train_gap: FORMAT={:?} | train_tools={} holdout_tools={} distractors={} abstain_frac={}",
        fmt,
        dataset.train_tools().len(),
        dataset.holdout_tools().len(),
        args.distractor_count,
        args.abstain_frac,
    );

    let train_examples = dataset
        .train_examples_with_abstention(args.train_examples, args.abstain_frac)
        .expect("train examples (mixed call + abstention)");
    let n_abstain = train_examples.iter().filter(|e| e.is_abstention()).count();
    println!(
        "train_gap: corpus n={} of which abstention={} ({:.1}%)",
        train_examples.len(),
        n_abstain,
        100.0 * n_abstain as f64 / train_examples.len().max(1) as f64,
    );

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer =
        load_or_train_tokenizer(&train_examples, &tokenizer_path, args.bpe_vocab_size, fmt);
    println!(
        "train_gap: BPE tokenizer vocab_size={}",
        tokenizer.vocab_size()
    );

    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);
    println!(
        "train_gap: base TinyTransformer ~{} params",
        burn::module::Module::num_params(&model)
    );

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        let (m, o, step) = checkpoint::load_checkpoint(resume_dir, model, optim, &device)
            .expect("checkpoint loads");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_gap: resumed at step {start_step}");
    }

    let call_holdout =
        FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("call holdout draws");
    let abstention_holdout =
        FrozenAbstentionHoldout::new(&dataset, args.holdout_examples, args.holdout_examples)
            .expect("abstention holdout draws");
    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };

    let eval_log_path = args.artifacts_dir.join("eval_gap.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!(
        "train_gap: {total_steps} steps ({} epochs), batch={}, lr={}",
        args.epochs, args.batch_size, args.learning_rate
    );

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<_> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch = build_batch_fmt::<ActiveBackend, _>(
            &tokenizer,
            &batch_examples,
            max_seq_len,
            &device,
            fmt,
        )
        .expect("encoding should not fail");

        let step_start = Instant::now();
        let (next_model, loss) = train_step(model, &mut optim, &batch, args.learning_rate);
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;
        last_loss = loss;

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} ({step_ms:.1} ms/step)");
            if loss.is_nan() {
                eprintln!("train_gap: NaN at step {step} — aborting.");
                break;
            }
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model(&model);
            let (b, a) = run_two_axis_eval_fmt(
                &eval_model,
                &tokenizer,
                &call_holdout,
                gen_config,
                &device,
                fmt,
            );
            let gap = run_abstention_eval_fmt(
                &eval_model,
                &tokenizer,
                &abstention_holdout,
                gen_config,
                &device,
                fmt,
            );
            log_gap_row(&eval_log_path, step, args.format.as_str(), &b, &a, &gap);
            println!(
                "  eval@{step}  call_axisB: schema={:.3} arg={:.3} | GAP: recall={:.3} false_abstain={:.3} precision={:.3} (n_call={} n_abstain={})",
                b.eval.schema_adherence_rate,
                b.eval.arg_correct_rate,
                gap.abstain_recall,
                gap.false_abstention_rate,
                gap.abstain_precision,
                gap.n_call,
                gap.n_abstain,
            );
            let _ = a; // axis-A whole-tool holdout stays flat (zero-shot ~0.03); logged, not printed.
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save");
        }
    }

    println!(
        "train_gap: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display(),
    );
}

/// Train (or load) the BPE tokenizer over the **abstention-mixed** corpus, so
/// the gap marker (`gap` / `{"gap":true}`) is represented in the vocabulary
/// alongside the call renders.
fn load_or_train_tokenizer(
    train_examples: &[grug_data::schema::Example],
    path: &Path,
    vocab_size: usize,
    fmt: TargetFormat,
) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json loads");
    }
    let corpus: Vec<String> = train_examples
        .iter()
        .take(60_000)
        .flat_map(|ex| {
            vec![
                grug_train::train::format_prompt_fmt(ex, fmt),
                grug_data::render_call(&ex.expected_call, fmt),
            ]
        })
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, vocab_size).expect("BPE training");
    tokenizer.save(path).expect("tokenizer save");
    tokenizer
}

fn log_gap_row(
    path: &Path,
    step: usize,
    fmt_label: &str,
    b: &grug_data::AxisReport,
    a: &grug_data::AxisReport,
    gap: &grug_data::AbstentionReport,
) {
    use std::io::Write as _;
    let row = serde_json::json!({
        "step": step,
        "format": fmt_label,
        "call_axis_b": {
            "schema_adherence_rate": b.eval.schema_adherence_rate,
            "arg_correct_rate": b.eval.arg_correct_rate,
        },
        "call_axis_a": {
            "schema_adherence_rate": a.eval.schema_adherence_rate,
            "arg_correct_rate": a.eval.arg_correct_rate,
        },
        "abstain_recall": gap.abstain_recall,
        "false_abstention_rate": gap.false_abstention_rate,
        "abstain_precision": gap.abstain_precision,
        "n_call": gap.n_call,
        "n_abstain": gap.n_abstain,
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
