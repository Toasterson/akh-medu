//! Phase 51 P2: JSON-vs-High-Grug format A/B on the base 13M model (ADR 088,
//! plan `docs/ai/plans/2026-07-07-phase51-p2-format-first-corpus.md`).
//!
//! Everything is held constant from the `p1a2-ep8` recipe (seed 42, 1000
//! procedural tools, 13M base [`TinyTransformer`] — NO pointer, 8 epochs, easy
//! 2-distractors); the ONLY variable is `--format {json,grug}`, which switches
//! *both* the prompt-side schema and the emitted call between JSON and the flat
//! High-Grug v0 format. The hypothesis: removing JSON's structural tax lets the
//! tiny model spend capacity on schema-reading + value-grounding, lifting
//! in-distribution arg-correct and — the real prize — zero-shot schema, where
//! every *model* lever (capacity, copy, copy-supervision) failed.
//!
//! Frequent snapshotting (`--ckpt-every`, default 500) per the P2 directive:
//! every capability jump is bisectable and resumable.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example train_format_ab --features gpu --release -- \
//!     --format grug --n-tools 1000 --holdout-frac 0.15 --train-examples 40000 \
//!     --epochs 8 --batch-size 16 --eval-every 500 --ckpt-every 500 --seed 42 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p2-grug
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::eval::{run_two_axis_eval_fmt, FrozenTwoAxisHoldout};
use grug_train::generate::GenerateConfig;
use grug_train::train::{build_batch_fmt, init_training, to_eval_model, train_step};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{checkpoint, BpeTokenizer};

#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_format_ab")]
struct Args {
    /// Boundary format: `json` (baseline) or `grug` (High-Grug v0, the treatment).
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
    #[arg(long, default_value_t = 500)]
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
        println!("train_format_ab: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_format_ab: CPU backend");
        Default::default()
    };

    let distractors = DistractorConfig::easy_with_count(args.distractor_count);
    let dataset = match args.n_tools {
        Some(n) => Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors),
        None => Dataset::new(args.seed),
    }
    .expect("dataset splits cleanly");
    println!(
        "train_format_ab: FORMAT={:?} | train_tools={} holdout_tools={} distractors={}",
        fmt,
        dataset.train_tools().len(),
        dataset.holdout_tools().len(),
        args.distractor_count,
    );

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(&dataset, &tokenizer_path, args.train_examples, args.bpe_vocab_size, fmt);
    println!("train_format_ab: BPE tokenizer vocab_size={}", tokenizer.vocab_size());

    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    // Prompt-length sanity (the Grug format is shorter than JSON, but check).
    {
        let sample = dataset.train_examples(256).unwrap_or_default();
        let mut lens: Vec<usize> = sample
            .iter()
            .filter_map(|ex| grug_train::train::encode_example_fmt(&tokenizer, ex, fmt).ok().map(|e| e.ids.len()))
            .collect();
        lens.sort_unstable();
        let max = lens.last().copied().unwrap_or(0);
        let median = lens.get(lens.len() / 2).copied().unwrap_or(0);
        let over = lens.iter().filter(|&&l| l > max_seq_len).count();
        println!("train_format_ab: SEQ-LEN (n={}) median={median} max={max} ctx={max_seq_len} over_ctx={over}", lens.len());
    }

    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);
    println!("train_format_ab: base TinyTransformer ~{} params", burn::module::Module::num_params(&model));

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        let (m, o, step) = checkpoint::load_checkpoint(resume_dir, model, optim, &device).expect("checkpoint loads");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_format_ab: resumed at step {start_step}");
    }

    let train_examples = dataset.train_examples(args.train_examples).expect("train examples");
    let holdout = FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("holdout draws");
    let gen_config = GenerateConfig { max_new_tokens: args.eval_max_new_tokens, max_seq_len };

    let eval_log_path = args.artifacts_dir.join("eval_axes.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!("train_format_ab: {total_steps} steps ({} epochs), batch={}, lr={}", args.epochs, args.batch_size, args.learning_rate);

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<_> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch = build_batch_fmt::<ActiveBackend, _>(&tokenizer, &batch_examples, max_seq_len, &device, fmt)
            .expect("encoding should not fail");

        let step_start = Instant::now();
        let (next_model, loss) = train_step(model, &mut optim, &batch, args.learning_rate);
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;
        last_loss = loss;

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} ({step_ms:.1} ms/step)");
            if loss.is_nan() {
                eprintln!("train_format_ab: NaN at step {step} — aborting.");
                break;
            }
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model(&model);
            let (b, a) = run_two_axis_eval_fmt(&eval_model, &tokenizer, &holdout, gen_config, &device, fmt);
            log_axis_rows(&eval_log_path, step, args.format.as_str(), &b, &a);
            println!(
                "  eval@{step}  axis_b: valid={:.3} schema={:.3} arg={:.3} | axis_a: valid={:.3} schema={:.3} arg={:.3} div={:.3}",
                b.eval.valid_json_rate, b.eval.schema_adherence_rate, b.eval.arg_correct_rate,
                a.eval.valid_json_rate, a.eval.schema_adherence_rate, a.eval.arg_correct_rate, a.diversity.distinct_call_ratio,
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save");
        }
    }

    println!(
        "train_format_ab: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display(),
    );
}

fn load_or_train_tokenizer(
    dataset: &Dataset,
    path: &Path,
    train_examples: usize,
    vocab_size: usize,
    fmt: TargetFormat,
) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json loads");
    }
    let corpus_examples = dataset
        .train_examples(train_examples.min(60_000).max(200))
        .expect("train examples for tokenizer corpus");
    let corpus: Vec<String> = corpus_examples
        .iter()
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

fn log_axis_rows(path: &Path, step: usize, fmt_label: &str, b: &grug_data::AxisReport, a: &grug_data::AxisReport) {
    use std::io::Write as _;
    let row = |axis: &str, r: &grug_data::AxisReport| {
        serde_json::json!({
            "step": step, "format": fmt_label, "axis": axis,
            "valid_json_rate": r.eval.valid_json_rate,
            "schema_adherence_rate": r.eval.schema_adherence_rate,
            "arg_correct_rate": r.eval.arg_correct_rate,
            "distinct_call_ratio": r.diversity.distinct_call_ratio,
            "tool_name_entropy": r.diversity.tool_name_entropy,
            "n": r.eval.n,
        })
        .to_string()
    };
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", row("held_out_phrasing", b));
        let _ = writeln!(f, "{}", row("whole_tool_holdout", a));
    }
}
