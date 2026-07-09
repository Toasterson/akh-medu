//! Phase 51 P3a — retrieval-scoped closed tool universe (ADR 089 mechanism 1,
//! plan `docs/ai/plans/2026-07-07-phase51-p2-format-first-corpus.md`).
//!
//! ADR 089's first mechanism: there is no genuine open-world zero-shot in normal
//! operation. The tool **registry** is closed and the model trains on all of it;
//! at runtime a **retriever** surfaces the top-K request-relevant tools and only
//! those are offered. So the whole-tool-holdout scenario (the neural floor at
//! ~0.03) is one the organ never faces — zero-shot is dissolved by *scoping*.
//!
//! This run realises that: the corpus is retrieval-scoped (offered set = the
//! retriever's top-K over the registry), and the label is a function of the
//! offered set — **call** when retrieval surfaced the target, **gap** when it
//! did not (a retrieval miss, or a genuine out-of-registry request). It couples
//! mechanism 1 to mechanism 3 (P3c): "call when the offered set covers the
//! request, abstain when it doesn't."
//!
//! Three things are measured every eval:
//!   1. **retrieval recall@k** (model-independent) — the premise: does the
//!      offered set contain the target? Printed once at startup + logged.
//!   2. **registry-wide call reliability** — schema/arg-correct given a
//!      retrieval-selected offered set that covered the request.
//!   3. **abstention** — gap recall (retrieval miss + genuine novelty) and
//!      false-abstention (wrongly gapping a covered request).
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example train_registry --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --retrieval-k 8 --novelty-frac 0.2 \
//!     --train-examples 40000 --epochs 8 --batch-size 16 --eval-every 500 \
//!     --ckpt-every 2000 --seed 42 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3a-registry
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::eval::{run_retrieval_eval_fmt, FrozenRetrievalHoldout};
use grug_train::generate::GenerateConfig;
use grug_train::train::{build_batch_fmt, init_training, to_eval_model, train_step};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{checkpoint, BpeTokenizer};

#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_registry")]
struct Args {
    #[arg(long, default_value = "grug")]
    format: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    n_tools: Option<usize>,
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,
    /// Top-K tools the retriever offers per request.
    #[arg(long, default_value_t = 8)]
    retrieval_k: usize,
    /// Fraction of training examples that are genuine out-of-registry novelty
    /// (must-abstain) examples.
    #[arg(long, default_value_t = 0.2)]
    novelty_frac: f64,
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
        println!("train_registry: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_registry: CPU backend");
        Default::default()
    };

    // `distractor_count` only seeds the base generator; the offered set is
    // replaced by the retriever's top-K, so the effective offered size is K.
    let distractors = DistractorConfig::easy_with_count(args.distractor_count);
    let dataset = match args.n_tools {
        Some(n) => Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors),
        None => Dataset::new(args.seed),
    }
    .expect("dataset splits cleanly");
    println!(
        "train_registry: FORMAT={:?} | registry(train)={} out_of_registry(holdout)={} retrieval_k={} novelty_frac={}",
        fmt,
        dataset.registry().len(),
        dataset.holdout_tools().len(),
        args.retrieval_k,
        args.novelty_frac,
    );

    let train_examples = dataset
        .retrieval_train_mix(args.retrieval_k, args.train_examples, args.novelty_frac)
        .expect("retrieval-scoped train mix");
    let n_gap = train_examples.iter().filter(|e| e.is_abstention()).count();
    println!(
        "train_registry: corpus n={} of which gap-labelled={} ({:.1}%) [novelty + retrieval-miss]",
        train_examples.len(),
        n_gap,
        100.0 * n_gap as f64 / train_examples.len().max(1) as f64,
    );

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(&train_examples, &tokenizer_path, args.bpe_vocab_size, fmt);
    println!("train_registry: BPE tokenizer vocab_size={}", tokenizer.vocab_size());

    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);
    println!("train_registry: base TinyTransformer ~{} params", burn::module::Module::num_params(&model));

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        let (m, o, step) = checkpoint::load_checkpoint(resume_dir, model, optim, &device).expect("checkpoint loads");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_registry: resumed at step {start_step}");
    }

    let holdout = FrozenRetrievalHoldout::new(
        &dataset,
        args.retrieval_k,
        args.holdout_examples,
        args.holdout_examples,
        args.holdout_examples.max(256),
    )
    .expect("retrieval holdout draws");
    println!(
        "train_registry: ★ retrieval recall@{} = {:.3} (model-independent premise: fraction of requests whose target is in the offered top-K)",
        args.retrieval_k,
        holdout.recall_at_k(),
    );
    let gen_config = GenerateConfig { max_new_tokens: args.eval_max_new_tokens, max_seq_len };

    let eval_log_path = args.artifacts_dir.join("eval_registry.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!("train_registry: {total_steps} steps ({} epochs), batch={}, lr={}", args.epochs, args.batch_size, args.learning_rate);

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
                eprintln!("train_registry: NaN at step {step} — aborting.");
                break;
            }
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model(&model);
            let r = run_retrieval_eval_fmt(&eval_model, &tokenizer, &holdout, gen_config, &device, fmt);
            log_registry_row(&eval_log_path, step, args.format.as_str(), args.retrieval_k, &r);
            println!(
                "  eval@{step}  recall@{}={:.3} | registry_call: schema={:.3} arg={:.3} → repaired arg={:.3} (n={}) | GAP: recall={:.3} false_abstain={:.3} precision={:.3} (n_abstain={})",
                args.retrieval_k, r.recall_at_k,
                r.call.schema_adherence_rate, r.call.arg_correct_rate, r.call_repaired.arg_correct_rate, r.n_call,
                r.abstention.abstain_recall, r.abstention.false_abstention_rate, r.abstention.abstain_precision, r.n_abstain,
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save");
        }
    }

    println!(
        "train_registry: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display(),
    );
}

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

fn log_registry_row(path: &Path, step: usize, fmt_label: &str, k: usize, r: &grug_train::RetrievalEval) {
    use std::io::Write as _;
    let row = serde_json::json!({
        "step": step,
        "format": fmt_label,
        "retrieval_k": k,
        "recall_at_k": r.recall_at_k,
        "registry_call": {
            "schema_adherence_rate": r.call.schema_adherence_rate,
            "arg_correct_rate": r.call.arg_correct_rate,
            "n": r.n_call,
        },
        "registry_call_repaired": {
            "schema_adherence_rate": r.call_repaired.schema_adherence_rate,
            "arg_correct_rate": r.call_repaired.arg_correct_rate,
        },
        "abstain_recall": r.abstention.abstain_recall,
        "false_abstention_rate": r.abstention.false_abstention_rate,
        "abstain_precision": r.abstention.abstain_precision,
        "n_abstain": r.n_abstain,
    })
    .to_string();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{row}");
    }
}
