//! Phase 51 P1a-4 Step B: train the pointer-generator model (ADR 087, plan
//! `docs/ai/plans/2026-07-05-phase51-p1a4-zeroshot-copy-mechanism.md`).
//!
//! Mirrors `train_p1a` (same data, tokenizer, 13M trunk, easy 2-distractor
//! distribution as `p1a2-ep8`) but trains a [`grug_train::PointerTransformer`]
//! (trunk + copy/pointer head) with an NLL loss over the mixed distribution.
//! The A/B question: does the copy mechanism move the zero-shot (whole-tool
//! holdout) axis off the ~0.03 floor without regressing in-distribution
//! arg-correct below ~0.75?
//!
//! Every eval logs **plain greedy** two-axis numbers (directly comparable to
//! the non-pointer baseline axis-B arg 0.78 / axis-A schema 0.03). The final
//! step additionally runs a **schema-constrained** two-axis pass (copy + Step-A
//! grammar together).
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example train_p1a_ptr --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --train-examples 40000 --epochs 8 \
//!     --batch-size 16 --learning-rate 3e-4 --eval-every 500 --ckpt-every 1000 \
//!     --seed 42 --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p1a4-ptr
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig};
use grug_train::eval::{
    FrozenTwoAxisHoldout, run_two_axis_eval_ptr, run_two_axis_eval_ptr_constrained,
};
use grug_train::generate::GenerateConfig;
use grug_train::tokenizer::Vocab;
use grug_train::train::{
    build_batch, init_training_ptr, to_eval_model_ptr, train_step_ptr, train_step_ptr_supervised,
};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, PointerTransformerConfig, checkpoint};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_p1a_ptr")]
struct Args {
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
    #[arg(long, default_value_t = 1000)]
    ckpt_every: usize,
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,
    #[arg(long, default_value = "/home/toasty/ws/agi/akh-medu/runs/p1a4-ptr")]
    artifacts_dir: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Distractor count (matches p1a2-ep8's 2). Uniform-random (easy).
    #[arg(long, default_value_t = 2)]
    distractor_count: usize,
    /// Phase 51 P1a-4 Step B2: copy-supervision BCE weight. `0.0` = plain
    /// marginal-likelihood pointer training (Step B); `>0` forces `p_gen`
    /// toward copy on copyable tokens (Step B2, the gate-collapse fix).
    #[arg(long, default_value_t = 0.0)]
    copy_supervision: f64,
}

fn build_dataset(args: &Args) -> grug_data::GrugResult<Dataset> {
    let distractors = DistractorConfig::easy_with_count(args.distractor_count);
    match args.n_tools {
        Some(n) => {
            Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, distractors)
        }
        None => Dataset::new(args.seed),
    }
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.artifacts_dir).expect("can create artifacts dir");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("train_p1a_ptr: GPU backend (Burn ROCm/HIP), device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_p1a_ptr: CPU backend (burn-ndarray)");
        Default::default()
    };

    let dataset = build_dataset(&args).expect("dataset splits cleanly");
    println!(
        "train_p1a_ptr: catalog train_tools={} holdout_tools={} (easy distractors: {})",
        dataset.train_tools().len(),
        dataset.holdout_tools().len(),
        args.distractor_count,
    );

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(
        &dataset,
        &tokenizer_path,
        args.train_examples,
        args.bpe_vocab_size,
    );
    println!(
        "train_p1a_ptr: BPE tokenizer vocab_size={}",
        tokenizer.vocab_size()
    );

    let base_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = base_config.max_seq_len;
    let model_config = PointerTransformerConfig::new(base_config);

    // Prompt-length sanity (same overflow risk as train_p1a: build_batch
    // silently truncates the target if prompt+target exceeds max_seq_len).
    {
        let sample = dataset.train_examples(256).unwrap_or_default();
        let mut lens: Vec<usize> = sample
            .iter()
            .filter_map(|ex| {
                grug_train::train::encode_example(&tokenizer, ex)
                    .ok()
                    .map(|e| e.ids.len())
            })
            .collect();
        lens.sort_unstable();
        let max = lens.last().copied().unwrap_or(0);
        let over = lens.iter().filter(|&&l| l > max_seq_len).count();
        println!(
            "train_p1a_ptr: PROMPT-LEN check (n={}) max={max} ctx={max_seq_len} over_ctx={over}",
            lens.len()
        );
    }

    let (mut model, mut optim) = init_training_ptr::<ActiveBackend>(&model_config, &device);
    let param_count = burn::module::Module::num_params(&model);
    println!("train_p1a_ptr: PointerTransformer(~{param_count} params over 13M trunk + copy head)");
    if args.copy_supervision > 0.0 {
        println!(
            "train_p1a_ptr: COPY-SUPERVISION ON (Step B2), lambda={}",
            args.copy_supervision
        );
    } else {
        println!("train_p1a_ptr: copy-supervision OFF (plain marginal-likelihood, Step B)");
    }

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        println!("train_p1a_ptr: resuming from {}", resume_dir.display());
        let (m, o, step) = checkpoint::load_checkpoint_ptr(resume_dir, model, optim, &device)
            .expect("checkpoint should load");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_p1a_ptr: resumed at step {start_step}");
    }

    let train_examples = dataset
        .train_examples(args.train_examples)
        .expect("generator produces train examples");
    let holdout =
        FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("holdout draws cleanly");
    let pieces = Vocab::token_pieces(&tokenizer);

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };
    let eval_log_path = args.artifacts_dir.join("eval_axes.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let train_log_path = args.artifacts_dir.join("train.jsonl");

    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!(
        "train_p1a_ptr: training {total_steps} steps ({} epochs), batch={}, lr={}",
        args.epochs, args.batch_size, args.learning_rate,
    );

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<_> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch =
            build_batch::<ActiveBackend, _>(&tokenizer, &batch_examples, max_seq_len, &device)
                .expect("BPE encoding should not fail");

        let step_start = Instant::now();
        let (next_model, loss) = if args.copy_supervision > 0.0 {
            train_step_ptr_supervised(
                model,
                &mut optim,
                &batch,
                args.learning_rate,
                args.copy_supervision,
            )
        } else {
            train_step_ptr(model, &mut optim, &batch, args.learning_rate)
        };
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;
        last_loss = loss;

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} ({step_ms:.1} ms/step)");
            append_train_log_row(&train_log_path, step, loss, step_ms);
            if loss.is_nan() {
                eprintln!(
                    "train_p1a_ptr: LOSS IS NaN at step {step} — aborting (see grad-clip / lr)."
                );
                break;
            }
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model_ptr(&model);
            let (b, a) =
                run_two_axis_eval_ptr(&eval_model, &tokenizer, &holdout, gen_config, &device);
            log_axis_rows(&eval_log_path, step, "plain", &b, &a);
            println!(
                "  eval@{step} PLAIN  axis_b: schema={:.3} arg={:.3} | axis_a: schema={:.3} arg={:.3} div={:.3}",
                b.eval.schema_adherence_rate,
                b.eval.arg_correct_rate,
                a.eval.schema_adherence_rate,
                a.eval.arg_correct_rate,
                a.diversity.distinct_call_ratio,
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint_ptr(&ckpt_dir, step, &model, &optim)
                .expect("checkpoint save should succeed");
            println!("  checkpoint -> {}", ckpt_dir.display());
        }
    }

    // Final: constrained (copy + Step-A grammar) two-axis pass.
    {
        let eval_model = to_eval_model_ptr(&model);
        let (b, a) = run_two_axis_eval_ptr_constrained(
            &eval_model,
            &tokenizer,
            &pieces,
            &holdout,
            gen_config,
            &device,
        );
        log_axis_rows(&eval_log_path, total_steps, "constrained", &b, &a);
        println!(
            "  FINAL CONSTRAINED  axis_b: schema={:.3} arg={:.3} | axis_a: schema={:.3} arg={:.3} div={:.3}",
            b.eval.schema_adherence_rate,
            b.eval.arg_correct_rate,
            a.eval.schema_adherence_rate,
            a.eval.arg_correct_rate,
            a.diversity.distinct_call_ratio,
        );
    }

    println!(
        "train_p1a_ptr: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display(),
    );
}

fn load_or_train_tokenizer(
    dataset: &Dataset,
    path: &Path,
    train_examples: usize,
    vocab_size: usize,
) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json should load");
    }
    let corpus_examples = dataset
        .train_examples(train_examples.clamp(200, 60_000))
        .expect("generator produces train examples for tokenizer corpus");
    let corpus: Vec<String> = corpus_examples
        .iter()
        .flat_map(|ex| {
            vec![
                grug_train::train::format_prompt(ex),
                ex.expected_call_as_json_string(),
            ]
        })
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, vocab_size).expect("BPE training should succeed");
    tokenizer.save(path).expect("tokenizer save should succeed");
    tokenizer
}

fn log_axis_rows(
    path: &Path,
    step: usize,
    mode: &str,
    b: &grug_data::AxisReport,
    a: &grug_data::AxisReport,
) {
    use std::io::Write as _;
    let row = |axis: &str, r: &grug_data::AxisReport| {
        serde_json::json!({
            "step": step, "mode": mode, "axis": axis,
            "valid_json_rate": r.eval.valid_json_rate,
            "schema_adherence_rate": r.eval.schema_adherence_rate,
            "arg_correct_rate": r.eval.arg_correct_rate,
            "distinct_call_ratio": r.diversity.distinct_call_ratio,
            "tool_name_entropy": r.diversity.tool_name_entropy,
            "n": r.eval.n,
        })
        .to_string()
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{}", row("held_out_phrasing", b));
        let _ = writeln!(f, "{}", row("whole_tool_holdout", a));
    }
}

fn append_train_log_row(path: &Path, step: usize, loss: f32, step_ms: f64) {
    use std::io::Write as _;
    let line = serde_json::json!({ "step": step, "loss": loss, "step_ms": step_ms }).to_string();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}
