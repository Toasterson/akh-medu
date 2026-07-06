//! Phase 51 P1a scale-up training run (ADR 087, Phase 51 plan P1a bullet).
//!
//! Self-contained CLI-configurable driver: generates training + both-axis
//! holdout data from `grug-data` by seed, trains (or loads) a BPE tokenizer
//! over the training corpus, trains [`grug_train::TinyTransformer`] with the
//! P1a-scale config, logs both holdout axes to JSONL every `--eval-every`
//! steps, and checkpoints (model + optimizer + step) every `--ckpt-every`
//! steps -- resumable via `--resume`.
//!
//! ```text
//! # CPU smoke test (default features):
//! cargo run -p grug-train --example train_p1a -- --steps 50 --train-examples 64
//!
//! # Real GPU run (source scripts/train-env.sh first):
//! cargo run -p grug-train --example train_p1a --features gpu --release -- \
//!     --steps 20000 --train-examples 20000 --epochs 5 \
//!     --eval-every 250 --ckpt-every 500 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p1a
//! ```
//!
//! Every artifact (trained tokenizer, eval JSONL, checkpoints, this run's
//! resolved config) is written under `--artifacts-dir`, which defaults to
//! `/home/toasty/ws/agi/akh-medu/runs/p1a` -- an absolute path outside any
//! worktree, per the Phase 51 plan's ops note, so a run survives whichever
//! worktree launched it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::Dataset;
use grug_train::eval::{run_two_axis_eval_and_log, FrozenTwoAxisHoldout};
use grug_train::generate::GenerateConfig;
use grug_train::train::{build_batch, init_training, train_step, TrainConfig};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{checkpoint, BpeTokenizer};

// Phase 51 P0 gate (R3) / P1a: the training backend is a pure type-alias
// swap -- default (no `gpu` feature) trains on CPU (`burn-ndarray`);
// `--features gpu` swaps in Burn's ROCm/HIP (CubeCL) backend targeting this
// box's 7900 XTX (gfx1100). No other line in this file changes. Source
// `scripts/train-env.sh` first so `hipconfig` (shimmed) and `ROCM_PATH`/
// `HIP_PATH` are on PATH/set before building or running with `--features gpu`.
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

/// Phase 51 P1a: scale-up training run for the Grug-Brain BabyLLM tool-calling
/// spike. See `docs/ai/plans/2026-07-05-phase51-grug-brain-baby-llm.md`.
#[derive(Parser, Debug)]
#[command(name = "train_p1a")]
struct Args {
    /// Base RNG seed for `grug_data::Dataset` (train/holdout streams derive
    /// their own independent seeds from this one).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Phase 51 P1a-2 (Wave 2): number of tools in the **procedural** catalog
    /// (`grug_data::Dataset::new_sized`), hundreds-to-thousands, replacing
    /// the hand-authored ~56-tool catalog so per-tool example counts stay low
    /// enough that memorizing tool->call shape is infeasible (see
    /// `grug_data::procedural` module docs -- this is the direct response to
    /// P1a's whole-tool-holdout collapse: schema=0.0, diversity=1/128).
    /// When absent, falls back to the original `Dataset::new` (hand-authored
    /// catalog) path for backward compatibility.
    #[arg(long)]
    n_tools: Option<usize>,

    /// Fraction of `--n-tools` reserved for the whole-tool holdout axis
    /// (axis A). Only meaningful when `--n-tools` is given.
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,

    /// Number of distinct training examples to generate (drawn once, then
    /// cycled over `--epochs` passes). The plan's P1a target is 10k-50k.
    #[arg(long, default_value_t = 20_000)]
    train_examples: usize,

    /// Number of passes over the generated training set.
    #[arg(long, default_value_t = 5)]
    epochs: usize,

    /// Batch size.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Adam learning rate.
    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,

    /// BPE vocabulary size ceiling (P1a target: ~2-4K).
    #[arg(long, default_value_t = 3_000)]
    bpe_vocab_size: usize,

    /// Number of examples per holdout axis (both axis A and axis B draw this
    /// many).
    #[arg(long, default_value_t = 64)]
    holdout_examples: usize,

    /// Run a two-axis eval pass every N steps.
    #[arg(long, default_value_t = 250)]
    eval_every: usize,

    /// Save a checkpoint every N steps.
    #[arg(long, default_value_t = 500)]
    ckpt_every: usize,

    /// Max tokens to greedily generate per eval example.
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,

    /// Where to write the trained tokenizer, eval JSONL, and checkpoints.
    /// Defaults to an absolute path outside any worktree so a detached run's
    /// artifacts survive the worktree that launched it.
    #[arg(long, default_value = "/home/toasty/ws/agi/akh-medu/runs/p1a")]
    artifacts_dir: PathBuf,

    /// Resume from a checkpoint directory (e.g.
    /// `<artifacts-dir>/checkpoints/step_00001500`) instead of training from
    /// scratch. The model config must match the checkpoint's -- this run's
    /// `--*` flags that affect architecture (currently none are exposed
    /// beyond the fixed P1a config) must be unchanged from the original run.
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.artifacts_dir).expect("can create artifacts dir");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("train_p1a: GPU feature enabled -- training on Burn ROCm/HIP (CubeCL) backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("train_p1a: CPU backend (burn-ndarray)");
        Default::default()
    };

    let dataset = build_dataset(args.seed, args.n_tools, args.holdout_frac).expect("dataset splits cleanly");
    match args.n_tools {
        Some(n) => println!(
            "train_p1a: procedural catalog -- n_tools={n}, holdout_frac={}, train_tools={}, holdout_tools={}",
            args.holdout_frac,
            dataset.train_tools().len(),
            dataset.holdout_tools().len(),
        ),
        None => println!(
            "train_p1a: hand-authored built-in catalog -- train_tools={}, holdout_tools={}",
            dataset.train_tools().len(),
            dataset.holdout_tools().len(),
        ),
    }

    // --- Tokenizer: train fresh, or reuse an already-trained one under this
    // artifacts dir (so re-running the same command twice, e.g. after a
    // crash before the first checkpoint, doesn't waste time retraining an
    // identical BPE vocabulary). ---
    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(&dataset, &tokenizer_path, args.train_examples, args.bpe_vocab_size);
    println!(
        "train_p1a: BPE tokenizer ready, vocab_size={} ({})",
        tokenizer.vocab_size(),
        if tokenizer_path.exists() { "loaded" } else { "freshly trained" }
    );

    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let train_cfg = TrainConfig {
        learning_rate: args.learning_rate,
        max_seq_len: model_config.max_seq_len,
    };

    // Prompt-length sanity: with hard distractors the offered-tool JSON grows;
    // if prompt+target exceeds max_seq_len, `build_batch` silently truncates the
    // TARGET off the end and corrupts the loss. Measure before a long run.
    {
        let sample = dataset.train_examples(256).unwrap_or_default();
        let mut lens: Vec<usize> = sample
            .iter()
            .filter_map(|ex| grug_train::train::encode_example(&tokenizer, ex).ok().map(|e| e.ids.len()))
            .collect();
        lens.sort_unstable();
        let max = lens.last().copied().unwrap_or(0);
        let p95 = lens.get(lens.len().saturating_mul(95) / 100).copied().unwrap_or(0);
        let over = lens.iter().filter(|&&l| l > model_config.max_seq_len).count();
        println!(
            "train_p1a: PROMPT-LEN check (n={}) max={max} p95={p95} ctx={} over_ctx={over}",
            lens.len(),
            model_config.max_seq_len
        );
    }

    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);
    let param_count = burn::module::Module::num_params(&model);
    println!(
        "train_p1a: TinyTransformer(vocab={}, d_model={}, n_heads={}, n_layers={}, d_ff={}, max_seq_len={}), ~{param_count} params",
        model_config.vocab_size,
        model_config.d_model,
        model_config.n_heads,
        model_config.n_layers,
        model_config.d_ff,
        model_config.max_seq_len,
    );

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        println!("train_p1a: resuming from checkpoint {}", resume_dir.display());
        let (loaded_model, loaded_optim, step) =
            checkpoint::load_checkpoint(resume_dir, model, optim, &device).expect("checkpoint should load");
        model = loaded_model;
        optim = loaded_optim;
        start_step = step + 1;
        println!("train_p1a: resumed at step {start_step}");
    }

    println!(
        "train_p1a: generating {} training examples (seed {})...",
        args.train_examples, args.seed
    );
    let train_examples = dataset
        .train_examples(args.train_examples)
        .expect("generator produces train examples");

    println!(
        "train_p1a: drawing {} frozen holdout examples per axis...",
        args.holdout_examples
    );
    let holdout = FrozenTwoAxisHoldout::new(&dataset, args.holdout_examples).expect("generator produces holdout examples");

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len: model_config.max_seq_len,
    };

    let eval_log_path = args.artifacts_dir.join("eval_axes.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let train_log_path = args.artifacts_dir.join("train.jsonl");

    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!(
        "train_p1a: training for {total_steps} steps ({} epochs x {} steps/epoch), batch_size={}, lr={}",
        args.epochs,
        total_steps / args.epochs.max(1),
        args.batch_size,
        args.learning_rate,
    );
    println!("train_p1a: eval JSONL -> {}", eval_log_path.display());
    println!("train_p1a: checkpoints -> {}", ckpt_root.display());

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<_> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch = build_batch::<ActiveBackend, _>(&tokenizer, &batch_examples, model_config.max_seq_len, &device)
            .expect("BPE tokenizer encoding should not fail on generator output");

        let step_start = Instant::now();
        let (next_model, loss) = train_step(model, &mut optim, &batch, train_cfg.learning_rate);
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;
        last_loss = loss;

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} ({step_ms:.1} ms/step)");
            append_train_log_row(&train_log_path, step, loss, step_ms);
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = grug_train::train::to_eval_model(&model);
            let (phrasing_row, whole_tool_row) = run_two_axis_eval_and_log(
                &eval_model,
                &tokenizer,
                &holdout,
                gen_config,
                &device,
                step,
                &eval_log_path,
            )
            .expect("eval log write should succeed");
            println!(
                "  eval@{step} [axis_b held_out_phrasing]: valid_json={:.3} schema={:.3} arg_correct={:.3} distinct_call_ratio={:.3} tool_entropy={:.3} (n={})",
                phrasing_row.valid_json_rate,
                phrasing_row.schema_adherence_rate,
                phrasing_row.arg_correct_rate,
                phrasing_row.distinct_call_ratio,
                phrasing_row.tool_name_entropy,
                phrasing_row.n,
            );
            println!(
                "  eval@{step} [axis_a whole_tool_holdout]: valid_json={:.3} schema={:.3} arg_correct={:.3} distinct_call_ratio={:.3} tool_entropy={:.3} (n={})",
                whole_tool_row.valid_json_rate,
                whole_tool_row.schema_adherence_rate,
                whole_tool_row.arg_correct_rate,
                whole_tool_row.distinct_call_ratio,
                whole_tool_row.tool_name_entropy,
                whole_tool_row.n,
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save should succeed");
            println!("  checkpoint saved -> {}", ckpt_dir.display());
        }
    }

    let total_elapsed = run_start.elapsed();
    println!(
        "train_p1a: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval JSONL: {}. checkpoints: {}",
        total_elapsed.as_secs_f64(),
        eval_log_path.display(),
        ckpt_root.display(),
    );
}

/// Build the training/holdout [`Dataset`]: the **procedural** catalog
/// (`Dataset::new_sized`) when `n_tools` is given (Phase 51 P1a-2, Wave 2),
/// or the original hand-authored ~56-tool catalog (`Dataset::new`) otherwise
/// -- keeps the pre-Wave-2 CLI invocation working unchanged.
fn build_dataset(seed: u64, n_tools: Option<usize>, holdout_frac: f64) -> grug_data::GrugResult<Dataset> {
    match n_tools {
        // 4 hard (same-family) distractors + target = 5 offered tools. Measured
        // via the PROMPT-LEN check: the hard() default of 6 distractors overflows
        // the 640-token ctx on ~4% of examples (max 785), which would silently
        // truncate targets in build_batch; 4 keeps max well under 640 while still
        // forcing in-context schema discrimination. (Phase 51 P1a-3, axis-A.)
        Some(n) => Dataset::new_sized_with_distractors(
            seed,
            n,
            holdout_frac,
            grug_data::DistractorConfig::hard_with_count(4),
        ),
        None => Dataset::new(seed),
    }
}

/// Load a previously-trained tokenizer from `path`, or train a fresh one over
/// `dataset`'s train stream (prompts + target JSON, `train_examples`-sized)
/// and save it to `path`.
fn load_or_train_tokenizer(dataset: &Dataset, path: &Path, train_examples: usize, vocab_size: usize) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json should load");
    }

    println!("train_p1a: training a fresh BPE tokenizer (vocab_size<={vocab_size})...");
    // The BPE corpus must cover the catalog's identifier + value surface, or
    // merges starve and the vocab ceiling is never reached (at 1000 procedural
    // tools a 5k sample under-covers tool/param names -> vocab capped ~2953 vs
    // an 8000 target, which fragments arg values and hurts copy-correctness).
    // Use (almost) the full training corpus; BPE training over it is still fast.
    let corpus_examples = dataset
        .train_examples(train_examples.min(60_000).max(200))
        .expect("generator produces train examples for tokenizer training corpus");
    let corpus: Vec<String> = corpus_examples
        .iter()
        .flat_map(|ex| {
            vec![
                grug_train::train::format_prompt(ex),
                ex.expected_call_as_json_string(),
            ]
        })
        .collect();

    let tokenizer = BpeTokenizer::train(corpus, vocab_size).expect("BPE training should succeed on a real corpus");
    tokenizer.save(path).expect("tokenizer save should succeed");
    tokenizer
}

/// Append one JSONL row of `{step, loss, step_ms}` to `path`, for a
/// lightweight training-loss time series alongside the eval axes file.
fn append_train_log_row(path: &Path, step: usize, loss: f32, step_ms: f64) {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let line = serde_json::json!({ "step": step, "loss": loss, "step_ms": step_ms }).to_string();
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}
