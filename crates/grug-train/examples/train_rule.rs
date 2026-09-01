//! Phase 51 **P4** slice 2 — the 11M deductive-arm ablation harness (ADR
//! 092-v2, `docs/ai/plans/2026-07-11-phase51-p4-350m-foundation.md`).
//!
//! Trains a `TinyTransformer` on one of the two High-Grug-v2 deductive corpus
//! arms (`--logic prop|rule`) with the Q2 slot-classifier ablation
//! (`--typed-slots`), and evaluates on a **held-out** stream (fresh
//! individuals over the same rules — genuine in-context deduction, since the
//! rules live in the prompt) scored on the `select/bind/fire/chain`
//! decomposition (Q5). `--model` selects the capacity-ladder rung
//! (p1a≈13M / p1a_large≈30M / p1a_50m / p1a_100m).
//!
//! This is the gate-2 measurement: does an 11M organ learn rule application at
//! all, and does the `chain` component's capacity slope go positive
//! (11M→50M→100M) — "scale it" vs "broken corpus".
//!
//! ```text
//! # CPU smoke:
//! cargo run -p grug-train --example train_rule -- \
//!     --logic rule --typed-slots true --train-examples 64 --epochs 1 \
//!     --batch-size 4 --eval-every 10 --ckpt-every 20 --artifacts-dir /tmp/p4-smoke
//!
//! # Real GPU run (source scripts/train-env.sh first):
//! cargo run -p grug-train --example train_rule --features gpu --release -- \
//!     --logic rule --typed-slots true --seed 42 --train-examples 40000 --epochs 8 \
//!     --batch-size 16 --eval-every 500 --ckpt-every 2000 \
//!     --artifacts-dir runs/p4-rule-typed
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use grug_data::{
    RuleCorpusConfig, RuleDataset, RuleEvalSummary, RuleExample, RuleWorldConfig, TraceArm,
    score_rule_outputs,
};
use grug_train::generate::GenerateConfig;
use grug_train::rule_train::{
    build_rule_batch, encode_rule_example, format_rule_prompt, generate_rule_outputs,
};
use grug_train::train::{init_training, to_eval_model, train_step_guarded};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{BpeTokenizer, checkpoint};

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_rule")]
struct Args {
    /// Which corpus arm: `prop` (boolean formula over ground atoms) or `rule`
    /// (modus-ponens chain over in-prompt rules).
    #[arg(long, default_value = "rule")]
    logic: String,

    /// Q2 slot-classifier ablation: typed (`x1·ANIMAL`) vs bare (`x1`). Takes
    /// an explicit value (`--typed-slots false` for the bare arm).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    typed_slots: bool,

    /// Training-stream target format: `trace` (full High-Grug-v2 trace,
    /// intermediate supervision) or `answer` (answer-only control). Eval is
    /// always replay-scored against whatever the model emits.
    #[arg(long, default_value = "trace")]
    arm: String,

    /// Model capacity rung: p1a | p1a_large | p1a_50m | p1a_100m.
    #[arg(long, default_value = "p1a")]
    model: String,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    // --- world / corpus knobs ---
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
    /// Inclusive prop-formula literal-count range "lo,hi".
    #[arg(long, default_value = "2,4")]
    prop_literals: String,

    /// Rule-arm training depths, e.g. `2,3,4`.
    #[arg(long, default_value = "2,3,4")]
    train_depths: String,
    /// Rule-arm length-generalization eval depths (unseen), e.g. `5,6`.
    #[arg(long, default_value = "5,6")]
    eval_depths: String,

    #[arg(long, default_value_t = 40_000)]
    train_examples: usize,
    #[arg(long, default_value_t = 8)]
    epochs: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    learning_rate: f64,
    /// Linear warmup steps (0 = no warmup, i.e. constant/cosine from step 0).
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Cosine decay floor as a fraction of the peak LR (post-warmup). 1.0
    /// disables decay entirely (LR stays at peak) — combined with
    /// `--warmup-steps 0` this reproduces the old constant-LR behavior exactly.
    #[arg(long, default_value_t = 0.1)]
    lr_min_ratio: f64,
    #[arg(long, default_value_t = 3_000)]
    bpe_vocab_size: usize,
    #[arg(long, default_value_t = 256)]
    holdout_examples: usize,
    #[arg(long, default_value_t = 500)]
    eval_every: usize,
    #[arg(long, default_value_t = 2_000)]
    ckpt_every: usize,
    /// Max greedy tokens per eval example (a depth-8 rule trace is the longest).
    #[arg(long, default_value_t = 220)]
    eval_max_new_tokens: usize,

    /// Override the model's context window (position-embedding table size). 0 =
    /// use the preset's default. The saturated BPE (this templated corpus has
    /// few unique substrings) leaves depth-6 sequences ~800 tokens, over p1a's
    /// 640 default — set 1024 to fit them.
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,

    #[arg(long)]
    artifacts_dir: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq)]
enum Logic {
    Prop,
    Rule,
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

/// Warmup + cosine-decay LR schedule (ADR: "~1e-4 + warmup + cosine" lesson
/// from the 50M LR-instability run, see `training/README.md#lr-at-scale`).
///
/// `warmup_steps == 0 && lr_min_ratio == 1.0` reproduces the old constant-LR
/// behavior exactly: outside warmup (vacuous when `warmup_steps == 0`) the
/// decay floor equals the peak, so `peak_lr - min_lr == 0.0` exactly and the
/// cosine term is multiplied away regardless of its value.
fn lr_at_step(
    step: usize,
    total_steps: usize,
    peak_lr: f64,
    warmup_steps: usize,
    lr_min_ratio: f64,
) -> f64 {
    if warmup_steps > 0 && step < warmup_steps {
        // Linear 0 -> peak, using (step+1)/warmup_steps so the very first
        // step isn't a wasted LR=0 step.
        return peak_lr * (step + 1) as f64 / warmup_steps as f64;
    }
    let min_lr = peak_lr * lr_min_ratio;
    let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
    let progress = ((step - warmup_steps) as f64 / decay_steps as f64).min(1.0);
    min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn model_config(name: &str, vocab: usize) -> TinyTransformerConfig {
    match name {
        "spike" => TinyTransformerConfig::spike(vocab),
        "p1a" => TinyTransformerConfig::p1a(vocab),
        "p1a_large" => TinyTransformerConfig::p1a_large(vocab),
        "p1a_50m" => TinyTransformerConfig::p1a_50m(vocab),
        "p1a_100m" => TinyTransformerConfig::p1a_100m(vocab),
        "p1a_350m" => TinyTransformerConfig::p1a_350m(vocab),
        other => panic!("--model must be spike|p1a|p1a_large|p1a_50m|p1a_100m|p1a_350m, got {other:?}"),
    }
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.artifacts_dir).expect("can create artifacts dir");

    let logic = match args.logic.as_str() {
        "prop" | "propositional" => Logic::Prop,
        "rule" | "rule_based" => Logic::Rule,
        other => panic!("--logic must be prop|rule, got {other:?}"),
    };
    let arm = TraceArm::parse_label(&args.arm).expect("--arm must be trace|answer");
    let train_depths = parse_depths(&args.train_depths, "--train-depths");
    let eval_depths = parse_depths(&args.eval_depths, "--eval-depths");
    let (plo, phi) = {
        let v = parse_depths(&args.prop_literals, "--prop-literals");
        assert_eq!(v.len(), 2, "--prop-literals must be lo,hi");
        (v[0], v[1])
    };

    // Vendor-agnostic GPU device (ROCm under `--features gpu`, CUDA under
    // `--features cuda`) so this example builds unchanged on the local 7900 XTX
    // and on an H100 cloud box.
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let device = {
        let d = grug_train::gpu_device();
        println!("train_rule: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let device = {
        println!("train_rule: CPU backend");
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
        prop_literals: (plo, phi),
    })
    .expect("RuleCorpusConfig builds a valid dataset");

    println!(
        "train_rule: LOGIC={} ARM={arm:?} TYPED={} MODEL={} seed={}",
        args.logic, args.typed_slots, args.model, args.seed
    );

    // --- training stream ---
    let train_examples: Vec<RuleExample> = match logic {
        Logic::Prop => dataset.propositional_examples(args.train_examples, arm, args.typed_slots),
        Logic::Rule => {
            dataset.rule_examples(args.train_examples, &train_depths, arm, args.typed_slots)
        }
    };
    let n_yes = train_examples
        .iter()
        .filter(|e| e.gold_answer == grug_data::GrugTruth::Yes)
        .count();
    println!(
        "train_rule: training stream: {} examples ({} yes / {} no)",
        train_examples.len(),
        n_yes,
        train_examples.len() - n_yes,
    );

    // --- frozen holdouts (always full traces, replay-scored) ---
    let in_dist_holdout: Vec<RuleExample> = match logic {
        Logic::Prop => dataset.propositional_examples_eval(
            args.holdout_examples,
            TraceArm::Trace,
            args.typed_slots,
        ),
        Logic::Rule => dataset.rule_examples_eval(
            args.holdout_examples,
            &train_depths,
            TraceArm::Trace,
            args.typed_slots,
        ),
    };
    // Length-generalization axis exists only for the rule arm.
    let len_gen_holdout: Vec<RuleExample> = match logic {
        Logic::Rule => dataset.rule_examples_eval(
            args.holdout_examples,
            &eval_depths,
            TraceArm::Trace,
            args.typed_slots,
        ),
        Logic::Prop => Vec::new(),
    };

    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = load_or_train_tokenizer(&train_examples, &tokenizer_path, args.bpe_vocab_size);
    println!(
        "train_rule: BPE vocab_size={} ({})",
        tokenizer.vocab_size(),
        if tokenizer_path.exists() {
            "loaded"
        } else {
            "trained"
        }
    );

    let mut cfg = model_config(&args.model, tokenizer.vocab_size());
    if args.max_seq_len > 0 {
        cfg.max_seq_len = args.max_seq_len;
    }
    let max_seq_len = cfg.max_seq_len;

    // PROMPT-LEN sanity: longest prompt+target vs ctx (a depth-8 rule example
    // with distractors is the longest sequence this corpus produces).
    {
        let probe: Vec<RuleExample> = match logic {
            Logic::Prop => {
                dataset.propositional_examples_eval(128, TraceArm::Trace, args.typed_slots)
            }
            Logic::Rule => {
                let mut v = dataset.rule_examples_eval(
                    96,
                    &train_depths,
                    TraceArm::Trace,
                    args.typed_slots,
                );
                v.extend(dataset.rule_examples_eval(
                    64,
                    &eval_depths,
                    TraceArm::Trace,
                    args.typed_slots,
                ));
                v
            }
        };
        let mut lens: Vec<usize> = probe
            .iter()
            .filter_map(|ex| {
                encode_rule_example(&tokenizer, ex)
                    .ok()
                    .map(|e| e.ids.len())
            })
            .collect();
        lens.sort_unstable();
        let max = lens.last().copied().unwrap_or(0);
        let over = lens.iter().filter(|&&l| l > max_seq_len).count();
        println!(
            "train_rule: PROMPT-LEN check (n={}) max={max} ctx={max_seq_len} over_ctx={over}",
            lens.len()
        );
    }

    let (mut model, mut optim) = init_training::<ActiveBackend>(&cfg, &device);
    println!(
        "train_rule: TinyTransformer ~{} params",
        burn::module::Module::num_params(&model)
    );

    let mut start_step = 0usize;
    if let Some(resume_dir) = &args.resume {
        let (m, o, step) = checkpoint::load_checkpoint(resume_dir, model, optim, &device)
            .expect("checkpoint loads");
        model = m;
        optim = o;
        start_step = step + 1;
        println!("train_rule: resumed at step {start_step}");
    }

    let gen_config = GenerateConfig {
        max_new_tokens: args.eval_max_new_tokens,
        max_seq_len,
    };
    let eval_log_path = args.artifacts_dir.join("eval_rule.jsonl");
    let ckpt_root = args.artifacts_dir.join("checkpoints");
    let total_steps = args.epochs * (args.train_examples / args.batch_size).max(1);
    println!(
        "train_rule: {total_steps} steps ({} epochs), batch={}, lr={} (warmup={}, lr_min_ratio={}), eval -> {}",
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.warmup_steps,
        args.lr_min_ratio,
        eval_log_path.display()
    );

    let run_start = Instant::now();
    let mut last_loss = f32::NAN;
    let mut skipped_steps = 0usize;
    let mut loss_ema = f32::NAN;

    for step in start_step..total_steps {
        let batch_examples: Vec<RuleExample> = (0..args.batch_size)
            .map(|i| train_examples[(step * args.batch_size + i) % train_examples.len()].clone())
            .collect();
        let batch =
            build_rule_batch::<ActiveBackend, _>(&tokenizer, &batch_examples, max_seq_len, &device)
                .expect("encoding should not fail on generator output");

        let current_lr = lr_at_step(
            step,
            total_steps,
            args.learning_rate,
            args.warmup_steps,
            args.lr_min_ratio,
        );

        // Spike guard: once past the settling-in phase, refuse to step on a
        // batch whose loss is far above the running EMA — that is the exact
        // precursor of the run-1/run-2 blowup (converged ~0.5 loss, one spike
        // to 7.4, inf grads, NaN weights through the norm-clip).
        let loss_cap = if step > 100 && loss_ema.is_finite() {
            (loss_ema * 4.0).max(loss_ema + 2.0)
        } else {
            f32::INFINITY
        };

        let step_start = Instant::now();
        let (next_model, loss) = train_step_guarded(model, &mut optim, &batch, current_lr, loss_cap);
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        model = next_model;

        if !loss.is_finite() || loss > loss_cap {
            // Optimizer update was skipped; the model is untouched. A few
            // dozen hard/poisoned batches per epoch are survivable; a flood
            // means something structural — abort so the run fails loudly.
            skipped_steps += 1;
            eprintln!(
                "train_rule: loss {loss:.4} vs cap {loss_cap:.4} at step {step} -- batch skipped ({skipped_steps} total)."
            );
            if skipped_steps > 500 {
                eprintln!("train_rule: {skipped_steps} skipped batches -- aborting.");
                break;
            }
        } else {
            last_loss = loss;
            loss_ema = if loss_ema.is_finite() { 0.98 * loss_ema + 0.02 * loss } else { loss };
        }

        if step % 10 == 0 || step == total_steps - 1 {
            println!("step {step:6}: loss = {loss:.4} lr = {current_lr:.6} ({step_ms:.1} ms/step)");
        }

        if step % args.eval_every == 0 || step == total_steps - 1 {
            let eval_model = to_eval_model(&model);
            let in_out = generate_rule_outputs(
                &eval_model,
                &tokenizer,
                &in_dist_holdout,
                gen_config,
                &device,
            );
            let in_summary = score_rule_outputs(&in_dist_holdout, &in_out);
            print_summary("in-dist ", step, &in_summary);

            let len_summary = if len_gen_holdout.is_empty() {
                None
            } else {
                let lg_out = generate_rule_outputs(
                    &eval_model,
                    &tokenizer,
                    &len_gen_holdout,
                    gen_config,
                    &device,
                );
                let s = score_rule_outputs(&len_gen_holdout, &lg_out);
                print_summary("len-gen ", step, &s);
                Some(s)
            };

            log_eval_row(
                &eval_log_path,
                step,
                &args,
                logic,
                &in_summary,
                len_summary.as_ref(),
            );
        }

        if (step > 0 && step % args.ckpt_every == 0) || step == total_steps - 1 {
            let ckpt_dir = ckpt_root.join(format!("step_{step:08}"));
            checkpoint::save_checkpoint(&ckpt_dir, step, &model, &optim).expect("checkpoint save");
            println!("  checkpoint -> {}", ckpt_dir.display());
        }
    }

    println!(
        "train_rule: done. {total_steps} steps in {:.2}s. final loss={last_loss:.4}. eval: {}",
        run_start.elapsed().as_secs_f64(),
        eval_log_path.display()
    );
}

fn print_summary(tag: &str, step: usize, s: &RuleEvalSummary) {
    let c = &s.components;
    println!(
        "  eval@{step} [{tag}] ans={:.3} exec={:.3} valid={:.3} | select={:.3} bind={:.3} fire={:.3} chain={:.3} (n={}, fires={})",
        s.answer_accuracy,
        s.executability_rate,
        s.trace_valid_rate,
        c.select_rate,
        c.bind_rate,
        c.fire_rate,
        c.chain_rate,
        s.n,
        c.fire_total,
    );
}

fn load_or_train_tokenizer(
    train_examples: &[RuleExample],
    path: &Path,
    vocab_size: usize,
) -> BpeTokenizer {
    if path.exists() {
        return BpeTokenizer::load(path).expect("existing tokenizer.json loads");
    }
    let corpus: Vec<String> = train_examples
        .iter()
        .take(60_000)
        .flat_map(|ex| vec![format_rule_prompt(ex).to_string(), ex.target.clone()])
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, vocab_size).expect("BPE training");
    tokenizer.save(path).expect("tokenizer save");
    tokenizer
}

#[allow(clippy::too_many_arguments)]
fn log_eval_row(
    path: &Path,
    step: usize,
    args: &Args,
    logic: Logic,
    in_dist: &RuleEvalSummary,
    len_gen: Option<&RuleEvalSummary>,
) {
    use std::io::Write as _;
    let comp = |s: &RuleEvalSummary| {
        serde_json::json!({
            "n": s.n,
            "answer_accuracy": s.answer_accuracy,
            "executability_rate": s.executability_rate,
            "trace_valid_rate": s.trace_valid_rate,
            "select_rate": s.components.select_rate,
            "bind_rate": s.components.bind_rate,
            "fire_rate": s.components.fire_rate,
            "chain_rate": s.components.chain_rate,
            "fire_total": s.components.fire_total,
            "chain_expected_total": s.components.chain_expected_total,
        })
    };
    let row = serde_json::json!({
        "step": step,
        "logic": if logic == Logic::Prop { "prop" } else { "rule" },
        "typed_slots": args.typed_slots,
        "arm": args.arm,
        "model": args.model,
        "in_dist": comp(in_dist),
        "len_gen": len_gen.map(comp),
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
