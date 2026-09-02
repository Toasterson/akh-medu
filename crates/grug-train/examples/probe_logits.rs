//! Throwaway forensic tool for the Phase 51 P4 350M loss/gradient "storm"
//! investigation: test the **output-logit divergence** hypothesis (Wortsman
//! et al. 2309.14322 §3.2 / the PaLM `z-loss` motivation) directly against a
//! real checkpoint -- do the epoch-periodic trigger batches drive output
//! logits to fp32-dangerous magnitudes (the precursor `masked_cross_entropy`'s
//! `log_softmax` would need to survive), while matched control batches stay
//! tame?
//!
//! Rebuilds the exact deterministic 40000-example rule-corpus stream + BPE
//! tokenizer in-process, identically to `train_rule.rs` / `forensic_batches.rs`
//! (seed 42, preds_per_cell 6, max_level 6, guard_prob 0.5, distractor_facts
//! 4, distractor_rules 6, yes_frac 0.5, train_depths [2,3,4], 40000 examples,
//! BPE vocab 3000), then loads a `--resume` checkpoint (P4's `p1a_350m`
//! config) via `checkpoint::load_checkpoint` (which requires the
//! `AutodiffBackend`-wrapped model/optimizer pair `train_rule.rs`'s resume
//! path uses) and immediately projects to the **inference-mode** model
//! (`train::to_eval_model` / `Module::valid()`, a plain non-Autodiff
//! backend) before running any forward passes.
//!
//! That projection matters, not just style: an Autodiff-wrapped forward pass
//! retains its full activation graph for a possible future `.backward()`
//! even when none is ever called, and across a 350M-param/24-layer model at
//! up to 1024 tokens that graph is tens of GB per batch -- the first version
//! of this probe left the model Autodiff-wrapped for the forward passes and
//! OOM'd an 80GB H100 on the *second* batch. The eval-mode model has no
//! graph to retain, matching `probe_lens.rs`'s inference-only pattern.
//!
//! Runs a plain forward pass (no backward -- this never calls `.backward()`)
//! over the TRIGGER and CONTROL batches, reporting the raw logit-magnitude
//! statistics per batch.
//!
//! CPU-capable (builds and runs with default features against a small/no-op
//! checkpoint), but the real 350M checkpoint only exists on the training pod
//! -- run there with `--features cuda --release`.
//!
//! ```text
//! # CPU compile check only (no real checkpoint available locally):
//! cargo check -p grug-train --example probe_logits
//!
//! # Real run, on the pod:
//! cargo run -p grug-train --example probe_logits --features cuda --release -- \
//!     --resume runs/p350m/checkpoints/step_00028000
//! ```

use std::path::PathBuf;

use clap::Parser;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::{RuleCorpusConfig, RuleDataset, RuleExample, RuleWorldConfig, TraceArm};
use grug_train::checkpoint;
use grug_train::rule_train::{encode_rule_example, format_rule_prompt};
use grug_train::tokenizer::Vocab;
use grug_train::train::{init_training, to_eval_model};
use grug_train::transformer::{TinyTransformer, TinyTransformerConfig};
use grug_train::BpeTokenizer;

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

/// Same corpus/stream constants as `forensic_batches.rs` / `train_rule.rs`'s
/// defaults -- must match exactly for step-indexed batches to line up with
/// the real training run's stream.
const SEED: u64 = 42;
const PREDS_PER_CELL: u32 = 6;
const MAX_LEVEL: u32 = 6;
const GUARD_PROB: f64 = 0.5;
const DISTRACTOR_FACTS: usize = 4;
const DISTRACTOR_RULES: usize = 6;
const YES_FRAC: f64 = 0.5;
const TRAIN_EXAMPLES: usize = 40_000;
const BPE_VOCAB_SIZE: usize = 3_000;
const BATCH_SIZE: usize = 16;

/// Epoch-periodic non-finite-gradient trigger steps from the run-log forensic
/// windows (same steps `forensic_batches.rs`'s trigger ranges are centered
/// on).
const TRIGGER_STEPS: &[usize] = &[874, 875, 880, 1220, 1225, 2084, 2090];
/// Matched control steps -- ordinary, apparently-healthy training steps at
/// comparable and surrounding positions in the same epoch.
const CONTROL_STEPS: &[usize] = &[500, 1000, 1500, 1800, 2200];

/// fp32 `exp()` overflows above this input magnitude (`f32::MAX.ln() ≈
/// 88.72`) -- `log_softmax`'s internal `exp(x - max)` is numerically stable
/// against the row's own max, but a per-row spread anywhere near this or a
/// raw `max|logit|` in the tens-of-thousands-plus is the fp32 danger zone
/// this probe is checking for.
const FP32_EXP_OVERFLOW_THRESHOLD: f32 = 88.0;

#[derive(Parser, Debug)]
#[command(name = "probe_logits")]
struct Args {
    /// Checkpoint directory to load (e.g. `runs/p350m/checkpoints/step_00028000`).
    #[arg(long)]
    resume: PathBuf,
    /// Model capacity rung -- defaults to the P4 350M target this
    /// investigation is about.
    #[arg(long, default_value = "p1a_350m")]
    model: String,
    /// Override the model's context window. 0 = use the preset's default
    /// (1024 for p1a_350m).
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,
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

/// Per-batch logit-magnitude statistics from one forward pass.
struct BatchStats {
    step: usize,
    group: &'static str,
    batch_size: usize,
    seq_len: usize,
    /// max(|logit|) over every position in the batch (all positions, not
    /// just loss-masked target positions -- the key headline number).
    max_abs_logit: f32,
    /// mean, over every (example, position) in the batch, of that position's
    /// own max (raw, signed) logit across the vocab.
    mean_of_per_pos_max: f64,
    /// max, over positions, of (row max - row min) -- the spread
    /// `log_softmax`'s internal `exp(x - max)` has to cover for that row.
    max_logsumexp_spread: f32,
    /// max, over positions, of (top1 logit - runner-up logit).
    max_margin: f32,
    nan_count: usize,
    inf_count: usize,
}

/// [`grug_train::rule_train::build_rule_batch`]'s truncate/pad logic,
/// reimplemented for a plain (non-Autodiff) `Backend` -- `Batch<B>` there is
/// generic over `B: AutodiffBackend` by construction (it exists to feed
/// `.backward()`), which the inference-mode eval model this probe uses
/// deliberately is not. Only `input_ids` is needed here (no loss mask, no
/// copy mask), so this only reproduces that slice of the logic.
fn build_probe_input<B: Backend, V: Vocab>(
    tokenizer: &V,
    examples: &[RuleExample],
    max_seq_len: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, usize, usize) {
    let pad_id = tokenizer.pad_id();
    let encoded: Vec<_> = examples
        .iter()
        .map(|ex| {
            encode_rule_example(tokenizer, ex).expect("encoding should not fail on generator output")
        })
        .collect();

    let longest = encoded.iter().map(|e| e.ids.len()).max().unwrap_or(1).min(max_seq_len);
    let batch_size = encoded.len();
    let seq_len = longest.saturating_sub(1).max(1);

    let mut input_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    for enc in &encoded {
        let mut ids = enc.ids.clone();
        ids.truncate(longest);
        while ids.len() < longest {
            ids.push(pad_id);
        }
        for i in 0..seq_len {
            input_flat.push(ids[i] as i64);
        }
    }

    let input_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [batch_size, seq_len]), device);
    (input_ids, batch_size, seq_len)
}

fn probe_batch<B: Backend>(
    model: &TinyTransformer<B>,
    tokenizer: &BpeTokenizer,
    train_examples: &[RuleExample],
    step: usize,
    group: &'static str,
    max_seq_len: usize,
    device: &B::Device,
) -> BatchStats {
    let batch_examples: Vec<RuleExample> = (0..BATCH_SIZE)
        .map(|i| train_examples[(step * BATCH_SIZE + i) % train_examples.len()].clone())
        .collect();
    let (input_ids, batch_size, seq_len) =
        build_probe_input::<B, _>(tokenizer, &batch_examples, max_seq_len, device);

    // Forward only, on the plain (non-Autodiff) inference-mode model -- no
    // computation graph is ever built, so nothing needs freeing between
    // batches; we just want the raw logits this batch would have produced
    // feeding into `masked_cross_entropy`'s `log_softmax`.
    let logits: Tensor<B, 3> = model.forward(input_ids); // [batch, seq, vocab]

    let vocab = tokenizer.vocab_size();
    let flat: Vec<f32> = logits
        .into_data()
        .to_vec::<f32>()
        .expect("logits tensor is f32");

    let n_positions = batch_size * seq_len;
    debug_assert_eq!(flat.len(), n_positions * vocab);

    // NB on NaN handling: IEEE-754 comparisons involving NaN are always
    // `false`, so plain `if x > running_max { running_max = x }` updates
    // naturally skip NaN values/derived quantities without any special-casing
    // -- no need to hand-roll NaN-aware max logic here.
    let mut max_abs_logit = f32::NEG_INFINITY;
    let mut per_pos_max_sum = 0f64;
    let mut n_pos_scored = 0usize;
    let mut max_spread = f32::NEG_INFINITY;
    let mut max_margin = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for pos in 0..n_positions {
        let row = &flat[pos * vocab..(pos + 1) * vocab];
        let mut row_max = f32::NEG_INFINITY;
        let mut row_second = f32::NEG_INFINITY;
        let mut row_min = f32::INFINITY;
        let mut row_has_nan = false;

        for &v in row {
            if v.is_nan() {
                nan_count += 1;
                row_has_nan = true;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
            }
            let av = v.abs();
            if av > max_abs_logit {
                max_abs_logit = av;
            }
            if v > row_max {
                row_second = row_max;
                row_max = v;
            } else if v > row_second {
                row_second = v;
            }
            if v < row_min {
                row_min = v;
            }
        }

        if row_has_nan {
            continue;
        }
        per_pos_max_sum += row_max as f64;
        n_pos_scored += 1;
        let spread = row_max - row_min;
        if spread > max_spread {
            max_spread = spread;
        }
        let margin = row_max - row_second;
        if margin > max_margin {
            max_margin = margin;
        }
    }

    BatchStats {
        step,
        group,
        batch_size,
        seq_len,
        max_abs_logit,
        mean_of_per_pos_max: if n_pos_scored > 0 {
            per_pos_max_sum / n_pos_scored as f64
        } else {
            f64::NAN
        },
        max_logsumexp_spread: max_spread,
        max_margin,
        nan_count,
        inf_count,
    }
}

fn notes_for(s: &BatchStats) -> String {
    let mut notes = Vec::new();
    if s.nan_count > 0 {
        notes.push(format!("NaN x{}", s.nan_count));
    }
    if s.inf_count > 0 {
        notes.push(format!("Inf x{}", s.inf_count));
    }
    if s.max_abs_logit > 1.0e4 {
        notes.push("EXTREME (>1e4)".to_string());
    } else if s.max_abs_logit > FP32_EXP_OVERFLOW_THRESHOLD {
        notes.push(format!("> fp32 exp-overflow ({FP32_EXP_OVERFLOW_THRESHOLD:.0})"));
    }
    if notes.is_empty() {
        "ok".to_string()
    } else {
        notes.join(", ")
    }
}

fn main() {
    let args = Args::parse();

    // --- vendor-agnostic GPU device seam (ROCm under --features gpu, CUDA
    // under --features cuda), CPU otherwise -- identical cfg pattern to
    // train_rule.rs so this one source builds unmodified on either box. ---
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let device = {
        let d = grug_train::gpu_device();
        println!("probe_logits: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let device = {
        println!("probe_logits: CPU backend");
        Default::default()
    };

    // --- 1. rebuild the exact deterministic 40000-example rule-corpus
    // stream, identically to train_rule.rs / forensic_batches.rs. ---
    let dataset = RuleDataset::new(RuleCorpusConfig {
        world: RuleWorldConfig {
            seed: SEED,
            preds_per_cell: PREDS_PER_CELL,
            max_level: MAX_LEVEL,
            guard_prob: GUARD_PROB,
        },
        distractor_facts: DISTRACTOR_FACTS,
        distractor_rules: DISTRACTOR_RULES,
        yes_frac: YES_FRAC,
        prop_literals: (2, 4), // unused by the rule arm, train_rule.rs default
    })
    .expect("RuleCorpusConfig builds a valid dataset");

    let train_depths = vec![2usize, 3, 4];
    let arm = TraceArm::Trace; // --arm trace (train_rule.rs default)
    let typed_slots = true;

    println!(
        "probe_logits: generating {TRAIN_EXAMPLES} rule examples (seed={SEED}, depths={train_depths:?}, arm=trace, typed_slots=true)..."
    );
    let train_examples: Vec<RuleExample> =
        dataset.rule_examples(TRAIN_EXAMPLES, &train_depths, arm, typed_slots);
    assert_eq!(train_examples.len(), TRAIN_EXAMPLES);

    // --- 2. train the BPE in-process, identically to
    // train_rule.rs::load_or_train_tokenizer / forensic_batches.rs. ---
    println!("probe_logits: training BPE tokenizer (vocab_size={BPE_VOCAB_SIZE}) from the corpus...");
    let corpus: Vec<String> = train_examples
        .iter()
        .take(60_000)
        .flat_map(|ex| vec![format_rule_prompt(ex).to_string(), ex.target.clone()])
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, BPE_VOCAB_SIZE).expect("BPE training");
    println!(
        "probe_logits: BPE vocab_size={} (requested {BPE_VOCAB_SIZE})",
        tokenizer.vocab_size()
    );

    // --- 3. load the checkpoint (model config = p1a_350m by default). ---
    let mut cfg = model_config(&args.model, tokenizer.vocab_size());
    if args.max_seq_len > 0 {
        cfg.max_seq_len = args.max_seq_len;
    }
    let max_seq_len = cfg.max_seq_len;

    let (model, optim) = init_training::<ActiveBackend>(&cfg, &device);
    println!(
        "probe_logits: {} model, ~{} params, max_seq_len={max_seq_len}",
        args.model,
        Module::num_params(&model)
    );

    let (model, _optim, loaded_step) =
        checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint loads");
    println!(
        "probe_logits: loaded checkpoint {} (step {loaded_step})",
        args.resume.display()
    );

    // Project to the plain inference-mode model *before* any forward pass --
    // see the module doc comment: leaving the model Autodiff-wrapped for
    // forward-only probing retains a full (unused) backward graph per batch
    // and OOMs an 80GB GPU within a couple of batches.
    let eval_model = to_eval_model(&model);
    drop(model);

    // --- 4. probe TRIGGER + CONTROL batches. ---
    let mut steps: Vec<(usize, &'static str)> = TRIGGER_STEPS
        .iter()
        .map(|&s| (s, "TRIGGER"))
        .chain(CONTROL_STEPS.iter().map(|&s| (s, "CONTROL")))
        .collect();
    steps.sort_by_key(|(s, _)| *s);

    let mut results: Vec<BatchStats> = Vec::with_capacity(steps.len());
    for (step, group) in steps {
        println!("probe_logits: forward pass, step {step} ({group})...");
        let stats = probe_batch(
            &eval_model,
            &tokenizer,
            &train_examples,
            step,
            group,
            max_seq_len,
            &device,
        );
        println!(
            "  step {step:6} [{group:7}] n={:>3}x{:<4} max|logit|={:>14.3} mean_per_pos_max={:>12.3} logsumexp_spread={:>12.3} margin={:>10.3} nan={} inf={}",
            stats.batch_size,
            stats.seq_len,
            stats.max_abs_logit,
            stats.mean_of_per_pos_max,
            stats.max_logsumexp_spread,
            stats.max_margin,
            stats.nan_count,
            stats.inf_count,
        );
        results.push(stats);
    }

    // --- 5. compact comparison table. ---
    println!();
    println!("=== probe_logits: output-logit divergence probe (Wortsman 2309.14322 / z-loss hypothesis) ===");
    println!("checkpoint: {} (loaded step {loaded_step})", args.resume.display());
    println!();
    println!(
        "{:>6} | {:<8} | {:>16} | {:>18} | {:<28}",
        "step", "trig/ctl", "max|logit|", "mean-of-per-pos-max", "notes"
    );
    println!("{}", "-".repeat(90));
    for s in &results {
        println!(
            "{:>6} | {:<8} | {:>16.3} | {:>18.3} | {:<28}",
            s.step,
            s.group,
            s.max_abs_logit,
            s.mean_of_per_pos_max,
            notes_for(s)
        );
    }
    println!();

    // --- 6. verdict. ---
    let trigger_max = results
        .iter()
        .filter(|s| s.group == "TRIGGER")
        .map(|s| s.max_abs_logit)
        .fold(f32::MIN, f32::max);
    let control_max = results
        .iter()
        .filter(|s| s.group == "CONTROL")
        .map(|s| s.max_abs_logit)
        .fold(f32::MIN, f32::max);
    let trigger_nonfinite: usize = results
        .iter()
        .filter(|s| s.group == "TRIGGER")
        .map(|s| s.nan_count + s.inf_count)
        .sum();
    let control_nonfinite: usize = results
        .iter()
        .filter(|s| s.group == "CONTROL")
        .map(|s| s.nan_count + s.inf_count)
        .sum();
    println!(
        "VERDICT-INPUTS: trigger_max_abs_logit={trigger_max:.3} control_max_abs_logit={control_max:.3} \
         trigger_nonfinite_count={trigger_nonfinite} control_nonfinite_count={control_nonfinite} \
         fp32_exp_overflow_threshold={FP32_EXP_OVERFLOW_THRESHOLD:.0}"
    );
}
