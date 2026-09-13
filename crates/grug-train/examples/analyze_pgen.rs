//! Phase 51 P1a-4 Step B2: instrument the pointer-generator's `p_gen` gate to
//! confirm (or refute) the gate-collapse hypothesis for Step B's null result.
//!
//! Loads a trained [`grug_train::PointerTransformer`] checkpoint and, on both
//! holdout axes, teacher-forces each example (the training framing) and reports
//! the mean `p_gen` at the **target-predicting positions** (where the model is
//! emitting the call), plus the fraction of those positions with `p_gen > 0.9`.
//!
//! `p_gen ≈ 1` means "always generate, never copy" — the copy pathway is
//! untrained/unused (gate collapse), which would explain why Step B gave no
//! zero-shot lift. `p_gen` well below 1 (especially on copyable tokens) would
//! refute the hypothesis and point elsewhere.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example analyze_pgen --features gpu --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p1a4-ptr \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p1a4-ptr/checkpoints/step_00019999 \
//!     --n-tools 1000 --holdout-frac 0.15 --samples 64
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig};
use grug_train::eval::FrozenTwoAxisHoldout;
use grug_train::tokenizer::Vocab;
use grug_train::train::{encode_example, init_training_ptr, to_eval_model_ptr};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{AttentionBackend, BpeTokenizer, PointerTransformerConfig, checkpoint};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "analyze_pgen")]
struct Args {
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
    #[arg(long, default_value_t = 2)]
    distractor_count: usize,
    /// Examples per axis to analyse.
    #[arg(long, default_value_t = 64)]
    samples: usize,
}

/// Mean p_gen and the fraction of target positions with p_gen > 0.9, over a
/// set of examples, computed by teacher-forcing each full sequence.
fn pgen_stats<B: Backend + AttentionBackend, V: Vocab>(
    model: &grug_train::PointerTransformer<B>,
    tokenizer: &V,
    examples: &[grug_data::schema::Example],
    device: &B::Device,
) -> (f64, f64, usize) {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    let mut high = 0usize;
    for ex in examples {
        let enc = match encode_example(tokenizer, ex) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let len = enc.ids.len();
        if enc.target_start == 0 || enc.target_start >= len {
            continue;
        }
        let ids: Vec<i64> = enc.ids.iter().map(|&t| t as i64).collect();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, len]), device);
        let parts = model.forward_parts(input);
        // p_gen: [1, len, 1] -> [len]
        let p_gen = parts.p_gen.reshape([len]);
        // Positions predicting a target token: i in [target_start-1, len-1).
        let start = enc.target_start - 1;
        #[allow(clippy::single_range_in_vec_init)]
        let vals: Vec<f32> = p_gen
            .slice([start..len - 1])
            .into_data()
            .to_vec()
            .unwrap_or_default();
        for v in vals {
            sum += v as f64;
            n += 1;
            if v > 0.9 {
                high += 1;
            }
        }
    }
    let mean = if n > 0 { sum / n as f64 } else { f64::NAN };
    let frac_high = if n > 0 {
        high as f64 / n as f64
    } else {
        f64::NAN
    };
    (mean, frac_high, n)
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("analyze_pgen: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("analyze_pgen: CPU backend");
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

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config =
        PointerTransformerConfig::new(TinyTransformerConfig::p1a(tokenizer.vocab_size()));
    let (model, optim) = init_training_ptr::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) =
        checkpoint::load_checkpoint_ptr(&args.resume, model, optim, &device)
            .expect("checkpoint loads");
    let model = to_eval_model_ptr(&model);
    println!("analyze_pgen: loaded checkpoint at step {step}");

    let holdout = FrozenTwoAxisHoldout::new(&dataset, args.samples).expect("holdout draws");

    let (b_mean, b_high, b_n) =
        pgen_stats(&model, &tokenizer, holdout.phrasing_examples(), &device);
    let (a_mean, a_high, a_n) =
        pgen_stats(&model, &tokenizer, holdout.whole_tool_examples(), &device);

    println!("\n=== p_gen at target positions (1.0 = always generate, 0.0 = always copy) ===");
    println!(
        "axis B (held_out_phrasing, in-dist):   mean p_gen = {b_mean:.3}   frac(p_gen>0.9) = {b_high:.3}   (n={b_n} positions)"
    );
    println!(
        "axis A (whole_tool_holdout, zero-shot): mean p_gen = {a_mean:.3}   frac(p_gen>0.9) = {a_high:.3}   (n={a_n} positions)"
    );
    println!(
        "\nInterpretation: mean p_gen near 1.0 with a high >0.9 fraction confirms gate collapse\n(the copy pathway is untrained/unused) — the leading explanation for Step B's null result."
    );
}
