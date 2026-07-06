//! Phase 51 Spike-0 runnable example: trains [`grug_train::TinyTransformer`]
//! from scratch on `grug-data`'s programmatic JSON tool-calling generator,
//! printing the loss curve and periodic [`grug_data::EvalReport`] /
//! [`grug_data::DiversityReport`] rows, and appending the same rows to a
//! JSONL eval time-series file.
//!
//! This is deliberately *not* a `#[test]` -- it is the "real training run"
//! the plan calls for, kept out of the fast test suite. Run it with:
//!
//! ```text
//! cargo run -p grug-train --example spike0
//! ```
//!
//! Everything here uses [`TinyTransformerConfig::spike`] (the plan's
//! "~2-10M transformer" P0 target) and trains on CPU (`CpuAutodiffBackend`
//! = `Autodiff<NdArray<f32>>`), per ADR 087 decision 2/3 -- CubeCL/ROCm is a
//! later swap of the backend type alias, not a change to this example.

use std::time::Instant;

use grug_data::Dataset;
use grug_train::eval::{default_log_path, run_eval_and_log, FrozenHoldout};
use grug_train::generate::GenerateConfig;
use grug_train::tokenizer::ByteTokenizer;
use grug_train::train::{build_batch, init_training, train_step, TrainConfig};
use grug_train::transformer::TinyTransformerConfig;

// Phase 51 P0 gate (R3): the training backend is a pure type-alias swap.
// Default (no `gpu` feature) trains on CPU (`burn-ndarray`); `--features gpu`
// swaps in Burn's ROCm/HIP (CubeCL) backend targeting this box's 7900 XTX
// (gfx1100) -- no other line in this file changes. See
// `docs/ai/decisions/087-grug-brain-baby-llm-training.md` and
// `GpuAutodiffBackend` / `CpuAutodiffBackend` in `src/lib.rs`.
#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

const TRAIN_EXAMPLES: usize = 128;
const HOLDOUT_EXAMPLES: usize = 16;
const TOTAL_STEPS: usize = 300;
const BATCH_SIZE: usize = 8;
const EVAL_EVERY: usize = 25;
const SEED: u64 = 42;

fn main() {
    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("Spike-0: GPU feature enabled -- training on Burn ROCm/HIP (CubeCL) backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("Spike-0: CPU backend (burn-ndarray)");
        Default::default()
    };
    let tokenizer = ByteTokenizer::new();
    let train_cfg = TrainConfig::default();

    let dataset = Dataset::new(SEED).expect("built-in catalog splits cleanly");
    let train_examples = dataset
        .train_examples(TRAIN_EXAMPLES)
        .expect("generator produces train examples");
    let holdout = FrozenHoldout::new(&dataset, HOLDOUT_EXAMPLES).expect("generator produces holdout examples");

    let model_config = TinyTransformerConfig::spike(grug_train::tokenizer::VOCAB_SIZE);
    let (mut model, mut optim) = init_training::<ActiveBackend>(&model_config, &device);

    let param_count = burn::module::Module::num_params(&model);
    println!(
        "Spike-0: TinyTransformer(vocab={}, d_model={}, n_heads={}, n_layers={}, d_ff={}, max_seq_len={}), ~{param_count} params",
        model_config.vocab_size,
        model_config.d_model,
        model_config.n_heads,
        model_config.n_layers,
        model_config.d_ff,
        model_config.max_seq_len,
    );
    println!(
        "Training on {} examples, {TOTAL_STEPS} steps, batch size {BATCH_SIZE}, lr {}",
        train_examples.len(),
        train_cfg.learning_rate
    );

    let gen_config = GenerateConfig {
        max_new_tokens: 64,
        max_seq_len: model_config.max_seq_len,
    };
    let log_path = default_log_path();
    // Start each run from a clean time-series file so it reflects only this
    // run's curve, not a concatenation of previous runs.
    let _ = std::fs::remove_file(&log_path);

    let run_start = Instant::now();
    let mut step_times_ns: Vec<u128> = Vec::with_capacity(TOTAL_STEPS);

    for step in 0..TOTAL_STEPS {
        let batch_examples: Vec<_> = (0..BATCH_SIZE)
            .map(|i| train_examples[(step * BATCH_SIZE + i) % train_examples.len()].clone())
            .collect();
        let batch = build_batch::<ActiveBackend, _>(
            &tokenizer,
            &batch_examples,
            model_config.max_seq_len,
            &device,
        )
        .expect("byte tokenizer encoding never fails");

        let step_start = Instant::now();
        let (next_model, loss) = train_step(model, &mut optim, &batch, train_cfg.learning_rate);
        step_times_ns.push(step_start.elapsed().as_nanos());
        model = next_model;

        if step % 10 == 0 || step == TOTAL_STEPS - 1 {
            println!("step {step:4}: loss = {loss:.4}");
        }

        if step % EVAL_EVERY == 0 || step == TOTAL_STEPS - 1 {
            let eval_model = grug_train::train::to_eval_model(&model);
            let row = run_eval_and_log(
                &eval_model,
                &tokenizer,
                &holdout,
                gen_config,
                &device,
                step,
                &log_path,
            )
            .expect("eval log write should succeed");
            println!(
                "  eval@{step}: valid_json={:.3} schema={:.3} arg_correct={:.3} distinct_call_ratio={:.3} tool_entropy={:.3} (n={})",
                row.valid_json_rate,
                row.schema_adherence_rate,
                row.arg_correct_rate,
                row.distinct_call_ratio,
                row.tool_name_entropy,
                row.n,
            );
        }
    }

    let total_elapsed = run_start.elapsed();
    let avg_step_ms = step_times_ns.iter().sum::<u128>() as f64 / step_times_ns.len() as f64 / 1e6;
    println!(
        "Done: {TOTAL_STEPS} steps in {:.2}s ({avg_step_ms:.2} ms/step avg). Eval time series: {}",
        total_elapsed.as_secs_f64(),
        log_path.display()
    );
}
