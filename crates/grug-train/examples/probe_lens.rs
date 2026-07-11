//! Phase 51 GWT workspace-emergence probe, Stage L: the **logit lens** per
//! layer (`docs/ai/research/2026-07-09-global-workspace-language-models.md`
//! §7.1).
//!
//! Inference only. Loads a trained checkpoint, rebuilds the frozen holdout the
//! same way the corresponding `eval_repair`/`eval_registry_repair` example
//! does, then for every holdout example applies the model's *own* final norm +
//! unembedding (`TinyTransformer::lens_logits`) to *every* intermediate hidden
//! state (`TinyTransformer::hidden_layers`), not just the final one. The
//! question: does an intermediate layer already "name" the eventual output
//! token before the model's last block does?
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example probe_lens --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p2-grug --format grug
//! ```

use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Serialize;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::schema::Example;
use grug_data::{Dataset, DistractorConfig, SignalMode, TargetFormat};
use grug_train::tokenizer::Vocab;
use grug_train::train::encode_example_fmt;
use grug_train::transformer::{TinyTransformer, TinyTransformerConfig};
use grug_train::{BpeTokenizer, checkpoint};

// Inference-only probe: a plain (non-autodiff) backend is all that's needed,
// since `load_model_only` never reconstructs optimizer state and nothing here
// calls `.backward()`. Mirrors the eval examples' GPU-feature cfg pattern,
// substituting the plain backend aliases for the Autodiff ones they use for
// training resumption.
#[cfg(not(feature = "gpu"))]
use grug_train::CpuBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "probe_lens")]
struct Args {
    #[arg(long)]
    artifacts_dir: PathBuf,
    /// Checkpoint subdirectory name under `<artifacts-dir>/checkpoints/`.
    /// Defaults to the lexicographically last entry (checkpoints are named
    /// `step_XXXXXXXX`, so this is the latest step).
    #[arg(long)]
    ckpt: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1000)]
    n_tools: usize,
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,
    #[arg(long, default_value_t = 2)]
    distractors: usize,
    /// Only consulted when `--registry` is set.
    #[arg(long, default_value = "name")]
    signal_mode: String,
    /// Reconstruct the dataset/holdout the way `eval_registry_repair.rs`
    /// does (retrieval-scoped registry) instead of `eval_repair.rs`'s plain
    /// P1a two-axis holdout.
    #[arg(long, default_value_t = false)]
    registry: bool,
    /// Retrieval top-k, only used when `--registry` is set (matches
    /// `eval_registry_repair.rs`'s `--retrieval-k` default).
    #[arg(long, default_value_t = 8)]
    retrieval_k: usize,
    #[arg(long, default_value = "json")]
    format: String,
    #[arg(long, default_value_t = 128)]
    examples: usize,
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Per-layer aggregate accumulator for one axis's probe pass.
#[derive(Debug, Clone, Copy, Default)]
struct LayerAgg {
    n: u64,
    top1_gold_hits: u64,
    top1_final_hits: u64,
    mrr_sum: f64,
    n_first: u64,
    first_top1_gold_hits: u64,
}

/// One layer's finished rates, ready for the human table and JSON output.
#[derive(Debug, Clone, Serialize)]
struct LayerStats {
    layer: usize,
    top1_gold: f64,
    top1_final: f64,
    mrr_gold: f64,
    first_tok_top1_gold: f64,
    n_positions: u64,
    n_first_positions: u64,
}

impl LayerStats {
    fn from_agg(layer: usize, agg: &LayerAgg) -> Self {
        let rate = |hits: u64, n: u64| if n == 0 { 0.0 } else { hits as f64 / n as f64 };
        Self {
            layer,
            top1_gold: rate(agg.top1_gold_hits, agg.n),
            top1_final: rate(agg.top1_final_hits, agg.n),
            mrr_gold: if agg.n == 0 {
                0.0
            } else {
                agg.mrr_sum / agg.n as f64
            },
            first_tok_top1_gold: rate(agg.first_top1_gold_hits, agg.n_first),
            n_positions: agg.n,
            n_first_positions: agg.n_first,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct AxisProbe {
    axis: String,
    n_examples: usize,
    n_skipped: usize,
    layers: Vec<LayerStats>,
    /// Smallest layer index with `top1_final >= 0.9`, or `None` if no layer
    /// reaches that threshold (no "verbalization crossover" found).
    crossover_layer: Option<usize>,
}

fn argmax_of(row: &[f32]) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}

/// `(argmax token id, 1-indexed rank of `gold` among the row's logits)`.
/// Rank ties (another token with a strictly higher logit) all count against
/// `gold`; equal logits do not.
fn row_metrics(row: &[f32], gold: u32) -> (u32, usize) {
    let argmax = argmax_of(row);
    let gold_val = row[gold as usize];
    let rank = row.iter().filter(|&&v| v > gold_val).count() + 1;
    (argmax, rank)
}

/// Run the Stage-L logit-lens probe over one axis's holdout examples: one
/// `hidden_layers` forward pass per example (teacher-forced on `prompt + gold
/// completion`), scoring every completion position at every layer.
fn probe_axis<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    examples: &[Example],
    fmt: TargetFormat,
    max_seq_len: usize,
    device: &B::Device,
    axis_name: &str,
) -> AxisProbe {
    let vocab_size = tokenizer.vocab_size();
    let mut aggs: Option<Vec<LayerAgg>> = None;
    let mut n_scored = 0usize;
    let mut n_skipped = 0usize;

    for example in examples {
        let encoded = match encode_example_fmt(tokenizer, example, fmt) {
            Ok(e) => e,
            Err(_) => {
                n_skipped += 1;
                continue;
            }
        };
        let ids = &encoded.ids;
        // Completion length excludes the trailing EOS `encode_example_fmt`
        // appends -- we score predictions of the gold *content* tokens, not
        // the stop token.
        let completion_len = ids.len().saturating_sub(encoded.target_start + 1);
        if ids.is_empty() || ids.len() > max_seq_len || completion_len == 0 {
            n_skipped += 1;
            continue;
        }

        let seq_len = ids.len();
        let ids_i64: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i64, [1, seq_len]), device);

        let layers = model.hidden_layers(input);
        let layer_flats: Vec<Vec<f32>> = layers
            .iter()
            .map(|h| {
                model
                    .lens_logits(h.clone())
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            })
            .collect();
        let final_flat = layer_flats
            .last()
            .expect("hidden_layers returns >= 1 entry");

        if aggs.is_none() {
            aggs = Some(vec![LayerAgg::default(); layer_flats.len()]);
        }
        let aggs = aggs.as_mut().unwrap();

        for c in 0..completion_len {
            let gold = ids[encoded.target_start + c];
            let pos = encoded.target_start + c - 1;
            let final_row = &final_flat[pos * vocab_size..(pos + 1) * vocab_size];
            let final_argmax = argmax_of(final_row);

            for (layer_idx, flat) in layer_flats.iter().enumerate() {
                let row = &flat[pos * vocab_size..(pos + 1) * vocab_size];
                let (row_argmax, rank) = row_metrics(row, gold);
                let agg = &mut aggs[layer_idx];
                agg.n += 1;
                if row_argmax == gold {
                    agg.top1_gold_hits += 1;
                }
                if row_argmax == final_argmax {
                    agg.top1_final_hits += 1;
                }
                agg.mrr_sum += 1.0 / rank as f64;
                if c == 0 {
                    agg.n_first += 1;
                    if row_argmax == gold {
                        agg.first_top1_gold_hits += 1;
                    }
                }
            }
        }
        n_scored += 1;
    }

    let aggs = aggs.unwrap_or_default();
    let layers: Vec<LayerStats> = aggs
        .iter()
        .enumerate()
        .map(|(i, agg)| LayerStats::from_agg(i, agg))
        .collect();
    let crossover_layer = layers.iter().find(|l| l.top1_final >= 0.9).map(|l| l.layer);

    AxisProbe {
        axis: axis_name.to_string(),
        n_examples: n_scored,
        n_skipped,
        layers,
        crossover_layer,
    }
}

fn print_axis_table(probe: &AxisProbe) {
    println!(
        "\n--- axis {} (n={}, skipped={}) ---",
        probe.axis, probe.n_examples, probe.n_skipped
    );
    println!(
        "{:>6} | {:>10} | {:>11} | {:>8} | {:>18}",
        "layer", "top1_gold", "top1_final", "mrr", "first_tok_top1_gold"
    );
    for l in &probe.layers {
        println!(
            "{:>6} | {:>10.3} | {:>11.3} | {:>8.3} | {:>18.3}",
            l.layer, l.top1_gold, l.top1_final, l.mrr_gold, l.first_tok_top1_gold
        );
    }
    match probe.crossover_layer {
        Some(l) => println!("crossover layer (top1_final >= 0.9): {l}"),
        None => println!("crossover layer (top1_final >= 0.9): none found"),
    }
}

/// Resolve the checkpoint directory: `--ckpt` if given, else the
/// lexicographically last `checkpoints/<name>` subdirectory (checkpoint dirs
/// are named `step_XXXXXXXX`, so this is the latest step).
fn resolve_checkpoint_dir(artifacts_dir: &Path, ckpt: Option<&str>) -> PathBuf {
    let checkpoints_dir = artifacts_dir.join("checkpoints");
    if let Some(name) = ckpt {
        return checkpoints_dir.join(name);
    }
    let mut entries: Vec<String> = std::fs::read_dir(&checkpoints_dir)
        .unwrap_or_else(|e| panic!("reading checkpoints dir {}: {e}", checkpoints_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    entries.sort();
    let last = entries.last().unwrap_or_else(|| {
        panic!(
            "no checkpoint subdirectories found under {}",
            checkpoints_dir.display()
        )
    });
    checkpoints_dir.join(last)
}

fn main() {
    let args = Args::parse();
    let fmt = TargetFormat::parse_label(&args.format).expect("--format json|grug");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("probe_lens: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("probe_lens: CPU backend");
        Default::default()
    };

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;

    let ckpt_dir = resolve_checkpoint_dir(&args.artifacts_dir, args.ckpt.as_deref());
    let model = model_config.init::<ActiveBackend>(&device);
    let (model, step) =
        checkpoint::load_model_only(&ckpt_dir, model, &device).expect("checkpoint loads");
    println!(
        "probe_lens: FORMAT={fmt:?} REGISTRY={} checkpoint {} (step {step})",
        args.registry,
        ckpt_dir.display()
    );

    let distractors = DistractorConfig::easy_with_count(args.distractors);

    let axes: Vec<AxisProbe> = if args.registry {
        let signal_mode = SignalMode::parse_label(&args.signal_mode)
            .expect("--signal-mode must be name|d0|description|d1|paraphrase|d2");
        let dataset = Dataset::new_sized_signal(
            args.seed,
            args.n_tools,
            args.holdout_frac,
            distractors,
            signal_mode,
        )
        .expect("dataset splits cleanly");

        // eval_registry_repair.rs's reconstruction: retrieval-scoped registry
        // holdout (`scoped`, a call/gap mix) and genuine out-of-registry
        // novelty (`novelty`, all-gap). `FrozenRetrievalHoldout` doesn't expose
        // its example vecs publicly (only `recall_at_k`), so this calls the
        // same underlying `Dataset` methods it wraps directly.
        let scoped = dataset
            .retrieval_holdout_examples(args.retrieval_k, args.examples)
            .expect("retrieval-scoped holdout draws");
        let novelty = dataset
            .retrieval_abstention_examples(args.retrieval_k, args.examples)
            .expect("retrieval-abstention (novelty) holdout draws");

        vec![
            probe_axis(
                &model,
                &tokenizer,
                &scoped,
                fmt,
                max_seq_len,
                &device,
                "registry_scoped",
            ),
            probe_axis(
                &model,
                &tokenizer,
                &novelty,
                fmt,
                max_seq_len,
                &device,
                "novelty_abstention",
            ),
        ]
    } else {
        let dataset = Dataset::new_sized_with_distractors(
            args.seed,
            args.n_tools,
            args.holdout_frac,
            distractors,
        )
        .expect("dataset splits cleanly");

        // eval_repair.rs's reconstruction: the plain P1a two-axis holdout.
        let phrasing = dataset
            .holdout_phrasing_examples(args.examples)
            .expect("phrasing holdout draws");
        let whole_tool = dataset
            .holdout_examples(args.examples)
            .expect("whole-tool holdout draws");

        vec![
            probe_axis(
                &model,
                &tokenizer,
                &phrasing,
                fmt,
                max_seq_len,
                &device,
                "axis_b_held_out_phrasing",
            ),
            probe_axis(
                &model,
                &tokenizer,
                &whole_tool,
                fmt,
                max_seq_len,
                &device,
                "axis_a_whole_tool_holdout",
            ),
        ]
    };

    println!("\n=== Phase 51 GWT probe -- Stage L (logit lens per layer) ===");
    for axis in &axes {
        print_axis_table(axis);
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.artifacts_dir.join("probe_lens.json"));
    let summary = serde_json::json!({
        "checkpoint_step": step,
        "checkpoint_dir": ckpt_dir.display().to_string(),
        "format": args.format,
        "registry": args.registry,
        "signal_mode": args.signal_mode,
        "seed": args.seed,
        "n_tools": args.n_tools,
        "holdout_frac": args.holdout_frac,
        "distractors": args.distractors,
        "retrieval_k": args.retrieval_k,
        "examples_requested": args.examples,
        "axes": axes,
    });
    match std::fs::write(&out_path, serde_json::to_string_pretty(&summary).unwrap()) {
        Ok(()) => println!("\nprobe_lens: wrote {}", out_path.display()),
        Err(e) => eprintln!("probe_lens: write failed: {e}"),
    }
}
