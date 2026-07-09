//! Phase 51 P3a — inspect the selection-error residual (design input for the
//! re-ranker). Loads a `train_registry` checkpoint, generates on the call
//! holdout, and for the arg-wrong cases prints the error structure: expected
//! tool vs the tool the model chose, whether the retriever ranks the expected
//! tool #1, and emitted-vs-expected arg names. Inference only, no scoring.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example inspect_registry --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --retrieval-k 8 --n 40 \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p3a-registry \
//!     --resume /home/toasty/ws/agi/akh-medu/runs/p3a-registry/checkpoints/step_00019999
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{Dataset, DistractorConfig, ParsedOutput, Retriever, TargetFormat};
use grug_train::generate::{generate, GenerateConfig};
use grug_train::train::{format_prompt_fmt, init_training, to_eval_model};
use grug_train::transformer::TinyTransformerConfig;
use grug_train::{checkpoint, BpeTokenizer};

#[cfg(feature = "gpu")]
use grug_train::GpuAutodiffBackend as ActiveBackend;
#[cfg(not(feature = "gpu"))]
use grug_train::CpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "inspect_registry")]
struct Args {
    #[arg(long, default_value = "grug")]
    format: String,
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
    #[arg(long, default_value_t = 8)]
    retrieval_k: usize,
    #[arg(long, default_value_t = 200)]
    n: usize,
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,
}

fn main() {
    let args = Args::parse();
    let fmt = TargetFormat::parse_label(&args.format).expect("--format json|grug");

    #[cfg(feature = "gpu")]
    let device = { let d = burn::backend::rocm::RocmDevice::default(); println!("inspect: GPU {d:?}"); d };
    #[cfg(not(feature = "gpu"))]
    let device = { println!("inspect: CPU"); Default::default() };

    let dataset = match args.n_tools {
        Some(n) => Dataset::new_sized_with_distractors(args.seed, n, args.holdout_frac, DistractorConfig::easy_with_count(2)),
        None => Dataset::new(args.seed),
    }
    .expect("dataset");
    let retriever = Retriever::new(dataset.registry());

    let tokenizer = BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;
    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) = checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint");
    let model = to_eval_model(&model);
    println!("inspect: step {step}, generating on {} call examples\n", args.n);

    let examples = dataset.retrieval_holdout_examples(args.retrieval_k, args.n).expect("holdout");
    let gen_config = GenerateConfig { max_new_tokens: args.eval_max_new_tokens, max_seq_len };

    let (mut n_call, mut n_wrong, mut wrong_tool, mut right_tool_wrong_args, mut retr_top1_right) = (0, 0, 0, 0, 0);
    // Residual breakdown among right-tool-wrong-args:
    let (mut value_only, mut invalid_name, mut presence_only) = (0, 0, 0);
    let mut shown = 0;
    for ex in &examples {
        if ex.is_abstention() {
            continue;
        }
        n_call += 1;
        let raw = generate(&model, &tokenizer, &format_prompt_fmt(ex, fmt), gen_config, &device);
        let parsed = grug_data::parse_output(&raw, fmt, &ex.tools);
        let expected = &ex.expected_call;
        let call = match parsed {
            ParsedOutput::Call(c) => c,
            _ => continue, // gap/unparseable on a call example: not a selection error
        };
        // arg-correct?
        let arg_ok = call.name == expected.name && call.arguments_map() == expected.arguments_map();
        if arg_ok {
            continue;
        }
        n_wrong += 1;

        // Where does the retriever rank the expected tool for this request?
        let exp_idx = dataset.registry().iter().position(|t| t.name == expected.name);
        let retr_top = retriever.top_k(&ex.user_request, args.retrieval_k);
        let retr_top1_name = retr_top.first().map(|&i| dataset.registry()[i].name.as_str());
        let retr_ranks_expected_first = retr_top1_name == Some(expected.name.as_str());
        if retr_ranks_expected_first {
            retr_top1_right += 1;
        }
        let _ = exp_idx;

        if call.name != expected.name {
            wrong_tool += 1;
        } else {
            right_tool_wrong_args += 1;
            // Classify the argument error against the (correct) tool's schema.
            let schema = ex.tools.iter().find(|t| t.name == call.name);
            let got = call.arguments_map();
            let want = expected.arguments_map();
            let got_keys: std::collections::BTreeSet<&String> = got.keys().collect();
            let want_keys: std::collections::BTreeSet<&String> = want.keys().collect();
            let has_invalid_name = schema
                .map(|s| got.keys().any(|k| s.param(k).is_none()))
                .unwrap_or(false);
            if got_keys == want_keys {
                value_only += 1; // same arg names, some value differs
            } else if has_invalid_name {
                invalid_name += 1; // emitted a param name not on the schema — M2-repairable
            } else {
                presence_only += 1; // all names valid schema params, but wrong optional subset
            }
        }

        if shown < 25 {
            shown += 1;
            let exp_arg_names: Vec<String> = expected.arguments_map().keys().cloned().collect();
            let got_arg_names: Vec<String> = call.arguments_map().keys().cloned().collect();
            println!("--- wrong #{shown} ---");
            println!("  request : {}", ex.user_request);
            println!("  expected: {} args={:?}", expected.name, exp_arg_names);
            println!("  model   : {} args={:?}", call.name, got_arg_names);
            println!("  offered : {:?}", ex.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>());
            println!("  retriever top1 = {:?}  (ranks expected first: {})", retr_top1_name, retr_ranks_expected_first);
        }
    }

    println!("\n=== summary (n_call={n_call}) ===");
    println!("arg-wrong (parsed as call): {n_wrong}");
    println!("  of which WRONG TOOL     : {wrong_tool}");
    println!("  of which right tool/args: {right_tool_wrong_args}");
    println!("  wrong-tool cases where retriever ranks the EXPECTED tool #1: {retr_top1_right}/{wrong_tool}");
    println!("  right-tool-wrong-args breakdown ({right_tool_wrong_args}):");
    println!("    value-only (names match, value differs)   : {value_only}  [generation error — needs value re-grounding, NOT re-rank/repair]");
    println!("    invalid-name (emitted a non-schema param) : {invalid_name}  [M2 param-grounding repairable]");
    println!("    presence-only (valid names, wrong optionals): {presence_only}  [optional-inclusion mismatch]");
}
