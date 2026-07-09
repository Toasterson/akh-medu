//! Phase 51 P3a — characterise the argument-fidelity residual using the REAL
//! eval scorer (`grug_data::score_output_fmt`), so the breakdown is exactly
//! what `arg_correct_rate` penalises — no raw-`==` discrepancy. For each
//! call-holdout example it buckets the outcome (arg-correct / schema-invalid /
//! gap-on-call / residual) and, for the schema-valid-but-arg-wrong residual,
//! tallies the scorer's own `ArgMismatch`es (ValueDiffers vs PresenceDiffers),
//! flagging two suspicious sub-cases: numeric-close value diffs (would mean the
//! 1e-6 tolerance failed — a scorer bug) and empty-string values (a generator
//! artifact from empty slots for omitted optional params). Inference only.
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example inspect_registry --features gpu --release -- \
//!     --n-tools 1000 --holdout-frac 0.15 --retrieval-k 8 --n 400 \
//!     --artifacts-dir /home/toasty/ws/akh-medu/runs/p3a-registry \
//!     --resume /home/toasty/ws/akh-medu/runs/p3a-registry/checkpoints/step_00019999
//! ```

use std::path::PathBuf;

use clap::Parser;
use grug_data::{ArgMismatch, Dataset, DistractorConfig, ParsedOutput, TargetFormat};
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
    #[arg(long, default_value_t = 400)]
    n: usize,
    #[arg(long, default_value_t = 96)]
    eval_max_new_tokens: usize,
}

fn is_empty_val(v: &serde_json::Value) -> bool {
    v.as_str() == Some("")
}

/// Render a JSON value the way it appears in the request text (unquoted).
fn val_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn type_tag(ty: &grug_data::ParamType) -> &'static str {
    match ty {
        grug_data::ParamType::String => "str",
        grug_data::ParamType::Integer => "int",
        grug_data::ParamType::Number => "num",
        grug_data::ParamType::Boolean => "bool",
        grug_data::ParamType::Enum(_) => "enum",
    }
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

    let tokenizer = BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;
    let (model, optim) = init_training::<ActiveBackend>(&model_config, &device);
    let (model, _optim, step) = checkpoint::load_checkpoint(&args.resume, model, optim, &device).expect("checkpoint");
    let model = to_eval_model(&model);
    println!("inspect: step {step}, scoring {} call examples with the real eval scorer\n", args.n);

    let examples = dataset.retrieval_holdout_examples(args.retrieval_k, args.n).expect("holdout");
    let gen_config = GenerateConfig { max_new_tokens: args.eval_max_new_tokens, max_seq_len };

    // Buckets (mutually exclusive, sum to n_call):
    let (mut n_call, mut arg_correct, mut schema_invalid, mut gap_on_call) = (0usize, 0usize, 0usize, 0usize);
    // Residual = schema-valid but arg-wrong:
    let (mut residual, mut case_value_only, mut case_presence_only, mut case_mixed, mut case_wrong_tool) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // Per-param mismatch tallies + suspicious sub-cases:
    let (mut value_diff_params, mut presence_diff_params, mut numeric_close_diffs, mut empty_val_diffs) =
        (0usize, 0usize, 0usize, 0usize);
    // Value-grounding design inputs: is the CORRECT value in the request (the
    // premise a grounding pass depends on)? Is the WRONG value in the request
    // (misalignment) or not (hallucination)? And are the errors a pure
    // permutation of the right values (→ type-aware re-alignment fixes them)?
    let (mut want_in_req, mut got_in_req, mut permutation_cases, mut value_cases_total) =
        (0usize, 0usize, 0usize, 0usize);
    use std::collections::BTreeMap;
    let mut err_by_type: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut shown = 0;

    for ex in &examples {
        if ex.is_abstention() {
            continue;
        }
        n_call += 1;
        let raw = generate(&model, &tokenizer, &format_prompt_fmt(ex, fmt), gen_config, &device);
        let outcome = grug_data::eval::score_output_fmt(&raw, ex, fmt);
        if outcome.arg_correct() {
            arg_correct += 1;
            continue;
        }
        // Not arg-correct. Distinguish gap-on-call / schema-invalid / residual.
        if matches!(grug_data::parse_output(&raw, fmt, &ex.tools), ParsedOutput::Abstain) {
            gap_on_call += 1;
            continue;
        }
        if !outcome.schema_valid() {
            schema_invalid += 1;
            continue;
        }
        residual += 1;

        let call = match &outcome.schema {
            Some(Ok(c)) => c.clone(),
            _ => continue,
        };
        let got = call.arguments_map();
        let want = ex.expected_call.arguments_map();
        let schema = ex.tools.iter().find(|t| t.name == call.name);
        let req = &ex.user_request;

        let (mut has_val, mut has_pres, mut wrong_tool_here) = (false, false, false);
        if let Some(Err(mismatches)) = &outcome.arg_correctness {
            for m in mismatches {
                match m {
                    ArgMismatch::ValueDiffers { param } => {
                        if param == "name" {
                            wrong_tool_here = true;
                            continue;
                        }
                        has_val = true;
                        value_diff_params += 1;
                        let w = want.get(param);
                        let g = got.get(param);
                        // Numeric-close: both parse as f64 within 1e-6 → the
                        // scorer's tolerance *should* have matched these; if this
                        // fires, it is a scorer/type bug worth fixing.
                        if let (Some(wf), Some(gf)) = (w.and_then(|v| v.as_f64()), g.and_then(|v| v.as_f64())) {
                            if (wf - gf).abs() < 1e-6 {
                                numeric_close_diffs += 1;
                            }
                        }
                        if w.is_some_and(is_empty_val) || g.is_some_and(is_empty_val) {
                            empty_val_diffs += 1;
                        }
                        // Value-grounding design inputs.
                        if let Some(wv) = w {
                            if req.contains(&val_text(wv)) {
                                want_in_req += 1;
                            }
                        }
                        if let Some(gv) = g {
                            if req.contains(&val_text(gv)) {
                                got_in_req += 1;
                            }
                        }
                        if let Some(ty) = schema.and_then(|s| s.param(param)).map(|p| &p.ty) {
                            *err_by_type.entry(type_tag(ty)).or_insert(0) += 1;
                        }
                    }
                    ArgMismatch::PresenceDiffers { param } => {
                        has_pres = true;
                        presence_diff_params += 1;
                        if want.get(param).is_some_and(is_empty_val) || got.get(param).is_some_and(is_empty_val) {
                            empty_val_diffs += 1;
                        }
                    }
                }
            }
        }
        if wrong_tool_here {
            case_wrong_tool += 1;
        } else if has_val && has_pres {
            case_mixed += 1;
        } else if has_val {
            case_value_only += 1;
        } else if has_pres {
            case_presence_only += 1;
        }

        // Permutation test (for cases with a value error): is the model emitting
        // the correct MULTISET of values, just assigned to the wrong params? If
        // so, a type-aware re-alignment could recover them; if not, values are
        // hallucinated/corrupted and re-alignment cannot.
        if has_val && !wrong_tool_here {
            value_cases_total += 1;
            let mut want_vals: Vec<String> = want.values().map(val_text).collect();
            let mut got_vals: Vec<String> = got.values().map(val_text).collect();
            want_vals.sort();
            got_vals.sort();
            if want_vals == got_vals {
                permutation_cases += 1;
            }
        }

        if shown < 20 && !wrong_tool_here {
            shown += 1;
            let mism: Vec<String> = outcome
                .arg_correctness
                .as_ref()
                .and_then(|r| r.as_ref().err())
                .map(|v| {
                    v.iter()
                        .map(|m| match m {
                            ArgMismatch::ValueDiffers { param } => {
                                format!("{param}: want={:?} got={:?}", want.get(param), got.get(param))
                            }
                            ArgMismatch::PresenceDiffers { param } => {
                                format!("{param}: PRESENCE want={:?} got={:?}", want.get(param), got.get(param))
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            println!("--- residual #{shown}  tool={} ---", call.name);
            println!("  request: {}", ex.user_request);
            for m in &mism {
                println!("    {m}");
            }
        }
    }

    let pct = |x: usize, d: usize| if d == 0 { 0.0 } else { 100.0 * x as f64 / d as f64 };
    println!("\n=== outcome buckets (n_call={n_call}) ===");
    println!("  arg-correct    : {arg_correct} ({:.1}%)", pct(arg_correct, n_call));
    println!("  schema-invalid : {schema_invalid} ({:.1}%)", pct(schema_invalid, n_call));
    println!("  gap-on-call    : {gap_on_call} ({:.1}%)  [model wrongly abstained]", pct(gap_on_call, n_call));
    println!("  residual (schema-valid, arg-wrong): {residual} ({:.1}%)", pct(residual, n_call));
    println!("\n=== residual case breakdown ({residual}) ===");
    println!("  wrong-tool (ValueDiffers on 'name'): {case_wrong_tool}");
    println!("  value-only                         : {case_value_only}");
    println!("  presence-only                      : {case_presence_only}");
    println!("  mixed (value + presence)           : {case_mixed}");
    println!("\n=== per-param mismatch tallies ===");
    println!("  ValueDiffers params   : {value_diff_params}  (numeric-close [scorer-bug flag]: {numeric_close_diffs})");
    println!("  PresenceDiffers params: {presence_diff_params}");
    println!("  mismatches involving an EMPTY-string value [generator artifact]: {empty_val_diffs}");
    println!("\n=== value-grounding design inputs (over {value_diff_params} ValueDiffers) ===");
    println!(
        "  CORRECT value appears in request : {want_in_req}/{value_diff_params} ({:.1}%)  [grounding premise: can we re-extract it?]",
        pct(want_in_req, value_diff_params)
    );
    println!(
        "  WRONG value appears in request   : {got_in_req}/{value_diff_params} ({:.1}%)  [in-req ⇒ misalignment; not-in-req ⇒ hallucination]",
        pct(got_in_req, value_diff_params)
    );
    println!(
        "  permutation cases (right value multiset, wrong assignment): {permutation_cases}/{value_cases_total}  [re-alignment could fix these]"
    );
    println!("  value errors by param type: {err_by_type:?}");
}
