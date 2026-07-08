//! Continuous eval harness (Phase 51 plan decision 9): every N training
//! steps, greedy-generate on a **frozen** holdout batch, score with
//! `grug_data::eval`, and append one row to a JSONL time-series file.
//!
//! The frozen holdout matters: it is drawn once (via
//! [`FrozenHoldout::new`]) and reused for every eval call over the whole
//! training run, so successive `EvalReport`/`DiversityReport` rows are
//! actually comparable to each other -- a moving eval set would confound
//! "the model got better" with "this batch happened to be easier".
//!
//! The anti-collapse [`grug_data::DiversityReport`] is logged on *every*
//! row unconditionally, per decision 9 ("the alarm") -- there is no
//! opt-out, because a collapsed model can otherwise look fine on
//! `valid_json_rate` alone (e.g. always emitting the same well-formed but
//! wrong call).
//!
//! # P1a: two orthogonal holdout axes
//!
//! The P0 [`FrozenHoldout`]/[`run_eval`] pair only ever drew from
//! `Dataset::holdout_examples` (whole-tool holdout, axis A). The P1a scale-up
//! plan requires **both** axes logged every eval: axis A (whole-tool -- can
//! the model read an unfamiliar schema at all?) and axis B (held-out-phrasing
//! -- does the model generalize to unseen wordings of tools it *was*
//! trained on?). [`FrozenTwoAxisHoldout`] freezes both batches once and
//! [`run_two_axis_eval_and_log`] scores + logs both every eval step, tagging
//! each JSONL row with an `axis` field so a single time-series file (or two
//! separate ones, at the caller's option) can be split back out per axis.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use grug_data::schema::Example;
use grug_data::{Dataset, DiversityReport, EvalAxis, EvalReport};
use serde::Serialize;

use grug_data::constrain::ToolCallGrammar;

use crate::error::{GrugError, GrugResult};
use crate::generate::{generate, generate_constrained, generate_ptr, generate_ptr_constrained, GenerateConfig};
use crate::pointer::PointerTransformer;
use crate::tokenizer::Vocab;
use crate::train::format_prompt;
use crate::transformer::TinyTransformer;

/// A holdout batch drawn once and reused for every eval call in a training
/// run, so the time series is measuring model improvement, not batch
/// difficulty drift.
pub struct FrozenHoldout {
    examples: Vec<Example>,
}

impl FrozenHoldout {
    /// Draw `n` examples from `dataset`'s holdout tool set (never seen in
    /// training, per `Dataset::holdout_examples`'s whole-tool split).
    pub fn new(dataset: &Dataset, n: usize) -> GrugResult<Self> {
        let examples = dataset.holdout_examples(n)?;
        Ok(Self { examples })
    }

    pub fn examples(&self) -> &[Example] {
        &self.examples
    }
}

/// One row of the eval time series: a training step number plus the scored
/// reports for that step, in a shape ready for `serde_json` line output.
#[derive(Debug, Clone, Serialize)]
pub struct EvalLogRow {
    pub step: usize,
    pub valid_json_rate: f64,
    pub schema_adherence_rate: f64,
    pub arg_correct_rate: f64,
    pub distinct_call_ratio: f64,
    pub tool_name_entropy: f64,
    pub n: usize,
}

impl EvalLogRow {
    fn from_reports(step: usize, eval: &EvalReport, diversity: &DiversityReport) -> Self {
        Self {
            step,
            valid_json_rate: eval.valid_json_rate,
            schema_adherence_rate: eval.schema_adherence_rate,
            arg_correct_rate: eval.arg_correct_rate,
            distinct_call_ratio: diversity.distinct_call_ratio,
            tool_name_entropy: diversity.tool_name_entropy,
            n: eval.n,
        }
    }
}

/// Run one eval pass: greedy-generate on every example in `holdout`, score
/// with `grug_data::eval`, and return the `(EvalReport, DiversityReport)`
/// pair -- without writing anything to disk (see [`run_eval_and_log`] for
/// the disk-writing wrapper used by the continuous-eval loop).
pub fn run_eval<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (EvalReport, DiversityReport) {
    let outputs = generate_outputs(model, tokenizer, holdout.examples(), gen_config, device);
    let eval_report = EvalReport::from_outputs(&outputs);
    let diversity_report = DiversityReport::from_outputs(&outputs);
    (eval_report, diversity_report)
}

/// Run one eval pass and append its result as one JSON line to `log_path`
/// (created if it does not exist, appended to otherwise).
pub fn run_eval_and_log<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
    step: usize,
    log_path: impl AsRef<Path>,
) -> GrugResult<EvalLogRow> {
    let (eval_report, diversity_report) = run_eval(model, tokenizer, holdout, gen_config, device);
    let row = EvalLogRow::from_reports(step, &eval_report, &diversity_report);
    append_row(&row, log_path.as_ref())?;
    Ok(row)
}

/// Greedy-generate on every example in `examples`, pairing each raw output
/// with its source `Example` -- the shared inner loop [`run_eval`] and the
/// two-axis eval below both build on.
fn generate_outputs<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    examples: &[Example],
    gen_config: GenerateConfig,
    device: &B::Device,
) -> Vec<(String, Example)> {
    examples
        .iter()
        .map(|example| {
            let prompt = format_prompt(example);
            let raw = generate(model, tokenizer, &prompt, gen_config, device);
            (raw, example.clone())
        })
        .collect()
}

/// Two-axis eval under an explicit [`TargetFormat`] (Phase 51 P2): the prompt
/// is rendered with `format_prompt_fmt` and outputs scored with
/// `Dataset::score_axis_fmt`, so a JSON-vs-Grug A/B is a pure format swap on an
/// otherwise-identical harness. Plain greedy decoding (`generate`).
pub fn run_two_axis_eval_fmt<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
    fmt: grug_data::TargetFormat,
) -> (grug_data::AxisReport, grug_data::AxisReport) {
    let axis = |examples: &[Example], which: grug_data::EvalAxis| {
        let outputs: Vec<(String, Example)> = examples
            .iter()
            .map(|ex| {
                let prompt = crate::train::format_prompt_fmt(ex, fmt);
                (generate(model, tokenizer, &prompt, gen_config, device), ex.clone())
            })
            .collect();
        Dataset::score_axis_fmt(which, &outputs, fmt)
    };
    (
        axis(holdout.phrasing_examples(), grug_data::EvalAxis::HeldOutPhrasing),
        axis(holdout.whole_tool_examples(), grug_data::EvalAxis::WholeToolHoldout),
    )
}

/// Two-axis eval that generates once and scores each axis **both raw and
/// KG-repaired** (Phase 51 P3b): returns `((raw_b, raw_a), (repaired_b,
/// repaired_a))`. The raw→repaired delta is what the symbolic grounding/repair
/// layer (`grug_data::repair`) recovers on top of the neural proposal.
pub fn run_two_axis_eval_raw_and_repaired_fmt<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
    fmt: grug_data::TargetFormat,
) -> (
    (grug_data::AxisReport, grug_data::AxisReport),
    (grug_data::AxisReport, grug_data::AxisReport),
) {
    let outputs = |examples: &[Example]| -> Vec<(String, Example)> {
        examples
            .iter()
            .map(|ex| {
                let prompt = crate::train::format_prompt_fmt(ex, fmt);
                (generate(model, tokenizer, &prompt, gen_config, device), ex.clone())
            })
            .collect()
    };
    let b_out = outputs(holdout.phrasing_examples());
    let a_out = outputs(holdout.whole_tool_examples());
    (
        (
            Dataset::score_axis_fmt(grug_data::EvalAxis::HeldOutPhrasing, &b_out, fmt),
            Dataset::score_axis_fmt(grug_data::EvalAxis::WholeToolHoldout, &a_out, fmt),
        ),
        (
            Dataset::score_axis_repaired_fmt(grug_data::EvalAxis::HeldOutPhrasing, &b_out, fmt),
            Dataset::score_axis_repaired_fmt(grug_data::EvalAxis::WholeToolHoldout, &a_out, fmt),
        ),
    )
}

/// Greedy-generate on every example under schema-constrained decoding
/// (Phase 51 P1a-4 Step A): the [`generate_outputs`] counterpart that builds a
/// per-example [`ToolCallGrammar`] over the offered tools and calls
/// [`generate_constrained`]. `pieces` is the tokenizer's per-id byte table
/// ([`Vocab::token_pieces`]), built once by the caller.
fn generate_outputs_constrained<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    pieces: &[Option<Vec<u8>>],
    examples: &[Example],
    gen_config: GenerateConfig,
    device: &B::Device,
) -> Vec<(String, Example)> {
    examples
        .iter()
        .map(|example| {
            let prompt = format_prompt(example);
            let grammar = ToolCallGrammar::new(&example.tools);
            let raw = generate_constrained(model, tokenizer, &prompt, &grammar, pieces, gen_config, device);
            (raw, example.clone())
        })
        .collect()
}

/// Run one two-axis eval pass under **schema-constrained decoding** and return
/// the `(axis_b_phrasing, axis_a_whole_tool)` `AxisReport` pair. The
/// unconstrained counterpart is [`run_two_axis_eval`]; running both on the same
/// frozen holdout is the Phase 51 P1a-4 Step A experiment (does forcing valid,
/// schema-consistent parameter names lift schema-adherence — especially the
/// zero-shot whole-tool axis — and how far does argument-correctness follow?).
pub fn run_two_axis_eval_constrained<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    pieces: &[Option<Vec<u8>>],
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (grug_data::AxisReport, grug_data::AxisReport) {
    let phrasing_outputs = generate_outputs_constrained(model, tokenizer, pieces, &holdout.phrasing, gen_config, device);
    let whole_tool_outputs =
        generate_outputs_constrained(model, tokenizer, pieces, &holdout.whole_tool, gen_config, device);

    (
        Dataset::score_axis(EvalAxis::HeldOutPhrasing, &phrasing_outputs),
        Dataset::score_axis(EvalAxis::WholeToolHoldout, &whole_tool_outputs),
    )
}

/// Two-axis eval for the [`PointerTransformer`] under **plain greedy** decoding
/// (Phase 51 P1a-4 Step B): the [`run_two_axis_eval`] counterpart that decodes
/// from the pointer-generator's mixed distribution (`generate_ptr`). Comparing
/// this to the non-pointer baseline (plain greedy: axis-B arg 0.78 / axis-A
/// schema 0.03) isolates the copy mechanism's contribution.
pub fn run_two_axis_eval_ptr<B: Backend, V: Vocab>(
    model: &PointerTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (grug_data::AxisReport, grug_data::AxisReport) {
    let axis = |examples: &[Example], which: EvalAxis| {
        let outputs: Vec<(String, Example)> = examples
            .iter()
            .map(|ex| (generate_ptr(model, tokenizer, &format_prompt(ex), gen_config, device), ex.clone()))
            .collect();
        Dataset::score_axis(which, &outputs)
    };
    (
        axis(&holdout.phrasing, EvalAxis::HeldOutPhrasing),
        axis(&holdout.whole_tool, EvalAxis::WholeToolHoldout),
    )
}

/// Two-axis eval for the [`PointerTransformer`] under **schema-constrained**
/// decoding (Phase 51 P1a-4 Step B): pointer copy mechanism *and* the Step A
/// grammar mask together (`generate_ptr_constrained`).
pub fn run_two_axis_eval_ptr_constrained<B: Backend, V: Vocab>(
    model: &PointerTransformer<B>,
    tokenizer: &V,
    pieces: &[Option<Vec<u8>>],
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (grug_data::AxisReport, grug_data::AxisReport) {
    let axis = |examples: &[Example], which: EvalAxis| {
        let outputs: Vec<(String, Example)> = examples
            .iter()
            .map(|ex| {
                let grammar = ToolCallGrammar::new(&ex.tools);
                let raw = generate_ptr_constrained(
                    model,
                    tokenizer,
                    &format_prompt(ex),
                    &grammar,
                    pieces,
                    gen_config,
                    device,
                );
                (raw, ex.clone())
            })
            .collect();
        Dataset::score_axis(which, &outputs)
    };
    (
        axis(&holdout.phrasing, EvalAxis::HeldOutPhrasing),
        axis(&holdout.whole_tool, EvalAxis::WholeToolHoldout),
    )
}

/// Both halves of the Phase 51 P3c abstention eval, drawn once and frozen for
/// the whole run (like [`FrozenTwoAxisHoldout`]):
///
/// - `call`: should-**call** examples (TRAIN tools × HOLDOUT phrasings, axis-B
///   shape) — used to measure the **false-abstention rate** (does gap-token
///   supervision make the model wrongly abstain on requests a tool *does* cover?).
/// - `abstain`: should-**abstain** examples ([`Dataset::abstention_examples`]) —
///   used to measure **abstain recall** (did the model learn to emit gap when no
///   offered tool fits?).
///
/// Scoring both together yields the full [`grug_data::AbstentionReport`]
/// confusion matrix.
pub struct FrozenAbstentionHoldout {
    call: Vec<Example>,
    abstain: Vec<Example>,
}

impl FrozenAbstentionHoldout {
    /// Draw `n_call` should-call and `n_abstain` should-abstain frozen examples.
    pub fn new(dataset: &Dataset, n_call: usize, n_abstain: usize) -> GrugResult<Self> {
        Ok(Self {
            call: dataset.holdout_phrasing_examples(n_call)?,
            abstain: dataset.abstention_examples(n_abstain)?,
        })
    }

    pub fn call_examples(&self) -> &[Example] {
        &self.call
    }

    pub fn abstain_examples(&self) -> &[Example] {
        &self.abstain
    }
}

/// Run one abstention eval pass (Phase 51 P3c): greedy-generate on both the
/// should-call and should-abstain frozen batches under `fmt`, then score the
/// combined confusion matrix. The two headline rates are `abstain_recall` (on
/// the abstain batch) and `false_abstention_rate` (on the call batch).
pub fn run_abstention_eval_fmt<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenAbstentionHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
    fmt: grug_data::TargetFormat,
) -> grug_data::AbstentionReport {
    let mut pairs: Vec<(String, Example)> = Vec::with_capacity(holdout.call.len() + holdout.abstain.len());
    for ex in holdout.call.iter().chain(holdout.abstain.iter()) {
        let prompt = crate::train::format_prompt_fmt(ex, fmt);
        let raw = generate(model, tokenizer, &prompt, gen_config, device);
        pairs.push((raw, ex.clone()));
    }
    grug_data::AbstentionReport::from_outputs_fmt(&pairs, fmt)
}

fn append_row<R: Serialize>(row: &R, path: &Path) -> GrugResult<()> {
    let line = serde_json::to_string(row).map_err(|e| GrugError::EvalLogSerializeFailed {
        message: e.to_string(),
    })?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| GrugError::EvalLogWriteFailed {
                message: format!("creating parent dir {}: {e}", parent.display()),
            })?;
        }
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| GrugError::EvalLogWriteFailed {
            message: format!("opening {}: {e}", path.display()),
        })?;

    writeln!(file, "{line}").map_err(|e| GrugError::EvalLogWriteFailed {
        message: format!("writing to {}: {e}", path.display()),
    })?;

    Ok(())
}

/// Default eval log path used by `examples/spike0.rs`, relative to the
/// crate root. Callers running from elsewhere should pass an explicit path
/// to [`run_eval_and_log`] instead.
pub fn default_log_path() -> PathBuf {
    PathBuf::from("eval_timeseries.jsonl")
}

/// A label for [`EvalLogRow`]/[`AxisEvalLogRow`] JSONL rows identifying which
/// of the two P1a holdout axes a row came from -- serializes to a short,
/// grep/`jq`-friendly string rather than `grug_data::EvalAxis`'s `Debug`
/// form.
fn axis_label(axis: EvalAxis) -> &'static str {
    match axis {
        EvalAxis::HeldOutPhrasing => "held_out_phrasing",
        EvalAxis::WholeToolHoldout => "whole_tool_holdout",
    }
}

/// One row of the P1a two-axis eval time series: identical fields to
/// [`EvalLogRow`], plus an `axis` tag so a single combined JSONL file (or
/// `jq`/`grep`-filtered per-axis files) can distinguish "train-tool /
/// held-out-phrasing" (axis B, the P1a gate's primary generalization signal)
/// from "whole-tool holdout" (axis A, the "off-the-floor" signal) rows.
#[derive(Debug, Clone, Serialize)]
pub struct AxisEvalLogRow {
    pub step: usize,
    pub axis: &'static str,
    pub valid_json_rate: f64,
    pub schema_adherence_rate: f64,
    pub arg_correct_rate: f64,
    pub distinct_call_ratio: f64,
    pub tool_name_entropy: f64,
    pub n: usize,
}

impl AxisEvalLogRow {
    fn from_axis_report(step: usize, report: &grug_data::AxisReport) -> Self {
        Self {
            step,
            axis: axis_label(report.axis),
            valid_json_rate: report.eval.valid_json_rate,
            schema_adherence_rate: report.eval.schema_adherence_rate,
            arg_correct_rate: report.eval.arg_correct_rate,
            distinct_call_ratio: report.diversity.distinct_call_ratio,
            tool_name_entropy: report.diversity.tool_name_entropy,
            n: report.eval.n,
        }
    }
}

/// Both P1a holdout axes' example batches, drawn once (via
/// [`FrozenTwoAxisHoldout::new`]) and reused for every eval call in a
/// training run -- the two-axis counterpart to [`FrozenHoldout`].
///
/// - `phrasing`: axis B, [`Dataset::holdout_phrasing_examples`] (TRAIN tools
///   x HOLDOUT phrasings) -- the P1a gate's primary signal (valid-JSON >=
///   0.9, arg-correct >= 0.7).
/// - `whole_tool`: axis A, [`Dataset::holdout_examples`] (HOLDOUT tools x any
///   phrasing) -- the "off the floor" signal (valid-JSON > 0).
pub struct FrozenTwoAxisHoldout {
    phrasing: Vec<Example>,
    whole_tool: Vec<Example>,
}

impl FrozenTwoAxisHoldout {
    /// Draw `n_per_axis` examples from each of `dataset`'s two holdout
    /// streams.
    pub fn new(dataset: &Dataset, n_per_axis: usize) -> GrugResult<Self> {
        Ok(Self {
            phrasing: dataset.holdout_phrasing_examples(n_per_axis)?,
            whole_tool: dataset.holdout_examples(n_per_axis)?,
        })
    }

    /// Axis B examples (TRAIN tools x HOLDOUT phrasings).
    pub fn phrasing_examples(&self) -> &[Example] {
        &self.phrasing
    }

    /// Axis A examples (HOLDOUT tools x any phrasing).
    pub fn whole_tool_examples(&self) -> &[Example] {
        &self.whole_tool
    }
}

/// Run one eval pass on **both** frozen holdout axes: greedy-generate on
/// every example in each of `holdout`'s two batches, score each with
/// `grug_data`, and return the `(axis_b_phrasing, axis_a_whole_tool)`
/// `AxisReport` pair -- without writing anything to disk (see
/// [`run_two_axis_eval_and_log`] for the disk-writing wrapper the P1a
/// training loop uses).
pub fn run_two_axis_eval<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (grug_data::AxisReport, grug_data::AxisReport) {
    let phrasing_outputs = generate_outputs(model, tokenizer, &holdout.phrasing, gen_config, device);
    let whole_tool_outputs = generate_outputs(model, tokenizer, &holdout.whole_tool, gen_config, device);

    (
        Dataset::score_axis(EvalAxis::HeldOutPhrasing, &phrasing_outputs),
        Dataset::score_axis(EvalAxis::WholeToolHoldout, &whole_tool_outputs),
    )
}

/// Run one two-axis eval pass and append **two** JSON lines (one per axis,
/// each tagged via [`AxisEvalLogRow::axis`]) to `log_path`. Returns both rows
/// as `(axis_b_phrasing, axis_a_whole_tool)`, mirroring
/// [`run_two_axis_eval`]'s ordering.
///
/// Writing both axes to the *same* file (rather than requiring two separate
/// paths) keeps a single time-series artifact per run; callers that prefer
/// split files can pass a different `log_path` per axis by calling
/// [`run_two_axis_eval`] directly and using [`AxisEvalLogRow`] themselves.
pub fn run_two_axis_eval_and_log<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    holdout: &FrozenTwoAxisHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
    step: usize,
    log_path: impl AsRef<Path>,
) -> GrugResult<(AxisEvalLogRow, AxisEvalLogRow)> {
    let (phrasing_report, whole_tool_report) = run_two_axis_eval(model, tokenizer, holdout, gen_config, device);

    let phrasing_row = AxisEvalLogRow::from_axis_report(step, &phrasing_report);
    let whole_tool_row = AxisEvalLogRow::from_axis_report(step, &whole_tool_report);

    let path = log_path.as_ref();
    append_row(&phrasing_row, path)?;
    append_row(&whole_tool_row, path)?;

    Ok((phrasing_row, whole_tool_row))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteTokenizer;
    use crate::transformer::TinyTransformerConfig;
    use crate::CpuBackend;

    #[test]
    fn eval_harness_runs_end_to_end_and_produces_well_formed_report() {
        let device = Default::default();
        let dataset = Dataset::new(7).unwrap();
        let holdout = FrozenHoldout::new(&dataset, 3).unwrap();

        let model_config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();

        let gen_config = GenerateConfig {
            max_new_tokens: 16,
            max_seq_len: 64,
        };

        let (eval_report, diversity_report) = run_eval(&model, &tokenizer, &holdout, gen_config, &device);

        assert_eq!(eval_report.n, 3);
        assert!((0.0..=1.0).contains(&eval_report.valid_json_rate));
        assert!((0.0..=1.0).contains(&eval_report.schema_adherence_rate));
        assert!((0.0..=1.0).contains(&eval_report.arg_correct_rate));
        assert_eq!(diversity_report.n, 3);
        assert!((0.0..=1.0).contains(&diversity_report.distinct_call_ratio));
        assert!((0.0..=1.0).contains(&diversity_report.tool_name_entropy));
    }

    #[test]
    fn run_eval_and_log_writes_a_jsonl_row() {
        let device = Default::default();
        let dataset = Dataset::new(11).unwrap();
        let holdout = FrozenHoldout::new(&dataset, 2).unwrap();

        let model_config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();
        let gen_config = GenerateConfig {
            max_new_tokens: 8,
            max_seq_len: 64,
        };

        let dir = std::env::temp_dir().join(format!("grug_train_eval_test_{}", std::process::id()));
        let log_path = dir.join("eval_timeseries.jsonl");
        // Ensure a clean slate if a previous run of this test left a file
        // behind (same pid is astronomically unlikely across runs, but a
        // stray dir from a killed run is plausible).
        let _ = std::fs::remove_file(&log_path);

        let row = run_eval_and_log(&model, &tokenizer, &holdout, gen_config, &device, 0, &log_path).unwrap();
        assert_eq!(row.step, 0);
        assert_eq!(row.n, 2);

        let contents = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["step"], 0);
        assert_eq!(parsed["n"], 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P1a: both holdout axes run end-to-end and produce well-formed,
    /// distinctly-tagged reports over a frozen batch drawn once.
    #[test]
    fn two_axis_eval_runs_end_to_end_and_produces_well_formed_reports() {
        let device = Default::default();
        let dataset = Dataset::new(13).unwrap();
        let holdout = FrozenTwoAxisHoldout::new(&dataset, 3).unwrap();

        let model_config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();

        let gen_config = GenerateConfig {
            max_new_tokens: 16,
            max_seq_len: 64,
        };

        let (phrasing_report, whole_tool_report) =
            run_two_axis_eval(&model, &tokenizer, &holdout, gen_config, &device);

        assert_eq!(phrasing_report.axis, EvalAxis::HeldOutPhrasing);
        assert_eq!(whole_tool_report.axis, EvalAxis::WholeToolHoldout);
        for report in [&phrasing_report, &whole_tool_report] {
            assert_eq!(report.eval.n, 3);
            assert!((0.0..=1.0).contains(&report.eval.valid_json_rate));
            assert!((0.0..=1.0).contains(&report.eval.schema_adherence_rate));
            assert!((0.0..=1.0).contains(&report.eval.arg_correct_rate));
            assert_eq!(report.diversity.n, 3);
            assert!((0.0..=1.0).contains(&report.diversity.distinct_call_ratio));
            assert!((0.0..=1.0).contains(&report.diversity.tool_name_entropy));
        }
    }

    /// P1a: logging both axes writes exactly two distinctly-tagged JSON lines
    /// per eval step to the same file.
    #[test]
    fn two_axis_eval_and_log_writes_two_tagged_jsonl_rows() {
        let device = Default::default();
        let dataset = Dataset::new(17).unwrap();
        let holdout = FrozenTwoAxisHoldout::new(&dataset, 2).unwrap();

        let model_config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();
        let gen_config = GenerateConfig {
            max_new_tokens: 8,
            max_seq_len: 64,
        };

        let dir = std::env::temp_dir().join(format!("grug_train_two_axis_eval_test_{}", std::process::id()));
        let log_path = dir.join("eval_axes.jsonl");
        let _ = std::fs::remove_file(&log_path);

        let (phrasing_row, whole_tool_row) =
            run_two_axis_eval_and_log(&model, &tokenizer, &holdout, gen_config, &device, 5, &log_path).unwrap();
        assert_eq!(phrasing_row.step, 5);
        assert_eq!(phrasing_row.axis, "held_out_phrasing");
        assert_eq!(whole_tool_row.step, 5);
        assert_eq!(whole_tool_row.axis, "whole_tool_holdout");

        let contents = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "expected exactly one JSONL row per axis");

        let axes: Vec<String> = lines
            .iter()
            .map(|line| {
                let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
                parsed["axis"].as_str().unwrap().to_string()
            })
            .collect();
        assert!(axes.contains(&"held_out_phrasing".to_string()));
        assert!(axes.contains(&"whole_tool_holdout".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
