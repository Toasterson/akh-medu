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

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use grug_data::schema::Example;
use grug_data::{Dataset, DiversityReport, EvalReport};
use serde::Serialize;

use crate::error::{GrugError, GrugResult};
use crate::generate::{generate, GenerateConfig};
use crate::tokenizer::ByteTokenizer;
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
pub fn run_eval<B: Backend>(
    model: &TinyTransformer<B>,
    tokenizer: &ByteTokenizer,
    holdout: &FrozenHoldout,
    gen_config: GenerateConfig,
    device: &B::Device,
) -> (EvalReport, DiversityReport) {
    let outputs: Vec<(String, Example)> = holdout
        .examples()
        .iter()
        .map(|example| {
            let prompt = format_prompt(example);
            let raw = generate(model, tokenizer, &prompt, gen_config, device);
            (raw, example.clone())
        })
        .collect();

    let eval_report = EvalReport::from_outputs(&outputs);
    let diversity_report = DiversityReport::from_outputs(&outputs);
    (eval_report, diversity_report)
}

/// Run one eval pass and append its result as one JSON line to `log_path`
/// (created if it does not exist, appended to otherwise).
pub fn run_eval_and_log<B: Backend>(
    model: &TinyTransformer<B>,
    tokenizer: &ByteTokenizer,
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

fn append_row(row: &EvalLogRow, path: &Path) -> GrugResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
