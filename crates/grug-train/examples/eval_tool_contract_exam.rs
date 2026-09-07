//! Execute a quarantined all-tool exam through the candidate model.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use grug_data::constrain::DecodeReport;
use grug_data::{
    TOOL_CONTRACT_EXAM_SCHEMA_VERSION, ToolContractEvaluationMode, ToolContractExamEvidence,
    ToolContractExamSet, score_tool_contract_exam,
};
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use grug_train::CpuKnowledgeModel as ActiveKnowledgeModel;
use grug_train::DecodingMode;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use grug_train::GpuKnowledgeModel as ActiveKnowledgeModel;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "eval_tool_contract_exam")]
struct Args {
    /// Training bundle containing candidate-model.json and immutable files.
    #[arg(long)]
    bundle: PathBuf,

    /// Quarantined checkpoint-bound ToolContractExamSet JSON.
    #[arg(long)]
    exam: PathBuf,

    /// Checkpoint-executed evidence consumed by model certification.
    #[arg(long)]
    evidence_out: PathBuf,

    /// Optional raw outputs for forensic review.
    #[arg(long)]
    predictions_out: Option<PathBuf>,

    /// Optional JSONL observations accepted directly by the adaptive exam
    /// controller, one independently steerable family per captured tool.
    #[arg(long)]
    observations_out: Option<PathBuf>,

    /// Optional per-case decode reports (which fields were value-copy
    /// constrained and whether the emitted value landed on a context
    /// candidate), for forensic review of ADR 104 D1 copy fidelity.
    #[arg(long)]
    decode_reports_out: Option<PathBuf>,

    #[arg(long, default_value_t = 384)]
    max_new_tokens: usize,

    /// `constrained` (default) decodes under the runtime `<tool_call>`
    /// grammar plus byte-level value-copy constraints (ADR 104 D1). `plain`
    /// reproduces the pre-D1 unconstrained greedy decoder, for comparing
    /// against historical certification numbers only -- it is never the
    /// deployed runtime's behavior.
    #[arg(long, value_enum, default_value_t = Decoding::Constrained)]
    decoding: Decoding,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Decoding {
    Constrained,
    Plain,
}

impl From<Decoding> for DecodingMode {
    fn from(value: Decoding) -> Self {
        match value {
            Decoding::Constrained => DecodingMode::Constrained,
            Decoding::Plain => DecodingMode::Plain,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let runner = ActiveKnowledgeModel::load_candidate(&args.bundle)?;
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let runner = ActiveKnowledgeModel::load_candidate_gpu(&args.bundle)?;
    let exam: ToolContractExamSet =
        serde_json::from_reader(BufReader::new(File::open(&args.exam)?))?;
    exam.verify(runner.contracts())?;
    let artifact = runner.artifact();
    let checkpoint_step = u64::try_from(artifact.checkpoint_step)
        .map_err(|_| invalid_data("candidate checkpoint step does not fit u64"))?;
    if exam.contract_hash != artifact.contract_hash
        || (exam.tier != grug_data::ExamTier::Sentinel
            && (exam.checkpoint.step != checkpoint_step
                || exam.checkpoint.artifact_hash != artifact.checkpoint_model_hash))
    {
        return Err(invalid_data(
            "tool exam is not bound to this exact candidate checkpoint and contract",
        ));
    }

    let mode: DecodingMode = args.decoding.into();
    let mut predictions = BTreeMap::new();
    let mut decode_reports: BTreeMap<String, DecodeReport> = BTreeMap::new();
    for (index, case) in exam.cases.iter().enumerate() {
        let (raw, _, _, report) = runner.generate_turn_with_system_mode(
            &case.tools,
            &case.conversation,
            None,
            args.max_new_tokens,
            mode,
        )?;
        predictions.insert(case.id.clone(), raw);
        decode_reports.insert(case.id.clone(), report);
        if index == 0 || (index + 1) % 100 == 0 || index + 1 == exam.cases.len() {
            eprintln!(
                "evaluated {}/{} all-tool cases",
                index + 1,
                exam.cases.len()
            );
        }
    }
    let mut report = score_tool_contract_exam(&exam, &predictions, runner.contracts());
    // The frozen sentinel deliberately has no candidate identity. Evaluation
    // binds its immutable cases to the checkpoint that actually executed them.
    report.checkpoint.step = checkpoint_step;
    report.checkpoint.artifact_hash = artifact.checkpoint_model_hash.clone();
    let observations = report.observations();
    let evidence = ToolContractExamEvidence {
        schema_version: TOOL_CONTRACT_EXAM_SCHEMA_VERSION,
        evaluation_mode: ToolContractEvaluationMode::CheckpointExecuted,
        report,
        observations,
    };
    if let Some(path) = &args.predictions_out {
        write_json_pretty_atomic(path, &predictions)?;
    }
    if let Some(path) = &args.decode_reports_out {
        write_json_pretty_atomic(path, &decode_reports)?;
    }
    if let Some(path) = &args.observations_out {
        write_jsonl_atomic(path, &evidence.observations)?;
    }
    write_json_pretty_atomic(&args.evidence_out, &evidence)?;

    let samples: u64 = evidence
        .observations
        .iter()
        .map(|observation| observation.sample_count)
        .sum();
    let passed: u64 = evidence
        .observations
        .iter()
        .map(|observation| observation.pass_count)
        .sum();
    let catastrophic: u64 = evidence
        .observations
        .iter()
        .map(|observation| observation.catastrophic_count)
        .sum();
    println!(
        "tools={} samples={} pass={:.3} catastrophic={:.3} evidence={}",
        evidence.observations.len(),
        samples,
        passed as f64 / samples.max(1) as f64,
        catastrophic as f64 / samples.max(1) as f64,
        args.evidence_out.display()
    );
    Ok(())
}

fn write_json_pretty_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!("{value}.tmp"))
        .unwrap_or_else(|| "tmp".into());
    let temporary = path.with_extension(extension);
    let mut writer = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn write_jsonl_atomic<T: Serialize>(path: &Path, values: &[T]) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = path.with_extension("jsonl.tmp");
    let mut writer = File::create(&temporary)?;
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
