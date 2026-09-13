//! Run a quarantined knowledge-workflow exam against a candidate checkpoint.
//!
//! The default is a closed loop: only an exact, schema-valid tool action earns
//! its fixture result and advances the episode. `--teacher-forced` is a
//! diagnostic mode that isolates failing stages with gold prior history, but
//! its evidence is explicitly ineligible for certification.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use grug_data::adaptive_exam::ADAPTIVE_EXAM_SCHEMA_VERSION;
use grug_data::constrain::DecodeReport;
use grug_data::{
    CheckpointRef, ExamTier, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeAction, KnowledgeEpisode,
    KnowledgeEvaluationMode, KnowledgeExamEvidence, KnowledgeExamSet, KnowledgeToolBindings,
    closed_loop_step_accepts, knowledge_exam_generator_supported, score_knowledge_exam,
};
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
use grug_train::{CpuBackend as ActiveBackend, CpuKnowledgeModel as ActiveKnowledgeModel};
use grug_train::{DecodingMode, KnowledgeModel};
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
use grug_train::{GpuBackend as ActiveBackend, GpuKnowledgeModel as ActiveKnowledgeModel};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "eval_knowledge_exam")]
struct Args {
    /// Training bundle containing candidate-model.json and its immutable files.
    #[arg(long)]
    bundle: PathBuf,

    /// Quarantined KnowledgeExamSet JSON generated for this exact checkpoint.
    #[arg(long)]
    exam: PathBuf,

    /// Checkpoint-bound KnowledgeExamEvidence written for certification.
    #[arg(long)]
    evidence_out: PathBuf,

    /// Optional raw per-episode outputs for forensic review.
    #[arg(long)]
    predictions_out: Option<PathBuf>,

    /// Optional per-episode decode reports (closed-loop mode only): which
    /// fields were value-copy constrained and whether the emitted value
    /// landed on a context candidate (ADR 104 D1).
    #[arg(long)]
    decode_reports_out: Option<PathBuf>,

    #[arg(long, default_value_t = 384)]
    max_new_tokens: usize,

    /// Diagnose stages independently with gold prior actions/results. Evidence
    /// from this mode is deliberately not eligible for model certification.
    #[arg(long)]
    teacher_forced: bool,

    /// `constrained` (default) decodes under the runtime `<tool_call>`
    /// grammar plus byte-level value-copy constraints (ADR 104 D1). `plain`
    /// reproduces the pre-D1 unconstrained greedy decoder, for comparing
    /// against historical certification numbers only.
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
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let runner = ActiveKnowledgeModel::load_candidate(&args.bundle)?;
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let runner = ActiveKnowledgeModel::load_candidate_gpu(&args.bundle)?;
    let artifact = runner.artifact();
    let contracts = runner.contracts();
    let exam: KnowledgeExamSet = serde_json::from_reader(BufReader::new(File::open(&args.exam)?))?;
    validate_exam(&exam, artifact, contracts.manifest_hash.as_str())?;

    let mode: DecodingMode = args.decoding.into();
    let bindings = KnowledgeToolBindings::default();
    let mut predictions = BTreeMap::new();
    let mut decode_reports: BTreeMap<String, Vec<DecodeReport>> = BTreeMap::new();
    for (index, episode) in exam.episodes.iter().enumerate() {
        let outputs = if args.teacher_forced {
            let records = episode.to_sft_records(contracts, &bindings)?;
            runner.generate_records(&records, args.max_new_tokens)?
        } else {
            let (outputs, reports) =
                generate_closed_loop(&runner, episode, &bindings, args.max_new_tokens, mode)?;
            decode_reports.insert(episode.id.clone(), reports);
            outputs
        };
        predictions.insert(episode.id.clone(), outputs);
        if index == 0 || (index + 1) % 10 == 0 || index + 1 == exam.episodes.len() {
            eprintln!(
                "evaluated {}/{} episodes for {}",
                index + 1,
                exam.episodes.len(),
                exam.plan.family
            );
        }
    }

    let report = score_knowledge_exam(&exam, &predictions, contracts, &bindings);
    let checkpoint = CheckpointRef {
        step: u64::try_from(artifact.checkpoint_step)
            .map_err(|_| invalid_data("checkpoint step does not fit the exam schema"))?,
        artifact_hash: artifact.checkpoint_model_hash.clone(),
    };
    let evidence = KnowledgeExamEvidence {
        schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
        evaluation_mode: if args.teacher_forced {
            KnowledgeEvaluationMode::TeacherForced
        } else {
            KnowledgeEvaluationMode::ClosedLoop
        },
        observation: report.observation(checkpoint, exam.plan.tier),
        report,
    };
    if let Some(path) = &args.predictions_out {
        write_json_pretty_atomic(path, &predictions)?;
    }
    if let Some(path) = &args.decode_reports_out {
        write_json_pretty_atomic(path, &decode_reports)?;
    }
    write_json_pretty_atomic(&args.evidence_out, &evidence)?;

    let samples = evidence.observation.sample_count.max(1) as f64;
    println!(
        "family={} difficulty={} pass={:.3} precursor={:.3} catastrophic={:.3} evidence={}",
        evidence.report.family,
        evidence.report.difficulty,
        evidence.observation.pass_count as f64 / samples,
        evidence.observation.precursor_count as f64 / samples,
        evidence.observation.catastrophic_count as f64 / samples,
        args.evidence_out.display()
    );
    Ok(())
}

fn generate_closed_loop(
    runner: &KnowledgeModel<ActiveBackend>,
    episode: &KnowledgeEpisode,
    bindings: &KnowledgeToolBindings,
    max_new_tokens: usize,
    mode: DecodingMode,
) -> Result<(Vec<String>, Vec<DecodeReport>), Box<dyn Error>> {
    let records = episode.to_sft_records(runner.contracts(), bindings)?;
    let mut conversation = vec![serde_json::json!({
        "role": "user",
        "content": episode.steps.first().map(|step| step.input.as_str()).unwrap_or("")
    })];
    let mut outputs = Vec::with_capacity(episode.steps.len());
    let mut reports = Vec::with_capacity(episode.steps.len());
    for (index, step) in episode.steps.iter().enumerate() {
        let Some(record) = records.get(index) else {
            break;
        };
        let (raw, _, _, report) = runner.generate_turn_with_system_mode(
            &record.tools,
            &conversation,
            None,
            max_new_tokens,
            mode,
        )?;
        outputs.push(raw.clone());
        reports.push(report);
        let Ok(action) = KnowledgeAction::parse_runtime(&raw) else {
            break;
        };
        // A fixed synthetic fixture can only return the declared result for
        // a call the grader would accept as this step. Use the grader's own
        // predicate rather than byte-exact equality, or a semantically
        // correct-but-differently-worded call gets laundered into a
        // `missing_prediction` catastrophic failure for every step after it
        // even though the grader would have passed it.
        if !closed_loop_step_accepts(
            step,
            &action,
            &episode.fixture,
            runner.contracts(),
            bindings,
        ) {
            break;
        }
        match action {
            KnowledgeAction::ToolCall { tool, arguments } => {
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {"name": tool, "arguments": arguments}
                    }]
                }));
                conversation.push(serde_json::json!({
                    "role": "tool",
                    "content": step.mock_result.as_deref().unwrap_or("")
                }));
            }
            KnowledgeAction::Answer { .. } | KnowledgeAction::Gap { .. } => break,
        }
    }
    Ok((outputs, reports))
}

fn validate_exam(
    exam: &KnowledgeExamSet,
    artifact: &grug_train::KnowledgeModelArtifact,
    contract_hash: &str,
) -> Result<(), Box<dyn Error>> {
    let checkpoint_step = u64::try_from(artifact.checkpoint_step)
        .map_err(|_| invalid_data("checkpoint step does not fit the exam schema"))?;
    if exam.schema_version != KNOWLEDGE_WORKFLOW_SCHEMA_VERSION
        || exam.plan.schema_version != ADAPTIVE_EXAM_SCHEMA_VERSION
        || !knowledge_exam_generator_supported(&exam.generator_version)
        || exam.plan.generator_version.trim().is_empty()
    {
        return Err(invalid_data(
            "exam schema or generator version is unsupported",
        ));
    }
    if exam.contract_hash != contract_hash || exam.contract_hash != artifact.contract_hash {
        return Err(invalid_data(format!(
            "exam contract {} does not match candidate contract {}",
            exam.contract_hash, artifact.contract_hash
        )));
    }
    // Sentinels are immutable and reusable across checkpoints; their plan
    // fingerprint deliberately excludes checkpoint identity. Moving tiers are
    // checkpoint-specific and must match the candidate exactly.
    if exam.plan.tier != ExamTier::Sentinel
        && (exam.plan.checkpoint.step != checkpoint_step
            || exam.plan.checkpoint.artifact_hash != artifact.checkpoint_model_hash)
    {
        return Err(invalid_data(
            "exam was not generated for this exact candidate checkpoint",
        ));
    }
    if exam.exam_fingerprint != exam.plan.fingerprint()
        || exam.plan.sample_count != exam.episodes.len() as u64
        || exam.episodes.iter().any(|episode| {
            episode.contract_hash != exam.contract_hash
                || episode.family != exam.plan.family
                || episode.difficulty != exam.plan.difficulty
        })
    {
        return Err(invalid_data(
            "exam fingerprint, sample count, or episode metadata is inconsistent",
        ));
    }
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

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
