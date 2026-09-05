//! Promote a checkpoint candidate only after checkpoint-bound quarantined
//! exams cover every canonical knowledge workflow and captured runtime tool.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use grug_data::{
    ExamTier, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeEvaluationMode, KnowledgeExamEvidence,
    TOOL_CONTRACT_EXAM_SCHEMA_VERSION, ToolContractEvaluationMode, ToolContractExamEvidence,
    ToolContractSnapshot, knowledge_capability_specs,
};
use grug_train::{
    CertifiedKnowledgeExam, CpuKnowledgeModel, KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION,
    KnowledgeModelArtifact, KnowledgeModelCertification,
};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "certify_knowledge_model")]
struct Args {
    /// Training bundle containing candidate-model.json.
    #[arg(long)]
    bundle: PathBuf,

    /// One scorer-emitted KnowledgeExamEvidence JSON file per capability.
    #[arg(long, required = true)]
    evidence: Vec<PathBuf>,

    /// One checkpoint-executed all-tool evidence bundle covering the exact
    /// captured runtime contract snapshot.
    #[arg(long)]
    tool_evidence: PathBuf,

    /// Release-gate tier. Ongoing frontier probes remain separate steering data.
    #[arg(long, default_value = "sentinel")]
    tier: String,

    #[arg(long, default_value_t = 1)]
    minimum_difficulty: u8,

    #[arg(long, default_value_t = 3)]
    minimum_tool_difficulty: u8,

    #[arg(long, default_value_t = 100)]
    minimum_samples_per_family: u64,

    #[arg(long, default_value_t = 24)]
    minimum_tool_samples_per_family: u64,

    #[arg(long, default_value_t = 500_000_000)]
    minimum_foundation_tokens_seen: u64,

    #[arg(long, default_value_t = 0.85)]
    minimum_pass_rate: f64,

    #[arg(long, default_value_t = 0.01)]
    maximum_catastrophic_rate: f64,
}

#[derive(Serialize)]
struct CertificationEvidenceBundle {
    schema_version: u32,
    certification: KnowledgeModelCertification,
    knowledge_evidence: Vec<KnowledgeExamEvidence>,
    tool_evidence: ToolContractExamEvidence,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_thresholds(&args)?;
    if args.bundle.join("active-model.json").exists() {
        return Err(invalid_data(
            "knowledge bundle already has an active model; activation is append-only",
        ));
    }
    let tier = parse_tier(&args.tier)?;
    // Use the same verified loader as evaluation and serving. Promotion then
    // proves structural runnability before considering behavioural evidence.
    let runner = CpuKnowledgeModel::load_candidate(&args.bundle)?;
    let mut artifact = runner.artifact().clone();
    let contracts = runner.contracts().clone();
    drop(runner);

    let mut evidence = Vec::with_capacity(args.evidence.len());
    for path in &args.evidence {
        let item: KnowledgeExamEvidence =
            serde_json::from_reader(BufReader::new(File::open(path)?)).map_err(|error| {
                invalid_data(format!("parse exam evidence {}: {error}", path.display()))
            })?;
        validate_evidence(&item, &artifact, tier)?;
        evidence.push(item);
    }
    let tool_evidence: ToolContractExamEvidence = serde_json::from_reader(BufReader::new(
        File::open(&args.tool_evidence)?,
    ))
    .map_err(|error| {
        invalid_data(format!(
            "parse tool exam evidence {}: {error}",
            args.tool_evidence.display()
        ))
    })?;
    validate_tool_evidence(&tool_evidence, &artifact, &contracts, tier)?;

    let required_knowledge_families: BTreeSet<String> = knowledge_capability_specs()
        .into_iter()
        .map(|capability| capability.id)
        .collect();
    let required_tool_families: BTreeSet<String> = contracts
        .tools
        .iter()
        .map(|tool| format!("tool_contract.{}", tool.name))
        .collect();
    let required_families: BTreeSet<String> = required_knowledge_families
        .union(&required_tool_families)
        .cloned()
        .collect();
    let mut selected = BTreeMap::new();
    for item in &evidence {
        let family = item.report.family.clone();
        if !required_knowledge_families.contains(&family) {
            return Err(invalid_data(format!(
                "exam evidence contains non-canonical family {family:?}"
            )));
        }
        if selected.insert(family.clone(), item).is_some() {
            return Err(invalid_data(format!(
                "multiple release-gate evidence files were supplied for {family:?}"
            )));
        }
    }
    let observed: BTreeSet<_> = selected.keys().cloned().collect();
    if observed != required_knowledge_families {
        let missing: Vec<_> = required_knowledge_families
            .difference(&observed)
            .cloned()
            .collect();
        return Err(invalid_data(format!(
            "certification is missing capability evidence: {missing:?}"
        )));
    }

    let mut exams: Vec<_> = selected
        .into_values()
        .map(|item| CertifiedKnowledgeExam {
            family: item.report.family.clone(),
            tier: item.observation.tier,
            difficulty: item.observation.difficulty,
            samples: item.observation.sample_count,
            pass_count: item.observation.pass_count,
            catastrophic_count: item.observation.catastrophic_count,
            exam_fingerprint: item.report.exam_fingerprint.clone(),
        })
        .collect();
    exams.extend(
        tool_evidence
            .report
            .families
            .iter()
            .map(|family| CertifiedKnowledgeExam {
                family: family.family.clone(),
                tier: tool_evidence.report.tier,
                difficulty: tool_evidence.report.difficulty,
                samples: family.cases.len() as u64,
                pass_count: family.pass_count,
                catastrophic_count: family.catastrophic_count,
                exam_fingerprint: format!(
                    "{}:{}",
                    tool_evidence.report.exam_fingerprint, family.family
                ),
            }),
    );
    let mut certification = KnowledgeModelCertification {
        schema_version: KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
        contract_hash: artifact.contract_hash.clone(),
        tier,
        minimum_difficulty: args.minimum_difficulty,
        minimum_tool_difficulty: args.minimum_tool_difficulty,
        minimum_samples_per_family: args.minimum_samples_per_family,
        minimum_tool_samples_per_family: args.minimum_tool_samples_per_family,
        minimum_foundation_tokens_seen: args.minimum_foundation_tokens_seen,
        minimum_pass_rate: args.minimum_pass_rate,
        maximum_catastrophic_rate: args.maximum_catastrophic_rate,
        required_families,
        required_tool_families,
        exams,
        evidence_hash: String::new(),
    };
    certification.evidence_hash = certification
        .compute_evidence_hash()
        .map_err(invalid_data)?;
    certification
        .verify_contract_coverage(&contracts)
        .map_err(invalid_data)?;
    artifact.certification = Some(certification.clone());
    // The resumable candidate retains Adam state. The inference-only active
    // manifest does not need to hash it on every daemon startup.
    artifact.checkpoint_optimizer_hash = None;
    artifact.verify().map_err(invalid_data)?;

    let evidence_bundle = CertificationEvidenceBundle {
        schema_version: KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION,
        certification,
        knowledge_evidence: evidence,
        tool_evidence,
    };
    write_or_verify_json_pretty(
        &args.bundle.join("certification-evidence.json"),
        &evidence_bundle,
    )?;
    // Write active last: readers either see the previous certified artifact or
    // this complete evidence-backed one, never a half-promoted candidate.
    write_json_pretty_atomic(&args.bundle.join("active-model.json"), &artifact)?;
    println!(
        "certified checkpoint step {} model={} contract={} -> {}",
        artifact.checkpoint_step,
        artifact.checkpoint_model_hash,
        artifact.contract_hash,
        args.bundle.join("active-model.json").display()
    );
    Ok(())
}

fn validate_thresholds(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.minimum_samples_per_family == 0
        || args.minimum_tool_samples_per_family == 0
        || args.minimum_foundation_tokens_seen == 0
        || !(1..=3).contains(&args.minimum_tool_difficulty)
        || !(0.0..=1.0).contains(&args.minimum_pass_rate)
        || !(0.0..=1.0).contains(&args.maximum_catastrophic_rate)
    {
        return Err(invalid_data(
            "sample count must be positive and rate thresholds must be in [0,1]",
        ));
    }
    Ok(())
}

fn validate_evidence(
    evidence: &KnowledgeExamEvidence,
    artifact: &KnowledgeModelArtifact,
    required_tier: ExamTier,
) -> Result<(), Box<dyn Error>> {
    let report = &evidence.report;
    let observation = &evidence.observation;
    let checkpoint_step = u64::try_from(artifact.checkpoint_step)
        .map_err(|_| invalid_data("artifact checkpoint step does not fit the exam schema"))?;
    if evidence.schema_version != KNOWLEDGE_WORKFLOW_SCHEMA_VERSION {
        return Err(invalid_data(format!(
            "exam evidence schema {} is unsupported",
            evidence.schema_version
        )));
    }
    if evidence.evaluation_mode != KnowledgeEvaluationMode::ClosedLoop {
        return Err(invalid_data(format!(
            "evidence for {:?} used {:?}; certification requires checkpoint-executed closed_loop evaluation",
            report.family, evidence.evaluation_mode
        )));
    }
    if report.contract_hash != artifact.contract_hash
        || observation.checkpoint.step != checkpoint_step
        || observation.checkpoint.artifact_hash != artifact.checkpoint_model_hash
    {
        return Err(invalid_data(format!(
            "evidence for {:?} is bound to a different contract/checkpoint",
            report.family
        )));
    }
    if observation.tier != required_tier
        || observation.family != report.family
        || observation.difficulty != report.difficulty
        || observation.sample_count != report.cases.len() as u64
        || observation.pass_count != report.pass_count
        || observation.precursor_count != report.precursor_count
        || observation.catastrophic_count != report.catastrophic_count
    {
        return Err(invalid_data(format!(
            "report/observation fields disagree for {:?}",
            report.family
        )));
    }
    if report.exam_fingerprint.is_empty()
        || report.pass_count > observation.sample_count
        || report.catastrophic_count > observation.sample_count
    {
        return Err(invalid_data(format!(
            "exam counts/fingerprint are invalid for {:?}",
            report.family
        )));
    }
    Ok(())
}

fn validate_tool_evidence(
    evidence: &ToolContractExamEvidence,
    artifact: &KnowledgeModelArtifact,
    contracts: &ToolContractSnapshot,
    required_tier: ExamTier,
) -> Result<(), Box<dyn Error>> {
    let report = &evidence.report;
    let checkpoint_step = u64::try_from(artifact.checkpoint_step)
        .map_err(|_| invalid_data("artifact checkpoint step does not fit the exam schema"))?;
    if evidence.schema_version != TOOL_CONTRACT_EXAM_SCHEMA_VERSION
        || evidence.evaluation_mode != ToolContractEvaluationMode::CheckpointExecuted
    {
        return Err(invalid_data(
            "tool evidence must be produced by checkpoint-executed evaluation",
        ));
    }
    if report.contract_hash != contracts.manifest_hash
        || report.contract_hash != artifact.contract_hash
        || report.checkpoint.step != checkpoint_step
        || report.checkpoint.artifact_hash != artifact.checkpoint_model_hash
        || report.tier != required_tier
        || report.offered_tool_count != contracts.tools.len().min(4)
        || report.exam_fingerprint.is_empty()
        || !report.exam_fingerprint.starts_with("blake3:")
    {
        return Err(invalid_data(
            "tool evidence is bound to a different tier, contract, checkpoint, or exam",
        ));
    }
    let expected_families: BTreeSet<String> = contracts
        .tools
        .iter()
        .map(|tool| format!("tool_contract.{}", tool.name))
        .collect();
    let observed_families: BTreeSet<String> = report
        .families
        .iter()
        .map(|family| family.family.clone())
        .collect();
    if report.families.len() != expected_families.len()
        || observed_families != expected_families
        || evidence.observations != report.observations()
    {
        return Err(invalid_data(
            "tool evidence does not contain one internally consistent result per captured tool",
        ));
    }
    for family in &report.families {
        if family.family != format!("tool_contract.{}", family.tool)
            || family.cases.is_empty()
            || family.cases.iter().any(|case| case.tool != family.tool)
        {
            return Err(invalid_data(format!(
                "tool evidence family {:?} has inconsistent case identity",
                family.family
            )));
        }
        let pass_count = family.cases.iter().filter(|case| case.passed).count() as u64;
        let precursor_count = family.cases.iter().filter(|case| case.precursor).count() as u64;
        let catastrophic_count =
            family.cases.iter().filter(|case| case.catastrophic).count() as u64;
        let invalid_partition = family.cases.iter().any(|case| {
            u8::from(case.passed) + u8::from(case.precursor) + u8::from(case.catastrophic) != 1
                || (case.passed
                    && !(case.tool_correct
                        && case.tool_offered
                        && case.schema_valid
                        && case.arguments_correct))
                || (case.precursor
                    && !(case.tool_correct
                        && case.tool_offered
                        && case.schema_valid
                        && !case.arguments_correct))
        });
        if family.pass_count != pass_count
            || family.precursor_count != precursor_count
            || family.catastrophic_count != catastrophic_count
            || invalid_partition
        {
            return Err(invalid_data(format!(
                "tool evidence counts or classifications disagree for {:?}",
                family.family
            )));
        }
    }
    Ok(())
}

fn parse_tier(value: &str) -> Result<ExamTier, Box<dyn Error>> {
    match value {
        "sentinel" => Ok(ExamTier::Sentinel),
        "rolling-probe" => Ok(ExamTier::RollingProbe),
        "frontier" => Ok(ExamTier::Frontier),
        "audit" => Ok(ExamTier::Audit),
        _ => Err(invalid_data(format!(
            "unknown --tier {value:?}; expected sentinel, rolling-probe, frontier, or audit"
        ))),
    }
}

fn write_json_pretty_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(invalid_data(format!(
            "refusing to overwrite append-only artifact {}",
            path.display()
        )));
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.sync_all()?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn write_or_verify_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return write_json_pretty_atomic(path, value);
    }
    let actual: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let expected = serde_json::to_value(value)?;
    if actual != expected {
        return Err(invalid_data(format!(
            "existing append-only artifact {} differs from recomputed certification evidence",
            path.display()
        )));
    }
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
