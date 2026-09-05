//! Run the complete held-out native OCR audit and text/tool retention exam.
//!
//! This binary has no provider client.  For promotion evidence, launch it in
//! an OS network namespace with only loopback and pass
//! `--require-network-denied`; it verifies `/proc`/`sysfs` state before loading
//! a model and repeats a deterministic audit slice byte-for-byte.

use clap::Parser;
use grug_data::{
    KnowledgeAction, KnowledgeSftRecord, VisionCorpusDomain, VisionTextManifest,
    VisionTextPrediction, load_manifest_rgb, score_vision_text, vision_perceptual_hash,
    vision_pixel_hash,
};
use grug_train::{
    MULTIMODAL_EVIDENCE_SCHEMA_VERSION, MultimodalAuditEvidence,
    MultimodalNetworkDeniedReplayEvidence, MultimodalTextRetentionEvidence,
    fingerprint_artifact_file, fingerprint_knowledge_records, score_knowledge_outputs,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
type CandidateModel = grug_train::CpuMultimodalModel;
#[cfg(any(feature = "gpu", feature = "cuda"))]
type CandidateModel = grug_train::GpuMultimodalModel;

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
type BaseTextModel = grug_train::CpuKnowledgeModel;
#[cfg(any(feature = "gpu", feature = "cuda"))]
type BaseTextModel = grug_train::GpuKnowledgeModel;

#[derive(Parser, Debug)]
#[command(name = "eval_native_ocr")]
struct Args {
    #[arg(long)]
    candidate_bundle: PathBuf,
    #[arg(long)]
    base_text_bundle: PathBuf,
    #[arg(long)]
    audit_manifest: PathBuf,
    #[arg(long, required = true)]
    retention_eval_jsonl: Vec<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 16_000_000)]
    maximum_image_pixels: u64,
    #[arg(long, default_value_t = 30)]
    network_replay_cases: usize,
    #[arg(long)]
    require_network_denied: bool,
}

#[derive(Debug, Serialize)]
struct CompletionReport {
    kind: &'static str,
    candidate_only: bool,
    active_model_written: bool,
    checkpoint_step: usize,
    checkpoint_model_hash: String,
    audit_cases: usize,
    exact_rate: f64,
    character_error_rate: f64,
    cross_semantic_collapse_rate: f64,
    metamorphic_consistency: f64,
    text_action_exact_before: f64,
    text_action_exact_after: f64,
    network_denied_verified: bool,
    audit_evidence: String,
    retention_evidence: String,
    network_replay_evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OutputRow {
    id: String,
    text: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_new_tokens == 0 || args.maximum_image_pixels == 0 || args.network_replay_cases == 0
    {
        return Err(invalid_data("positive limits are required"));
    }
    if args.output_dir.exists() && args.output_dir.read_dir()?.next().is_some() {
        return Err(invalid_data(format!(
            "output directory {} is not empty; evaluation evidence is append-only",
            args.output_dir.display()
        )));
    }
    std::fs::create_dir_all(&args.output_dir)?;

    let network = inspect_network_denial()?;
    if args.require_network_denied && !network.denied {
        return Err(invalid_data(format!(
            "network-denied audit requested, but visible interfaces/routes are {:?}/{}",
            network.interfaces, network.route_entries
        )));
    }

    let candidate = load_candidate(&args.candidate_bundle)?
        .with_maximum_image_pixels(args.maximum_image_pixels)?;
    let artifact = candidate.artifact().clone();
    let audit_hash = fingerprint_artifact_file(&args.audit_manifest)?;
    if audit_hash != artifact.audit_corpus.manifest_hash {
        return Err(invalid_data(format!(
            "audit manifest hash {audit_hash} differs from candidate {}",
            artifact.audit_corpus.manifest_hash
        )));
    }
    let audit_cases = read_audit_manifest(&args.audit_manifest)?;
    if audit_cases.len() != artifact.audit_corpus.cases || audit_cases.len() < 1_050 {
        return Err(invalid_data(format!(
            "audit contains {} cases; candidate binds {} and promotion requires at least 1050",
            audit_cases.len(),
            artifact.audit_corpus.cases
        )));
    }

    let predictions_path = args.output_dir.join("audit-predictions.jsonl");
    let mut predictions_writer = BufWriter::new(new_file(&predictions_path)?);
    let mut predictions = Vec::with_capacity(audit_cases.len());
    for case in &audit_cases {
        let prediction = predict_case(
            &candidate,
            &args.audit_manifest,
            case,
            args.max_new_tokens,
            args.maximum_image_pixels,
        )?;
        serde_json::to_writer(&mut predictions_writer, &prediction)?;
        predictions_writer.write_all(b"\n")?;
        predictions.push(prediction);
    }
    predictions_writer.flush()?;
    drop(predictions_writer);
    let predictions_hash = fingerprint_artifact_file(&predictions_path)?;
    let score = score_vision_text(&audit_cases, &predictions)?;
    let audit_evidence = MultimodalAuditEvidence {
        schema_version: MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
        audit_manifest_hash: audit_hash.clone(),
        predictions_hash,
        score,
    };
    let audit_evidence_path = args.output_dir.join("multimodal-audit-evidence.json");
    write_json_atomic(&audit_evidence_path, &audit_evidence)?;

    let retention_records = read_retention_records(&args.retention_eval_jsonl)?;
    let retention_hash = fingerprint_knowledge_records(&retention_records)?;
    if retention_hash != artifact.retention_eval_corpus_hash
        || retention_records.len() != artifact.retention_eval_records
    {
        return Err(invalid_data(
            "retention exam differs from the quarantined corpus bound to the candidate",
        ));
    }
    validate_retention(&retention_records, &artifact.contract_hash)?;
    let base_manifest = args.base_text_bundle.join("active-model.json");
    let base_artifact_hash = fingerprint_artifact_file(&base_manifest)?;
    if base_artifact_hash != artifact.base_text_artifact_hash {
        return Err(invalid_data(
            "base text artifact differs from the candidate's certified warm start",
        ));
    }
    let base = load_base(&args.base_text_bundle)?;
    if base.artifact().checkpoint_model_hash != artifact.base_text_checkpoint_hash {
        return Err(invalid_data(
            "base text checkpoint differs from the candidate lineage",
        ));
    }
    let before_outputs = base.generate_records(&retention_records, args.max_new_tokens)?;
    let after_outputs = candidate.generate_records(&retention_records, args.max_new_tokens)?;
    let before_path = args.output_dir.join("retention-before-outputs.jsonl");
    let after_path = args.output_dir.join("retention-after-outputs.jsonl");
    write_output_rows(
        &before_path,
        retention_records
            .iter()
            .zip(&before_outputs)
            .map(|(record, text)| OutputRow {
                id: record.id.clone(),
                text: text.clone(),
            }),
    )?;
    write_output_rows(
        &after_path,
        retention_records
            .iter()
            .zip(&after_outputs)
            .map(|(record, text)| OutputRow {
                id: record.id.clone(),
                text: text.clone(),
            }),
    )?;
    let retention_evidence = MultimodalTextRetentionEvidence {
        schema_version: MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
        base_text_artifact_hash: base_artifact_hash,
        base_text_checkpoint_hash: artifact.base_text_checkpoint_hash.clone(),
        retention_corpus_hash: retention_hash,
        records: retention_records.len(),
        before_outputs_hash: fingerprint_artifact_file(&before_path)?,
        after_outputs_hash: fingerprint_artifact_file(&after_path)?,
        before: score_knowledge_outputs(&retention_records, &before_outputs),
        after: score_knowledge_outputs(&retention_records, &after_outputs),
    };
    let retention_evidence_path = args
        .output_dir
        .join("multimodal-text-retention-evidence.json");
    write_json_atomic(&retention_evidence_path, &retention_evidence)?;

    let replay_indices = evenly_spaced_indices(
        audit_cases.len(),
        args.network_replay_cases.min(audit_cases.len()),
    );
    let first = replay_outputs(
        &candidate,
        &args.audit_manifest,
        &audit_cases,
        &replay_indices,
        args.max_new_tokens,
        args.maximum_image_pixels,
    )?;
    let second = replay_outputs(
        &candidate,
        &args.audit_manifest,
        &audit_cases,
        &replay_indices,
        args.max_new_tokens,
        args.maximum_image_pixels,
    )?;
    let first_path = args.output_dir.join("network-replay-first.jsonl");
    let second_path = args.output_dir.join("network-replay-second.jsonl");
    write_output_rows(&first_path, first.iter().cloned())?;
    write_output_rows(&second_path, second.iter().cloned())?;
    let first_hash = fingerprint_artifact_file(&first_path)?;
    let second_hash = fingerprint_artifact_file(&second_path)?;
    let outputs_identical = first == second && first_hash == second_hash;
    let replay_evidence = MultimodalNetworkDeniedReplayEvidence {
        schema_version: MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
        audit_manifest_hash: audit_hash,
        network_denied_verified: network.denied,
        verification_method: network.method,
        network_namespace: network.namespace,
        visible_interfaces: network.interfaces,
        kernel_route_entries: network.route_entries,
        replay_cases: replay_indices.len(),
        first_outputs_hash: first_hash.clone(),
        second_outputs_hash: second_hash,
        outputs_identical,
    };
    let replay_evidence_path = args
        .output_dir
        .join("multimodal-network-denied-replay.json");
    write_json_atomic(&replay_evidence_path, &replay_evidence)?;

    let report = CompletionReport {
        kind: "native_ocr_frozen_audit",
        candidate_only: true,
        active_model_written: false,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash,
        audit_cases: audit_cases.len(),
        exact_rate: audit_evidence.score.overall.exact_rate,
        character_error_rate: audit_evidence.score.overall.character_error_rate,
        cross_semantic_collapse_rate: audit_evidence.score.cross_semantic_collapse_rate,
        metamorphic_consistency: audit_evidence.score.metamorphic_consistency,
        text_action_exact_before: retention_evidence.before.action_exact_rate,
        text_action_exact_after: retention_evidence.after.action_exact_rate,
        network_denied_verified: replay_evidence.network_denied_verified,
        audit_evidence: audit_evidence_path.display().to_string(),
        retention_evidence: retention_evidence_path.display().to_string(),
        network_replay_evidence: replay_evidence_path.display().to_string(),
    };
    write_json_atomic(&args.output_dir.join("completion.json"), &report)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn load_candidate(path: &Path) -> Result<CandidateModel, grug_train::MultimodalRuntimeError> {
    CandidateModel::load_candidate(path)
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn load_candidate(path: &Path) -> Result<CandidateModel, grug_train::MultimodalRuntimeError> {
    CandidateModel::load_candidate_gpu(path)
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn load_base(path: &Path) -> Result<BaseTextModel, grug_train::KnowledgeRuntimeError> {
    BaseTextModel::load_active(path)
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn load_base(path: &Path) -> Result<BaseTextModel, grug_train::KnowledgeRuntimeError> {
    BaseTextModel::load_active_gpu(path)
}

fn read_audit_manifest(path: &Path) -> Result<Vec<VisionTextManifest>, Box<dyn Error>> {
    let mut cases = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: VisionTextManifest = serde_json::from_str(&line)
            .map_err(|error| invalid_data(format!("{}:{}: {error}", path.display(), index + 1)))?;
        if case.domain != VisionCorpusDomain::Audit || !ids.insert(case.case_id.clone()) {
            return Err(invalid_data(format!(
                "case {} is not a unique audit case",
                case.case_id
            )));
        }
        cases.push(case);
    }
    Ok(cases)
}

fn predict_case(
    model: &CandidateModel,
    manifest_path: &Path,
    case: &VisionTextManifest,
    max_new_tokens: usize,
    maximum_image_pixels: u64,
) -> Result<VisionTextPrediction, Box<dyn Error>> {
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let (width, height, rgb) = load_manifest_rgb(root, case, maximum_image_pixels)?;
    if width != case.width
        || height != case.height
        || case.channels != 3
        || vision_pixel_hash(&rgb) != case.pixel_hash_fnv1a64
        || case.pixel_hash_blake3.as_deref()
            != Some(format!("blake3:{}", blake3::hash(&rgb).to_hex()).as_str())
        || vision_perceptual_hash(&rgb, width, height) != case.perceptual_hash
    {
        return Err(invalid_data(format!(
            "case {} pixels do not match the frozen manifest",
            case.case_id
        )));
    }
    Ok(VisionTextPrediction {
        case_id: case.case_id.clone(),
        text: model.generate_ocr_rgb(&rgb, width, height, max_new_tokens)?,
        confidence: None,
    })
}

fn replay_outputs(
    model: &CandidateModel,
    manifest_path: &Path,
    cases: &[VisionTextManifest],
    indices: &[usize],
    max_new_tokens: usize,
    maximum_image_pixels: u64,
) -> Result<Vec<OutputRow>, Box<dyn Error>> {
    indices
        .iter()
        .map(|index| {
            predict_case(
                model,
                manifest_path,
                &cases[*index],
                max_new_tokens,
                maximum_image_pixels,
            )
            .map(|prediction| OutputRow {
                id: prediction.case_id,
                text: prediction.text,
            })
        })
        .collect()
}

fn read_retention_records(paths: &[PathBuf]) -> Result<Vec<KnowledgeSftRecord>, Box<dyn Error>> {
    let mut records = Vec::new();
    for path in paths {
        for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(&line).map_err(|error| {
                invalid_data(format!("{}:{}: {error}", path.display(), index + 1))
            })?);
        }
    }
    if records.is_empty() {
        return Err(invalid_data("retention exam is empty"));
    }
    Ok(records)
}

fn validate_retention(
    records: &[KnowledgeSftRecord],
    contract_hash: &str,
) -> Result<(), Box<dyn Error>> {
    let mut ids = HashSet::new();
    for record in records {
        if record.contract_hash != contract_hash || !ids.insert(&record.id) {
            return Err(invalid_data(format!(
                "retention record {} has duplicate identity or wrong contract",
                record.id
            )));
        }
        KnowledgeAction::parse_runtime(&record.target)
            .map_err(|error| invalid_data(error.to_string()))?;
    }
    Ok(())
}

fn evenly_spaced_indices(length: usize, limit: usize) -> Vec<usize> {
    (0..limit).map(|index| index * length / limit).collect()
}

struct NetworkState {
    denied: bool,
    method: String,
    namespace: String,
    interfaces: Vec<String>,
    route_entries: usize,
}

#[cfg(target_os = "linux")]
fn inspect_network_denial() -> Result<NetworkState, Box<dyn Error>> {
    let mut interfaces: Vec<String> = std::fs::read_dir("/sys/class/net")?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    interfaces.sort();
    let route_entries = std::fs::read_to_string("/proc/net/route")?
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .count();
    let namespace = std::fs::read_link("/proc/self/ns/net")?
        .to_string_lossy()
        .into_owned();
    let denied = interfaces.iter().all(|name| name == "lo") && route_entries == 0;
    Ok(NetworkState {
        denied,
        method: "linux_net_namespace_sysfs_and_proc_route_v1".into(),
        namespace,
        interfaces,
        route_entries,
    })
}

#[cfg(not(target_os = "linux"))]
fn inspect_network_denial() -> Result<NetworkState, Box<dyn Error>> {
    Ok(NetworkState {
        denied: false,
        method: "unsupported_platform_requires_linux_network_namespace".into(),
        namespace: String::new(),
        interfaces: Vec::new(),
        route_entries: usize::MAX,
    })
}

fn new_file(path: &Path) -> Result<File, std::io::Error> {
    OpenOptions::new().create_new(true).write(true).open(path)
}

fn write_output_rows(
    path: &Path,
    rows: impl IntoIterator<Item = OutputRow>,
) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(new_file(path)?);
    for row in rows {
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension("json.tmp");
    if temporary.exists() {
        return Err(invalid_data(format!(
            "temporary output {} already exists",
            temporary.display()
        )));
    }
    {
        let mut writer = BufWriter::new(new_file(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "id": id,
            "episode_id": format!("episode-{id}"),
            "step": 0,
            "stage": "answer",
            "family": "retention.test",
            "difficulty": 1,
            "language": "english",
            "contract_hash": "blake3:test-contract",
            "tools": [],
            "conversation": [],
            "prompt": "prompt",
            "target": "answer"
        })
    }

    #[test]
    fn retention_reader_concatenates_every_file_in_argument_order() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("knowledge.jsonl");
        let second = directory.path().join("dojo.jsonl");
        std::fs::write(
            &first,
            format!("{}\n\n{}\n", record("knowledge-1"), record("knowledge-2")),
        )
        .unwrap();
        std::fs::write(&second, format!("{}\n", record("dojo-1"))).unwrap();

        let records = read_retention_records(&[first, second]).unwrap();
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["knowledge-1", "knowledge-2", "dojo-1"]
        );
    }

    #[test]
    fn retention_reader_refuses_an_empty_file_set() {
        assert!(read_retention_records(&[]).is_err());
    }
}
