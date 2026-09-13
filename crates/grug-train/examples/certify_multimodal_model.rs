//! Promote a multimodal BabyLLM candidate only when all frozen, checkpoint-
//! bound OCR, text-retention, and provider-free replay gates pass.

use clap::Parser;
use grug_data::{KnowledgeSftRecord, VisionTextManifest, VisionTextPrediction, score_vision_text};
use grug_train::{
    MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION, MultimodalAuditEvidence,
    MultimodalModelCertification, MultimodalNetworkDeniedReplayEvidence,
    MultimodalTextRetentionEvidence, fingerprint_artifact_file, fingerprint_knowledge_records,
    score_knowledge_outputs,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
type CandidateModel = grug_train::CpuMultimodalModel;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
type CandidateModel = grug_train::GpuMultimodalModel;

#[derive(Parser, Debug)]
#[command(name = "certify_multimodal_model")]
struct Args {
    #[arg(long)]
    candidate_bundle: PathBuf,
    #[arg(long)]
    audit_evidence: PathBuf,
    #[arg(long)]
    audit_manifest: PathBuf,
    #[arg(long)]
    audit_predictions: PathBuf,
    #[arg(long)]
    retention_evidence: PathBuf,
    #[arg(long, required = true)]
    retention_eval_jsonl: Vec<PathBuf>,
    #[arg(long)]
    retention_before_outputs: PathBuf,
    #[arg(long)]
    retention_after_outputs: PathBuf,
    #[arg(long)]
    network_replay_evidence: PathBuf,
    #[arg(long)]
    network_replay_first: PathBuf,
    #[arg(long)]
    network_replay_second: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OutputRow {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct PromotionReport<'a> {
    kind: &'static str,
    promoted: bool,
    active_pointer: String,
    checkpoint_step: usize,
    checkpoint_model_hash: &'a str,
    audit_evidence_hash: &'a str,
    retention_evidence_hash: &'a str,
    network_replay_evidence_hash: &'a str,
    audit_cases: usize,
    exact_rate: f64,
    character_error_rate: f64,
    cross_semantic_collapse_rate: f64,
    metamorphic_consistency: f64,
    minimum_required_slice_exact_rate: f64,
    text_action_exact_before: f64,
    text_action_exact_after: f64,
    text_action_exact_regression: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let active_path = args.candidate_bundle.join("multimodal-active.json");
    let report_path = args.candidate_bundle.join("multimodal-promotion.json");
    if active_path.exists() {
        return Err(invalid_data(
            "active pointer already exists; certification is append-only",
        ));
    }

    // Use the same loader as serving before considering scores. This verifies
    // every path/hash and proves Burn can deserialize the exact checkpoint.
    let runtime = load_candidate(&args.candidate_bundle)?;
    let mut artifact = runtime.artifact().clone();
    drop(runtime);
    if artifact.certification.is_some() {
        return Err(invalid_data("candidate already carries a certification"));
    }
    if !artifact.base_text_certified {
        return Err(invalid_data(
            "candidate did not warm-start from a certified text/tool model",
        ));
    }

    let audit: MultimodalAuditEvidence = read_json(&args.audit_evidence)?;
    let retention: MultimodalTextRetentionEvidence = read_json(&args.retention_evidence)?;
    let replay: MultimodalNetworkDeniedReplayEvidence = read_json(&args.network_replay_evidence)?;
    audit.verify_for(&artifact).map_err(invalid_data)?;
    retention.verify_for(&artifact).map_err(invalid_data)?;
    replay.verify_for(&artifact).map_err(invalid_data)?;

    if fingerprint_artifact_file(&args.audit_manifest)? != audit.audit_manifest_hash
        || fingerprint_artifact_file(&args.audit_predictions)? != audit.predictions_hash
    {
        return Err(invalid_data(
            "audit manifest/prediction files differ from their evidence hashes",
        ));
    }
    let audit_cases: Vec<VisionTextManifest> = read_json_lines(&args.audit_manifest)?;
    let audit_predictions: Vec<VisionTextPrediction> = read_json_lines(&args.audit_predictions)?;
    if score_vision_text(&audit_cases, &audit_predictions)? != audit.score {
        return Err(invalid_data(
            "recomputed OCR score differs from the submitted audit evidence",
        ));
    }

    let mut retention_records: Vec<KnowledgeSftRecord> = Vec::new();
    for path in &args.retention_eval_jsonl {
        retention_records.extend(read_json_lines(path)?);
    }
    if fingerprint_knowledge_records(&retention_records)? != retention.retention_corpus_hash
        || fingerprint_artifact_file(&args.retention_before_outputs)?
            != retention.before_outputs_hash
        || fingerprint_artifact_file(&args.retention_after_outputs)? != retention.after_outputs_hash
    {
        return Err(invalid_data(
            "retention records/output files differ from their evidence hashes",
        ));
    }
    let before_rows: Vec<OutputRow> = read_json_lines(&args.retention_before_outputs)?;
    let after_rows: Vec<OutputRow> = read_json_lines(&args.retention_after_outputs)?;
    let before_outputs = ordered_outputs(&retention_records, &before_rows)?;
    let after_outputs = ordered_outputs(&retention_records, &after_rows)?;
    if score_knowledge_outputs(&retention_records, &before_outputs) != retention.before
        || score_knowledge_outputs(&retention_records, &after_outputs) != retention.after
    {
        return Err(invalid_data(
            "recomputed text-retention score differs from the submitted evidence",
        ));
    }

    if fingerprint_artifact_file(&args.network_replay_first)? != replay.first_outputs_hash
        || fingerprint_artifact_file(&args.network_replay_second)? != replay.second_outputs_hash
    {
        return Err(invalid_data(
            "network replay streams differ from their evidence hashes",
        ));
    }
    let first_replay: Vec<OutputRow> = read_json_lines(&args.network_replay_first)?;
    let second_replay: Vec<OutputRow> = read_json_lines(&args.network_replay_second)?;
    if first_replay != second_replay || first_replay.len() != replay.replay_cases {
        return Err(invalid_data(
            "network-denied output streams are not identical and complete",
        ));
    }

    let minimum_slice = audit
        .minimum_required_slice_exact_rate()
        .map_err(invalid_data)?;
    let regression =
        (retention.before.action_exact_rate - retention.after.action_exact_rate).max(0.0);
    let audit_hash = fingerprint_artifact_file(&args.audit_evidence)?;
    let retention_hash = fingerprint_artifact_file(&args.retention_evidence)?;
    let replay_hash = fingerprint_artifact_file(&args.network_replay_evidence)?;
    let certification = MultimodalModelCertification {
        schema_version: MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION,
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
        audit_manifest_hash: artifact.audit_corpus.manifest_hash.clone(),
        audit_score_hash: audit_hash.clone(),
        text_retention_score_hash: retention_hash.clone(),
        audit_cases: audit.score.overall.cases as usize,
        minimum_matrix_cell_cases: artifact
            .audit_corpus
            .minimum_language_layout_corruption_cases,
        exact_rate: audit.score.overall.exact_rate,
        character_error_rate: audit.score.overall.character_error_rate,
        cross_semantic_collapse_rate: audit.score.cross_semantic_collapse_rate,
        metamorphic_consistency: audit.score.metamorphic_consistency,
        minimum_required_slice_exact_rate: minimum_slice,
        text_action_exact_before: retention.before.action_exact_rate,
        text_action_exact_after: retention.after.action_exact_rate,
        maximum_text_action_regression: regression,
        network_denied_replay_hash: replay_hash.clone(),
    };
    certification.verify_for(&artifact).map_err(invalid_data)?;
    artifact.certification = Some(certification);
    // Adam state is required to resume a candidate training run, but it is
    // deliberately not part of the serving surface.  Match the text/tool
    // activation contract by dropping the optimizer reference only after the
    // exact candidate checkpoint and all checkpoint-bound evidence have been
    // verified above.
    artifact.checkpoint_optimizer_hash = None;
    artifact.verify().map_err(invalid_data)?;

    let report = PromotionReport {
        kind: "multimodal_model_promotion",
        promoted: true,
        active_pointer: active_path.display().to_string(),
        checkpoint_step: artifact.checkpoint_step,
        checkpoint_model_hash: &artifact.checkpoint_model_hash,
        audit_evidence_hash: &audit_hash,
        retention_evidence_hash: &retention_hash,
        network_replay_evidence_hash: &replay_hash,
        audit_cases: audit.score.overall.cases as usize,
        exact_rate: audit.score.overall.exact_rate,
        character_error_rate: audit.score.overall.character_error_rate,
        cross_semantic_collapse_rate: audit.score.cross_semantic_collapse_rate,
        metamorphic_consistency: audit.score.metamorphic_consistency,
        minimum_required_slice_exact_rate: minimum_slice,
        text_action_exact_before: retention.before.action_exact_rate,
        text_action_exact_after: retention.after.action_exact_rate,
        text_action_exact_regression: regression,
    };
    if report_path.exists() {
        let existing: serde_json::Value = read_json(&report_path)?;
        if existing != serde_json::to_value(&report)? {
            return Err(invalid_data(
                "existing promotion report does not match recomputed evidence",
            ));
        }
    } else {
        write_json_atomic(&report_path, &report)?;
    }
    // Active pointer is intentionally the final write: observers either see no
    // active model or a complete pointer whose evidence already exists. If a
    // prior process stopped after the matching report write, this final step is
    // safely resumable without rewriting the immutable report.
    write_json_atomic(&active_path, &artifact)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
fn load_candidate(path: &Path) -> Result<CandidateModel, grug_train::MultimodalRuntimeError> {
    CandidateModel::load_candidate(path)
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
fn load_candidate(path: &Path) -> Result<CandidateModel, grug_train::MultimodalRuntimeError> {
    CandidateModel::load_candidate_gpu(path)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn read_json_lines<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, Box<dyn Error>> {
    use std::io::BufRead as _;

    let mut values = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(
            serde_json::from_str(&line).map_err(|error| {
                invalid_data(format!("{}:{}: {error}", path.display(), index + 1))
            })?,
        );
    }
    if values.is_empty() {
        return Err(invalid_data(format!("{} is empty", path.display())));
    }
    Ok(values)
}

fn ordered_outputs(
    records: &[KnowledgeSftRecord],
    rows: &[OutputRow],
) -> Result<Vec<String>, Box<dyn Error>> {
    if rows.len() != records.len() {
        return Err(invalid_data("retention output count differs from exam"));
    }
    let mut seen = HashSet::new();
    for (record, row) in records.iter().zip(rows) {
        if row.id != record.id || !seen.insert(&row.id) {
            return Err(invalid_data(
                "retention outputs are reordered, duplicated, or bound to other records",
            ));
        }
    }
    Ok(rows.iter().map(|row| row.text.clone()).collect())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension("json.tmp");
    let mut serialized = serde_json::to_vec_pretty(value)?;
    serialized.push(b'\n');
    if temporary.exists() {
        if std::fs::read(&temporary)? != serialized {
            return Err(invalid_data(format!(
                "temporary output {} exists with different content",
                temporary.display()
            )));
        }
    } else {
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)?,
        );
        writer.write_all(&serialized)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    std::fs::rename(&temporary, path)?;
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

    #[test]
    fn atomic_json_write_finishes_an_identical_interrupted_temporary() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("multimodal-active.json");
        let temporary = path.with_extension("json.tmp");
        let value = serde_json::json!({"checkpoint": "blake3:model"});
        let mut expected = serde_json::to_vec_pretty(&value).unwrap();
        expected.push(b'\n');
        std::fs::write(&temporary, &expected).unwrap();

        write_json_atomic(&path, &value).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        assert!(!temporary.exists());
    }

    #[test]
    fn atomic_json_write_rejects_a_conflicting_interrupted_temporary() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("multimodal-active.json");
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, b"different\n").unwrap();

        let error = write_json_atomic(&path, &serde_json::json!({"checkpoint": "expected"}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("different content"));
        assert!(!path.exists());
        assert!(temporary.exists());
    }
}
