//! P6.4 local remediation: continuing an *uncertified* `train_knowledge`
//! candidate's retained Adam optimizer state on the original base corpus
//! plus one or more engine-generated supplemental corpora (P6.3's
//! `akh_episode_generator` siblings), rather than the certified-active
//! `--adapt-from-bundle` path or the controller-authorized
//! `--steer-from-bundle` path.
//!
//! Unlike [`crate::knowledge_train::KnowledgeAdaptationLineage`] /
//! [`crate::knowledge_train::KnowledgeSteeringLineage`] -- both portable
//! (content-hash-only, no local paths) because they mint a lineage that must
//! travel with a promoted or controller-authorized artifact -- a remediation
//! segment is a locally-tracked continuation of one specific uncertified
//! candidate bundle on this machine, so [`RemediationLineage`] deliberately
//! keeps that bundle's local path alongside its content hashes.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::GrugError;
use crate::knowledge_train::MAX_KNOWLEDGE_STEERING_SHARE_PPM;
use crate::train::RecordWeightsMode;

/// Default ceiling on how much of a remediation segment's mixture an
/// engine-generated supplemental corpus may occupy, in parts per million.
/// Matches [`MAX_KNOWLEDGE_STEERING_SHARE_PPM`] (50_000 ppm = 5%), the same
/// bound the adaptive loop's controller-directed steering already enforces.
/// `train_knowledge --remediation-max-share-ppm` may override it; whatever
/// value is actually used is always recorded in [`RemediationLineage`].
pub const DEFAULT_REMEDIATION_MAX_SHARE_PPM: u64 = MAX_KNOWLEDGE_STEERING_SHARE_PPM;

/// Validate a remediation segment's supplemental share against `ceiling_ppm`
/// and return the actual share in parts per million. Integer cross
/// multiplication keeps the ceiling exact -- no floating-point rounding can
/// let a share creep past the limit.
pub fn validate_remediation_share(
    base_records: usize,
    supplemental_records: usize,
    ceiling_ppm: u64,
) -> Result<u64, GrugError> {
    if ceiling_ppm == 0 || ceiling_ppm > 1_000_000 {
        return Err(GrugError::RemediationInvalidCeiling { ceiling_ppm });
    }
    if base_records == 0 || supplemental_records == 0 {
        return Err(GrugError::RemediationEmptyMixture {
            base_records,
            supplemental_records,
        });
    }
    let total = (base_records as u128)
        .checked_add(supplemental_records as u128)
        .ok_or(GrugError::RemediationRecordCountOverflow)?;
    let scaled_supplemental = (supplemental_records as u128) * 1_000_000u128;
    let actual_ppm = u64::try_from((scaled_supplemental + total / 2) / total)
        .map_err(|_| GrugError::RemediationRecordCountOverflow)?;
    if scaled_supplemental > total * u128::from(ceiling_ppm) {
        return Err(GrugError::RemediationShareExceeded {
            actual_ppm,
            ceiling_ppm,
            base_records,
            supplemental_records,
        });
    }
    Ok(actual_ppm)
}

/// One supplemental (engine-generated sibling) corpus mixed into a
/// remediation segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemediationSupplement {
    pub path: String,
    pub sft_hash: String,
    pub records: usize,
    /// Read from a sibling `manifest.json`, if one exists next to `path`.
    /// See [`read_sibling_generator_metadata`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_git_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Content-addressed evidence for one local P6.4 remediation segment: an
/// uncertified candidate's weights *and* Adam state continued on
/// `--train-jsonl` plus one or more `--supplemental-jsonl` engine-generated
/// siblings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemediationLineage {
    /// Local path to the parent candidate bundle directory (the argument to
    /// `--remediate-from-bundle`). Deliberately not content-addressed-only:
    /// remediation is a machine-local continuation, not a portable artifact.
    pub parent_bundle: String,
    pub parent_checkpoint_step: usize,
    pub parent_checkpoint_model_hash: String,
    pub parent_checkpoint_optimizer_hash: String,
    pub contract_hash: String,
    pub tokenizer_hash: String,
    pub supplemental: Vec<RemediationSupplement>,
    pub base_training_records: usize,
    pub supplemental_training_records: usize,
    pub supplemental_share_ppm: u64,
    /// The `--remediation-max-share-ppm` ceiling this segment was validated
    /// against (default [`DEFAULT_REMEDIATION_MAX_SHARE_PPM`]).
    pub max_share_ppm: u64,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub epochs: usize,
    /// Whether this segment read per-record reward weights from
    /// `KnowledgeSftRecord::weight` (Phase 51 P6.5 experiment E3). Absent
    /// from `RemediationLineage` blocks written before this field existed --
    /// `#[serde(default)]` resolves those to [`RecordWeightsMode::Ignore`],
    /// exactly matching their actual (pre-E3) behaviour.
    #[serde(default)]
    pub record_weights: RecordWeightsMode,
    /// How many training records in this segment's full mixture (base +
    /// supplemental) carried a `weight` value. Always `0` when
    /// `record_weights` is `Ignore` (weights are never read, so none are
    /// counted as "used" regardless of what the source JSONL happens to
    /// contain) -- this is what lets an E3 treatment candidate
    /// (`record_weights: "use"`, `weighted_record_count > 0`) and its
    /// control (`record_weights: "ignore"`, `weighted_record_count == 0`)
    /// be told apart from `candidate-model.json` alone.
    #[serde(default)]
    pub weighted_record_count: usize,
}

impl RemediationLineage {
    pub fn verify(&self) -> Result<(), String> {
        if self.parent_bundle.trim().is_empty() {
            return Err("remediation lineage has a blank parent bundle path".into());
        }
        if !self.parent_checkpoint_model_hash.starts_with("blake3:")
            || !self.parent_checkpoint_optimizer_hash.starts_with("blake3:")
            || !self.contract_hash.starts_with("blake3:")
            || !self.tokenizer_hash.starts_with("blake3:")
        {
            return Err("remediation lineage is missing a BLAKE3 content hash".into());
        }
        if self.supplemental.is_empty() {
            return Err("remediation lineage has no supplemental corpora".into());
        }
        let mut summed = 0usize;
        for supplement in &self.supplemental {
            if supplement.path.trim().is_empty()
                || !supplement.sft_hash.starts_with("blake3:")
                || supplement.records == 0
            {
                return Err("remediation lineage has an incomplete supplemental entry".into());
            }
            summed = summed
                .checked_add(supplement.records)
                .ok_or_else(|| "remediation supplemental record count overflow".to_string())?;
        }
        if summed != self.supplemental_training_records {
            return Err(
                "remediation lineage's supplemental record total does not match its entries \
                 sum"
                .into(),
            );
        }
        if self.base_training_records == 0 {
            return Err("remediation lineage has zero base training records".into());
        }
        let recomputed = validate_remediation_share(
            self.base_training_records,
            self.supplemental_training_records,
            self.max_share_ppm,
        )
        .map_err(|error| error.to_string())?;
        if recomputed != self.supplemental_share_ppm {
            return Err("remediation lineage has an inconsistent supplemental share".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("remediation lineage learning rate must be finite and positive".into());
        }
        if self.epochs == 0 {
            return Err("remediation lineage epochs must be positive".into());
        }
        if self.record_weights == RecordWeightsMode::Ignore && self.weighted_record_count != 0 {
            return Err(
                "remediation lineage has record_weights=ignore but a nonzero \
                 weighted_record_count -- weights that are never read cannot have been used"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Best-effort read of a P6.3-generator-shaped `manifest.json` sitting next
/// to a supplemental JSONL file (same directory). Two real shapes exist on
/// disk (see `training/results/p6-3-episode-generator-20260913-RESULT.md`
/// §sidecar): the run-level `Manifest` (top-level `manifest.json` beside a
/// primary `<family>.jsonl`, carries `generator_git_hash` + `seed`) and the
/// per-family `SteeringManifest` sidecar (`<family>/manifest.json`, carries
/// neither). Both are read permissively as JSON -- a missing file, a parse
/// failure, or absent fields are all silently `(None, None)`, never a hard
/// error: this metadata is provenance-only context recorded on the lineage,
/// not a training input that gates the run.
pub fn read_sibling_generator_metadata(supplemental_path: &Path) -> (Option<String>, Option<u64>) {
    let Some(parent) = supplemental_path.parent() else {
        return (None, None);
    };
    let manifest_path = parent.join("manifest.json");
    let Ok(bytes) = std::fs::read(&manifest_path) else {
        return (None, None);
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return (None, None);
    };
    let generator_git_hash = value
        .get("generator_git_hash")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let seed = value.get("seed").and_then(serde_json::Value::as_u64);
    (generator_git_hash, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_at_exactly_the_ceiling_is_accepted() {
        // 50 supplemental of 950 base + 50 supplemental = 1000 total is
        // exactly 50_000 ppm (5%) -- the default ceiling must accept the
        // boundary value, not just values strictly below it.
        let share = validate_remediation_share(950, 50, 50_000).expect("must accept boundary");
        assert_eq!(share, 50_000);
    }

    #[test]
    fn one_record_over_the_ceiling_is_rejected() {
        // 51 supplemental of 949 base + 51 supplemental = 1000 total is
        // 51_000 ppm, one record past the 50_000 ppm ceiling.
        let error = validate_remediation_share(949, 51, 50_000).unwrap_err();
        assert!(matches!(
            error,
            GrugError::RemediationShareExceeded {
                base_records: 949,
                supplemental_records: 51,
                ceiling_ppm: 50_000,
                ..
            }
        ));
    }

    #[test]
    fn empty_base_or_supplemental_is_rejected() {
        assert!(matches!(
            validate_remediation_share(0, 10, 50_000),
            Err(GrugError::RemediationEmptyMixture { .. })
        ));
        assert!(matches!(
            validate_remediation_share(10, 0, 50_000),
            Err(GrugError::RemediationEmptyMixture { .. })
        ));
    }

    #[test]
    fn zero_or_overlarge_ceiling_is_rejected() {
        assert!(matches!(
            validate_remediation_share(10, 1, 0),
            Err(GrugError::RemediationInvalidCeiling { ceiling_ppm: 0 })
        ));
        assert!(matches!(
            validate_remediation_share(10, 1, 1_000_001),
            Err(GrugError::RemediationInvalidCeiling {
                ceiling_ppm: 1_000_001
            })
        ));
    }

    fn sample_lineage() -> RemediationLineage {
        RemediationLineage {
            parent_bundle: "runs/candidate".into(),
            parent_checkpoint_step: 500,
            parent_checkpoint_model_hash: "blake3:model".into(),
            parent_checkpoint_optimizer_hash: "blake3:optim".into(),
            contract_hash: "blake3:contract".into(),
            tokenizer_hash: "blake3:tokenizer".into(),
            supplemental: vec![RemediationSupplement {
                path: "training/episodes/p6-3/memory_write.jsonl".into(),
                sft_hash: "blake3:sibling".into(),
                records: 50,
                generator_git_hash: Some("95368800".into()),
                seed: Some(20260913),
            }],
            base_training_records: 950,
            supplemental_training_records: 50,
            supplemental_share_ppm: 50_000,
            max_share_ppm: 50_000,
            learning_rate: 1.5e-4,
            warmup_steps: 100,
            epochs: 2,
            record_weights: RecordWeightsMode::Ignore,
            weighted_record_count: 0,
        }
    }

    #[test]
    fn lineage_round_trips_through_json_and_verifies() {
        let lineage = sample_lineage();
        lineage.verify().expect("sample lineage must verify");
        let json = serde_json::to_string_pretty(&lineage).expect("serialize");
        let decoded: RemediationLineage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(lineage, decoded);
        decoded.verify().expect("round-tripped lineage must verify");
    }

    #[test]
    fn lineage_rejects_inconsistent_share() {
        let mut lineage = sample_lineage();
        lineage.supplemental_share_ppm = 40_000;
        assert!(lineage.verify().is_err());
    }

    #[test]
    fn lineage_rejects_mismatched_supplemental_total() {
        let mut lineage = sample_lineage();
        lineage.supplemental_training_records = 51;
        assert!(lineage.verify().is_err());
    }

    #[test]
    fn lineage_rejects_ignore_mode_with_nonzero_weighted_count() {
        let mut lineage = sample_lineage();
        lineage.record_weights = RecordWeightsMode::Ignore;
        lineage.weighted_record_count = 3;
        assert!(lineage.verify().is_err());
    }

    #[test]
    fn lineage_accepts_use_mode_with_weighted_count() {
        let mut lineage = sample_lineage();
        lineage.record_weights = RecordWeightsMode::Use;
        lineage.weighted_record_count = 3;
        lineage
            .verify()
            .expect("use mode with weighted records verifies");
    }

    /// A `RemediationLineage` JSON blob written before `record_weights`/
    /// `weighted_record_count` existed must still deserialize, defaulting to
    /// `Ignore`/`0` -- exactly the pre-E3 behaviour those blocks recorded.
    #[test]
    fn lineage_defaults_record_weights_when_absent_from_legacy_json() {
        let mut value = serde_json::to_value(sample_lineage()).expect("serialize sample lineage");
        let object = value.as_object_mut().expect("lineage is a JSON object");
        object.remove("record_weights");
        object.remove("weighted_record_count");
        let decoded: RemediationLineage =
            serde_json::from_value(value).expect("legacy lineage without record_weights fields");
        assert_eq!(decoded.record_weights, RecordWeightsMode::Ignore);
        assert_eq!(decoded.weighted_record_count, 0);
        decoded.verify().expect("legacy-defaulted lineage verifies");
    }

    #[test]
    fn read_sibling_generator_metadata_is_none_when_manifest_absent() {
        let directory = tempfile::tempdir().unwrap();
        let supplemental = directory.path().join("family.jsonl");
        std::fs::write(&supplemental, "{}\n").unwrap();
        assert_eq!(read_sibling_generator_metadata(&supplemental), (None, None));
    }

    #[test]
    fn read_sibling_generator_metadata_reads_run_level_manifest() {
        let directory = tempfile::tempdir().unwrap();
        let supplemental = directory.path().join("memory_write.jsonl");
        std::fs::write(&supplemental, "{}\n").unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "generator": "akh-episode-generator-engine-v1",
                "generator_git_hash": "95368800c21f5b4f550ec59f56e932cd22cd7598",
                "seed": 20260913,
            })
            .to_string(),
        )
        .unwrap();
        let (generator_git_hash, seed) = read_sibling_generator_metadata(&supplemental);
        assert_eq!(
            generator_git_hash.as_deref(),
            Some("95368800c21f5b4f550ec59f56e932cd22cd7598")
        );
        assert_eq!(seed, Some(20260913));
    }

    #[test]
    fn read_sibling_generator_metadata_is_none_for_per_family_sidecar_shape() {
        // The per-family `SteeringManifest` sidecar shares the filename but
        // carries neither field -- must degrade to `(None, None)`, not error.
        let directory = tempfile::tempdir().unwrap();
        let supplemental = directory.path().join("train.sft.jsonl");
        std::fs::write(&supplemental, "{}\n").unwrap();
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "complete": true,
                "contract_hash": "blake3:contract",
                "family": "tool_contract.memory_write",
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(read_sibling_generator_metadata(&supplemental), (None, None));
    }
}
