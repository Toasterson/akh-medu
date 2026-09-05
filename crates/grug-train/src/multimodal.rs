//! Content-addressed artifact contract for the native OCR + text/tool model.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{KnowledgeModelShape, KnowledgeModelSize, KnowledgeTeacherForcedEval};

pub const MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION: u32 = 1;
pub const MULTIMODAL_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Full frozen-audit result bound to one candidate checkpoint and its exact
/// prediction stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalAuditEvidence {
    pub schema_version: u32,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    pub audit_manifest_hash: String,
    pub predictions_hash: String,
    pub score: grug_data::VisionTextScoreSummary,
}

/// Before/after text-and-tool retention result over the same quarantined
/// records, with the base artifact and multimodal candidate both identified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalTextRetentionEvidence {
    pub schema_version: u32,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    pub base_text_artifact_hash: String,
    pub base_text_checkpoint_hash: String,
    pub retention_corpus_hash: String,
    pub records: usize,
    pub before_outputs_hash: String,
    pub after_outputs_hash: String,
    pub before: KnowledgeTeacherForcedEval,
    pub after: KnowledgeTeacherForcedEval,
}

/// Deterministic replay performed while the process can observe no network
/// interface except loopback and no kernel route entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalNetworkDeniedReplayEvidence {
    pub schema_version: u32,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    pub audit_manifest_hash: String,
    pub network_denied_verified: bool,
    pub verification_method: String,
    pub network_namespace: String,
    pub visible_interfaces: Vec<String>,
    pub kernel_route_entries: usize,
    pub replay_cases: usize,
    pub first_outputs_hash: String,
    pub second_outputs_hash: String,
    pub outputs_identical: bool,
}

impl MultimodalAuditEvidence {
    pub fn verify_for(&self, artifact: &MultimodalModelArtifact) -> Result<(), String> {
        if self.schema_version != MULTIMODAL_EVIDENCE_SCHEMA_VERSION
            || self.checkpoint_step != artifact.checkpoint_step
            || self.checkpoint_model_hash != artifact.checkpoint_model_hash
            || self.audit_manifest_hash != artifact.audit_corpus.manifest_hash
            || self.score.overall.cases as usize != artifact.audit_corpus.cases
        {
            return Err("OCR audit evidence is not bound to this complete candidate".into());
        }
        if !self.predictions_hash.starts_with("blake3:") {
            return Err("OCR audit evidence lacks a prediction-stream hash".into());
        }
        Ok(())
    }

    pub fn minimum_required_slice_exact_rate(&self) -> Result<f64, String> {
        for language in ["en", "de", "fr"] {
            if !self.score.per_language.contains_key(language) {
                return Err(format!("OCR score lacks language slice {language}"));
            }
        }
        for layout in ["line", "paragraph", "receipt", "terminal", "api_error"] {
            if !self.score.per_layout.contains_key(layout) {
                return Err(format!("OCR score lacks layout slice {layout}"));
            }
        }
        for corruption in [
            "clean",
            "low_contrast",
            "salt_pepper",
            "box_blur",
            "occlusion",
            "uneven_lighting",
            "horizontal_shear",
        ] {
            if !self.score.per_corruption.contains_key(corruption) {
                return Err(format!("OCR score lacks corruption slice {corruption}"));
            }
        }
        self.score
            .per_language
            .values()
            .chain(self.score.per_layout.values())
            .chain(self.score.per_corruption.values())
            .map(|slice| slice.exact_rate)
            .reduce(f64::min)
            .ok_or_else(|| "OCR score contains no required slices".into())
    }
}

impl MultimodalTextRetentionEvidence {
    pub fn verify_for(&self, artifact: &MultimodalModelArtifact) -> Result<(), String> {
        if self.schema_version != MULTIMODAL_EVIDENCE_SCHEMA_VERSION
            || self.checkpoint_step != artifact.checkpoint_step
            || self.checkpoint_model_hash != artifact.checkpoint_model_hash
            || self.base_text_artifact_hash != artifact.base_text_artifact_hash
            || self.base_text_checkpoint_hash != artifact.base_text_checkpoint_hash
            || self.retention_corpus_hash != artifact.retention_eval_corpus_hash
            || self.records != artifact.retention_eval_records
            || self.before.samples != self.records
            || self.after.samples != self.records
        {
            return Err("text-retention evidence is not bound to this candidate/base/exam".into());
        }
        if !self.before_outputs_hash.starts_with("blake3:")
            || !self.after_outputs_hash.starts_with("blake3:")
        {
            return Err("text-retention evidence lacks output-stream hashes".into());
        }
        Ok(())
    }
}

impl MultimodalNetworkDeniedReplayEvidence {
    pub fn verify_for(&self, artifact: &MultimodalModelArtifact) -> Result<(), String> {
        if self.schema_version != MULTIMODAL_EVIDENCE_SCHEMA_VERSION
            || self.checkpoint_step != artifact.checkpoint_step
            || self.checkpoint_model_hash != artifact.checkpoint_model_hash
            || self.audit_manifest_hash != artifact.audit_corpus.manifest_hash
        {
            return Err("network-denied replay is not bound to this candidate/audit".into());
        }
        if !self.network_denied_verified
            || self.visible_interfaces.iter().any(|name| name != "lo")
            || self.kernel_route_entries != 0
            || self.replay_cases < 30
            || !self.outputs_identical
            || self.first_outputs_hash != self.second_outputs_hash
            || !self.first_outputs_hash.starts_with("blake3:")
        {
            return Err(
                "network-denied replay did not prove a deterministic provider-free run".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactVisionShape {
    pub input_channels: usize,
    pub hidden_channels: usize,
    pub output_stride: usize,
    pub max_grid_rows: usize,
    pub max_grid_columns: usize,
    pub canvas_width: usize,
    pub canvas_height: usize,
}

impl CompactVisionShape {
    pub fn visual_tokens(&self) -> usize {
        self.canvas_width.div_ceil(self.output_stride)
            * self.canvas_height.div_ceil(self.output_stride)
    }

    pub fn verify(&self, text_shape: &KnowledgeModelShape) -> Result<(), String> {
        if self.input_channels != 3
            || self.hidden_channels == 0
            || self.output_stride != 16
            || self.max_grid_rows == 0
            || self.max_grid_columns == 0
            || self.canvas_width == 0
            || self.canvas_height == 0
        {
            return Err("multimodal artifact has an invalid compact vision shape".into());
        }
        let rows = self.canvas_height.div_ceil(self.output_stride);
        let columns = self.canvas_width.div_ceil(self.output_stride);
        if rows > self.max_grid_rows || columns > self.max_grid_columns {
            return Err(format!(
                "canvas produces {rows}x{columns} visual tokens, above configured {}x{}",
                self.max_grid_rows, self.max_grid_columns
            ));
        }
        if self.visual_tokens() >= text_shape.max_seq_len {
            return Err(format!(
                "{} visual tokens leave no text room in context {}",
                self.visual_tokens(),
                text_shape.max_seq_len
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionCorpusLineage {
    pub manifest_hash: String,
    pub cases: usize,
    pub semantic_items: usize,
    pub minimum_language_layout_corruption_cases: usize,
    pub generator_versions: BTreeSet<String>,
    pub renderer_ids: BTreeSet<String>,
    pub source_hashes: BTreeSet<String>,
    pub languages: BTreeSet<String>,
    pub layouts: BTreeSet<String>,
    pub corruptions: BTreeSet<String>,
    #[serde(default)]
    pub held_out_axes: BTreeSet<String>,
}

impl VisionCorpusLineage {
    pub fn verify(&self, audit: bool) -> Result<(), String> {
        if !self.manifest_hash.starts_with("blake3:")
            || self.cases == 0
            || self.semantic_items == 0
            || self.generator_versions.is_empty()
            || self.renderer_ids.is_empty()
            || self.source_hashes.is_empty()
            || self
                .source_hashes
                .iter()
                .any(|hash| !hash.starts_with("blake3:"))
        {
            return Err("vision corpus lineage is incomplete or lacks content hashes".into());
        }
        for required in ["en", "de", "fr"] {
            if !self.languages.contains(required) {
                return Err(format!(
                    "vision corpus is missing required language {required}"
                ));
            }
        }
        for required in ["line", "paragraph", "receipt", "terminal", "api_error"] {
            if !self.layouts.contains(required) {
                return Err(format!(
                    "vision corpus is missing required layout {required}"
                ));
            }
        }
        for required in [
            "clean",
            "low_contrast",
            "salt_pepper",
            "box_blur",
            "occlusion",
            "uneven_lighting",
            "horizontal_shear",
        ] {
            if !self.corruptions.contains(required) {
                return Err(format!(
                    "vision corpus is missing required corruption {required}"
                ));
            }
        }
        if audit {
            if self.minimum_language_layout_corruption_cases < 10 {
                return Err(
                    "audit corpus has fewer than 10 cases in a required matrix cell".into(),
                );
            }
            if !self.held_out_axes.contains("renderer")
                || !self.held_out_axes.contains("content_bank")
            {
                return Err(
                    "audit corpus must hold out both renderer and content-bank axes".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalModelCertification {
    pub schema_version: u32,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    pub audit_manifest_hash: String,
    pub audit_score_hash: String,
    pub text_retention_score_hash: String,
    pub audit_cases: usize,
    pub minimum_matrix_cell_cases: usize,
    pub exact_rate: f64,
    pub character_error_rate: f64,
    pub cross_semantic_collapse_rate: f64,
    pub metamorphic_consistency: f64,
    pub minimum_required_slice_exact_rate: f64,
    pub text_action_exact_before: f64,
    pub text_action_exact_after: f64,
    pub maximum_text_action_regression: f64,
    pub network_denied_replay_hash: String,
}

impl MultimodalModelCertification {
    pub fn verify_for(&self, artifact: &MultimodalModelArtifact) -> Result<(), String> {
        if self.schema_version != MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION
            || self.checkpoint_step != artifact.checkpoint_step
            || self.checkpoint_model_hash != artifact.checkpoint_model_hash
            || self.audit_manifest_hash != artifact.audit_corpus.manifest_hash
        {
            return Err("multimodal certification is not bound to this checkpoint/audit".into());
        }
        if self.audit_cases < 1_050
            || self.minimum_matrix_cell_cases < 10
            || self.exact_rate < 0.80
            || self.character_error_rate > 0.08
            || self.cross_semantic_collapse_rate > 0.01
            || self.metamorphic_consistency < 0.80
            || self.minimum_required_slice_exact_rate < 0.70
        {
            return Err("multimodal certification does not clear the frozen OCR gates".into());
        }
        let regression = (self.text_action_exact_before - self.text_action_exact_after).max(0.0);
        // Rates are recomputed from integer counts, so an exact two-point
        // boundary can land a few ulps above `0.02` (for example 0.90-0.88).
        // Keep the documented threshold inclusive without admitting a
        // materially larger regression.
        const RATE_EPSILON: f64 = 1e-12;
        if self.maximum_text_action_regression > 0.02 + RATE_EPSILON
            || regression > 0.02 + RATE_EPSILON
        {
            return Err(format!(
                "text/tool action-exact regression {regression:.4} exceeds 0.02"
            ));
        }
        for hash in [
            &self.audit_score_hash,
            &self.text_retention_score_hash,
            &self.network_denied_replay_hash,
        ] {
            if !hash.starts_with("blake3:") {
                return Err("multimodal certification is missing a BLAKE3 evidence hash".into());
            }
        }
        Ok(())
    }
}

/// Candidate/active runtime pointer for one shared text-and-vision decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalModelArtifact {
    pub schema_version: u32,
    pub model_size: KnowledgeModelSize,
    pub text_shape: KnowledgeModelShape,
    pub vision_shape: CompactVisionShape,
    pub parameter_count: usize,
    pub tokenizer_file: String,
    pub tokenizer_hash: String,
    pub contracts_file: String,
    pub contract_hash: String,
    pub checkpoint_dir: String,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_optimizer_hash: Option<String>,
    pub base_text_artifact_hash: String,
    pub base_text_checkpoint_step: usize,
    pub base_text_checkpoint_hash: String,
    pub base_text_certified: bool,
    pub vision_training_corpus: VisionCorpusLineage,
    pub audit_corpus: VisionCorpusLineage,
    pub retention_corpus_hash: String,
    pub retention_records: usize,
    pub retention_eval_corpus_hash: String,
    pub retention_eval_records: usize,
    pub stem_warmup_updates: usize,
    pub joint_vision_updates: usize,
    pub joint_text_updates: usize,
    pub skipped_updates: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certification: Option<MultimodalModelCertification>,
}

impl MultimodalModelArtifact {
    pub fn verify(&self) -> Result<(), String> {
        if self.schema_version != MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION {
            return Err(format!(
                "multimodal artifact schema {} is unsupported; expected {}",
                self.schema_version, MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION
            ));
        }
        let expected_shape = self.model_size.config(self.text_shape.vocab_size);
        if KnowledgeModelShape::from_config(&expected_shape) != self.text_shape {
            return Err("multimodal text shape is not a canonical model-size shape".into());
        }
        self.vision_shape.verify(&self.text_shape)?;
        if self.parameter_count == 0
            || self.tokenizer_file.is_empty()
            || self.contracts_file.is_empty()
            || self.checkpoint_dir.is_empty()
            || self.retention_records == 0
            || self.retention_eval_records == 0
            || self.joint_vision_updates == 0
            || self.joint_text_updates == 0
        {
            return Err("multimodal artifact is missing runtime or training state".into());
        }
        for hash in [
            &self.tokenizer_hash,
            &self.contract_hash,
            &self.checkpoint_model_hash,
            &self.base_text_artifact_hash,
            &self.base_text_checkpoint_hash,
            &self.retention_corpus_hash,
            &self.retention_eval_corpus_hash,
        ] {
            if !hash.starts_with("blake3:") {
                return Err("multimodal artifact is missing a BLAKE3 content hash".into());
            }
        }
        if self
            .checkpoint_optimizer_hash
            .as_ref()
            .is_some_and(|hash| !hash.starts_with("blake3:"))
        {
            return Err("multimodal optimizer hash is not BLAKE3".into());
        }
        self.vision_training_corpus.verify(false)?;
        self.audit_corpus.verify(true)?;
        if self
            .vision_training_corpus
            .renderer_ids
            .iter()
            .any(|renderer| self.audit_corpus.renderer_ids.contains(renderer))
        {
            return Err("training and audit renderer identities overlap".into());
        }
        if let Some(certification) = &self.certification {
            if !self.base_text_certified {
                return Err(
                    "a multimodal model cannot activate from an uncertified text base".into(),
                );
            }
            certification.verify_for(self)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineage(renderer: &str, audit: bool) -> VisionCorpusLineage {
        VisionCorpusLineage {
            manifest_hash: format!("blake3:{}", if audit { "audit" } else { "train" }),
            cases: 1_050,
            semantic_items: 150,
            minimum_language_layout_corruption_cases: if audit { 10 } else { 1 },
            generator_versions: BTreeSet::from(["generator-v1".into()]),
            renderer_ids: BTreeSet::from([renderer.into()]),
            source_hashes: BTreeSet::from(["blake3:source".into()]),
            languages: BTreeSet::from(["en".into(), "de".into(), "fr".into()]),
            layouts: BTreeSet::from([
                "line".into(),
                "paragraph".into(),
                "receipt".into(),
                "terminal".into(),
                "api_error".into(),
            ]),
            corruptions: BTreeSet::from([
                "clean".into(),
                "low_contrast".into(),
                "salt_pepper".into(),
                "box_blur".into(),
                "occlusion".into(),
                "uneven_lighting".into(),
                "horizontal_shear".into(),
            ]),
            held_out_axes: if audit {
                BTreeSet::from(["renderer".into(), "content_bank".into()])
            } else {
                BTreeSet::new()
            },
        }
    }

    fn artifact() -> MultimodalModelArtifact {
        let text_shape = KnowledgeModelShape::from_config(&KnowledgeModelSize::Small.config(8_000));
        MultimodalModelArtifact {
            schema_version: MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION,
            model_size: KnowledgeModelSize::Small,
            text_shape,
            vision_shape: CompactVisionShape {
                input_channels: 3,
                hidden_channels: 64,
                output_stride: 16,
                max_grid_rows: 12,
                max_grid_columns: 30,
                canvas_width: 480,
                canvas_height: 192,
            },
            parameter_count: 1,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_hash: "blake3:tokenizer".into(),
            contracts_file: "contracts.json".into(),
            contract_hash: "blake3:contracts".into(),
            checkpoint_dir: "checkpoint".into(),
            checkpoint_step: 42,
            checkpoint_model_hash: "blake3:model".into(),
            checkpoint_optimizer_hash: Some("blake3:optimizer".into()),
            base_text_artifact_hash: "blake3:base-artifact".into(),
            base_text_checkpoint_step: 41,
            base_text_checkpoint_hash: "blake3:base-model".into(),
            base_text_certified: true,
            vision_training_corpus: lineage("training-renderer", false),
            audit_corpus: lineage("audit-renderer", true),
            retention_corpus_hash: "blake3:retention-train".into(),
            retention_records: 100,
            retention_eval_corpus_hash: "blake3:retention-eval".into(),
            retention_eval_records: 60,
            stem_warmup_updates: 500,
            joint_vision_updates: 100,
            joint_text_updates: 100,
            skipped_updates: 0,
            certification: None,
        }
    }

    fn passing_certification(artifact: &MultimodalModelArtifact) -> MultimodalModelCertification {
        MultimodalModelCertification {
            schema_version: MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION,
            checkpoint_step: artifact.checkpoint_step,
            checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
            audit_manifest_hash: artifact.audit_corpus.manifest_hash.clone(),
            audit_score_hash: "blake3:audit-score".into(),
            text_retention_score_hash: "blake3:retention-score".into(),
            audit_cases: 1_050,
            minimum_matrix_cell_cases: 10,
            exact_rate: 0.80,
            character_error_rate: 0.08,
            cross_semantic_collapse_rate: 0.01,
            metamorphic_consistency: 0.80,
            minimum_required_slice_exact_rate: 0.70,
            text_action_exact_before: 0.90,
            text_action_exact_after: 0.88,
            maximum_text_action_regression: 0.02,
            network_denied_replay_hash: "blake3:network-replay".into(),
        }
    }

    #[test]
    fn compact_shape_accounts_for_visual_context() {
        let shape = CompactVisionShape {
            input_channels: 3,
            hidden_channels: 64,
            output_stride: 16,
            max_grid_rows: 12,
            max_grid_columns: 30,
            canvas_width: 480,
            canvas_height: 192,
        };
        assert_eq!(shape.visual_tokens(), 360);
        assert!(
            shape
                .verify(&KnowledgeModelShape::from_config(
                    &KnowledgeModelSize::Small.config(8_000)
                ))
                .is_ok()
        );
    }

    #[test]
    fn artifact_rejects_training_and_audit_renderer_overlap() {
        let mut artifact = artifact();
        assert!(artifact.verify().is_ok());
        artifact.audit_corpus.renderer_ids = artifact.vision_training_corpus.renderer_ids.clone();
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("renderer identities overlap")
        );
    }

    #[test]
    fn certification_thresholds_are_inclusive_but_regression_is_bounded() {
        let artifact = artifact();
        let mut certification = passing_certification(&artifact);
        assert!(certification.verify_for(&artifact).is_ok());

        certification.text_action_exact_after = 0.879;
        assert!(
            certification
                .verify_for(&artifact)
                .unwrap_err()
                .contains("exceeds 0.02")
        );
    }

    #[test]
    fn provider_free_replay_requires_a_real_network_deny_and_thirty_cases() {
        let artifact = artifact();
        let mut replay = MultimodalNetworkDeniedReplayEvidence {
            schema_version: MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
            checkpoint_step: artifact.checkpoint_step,
            checkpoint_model_hash: artifact.checkpoint_model_hash.clone(),
            audit_manifest_hash: artifact.audit_corpus.manifest_hash.clone(),
            network_denied_verified: true,
            verification_method: "linux-netns".into(),
            network_namespace: "net:[1]".into(),
            visible_interfaces: vec!["lo".into()],
            kernel_route_entries: 0,
            replay_cases: 30,
            first_outputs_hash: "blake3:outputs".into(),
            second_outputs_hash: "blake3:outputs".into(),
            outputs_identical: true,
        };
        assert!(replay.verify_for(&artifact).is_ok());

        replay.replay_cases = 29;
        assert!(replay.verify_for(&artifact).is_err());
        replay.replay_cases = 30;
        replay.visible_interfaces.push("eth0".into());
        assert!(replay.verify_for(&artifact).is_err());
    }
}
