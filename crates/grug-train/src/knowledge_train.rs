//! Target-masked training plumbing for stateful Akh knowledge workflows.
//!
//! A KnowledgeSftRecord already carries a prompt rendered from captured tool
//! contracts and a runtime-native target. This module feeds those strings into
//! the same TinyTransformer optimizer used by the tool and reasoning corpora.
//! Long prompts are trimmed from both middle sides while preserving the target,
//! avoiding the old failure mode where a large tool catalog truncated away
//! every supervised token.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use grug_data::{KnowledgeAction, KnowledgeSftRecord};

use crate::error::GrugResult;
use crate::foundation::FoundationLineage;
use crate::generate::{GenerateConfig, generate_from_prompt_ids_cached};
use crate::tokenizer::Vocab;
use crate::train::{Batch, Encoded};
use crate::transformer::TinyTransformer;
use crate::transformer::TinyTransformerConfig;

pub const KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION: u32 = 2;
/// A controller-directed sibling may add at most five percentage points to a
/// candidate segment's record mixture. The limit is represented in parts per
/// million so artifact verification does not depend on floating-point math.
pub const MAX_KNOWLEDGE_STEERING_SHARE_PPM: u64 = 50_000;

/// Supported BabyLLM capacity rungs for the tool-native knowledge workflow.
///
/// `Small` is the normal development/deployment target. `Model350m` keeps the
/// exact same artifact and prompt contracts, allowing a capacity experiment
/// without forking the runtime integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeModelSize {
    Small,
    #[serde(rename = "350m")]
    Model350m,
}

impl KnowledgeModelSize {
    pub fn config(self, vocab_size: usize) -> TinyTransformerConfig {
        match self {
            Self::Small => TinyTransformerConfig::akh_tool_small(vocab_size),
            Self::Model350m => TinyTransformerConfig::akh_tool_350m(vocab_size),
        }
    }
}

impl fmt::Display for KnowledgeModelSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Small => "small",
            Self::Model350m => "350m",
        })
    }
}

impl FromStr for KnowledgeModelSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "small" => Ok(Self::Small),
            "350m" | "model350m" | "tool-350m" => Ok(Self::Model350m),
            other => Err(format!(
                "unknown knowledge model size {other:?}; expected small or 350m"
            )),
        }
    }
}

/// Serializable architecture fields needed to reconstruct a Burn checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeModelShape {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

impl KnowledgeModelShape {
    pub fn from_config(config: &TinyTransformerConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_heads: config.n_heads,
            n_layers: config.n_layers,
            d_ff: config.d_ff,
            max_seq_len: config.max_seq_len,
        }
    }

    pub fn config(&self) -> TinyTransformerConfig {
        TinyTransformerConfig::new(
            self.vocab_size,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
            self.max_seq_len,
        )
    }
}

/// The small, atomic pointer Akh-Medu needs to load a trained knowledge model.
/// All paths are relative to the directory containing this JSON document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeModelArtifact {
    pub schema_version: u32,
    pub model_size: KnowledgeModelSize,
    pub shape: KnowledgeModelShape,
    pub parameter_count: usize,
    pub tokenizer_file: String,
    pub tokenizer_hash: String,
    pub contracts_file: String,
    pub checkpoint_dir: String,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    /// Present in resumable training bundles; inference-only distributions may
    /// omit Adam's much larger state file deliberately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_optimizer_hash: Option<String>,
    pub contract_hash: String,
    pub training_corpus_hash: String,
    pub training_records: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_corpus_hash: Option<String>,
    #[serde(default)]
    pub capability_families: BTreeSet<String>,
    #[serde(default)]
    pub languages: BTreeSet<String>,
    #[serde(default)]
    pub stages: BTreeSet<String>,
    /// Present when knowledge/tool SFT warm-started from a separately audited
    /// causal-language foundation. Hashes make the ancestry portable without
    /// embedding a machine-local bundle path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foundation: Option<FoundationLineage>,
    /// Present when this candidate reset Adam and adapted the weights of a
    /// previously certified active knowledge model to a new immutable corpus
    /// or tool contract. The parent is content-addressed rather than referred
    /// to by a machine-local path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapted_from: Option<KnowledgeAdaptationLineage>,
    /// Present on a new bounded training segment authorized by a checkpoint-
    /// bound adaptive-controller `steer` decision and a separation-validated
    /// sibling corpus. Unlike active-model adaptation, this parent is an
    /// unpromoted candidate and its optimizer state is continued.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steered_from: Option<KnowledgeSteeringLineage>,
    /// Absent on ordinary training checkpoints. `certify_knowledge_model`
    /// populates it only after checkpoint-bound, quarantined exams cover every
    /// required knowledge workflow and every captured runtime tool contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certification: Option<KnowledgeModelCertification>,
}

impl KnowledgeModelArtifact {
    pub fn verify(&self) -> Result<(), String> {
        if self.schema_version != KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION {
            return Err(format!(
                "knowledge model artifact schema {} is unsupported; expected {}",
                self.schema_version, KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION
            ));
        }
        if self.shape.vocab_size == 0 || self.shape.max_seq_len < 3 {
            return Err("knowledge model artifact has an invalid model shape".into());
        }
        let expected =
            KnowledgeModelShape::from_config(&self.model_size.config(self.shape.vocab_size));
        if self.shape != expected {
            return Err(format!(
                "artifact shape {:?} does not match the canonical {} shape {:?}",
                self.shape, self.model_size, expected
            ));
        }
        if self.tokenizer_file.is_empty()
            || self.contracts_file.is_empty()
            || self.checkpoint_dir.is_empty()
        {
            return Err("artifact tokenizer/contracts/checkpoint path is blank".into());
        }
        if !self.contract_hash.starts_with("blake3:")
            || !self.training_corpus_hash.starts_with("blake3:")
            || !self.tokenizer_hash.starts_with("blake3:")
            || !self.checkpoint_model_hash.starts_with("blake3:")
            || self
                .checkpoint_optimizer_hash
                .as_ref()
                .is_some_and(|hash| !hash.starts_with("blake3:"))
        {
            return Err("artifact is missing a BLAKE3 content hash".into());
        }
        if self.training_records == 0 {
            return Err("artifact says it was trained on zero records".into());
        }
        if let Some(foundation) = &self.foundation {
            foundation.verify()?;
            if foundation.tokenizer_hash != self.tokenizer_hash {
                return Err(
                    "knowledge artifact tokenizer differs from its foundation lineage".into(),
                );
            }
        }
        if let Some(parent) = &self.adapted_from {
            parent.verify()?;
            if parent.model_size != self.model_size
                || parent.shape != self.shape
                || parent.parameter_count != self.parameter_count
                || parent.tokenizer_hash != self.tokenizer_hash
            {
                return Err(
                    "knowledge adaptation changed the parent architecture or tokenizer".into(),
                );
            }
            let foundation_hash = self
                .foundation
                .as_ref()
                .map(|foundation| foundation.artifact_hash.as_str());
            if foundation_hash != Some(parent.foundation_artifact_hash.as_str()) {
                return Err(
                    "knowledge adaptation did not preserve parent foundation lineage".into(),
                );
            }
        }
        if self.adapted_from.is_some() && self.steered_from.is_some() {
            return Err("artifact cannot have two immediate knowledge-model parents".into());
        }
        if let Some(parent) = &self.steered_from {
            parent.verify()?;
            if parent.model_size != self.model_size
                || parent.shape != self.shape
                || parent.parameter_count != self.parameter_count
                || parent.tokenizer_hash != self.tokenizer_hash
                || parent.contract_hash != self.contract_hash
            {
                return Err(
                    "steered knowledge segment changed the parent architecture, tokenizer, or contract"
                        .into(),
                );
            }
            let foundation_hash = self
                .foundation
                .as_ref()
                .map(|foundation| foundation.artifact_hash.as_str());
            if foundation_hash != Some(parent.foundation_artifact_hash.as_str()) {
                return Err("steered knowledge segment did not preserve foundation lineage".into());
            }
            if !self.capability_families.contains(&parent.family) {
                return Err("steered training sibling family is absent from the new corpus".into());
            }
        }
        if let Some(certification) = &self.certification {
            certification.verify_for(self)?;
        }
        Ok(())
    }
}

/// Portable ancestry for a continual-learning SFT run. Only an
/// exam-certified active parent may mint this record; the certification hash
/// makes that fact independently visible without retaining a local path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeAdaptationLineage {
    pub artifact_hash: String,
    pub checkpoint_model_hash: String,
    pub checkpoint_step: usize,
    pub contract_hash: String,
    pub training_corpus_hash: String,
    pub tokenizer_hash: String,
    pub model_size: KnowledgeModelSize,
    pub shape: KnowledgeModelShape,
    pub parameter_count: usize,
    pub certification_evidence_hash: String,
    pub foundation_artifact_hash: String,
}

impl KnowledgeAdaptationLineage {
    pub fn verify(&self) -> Result<(), String> {
        if [
            self.artifact_hash.as_str(),
            self.checkpoint_model_hash.as_str(),
            self.contract_hash.as_str(),
            self.training_corpus_hash.as_str(),
            self.tokenizer_hash.as_str(),
            self.certification_evidence_hash.as_str(),
            self.foundation_artifact_hash.as_str(),
        ]
        .iter()
        .any(|hash| !hash.starts_with("blake3:"))
            || self.parameter_count == 0
        {
            return Err("knowledge adaptation lineage is incomplete".into());
        }
        let expected =
            KnowledgeModelShape::from_config(&self.model_size.config(self.shape.vocab_size));
        if self.shape != expected {
            return Err("knowledge adaptation parent has a non-canonical shape".into());
        }
        Ok(())
    }
}

/// Content-addressed evidence for one candidate-to-candidate adaptive segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeSteeringLineage {
    pub artifact_hash: String,
    pub checkpoint_model_hash: String,
    pub checkpoint_step: usize,
    pub contract_hash: String,
    pub training_corpus_hash: String,
    pub tokenizer_hash: String,
    pub model_size: KnowledgeModelSize,
    pub shape: KnowledgeModelShape,
    pub parameter_count: usize,
    pub foundation_artifact_hash: String,
    pub controller_step_hash: String,
    pub frontier_plan_fingerprint: String,
    pub family: String,
    pub source_exam_fingerprint: String,
    pub sibling_manifest_hash: String,
    pub sibling_sft_hash: String,
    pub base_training_records: usize,
    pub sibling_training_records: usize,
    pub sibling_share_ppm: u64,
}

impl KnowledgeSteeringLineage {
    pub fn verify(&self) -> Result<(), String> {
        if [
            self.artifact_hash.as_str(),
            self.checkpoint_model_hash.as_str(),
            self.contract_hash.as_str(),
            self.training_corpus_hash.as_str(),
            self.tokenizer_hash.as_str(),
            self.foundation_artifact_hash.as_str(),
            self.controller_step_hash.as_str(),
            self.sibling_manifest_hash.as_str(),
            self.sibling_sft_hash.as_str(),
        ]
        .iter()
        .any(|hash| !hash.starts_with("blake3:"))
            || self.parameter_count == 0
            || self.frontier_plan_fingerprint.trim().is_empty()
            || self.family.trim().is_empty()
            || self.source_exam_fingerprint.trim().is_empty()
            || self.base_training_records == 0
            || self.sibling_training_records == 0
        {
            return Err("knowledge steering lineage is incomplete".into());
        }
        let share = validate_knowledge_steering_share(
            self.base_training_records,
            self.sibling_training_records,
        )?;
        if share != self.sibling_share_ppm {
            return Err("knowledge steering lineage has an inconsistent sibling share".into());
        }
        let expected =
            KnowledgeModelShape::from_config(&self.model_size.config(self.shape.vocab_size));
        if self.shape != expected {
            return Err("knowledge steering parent has a non-canonical shape".into());
        }
        Ok(())
    }
}

/// Validate and return the controller-directed sibling share in parts per
/// million. Integer cross multiplication makes the five-point ceiling exact.
pub fn validate_knowledge_steering_share(
    base_records: usize,
    sibling_records: usize,
) -> Result<u64, String> {
    if base_records == 0 || sibling_records == 0 {
        return Err("knowledge steering requires non-empty base and sibling corpora".into());
    }
    let total = (base_records as u128)
        .checked_add(sibling_records as u128)
        .ok_or_else(|| "knowledge steering record count overflow".to_string())?;
    let scaled_sibling = (sibling_records as u128) * 1_000_000u128;
    if scaled_sibling > total * u128::from(MAX_KNOWLEDGE_STEERING_SHARE_PPM) {
        return Err(format!(
            "knowledge steering sibling would occupy more than {} basis points of the segment",
            MAX_KNOWLEDGE_STEERING_SHARE_PPM / 100
        ));
    }
    u64::try_from((scaled_sibling + total / 2) / total)
        .map_err(|_| "knowledge steering share does not fit u64".into())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertifiedKnowledgeExam {
    pub family: String,
    pub tier: grug_data::ExamTier,
    pub difficulty: u8,
    pub samples: u64,
    pub pass_count: u64,
    pub catastrophic_count: u64,
    pub exam_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeModelCertification {
    pub schema_version: u32,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    pub contract_hash: String,
    pub tier: grug_data::ExamTier,
    pub minimum_difficulty: u8,
    pub minimum_tool_difficulty: u8,
    /// Minimum release-gate cases for each stateful knowledge workflow.
    pub minimum_samples_per_family: u64,
    /// Minimum single-turn cases for each captured runtime tool contract.
    pub minimum_tool_samples_per_family: u64,
    /// Causal tokens that must have reached the optimizer before a warm-started
    /// knowledge checkpoint can become an active language model.
    pub minimum_foundation_tokens_seen: u64,
    pub minimum_pass_rate: f64,
    pub maximum_catastrophic_rate: f64,
    /// Canonical knowledge families plus every `tool_contract.<name>` family.
    pub required_families: BTreeSet<String>,
    pub required_tool_families: BTreeSet<String>,
    pub exams: Vec<CertifiedKnowledgeExam>,
    /// BLAKE3 over the normalized certification policy and exam summaries.
    pub evidence_hash: String,
}

impl KnowledgeModelCertification {
    pub fn compute_evidence_hash(&self) -> Result<String, String> {
        let mut exams = self.exams.clone();
        exams.sort_by(|left, right| left.family.cmp(&right.family));
        let payload = serde_json::to_vec(&serde_json::json!({
            "schema_version": self.schema_version,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_model_hash": self.checkpoint_model_hash,
            "contract_hash": self.contract_hash,
            "tier": self.tier,
            "minimum_difficulty": self.minimum_difficulty,
            "minimum_tool_difficulty": self.minimum_tool_difficulty,
            "minimum_samples_per_family": self.minimum_samples_per_family,
            "minimum_tool_samples_per_family": self.minimum_tool_samples_per_family,
            "minimum_foundation_tokens_seen": self.minimum_foundation_tokens_seen,
            "minimum_pass_rate": self.minimum_pass_rate,
            "maximum_catastrophic_rate": self.maximum_catastrophic_rate,
            "required_families": self.required_families,
            "required_tool_families": self.required_tool_families,
            "exams": exams,
        }))
        .map_err(|error| format!("serialize certification evidence: {error}"))?;
        Ok(format!("blake3:{}", blake3::hash(&payload).to_hex()))
    }

    pub fn verify_for(&self, artifact: &KnowledgeModelArtifact) -> Result<(), String> {
        if self.schema_version != KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION {
            return Err(format!(
                "knowledge certification schema {} is unsupported; expected {}",
                self.schema_version, KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION
            ));
        }
        if self.checkpoint_step != artifact.checkpoint_step
            || self.checkpoint_model_hash != artifact.checkpoint_model_hash
            || self.contract_hash != artifact.contract_hash
        {
            return Err("certification is bound to a different checkpoint or tool contract".into());
        }
        if self.minimum_samples_per_family == 0
            || self.minimum_tool_samples_per_family == 0
            || self.minimum_foundation_tokens_seen == 0
            || !(1..=3).contains(&self.minimum_tool_difficulty)
            || !(0.0..=1.0).contains(&self.minimum_pass_rate)
            || !(0.0..=1.0).contains(&self.maximum_catastrophic_rate)
        {
            return Err("certification thresholds are invalid".into());
        }
        let canonical_knowledge_families: BTreeSet<String> =
            grug_data::knowledge_capability_specs()
                .into_iter()
                .map(|capability| capability.id)
                .collect();
        if self.required_tool_families.is_empty()
            || self
                .required_tool_families
                .iter()
                .any(|family| !family.starts_with("tool_contract."))
        {
            return Err("certification has no valid runtime tool-contract families".into());
        }
        let expected_families: BTreeSet<String> = canonical_knowledge_families
            .union(&self.required_tool_families)
            .cloned()
            .collect();
        if self.required_families != expected_families {
            return Err(
                "certification does not cover canonical knowledge and runtime tool families".into(),
            );
        }
        if !artifact
            .capability_families
            .is_superset(&self.required_families)
        {
            return Err(
                "certified artifact was not trained across every required capability family".into(),
            );
        }
        let foundation = artifact.foundation.as_ref().ok_or_else(|| {
            "certified artifact has no verified causal-language foundation".to_string()
        })?;
        if foundation.training_tokens_seen < self.minimum_foundation_tokens_seen {
            return Err(format!(
                "foundation saw {} tokens; certification requires at least {}",
                foundation.training_tokens_seen, self.minimum_foundation_tokens_seen
            ));
        }
        let required_foundation_languages =
            BTreeSet::from(["en".to_string(), "de".to_string(), "fr".to_string()]);
        if foundation.languages != required_foundation_languages {
            return Err(
                "certified foundation language set is not exactly canonical en/de/fr".into(),
            );
        }
        let required_languages: BTreeSet<String> = grug_data::WorkflowLanguage::all()
            .into_iter()
            .map(|language| language.as_str().to_string())
            .collect();
        if !artifact.languages.is_superset(&required_languages) {
            return Err(
                "certified artifact was not trained across English, German and French".into(),
            );
        }
        let required_stages: BTreeSet<String> = grug_data::WorkflowStage::all()
            .into_iter()
            .map(|stage| stage.as_str().to_string())
            .collect();
        if !artifact.stages.is_superset(&required_stages) {
            return Err("certified artifact was not trained across every workflow stage".into());
        }
        let mut observed = BTreeSet::new();
        for exam in &self.exams {
            if !self.required_families.contains(&exam.family) {
                return Err(format!(
                    "certification contains unknown family {:?}",
                    exam.family
                ));
            }
            if !observed.insert(exam.family.clone()) {
                return Err(format!(
                    "certification contains multiple selected exams for {:?}",
                    exam.family
                ));
            }
            let is_tool_family = self.required_tool_families.contains(&exam.family);
            let minimum_samples = if is_tool_family {
                self.minimum_tool_samples_per_family
            } else {
                self.minimum_samples_per_family
            };
            let minimum_difficulty = if is_tool_family {
                self.minimum_tool_difficulty
            } else {
                self.minimum_difficulty
            };
            if exam.tier != self.tier
                || exam.difficulty < minimum_difficulty
                || exam.samples < minimum_samples
                || exam.pass_count > exam.samples
                || exam.catastrophic_count > exam.samples
            {
                return Err(format!(
                    "exam for {:?} does not satisfy certification shape/size constraints",
                    exam.family
                ));
            }
            let denominator = exam.samples as f64;
            if exam.pass_count as f64 / denominator < self.minimum_pass_rate {
                return Err(format!(
                    "exam for {:?} is below the minimum pass rate",
                    exam.family
                ));
            }
            if exam.catastrophic_count as f64 / denominator > self.maximum_catastrophic_rate {
                return Err(format!(
                    "exam for {:?} exceeds the catastrophic failure limit",
                    exam.family
                ));
            }
            if exam.exam_fingerprint.is_empty() {
                return Err("certification has a blank exam fingerprint".into());
            }
        }
        if observed != self.required_families {
            return Err("certification is missing one or more required capability families".into());
        }
        let computed = self.compute_evidence_hash()?;
        if computed != self.evidence_hash {
            return Err(format!(
                "certification evidence hash mismatch: declared {}, computed {computed}",
                self.evidence_hash
            ));
        }
        Ok(())
    }

    /// Bind the serialized certification to the exact tool names in the
    /// content-addressed snapshot loaded beside the model. Artifact-only
    /// validation cannot derive this set because it intentionally stores the
    /// snapshot by hash/path rather than duplicating its mutable contents.
    pub fn verify_contract_coverage(
        &self,
        contracts: &grug_data::ToolContractSnapshot,
    ) -> Result<(), String> {
        if contracts.manifest_hash != self.contract_hash {
            return Err("certification and loaded tool snapshot hashes differ".into());
        }
        let expected: BTreeSet<String> = contracts
            .tools
            .iter()
            .map(|tool| format!("tool_contract.{}", tool.name))
            .collect();
        if self.required_tool_families != expected {
            let missing: Vec<_> = expected
                .difference(&self.required_tool_families)
                .cloned()
                .collect();
            let unexpected: Vec<_> = self
                .required_tool_families
                .difference(&expected)
                .cloned()
                .collect();
            return Err(format!(
                "certification tool coverage differs from the loaded contract snapshot; missing={missing:?} unexpected={unexpected:?}"
            ));
        }
        Ok(())
    }
}

/// Stream a potentially-large artifact through BLAKE3 without reading it all
/// into memory. Runtime loading uses the same helper before deserialization.
pub fn fingerprint_artifact_file(path: impl AsRef<Path>) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

/// Stable, order-sensitive identity of the exact SFT stream consumed by a run.
pub fn fingerprint_knowledge_records(
    records: &[KnowledgeSftRecord],
) -> Result<String, serde_json::Error> {
    let mut hasher = blake3::Hasher::new();
    for record in records {
        serde_json::to_writer(&mut hasher, record)?;
        hasher.update(b"\n");
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeEvalSlice {
    pub samples: usize,
    pub parse_rate: f64,
    pub action_exact_rate: f64,
    pub target_exact_rate: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeTeacherForcedEval {
    pub samples: usize,
    pub parse_rate: f64,
    pub action_exact_rate: f64,
    pub target_exact_rate: f64,
    pub by_family: BTreeMap<String, KnowledgeEvalSlice>,
    pub by_stage: BTreeMap<String, KnowledgeEvalSlice>,
}

#[derive(Default)]
struct EvalCounts {
    samples: usize,
    parsed: usize,
    action_exact: usize,
    target_exact: usize,
}

impl EvalCounts {
    fn observe(&mut self, parsed: bool, action_exact: bool, target_exact: bool) {
        self.samples += 1;
        self.parsed += usize::from(parsed);
        self.action_exact += usize::from(action_exact);
        self.target_exact += usize::from(target_exact);
    }

    fn finish(&self) -> KnowledgeEvalSlice {
        let denominator = self.samples.max(1) as f64;
        KnowledgeEvalSlice {
            samples: self.samples,
            parse_rate: self.parsed as f64 / denominator,
            action_exact_rate: self.action_exact as f64 / denominator,
            target_exact_rate: self.target_exact as f64 / denominator,
        }
    }
}

/// Score isolated action predictions. This is deliberately named
/// "teacher-forced": earlier gold actions/results remain in each record's
/// prompt. Stateful rollout grading remains the responsibility of
/// `grug_data::score_knowledge_exam`.
pub fn score_knowledge_outputs(
    records: &[KnowledgeSftRecord],
    outputs: &[String],
) -> KnowledgeTeacherForcedEval {
    let mut overall = EvalCounts::default();
    let mut families: BTreeMap<String, EvalCounts> = BTreeMap::new();
    let mut stages: BTreeMap<String, EvalCounts> = BTreeMap::new();

    for (index, record) in records.iter().enumerate() {
        let output = outputs.get(index).map(String::as_str).unwrap_or("");
        let expected = KnowledgeAction::parse_runtime(&record.target).ok();
        let predicted = KnowledgeAction::parse_runtime(output).ok();
        let parsed = predicted.is_some();
        let action_exact = predicted.is_some() && predicted == expected;
        let target_exact = output.trim() == record.target.trim();
        overall.observe(parsed, action_exact, target_exact);
        families.entry(record.family.clone()).or_default().observe(
            parsed,
            action_exact,
            target_exact,
        );
        stages
            .entry(stage_label(record))
            .or_default()
            .observe(parsed, action_exact, target_exact);
    }

    let all = overall.finish();
    KnowledgeTeacherForcedEval {
        samples: all.samples,
        parse_rate: all.parse_rate,
        action_exact_rate: all.action_exact_rate,
        target_exact_rate: all.target_exact_rate,
        by_family: families
            .into_iter()
            .map(|(name, counts)| (name, counts.finish()))
            .collect(),
        by_stage: stages
            .into_iter()
            .map(|(name, counts)| (name, counts.finish()))
            .collect(),
    }
}

fn stage_label(record: &KnowledgeSftRecord) -> String {
    serde_json::to_value(record.stage)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .unwrap_or_else(|| format!("{:?}", record.stage))
}

pub fn format_knowledge_prompt(record: &KnowledgeSftRecord) -> &str {
    &record.prompt
}

pub fn encode_knowledge_record<V: Vocab>(
    tokenizer: &V,
    record: &KnowledgeSftRecord,
) -> GrugResult<Encoded> {
    let mut ids = Vec::with_capacity(record.prompt.len() + record.target.len() + 3);
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(&record.prompt)?);
    ids.push(tokenizer.sep_id());
    let target_start = ids.len();
    ids.extend(tokenizer.encode(&record.target)?);
    ids.push(tokenizer.eos_id());
    Ok(Encoded { ids, target_start })
}

pub fn build_knowledge_batch<B: Backend, V: Vocab>(
    tokenizer: &V,
    records: &[KnowledgeSftRecord],
    max_seq_len: usize,
    device: &B::Device,
) -> GrugResult<Batch<B>> {
    let encoded: Vec<Encoded> = records
        .iter()
        .map(|record| encode_knowledge_record(tokenizer, record))
        .collect::<GrugResult<Vec<_>>>()?
        .into_iter()
        .map(|encoded| fit_target_preserving(encoded, max_seq_len.max(3)))
        .collect();
    let longest = encoded.iter().map(|item| item.ids.len()).max().unwrap_or(1);
    let batch_size = encoded.len();
    let seq_len = longest.saturating_sub(1).max(1);
    let pad_id = tokenizer.pad_id();

    let mut input_flat = Vec::with_capacity(batch_size * seq_len);
    let mut target_flat = Vec::with_capacity(batch_size * seq_len);
    let mut loss_mask = Vec::with_capacity(batch_size * seq_len);
    let mut copy_mask = Vec::with_capacity(batch_size * seq_len);
    for encoded in &encoded {
        let mut ids = encoded.ids.clone();
        while ids.len() < longest {
            ids.push(pad_id);
        }
        let prompt_tokens: std::collections::HashSet<u32> = ids
            [..encoded.target_start.min(ids.len())]
            .iter()
            .copied()
            .collect();
        for index in 0..seq_len {
            input_flat.push(i64::from(ids[index]));
            target_flat.push(i64::from(ids[index + 1]));
            let supervised = index + 1 >= encoded.target_start && ids[index + 1] != pad_id;
            loss_mask.push(supervised);
            copy_mask.push(supervised && prompt_tokens.contains(&ids[index + 1]));
        }
    }

    Ok(Batch {
        input_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_flat, [batch_size, seq_len]),
            device,
        ),
        targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_flat, [batch_size, seq_len]),
            device,
        ),
        loss_mask,
        copy_mask,
        batch_size,
        seq_len,
    })
}

/// Keep the beginning of the system/tool catalog, the newest conversation
/// tail, and every target token that fits. The target is never accidentally
/// dropped merely because the captured schemas are long.
fn fit_target_preserving(encoded: Encoded, max_seq_len: usize) -> Encoded {
    if encoded.ids.len() <= max_seq_len {
        return encoded;
    }
    let bos = encoded.ids[0];
    let separator = encoded.ids[encoded.target_start - 1];
    let target = &encoded.ids[encoded.target_start..];
    if target.len() + 2 >= max_seq_len {
        let eos = *target.last().unwrap_or(&separator);
        let mut kept_target = target[..(max_seq_len - 2).min(target.len())].to_vec();
        if let Some(last) = kept_target.last_mut() {
            *last = eos;
        }
        let mut ids = vec![bos, separator];
        ids.extend(kept_target);
        return Encoded {
            ids,
            target_start: 2,
        };
    }

    let prompt = &encoded.ids[..encoded.target_start - 1];
    let prompt_budget = max_seq_len - target.len() - 1;
    let mut kept_prompt = Vec::with_capacity(prompt_budget);
    if prompt.len() <= prompt_budget {
        kept_prompt.extend_from_slice(prompt);
    } else {
        let head = (prompt_budget / 2).max(1);
        let tail = prompt_budget - head;
        kept_prompt.extend_from_slice(&prompt[..head]);
        if tail > 0 {
            kept_prompt.extend_from_slice(&prompt[prompt.len() - tail..]);
        }
    }
    let mut ids = kept_prompt;
    ids.push(separator);
    let target_start = ids.len();
    ids.extend_from_slice(target);
    Encoded { ids, target_start }
}

pub fn generate_knowledge_outputs<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    records: &[KnowledgeSftRecord],
    config: GenerateConfig,
    device: &B::Device,
) -> Vec<String> {
    records
        .iter()
        .map(|record| {
            let prompt_ids = tokenizer.encode(&record.prompt).unwrap_or_default();
            let prompt_ids = compact_knowledge_prompt_ids(&prompt_ids, config);
            generate_from_prompt_ids_cached(model, tokenizer, &prompt_ids, config, device)
        })
        .collect()
}

/// Reserve the requested generation budget, retaining the stable tool-policy
/// head and the newest conversation tail. This mirrors the target-preserving
/// training truncation and prevents autoregressive generation from gradually
/// sliding the tool schemas out of context.
pub fn compact_knowledge_prompt_ids(prompt_ids: &[u32], config: GenerateConfig) -> Vec<u32> {
    let prompt_budget = config
        .max_seq_len
        .saturating_sub(config.max_new_tokens)
        .saturating_sub(2)
        .max(1);
    if prompt_ids.len() <= prompt_budget {
        return prompt_ids.to_vec();
    }
    let head = (prompt_budget / 2).max(1);
    let tail = prompt_budget - head;
    let mut compacted = Vec::with_capacity(prompt_budget);
    compacted.extend_from_slice(&prompt_ids[..head]);
    if tail > 0 {
        compacted.extend_from_slice(&prompt_ids[prompt_ids.len() - tail..]);
    }
    compacted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::tokenizer::{ByteTokenizer, EOS, SEP};
    use crate::train::{init_training, train_step};
    use crate::transformer::TinyTransformerConfig;
    use grug_data::{WorkflowLanguage, WorkflowStage};

    fn record(prompt: &str, target: &str) -> KnowledgeSftRecord {
        KnowledgeSftRecord {
            schema_version: 1,
            id: "record-1".into(),
            episode_id: "episode-1".into(),
            step: 0,
            stage: WorkflowStage::Answer,
            family: "knowledge.test".into(),
            difficulty: 0,
            language: WorkflowLanguage::English,
            contract_hash: "blake3:test".into(),
            tools: Vec::new(),
            conversation: Vec::new(),
            system_context: None,
            prompt: prompt.into(),
            target: target.into(),
        }
    }

    #[test]
    fn encoder_places_target_after_separator() {
        let tokenizer = ByteTokenizer::new();
        let encoded = encode_knowledge_record(&tokenizer, &record("question", "answer")).unwrap();
        assert_eq!(encoded.ids[encoded.target_start - 1], SEP);
        assert_eq!(*encoded.ids.last().unwrap(), EOS);
    }

    #[test]
    fn long_tool_prompt_cannot_truncate_all_supervision() {
        let tokenizer = ByteTokenizer::new();
        let device = Default::default();
        let long_prompt = format!("TOOLS:{}\nlatest question", "schema ".repeat(500));
        let records = [record(&long_prompt, "short target")];
        let batch =
            build_knowledge_batch::<CpuAutodiffBackend, _>(&tokenizer, &records, 128, &device)
                .unwrap();
        assert_eq!(batch.seq_len, 127);
        assert!(batch.loss_mask.iter().any(|supervised| *supervised));
    }

    #[test]
    fn generation_compaction_keeps_policy_head_and_latest_state() {
        let prompt: Vec<u32> = (0..100).collect();
        let compacted = compact_knowledge_prompt_ids(
            &prompt,
            GenerateConfig {
                max_new_tokens: 20,
                max_seq_len: 42,
            },
        );
        assert_eq!(compacted.len(), 20);
        assert_eq!(&compacted[..10], &prompt[..10]);
        assert_eq!(&compacted[10..], &prompt[90..]);
    }

    #[test]
    fn knowledge_batch_runs_the_real_optimizer() {
        let tokenizer = ByteTokenizer::new();
        let device = Default::default();
        let records = vec![
            record("Question A", "answer"),
            record("Question B", "answer"),
        ];
        let batch =
            build_knowledge_batch::<CpuAutodiffBackend, _>(&tokenizer, &records, 128, &device)
                .unwrap();
        let config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 128);
        let (mut model, mut optimizer) = init_training::<CpuAutodiffBackend>(&config, &device);
        let mut losses = Vec::new();
        for _ in 0..20 {
            let (next, loss) = train_step(model, &mut optimizer, &batch, 1e-2);
            model = next;
            losses.push(loss);
        }
        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "expected target-masked loss to fall: {losses:?}"
        );
    }

    #[test]
    fn model_artifact_rejects_shape_drift() {
        let config = KnowledgeModelSize::Small.config(3_000);
        let mut artifact = KnowledgeModelArtifact {
            schema_version: KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION,
            model_size: KnowledgeModelSize::Small,
            shape: KnowledgeModelShape::from_config(&config),
            parameter_count: 30_000_000,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_hash: "blake3:tokenizer".into(),
            contracts_file: "contracts.json".into(),
            checkpoint_dir: "checkpoints/step_00000042".into(),
            checkpoint_step: 42,
            checkpoint_model_hash: "blake3:model".into(),
            checkpoint_optimizer_hash: Some("blake3:optimizer".into()),
            contract_hash: "blake3:contract".into(),
            training_corpus_hash: "blake3:corpus".into(),
            training_records: 100,
            eval_corpus_hash: None,
            capability_families: BTreeSet::new(),
            languages: BTreeSet::new(),
            stages: BTreeSet::new(),
            foundation: None,
            adapted_from: None,
            steered_from: None,
            certification: None,
        };
        artifact.verify().unwrap();
        artifact.shape.max_seq_len -= 1;
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("canonical small shape")
        );
    }

    #[test]
    fn model_artifact_rejects_foundation_tokenizer_drift() {
        let config = KnowledgeModelSize::Small.config(3_000);
        let mut artifact = KnowledgeModelArtifact {
            schema_version: KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION,
            model_size: KnowledgeModelSize::Small,
            shape: KnowledgeModelShape::from_config(&config),
            parameter_count: 30_000_000,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_hash: "blake3:knowledge-tokenizer".into(),
            contracts_file: "contracts.json".into(),
            checkpoint_dir: "checkpoints/step_00000042".into(),
            checkpoint_step: 42,
            checkpoint_model_hash: "blake3:model".into(),
            checkpoint_optimizer_hash: Some("blake3:optimizer".into()),
            contract_hash: "blake3:contract".into(),
            training_corpus_hash: "blake3:corpus".into(),
            training_records: 100,
            eval_corpus_hash: None,
            capability_families: BTreeSet::new(),
            languages: BTreeSet::new(),
            stages: BTreeSet::new(),
            foundation: Some(FoundationLineage {
                artifact_hash: "blake3:foundation-artifact".into(),
                checkpoint_model_hash: "blake3:foundation-model".into(),
                checkpoint_step: 7,
                training_corpus_hash: "blake3:foundation-corpus".into(),
                training_tokens_seen: 1_000_000,
                tokenizer_hash: "blake3:foundation-tokenizer".into(),
                languages: BTreeSet::from(["english".into()]),
                sources: BTreeSet::from(["source@revision|license".into()]),
            }),
            adapted_from: None,
            steered_from: None,
            certification: None,
        };
        assert!(artifact.verify().unwrap_err().contains("tokenizer differs"));
        artifact.tokenizer_hash = "blake3:foundation-tokenizer".into();
        artifact.verify().unwrap();

        artifact.adapted_from = Some(KnowledgeAdaptationLineage {
            artifact_hash: "blake3:active-parent-artifact".into(),
            checkpoint_model_hash: "blake3:active-parent-model".into(),
            checkpoint_step: 41,
            contract_hash: "blake3:parent-contract".into(),
            training_corpus_hash: "blake3:parent-corpus".into(),
            tokenizer_hash: artifact.tokenizer_hash.clone(),
            model_size: artifact.model_size,
            shape: artifact.shape.clone(),
            parameter_count: artifact.parameter_count,
            certification_evidence_hash: "blake3:parent-certification".into(),
            foundation_artifact_hash: artifact.foundation.as_ref().unwrap().artifact_hash.clone(),
        });
        artifact.verify().unwrap();
        artifact.adapted_from.as_mut().unwrap().tokenizer_hash = "blake3:wrong".into();
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("architecture or tokenizer")
        );

        artifact.adapted_from = None;
        artifact
            .capability_families
            .insert("knowledge.provider_fallback".into());
        artifact.steered_from = Some(KnowledgeSteeringLineage {
            artifact_hash: "blake3:candidate-parent-artifact".into(),
            checkpoint_model_hash: "blake3:candidate-parent-model".into(),
            checkpoint_step: 41,
            contract_hash: artifact.contract_hash.clone(),
            training_corpus_hash: "blake3:parent-corpus".into(),
            tokenizer_hash: artifact.tokenizer_hash.clone(),
            model_size: artifact.model_size,
            shape: artifact.shape.clone(),
            parameter_count: artifact.parameter_count,
            foundation_artifact_hash: artifact.foundation.as_ref().unwrap().artifact_hash.clone(),
            controller_step_hash: "blake3:controller".into(),
            frontier_plan_fingerprint: "exam-fnv1a64-0123456789abcdef".into(),
            family: "knowledge.provider_fallback".into(),
            source_exam_fingerprint: "exam-fnv1a64-fedcba9876543210".into(),
            sibling_manifest_hash: "blake3:sibling-manifest".into(),
            sibling_sft_hash: "blake3:sibling-sft".into(),
            base_training_records: 1_900,
            sibling_training_records: 100,
            sibling_share_ppm: 50_000,
        });
        artifact.verify().unwrap();
        artifact.steered_from.as_mut().unwrap().contract_hash = "blake3:other-contract".into();
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("architecture, tokenizer, or contract")
        );
    }

    #[test]
    fn steering_share_is_exactly_bounded_to_five_percentage_points() {
        assert_eq!(
            validate_knowledge_steering_share(1_900, 100).unwrap(),
            50_000
        );
        assert_eq!(
            validate_knowledge_steering_share(9_917, 500).unwrap(),
            47_998
        );
        assert!(validate_knowledge_steering_share(1_899, 100).is_err());
        assert!(validate_knowledge_steering_share(1_900, 0).is_err());
    }

    #[test]
    fn corpus_fingerprint_is_order_and_content_sensitive() {
        let one = record("one", "answer");
        let two = record("two", "answer");
        let first = fingerprint_knowledge_records(&[one.clone(), two.clone()]).unwrap();
        let repeat = fingerprint_knowledge_records(&[one.clone(), two.clone()]).unwrap();
        let reversed = fingerprint_knowledge_records(&[two, one]).unwrap();
        assert_eq!(first, repeat);
        assert_ne!(first, reversed);
    }

    #[test]
    fn certification_is_bound_to_all_capabilities_and_checkpoint() {
        let config = KnowledgeModelSize::Small.config(3_000);
        let knowledge_families: BTreeSet<String> = grug_data::knowledge_capability_specs()
            .into_iter()
            .map(|capability| capability.id)
            .collect();
        let contracts = grug_data::ToolContractSnapshot::from_contracts(vec![
            grug_data::ToolContract {
                name: "search".into(),
                description: "Search".into(),
                parameters: vec![],
                danger: serde_json::Value::Null,
                source: serde_json::Value::Null,
            },
            grug_data::ToolContract {
                name: "write".into(),
                description: "Write".into(),
                parameters: vec![],
                danger: serde_json::Value::Null,
                source: serde_json::Value::Null,
            },
        ])
        .unwrap();
        let required_tool_families = BTreeSet::from([
            "tool_contract.search".to_string(),
            "tool_contract.write".to_string(),
        ]);
        let required_families: BTreeSet<String> = knowledge_families
            .union(&required_tool_families)
            .cloned()
            .collect();
        let exams = required_families
            .iter()
            .enumerate()
            .map(|(index, family)| CertifiedKnowledgeExam {
                family: family.clone(),
                tier: grug_data::ExamTier::Sentinel,
                difficulty: if family.starts_with("tool_contract.") {
                    3
                } else {
                    1
                },
                samples: 100,
                pass_count: 90,
                catastrophic_count: 0,
                exam_fingerprint: format!("exam-{index}"),
            })
            .collect();
        let mut certification = KnowledgeModelCertification {
            schema_version: KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION,
            checkpoint_step: 42,
            checkpoint_model_hash: "blake3:model".into(),
            contract_hash: contracts.manifest_hash.clone(),
            tier: grug_data::ExamTier::Sentinel,
            minimum_difficulty: 1,
            minimum_tool_difficulty: 3,
            minimum_samples_per_family: 100,
            minimum_tool_samples_per_family: 50,
            minimum_foundation_tokens_seen: 500_000_000,
            minimum_pass_rate: 0.85,
            maximum_catastrophic_rate: 0.01,
            required_families: required_families.clone(),
            required_tool_families,
            exams,
            evidence_hash: String::new(),
        };
        certification.evidence_hash = certification.compute_evidence_hash().unwrap();
        let mut artifact = KnowledgeModelArtifact {
            schema_version: KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION,
            model_size: KnowledgeModelSize::Small,
            shape: KnowledgeModelShape::from_config(&config),
            parameter_count: 30_000_000,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_hash: "blake3:tokenizer".into(),
            contracts_file: "contracts.json".into(),
            checkpoint_dir: "checkpoints/step_00000042".into(),
            checkpoint_step: 42,
            checkpoint_model_hash: "blake3:model".into(),
            checkpoint_optimizer_hash: None,
            contract_hash: contracts.manifest_hash.clone(),
            training_corpus_hash: "blake3:corpus".into(),
            training_records: 1_000,
            eval_corpus_hash: Some("blake3:eval".into()),
            capability_families: required_families,
            languages: grug_data::WorkflowLanguage::all()
                .into_iter()
                .map(|language| language.as_str().into())
                .collect(),
            stages: grug_data::WorkflowStage::all()
                .into_iter()
                .map(|stage| stage.as_str().into())
                .collect(),
            foundation: Some(FoundationLineage {
                artifact_hash: "blake3:foundation-artifact".into(),
                checkpoint_model_hash: "blake3:foundation-model".into(),
                checkpoint_step: 49_999,
                training_corpus_hash: "blake3:foundation-corpus".into(),
                training_tokens_seen: 819_200_000,
                tokenizer_hash: "blake3:tokenizer".into(),
                languages: BTreeSet::from(["en".into(), "de".into(), "fr".into()]),
                sources: BTreeSet::from(["source@revision|license".into()]),
            }),
            adapted_from: None,
            steered_from: None,
            certification: Some(certification),
        };
        artifact.verify().unwrap();
        artifact
            .certification
            .as_ref()
            .unwrap()
            .verify_contract_coverage(&contracts)
            .unwrap();
        let foundation = artifact.foundation.take();
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("no verified causal-language foundation")
        );
        artifact.foundation = foundation;
        artifact.checkpoint_model_hash = "blake3:different".into();
        assert!(
            artifact
                .verify()
                .unwrap_err()
                .contains("different checkpoint")
        );
    }

    #[test]
    fn teacher_forced_score_distinguishes_wire_and_semantic_exactness() {
        let tool_target =
            "<tool_call>\n{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}\n</tool_call>";
        let records = vec![
            record("question", tool_target),
            record("question", "grounded"),
        ];
        let outputs = vec![
            "<tool_call>{\"arguments\":{\"q\":\"x\"},\"name\":\"search\"}</tool_call>".into(),
            "".into(),
        ];
        let score = score_knowledge_outputs(&records, &outputs);
        assert_eq!(score.samples, 2);
        assert_eq!(score.parse_rate, 0.5);
        assert_eq!(score.action_exact_rate, 0.5);
        assert_eq!(score.target_exact_rate, 0.0);
    }
}
