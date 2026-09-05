//! Verified CPU inference loader for tool-native BabyLLM artifacts.
//!
//! Both the offline exam runner and Akh-Medu's `GrugChatBackend` use this
//! implementation, so artifact/path/hash checks and prompt compaction cannot
//! drift between evaluation and deployment.

use std::path::{Component, Path, PathBuf};

use burn::tensor::backend::Backend;
use grug_data::{KnowledgeSftRecord, ToolContractSnapshot, render_knowledge_prompt_with_system};
use serde_json::Value;
use thiserror::Error;

use crate::{
    BpeTokenizer, CpuBackend, GenerateConfig, KnowledgeAdaptationLineage, KnowledgeModelArtifact,
    TinyTransformer, compact_knowledge_prompt_ids, fingerprint_artifact_file,
    generate_from_prompt_ids_cached, load_model_only,
};

#[derive(Debug, Error)]
pub enum KnowledgeRuntimeError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid BabyLLM artifact: {0}")]
    InvalidArtifact(String),
    #[error("unsafe artifact path {path:?}: paths must stay inside the bundle")]
    UnsafePath { path: String },
    #[error("artifact hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Contract(#[from] grug_data::ToolContractError),
    #[error(transparent)]
    Train(#[from] crate::GrugError),
}

/// A fully verified inference model. Serving uses the CPU alias; accelerated
/// exam binaries use the same loader and methods with the configured GPU
/// backend so release matrices do not take days merely because decoding is
/// checkpoint-bound.
pub struct KnowledgeModel<B: Backend> {
    artifact: KnowledgeModelArtifact,
    contracts: ToolContractSnapshot,
    tokenizer: BpeTokenizer,
    model: TinyTransformer<B>,
    device: B::Device,
}

pub type CpuKnowledgeModel = KnowledgeModel<CpuBackend>;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub type GpuKnowledgeModel = KnowledgeModel<crate::GpuBackend>;

/// Hash- and path-verified model bundle shared by serving, offline evaluation,
/// and continual-learning warm starts. Weight deserialization remains generic
/// over the caller's Burn backend.
pub struct VerifiedKnowledgeBundle {
    artifact: KnowledgeModelArtifact,
    artifact_hash: String,
    contracts: ToolContractSnapshot,
    tokenizer: BpeTokenizer,
    tokenizer_path: PathBuf,
    checkpoint_dir: PathBuf,
}

impl VerifiedKnowledgeBundle {
    pub fn artifact(&self) -> &KnowledgeModelArtifact {
        &self.artifact
    }

    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    pub fn contracts(&self) -> &ToolContractSnapshot {
        &self.contracts
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    pub fn adaptation_lineage(&self) -> Result<KnowledgeAdaptationLineage, KnowledgeRuntimeError> {
        let certification = self.artifact.certification.as_ref().ok_or_else(|| {
            KnowledgeRuntimeError::InvalidArtifact(
                "only an exam-certified active model can be an adaptation parent".into(),
            )
        })?;
        let foundation = self.artifact.foundation.as_ref().ok_or_else(|| {
            KnowledgeRuntimeError::InvalidArtifact(
                "adaptation parent has no verified causal foundation".into(),
            )
        })?;
        let lineage = KnowledgeAdaptationLineage {
            artifact_hash: self.artifact_hash.clone(),
            checkpoint_model_hash: self.artifact.checkpoint_model_hash.clone(),
            checkpoint_step: self.artifact.checkpoint_step,
            contract_hash: self.artifact.contract_hash.clone(),
            training_corpus_hash: self.artifact.training_corpus_hash.clone(),
            tokenizer_hash: self.artifact.tokenizer_hash.clone(),
            model_size: self.artifact.model_size,
            shape: self.artifact.shape.clone(),
            parameter_count: self.artifact.parameter_count,
            certification_evidence_hash: certification.evidence_hash.clone(),
            foundation_artifact_hash: foundation.artifact_hash.clone(),
        };
        lineage
            .verify()
            .map_err(KnowledgeRuntimeError::InvalidArtifact)?;
        Ok(lineage)
    }

    pub fn load_model_only<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<TinyTransformer<B>, KnowledgeRuntimeError> {
        let model = self.artifact.shape.config().init::<B>(device);
        let (model, checkpoint_step) = load_model_only(&self.checkpoint_dir, model, device)?;
        if checkpoint_step != self.artifact.checkpoint_step {
            return Err(KnowledgeRuntimeError::InvalidArtifact(format!(
                "checkpoint metadata says step {checkpoint_step}, artifact says {}",
                self.artifact.checkpoint_step
            )));
        }
        let actual_parameters = model.num_params();
        if actual_parameters != self.artifact.parameter_count {
            return Err(KnowledgeRuntimeError::InvalidArtifact(format!(
                "loaded model has {actual_parameters} parameters, artifact says {}",
                self.artifact.parameter_count
            )));
        }
        Ok(model)
    }
}

pub fn verify_active_knowledge_bundle(
    bundle: impl AsRef<Path>,
) -> Result<VerifiedKnowledgeBundle, KnowledgeRuntimeError> {
    verify_knowledge_bundle(bundle.as_ref(), "active-model.json", true)
}

pub fn verify_candidate_knowledge_bundle(
    bundle: impl AsRef<Path>,
) -> Result<VerifiedKnowledgeBundle, KnowledgeRuntimeError> {
    verify_knowledge_bundle(bundle.as_ref(), "candidate-model.json", false)
}

impl KnowledgeModel<CpuBackend> {
    /// Load only an exam-certified `active-model.json`.
    pub fn load_active(bundle: impl AsRef<Path>) -> Result<Self, KnowledgeRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "active-model.json",
            true,
            Default::default(),
        )
    }

    /// Load `candidate-model.json` for offline evaluation; never use this path
    /// to serve an Akh workspace.
    pub fn load_candidate(bundle: impl AsRef<Path>) -> Result<Self, KnowledgeRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "candidate-model.json",
            false,
            Default::default(),
        )
    }
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl KnowledgeModel<crate::GpuBackend> {
    pub fn load_active_gpu(bundle: impl AsRef<Path>) -> Result<Self, KnowledgeRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "active-model.json",
            true,
            crate::gpu_device(),
        )
    }

    pub fn load_candidate_gpu(bundle: impl AsRef<Path>) -> Result<Self, KnowledgeRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "candidate-model.json",
            false,
            crate::gpu_device(),
        )
    }
}

impl<B: Backend> KnowledgeModel<B> {
    fn load_manifest_with_device(
        bundle: &Path,
        manifest_name: &str,
        require_certification: bool,
        device: B::Device,
    ) -> Result<Self, KnowledgeRuntimeError> {
        let verified = verify_knowledge_bundle(bundle, manifest_name, require_certification)?;
        let model = verified.load_model_only(&device)?;
        let VerifiedKnowledgeBundle {
            artifact,
            contracts,
            tokenizer,
            ..
        } = verified;
        Ok(Self {
            artifact,
            contracts,
            tokenizer,
            model,
            device,
        })
    }

    pub fn artifact(&self) -> &KnowledgeModelArtifact {
        &self.artifact
    }

    pub fn contracts(&self) -> &ToolContractSnapshot {
        &self.contracts
    }

    pub fn encode_len(&self, text: &str) -> Result<usize, KnowledgeRuntimeError> {
        Ok(self.tokenizer.encode(text)?.len())
    }

    /// Generate one runtime action through the canonical corpus prompt.
    /// Returns `(raw_output, retained_prompt_tokens, completion_tokens)`.
    pub fn generate_turn(
        &self,
        tools: &[Value],
        conversation: &[Value],
        max_new_tokens: usize,
    ) -> Result<(String, usize, usize), KnowledgeRuntimeError> {
        self.generate_turn_with_system(tools, conversation, None, max_new_tokens)
    }

    pub fn generate_turn_with_system(
        &self,
        tools: &[Value],
        conversation: &[Value],
        system_context: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<(String, usize, usize), KnowledgeRuntimeError> {
        let maximum = self.artifact.shape.max_seq_len.saturating_sub(3);
        if max_new_tokens == 0 || max_new_tokens > maximum {
            return Err(KnowledgeRuntimeError::InvalidArtifact(format!(
                "max_new_tokens must be in 1..={maximum}, got {max_new_tokens}"
            )));
        }
        let prompt = render_knowledge_prompt_with_system(tools, conversation, system_context);
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let config = GenerateConfig {
            max_new_tokens,
            max_seq_len: self.artifact.shape.max_seq_len,
        };
        let compacted = compact_knowledge_prompt_ids(&prompt_ids, config);
        let output = generate_from_prompt_ids_cached(
            &self.model,
            &self.tokenizer,
            &compacted,
            config,
            &self.device,
        );
        let completion_tokens = self.tokenizer.encode(&output)?.len();
        Ok((output, compacted.len() + 2, completion_tokens))
    }

    pub fn generate_records(
        &self,
        records: &[KnowledgeSftRecord],
        max_new_tokens: usize,
    ) -> Result<Vec<String>, KnowledgeRuntimeError> {
        records
            .iter()
            .map(|record| {
                self.generate_turn_with_system(
                    &record.tools,
                    &record.conversation,
                    record.system_context.as_deref(),
                    max_new_tokens,
                )
                .map(|(output, _, _)| output)
            })
            .collect()
    }
}

fn verify_knowledge_bundle(
    bundle: &Path,
    manifest_name: &str,
    require_certification: bool,
) -> Result<VerifiedKnowledgeBundle, KnowledgeRuntimeError> {
    let manifest_path = resolve_bundle_child(bundle, manifest_name)?;
    let artifact: KnowledgeModelArtifact = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&manifest_path).map_err(|source| KnowledgeRuntimeError::Read {
            path: manifest_path.display().to_string(),
            source,
        })?,
    ))
    .map_err(|source| KnowledgeRuntimeError::Parse {
        path: manifest_path.display().to_string(),
        source,
    })?;
    artifact
        .verify()
        .map_err(KnowledgeRuntimeError::InvalidArtifact)?;
    if require_certification && artifact.certification.is_none() {
        return Err(KnowledgeRuntimeError::InvalidArtifact(
            "active model has no valid knowledge-and-tool exam certification".into(),
        ));
    }
    let artifact_hash = fingerprint_artifact_file(&manifest_path).map_err(|source| {
        KnowledgeRuntimeError::Read {
            path: manifest_path.display().to_string(),
            source,
        }
    })?;

    let tokenizer_path = resolve_bundle_child(bundle, &artifact.tokenizer_file)?;
    verify_file_hash(&tokenizer_path, &artifact.tokenizer_hash)?;
    let tokenizer = BpeTokenizer::load(&tokenizer_path)?;
    if tokenizer.vocab_size() != artifact.shape.vocab_size {
        return Err(KnowledgeRuntimeError::InvalidArtifact(format!(
            "tokenizer vocabulary {} does not match artifact shape {}",
            tokenizer.vocab_size(),
            artifact.shape.vocab_size
        )));
    }

    let contracts_path = resolve_bundle_child(bundle, &artifact.contracts_file)?;
    let contracts: ToolContractSnapshot = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&contracts_path).map_err(|source| KnowledgeRuntimeError::Read {
            path: contracts_path.display().to_string(),
            source,
        })?,
    ))
    .map_err(|source| KnowledgeRuntimeError::Parse {
        path: contracts_path.display().to_string(),
        source,
    })?;
    contracts.verify_integrity()?;
    if contracts.manifest_hash != artifact.contract_hash {
        return Err(KnowledgeRuntimeError::InvalidArtifact(format!(
            "contracts file identifies {}, artifact declares {}",
            contracts.manifest_hash, artifact.contract_hash
        )));
    }
    if let Some(certification) = &artifact.certification {
        certification
            .verify_contract_coverage(&contracts)
            .map_err(KnowledgeRuntimeError::InvalidArtifact)?;
    }

    let checkpoint_dir = resolve_bundle_child(bundle, &artifact.checkpoint_dir)?;
    let model_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("model.mpk"))?;
    verify_file_hash(&model_path, &artifact.checkpoint_model_hash)?;
    if let Some(expected) = &artifact.checkpoint_optimizer_hash {
        let optimizer_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("optim.mpk"))?;
        verify_file_hash(&optimizer_path, expected)?;
    }

    Ok(VerifiedKnowledgeBundle {
        artifact,
        artifact_hash,
        contracts,
        tokenizer,
        tokenizer_path,
        checkpoint_dir,
    })
}

fn resolve_bundle_child(bundle: &Path, relative: &str) -> Result<PathBuf, KnowledgeRuntimeError> {
    let relative_path = Path::new(relative);
    if relative_path.as_os_str().is_empty()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(KnowledgeRuntimeError::UnsafePath {
            path: relative.into(),
        });
    }
    let canonical_bundle = bundle
        .canonicalize()
        .map_err(|source| KnowledgeRuntimeError::Read {
            path: bundle.display().to_string(),
            source,
        })?;
    let unresolved = canonical_bundle.join(relative_path);
    let child = ensure_bundle_descendant(&canonical_bundle, &unresolved)?;
    Ok(child)
}

fn ensure_bundle_descendant(
    bundle: &Path,
    unresolved: &Path,
) -> Result<PathBuf, KnowledgeRuntimeError> {
    let canonical_bundle = bundle
        .canonicalize()
        .map_err(|source| KnowledgeRuntimeError::Read {
            path: bundle.display().to_string(),
            source,
        })?;
    let child = unresolved
        .canonicalize()
        .map_err(|source| KnowledgeRuntimeError::Read {
            path: unresolved.display().to_string(),
            source,
        })?;
    if !child.starts_with(&canonical_bundle) {
        return Err(KnowledgeRuntimeError::UnsafePath {
            path: unresolved.display().to_string(),
        });
    }
    Ok(child)
}

fn verify_file_hash(path: &Path, expected: &str) -> Result<(), KnowledgeRuntimeError> {
    let actual = fingerprint_artifact_file(path).map_err(|source| KnowledgeRuntimeError::Read {
        path: path.display().to_string(),
        source,
    })?;
    if actual != expected {
        return Err(KnowledgeRuntimeError::HashMismatch {
            path: path.display().to_string(),
            expected: expected.into(),
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_relative_path_rejects_parent_and_absolute_components() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            resolve_bundle_child(directory.path(), "../outside"),
            Err(KnowledgeRuntimeError::UnsafePath { .. })
        ));
        assert!(matches!(
            resolve_bundle_child(directory.path(), "/tmp/outside"),
            Err(KnowledgeRuntimeError::UnsafePath { .. })
        ));
    }
}
