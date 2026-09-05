//! Causal-language pretraining primitives and verified foundation artifacts.
//!
//! Foundation training and knowledge/tool SFT deliberately share the exact
//! tokenizer and transformer shape.  A knowledge run may warm-start only from
//! a bundle that passes the path, hash, corpus-lineage, checkpoint-step and
//! parameter-count checks in this module.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use grug_data::{FoundationCorpusManifest, FoundationSourceSummary};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::checkpoint::load_model_only;
use crate::knowledge_train::{KnowledgeModelShape, KnowledgeModelSize, fingerprint_artifact_file};
use crate::train::{Batch, masked_cross_entropy};
use crate::{BpeTokenizer, GrugError, TinyTransformer};

pub const FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const FOUNDATION_MODEL_MANIFEST: &str = "foundation-model.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationModelArtifact {
    pub schema_version: u32,
    pub model_size: KnowledgeModelSize,
    pub shape: KnowledgeModelShape,
    pub parameter_count: usize,
    pub tokenizer_file: String,
    pub tokenizer_hash: String,
    pub corpus_manifest_file: String,
    pub corpus_manifest_hash: String,
    pub training_corpus_hash: String,
    pub evaluation_corpus_hash: String,
    pub training_records: u64,
    pub evaluation_records: u64,
    /// Tokens in the packed corpus, including BOS/EOS document boundaries.
    pub corpus_tokens: u64,
    pub evaluation_tokens: u64,
    /// Actual causal tokens presented to the optimizer, including repeats.
    pub training_tokens_seen: u64,
    pub languages: BTreeSet<String>,
    /// Stable `source@revision|license` identifiers.
    pub sources: BTreeSet<String>,
    pub checkpoint_dir: String,
    pub checkpoint_step: usize,
    pub checkpoint_model_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_optimizer_hash: Option<String>,
}

impl FoundationModelArtifact {
    pub fn verify(&self) -> Result<(), String> {
        if self.schema_version != FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION {
            return Err(format!(
                "foundation artifact schema {} is unsupported; expected {}",
                self.schema_version, FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION
            ));
        }
        let expected =
            KnowledgeModelShape::from_config(&self.model_size.config(self.shape.vocab_size));
        if self.shape != expected {
            return Err(format!(
                "foundation artifact shape {:?} does not match canonical {} shape {:?}",
                self.shape, self.model_size, expected
            ));
        }
        if self.parameter_count == 0
            || self.training_records == 0
            || self.evaluation_records == 0
            || self.corpus_tokens < 2
            || self.evaluation_tokens < 2
            || self.training_tokens_seen == 0
            || self.languages.is_empty()
            || self.sources.is_empty()
        {
            return Err("foundation artifact has empty training metadata".into());
        }
        if [
            self.tokenizer_file.as_str(),
            self.corpus_manifest_file.as_str(),
            self.checkpoint_dir.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            return Err("foundation artifact has a blank file path".into());
        }
        if [
            self.tokenizer_hash.as_str(),
            self.corpus_manifest_hash.as_str(),
            self.training_corpus_hash.as_str(),
            self.evaluation_corpus_hash.as_str(),
            self.checkpoint_model_hash.as_str(),
        ]
        .iter()
        .any(|hash| !hash.starts_with("blake3:"))
            || self
                .checkpoint_optimizer_hash
                .as_ref()
                .is_some_and(|hash| !hash.starts_with("blake3:"))
        {
            return Err("foundation artifact is missing a BLAKE3 content hash".into());
        }
        Ok(())
    }

    pub fn lineage(&self, artifact_hash: String) -> FoundationLineage {
        FoundationLineage {
            artifact_hash,
            checkpoint_model_hash: self.checkpoint_model_hash.clone(),
            checkpoint_step: self.checkpoint_step,
            training_corpus_hash: self.training_corpus_hash.clone(),
            training_tokens_seen: self.training_tokens_seen,
            tokenizer_hash: self.tokenizer_hash.clone(),
            languages: self.languages.clone(),
            sources: self.sources.clone(),
        }
    }
}

/// Portable lineage embedded into every knowledge-SFT candidate warm-started
/// from a foundation bundle. It intentionally contains hashes rather than a
/// machine-local bundle path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationLineage {
    pub artifact_hash: String,
    pub checkpoint_model_hash: String,
    pub checkpoint_step: usize,
    pub training_corpus_hash: String,
    pub training_tokens_seen: u64,
    pub tokenizer_hash: String,
    pub languages: BTreeSet<String>,
    pub sources: BTreeSet<String>,
}

impl FoundationLineage {
    pub fn verify(&self) -> Result<(), String> {
        if [
            self.artifact_hash.as_str(),
            self.checkpoint_model_hash.as_str(),
            self.training_corpus_hash.as_str(),
            self.tokenizer_hash.as_str(),
        ]
        .iter()
        .any(|hash| !hash.starts_with("blake3:"))
            || self.training_tokens_seen == 0
            || self.languages.is_empty()
            || self.sources.is_empty()
        {
            return Err("knowledge artifact has invalid foundation lineage".into());
        }
        Ok(())
    }
}

pub struct VerifiedFoundationBundle {
    artifact: FoundationModelArtifact,
    artifact_hash: String,
    tokenizer_path: PathBuf,
    checkpoint_dir: PathBuf,
}

impl VerifiedFoundationBundle {
    pub fn artifact(&self) -> &FoundationModelArtifact {
        &self.artifact
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub fn lineage(&self) -> FoundationLineage {
        self.artifact.lineage(self.artifact_hash.clone())
    }

    pub fn load_model_only<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<TinyTransformer<B>, FoundationError> {
        let model = self.artifact.shape.config().init::<B>(device);
        let (model, step) = load_model_only(&self.checkpoint_dir, model, device)?;
        if step != self.artifact.checkpoint_step {
            return Err(FoundationError::InvalidArtifact(format!(
                "checkpoint metadata says step {step}, artifact says {}",
                self.artifact.checkpoint_step
            )));
        }
        let parameters = model.num_params();
        if parameters != self.artifact.parameter_count {
            return Err(FoundationError::InvalidArtifact(format!(
                "loaded model has {parameters} parameters, artifact says {}",
                self.artifact.parameter_count
            )));
        }
        Ok(model)
    }
}

/// Verify a foundation bundle without trusting paths from its manifest.
pub fn verify_foundation_bundle(
    bundle: impl AsRef<Path>,
) -> Result<VerifiedFoundationBundle, FoundationError> {
    let bundle = bundle.as_ref();
    let artifact_path = resolve_bundle_child(bundle, FOUNDATION_MODEL_MANIFEST)?;
    let artifact: FoundationModelArtifact = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&artifact_path).map_err(|source| FoundationError::Read {
            path: artifact_path.display().to_string(),
            source,
        })?,
    ))
    .map_err(|source| FoundationError::Parse {
        path: artifact_path.display().to_string(),
        source,
    })?;
    artifact
        .verify()
        .map_err(FoundationError::InvalidArtifact)?;
    let artifact_hash = file_hash(&artifact_path)?;

    let tokenizer_path = resolve_bundle_child(bundle, &artifact.tokenizer_file)?;
    verify_file_hash(&tokenizer_path, &artifact.tokenizer_hash)?;
    let tokenizer = BpeTokenizer::load(&tokenizer_path)?;
    if tokenizer.vocab_size() != artifact.shape.vocab_size {
        return Err(FoundationError::InvalidArtifact(format!(
            "tokenizer vocabulary {} does not match model vocabulary {}",
            tokenizer.vocab_size(),
            artifact.shape.vocab_size
        )));
    }

    let corpus_path = resolve_bundle_child(bundle, &artifact.corpus_manifest_file)?;
    verify_file_hash(&corpus_path, &artifact.corpus_manifest_hash)?;
    let corpus: FoundationCorpusManifest = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&corpus_path).map_err(|source| FoundationError::Read {
            path: corpus_path.display().to_string(),
            source,
        })?,
    ))
    .map_err(|source| FoundationError::Parse {
        path: corpus_path.display().to_string(),
        source,
    })?;
    corpus.verify()?;
    let corpus_languages = foundation_training_languages(&corpus);
    let corpus_sources = foundation_training_sources(&corpus);
    if artifact.training_corpus_hash != corpus.training.fingerprint
        || artifact.evaluation_corpus_hash != corpus.evaluation.fingerprint
        || artifact.training_records != corpus.training.records
        || artifact.evaluation_records != corpus.evaluation.records
        || artifact.languages != corpus_languages
        || artifact.sources != corpus_sources
    {
        return Err(FoundationError::InvalidArtifact(
            "foundation artifact metadata does not match its corpus manifest".into(),
        ));
    }

    let checkpoint_dir = resolve_bundle_child(bundle, &artifact.checkpoint_dir)?;
    let model_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("model.mpk"))?;
    verify_file_hash(&model_path, &artifact.checkpoint_model_hash)?;
    if let Some(expected) = &artifact.checkpoint_optimizer_hash {
        let optimizer_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("optim.mpk"))?;
        verify_file_hash(&optimizer_path, expected)?;
    }

    Ok(VerifiedFoundationBundle {
        artifact,
        artifact_hash,
        tokenizer_path,
        checkpoint_dir,
    })
}

pub fn source_identity(source: &FoundationSourceSummary) -> String {
    format!(
        "{}@{}|{}",
        source.source, source.source_revision, source.license
    )
}

pub fn foundation_training_languages(corpus: &FoundationCorpusManifest) -> BTreeSet<String> {
    corpus
        .sources
        .iter()
        .filter(|source| source.split == grug_data::FoundationSplit::Training)
        .map(|source| source.language.clone())
        .collect()
}

pub fn foundation_training_sources(corpus: &FoundationCorpusManifest) -> BTreeSet<String> {
    corpus
        .sources
        .iter()
        .filter(|source| source.split == grug_data::FoundationSplit::Training)
        .map(source_identity)
        .collect()
}

/// Turn packed token windows into the same `Batch` consumed by the guarded
/// optimizer. Every non-padding next token is supervised (causal LM loss).
pub fn build_causal_batch<B: Backend>(
    sequences: &[Vec<u32>],
    pad_id: u32,
    max_seq_len: usize,
    device: &B::Device,
) -> Result<Batch<B>, FoundationError> {
    if sequences.is_empty() || max_seq_len == 0 {
        return Err(FoundationError::InvalidBatch(
            "causal batch and max sequence length must be non-empty".into(),
        ));
    }
    if sequences.iter().any(|sequence| sequence.len() < 2) {
        return Err(FoundationError::InvalidBatch(
            "every causal sequence needs at least two tokens".into(),
        ));
    }
    let longest = sequences
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(2)
        .min(max_seq_len + 1);
    let batch_size = sequences.len();
    let seq_len = longest - 1;
    let mut input_flat = Vec::with_capacity(batch_size * seq_len);
    let mut target_flat = Vec::with_capacity(batch_size * seq_len);
    let mut loss_mask = Vec::with_capacity(batch_size * seq_len);
    for sequence in sequences {
        let mut ids = sequence[..sequence.len().min(longest)].to_vec();
        while ids.len() < longest {
            ids.push(pad_id);
        }
        for index in 0..seq_len {
            input_flat.push(i64::from(ids[index]));
            target_flat.push(i64::from(ids[index + 1]));
            loss_mask.push(ids[index + 1] != pad_id);
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
        copy_mask: vec![false; batch_size * seq_len],
        loss_mask,
        batch_size,
        seq_len,
    })
}

pub fn causal_batch_loss<B: Backend>(model: &TinyTransformer<B>, batch: &Batch<B>) -> f32 {
    use burn::tensor::ElementConversion;

    let mask: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|value| i64::from(*value))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask, [batch.batch_size * batch.seq_len]),
        &batch.input_ids.device(),
    );
    masked_cross_entropy(
        model.forward(batch.input_ids.clone()),
        batch.targets.clone(),
        &mask,
    )
    .into_scalar()
    .elem()
}

fn verify_file_hash(path: &Path, expected: &str) -> Result<(), FoundationError> {
    let actual = file_hash(path)?;
    if actual != expected {
        return Err(FoundationError::HashMismatch {
            path: path.display().to_string(),
            expected: expected.into(),
            actual,
        });
    }
    Ok(())
}

fn file_hash(path: &Path) -> Result<String, FoundationError> {
    fingerprint_artifact_file(path).map_err(|source| FoundationError::Read {
        path: path.display().to_string(),
        source,
    })
}

fn resolve_bundle_child(bundle: &Path, relative: &str) -> Result<PathBuf, FoundationError> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(FoundationError::UnsafePath {
            path: relative.into(),
        });
    }
    ensure_bundle_descendant(bundle, &bundle.join(path))
}

fn ensure_bundle_descendant(bundle: &Path, path: &Path) -> Result<PathBuf, FoundationError> {
    let canonical_bundle = bundle
        .canonicalize()
        .map_err(|source| FoundationError::Read {
            path: bundle.display().to_string(),
            source,
        })?;
    let canonical = path
        .canonicalize()
        .map_err(|source| FoundationError::Read {
            path: path.display().to_string(),
            source,
        })?;
    if !canonical.starts_with(&canonical_bundle) {
        return Err(FoundationError::UnsafePath {
            path: path.display().to_string(),
        });
    }
    Ok(canonical)
}

#[derive(Debug, Error)]
pub enum FoundationError {
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
    #[error("invalid foundation artifact: {0}")]
    InvalidArtifact(String),
    #[error("invalid causal foundation batch: {0}")]
    InvalidBatch(String),
    #[error("unsafe foundation bundle path {path:?}")]
    UnsafePath { path: String },
    #[error("artifact hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Corpus(#[from] grug_data::FoundationCorpusError),
    #[error(transparent)]
    Train(#[from] GrugError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuAutodiffBackend, CpuBackend, TinyTransformerConfig};

    fn artifact() -> FoundationModelArtifact {
        let shape = KnowledgeModelShape::from_config(&KnowledgeModelSize::Small.config(3_000));
        FoundationModelArtifact {
            schema_version: FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION,
            model_size: KnowledgeModelSize::Small,
            shape,
            parameter_count: 30_000_000,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_hash: "blake3:tokenizer".into(),
            corpus_manifest_file: "foundation-corpus.json".into(),
            corpus_manifest_hash: "blake3:manifest".into(),
            training_corpus_hash: "blake3:train".into(),
            evaluation_corpus_hash: "blake3:eval".into(),
            training_records: 10,
            evaluation_records: 2,
            corpus_tokens: 1_000,
            evaluation_tokens: 100,
            training_tokens_seen: 10_000,
            languages: BTreeSet::from(["english".into()]),
            sources: BTreeSet::from(["source@rev|license".into()]),
            checkpoint_dir: "checkpoints/step_00000001".into(),
            checkpoint_step: 1,
            checkpoint_model_hash: "blake3:model".into(),
            checkpoint_optimizer_hash: Some("blake3:optimizer".into()),
        }
    }

    #[test]
    fn artifact_and_lineage_are_self_consistent() {
        let artifact = artifact();
        artifact.verify().unwrap();
        artifact.lineage("blake3:artifact".into()).verify().unwrap();
    }

    #[test]
    fn causal_batch_supervises_non_padding_targets() {
        let device = Default::default();
        let batch = build_causal_batch::<CpuAutodiffBackend>(
            &[vec![1, 2, 3, 4], vec![5, 6]],
            99,
            8,
            &device,
        )
        .unwrap();
        assert_eq!(batch.input_ids.dims(), [2, 3]);
        assert_eq!(batch.loss_mask, vec![true, true, true, true, false, false]);
    }

    #[test]
    fn causal_loss_runs_on_the_non_autodiff_evaluation_backend() {
        let device = Default::default();
        let model = TinyTransformerConfig::spike(16).init::<CpuBackend>(&device);
        let batch =
            build_causal_batch::<CpuBackend>(&[vec![1, 2, 3, 4], vec![5, 6, 7]], 0, 8, &device)
                .unwrap();
        assert!(causal_batch_loss(&model, &batch).is_finite());
    }

    #[test]
    fn bundle_paths_reject_parent_components() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            resolve_bundle_child(directory.path(), "../tokenizer.json"),
            Err(FoundationError::UnsafePath { .. })
        ));
    }
}
