//! Canonical, provenance-bearing natural-language corpus records.
//!
//! The knowledge curriculum teaches Akh-Medu workflows, but a model trained
//! only on those synthetic trajectories is not a language model in the useful
//! sense.  This module defines the boundary for the preceding causal-language
//! stage.  Records retain the upstream dataset revision and licence, while a
//! prepared manifest binds the exact train/evaluation shards and records the
//! decontamination policy applied between them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FOUNDATION_RECORD_SCHEMA_VERSION: u32 = 1;
pub const FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const FOUNDATION_CORPUS_GENERATOR_VERSION: &str = "akh-foundation-corpus-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoundationSplit {
    Training,
    Evaluation,
}

impl FoundationSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Evaluation => "evaluation",
        }
    }
}

/// One normalized document entering causal pretraining.
///
/// `id` must be stable at the declared source revision. The text is kept
/// verbatim; normalization is used only for duplicate/leakage detection by
/// the preparer so the published corpus remains auditable against upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationRecord {
    pub schema_version: u32,
    pub id: String,
    pub source: String,
    pub source_revision: String,
    pub license: String,
    pub language: String,
    pub text: String,
}

impl FoundationRecord {
    pub fn validate(&self) -> Result<(), FoundationCorpusError> {
        if self.schema_version != FOUNDATION_RECORD_SCHEMA_VERSION {
            return Err(FoundationCorpusError::UnsupportedRecordVersion {
                actual: self.schema_version,
                expected: FOUNDATION_RECORD_SCHEMA_VERSION,
            });
        }
        for (field, value) in [
            ("id", self.id.as_str()),
            ("source", self.source.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("license", self.license.as_str()),
            ("language", self.language.as_str()),
            ("text", self.text.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(FoundationCorpusError::BlankRecordField {
                    id: self.id.clone(),
                    field,
                });
            }
        }
        if self.text.contains('\0') {
            return Err(FoundationCorpusError::NulText {
                id: self.id.clone(),
            });
        }
        Ok(())
    }
}

/// Feed one record into the order-sensitive corpus identity used by prepared
/// manifests. Length delimiting prevents concatenation ambiguities.
pub fn update_foundation_fingerprint(
    hasher: &mut blake3::Hasher,
    record: &FoundationRecord,
) -> Result<(), serde_json::Error> {
    let encoded = serde_json::to_vec(record)?;
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationShard {
    /// Relative to the directory containing `foundation-corpus.json`.
    pub file: String,
    pub file_hash: String,
    pub records: u64,
    pub jsonl_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FoundationSourceSummary {
    pub split: FoundationSplit,
    pub source: String,
    pub source_revision: String,
    pub license: String,
    pub language: String,
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationSplitManifest {
    pub records: u64,
    pub jsonl_bytes: u64,
    /// Order-sensitive BLAKE3 over length-delimited canonical record JSON.
    pub fingerprint: String,
    pub shards: Vec<FoundationShard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationDecontaminationSummary {
    pub method: String,
    pub word_ngram_size: usize,
    pub evaluation_texts: u64,
    pub evaluation_ngrams: u64,
    pub removed_duplicate_text: u64,
    pub removed_evaluation_exact: u64,
    pub removed_evaluation_ngram: u64,
}

/// Immutable output of `prepare_foundation`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundationCorpusManifest {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub training: FoundationSplitManifest,
    pub evaluation: FoundationSplitManifest,
    /// Counts across both splits, useful for catching a drowned-out language
    /// before an expensive tokenizer/model run starts.
    pub languages: BTreeMap<String, u64>,
    pub sources: Vec<FoundationSourceSummary>,
    pub decontamination: FoundationDecontaminationSummary,
}

impl FoundationCorpusManifest {
    pub fn verify(&self) -> Result<(), FoundationCorpusError> {
        if self.schema_version != FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION {
            return Err(FoundationCorpusError::UnsupportedManifestVersion {
                actual: self.schema_version,
                expected: FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION,
            });
        }
        if self.generator_version != FOUNDATION_CORPUS_GENERATOR_VERSION {
            return Err(FoundationCorpusError::GeneratorVersion {
                actual: self.generator_version.clone(),
                expected: FOUNDATION_CORPUS_GENERATOR_VERSION,
            });
        }
        if self.run_id.trim().is_empty() {
            return Err(FoundationCorpusError::InvalidManifest(
                "run_id must not be blank".into(),
            ));
        }
        verify_split(FoundationSplit::Training, &self.training)?;
        verify_split(FoundationSplit::Evaluation, &self.evaluation)?;

        let mut paths = BTreeSet::new();
        for shard in self.training.shards.iter().chain(&self.evaluation.shards) {
            if !paths.insert(shard.file.as_str()) {
                return Err(FoundationCorpusError::InvalidManifest(format!(
                    "duplicate shard path {:?}",
                    shard.file
                )));
            }
        }

        let source_records: u64 = self.sources.iter().map(|source| source.records).sum();
        let total_records = self.training.records + self.evaluation.records;
        if source_records != total_records {
            return Err(FoundationCorpusError::InvalidManifest(format!(
                "source summaries count {source_records} records, splits count {total_records}"
            )));
        }
        if self.sources.iter().any(|source| {
            source.source.trim().is_empty()
                || source.source_revision.trim().is_empty()
                || source.license.trim().is_empty()
                || source.language.trim().is_empty()
                || source.records == 0
        }) {
            return Err(FoundationCorpusError::InvalidManifest(
                "source summaries must have non-blank lineage and positive counts".into(),
            ));
        }
        let language_records: u64 = self.languages.values().sum();
        if language_records != total_records || self.languages.is_empty() {
            return Err(FoundationCorpusError::InvalidManifest(format!(
                "language counts sum to {language_records}, expected {total_records}"
            )));
        }
        if self.decontamination.word_ngram_size < 2 || self.decontamination.method.trim().is_empty()
        {
            return Err(FoundationCorpusError::InvalidManifest(
                "decontamination policy is invalid".into(),
            ));
        }
        Ok(())
    }
}

fn verify_split(
    split: FoundationSplit,
    manifest: &FoundationSplitManifest,
) -> Result<(), FoundationCorpusError> {
    if manifest.records == 0 || manifest.shards.is_empty() {
        return Err(FoundationCorpusError::InvalidManifest(format!(
            "{} split must contain records and shards",
            split.as_str()
        )));
    }
    if !manifest.fingerprint.starts_with("blake3:") {
        return Err(FoundationCorpusError::InvalidManifest(format!(
            "{} fingerprint is not BLAKE3",
            split.as_str()
        )));
    }
    let records: u64 = manifest.shards.iter().map(|shard| shard.records).sum();
    let bytes: u64 = manifest.shards.iter().map(|shard| shard.jsonl_bytes).sum();
    if records != manifest.records || bytes != manifest.jsonl_bytes {
        return Err(FoundationCorpusError::InvalidManifest(format!(
            "{} shard totals are records={records}, bytes={bytes}; split says records={}, bytes={}",
            split.as_str(),
            manifest.records,
            manifest.jsonl_bytes
        )));
    }
    for shard in &manifest.shards {
        if shard.records == 0
            || shard.jsonl_bytes == 0
            || !shard.file_hash.starts_with("blake3:")
            || !safe_relative_path(&shard.file)
        {
            return Err(FoundationCorpusError::InvalidManifest(format!(
                "{} has an invalid shard {:?}",
                split.as_str(),
                shard.file
            )));
        }
    }
    Ok(())
}

pub fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[derive(Debug, Error)]
pub enum FoundationCorpusError {
    #[error("foundation record schema {actual} is unsupported; expected {expected}")]
    UnsupportedRecordVersion { actual: u32, expected: u32 },
    #[error("foundation corpus manifest schema {actual} is unsupported; expected {expected}")]
    UnsupportedManifestVersion { actual: u32, expected: u32 },
    #[error("foundation corpus generator {actual:?} is unsupported; expected {expected}")]
    GeneratorVersion {
        actual: String,
        expected: &'static str,
    },
    #[error("foundation record {id:?} has blank field {field}")]
    BlankRecordField { id: String, field: &'static str },
    #[error("foundation record {id:?} contains a NUL byte")]
    NulText { id: String },
    #[error("invalid foundation corpus manifest: {0}")]
    InvalidManifest(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(file: &str) -> FoundationSplitManifest {
        FoundationSplitManifest {
            records: 1,
            jsonl_bytes: 100,
            fingerprint: "blake3:split".into(),
            shards: vec![FoundationShard {
                file: file.into(),
                file_hash: "blake3:file".into(),
                records: 1,
                jsonl_bytes: 100,
            }],
        }
    }

    fn manifest() -> FoundationCorpusManifest {
        FoundationCorpusManifest {
            schema_version: FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION,
            generator_version: FOUNDATION_CORPUS_GENERATOR_VERSION.into(),
            run_id: "test".into(),
            training: split("train/part-00000.jsonl"),
            evaluation: split("evaluation/part-00000.jsonl"),
            languages: BTreeMap::from([("english".into(), 2)]),
            sources: vec![
                FoundationSourceSummary {
                    split: FoundationSplit::Training,
                    source: "example/source".into(),
                    source_revision: "abc".into(),
                    license: "Apache-2.0".into(),
                    language: "english".into(),
                    records: 1,
                },
                FoundationSourceSummary {
                    split: FoundationSplit::Evaluation,
                    source: "example/source".into(),
                    source_revision: "abc".into(),
                    license: "Apache-2.0".into(),
                    language: "english".into(),
                    records: 1,
                },
            ],
            decontamination: FoundationDecontaminationSummary {
                method: "exact_text+word_13gram".into(),
                word_ngram_size: 13,
                evaluation_texts: 1,
                evaluation_ngrams: 1,
                removed_duplicate_text: 0,
                removed_evaluation_exact: 0,
                removed_evaluation_ngram: 0,
            },
        }
    }

    #[test]
    fn manifest_verifies_consistent_totals_and_paths() {
        manifest().verify().unwrap();
    }

    #[test]
    fn manifest_rejects_parent_traversal() {
        let mut value = manifest();
        value.training.shards[0].file = "../secret.jsonl".into();
        assert!(value.verify().is_err());
    }

    #[test]
    fn record_requires_explicit_lineage() {
        let record = FoundationRecord {
            schema_version: FOUNDATION_RECORD_SCHEMA_VERSION,
            id: "doc-1".into(),
            source: "dataset".into(),
            source_revision: "".into(),
            license: "ODC-By-1.0".into(),
            language: "english".into(),
            text: "Useful prose.".into(),
        };
        assert!(record.validate().is_err());
    }
}
