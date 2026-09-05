//! Append-only evidence for model-visible runtime tool-contract drift.
//!
//! A warning line is not enough to drive a later adaptation run: it can be
//! rotated away and cannot reconstruct nested JSON Schema. This journal writes
//! the trained and live endpoint snapshots plus their verified delta under a
//! content-addressed directory. Entries are immutable by API and replayable by
//! `prepare_tool_contract_delta`; metadata-only changes are retained as
//! evidence but explicitly marked as not requiring weight adaptation.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use grug_data::{ToolContractDelta, ToolContractError, ToolContractSnapshot};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

const JOURNAL_SCHEMA_VERSION: u32 = 1;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum ContractDriftJournalError {
    #[error("tool-contract drift is invalid: {0}")]
    Contract(#[from] ToolContractError),
    #[error("contract-drift journal I/O at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("contract-drift journal JSON at {path}: {message}")]
    Json { path: String, message: String },
    #[error("contract-drift journal entry is invalid: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct ContractDriftJournal {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractDriftReceipt {
    pub directory: PathBuf,
    pub delta_hash: String,
    pub model_adaptation_required: bool,
    pub created: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactIdentity {
    file: String,
    file_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotIdentity {
    file: String,
    file_hash: String,
    contract_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct JournalManifest {
    schema_version: u32,
    kind: String,
    previous_contracts: SnapshotIdentity,
    current_contracts: SnapshotIdentity,
    delta: ArtifactIdentity,
    delta_hash: String,
    model_adaptation_required: bool,
}

struct VerifiedEntry {
    previous: ToolContractSnapshot,
    current: ToolContractSnapshot,
    delta: ToolContractDelta,
    manifest: JournalManifest,
}

impl ContractDriftJournal {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ContractDriftJournalError> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|source| io_error(&root, source))?;
        if !root.is_dir() || root.is_symlink() {
            return Err(ContractDriftJournalError::Invalid(format!(
                "journal root is not a regular directory: {}",
                root.display()
            )));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Persist one trained-to-live delta. Identical endpoints produce no entry;
    /// an already-recorded exact entry is verified and returned idempotently.
    pub fn record(
        &self,
        previous: &ToolContractSnapshot,
        current: &ToolContractSnapshot,
    ) -> Result<Option<ContractDriftReceipt>, ContractDriftJournalError> {
        let delta = ToolContractDelta::between(previous, current)?;
        if delta.is_empty() {
            return Ok(None);
        }
        let digest = canonical_digest(&delta.delta_hash)?;
        let destination = self.root.join(format!("delta-{digest}"));
        if destination.exists() {
            return Ok(Some(self.verify_expected(
                &destination,
                previous,
                current,
                &delta,
                false,
            )?));
        }

        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = self.root.join(format!(
            ".delta-{digest}.tmp-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&temporary).map_err(|source| io_error(&temporary, source))?;
        let write_result = self.write_entry(&temporary, previous, current, &delta);
        if let Err(error) = write_result {
            let _ = std::fs::remove_dir_all(&temporary);
            return Err(error);
        }
        match std::fs::rename(&temporary, &destination) {
            Ok(()) => {
                sync_directory(&self.root)?;
                Ok(Some(self.verify_expected(
                    &destination,
                    previous,
                    current,
                    &delta,
                    true,
                )?))
            }
            Err(_source) if destination.exists() => {
                // Another thread/process may have won the same content-addressed
                // write. Remove only our uniquely named temporary directory and
                // accept the winner only after full byte/semantic verification.
                let _ = std::fs::remove_dir_all(&temporary);
                Ok(Some(self.verify_expected(
                    &destination,
                    previous,
                    current,
                    &delta,
                    false,
                )?))
            }
            Err(source) => {
                let _ = std::fs::remove_dir_all(&temporary);
                Err(io_error(&destination, source))
            }
        }
    }

    fn write_entry(
        &self,
        directory: &Path,
        previous: &ToolContractSnapshot,
        current: &ToolContractSnapshot,
        delta: &ToolContractDelta,
    ) -> Result<(), ContractDriftJournalError> {
        let previous_path = directory.join("previous-contracts.json");
        let current_path = directory.join("contracts.json");
        let delta_path = directory.join("contract-delta.json");
        write_json_new(&previous_path, previous)?;
        write_json_new(&current_path, current)?;
        write_json_new(&delta_path, delta)?;
        let manifest = JournalManifest {
            schema_version: JOURNAL_SCHEMA_VERSION,
            kind: "tool_contract_drift".into(),
            previous_contracts: SnapshotIdentity {
                file: "previous-contracts.json".into(),
                file_hash: fingerprint_file(&previous_path)?,
                contract_hash: previous.manifest_hash.clone(),
            },
            current_contracts: SnapshotIdentity {
                file: "contracts.json".into(),
                file_hash: fingerprint_file(&current_path)?,
                contract_hash: current.manifest_hash.clone(),
            },
            delta: ArtifactIdentity {
                file: "contract-delta.json".into(),
                file_hash: fingerprint_file(&delta_path)?,
            },
            delta_hash: delta.delta_hash.clone(),
            model_adaptation_required: delta.model_adaptation_required(),
        };
        // The manifest is the commit marker and is always written last.
        write_json_new(&directory.join("manifest.json"), &manifest)?;
        sync_directory(directory)
    }

    fn verify_expected(
        &self,
        directory: &Path,
        expected_previous: &ToolContractSnapshot,
        expected_current: &ToolContractSnapshot,
        expected_delta: &ToolContractDelta,
        created: bool,
    ) -> Result<ContractDriftReceipt, ContractDriftJournalError> {
        let entry = verify_entry(directory)?;
        if &entry.previous != expected_previous
            || &entry.current != expected_current
            || &entry.delta != expected_delta
        {
            return Err(ContractDriftJournalError::Invalid(format!(
                "content-addressed path {} contains different endpoints",
                directory.display()
            )));
        }
        Ok(ContractDriftReceipt {
            directory: directory.to_path_buf(),
            delta_hash: entry.manifest.delta_hash,
            model_adaptation_required: entry.manifest.model_adaptation_required,
            created,
        })
    }
}

fn verify_entry(directory: &Path) -> Result<VerifiedEntry, ContractDriftJournalError> {
    if !directory.is_dir() || directory.is_symlink() {
        return Err(ContractDriftJournalError::Invalid(format!(
            "entry is not a regular directory: {}",
            directory.display()
        )));
    }
    let expected = BTreeSet::from([
        "contract-delta.json".to_string(),
        "contracts.json".to_string(),
        "manifest.json".to_string(),
        "previous-contracts.json".to_string(),
    ]);
    let mut observed = BTreeSet::new();
    for entry in std::fs::read_dir(directory).map_err(|source| io_error(directory, source))? {
        let entry = entry.map_err(|source| io_error(directory, source))?;
        let file_type = entry
            .file_type()
            .map_err(|source| io_error(&entry.path(), source))?;
        if !file_type.is_file() || file_type.is_symlink() {
            return Err(ContractDriftJournalError::Invalid(format!(
                "entry contains a non-regular file: {}",
                entry.path().display()
            )));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            ContractDriftJournalError::Invalid("entry has a non-UTF-8 filename".into())
        })?;
        observed.insert(name);
    }
    if observed != expected {
        return Err(ContractDriftJournalError::Invalid(format!(
            "entry inventory differs: observed={observed:?}"
        )));
    }

    let manifest: JournalManifest = read_json(&directory.join("manifest.json"))?;
    if manifest.schema_version != JOURNAL_SCHEMA_VERSION
        || manifest.kind != "tool_contract_drift"
        || manifest.previous_contracts.file != "previous-contracts.json"
        || manifest.current_contracts.file != "contracts.json"
        || manifest.delta.file != "contract-delta.json"
    {
        return Err(ContractDriftJournalError::Invalid(
            "entry manifest header or filenames are invalid".into(),
        ));
    }
    let previous_path = directory.join(&manifest.previous_contracts.file);
    let current_path = directory.join(&manifest.current_contracts.file);
    let delta_path = directory.join(&manifest.delta.file);
    for (path, expected_hash) in [
        (&previous_path, &manifest.previous_contracts.file_hash),
        (&current_path, &manifest.current_contracts.file_hash),
        (&delta_path, &manifest.delta.file_hash),
    ] {
        if fingerprint_file(path)? != *expected_hash {
            return Err(ContractDriftJournalError::Invalid(format!(
                "entry file hash differs: {}",
                path.display()
            )));
        }
    }
    let previous: ToolContractSnapshot = read_json(&previous_path)?;
    let current: ToolContractSnapshot = read_json(&current_path)?;
    let delta: ToolContractDelta = read_json(&delta_path)?;
    delta.verify(&previous, &current)?;
    if manifest.previous_contracts.contract_hash != previous.manifest_hash
        || manifest.current_contracts.contract_hash != current.manifest_hash
        || manifest.delta_hash != delta.delta_hash
        || manifest.model_adaptation_required != delta.model_adaptation_required()
    {
        return Err(ContractDriftJournalError::Invalid(
            "entry manifest differs from verified endpoints/delta".into(),
        ));
    }
    let digest = canonical_digest(&delta.delta_hash)?;
    if directory.file_name().and_then(|name| name.to_str())
        != Some(format!("delta-{digest}").as_str())
    {
        return Err(ContractDriftJournalError::Invalid(
            "entry directory is not named by its delta hash".into(),
        ));
    }
    Ok(VerifiedEntry {
        previous,
        current,
        delta,
        manifest,
    })
}

fn canonical_digest(hash: &str) -> Result<&str, ContractDriftJournalError> {
    let Some(digest) = hash.strip_prefix("blake3:") else {
        return Err(ContractDriftJournalError::Invalid(
            "delta hash is not BLAKE3".into(),
        ));
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ContractDriftJournalError::Invalid(
            "delta hash is not canonical lowercase hex".into(),
        ));
    }
    Ok(digest)
}

fn write_json_new<T: Serialize>(path: &Path, value: &T) -> Result<(), ContractDriftJournalError> {
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|error| ContractDriftJournalError::Json {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error(path, source))?;
    file.write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .map_err(|source| io_error(path, source))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, ContractDriftJournalError> {
    serde_json::from_reader(BufReader::new(
        File::open(path).map_err(|source| io_error(path, source))?,
    ))
    .map_err(|error| ContractDriftJournalError::Json {
        path: path.display().to_string(),
        message: error.to_string(),
    })
}

fn fingerprint_file(path: &Path) -> Result<String, ContractDriftJournalError> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| io_error(path, source))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn sync_directory(path: &Path) -> Result<(), ContractDriftJournalError> {
    #[cfg(unix)]
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error(path, source))?;
    Ok(())
}

fn io_error(path: &Path, source: std::io::Error) -> ContractDriftJournalError {
    ContractDriftJournalError::Io {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grug_data::{ContractParameter, ToolContract};
    use serde_json::Value;

    fn snapshot(description: &str, danger: Value) -> ToolContractSnapshot {
        ToolContractSnapshot::from_contracts(vec![ToolContract {
            name: "search".into(),
            description: description.into(),
            parameters: vec![ContractParameter {
                name: "query".into(),
                description: "search query".into(),
                required: true,
                param_type: "string".into(),
                schema: Some(serde_json::json!({"type":"string"})),
            }],
            danger,
            source: Value::Null,
        }])
        .unwrap()
    }

    #[test]
    fn writes_replayable_content_addressed_delta_once() {
        let directory = tempfile::tempdir().unwrap();
        let journal = ContractDriftJournal::open(directory.path()).unwrap();
        let previous = snapshot("search archive", Value::Null);
        let current = snapshot("search current archive", Value::Null);
        let first = journal.record(&previous, &current).unwrap().unwrap();
        assert!(first.created);
        assert!(first.model_adaptation_required);
        assert!(first.directory.join("manifest.json").is_file());

        let second = journal.record(&previous, &current).unwrap().unwrap();
        assert!(!second.created);
        assert_eq!(first.directory, second.directory);
        assert_eq!(first.delta_hash, second.delta_hash);

        let loaded_previous: ToolContractSnapshot =
            read_json(&first.directory.join("previous-contracts.json")).unwrap();
        let loaded_current: ToolContractSnapshot =
            read_json(&first.directory.join("contracts.json")).unwrap();
        let loaded_delta: ToolContractDelta =
            read_json(&first.directory.join("contract-delta.json")).unwrap();
        loaded_delta
            .verify(&loaded_previous, &loaded_current)
            .unwrap();
    }

    #[test]
    fn records_metadata_only_drift_without_requesting_weight_update() {
        let directory = tempfile::tempdir().unwrap();
        let journal = ContractDriftJournal::open(directory.path()).unwrap();
        let previous = snapshot("search archive", Value::Null);
        let current = snapshot("search archive", serde_json::json!({"level":"safe"}));
        let receipt = journal.record(&previous, &current).unwrap().unwrap();
        assert!(!receipt.model_adaptation_required);
    }

    #[test]
    fn identical_contract_has_no_journal_entry() {
        let directory = tempfile::tempdir().unwrap();
        let journal = ContractDriftJournal::open(directory.path()).unwrap();
        let snapshot = snapshot("search archive", Value::Null);
        assert!(journal.record(&snapshot, &snapshot).unwrap().is_none());
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn tampered_existing_entry_fails_closed() {
        let directory = tempfile::tempdir().unwrap();
        let journal = ContractDriftJournal::open(directory.path()).unwrap();
        let previous = snapshot("search archive", Value::Null);
        let current = snapshot("search current archive", Value::Null);
        let receipt = journal.record(&previous, &current).unwrap().unwrap();
        std::fs::write(receipt.directory.join("contracts.json"), b"{}\n").unwrap();
        assert!(journal.record(&previous, &current).is_err());
    }
}
