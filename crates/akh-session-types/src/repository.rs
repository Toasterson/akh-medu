//! Immutable commit evidence captured by the workspace executor.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaptureGitSnapshot {
    pub base_revision: String,
    pub head_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_patch: Option<String>,
}

pub fn is_commit_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value.bytes().any(|b| b != b'0')
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl CaptureGitSnapshot {
    pub fn validate(&self) -> crate::Result<()> {
        if self
            .expected_patch
            .as_ref()
            .is_some_and(|patch| patch.is_empty() || patch.len() > 1024 * 1024)
        {
            return Err(crate::Fault::invalid(
                "expected patch must contain 1 byte to 1 MiB",
            ));
        }
        if !is_commit_id(&self.base_revision)
            || !is_commit_id(&self.head_revision)
            || self.base_revision.len() != self.head_revision.len()
        {
            return Err(crate::Fault::invalid(
                "snapshot requires full nonzero commit IDs of the same object format",
            ));
        }
        Ok(())
    }
}

/// The session manager supplies the base from its verified bootstrap, and the
/// executor/workspace/generation from the existing binding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SnapshotSessionRepository {
    pub request_id: String,
    pub head_revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_patch: Option<String>,
}

/// This proves a selected local commit's contents, not publication, current
/// checkout cleanliness, or review/CI approval. Its enclosing OperationReceipt
/// supplies the session, workspace, generation and durable operation identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GitSnapshot {
    pub schema_version: u16,
    pub request: CaptureGitSnapshot,
    pub base_tree: String,
    pub head_tree: String,
    pub patch: String,
    pub patch_sha256: String,
    /// Present only when the submitted patch was applied to a private base
    /// index and produced the exact head tree. Raw input is in the operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_patch_sha256: Option<String>,
    pub files_changed: i32,
    pub lines_added: i32,
    pub lines_removed: i32,
    pub binary_files: i32,
    pub git_version: String,
    pub executable_sha256: String,
}
