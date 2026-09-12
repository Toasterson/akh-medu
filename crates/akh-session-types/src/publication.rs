//! Controller-owned publication of a verified local Git result.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublishSessionRepository {
    pub request_id: String,
    pub snapshot_request_id: String,
    pub branch_ref: String,
}

/// Session-owned intent. An absent executor receipt is never publication proof.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SessionPublication {
    pub schema_version: u16,
    pub request: PublishSessionRepository,
    pub operation: crate::WorkspaceOperation,
    pub dispatched_at: Option<u64>,
    pub cancelled_at: Option<u64>,
    pub receipt: Option<crate::OperationReceipt>,
}

impl SessionPublication {
    pub fn settled(&self) -> bool {
        self.cancelled_at.is_some() || self.receipt.as_ref().is_some_and(|r| r.status.terminal())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublishGitSnapshot {
    pub snapshot_operation_id: String,
    pub repository: String,
    pub branch_ref: String,
}

/// A single remote branch observed at the selected commit. This proves remote
/// availability at `at`, not a PR, review, CI, or who first published the commit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GitPublication {
    pub schema_version: u16,
    pub request: PublishGitSnapshot,
    pub base_revision: String,
    pub head_revision: String,
    pub previous_revision: Option<String>,
    pub observed_revision: String,
    pub push_acknowledged: bool,
    pub already_present: bool,
    pub policy_sha256: String,
    pub at: u64,
}

pub fn validate_branch(branch: &str) -> crate::Result<()> {
    if !branch.starts_with("refs/heads/")
        || branch.len() > 512
        || !branch
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'))
        || branch.contains("..")
        || branch
            .split('/')
            .any(|p| p.is_empty() || p.starts_with('.') || p.ends_with('.') || p.ends_with(".lock"))
    {
        return Err(crate::Fault::invalid(
            "publication requires a bounded full work-branch ref",
        ));
    }
    Ok(())
}
