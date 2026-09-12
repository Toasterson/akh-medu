//! Repository bootstrap intent and executor-backed completion evidence.
use serde::{Deserialize, Serialize};

/// Required admission gates for a fresh, per-occurrence workspace session.
/// These are checked atomically before the automation creates its first turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BootstrapRequirements {
    pub repository: String,
    pub mcp_credentials: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RepositoryCheckout {
    pub repository: String,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BootstrapRepository {
    pub request_id: String,
    pub checkout: RepositoryCheckout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RepositoryBootstrapReceipt {
    pub schema_version: u16,
    pub session_id: String,
    pub request: BootstrapRepository,
    pub principal: String,
    pub at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RepositoryBootstrapResult {
    pub request: BootstrapRepository,
    pub session_id: String,
    pub workspace_id: String,
    pub generation: u64,
    pub verified_revision: String,
    pub operation_receipts: Vec<String>,
    pub at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionRepositoryBootstrap {
    pub policy_sha256: String,
    pub receipt: Option<RepositoryBootstrapReceipt>,
    pub result: Option<RepositoryBootstrapResult>,
}
