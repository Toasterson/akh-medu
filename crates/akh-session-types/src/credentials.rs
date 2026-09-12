//! Private provisioning request and its secret-free durable acknowledgement.
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvisionMcpCredential {
    pub request_id: String,
    pub token: String,
}

impl std::fmt::Debug for ProvisionMcpCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvisionMcpCredential")
            .field("request_id", &self.request_id)
            .field("token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct McpCredentialReceipt {
    pub schema_version: u16,
    pub session_id: String,
    pub service: String,
    pub endpoint_sha256: String,
    pub request_id: String,
    pub token_sha256: String,
    pub principal: String,
    pub at: u64,
}
