//! OpenClaw API client: fetch daily memory journals for episodic import.
//!
//! Configuration is read from environment variables so that credentials never
//! appear in source code or config files:
//!   `OPENCLAW_API_KEY`   — bearer token (required when enabled)
//!   `OPENCLAW_ENDPOINT`  — base URL (default: `https://api.openclaw.dev`)

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum OpenClawError {
    #[error("OpenClaw API key not configured")]
    #[diagnostic(
        code(akh::openclaw::no_api_key),
        help(
            "Set the OPENCLAW_API_KEY environment variable to your OpenClaw bearer token."
        )
    )]
    NoApiKey,

    #[error("OpenClaw API request failed (HTTP {status}): {message}")]
    #[diagnostic(
        code(akh::openclaw::api_error),
        help("Check your API key, endpoint, and network connectivity.")
    )]
    ApiError { status: u16, message: String },

    #[error("OpenClaw transport error: {message}")]
    #[diagnostic(
        code(akh::openclaw::transport),
        help("Verify the endpoint is reachable and OPENCLAW_ENDPOINT is correct.")
    )]
    Transport { message: String },

    #[error("OpenClaw response parse error: {message}")]
    #[diagnostic(
        code(akh::openclaw::parse),
        help("The server returned unexpected JSON. Check for API version mismatches.")
    )]
    Parse { message: String },
}

pub type OpenClawResult<T> = std::result::Result<T, OpenClawError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Runtime configuration for the OpenClaw client.
///
/// Load from env vars at daemon startup; do not persist secrets in the KG.
#[derive(Debug, Clone)]
pub struct OpenClawConfig {
    /// Bearer token for API authentication.
    pub api_key: String,
    /// Base URL for the OpenClaw API.
    pub endpoint: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl OpenClawConfig {
    /// Read config from environment variables.
    ///
    /// Returns `None` when `OPENCLAW_API_KEY` is unset, so the caller can
    /// silently skip the sync rather than failing.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENCLAW_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }
        let endpoint = std::env::var("OPENCLAW_ENDPOINT")
            .unwrap_or_else(|_| "https://api.openclaw.dev".into());
        Some(Self {
            api_key,
            endpoint,
            timeout_secs: 30,
        })
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A single daily memory journal returned by the OpenClaw API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenClawJournal {
    /// Stable unique identifier (used for idempotency checks).
    pub id: String,
    /// ISO-8601 date string (`YYYY-MM-DD`) for the journal's day.
    pub date: String,
    /// Human-readable title.
    pub title: String,
    /// Individual memory entries / observations captured that day.
    pub entries: Vec<String>,
    /// Categorical tags supplied by OpenClaw.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Unix timestamp when this journal was created.
    #[serde(default)]
    pub created_at: u64,
}

/// Wire-level response from `GET /api/v1/journals`.
#[derive(Debug, Deserialize)]
struct JournalsResponse {
    journals: Vec<OpenClawJournal>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous HTTP client for the OpenClaw journal API.
pub struct OpenClawClient {
    config: OpenClawConfig,
    agent: ureq::Agent,
}

impl OpenClawClient {
    /// Create a new client from the given config.
    pub fn new(config: OpenClawConfig) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build();
        Self { config, agent }
    }

    /// Fetch all journals updated since `since_secs` (Unix timestamp).
    ///
    /// Passes `since` as a query parameter so the server can filter
    /// incrementally; a `since` value of `0` returns all journals.
    pub fn fetch_since(&self, since_secs: u64) -> OpenClawResult<Vec<OpenClawJournal>> {
        let url = format!("{}/api/v1/journals", self.config.endpoint);
        let auth = format!("Bearer {}", self.config.api_key);

        let response = self
            .agent
            .get(&url)
            .set("Authorization", &auth)
            .set("Accept", "application/json")
            .query("since", &since_secs.to_string())
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(status, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    OpenClawError::ApiError {
                        status,
                        message: body.chars().take(500).collect(),
                    }
                }
                ureq::Error::Transport(t) => OpenClawError::Transport {
                    message: t.to_string(),
                },
            })?;

        let parsed: JournalsResponse = response
            .into_json()
            .map_err(|e| OpenClawError::Parse {
                message: e.to_string(),
            })?;

        Ok(parsed.journals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_env_missing_key() {
        // Without OPENCLAW_API_KEY set the builder returns None.
        // SAFETY: single-threaded test; no other threads read this env var.
        unsafe { std::env::remove_var("OPENCLAW_API_KEY") };
        assert!(OpenClawConfig::from_env().is_none());
    }

    #[test]
    fn journal_deserializes() {
        let json = r#"{
            "journals": [{
                "id": "2026-05-19-abc",
                "date": "2026-05-19",
                "title": "Test Journal",
                "entries": ["Observation one", "Observation two"],
                "tags": ["daily"],
                "created_at": 1747612800
            }]
        }"#;
        let resp: JournalsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.journals.len(), 1);
        assert_eq!(resp.journals[0].id, "2026-05-19-abc");
        assert_eq!(resp.journals[0].entries.len(), 2);
    }
}
