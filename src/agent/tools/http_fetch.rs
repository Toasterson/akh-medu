//! HTTP fetch tool: retrieve content from URLs.
//!
//! Uses `ureq` for synchronous HTTP requests. Enforces a timeout and
//! maximum response size to prevent the agent from hanging or consuming
//! excessive memory.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Maximum response body size (256 KB).
const MAX_RESPONSE_SIZE: u64 = 256 * 1024;

/// Fetch content from a URL via HTTP GET.
pub struct HttpFetchTool;

impl Tool for HttpFetchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "http_fetch".into(),
            description: "Fetch content from a URL via HTTP GET (max 256 KB response).".into(),
            parameters: vec![
                ToolParam {
                    name: "url".into(),
                    description: "URL to fetch.".into(),
                    required: true,
                },
                ToolParam {
                    name: "timeout".into(),
                    description: "Timeout in seconds (default: 10).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let url = input.require("url", "http_fetch")?;
        let timeout_secs: u64 = input
            .get("timeout")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        // Basic URL validation.
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolOutput::err(format!(
                "Invalid URL: \"{url}\". Must start with http:// or https://."
            )));
        }

        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build();

        match agent.get(url).call() {
            Ok(response) => {
                let status = response.status();
                let content_type = response
                    .header("Content-Type")
                    .unwrap_or("unknown")
                    .to_string();

                match response.into_string() {
                    Ok(body) => {
                        let truncated = if body.len() as u64 > MAX_RESPONSE_SIZE {
                            format!(
                                "{}... [truncated at {} bytes, total: {}]",
                                &body[..MAX_RESPONSE_SIZE as usize],
                                MAX_RESPONSE_SIZE,
                                body.len()
                            )
                        } else {
                            body.clone()
                        };

                        Ok(ToolOutput::ok(format!(
                            "HTTP {} ({}), {} bytes:\n{}",
                            status,
                            content_type,
                            body.len(),
                            truncated,
                        )))
                    }
                    Err(e) => Ok(ToolOutput::err(format!(
                        "HTTP {status} but failed to read body: {e}"
                    ))),
                }
            }
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                let preview = if body.len() > 500 {
                    format!("{}...", &body[..500])
                } else {
                    body
                };
                Ok(ToolOutput::err(format!("HTTP error {code}: {preview}")))
            }
            Err(ureq::Error::Transport(transport)) => Ok(ToolOutput::err(format!(
                "Transport error fetching \"{url}\": {transport}"
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "http_fetch".into(),
            description: "Fetches URLs via HTTP GET — network access.".into(),
            parameters: vec![
                ToolParamSchema::required("url", "URL to fetch."),
                ToolParamSchema::optional("timeout", "Timeout in seconds (default: 10)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network]),
                description: "Fetches URLs via HTTP GET — network access.".into(),
                shadow_triggers: vec![
                    "http".into(),
                    "url".into(),
                    "fetch".into(),
                    "download".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}
