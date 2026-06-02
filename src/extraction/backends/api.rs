//! External OpenAI-compatible API backend for triple extraction.

use crate::extraction::config::ExternalApiConfig;
use crate::extraction::error::{ExtractionError, ExtractionResult};
use crate::extraction::prompt;
use crate::extraction::RawTriple;

/// Backend that calls an OpenAI-compatible `/v1/chat/completions` endpoint.
pub struct ExternalApiBackend {
    config: ExternalApiConfig,
}

impl ExternalApiBackend {
    pub fn new(config: ExternalApiConfig) -> Self {
        Self { config }
    }

    /// Extract triples from text via the external API.
    pub fn extract(&self, text: &str, max_triples: usize) -> ExtractionResult<Vec<RawTriple>> {
        let api_key = self.config.resolve_api_key().ok_or_else(|| {
            ExtractionError::LlmUnavailable {
                reason: "no API key: set AKH_EXTRACTION_API_KEY or api_key in config".into(),
            }
        })?;

        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let messages = prompt::openai_messages(text, max_triples);

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(status, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    ExtractionError::ApiRequestFailed {
                        url: url.clone(),
                        status,
                        body,
                    }
                }
                _ => ExtractionError::ApiTimeout {
                    url: url.clone(),
                    timeout_secs: self.config.timeout_secs,
                },
            })?;

        let resp_body: serde_json::Value = response
            .into_json()
            .map_err(|e| ExtractionError::ParseFailed {
                raw_output: String::new(),
                reason: format!("failed to parse API response: {e}"),
            })?;

        // Extract the assistant's message content.
        let content = resp_body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("[]");

        Ok(prompt::parse_triples(content))
    }
}
