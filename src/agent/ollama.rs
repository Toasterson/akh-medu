//! Ollama REST API client for local Gemma inference.
//!
//! Provides synchronous HTTP access to an Ollama server using the `/api/chat`
//! endpoint with non-streaming response mode, matching the rest of the codebase's
//! `ureq`-based sync HTTP pattern.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Ollama client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server base URL (default: "http://localhost:11434").
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Model to load, e.g. "gemma4:e4b" (default: "gemma4:e4b").
    #[serde(default = "default_model")]
    pub model: String,
    /// Request timeout in seconds (default: 120).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Ollama keep-alive duration after a request (default: "5m").
    /// Set to "0" to unload the model immediately after the request.
    #[serde(default = "default_keep_alive")]
    pub keep_alive: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            model: default_model(),
            timeout_secs: default_timeout(),
            keep_alive: default_keep_alive(),
        }
    }
}

fn default_base_url() -> String { "http://localhost:11434".into() }
fn default_model() -> String { "gemma4:e4b".into() }
fn default_timeout() -> u64 { 120 }
fn default_keep_alive() -> String { "5m".into() }

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from the Ollama client.
#[derive(Debug, Error, Diagnostic)]
pub enum OllamaError {
    #[error("Ollama server unreachable at {url}: {reason}")]
    #[diagnostic(
        code(akh::ollama::unreachable),
        help("Ensure Ollama is running: `ollama serve`")
    )]
    Unreachable { url: String, reason: String },

    #[error("Ollama request failed (HTTP {status}) at {url}: {body}")]
    #[diagnostic(
        code(akh::ollama::request_failed),
        help("Check Ollama server logs. Ensure the model is pulled: `ollama pull {model}`"),
    )]
    RequestFailed { url: String, status: u16, body: String, model: String },

    #[error("Ollama returned unexpected response: {reason}")]
    #[diagnostic(
        code(akh::ollama::parse_failed),
        help("The Ollama /api/chat endpoint returned JSON without message.content.")
    )]
    ParseFailed { reason: String },

    #[error("Ollama request timed out after {timeout_secs}s")]
    #[diagnostic(
        code(akh::ollama::timeout),
        help(
            "Increase timeout_secs in OllamaConfig, or configure a faster model, \
             or use the remote LLM fallback."
        )
    )]
    Timeout { timeout_secs: u64 },
}

/// Convenience alias.
pub type OllamaResult<T> = std::result::Result<T, OllamaError>;

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous Ollama REST client.
pub struct OllamaClient {
    config: OllamaConfig,
}

impl OllamaClient {
    /// Create a new client with the given config.
    pub fn new(config: OllamaConfig) -> Self {
        Self { config }
    }

    /// Returns `true` if the Ollama server is reachable.
    ///
    /// Uses a short 5-second timeout so callers are not blocked long
    /// when Ollama is not running.
    pub fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        ureq::get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()
            .is_ok()
    }

    /// Generate a chat response from the model.
    ///
    /// Sends a non-streaming `/api/chat` request with a `system` turn followed
    /// by the `user` turn. Returns the assistant's reply as a `String`.
    ///
    /// Error variants callers can handle:
    /// - `Unreachable` — Ollama is not running; fall back to remote API.
    /// - `RequestFailed` — model not found or server error; check model name.
    /// - `Timeout` — model too slow; reduce prompt or increase `timeout_secs`.
    pub fn generate(&self, system_prompt: &str, user_prompt: &str) -> OllamaResult<String> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": false,
            "keep_alive": self.config.keep_alive,
        });

        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(status, resp) => OllamaError::RequestFailed {
                    url: url.clone(),
                    status,
                    body: resp.into_string().unwrap_or_default(),
                    model: self.config.model.clone(),
                },
                other => OllamaError::Unreachable {
                    url: url.clone(),
                    reason: other.to_string(),
                },
            })?;

        let chat_resp: ChatResponse = response.into_json().map_err(|e| OllamaError::ParseFailed {
            reason: format!("JSON decode failed: {e}"),
        })?;

        chat_resp
            .message
            .and_then(|m| m.content)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| OllamaError::ParseFailed {
                reason: "message.content missing or empty in Ollama response".into(),
            })
    }

    /// Hint Ollama to unload the model immediately by sending a request with
    /// `keep_alive: "0"`. This is best-effort — failure is logged, not returned.
    pub fn hint_unload(&self) {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [{"role": "user", "content": ""}],
            "keep_alive": "0",
            "stream": false,
        });
        if let Err(e) = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(5))
            .send_json(&body)
        {
            tracing::debug!("hint_unload: Ollama did not acknowledge unload hint: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (pure logic, no live Ollama required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = OllamaConfig::default();
        assert_eq!(cfg.base_url, "http://localhost:11434");
        assert_eq!(cfg.model, "gemma4:e4b");
        assert_eq!(cfg.timeout_secs, 120);
        assert_eq!(cfg.keep_alive, "5m");
    }

    #[test]
    fn health_check_returns_false_when_unreachable() {
        let cfg = OllamaConfig {
            base_url: "http://127.0.0.1:19999".into(), // nothing listening here
            ..OllamaConfig::default()
        };
        let client = OllamaClient::new(cfg);
        assert!(!client.health_check());
    }
}
