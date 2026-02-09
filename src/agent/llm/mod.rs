//! Ollama client for optional natural language I/O.
//!
//! The LLM is used **only** for:
//! - Richer triple extraction from ambiguous text
//! - Synthesizing natural language answers from KG query results
//! - Conversational interaction in chat mode
//!
//! Core reasoning (tool selection, plan generation, criteria evaluation) is
//! VSA-native and does NOT use the LLM.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from the LLM subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum LlmError {
    #[error("Ollama is not available at {url}")]
    #[diagnostic(
        code(akh::llm::unavailable),
        help("Start Ollama with `ollama serve` or set --no-ollama to use regex-only mode.")
    )]
    Unavailable { url: String },

    #[error("Ollama request failed: {message}")]
    #[diagnostic(
        code(akh::llm::request_failed),
        help("Check that Ollama is running and the model is pulled.")
    )]
    RequestFailed { message: String },

    #[error("Failed to parse Ollama response: {message}")]
    #[diagnostic(
        code(akh::llm::parse_error),
        help("The model returned an unexpected response format.")
    )]
    ParseError { message: String },

    #[error("Ollama request timed out after {timeout_secs}s")]
    #[diagnostic(
        code(akh::llm::timeout),
        help("Increase the timeout or use a smaller model.")
    )]
    Timeout { timeout_secs: u64 },
}

/// Configuration for the Ollama client.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL for the Ollama API.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "llama3.2".into(),
            timeout_secs: 120,
        }
    }
}

/// A chat message for multi-turn conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant".
    pub role: String,
    /// Message content.
    pub content: String,
}

/// An extracted triple from LLM-assisted text processing.
#[derive(Debug, Clone)]
pub struct ExtractedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// Client for the Ollama REST API.
pub struct OllamaClient {
    config: OllamaConfig,
    available: bool,
}

impl OllamaClient {
    /// Create a new Ollama client with the given configuration.
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            available: false,
        }
    }

    /// Probe the Ollama server to check availability.
    ///
    /// Sends a lightweight request to the health endpoint.
    /// Sets the internal `available` flag.
    pub fn probe(&mut self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(5))
            .build();

        match agent.get(&url).call() {
            Ok(resp) => {
                self.available = resp.status() == 200;
                self.available
            }
            Err(_) => {
                self.available = false;
                false
            }
        }
    }

    /// Whether the Ollama server is available.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Generate a completion from a prompt.
    pub fn generate(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, LlmError> {
        if !self.available {
            return Err(LlmError::Unavailable {
                url: self.config.base_url.clone(),
            });
        }

        let url = format!("{}/api/generate", self.config.base_url);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
        });

        if let Some(sys) = system {
            body["system"] = serde_json::Value::String(sys.to_string());
        }

        let body_str = serde_json::to_string(&body).map_err(|e| LlmError::RequestFailed {
            message: format!("JSON serialize error: {e}"),
        })?;

        let resp = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .map_err(|e: ureq::Error| LlmError::RequestFailed {
                message: e.to_string(),
            })?;

        let resp_str = resp.into_string().map_err(|e| LlmError::ParseError {
            message: e.to_string(),
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&resp_str).map_err(|e| LlmError::ParseError {
                message: e.to_string(),
            })?;

        json["response"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::ParseError {
                message: "missing 'response' field".into(),
            })
    }

    /// Multi-turn chat completion.
    pub fn chat(&self, messages: &[ChatMessage]) -> Result<ChatMessage, LlmError> {
        if !self.available {
            return Err(LlmError::Unavailable {
                url: self.config.base_url.clone(),
            });
        }

        let url = format!("{}/api/chat", self.config.base_url);
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build();

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": msgs,
            "stream": false,
        });

        let body_str = serde_json::to_string(&body).map_err(|e| LlmError::RequestFailed {
            message: format!("JSON serialize error: {e}"),
        })?;

        let resp = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .map_err(|e: ureq::Error| LlmError::RequestFailed {
                message: e.to_string(),
            })?;

        let resp_str = resp.into_string().map_err(|e| LlmError::ParseError {
            message: e.to_string(),
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&resp_str).map_err(|e| LlmError::ParseError {
                message: e.to_string(),
            })?;

        let content = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(ChatMessage {
            role: "assistant".into(),
            content,
        })
    }

    /// Extract triples from text using the LLM for structured JSON output.
    pub fn extract_triples(&self, text: &str) -> Result<Vec<ExtractedTriple>, LlmError> {
        let system = "You are a knowledge extraction assistant. \
            Extract factual triples from the given text. \
            Return a JSON array of objects with fields: subject, predicate, object, confidence (0-1). \
            Only return the JSON array, no other text.";

        let response = self.generate(text, Some(system))?;

        // Try to parse JSON from the response.
        let trimmed = response.trim();
        let json_str = if trimmed.starts_with('[') {
            trimmed
        } else {
            // Try to find JSON array in the response.
            let start = trimmed.find('[');
            let end = trimmed.rfind(']');
            match (start, end) {
                (Some(s), Some(e)) if e > s => &trimmed[s..=e],
                _ => {
                    return Err(LlmError::ParseError {
                        message: "no JSON array found in response".into(),
                    })
                }
            }
        };

        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| LlmError::ParseError {
                message: format!("JSON parse error: {e}"),
            })?;

        let mut triples = Vec::new();
        for val in &parsed {
            let subject = val["subject"].as_str().unwrap_or("").to_string();
            let predicate = val["predicate"].as_str().unwrap_or("").to_string();
            let object = val["object"].as_str().unwrap_or("").to_string();
            let confidence = val["confidence"].as_f64().unwrap_or(0.7) as f32;

            if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                triples.push(ExtractedTriple {
                    subject,
                    predicate,
                    object,
                    confidence,
                });
            }
        }

        Ok(triples)
    }

    /// Get the model name being used.
    pub fn model(&self) -> &str {
        &self.config.model
    }
}

impl std::fmt::Debug for OllamaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaClient")
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("available", &self.available)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_unreachable_returns_false() {
        let config = OllamaConfig {
            base_url: "http://127.0.0.1:1".into(), // unreachable port
            ..Default::default()
        };
        let mut client = OllamaClient::new(config);
        assert!(!client.probe());
        assert!(!client.is_available());
    }

    #[test]
    fn generate_when_unavailable_returns_error() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);
        let result = client.generate("test", None);
        assert!(result.is_err());
    }

    #[test]
    fn chat_when_unavailable_returns_error() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);
        let result = client.chat(&[ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }]);
        assert!(result.is_err());
    }

    #[test]
    fn default_config_values() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.timeout_secs, 120);
    }
}
