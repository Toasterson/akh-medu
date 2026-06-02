//! Pattern B comparator: OpenRouter thinking-trace fetch + ground-truth scoring.
//!
//! When `AKH_OPENROUTER_API_KEY` is set the client makes a real HTTPS call to
//! `openrouter.ai/api/v1/chat/completions` with `thinking: {type: "enabled"}`.
//! When the key is absent the run is skipped and marked as `PatternBResult::skipped`.
//!
//! Thinking-block normalization:
//! - Anthropic style: `choices[0].message.thinking`
//! - DeepSeek style:  `choices[0].message.reasoning_content`
//! - Fallback:        empty string (non-thinking model; result scored on `content` only)

use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{BenchPrompt, HaltReason};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Pattern B OpenRouter client.
#[derive(Debug, Clone)]
pub struct PatternBConfig {
    /// OpenRouter API key; `None` ⇒ all runs are skipped.
    pub api_key: Option<String>,
    /// Provider-prefixed model name, e.g. `"anthropic/claude-opus-4-5"`.
    pub model: String,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for PatternBConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("AKH_OPENROUTER_API_KEY").ok(),
            model: "anthropic/claude-opus-4-5".into(),
            timeout_ms: 120_000,
        }
    }
}

impl PatternBConfig {
    pub fn has_key(&self) -> bool {
        self.api_key.is_some()
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome of a single Pattern B run.
#[derive(Debug, Clone, Serialize)]
pub struct PatternBResult {
    pub prompt_id: String,
    pub model: String,
    /// `true` iff all acceptance predicates passed against the model's output.
    pub success: bool,
    /// Wall time for the API round-trip, ms.
    pub wall_ms: u64,
    /// Input + output tokens reported by the API.
    pub tokens_used: u64,
    /// `true` iff no API key was configured (run was skipped).
    pub skipped: bool,
    /// Text extracted from the thinking block (normalized across providers).
    pub thinking_text: String,
    /// Final answer from the model.
    pub answer_text: String,
}

impl PatternBResult {
    fn skipped(prompt_id: &str, model: &str) -> Self {
        Self {
            prompt_id: prompt_id.into(),
            model: model.into(),
            success: false,
            wall_ms: 0,
            tokens_used: 0,
            skipped: true,
            thinking_text: String::new(),
            answer_text: String::new(),
        }
    }

    fn error(prompt_id: &str, model: &str, msg: &str) -> Self {
        Self {
            prompt_id: prompt_id.into(),
            model: model.into(),
            success: false,
            wall_ms: 0,
            tokens_used: 0,
            skipped: false,
            thinking_text: String::new(),
            answer_text: format!("[error] {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenRouter response schema (minimal)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OrResponse {
    choices: Vec<OrChoice>,
    #[serde(default)]
    usage: OrUsage,
}

#[derive(Deserialize)]
struct OrChoice {
    message: OrMessage,
}

#[derive(Deserialize, Default)]
struct OrMessage {
    #[serde(default)]
    content: String,
    /// Anthropic extended-thinking blocks.
    #[serde(default)]
    thinking: String,
    /// DeepSeek reasoning_content field.
    #[serde(default)]
    reasoning_content: String,
}

#[derive(Deserialize, Default)]
struct OrUsage {
    #[serde(default)]
    total_tokens: u64,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

pub struct PatternBClient {
    config: PatternBConfig,
}

impl PatternBClient {
    pub fn new(config: PatternBConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(PatternBConfig::default())
    }

    /// Run Pattern B for `prompt`.  Returns immediately with a skipped result
    /// if no API key is configured.
    pub fn run(&self, prompt: &BenchPrompt) -> PatternBResult {
        let Some(ref key) = self.config.api_key else {
            return PatternBResult::skipped(&prompt.id, &self.config.model);
        };

        let start = Instant::now();

        let body = serde_json::json!({
            "model": self.config.model,
            "thinking": { "type": "enabled", "budget_tokens": 8000 },
            "messages": [{ "role": "user", "content": prompt.prompt }],
            "max_tokens": 4096,
        });

        let response = match ureq::post("https://openrouter.ai/api/v1/chat/completions")
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json")
            .set("HTTP-Referer", "https://akh-medu.dev")
            .set("X-Title", "akh-medu phase-41g bench")
            .send_json(&body)
        {
            Ok(r) => r,
            Err(e) => {
                return PatternBResult::error(
                    &prompt.id,
                    &self.config.model,
                    &format!("HTTP error: {e}"),
                );
            }
        };

        let wall_ms = start.elapsed().as_millis() as u64;

        let parsed: OrResponse = match response.into_json() {
            Ok(r) => r,
            Err(e) => {
                return PatternBResult::error(
                    &prompt.id,
                    &self.config.model,
                    &format!("JSON parse error: {e}"),
                );
            }
        };

        let msg = match parsed.choices.into_iter().next() {
            Some(c) => c.message,
            None => {
                return PatternBResult::error(
                    &prompt.id,
                    &self.config.model,
                    "empty choices array",
                );
            }
        };

        // Normalise thinking block across providers.
        let thinking_text = if !msg.thinking.is_empty() {
            msg.thinking
        } else {
            msg.reasoning_content
        };

        let answer_text = msg.content;
        let combined = format!("{thinking_text}\n{answer_text}");

        // Evaluate acceptance predicates using the combined text as output.
        // Step count is not meaningful for Pattern B; use usize::MAX so
        // PlanHasMinSteps predicates always pass (they constrain the symbolic side).
        let success = prompt.accepts(&combined, usize::MAX, &HaltReason::Converged);

        PatternBResult {
            prompt_id: prompt.id.clone(),
            model: self.config.model.clone(),
            success,
            wall_ms,
            tokens_used: parsed.usage.total_tokens,
            skipped: false,
            thinking_text,
            answer_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_phase41g::{AcceptancePredicate, BenchPrompt, PromptCategory};

    #[test]
    fn skips_without_api_key() {
        let config = PatternBConfig {
            api_key: None,
            model: "anthropic/claude-opus-4-5".into(),
            timeout_ms: 5_000,
        };
        let client = PatternBClient::new(config);
        let prompt = BenchPrompt {
            id: "test_skip".into(),
            category: PromptCategory::CodeAdd,
            prompt: "Add struct Foo.".into(),
            acceptance: vec![AcceptancePredicate::OutputContains {
                pattern: "struct Foo".into(),
            }],
        };
        let result = client.run(&prompt);
        assert!(result.skipped);
        assert!(!result.success);
    }
}
