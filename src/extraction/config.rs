//! Extraction configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the triple extraction subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Whether LLM extraction is enabled (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Prefer local LLM over external API (default: true).
    #[serde(default = "default_true")]
    pub prefer_local: bool,

    /// External API configuration (optional).
    #[serde(default)]
    pub external_api: Option<ExternalApiConfig>,

    /// Maximum text chunk size in chars before splitting (default: 2000).
    #[serde(default = "default_max_chunk")]
    pub max_chunk_chars: usize,

    /// Minimum confidence to accept an extracted triple (default: 0.5).
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,

    /// Maximum triples to extract per chunk (default: 20).
    #[serde(default = "default_max_triples")]
    pub max_triples_per_chunk: usize,

    /// Maximum external API calls per extraction cycle (default: 10).
    #[serde(default = "default_max_api_calls")]
    pub max_api_calls_per_cycle: usize,
}

/// External OpenAI-compatible API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalApiConfig {
    /// API base URL (e.g., "https://openrouter.ai/api/v1").
    pub base_url: String,

    /// API key. Falls back to `AKH_EXTRACTION_API_KEY` env var if absent.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Model identifier (e.g., "deepseek/deepseek-chat").
    pub model: String,

    /// Max tokens for completion (default: 512).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature (default: 0.1 — low for structured extraction).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Timeout in seconds (default: 30).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl ExternalApiConfig {
    /// Resolve the API key from config or environment.
    pub fn resolve_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("AKH_EXTRACTION_API_KEY").ok())
    }
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefer_local: true,
            external_api: None,
            max_chunk_chars: 2000,
            min_confidence: 0.5,
            max_triples_per_chunk: 20,
            max_api_calls_per_cycle: 10,
        }
    }
}

fn default_true() -> bool { true }
fn default_max_chunk() -> usize { 2000 }
fn default_min_confidence() -> f32 { 0.5 }
fn default_max_triples() -> usize { 20 }
fn default_max_api_calls() -> usize { 10 }
fn default_max_tokens() -> u32 { 512 }
fn default_temperature() -> f32 { 0.1 }
fn default_timeout() -> u64 { 30 }
