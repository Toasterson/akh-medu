//! Seshat service configuration.

use serde::{Deserialize, Serialize};

/// Top-level Seshat service configuration, parsed from `seshat.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeshatConfig {
    pub database: DatabaseConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub synthesis: SynthesisConfig,
}

/// PostgreSQL connection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL.
    /// Example: `postgres://user:pass@localhost:5432/akhmedu`
    pub url: String,

    /// Maximum number of connections in the pool.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

/// Embedding worker settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Sentence embedding model name (for model directory lookup).
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Number of chunks to embed per batch.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Seconds between embed worker polls for unembedded chunks.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,

    /// Directory for downloaded models.
    /// Defaults to `~/.local/share/seshat/models/`.
    pub model_dir: Option<String>,
}

/// MCP server settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind the MCP server.
    #[serde(default = "default_bind")]
    pub bind: String,

    /// MCP transport type.
    #[serde(default = "default_transport")]
    pub transport: String,
}

/// Optional LLM synthesis settings (Candle backend).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Whether LLM synthesis is enabled in the service.
    #[serde(default)]
    pub enabled: bool,

    /// Inference backend: "candle" (GGUF via Candle).
    #[serde(default = "default_synthesis_backend")]
    pub backend: String,

    /// Path to the GGUF model file.
    /// Interim: Qwen2.5-1.5B or Phi-3-mini.
    /// Target: T5 Knowledge model from Phase 35.
    #[serde(default)]
    pub model_path: String,

    /// Maximum triples to extract per wish.
    #[serde(default = "default_max_triples")]
    pub max_triples: usize,
}

// --- Defaults ---

fn default_max_connections() -> u32 {
    10
}

fn default_embedding_model() -> String {
    "all-MiniLM-L6-v2".into()
}

fn default_batch_size() -> usize {
    32
}

fn default_poll_interval() -> u64 {
    30
}

fn default_bind() -> String {
    "127.0.0.1:9090".into()
}

fn default_transport() -> String {
    "streamable-http".into()
}

fn default_synthesis_backend() -> String {
    "candle".into()
}

fn default_max_triples() -> usize {
    30
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            batch_size: default_batch_size(),
            poll_interval_secs: default_poll_interval(),
            model_dir: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            transport: default_transport(),
        }
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: default_synthesis_backend(),
            model_path: String::new(),
            max_triples: default_max_triples(),
        }
    }
}

impl SeshatConfig {
    /// Load config from a TOML file path.
    pub fn from_file(path: &std::path::Path) -> crate::SeshatResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::SeshatError::Config(format!("cannot read {}: {e}", path.display()))
        })?;
        toml::from_str(&content)
            .map_err(|e| crate::SeshatError::Config(format!("invalid TOML: {e}")))
    }

    /// Resolve the model directory, defaulting to `~/.local/share/seshat/models/`.
    pub fn model_dir(&self) -> std::path::PathBuf {
        if let Some(ref dir) = self.embedding.model_dir {
            std::path::PathBuf::from(dir)
        } else if let Some(data) = dirs_fallback() {
            data.join("seshat").join("models")
        } else {
            std::path::PathBuf::from("models")
        }
    }
}

/// Best-effort XDG data home without pulling in the `dirs` crate.
fn dirs_fallback() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share"))
        })
}
