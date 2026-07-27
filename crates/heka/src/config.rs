//! Heka service configuration, parsed from `heka.toml`.
//!
//! Mirrors `crates/seshat/src/config.rs` — TOML config with environment-variable
//! overrides for K8s secrets.

use serde::{Deserialize, Serialize};

/// Top-level Heka service configuration, parsed from `heka.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HekaConfig {
    pub database: DatabaseConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub eval: EvalConfig,
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

/// Service (hekad) settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Address to bind the hekad service (reserved for Phase 42b MCP server).
    #[serde(default = "default_bind")]
    pub bind: String,
}

/// Embedding model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Sentence embedding model name (for model directory lookup).
    #[serde(default = "default_embedding_model")]
    pub model: String,

    /// Directory for downloaded models.
    /// Defaults to `~/.local/share/heka/models/`.
    pub model_dir: Option<String>,
}

/// Evaluation data settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Path to the evaluation dataset (JSONL or CSV) for recall/precision scoring.
    #[serde(default)]
    pub data_path: String,
}

// --- Defaults ---

fn default_max_connections() -> u32 {
    10
}

fn default_bind() -> String {
    "127.0.0.1:9091".into()
}

fn default_embedding_model() -> String {
    "all-MiniLM-L6-v2".into()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: default_embedding_model(),
            model_dir: None,
        }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            data_path: String::new(),
        }
    }
}

impl HekaConfig {
    /// Load config from a TOML file path.
    ///
    /// After parsing, environment variable overrides are applied:
    /// - `DATABASE_URL` overrides `database.url`
    /// - `HEKA_BIND` overrides `server.bind`
    pub fn from_file(path: &std::path::Path) -> crate::HekaResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::HekaError::Config(format!("cannot read {}: {e}", path.display())))?;
        let mut config: Self = toml::from_str(&content)
            .map_err(|e| crate::HekaError::Config(format!("invalid TOML: {e}")))?;

        // Allow DATABASE_URL env var to override the TOML value (K8s secrets).
        if let Ok(url) = std::env::var("DATABASE_URL") {
            config.database.url = url;
        }
        // Allow HEKA_BIND env var to override the bind address.
        if let Ok(bind) = std::env::var("HEKA_BIND") {
            config.server.bind = bind;
        }

        Ok(config)
    }

    /// Resolve the model directory, defaulting to `~/.local/share/heka/models/`.
    pub fn model_dir(&self) -> std::path::PathBuf {
        if let Some(ref dir) = self.embedding.model_dir {
            std::path::PathBuf::from(dir)
        } else if let Some(data) = dirs_fallback() {
            data.join("heka").join("models")
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