//! LoRA adapter registry and hot-swap infrastructure for psyche-encoded agent adapters.
//!
//! Each of the six Ennead agents (Tecton, Thoth, Horus, Maat, Set, Khnum) can
//! have a trained LoRA adapter (~10–50 MB) encoding their psyche in model weights.
//!
//! Design goals:
//! - Adapter loading overhead < 100 ms (safetensors mmap + cached registry)
//! - Per-agent `.safetensors` files: `{data_dir}/adapters/{agent_id}/adapter.safetensors`
//! - Metadata sidecar: `{data_dir}/adapters/{agent_id}/meta.json`
//! - Hot-swap: the registry caches the last loaded adapter; swapping costs only
//!   a single Arc clone when the same agent is reused back-to-back.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors from LoRA adapter loading and registry operations.
#[derive(Debug, Error, Diagnostic)]
pub enum LoraError {
    #[error("adapter not found for agent '{agent_id}' at {path}")]
    #[diagnostic(
        code(akh::lora::not_found),
        help(
            "Run `python scripts/train_lora.py --agent {agent_id}` to train the adapter first. \
             Trained adapters are stored under <data-dir>/adapters/<agent-id>/."
        )
    )]
    NotFound { agent_id: String, path: PathBuf },

    #[error("adapter metadata is missing or corrupt for agent '{agent_id}': {reason}")]
    #[diagnostic(
        code(akh::lora::meta_corrupt),
        help(
            "Delete the adapter directory and retrain: \
             `python scripts/train_lora.py --agent {agent_id}`"
        )
    )]
    MetaCorrupt { agent_id: String, reason: String },

    #[error("safetensors load failed for agent '{agent_id}': {reason}")]
    #[diagnostic(
        code(akh::lora::load_failed),
        help(
            "The adapter file exists but could not be loaded. \
             Check disk space and retrain if the file is corrupt."
        )
    )]
    LoadFailed { agent_id: String, reason: String },

    #[error("adapter load exceeded 100 ms budget: {elapsed_ms} ms for agent '{agent_id}'")]
    #[diagnostic(
        code(akh::lora::slow_load),
        help(
            "The adapter file may be too large or on a slow storage device. \
             Expected 10–50 MB; actual size may exceed this budget."
        )
    )]
    SlowLoad { agent_id: String, elapsed_ms: u64 },

    #[error("I/O error for agent '{agent_id}': {reason}")]
    #[diagnostic(
        code(akh::lora::io),
        help("Check that the data directory is readable.")
    )]
    Io { agent_id: String, reason: String },
}

/// Result alias for LoRA operations.
pub type LoraResult<T> = Result<T, LoraError>;

// ── AgentId ──────────────────────────────────────────────────────────────────

/// The six Ennead agents, each with a distinct psyche encoded in a LoRA adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentId {
    /// Master builder — Ptah / Sage + Creator.
    Tecton,
    /// Knowledge seeker — Thoth / Sage + Explorer.
    Thoth,
    /// Decisive executor — Horus / Hero + Guardian.
    Horus,
    /// Balance keeper — Maat / Guardian + Healer.
    Maat,
    /// Disruptor — Set / Rebel + Explorer.
    Set,
    /// Craftsman and pattern maker — Khnum / Creator + Sage.
    Khnum,
}

impl AgentId {
    /// Canonical lowercase identifier used in file paths and config.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tecton => "tecton",
            Self::Thoth => "thoth",
            Self::Horus => "horus",
            Self::Maat => "maat",
            Self::Set => "set",
            Self::Khnum => "khnum",
        }
    }

    /// All six agents.
    pub fn all() -> &'static [AgentId] {
        &[
            Self::Tecton,
            Self::Thoth,
            Self::Horus,
            Self::Maat,
            Self::Set,
            Self::Khnum,
        ]
    }

    /// Parse from a string slice (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tecton" => Some(Self::Tecton),
            "thoth" => Some(Self::Thoth),
            "horus" => Some(Self::Horus),
            "maat" => Some(Self::Maat),
            "set" => Some(Self::Set),
            "khnum" => Some(Self::Khnum),
            _ => None,
        }
    }

    /// Primary and secondary Jungian archetype labels.
    pub fn archetypes(self) -> (&'static str, &'static str) {
        match self {
            Self::Tecton => ("sage", "creator"),
            Self::Thoth => ("sage", "explorer"),
            Self::Horus => ("hero", "guardian"),
            Self::Maat => ("guardian", "healer"),
            Self::Set => ("rebel", "explorer"),
            Self::Khnum => ("creator", "sage"),
        }
    }

    /// Egyptian deity inspiration for this agent.
    pub fn deity(self) -> &'static str {
        match self {
            Self::Tecton => "Ptah",
            Self::Thoth => "Thoth",
            Self::Horus => "Horus",
            Self::Maat => "Maat",
            Self::Set => "Set",
            Self::Khnum => "Khnum",
        }
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── LoRA configuration ────────────────────────────────────────────────────────

/// LoRA hyperparameters for training an agent adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank (default: 16).
    pub rank: usize,
    /// LoRA alpha — scales the adapter contribution (default: 32).
    pub alpha: f32,
    /// Target module names within the base model.
    pub target_modules: Vec<String>,
    /// Base model identifier (GGUF filename or Ollama tag).
    pub base_model: String,
    /// Training batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f32,
    /// Maximum sequence length in tokens.
    pub max_seq_len: usize,
    /// Warmup steps for learning-rate scheduler.
    pub warmup_steps: usize,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            base_model: "gemma2:2b-instruct-q4_K_M".into(),
            batch_size: 4,
            epochs: 3,
            learning_rate: 2e-4,
            max_seq_len: 512,
            warmup_steps: 50,
        }
    }
}

// ── Adapter metadata ──────────────────────────────────────────────────────────

/// Metadata sidecar stored alongside the `.safetensors` adapter file.
///
/// Written by `scripts/train_lora.py` after a successful training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapterMeta {
    /// Which agent this adapter belongs to.
    pub agent_id: AgentId,
    /// Adapter file name (relative to adapter directory).
    pub adapter_file: String,
    /// Total file size in bytes.
    pub file_size_bytes: u64,
    /// LoRA configuration used for training.
    pub lora_config: LoraConfig,
    /// Number of QA pairs used for training.
    pub training_pairs: usize,
    /// Agent-specific QA accuracy on the evaluation set [0.0, 1.0].
    pub eval_accuracy: f32,
    /// Training duration in seconds.
    pub training_duration_secs: f64,
    /// Unix timestamp (seconds) when training completed.
    pub trained_at: u64,
    /// Hash or tag of the base model for reproducibility.
    pub base_model_hash: String,
}

// ── Loaded adapter ────────────────────────────────────────────────────────────

/// A loaded adapter: raw tensor bytes + metadata, ready for hot-swap.
///
/// Bytes are kept as raw safetensors wire format so that downstream inference
/// backends (candle, llama-cpp) can deserialize into their own tensor types
/// without this module depending on those crates.
#[derive(Debug)]
pub struct LoadedAdapter {
    /// Adapter metadata.
    pub meta: LoraAdapterMeta,
    /// Raw safetensors bytes.
    pub bytes: Arc<Vec<u8>>,
    /// Wall-clock time at which this adapter was loaded.
    pub loaded_at: Instant,
}

impl LoadedAdapter {
    /// The agent this adapter was trained for.
    pub fn agent_id(&self) -> AgentId {
        self.meta.agent_id
    }

    /// Adapter file size in bytes (from the loaded buffer).
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}

// ── LoraRegistry ──────────────────────────────────────────────────────────────

/// Registry that manages per-agent LoRA adapters with < 100 ms hot-swap.
///
/// The last loaded adapter is cached in `current`.  Swapping to the same agent
/// is a no-op; swapping to a different agent loads from disk and updates the cache.
pub struct LoraRegistry {
    adapters_dir: PathBuf,
    current: RwLock<Option<Arc<LoadedAdapter>>>,
    meta_cache: RwLock<HashMap<AgentId, LoraAdapterMeta>>,
}

impl LoraRegistry {
    /// Create a registry rooted at `adapters_dir`.
    ///
    /// `adapters_dir` should be `<data-dir>/adapters/`.
    pub fn new(adapters_dir: impl Into<PathBuf>) -> Self {
        Self {
            adapters_dir: adapters_dir.into(),
            current: RwLock::new(None),
            meta_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Directory for a specific agent's adapter files.
    fn agent_dir(&self, agent: AgentId) -> PathBuf {
        self.adapters_dir.join(agent.as_str())
    }

    /// Path to the adapter safetensors file for `agent`.
    pub fn adapter_path(&self, agent: AgentId) -> PathBuf {
        self.agent_dir(agent).join("adapter.safetensors")
    }

    /// Path to the metadata JSON for `agent`.
    pub fn meta_path(&self, agent: AgentId) -> PathBuf {
        self.agent_dir(agent).join("meta.json")
    }

    /// Whether a trained adapter exists for `agent`.
    pub fn has_adapter(&self, agent: AgentId) -> bool {
        self.adapter_path(agent).exists() && self.meta_path(agent).exists()
    }

    /// Load adapter metadata from disk (returns cached copy if available).
    pub fn load_meta(&self, agent: AgentId) -> LoraResult<LoraAdapterMeta> {
        {
            let cache = self.meta_cache.read().unwrap();
            if let Some(meta) = cache.get(&agent) {
                return Ok(meta.clone());
            }
        }

        let path = self.meta_path(agent);
        let json = std::fs::read_to_string(&path).map_err(|e| LoraError::Io {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;
        let meta: LoraAdapterMeta =
            serde_json::from_str(&json).map_err(|e| LoraError::MetaCorrupt {
                agent_id: agent.to_string(),
                reason: e.to_string(),
            })?;

        self.meta_cache.write().unwrap().insert(agent, meta.clone());
        Ok(meta)
    }

    /// Hot-swap to `agent`'s adapter in < 100 ms.
    ///
    /// Returns the cached adapter immediately when the same agent is already active.
    /// Otherwise loads from disk, verifies the load time budget, caches, and returns.
    pub fn activate(&self, agent: AgentId) -> LoraResult<Arc<LoadedAdapter>> {
        {
            let guard = self.current.read().unwrap();
            if let Some(ref loaded) = *guard {
                if loaded.agent_id() == agent {
                    return Ok(Arc::clone(loaded));
                }
            }
        }

        let t0 = Instant::now();
        let loaded = self.load_from_disk(agent)?;
        let elapsed = t0.elapsed();

        if elapsed > Duration::from_millis(100) {
            return Err(LoraError::SlowLoad {
                agent_id: agent.to_string(),
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }

        let arc = Arc::new(loaded);
        *self.current.write().unwrap() = Some(Arc::clone(&arc));
        Ok(arc)
    }

    fn load_from_disk(&self, agent: AgentId) -> LoraResult<LoadedAdapter> {
        let meta = self.load_meta(agent)?;
        let adapter_path = self.agent_dir(agent).join(&meta.adapter_file);

        if !adapter_path.exists() {
            return Err(LoraError::NotFound {
                agent_id: agent.to_string(),
                path: adapter_path,
            });
        }

        let bytes = std::fs::read(&adapter_path).map_err(|e| LoraError::LoadFailed {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;

        Ok(LoadedAdapter {
            meta,
            bytes: Arc::new(bytes),
            loaded_at: Instant::now(),
        })
    }

    /// Deactivate the current adapter (release cached bytes).
    pub fn deactivate(&self) {
        *self.current.write().unwrap() = None;
    }

    /// List agents for which adapters are available on disk.
    pub fn available_agents(&self) -> Vec<AgentId> {
        AgentId::all()
            .iter()
            .copied()
            .filter(|&a| self.has_adapter(a))
            .collect()
    }

    /// Write adapter metadata to disk (called by the training pipeline after training).
    pub fn save_meta(&self, meta: &LoraAdapterMeta) -> LoraResult<()> {
        let agent = meta.agent_id;
        let dir = self.agent_dir(agent);
        std::fs::create_dir_all(&dir).map_err(|e| LoraError::Io {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;
        let json = serde_json::to_string_pretty(meta).map_err(|e| LoraError::MetaCorrupt {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;
        std::fs::write(self.meta_path(agent), json).map_err(|e| LoraError::Io {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;
        self.meta_cache.write().unwrap().remove(&agent);
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_id_round_trip() {
        for &agent in AgentId::all() {
            assert_eq!(AgentId::from_str(agent.as_str()), Some(agent));
        }
    }

    #[test]
    fn agent_id_display() {
        assert_eq!(AgentId::Tecton.to_string(), "tecton");
        assert_eq!(AgentId::Maat.to_string(), "maat");
    }

    #[test]
    fn agent_id_archetypes_non_empty() {
        for &agent in AgentId::all() {
            let (primary, secondary) = agent.archetypes();
            assert!(!primary.is_empty());
            assert!(!secondary.is_empty());
            assert_ne!(primary, secondary);
        }
    }

    #[test]
    fn lora_config_defaults() {
        let cfg = LoraConfig::default();
        assert_eq!(cfg.rank, 16);
        assert!((cfg.alpha - 32.0).abs() < f32::EPSILON);
        assert!(cfg.target_modules.contains(&"q_proj".to_string()));
        assert!(cfg.target_modules.contains(&"v_proj".to_string()));
    }

    #[test]
    fn registry_no_adapters_initially() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));
        assert!(reg.available_agents().is_empty());
        assert!(!reg.has_adapter(AgentId::Tecton));
    }

    #[test]
    fn registry_save_and_reload_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));

        let meta = LoraAdapterMeta {
            agent_id: AgentId::Thoth,
            adapter_file: "adapter.safetensors".into(),
            file_size_bytes: 12_000_000,
            lora_config: LoraConfig::default(),
            training_pairs: 500,
            eval_accuracy: 0.87,
            training_duration_secs: 3600.0,
            trained_at: 0,
            base_model_hash: "abc123".into(),
        };

        reg.save_meta(&meta).unwrap();
        let loaded = reg.load_meta(AgentId::Thoth).unwrap();
        assert_eq!(loaded.agent_id, AgentId::Thoth);
        assert!((loaded.eval_accuracy - 0.87).abs() < 1e-4);
        assert_eq!(loaded.training_pairs, 500);
    }

    #[test]
    fn registry_meta_cached_on_second_read() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));

        let meta = LoraAdapterMeta {
            agent_id: AgentId::Set,
            adapter_file: "adapter.safetensors".into(),
            file_size_bytes: 8_000_000,
            lora_config: LoraConfig::default(),
            training_pairs: 600,
            eval_accuracy: 0.72,
            training_duration_secs: 2700.0,
            trained_at: 1716000000,
            base_model_hash: "def456".into(),
        };
        reg.save_meta(&meta).unwrap();

        let first = reg.load_meta(AgentId::Set).unwrap();
        let second = reg.load_meta(AgentId::Set).unwrap();
        assert_eq!(first.training_pairs, second.training_pairs);
    }

    #[test]
    fn registry_activate_missing_returns_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));
        let result = reg.activate(AgentId::Horus);
        assert!(matches!(
            result,
            Err(LoraError::NotFound { .. } | LoraError::MetaCorrupt { .. } | LoraError::Io { .. })
        ));
    }

    #[test]
    fn registry_activate_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));

        // Write a fake safetensors file and metadata.
        let agent = AgentId::Khnum;
        let meta = LoraAdapterMeta {
            agent_id: agent,
            adapter_file: "adapter.safetensors".into(),
            file_size_bytes: 4,
            lora_config: LoraConfig::default(),
            training_pairs: 510,
            eval_accuracy: 0.81,
            training_duration_secs: 3200.0,
            trained_at: 1716000001,
            base_model_hash: "ghi789".into(),
        };
        reg.save_meta(&meta).unwrap();

        // Write dummy adapter bytes.
        let adapter_path = reg.adapter_path(agent);
        std::fs::write(&adapter_path, b"test").unwrap();

        let loaded = reg.activate(agent).unwrap();
        assert_eq!(loaded.agent_id(), agent);
        assert_eq!(loaded.size_bytes(), 4);

        // Second activate returns cached.
        let cached = reg.activate(agent).unwrap();
        assert_eq!(Arc::ptr_eq(&loaded.bytes, &cached.bytes), true);

        reg.deactivate();
    }

    #[test]
    fn registry_available_agents_returns_only_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = LoraRegistry::new(tmp.path().join("adapters"));

        let meta = LoraAdapterMeta {
            agent_id: AgentId::Maat,
            adapter_file: "adapter.safetensors".into(),
            file_size_bytes: 4,
            lora_config: LoraConfig::default(),
            training_pairs: 505,
            eval_accuracy: 0.75,
            training_duration_secs: 3000.0,
            trained_at: 0,
            base_model_hash: "xyz".into(),
        };
        reg.save_meta(&meta).unwrap();
        // No adapter file yet → not available.
        assert!(!reg.available_agents().contains(&AgentId::Maat));

        // Write the adapter file.
        std::fs::write(reg.adapter_path(AgentId::Maat), b"x").unwrap();
        assert!(reg.available_agents().contains(&AgentId::Maat));
        assert!(!reg.available_agents().contains(&AgentId::Tecton));
    }
}
