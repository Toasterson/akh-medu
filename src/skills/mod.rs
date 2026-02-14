//! Skillpack manager: MoE-style modular expert loading.
//!
//! Skillpacks are loadable modules containing domain-specific knowledge
//! (triples), rewrite rules, and metadata. They follow a lifecycle:
//! Cold (on disk) → Warm (manifest parsed) → Hot (active in memory).

pub mod manager;

use serde::{Deserialize, Serialize};

use crate::error::SkillResult;

/// The lifecycle state of a skillpack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillState {
    /// On disk, not loaded.
    Cold,
    /// Manifest loaded, ready to activate.
    Warm,
    /// Fully loaded and active in memory.
    Hot,
}

impl std::fmt::Display for SkillState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cold => write!(f, "Cold"),
            Self::Warm => write!(f, "Warm"),
            Self::Hot => write!(f, "Hot"),
        }
    }
}

/// Metadata describing a skillpack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique skillpack identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string.
    pub version: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Domain tags (e.g., "astronomy", "linguistics").
    pub domains: Vec<String>,
    /// Size of weight data in bytes (0 if no neural component).
    #[serde(default)]
    pub weight_size_bytes: u64,
    /// Path to triples JSON file relative to skill directory.
    #[serde(default)]
    pub triples_file: Option<String>,
    /// Path to rules file relative to skill directory.
    #[serde(default)]
    pub rules_file: Option<String>,
    /// Paths to CLI tool manifest JSON files relative to skill directory.
    #[serde(default)]
    pub cli_tools: Vec<String>,
    /// Paths to WASM component modules relative to skill directory.
    #[serde(default)]
    pub wasm_tools: Vec<String>,
    /// Key-value configuration passed to tools (e.g., API keys, base URLs).
    #[serde(default)]
    pub tool_config: std::collections::HashMap<String, String>,
}

/// A skillpack with its manifest and current state.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// The parsed manifest.
    pub manifest: SkillManifest,
    /// Current lifecycle state.
    pub state: SkillState,
    /// Number of triples loaded from this skill.
    pub triple_count: usize,
    /// Number of rules loaded from this skill.
    pub rule_count: usize,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
}

/// Summary info for listing skills.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    /// Skill identifier.
    pub id: String,
    /// Skill name.
    pub name: String,
    /// Version.
    pub version: String,
    /// Description.
    pub description: String,
    /// Current state.
    pub state: SkillState,
    /// Domain tags.
    pub domains: Vec<String>,
    /// Triples loaded.
    pub triple_count: usize,
    /// Rules loaded.
    pub rule_count: usize,
}

/// Result of activating a skillpack.
#[derive(Debug, Clone)]
pub struct SkillActivation {
    /// Skill identifier.
    pub skill_id: String,
    /// Number of triples loaded.
    pub triples_loaded: usize,
    /// Number of rules loaded.
    pub rules_loaded: usize,
    /// Memory used in bytes.
    pub memory_bytes: usize,
}

/// Result type alias for skill operations.
pub type SkillResultType<T> = SkillResult<T>;
