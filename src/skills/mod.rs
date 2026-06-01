//! Skillpack manager: MoE-style modular expert loading.
//!
//! Skillpacks are loadable modules containing domain-specific knowledge
//! (triples), rewrite rules, and metadata. They follow a lifecycle:
//! Cold (on disk) → Warm (manifest parsed) → Hot (active in memory).

pub mod bridge;
pub mod curriculum;
pub mod discover;
pub mod graph;
pub mod manager;
pub mod share;
pub mod synthesis;

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
///
/// Extended in Phase 31 with dependency tracking (31a), curriculum support (31h),
/// and remote sharing (31f).
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

    // ── Phase 31a: Dependency graph ────────────────────────────────────────
    /// Skill IDs that must be active before this skill can be activated.
    #[serde(default)]
    pub depends: Vec<String>,
    /// Skill IDs that cannot be co-active with this skill.
    #[serde(default)]
    pub conflicts: Vec<String>,
    /// Capability tags this skill provides (for loose-coupling via `requires`).
    #[serde(default)]
    pub provides: Vec<String>,
    /// Capability tags this skill needs; resolved to provider skill IDs at activation.
    #[serde(default)]
    pub requires: Vec<String>,

    // ── Phase 31f: Sharing / remote URL ───────────────────────────────────
    /// Optional canonical remote URL for this skill archive (e.g., from a registry).
    #[serde(default)]
    pub remote_url: Option<String>,

    // ── Phase 31h: Curriculum ──────────────────────────────────────────────
    /// Path to the curriculum JSON file relative to the skill directory.
    #[serde(default)]
    pub curriculum_file: Option<String>,
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
    /// Timestamp of the most recent access (Phase 31c idle-deactivation tracking).
    pub last_used_at: Option<std::time::Instant>,
}

/// Summary info for listing skills.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Capabilities provided by this skill.
    #[serde(default)]
    pub provides: Vec<String>,
    /// Capabilities required by this skill.
    #[serde(default)]
    pub requires: Vec<String>,
}

/// Result of activating a skillpack.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Payload for installing a skill remotely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInstallPayload {
    pub manifest: SkillManifest,
    pub triples: Vec<LabelTriple>,
    #[serde(default)]
    pub rules: String,
}

/// A single label-based triple for skill installation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default = "default_label_confidence")]
    pub confidence: f32,
}

fn default_label_confidence() -> f32 {
    1.0
}

/// Result type alias for skill operations.
pub type SkillResultType<T> = SkillResult<T>;
