//! Skillpack manager: MoE-style modular expert loading.
//!
//! Skillpacks are loadable modules containing domain-specific knowledge,
//! rewrite rules, and optional neural network weights.
//! Full implementation in Phase 3.

use serde::{Deserialize, Serialize};

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

/// Metadata describing a skillpack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique skillpack identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string.
    pub version: String,
    /// Domain tags (e.g., "astronomy", "linguistics").
    pub domains: Vec<String>,
    /// Size of weight data in bytes (0 if no neural component).
    pub weight_size_bytes: u64,
}
