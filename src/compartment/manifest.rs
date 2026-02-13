//! Compartment manifest and state types.

use serde::{Deserialize, Serialize};

/// What kind of compartment this is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompartmentKind {
    /// Always-active compartments (personality, psyche).
    Core,
    /// Travels with skill packs — loaded/unloaded with skills.
    Skill,
    /// Attached to a project context.
    Project,
}

impl std::fmt::Display for CompartmentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "Core"),
            Self::Skill => write!(f, "Skill"),
            Self::Project => write!(f, "Project"),
        }
    }
}

/// Lifecycle state of a compartment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompartmentState {
    /// On disk, not loaded into KG.
    Dormant,
    /// Triples loaded into KG, queries can be scoped to it.
    Loaded,
    /// Loaded AND actively influencing the OODA loop.
    Active,
}

impl std::fmt::Display for CompartmentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dormant => write!(f, "Dormant"),
            Self::Loaded => write!(f, "Loaded"),
            Self::Active => write!(f, "Active"),
        }
    }
}

/// Manifest describing a compartment, parsed from `compartment.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentManifest {
    /// Unique identifier (e.g., "psyche", "personality", "skill:astronomy").
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// What kind of compartment this is.
    pub kind: CompartmentKind,
    /// Description of the compartment's purpose.
    #[serde(default)]
    pub description: String,
    /// Optional path to a triples JSON file within the compartment directory.
    #[serde(default)]
    pub triples_file: Option<String>,
    /// Optional path to a rules file within the compartment directory.
    #[serde(default)]
    pub rules_file: Option<String>,
    /// Grammar reference: built-in grammar name OR path to custom grammar TOML.
    #[serde(default)]
    pub grammar_ref: Option<String>,
    /// Domain tags for search and categorization.
    #[serde(default)]
    pub tags: Vec<String>,
}
