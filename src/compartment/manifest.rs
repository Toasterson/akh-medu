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
    /// Shared content library document.
    Library,
    /// Per-person knowledge compartment (identity, preferences, history).
    Person,
    /// Workspace-level compartment scoping an entire named Engine instance.
    ///
    /// Workspace compartments act as the root of a compartment hierarchy —
    /// they group Project, Skill, and Person compartments under a single
    /// organizational boundary and carry RBAC metadata for the workspace.
    Workspace,
}

/// Who can read/write a compartment's triples.
///
/// Scope is advisory at this layer — enforcement depends on the caller
/// checking [`CompartmentManifest::scope`] before granting access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CompartmentScope {
    /// Readable and writable by any workspace agent.
    #[default]
    Public,
    /// Readable by any agent; writable only by the owning workspace.
    Protected,
    /// Readable and writable only by the owning workspace.
    Private,
}

impl std::fmt::Display for CompartmentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "Core"),
            Self::Skill => write!(f, "Skill"),
            Self::Project => write!(f, "Project"),
            Self::Library => write!(f, "Library"),
            Self::Person => write!(f, "Person"),
            Self::Workspace => write!(f, "Workspace"),
        }
    }
}

impl std::fmt::Display for CompartmentScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public => write!(f, "Public"),
            Self::Protected => write!(f, "Protected"),
            Self::Private => write!(f, "Private"),
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
    /// Parent compartment id, forming a hierarchy rooted at a Workspace compartment.
    ///
    /// `None` means this compartment is a root (typically `Core` or `Workspace` kind).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Access scope for this compartment's triples.
    #[serde(default)]
    pub scope: CompartmentScope,
}
