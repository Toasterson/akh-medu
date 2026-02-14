//! Tool manifest types: danger metadata, capabilities, and source tracking.
//!
//! Every tool (native, WASM, CLI) declares a [`ToolManifest`] that describes
//! its interface, danger level, capabilities, and origin. The Shadow system
//! inspects these manifests for veto/bias decisions.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// How dangerous a tool's actions are.
///
/// Ordered from safest to most dangerous; comparisons use this ordering
/// (e.g., `DangerLevel::Safe < DangerLevel::Critical`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DangerLevel {
    /// No side effects, read-only or pure computation.
    Safe,
    /// Minor side effects or external reads that merit awareness.
    Cautious,
    /// Significant side effects (filesystem writes, network calls).
    Dangerous,
    /// Arbitrary execution, destructive potential.
    Critical,
}

impl std::fmt::Display for DangerLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "Safe"),
            Self::Cautious => write!(f, "Cautious"),
            Self::Dangerous => write!(f, "Dangerous"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// A capability that a tool may exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    ReadKg,
    WriteKg,
    ReadFilesystem,
    WriteFilesystem,
    Network,
    ProcessExec,
    UserInteraction,
    Reason,
    VsaAccess,
    ProvenanceAccess,
    MemoryAccess,
}

/// Danger metadata attached to every tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DangerInfo {
    /// Overall danger level.
    pub level: DangerLevel,
    /// Capabilities this tool exercises.
    pub capabilities: HashSet<Capability>,
    /// Human-readable description of the danger.
    pub description: String,
    /// Keywords the Shadow can match against action descriptions.
    pub shadow_triggers: Vec<String>,
}

/// Where a tool comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolSource {
    /// Compiled into the binary.
    Native,
    /// Loaded from a WASM component module.
    Wasm { skill_id: String, wasm_path: String },
    /// An external binary executed directly (no shell).
    Cli {
        binary_path: String,
        skill_id: Option<String>,
    },
}

/// Complete manifest describing a tool's interface and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    /// Unique tool name.
    pub name: String,
    /// What this tool does.
    pub description: String,
    /// Parameter schema.
    pub parameters: Vec<ToolParamSchema>,
    /// Danger metadata.
    pub danger: DangerInfo,
    /// Where this tool comes from.
    pub source: ToolSource,
}

/// Schema for a single tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParamSchema {
    /// Parameter name.
    pub name: String,
    /// What this parameter controls.
    pub description: String,
    /// Whether this parameter must be provided.
    pub required: bool,
    /// Type hint (default: "string").
    pub param_type: String,
}

impl ToolParamSchema {
    /// Create a required string parameter.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            param_type: "string".into(),
        }
    }

    /// Create an optional string parameter.
    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            param_type: "string".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn danger_level_ordering() {
        assert!(DangerLevel::Safe < DangerLevel::Cautious);
        assert!(DangerLevel::Cautious < DangerLevel::Dangerous);
        assert!(DangerLevel::Dangerous < DangerLevel::Critical);
    }

    #[test]
    fn danger_level_display() {
        assert_eq!(DangerLevel::Safe.to_string(), "Safe");
        assert_eq!(DangerLevel::Critical.to_string(), "Critical");
    }

    #[test]
    fn tool_param_schema_constructors() {
        let req = ToolParamSchema::required("name", "desc");
        assert!(req.required);
        assert_eq!(req.param_type, "string");

        let opt = ToolParamSchema::optional("opt", "desc");
        assert!(!opt.required);
    }

    #[test]
    fn danger_info_serialization_roundtrip() {
        let info = DangerInfo {
            level: DangerLevel::Cautious,
            capabilities: HashSet::from([Capability::ReadKg, Capability::WriteKg]),
            description: "test".into(),
            shadow_triggers: vec!["mutate".into()],
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: DangerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level, DangerLevel::Cautious);
        assert!(back.capabilities.contains(&Capability::ReadKg));
    }
}
