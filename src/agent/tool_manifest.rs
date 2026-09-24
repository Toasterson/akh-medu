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

/// Whether invoking a tool can change state.
///
/// Read-only is an explicit declaration, never an inference from the danger
/// level or capability set: a tool that says nothing is [`ToolEffect::Mutating`].
/// The `/v1` core API derives its `read_only` catalog flag and the default
/// delegation grant from this (Phase 53 review, ADR 113), so a `Safe` tool
/// with only read capabilities that quietly writes (a scratchpad, a shared
/// checklist, a process-wide option) can never slip into a read-only set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    /// May change akh, session, process or external state. The default.
    #[default]
    Mutating,
    /// Declared to change no state at all: every call is a pure read.
    ReadOnly,
}

impl ToolEffect {
    /// Whether the tool declared itself read-only.
    pub fn is_read_only(self) -> bool {
        matches!(self, Self::ReadOnly)
    }
}

/// Danger metadata attached to every tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DangerInfo {
    /// Whether the tool changes state. Absent in older serialized manifests,
    /// which therefore read as [`ToolEffect::Mutating`].
    #[serde(default)]
    pub effect: ToolEffect,
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
    /// A proxied tool from a remote MCP server (Phase 44m, ADR 065 D12.1). The
    /// `server` is the session-registered server name, so the ACP `_akh/usage`
    /// panel can group MCP tools by server (`MCP tools` → per-server children)
    /// by reading the manifest rather than guessing from a `mcp__{server}__{tool}`
    /// name prefix.
    Mcp { server: String },
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
    /// Optional JSON-Schema fragment for a structured (nested) parameter — e.g.
    /// an array of objects (ADR 076 §2). When present, `ollama_tool_catalog`
    /// emits it verbatim (merging in `description`) instead of the flat
    /// `{type: param_type}` rendering, so the LLM is handed precise nested types.
    /// `None` → the historical flat rendering (existing tools are unaffected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

impl ToolParamSchema {
    /// Create a required string parameter.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            param_type: "string".into(),
            schema: None,
        }
    }

    /// Create an optional string parameter.
    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            param_type: "string".into(),
            schema: None,
        }
    }

    /// Create a required parameter carrying an explicit JSON-Schema fragment for
    /// a structured (nested) value.
    pub fn required_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            param_type: "object".into(),
            schema: Some(schema),
        }
    }

    /// Create an optional parameter carrying an explicit JSON-Schema fragment for
    /// a structured (nested) value.
    pub fn optional_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            param_type: "object".into(),
            schema: Some(schema),
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
            effect: ToolEffect::Mutating,
            level: DangerLevel::Cautious,
            capabilities: HashSet::from([Capability::ReadKg, Capability::WriteKg]),
            description: "test".into(),
            shadow_triggers: vec!["mutate".into()],
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: DangerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level, DangerLevel::Cautious);
        assert!(back.capabilities.contains(&Capability::ReadKg));
        assert_eq!(back.effect, ToolEffect::Mutating);
    }

    #[test]
    fn danger_info_without_effect_reads_as_mutating() {
        let json =
            r#"{"level":"Safe","capabilities":["ReadKg"],"description":"","shadow_triggers":[]}"#;
        let info: DangerInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.effect, ToolEffect::Mutating);
        assert!(!info.effect.is_read_only());
    }
}
