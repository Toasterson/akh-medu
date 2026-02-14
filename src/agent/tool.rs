//! Tool system: trait-based tools with runtime registration.
//!
//! Tools are the agent's interface to the engine's capabilities.
//! Each tool implements the [`Tool`] trait and is registered in a [`ToolRegistry`].

use std::collections::HashMap;

use crate::engine::Engine;

use super::error::{AgentError, AgentResult};
use super::tool_manifest::ToolManifest;

/// Description of a tool's interface.
#[derive(Debug, Clone)]
pub struct ToolSignature {
    /// Unique name of the tool.
    pub name: String,
    /// What this tool does.
    pub description: String,
    /// Parameters the tool accepts.
    pub parameters: Vec<ToolParam>,
}

/// A single parameter in a tool's signature.
#[derive(Debug, Clone)]
pub struct ToolParam {
    /// Parameter name.
    pub name: String,
    /// What this parameter controls.
    pub description: String,
    /// Whether this parameter must be provided.
    pub required: bool,
}

/// Input to a tool execution.
#[derive(Debug, Clone, Default)]
pub struct ToolInput {
    /// Named parameters.
    pub params: HashMap<String, String>,
}

impl ToolInput {
    /// Create a new empty input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter.
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }

    /// Get a parameter value.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.params.get(name).map(|s| s.as_str())
    }

    /// Get a required parameter, returning an error if missing.
    pub fn require(&self, name: &str, tool_name: &str) -> AgentResult<&str> {
        self.get(name).ok_or(AgentError::ToolExecution {
            tool_name: tool_name.into(),
            message: format!("missing required parameter: {name}"),
        })
    }
}

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Whether the tool succeeded.
    pub success: bool,
    /// Human-readable result summary.
    pub result: String,
    /// Symbols involved in the operation (for WM linking).
    pub symbols_involved: Vec<crate::symbol::SymbolId>,
}

impl ToolOutput {
    /// Create a successful output.
    pub fn ok(result: impl Into<String>) -> Self {
        Self {
            success: true,
            result: result.into(),
            symbols_involved: Vec::new(),
        }
    }

    /// Create a successful output with associated symbols.
    pub fn ok_with_symbols(
        result: impl Into<String>,
        symbols: Vec<crate::symbol::SymbolId>,
    ) -> Self {
        Self {
            success: true,
            result: result.into(),
            symbols_involved: symbols,
        }
    }

    /// Create a failed output.
    pub fn err(result: impl Into<String>) -> Self {
        Self {
            success: false,
            result: result.into(),
            symbols_involved: Vec::new(),
        }
    }
}

/// A tool the agent can execute.
pub trait Tool: Send + Sync {
    /// Describe this tool's interface.
    fn signature(&self) -> ToolSignature;

    /// Execute the tool with the given input against the engine.
    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput>;

    /// Return structured manifest describing this tool's danger metadata,
    /// capabilities, and source.
    fn manifest(&self) -> ToolManifest;
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    manifests: HashMap<String, ToolManifest>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            manifests: HashMap::new(),
        }
    }

    /// Register a tool. Caches its manifest on registration.
    /// If a tool with the same name exists, it is replaced.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let manifest = tool.manifest();
        let name = manifest.name.clone();
        self.manifests.insert(name.clone(), manifest);
        self.tools.insert(name, tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Get a tool's manifest by name.
    pub fn manifest(&self, name: &str) -> Option<&ToolManifest> {
        self.manifests.get(name)
    }

    /// List all registered tool manifests.
    pub fn list_manifests(&self) -> Vec<&ToolManifest> {
        self.manifests.values().collect()
    }

    /// List all registered tool signatures.
    pub fn list(&self) -> Vec<ToolSignature> {
        self.tools.values().map(|t| t.signature()).collect()
    }

    /// Execute a tool by name.
    pub fn execute(
        &self,
        name: &str,
        input: ToolInput,
        engine: &Engine,
    ) -> AgentResult<ToolOutput> {
        let tool = self
            .get(name)
            .ok_or_else(|| AgentError::ToolNotFound { name: name.into() })?;
        tool.execute(engine, input)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::agent::tool_manifest::{Capability, DangerInfo, DangerLevel, ToolSource};

    struct DummyTool;
    impl Tool for DummyTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "dummy".into(),
                description: "A test tool".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok("dummy result"))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "dummy".into(),
                description: "A test tool".into(),
                parameters: vec![],
                danger: DangerInfo {
                    level: DangerLevel::Safe,
                    capabilities: HashSet::from([Capability::ReadKg]),
                    description: "Safe test tool".into(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    #[test]
    fn register_and_list() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        assert_eq!(reg.len(), 1);
        let sigs = reg.list();
        assert_eq!(sigs[0].name, "dummy");
    }

    #[test]
    fn manifest_cached_on_register() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let m = reg.manifest("dummy").unwrap();
        assert_eq!(m.name, "dummy");
        assert_eq!(m.danger.level, DangerLevel::Safe);
    }

    #[test]
    fn list_manifests() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let manifests = reg.list_manifests();
        assert_eq!(manifests.len(), 1);
    }

    #[test]
    fn manifest_missing_returns_none() {
        let reg = ToolRegistry::new();
        assert!(reg.manifest("nonexistent").is_none());
    }

    #[test]
    fn get_missing_tool() {
        let reg = ToolRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn tool_input_builder() {
        let input = ToolInput::new()
            .with_param("symbol", "Sun")
            .with_param("direction", "from");
        assert_eq!(input.get("symbol"), Some("Sun"));
        assert_eq!(input.get("direction"), Some("from"));
        assert_eq!(input.get("missing"), None);
    }
}
