//! Tool manifest registration in the knowledge graph.
//!
//! Persists every tool manifest as triples so peer agents and federation
//! nodes can discover which tools are available and who may invoke them.
//!
//! Triple schema per tool:
//! ```text
//! tool:{name} -> tool:is_tool      -> akhomed:tool
//! tool:{name} -> tool:schema       -> "{compact_json_schema}"
//! tool:{name} -> tool:provider     -> "{provider}"
//! tool:{name} -> tool:danger_level -> "{Safe|Cautious|Dangerous|Critical}"
//! tool:{name} -> tool:access       -> "{agent_label}"   (one per agent)
//! tool:{name} -> tool:endpoint     -> "{url}"           (remote tools only)
//! ```

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::tool::ToolRegistry;
use super::tool_manifest::{ToolManifest, ToolParamSchema};

/// Well-known relation SymbolIds used for tool manifest triples.
#[derive(Debug, Clone)]
pub struct ToolGraphPredicates {
    /// `tool:schema` — JSON parameter schema.
    pub schema: SymbolId,
    /// `tool:provider` — who supplies this tool (e.g. "akhomed").
    pub provider: SymbolId,
    /// `tool:access` — agent label that may invoke this tool.
    pub access: SymbolId,
    /// `tool:endpoint` — remote endpoint URL (remote/federated tools only).
    pub endpoint: SymbolId,
    /// `tool:danger_level` — DangerLevel string (Safe/Cautious/Dangerous/Critical).
    pub danger_level: SymbolId,
    /// `tool:is_tool` — type-marker relation: subject is an invocable tool.
    pub is_tool: SymbolId,
}

impl ToolGraphPredicates {
    /// Resolve (or create) all well-known tool predicates in the engine.
    pub fn init(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            schema: engine.resolve_or_create_relation("tool:schema")?,
            provider: engine.resolve_or_create_relation("tool:provider")?,
            access: engine.resolve_or_create_relation("tool:access")?,
            endpoint: engine.resolve_or_create_relation("tool:endpoint")?,
            danger_level: engine.resolve_or_create_relation("tool:danger_level")?,
            is_tool: engine.resolve_or_create_relation("tool:is_tool")?,
        })
    }
}

/// Registers and queries tool manifests as knowledge-graph triples.
pub struct ToolGraphRegistrar<'e> {
    engine: &'e Engine,
    predicates: ToolGraphPredicates,
    /// Entity symbol for the type-marker object `akhomed:tool`.
    tool_type: SymbolId,
}

impl<'e> ToolGraphRegistrar<'e> {
    /// Initialise: resolve predicates and the `akhomed:tool` type-marker entity.
    pub fn init(engine: &'e Engine) -> AkhResult<Self> {
        let predicates = ToolGraphPredicates::init(engine)?;
        let tool_type = engine.resolve_or_create_entity("akhomed:tool")?;
        Ok(Self {
            engine,
            predicates,
            tool_type,
        })
    }

    /// Register one tool manifest as KG triples.
    ///
    /// - `provider` — e.g. `"akhomed"`
    /// - `access_agents` — labels of agents that may invoke this tool.
    pub fn register_tool(
        &self,
        manifest: &ToolManifest,
        provider: &str,
        access_agents: &[&str],
    ) -> AkhResult<()> {
        let tool_label = format!("tool:{}", manifest.name);
        let subject = self.engine.resolve_or_create_entity(&tool_label)?;

        // tool:is_tool -> akhomed:tool
        self.triple(subject, self.predicates.is_tool, self.tool_type)?;

        // tool:schema -> "{compact_json_schema}"
        let schema_json = compact_schema_json(&manifest.parameters);
        let schema_sym = self.engine.resolve_or_create_entity(&schema_json)?;
        self.triple(subject, self.predicates.schema, schema_sym)?;

        // tool:provider -> "{provider}"
        let provider_sym = self.engine.resolve_or_create_entity(provider)?;
        self.triple(subject, self.predicates.provider, provider_sym)?;

        // tool:danger_level -> "{level}"
        let danger_sym = self
            .engine
            .resolve_or_create_entity(&manifest.danger.level.to_string())?;
        self.triple(subject, self.predicates.danger_level, danger_sym)?;

        // tool:access -> "{agent_label}"  — one triple per agent
        for agent_label in access_agents {
            let agent_sym = self.engine.resolve_or_create_entity(agent_label)?;
            self.triple(subject, self.predicates.access, agent_sym)?;
        }

        Ok(())
    }

    /// Register a remote MCP tool discovered via federation.
    ///
    /// Adds a `tool:endpoint` triple in addition to the standard triples so
    /// agents know how to reach the tool.
    pub fn register_remote(
        &self,
        name: &str,
        endpoint: &str,
        schema_json: &str,
        provider: &str,
        access_agents: &[&str],
    ) -> AkhResult<()> {
        let tool_label = format!("tool:{name}");
        let subject = self.engine.resolve_or_create_entity(&tool_label)?;

        self.triple(subject, self.predicates.is_tool, self.tool_type)?;

        let endpoint_sym = self.engine.resolve_or_create_entity(endpoint)?;
        self.triple(subject, self.predicates.endpoint, endpoint_sym)?;

        let schema_sym = self.engine.resolve_or_create_entity(schema_json)?;
        self.triple(subject, self.predicates.schema, schema_sym)?;

        let provider_sym = self.engine.resolve_or_create_entity(provider)?;
        self.triple(subject, self.predicates.provider, provider_sym)?;

        for agent_label in access_agents {
            let agent_sym = self.engine.resolve_or_create_entity(agent_label)?;
            self.triple(subject, self.predicates.access, agent_sym)?;
        }

        Ok(())
    }

    /// Register all manifests in a [`ToolRegistry`].
    ///
    /// Returns the number of tools successfully registered.
    /// Errors on individual tools are logged as warnings but do not abort
    /// the batch — the caller can still use the partial result.
    pub fn register_all(
        &self,
        registry: &ToolRegistry,
        provider: &str,
        access_agents: &[&str],
    ) -> usize {
        let mut registered = 0usize;
        for manifest in registry.list_manifests() {
            match self.register_tool(manifest, provider, access_agents) {
                Ok(()) => registered += 1,
                Err(e) => tracing::warn!(
                    tool = %manifest.name,
                    error = %e,
                    "tool_graph: failed to register tool manifest"
                ),
            }
        }
        registered
    }

    /// Return the labels of all tools that `agent_label` has access to.
    ///
    /// Walks all `tool:access` triples whose object is the agent symbol and
    /// collects the subject labels.  Returns an empty vec if the agent does
    /// not exist or has no tool grants.
    pub fn query_tools_for_agent(&self, agent_label: &str) -> Vec<String> {
        let Ok(agent_sym) = self.engine.lookup_symbol(agent_label) else {
            return Vec::new();
        };
        self.engine
            .triples_to(agent_sym)
            .into_iter()
            .filter(|t| t.predicate == self.predicates.access)
            .map(|t| self.engine.resolve_label(t.subject))
            .collect()
    }

    /// Return labels of all registered tools regardless of access scope.
    pub fn list_all_tools(&self) -> Vec<String> {
        let Ok(tool_type_sym) = self.engine.lookup_symbol("akhomed:tool") else {
            return Vec::new();
        };
        self.engine
            .triples_to(tool_type_sym)
            .into_iter()
            .filter(|t| t.predicate == self.predicates.is_tool)
            .map(|t| self.engine.resolve_label(t.subject))
            .collect()
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    fn triple(&self, subject: SymbolId, predicate: SymbolId, object: SymbolId) -> AkhResult<()> {
        self.engine.add_triple(&Triple::new(subject, predicate, object))
    }
}

/// Build a compact JSON schema string from a list of parameter schemas.
///
/// Output example:
/// `{"type":"object","properties":{"term":{"type":"string"}},"required":["term"]}`
fn compact_schema_json(params: &[ToolParamSchema]) -> String {
    let mut props = String::new();
    let mut required: Vec<&str> = Vec::new();

    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            props.push(',');
        }
        let desc = serde_json::to_string(&p.description).unwrap_or_else(|_| "\"\"".into());
        props.push_str(&format!(
            r#""{}":{{"type":"{}","description":{}}}"#,
            p.name, p.param_type, desc
        ));
        if p.required {
            required.push(&p.name);
        }
    }

    let mut out = format!(r#"{{"type":"object","properties":{{{props}}}"#);
    if !required.is_empty() {
        out.push_str(r#","required":["#);
        for (i, r) in required.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(r);
            out.push('"');
        }
        out.push(']');
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::agent::tool_manifest::{DangerInfo, DangerLevel, ToolSource};
    use crate::engine::{Engine, EngineConfig};

    fn make_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension(64),
            ..Default::default()
        })
        .expect("engine")
    }

    fn dummy_manifest(name: &str) -> ToolManifest {
        ToolManifest {
            name: name.into(),
            description: "A test tool.".into(),
            parameters: vec![
                ToolParamSchema::required("term", "Search term."),
                ToolParamSchema::optional("limit", "Max results."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([crate::agent::tool_manifest::Capability::ReadKg]),
                description: "Read-only.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }

    #[test]
    fn register_tool_creates_triples() {
        let engine = make_engine();
        let reg = ToolGraphRegistrar::init(&engine).unwrap();
        let manifest = dummy_manifest("search");
        reg.register_tool(&manifest, "akhomed", &["agent:tecton"]).unwrap();

        // tool:search should now exist
        let subject = engine.lookup_symbol("tool:search").unwrap();
        // At least one triple from subject
        assert!(!engine.triples_from(subject).is_empty());
    }

    #[test]
    fn query_tools_for_agent_returns_registered_tools() {
        let engine = make_engine();
        let reg = ToolGraphRegistrar::init(&engine).unwrap();
        reg.register_tool(&dummy_manifest("kg_query"), "akhomed", &["agent:tecton"])
            .unwrap();
        reg.register_tool(&dummy_manifest("kg_mutate"), "akhomed", &["agent:tecton"])
            .unwrap();
        // Tool not accessible to another agent
        reg.register_tool(&dummy_manifest("private_tool"), "akhomed", &["agent:thoth"])
            .unwrap();

        let tecton_tools = reg.query_tools_for_agent("agent:tecton");
        assert!(tecton_tools.contains(&"tool:kg_query".to_string()));
        assert!(tecton_tools.contains(&"tool:kg_mutate".to_string()));
        assert!(!tecton_tools.contains(&"tool:private_tool".to_string()));
    }

    #[test]
    fn register_remote_adds_endpoint_triple() {
        let engine = make_engine();
        let reg = ToolGraphRegistrar::init(&engine).unwrap();
        let schema = r#"{"type":"object","properties":{"term":{"type":"string"}}}"#;
        reg.register_remote(
            "seshat_search",
            "http://localhost:8200",
            schema,
            "seshat",
            &["agent:tecton"],
        )
        .unwrap();

        let subject = engine.lookup_symbol("tool:seshat_search").unwrap();
        let triples = engine.triples_from(subject);
        let endpoint_pred = engine.lookup_symbol("tool:endpoint").unwrap();
        assert!(triples.iter().any(|t| t.predicate == endpoint_pred));
    }

    #[test]
    fn compact_schema_json_roundtrip() {
        let params = vec![
            ToolParamSchema::required("term", "Search term."),
            ToolParamSchema::optional("limit", "Max results."),
        ];
        let json = compact_schema_json(&params);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "object");
        assert!(parsed["properties"]["term"].is_object());
        assert!(parsed["properties"]["limit"].is_object());
        let req = parsed["required"].as_array().unwrap();
        assert_eq!(req.len(), 1);
        assert_eq!(req[0], "term");
    }

    #[test]
    fn list_all_tools_returns_registered() {
        let engine = make_engine();
        let reg = ToolGraphRegistrar::init(&engine).unwrap();
        reg.register_tool(&dummy_manifest("alpha"), "akhomed", &[]).unwrap();
        reg.register_tool(&dummy_manifest("beta"), "akhomed", &[]).unwrap();
        let all = reg.list_all_tools();
        assert!(all.contains(&"tool:alpha".to_string()));
        assert!(all.contains(&"tool:beta".to_string()));
    }
}
