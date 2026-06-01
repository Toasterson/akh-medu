//! Tool-query tool: list which tools are available to a given agent scope.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_graph::ToolGraphRegistrar;
use crate::agent::tool_manifest::{Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource};
use crate::engine::Engine;

/// Query which tools are registered in the KG for a given agent scope.
///
/// Accepts an optional `agent` parameter (e.g. `agent:tecton`).
/// When omitted, returns all registered tools regardless of access scope.
pub struct ToolQueryTool;

impl Tool for ToolQueryTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "tool_query".into(),
            description: "List tools registered in the knowledge graph, optionally filtered by agent scope.".into(),
            parameters: vec![
                ToolParam {
                    name: "agent".into(),
                    description: "Agent label to filter by (e.g. \"agent:tecton\"). Omit to list all tools.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let registrar = ToolGraphRegistrar::init(engine)?;

        let tools = if let Some(agent) = input.get("agent") {
            registrar.query_tools_for_agent(agent)
        } else {
            registrar.list_all_tools()
        };

        if tools.is_empty() {
            let msg = input.get("agent")
                .map(|a| format!("No tools registered for agent scope \"{a}\"."))
                .unwrap_or_else(|| "No tools registered in the knowledge graph.".into());
            return Ok(ToolOutput::ok(msg));
        }

        let mut sorted = tools;
        sorted.sort();
        let result = format!(
            "{} tool(s) found:\n{}",
            sorted.len(),
            sorted.join("\n")
        );
        Ok(ToolOutput::ok(result))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "tool_query".into(),
            description: "List tools registered in the knowledge graph, optionally filtered by agent scope.".into(),
            parameters: vec![
                ToolParamSchema::optional(
                    "agent",
                    "Agent label to filter by (e.g. \"agent:tecton\"). Omit to list all tools.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only query of tool manifests stored in the KG.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
