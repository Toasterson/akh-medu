//! Tool-query tool: list which tools are available to a given agent scope.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_graph::ToolGraphRegistrar;
use crate::agent::tool_manifest::{Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource};
use crate::engine::Engine;

/// Introspect this akh's tools.
///
/// Unscoped (no `agent`), it reports the **live** registry — every tool callable
/// right now, including per-turn session tools (e.g. MCP servers pushed over ACP)
/// that never land in the durable KG — then notes any tools known only to the KG
/// (durable/peer entries not locally loaded). With an `agent` label it answers
/// the KG access-scope question "which tools may agent X invoke?" for peer
/// discovery.
pub struct ToolQueryTool;

impl Tool for ToolQueryTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "tool_query".into(),
            description: "List the tools this akh can call right now (incl. session MCP tools); with an agent label, list tools registered for that agent in the KG.".into(),
            parameters: vec![
                ToolParam {
                    name: "agent".into(),
                    description: "Agent label to filter by (e.g. \"agent:tecton\"). Omit to list this akh's live tools.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let registrar = ToolGraphRegistrar::init(engine)?;

        // Scoped: "which tools may agent X invoke?" is a KG access-scope question
        // (peer discovery) — answer from the durable graph.
        if let Some(agent) = input.get("agent") {
            let mut tools = registrar.query_tools_for_agent(agent);
            if tools.is_empty() {
                return Ok(ToolOutput::ok(format!(
                    "No tools registered for agent scope \"{agent}\"."
                )));
            }
            tools.sort();
            return Ok(ToolOutput::ok(format!(
                "{} tool(s) registered for \"{agent}\":\n{}",
                tools.len(),
                tools.join("\n")
            )));
        }

        // Unscoped: the live registry is authoritative for what is callable now.
        let live = engine.live_tools();
        if live.is_empty() {
            // No live view populated (e.g. a non-agent engine context) — fall
            // back to the durable KG projection so we never regress to nothing.
            let mut kg = registrar.list_all_tools();
            if kg.is_empty() {
                return Ok(ToolOutput::ok("No tools registered."));
            }
            kg.sort();
            return Ok(ToolOutput::ok(format!(
                "{} tool(s) in the knowledge graph:\n{}",
                kg.len(),
                kg.join("\n")
            )));
        }

        let live_names: HashSet<&str> = live.iter().map(|t| t.name.as_str()).collect();
        let mut live_sorted = live.clone();
        live_sorted.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = format!("{} tool(s) callable now:\n", live_sorted.len());
        for t in &live_sorted {
            out.push_str(&format!("  {} [{}] — {}\n", t.name, t.origin, t.description));
        }

        // KG-only extras: labels are `tool:{name}`; compare on the bare name so a
        // durable/peer tool not in the live registry is surfaced (and stale ones
        // that survived reconciliation are visible rather than silently missing).
        let mut extras: Vec<String> = registrar
            .list_all_tools()
            .into_iter()
            .filter(|label| {
                let bare = label.strip_prefix("tool:").unwrap_or(label);
                !live_names.contains(bare)
            })
            .collect();
        if !extras.is_empty() {
            extras.sort();
            out.push_str(&format!(
                "\n{} tool(s) known to the knowledge graph but not locally callable:\n",
                extras.len()
            ));
            for label in &extras {
                out.push_str(&format!("  {label}\n"));
            }
        }

        Ok(ToolOutput::ok(out))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "tool_query".into(),
            description: "List the tools this akh can call right now (incl. session MCP tools); with an agent label, list tools registered for that agent in the KG.".into(),
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
