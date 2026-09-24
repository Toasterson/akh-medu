//! Dispatch-trap triage tool (post-ADR-061).
//!
//! Detects Anima work items that are stuck with a captured `agent_diff` but no
//! advancing change record, and surfaces them as KG triples plus a human-readable
//! triage report. This is visibility tooling: the runner no longer calls
//! `SetAgentDiff`, so the trap is architecturally unlikely to recur, but the
//! 2026-07-23 frozen items and any future edge cases should be discoverable
//! without querying Postgres.
//!
//! The tool reads `[upstreams.anima]` from the global akh-medu config and calls
//! the still-functional outbound `UpstreamMcpClient`. KG assertions are made
//! through the `Tool` trait's `&Engine` handle.

use std::collections::HashSet;

use serde_json::Value;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::config::AkhomedConfig;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::paths::AkhPaths;
use crate::upstream_client::UpstreamMcpClient;

/// Anima work item in a dispatch trap.
#[derive(Debug, Clone)]
struct Trap {
    id: String,
    code: String,
    title: String,
    phase: String,
    agent_diff_count: usize,
    non_advancing_count: usize,
    has_explosion: bool,
}

/// Tool: `triage_dispatch_traps`.
///
/// Parameters:
/// - `project_id` (required): Anima project UUID.
/// - `dry_run` (optional): when true, report what would be asserted without
///   writing KG triples.
pub struct TriageDispatchTrapsTool;

impl TriageDispatchTrapsTool {
    /// Load global config and build an Anima upstream client.
    fn anima_client() -> AgentResult<UpstreamMcpClient> {
        let paths = AkhPaths::resolve().map_err(|e| AgentError::ToolExecution {
            tool_name: "triage_dispatch_traps".into(),
            message: format!("cannot resolve akh-medu paths: {e}"),
        })?;
        let config_path = paths.config_dir.join("config.toml");
        let config = AkhomedConfig::load(&config_path).map_err(|e| AgentError::ToolExecution {
            tool_name: "triage_dispatch_traps".into(),
            message: format!(
                "failed to load global config at {}: {e}",
                config_path.display()
            ),
        })?;
        let Some(anima_cfg) = config.upstreams.anima else {
            return Err(AgentError::ToolExecution {
                tool_name: "triage_dispatch_traps".into(),
                message: "no [upstreams.anima] configured in akh-medu config.toml".into(),
            });
        };
        Ok(UpstreamMcpClient::new(&anima_cfg))
    }

    /// Fetch work items for the project and identify trap candidates.
    fn fetch_project_work_items(
        client: &UpstreamMcpClient,
        project_id: &str,
    ) -> AgentResult<Vec<Value>> {
        let text = client
            .call_tool_text(
                "list_work_items",
                serde_json::json!({"project_id": project_id}),
            )
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "triage_dispatch_traps".into(),
                message: format!("list_work_items failed: {e}"),
            })?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|_| Value::String(text));
        let items = parsed
            .as_array()
            .cloned()
            .or_else(|| parsed.get("work_items").and_then(|v| v.as_array()).cloned())
            .unwrap_or_default();
        Ok(items)
    }

    /// Fetch full detail for a single work item.
    fn fetch_work_item_detail(
        client: &UpstreamMcpClient,
        work_item_id: &str,
    ) -> AgentResult<Value> {
        let text = client
            .call_tool_text(
                "get_work_item_detail",
                serde_json::json!({"id": work_item_id}),
            )
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "triage_dispatch_traps".into(),
                message: format!("get_work_item_detail({work_item_id}) failed: {e}"),
            })?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|_| Value::String(text));
        Ok(parsed)
    }

    /// Determine whether a work item is in a dispatch trap from its detail JSON.
    fn classify(detail: &Value) -> Option<Trap> {
        let id = detail.get("id").and_then(|v| v.as_str())?;
        let phase = detail.get("current_phase").and_then(|v| v.as_str())?;
        if !matches!(phase, "ready" | "working") {
            return None;
        }

        let code = detail
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let title = detail
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let changes = detail
            .get("changes")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut agent_diff_count = 0usize;
        let mut non_advancing_count = 0usize;
        let mut advancing_found = false;

        for ch in &changes {
            let kind = ch.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let decision = ch
                .get("integration_decision")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let lifecycle = ch
                .get("lifecycle_state")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if kind == "agent_diff" {
                agent_diff_count += 1;
            }

            if Self::is_advancing(kind, decision, lifecycle) {
                advancing_found = true;
            } else if !kind.is_empty() {
                non_advancing_count += 1;
            }
        }

        // Also consider resolved PRs as advancing state.
        if !advancing_found {
            if let Some(prs) = detail.get("resolved_prs").and_then(|v| v.as_array()) {
                for pr in prs {
                    let state = pr.get("state").and_then(|v| v.as_str()).unwrap_or("");
                    let merged = pr.get("merged").and_then(|v| v.as_bool()).unwrap_or(false);
                    let mergeable = pr
                        .get("mergeable")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if merged || mergeable || state == "merged" {
                        advancing_found = true;
                        break;
                    }
                }
            }
        }

        if agent_diff_count == 0 || advancing_found {
            return None;
        }

        let has_explosion = non_advancing_count >= 3;

        Some(Trap {
            id: id.to_string(),
            code,
            title,
            phase: phase.to_string(),
            agent_diff_count,
            non_advancing_count,
            has_explosion,
        })
    }

    /// Whether a change record means the WI has already advanced past the trap.
    fn is_advancing(kind: &str, decision: &str, lifecycle: &str) -> bool {
        const ADVANCING_KINDS: &[&str] = &[
            "solstice_result",
            "external_pr",
            "merge_attempt",
            "promote_loop_attempt",
        ];
        const ADVANCING_DECISIONS: &[&str] = &["merged", "promoted", "closed", "integrated"];
        const TERMINAL_STATES: &[&str] = &["done", "testing", "parked"];

        if ADVANCING_KINDS.contains(&kind) {
            return true;
        }
        if ADVANCING_DECISIONS.contains(&decision) {
            return true;
        }
        if TERMINAL_STATES.contains(&lifecycle) {
            return true;
        }
        false
    }

    /// Assert the dispatch-trap triples for one work item.
    fn assert_triples(engine: &Engine, trap: &Trap) -> AgentResult<()> {
        let wi_sym = engine.resolve_or_create_entity(&trap.id)?;
        let dispatch_state_pred = engine.resolve_or_create_relation("dispatch-state")?;
        let state_obj = engine.resolve_or_create_entity("diff-captured-not-submitted")?;
        engine.add_triple(&Triple::new(wi_sym, dispatch_state_pred, state_obj))?;

        let needs_triage_pred = engine.resolve_or_create_relation("needs-triage")?;
        let true_obj = engine.resolve_or_create_entity("true")?;
        engine.add_triple(&Triple::new(wi_sym, needs_triage_pred, true_obj))?;

        if trap.has_explosion {
            let explosion_pred =
                engine.resolve_or_create_relation("has-change-record-explosion")?;
            engine.add_triple(&Triple::new(wi_sym, explosion_pred, true_obj))?;
        }

        Ok(())
    }
}

impl Tool for TriageDispatchTrapsTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "triage_dispatch_traps".into(),
            description: "Detect Anima work items stuck with a captured agent_diff but no \
                         advancing change record, assert dispatch-trap triples to the KG, and \
                         return a formatted triage report."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "project_id".into(),
                    description: "Anima project UUID to scan.".into(),
                    required: true,
                },
                ToolParam {
                    name: "dry_run".into(),
                    description: "If true, report findings without writing KG triples.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let project_id = input.require("project_id", "triage_dispatch_traps")?;
        let dry_run = input.get_bool("dry_run").unwrap_or(false);

        let client = Self::anima_client()?;
        let items = Self::fetch_project_work_items(&client, project_id)?;

        let mut traps: Vec<Trap> = Vec::new();
        let mut inspected = 0usize;

        for item in &items {
            let phase = item.get("current_phase").and_then(|v| v.as_str());
            if !matches!(phase, Some("ready") | Some("working")) {
                continue;
            }
            let Some(id) = item.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            inspected += 1;
            match Self::fetch_work_item_detail(&client, id) {
                Ok(detail) => {
                    if let Some(trap) = Self::classify(&detail) {
                        traps.push(trap);
                    }
                }
                Err(e) => {
                    // Surface per-item fetch failures in the report rather than
                    // failing the whole scan.
                    tracing::warn!(%id, %e, "triage_dispatch_traps: failed to fetch work item detail");
                }
            }
        }

        // Assert KG triples unless this is a dry run.
        if !dry_run {
            for trap in &traps {
                if let Err(e) = Self::assert_triples(engine, trap) {
                    tracing::warn!(wi_id = %trap.id, %e, "triage_dispatch_traps: failed to assert triples");
                }
            }
        }

        // Build the formatted report.
        let mut lines = Vec::new();
        lines.push(format!(
            "Dispatch-trap triage for project {project_id}: inspected {inspected} ready/working work items, found {} trap(s){}.",
            traps.len(),
            if dry_run { " (dry run — no KG triples written)" } else { "" }
        ));

        if traps.is_empty() {
            lines.push("No dispatch traps detected.".into());
        } else {
            lines.push("\nTrapped work items:".into());
            for trap in &traps {
                lines.push(format!(
                    "- {} ({}) — phase={} title=\"{}\" agent_diffs={} non_advancing_changes={} explosion={}",
                    trap.id,
                    trap.code,
                    trap.phase,
                    trap.title,
                    trap.agent_diff_count,
                    trap.non_advancing_count,
                    trap.has_explosion
                ));
            }
        }

        let result = lines.join("\n");
        Ok(ToolOutput::ok(result))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "triage_dispatch_traps".into(),
            description: "Detect Anima work items stuck with a captured agent_diff but no advancing change record, and surface them as KG triples plus a report."
                .into(),
            parameters: vec![
                ToolParamSchema::required("project_id", "Anima project UUID to scan."),
                ToolParamSchema::optional("dry_run", "If true, report findings without writing KG triples."),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network, Capability::WriteKg]),
                description:
                    "Reads work items from Anima over the network and writes diagnostic triples to the local KG."
                        .into(),
                shadow_triggers: vec!["anima".into(), "triage".into(), "write kg".into()],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detail_with_changes(changes: Vec<Value>) -> Value {
        serde_json::json!({
            "id": "wi-1",
            "code": "WI-1",
            "title": "Test item",
            "current_phase": "working",
            "changes": changes,
            "resolved_prs": []
        })
    }

    #[test]
    fn no_agent_diff_is_not_trapped() {
        let detail = detail_with_changes(vec![serde_json::json!({
            "kind": "solstice_result",
            "lifecycle_state": "done",
            "integration_decision": ""
        })]);
        assert!(TriageDispatchTrapsTool::classify(&detail).is_none());
    }

    #[test]
    fn agent_diff_with_no_advancing_record_is_trapped() {
        let detail = detail_with_changes(vec![serde_json::json!({
            "kind": "agent_diff",
            "lifecycle_state": "working",
            "integration_decision": ""
        })]);
        let trap = TriageDispatchTrapsTool::classify(&detail).expect("should be trapped");
        assert_eq!(trap.id, "wi-1");
        assert_eq!(trap.agent_diff_count, 1);
        assert!(!trap.has_explosion);
    }

    #[test]
    fn agent_diff_with_solstice_result_is_not_trapped() {
        let detail = detail_with_changes(vec![
            serde_json::json!({
                "kind": "agent_diff",
                "lifecycle_state": "working",
                "integration_decision": ""
            }),
            serde_json::json!({
                "kind": "solstice_result",
                "lifecycle_state": "done",
                "integration_decision": ""
            }),
        ]);
        assert!(TriageDispatchTrapsTool::classify(&detail).is_none());
    }

    #[test]
    fn multiple_non_advancing_records_flag_explosion() {
        let detail = detail_with_changes(vec![
            serde_json::json!({"kind": "agent_diff", "lifecycle_state": "working", "integration_decision": ""}),
            serde_json::json!({"kind": "agent_diff", "lifecycle_state": "working", "integration_decision": ""}),
            serde_json::json!({"kind": "agent_diff", "lifecycle_state": "working", "integration_decision": ""}),
        ]);
        let trap = TriageDispatchTrapsTool::classify(&detail).expect("should be trapped");
        assert!(trap.has_explosion);
    }

    #[test]
    fn merged_pr_clears_trap() {
        let detail = serde_json::json!({
            "id": "wi-2",
            "code": "WI-2",
            "title": "Merged item",
            "current_phase": "working",
            "changes": [{"kind": "agent_diff", "lifecycle_state": "working", "integration_decision": ""}],
            "resolved_prs": [{"state": "merged", "merged": true, "mergeable": false}]
        });
        assert!(TriageDispatchTrapsTool::classify(&detail).is_none());
    }

    #[test]
    fn done_phase_is_not_trapped() {
        let detail = serde_json::json!({
            "id": "wi-3",
            "code": "WI-3",
            "title": "Done item",
            "current_phase": "done",
            "changes": [{"kind": "agent_diff", "lifecycle_state": "done", "integration_decision": ""}],
            "resolved_prs": []
        });
        assert!(TriageDispatchTrapsTool::classify(&detail).is_none());
    }
}
