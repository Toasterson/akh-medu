//! Agent management tools: multi-agent orchestration via akhomed workspaces.
//!
//! Four tools that let one agent create, list, message, and retire other
//! agent workspaces. All calls route through akhomed's REST API using `ureq`.
//! When akhomed is not running, tools return a descriptive error.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Discover the akhomed base URL, returning an error output if unavailable.
fn discover_base_url() -> Result<String, ToolOutput> {
    let paths = crate::paths::AkhPaths::resolve().map_err(|e| {
        ToolOutput::err(format!("Cannot resolve akh paths: {e}"))
    })?;
    match crate::client::discover_server(&paths) {
        Some(info) => Ok(info.base_url()),
        None => Err(ToolOutput::err(
            "akhomed is not running. Start it with `akh serve` first.",
        )),
    }
}

/// Derive the current workspace name from the engine's data_dir path.
///
/// Workspace data dirs follow `…/workspaces/<name>/kg`.  We extract `<name>`
/// from the grandparent of the `kg` directory.  Falls back to `"unknown"`.
fn current_workspace_name(engine: &Engine) -> String {
    engine
        .config()
        .data_dir
        .as_ref()
        .and_then(|p| p.parent()) // workspaces/<name>
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

// ===========================================================================
// AgentListTool
// ===========================================================================

/// List all agent workspaces managed by akhomed.
pub struct AgentListTool;

impl Tool for AgentListTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "agent_list".into(),
            description: "List all agent workspaces managed by akhomed.".into(),
            parameters: vec![],
        }
    }

    fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
        let base_url = match discover_base_url() {
            Ok(u) => u,
            Err(out) => return Ok(out),
        };

        let url = format!("{base_url}/workspaces");
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(5))
            .build();

        match agent.get(&url).call() {
            Ok(resp) => match resp.into_string() {
                Ok(body) => Ok(ToolOutput::ok(format!("Agent workspaces:\n{body}"))),
                Err(e) => Ok(ToolOutput::err(format!("Failed to read response: {e}"))),
            },
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(ToolOutput::err(format!("HTTP {code}: {body}")))
            }
            Err(ureq::Error::Transport(t)) => {
                Ok(ToolOutput::err(format!("Transport error: {t}")))
            }
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "agent_list".into(),
            description: "Lists all agent workspaces via akhomed — read-only network call.".into(),
            parameters: vec![],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::Network]),
                description: "Read-only listing of agent workspaces.".into(),
                shadow_triggers: vec![
                    "list".into(),
                    "agents".into(),
                    "workspaces".into(),
                    "inventory".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}

// ===========================================================================
// AgentSpawnTool
// ===========================================================================

/// Create a new agent workspace and optionally assign it an Ennead role.
pub struct AgentSpawnTool;

impl Tool for AgentSpawnTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "agent_spawn".into(),
            description: "Create a new agent workspace, optionally with an Ennead archetype role."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "name".into(),
                    description: "Name for the new workspace.".into(),
                    required: true,
                },
                ToolParam {
                    name: "role".into(),
                    description: "Ennead archetype name (e.g. Investigator, Executor).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let name = input.require("name", "agent_spawn")?;
        let role = input.get("role");

        let base_url = match discover_base_url() {
            Ok(u) => u,
            Err(out) => return Ok(out),
        };

        let http = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(10))
            .build();

        // 1. Create the workspace.
        let create_url = format!("{base_url}/workspaces/{name}");
        match http.post(&create_url).call() {
            Ok(resp) if resp.status() == 200 => { /* created */ }
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                return Ok(ToolOutput::err(format!(
                    "Failed to create workspace \"{name}\": {body}"
                )));
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                return Ok(ToolOutput::err(format!(
                    "HTTP {code} creating workspace \"{name}\": {body}"
                )));
            }
            Err(ureq::Error::Transport(t)) => {
                return Ok(ToolOutput::err(format!("Transport error: {t}")));
            }
        }

        // 2. Assign role via the write-once assign-role endpoint if provided.
        if let Some(role_name) = role {
            let assign_url = format!("{base_url}/workspaces/{name}/assign-role");
            let payload = serde_json::json!({ "role": role_name });

            match http
                .post(&assign_url)
                .set("Content-Type", "application/json")
                .send_string(&payload.to_string())
            {
                Ok(_) => {}
                Err(ureq::Error::Status(code, resp)) => {
                    let body = resp.into_string().unwrap_or_default();
                    return Ok(ToolOutput::err(format!(
                        "Workspace \"{name}\" created but failed to assign role: HTTP {code}: {body}"
                    )));
                }
                Err(ureq::Error::Transport(t)) => {
                    return Ok(ToolOutput::err(format!(
                        "Workspace \"{name}\" created but failed to assign role: {t}"
                    )));
                }
            }
        }

        let role_msg = role
            .map(|r| format!(" with role \"{r}\""))
            .unwrap_or_default();
        Ok(ToolOutput::ok(format!(
            "Agent workspace \"{name}\" created{role_msg}."
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "agent_spawn".into(),
            description: "Creates a new agent workspace via akhomed.".into(),
            parameters: vec![
                ToolParamSchema::required("name", "Name for the new workspace."),
                ToolParamSchema::optional(
                    "role",
                    "Ennead archetype name (e.g. Investigator, Executor).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network, Capability::WriteKg]),
                description: "Creates a workspace and optionally writes role triples.".into(),
                shadow_triggers: vec![
                    "spawn".into(),
                    "create".into(),
                    "new".into(),
                    "agent".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}

// ===========================================================================
// AgentMessageTool
// ===========================================================================

/// Send a message to another agent workspace by ingesting triples into its KG.
pub struct AgentMessageTool;

impl Tool for AgentMessageTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "agent_message".into(),
            description:
                "Send a message to another agent workspace by ingesting triples into its KG."
                    .into(),
            parameters: vec![
                ToolParam {
                    name: "workspace".into(),
                    description: "Target workspace name.".into(),
                    required: true,
                },
                ToolParam {
                    name: "message".into(),
                    description: "Message text to deliver.".into(),
                    required: true,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let workspace = input.require("workspace", "agent_message")?;
        let message = input.require("message", "agent_message")?;

        let base_url = match discover_base_url() {
            Ok(u) => u,
            Err(out) => return Ok(out),
        };

        let sender = current_workspace_name(engine);

        let triples: Vec<(String, String, String, f32)> = vec![
            (
                "agent:inbox".into(),
                "agent:message".into(),
                message.to_string(),
                1.0,
            ),
            (
                "agent:inbox".into(),
                "agent:from".into(),
                sender.clone(),
                1.0,
            ),
        ];

        let url = format!("{base_url}/workspaces/{workspace}/ingest");
        let payload = serde_json::json!({ "triples": triples });

        let http = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(10))
            .build();

        match http
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&payload.to_string())
        {
            Ok(resp) if resp.status() == 200 => Ok(ToolOutput::ok(format!(
                "Message delivered to \"{workspace}\" from \"{sender}\"."
            ))),
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(ToolOutput::err(format!(
                    "Failed to deliver message to \"{workspace}\": {body}"
                )))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(ToolOutput::err(format!(
                    "HTTP {code} delivering message to \"{workspace}\": {body}"
                )))
            }
            Err(ureq::Error::Transport(t)) => {
                Ok(ToolOutput::err(format!("Transport error: {t}")))
            }
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "agent_message".into(),
            description: "Sends a message to another workspace by ingesting triples.".into(),
            parameters: vec![
                ToolParamSchema::required("workspace", "Target workspace name."),
                ToolParamSchema::required("message", "Message text to deliver."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network, Capability::WriteKg]),
                description: "Writes message triples into another workspace's KG.".into(),
                shadow_triggers: vec![
                    "message".into(),
                    "send".into(),
                    "delegate".into(),
                    "tell".into(),
                    "communicate".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}

// ===========================================================================
// AgentRetireTool
// ===========================================================================

/// Delete an agent workspace, removing all its data.
pub struct AgentRetireTool;

impl Tool for AgentRetireTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "agent_retire".into(),
            description: "Delete an agent workspace, permanently removing it and all its data."
                .into(),
            parameters: vec![ToolParam {
                name: "workspace".into(),
                description: "Workspace name to delete.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let workspace = input.require("workspace", "agent_retire")?;

        let base_url = match discover_base_url() {
            Ok(u) => u,
            Err(out) => return Ok(out),
        };

        let url = format!("{base_url}/workspaces/{workspace}");
        let http = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(10))
            .build();

        match http.delete(&url).call() {
            Ok(resp) if resp.status() == 200 => Ok(ToolOutput::ok(format!(
                "Agent workspace \"{workspace}\" retired."
            ))),
            Ok(resp) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(ToolOutput::err(format!(
                    "Failed to retire workspace \"{workspace}\": {body}"
                )))
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                Ok(ToolOutput::err(format!(
                    "HTTP {code} retiring workspace \"{workspace}\": {body}"
                )))
            }
            Err(ureq::Error::Transport(t)) => {
                Ok(ToolOutput::err(format!("Transport error: {t}")))
            }
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "agent_retire".into(),
            description: "Deletes an agent workspace — destructive, permanent.".into(),
            parameters: vec![ToolParamSchema::required(
                "workspace",
                "Workspace name to delete.",
            )],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::Network]),
                description: "Permanently deletes a workspace and all its data.".into(),
                shadow_triggers: vec![
                    "delete".into(),
                    "remove".into(),
                    "destroy".into(),
                    "retire".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}
