//! Escalation tools: `escalate`, `dispatch_merge_repair`, `escalation_status`
//! (Phase 53g PR 7).
//!
//! Registered only for an akh whose `[delegation.native.escalation]` enables
//! them (Janus). Each wraps the shared
//! [`EscalationService`](crate::agent::delegation::escalation::EscalationService).
//! Validation errors come back as a failed tool output carrying the
//! diagnostic's code and help, so the model can correct the call.
//!
//! None of them is in the `/v1` legacy cognitive scope and none is
//! read-only, so a `/v1` principal or a delegation grant reaches them only
//! by naming them explicitly (and the default worker grant never does).

use std::collections::HashSet;
use std::sync::Arc;

use miette::Diagnostic;
use serde::Serialize;
use serde_json::json;

use crate::agent::delegation::escalation::{
    EscalationError, EscalationRequest, EscalationService, MergeRepairRequest,
};
use crate::agent::delegation::ledger::IncidentKey;
use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

/// Tool name of [`EscalateTool`].
pub const ESCALATE_TOOL: &str = "escalate";
/// Tool name of [`DispatchMergeRepairTool`].
pub const DISPATCH_MERGE_REPAIR_TOOL: &str = "dispatch_merge_repair";
/// Tool name of [`EscalationStatusTool`].
pub const ESCALATION_STATUS_TOOL: &str = "escalation_status";

/// A successful answer: the value as JSON.
fn answer<T: Serialize>(value: &T) -> ToolOutput {
    match serde_json::to_string(value) {
        Ok(text) => ToolOutput::ok(text),
        Err(error) => ToolOutput::err(format!("could not encode the answer: {error}")),
    }
}

/// A failed answer carrying the diagnostic's code, message and help.
fn refusal(error: &EscalationError) -> ToolOutput {
    let help = error.help().map(|help| help.to_string());
    ToolOutput::err(
        json!({"error": {
            "code": error.code_str(),
            "message": error.to_string(),
            "help": help,
        }})
        .to_string(),
    )
}

fn string_list_schema() -> serde_json::Value {
    json!({"type": "array", "minItems": 1, "maxItems": 12,
           "items": {"type": "string", "minLength": 1, "maxLength": 4000}})
}

fn sha_schema() -> serde_json::Value {
    json!({"type": "string", "pattern": "^[0-9a-fA-F]{40}$"})
}

fn uuid_schema() -> serde_json::Value {
    json!({"type": "string", "format": "uuid"})
}

fn pr_schema() -> serde_json::Value {
    json!({"type": "integer", "minimum": 1})
}

fn delegating_danger(description: &str) -> DangerInfo {
    DangerInfo {
        effect: ToolEffect::Mutating,
        level: DangerLevel::Dangerous,
        capabilities: HashSet::from([Capability::ProcessExec, Capability::Network]),
        description: description.into(),
        shadow_triggers: vec!["delegate".into(), "external".into(), "llm".into()],
    }
}

fn signature_of(manifest: &ToolManifest) -> ToolSignature {
    ToolSignature {
        name: manifest.name.clone(),
        description: manifest.description.clone(),
        parameters: manifest
            .parameters
            .iter()
            .map(|p| ToolParam {
                name: p.name.clone(),
                description: p.description.clone(),
                required: p.required,
            })
            .collect(),
    }
}

/// `escalate`: hand one bounded blocker to a contained native worker.
pub struct EscalateTool {
    service: Arc<EscalationService>,
}

impl EscalateTool {
    /// The tool over `service`.
    pub fn new(service: Arc<EscalationService>) -> Self {
        Self { service }
    }
}

impl Tool for EscalateTool {
    fn signature(&self) -> ToolSignature {
        signature_of(&self.manifest())
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let outcome = EscalationRequest::from_params(&input.params)
            .and_then(|request| self.service.escalate(&request));
        Ok(match outcome {
            Ok(receipt) => answer(&receipt),
            Err(error) => refusal(&error),
        })
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: ESCALATE_TOOL.into(),
            description: format!(
                "Escalate one bounded blocker to a contained native worker session. The same \
                 incident_key always returns the same incident (duplicate: true); at most the \
                 configured number of new incidents per UTC day. A pending or running receipt is \
                 not proof of resolution: check escalation_status on a later wake. Registered \
                 projects: {}.",
                self.service.projects().join(", ")
            ),
            parameters: vec![
                ToolParamSchema::required(
                    "incident_key",
                    "Stable ASCII key for this blocker, reused across wakes (not merge-repair-*)",
                ),
                ToolParamSchema::required("project", "A registered project"),
                ToolParamSchema::required("summary", "One line, at most 160 characters"),
                ToolParamSchema::required_schema(
                    "evidence",
                    "1-12 exact evidence entries (IDs, timestamps, observations; no secrets)",
                    string_list_schema(),
                ),
                ToolParamSchema::required_schema(
                    "attempted",
                    "1-12 bounded recovery attempts already made",
                    string_list_schema(),
                ),
                ToolParamSchema::required(
                    "requested_action",
                    "The bounded action the worker may take (at most 4000 characters)",
                ),
                ToolParamSchema::optional_schema(
                    "work_item_id",
                    "Anima work-item UUID",
                    uuid_schema(),
                ),
                ToolParamSchema::optional_schema("pr_number", "PR number", pr_schema()),
                ToolParamSchema::optional_schema(
                    "head_sha",
                    "Full 40-hex PR head SHA",
                    sha_schema(),
                ),
                ToolParamSchema::optional_schema(
                    "anima_session_id",
                    "Anima session UUID",
                    uuid_schema(),
                ),
            ],
            danger: delegating_danger(
                "Starts a native worker session (Codex) under a scoped delegation grant; the \
                 worker can edit its workspace and call the granted cognitive tools.",
            ),
            source: ToolSource::Native,
        }
    }
}

/// `dispatch_merge_repair`: hand one PR merge conflict to a contained worker.
pub struct DispatchMergeRepairTool {
    service: Arc<EscalationService>,
}

impl DispatchMergeRepairTool {
    /// The tool over `service`.
    pub fn new(service: Arc<EscalationService>) -> Self {
        Self { service }
    }
}

impl Tool for DispatchMergeRepairTool {
    fn signature(&self) -> ToolSignature {
        signature_of(&self.manifest())
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let outcome = MergeRepairRequest::from_params(&input.params)
            .and_then(|request| self.service.dispatch_merge_repair(&request));
        Ok(match outcome {
            Ok(receipt) => answer(&receipt),
            Err(error) => refusal(&error),
        })
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: DISPATCH_MERGE_REPAIR_TOOL.into(),
            description: format!(
                "Dispatch a merge-conflict repair worker for one PR at exact heads. The worker \
                 may only fast-forward the existing PR source branch and never merges. The same \
                 project, repository, PR and heads always return the same incident (its derived \
                 incident_key); shares the daily limit with escalate. Registered projects: {}.",
                self.service.projects().join(", ")
            ),
            parameters: vec![
                ToolParamSchema::required("project", "A registered project"),
                ToolParamSchema::required(
                    "repository",
                    "The PR's repository: owner/name or an https:// or ssh:// URL",
                ),
                ToolParamSchema::required_schema(
                    "work_item_id",
                    "Exact Anima work-item UUID",
                    uuid_schema(),
                ),
                ToolParamSchema::required_schema("pr_number", "PR number", pr_schema()),
                ToolParamSchema::required_schema(
                    "head_sha",
                    "Verified current PR source-branch head (full 40-hex SHA)",
                    sha_schema(),
                ),
                ToolParamSchema::required_schema(
                    "base_sha",
                    "Verified current base commit to merge (full 40-hex SHA)",
                    sha_schema(),
                ),
                ToolParamSchema::required_schema(
                    "evidence",
                    "1-12 conflict and ownership evidence entries",
                    string_list_schema(),
                ),
            ],
            danger: delegating_danger(
                "Starts a native merge-repair worker session (Codex) under a scoped delegation \
                 grant; its only publication is a fast-forward of the PR source branch.",
            ),
            source: ToolSource::Native,
        }
    }
}

/// `escalation_status`: read (and reconcile) one incident.
pub struct EscalationStatusTool {
    service: Arc<EscalationService>,
}

impl EscalationStatusTool {
    /// The tool over `service`.
    pub fn new(service: Arc<EscalationService>) -> Self {
        Self { service }
    }
}

impl Tool for EscalationStatusTool {
    fn signature(&self) -> ToolSignature {
        signature_of(&self.manifest())
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let outcome = (|| {
            let raw = input
                .get("incident_key")
                .ok_or_else(|| EscalationError::InvalidInput {
                    field: "incident_key".into(),
                    reason: "is required".into(),
                })?;
            if let Some(extra) = input.params.keys().find(|k| k.as_str() != "incident_key") {
                return Err(EscalationError::InvalidInput {
                    field: extra.chars().take(64).collect(),
                    reason: "is not a field of this tool (fields: incident_key)".into(),
                });
            }
            let key = IncidentKey::parse(raw)?;
            self.service.status(&key)
        })();
        Ok(match outcome {
            Ok(status) => answer(&status),
            Err(error) => refusal(&error),
        })
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: ESCALATION_STATUS_TOOL.into(),
            description: "Read one escalation or merge-repair incident: its state, the worker's \
                 turn and final output, and a failure's provider class and next owner. Retries \
                 a pending dispatch and settles an ended turn (revoking the worker's grant). A \
                 completed turn is not resolution; verify the result."
                .into(),
            parameters: vec![ToolParamSchema::required(
                "incident_key",
                "The incident_key an escalate or dispatch_merge_repair receipt returned",
            )],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::Network]),
                description: "Reads akhsessd; settles the incident ledger and closes ended \
                     worker sessions."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// The three tools over one service, for registration.
pub fn escalation_tools(service: &Arc<EscalationService>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(EscalateTool::new(Arc::clone(service))),
        Box::new(DispatchMergeRepairTool::new(Arc::clone(service))),
        Box::new(EscalationStatusTool::new(Arc::clone(service))),
    ]
}
