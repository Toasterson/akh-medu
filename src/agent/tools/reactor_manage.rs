//! Management tool for durable cron, interval, once, and event schedules.

use std::collections::HashSet;

use serde::Deserialize;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::reactor::{
    MisfirePolicy, ReactorAction, ReactorEvent, ReactorSchedule, ReactorStore, RetryPolicy,
    ScheduleRule,
};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

pub struct ReactorManageTool;

const ENABLE_ENV: &str = "AKH_REACTOR_ENABLE";

#[derive(Debug, Deserialize)]
struct AddRequest {
    #[serde(default)]
    id: Option<String>,
    name: String,
    rule: ScheduleRule,
    action: ReactorAction,
    #[serde(default)]
    misfire_policy: Option<MisfirePolicy>,
    #[serde(default)]
    misfire_grace_seconds: Option<u64>,
    #[serde(default)]
    lease_seconds: Option<u64>,
    #[serde(default)]
    retry: Option<RetryPolicy>,
    /// Event schedules: keep only the newest pending event.
    #[serde(default)]
    coalesce: Option<bool>,
}

/// `update`: replace the action (and optionally coalescing) of a stored
/// schedule, for example to add the `exit_condition` a paused legacy
/// schedule lacks (ADR 105).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateRequest {
    action: ReactorAction,
    #[serde(default)]
    coalesce: Option<bool>,
}

fn tool_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "reactor_manage".into(),
        message: message.into(),
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl Tool for ReactorManageTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "reactor_manage".into(),
            description: "Manage durable cron/interval/once/event schedules whose actions can include real LLM turns.".into(),
            parameters: vec![
                ToolParam {
                    name: "operation".into(),
                    description: "list, add, update, remove, enable, disable, or wake".into(),
                    required: true,
                },
                ToolParam {
                    name: "id".into(),
                    description: "Schedule id for update/remove/enable/disable/wake".into(),
                    required: false,
                },
                ToolParam {
                    name: "schedule".into(),
                    description: "Structured schedule object for add; {action, coalesce} for update".into(),
                    required: false,
                },
                ToolParam {
                    name: "event".into(),
                    description: "Structured event object for wake".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let operation = input.require("operation", "reactor_manage")?;
        let store = ReactorStore::new(engine);
        match operation {
            "list" => serde_json::to_string_pretty(&store.list_checked()?)
                .map(ToolOutput::ok)
                .map_err(|error| tool_error(format!("serialize schedules: {error}"))),
            "add" => {
                let value = input
                    .get_value("schedule")
                    .ok_or_else(|| tool_error("schedule object is required for add"))?;
                let request: AddRequest = serde_json::from_value(value.clone())
                    .map_err(|error| tool_error(format!("invalid schedule object: {error}")))?;
                let now = now_secs();
                let id = request.id.unwrap_or_else(|| {
                    let digest = blake3::hash(format!("{}\0{now}", request.name).as_bytes());
                    digest.to_hex()[..24].to_string()
                });
                if store.get(&id).is_some() {
                    return Err(tool_error(format!("schedule {id:?} already exists")));
                }
                let mut schedule =
                    ReactorSchedule::new(&id, request.name, request.rule, request.action, now)?;
                if let Some(policy) = request.misfire_policy {
                    schedule.misfire_policy = policy;
                }
                if let Some(grace) = request.misfire_grace_seconds {
                    schedule.misfire_grace_seconds = grace;
                }
                if let Some(lease) = request.lease_seconds {
                    schedule.lease_seconds = lease.max(1);
                }
                if let Some(retry) = request.retry {
                    schedule.retry = retry;
                }
                if let Some(coalesce) = request.coalesce {
                    schedule.coalesce = coalesce;
                }
                schedule.validate_for_admission()?;
                // Model-authored schedules are always staged. Enabling a
                // persistent autonomous action is a separate operator gate.
                schedule.enabled = false;
                store.put(&schedule)?;
                Ok(ToolOutput::ok(format!(
                    "durable schedule {id} added disabled; set {ENABLE_ENV}=1 and enable it explicitly"
                )))
            }
            "update" => {
                let id = input.require("id", "reactor_manage")?;
                let value = input.get_value("schedule").ok_or_else(|| {
                    tool_error("schedule {action, coalesce} is required for update")
                })?;
                let request: UpdateRequest = serde_json::from_value(value.clone())
                    .map_err(|error| tool_error(format!("invalid update object: {error}")))?;
                store.update_action(id, request.action, request.coalesce)?;
                Ok(ToolOutput::ok(format!(
                    "schedule {id} updated and left disabled; set {ENABLE_ENV}=1 and enable it explicitly"
                )))
            }
            "remove" => {
                let id = input.require("id", "reactor_manage")?;
                store.remove(id)?;
                Ok(ToolOutput::ok(format!("schedule {id} removed")))
            }
            "enable" | "disable" => {
                let id = input.require("id", "reactor_manage")?;
                let enabled = operation == "enable";
                if enabled && std::env::var(ENABLE_ENV).as_deref() != Ok("1") {
                    return Err(tool_error(format!(
                        "enabling autonomous schedules requires operator gate {ENABLE_ENV}=1"
                    )));
                }
                store.set_enabled(id, enabled)?;
                Ok(ToolOutput::ok(format!(
                    "schedule {id} {}",
                    if enabled { "enabled" } else { "disabled" }
                )))
            }
            "wake" => {
                let id = input.require("id", "reactor_manage")?;
                let value = input
                    .get_value("event")
                    .ok_or_else(|| tool_error("event object is required for wake"))?;
                let event: ReactorEvent = serde_json::from_value(value.clone())
                    .map_err(|error| tool_error(format!("invalid event object: {error}")))?;
                let queued = store.wake(id, event)?;
                Ok(ToolOutput::ok(if queued {
                    format!("event queued for schedule {id}")
                } else {
                    format!("duplicate event ignored for schedule {id}")
                }))
            }
            other => Err(tool_error(format!("unknown operation {other:?}"))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        let schedule_schema = serde_json::json!({
            "type": "object",
            "required": ["name", "rule", "action"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "rule": {
                    "type": "object",
                    "description": "Tagged rule: once {at}, interval {every_seconds,anchor}, cron {expression}, or event {source}. Events admitted through POST /v1/akhs/{akh}/events wake event schedules whose source is \"<event source>/<event type>\" (for example \"djehuty-inbox/delegation\")."
                },
                "action": {
                    "type": "object",
                    "description": "Tagged action: llm_turn {prompt, agent_id?, allow_tools?, require_mutation?, exit_condition, report_thread?}, djehuty_reply, trigger, or training_monitor. llm_turn is tool-free by default; with allow_tools it REQUIRES exit_condition (ADR 105: what done/failed look like). report_thread is an Agent Board thread UUID that receives a coordination/status@v1 message after every run. A woken event's payload is appended to the prompt as untrusted data."
                },
                "coalesce": {"type": "boolean", "description": "Event schedules only: keep just the newest pending event (replaceable state notifications, never commands)."},
                "misfire_policy": {"type": "string", "enum": ["skip", "run_once"]},
                "misfire_grace_seconds": {"type": "integer", "minimum": 0},
                "lease_seconds": {"type": "integer", "minimum": 1},
                "retry": {"type": "object"}
            }
        });
        let event_schema = serde_json::json!({
            "type": "object",
            "required": ["source", "id", "observed_at"],
            "properties": {
                "source": {"type": "string"},
                "id": {"type": "string"},
                "observed_at": {"type": "integer"},
                "payload": {"type": "string"}
            }
        });
        ToolManifest {
            name: "reactor_manage".into(),
            description: "CRUD and event wake operations for the durable daemon reactor.".into(),
            parameters: vec![
                ToolParamSchema::required(
                    "operation",
                    "list, add, update, remove, enable, disable, or wake",
                ),
                ToolParamSchema::optional("id", "Schedule id"),
                ToolParamSchema::optional_schema(
                    "schedule",
                    "Schedule object for add, or {action, coalesce} for update",
                    schedule_schema,
                ),
                ToolParamSchema::optional_schema("event", "Event object for wake", event_schema),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Critical,
                capabilities: HashSet::from([
                    Capability::WriteKg,
                    Capability::MemoryAccess,
                    Capability::Reason,
                    Capability::Network,
                    Capability::WriteFilesystem,
                ]),
                description: "Schedules future autonomous actions, including LLM/tool turns and a bounded checkpoint-and-stop control request.".into(),
                shadow_triggers: vec!["autonomous".into(), "scheduled".into(), "stop".into()],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    #[test]
    fn tool_can_create_real_llm_interval_action() {
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tempfile::tempdir().unwrap().keep()),
            ..EngineConfig::default()
        })
        .unwrap();
        let input = ToolInput::new().with_param("operation", "add").with_value(
            "schedule",
            serde_json::json!({
                "id": "daily-review",
                "name": "daily review",
                "rule": {"kind": "interval", "every_seconds": 3600, "anchor": 1},
                "action": {"kind": "llm_turn", "prompt": "Review the latest durable events"}
            }),
        );
        let output = ReactorManageTool.execute(&engine, input).unwrap();
        assert!(output.success);
        let schedule = ReactorStore::new(&engine).get("daily-review").unwrap();
        assert!(matches!(schedule.action, ReactorAction::LlmTurn { .. }));
        assert!(!schedule.enabled);
    }

    #[test]
    fn tool_creates_governed_event_schedules_and_fixes_legacy_ones() {
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tempfile::tempdir().unwrap().keep()),
            ..EngineConfig::default()
        })
        .unwrap();
        let add = |schedule: serde_json::Value| {
            ReactorManageTool.execute(
                &engine,
                ToolInput::new()
                    .with_param("operation", "add")
                    .with_value("schedule", schedule),
            )
        };
        // Tool-capable without an exit condition: refused with the diagnostic.
        let refused = add(serde_json::json!({
            "id": "inbox", "name": "inbox",
            "rule": {"kind": "event", "source": "djehuty-inbox/delegation"},
            "action": {"kind": "llm_turn", "prompt": "Handle it", "allow_tools": true}
        }))
        .unwrap_err();
        assert!(refused.to_string().contains("exit condition"), "{refused}");

        add(serde_json::json!({
            "id": "inbox", "name": "inbox", "coalesce": true,
            "rule": {"kind": "event", "source": "djehuty-inbox/delegation"},
            "action": {"kind": "llm_turn", "prompt": "Handle it", "allow_tools": true,
                "exit_condition": "delegation acknowledged on the board",
                "report_thread": "01a0ab49-9cc4-7012-8785-1eada26fbd46"}
        }))
        .unwrap();
        let stored = ReactorStore::new(&engine).get("inbox").unwrap();
        assert!(stored.coalesce && !stored.enabled);
        let ReactorAction::LlmTurn {
            exit_condition,
            report_thread,
            ..
        } = &stored.action
        else {
            panic!("llm_turn expected");
        };
        assert_eq!(exit_condition, "delegation acknowledged on the board");
        assert!(report_thread.is_some());

        let updated = ReactorManageTool
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("operation", "update")
                    .with_param("id", "inbox")
                    .with_value(
                        "schedule",
                        serde_json::json!({"coalesce": false, "action": {"kind": "llm_turn",
                            "prompt": "Handle it", "allow_tools": true,
                            "exit_condition": "delegation closed"}}),
                    ),
            )
            .unwrap();
        assert!(updated.success);
        assert!(!ReactorStore::new(&engine).get("inbox").unwrap().coalesce);
    }
}
