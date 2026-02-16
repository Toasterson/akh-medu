//! Trigger management tool: CRUD operations on autonomous triggers.
//!
//! Allows the agent (or user) to list, add, and remove condition→action triggers
//! that fire autonomously during daemon operation.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::agent::trigger::{Trigger, TriggerAction, TriggerCondition, TriggerStore};
use crate::engine::Engine;

/// Agent tool for managing triggers (list/add/remove).
pub struct TriggerManageTool;

impl Tool for TriggerManageTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "trigger_manage".into(),
            description: "Manage autonomous triggers: list, add, or remove \
                          condition→action rules that fire during background daemon operation."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "action".into(),
                    description: "Action: list, add, remove.".into(),
                    required: true,
                },
                ToolParam {
                    name: "name".into(),
                    description: "Trigger name (required for add).".into(),
                    required: false,
                },
                ToolParam {
                    name: "condition_type".into(),
                    description: "Condition type: interval, goal_stalled, memory_pressure, new_triples.".into(),
                    required: false,
                },
                ToolParam {
                    name: "condition_value".into(),
                    description: "Condition threshold value (numeric).".into(),
                    required: false,
                },
                ToolParam {
                    name: "action_type".into(),
                    description: "Action type: run_cycles, reflect, learn_equivalences, run_rules, analyze_gaps, add_goal, execute_tool.".into(),
                    required: false,
                },
                ToolParam {
                    name: "action_value".into(),
                    description: "Action value (e.g., cycle count, goal description).".into(),
                    required: false,
                },
                ToolParam {
                    name: "id".into(),
                    description: "Trigger ID (required for remove).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let action = input.require("action", "trigger_manage")?;
        let store = TriggerStore::new(engine);

        match action {
            "list" => {
                let triggers = store.list();
                if triggers.is_empty() {
                    return Ok(ToolOutput::ok("No triggers registered."));
                }
                let mut lines = vec![format!("{} trigger(s):", triggers.len())];
                for t in &triggers {
                    lines.push(format!(
                        "  [{}] \"{}\" — {:?} → {:?} (enabled={}, last_fired={})",
                        t.id, t.name, t.condition, t.action, t.enabled, t.last_fired,
                    ));
                }
                Ok(ToolOutput::ok(lines.join("\n")))
            }

            "add" => {
                let name = input
                    .get("name")
                    .ok_or_else(|| crate::agent::error::AgentError::ToolExecution {
                        tool_name: "trigger_manage".into(),
                        message: "\"name\" is required for add".into(),
                    })?
                    .to_string();

                let condition = parse_condition(&input)?;
                let action = parse_action(&input)?;

                let trigger = Trigger {
                    id: String::new(), // will be assigned by store
                    name: name.clone(),
                    condition,
                    action,
                    enabled: true,
                    last_fired: 0,
                };

                let saved = store.add(trigger)?;
                Ok(ToolOutput::ok(format!(
                    "Trigger \"{}\" added with id {}.",
                    name, saved.id
                )))
            }

            "remove" => {
                let id = input
                    .get("id")
                    .ok_or_else(|| crate::agent::error::AgentError::ToolExecution {
                        tool_name: "trigger_manage".into(),
                        message: "\"id\" is required for remove".into(),
                    })?;
                store.remove(id)?;
                Ok(ToolOutput::ok(format!("Trigger \"{id}\" removed.")))
            }

            other => Ok(ToolOutput::err(format!(
                "Unknown action: \"{other}\". Use list, add, or remove."
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "trigger_manage".into(),
            description: "CRUD for autonomous triggers — modifies durable trigger store.".into(),
            parameters: vec![
                ToolParamSchema::required("action", "Action: list, add, remove."),
                ToolParamSchema::optional("name", "Trigger name (required for add)."),
                ToolParamSchema::optional(
                    "condition_type",
                    "interval, goal_stalled, memory_pressure, new_triples.",
                ),
                ToolParamSchema::optional("condition_value", "Condition threshold (numeric)."),
                ToolParamSchema::optional(
                    "action_type",
                    "run_cycles, reflect, learn_equivalences, run_rules, analyze_gaps, add_goal, execute_tool.",
                ),
                ToolParamSchema::optional("action_value", "Action value (e.g., cycle count)."),
                ToolParamSchema::optional("id", "Trigger ID (required for remove)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg]),
                description:
                    "Creates/removes triggers that can autonomously execute tools and modify KG state."
                        .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_condition(input: &ToolInput) -> AgentResult<TriggerCondition> {
    let ctype = input.get("condition_type").unwrap_or("interval");
    let cval: u64 = input
        .get("condition_value")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    match ctype {
        "interval" => Ok(TriggerCondition::Interval { seconds: cval }),
        "goal_stalled" => Ok(TriggerCondition::GoalStalled {
            threshold: cval as u32,
        }),
        "memory_pressure" => Ok(TriggerCondition::MemoryPressure {
            threshold: cval as usize,
        }),
        "new_triples" => Ok(TriggerCondition::NewTriples {
            min_count: cval as usize,
        }),
        other => Err(crate::agent::error::AgentError::ToolExecution {
            tool_name: "trigger_manage".into(),
            message: format!("unknown condition_type: \"{other}\""),
        }),
    }
}

fn parse_action(input: &ToolInput) -> AgentResult<TriggerAction> {
    let atype = input.get("action_type").unwrap_or("reflect");
    let aval = input.get("action_value").unwrap_or("1");

    match atype {
        "run_cycles" => Ok(TriggerAction::RunCycles {
            count: aval.parse().unwrap_or(1),
        }),
        "reflect" => Ok(TriggerAction::Reflect),
        "learn_equivalences" => Ok(TriggerAction::LearnEquivalences),
        "run_rules" => Ok(TriggerAction::RunRules),
        "analyze_gaps" => Ok(TriggerAction::AnalyzeGaps),
        "add_goal" => Ok(TriggerAction::AddGoal {
            description: aval.to_string(),
            priority: 128,
            criteria: String::new(),
        }),
        "execute_tool" => Ok(TriggerAction::ExecuteTool {
            name: aval.to_string(),
            params: HashMap::new(),
        }),
        other => Err(crate::agent::error::AgentError::ToolExecution {
            tool_name: "trigger_manage".into(),
            message: format!("unknown action_type: \"{other}\""),
        }),
    }
}
