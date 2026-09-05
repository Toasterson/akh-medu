//! Structured task tool (ADR 081 slice 6): four verbs (`create`/`update`/
//! `complete`/`list`) over the KG-backed [`crate::agent::tasks`] store.
//!
//! One `Tool` struct per verb — the `AgentListTool`/`AgentSpawnTool`/
//! `AgentMessageTool`/`AgentRetireTool` idiom (`agent_management.rs`), not a
//! single tool with an `op` discriminator. Named `structured_task_*` (not
//! `task_*`) to avoid colliding with the pre-existing ephemeral
//! `task_list` tool (`tools/task_list.rs`) — that one is an in-memory
//! TodoWrite-style scratch checklist with no KG persistence; this one is the
//! durable, LLM-driven task store ADR 081 calls for. They intentionally
//! coexist under different names.
//!
//! Reliability comes from the tool schema, not model judgment: every
//! argument set is deserialized into a typed `*Args` struct
//! (`serde_json::from_value`), so a malformed call fails closed with a
//! miette-coded [`crate::agent::tasks::TaskError`] diagnostic before any KG
//! mutation happens — never a partially-written or corrupt task.

use serde::Deserialize;
use std::collections::HashSet;

use crate::agent::agent::AgentPredicates;
use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tasks::{
    self, NewTask, Task, TaskError, TaskListFilter, TaskPredicates, TaskPriority, TaskResult,
    TaskUpdate, UpdateStatus,
};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Deserialize a tool's typed argument struct from the raw JSON param map.
/// A shape mismatch (missing required field, wrong type, …) becomes a
/// [`TaskError::BadInput`] diagnostic — this is the "grammar-constrained,
/// schema-validated" reliability mechanism the ADR calls for, not a free-form
/// string the tool has to guess at.
fn parse_args<T: serde::de::DeserializeOwned>(
    input: &ToolInput,
    tool_name: &str,
) -> AgentResult<T> {
    let value = serde_json::Value::Object(input.params.clone());
    serde_json::from_value(value).map_err(|e| {
        AgentError::Task(TaskError::BadInput {
            message: format!("{tool_name}: {e}"),
        })
    })
}

/// Parse a task id argument (the decimal `SymbolId`, as returned by `create`/`list`).
fn parse_task_id(s: &str) -> TaskResult<SymbolId> {
    s.trim()
        .parse::<u64>()
        .ok()
        .and_then(SymbolId::new)
        .ok_or_else(|| TaskError::BadInput {
            message: format!(
                "invalid task id: \"{s}\" (expected the numeric id from `create`/`list`)"
            ),
        })
}

/// Render a list of tasks as Obsidian-Tasks emoji lines plus a summary count
/// (mirrors `tools/task_list.rs::render`'s shape for a consistent feel).
fn render_tasks(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "No tasks match.".to_string();
    }
    let mut out = String::new();
    for task in tasks {
        out.push_str(&format!("{} (id={})\n", task, task.symbol_id.get()));
    }
    out.push_str(&format!("{} task(s).", tasks.len()));
    out
}

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CreateArgs {
    title: String,
    #[serde(default)]
    due: Option<String>,
    #[serde(default)]
    scheduled: Option<String>,
    #[serde(default)]
    recurrence: Option<String>,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    source_note_id: Option<String>,
}

fn build_new_task(args: &CreateArgs) -> TaskResult<NewTask> {
    let due = args
        .due
        .as_deref()
        .map(|s| tasks::parse_task_date("due", s))
        .transpose()?;
    let scheduled = args
        .scheduled
        .as_deref()
        .map(|s| tasks::parse_task_date("scheduled", s))
        .transpose()?;
    let recurrence = args
        .recurrence
        .as_deref()
        .map(super::super::pim::Recurrence::parse)
        .transpose()?;
    let priority = args
        .priority
        .as_deref()
        .map(TaskPriority::parse)
        .transpose()?
        .unwrap_or_default();
    let source_note = match &args.source_note_id {
        Some(s) => Some(parse_task_id(s)?),
        None => None,
    };
    Ok(NewTask {
        title: args.title.clone(),
        due,
        scheduled,
        recurrence,
        priority,
        source_note,
    })
}

/// Create a new structured task.
pub struct StructuredTaskCreateTool {
    task_predicates: TaskPredicates,
    agent_predicates: AgentPredicates,
}

impl StructuredTaskCreateTool {
    pub fn new(task_predicates: TaskPredicates, agent_predicates: AgentPredicates) -> Self {
        Self {
            task_predicates,
            agent_predicates,
        }
    }
}

impl Tool for StructuredTaskCreateTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "structured_task_create".into(),
            description: "Create a structured task (the KG-backed store, not a markdown \
                          checkbox). Optionally carries a due date, a scheduled date, a \
                          recurrence rule, and a priority."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "title".into(),
                    description: "Task title.".into(),
                    required: true,
                },
                ToolParam {
                    name: "due".into(),
                    description: "Due date, \"YYYY-MM-DD\".".into(),
                    required: false,
                },
                ToolParam {
                    name: "scheduled".into(),
                    description: "Scheduled (start-work) date, \"YYYY-MM-DD\".".into(),
                    required: false,
                },
                ToolParam {
                    name: "recurrence".into(),
                    description: "Recurrence rule: \"daily\", \"weekly:mon,wed,fri\", \
                                  \"monthly:15\", \"yearly:3-21\", or \"every:3d\"."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "priority".into(),
                    description: "\"low\", \"medium\" (default), or \"high\".".into(),
                    required: false,
                },
                ToolParam {
                    name: "source_note_id".into(),
                    description: "Numeric symbol id of a note this task was captured from.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let args: CreateArgs = parse_args(&input, "structured_task_create")?;
        let new = build_new_task(&args)?;
        let task = tasks::create_task(engine, &self.task_predicates, &self.agent_predicates, new)?;
        Ok(ToolOutput::ok_with_symbols(
            format!("Created task {}: {task}", task.symbol_id.get()),
            vec![task.symbol_id],
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "structured_task_create".into(),
            description: "Creates a structured task in the knowledge graph.".into(),
            parameters: vec![
                ToolParamSchema::required("title", "Task title."),
                ToolParamSchema::optional("due", "Due date, \"YYYY-MM-DD\"."),
                ToolParamSchema::optional("scheduled", "Scheduled date, \"YYYY-MM-DD\"."),
                ToolParamSchema::optional(
                    "recurrence",
                    "Recurrence rule: daily, weekly:mon,wed,fri, monthly:15, yearly:3-21, every:3d.",
                ),
                ToolParamSchema::optional("priority", "low, medium (default), or high."),
                ToolParamSchema::optional("source_note_id", "Numeric symbol id of a source note."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg, Capability::MemoryAccess]),
                description: "Writes a new task entity to the knowledge graph.".into(),
                shadow_triggers: vec!["remind me".into(), "add a task".into(), "todo".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct UpdateArgs {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    due: Option<String>,
    #[serde(default)]
    scheduled: Option<String>,
    #[serde(default)]
    recurrence: Option<String>,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

fn build_update(args: &UpdateArgs) -> TaskResult<TaskUpdate> {
    let due = args
        .due
        .as_deref()
        .map(|s| tasks::parse_task_date("due", s))
        .transpose()?;
    let scheduled = args
        .scheduled
        .as_deref()
        .map(|s| tasks::parse_task_date("scheduled", s))
        .transpose()?;
    let recurrence = args
        .recurrence
        .as_deref()
        .map(super::super::pim::Recurrence::parse)
        .transpose()?;
    let priority = args
        .priority
        .as_deref()
        .map(TaskPriority::parse)
        .transpose()?;
    let status = args
        .status
        .as_deref()
        .map(UpdateStatus::parse)
        .transpose()?;
    Ok(TaskUpdate {
        title: args.title.clone(),
        due,
        scheduled,
        recurrence,
        priority,
        status,
    })
}

/// Update fields on an existing task (id unchanged). Cannot mark a task
/// done — use `structured_task_complete` for that (it also handles
/// recurrence and the completion episode).
pub struct StructuredTaskUpdateTool {
    task_predicates: TaskPredicates,
}

impl StructuredTaskUpdateTool {
    pub fn new(task_predicates: TaskPredicates) -> Self {
        Self { task_predicates }
    }
}

impl Tool for StructuredTaskUpdateTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "structured_task_update".into(),
            description: "Update fields on an existing task (title, due, scheduled, \
                          recurrence, priority, or status open/cancelled). Use \
                          `structured_task_complete` to finish a task."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "id".into(),
                    description: "Numeric task id (from `create`/`list`).".into(),
                    required: true,
                },
                ToolParam {
                    name: "title".into(),
                    description: "New title.".into(),
                    required: false,
                },
                ToolParam {
                    name: "due".into(),
                    description: "New due date, \"YYYY-MM-DD\".".into(),
                    required: false,
                },
                ToolParam {
                    name: "scheduled".into(),
                    description: "New scheduled date, \"YYYY-MM-DD\".".into(),
                    required: false,
                },
                ToolParam {
                    name: "recurrence".into(),
                    description: "New recurrence rule.".into(),
                    required: false,
                },
                ToolParam {
                    name: "priority".into(),
                    description: "New priority: low, medium, or high.".into(),
                    required: false,
                },
                ToolParam {
                    name: "status".into(),
                    description: "\"open\" or \"cancelled\" (not \"done\" — use `complete`)."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let args: UpdateArgs = parse_args(&input, "structured_task_update")?;
        let id = parse_task_id(&args.id)?;
        let update = build_update(&args)?;
        let task = tasks::update_task(engine, &self.task_predicates, id, update)?;
        Ok(ToolOutput::ok_with_symbols(
            format!("Updated task {}: {task}", task.symbol_id.get()),
            vec![task.symbol_id],
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "structured_task_update".into(),
            description: "Updates fields on an existing task in the knowledge graph.".into(),
            parameters: vec![
                ToolParamSchema::required("id", "Numeric task id."),
                ToolParamSchema::optional("title", "New title."),
                ToolParamSchema::optional("due", "New due date, \"YYYY-MM-DD\"."),
                ToolParamSchema::optional("scheduled", "New scheduled date, \"YYYY-MM-DD\"."),
                ToolParamSchema::optional("recurrence", "New recurrence rule."),
                ToolParamSchema::optional("priority", "low, medium, or high."),
                ToolParamSchema::optional("status", "\"open\" or \"cancelled\"."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Modifies an existing task entity in the knowledge graph.".into(),
                shadow_triggers: vec!["update task".into(), "cancel task".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// complete
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CompleteArgs {
    id: String,
}

/// Mark a task done: mints a past completion episode and, if the task
/// recurs, spawns the next occurrence.
pub struct StructuredTaskCompleteTool {
    task_predicates: TaskPredicates,
    agent_predicates: AgentPredicates,
}

impl StructuredTaskCompleteTool {
    pub fn new(task_predicates: TaskPredicates, agent_predicates: AgentPredicates) -> Self {
        Self {
            task_predicates,
            agent_predicates,
        }
    }
}

impl Tool for StructuredTaskCompleteTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "structured_task_complete".into(),
            description: "Mark a task done. Records a past episodic memory entry for the \
                          completion, and if the task recurs, creates the next occurrence."
                .into(),
            parameters: vec![ToolParam {
                name: "id".into(),
                description: "Numeric task id (from `create`/`list`).".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let args: CompleteArgs = parse_args(&input, "structured_task_complete")?;
        let id = parse_task_id(&args.id)?;
        let outcome =
            tasks::complete_task(engine, &self.task_predicates, &self.agent_predicates, id)?;

        let mut symbols = vec![outcome.task.symbol_id, outcome.episode_id];
        let mut msg = format!(
            "Completed task {}: {}",
            outcome.task.symbol_id.get(),
            outcome.task
        );
        if let Some(next) = &outcome.next_occurrence {
            symbols.push(next.symbol_id);
            msg.push_str(&format!(
                "\nSpawned next occurrence {}: {next}",
                next.symbol_id.get()
            ));
        }
        Ok(ToolOutput::ok_with_symbols(msg, symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "structured_task_complete".into(),
            description: "Marks a task done, records a completion episode, and spawns the \
                          next occurrence if recurring."
                .into(),
            parameters: vec![ToolParamSchema::required("id", "Numeric task id.")],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg, Capability::MemoryAccess]),
                description: "Writes a completion episode and updates task state in the \
                              knowledge graph."
                    .into(),
                shadow_triggers: vec!["done".into(), "finished".into(), "complete task".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ListArgs {
    #[serde(default)]
    filter: Option<String>,
}

/// List tasks (default: open only).
pub struct StructuredTaskListTool {
    task_predicates: TaskPredicates,
}

impl StructuredTaskListTool {
    pub fn new(task_predicates: TaskPredicates) -> Self {
        Self { task_predicates }
    }
}

impl Tool for StructuredTaskListTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "structured_task_list".into(),
            description: "List structured tasks. Default filter is \"open\"; pass \"all\", \
                          \"done\", or \"cancelled\" to see others."
                .into(),
            parameters: vec![ToolParam {
                name: "filter".into(),
                description: "\"open\" (default), \"done\", \"cancelled\", or \"all\".".into(),
                required: false,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let args: ListArgs = parse_args(&input, "structured_task_list")?;
        let filter = args
            .filter
            .as_deref()
            .map(TaskListFilter::parse)
            .transpose()
            .map_err(AgentError::from)?
            .unwrap_or_default();
        let tasks = tasks::list_tasks(engine, &self.task_predicates, filter)?;
        let symbols = tasks.iter().map(|t| t.symbol_id).collect();
        Ok(ToolOutput::ok_with_symbols(render_tasks(&tasks), symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "structured_task_list".into(),
            description: "Lists structured tasks from the knowledge graph.".into(),
            parameters: vec![ToolParamSchema::optional(
                "filter",
                "\"open\" (default), \"done\", \"cancelled\", or \"all\".",
            )],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only query over task entities in the knowledge graph.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;
    use serde_json::json;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn setup() -> (Engine, TaskPredicates, AgentPredicates) {
        let engine = test_engine();
        let tp = TaskPredicates::init(&engine).unwrap();
        let ap = AgentPredicates::init(&engine).unwrap();
        (engine, tp, ap)
    }

    #[test]
    fn create_update_complete_list_end_to_end() {
        let (engine, tp, ap) = setup();

        let create = StructuredTaskCreateTool::new(tp.clone(), ap.clone());
        let out = create
            .execute(
                &engine,
                ToolInput::new()
                    .with_value("title", json!("Buy groceries"))
                    .with_value("due", json!("2026-07-15"))
                    .with_value("recurrence", json!("weekly:mon"))
                    .with_value("priority", json!("high")),
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        let task_id = out.symbols_involved[0];

        let list = StructuredTaskListTool::new(tp.clone());
        let out = list.execute(&engine, ToolInput::new()).unwrap();
        assert!(out.result.contains("Buy groceries"));
        assert!(out.result.contains("1 task(s)"));

        let update = StructuredTaskUpdateTool::new(tp.clone());
        let out = update
            .execute(
                &engine,
                ToolInput::new()
                    .with_value("id", json!(task_id.get().to_string()))
                    .with_value("priority", json!("low")),
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        assert!(out.result.contains("Updated task"));

        let complete = StructuredTaskCompleteTool::new(tp.clone(), ap.clone());
        let out = complete
            .execute(
                &engine,
                ToolInput::new().with_value("id", json!(task_id.get().to_string())),
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        assert!(out.result.contains("Completed task"));
        assert!(out.result.contains("Spawned next occurrence"));

        let open = list.execute(&engine, ToolInput::new()).unwrap();
        assert!(!open.result.contains("Buy groceries") || open.result.contains("🔁"));
        let done = list
            .execute(
                &engine,
                ToolInput::new().with_value("filter", json!("done")),
            )
            .unwrap();
        assert_eq!(done.symbols_involved.len(), 1);
    }

    #[test]
    fn create_missing_title_is_a_bad_input_diagnostic() {
        let (engine, tp, ap) = setup();
        let create = StructuredTaskCreateTool::new(tp, ap);
        let err = create.execute(&engine, ToolInput::new()).unwrap_err();
        match err {
            AgentError::Task(TaskError::BadInput { .. }) => {}
            other => panic!("expected AgentError::Task(BadInput), got {other:?}"),
        }
    }

    #[test]
    fn create_bad_date_is_a_diagnostic_not_a_corrupt_task() {
        let (engine, tp, ap) = setup();
        let create = StructuredTaskCreateTool::new(tp.clone(), ap);
        let err = create
            .execute(
                &engine,
                ToolInput::new()
                    .with_value("title", json!("X"))
                    .with_value("due", json!("not-a-date")),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            AgentError::Task(TaskError::InvalidDate { .. })
        ));

        // No task was created — the store has nothing.
        let tasks = tasks::list_tasks(&engine, &tp, TaskListFilter::All).unwrap();
        assert!(
            tasks.is_empty(),
            "invalid input must not leave a corrupt task behind"
        );
    }

    #[test]
    fn update_unknown_id_is_a_not_found_diagnostic() {
        let (engine, tp, _ap) = setup();
        let update = StructuredTaskUpdateTool::new(tp);
        let err = update
            .execute(
                &engine,
                ToolInput::new()
                    .with_value("id", json!("999999"))
                    .with_value("title", json!("whatever")),
            )
            .unwrap_err();
        assert!(matches!(err, AgentError::Task(TaskError::NotFound { .. })));
    }
}
