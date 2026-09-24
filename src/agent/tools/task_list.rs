//! Task-list tool: TodoWrite-style structured task tracking.
//!
//! Maintains an in-memory checklist of tasks the agent is working on. Each
//! task has `content` (imperative form, e.g. "Run the tests"), a `status`
//! (`pending` / `in_progress` / `completed`), and an `active_form` (present
//! continuous, e.g. "Running the tests") shown while the task is in progress.
//!
//! This is the headline consumer of the structured [`ToolInput`] API: the
//! `tasks` parameter is a JSON *array of objects* read via
//! [`ToolInput::get_value`], not a flat string.
//!
//! Semantics mirror TodoWrite: supplying `tasks` REPLACES the entire stored
//! list. Omitting `tasks` returns the current list as a read-only view. The
//! list lives behind an `Arc<Mutex<Vec<TaskItem>>>` (the same shared-state
//! pattern used by `DelegateTool`) so the agent and any external observer
//! share one source of truth.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Lifecycle status of a single task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Not yet started.
    Pending,
    /// Currently being worked on.
    InProgress,
    /// Finished.
    Completed,
}

impl TaskStatus {
    /// Parse a status string into a [`TaskStatus`], accepting both snake_case
    /// (`in_progress`) and camelCase (`inProgress`) spellings, case-insensitively.
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "in_progress" | "inprogress" => Some(Self::InProgress),
            "completed" | "complete" | "done" => Some(Self::Completed),
            _ => None,
        }
    }
}

/// A single task in the agent's checklist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskItem {
    /// Imperative description of the task (e.g. "Run the tests").
    pub content: String,
    /// Current status.
    pub status: TaskStatus,
    /// Present-continuous form shown while in progress (e.g. "Running the tests").
    pub active_form: String,
}

/// TodoWrite-style task tracker backed by shared in-memory state.
pub struct TaskListTool {
    /// Shared task list; replaced wholesale on each write (TodoWrite semantics).
    tasks: Arc<Mutex<Vec<TaskItem>>>,
}

impl TaskListTool {
    /// Create a task-list tool over a shared task vector.
    pub fn new(tasks: Arc<Mutex<Vec<TaskItem>>>) -> Self {
        Self { tasks }
    }
}

/// Render a slice of tasks as a checklist plus a summary count line.
fn render(tasks: &[TaskItem]) -> String {
    if tasks.is_empty() {
        return "Task list is empty.".to_string();
    }

    let mut pending = 0usize;
    let mut in_progress = 0usize;
    let mut completed = 0usize;

    let mut out = String::new();
    for task in tasks {
        match task.status {
            TaskStatus::Pending => {
                pending += 1;
                out.push_str(&format!("[ ] {}\n", task.content));
            }
            TaskStatus::InProgress => {
                in_progress += 1;
                out.push_str(&format!("[~] {} ({})\n", task.content, task.active_form));
            }
            TaskStatus::Completed => {
                completed += 1;
                out.push_str(&format!("[x] {}\n", task.content));
            }
        }
    }

    out.push_str(&format!(
        "{} task(s): {} pending, {} in progress, {} completed.",
        tasks.len(),
        pending,
        in_progress,
        completed
    ));
    out
}

/// Parse one JSON array element into a [`TaskItem`].
///
/// Returns `Err(detail)` on any malformed shape; the caller maps this to an
/// `Ok(ToolOutput::err(...))`.
fn parse_task(value: &serde_json::Value) -> Result<TaskItem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| format!("task entry is not an object: {value}"))?;

    let content = obj
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "task entry missing string field \"content\"".to_string())?
        .to_string();

    let status_str = obj
        .get("status")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "task entry missing string field \"status\"".to_string())?;
    let status = TaskStatus::parse(status_str).ok_or_else(|| {
        format!("unknown status \"{status_str}\" (expected pending, in_progress, or completed)")
    })?;

    // Accept either snake_case `active_form` or camelCase `activeForm`.
    let active_form = obj
        .get("active_form")
        .or_else(|| obj.get("activeForm"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(TaskItem {
        content,
        status,
        active_form,
    })
}

impl Tool for TaskListTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "task_list".into(),
            description:
                "Track a TodoWrite-style checklist of tasks. Supply `tasks` (a JSON array of \
                 {content, status, active_form} objects) to replace the entire list; omit it \
                 to view the current list. Status is one of pending, in_progress, completed."
                    .into(),
            parameters: vec![ToolParam {
                name: "tasks".into(),
                description: "JSON array of task objects {content, status, active_form}. \
                     When present, replaces the stored list. When absent, returns the \
                     current list."
                    .into(),
                required: false,
            }],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        // Read-only view when `tasks` is absent.
        let Some(value) = input.get_value("tasks") else {
            let guard = match self.tasks.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            return Ok(ToolOutput::ok(render(&guard)));
        };

        // `tasks` MUST be a JSON array.
        let serde_json::Value::Array(items) = value else {
            return Ok(ToolOutput::err(
                "tasks must be an array of {content, status, active_form} objects: \
                 received a non-array value"
                    .to_string(),
            ));
        };

        // Parse every entry before mutating shared state (all-or-nothing).
        let mut parsed = Vec::with_capacity(items.len());
        for item in items {
            match parse_task(item) {
                Ok(task) => parsed.push(task),
                Err(detail) => {
                    return Ok(ToolOutput::err(format!(
                        "tasks must be an array of {{content, status, active_form}} objects: \
                         {detail}"
                    )));
                }
            }
        }

        // Replace the entire stored list (TodoWrite semantics).
        let mut guard = match self.tasks.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = parsed;
        Ok(ToolOutput::ok(render(&guard)))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "task_list".into(),
            description:
                "Maintain an in-memory TodoWrite-style task checklist. No side effects beyond \
                 the shared in-process task list."
                    .into(),
            parameters: vec![ToolParamSchema::optional(
                "tasks",
                "JSON array of {content, status, active_form} objects; replaces the list when \
                 present, reads it when absent.",
            )],
            danger: DangerInfo {
                // Mutating: a call with `tasks` replaces the list every
                // session of this process shares.
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::new(),
                description: "Replaces the shared in-process task list — no filesystem, network, \
                              or KG access."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn tool() -> TaskListTool {
        TaskListTool::new(Arc::new(Mutex::new(Vec::new())))
    }

    #[test]
    fn task_list_write_then_read() {
        let engine = test_engine();
        let tool = tool();

        // Write two tasks via the structured array input.
        let input = ToolInput::new().with_value(
            "tasks",
            serde_json::json!([
                {"content": "Read the file", "status": "completed", "active_form": "Reading the file"},
                {"content": "Run the tests", "status": "in_progress", "active_form": "Running the tests"}
            ]),
        );
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success);
        assert!(out.result.contains("[x] Read the file"));
        assert!(out.result.contains("[~] Run the tests (Running the tests)"));
        assert!(out.result.contains("2 task(s)"));
        assert!(out.result.contains("1 in progress"));
        assert!(out.result.contains("1 completed"));

        // Calling again with no `tasks` param returns the persisted list.
        let view = tool.execute(&engine, ToolInput::new()).unwrap();
        assert!(view.success);
        assert!(view.result.contains("[x] Read the file"));
        assert!(
            view.result
                .contains("[~] Run the tests (Running the tests)")
        );
        assert!(view.result.contains("2 task(s)"));
    }

    #[test]
    fn task_list_rejects_bad_status() {
        let engine = test_engine();
        let tool = tool();

        let input = ToolInput::new().with_value(
            "tasks",
            serde_json::json!([
                {"content": "Do a thing", "status": "halfway", "active_form": "Doing a thing"}
            ]),
        );
        let out = tool.execute(&engine, input).unwrap();
        assert!(!out.success);
        assert!(out.result.contains("tasks must be an array"));
        assert!(out.result.contains("halfway"));

        // A non-array `tasks` value is also rejected.
        let bad = ToolInput::new().with_value("tasks", serde_json::json!("not an array"));
        let out = tool.execute(&engine, bad).unwrap();
        assert!(!out.success);
        assert!(out.result.contains("tasks must be an array"));
    }

    #[test]
    fn task_list_empty_message() {
        let engine = test_engine();
        let tool = tool();

        // No tasks written yet — read-only view reports empty.
        let out = tool.execute(&engine, ToolInput::new()).unwrap();
        assert!(out.success);
        assert_eq!(out.result, "Task list is empty.");

        // Writing an empty array also yields the empty message.
        let input = ToolInput::new().with_value("tasks", serde_json::json!([]));
        let out = tool.execute(&engine, input).unwrap();
        assert!(out.success);
        assert_eq!(out.result, "Task list is empty.");
    }
}
