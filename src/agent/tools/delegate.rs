//! Delegate tool: dispatch a task to an external Claude Code worker or peer.
//!
//! This tool enables the OODA loop to offload tasks that require LLM-class
//! capability (code generation, document synthesis, deep research) to an
//! external Claude Code subprocess.
//!
//! The tool is synchronous: it enqueues the task into the DelegationManager's
//! pending queue. The daemon's delegation tick drains the queue and performs
//! the async subprocess dispatch.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

#[cfg(feature = "delegation")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "delegation")]
use crate::agent::delegation::DelegationTask;

/// Delegate a task to an external worker (Claude Code subprocess or peer).
pub struct DelegateTool {
    /// Shared pending task queue; the daemon drains this on each delegation tick.
    #[cfg(feature = "delegation")]
    pending: Arc<Mutex<Vec<DelegationTask>>>,
}

impl DelegateTool {
    /// Create without a pending queue (no-op delegation).
    pub fn new_noop() -> Self {
        Self {
            #[cfg(feature = "delegation")]
            pending: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create with a shared pending queue from the DelegationManager.
    #[cfg(feature = "delegation")]
    pub fn new(pending: Arc<Mutex<Vec<DelegationTask>>>) -> Self {
        Self { pending }
    }
}

impl Tool for DelegateTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "delegate".into(),
            description:
                "Delegate a task to an external Claude Code process or peer agent. \
                 The task will be executed asynchronously and results flow back \
                 through the delegation manager."
                    .into(),
            parameters: vec![
                ToolParam {
                    name: "task".into(),
                    description: "Description of the task to delegate.".into(),
                    required: true,
                },
                ToolParam {
                    name: "target".into(),
                    description:
                        "Target hint: 'local' for Claude Code subprocess (default), \
                         'remote' for cloud trigger, 'peer' for another akh-medu instance."
                            .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let task_desc = input.require("task", "delegate")?;
        let target_hint = input.get("target").map(str::to_string);

        #[cfg(feature = "delegation")]
        {
            let task = DelegationTask {
                description: task_desc.to_string(),
                context_triples: Vec::new(),
                required_capabilities: Vec::new(),
                goal_id: 0,
                priority: 0.5,
                target_hint: target_hint.clone(),
            };
            if let Ok(mut q) = self.pending.lock() {
                q.push(task);
                return Ok(ToolOutput::ok(format!(
                    "Delegation queued: \"{task_desc}\" → target: {} \
                     (will dispatch on next daemon tick)",
                    target_hint.as_deref().unwrap_or("auto"),
                )));
            }
        }

        // Feature not enabled or lock poisoned — report queued anyway so OODA
        // continues rather than failing the whole cycle.
        Ok(ToolOutput::ok(format!(
            "Delegation noted: \"{task_desc}\" (delegation feature not active)"
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "delegate".into(),
            description:
                "Delegate a task to an external Claude Code process or peer agent."
                    .into(),
            parameters: vec![
                ToolParamSchema::required("task", "Description of the task to delegate"),
                ToolParamSchema::optional(
                    "target",
                    "Target hint: 'local', 'remote', or 'peer'",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([
                    Capability::ProcessExec,
                    Capability::Network,
                ]),
                description:
                    "Spawns an external Claude Code process that can read/write files \
                     and query the knowledge graph."
                        .into(),
                shadow_triggers: vec![
                    "delegate".into(),
                    "external".into(),
                    "subprocess".into(),
                    "llm".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}
