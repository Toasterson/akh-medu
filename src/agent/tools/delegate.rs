//! Delegate tool: dispatch a task to a native-runtime session or a peer.
//!
//! This tool enables the OODA loop to offload tasks that require LLM-class
//! capability (code generation, document synthesis, deep research) to a
//! native runtime session brokered by akhsessd, carrying a per-delegation
//! grant (Phase 53e).
//!
//! The tool is synchronous: it enqueues the task into the DelegationManager's
//! pending queue. The daemon's delegation tick drains the queue and performs
//! the async subprocess dispatch.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

#[cfg(feature = "delegation")]
use crate::agent::delegation::DelegationTask;
#[cfg(feature = "delegation")]
use std::sync::{Arc, Mutex};

/// Delegate a task to an external worker (native-runtime session or peer).
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
            description: "Delegate a task to a native-runtime session or a peer agent. \
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
                    description: "Target hint: 'native' for a native-runtime session \
                         carrying a scoped grant (default), 'remote' for cloud trigger, \
                         'peer' for another akh-medu instance."
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
            let target_workspace = input.get("workspace").map(str::to_string);
            let task = DelegationTask {
                description: task_desc.to_string(),
                context_triples: Vec::new(),
                required_capabilities: Vec::new(),
                goal_id: 0,
                priority: 0.5,
                target_hint: target_hint.clone(),
                target_workspace,
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
            description: "Delegate a task to a native-runtime session or peer agent.".into(),
            parameters: vec![
                ToolParamSchema::required("task", "Description of the task to delegate"),
                ToolParamSchema::optional("target", "Target hint: 'native', 'remote', or 'peer'"),
                ToolParamSchema::optional(
                    "workspace",
                    "Target workspace for peer delegation (the peer's default when omitted)",
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::ProcessExec, Capability::Network]),
                description: "Starts an external native-runtime session that can read/write \
                     files and call a granted subset of the akh's cognitive tools."
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
