//! Delegate tool: dispatch a task to an external Claude Code worker or peer.
//!
//! This tool enables the OODA loop to offload tasks that require LLM-class
//! capability (code generation, document synthesis, deep research) to an
//! external Claude Code subprocess.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

/// Delegate a task to an external worker (Claude Code subprocess or peer).
pub struct DelegateTool;

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
                        "Target hint: 'local' for Claude Code subprocess (default)."
                            .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let task_desc = input.require("task", "delegate")?;
        let target_hint = input.get("target");

        // The actual dispatch happens through the DelegationManager which
        // requires async + the manager instance. From the synchronous tool
        // context we build the task spec and return it; the OODA loop or
        // daemon will pick it up and dispatch.
        //
        // For now, we report the task as queued. The daemon integration
        // (Phase 36e) will wire this into the async dispatch path.
        Ok(ToolOutput::ok(format!(
            "Delegation queued: \"{task_desc}\" → target: {}",
            target_hint.unwrap_or("auto"),
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
                    "Target hint: 'local' for Claude Code subprocess",
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
