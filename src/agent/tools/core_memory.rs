//! Self-edit memory tools for the VirtualContextManager (Phase 44i).
//!
//! These let the model manage its own context across MemGPT's tiers through the
//! ADR-062 tool loop:
//! - [`CoreMemoryAppendTool`] / [`CoreMemoryReplaceTool`] edit the always-in-
//!   context **core** tier ([`crate::agent::context::core_memory`]).
//! - [`ArchivalSearchTool`] pages relevant facts in from the long-term **archival**
//!   tier (the KG), so the model can retrieve without the operator re-stating.
//!
//! Each editing tool is constructed with the akh's `agent_id` baked in (the tool
//! API only passes `&Engine`), so a registered tool always edits *its own* akh's
//! core memory. They are registered only when the akh runs the `virtual` context
//! strategy. All are `Safe`: core memory is the akh's own scratchpad and archival
//! search is read-only.

use std::collections::HashSet;

use crate::agent::context::core_memory;
use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

/// Append a line to the akh's core memory.
pub struct CoreMemoryAppendTool {
    agent_id: String,
}

impl CoreMemoryAppendTool {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self { agent_id: agent_id.into() }
    }
}

impl Tool for CoreMemoryAppendTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "core_memory_append".into(),
            description: "Append a durable fact to your core memory (always kept in context). \
                          Use for the user's name, standing preferences, the current task."
                .into(),
            parameters: vec![ToolParam {
                name: "text".into(),
                description: "The line to remember.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let text = input.require("text", "core_memory_append")?;
        let count = core_memory::append(engine, &self.agent_id, text).map_err(|e| {
            crate::agent::error::AgentError::ToolExecution {
                tool_name: "core_memory_append".into(),
                message: e.to_string(),
            }
        })?;
        Ok(ToolOutput::ok(format!("Core memory updated ({count} line(s)).")))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "core_memory_append".into(),
            description: "Append a durable fact to your core memory (always kept in context)."
                .into(),
            parameters: vec![ToolParamSchema::required("text", "The line to remember.")],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Writes to the akh's own core-memory scratchpad.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Replace an existing core-memory line.
pub struct CoreMemoryReplaceTool {
    agent_id: String,
}

impl CoreMemoryReplaceTool {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self { agent_id: agent_id.into() }
    }
}

impl Tool for CoreMemoryReplaceTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "core_memory_replace".into(),
            description: "Replace a line in your core memory (e.g. when a fact changes). \
                          `old` must match an existing line exactly."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "old".into(),
                    description: "The existing line to replace (exact text).".into(),
                    required: true,
                },
                ToolParam {
                    name: "new".into(),
                    description: "The replacement text (empty to delete the line).".into(),
                    required: true,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let old = input.require("old", "core_memory_replace")?;
        let new = input.get("new").unwrap_or("");
        let replaced = core_memory::replace(engine, &self.agent_id, old, new).map_err(|e| {
            crate::agent::error::AgentError::ToolExecution {
                tool_name: "core_memory_replace".into(),
                message: e.to_string(),
            }
        })?;
        if replaced {
            Ok(ToolOutput::ok("Core memory line replaced."))
        } else {
            Ok(ToolOutput::err(format!("No core-memory line matched: {old:?}")))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "core_memory_replace".into(),
            description: "Replace a line in your core memory (exact match on `old`).".into(),
            parameters: vec![
                ToolParamSchema::required("old", "The existing line to replace (exact text)."),
                ToolParamSchema::required("new", "The replacement text (empty to delete)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Edits the akh's own core-memory scratchpad.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Search the long-term archival tier (the KG) for facts relevant to a query.
pub struct ArchivalSearchTool;

impl Tool for ArchivalSearchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "archival_search".into(),
            description: "Search your long-term archival memory (knowledge graph) for facts \
                          relevant to a query, paging them into context on demand."
                .into(),
            parameters: vec![ToolParam {
                name: "query".into(),
                description: "What to look up.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query = input.require("query", "archival_search")?;
        let mut cache = crate::nlu::kg_context::KgContextCache::default();
        match crate::nlu::kg_context::build_kg_context(engine, query, &mut cache) {
            Some(block) => Ok(ToolOutput::ok(block)),
            None => Ok(ToolOutput::ok("(no archival matches)")),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "archival_search".into(),
            description: "Search long-term archival memory (the knowledge graph) for relevant facts."
                .into(),
            parameters: vec![ToolParamSchema::required("query", "What to look up.")],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg, Capability::MemoryAccess]),
                description: "Read-only knowledge-graph search — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let e = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (e, tmp)
    }

    #[test]
    fn append_then_read_back_via_store() {
        let (e, _t) = engine();
        let tool = CoreMemoryAppendTool::new("akh");
        let out = tool
            .execute(&e, ToolInput::new().with_param("text", "user likes Rust"))
            .unwrap();
        assert!(out.success);
        assert_eq!(core_memory::read(&e, "akh"), vec!["user likes Rust"]);
    }

    #[test]
    fn replace_reports_miss() {
        let (e, _t) = engine();
        let tool = CoreMemoryReplaceTool::new("akh");
        let out = tool
            .execute(
                &e,
                ToolInput::new().with_param("old", "nope").with_param("new", "x"),
            )
            .unwrap();
        assert!(!out.success, "replacing a non-existent line fails cleanly");
    }
}
