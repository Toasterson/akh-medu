//! Memory recall tool: recall episodic memories by tag or symbol.

use crate::agent::agent::AgentPredicates;
use crate::agent::error::{AgentError, AgentResult};
use crate::agent::memory;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::symbol::SymbolId;
use std::collections::HashSet;

/// Recall episodic memories from long-term storage.
pub struct MemoryRecallTool {
    predicates: AgentPredicates,
}

impl MemoryRecallTool {
    /// Create a new recall tool with the given agent predicates.
    pub fn new(predicates: AgentPredicates) -> Self {
        Self { predicates }
    }
}

impl Tool for MemoryRecallTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "memory_recall".into(),
            description: "Recall episodic memories by query symbols or tags.".into(),
            parameters: vec![
                ToolParam {
                    name: "query_symbols".into(),
                    description: "Comma-separated symbol names or IDs to query.".into(),
                    required: true,
                },
                ToolParam {
                    name: "top_k".into(),
                    description: "Maximum number of episodes to return (default: 5).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query_str = input.require("query_symbols", "memory_recall")?;
        let top_k: usize = input
            .get("top_k")
            .and_then(|k| k.parse().ok())
            .unwrap_or(5);

        let query_ids: Vec<SymbolId> = query_str
            .split(',')
            .filter_map(|s| engine.resolve_symbol(s.trim()).ok())
            .collect();

        if query_ids.is_empty() {
            return Ok(ToolOutput::ok("No valid query symbols resolved."));
        }

        let episodes =
            memory::recall_episodes(engine, &query_ids, &self.predicates, top_k)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "memory_recall".into(),
                    message: format!("{e}"),
                })?;

        if episodes.is_empty() {
            return Ok(ToolOutput::ok_with_symbols(
                "No episodic memories found for the query symbols.",
                query_ids,
            ));
        }

        let mut lines = Vec::new();
        let mut all_symbols = query_ids;
        for ep in &episodes {
            lines.push(format!(
                "- [{}] {} (learnings: {}, tags: {})",
                engine.resolve_label(ep.symbol_id),
                ep.summary,
                ep.learnings.len(),
                ep.tags.len(),
            ));
            all_symbols.push(ep.symbol_id);
            all_symbols.extend(&ep.learnings);
        }

        let result = format!("Recalled {} episode(s):\n{}", lines.len(), lines.join("\n"));
        Ok(ToolOutput::ok_with_symbols(result, all_symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "memory_recall".into(),
            description: "Recall episodic memories by query symbols or tags.".into(),
            parameters: vec![
                ToolParamSchema::required(
                    "query_symbols",
                    "Comma-separated symbol names or IDs to query.",
                ),
                ToolParamSchema::optional(
                    "top_k",
                    "Maximum number of episodes to return (default: 5).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess, Capability::ReadKg]),
                description: "Read-only episodic memory recall — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
