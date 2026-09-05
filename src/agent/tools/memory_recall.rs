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
            description:
                "Recall episodic memories matching query terms (fuzzy -- terms need not be \
                          exact symbol names; 'cluster' matches `cluster-topology`). Returns \
                          each episode's stored FACTS and tags inline, so you can answer \
                          directly without further kg_query drill-down. Recency fallback if \
                          nothing matches."
                    .into(),
            parameters: vec![
                ToolParam {
                    name: "query_symbols".into(),
                    description: "Comma-separated symbol names or IDs to query. Generic terms \
                                  like 'recent' trigger a temporal scan of all episodes."
                        .into(),
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
        let top_k: usize = input.get("top_k").and_then(|k| k.parse().ok()).unwrap_or(5);

        // Normalized query terms (drop <3-char tokens that would match noise).
        let terms: Vec<String> = query_str
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s.len() >= 3)
            .collect();

        // 1. Exact symbol resolves (the original fast path).
        let mut query_ids: Vec<SymbolId> = query_str
            .split(',')
            .filter_map(|s| engine.resolve_symbol(s.trim()).ok())
            .collect();

        // 2. Fuzzy expansion — the fix for query DISCRIMINATION. The LLM rarely
        //    names the exact stored symbol ("cluster" must still reach
        //    `cluster-topology`/`tag:cluster-topology`), so also pull in any symbol
        //    whose label contains a query term. Capped so a common term can't drag
        //    in the whole graph and collapse back to a recency-like scan.
        if !terms.is_empty() {
            const MAX_EXPANSION: usize = 64;
            let mut matched = 0usize;
            for meta in engine.all_symbols() {
                if matched >= MAX_EXPANSION {
                    break;
                }
                let label = meta.label.to_lowercase();
                if terms.iter().any(|t| label.contains(t.as_str())) {
                    query_ids.push(meta.id);
                    matched += 1;
                }
            }
        }
        query_ids.sort_unstable_by_key(|s| s.get());
        query_ids.dedup();

        // 3. Recall episodes linked to / tagged with the matched symbols; fall back
        //    to recency only when nothing matched. Over-fetch a pool so step 4 can
        //    re-rank by relevance before truncating to top_k.
        let pool = (top_k * 4).max(16);
        let recent = || {
            memory::recall_recent_episodes(engine, &self.predicates, top_k).map_err(|e| {
                AgentError::ToolExecution {
                    tool_name: "memory_recall".into(),
                    message: format!("{e}"),
                }
            })
        };
        let (mut episodes, mode) = if query_ids.is_empty() {
            (recent()?, "recent")
        } else {
            let eps = memory::recall_episodes(engine, &query_ids, &self.predicates, pool, None)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "memory_recall".into(),
                    message: format!("{e}"),
                })?;
            if eps.is_empty() {
                (recent()?, "recent (no symbol match)")
            } else {
                (eps, "matched")
            }
        };

        if episodes.is_empty() {
            return Ok(ToolOutput::ok_with_symbols(
                "No episodic memories found.",
                query_ids,
            ));
        }

        // Resolve each episode's facts + tags up front (also reused for ranking).
        const MAX_FACTS: usize = 18;
        const MAX_TAGS: usize = 12;
        struct Rendered {
            symbol_id: SymbolId,
            summary: String,
            facts: Vec<String>,
            tags: Vec<String>,
            learnings: Vec<SymbolId>,
            score: usize,
        }
        let mut rendered: Vec<Rendered> = episodes
            .drain(..)
            .map(|ep| {
                // The episode's learned-subject concepts (full S-P-O rendering was
                // tried and regressed 6/8 → 2/8: 24 statements × 5 episodes bloated
                // the context and `triples_from` pulled in cross-episode noise).
                let facts: Vec<String> = ep
                    .learnings
                    .iter()
                    .take(MAX_FACTS)
                    .map(|l| engine.resolve_label(*l))
                    .collect();
                let tags: Vec<String> = ep
                    .tags
                    .iter()
                    .take(MAX_TAGS)
                    .map(|t| engine.resolve_label(*t))
                    .collect();
                // 4. Relevance score = how many query terms appear anywhere in the
                //    episode's text (summary + facts + tags). Surfaces the on-topic
                //    episode even when the recency pool is large.
                let blob =
                    format!("{} {} {}", ep.summary, facts.join(" "), tags.join(" ")).to_lowercase();
                let score = terms.iter().filter(|t| blob.contains(t.as_str())).count();
                Rendered {
                    symbol_id: ep.symbol_id,
                    summary: ep.summary,
                    facts,
                    tags,
                    learnings: ep.learnings,
                    score,
                }
            })
            .collect();
        // Stable sort: keep recall's recency order, then float higher-scoring up.
        rendered.sort_by(|a, b| b.score.cmp(&a.score));
        rendered.truncate(top_k);

        // 5. Output the actual FACTS inline (not just counts) so the agent can
        //    answer without extra kg_query drill-down rounds.
        let mut all_symbols = query_ids;
        let mut lines = Vec::new();
        for r in &rendered {
            let facts = if r.facts.is_empty() {
                "(none recorded)".to_string()
            } else {
                r.facts.join("; ")
            };
            lines.push(format!(
                "- [{}] {}\n    facts: {}\n    tags: {}",
                engine.resolve_label(r.symbol_id),
                r.summary,
                facts,
                r.tags.join(", "),
            ));
            all_symbols.push(r.symbol_id);
            all_symbols.extend(&r.learnings);
        }

        let result = format!(
            "Recalled {} episode(s) [{}]:\n{}",
            lines.len(),
            mode,
            lines.join("\n")
        );
        Ok(ToolOutput::ok_with_symbols(result, all_symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "memory_recall".into(),
            description:
                "Recall episodic memories matching query terms (fuzzy -- terms need not be \
                          exact symbol names; 'cluster' matches `cluster-topology`). Returns \
                          each episode's stored FACTS and tags inline, so you can answer \
                          directly without further kg_query drill-down. Recency fallback if \
                          nothing matches."
                    .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "query_symbols",
                    "Comma-separated symbol names or IDs to query. Generic terms like 'recent' \
                     trigger a temporal scan of all episodes.",
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
