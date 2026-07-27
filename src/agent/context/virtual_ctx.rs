//! The [`VirtualContextManager`] strategy (Phase 44i).
//!
//! MemGPT-style tiered memory: the model manages its own context across three
//! tiers —
//! - **core** (always in context, small, model-editable via the
//!   `core_memory_append` / `core_memory_replace` self-edit tools — see
//!   [`super::core_memory`] and `agent::tools::core_memory`),
//! - **recall** (recent conversation, paged in as the bounded history tail), and
//! - **archival** (the long-term KG, searched on demand via `archival_search`).
//!
//! [`assemble`](ContextManager::assemble) renders the core-memory block first,
//! then defers to the curating assembly with a short history tail (the recall
//! tier). If the model never edits core memory, the result degrades to a
//! [`PassiveManager`](super::passive::PassiveManager)-like lean context — recall +
//! tail + the current prompt — which is the documented "passive safety net". Like
//! every shipped manager it is wrapped in
//! [`SafetyEnforced`](super::safety::SafetyEnforced).

use serde_json::json;

use super::compacting::assemble_with_cap;
use super::core_memory;
use super::error::ContextResult;
use super::manager::{ContextManager, TokenCounter};
use super::types::{
    AssembledContext, ContextSection, ContextWindowStats, EpisodeState, StrategyId, TokenBudget,
};

/// Recall-tier history tail kept verbatim (the model pages older context in via
/// `archival_search` / `memory_recall` instead of carrying it in every prompt).
pub const DEFAULT_TAIL: usize = 4;

/// Tiered self-managed working-context strategy (44i).
#[derive(Debug, Clone, Copy, Default)]
pub struct VirtualContextManager {
    tail: Option<usize>,
}

impl VirtualContextManager {
    /// Manager keeping the last `tail` turns as the recall tier.
    pub fn with_tail(tail: usize) -> Self {
        Self { tail: Some(tail.max(1)) }
    }

    fn tail(&self) -> usize {
        self.tail.unwrap_or(DEFAULT_TAIL)
    }
}

impl ContextManager for VirtualContextManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("virtual")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        // Core tier: the model-editable always-in-context block.
        let lines = core_memory::read(episode.engine, episode.agent_id);
        let core_block = (!lines.is_empty()).then(|| {
            format!(
                "Core memory (persistent; edit with core_memory_append / core_memory_replace):\n- {}",
                lines.join("\n- ")
            )
        });
        let core_tokens = core_block.as_ref().map(|b| counter.count(b)).unwrap_or(0);

        // Recall + archival + the rest, assembled within the budget left after the
        // core block, with only the recent tail kept verbatim.
        let inner = TokenBudget::new(
            budget.window.saturating_sub(core_tokens),
            budget.reserved_output,
        );
        let mut assembled = assemble_with_cap(episode, &inner, counter, Some(self.tail()))?;

        if let Some(block) = core_block {
            assembled.messages.insert(0, json!({ "role": "system", "content": block }));
            assembled
                .section_tokens
                .entry(ContextSection::System)
                .and_modify(|t| *t += core_tokens)
                .or_insert(core_tokens);
        }
        Ok(assembled)
    }

    /// Override to fill the virtual-context extras (ADR 065 D12). Today there
    /// is no live VSA paging in the assembler — the model pages archival
    /// context in via the `archival_search` tool, which runs *during* the
    /// turn, not in `assemble`. So all three VSA extras are `Some(0)` (the
    /// manager *is* a virtual one, just not yet paging at assemble-time);
    /// the compacting extras inherit from the shared `assemble_with_cap`.
    fn stats(&self, ctx: &AssembledContext, budget: &TokenBudget) -> ContextWindowStats {
        let mut s = super::manager::default_stats(self.id(), ctx, budget, 0);
        // Inherit the compacting-style extras (this manager delegates to
        // assemble_with_cap, so history compaction is reported the same way).
        let compacted = ctx
            .dropped
            .iter()
            .filter(|d| d.section == ContextSection::History)
            .count();
        s.compacted = Some(compacted);
        let hist_kept = ctx
            .section_tokens
            .get(&ContextSection::History)
            .copied()
            .unwrap_or(0);
        let hist_dropped: usize = ctx
            .dropped
            .iter()
            .filter(|d| d.section == ContextSection::History)
            .map(|d| d.text.len().div_ceil(4))
            .sum();
        let hist_in = hist_kept + hist_dropped;
        s.compaction_ratio = Some(if hist_in == 0 {
            1.0
        } else {
            (hist_kept as f32) / (hist_in as f32)
        });
        // VSA extras — reserved for the future paging work; present so the
        // panel can tell this is a virtual manager.
        s.virtual_referenced = Some(0);
        s.physical_materialized = Some(0);
        s.vsa_hit_rate = Some(0.0);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::core_memory;
    use crate::agent::context::manager::HeuristicTokenCounter;
    use crate::engine::{Engine, EngineConfig};

    fn test_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, tmp)
    }

    #[test]
    fn renders_core_memory_first() {
        let (engine, _t) = test_engine();
        core_memory::append(&engine, "akh", "user's name is Toni").unwrap();
        let ep = EpisodeState::new("akh", "what's my name?", &engine);
        let out = VirtualContextManager::default()
            .assemble(&ep, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        // Core memory leads as a system block; the prompt is the final user turn.
        assert_eq!(out.messages[0]["role"], "system");
        assert!(out.messages[0]["content"].as_str().unwrap().contains("Toni"));
        assert_eq!(out.messages.last().unwrap()["role"], "user");
        assert!(out.section_tokens.contains_key(&ContextSection::System));
    }

    #[test]
    fn degrades_to_lean_without_core_memory() {
        let (engine, _t) = test_engine();
        // No core memory written ⇒ behaves like a lean passive context.
        let ep = EpisodeState::new("akh", "hello", &engine);
        let out = VirtualContextManager::default()
            .assemble(&ep, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert_eq!(out.messages, vec![json!({"role":"user","content":"hello"})]);
    }
}
