//! The [`PassiveManager`] strategy (Phase 44h).
//!
//! MemGPT / generative-agents "passive" memory: the working context stays **lean**
//! — continuity seed + KG facts + recalled episodes + the immediate exchange (a
//! bounded `tail` of recent turns) + the current prompt — and the longer-term
//! thread is carried by *background consolidation* (the existing sleep/idle cycle,
//! [`agent::memory::consolidate`](crate::agent::memory::consolidate)) surfaced
//! through recall, not by stuffing verbatim history into every prompt. The main
//! agentic loop therefore spends **no extra tokens** growing history over a
//! session — that work happens off the hot path.
//!
//! [`assemble`](ContextManager::assemble) is a pure function of the borrowed
//! state, so it is idempotent; [`commit`](ContextManager::commit) is a no-op
//! (consolidation is the background cycle). Like every shipped manager it is
//! wrapped in [`SafetyEnforced`](super::safety::SafetyEnforced) before use, with a
//! `CompactingManager`-style fallback available if recall ever comes up empty.

use super::compacting::assemble_with_cap;
use super::error::ContextResult;
use super::manager::{ContextManager, TokenCounter};
use super::types::{
    AssembledContext, ContextSection, ContextWindowStats, EpisodeState, StrategyId, TokenBudget,
};

/// Most-recent turns kept verbatim (one user+agent pair plus the current prompt).
pub const DEFAULT_TAIL: usize = 3;

/// Lean working-context strategy backed by background consolidation (44h).
#[derive(Debug, Clone, Copy)]
pub struct PassiveManager {
    /// How many most-recent turns stay verbatim; older turns are left to recall.
    tail: usize,
}

impl Default for PassiveManager {
    fn default() -> Self {
        Self { tail: DEFAULT_TAIL }
    }
}

impl PassiveManager {
    /// Manager keeping the last `tail` turns verbatim (clamped to ≥ 1 so the
    /// current prompt is never dropped).
    pub fn with_tail(tail: usize) -> Self {
        Self { tail: tail.max(1) }
    }
}

impl ContextManager for PassiveManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("passive")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        assemble_with_cap(episode, budget, counter, Some(self.tail))
    }

    /// Override to fill the compacting extras the same way CompactingManager
    /// does (the two share `assemble_with_cap`). PassiveManager's distinct
    /// character (age-capping history to the tail) is already visible in the
    /// per-section `drop_reason` — History drops under the tail cap are
    /// `Age`, not `Budget`.
    fn stats(&self, ctx: &AssembledContext, budget: &TokenBudget) -> ContextWindowStats {
        let mut s = super::manager::default_stats(self.id(), ctx, budget, 0);
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
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::ContextSection;
    use crate::agent::context::manager::HeuristicTokenCounter;
    use crate::agent::conversation::ConversationState;
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
    fn keeps_only_the_immediate_tail() {
        let (engine, _t) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        for i in 0..8 {
            conv.record_operator_turn(&format!("turn {i}"));
        }
        let ep = EpisodeState::new("a", "turn 7", &engine).with_conversation(Some(&conv));
        // Huge budget, yet the tail cap (not the budget) bounds history.
        let out = PassiveManager::with_tail(2)
            .assemble(&ep, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        let user_msgs = out.messages.iter().filter(|m| m["role"] == "user").count();
        assert_eq!(
            user_msgs, 2,
            "passive keeps only the tail: {:?}",
            out.messages
        );
        assert_eq!(out.messages.last().unwrap()["content"], "turn 7");
        let dropped_history = out
            .dropped
            .iter()
            .filter(|d| d.section == ContextSection::History)
            .count();
        assert!(
            dropped_history >= 6,
            "older turns left to recall, not verbatim"
        );
    }

    #[test]
    fn assemble_is_idempotent() {
        let (engine, _t) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        conv.record_operator_turn("hello");
        conv.record_agent_turn("hi");
        conv.record_operator_turn("again");
        let ep = EpisodeState::new("a", "again", &engine).with_conversation(Some(&conv));
        let m = PassiveManager::default();
        let a = m
            .assemble(&ep, &TokenBudget::default(), &HeuristicTokenCounter)
            .unwrap();
        let b = m
            .assemble(&ep, &TokenBudget::default(), &HeuristicTokenCounter)
            .unwrap();
        assert_eq!(a.messages, b.messages, "assemble has no hidden state");
    }
}
