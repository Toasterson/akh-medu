//! The default curating context manager (Phase 44c — first increment).
//!
//! Measure-first Q1 showed conversation history demonstrably improves NL quality
//! (a follow-up that needs prior context fails without it), and the agentic path
//! shipped *no* history. This manager fixes that: it renders the recent
//! conversation turns into the prompt, newest-first up to the token budget. The
//! windows are large (128K–1M, read from the model — see the measure-first
//! report), so in practice all recent turns fit; the budget check is the cap for
//! the rare long transcript.
//!
//! Graduated fidelity (full → summary → reference-handle → dropped) and the KG /
//! recalled-episode sections are later increments (44c-full / 44d / 44e / 44g);
//! this increment is verbatim recent history, the highest-value piece.

use std::collections::BTreeMap;

use serde_json::json;

use crate::agent::conversation::Speaker;

use super::error::ContextResult;
use super::manager::{ContextManager, TokenCounter};
use super::types::{AssembledContext, ContextSection, EpisodeState, StrategyId, TokenBudget};

/// Assembles recent conversation history + the current turn within budget.
///
/// Expects the caller to have recorded the current user turn into the
/// [`ConversationState`](crate::agent::conversation::ConversationState) before
/// assembly, so the conversation's last turn *is* the current prompt — this
/// manager renders the turns as-is (no separate append, no duplication).
#[derive(Debug, Default, Clone, Copy)]
pub struct CompactingManager;

impl ContextManager for CompactingManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("compacting")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        let available = budget.available();
        let mut section_tokens = BTreeMap::new();

        // Select recent turns newest-first until the budget is spent, then render
        // them chronologically. Falls back to just the current user prompt when
        // there is no conversation state (embedded callers).
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let mut history_tokens = 0usize;
        let mut user_prompt_tokens = counter.count(episode.user_prompt);

        if let Some(conv) = episode.conversation.filter(|c| !c.turns.is_empty()) {
            let mut chosen = Vec::new();
            for turn in conv.turns.iter().rev() {
                let t = counter.count(&turn.text);
                if !chosen.is_empty() && history_tokens + t > available {
                    break; // keep at least the most-recent turn (the current prompt)
                }
                history_tokens += t;
                chosen.push(turn);
            }
            chosen.reverse();
            for turn in chosen {
                let role = match turn.speaker {
                    Speaker::Operator => "user",
                    Speaker::Agent => "assistant",
                };
                messages.push(json!({ "role": role, "content": turn.text }));
            }
            // The current prompt is the last recorded turn; its tokens are already
            // counted in `history_tokens`, so don't double-count the section.
            user_prompt_tokens = 0;
            section_tokens.insert(ContextSection::History, history_tokens);
        }

        // No conversation state ⇒ behave like pass-through (current prompt only).
        if messages.is_empty() {
            messages.push(json!({ "role": "user", "content": episode.user_prompt }));
        }
        if user_prompt_tokens > 0 {
            section_tokens.insert(ContextSection::UserPrompt, user_prompt_tokens);
        }

        Ok(AssembledContext {
            messages,
            section_tokens,
            dropped: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::conversation::ConversationState;
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
    fn renders_recorded_history_as_roles() {
        let (engine, _tmp) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        conv.record_operator_turn("I'm thinking of the Eiffel Tower.");
        conv.record_agent_turn("Noted.");
        conv.record_operator_turn("What city is it in?"); // the current turn

        let episode = EpisodeState {
            agent_id: "a",
            user_prompt: "What city is it in?",
            engine: &engine,
            conversation: Some(&conv),
        };
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");

        // All three turns rendered, in order, with the right roles.
        assert_eq!(out.messages.len(), 3);
        assert_eq!(out.messages[0]["role"], "user");
        assert_eq!(out.messages[0]["content"], "I'm thinking of the Eiffel Tower.");
        assert_eq!(out.messages[1]["role"], "assistant");
        assert_eq!(out.messages[2]["role"], "user");
        assert_eq!(out.messages[2]["content"], "What city is it in?");
    }

    #[test]
    fn empty_conversation_is_passthrough_equivalent() {
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState {
            agent_id: "a",
            user_prompt: "hello",
            engine: &engine,
            conversation: None,
        };
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert_eq!(out.messages, vec![json!({"role":"user","content":"hello"})]);
    }

    #[test]
    fn budget_keeps_most_recent_turns() {
        let (engine, _tmp) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        // 20 turns of ~4 tokens each; a tiny budget keeps only the newest few.
        for i in 0..20 {
            conv.record_operator_turn(&format!("turn number {i} here"));
        }
        let episode = EpisodeState {
            agent_id: "a",
            user_prompt: "turn number 19 here",
            engine: &engine,
            conversation: Some(&conv),
        };
        // available = window - reserved; use a tiny window to force trimming.
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::new(40, 16), &HeuristicTokenCounter)
            .expect("assemble");
        // Fewer than all 20 turns, and the LAST rendered is the most recent.
        assert!(out.messages.len() < 20, "history trimmed to budget");
        assert_eq!(out.messages.last().unwrap()["content"], "turn number 19 here");
    }
}
