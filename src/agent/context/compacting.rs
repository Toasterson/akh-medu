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
        let mut section_tokens: BTreeMap<ContextSection, usize> = BTreeMap::new();
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let mut spent = 0usize;

        // Leading context blocks, highest-priority first, each skipped if it would
        // bust the budget: 44e session seed → 44d KG facts → 44g recalled episodes.
        // Rendered as `system` messages (both backends surface them; ADR 065).
        let add_block = |section: ContextSection,
                             header: &str,
                             body: &str,
                             section_tokens: &mut BTreeMap<ContextSection, usize>,
                             messages: &mut Vec<serde_json::Value>,
                             spent: &mut usize| {
            let body = body.trim();
            if body.is_empty() {
                return;
            }
            let block = format!("{header}\n{body}");
            let t = counter.count(&block);
            if *spent + t > available {
                return;
            }
            *spent += t;
            section_tokens.insert(section, t);
            messages.push(json!({ "role": "system", "content": block }));
        };
        if let Some(seed) = episode.session_seed {
            add_block(ContextSection::System, "Session so far:", seed,
                &mut section_tokens, &mut messages, &mut spent);
        }
        if let Some(kg) = episode.kg_context {
            add_block(ContextSection::KnowledgeGraph, "Relevant knowledge from your memory:", kg,
                &mut section_tokens, &mut messages, &mut spent);
        }
        if !episode.recalled.is_empty() {
            let body = format!("- {}", episode.recalled.join("\n- "));
            add_block(ContextSection::RecalledEpisodes, "Relevant past episodes:", &body,
                &mut section_tokens, &mut messages, &mut spent);
        }

        // Conversation history within the remaining budget. Select newest-first
        // (always keeping the most recent turn — the current prompt), then render
        // chronologically. The current prompt is the last recorded turn, so no
        // separate append is needed.
        let has_turns = episode.conversation.map(|c| !c.turns.is_empty()).unwrap_or(false);
        if let Some(conv) = episode.conversation.filter(|_| has_turns) {
            let remaining = available.saturating_sub(spent);
            let mut history_tokens = 0usize;
            let mut chosen = Vec::new();
            for turn in conv.turns.iter().rev() {
                let t = counter.count(&turn.text);
                if !chosen.is_empty() && history_tokens + t > remaining {
                    break;
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
            section_tokens.insert(ContextSection::History, history_tokens);
        } else {
            // No conversation state ⇒ the current prompt as a single user turn.
            messages.push(json!({ "role": "user", "content": episode.user_prompt }));
            section_tokens.insert(ContextSection::UserPrompt, counter.count(episode.user_prompt));
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

        let episode =
            EpisodeState::new("a", "What city is it in?", &engine).with_conversation(Some(&conv));
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
    fn renders_kg_and_recalled_context_blocks() {
        let (engine, _tmp) = test_engine();
        let recalled = vec!["learned that Paris is the capital of France".to_string()];
        let episode = EpisodeState::new("a", "tell me about France", &engine).with_context(
            Some("France capital-is Paris"),
            &recalled,
            Some("Earlier you discussed travel plans."),
        );
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        // Three leading `system` context blocks (seed → kg → recalled), then the
        // current prompt as the final user turn.
        let systems = out.messages.iter().filter(|m| m["role"] == "system").count();
        assert_eq!(systems, 3, "seed + kg + recalled blocks: {:?}", out.messages);
        assert!(
            out.messages.iter().any(|m| m["content"].as_str().unwrap_or("").contains("Paris")),
            "KG/recalled content present"
        );
        assert_eq!(out.messages.last().unwrap()["role"], "user");
        assert!(out.section_tokens.contains_key(&ContextSection::KnowledgeGraph));
        assert!(out.section_tokens.contains_key(&ContextSection::RecalledEpisodes));
    }

    #[test]
    fn empty_conversation_is_passthrough_equivalent() {
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState::new("a", "hello", &engine);
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
        let episode = EpisodeState::new("a", "turn number 19 here", &engine)
            .with_conversation(Some(&conv));
        // available = window - reserved; use a tiny window to force trimming.
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::new(40, 16), &HeuristicTokenCounter)
            .expect("assemble");
        // Fewer than all 20 turns, and the LAST rendered is the most recent.
        assert!(out.messages.len() < 20, "history trimmed to budget");
        assert_eq!(out.messages.last().unwrap()["content"], "turn number 19 here");
    }
}
