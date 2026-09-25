//! The [`SelectingManager`]: context by selection, never by rewriting.
//!
//! Reading the literature (Zhang et al., arXiv:2404.13501) separates memory
//! *reading* — choosing what enters the prompt by similarity, recency or
//! importance — from memory *management*, which summarizes, merges and forgets.
//! Retrieval-based selection is the mainstream reading mechanism, and MemGPT
//! adds a paging hierarchy above it rather than replacing it.
//!
//! Selection also has a property rewriting cannot have on a resident runtime:
//! it preserves an exact token prefix, so the keyed KV cache still matches. On
//! this model that is worth roughly 7x on time to first token. Any strategy that
//! rewrites earlier turns invalidates the prefix and pays a full prefill.
//!
//! This manager selects over a message array, so a native turn's tool calls,
//! tool results, pinned system instructions and `neural_recall` markers survive
//! intact — none of which the speaker/text conversation model can represent.

use serde_json::Value;

use super::error::{ContextError, ContextResult};
use super::manager::{ContextManager, TokenCounter};
use super::types::{
    AssembledContext, ContextSection, DropReason, DroppedSpan, EpisodeState, StrategyId,
    TokenBudget,
};

/// Selects whole turns from a message array to fit a budget.
///
/// Oldest turns go first. Unmarked system messages are pinned, because they are
/// standing instructions rather than history. A `neural_recall` system message
/// belongs to the user turn it precedes and is dropped with it. The newest user
/// turn and everything after it always survive; if that alone overflows, the
/// turn fails rather than silently losing the current request.
#[derive(Debug, Default, Clone, Copy)]
pub struct SelectingManager;

/// Index of the first message of each user turn, with recall markers pulled in.
fn turn_starts(messages: &[Value]) -> Vec<usize> {
    let mut starts = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message["role"] != "user" {
            continue;
        }
        let mut start = index;
        while start > 0
            && messages[start - 1]["role"] == "system"
            && messages[start - 1].get("neural_recall").is_some()
        {
            start -= 1;
        }
        starts.push(start);
    }
    starts
}

/// True when a message is a standing instruction rather than history.
fn is_pinned(message: &Value) -> bool {
    message["role"] == "system" && message.get("neural_recall").is_none()
}

/// Which section a message belongs to, for accounting.
fn section_of(message: &Value) -> ContextSection {
    if is_pinned(message) {
        ContextSection::System
    } else if message.get("neural_recall").is_some() {
        ContextSection::RecalledEpisodes
    } else {
        ContextSection::History
    }
}

/// A compact rendering used only for token counting and drop records.
fn render(message: &Value) -> String {
    let role = message["role"].as_str().unwrap_or("unknown");
    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| message.to_string());
    format!("{role}: {content}")
}

impl SelectingManager {
    /// Messages that survive, and those dropped, for `budget`.
    fn select(
        messages: &[Value],
        budget: usize,
        counter: &dyn TokenCounter,
    ) -> ContextResult<(Vec<usize>, Vec<usize>)> {
        let cost = |index: usize| counter.count(&render(&messages[index]));
        let mut keep = vec![true; messages.len()];
        let mut total: usize = (0..messages.len()).map(cost).sum();

        let starts = turn_starts(messages);
        // Everything from the last user turn onward is current, never dropped.
        for turns in starts.windows(2) {
            if total <= budget {
                break;
            }
            for index in turns[0]..turns[1] {
                if keep[index] && !is_pinned(&messages[index]) {
                    keep[index] = false;
                    total -= cost(index);
                }
            }
        }

        if total > budget {
            return Err(ContextError::BudgetExceeded {
                needed: total,
                available: budget,
            });
        }
        Ok((
            (0..messages.len()).filter(|i| keep[*i]).collect(),
            (0..messages.len()).filter(|i| !keep[*i]).collect(),
        ))
    }
}

impl ContextManager for SelectingManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("selecting")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        let messages = episode
            .messages
            .ok_or_else(|| ContextError::MissingMessages {
                strategy: self.id().as_str().to_string(),
            })?;
        let (kept, dropped) = Self::select(messages, budget.available(), counter)?;

        let mut section_tokens = std::collections::BTreeMap::new();
        for index in &kept {
            *section_tokens
                .entry(section_of(&messages[*index]))
                .or_insert(0) += counter.count(&render(&messages[*index]));
        }
        Ok(AssembledContext {
            messages: kept.iter().map(|i| messages[*i].clone()).collect(),
            tools: episode.tool_catalog.to_vec(),
            section_tokens,
            dropped: dropped
                .iter()
                .map(|index| DroppedSpan {
                    section: section_of(&messages[*index]),
                    text: render(&messages[*index]),
                    anchor: None,
                    reason: DropReason::Budget,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::manager::HeuristicTokenCounter;
    use serde_json::json;

    /// One token per message, so budgets are message counts.
    struct PerMessage;
    impl TokenCounter for PerMessage {
        fn count(&self, _text: &str) -> usize {
            1
        }
        fn label(&self) -> &str {
            "per-message"
        }
    }

    fn history() -> Vec<Value> {
        vec![
            json!({"role":"system","content":"standing rules"}),
            json!({"role":"system","neural_recall":{},"content":"recalled fact"}),
            json!({"role":"user","content":"old question"}),
            json!({"role":"assistant","tool_calls":[{"id":"a"}]}),
            json!({"role":"tool","tool_call_id":"a","content":"tool result"}),
            json!({"role":"assistant","content":"old answer"}),
            json!({"role":"user","content":"current question"}),
        ]
    }

    #[test]
    fn a_dropped_turn_takes_its_tool_traffic_and_recall_marker_with_it() {
        let messages = history();
        let (kept, dropped) = SelectingManager::select(&messages, 2, &PerMessage).unwrap();
        // Pinned instruction and the current turn survive; the old turn goes
        // whole, including its recall marker and both tool messages.
        assert_eq!(kept, vec![0, 6]);
        assert_eq!(dropped, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn nothing_is_dropped_when_it_already_fits() {
        let messages = history();
        let (kept, dropped) = SelectingManager::select(&messages, 100, &PerMessage).unwrap();
        assert_eq!(kept.len(), messages.len());
        assert!(dropped.is_empty());
    }

    #[test]
    fn an_overflowing_current_turn_fails_rather_than_losing_the_request() {
        let messages = history();
        // Budget below what the pinned rules plus the current turn need.
        let error = SelectingManager::select(&messages, 1, &PerMessage).unwrap_err();
        assert!(
            matches!(error, ContextError::BudgetExceeded { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn tool_messages_are_preserved_verbatim_when_kept() {
        let messages = history();
        let (kept, _) = SelectingManager::select(&messages, 100, &PerMessage).unwrap();
        let tool = &messages[kept[4]];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "a");
        // Selection copies messages; it never rewrites their content.
        assert_eq!(tool["content"], "tool result");
    }

    #[test]
    fn accounting_separates_pinned_instructions_recall_and_history() {
        let messages = history();
        let counter = HeuristicTokenCounter;
        let manager = SelectingManager;
        let sections: Vec<ContextSection> = messages.iter().map(section_of).collect();
        assert_eq!(sections[0], ContextSection::System);
        assert_eq!(sections[1], ContextSection::RecalledEpisodes);
        assert_eq!(sections[2], ContextSection::History);
        assert_eq!(manager.id().as_str(), "selecting");
        // The counter is only exercised through assemble, which needs an
        // engine; section attribution is the part testable in isolation.
        let _ = counter.label();
    }
}
