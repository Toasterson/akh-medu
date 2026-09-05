//! Discourse selection (Phase 44e): *which* prior turns to keep under budget.
//!
//! The default is recency (newest-first), which measure-first Q1 showed resolves
//! coreference. When the conversation carries a focus (active referents) and its
//! turns carry resolved entities, this ranks turns by **graph proximity** — entity
//! overlap with the focus — blended with recency, so a relevant older turn can be
//! kept over a recent off-topic one when the budget forces a choice. The current
//! prompt (the most recent turn) always ranks first and is never dropped.
//!
//! The agentic chat path records turns without per-turn entity resolution, so it
//! takes the recency fallback today; the entity-aware (symbolic) path activates
//! proximity. Either way it only changes behaviour when history is *trimmed*,
//! which the large model windows make rare (the measure-first finding).

use crate::agent::conversation::ConversationState;
use crate::symbol::SymbolId;

/// The focus the discourse ranker measures proximity against: the single current
/// topic when one is set (avoids off-topic referents that merely lingered on the
/// table), else the full active-referent set.
fn focus_entities(conv: &ConversationState) -> Vec<SymbolId> {
    match conv.active_topic {
        Some(topic) => vec![topic],
        None => conv.active_referents.clone(),
    }
}

/// Whether proximity ranking applies: a focus exists and turns carry entities.
pub fn has_proximity_signal(conv: &ConversationState) -> bool {
    !focus_entities(conv).is_empty() && conv.turns.iter().any(|t| !t.resolved_entities.is_empty())
}

/// Turn indices in **keep-priority order** (highest priority first).
///
/// Pure recency (newest-first) unless [`has_proximity_signal`]; then proximity ⊕
/// recency, with the most recent turn pinned first.
pub fn rank_turns_keep_first(conv: &ConversationState) -> Vec<usize> {
    let n = conv.turns.len();
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();

    if !has_proximity_signal(conv) {
        idx.sort_unstable_by(|a, b| b.cmp(a)); // newest first
        return idx;
    }

    let focus = focus_entities(conv);
    let score = |i: usize| -> f32 {
        let t = &conv.turns[i];
        let overlap = t
            .resolved_entities
            .iter()
            .filter(|e| focus.contains(e))
            .count() as f32;
        let proximity = overlap / focus.len() as f32;
        let recency = (i + 1) as f32 / n as f32; // newest → ~1.0
        proximity + recency
    };
    let last = n - 1;
    idx.sort_by(|a, b| {
        // Pin the current prompt (most recent turn) to the front.
        if *a == last {
            return std::cmp::Ordering::Less;
        }
        if *b == last {
            return std::cmp::Ordering::Greater;
        }
        score(*b)
            .partial_cmp(&score(*a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::conversation::{ConversationState, Speaker};
    use crate::symbol::SymbolId;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    #[test]
    fn recency_fallback_when_no_entities() {
        let mut conv = ConversationState::new("ch", "narrative");
        conv.record_operator_turn("a");
        conv.record_operator_turn("b");
        conv.record_operator_turn("c"); // most recent
        // No resolved entities anywhere → pure recency, newest first.
        assert_eq!(rank_turns_keep_first(&conv), vec![2, 1, 0]);
    }

    #[test]
    fn proximity_lifts_on_topic_turn_above_recent_off_topic() {
        let mut conv = ConversationState::new("ch", "narrative");
        // Turn 0: on the focus topic (sym 42). Turn 1: off-topic. Turn 2: current.
        conv.add_turn(Speaker::Operator, "about the tower", vec![sym(42)]);
        conv.add_turn(Speaker::Operator, "unrelated aside", vec![sym(7)]);
        conv.add_turn(Speaker::Operator, "what about it?", vec![sym(42)]);
        // Focus is sym(42) (set by the last entity-bearing turn).
        let order = rank_turns_keep_first(&conv);
        assert_eq!(order[0], 2, "current prompt pinned first");
        // The on-topic older turn (0) outranks the off-topic turn (1).
        let pos0 = order.iter().position(|&i| i == 0).unwrap();
        let pos1 = order.iter().position(|&i| i == 1).unwrap();
        assert!(
            pos0 < pos1,
            "on-topic turn kept before off-topic: {order:?}"
        );
    }
}
