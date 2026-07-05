//! Budget allocation skeleton + per-turn telemetry (Phase 44a; allocator
//! foundation for ADR 064 P5 / lesson 2).
//!
//! 44a lands the seam and the instrumentation; the graduated-degradation logic
//! (render at lower fidelity instead of a hard cap) is 44c. Here the allocator
//! grants each section its request in keep-first priority order until the budget
//! is spent — replacing the three independent caps with one ordered division.

use std::collections::BTreeMap;

use super::types::{AssembledContext, ContextSection, StrategyId, TokenBudget};

/// Divides a token budget across context sections by priority.
#[derive(Debug, Clone)]
pub struct BudgetAllocator {
    /// Sections in keep-first order: earlier sections are funded before later
    /// ones when the budget is tight.
    priority: Vec<ContextSection>,
}

impl Default for BudgetAllocator {
    fn default() -> Self {
        Self {
            priority: vec![
                ContextSection::System,
                ContextSection::UserPrompt,
                ContextSection::KnowledgeGraph,
                ContextSection::History,
                ContextSection::RecalledEpisodes,
                ContextSection::Tools,
            ],
        }
    }
}

impl BudgetAllocator {
    /// Allocator with an explicit section priority order.
    pub fn new(priority: Vec<ContextSection>) -> Self {
        Self { priority }
    }

    /// Grant each requested section up to its ask, in priority order, until
    /// `available` tokens are exhausted. Sections not in the priority list are
    /// funded last (in the requested map's order). 44c replaces the hard cap with
    /// graduated fidelity (full → summary → handle → dropped).
    pub fn allocate(
        &self,
        available: usize,
        requested: &BTreeMap<ContextSection, usize>,
    ) -> BTreeMap<ContextSection, usize> {
        let mut granted = BTreeMap::new();
        let mut remaining = available;
        for sec in &self.priority {
            if let Some(&want) = requested.get(sec) {
                let give = want.min(remaining);
                remaining -= give;
                granted.insert(*sec, give);
            }
        }
        for (sec, &want) in requested {
            if !granted.contains_key(sec) {
                let give = want.min(remaining);
                remaining -= give;
                granted.insert(*sec, give);
            }
        }
        granted
    }

    /// Like [`allocate`](Self::allocate) but sections listed in `atomic` are
    /// granted **all-or-nothing**: a section that does not fit whole gets `0`
    /// (rather than a partial grant that can't be rendered without summarising —
    /// that's the 44c-full graduated-fidelity refinement). Divisible sections
    /// (history, tools) still get `want.min(remaining)` so they degrade smoothly.
    ///
    /// This is what [`CompactingManager`](super::compacting::CompactingManager)
    /// uses: KG / recalled / seed blocks are atomic; history + tools are divisible.
    pub fn allocate_keep_first(
        &self,
        available: usize,
        requested: &BTreeMap<ContextSection, usize>,
        atomic: &[ContextSection],
    ) -> BTreeMap<ContextSection, usize> {
        let mut granted = BTreeMap::new();
        let mut remaining = available;
        // Pure: how much this section gets given `remaining` (atomic ⇒ whole-or-0).
        let give = |sec: &ContextSection, want: usize, remaining: usize| -> usize {
            if atomic.contains(sec) {
                if want <= remaining { want } else { 0 }
            } else {
                want.min(remaining)
            }
        };
        for sec in &self.priority {
            if let Some(&want) = requested.get(sec) {
                let g = give(sec, want, remaining);
                remaining -= g;
                granted.insert(*sec, g);
            }
        }
        for (sec, &want) in requested {
            if !granted.contains_key(sec) {
                let g = give(sec, want, remaining);
                remaining -= g;
                granted.insert(*sec, g);
            }
        }
        granted
    }
}

/// Emit a structured per-turn telemetry event with token usage per section.
///
/// Answers measure-first Q2 (the real window vs. what the legacy caps sum to)
/// from a live run: enable `akh_medu::agent::context=debug` and read the
/// `sections` field per turn.
pub fn record_assembly_telemetry(
    strategy: &StrategyId,
    ctx: &AssembledContext,
    budget: &TokenBudget,
    counter_label: &str,
) {
    tracing::debug!(
        strategy = strategy.as_str(),
        counter = counter_label,
        total_tokens = ctx.total_tokens(),
        available = budget.available(),
        window = budget.window,
        dropped = ctx.dropped.len(),
        sections = ?ctx.section_tokens,
        "phase44: context assembled",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_in_priority_order_and_clamps_to_budget() {
        let alloc = BudgetAllocator::default();
        let mut want = BTreeMap::new();
        want.insert(ContextSection::System, 100);
        want.insert(ContextSection::UserPrompt, 50);
        want.insert(ContextSection::History, 400);
        want.insert(ContextSection::Tools, 200);

        // Budget of 300: System (100) + UserPrompt (50) fully funded, then KG (0,
        // not requested), History gets the remaining 150, Tools gets 0.
        let granted = alloc.allocate(300, &want);
        assert_eq!(granted[&ContextSection::System], 100);
        assert_eq!(granted[&ContextSection::UserPrompt], 50);
        assert_eq!(granted[&ContextSection::History], 150);
        assert_eq!(granted[&ContextSection::Tools], 0);
        let total: usize = granted.values().sum();
        assert_eq!(total, 300, "never exceeds the budget");
    }

    #[test]
    fn atomic_sections_are_all_or_nothing() {
        let alloc = BudgetAllocator::default();
        let mut want = BTreeMap::new();
        want.insert(ContextSection::UserPrompt, 20);
        want.insert(ContextSection::KnowledgeGraph, 100); // atomic, won't fit whole
        want.insert(ContextSection::History, 400); // divisible
        // Budget 120: UserPrompt (20) funded; KG wants 100 but only 100 left → fits
        // exactly; History gets 0. Drop KG by shrinking the budget instead:
        let granted = alloc.allocate_keep_first(60, &want, &[ContextSection::KnowledgeGraph]);
        assert_eq!(granted[&ContextSection::UserPrompt], 20);
        assert_eq!(granted[&ContextSection::KnowledgeGraph], 0, "atomic KG didn't fit → dropped whole");
        assert_eq!(granted[&ContextSection::History], 40, "divisible history takes the remainder");
        let total: usize = granted.values().sum();
        assert!(total <= 60);
    }

    #[test]
    fn unbounded_budget_grants_every_request() {
        let alloc = BudgetAllocator::default();
        let mut want = BTreeMap::new();
        want.insert(ContextSection::System, 100);
        want.insert(ContextSection::History, 400);
        let granted = alloc.allocate(10_000, &want);
        assert_eq!(granted[&ContextSection::System], 100);
        assert_eq!(granted[&ContextSection::History], 400);
    }
}
