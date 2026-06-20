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
