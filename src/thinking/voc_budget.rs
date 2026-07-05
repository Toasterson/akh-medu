//! VOC-driven adaptive budget (Phase 41e).
//!
//! Composes the Phase 11g Value-of-Computation machinery
//! (`agent::resource::{ImprovementHistory, record_improvement,
//! thinking_marginal_value, history_diminishing_returns}`) into a controller
//! the loop consults every `check_every` steps:
//!
//! - **Halt** when the marginal value of another step drops below threshold
//!   (the Apple *illusion-of-thinking* fix: stop when more thinking won't
//!   help).
//! - **Extend** the step budget when marginal value stays high near
//!   exhaustion, capped by a global ceiling.

use crate::agent::resource::{
    ImprovementHistory, history_diminishing_returns, record_improvement, thinking_marginal_value,
};
use crate::symbol::SymbolId;

use super::budget::ThinkingBudget;

/// What the controller wants the loop to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VocDecision {
    /// Keep thinking.
    Continue,
    /// Marginal value below threshold — halt with `DiminishingReturns`.
    Halt { marginal_value: f32 },
    /// Marginal value high and budget nearly exhausted — extend by `extra`.
    Extend { extra: u32, marginal_value: f32 },
}

/// VOC controller configuration + state.
#[derive(Debug, Clone)]
pub struct VocBudgetController {
    /// Consult the controller every N primitive steps.
    pub check_every: u32,
    /// Halt when marginal value falls below this (Phase 11g semantics:
    /// negative VOC means the computation is no longer worth its cost).
    pub halt_threshold: f32,
    /// Extend when marginal value is at least this and the step budget is
    /// nearly used up.
    pub extend_threshold: f32,
    /// Extend by this fraction of the original step budget.
    pub extend_fraction: f32,
    /// Hard ceiling on total steps regardless of extensions.
    pub global_step_cap: u32,
    /// Per-step cost in the VOC formula.
    pub step_cost: f32,
    /// How many verifier-score observations to retain per goal.
    pub max_history: usize,

    history: ImprovementHistory,
    last_score: Option<f32>,
    observations: u64,
}

impl VocBudgetController {
    /// Controller sized for an initial budget of `max_steps`.
    pub fn for_budget(budget: &ThinkingBudget) -> Self {
        Self {
            check_every: 16,
            halt_threshold: 0.0,
            extend_threshold: 0.25,
            extend_fraction: 0.25,
            global_step_cap: budget.max_steps.saturating_mul(4),
            step_cost: 0.05,
            max_history: 32,
            history: ImprovementHistory::default(),
            last_score: None,
            observations: 0,
        }
    }

    /// Record a verifier score for `goal`. The improvement *rate* fed into
    /// the Phase 11g history is the positive part of the score delta.
    pub fn observe_score(&mut self, goal: SymbolId, score: f32) {
        let rate = match self.last_score {
            Some(prev) => (score - prev).max(0.0),
            // First observation: treat the score itself as headroom-scaled
            // progress so a strong early answer registers as improvement.
            None => score.clamp(0.0, 1.0),
        };
        self.last_score = Some(score);
        self.observations += 1;
        record_improvement(
            &mut self.history,
            goal,
            self.observations,
            rate,
            self.max_history,
        );
    }

    /// Should the loop halt, extend, or continue?
    ///
    /// `top_belief` is the current best hypothesis belief; the marginal-value
    /// magnitude is its complement (little left to gain once belief is high).
    pub fn decide(&self, goal: SymbolId, budget: &ThinkingBudget, top_belief: f32) -> VocDecision {
        // No observations yet: nothing to reason about.
        if self.observations == 0 {
            return VocDecision::Continue;
        }

        let magnitude = (1.0 - top_belief).clamp(0.0, 1.0);
        let marginal_value =
            thinking_marginal_value(&self.history, goal, magnitude, self.step_cost);

        // Halt on negative marginal value, but only once the improvement
        // signal has actually stalled — a single flat check right after a
        // strong step should not kill the deliberation.
        if marginal_value < self.halt_threshold
            && (self.observations >= 2 || history_diminishing_returns(&self.history, goal))
        {
            return VocDecision::Halt { marginal_value };
        }

        // Extend when value remains high near exhaustion (within global cap).
        if marginal_value >= self.extend_threshold
            && budget.step_fraction_used() >= 0.9
            && budget.max_steps < self.global_step_cap
        {
            let extra = ((budget.max_steps as f32 * self.extend_fraction) as u32).max(1);
            return VocDecision::Extend {
                extra,
                marginal_value,
            };
        }

        VocDecision::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goal() -> SymbolId {
        SymbolId::new(42).unwrap()
    }

    #[test]
    fn no_observations_means_continue() {
        let budget = ThinkingBudget::steps(100);
        let ctl = VocBudgetController::for_budget(&budget);
        assert_eq!(ctl.decide(goal(), &budget, 0.0), VocDecision::Continue);
    }

    #[test]
    fn stalled_scores_trigger_halt() {
        let budget = ThinkingBudget::steps(100);
        let mut ctl = VocBudgetController::for_budget(&budget);
        // Verifier score flatlines: deltas are zero → P(improvement) ~ 0.
        ctl.observe_score(goal(), 0.5);
        ctl.observe_score(goal(), 0.5);
        ctl.observe_score(goal(), 0.5);
        match ctl.decide(goal(), &budget, 0.5) {
            VocDecision::Halt { marginal_value } => {
                assert!(marginal_value < 0.0, "flat curve should have negative VOC");
            }
            other => panic!("expected Halt, got {other:?}"),
        }
    }

    #[test]
    fn rising_scores_continue() {
        let budget = ThinkingBudget::steps(100);
        let mut ctl = VocBudgetController::for_budget(&budget);
        ctl.observe_score(goal(), 0.1);
        ctl.observe_score(goal(), 0.4);
        ctl.observe_score(goal(), 0.7);
        assert_eq!(ctl.decide(goal(), &budget, 0.3), VocDecision::Continue);
    }

    #[test]
    fn high_value_near_exhaustion_extends() {
        let mut budget = ThinkingBudget::steps(10);
        let mut ctl = VocBudgetController::for_budget(&budget);
        ctl.observe_score(goal(), 0.2);
        ctl.observe_score(goal(), 0.8); // big improvement → high P(improvement)
        for _ in 0..9 {
            budget.charge(1.0, 0);
        }
        match ctl.decide(goal(), &budget, 0.2) {
            VocDecision::Extend { extra, .. } => assert!(extra >= 1),
            other => panic!("expected Extend, got {other:?}"),
        }
    }

    #[test]
    fn extension_respects_global_cap() {
        let mut budget = ThinkingBudget::steps(10);
        let mut ctl = VocBudgetController::for_budget(&budget);
        ctl.global_step_cap = 10; // no headroom at all
        ctl.observe_score(goal(), 0.2);
        ctl.observe_score(goal(), 0.9);
        for _ in 0..9 {
            budget.charge(1.0, 0);
        }
        assert_eq!(ctl.decide(goal(), &budget, 0.2), VocDecision::Continue);
    }
}
