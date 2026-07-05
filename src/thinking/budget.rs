//! Thinking budget: the "how hard may I think" dial (Phase 41a).
//!
//! A [`ThinkingBudget`] bounds a deliberation along three axes:
//! primitive steps, wall-clock time, and Phase 11g VOC credits.
//! The loop charges the budget after every primitive step and halts with
//! [`super::state::HaltReason::BudgetExhausted`] when any axis runs out.

use serde::{Deserialize, Serialize};

use super::error::{ThinkingError, ThinkingResult};

/// Which budget axis ran out first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetResource {
    /// `used_steps` reached `max_steps`.
    Steps,
    /// `used_wall_ms` reached `max_wall_ms`.
    WallClock,
    /// `used_voc` reached `voc_credits`.
    VocCredits,
}

impl std::fmt::Display for BudgetResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Steps => "steps",
            Self::WallClock => "wall-clock",
            Self::VocCredits => "voc-credits",
        };
        f.write_str(s)
    }
}

/// Remaining and consumed deliberation budget.
///
/// Wall-clock and VOC limits are optional in practice: set them to
/// `u64::MAX` / `f32::INFINITY` (the [`ThinkingBudget::steps`] constructor
/// does) to bound by steps only — the mode the Phase 41f scaling-law
/// benchmark uses so accuracy depends on steps, not on host speed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingBudget {
    pub max_steps: u32,
    pub max_wall_ms: u64,
    /// Phase 11g VOC currency available for this deliberation.
    pub voc_credits: f32,
    pub used_steps: u32,
    pub used_wall_ms: u64,
    pub used_voc: f32,
}

impl Default for ThinkingBudget {
    /// Interactive default: 300 steps, 5 s wall clock, 300 VOC credits.
    fn default() -> Self {
        Self {
            max_steps: 300,
            max_wall_ms: 5_000,
            voc_credits: 300.0,
            used_steps: 0,
            used_wall_ms: 0,
            used_voc: 0.0,
        }
    }
}

impl ThinkingBudget {
    /// A budget bounded only by primitive step count.
    pub fn steps(max_steps: u32) -> Self {
        Self {
            max_steps,
            max_wall_ms: u64::MAX,
            voc_credits: f32::INFINITY,
            used_steps: 0,
            used_wall_ms: 0,
            used_voc: 0.0,
        }
    }

    /// Add a wall-clock ceiling (milliseconds).
    pub fn with_wall_ms(mut self, max_wall_ms: u64) -> Self {
        self.max_wall_ms = max_wall_ms;
        self
    }

    /// Add a VOC-credit ceiling (Phase 11g currency).
    pub fn with_voc_credits(mut self, voc_credits: f32) -> Self {
        self.voc_credits = voc_credits;
        self
    }

    /// Validate that the budget permits at least one step.
    pub fn validate(&self) -> ThinkingResult<()> {
        if self.max_steps == 0 {
            return Err(ThinkingError::InvalidBudget {
                reason: "max_steps = 0 — the loop could never run a primitive".into(),
            });
        }
        if self.voc_credits <= 0.0 {
            return Err(ThinkingError::InvalidBudget {
                reason: format!(
                    "voc_credits = {} — must be strictly positive (use f32::INFINITY to disable)",
                    self.voc_credits
                ),
            });
        }
        Ok(())
    }

    /// Charge one primitive step against the budget.
    pub fn charge(&mut self, voc_cost: f32, wall_ms: u64) {
        self.used_steps = self.used_steps.saturating_add(1);
        self.used_wall_ms = self.used_wall_ms.saturating_add(wall_ms);
        self.used_voc += voc_cost;
    }

    /// Which axis (if any) is exhausted?
    pub fn exhausted(&self) -> Option<BudgetResource> {
        if self.used_steps >= self.max_steps {
            Some(BudgetResource::Steps)
        } else if self.used_wall_ms >= self.max_wall_ms {
            Some(BudgetResource::WallClock)
        } else if self.used_voc >= self.voc_credits {
            Some(BudgetResource::VocCredits)
        } else {
            None
        }
    }

    /// Steps still available.
    pub fn remaining_steps(&self) -> u32 {
        self.max_steps.saturating_sub(self.used_steps)
    }

    /// Fraction of the step budget consumed, in `[0, 1]`.
    pub fn step_fraction_used(&self) -> f32 {
        if self.max_steps == 0 {
            return 1.0;
        }
        (self.used_steps as f32 / self.max_steps as f32).min(1.0)
    }

    /// Extend the step ceiling (used by the Phase 41e VOC controller when
    /// marginal value stays high near exhaustion). Capped by `global_cap`.
    pub fn extend_steps(&mut self, extra: u32, global_cap: u32) {
        self.max_steps = self.max_steps.saturating_add(extra).min(global_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_step_budget_is_invalid() {
        let b = ThinkingBudget::steps(0);
        assert!(b.validate().is_err());
    }

    #[test]
    fn steps_budget_ignores_wall_and_voc() {
        let mut b = ThinkingBudget::steps(2);
        b.validate().unwrap();
        b.charge(1_000.0, 1_000_000);
        assert_eq!(b.exhausted(), None);
        b.charge(1_000.0, 1_000_000);
        assert_eq!(b.exhausted(), Some(BudgetResource::Steps));
    }

    #[test]
    fn wall_clock_exhaustion_detected() {
        let mut b = ThinkingBudget::steps(100).with_wall_ms(10);
        b.charge(0.0, 11);
        assert_eq!(b.exhausted(), Some(BudgetResource::WallClock));
    }

    #[test]
    fn voc_exhaustion_detected() {
        let mut b = ThinkingBudget::steps(100).with_voc_credits(1.5);
        b.charge(1.0, 0);
        assert_eq!(b.exhausted(), None);
        b.charge(1.0, 0);
        assert_eq!(b.exhausted(), Some(BudgetResource::VocCredits));
    }

    #[test]
    fn extend_steps_respects_global_cap() {
        let mut b = ThinkingBudget::steps(10);
        b.extend_steps(100, 50);
        assert_eq!(b.max_steps, 50);
    }

    #[test]
    fn step_fraction_tracks_usage() {
        let mut b = ThinkingBudget::steps(4);
        assert_eq!(b.step_fraction_used(), 0.0);
        b.charge(0.0, 0);
        b.charge(0.0, 0);
        assert!((b.step_fraction_used() - 0.5).abs() < f32::EPSILON);
    }
}
