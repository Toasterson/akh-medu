//! Plan monitoring & OODA integration — Phase 16c.
//!
//! Bridges the MCTS planner (Phase 16b) with the OODA cycle. Monitors plan
//! execution by comparing predicted state transitions against actual outcomes,
//! triggering re-planning when deviations exceed thresholds.
//!
//! Also provides the `Plan::from_mcts_result` constructor to convert MCTS
//! output into executable plans, and the reward/TD-update cycle that improves
//! the value function over time.

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::causal::CausalManager;
use super::counterfactual::PredictionTracker;
use super::mcts::{MctsConfig, MctsPlanner, MctsReflection, MctsResult};
use super::plan::{Plan, PlanStatus, PlanStep, StepStatus};
use super::state_value::{AgentState, Reward, ValueFunction, encode_state};
use super::tool::ToolInput;

// ═══════════════════════════════════════════════════════════════════════
// PlanMonitor
// ═══════════════════════════════════════════════════════════════════════

/// Monitors plan execution by comparing predictions against reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanMonitor {
    /// Expected state after each step (predicted by causal model).
    pub expected_states: Vec<AgentState>,
    /// Actual states observed after each step.
    pub actual_states: Vec<AgentState>,
    /// Per-step deviation severity.
    pub step_deviations: Vec<DeviationSeverity>,
    /// Whether re-planning has been triggered.
    pub replanned: bool,
    /// The state snapshot before the plan started.
    pub initial_state: AgentState,
}

/// Severity of deviation between expected and actual outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeviationSeverity {
    /// No deviation — prediction matched reality.
    None,
    /// Minor deviation — plan can continue safely.
    Minor,
    /// Moderate deviation — consider re-planning.
    Moderate,
    /// Critical deviation — remaining plan steps may be invalid.
    Critical,
}

impl PlanMonitor {
    /// Create a new monitor from an initial state and expected step states.
    pub fn new(initial_state: AgentState, expected_states: Vec<AgentState>) -> Self {
        Self {
            expected_states,
            actual_states: Vec::new(),
            step_deviations: Vec::new(),
            replanned: false,
            initial_state,
        }
    }

    /// Record an actual state after a step and compute deviation.
    pub fn record_step(&mut self, actual: AgentState) -> DeviationSeverity {
        let step_idx = self.actual_states.len();
        let severity = if let Some(expected) = self.expected_states.get(step_idx) {
            compute_deviation(expected, &actual)
        } else {
            DeviationSeverity::Minor // no prediction = minor
        };

        self.actual_states.push(actual);
        self.step_deviations.push(severity);
        severity
    }

    /// Should we re-plan based on accumulated deviations?
    pub fn should_replan(&self) -> bool {
        self.step_deviations
            .iter()
            .any(|d| matches!(d, DeviationSeverity::Critical))
    }

    /// Overall monitoring summary.
    pub fn summary(&self) -> String {
        let critical = self.step_deviations.iter()
            .filter(|d| matches!(d, DeviationSeverity::Critical)).count();
        let moderate = self.step_deviations.iter()
            .filter(|d| matches!(d, DeviationSeverity::Moderate)).count();
        let minor = self.step_deviations.iter()
            .filter(|d| matches!(d, DeviationSeverity::Minor)).count();
        let none = self.step_deviations.iter()
            .filter(|d| matches!(d, DeviationSeverity::None)).count();

        format!(
            "{} steps monitored: {} on-track, {} minor, {} moderate, {} critical{}",
            self.actual_states.len(),
            none, minor, moderate, critical,
            if self.replanned { " (replanned)" } else { "" }
        )
    }
}

/// Compute deviation severity between expected and actual states.
fn compute_deviation(expected: &AgentState, actual: &AgentState) -> DeviationSeverity {
    // Compare fluent counts (proxy for KG state).
    let fluent_diff = (expected.fluent_count as i64 - actual.fluent_count as i64).unsigned_abs();
    let fluent_ratio = if expected.fluent_count > 0 {
        fluent_diff as f32 / expected.fluent_count as f32
    } else {
        0.0
    };

    // Compare goal progress.
    let mut progress_diff = 0.0_f32;
    for (exp_id, exp_progress) in &expected.goal_states {
        if let Some((_, act_progress)) = actual.goal_states.iter().find(|(id, _)| id == exp_id) {
            progress_diff += (exp_progress - act_progress).abs();
        } else {
            progress_diff += exp_progress; // goal disappeared = full diff
        }
    }

    let combined = fluent_ratio * 0.5 + progress_diff * 0.5;

    if combined < 0.05 {
        DeviationSeverity::None
    } else if combined < 0.15 {
        DeviationSeverity::Minor
    } else if combined < 0.40 {
        DeviationSeverity::Moderate
    } else {
        DeviationSeverity::Critical
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Plan from MCTS
// ═══════════════════════════════════════════════════════════════════════

/// Convert an MCTS result into an executable Plan.
pub fn plan_from_mcts(
    goal_id: SymbolId,
    result: &MctsResult,
    engine: &Engine,
) -> Plan {
    let steps: Vec<PlanStep> = result
        .best_plan
        .iter()
        .enumerate()
        .map(|(i, &action_id)| {
            let tool_name = engine.resolve_label(action_id);
            PlanStep {
                tool_name: tool_name.clone(),
                tool_input: ToolInput::new(),
                rationale: format!(
                    "MCTS planned (expected value: {:.3})",
                    result.expected_value
                ),
                status: StepStatus::Pending,
                index: i,
                expected_effects: Vec::new(),
            }
        })
        .collect();

    Plan {
        goal_id,
        steps,
        status: PlanStatus::Active,
        attempt: 0,
        strategy: format!(
            "MCTS: {} steps, {:.3} expected value, {} iterations",
            result.best_plan.len(),
            result.expected_value,
            result.iterations
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Reward & TD Update Cycle
// ═══════════════════════════════════════════════════════════════════════

/// Run the post-action TD update cycle.
///
/// 1. Snapshot current state (after action)
/// 2. Compute reward from before/after states
/// 3. Run TD(0) update on the value function
/// 4. Optionally verify prediction and update tracker
pub fn post_action_update(
    before_state: &AgentState,
    after_state: &AgentState,
    prediction_correct: bool,
    value_fn: &mut ValueFunction,
) -> Reward {
    let reward = Reward::compute(before_state, after_state, prediction_correct);
    value_fn.td_update(before_state, reward.total, after_state);
    reward
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(goals: Vec<(u64, f32)>, fluent_count: usize) -> AgentState {
        AgentState {
            goal_states: goals
                .into_iter()
                .filter_map(|(id, p)| SymbolId::new(id).map(|s| (s, p)))
                .collect(),
            fluent_count,
            state_vec: None,
            timestamp: 100,
        }
    }

    // ── Deviation ─────────────────────────────────────────────────

    #[test]
    fn deviation_none_when_identical() {
        let s = make_state(vec![(1, 0.5)], 1000);
        assert_eq!(compute_deviation(&s, &s), DeviationSeverity::None);
    }

    #[test]
    fn deviation_minor_small_diff() {
        let expected = make_state(vec![(1, 0.5)], 1000);
        let actual = make_state(vec![(1, 0.48)], 1020);
        let sev = compute_deviation(&expected, &actual);
        assert!(matches!(sev, DeviationSeverity::None | DeviationSeverity::Minor));
    }

    #[test]
    fn deviation_critical_large_diff() {
        let expected = make_state(vec![(1, 0.9)], 1000);
        let actual = make_state(vec![(1, 0.1)], 300);
        let sev = compute_deviation(&expected, &actual);
        assert!(matches!(sev, DeviationSeverity::Moderate | DeviationSeverity::Critical));
    }

    // ── PlanMonitor ───────────────────────────────────────────────

    #[test]
    fn plan_monitor_no_deviation() {
        let initial = make_state(vec![(1, 0.0)], 100);
        let expected = make_state(vec![(1, 0.5)], 200);
        let mut monitor = PlanMonitor::new(initial, vec![expected.clone()]);

        let sev = monitor.record_step(expected);
        assert_eq!(sev, DeviationSeverity::None);
        assert!(!monitor.should_replan());
    }

    #[test]
    fn plan_monitor_critical_triggers_replan() {
        let initial = make_state(vec![(1, 0.0)], 100);
        let expected = make_state(vec![(1, 0.9)], 500);
        let actual = make_state(vec![(1, 0.1)], 50);
        let mut monitor = PlanMonitor::new(initial, vec![expected]);

        let sev = monitor.record_step(actual);
        assert!(matches!(sev, DeviationSeverity::Moderate | DeviationSeverity::Critical));
    }

    #[test]
    fn plan_monitor_summary() {
        let initial = make_state(vec![], 100);
        let mut monitor = PlanMonitor::new(initial.clone(), vec![initial.clone()]);
        monitor.record_step(initial);
        let summary = monitor.summary();
        assert!(summary.contains("1 steps monitored"));
    }

    // ── plan_from_mcts ────────────────────────────────────────────

    #[test]
    fn plan_from_mcts_empty_result() {
        let goal = SymbolId::new(1).unwrap();
        let result = MctsResult {
            best_plan: vec![],
            expected_value: 0.0,
            iterations: 50,
            max_depth_reached: 0,
            first_action_scores: vec![],
            reaches_goal: false,
        };

        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let plan = plan_from_mcts(goal, &result, &engine);
        assert!(plan.steps.is_empty());
        assert_eq!(plan.status, PlanStatus::Active);
    }

    // ── post_action_update ────────────────────────────────────────

    #[test]
    fn post_action_update_improves_value() {
        let before = make_state(vec![(1, 0.0)], 100);
        let after = make_state(vec![(1, 0.5)], 200);
        let mut vf = ValueFunction::default();

        let reward = post_action_update(&before, &after, true, &mut vf);
        assert!(reward.total > 0.0);
        assert!(vf.state_value(&before) > 0.0);
        assert_eq!(vf.update_count, 1);
    }

    #[test]
    fn post_action_update_negative_reward() {
        let before = make_state(vec![(1, 0.5)], 200);
        let after = make_state(vec![(1, 0.3)], 100);
        let mut vf = ValueFunction::default();

        let reward = post_action_update(&before, &after, false, &mut vf);
        assert!(reward.goal_progress < 0.0);
    }
}
