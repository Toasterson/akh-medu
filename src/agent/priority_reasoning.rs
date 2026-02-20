//! Priority reasoning via value-based argumentation.
//!
//! Each goal's priority is justified by pro/con arguments promoting values
//! (timeliness, thoroughness, efficiency, accuracy), weighted by an audience
//! (operational mode). This grounds priorities in transparent reasoning
//! rather than magic numbers.
//!
//! Based on Dung argumentation frameworks and Value-based Argumentation
//! Frameworks (VAFs) from Bench-Capon.

use serde::{Deserialize, Serialize};

use super::goal::{Goal, GoalSource, GoalStatus};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// An abstract value that an argument promotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Value {
    /// Urgency: blocks others, deadlines, time-sensitive.
    Timeliness,
    /// Knowledge quality: gap coverage, learning, depth.
    Thoroughness,
    /// Low effort relative to impact, resource conservation.
    Efficiency,
    /// Contradiction resolution, confidence improvement.
    Accuracy,
}

impl Value {
    /// All values in canonical order.
    pub const ALL: [Value; 4] = [
        Value::Timeliness,
        Value::Thoroughness,
        Value::Efficiency,
        Value::Accuracy,
    ];

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Timeliness => "timeliness",
            Self::Thoroughness => "thoroughness",
            Self::Efficiency => "efficiency",
            Self::Accuracy => "accuracy",
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Audience
// ---------------------------------------------------------------------------

/// An audience represents a value ordering (operational mode).
///
/// The ordering determines how much weight each value carries.
/// Position 0 is most preferred, position 3 is least.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Audience {
    /// Human-readable name for this audience.
    pub name: String,
    /// Values ordered from most preferred to least preferred.
    pub ordering: Vec<Value>,
}

impl Audience {
    /// Exploration mode: thoroughness > accuracy > efficiency > timeliness.
    pub fn exploration() -> Self {
        Self {
            name: "exploration".into(),
            ordering: vec![
                Value::Thoroughness,
                Value::Accuracy,
                Value::Efficiency,
                Value::Timeliness,
            ],
        }
    }

    /// Deadline mode: timeliness > efficiency > accuracy > thoroughness.
    pub fn deadline() -> Self {
        Self {
            name: "deadline".into(),
            ordering: vec![
                Value::Timeliness,
                Value::Efficiency,
                Value::Accuracy,
                Value::Thoroughness,
            ],
        }
    }

    /// Quality mode: accuracy > thoroughness > timeliness > efficiency.
    pub fn quality() -> Self {
        Self {
            name: "quality".into(),
            ordering: vec![
                Value::Accuracy,
                Value::Thoroughness,
                Value::Timeliness,
                Value::Efficiency,
            ],
        }
    }

    /// Weight of a value in this audience's ordering.
    ///
    /// Exponential decay: position 0 → 8.0, position 1 → 4.0,
    /// position 2 → 2.0, position 3 → 1.0.
    pub fn weight_of(&self, value: Value) -> f64 {
        let pos = self
            .ordering
            .iter()
            .position(|v| *v == value)
            .unwrap_or(self.ordering.len());
        match pos {
            0 => 8.0,
            1 => 4.0,
            2 => 2.0,
            3 => 1.0,
            _ => 0.5, // unlisted value
        }
    }

    /// Whether an attacker's value defeats a target's value.
    ///
    /// Attack succeeds if attacker value is preferred or equal in the ordering.
    pub fn attack_succeeds(&self, attacker_val: Value, target_val: Value) -> bool {
        let attacker_pos = self
            .ordering
            .iter()
            .position(|v| *v == attacker_val)
            .unwrap_or(usize::MAX);
        let target_pos = self
            .ordering
            .iter()
            .position(|v| *v == target_val)
            .unwrap_or(usize::MAX);
        attacker_pos <= target_pos
    }
}

// ---------------------------------------------------------------------------
// PriorityArgument
// ---------------------------------------------------------------------------

/// Direction of an argument relative to priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorityDirection {
    /// Argues for increasing priority.
    Pro,
    /// Argues for decreasing priority.
    Con,
}

/// The source/evidence behind a priority argument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArgumentSource {
    /// This goal blocks N other active goals.
    BlocksOthers { blocked_count: usize },
    /// This goal is blocked by N incomplete goals.
    IsBlocked { blocker_count: usize },
    /// A motivational drive exceeded its threshold.
    DriveExceeded { drive: String, strength: f32 },
    /// The goal has been stalled for too long.
    Stalled { cycles_stalled: u64 },
    /// The goal made progress recently.
    MakingProgress { recent_progress_cycle: u64 },
    /// This goal fills a knowledge gap.
    FillsGap { severity: f32 },
    /// This goal resolves a contradiction.
    ResolvesContradiction,
    /// This goal is estimated to require few cycles.
    LowEffort { estimated_cycles: u32 },
    /// This goal is estimated to require many cycles.
    HighEffort { estimated_cycles: u32 },
    /// This goal is a child of a high-priority parent.
    ChildOfHighPriority { parent_priority: u8 },
    /// Most siblings of this goal are already complete.
    NearCompletion {
        completed_siblings: usize,
        total_siblings: usize,
    },
    /// The user explicitly set this goal's priority.
    UserSet { original_priority: u8 },
}

/// A single argument for or against a goal's priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityArgument {
    /// Whether this argues for higher or lower priority.
    pub direction: PriorityDirection,
    /// Which value this argument promotes.
    pub value: Value,
    /// What evidence supports this argument.
    pub source: ArgumentSource,
    /// Base weight before audience modulation, in [0.0, 1.0].
    pub base_weight: f64,
    /// Human-readable explanation.
    pub reasoning: String,
}

impl PriorityArgument {
    /// Effective weight after audience modulation.
    pub fn effective_weight(&self, audience: &Audience) -> f64 {
        self.base_weight * audience.weight_of(self.value)
    }
}

// ---------------------------------------------------------------------------
// PriorityVerdict
// ---------------------------------------------------------------------------

/// The computed priority verdict for a single goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityVerdict {
    /// The final computed priority (0–255).
    pub computed_priority: u8,
    /// Arguments in favor of higher priority.
    pub pro: Vec<PriorityArgument>,
    /// Arguments in favor of lower priority.
    pub con: Vec<PriorityArgument>,
    /// Net score before normalization.
    pub net_score: f64,
    /// Which audience was used for weighting.
    pub audience_name: String,
    /// Human-readable summary.
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// Argument generation
// ---------------------------------------------------------------------------

/// Map a drive name to its corresponding value.
fn drive_to_value(drive: &str) -> Value {
    match drive {
        "curiosity" => Value::Thoroughness,
        "coherence" => Value::Accuracy,
        "completeness" => Value::Thoroughness,
        "efficiency" => Value::Efficiency,
        _ => Value::Thoroughness,
    }
}

/// Count how many active goals are blocked by a given goal.
fn count_goals_blocked_by(goal_id: SymbolId, all_goals: &[Goal]) -> usize {
    all_goals
        .iter()
        .filter(|g| {
            matches!(g.status, GoalStatus::Active) && g.blocked_by.contains(&goal_id)
        })
        .count()
}

/// Generate arguments for a goal's priority based on its current state.
pub fn generate_arguments(
    goal: &Goal,
    all_goals: &[Goal],
    current_cycle: u64,
    stall_threshold: u32,
) -> Vec<PriorityArgument> {
    let mut args = Vec::new();

    // ── BlocksOthers: Pro/Timeliness ──
    let blocked_count = count_goals_blocked_by(goal.symbol_id, all_goals);
    if blocked_count > 0 {
        let weight = (0.3 + 0.15 * blocked_count as f64).min(1.0);
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Timeliness,
            source: ArgumentSource::BlocksOthers { blocked_count },
            base_weight: weight,
            reasoning: format!("Blocks {blocked_count} other active goal(s)."),
        });
    }

    // ── IsBlocked: Con/Efficiency ──
    let blocker_count = goal
        .blocked_by
        .iter()
        .filter(|bid| {
            all_goals
                .iter()
                .find(|g| g.symbol_id == **bid)
                .is_some_and(|g| {
                    !matches!(g.status, GoalStatus::Completed | GoalStatus::Failed { .. })
                })
        })
        .count();
    if blocker_count > 0 {
        args.push(PriorityArgument {
            direction: PriorityDirection::Con,
            value: Value::Efficiency,
            source: ArgumentSource::IsBlocked { blocker_count },
            base_weight: 0.8,
            reasoning: format!("Blocked by {blocker_count} incomplete goal(s)."),
        });
    }

    // ── DriveExceeded: Pro/mapped value ──
    if let Some(GoalSource::DriveExceeded { drive, strength }) = &goal.source {
        let value = drive_to_value(drive);
        let weight = (*strength as f64 * 0.8).min(1.0);
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value,
            source: ArgumentSource::DriveExceeded {
                drive: drive.clone(),
                strength: *strength,
            },
            base_weight: weight,
            reasoning: format!("Drive \"{drive}\" exceeded threshold (strength {strength:.2})."),
        });
    }

    // ── Stalled: Con/Efficiency ──
    let cycles_since_progress = current_cycle.saturating_sub(goal.last_progress_cycle);
    if goal.cycles_worked >= stall_threshold
        && cycles_since_progress >= stall_threshold as u64
    {
        let over = cycles_since_progress.saturating_sub(stall_threshold as u64);
        let weight = (0.4 + 0.05 * over as f64).min(0.9);
        args.push(PriorityArgument {
            direction: PriorityDirection::Con,
            value: Value::Efficiency,
            source: ArgumentSource::Stalled {
                cycles_stalled: cycles_since_progress,
            },
            base_weight: weight,
            reasoning: format!(
                "Stalled for {cycles_since_progress} cycles (threshold: {stall_threshold})."
            ),
        });
    }

    // ── MakingProgress: Pro/Efficiency ──
    if goal.last_progress_cycle > 0 && current_cycle.saturating_sub(goal.last_progress_cycle) <= 3
    {
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Efficiency,
            source: ArgumentSource::MakingProgress {
                recent_progress_cycle: goal.last_progress_cycle,
            },
            base_weight: 0.6,
            reasoning: format!(
                "Made progress at cycle {} (current: {current_cycle}).",
                goal.last_progress_cycle
            ),
        });
    }

    // ── FillsGap: Pro/Thoroughness ──
    if let Some(GoalSource::GapDetection { severity, .. }) = &goal.source {
        let weight = (*severity as f64 * 0.7).min(1.0);
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Thoroughness,
            source: ArgumentSource::FillsGap {
                severity: *severity,
            },
            base_weight: weight,
            reasoning: format!("Fills knowledge gap (severity: {severity:.2})."),
        });
    }

    // ── ResolvesContradiction: Pro/Accuracy ──
    if matches!(
        &goal.source,
        Some(GoalSource::ContradictionDetected { .. })
    ) {
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Accuracy,
            source: ArgumentSource::ResolvesContradiction,
            base_weight: 0.7,
            reasoning: "Resolves a detected contradiction.".into(),
        });
    }

    // ── ChildOfHighPriority: Pro/Timeliness ──
    if let Some(parent_id) = goal.parent {
        if let Some(parent) = all_goals.iter().find(|g| g.symbol_id == parent_id) {
            let parent_p = parent.computed_priority();
            if parent_p > 200 {
                let weight = parent_p as f64 / 255.0 * 0.6;
                args.push(PriorityArgument {
                    direction: PriorityDirection::Pro,
                    value: Value::Timeliness,
                    source: ArgumentSource::ChildOfHighPriority {
                        parent_priority: parent_p,
                    },
                    base_weight: weight,
                    reasoning: format!(
                        "Child of high-priority parent (priority {parent_p})."
                    ),
                });
            }
        }
    }

    // ── NearCompletion: Pro/Timeliness ──
    if let Some(parent_id) = goal.parent {
        if let Some(parent) = all_goals.iter().find(|g| g.symbol_id == parent_id) {
            if !parent.children.is_empty() {
                let total = parent.children.len();
                let completed = parent
                    .children
                    .iter()
                    .filter(|cid| {
                        all_goals
                            .iter()
                            .find(|g| g.symbol_id == **cid)
                            .is_some_and(|g| matches!(g.status, GoalStatus::Completed))
                    })
                    .count();
                if total > 1 && completed * 4 >= total * 3 {
                    // >75% siblings done
                    args.push(PriorityArgument {
                        direction: PriorityDirection::Pro,
                        value: Value::Timeliness,
                        source: ArgumentSource::NearCompletion {
                            completed_siblings: completed,
                            total_siblings: total,
                        },
                        base_weight: 0.7,
                        reasoning: format!(
                            "Near completion: {completed}/{total} siblings done."
                        ),
                    });
                }
            }
        }
    }

    // ── UserSet: Pro/Timeliness ──
    if goal.source.is_none() && goal.priority > 0 {
        args.push(PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Timeliness,
            source: ArgumentSource::UserSet {
                original_priority: goal.priority,
            },
            base_weight: goal.priority as f64 / 255.0,
            reasoning: format!(
                "User-set priority: {} (base weight {:.2}).",
                goal.priority,
                goal.priority as f64 / 255.0
            ),
        });
    }

    args
}

// ---------------------------------------------------------------------------
// Scoring formula
// ---------------------------------------------------------------------------

/// Influence range: arguments can shift priority by at most this many units.
const INFLUENCE_RANGE: f64 = 80.0;

/// Compute a priority verdict for a single goal.
pub fn compute_priority(
    goal: &Goal,
    arguments: Vec<PriorityArgument>,
    audience: &Audience,
) -> PriorityVerdict {
    let mut pro = Vec::new();
    let mut con = Vec::new();

    for arg in arguments {
        match arg.direction {
            PriorityDirection::Pro => pro.push(arg),
            PriorityDirection::Con => con.push(arg),
        }
    }

    let pro_sum: f64 = pro.iter().map(|a| a.effective_weight(audience)).sum();
    let con_sum: f64 = con.iter().map(|a| a.effective_weight(audience)).sum();
    let net_score = pro_sum - con_sum;

    // Normalize to [-1.0, 1.0] — division by 32.0 gives a reasonable dynamic range.
    let normalized = (net_score / 32.0).clamp(-1.0, 1.0);

    let shift = normalized * INFLUENCE_RANGE;
    let raw = goal.priority as f64 + shift;
    let computed_priority = raw.round().clamp(0.0, 255.0) as u8;

    let reasoning = format!(
        "Audience \"{}\": {} pro ({:.1}), {} con ({:.1}), net={:.2}, shift={:+.0} → {}",
        audience.name,
        pro.len(),
        pro_sum,
        con.len(),
        con_sum,
        net_score,
        shift,
        computed_priority,
    );

    PriorityVerdict {
        computed_priority,
        pro,
        con,
        net_score,
        audience_name: audience.name.clone(),
        reasoning,
    }
}

/// Reprioritize all active goals using argumentation.
///
/// Returns a list of `(goal_id, old_priority, new_priority, verdict)`.
pub fn reprioritize_all(
    goals: &[Goal],
    current_cycle: u64,
    stall_threshold: u32,
    audience: &Audience,
) -> Vec<(SymbolId, u8, u8, PriorityVerdict)> {
    goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .map(|goal| {
            let args = generate_arguments(goal, goals, current_cycle, stall_threshold);
            let verdict = compute_priority(goal, args, audience);
            let old = goal.priority;
            let new = verdict.computed_priority;
            (goal.symbol_id, old, new, verdict)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolId;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn make_goal(id: u64, priority: u8) -> Goal {
        Goal {
            symbol_id: sym(id),
            description: format!("goal-{id}"),
            status: GoalStatus::Active,
            priority,
            success_criteria: String::new(),
            parent: None,
            children: Vec::new(),
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            blocked_by: Vec::new(),
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
        }
    }

    // ── Value tests ──

    #[test]
    fn value_display() {
        assert_eq!(Value::Timeliness.to_string(), "timeliness");
        assert_eq!(Value::Thoroughness.to_string(), "thoroughness");
        assert_eq!(Value::Efficiency.to_string(), "efficiency");
        assert_eq!(Value::Accuracy.to_string(), "accuracy");
    }

    // ── Audience tests ──

    #[test]
    fn audience_exploration_ordering() {
        let a = Audience::exploration();
        assert_eq!(a.ordering[0], Value::Thoroughness);
        assert_eq!(a.ordering[3], Value::Timeliness);
    }

    #[test]
    fn audience_deadline_ordering() {
        let a = Audience::deadline();
        assert_eq!(a.ordering[0], Value::Timeliness);
    }

    #[test]
    fn audience_quality_ordering() {
        let a = Audience::quality();
        assert_eq!(a.ordering[0], Value::Accuracy);
    }

    #[test]
    fn weight_of_most_preferred() {
        let a = Audience::exploration();
        assert!((a.weight_of(Value::Thoroughness) - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn weight_of_least_preferred() {
        let a = Audience::exploration();
        assert!((a.weight_of(Value::Timeliness) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn attack_succeeds_preferred() {
        let a = Audience::exploration();
        // Thoroughness (pos 0) attacks Timeliness (pos 3)
        assert!(a.attack_succeeds(Value::Thoroughness, Value::Timeliness));
    }

    #[test]
    fn attack_fails_when_less_preferred() {
        let a = Audience::exploration();
        // Timeliness (pos 3) attacks Thoroughness (pos 0)
        assert!(!a.attack_succeeds(Value::Timeliness, Value::Thoroughness));
    }

    // ── Argument generation tests ──

    #[test]
    fn blocks_others_generates_pro_timeliness() {
        let g1 = make_goal(1, 128);
        let mut g2 = make_goal(2, 128);
        g2.blocked_by = vec![sym(1)];

        let args = generate_arguments(&g1, &[g1.clone(), g2], 10, 5);
        let pro = args
            .iter()
            .find(|a| matches!(a.source, ArgumentSource::BlocksOthers { .. }));
        assert!(pro.is_some());
        let p = pro.unwrap();
        assert_eq!(p.direction, PriorityDirection::Pro);
        assert_eq!(p.value, Value::Timeliness);
    }

    #[test]
    fn stalled_generates_con_efficiency() {
        let mut g = make_goal(1, 128);
        g.cycles_worked = 10;
        g.last_progress_cycle = 0;

        let args = generate_arguments(&g, &[g.clone()], 15, 5);
        let con = args
            .iter()
            .find(|a| matches!(a.source, ArgumentSource::Stalled { .. }));
        assert!(con.is_some());
        let c = con.unwrap();
        assert_eq!(c.direction, PriorityDirection::Con);
        assert_eq!(c.value, Value::Efficiency);
    }

    #[test]
    fn gap_generates_pro_thoroughness() {
        let mut g = make_goal(1, 128);
        g.source = Some(GoalSource::GapDetection {
            gap_entity: sym(99),
            gap_kind: "missing_def".into(),
            severity: 0.8,
        });

        let args = generate_arguments(&g, &[g.clone()], 10, 5);
        let pro = args
            .iter()
            .find(|a| matches!(a.source, ArgumentSource::FillsGap { .. }));
        assert!(pro.is_some());
        assert_eq!(pro.unwrap().value, Value::Thoroughness);
    }

    #[test]
    fn user_set_generates_pro_timeliness() {
        let g = make_goal(1, 200);
        let args = generate_arguments(&g, &[g.clone()], 10, 5);
        let user_arg = args
            .iter()
            .find(|a| matches!(a.source, ArgumentSource::UserSet { .. }));
        assert!(user_arg.is_some());
        assert_eq!(user_arg.unwrap().value, Value::Timeliness);
    }

    // ── Scoring tests ──

    #[test]
    fn pure_pro_raises_priority() {
        let g = make_goal(1, 128);
        let args = vec![PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Timeliness,
            source: ArgumentSource::BlocksOthers { blocked_count: 3 },
            base_weight: 0.8,
            reasoning: "test".into(),
        }];
        let audience = Audience::deadline(); // Timeliness is most preferred
        let verdict = compute_priority(&g, args, &audience);
        assert!(verdict.computed_priority > 128);
    }

    #[test]
    fn pure_con_lowers_priority() {
        let g = make_goal(1, 128);
        let args = vec![PriorityArgument {
            direction: PriorityDirection::Con,
            value: Value::Efficiency,
            source: ArgumentSource::Stalled { cycles_stalled: 20 },
            base_weight: 0.8,
            reasoning: "test".into(),
        }];
        let audience = Audience::exploration();
        let verdict = compute_priority(&g, args, &audience);
        assert!(verdict.computed_priority < 128);
    }

    #[test]
    fn balanced_preserves_approximately() {
        let g = make_goal(1, 128);
        let args = vec![
            PriorityArgument {
                direction: PriorityDirection::Pro,
                value: Value::Timeliness,
                source: ArgumentSource::BlocksOthers { blocked_count: 1 },
                base_weight: 0.5,
                reasoning: "pro".into(),
            },
            PriorityArgument {
                direction: PriorityDirection::Con,
                value: Value::Timeliness,
                source: ArgumentSource::Stalled { cycles_stalled: 10 },
                base_weight: 0.5,
                reasoning: "con".into(),
            },
        ];
        let audience = Audience::exploration();
        let verdict = compute_priority(&g, args, &audience);
        // Both same value, same weight → net ~0 → priority near base
        assert!((verdict.computed_priority as i16 - 128).unsigned_abs() <= 5);
    }

    #[test]
    fn audience_modulates_scoring() {
        let g = make_goal(1, 128);
        let args_fn = || {
            vec![PriorityArgument {
                direction: PriorityDirection::Pro,
                value: Value::Thoroughness,
                source: ArgumentSource::FillsGap { severity: 0.9 },
                base_weight: 0.7,
                reasoning: "gap".into(),
            }]
        };

        // Exploration audience favors thoroughness (weight 8.0)
        let explore = compute_priority(&g, args_fn(), &Audience::exploration());
        // Deadline audience has thoroughness as last (weight 1.0)
        let deadline = compute_priority(&g, args_fn(), &Audience::deadline());

        assert!(explore.computed_priority > deadline.computed_priority);
    }

    #[test]
    fn clamped_to_bounds() {
        // Priority 0 with strong con should not go negative
        let g = make_goal(1, 0);
        let args = vec![PriorityArgument {
            direction: PriorityDirection::Con,
            value: Value::Efficiency,
            source: ArgumentSource::Stalled { cycles_stalled: 100 },
            base_weight: 1.0,
            reasoning: "max con".into(),
        }];
        let verdict = compute_priority(&g, args, &Audience::exploration());
        assert_eq!(verdict.computed_priority, 0);

        // Priority 255 with strong pro should not exceed 255
        let g2 = make_goal(2, 255);
        let args2 = vec![PriorityArgument {
            direction: PriorityDirection::Pro,
            value: Value::Timeliness,
            source: ArgumentSource::BlocksOthers { blocked_count: 10 },
            base_weight: 1.0,
            reasoning: "max pro".into(),
        }];
        let verdict2 = compute_priority(&g2, args2, &Audience::deadline());
        assert_eq!(verdict2.computed_priority, 255);
    }

    #[test]
    fn influence_limited_to_range() {
        let g = make_goal(1, 128);
        // Massive pro arguments
        let args: Vec<PriorityArgument> = (0..20)
            .map(|i| PriorityArgument {
                direction: PriorityDirection::Pro,
                value: Value::Timeliness,
                source: ArgumentSource::BlocksOthers { blocked_count: i },
                base_weight: 1.0,
                reasoning: format!("pro-{i}"),
            })
            .collect();
        let verdict = compute_priority(&g, args, &Audience::deadline());
        // Max shift is +80 from base 128 = 208
        assert!(verdict.computed_priority <= 208);
    }

    #[test]
    fn empty_arguments_returns_base() {
        let g = make_goal(1, 128);
        let verdict = compute_priority(&g, vec![], &Audience::exploration());
        assert_eq!(verdict.computed_priority, 128);
    }

    #[test]
    fn reasoning_contains_audience_name() {
        let g = make_goal(1, 128);
        let verdict = compute_priority(&g, vec![], &Audience::deadline());
        assert!(verdict.reasoning.contains("deadline"));
    }

    // ── Batch reprioritize test ──

    #[test]
    fn reprioritize_all_processes_active_goals() {
        let g1 = make_goal(1, 128);
        let mut g2 = make_goal(2, 200);
        g2.status = GoalStatus::Completed;
        let g3 = make_goal(3, 100);

        let results = reprioritize_all(&[g1, g2, g3], 10, 5, &Audience::exploration());
        // Only active goals processed (g1, g3; g2 is completed)
        assert_eq!(results.len(), 2);
    }
}
