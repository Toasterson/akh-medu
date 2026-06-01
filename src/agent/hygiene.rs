//! Goal queue hygiene: retires stale goals, deduplicates identical intents,
//! and enforces max child-goal depth to prevent OODA loop paralysis.
//!
//! The hygiene pass runs at the beginning of each agent cycle (every
//! `run_every_n_cycles` cycles) and performs three sub-passes in order:
//!
//! 1. **Staleness retirement** — cancels goals that have been idle (no progress)
//!    for more than `max_idle_cycles` cycles.
//! 2. **Deduplication** — collapses goals with highly overlapping descriptions
//!    into a single primary goal; extras are marked `Duplicate`.
//! 3. **Child-depth cap** — cancels goals whose nesting depth exceeds
//!    `max_child_depth` to stop recursive explosion.
//!
//! Protected goals (never auto-cancelled):
//! * `GoalJustification::UserRequested` (entrenchment 3)
//! * Goals still blocking another non-terminal goal (HTN dependencies)

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::agent::AgentPredicates;
use super::goal::{Goal, GoalJustification, GoalStatus, update_goal_status};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the goal hygiene pass.
#[derive(Debug, Clone)]
pub struct HygieneConfig {
    /// Run the hygiene pass every N cycles (0 = disabled). Default: 10.
    pub run_every_n_cycles: u64,
    /// Cancel goals that have been idle for more than this many cycles. Default: 20.
    pub max_idle_cycles: u64,
    /// Maximum child-goal nesting depth. Goals deeper than this are cancelled. Default: 3.
    pub max_child_depth: u32,
    /// Word-overlap ratio threshold for considering two goals duplicates. Default: 0.75.
    pub dedup_overlap_threshold: f32,
    /// Minimum number of active + pending goals before deduplication runs.
    /// Avoids churn when the queue is already small. Default: 5.
    pub dedup_min_queue_size: usize,
}

impl Default for HygieneConfig {
    fn default() -> Self {
        Self {
            run_every_n_cycles: 10,
            max_idle_cycles: 20,
            max_child_depth: 3,
            dedup_overlap_threshold: 0.75,
            dedup_min_queue_size: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Summary of what the hygiene pass changed.
#[derive(Debug, Clone, Default)]
pub struct HygieneResult {
    /// Goals cancelled because they were idle too long.
    pub cancelled_stale: usize,
    /// Goals cancelled because they exceeded the child depth limit.
    pub cancelled_deep: usize,
    /// Goals marked as duplicates of a primary goal.
    pub deduplicated: usize,
    /// Total live goals before the pass.
    pub total_before: usize,
    /// Total live goals after the pass (Active + Pending only).
    pub total_after: usize,
}

// ---------------------------------------------------------------------------
// Word-overlap similarity
// ---------------------------------------------------------------------------

/// Normalise a goal description: lowercase, strip non-alphanumeric characters,
/// return sorted word set so word order doesn't matter.
fn normalise_words(text: &str) -> std::collections::HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(String::from)
        .collect()
}

/// Jaccard similarity over word sets.
fn word_overlap(a: &str, b: &str) -> f32 {
    let wa = normalise_words(a);
    let wb = normalise_words(b);
    if wa.is_empty() && wb.is_empty() {
        return 1.0;
    }
    let intersection = wa.intersection(&wb).count();
    let union = wa.union(&wb).count();
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

// ---------------------------------------------------------------------------
// Protection predicates
// ---------------------------------------------------------------------------

/// Return `true` if the goal should never be auto-cancelled.
fn is_protected(goal: &Goal, all_goals: &[Goal]) -> bool {
    // User-requested goals are sacred.
    if matches!(goal.justification, Some(GoalJustification::UserRequested)) {
        return true;
    }

    // Goals that are blockers for at least one live (non-terminal) goal.
    let is_blocker = all_goals.iter().any(|g| {
        let live = matches!(g.status, GoalStatus::Active | GoalStatus::Pending);
        live && g.blocked_by.contains(&goal.symbol_id)
    });
    if is_blocker {
        return true;
    }

    false
}

/// Return `true` if the goal status is live (active or pending — eligible for hygiene).
fn is_live(status: &GoalStatus) -> bool {
    matches!(status, GoalStatus::Active | GoalStatus::Pending)
}

// ---------------------------------------------------------------------------
// Sub-pass 1: Staleness retirement
// ---------------------------------------------------------------------------

/// Identify goals that should be cancelled for staleness.
///
/// `session_start_cycle` is the cycle count at the start of the current session.
/// Goals whose `last_progress_cycle` is 0 (never recorded progress, which includes
/// all goals restored from a previous session that did not persist cycle metadata)
/// are measured from `session_start_cycle` instead of cycle 0, so a session restart
/// does not immediately mark every restored goal as stale.
///
/// Returns the SymbolIds of goals to cancel and the reason string.
fn find_stale_goals(
    goals: &[Goal],
    current_cycle: u64,
    session_start_cycle: u64,
    max_idle: u64,
) -> Vec<(SymbolId, String)> {
    let mut stale = Vec::new();

    for goal in goals {
        if !is_live(&goal.status) {
            continue;
        }
        if is_protected(goal, goals) {
            continue;
        }

        let idle_cycles = if goal.last_progress_cycle == 0 {
            // Never recorded progress (or restored from a session without cycle
            // metadata).  Measure idle time from session start so we don't
            // immediately cancel goals that have simply not been worked on yet
            // in the current session.
            current_cycle.saturating_sub(session_start_cycle)
        } else {
            current_cycle.saturating_sub(goal.last_progress_cycle)
        };

        if idle_cycles >= max_idle {
            let reason = format!(
                "idle for {idle_cycles} cycles (threshold: {max_idle})"
            );
            stale.push((goal.symbol_id, reason));
        }
    }

    stale
}

// ---------------------------------------------------------------------------
// Sub-pass 2: Deduplication
// ---------------------------------------------------------------------------

/// Identify duplicate goals: for each pair of live goals whose descriptions
/// overlap above the threshold, mark the lower-priority one as a duplicate
/// of the higher-priority one.
///
/// Returns `(duplicate_id, primary_id)` pairs.
fn find_duplicate_goals(goals: &[Goal], threshold: f32) -> Vec<(SymbolId, SymbolId)> {
    let live: Vec<&Goal> = goals.iter().filter(|g| is_live(&g.status)).collect();
    let n = live.len();

    // Track which indices are already marked as duplicates in this pass.
    let mut is_dup = vec![false; n];
    let mut pairs = Vec::new();

    for i in 0..n {
        if is_dup[i] {
            continue;
        }
        for j in (i + 1)..n {
            if is_dup[j] {
                continue;
            }
            let sim = word_overlap(&live[i].description, &live[j].description);
            if sim >= threshold {
                // Keep the higher-priority goal (i or j).
                let (primary_idx, dup_idx) =
                    if live[i].computed_priority() >= live[j].computed_priority() {
                        (i, j)
                    } else {
                        (j, i)
                    };
                // Never mark a protected goal as duplicate.
                if is_protected(live[dup_idx], goals) {
                    continue;
                }
                is_dup[dup_idx] = true;
                pairs.push((live[dup_idx].symbol_id, live[primary_idx].symbol_id));
            }
        }
    }

    pairs
}

// ---------------------------------------------------------------------------
// Sub-pass 3: Child depth cap
// ---------------------------------------------------------------------------

/// Compute the nesting depth of `goal_id` by following parent links.
/// Returns 0 for root goals (no parent), 1 for direct children, etc.
fn goal_depth(goal_id: SymbolId, goals: &[Goal]) -> u32 {
    let mut depth = 0u32;
    let mut current = goal_id;
    // Guard against cycles.
    let mut visited = std::collections::HashSet::new();
    while let Some(g) = goals.iter().find(|g| g.symbol_id == current) {
        if !visited.insert(current) {
            break;
        }
        if let Some(parent_id) = g.parent {
            depth += 1;
            current = parent_id;
        } else {
            break;
        }
    }
    depth
}

/// Identify goals that are too deeply nested.
fn find_deep_goals(goals: &[Goal], max_depth: u32) -> Vec<(SymbolId, String)> {
    let mut deep = Vec::new();
    for goal in goals {
        if !is_live(&goal.status) {
            continue;
        }
        if is_protected(goal, goals) {
            continue;
        }
        let depth = goal_depth(goal.symbol_id, goals);
        if depth > max_depth {
            let reason = format!(
                "child depth {depth} exceeds max {max_depth}"
            );
            deep.push((goal.symbol_id, reason));
        }
    }
    deep
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the full hygiene pass on the agent's goal list.
///
/// `session_start_cycle` is the cycle count at the start of the current session.
/// It anchors staleness detection so goals restored from a previous session are
/// not immediately retired; they get a fresh idle window from session start.
///
/// Applies changes directly via `update_goal_status` so the KG stays in sync.
pub fn run_hygiene_pass(
    goals: &mut [Goal],
    engine: &Engine,
    predicates: &AgentPredicates,
    current_cycle: u64,
    session_start_cycle: u64,
    config: &HygieneConfig,
) -> HygieneResult {
    let total_before = goals
        .iter()
        .filter(|g| is_live(&g.status))
        .count();

    let mut result = HygieneResult {
        total_before,
        ..Default::default()
    };

    // ── Sub-pass 1: Staleness ────────────────────────────────────────────
    let stale = find_stale_goals(goals, current_cycle, session_start_cycle, config.max_idle_cycles);
    for (gid, reason) in &stale {
        if let Some(g) = goals.iter_mut().find(|g| g.symbol_id == *gid) {
            let status = GoalStatus::Cancelled { reason: reason.clone() };
            if let Err(e) = update_goal_status(engine, g, status, predicates) {
                tracing::warn!(goal = %gid, "hygiene: failed to cancel stale goal: {e}");
            } else {
                tracing::debug!(goal = %gid, %reason, "hygiene: cancelled stale goal");
                result.cancelled_stale += 1;
            }
        }
    }

    // ── Sub-pass 2: Deduplication ────────────────────────────────────────
    if total_before >= config.dedup_min_queue_size {
        let dups = find_duplicate_goals(goals, config.dedup_overlap_threshold);
        for (dup_id, primary_id) in &dups {
            if let Some(g) = goals.iter_mut().find(|g| g.symbol_id == *dup_id) {
                let status = GoalStatus::Duplicate { primary: *primary_id };
                if let Err(e) = update_goal_status(engine, g, status, predicates) {
                    tracing::warn!(goal = %dup_id, "hygiene: failed to mark goal as duplicate: {e}");
                } else {
                    tracing::debug!(
                        duplicate = %dup_id,
                        primary = %primary_id,
                        "hygiene: deduplicated goal"
                    );
                    result.deduplicated += 1;
                }
            }
        }
    }

    // ── Sub-pass 3: Child depth cap ──────────────────────────────────────
    let deep = find_deep_goals(goals, config.max_child_depth);
    for (gid, reason) in &deep {
        if let Some(g) = goals.iter_mut().find(|g| g.symbol_id == *gid) {
            let status = GoalStatus::Cancelled { reason: reason.clone() };
            if let Err(e) = update_goal_status(engine, g, status, predicates) {
                tracing::warn!(goal = %gid, "hygiene: failed to cancel deep goal: {e}");
            } else {
                tracing::debug!(goal = %gid, %reason, "hygiene: cancelled deep goal");
                result.cancelled_deep += 1;
            }
        }
    }

    result.total_after = goals.iter().filter(|g| is_live(&g.status)).count();
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(id: u64, desc: &str, status: GoalStatus, priority: u8) -> Goal {
        Goal {
            symbol_id: SymbolId::new(id).unwrap(),
            description: desc.into(),
            status,
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
            estimated_effort: None,
        }
    }

    #[test]
    fn word_overlap_identical() {
        assert!((word_overlap("explore the knowledge graph", "explore the knowledge graph") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn word_overlap_disjoint() {
        let s = word_overlap("explore knowledge", "retire goals");
        assert!(s < 0.1, "got {s}");
    }

    #[test]
    fn word_overlap_partial() {
        // Three identical content words out of a slightly larger union — moderate overlap.
        let s = word_overlap(
            "what is the Ennead and its members",
            "what is the Ennead",
        );
        // Union = {what, the, ennead, and, its, members}, intersection = {what, the, ennead}
        // Jaccard = 3/6 for the first sentence alone but "is" is filtered (len 2).
        // Actual: intersection 3, union ~6..8 → between 0.3 and 0.6
        assert!(s > 0.3 && s <= 1.0, "got {s}");
    }

    #[test]
    fn find_stale_detects_never_progressed() {
        let goals = vec![
            make_goal(1, "find is-a for rate:0.8", GoalStatus::Pending, 128),
        ];
        // session_start=0, current=25, max_idle=20 → idle=25 → stale
        let stale = find_stale_goals(&goals, 25, 0, 20);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0, SymbolId::new(1).unwrap());
    }

    #[test]
    fn find_stale_skips_restored_goal_in_fresh_session() {
        // A goal restored at session start has last_progress_cycle=0.
        // With session_start=100 and current=110, idle=10 which is below threshold.
        let goals = vec![
            make_goal(1, "restored but recently started", GoalStatus::Pending, 128),
        ];
        // session_start=100, current=110, max_idle=20 → idle=10 → not stale
        let stale = find_stale_goals(&goals, 110, 100, 20);
        assert!(stale.is_empty(), "restored goals should not be immediately stale");
    }

    #[test]
    fn find_stale_detects_restored_goal_after_idle() {
        // A restored goal (last_progress_cycle=0) should become stale after
        // max_idle_cycles have passed in the current session.
        let goals = vec![
            make_goal(1, "zombie goal never worked on", GoalStatus::Pending, 128),
        ];
        // session_start=100, current=121, max_idle=20 → idle=21 → stale
        let stale = find_stale_goals(&goals, 121, 100, 20);
        assert_eq!(stale.len(), 1, "goal should be stale after idle threshold passed");
    }

    #[test]
    fn find_stale_skips_recently_progressed() {
        let mut g = make_goal(1, "active goal", GoalStatus::Active, 128);
        g.last_progress_cycle = 20;
        let goals = vec![g];
        // current_cycle = 25, session_start=0, max_idle = 20 → idle = 5 → not stale
        let stale = find_stale_goals(&goals, 25, 0, 20);
        assert!(stale.is_empty());
    }

    #[test]
    fn find_stale_skips_user_requested() {
        let mut g = make_goal(1, "user goal", GoalStatus::Pending, 200);
        g.justification = Some(GoalJustification::UserRequested);
        let goals = vec![g];
        // session_start=0, current=100 → idle=100, but user-requested is protected
        let stale = find_stale_goals(&goals, 100, 0, 20);
        assert!(stale.is_empty(), "should not cancel user-requested goals");
    }

    #[test]
    fn find_duplicates_groups_similar() {
        // Two goals with nearly identical wording — only a trailing word differs.
        let goals = vec![
            make_goal(1, "What is the Ennead and its role", GoalStatus::Pending, 200),
            make_goal(2, "What is the Ennead and its role here", GoalStatus::Pending, 100),
            make_goal(3, "explore the knowledge graph structure", GoalStatus::Pending, 150),
        ];
        // threshold 0.75 — goals 1 & 2 share {what, the, ennead, and, its, role} out of
        // {what, the, ennead, and, its, role, here} → 6/7 ≈ 0.86 → should be flagged.
        let dups = find_duplicate_goals(&goals, 0.75);
        assert_eq!(dups.len(), 1);
        let (dup_id, primary_id) = dups[0];
        // Goal 2 is the duplicate (lower priority); goal 1 is primary.
        assert_eq!(dup_id, SymbolId::new(2).unwrap());
        assert_eq!(primary_id, SymbolId::new(1).unwrap());
    }

    #[test]
    fn find_duplicates_no_match_disjoint() {
        let goals = vec![
            make_goal(1, "explore knowledge graph", GoalStatus::Pending, 128),
            make_goal(2, "retire stale goals queue", GoalStatus::Pending, 128),
        ];
        let dups = find_duplicate_goals(&goals, 0.75);
        assert!(dups.is_empty());
    }

    #[test]
    fn goal_depth_root_is_zero() {
        let goals = vec![make_goal(1, "root", GoalStatus::Active, 128)];
        assert_eq!(goal_depth(SymbolId::new(1).unwrap(), &goals), 0);
    }

    #[test]
    fn goal_depth_child_is_one() {
        let mut parent = make_goal(1, "parent", GoalStatus::Active, 200);
        let mut child = make_goal(2, "child", GoalStatus::Active, 150);
        child.parent = Some(SymbolId::new(1).unwrap());
        parent.children.push(SymbolId::new(2).unwrap());
        let goals = vec![parent, child];
        assert_eq!(goal_depth(SymbolId::new(2).unwrap(), &goals), 1);
    }

    #[test]
    fn find_deep_goals_caps_at_max() {
        // depth-3 chain: root → a → b → c
        let root = make_goal(1, "root", GoalStatus::Active, 200);
        let mut a = make_goal(2, "a", GoalStatus::Active, 150);
        a.parent = Some(SymbolId::new(1).unwrap());
        let mut b = make_goal(3, "b", GoalStatus::Active, 130);
        b.parent = Some(SymbolId::new(2).unwrap());
        let mut c = make_goal(4, "c", GoalStatus::Active, 110);
        c.parent = Some(SymbolId::new(3).unwrap());
        let goals = vec![root, a, b, c];
        // max_depth = 2 → goal 4 (depth 3) should be flagged
        let deep = find_deep_goals(&goals, 2);
        assert_eq!(deep.len(), 1);
        assert_eq!(deep[0].0, SymbolId::new(4).unwrap());
    }

    #[test]
    fn hygiene_result_default_is_zero() {
        let r = HygieneResult::default();
        assert_eq!(r.cancelled_stale, 0);
        assert_eq!(r.cancelled_deep, 0);
        assert_eq!(r.deduplicated, 0);
    }
}
