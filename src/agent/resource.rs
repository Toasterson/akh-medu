//! Resource awareness: effort estimation, VOC analysis, and diminishing returns (Phase 11g).
//!
//! Provides CBR-based effort estimation from past goal completions, Value of
//! Computation (VOC) scoring to decide if a goal is worth continued investment,
//! dynamic stall thresholds, diminishing-returns detection, and marginal-value
//! ranking for goal selection.

use std::collections::HashMap;
use std::sync::RwLock;

use anndists::dist::DistHamming;
use hnsw_rs::prelude::*;
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::goal::{Goal, GoalStatus, DEFAULT_STALL_THRESHOLD};
use super::memory::{WorkingMemory, WorkingMemoryKind};
use super::metacognition::CompetenceModel;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to the resource-awareness subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum ResourceError {
    /// No completed effort cases are available for estimation.
    #[error("no effort cases available for estimation")]
    #[diagnostic(
        code(akh::agent::resource::no_effort_cases),
        help("complete at least one goal to seed the effort case base")
    )]
    NoEffortCases,

    /// A project's cycle budget has been exceeded.
    #[error("project cycle budget exceeded: consumed {consumed}, budget {budget}")]
    #[diagnostic(
        code(akh::agent::resource::budget_exceeded),
        help("consider suspending or reprioritizing goals in this project")
    )]
    BudgetExceeded { consumed: u32, budget: u32 },
}

/// Result type alias for the resource subsystem.
pub type ResourceResult<T> = Result<T, ResourceError>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Record of a completed goal's effort for CBR retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortCase {
    /// Human-readable description of the goal.
    pub description: String,
    /// VSA-encoded goal vector for similarity search.
    pub goal_vector: Vec<u32>,
    /// Number of OODA cycles consumed to complete/fail.
    pub cycles_consumed: u32,
    /// Whether the goal succeeded.
    pub succeeded: bool,
    /// Per-tool invocation counts during execution.
    pub tool_usage: HashMap<String, u32>,
    /// KG triple count at goal creation (proxy for initial knowledge coverage).
    pub initial_coverage: u32,
}

/// Estimated effort for a goal, derived from similar past cases.
#[derive(Debug, Clone)]
pub struct EffortEstimate {
    /// Predicted number of OODA cycles to completion.
    pub estimated_cycles: u32,
    /// Confidence in the estimate (0.0–1.0), based on case similarity.
    pub confidence: f32,
    /// SymbolIds of the basis cases used for estimation.
    pub basis: Vec<SymbolId>,
}

/// Full per-goal resource assessment.
#[derive(Debug, Clone)]
pub struct ResourceReport {
    /// Goal being assessed.
    pub goal_id: SymbolId,
    /// Cycles already consumed on this goal.
    pub cycles_consumed: u32,
    /// Remaining budget (from project or estimated effort).
    pub budget_remaining: Option<u32>,
    /// Value of continued computation.
    pub voc: f32,
    /// Recent improvement rate (0.0–1.0).
    pub improvement_rate: f32,
    /// Whether diminishing returns were detected.
    pub diminishing_returns: bool,
    /// Dynamic stall threshold for this goal.
    pub dynamic_stall_threshold: u32,
}

/// Sliding-window history of per-goal improvement rates.
///
/// Keyed by goal SymbolId (as u64), each entry is a `(cycle, rate)` pair.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImprovementHistory {
    /// Per-goal history: goal_id → Vec<(cycle, improvement_rate)>.
    pub entries: HashMap<u64, Vec<(u64, f32)>>,
}

// ---------------------------------------------------------------------------
// EffortIndex (HNSW-backed CBR)
// ---------------------------------------------------------------------------

/// HNSW-backed index of past effort cases for similarity-based retrieval.
///
/// Follows the `FailureIndex` pattern from metacognition.rs.
pub struct EffortIndex {
    /// HNSW index for nearest-neighbor effort lookup.
    hnsw: RwLock<Hnsw<'static, u32, DistHamming>>,
    /// All stored effort cases (for persistence and result lookup).
    cases: Vec<EffortCase>,
    /// Next HNSW point ID.
    next_id: usize,
}

impl std::fmt::Debug for EffortIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EffortIndex")
            .field("cases", &self.cases.len())
            .finish()
    }
}

impl EffortIndex {
    /// Create a new empty effort index.
    pub fn new() -> Self {
        let max_elements = 1000;
        let max_layer = 8;
        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});
        Self {
            hnsw: RwLock::new(hnsw),
            cases: Vec::new(),
            next_id: 0,
        }
    }

    /// Insert an effort case into the index.
    pub fn insert(&mut self, case: EffortCase) {
        let id = self.next_id;
        self.next_id += 1;
        if let Ok(hnsw) = self.hnsw.read() {
            hnsw.insert((&case.goal_vector, id));
        }
        self.cases.push(case);
    }

    /// Search for the k most similar past effort cases.
    pub fn search_similar(&self, query_vec: &[u32], k: usize) -> Vec<&EffortCase> {
        if self.cases.is_empty() || k == 0 {
            return Vec::new();
        }
        let ef_search = (k * 2).max(32);

        let neighbours = if let Ok(hnsw) = self.hnsw.read() {
            hnsw.search(query_vec, k, ef_search)
        } else {
            return Vec::new();
        };

        neighbours
            .into_iter()
            .filter_map(|n| self.cases.get(n.d_id))
            .collect()
    }

    /// Number of stored effort cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// Get all cases (for persistence).
    pub fn cases(&self) -> &[EffortCase] {
        &self.cases
    }

    /// Rebuild from persisted cases.
    pub fn from_cases(cases: Vec<EffortCase>) -> Self {
        let max_elements = cases.len().max(1000);
        let max_layer = (max_elements as f64).log2().ceil() as usize;
        let max_layer = max_layer.max(4).min(16);
        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});

        // Re-insert all cases into the HNSW index.
        for (id, case) in cases.iter().enumerate() {
            hnsw.insert((&case.goal_vector, id));
        }

        let next_id = cases.len();
        Self {
            hnsw: RwLock::new(hnsw),
            cases,
            next_id,
        }
    }
}

impl Default for EffortIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Effort estimation (CBR)
// ---------------------------------------------------------------------------

/// Estimate effort for a goal using k-nearest neighbor retrieval from the case base.
///
/// Returns median cycles of the k nearest cases, scaled by the ratio of the
/// goal's current KG triple count to the cases' initial coverage. Falls back
/// to `Err(NoEffortCases)` when the index is empty.
pub fn estimate_effort(
    goal: &Goal,
    effort_index: &EffortIndex,
    engine: &Engine,
    k: usize,
) -> ResourceResult<EffortEstimate> {
    if effort_index.is_empty() {
        return Err(ResourceError::NoEffortCases);
    }

    // Encode goal description for similarity search.
    let goal_vec = goal_to_u32_vec(&goal.description);
    let neighbours = effort_index.search_similar(&goal_vec, k);

    if neighbours.is_empty() {
        return Err(ResourceError::NoEffortCases);
    }

    // Median of neighbour cycles.
    let mut cycles: Vec<u32> = neighbours.iter().map(|c| c.cycles_consumed).collect();
    cycles.sort_unstable();
    let median = cycles[cycles.len() / 2];

    // KG richness scaling: if the current KG has more triples than the average
    // case's initial_coverage, the goal may need fewer cycles.
    let current_triples = engine.all_symbols().len() as u32;
    let avg_coverage: u32 = neighbours.iter().map(|c| c.initial_coverage).sum::<u32>()
        / neighbours.len().max(1) as u32;
    let scale = if avg_coverage > 0 && current_triples > avg_coverage {
        (avg_coverage as f32 / current_triples as f32).max(0.5)
    } else {
        1.0
    };

    let estimated = (median as f32 * scale).ceil() as u32;
    let confidence = (neighbours.len() as f32 / k as f32).min(1.0);

    Ok(EffortEstimate {
        estimated_cycles: estimated.max(1),
        confidence,
        basis: Vec::new(), // No SymbolIds for case base entries.
    })
}

// ---------------------------------------------------------------------------
// VOC (Value of Computation)
// ---------------------------------------------------------------------------

/// Compute Value of Continued Computation for a goal.
///
/// `VOC = P(improvement) * magnitude - cycle_cost`
///
/// A positive VOC means more cycles are worthwhile; negative means the
/// agent should switch to another goal.
pub fn compute_voc(
    goal: &Goal,
    goals: &[Goal],
    competence: &CompetenceModel,
    improvement_history: &ImprovementHistory,
    cycle: u64,
    window: u64,
) -> f32 {
    let category = super::metacognition::categorize_goal(goal);
    let comp = competence.category_competence(&category);

    // P(improvement): derived from recent improvement rate + competence.
    let rate = current_improvement_rate(goal, improvement_history, cycle, window);
    let p_improvement = ((rate + comp) / 2.0).clamp(0.0, 1.0);

    // Magnitude: how much priority would completing this goal add?
    let priority_fraction = goal.computed_priority() as f32 / 255.0;
    let magnitude = priority_fraction;

    // Cycle cost: base cost of one cycle, scaled down by the number of active goals
    // (more goals = lower per-goal cost since we're context-switching anyway).
    let active_count = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .count()
        .max(1);
    let cycle_cost = 0.1 / active_count as f32;

    p_improvement * magnitude - cycle_cost
}

/// Dynamic stall threshold: adapts based on estimated effort.
///
/// Uses `1.5 * estimated_effort` if available, otherwise falls back
/// to `DEFAULT_STALL_THRESHOLD`.
pub fn dynamic_stall_threshold(estimate: Option<&EffortEstimate>) -> u32 {
    match estimate {
        Some(est) => ((est.estimated_cycles as f32 * 1.5).ceil() as u32).max(DEFAULT_STALL_THRESHOLD),
        None => DEFAULT_STALL_THRESHOLD,
    }
}

// ---------------------------------------------------------------------------
// Diminishing returns
// ---------------------------------------------------------------------------

/// Detect diminishing returns for a goal.
///
/// Returns `true` if the current improvement rate is less than half the rate
/// observed in the previous window.
pub fn detect_diminishing_returns(
    goal: &Goal,
    history: &ImprovementHistory,
    cycle: u64,
    window: u64,
) -> bool {
    let entries = match history.entries.get(&goal.symbol_id.get()) {
        Some(e) if e.len() >= 2 => e,
        _ => return false,
    };

    let current_rate = current_improvement_rate(goal, history, cycle, window);

    // Previous rate: the second-to-last entry.
    let prev_rate = entries
        .iter()
        .rev()
        .nth(1)
        .map(|&(_, r)| r)
        .unwrap_or(0.0);

    prev_rate > 0.01 && current_rate < prev_rate * 0.5
}

/// Record an improvement rate snapshot.
pub fn record_improvement(
    history: &mut ImprovementHistory,
    goal_id: SymbolId,
    cycle: u64,
    rate: f32,
    max_entries: usize,
) {
    let entries = history.entries.entry(goal_id.get()).or_default();
    entries.push((cycle, rate));
    // Trim to sliding window.
    if entries.len() > max_entries {
        let excess = entries.len() - max_entries;
        entries.drain(..excess);
    }
}

/// Compute current improvement rate for a goal from working memory decision entries.
///
/// Defined as the fraction of recent cycles that advanced the goal (non-zero
/// `last_progress_cycle` delta over the window).
fn current_improvement_rate(
    goal: &Goal,
    history: &ImprovementHistory,
    cycle: u64,
    window: u64,
) -> f32 {
    // Try to use the latest recorded rate from history.
    if let Some(entries) = history.entries.get(&goal.symbol_id.get()) {
        if let Some(&(_, rate)) = entries.last() {
            return rate;
        }
    }

    // Fallback: heuristic based on goal's last_progress_cycle.
    if goal.cycles_worked == 0 {
        return 0.5; // Default for new goals.
    }
    let cycles_since_progress = cycle.saturating_sub(goal.last_progress_cycle);
    if cycles_since_progress == 0 {
        return 1.0;
    }
    (1.0 / cycles_since_progress as f32).min(1.0)
}

// ---------------------------------------------------------------------------
// Goal ranking
// ---------------------------------------------------------------------------

/// Rank goals by marginal value: `VOC * priority / 255`.
///
/// Returns `(goal_id, marginal_value, voc)` tuples sorted descending.
pub fn rank_goals_by_marginal_value(
    goals: &[Goal],
    competence: &CompetenceModel,
    history: &ImprovementHistory,
    cycle: u64,
    window: u64,
) -> Vec<(SymbolId, f32, f32)> {
    let mut ranked: Vec<(SymbolId, f32, f32)> = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .map(|g| {
            let voc = compute_voc(g, goals, competence, history, cycle, window);
            let marginal = voc * (g.computed_priority() as f32 / 255.0);
            (g.symbol_id, marginal, voc)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

// ---------------------------------------------------------------------------
// Opportunity cost recording
// ---------------------------------------------------------------------------

/// Record the opportunity cost of selecting one goal over alternatives.
///
/// Stores a KG triple: `selected_goal → agent:opportunity_cost → cost_entity`.
pub fn record_opportunity_cost(
    engine: &Engine,
    cycle: u64,
    selected: SymbolId,
    cost: f32,
    alternative: Option<SymbolId>,
) {
    let cost_label = format!("cost:{cost:.3}@cycle:{cycle}");
    if let Ok(cost_entity) = engine.resolve_or_create_entity(&cost_label) {
        if let Ok(predicate) = engine.resolve_or_create_relation("agent:opportunity_cost") {
            let _ = engine.add_triple(&Triple::new(selected, predicate, cost_entity));
        }
    }

    // Link to the passed-over alternative.
    if let Some(alt) = alternative {
        let alt_label = format!("alt_goal:{}", alt.get());
        if let Ok(alt_entity) = engine.resolve_or_create_entity(&alt_label) {
            if let Ok(predicate) = engine.resolve_or_create_relation("agent:passed_over") {
                let _ = engine.add_triple(&Triple::new(selected, predicate, alt_entity));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Effort case creation
// ---------------------------------------------------------------------------

/// Create an EffortCase from a completed/failed goal and current WM state.
pub fn create_effort_case(
    goal: &Goal,
    wm: &WorkingMemory,
    engine: &Engine,
) -> EffortCase {
    // Count tool usage from WM decision entries.
    let mut tool_usage: HashMap<String, u32> = HashMap::new();
    for entry in wm.by_kind(WorkingMemoryKind::Decision) {
        // WM Decision entries have format: "Decide: tool=X, goal=\"Y\", ..."
        if let Some(rest) = entry.content.strip_prefix("Decide: tool=") {
            if let Some(comma) = rest.find(',') {
                let tool = &rest[..comma];
                *tool_usage.entry(tool.to_string()).or_insert(0) += 1;
            }
        }
    }

    let goal_vector = goal_to_u32_vec(&goal.description);
    let initial_coverage = engine.all_symbols().len() as u32;

    EffortCase {
        description: goal.description.clone(),
        goal_vector,
        cycles_consumed: goal.cycles_worked,
        succeeded: matches!(goal.status, GoalStatus::Completed),
        tool_usage,
        initial_coverage,
    }
}

// ---------------------------------------------------------------------------
// Full assessment
// ---------------------------------------------------------------------------

/// Compute a full resource report for a goal.
pub fn assess_goal_resources(
    goal: &Goal,
    goals: &[Goal],
    competence: &CompetenceModel,
    improvement_history: &ImprovementHistory,
    effort_index: &EffortIndex,
    engine: &Engine,
    cycle: u64,
    window: u64,
    project_budget: Option<(u32, u32)>, // (budget, consumed)
) -> ResourceReport {
    let estimate = estimate_effort(goal, effort_index, engine, 3).ok();
    let threshold = dynamic_stall_threshold(estimate.as_ref());
    let voc = compute_voc(goal, goals, competence, improvement_history, cycle, window);
    let rate = current_improvement_rate(goal, improvement_history, cycle, window);
    let diminishing = detect_diminishing_returns(goal, improvement_history, cycle, window);

    let budget_remaining = project_budget.map(|(budget, consumed)| budget.saturating_sub(consumed));

    ResourceReport {
        goal_id: goal.symbol_id,
        cycles_consumed: goal.cycles_worked,
        budget_remaining,
        voc,
        improvement_rate: rate,
        diminishing_returns: diminishing,
        dynamic_stall_threshold: threshold,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode a goal description string as a u32 vector for HNSW search.
///
/// Uses the same deterministic hash approach as `simple_text_hash` but
/// produces u32 elements directly.
fn goal_to_u32_vec(description: &str) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // 256 u32 elements = 1024 bytes of hash-based encoding.
    let dim = 256;
    let mut vec = vec![0u32; dim];
    let words: Vec<&str> = description.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        let idx = (h as usize) % dim;
        vec[idx] ^= h as u32;

        // Second hash for spread.
        let mut hasher2 = DefaultHasher::new();
        h.hash(&mut hasher2);
        let h2 = hasher2.finish();
        let idx2 = (h2 as usize) % dim;
        vec[idx2] ^= h2 as u32;
    }

    vec
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolId;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    fn make_goal(id: u64, desc: &str, priority: u8, status: GoalStatus) -> Goal {
        Goal {
            symbol_id: sym(id),
            description: desc.into(),
            status,
            priority,
            success_criteria: "done".into(),
            parent: None,
            children: vec![],
            created_at: 0,
            cycles_worked: 5,
            last_progress_cycle: 3,
            source: None,
            blocked_by: vec![],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        }
    }

    fn make_effort_case(desc: &str, cycles: u32, succeeded: bool) -> EffortCase {
        EffortCase {
            description: desc.into(),
            goal_vector: goal_to_u32_vec(desc),
            cycles_consumed: cycles,
            succeeded,
            tool_usage: HashMap::new(),
            initial_coverage: 100,
        }
    }

    // -- EffortIndex --

    #[test]
    fn effort_index_empty() {
        let idx = EffortIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn effort_index_insert_and_search() {
        let mut idx = EffortIndex::new();
        idx.insert(make_effort_case("explore knowledge graph", 10, true));
        idx.insert(make_effort_case("reason about types", 20, true));
        idx.insert(make_effort_case("explore knowledge base", 12, true));

        assert_eq!(idx.len(), 3);

        let query = goal_to_u32_vec("explore knowledge");
        let results = idx.search_similar(&query, 2);
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn effort_index_from_cases_roundtrip() {
        let cases = vec![
            make_effort_case("goal A", 5, true),
            make_effort_case("goal B", 10, false),
        ];
        let idx = EffortIndex::from_cases(cases);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    // -- Effort estimation --

    #[test]
    fn estimate_effort_empty_index_errors() {
        let idx = EffortIndex::new();
        let goal = make_goal(1, "test", 100, GoalStatus::Active);
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let result = estimate_effort(&goal, &idx, &engine, 3);
        assert!(result.is_err());
    }

    #[test]
    fn estimate_effort_returns_median() {
        let mut idx = EffortIndex::new();
        idx.insert(make_effort_case("explore data", 8, true));
        idx.insert(make_effort_case("explore types", 12, true));
        idx.insert(make_effort_case("explore facts", 10, true));

        let goal = make_goal(1, "explore information", 100, GoalStatus::Active);
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let est = estimate_effort(&goal, &idx, &engine, 3).unwrap();
        assert!(est.estimated_cycles >= 1);
        assert!(est.confidence > 0.0);
    }

    // -- VOC --

    #[test]
    fn voc_positive_for_high_priority_progressing_goal() {
        let goal = make_goal(1, "important task", 200, GoalStatus::Active);
        let goals = vec![goal.clone()];
        let competence = CompetenceModel::default();
        let mut history = ImprovementHistory::default();
        record_improvement(&mut history, sym(1), 5, 0.8, 20);

        let voc = compute_voc(&goal, &goals, &competence, &history, 6, 5);
        // High priority + good improvement rate should give positive VOC.
        assert!(voc > 0.0, "expected positive VOC, got {voc}");
    }

    #[test]
    fn voc_negative_for_stalled_low_priority_goal() {
        let mut goal = make_goal(1, "low priority stalled", 10, GoalStatus::Active);
        goal.cycles_worked = 20;
        goal.last_progress_cycle = 0;

        let other = make_goal(2, "another goal", 200, GoalStatus::Active);
        let goals = vec![goal.clone(), other];
        let competence = CompetenceModel::default();
        let mut history = ImprovementHistory::default();
        record_improvement(&mut history, sym(1), 20, 0.0, 20);

        let voc = compute_voc(&goal, &goals, &competence, &history, 21, 5);
        // Low priority + zero improvement + multiple goals => negative VOC.
        assert!(voc <= 0.0, "expected non-positive VOC, got {voc}");
    }

    // -- Dynamic threshold --

    #[test]
    fn dynamic_threshold_with_estimate() {
        let est = EffortEstimate {
            estimated_cycles: 10,
            confidence: 0.8,
            basis: vec![],
        };
        let threshold = dynamic_stall_threshold(Some(&est));
        assert_eq!(threshold, 15); // 10 * 1.5
    }

    #[test]
    fn dynamic_threshold_without_estimate() {
        let threshold = dynamic_stall_threshold(None);
        assert_eq!(threshold, DEFAULT_STALL_THRESHOLD);
    }

    #[test]
    fn dynamic_threshold_minimum_clamp() {
        let est = EffortEstimate {
            estimated_cycles: 2,
            confidence: 0.5,
            basis: vec![],
        };
        let threshold = dynamic_stall_threshold(Some(&est));
        // 2 * 1.5 = 3, but DEFAULT_STALL_THRESHOLD = 5, so clamp to 5.
        assert_eq!(threshold, DEFAULT_STALL_THRESHOLD);
    }

    // -- Diminishing returns --

    #[test]
    fn diminishing_returns_detected() {
        let goal = make_goal(1, "test", 100, GoalStatus::Active);
        let mut history = ImprovementHistory::default();
        record_improvement(&mut history, sym(1), 5, 0.8, 20);
        record_improvement(&mut history, sym(1), 10, 0.2, 20);

        assert!(detect_diminishing_returns(&goal, &history, 11, 5));
    }

    #[test]
    fn no_diminishing_returns_when_improving() {
        let goal = make_goal(1, "test", 100, GoalStatus::Active);
        let mut history = ImprovementHistory::default();
        record_improvement(&mut history, sym(1), 5, 0.4, 20);
        record_improvement(&mut history, sym(1), 10, 0.6, 20);

        assert!(!detect_diminishing_returns(&goal, &history, 11, 5));
    }

    #[test]
    fn no_diminishing_returns_insufficient_history() {
        let goal = make_goal(1, "test", 100, GoalStatus::Active);
        let history = ImprovementHistory::default();

        assert!(!detect_diminishing_returns(&goal, &history, 5, 5));
    }

    // -- Improvement history --

    #[test]
    fn record_improvement_window_trimming() {
        let mut history = ImprovementHistory::default();
        for i in 0..10 {
            record_improvement(&mut history, sym(1), i, i as f32 * 0.1, 5);
        }
        assert_eq!(history.entries[&1].len(), 5);
        // Should keep the last 5 entries.
        assert_eq!(history.entries[&1][0].0, 5);
    }

    // -- Marginal ranking --

    #[test]
    fn rank_goals_by_marginal_value_sorted() {
        let goals = vec![
            make_goal(1, "low", 50, GoalStatus::Active),
            make_goal(2, "high", 200, GoalStatus::Active),
            make_goal(3, "done", 255, GoalStatus::Completed),
        ];
        let competence = CompetenceModel::default();
        let history = ImprovementHistory::default();

        let ranked = rank_goals_by_marginal_value(&goals, &competence, &history, 5, 5);
        // Only active goals should appear.
        assert_eq!(ranked.len(), 2);
        // Higher priority goal should rank first.
        assert_eq!(ranked[0].0, sym(2));
    }

    // -- Budget check --

    #[test]
    fn assess_detects_budget_state() {
        let goal = make_goal(1, "test", 100, GoalStatus::Active);
        let goals = vec![goal.clone()];
        let competence = CompetenceModel::default();
        let history = ImprovementHistory::default();
        let idx = EffortIndex::new();
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let report = assess_goal_resources(
            &goal,
            &goals,
            &competence,
            &history,
            &idx,
            &engine,
            5,
            5,
            Some((100, 80)),
        );

        assert_eq!(report.budget_remaining, Some(20));
        assert_eq!(report.dynamic_stall_threshold, DEFAULT_STALL_THRESHOLD);
    }

    // -- Serialization roundtrip --

    #[test]
    fn effort_case_serde_roundtrip() {
        let case = make_effort_case("test goal", 10, true);
        let bytes = bincode::serialize(&case).unwrap();
        let restored: EffortCase = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.description, "test goal");
        assert_eq!(restored.cycles_consumed, 10);
        assert!(restored.succeeded);
    }

    #[test]
    fn improvement_history_serde_roundtrip() {
        let mut history = ImprovementHistory::default();
        record_improvement(&mut history, sym(1), 5, 0.5, 20);
        record_improvement(&mut history, sym(1), 10, 0.8, 20);

        let bytes = bincode::serialize(&history).unwrap();
        let restored: ImprovementHistory = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.entries[&1].len(), 2);
    }

    // -- Helper --

    #[test]
    fn goal_to_u32_vec_deterministic() {
        let a = goal_to_u32_vec("explore the knowledge graph");
        let b = goal_to_u32_vec("explore the knowledge graph");
        assert_eq!(a, b);
    }

    #[test]
    fn goal_to_u32_vec_different_inputs() {
        let a = goal_to_u32_vec("explore the knowledge graph");
        let b = goal_to_u32_vec("reason about types");
        assert_ne!(a, b);
    }
}
