//! Metacognitive monitoring and control (Nelson-Narens inspired).
//!
//! Provides self-evaluation capabilities:
//! - **CompetenceModel**: cumulative per-tool and per-category success tracking
//! - **FailureIndex**: HNSW-backed similarity search over past failure patterns
//! - **MetacognitionConfig**: tuning parameters for monitoring thresholds
//! - **GoalEvaluation / MetacognitiveControl**: monitoring signals and control actions
//!
//! The metacognitive layer runs during reflection to question whether goals are
//! still worth pursuing, tracks competence via calibrated predictions, encodes
//! failure patterns as VSA hypervectors for similarity search, and uses e-graph
//! equality saturation to reformulate infeasible goals into simpler versions.

use std::collections::HashMap;
use std::sync::RwLock;

use anndists::dist::DistHamming;
use hnsw_rs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::error::AgentResult;
use super::goal::{Goal, GoalJustification, GoalSource, GoalStatus};
use super::memory::{WorkingMemory, WorkingMemoryKind};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the metacognitive monitoring/control layer.
#[derive(Debug, Clone)]
pub struct MetacognitionConfig {
    /// Sliding window size for improvement rate computation (default: 5 cycles).
    pub progress_window: u64,
    /// Autoepistemic effort multiplier: question a goal if cycles_worked exceeds
    /// this factor times the estimated effort (default: 2.0).
    pub autoepistemic_effort_multiplier: f32,
    /// Minimum improvement rate threshold below which a goal may be questioned
    /// (default: 0.0 — any zero-progress goal is eligible).
    pub min_improvement_rate: f32,
    /// ZPD lower bound factor: difficulty >= competence × this means "in zone" (default: 0.8).
    pub zpd_low_factor: f32,
    /// ZPD upper bound factor: difficulty <= competence × this means "in zone" (default: 1.3).
    pub zpd_high_factor: f32,
    /// Priority boost for goals in ZPD sweet spot (default: 20).
    pub zpd_boost: i16,
    /// Priority penalty for goals beyond competence (default: 30).
    pub zpd_penalty: i16,
    /// Maximum calibration history entries (default: 100).
    pub max_calibration_history: usize,
    /// Number of nearest failures to search for in HNSW (default: 3).
    pub failure_search_k: usize,
}

impl Default for MetacognitionConfig {
    fn default() -> Self {
        Self {
            progress_window: 5,
            autoepistemic_effort_multiplier: 2.0,
            min_improvement_rate: 0.0,
            zpd_low_factor: 0.8,
            zpd_high_factor: 1.3,
            zpd_boost: 20,
            zpd_penalty: 30,
            max_calibration_history: 100,
            failure_search_k: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// CompetenceModel
// ---------------------------------------------------------------------------

/// Per-tool success tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolRecord {
    pub successes: u32,
    pub attempts: u32,
}

/// Per-goal-category success tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryRecord {
    pub completed: u32,
    pub attempted: u32,
}

/// Calibration observation: (predicted probability, actual outcome).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub predicted: f32,
    pub actual: f32, // 0.0 or 1.0
}

/// Cumulative competence tracker across sessions.
///
/// Tracks per-tool success rates, per-goal-category completion rates, and
/// calibration history for Brier score computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompetenceModel {
    /// Per-tool success counters.
    pub tools: HashMap<String, ToolRecord>,
    /// Per-goal-category success counters.
    pub categories: HashMap<String, CategoryRecord>,
    /// Calibration history for Brier score.
    pub calibration_history: Vec<CalibrationPoint>,
}

impl CompetenceModel {
    /// Update competence after a tool execution.
    pub fn update_tool(&mut self, tool_name: &str, success: bool) {
        let record = self
            .tools
            .entry(tool_name.to_string())
            .or_default();
        record.attempts += 1;
        if success {
            record.successes += 1;
        }
    }

    /// Update competence after a goal completes or fails.
    pub fn update_category(&mut self, category: &str, completed: bool) {
        let record = self
            .categories
            .entry(category.to_string())
            .or_default();
        record.attempted += 1;
        if completed {
            record.completed += 1;
        }
    }

    /// Record a calibration observation.
    pub fn record_calibration(&mut self, predicted: f32, actual: f32, max_history: usize) {
        self.calibration_history.push(CalibrationPoint { predicted, actual });
        if self.calibration_history.len() > max_history {
            self.calibration_history.remove(0);
        }
    }

    /// Laplace-smoothed success rate for a tool: (successes + 1) / (attempts + 2).
    pub fn tool_success_rate(&self, tool_name: &str) -> f32 {
        match self.tools.get(tool_name) {
            Some(r) => (r.successes as f32 + 1.0) / (r.attempts as f32 + 2.0),
            None => 0.5, // uninformative prior
        }
    }

    /// Laplace-smoothed completion rate for a goal category.
    pub fn category_competence(&self, category: &str) -> f32 {
        match self.categories.get(category) {
            Some(r) => (r.completed as f32 + 1.0) / (r.attempted as f32 + 2.0),
            None => 0.5,
        }
    }

    /// Brier score: mean((predicted - actual)^2). Lower is better.
    /// Returns 0.0 if no calibration history.
    pub fn calibration_error(&self) -> f32 {
        if self.calibration_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .calibration_history
            .iter()
            .map(|p| (p.predicted - p.actual).powi(2))
            .sum();
        sum / self.calibration_history.len() as f32
    }

    /// Store per-tool success rates as KG triples for SPARQL queryability.
    pub fn store_competence_triples(
        &self,
        engine: &Engine,
        success_rate_pred: SymbolId,
    ) -> AgentResult<()> {
        for (tool_name, record) in &self.tools {
            let rate = (record.successes as f32 + 1.0) / (record.attempts as f32 + 2.0);
            let tool_sym = engine.resolve_or_create_entity(&format!("tool:{tool_name}"))?;
            let rate_sym =
                engine.resolve_or_create_entity(&format!("rate:{rate:.4}"))?;
            let _ = engine.add_triple(&Triple::new(tool_sym, success_rate_pred, rate_sym));
        }
        Ok(())
    }
}

/// Derive a category string from a goal's source and description.
///
/// Uses the GoalSource discriminant + first significant keyword from description.
pub fn categorize_goal(goal: &Goal) -> String {
    let prefix = match &goal.source {
        Some(GoalSource::GapDetection { .. }) => "gap",
        Some(GoalSource::ContradictionDetected { .. }) => "contradiction",
        Some(GoalSource::OpportunityDetected { .. }) => "opportunity",
        Some(GoalSource::DriveExceeded { drive, .. }) => {
            return format!("drive:{drive}");
        }
        Some(GoalSource::ImpasseDetected { .. }) => "impasse",
        Some(GoalSource::ReflectionInsight { .. }) => "reflection",
        Some(GoalSource::WorldChange { .. }) => "worldchange",
        None => "user",
    };

    // Extract first significant keyword (>3 chars, lowercase, not a stop word).
    let keyword = goal
        .description
        .split_whitespace()
        .find(|w| w.len() > 3 && !matches!(w.to_lowercase().as_str(), "find" | "make" | "this" | "that" | "with" | "from" | "about"))
        .unwrap_or("general");

    format!("{prefix}:{}", keyword.to_lowercase())
}

// ---------------------------------------------------------------------------
// FailureCase & FailureIndex
// ---------------------------------------------------------------------------

/// A recorded failure context with its encoded hypervector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCase {
    /// Goal description at time of failure.
    pub goal_description: String,
    /// Tool that failed.
    pub tool_name: String,
    /// Error message or failure reason.
    pub error_message: String,
    /// Cycle at which the failure occurred.
    pub cycle: u64,
    /// Whether a resolution was later found.
    pub resolved: bool,
    /// Resolution tool (if resolved).
    pub resolution_tool: Option<String>,
    /// Encoded hypervector (serialized for persistence).
    #[serde(with = "hypervec_serde")]
    pub vector: Vec<u8>,
}

/// Serde helpers for raw byte vectors (HyperVec data).
mod hypervec_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], s: S) -> Result<S::Ok, S::Error> {
        data.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        Vec::<u8>::deserialize(d)
    }
}

/// Convert byte slice to u32 slice for HNSW (same logic as item_memory).
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(buf)
        })
        .collect()
}

/// HNSW-backed index of past failure patterns for similarity search.
pub struct FailureIndex {
    /// HNSW index for nearest-neighbor failure lookup.
    hnsw: RwLock<Hnsw<'static, u32, DistHamming>>,
    /// All stored failure cases (for persistence and result lookup).
    cases: Vec<FailureCase>,
    /// Next HNSW point ID.
    next_id: usize,
}

impl std::fmt::Debug for FailureIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailureIndex")
            .field("cases", &self.cases.len())
            .finish()
    }
}

impl FailureIndex {
    /// Create a new empty failure index.
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

    /// Insert a failure case into the index.
    pub fn insert(&mut self, case: FailureCase) {
        let data_u32 = bytes_to_u32_vec(&case.vector);
        let id = self.next_id;
        self.next_id += 1;
        if let Ok(hnsw) = self.hnsw.read() {
            hnsw.insert((&data_u32, id));
        }
        self.cases.push(case);
    }

    /// Search for the k most similar past failures.
    pub fn search_similar(&self, query_vec: &[u8], k: usize) -> Vec<&FailureCase> {
        if self.cases.is_empty() || k == 0 {
            return Vec::new();
        }
        let query_u32 = bytes_to_u32_vec(query_vec);
        let ef_search = (k * 2).max(32);

        let neighbours = if let Ok(hnsw) = self.hnsw.read() {
            hnsw.search(&query_u32, k, ef_search)
        } else {
            return Vec::new();
        };

        neighbours
            .into_iter()
            .filter_map(|n| self.cases.get(n.d_id))
            .collect()
    }

    /// Number of stored failure cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// Get all cases (for persistence).
    pub fn cases(&self) -> &[FailureCase] {
        &self.cases
    }

    /// Rebuild from persisted cases.
    pub fn from_cases(cases: Vec<FailureCase>) -> Self {
        let max_elements = cases.len().max(1000);
        let max_layer = (max_elements as f64).log2().ceil() as usize;
        let max_layer = max_layer.max(4).min(16);
        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});

        // Re-insert all cases into the HNSW index.
        for (id, case) in cases.iter().enumerate() {
            let data_u32 = bytes_to_u32_vec(&case.vector);
            hnsw.insert((&data_u32, id));
        }

        let next_id = cases.len();
        Self {
            hnsw: RwLock::new(hnsw),
            cases,
            next_id,
        }
    }

    /// Mark a failure as resolved (when the same goal later succeeded with a different tool).
    pub fn mark_resolved(&mut self, goal_description: &str, resolution_tool: &str) {
        for case in &mut self.cases {
            if case.goal_description == goal_description && !case.resolved {
                case.resolved = true;
                case.resolution_tool = Some(resolution_tool.to_string());
            }
        }
    }
}

impl Default for FailureIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GoalEvaluation & MetacognitiveControl
// ---------------------------------------------------------------------------

/// Per-goal metacognitive assessment produced by `evaluate_goals()`.
#[derive(Debug, Clone)]
pub struct GoalEvaluation {
    /// Goal being evaluated.
    pub goal_id: SymbolId,
    /// Progress events per cycle in the sliding window.
    pub improvement_rate: f32,
    /// Whether progress is diminishing (rate is declining).
    pub diminishing_returns: bool,
    /// Estimated difficulty from similar past goals [0.0, 1.0].
    pub estimated_difficulty: f32,
    /// Competence score for this goal's category [0.0, 1.0].
    pub competence: f32,
    /// Whether the goal is within the Zone of Proximal Development.
    pub in_zpd: bool,
    /// Priority adjustment from ZPD analysis.
    pub zpd_adjustment: i16,
    /// Whether the autoepistemic check recommends questioning this goal.
    pub should_question: bool,
    /// Reason for questioning (empty if should_question is false).
    pub questioning_reason: String,
    /// Similar past failures found via HNSW search.
    pub similar_failures: Vec<FailureCase>,
}

/// Control signals produced by metacognitive evaluation.
#[derive(Debug, Clone)]
pub enum MetacognitiveControl {
    /// No action needed — goal is progressing normally.
    Continue,
    /// Suggest reformulating the goal with relaxed criteria (e-graph output).
    SuggestReformulate {
        goal_id: SymbolId,
        relaxed_criteria: String,
    },
    /// Question whether this goal is still worth pursuing.
    QuestionGoal { goal_id: SymbolId, reason: String },
    /// Adjust priority based on ZPD analysis.
    AdjustZpd { goal_id: SymbolId, delta: i16 },
    /// A goal's justification has been invalidated (AGM cascade).
    InvalidateJustification { goal_id: SymbolId, reason: String },
    /// Beliefs should be revised: retract and assert the listed symbols.
    ReviseBeliefs {
        goal_id: SymbolId,
        retract: Vec<SymbolId>,
        assert: Vec<SymbolId>,
    },
}

// ---------------------------------------------------------------------------
// Monitoring functions
// ---------------------------------------------------------------------------

/// Compute improvement rate: progress events per cycle in a sliding window.
///
/// Counts GoalUpdate and ToolResult WM entries relevant to a goal within
/// the last `window` cycles.
pub fn compute_improvement_rate(
    goal: &Goal,
    working_memory: &WorkingMemory,
    current_cycle: u64,
    window: u64,
) -> f32 {
    if window == 0 {
        return 0.0;
    }
    let window_start = current_cycle.saturating_sub(window);

    let progress_events = working_memory
        .entries()
        .iter()
        .filter(|e| {
            e.source_cycle >= window_start
                && e.source_cycle <= current_cycle
                && (e.kind == WorkingMemoryKind::GoalUpdate
                    || e.kind == WorkingMemoryKind::ToolResult)
                && e.symbols.contains(&goal.symbol_id)
        })
        .count();

    progress_events as f32 / window as f32
}

/// Estimate difficulty from similar completed/failed goals.
///
/// Uses keyword overlap between the current goal's description and past goals.
/// Returns difficulty in [0.0, 1.0].
pub fn estimate_difficulty(goal: &Goal, all_goals: &[Goal]) -> f32 {
    let keywords: Vec<&str> = goal
        .description
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();

    if keywords.is_empty() {
        return 0.5; // unknown difficulty
    }

    let mut total_weight = 0.0f32;
    let mut weighted_difficulty = 0.0f32;

    for other in all_goals {
        if other.symbol_id == goal.symbol_id {
            continue;
        }
        // Only learn from completed or failed goals.
        let outcome_known = matches!(
            other.status,
            GoalStatus::Completed | GoalStatus::Failed { .. }
        );
        if !outcome_known {
            continue;
        }

        // Keyword overlap score.
        let overlap = keywords
            .iter()
            .filter(|kw| other.description.to_lowercase().contains(&kw.to_lowercase()))
            .count();
        if overlap == 0 {
            continue;
        }
        let similarity = overlap as f32 / keywords.len() as f32;

        // Difficulty: cycles consumed normalized (max 50 cycles = 1.0).
        let raw_difficulty = (other.cycles_worked as f32 / 50.0).min(1.0);

        // Weight by similarity.
        total_weight += similarity;
        weighted_difficulty += similarity * raw_difficulty;
    }

    if total_weight > 0.0 {
        weighted_difficulty / total_weight
    } else {
        0.5 // no similar goals found
    }
}

/// Autoepistemic check: Moore-style self-reflection.
///
/// If cycles_worked exceeds `effort_multiplier × estimated_effort` AND
/// improvement_rate is at or below `threshold`, returns a questioning reason.
pub fn autoepistemic_check(
    goal: &Goal,
    improvement_rate: f32,
    estimated_difficulty: f32,
    config: &MetacognitionConfig,
) -> Option<String> {
    // Estimated effort: difficulty × 50 (max cycle reference).
    let estimated_effort = (estimated_difficulty * 50.0).max(1.0);
    let threshold = estimated_effort * config.autoepistemic_effort_multiplier;

    if goal.cycles_worked as f32 >= threshold && improvement_rate <= config.min_improvement_rate {
        Some(format!(
            "Goal has consumed {} cycles (expected ~{:.0}, threshold {:.0}) \
             with improvement rate {:.2} — questioning value of continued pursuit",
            goal.cycles_worked,
            estimated_effort,
            threshold,
            improvement_rate,
        ))
    } else {
        None
    }
}

/// Check whether a goal's justification is still valid.
///
/// - `UserRequested` is always valid (highest entrenchment).
/// - `DecomposedFrom` checks parent status: valid if Active/Pending/Suspended.
/// - `InferredFromKG` checks that at least half the supporting symbols still have triples.
/// - `DefaultAssumption` is always structurally valid.
pub fn is_justification_valid(
    justification: &GoalJustification,
    all_goals: &[Goal],
    engine: &Engine,
) -> bool {
    match justification {
        GoalJustification::UserRequested => true,
        GoalJustification::DecomposedFrom { parent } => {
            all_goals.iter().any(|g| {
                g.symbol_id == *parent
                    && matches!(
                        g.status,
                        GoalStatus::Active | GoalStatus::Pending | GoalStatus::Suspended
                    )
            })
        }
        GoalJustification::InferredFromKG { supporting } => {
            if supporting.is_empty() {
                return true;
            }
            let valid_count = supporting
                .iter()
                .filter(|s| !engine.triples_from(**s).is_empty())
                .count();
            valid_count * 2 >= supporting.len() // at least half still valid
        }
        GoalJustification::DefaultAssumption { .. } => true,
    }
}

/// Cascade invalidation when a goal is suspended/failed/reformulated.
///
/// Finds all goals with DecomposedFrom pointing to the invalidated goal
/// and produces InvalidateJustification signals. Follows entrenchment ordering:
/// lowest-entrenchment goals are invalidated first.
pub fn cascade_invalidation(
    invalidated_goal_id: SymbolId,
    all_goals: &[Goal],
) -> Vec<MetacognitiveControl> {
    let mut affected: Vec<&Goal> = all_goals
        .iter()
        .filter(|g| {
            matches!(g.status, GoalStatus::Active | GoalStatus::Pending)
                && g.justification.as_ref().is_some_and(|j| matches!(
                    j,
                    GoalJustification::DecomposedFrom { parent } if *parent == invalidated_goal_id
                ))
        })
        .collect();

    // Sort by entrenchment (lowest first — they get invalidated first).
    affected.sort_by_key(|g| {
        g.justification
            .as_ref()
            .map(|j| j.entrenchment())
            .unwrap_or(0)
    });

    affected
        .into_iter()
        .map(|g| MetacognitiveControl::InvalidateJustification {
            goal_id: g.symbol_id,
            reason: format!(
                "Parent goal {} was invalidated — cascading to child",
                invalidated_goal_id
            ),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ZPD computation
// ---------------------------------------------------------------------------

/// Compute ZPD (Zone of Proximal Development) adjustment for a goal.
///
/// If difficulty is within [competence × low_factor, competence × high_factor],
/// the goal is in the ZPD sweet spot and gets a priority boost.
/// If difficulty exceeds the upper bound, the goal gets a priority penalty.
pub fn compute_zpd(
    competence: f32,
    difficulty: f32,
    config: &MetacognitionConfig,
) -> (bool, i16) {
    let low = competence * config.zpd_low_factor;
    let high = competence * config.zpd_high_factor;

    if difficulty >= low && difficulty <= high {
        (true, config.zpd_boost)
    } else if difficulty > high {
        (false, -config.zpd_penalty)
    } else {
        // Below ZPD — too easy, no adjustment
        (false, 0)
    }
}

// ---------------------------------------------------------------------------
// E-graph goal reformulation
// ---------------------------------------------------------------------------

/// Create reformulation-specific rewrite rules for goal criteria simplification.
///
/// Supplements `reason::builtin_rules()` with domain-specific rules:
/// - relax-full-to-partial: full achievement → partial
/// - simplify-and: (and X (and Y Z)) → (and X Y) when Z is "trivial"
pub fn reformulation_rules() -> Vec<egg::Rewrite<crate::reason::AkhLang, ()>> {
    use egg::rewrite;
    vec![
        // (and (and ?a ?b) ?c) can be relaxed to (and ?a ?b) — drop outer clause
        rewrite!("relax-and-outer"; "(and (and ?a ?b) ?c)" => "(and ?a ?b)"),
        // (and ?a (and ?b ?c)) can be relaxed to (and ?a ?b) — drop inner tail
        rewrite!("relax-and-inner"; "(and ?a (and ?b ?c))" => "(and ?a ?b)"),
    ]
}

/// Attempt to reformulate goal criteria via e-graph equality saturation.
///
/// Parses criteria into an AkhLang expression, runs equality saturation with
/// reformulation rules, and extracts the simplest equivalent via AstSize.
/// Returns `None` if parsing fails (criteria too free-form for s-expression).
pub fn reformulate_via_egraph(criteria: &str) -> Option<String> {
    use egg::{AstSize, Extractor, Runner};

    // Try to parse criteria as an s-expression.
    let expr: egg::RecExpr<crate::reason::AkhLang> = criteria.parse().ok()?;
    let original_cost = {
        let runner = Runner::<crate::reason::AkhLang, ()>::default()
            .with_expr(&expr)
            .run(&[]);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (cost, _) = extractor.find_best(runner.roots[0]);
        cost
    };

    // Build combined rules: builtin + reformulation-specific.
    let mut rules = crate::reason::builtin_rules();
    rules.extend(reformulation_rules());

    let runner = Runner::default()
        .with_expr(&expr)
        .with_iter_limit(10)
        .with_node_limit(5000)
        .run(&rules);

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (cost, best) = extractor.find_best(runner.roots[0]);

    // Only return if actually simplified.
    if cost < original_cost {
        Some(best.to_string())
    } else {
        None
    }
}

/// Fallback reformulation: drop the hardest clause from criteria.
///
/// Splits criteria on comma/and, scores each clause by keyword match count
/// against existing KG symbols, drops the lowest-scoring clause.
pub fn relax_criteria(criteria: &str, engine: &Engine) -> String {
    let clauses: Vec<&str> = criteria
        .split(',')
        .flat_map(|p| p.split(" and "))
        .map(|p| p.trim())
        .filter(|p| p.len() > 3)
        .collect();

    if clauses.len() <= 1 {
        return criteria.to_string();
    }

    // Score each clause by how many of its keywords match existing symbols.
    let scored: Vec<(&str, usize)> = clauses
        .iter()
        .map(|clause| {
            let score = clause
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .filter(|w| {
                    engine.lookup_symbol(w).is_ok()
                        || engine
                            .lookup_symbol(&w.to_lowercase())
                            .is_ok()
                })
                .count();
            (*clause, score)
        })
        .collect();

    // Drop the lowest-scoring clause.
    let min_score = scored.iter().map(|(_, s)| *s).min().unwrap_or(0);
    let remaining: Vec<&str> = scored
        .iter()
        .filter(|(_, s)| *s > min_score || scored.len() <= 2)
        .take(clauses.len() - 1)
        .map(|(c, _)| *c)
        .collect();

    if remaining.is_empty() {
        clauses[0].to_string()
    } else {
        remaining.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Orchestrator: evaluate_goals
// ---------------------------------------------------------------------------

/// Evaluate all active goals and produce metacognitive control signals.
///
/// For each active goal:
/// 1. Compute improvement rate (sliding window)
/// 2. Estimate difficulty from similar past goals
/// 3. Check ZPD bounds
/// 4. Run autoepistemic check
/// 5. Validate justification
/// 6. Search for similar failures
///
/// Returns evaluations and control signals.
pub fn evaluate_goals(
    goals: &[Goal],
    working_memory: &WorkingMemory,
    competence: &CompetenceModel,
    failure_index: &FailureIndex,
    engine: &Engine,
    config: &MetacognitionConfig,
    current_cycle: u64,
) -> (Vec<GoalEvaluation>, Vec<MetacognitiveControl>) {
    let mut evaluations = Vec::new();
    let mut signals = Vec::new();

    for goal in goals {
        if !matches!(goal.status, GoalStatus::Active) {
            continue;
        }

        let category = categorize_goal(goal);
        let comp = competence.category_competence(&category);
        let improvement_rate =
            compute_improvement_rate(goal, working_memory, current_cycle, config.progress_window);
        let difficulty = estimate_difficulty(goal, goals);
        let (in_zpd, zpd_adjustment) = compute_zpd(comp, difficulty, config);

        // Autoepistemic check.
        let autoepistemic = autoepistemic_check(goal, improvement_rate, difficulty, config);
        let should_question = autoepistemic.is_some();
        let questioning_reason = autoepistemic.unwrap_or_default();

        // Search for similar failures (encode current goal context).
        let similar_failures = if goal.cycles_worked > 0 {
            let query = format!("{} {}", goal.description, goal.success_criteria);
            let query_bytes = simple_text_hash(&query);
            failure_index
                .search_similar(&query_bytes, config.failure_search_k)
                .into_iter()
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Compute diminishing returns: improvement rate < previous window would indicate decline.
        let diminishing_returns = improvement_rate == 0.0 && goal.cycles_worked >= config.progress_window as u32;

        let eval = GoalEvaluation {
            goal_id: goal.symbol_id,
            improvement_rate,
            diminishing_returns,
            estimated_difficulty: difficulty,
            competence: comp,
            in_zpd,
            zpd_adjustment,
            should_question,
            questioning_reason: questioning_reason.clone(),
            similar_failures: similar_failures.clone(),
        };
        evaluations.push(eval);

        // Generate control signals.

        // ZPD adjustment.
        if zpd_adjustment != 0 {
            signals.push(MetacognitiveControl::AdjustZpd {
                goal_id: goal.symbol_id,
                delta: zpd_adjustment,
            });
        }

        // Autoepistemic questioning.
        if should_question {
            // Check entrenchment before deciding action.
            let entrenchment = goal
                .justification
                .as_ref()
                .map(|j| j.entrenchment())
                .unwrap_or(0);

            if entrenchment <= 1 {
                // Low entrenchment — try to reformulate first.
                let relaxed = reformulate_via_egraph(&goal.success_criteria)
                    .unwrap_or_else(|| relax_criteria(&goal.success_criteria, engine));

                if relaxed != goal.success_criteria {
                    signals.push(MetacognitiveControl::SuggestReformulate {
                        goal_id: goal.symbol_id,
                        relaxed_criteria: relaxed,
                    });
                } else {
                    signals.push(MetacognitiveControl::QuestionGoal {
                        goal_id: goal.symbol_id,
                        reason: questioning_reason.clone(),
                    });
                }
            } else {
                // Higher entrenchment — only question, never auto-suspend.
                signals.push(MetacognitiveControl::QuestionGoal {
                    goal_id: goal.symbol_id,
                    reason: questioning_reason.clone(),
                });
            }
        }

        // Justification validation.
        if let Some(ref justification) = goal.justification {
            if !is_justification_valid(justification, goals, engine) {
                signals.push(MetacognitiveControl::InvalidateJustification {
                    goal_id: goal.symbol_id,
                    reason: format!(
                        "Justification no longer holds for goal \"{}\"",
                        goal.description.chars().take(40).collect::<String>()
                    ),
                });
            }
        }
    }

    (evaluations, signals)
}

/// Simple text-to-bytes hash for failure pattern encoding.
///
/// When the full VSA encode pipeline is unavailable (no Engine/VsaOps/ItemMemory
/// accessible), uses a deterministic hash-based approach.
pub(crate) fn simple_text_hash(text: &str) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate a 1250-byte vector (10000 bits) from text hash chain.
    let mut bytes = vec![0u8; 1250]; // 10000 bits
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        let h_bytes = h.to_le_bytes();
        for (j, &b) in h_bytes.iter().enumerate() {
            let idx = (i * 8 + j) % bytes.len();
            bytes[idx] ^= b;
        }
    }
    bytes
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

    fn make_goal(id: u64, desc: &str, cycles_worked: u32) -> Goal {
        Goal {
            symbol_id: sym(id),
            description: desc.into(),
            status: GoalStatus::Active,
            priority: 128,
            success_criteria: String::new(),
            parent: None,
            children: Vec::new(),
            created_at: 0,
            cycles_worked,
            last_progress_cycle: 0,
            source: None,
            blocked_by: Vec::new(),
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
        }
    }

    #[test]
    fn update_competence_tracks_tool() {
        let mut model = CompetenceModel::default();
        model.update_tool("kg_query", true);
        model.update_tool("kg_query", true);
        model.update_tool("kg_query", false);
        assert_eq!(model.tools["kg_query"].successes, 2);
        assert_eq!(model.tools["kg_query"].attempts, 3);
        // Laplace: (2+1)/(3+2) = 0.6
        let rate = model.tool_success_rate("kg_query");
        assert!((rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn update_competence_tracks_category() {
        let mut model = CompetenceModel::default();
        model.update_category("gap:explore", true);
        model.update_category("gap:explore", false);
        assert_eq!(model.categories["gap:explore"].completed, 1);
        assert_eq!(model.categories["gap:explore"].attempted, 2);
    }

    #[test]
    fn calibration_error_perfect() {
        let mut model = CompetenceModel::default();
        model.record_calibration(1.0, 1.0, 100);
        model.record_calibration(0.0, 0.0, 100);
        assert_eq!(model.calibration_error(), 0.0);
    }

    #[test]
    fn calibration_error_overconfident() {
        let mut model = CompetenceModel::default();
        // Predicted 1.0, actual 0.0 → error = 1.0
        model.record_calibration(1.0, 0.0, 100);
        assert!((model.calibration_error() - 1.0).abs() < 0.01);
    }

    #[test]
    fn goal_competence_laplace() {
        let model = CompetenceModel::default();
        // Unknown category: Laplace gives 0.5
        assert!((model.category_competence("unknown") - 0.5).abs() < 0.01);
    }

    #[test]
    fn categorize_goal_from_source() {
        let mut goal = make_goal(1, "explore the galaxy structures", 0);
        goal.source = Some(GoalSource::DriveExceeded {
            drive: "curiosity".into(),
            strength: 0.8,
        });
        assert_eq!(categorize_goal(&goal), "drive:curiosity");

        let goal2 = make_goal(2, "find knowledge gaps in astronomy", 0);
        let cat = categorize_goal(&goal2);
        assert!(cat.starts_with("user:"));
    }

    #[test]
    fn failure_index_insert_and_search() {
        let mut index = FailureIndex::new();
        let case = FailureCase {
            goal_description: "explore stars".into(),
            tool_name: "kg_query".into(),
            error_message: "no triples found".into(),
            cycle: 5,
            resolved: false,
            resolution_tool: None,
            vector: simple_text_hash("explore stars kg_query no triples found"),
        };
        index.insert(case);
        assert_eq!(index.len(), 1);

        // Search with similar text.
        let query = simple_text_hash("explore stars kg_query error");
        let results = index.search_similar(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].goal_description, "explore stars");
    }

    #[test]
    fn failure_case_serialization_roundtrip() {
        let case = FailureCase {
            goal_description: "test goal".into(),
            tool_name: "reason".into(),
            error_message: "parse failed".into(),
            cycle: 10,
            resolved: true,
            resolution_tool: Some("kg_query".into()),
            vector: vec![1, 2, 3, 4],
        };
        let bytes = bincode::serialize(&case).unwrap();
        let restored: FailureCase = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.goal_description, "test goal");
        assert_eq!(restored.resolved, true);
        assert_eq!(restored.resolution_tool, Some("kg_query".into()));
    }

    #[test]
    fn estimate_difficulty_from_similar_goals() {
        let goal = make_goal(1, "explore the star systems", 0);
        let completed = Goal {
            symbol_id: sym(2),
            description: "explore the star clusters".into(),
            status: GoalStatus::Completed,
            cycles_worked: 25,
            ..make_goal(2, "", 0)
        };
        let difficulty = estimate_difficulty(&goal, &[goal.clone(), completed]);
        // "star" overlaps, difficulty = 25/50 = 0.5 (weighted by similarity)
        assert!(difficulty > 0.0 && difficulty <= 1.0);
    }

    #[test]
    fn autoepistemic_triggers_for_stalled_goal() {
        let goal = make_goal(1, "find all galaxies", 31);
        let config = MetacognitionConfig::default();
        let result = autoepistemic_check(&goal, 0.0, 0.3, &config);
        assert!(result.is_some());
        assert!(result.unwrap().contains("questioning"));
    }

    #[test]
    fn autoepistemic_not_triggered_early() {
        let goal = make_goal(1, "find all galaxies", 2);
        let config = MetacognitionConfig::default();
        let result = autoepistemic_check(&goal, 0.5, 0.3, &config);
        assert!(result.is_none());
    }

    #[test]
    fn zpd_boost_in_sweet_spot() {
        let config = MetacognitionConfig::default();
        let (in_zpd, adj) = compute_zpd(0.5, 0.5, &config);
        assert!(in_zpd);
        assert_eq!(adj, config.zpd_boost);
    }

    #[test]
    fn zpd_penalty_beyond_competence() {
        let config = MetacognitionConfig::default();
        let (in_zpd, adj) = compute_zpd(0.3, 0.9, &config);
        assert!(!in_zpd);
        assert_eq!(adj, -config.zpd_penalty);
    }

    #[test]
    fn reformulate_via_egraph_simplifies() {
        // (and (and x y) z) should simplify to (and x y)
        let result = reformulate_via_egraph("(and (and x y) z)");
        assert!(result.is_some());
        let simplified = result.unwrap();
        // Should be shorter than the original
        assert!(simplified.len() < "(and (and x y) z)".len());
    }

    #[test]
    fn cascade_invalidation_follows_entrenchment() {
        let parent = make_goal(1, "parent goal", 10);
        let mut child_low = make_goal(2, "child low entrenchment", 5);
        child_low.justification = Some(GoalJustification::DefaultAssumption {
            rationale: "just because".into(),
        });
        let mut child_decomposed = make_goal(3, "child decomposed", 5);
        child_decomposed.justification = Some(GoalJustification::DecomposedFrom {
            parent: sym(1),
        });

        let goals = vec![parent, child_low, child_decomposed];
        let signals = cascade_invalidation(sym(1), &goals);

        // Only child_decomposed has DecomposedFrom pointing to parent.
        assert_eq!(signals.len(), 1);
        if let MetacognitiveControl::InvalidateJustification { goal_id, .. } = &signals[0] {
            assert_eq!(*goal_id, sym(3));
        } else {
            panic!("Expected InvalidateJustification");
        }
    }

    #[test]
    fn failure_index_rebuild_from_cases() {
        let cases = vec![
            FailureCase {
                goal_description: "goal a".into(),
                tool_name: "tool1".into(),
                error_message: "err1".into(),
                cycle: 1,
                resolved: false,
                resolution_tool: None,
                vector: simple_text_hash("goal a tool1 err1"),
            },
            FailureCase {
                goal_description: "goal b".into(),
                tool_name: "tool2".into(),
                error_message: "err2".into(),
                cycle: 2,
                resolved: false,
                resolution_tool: None,
                vector: simple_text_hash("goal b tool2 err2"),
            },
        ];
        let index = FailureIndex::from_cases(cases);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn relax_criteria_drops_clause() {
        // Without engine, all keyword lookups fail → all clauses score 0.
        // Should still drop one clause.
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let result = relax_criteria("find stars, map galaxies, count nebulae", &engine);
        // Should have fewer clauses than the original.
        let original_clauses = 3;
        let result_clauses = result.split(',').count();
        assert!(result_clauses < original_clauses);
    }

    #[test]
    fn competence_model_serialization_roundtrip() {
        let mut model = CompetenceModel::default();
        model.update_tool("kg_query", true);
        model.update_category("gap:explore", true);
        model.record_calibration(0.8, 1.0, 100);

        let bytes = bincode::serialize(&model).unwrap();
        let restored: CompetenceModel = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.tools["kg_query"].successes, 1);
        assert_eq!(restored.categories["gap:explore"].completed, 1);
        assert_eq!(restored.calibration_history.len(), 1);
    }
}
