//! Procedural learning via chunking (Phase 11h).
//!
//! Implements Soar-inspired compilation of successful goal-solving traces
//! into reusable `LearnedMethod`s that integrate with the HTN decomposition
//! registry. When the agent repeatedly solves similar goals with the same
//! tool sequence, it compiles the trace into a method, refining and retracting
//! methods based on empirical success/failure.
//!
//! # Architecture
//!
//! - **Trace extraction**: pull tool invocation sequences from WM `Decision` entries.
//! - **Generalization**: replace concrete SymbolIds with type-based patterns via KG hierarchy.
//! - **Compilation**: encode preconditions as VSA vector, bundle steps into a `LearnedMethod`.
//! - **MethodIndex**: HNSW-backed similarity index for fast retrieval.
//! - **Lifecycle**: success/failure tracking, retraction, dormancy pruning.

use std::sync::RwLock;

use anndists::dist::DistHamming;
use hnsw_rs::prelude::*;
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::decomposition::{DecompositionMethod, DecompositionStrategy, SubtaskTemplate};
use super::goal::Goal;
use super::memory::{WorkingMemory, WorkingMemoryKind};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to the chunking/procedural-learning subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum ChunkingError {
    /// Could not extract a tool trace from working memory for the given goal.
    #[error("trace extraction failed for goal \"{goal_description}\": {reason}")]
    #[diagnostic(
        code(akh::agent::chunking::trace_extraction_failed),
        help("ensure the goal was executed with at least 2 OODA cycles before attempting compilation")
    )]
    TraceExtractionFailed {
        goal_description: String,
        reason: String,
    },

    /// Generalization of a trace into abstract steps failed.
    #[error("generalization failed for trace of {step_count} steps: {reason}")]
    #[diagnostic(
        code(akh::agent::chunking::generalization_failed),
        help("the goal's tool trace may use tools not present in the KG hierarchy")
    )]
    GeneralizationFailed { step_count: usize, reason: String },

    /// The learned method library has reached its configured maximum.
    #[error("method library full: {current}/{max} methods")]
    #[diagnostic(
        code(akh::agent::chunking::library_full),
        help("prune dormant or low-confidence methods to make room")
    )]
    LibraryFull { current: usize, max: usize },
}

/// Result type alias for the chunking subsystem.
pub type ChunkingResult<T> = Result<T, ChunkingError>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single step in a generalized method (tool invocation pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralizedStep {
    /// Tool name (e.g., "kg_query", "reason").
    pub tool: String,
    /// Argument pattern (generalized from concrete inputs).
    pub arg_pattern: String,
    /// Expected outcome pattern.
    pub expected_outcome: String,
}

/// A learned method compiled from successful goal traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedMethod {
    /// Unique identifier in the symbol table.
    pub id: SymbolId,
    /// Human-readable name.
    pub name: String,
    /// VSA-encoded precondition vector for similarity search.
    pub precondition_vector: Vec<u8>,
    /// Natural-language precondition pattern.
    pub precondition_pattern: String,
    /// Ordered generalized steps.
    pub steps: Vec<GeneralizedStep>,
    /// Confidence in this method (0.0–1.0).
    pub confidence: f32,
    /// How many times this method has been used.
    pub usage_count: u32,
    /// How many times this method led to goal success.
    pub success_count: u32,
    /// How many times this method led to goal failure.
    pub failure_count: u32,
    /// Cycle at which this method was learned.
    pub learned_at: u64,
    /// Cycle at which this method was last used.
    pub last_used: u64,
    /// Description of the source goal from which this method was compiled.
    pub source_goal_description: String,
}

impl LearnedMethod {
    /// Empirical success rate [0.0, 1.0].
    pub fn success_rate(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            self.confidence
        } else {
            self.success_count as f32 / total as f32
        }
    }
}

/// Configuration for the chunking subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Minimum trace length required for compilation.
    pub min_trace_length: usize,
    /// Maximum number of learned methods in the library.
    pub max_methods: usize,
    /// Number of sessions without use before a method is considered dormant.
    pub dormant_sessions: u64,
    /// Success rate below which a method is retracted.
    pub retraction_threshold: f32,
    /// Number of similar traces required before compilation is triggered.
    pub similarity_compilation_count: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            min_trace_length: 2,
            max_methods: 100,
            dormant_sessions: 10,
            retraction_threshold: 0.2,
            similarity_compilation_count: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// MethodIndex (HNSW-backed)
// ---------------------------------------------------------------------------

/// HNSW-backed index of learned methods for similarity-based retrieval.
///
/// Follows the `FailureIndex` / `EffortIndex` pattern.
pub struct MethodIndex {
    /// HNSW index for nearest-neighbor method lookup.
    hnsw: RwLock<Hnsw<'static, u32, DistHamming>>,
    /// All stored methods (for persistence and result lookup).
    methods: Vec<LearnedMethod>,
    /// Next HNSW point ID.
    next_id: usize,
}

impl std::fmt::Debug for MethodIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MethodIndex")
            .field("methods", &self.methods.len())
            .finish()
    }
}

impl MethodIndex {
    /// Create a new empty method index.
    pub fn new() -> Self {
        let max_elements = 1000;
        let max_layer = 8;
        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});
        Self {
            hnsw: RwLock::new(hnsw),
            methods: Vec::new(),
            next_id: 0,
        }
    }

    /// Insert a learned method into the index.
    pub fn insert(&mut self, method: LearnedMethod) {
        let data_u32 = bytes_to_u32_vec(&method.precondition_vector);
        let id = self.next_id;
        self.next_id += 1;
        if let Ok(hnsw) = self.hnsw.read() {
            hnsw.insert((&data_u32, id));
        }
        self.methods.push(method);
    }

    /// Search for the k most similar learned methods.
    pub fn search_similar(&self, query_vec: &[u8], k: usize) -> Vec<&LearnedMethod> {
        if self.methods.is_empty() || k == 0 {
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
            .filter_map(|n| self.methods.get(n.d_id))
            .collect()
    }

    /// Number of stored methods.
    pub fn len(&self) -> usize {
        self.methods.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.methods.is_empty()
    }

    /// Get all methods (for persistence).
    pub fn methods(&self) -> &[LearnedMethod] {
        &self.methods
    }

    /// Get a mutable reference to a method by its SymbolId.
    pub fn get_mut(&mut self, id: SymbolId) -> Option<&mut LearnedMethod> {
        self.methods.iter_mut().find(|m| m.id == id)
    }

    /// Remove a method by its SymbolId. Returns whether it was found.
    pub fn remove(&mut self, id: SymbolId) -> bool {
        let before = self.methods.len();
        self.methods.retain(|m| m.id != id);
        self.methods.len() < before
        // Note: HNSW index is not rebuilt — stale entries are harmless and
        // the index is rebuilt from scratch on persistence/restore.
    }

    /// Rebuild from persisted methods.
    pub fn from_methods(methods: Vec<LearnedMethod>) -> Self {
        let max_elements = methods.len().max(1000);
        let max_layer = (max_elements as f64).log2().ceil() as usize;
        let max_layer = max_layer.max(4).min(16);
        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});

        // Re-insert all methods into the HNSW index.
        for (id, method) in methods.iter().enumerate() {
            let data_u32 = bytes_to_u32_vec(&method.precondition_vector);
            hnsw.insert((&data_u32, id));
        }

        let next_id = methods.len();
        Self {
            hnsw: RwLock::new(hnsw),
            methods,
            next_id,
        }
    }
}

impl Default for MethodIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trace extraction
// ---------------------------------------------------------------------------

/// A raw trace step extracted from working memory.
#[derive(Debug, Clone)]
pub struct TraceStep {
    /// Tool name.
    pub tool: String,
    /// Summary of the tool's input.
    pub input_summary: String,
    /// Cycle at which this step occurred.
    pub cycle: u64,
}

/// Extract the tool invocation trace for a goal from working memory.
///
/// Parses `Decision` entries with format `"Decide: tool=X, goal=\"Y\", ..."`.
pub fn extract_trace(goal_id: SymbolId, wm: &WorkingMemory) -> ChunkingResult<Vec<TraceStep>> {
    // Match goal references precisely: look for `goal="N"` pattern.
    let goal_pattern = format!("goal=\"{}\"", goal_id.get());
    let mut steps = Vec::new();

    for entry in wm.by_kind(WorkingMemoryKind::Decision) {
        // Match entries that reference this goal precisely.
        if !entry.content.contains(&goal_pattern) {
            continue;
        }

        // Parse "Decide: tool=X, goal=\"Y\", reasoning=\"Z\""
        if let Some(rest) = entry.content.strip_prefix("Decide: tool=") {
            if let Some(comma) = rest.find(',') {
                let tool = rest[..comma].to_string();
                // Extract a brief input summary from the reasoning.
                let input_summary = rest
                    .find("reasoning=\"")
                    .and_then(|start| {
                        let begin = start + 11;
                        rest[begin..].find('"').map(|end| rest[begin..begin + end].to_string())
                    })
                    .unwrap_or_default();

                steps.push(TraceStep {
                    tool,
                    input_summary,
                    cycle: entry.source_cycle,
                });
            }
        }
    }

    if steps.len() < 2 {
        return Err(ChunkingError::TraceExtractionFailed {
            goal_description: format!("goal:{}", goal_id.get()),
            reason: format!("only {} steps found, need at least 2", steps.len()),
        });
    }

    // Sort by cycle order.
    steps.sort_by_key(|s| s.cycle);
    Ok(steps)
}

// ---------------------------------------------------------------------------
// Generalization
// ---------------------------------------------------------------------------

/// Generalize a concrete trace into abstract `GeneralizedStep`s.
///
/// Replaces specific SymbolIds in input summaries with type patterns
/// (e.g., `sym:42` → `{Entity}`) using the KG hierarchy.
pub fn generalize_trace(
    trace: &[TraceStep],
    engine: &Engine,
) -> ChunkingResult<Vec<GeneralizedStep>> {
    if trace.is_empty() {
        return Err(ChunkingError::GeneralizationFailed {
            step_count: 0,
            reason: "empty trace".into(),
        });
    }

    let mut steps = Vec::with_capacity(trace.len());

    for step in trace {
        // Generalize the input summary by replacing SymbolId references
        // with their type category from the KG.
        let arg_pattern = generalize_input(&step.input_summary, engine);
        let expected_outcome = format!("{}_result", step.tool);

        steps.push(GeneralizedStep {
            tool: step.tool.clone(),
            arg_pattern,
            expected_outcome,
        });
    }

    Ok(steps)
}

/// Replace concrete SymbolId references in an input string with type patterns.
fn generalize_input(input: &str, engine: &Engine) -> String {
    let mut result = input.to_string();

    // Find numeric patterns that could be SymbolIds and try to resolve their type.
    for word in input.split_whitespace() {
        if let Ok(num) = word.parse::<u64>() {
            if let Some(sym_id) = SymbolId::new(num) {
                if let Ok(meta) = engine.get_symbol_meta(sym_id) {
                    // Use the symbol's kind label as the type pattern.
                    let kind_label = format!("{{{}}}", meta.kind);
                    result = result.replace(word, &kind_label);
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Compile a generalized trace into a `LearnedMethod`.
pub fn compile_method(
    goal: &Goal,
    steps: Vec<GeneralizedStep>,
    engine: &Engine,
    cycle: u64,
) -> LearnedMethod {
    // Encode precondition as a VSA-like vector from the goal description.
    let precondition_vector = simple_text_hash(&goal.description);
    let precondition_pattern = goal.description.clone();

    // Generate a descriptive name from the tool sequence.
    let tool_sequence: Vec<&str> = steps.iter().map(|s| s.tool.as_str()).collect();
    let name = format!("learned:{}", tool_sequence.join("→"));

    // Use the engine to mint a fresh SymbolId for the method.
    let id = engine
        .resolve_or_create_entity(&format!("agent:method:{name}"))
        .unwrap_or_else(|_| SymbolId::new(1).unwrap());

    LearnedMethod {
        id,
        name,
        precondition_vector,
        precondition_pattern,
        steps,
        confidence: 0.5,
        usage_count: 0,
        success_count: 0,
        failure_count: 0,
        learned_at: cycle,
        last_used: cycle,
        source_goal_description: goal.description.clone(),
    }
}

/// Full pipeline: extract trace → generalize → compile into a `LearnedMethod`.
///
/// Returns `None` if the trace is too short or extraction fails.
pub fn chunk_completed_goal(
    goal: &Goal,
    wm: &WorkingMemory,
    engine: &Engine,
    config: &ChunkingConfig,
    cycle: u64,
) -> Option<LearnedMethod> {
    let trace = extract_trace(goal.symbol_id, wm).ok()?;

    if trace.len() < config.min_trace_length {
        return None;
    }

    let steps = generalize_trace(&trace, engine).ok()?;
    Some(compile_method(goal, steps, engine, cycle))
}

// ---------------------------------------------------------------------------
// Success / failure tracking
// ---------------------------------------------------------------------------

/// Record a successful use of a learned method.
pub fn record_success(method: &mut LearnedMethod) {
    method.usage_count += 1;
    method.success_count += 1;
    // Boost confidence toward empirical rate.
    method.confidence = method.success_rate();
}

/// Record a failed use of a learned method.
///
/// Returns `true` if the method should be retracted (success rate below threshold).
pub fn record_failure(method: &mut LearnedMethod, threshold: f32) -> bool {
    method.usage_count += 1;
    method.failure_count += 1;
    method.confidence = method.success_rate();
    method.success_rate() < threshold
}

// ---------------------------------------------------------------------------
// HTN integration
// ---------------------------------------------------------------------------

/// Convert a `LearnedMethod` into a `DecompositionMethod` for the HTN registry.
pub fn to_decomposition_method(learned: &LearnedMethod) -> DecompositionMethod {
    let subtask_templates: Vec<SubtaskTemplate> = learned
        .steps
        .iter()
        .enumerate()
        .map(|(i, step)| SubtaskTemplate {
            description_template: format!("Step {}: {} with {}", i + 1, step.tool, step.arg_pattern),
            priority_offset: -(i as i8 * 5),
            criteria_template: step.expected_outcome.clone(),
        })
        .collect();

    // Sequential ordering: each step depends on the previous.
    let ordering: Vec<(usize, usize)> = (0..subtask_templates.len().saturating_sub(1))
        .map(|i| (i, i + 1))
        .collect();

    // Build keyword hints from tool names and the precondition pattern.
    let mut keyword_hints: Vec<String> = learned
        .steps
        .iter()
        .map(|s| s.tool.clone())
        .collect();
    // Add a few words from the precondition pattern.
    for word in learned.precondition_pattern.split_whitespace().take(3) {
        let w = word.to_lowercase();
        if w.len() > 2 && !keyword_hints.contains(&w) {
            keyword_hints.push(w);
        }
    }

    DecompositionMethod {
        name: learned.name.clone(),
        strategy: DecompositionStrategy::Custom(format!("learned@{}", learned.learned_at)),
        precondition_sparql: String::new(),
        keyword_hints,
        subtask_templates,
        ordering,
        semantic_vector: None,
        usage_count: learned.usage_count,
        success_rate: learned.success_rate(),
    }
}

// ---------------------------------------------------------------------------
// Dormancy pruning
// ---------------------------------------------------------------------------

/// Remove methods that have not been used for `max_dormant` sessions.
///
/// `session_cycles` is the approximate number of cycles per session.
/// Returns the SymbolIds of pruned methods.
pub fn prune_dormant(
    methods: &mut Vec<LearnedMethod>,
    cycle: u64,
    session_cycles: u64,
    max_dormant: u64,
) -> Vec<SymbolId> {
    let dormancy_cutoff = cycle.saturating_sub(max_dormant * session_cycles.max(1));
    let mut pruned = Vec::new();

    methods.retain(|m| {
        if m.last_used < dormancy_cutoff && m.usage_count > 0 {
            pruned.push(m.id);
            false
        } else {
            true
        }
    });

    pruned
}

// ---------------------------------------------------------------------------
// Trace similarity
// ---------------------------------------------------------------------------

/// Check whether two traces have the same tool sequence.
pub fn traces_similar(trace_a: &[TraceStep], trace_b: &[TraceStep]) -> bool {
    if trace_a.len() != trace_b.len() {
        return false;
    }
    trace_a
        .iter()
        .zip(trace_b.iter())
        .all(|(a, b)| a.tool == b.tool)
}

// ---------------------------------------------------------------------------
// Compilation opportunity detection
// ---------------------------------------------------------------------------

/// Detect whether recent completed goals present compilation opportunities.
///
/// ACT-R analog: triggers when `similarity_compilation_count` goals solved
/// with similar tool sequences are found.
pub fn detect_compilation_opportunity(
    recent_goals: &[Goal],
    wm: &WorkingMemory,
    config: &ChunkingConfig,
) -> Vec<SymbolId> {
    // Collect traces from recently completed goals.
    let mut traces: Vec<(SymbolId, Vec<TraceStep>)> = Vec::new();

    for goal in recent_goals {
        if let Ok(trace) = extract_trace(goal.symbol_id, wm) {
            if trace.len() >= config.min_trace_length {
                traces.push((goal.symbol_id, trace));
            }
        }
    }

    // Find groups of similar traces.
    let mut candidates = Vec::new();
    for i in 0..traces.len() {
        let mut similar_count = 1;
        for j in (i + 1)..traces.len() {
            if traces_similar(&traces[i].1, &traces[j].1) {
                similar_count += 1;
            }
        }
        if similar_count >= config.similarity_compilation_count {
            candidates.push(traces[i].0);
        }
    }

    candidates
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode text as a u8 vector for HNSW precondition matching.
///
/// Uses a deterministic hash-based encoding (same approach as `metacognition::simple_text_hash`).
fn simple_text_hash(text: &str) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let dim = 256;
    let mut vec = vec![0u8; dim];
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        let idx = (h as usize) % dim;
        vec[idx] ^= (h & 0xFF) as u8;

        let mut hasher2 = DefaultHasher::new();
        h.hash(&mut hasher2);
        let h2 = hasher2.finish();
        let idx2 = (h2 as usize) % dim;
        vec[idx2] ^= (h2 & 0xFF) as u8;
    }

    vec
}

/// Convert a u8 slice to u32 vector for HNSW distance computation.
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut arr = [0u8; 4];
            arr[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(arr)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::goal::GoalStatus;
    use crate::agent::memory::{WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind};
    use crate::symbol::SymbolId;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    fn make_goal(id: u64, desc: &str) -> Goal {
        Goal {
            symbol_id: sym(id),
            description: desc.into(),
            status: GoalStatus::Completed,
            priority: 128,
            success_criteria: "done".into(),
            parent: None,
            children: vec![],
            created_at: 0,
            cycles_worked: 5,
            last_progress_cycle: 4,
            source: None,
            blocked_by: vec![],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        }
    }

    fn make_wm_with_decisions(goal_id: u64, tools: &[&str]) -> WorkingMemory {
        let mut wm = WorkingMemory::new(100);
        for (i, tool) in tools.iter().enumerate() {
            let content = format!(
                "Decide: tool={tool}, goal=\"{goal_id}\", reasoning=\"step {i}\""
            );
            let _ = wm.push(WorkingMemoryEntry {
                id: 0,
                content,
                symbols: vec![],
                kind: WorkingMemoryKind::Decision,
                timestamp: 0,
                relevance: 1.0,
                source_cycle: i as u64 + 1,
                reference_count: 0,
                access_timestamps: vec![],
            });
        }
        wm
    }

    // -- Trace extraction --

    #[test]
    fn extract_trace_success() {
        let wm = make_wm_with_decisions(1, &["kg_query", "reason", "kg_mutate"]);
        let trace = extract_trace(sym(1), &wm).unwrap();
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].tool, "kg_query");
        assert_eq!(trace[1].tool, "reason");
        assert_eq!(trace[2].tool, "kg_mutate");
    }

    #[test]
    fn extract_trace_too_short() {
        let wm = make_wm_with_decisions(1, &["kg_query"]);
        let result = extract_trace(sym(1), &wm);
        assert!(result.is_err());
    }

    #[test]
    fn extract_trace_wrong_goal() {
        let wm = make_wm_with_decisions(1, &["kg_query", "reason"]);
        let result = extract_trace(sym(999), &wm);
        assert!(result.is_err());
    }

    // -- Generalization --

    #[test]
    fn generalize_trace_produces_steps() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let trace = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "find concepts".into(), cycle: 1 },
            TraceStep { tool: "reason".into(), input_summary: "analyze".into(), cycle: 2 },
        ];

        let steps = generalize_trace(&trace, &engine).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].tool, "kg_query");
        assert_eq!(steps[1].tool, "reason");
        assert_eq!(steps[0].expected_outcome, "kg_query_result");
    }

    #[test]
    fn generalize_empty_trace_errors() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let result = generalize_trace(&[], &engine);
        assert!(result.is_err());
    }

    // -- Compilation --

    #[test]
    fn compile_method_creates_valid_method() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let goal = make_goal(1, "explore knowledge graph");
        let steps = vec![
            GeneralizedStep { tool: "kg_query".into(), arg_pattern: "find {Entity}".into(), expected_outcome: "kg_query_result".into() },
            GeneralizedStep { tool: "reason".into(), arg_pattern: "analyze results".into(), expected_outcome: "reason_result".into() },
        ];

        let method = compile_method(&goal, steps, &engine, 10);
        assert!(method.name.contains("kg_query"));
        assert!(method.name.contains("reason"));
        assert_eq!(method.steps.len(), 2);
        assert_eq!(method.learned_at, 10);
        assert!((method.confidence - 0.5).abs() < f32::EPSILON);
    }

    // -- Full pipeline --

    #[test]
    fn chunk_completed_goal_pipeline() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let goal = make_goal(1, "learn about types");
        let wm = make_wm_with_decisions(1, &["kg_query", "reason", "kg_mutate"]);
        let config = ChunkingConfig::default();

        let method = chunk_completed_goal(&goal, &wm, &engine, &config, 5);
        assert!(method.is_some());
        let m = method.unwrap();
        assert_eq!(m.steps.len(), 3);
    }

    #[test]
    fn chunk_completed_goal_too_short() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let goal = make_goal(1, "simple task");
        let wm = make_wm_with_decisions(1, &["kg_query"]);
        let config = ChunkingConfig::default();

        let method = chunk_completed_goal(&goal, &wm, &engine, &config, 5);
        assert!(method.is_none());
    }

    // -- Success/failure tracking --

    #[test]
    fn record_success_updates_stats() {
        let mut method = LearnedMethod {
            id: sym(1),
            name: "test".into(),
            precondition_vector: vec![0; 256],
            precondition_pattern: "test".into(),
            steps: vec![],
            confidence: 0.5,
            usage_count: 0,
            success_count: 0,
            failure_count: 0,
            learned_at: 0,
            last_used: 0,
            source_goal_description: "test".into(),
        };

        record_success(&mut method);
        assert_eq!(method.usage_count, 1);
        assert_eq!(method.success_count, 1);
        assert!((method.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn record_failure_triggers_retraction() {
        let mut method = LearnedMethod {
            id: sym(1),
            name: "test".into(),
            precondition_vector: vec![0; 256],
            precondition_pattern: "test".into(),
            steps: vec![],
            confidence: 0.5,
            usage_count: 4,
            success_count: 0,
            failure_count: 4,
            learned_at: 0,
            last_used: 0,
            source_goal_description: "test".into(),
        };

        let should_retract = record_failure(&mut method, 0.2);
        assert!(should_retract);
        assert_eq!(method.failure_count, 5);
    }

    #[test]
    fn record_failure_no_retraction_above_threshold() {
        let mut method = LearnedMethod {
            id: sym(1),
            name: "test".into(),
            precondition_vector: vec![0; 256],
            precondition_pattern: "test".into(),
            steps: vec![],
            confidence: 0.5,
            usage_count: 3,
            success_count: 3,
            failure_count: 0,
            learned_at: 0,
            last_used: 0,
            source_goal_description: "test".into(),
        };

        let should_retract = record_failure(&mut method, 0.2);
        assert!(!should_retract);
    }

    // -- Dormancy pruning --

    #[test]
    fn prune_dormant_removes_old_methods() {
        let mut methods = vec![
            LearnedMethod {
                id: sym(1),
                name: "old".into(),
                precondition_vector: vec![0; 256],
                precondition_pattern: "old".into(),
                steps: vec![],
                confidence: 0.5,
                usage_count: 1,
                success_count: 1,
                failure_count: 0,
                learned_at: 0,
                last_used: 10,
                source_goal_description: "old".into(),
            },
            LearnedMethod {
                id: sym(2),
                name: "recent".into(),
                precondition_vector: vec![0; 256],
                precondition_pattern: "recent".into(),
                steps: vec![],
                confidence: 0.8,
                usage_count: 5,
                success_count: 4,
                failure_count: 1,
                learned_at: 90,
                last_used: 95,
                source_goal_description: "recent".into(),
            },
        ];

        let pruned = prune_dormant(&mut methods, 100, 10, 5);
        // "old" was last used at cycle 10, dormancy cutoff = 100 - 5*10 = 50 → pruned.
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], sym(1));
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name, "recent");
    }

    // -- Trace similarity --

    #[test]
    fn traces_similar_same_sequence() {
        let a = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "a".into(), cycle: 1 },
            TraceStep { tool: "reason".into(), input_summary: "b".into(), cycle: 2 },
        ];
        let b = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "x".into(), cycle: 3 },
            TraceStep { tool: "reason".into(), input_summary: "y".into(), cycle: 4 },
        ];
        assert!(traces_similar(&a, &b));
    }

    #[test]
    fn traces_similar_different_sequence() {
        let a = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "a".into(), cycle: 1 },
            TraceStep { tool: "reason".into(), input_summary: "b".into(), cycle: 2 },
        ];
        let b = vec![
            TraceStep { tool: "reason".into(), input_summary: "x".into(), cycle: 3 },
            TraceStep { tool: "kg_query".into(), input_summary: "y".into(), cycle: 4 },
        ];
        assert!(!traces_similar(&a, &b));
    }

    #[test]
    fn traces_similar_different_length() {
        let a = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "a".into(), cycle: 1 },
        ];
        let b = vec![
            TraceStep { tool: "kg_query".into(), input_summary: "x".into(), cycle: 3 },
            TraceStep { tool: "reason".into(), input_summary: "y".into(), cycle: 4 },
        ];
        assert!(!traces_similar(&a, &b));
    }

    // -- MethodIndex --

    #[test]
    fn method_index_empty() {
        let idx = MethodIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn method_index_insert_and_search() {
        let mut idx = MethodIndex::new();
        let method = LearnedMethod {
            id: sym(1),
            name: "test".into(),
            precondition_vector: simple_text_hash("explore knowledge"),
            precondition_pattern: "explore knowledge".into(),
            steps: vec![],
            confidence: 0.5,
            usage_count: 0,
            success_count: 0,
            failure_count: 0,
            learned_at: 0,
            last_used: 0,
            source_goal_description: "explore knowledge".into(),
        };
        idx.insert(method);
        assert_eq!(idx.len(), 1);

        let query = simple_text_hash("explore knowledge graph");
        let results = idx.search_similar(&query, 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn method_index_from_methods_roundtrip() {
        let methods = vec![
            LearnedMethod {
                id: sym(1),
                name: "a".into(),
                precondition_vector: simple_text_hash("method a"),
                precondition_pattern: "method a".into(),
                steps: vec![],
                confidence: 0.5,
                usage_count: 0,
                success_count: 0,
                failure_count: 0,
                learned_at: 0,
                last_used: 0,
                source_goal_description: "a".into(),
            },
            LearnedMethod {
                id: sym(2),
                name: "b".into(),
                precondition_vector: simple_text_hash("method b"),
                precondition_pattern: "method b".into(),
                steps: vec![],
                confidence: 0.8,
                usage_count: 3,
                success_count: 2,
                failure_count: 1,
                learned_at: 5,
                last_used: 10,
                source_goal_description: "b".into(),
            },
        ];
        let idx = MethodIndex::from_methods(methods);
        assert_eq!(idx.len(), 2);
    }

    // -- to_decomposition_method --

    #[test]
    fn to_decomposition_method_converts() {
        let learned = LearnedMethod {
            id: sym(1),
            name: "learned:kg_query→reason".into(),
            precondition_vector: vec![0; 256],
            precondition_pattern: "explore knowledge".into(),
            steps: vec![
                GeneralizedStep {
                    tool: "kg_query".into(),
                    arg_pattern: "find {Entity}".into(),
                    expected_outcome: "kg_query_result".into(),
                },
                GeneralizedStep {
                    tool: "reason".into(),
                    arg_pattern: "analyze results".into(),
                    expected_outcome: "reason_result".into(),
                },
            ],
            confidence: 0.8,
            usage_count: 5,
            success_count: 4,
            failure_count: 1,
            learned_at: 10,
            last_used: 20,
            source_goal_description: "explore knowledge".into(),
        };

        let dm = to_decomposition_method(&learned);
        assert_eq!(dm.name, "learned:kg_query→reason");
        assert_eq!(dm.subtask_templates.len(), 2);
        assert_eq!(dm.ordering.len(), 1);
        assert_eq!(dm.ordering[0], (0, 1));
        assert_eq!(dm.usage_count, 5);
        assert!((dm.success_rate - 0.8).abs() < f32::EPSILON);
    }

    // -- Serialization --

    #[test]
    fn learned_method_serde_roundtrip() {
        let method = LearnedMethod {
            id: sym(42),
            name: "test_method".into(),
            precondition_vector: vec![1, 2, 3, 4],
            precondition_pattern: "explore things".into(),
            steps: vec![GeneralizedStep {
                tool: "kg_query".into(),
                arg_pattern: "find {Entity}".into(),
                expected_outcome: "kg_query_result".into(),
            }],
            confidence: 0.75,
            usage_count: 10,
            success_count: 8,
            failure_count: 2,
            learned_at: 5,
            last_used: 15,
            source_goal_description: "explore things".into(),
        };

        let bytes = bincode::serialize(&method).unwrap();
        let restored: LearnedMethod = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.name, "test_method");
        assert_eq!(restored.steps.len(), 1);
        assert_eq!(restored.usage_count, 10);
        assert_eq!(restored.id, sym(42));
    }

    // -- Compilation opportunity detection --

    #[test]
    fn detect_compilation_opportunity_finds_similar() {
        let goals = vec![
            make_goal(1, "learn about types"),
            make_goal(2, "learn about concepts"),
        ];

        let mut wm = WorkingMemory::new(100);
        // Add decision entries for both goals with same tool sequence.
        for goal_id in &[1u64, 2u64] {
            for (i, tool) in ["kg_query", "reason", "kg_mutate"].iter().enumerate() {
                let _ = wm.push(WorkingMemoryEntry {
                    id: 0,
                    content: format!("Decide: tool={tool}, goal=\"{goal_id}\", reasoning=\"step {i}\""),
                    symbols: vec![],
                    kind: WorkingMemoryKind::Decision,
                    timestamp: 0,
                    relevance: 1.0,
                    source_cycle: (goal_id * 10) + i as u64,
                    reference_count: 0,
                    access_timestamps: vec![],
                });
            }
        }

        let config = ChunkingConfig::default();
        let candidates = detect_compilation_opportunity(&goals, &wm, &config);
        assert!(!candidates.is_empty());
    }
}
