//! OODA loop: Observe → Orient → Decide → Act cycle.
//!
//! Each cycle gathers state, builds context, chooses an action, and executes it.
//! The agent's working memory is updated throughout the cycle.

use std::collections::{HashMap, HashSet};

use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::agent::Agent;
use super::error::{AgentError, AgentResult};
use super::goal::{self, Goal, GoalStatus};
use super::memory::{EpisodicEntry, WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind};
use super::tool::{ToolInput, ToolOutput};
use super::tool_semantics::{encode_criteria, encode_goal_semantics};

// ---------------------------------------------------------------------------
// OODA cycle types
// ---------------------------------------------------------------------------

/// Result of a single OODA cycle.
#[derive(Debug, Clone)]
pub struct OodaCycleResult {
    /// Which cycle this was.
    pub cycle_number: u64,
    /// What was observed.
    pub observation: Observation,
    /// How the observation was interpreted.
    pub orientation: Orientation,
    /// What action was chosen.
    pub decision: Decision,
    /// Result of the chosen action.
    pub action_result: ActionResult,
}

/// State gathered during the Observe phase.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Active goal symbol IDs.
    pub active_goals: Vec<SymbolId>,
    /// Current working memory size.
    pub working_memory_size: usize,
    /// WM entry IDs from the most recent cycle.
    pub recent_entries: Vec<u64>,
    /// Recalled episodic memories (full entries, not just IDs).
    pub recalled_episodes: Vec<EpisodicEntry>,
}

/// Context built during the Orient phase.
#[derive(Debug, Clone)]
pub struct Orientation {
    /// KG triples relevant to current goals.
    pub relevant_knowledge: Vec<Triple>,
    /// Inferences from spreading activation.
    pub inferences: Vec<(SymbolId, f32)>,
    /// Memory pressure (0.0–1.0).
    pub memory_pressure: f32,
}

/// Action chosen during the Decide phase.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Which tool was selected.
    pub chosen_tool: String,
    /// Input constructed for the tool.
    pub tool_input: ToolInput,
    /// Why this action was chosen.
    pub reasoning: String,
    /// Which goal this serves.
    pub goal_id: SymbolId,
}

/// Result of executing the chosen action.
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// Tool output.
    pub tool_output: ToolOutput,
    /// How the goal progressed.
    pub goal_progress: GoalProgress,
    /// WM entries created during this cycle.
    pub new_wm_entries: Vec<u64>,
}

/// How a goal progressed as a result of the action.
#[derive(Debug, Clone)]
pub enum GoalProgress {
    NoChange,
    Advanced { detail: String },
    Completed,
    Failed { reason: String },
}

// ---------------------------------------------------------------------------
// Impasse detection
// ---------------------------------------------------------------------------

/// A decision impasse: the agent couldn't choose a tool with confidence.
#[derive(Debug, Clone)]
pub struct DecisionImpasse {
    /// Which goal was being targeted.
    pub goal_id: SymbolId,
    /// What kind of impasse occurred.
    pub kind: ImpasseKind,
    /// Best score among all candidates.
    pub best_score: f32,
}

/// What kind of decision impasse occurred.
#[derive(Debug, Clone)]
pub enum ImpasseKind {
    /// All tool candidates scored below the usability threshold.
    AllBelowThreshold { threshold: f32 },
    /// Top two candidates tied (within epsilon).
    Tie {
        tool_a: String,
        tool_b: String,
        epsilon: f32,
    },
}

// ---------------------------------------------------------------------------
// OODA cycle implementation
// ---------------------------------------------------------------------------

/// Run one full OODA cycle on the agent.
pub fn run_ooda_cycle(agent: &mut Agent) -> AgentResult<OodaCycleResult> {
    agent.cycle_count += 1;
    let cycle = agent.cycle_count;

    // ── Observe ─────────────────────────────────────────────────────────
    let observation = observe(agent, cycle)?;

    // ── Orient ──────────────────────────────────────────────────────────
    let orientation = orient(agent, &observation)?;

    // ── Decide ──────────────────────────────────────────────────────────
    let decision = decide(agent, &observation, &orientation, cycle)?;

    // ── Act ─────────────────────────────────────────────────────────────
    let action_result = act(agent, &decision, cycle)?;

    Ok(OodaCycleResult {
        cycle_number: cycle,
        observation,
        orientation,
        decision,
        action_result,
    })
}

/// Observe: gather current state.
fn observe(agent: &mut Agent, cycle: u64) -> AgentResult<Observation> {
    let active = goal::active_goals(&agent.goals);
    let active_goals: Vec<SymbolId> = active.iter().map(|g| g.symbol_id).collect();

    let wm_size = agent.working_memory.len();
    let recent: Vec<u64> = agent
        .working_memory
        .recent(5)
        .iter()
        .map(|e| e.id)
        .collect();

    // Recall relevant episodes if we have active goals.
    let mut recalled = Vec::new();
    if !active_goals.is_empty() {
        if let Ok(episodes) =
            super::memory::recall_episodes(&agent.engine, &active_goals, &agent.predicates, 3)
        {
            recalled = episodes;
        }
    }

    // Push observation to WM.
    let obs_content = format!(
        "Cycle {}: {} active goals, {} WM entries, {} recalled episodes",
        cycle,
        active_goals.len(),
        wm_size,
        recalled.len()
    );
    let _ = agent.working_memory.push(WorkingMemoryEntry {
        id: 0,
        content: obs_content,
        symbols: active_goals.clone(),
        kind: WorkingMemoryKind::Observation,
        timestamp: 0,
        relevance: 0.6,
        source_cycle: cycle,
        reference_count: 0,
    });

    Ok(Observation {
        active_goals,
        working_memory_size: wm_size,
        recent_entries: recent,
        recalled_episodes: recalled,
    })
}

/// Orient: build context for the current goals.
fn orient(agent: &mut Agent, observation: &Observation) -> AgentResult<Orientation> {
    let mut relevant_knowledge = Vec::new();
    let mut inferences = Vec::new();

    // For each active goal, gather adjacent KG triples.
    for goal_id in &observation.active_goals {
        let from = agent.engine.triples_from(*goal_id);
        let to = agent.engine.triples_to(*goal_id);
        relevant_knowledge.extend(from);
        relevant_knowledge.extend(to);

        // Try inference from goal symbol.
        let query = crate::infer::InferenceQuery {
            seeds: vec![*goal_id],
            top_k: 5,
            max_depth: 1,
            ..Default::default()
        };
        if let Ok(result) = agent.engine.infer(&query) {
            inferences.extend(result.activations);
        }
    }

    // Incorporate knowledge from recalled episodic memories.
    // Episodes carry `learnings` — symbols the agent previously found relevant.
    // Gather triples around those learned symbols for richer context.
    for episode in &observation.recalled_episodes {
        for learned_sym in &episode.learnings {
            let from = agent.engine.triples_from(*learned_sym);
            relevant_knowledge.extend(from);
        }
    }

    let memory_pressure = agent.working_memory.pressure();

    // Push orientation summary to WM if we found knowledge.
    if !relevant_knowledge.is_empty() || !inferences.is_empty() {
        let orient_content = format!(
            "Orient: {} relevant triples, {} inferences, {} episodes, pressure {:.2}",
            relevant_knowledge.len(),
            inferences.len(),
            observation.recalled_episodes.len(),
            memory_pressure,
        );
        let syms: Vec<SymbolId> = inferences.iter().map(|(s, _)| *s).collect();
        let _ = agent.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: orient_content,
            symbols: syms,
            kind: WorkingMemoryKind::Inference,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: agent.cycle_count,
            reference_count: 0,
        });
    }

    Ok(Orientation {
        relevant_knowledge,
        inferences,
        memory_pressure,
    })
}

/// Decide: choose a tool and construct input based on the top-priority goal.
fn decide(
    agent: &mut Agent,
    observation: &Observation,
    orientation: &Orientation,
    cycle: u64,
) -> AgentResult<Decision> {
    // Get the top-priority active goal, skipping any that are blocked.
    let active = goal::active_goals(&agent.goals);
    let top_goal = active
        .iter()
        .find(|g| !g.is_blocked(&agent.goals))
        .or(active.first())
        .ok_or(AgentError::NoGoals)?;
    let goal_id = top_goal.symbol_id;
    let goal_desc = &top_goal.description;

    // Increment reference counts on WM entries we're consulting for this decision.
    for entry_id in &observation.recent_entries {
        agent.working_memory.increment_reference(*entry_id);
    }

    // Rule-based strategy to select tool + build input.
    let (tool_name, tool_input, reasoning, impasse) = select_tool(
        top_goal,
        observation,
        orientation,
        &agent.engine,
        &agent.working_memory,
        agent.psyche.as_ref(),
        &agent.tool_registry,
    );

    // Store impasse on agent for goal generation to pick up.
    agent.last_impasse = impasse;

    // Record decision in WM.
    let dec_content =
        format!("Decide: tool={tool_name}, goal=\"{goal_desc}\", reason={reasoning}",);
    let _ = agent.working_memory.push(WorkingMemoryEntry {
        id: 0,
        content: dec_content,
        symbols: vec![goal_id],
        kind: WorkingMemoryKind::Decision,
        timestamp: 0,
        relevance: 0.7,
        source_cycle: cycle,
        reference_count: 0,
    });

    // Store provenance for the decision.
    let mut prov = ProvenanceRecord::new(
        goal_id,
        DerivationKind::AgentDecision {
            goal: goal_id,
            cycle,
        },
    )
    .with_confidence(0.8);
    let _ = agent.engine.store_provenance(&mut prov);

    Ok(Decision {
        chosen_tool: tool_name,
        tool_input,
        reasoning,
        goal_id,
    })
}

// ---------------------------------------------------------------------------
// Utility-based tool selection
// ---------------------------------------------------------------------------

/// Tool usage history for a specific goal, extracted from working memory.
struct GoalToolHistory {
    /// How many times each tool was used for this goal.
    usage_counts: HashMap<String, usize>,
    /// Ordered list of tools used (most recent last), for recency tracking.
    recent_tools: Vec<String>,
}

impl GoalToolHistory {
    /// Extract tool history for a goal from working memory Decision entries.
    fn from_working_memory(wm: &WorkingMemory, goal_id: SymbolId) -> Self {
        let mut usage_counts: HashMap<String, usize> = HashMap::new();
        let mut recent_tools = Vec::new();

        let mut decisions: Vec<&WorkingMemoryEntry> = wm
            .by_kind(WorkingMemoryKind::Decision)
            .into_iter()
            .filter(|e| e.symbols.contains(&goal_id))
            .collect();
        decisions.sort_by_key(|e| e.id);

        for entry in decisions {
            if let Some(tool_name) = parse_tool_from_decision(&entry.content) {
                *usage_counts.entry(tool_name.clone()).or_insert(0) += 1;
                recent_tools.push(tool_name);
            }
        }

        Self {
            usage_counts,
            recent_tools,
        }
    }

    /// How many times a tool has been used for this goal.
    fn count(&self, tool: &str) -> usize {
        self.usage_counts.get(tool).copied().unwrap_or(0)
    }

    /// Whether a tool has ever been used for this goal.
    fn ever_used(&self, tool: &str) -> bool {
        self.count(tool) > 0
    }

    /// How recently a tool was used (0 = most recent, 1 = one before, etc).
    /// Returns None if never used for this goal.
    fn recency(&self, tool: &str) -> Option<usize> {
        self.recent_tools.iter().rev().position(|t| t == tool)
    }
}

/// Parse tool name from a Decision WM entry content string.
fn parse_tool_from_decision(content: &str) -> Option<String> {
    content
        .strip_prefix("Decide: tool=")
        .and_then(|s| s.find(',').map(|i| s[..i].to_string()))
}

/// A scored tool candidate with utility breakdown.
struct ToolCandidate {
    name: String,
    input: ToolInput,
    /// How appropriate this tool is given the current state [0.0, 1.0].
    base_score: f32,
    /// Penalty for recent use on the same goal [0.0, 0.5].
    recency_penalty: f32,
    /// Bonus for being an unexplored tool for this goal [0.0, 0.2].
    novelty_bonus: f32,
    /// Bonus from episodic memory suggesting this tool worked before [0.0, 0.2].
    episodic_bonus: f32,
    /// Bonus for being appropriate given memory pressure [0.0, 0.2].
    pressure_bonus: f32,
    /// Bonus from the Jungian psyche archetype weights.
    archetype_bonus: f32,
    /// ACT-R utility scaling: computed_priority / 255.0 (Phase 11c).
    goal_value_factor: f32,
    /// Why this tool was considered.
    reasoning: String,
}

impl ToolCandidate {
    fn total_score(&self) -> f32 {
        let raw = (self.base_score - self.recency_penalty
            + self.novelty_bonus
            + self.episodic_bonus
            + self.pressure_bonus
            + self.archetype_bonus)
            .max(0.0);
        // ACT-R utility scaling: 0.5× at priority 0, 1.0× at priority 255.
        raw * (0.5 + 0.5 * self.goal_value_factor)
    }

    fn new(name: &str, input: ToolInput, base_score: f32, reasoning: String) -> Self {
        Self {
            name: name.into(),
            input,
            base_score,
            recency_penalty: 0.0,
            novelty_bonus: 0.0,
            episodic_bonus: 0.0,
            pressure_bonus: 0.0,
            archetype_bonus: 0.0,
            goal_value_factor: 1.0, // default: no scaling
            reasoning,
        }
    }
}

/// Extract tool names mentioned in recalled episodic memory summaries.
fn extract_episodic_tool_hints(episodes: &[EpisodicEntry]) -> HashSet<String> {
    let mut tools = HashSet::new();
    for ep in episodes {
        // Summaries contain WM content like "summary:Tool result (kg_query):..."
        // or "summary:Decide: tool=kg_query, ..."
        let text = ep.summary.strip_prefix("summary:").unwrap_or(&ep.summary);
        if let Some(name) = text
            .strip_prefix("Tool result (")
            .and_then(|s| s.find(')').map(|i| s[..i].to_string()))
        {
            tools.insert(name);
        }
        if let Some(name) = parse_tool_from_decision(text) {
            tools.insert(name);
        }
    }
    tools
}

/// Compute recency penalty for a tool given its per-goal history.
///
/// Most recently used: 0.4, second-most-recent: 0.2, third: 0.1, beyond: 0.0.
fn compute_recency_penalty(tool: &str, history: &GoalToolHistory) -> f32 {
    match history.recency(tool) {
        Some(0) => 0.4,
        Some(1) => 0.2,
        Some(2) => 0.1,
        _ => 0.0,
    }
}

/// Apply modifiers (recency, novelty, episodic, pressure) to a base candidate.
fn apply_modifiers(
    mut c: ToolCandidate,
    history: &GoalToolHistory,
    episodic_tools: &HashSet<String>,
    memory_pressure: f32,
) -> ToolCandidate {
    c.recency_penalty = compute_recency_penalty(&c.name, history);

    if !history.ever_used(&c.name) {
        c.novelty_bonus = 0.15;
    }

    if episodic_tools.contains(&c.name) {
        c.episodic_bonus = 0.2;
    }

    // memory_recall benefits from high pressure; others don't.
    if c.name == "memory_recall" && memory_pressure > 0.7 {
        c.pressure_bonus = 0.15;
    }

    c
}

/// Try to synthesize a new triple from orientation context.
///
/// Looks for an inference symbol that isn't already connected to the goal,
/// and proposes connecting it via a relation found in existing knowledge.
fn synthesize_triple(
    goal: &Goal,
    orientation: &Orientation,
    engine: &crate::engine::Engine,
) -> Option<(String, String, String)> {
    let connected: HashSet<SymbolId> = orientation
        .relevant_knowledge
        .iter()
        .flat_map(|t| [t.subject, t.object])
        .collect();

    let novel_inference = orientation
        .inferences
        .iter()
        .find(|(sym, _)| !connected.contains(sym) && *sym != goal.symbol_id)?;

    let predicate = orientation
        .relevant_knowledge
        .first()
        .map(|t| t.predicate)?;

    let subject = engine.resolve_label(goal.symbol_id);
    let pred_label = engine.resolve_label(predicate);
    let object = engine.resolve_label(novel_inference.0);

    Some((subject, pred_label, object))
}

/// Utility-based tool selection.
///
/// Scores all 5 tools based on: state-dependent base score, recency penalty
/// (avoids repeating tools on the same goal), novelty bonus (explores
/// untried tools), episodic bonus (replicates past successful strategies),
/// and memory pressure bonus. Picks the highest-scored candidate.
fn select_tool(
    goal: &Goal,
    observation: &Observation,
    orientation: &Orientation,
    engine: &crate::engine::Engine,
    working_memory: &WorkingMemory,
    psyche: Option<&crate::compartment::psyche::Psyche>,
    tool_registry: &super::tool::ToolRegistry,
) -> (String, ToolInput, String, Option<DecisionImpasse>) {
    let history = GoalToolHistory::from_working_memory(working_memory, goal.symbol_id);
    let episodic_tools = extract_episodic_tool_hints(&observation.recalled_episodes);
    let pressure = orientation.memory_pressure;

    let goal_label = engine.resolve_label(goal.symbol_id);
    let has_knowledge = !orientation.relevant_knowledge.is_empty();
    let has_inferences = !orientation.inferences.is_empty();
    let has_episodes = !observation.recalled_episodes.is_empty();

    // Detect code-related queries for smarter tool selection.
    let is_code_query = is_code_goal(&goal.description);

    // Pre-compute unexplored child once to avoid redundant WM scans.
    let unexplored_child = if is_code_query {
        find_unexplored_code_child(working_memory, engine)
    } else {
        None
    };

    let mut candidates: Vec<ToolCandidate> = Vec::new();

    // ── kg_query: always applicable, most valuable when knowledge is scarce ──
    {
        // For code queries, prefer unexplored child entities discovered in
        // previous kg_query results, then fall back to top-level match.
        let query_symbol = if is_code_query {
            unexplored_child
                .clone()
                .or_else(|| find_code_entity_for_query(engine, &goal.description))
                .unwrap_or_else(|| {
                    orientation
                        .inferences
                        .iter()
                        .find(|(sym, _)| {
                            let triples = engine.triples_from(*sym);
                            triples.len() < 3 && *sym != goal.symbol_id
                        })
                        .map(|(sym, _)| engine.resolve_label(*sym))
                        .unwrap_or_else(|| goal_label.clone())
                })
        } else {
            // Pick a query target: prefer an unexplored inference symbol, else goal.
            orientation
                .inferences
                .iter()
                .find(|(sym, _)| {
                    let triples = engine.triples_from(*sym);
                    triples.len() < 3 && *sym != goal.symbol_id
                })
                .map(|(sym, _)| engine.resolve_label(*sym))
                .unwrap_or_else(|| goal_label.clone())
        };

        let base = if is_code_query {
            if has_knowledge { 0.7 } else { 0.85 }
        } else if has_knowledge {
            0.4
        } else {
            0.8
        };
        let reason = if is_code_query {
            format!("Code query — querying KG for \"{query_symbol}\".")
        } else if has_knowledge {
            format!("Deeper KG exploration of \"{query_symbol}\".")
        } else {
            format!("No knowledge yet — querying KG for \"{query_symbol}\".")
        };

        let mut kg_candidate = apply_modifiers(
            ToolCandidate::new(
                "kg_query",
                ToolInput::new()
                    .with_param("symbol", &query_symbol)
                    .with_param("direction", "both"),
                base,
                reason,
            ),
            &history,
            &episodic_tools,
            pressure,
        );

        // When exploring code with an unexplored child, override recency penalty.
        // The agent found a NEW entity to query — querying a different entity isn't repetition.
        if is_code_query && unexplored_child.is_some() {
            kg_candidate.recency_penalty = 0.0;
        }

        candidates.push(kg_candidate);
    }

    // ── kg_mutate: only when we can synthesize a new triple ──
    if has_knowledge && has_inferences {
        if let Some((subj, pred, obj)) = synthesize_triple(goal, orientation, engine) {
            candidates.push(apply_modifiers(
                ToolCandidate::new(
                    "kg_mutate",
                    ToolInput::new()
                        .with_param("subject", &subj)
                        .with_param("predicate", &pred)
                        .with_param("object", &obj)
                        .with_param("confidence", "0.7"),
                    0.7,
                    format!("Synthesizing new triple: {subj} -> {pred} -> {obj}."),
                ),
                &history,
                &episodic_tools,
                pressure,
            ));
        }
    }

    // ── memory_recall: only when episodes exist ──
    if has_episodes {
        let query_syms: Vec<String> = observation
            .recalled_episodes
            .iter()
            .flat_map(|ep| ep.learnings.iter())
            .take(3)
            .map(|s| engine.resolve_label(*s))
            .collect();
        let query_str = if query_syms.is_empty() {
            goal_label.clone()
        } else {
            query_syms.join(",")
        };

        let base = 0.55;
        candidates.push(apply_modifiers(
            ToolCandidate::new(
                "memory_recall",
                ToolInput::new()
                    .with_param("query_symbols", &query_str)
                    .with_param("top_k", "3"),
                base,
                "Recalling past experience for this goal.".into(),
            ),
            &history,
            &episodic_tools,
            pressure,
        ));
    }

    // ── reason: only when knowledge triples exist to reason about ──
    if has_knowledge {
        let t = &orientation.relevant_knowledge[0];
        // Sanitise labels so each is a single s-expression token for egg.
        // egg's parser splits on whitespace, so spaces/special chars in KG
        // labels (e.g. "VSA module", "code:defines-fn") must become valid
        // atomic symbols.
        let sanitize = |label: String| -> String {
            label
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        };
        let expr = format!(
            "(triple {} {} {})",
            sanitize(engine.resolve_label(t.subject)),
            sanitize(engine.resolve_label(t.predicate)),
            sanitize(engine.resolve_label(t.object)),
        );

        candidates.push(apply_modifiers(
            ToolCandidate::new(
                "reason",
                ToolInput::new().with_param("expression", &expr),
                0.5,
                "Applying symbolic reasoning to known triples.".into(),
            ),
            &history,
            &episodic_tools,
            pressure,
        ));
    }

    // ── similarity_search: only when inferences give us symbols to explore ──
    // Suppress for code queries (mostly finds metadata like priority:128).
    if has_inferences {
        let top_sym = orientation.inferences[0].0;
        let top_label = engine.resolve_label(top_sym);

        let sim_base = if is_code_query { 0.25 } else { 0.55 };

        candidates.push(apply_modifiers(
            ToolCandidate::new(
                "similarity_search",
                ToolInput::new()
                    .with_param("symbol", &top_label)
                    .with_param("top_k", "5"),
                sim_base,
                format!("Exploring similar symbols around \"{top_label}\"."),
            ),
            &history,
            &episodic_tools,
            pressure,
        ));
    }

    // ── VSA-based tool scoring for remaining tools ──
    // Encode the goal as a hypervector, then score each tool via semantic similarity.
    // This replaces all keyword-matching blocks with interference-based scoring.
    {
        let ops = engine.ops();
        let im = engine.item_memory();

        if let Ok(goal_vec) =
            encode_goal_semantics(&goal.description, &goal.success_criteria, engine, ops, im)
        {
            // Score tools by semantic similarity to the goal vector.
            let vsa_tools: &[(&str, &[&str])] = &[
                (
                    "file_io",
                    &[
                        "file",
                        "read",
                        "write",
                        "save",
                        "export",
                        "data",
                        "disk",
                        "load",
                        "document",
                        "import",
                        "open",
                        "close",
                        "directory",
                        "folder",
                        "path",
                        "output",
                        "input",
                    ],
                ),
                (
                    "http_fetch",
                    &[
                        "http", "url", "fetch", "web", "api", "download", "request", "network",
                        "website", "page", "online", "internet", "get", "endpoint", "link",
                        "browse",
                    ],
                ),
                (
                    "shell_exec",
                    &[
                        "command", "shell", "execute", "run", "process", "script", "system",
                        "terminal", "bash", "program", "invoke", "launch", "pipe", "cli", "binary",
                    ],
                ),
                (
                    "infer_rules",
                    &[
                        "infer",
                        "deduce",
                        "derive",
                        "transitive",
                        "type",
                        "hierarchy",
                        "classify",
                        "forward",
                        "chain",
                        "reason",
                        "logic",
                        "imply",
                        "conclude",
                        "rule",
                        "ontology",
                        "propagate",
                    ],
                ),
                (
                    "gap_analysis",
                    &[
                        "gap",
                        "missing",
                        "incomplete",
                        "discover",
                        "explore",
                        "what",
                        "unknown",
                        "coverage",
                        "lack",
                        "absent",
                        "need",
                        "require",
                        "insufficient",
                        "sparse",
                    ],
                ),
                (
                    "user_interact",
                    &[
                        "ask", "user", "input", "question", "interact", "human", "prompt",
                        "dialog", "clarify", "confirm", "respond", "answer", "feedback", "help",
                    ],
                ),
                (
                    "content_ingest",
                    &[
                        "ingest",
                        "document",
                        "book",
                        "pdf",
                        "epub",
                        "html",
                        "article",
                        "website",
                        "library",
                        "read",
                        "parse",
                        "content",
                        "import",
                        "fetch",
                        "download",
                        "add",
                        "store",
                        "learn",
                        "absorb",
                        "paper",
                        "capture",
                        "save",
                        "publication",
                    ],
                ),
                (
                    "library_search",
                    &[
                        "search",
                        "library",
                        "find",
                        "document",
                        "paragraph",
                        "content",
                        "lookup",
                        "recall",
                        "retrieve",
                        "what",
                        "about",
                        "said",
                        "mention",
                        "topic",
                        "learn",
                        "reference",
                        "quote",
                        "knowledge",
                        "look",
                        "paper",
                        "book",
                        "article",
                    ],
                ),
            ];

            for (tool_name, concepts) in vsa_tools {
                let profile_vec =
                    match crate::vsa::grounding::bundle_symbols(engine, ops, im, concepts) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                let semantic_score = ops.similarity(&goal_vec, &profile_vec).unwrap_or(0.5);

                // Apply context-aware adjustments on top of raw semantic score.
                let (adjusted_base, input, reason) = match *tool_name {
                    "file_io" => {
                        if semantic_score <= 0.55 {
                            continue;
                        }
                        (
                            semantic_score * 0.75,
                            ToolInput::new().with_param("action", "read").with_param(
                                "path",
                                &format!("{}.txt", goal_label.replace(' ', "_")),
                            ),
                            format!("VSA semantic match for file I/O: {semantic_score:.3}"),
                        )
                    }
                    "http_fetch" => {
                        if semantic_score <= 0.55 {
                            continue;
                        }
                        (
                            semantic_score * 0.75,
                            ToolInput::new().with_param("url", "https://example.com"),
                            format!("VSA semantic match for HTTP fetch: {semantic_score:.3}"),
                        )
                    }
                    "shell_exec" => {
                        if semantic_score <= 0.55 {
                            continue;
                        }
                        (
                            semantic_score * 0.75,
                            ToolInput::new()
                                .with_param("command", "echo 'agent shell exec'")
                                .with_param("timeout", "10"),
                            format!("VSA semantic match for shell exec: {semantic_score:.3}"),
                        )
                    }
                    "infer_rules" => {
                        if !has_knowledge {
                            continue;
                        }
                        let recent_infer_count = history.count("infer_rules");
                        let context_boost = if recent_infer_count == 0 { 0.1 } else { 0.0 };
                        // Suppress when code children still need exploring.
                        let base = if is_code_query && unexplored_child.is_some() {
                            0.1
                        } else {
                            // Always-available reasoning tool: use VSA score with a floor
                            (semantic_score * 0.7 + context_boost).max(0.35).min(0.75)
                        };
                        (
                            base,
                            ToolInput::new()
                                .with_param("max_iterations", "5")
                                .with_param("min_confidence", "0.1"),
                            format!("VSA inference relevance: {semantic_score:.3}"),
                        )
                    }
                    "gap_analysis" => {
                        let query_count = history.count("kg_query");
                        let stall_boost = if query_count >= 2 { 0.1 } else { 0.0 };
                        // Suppress when code children still need exploring.
                        let base = if is_code_query && unexplored_child.is_some() {
                            0.1
                        } else {
                            // Always-available analysis tool: use VSA score with a floor
                            (semantic_score * 0.7 + stall_boost).max(0.3).min(0.75)
                        };
                        (
                            base,
                            ToolInput::new()
                                .with_param("goal", &goal_label)
                                .with_param("max_gaps", "10"),
                            format!("VSA gap analysis relevance: {semantic_score:.3}"),
                        )
                    }
                    "user_interact" => {
                        if semantic_score <= 0.55 {
                            continue;
                        }
                        (
                            semantic_score * 0.75,
                            ToolInput::new().with_param(
                                "question",
                                &format!("What should I know about: {}?", goal.description),
                            ),
                            format!("VSA semantic match for user interaction: {semantic_score:.3}"),
                        )
                    }
                    "content_ingest" => {
                        if semantic_score <= 0.50 {
                            continue;
                        }
                        (
                            semantic_score * 0.75,
                            ToolInput::new().with_param("source", &goal_label),
                            format!("VSA semantic match for content ingest: {semantic_score:.3}"),
                        )
                    }
                    "library_search" => {
                        if semantic_score <= 0.50 {
                            continue;
                        }
                        (
                            semantic_score * 0.80,
                            ToolInput::new()
                                .with_param("query", &goal.description)
                                .with_param("top_k", "5"),
                            format!("VSA semantic match for library search: {semantic_score:.3}"),
                        )
                    }
                    _ => continue,
                };

                candidates.push(apply_modifiers(
                    ToolCandidate::new(tool_name, input, adjusted_base, reason),
                    &history,
                    &episodic_tools,
                    pressure,
                ));
            }

            // ── Generic VSA scoring for unscored registered tools ──
            // Scores tools via KG-grounded vectors (from skillpack triples)
            // or text-encoded fallback.  This makes tool selection data-driven:
            // tools only score high when the loaded skillpack creates semantic
            // links between tool names and goal-relevant concepts.
            {
                let already_scored: std::collections::HashSet<String> =
                    candidates.iter().map(|c| c.name.clone()).collect();

                for sig in tool_registry.list() {
                    if already_scored.contains(&sig.name) {
                        continue;
                    }

                    // Prefer KG-grounded vector (rich from skillpack triples),
                    // fall back to text encoding (sparse, low scores).
                    let profile_vec = if let Ok(sym_id) = engine.lookup_symbol(&sig.name) {
                        im.get_or_create(ops, sym_id)
                    } else {
                        match crate::vsa::grounding::encode_text_as_vector(
                            &format!("{} {}", sig.name, sig.description),
                            engine,
                            ops,
                            im,
                        ) {
                            Ok(v) => v,
                            Err(_) => continue,
                        }
                    };

                    let semantic_score =
                        ops.similarity(&goal_vec, &profile_vec).unwrap_or(0.5);
                    if semantic_score <= 0.55 {
                        continue;
                    }

                    // Build ToolInput from parameter schema (best-effort).
                    let mut input = ToolInput::new();
                    for param in &sig.parameters {
                        if param.required {
                            if param.name == "message" || param.name == "goal" {
                                input = input.with_param(&param.name, &goal.description);
                            } else if param.name == "name" || param.name == "workspace" {
                                let id = orientation
                                    .inferences
                                    .iter()
                                    .find_map(|(sym, _)| {
                                        let label = engine.resolve_label(*sym);
                                        if orientation.relevant_knowledge.iter().any(|t| {
                                            t.subject == *sym || t.object == *sym
                                        }) {
                                            Some(label.to_lowercase())
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or_else(|| goal_label.clone());
                                input = input.with_param(&param.name, &id);
                            }
                        }
                    }

                    candidates.push(apply_modifiers(
                        ToolCandidate::new(
                            &sig.name,
                            input,
                            semantic_score * 0.75,
                            format!(
                                "VSA semantic match for \"{}\": {semantic_score:.3}",
                                sig.name
                            ),
                        ),
                        &history,
                        &episodic_tools,
                        pressure,
                    ));
                }
            }
        }
    }

    // Apply psyche archetype bonus if available.
    if let Some(psyche) = psyche {
        for candidate in &mut candidates {
            candidate.archetype_bonus = psyche.archetype_bias(&candidate.name);
        }
    }

    // Apply ACT-R goal-value scaling from computed priority.
    let gvf = goal.computed_priority() as f32 / 255.0;
    for candidate in &mut candidates {
        candidate.goal_value_factor = gvf;
    }

    // Pick the highest-scored candidate.
    candidates.sort_by(|a, b| {
        b.total_score()
            .partial_cmp(&a.total_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Detect impasse before consuming candidates.
    let impasse = detect_impasse(&candidates, goal.symbol_id);

    if let Some(best) = candidates.into_iter().next() {
        let score = best.total_score();
        let reasoning = format!(
            "{} [score={:.2}: base={:.2} recency=-{:.2} novelty=+{:.2} episodic=+{:.2} pressure=+{:.2} archetype={:+.3} gvf={:.2}]",
            best.reasoning,
            score,
            best.base_score,
            best.recency_penalty,
            best.novelty_bonus,
            best.episodic_bonus,
            best.pressure_bonus,
            best.archetype_bonus,
            best.goal_value_factor,
        );
        (best.name, best.input, reasoning, impasse)
    } else {
        // Absolute fallback (no candidates at all — shouldn't happen).
        (
            "kg_query".into(),
            ToolInput::new()
                .with_param("symbol", &goal_label)
                .with_param("direction", "both"),
            "No candidates scored — falling back to KG query.".into(),
            impasse,
        )
    }
}

/// Detect whether the tool selection is in an impasse state.
///
/// Returns `Some(DecisionImpasse)` if:
/// - All candidates scored below 0.15 (nothing seems useful)
/// - Top two candidates are within 0.02 of each other (can't decide)
fn detect_impasse(candidates: &[ToolCandidate], goal_id: SymbolId) -> Option<DecisionImpasse> {
    if candidates.is_empty() {
        return Some(DecisionImpasse {
            goal_id,
            kind: ImpasseKind::AllBelowThreshold { threshold: 0.15 },
            best_score: 0.0,
        });
    }

    let best_score = candidates[0].total_score();

    // All below usability threshold.
    if best_score < 0.15 {
        return Some(DecisionImpasse {
            goal_id,
            kind: ImpasseKind::AllBelowThreshold { threshold: 0.15 },
            best_score,
        });
    }

    // Top two tied (within epsilon).
    if candidates.len() >= 2 {
        let second_score = candidates[1].total_score();
        let epsilon = 0.02;
        if (best_score - second_score).abs() < epsilon {
            return Some(DecisionImpasse {
                goal_id,
                kind: ImpasseKind::Tie {
                    tool_a: candidates[0].name.clone(),
                    tool_b: candidates[1].name.clone(),
                    epsilon,
                },
                best_score,
            });
        }
    }

    None
}

/// Act: execute the selected tool and update goal status.
fn act(agent: &mut Agent, decision: &Decision, cycle: u64) -> AgentResult<ActionResult> {
    // Shadow veto check: if the psyche vetoes this action, block it.
    if let Some(ref psyche) = agent.psyche {
        let action_desc = format!(
            "tool={} input={:?}",
            decision.chosen_tool, decision.tool_input
        );
        if let Some(manifest) = agent.tool_registry.manifest(&decision.chosen_tool) {
            // Structured veto check using manifest.
            if let Some(veto) = psyche.check_veto(manifest, &action_desc) {
                // Clone veto data before releasing the immutable borrow.
                let veto_name = veto.name.clone();
                let veto_explanation = veto.explanation.clone();
                let veto_severity = veto.severity;

                // Record provenance for the veto.
                let mut prov = ProvenanceRecord::new(
                    decision.goal_id,
                    DerivationKind::ShadowVeto {
                        pattern_name: veto_name.clone(),
                        severity: veto_severity,
                    },
                );
                let _ = agent.engine.store_provenance(&mut prov);

                // Record shadow encounter on psyche.
                if let Some(ref mut p) = agent.psyche {
                    p.record_shadow_encounter();
                }

                let veto_output = ToolOutput {
                    result: format!(
                        "VETOED by Shadow pattern '{}': {}",
                        veto_name, veto_explanation
                    ),
                    success: false,
                    symbols_involved: Vec::new(),
                };

                let wm_id = agent
                    .working_memory
                    .push(WorkingMemoryEntry {
                        id: 0,
                        content: format!(
                            "Tool result ({}):\n{}",
                            decision.chosen_tool, veto_output.result
                        ),
                        symbols: vec![decision.goal_id],
                        kind: WorkingMemoryKind::ToolResult,
                        timestamp: 0,
                        relevance: 0.8,
                        source_cycle: cycle,
                        reference_count: 0,
                    })
                    .ok();

                return Ok(ActionResult {
                    tool_output: veto_output,
                    goal_progress: GoalProgress::Failed {
                        reason: format!("Shadow veto: {}", veto_name),
                    },
                    new_wm_entries: wm_id.into_iter().collect(),
                });
            }

            // Structured bias check (non-blocking, just note it in reasoning).
            let bias = psyche.check_bias(manifest, &action_desc);
            if bias > 0.0 {
                tracing::debug!(
                    tool = %decision.chosen_tool,
                    shadow_bias = bias,
                    "Shadow bias applied (non-blocking)"
                );
            }
        }
    }

    // Tool errors should NOT be fatal to the OODA loop — convert them into a
    // failed ToolOutput so the agent can adapt (retry, switch tools, etc.).
    let tool_output = match agent.tool_registry.execute(
        &decision.chosen_tool,
        decision.tool_input.clone(),
        &agent.engine,
    ) {
        Ok(output) => output,
        Err(e) => ToolOutput {
            result: format!("{e}"),
            success: false,
            symbols_involved: Vec::new(),
        },
    };

    // Record tool result in WM (truncate very large results to keep WM manageable).
    let result_content = if tool_output.result.len() > 4096 {
        format!("{}...", &tool_output.result[..4096])
    } else {
        tool_output.result.clone()
    };
    let wm_id = agent
        .working_memory
        .push(WorkingMemoryEntry {
            id: 0,
            content: format!(
                "Tool result ({}):\n{}",
                decision.chosen_tool, result_content
            ),
            symbols: tool_output.symbols_involved.clone(),
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 0.6,
            source_cycle: cycle,
            reference_count: 0,
        })
        .ok();

    // Determine goal progress by evaluating success criteria.
    let goal_progress =
        if let Some(goal) = agent.goals.iter().find(|g| g.symbol_id == decision.goal_id) {
            evaluate_goal_progress(goal, &tool_output, &agent.engine)
        } else {
            GoalProgress::NoChange
        };

    // Update goal tracking and status.
    if let Some(goal) = agent
        .goals
        .iter_mut()
        .find(|g| g.symbol_id == decision.goal_id)
    {
        // Track cycles worked on this goal.
        goal.cycles_worked += 1;

        match &goal_progress {
            GoalProgress::Completed => {
                goal.last_progress_cycle = cycle;
                let _ = goal::update_goal_status(
                    &agent.engine,
                    goal,
                    GoalStatus::Completed,
                    &agent.predicates,
                );
            }
            GoalProgress::Advanced { .. } => {
                goal.last_progress_cycle = cycle;
            }
            GoalProgress::Failed { reason } => {
                let _ = goal::update_goal_status(
                    &agent.engine,
                    goal,
                    GoalStatus::Failed {
                        reason: reason.clone(),
                    },
                    &agent.predicates,
                );
            }
            GoalProgress::NoChange => {}
        }
    }

    Ok(ActionResult {
        tool_output,
        goal_progress,
        new_wm_entries: wm_id.into_iter().collect(),
    })
}

/// Extract meaningful keywords from a criteria string (words > 3 chars, lowercased,
/// stripped of punctuation).
fn parse_criteria_keywords(criteria: &str) -> Vec<String> {
    criteria
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Whether a label is agent-internal metadata (should be excluded from criteria matching).
fn is_metadata_label(label: &str) -> bool {
    super::synthesize::is_metadata_label(label)
}

/// Code-related keywords that indicate a code-structure query.
const CODE_KEYWORDS: &[&str] = &[
    "module",
    "function",
    "struct",
    "trait",
    "enum",
    "type",
    "impl",
    "architecture",
    "code",
    "crate",
    "method",
    "field",
    "vsa",
    "engine",
    "agent",
    "ooda",
    "tool",
    "graph",
    "symbol",
    "triple",
    "fn",
    "mod",
    "defines",
    "depends",
    "contains",
];

/// Whether a goal description is asking about code structure.
fn is_code_goal(description: &str) -> bool {
    let lower = description.to_lowercase();
    CODE_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Extract code entity labels that appeared in kg_query results (as children)
/// but haven't been queried themselves yet.
///
/// Parses WM ToolResult entries for kg_query output lines like:
/// `"Vsa" -> code:contains-mod -> "encode"  [1.00]`
/// and collects the object labels. Then checks which of those haven't been
/// queried (i.e., don't appear as subjects in other kg_query results).
fn find_unexplored_code_child(
    working_memory: &WorkingMemory,
    engine: &crate::engine::Engine,
) -> Option<String> {
    use super::memory::WorkingMemoryKind;

    let code_child_predicates = [
        "code:contains-mod",
        "code:defines-fn",
        "code:defines-struct",
        "code:defines-enum",
        "code:defines-type",
    ];

    let mut discovered_children: Vec<String> = Vec::new();
    let mut queried_subjects: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in working_memory.entries() {
        if !matches!(entry.kind, WorkingMemoryKind::ToolResult) {
            continue;
        }
        let content = &entry.content;
        if !content.contains("Tool result (kg_query):") {
            continue;
        }

        for line in content.lines() {
            let trimmed = line.trim();
            // Parse `"Subject" -> predicate -> "Object"  [conf]`
            let parts: Vec<&str> = trimmed.split(" -> ").collect();
            if parts.len() < 3 {
                continue;
            }
            let subject = parts[0].trim().trim_matches('"').to_string();
            let predicate = parts[1].trim();
            let object_raw = parts[2..].join(" -> ");
            // Strip trailing confidence
            let object = if let Some(bracket_pos) = object_raw.rfind("  [") {
                object_raw[..bracket_pos]
                    .trim()
                    .trim_matches('"')
                    .to_string()
            } else {
                object_raw.trim().trim_matches('"').to_string()
            };

            queried_subjects.insert(subject);

            if code_child_predicates.iter().any(|&p| p == predicate) {
                if !object.is_empty()
                    && !super::synthesize::is_metadata_label(&object)
                    && object != "tests"
                {
                    discovered_children.push(object);
                }
            }
        }
    }

    // Collect unexplored children (not yet queried, exists in KG).
    let mut unexplored: Vec<(String, f32)> = Vec::new();

    // Try to load semantic importance for ranking.
    let sem_preds = super::semantic_enrichment::SemanticPredicates::init(engine).ok();

    for child in &discovered_children {
        if queried_subjects.contains(child) {
            continue;
        }

        let resolved_label = if engine.resolve_symbol(child).is_ok() {
            Some(child.clone())
        } else {
            // Try module-qualified form (e.g., "vsa::encode")
            let symbols = engine.all_symbols();
            symbols.iter().find_map(|sym| {
                let label = &sym.label;
                if label.ends_with(child.as_str())
                    && (label.ends_with(&format!("::{child}")) || label == child)
                    && !queried_subjects.contains(label)
                {
                    Some(label.clone())
                } else {
                    None
                }
            })
        };

        if let Some(label) = resolved_label {
            // Look up importance if semantic enrichment is available.
            let importance = sem_preds
                .as_ref()
                .and_then(|preds| {
                    engine.resolve_symbol(&label).ok().and_then(|sym| {
                        super::semantic_enrichment::lookup_importance(engine, sym, preds)
                    })
                })
                .unwrap_or(0.0);

            unexplored.push((label, importance));
        }
    }

    // Sort by importance descending — prefer high-importance children first.
    unexplored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    unexplored.into_iter().next().map(|(label, _)| label)
}

/// Stop words that should not contribute to entity matching scores.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "describe", "explain", "show", "list", "what", "how", "why", "when",
    "where", "which", "who", "whom", "that", "this", "these", "those", "it", "its", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "about", "into", "through", "during", "before",
    "after", "and", "or", "but", "not", "no", "all", "each", "every", "any", "me", "my", "we",
    "our",
];

/// Find the best matching code entity for a goal description.
///
/// Scores all non-metadata symbols by keyword overlap with the description.
/// Uses exact word-boundary matching (not substring), filters stop words,
/// and prefers shorter labels (more specific entities).
fn find_code_entity_for_query(engine: &crate::engine::Engine, description: &str) -> Option<String> {
    let desc_words: Vec<String> = description
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty() && w.len() > 1 && !STOP_WORDS.contains(&w.as_str()))
        .collect();

    if desc_words.is_empty() {
        return None;
    }

    let symbols = engine.all_symbols();
    let mut best_label: Option<String> = None;
    let mut best_score = 0i32;

    for sym in &symbols {
        let label = &sym.label;
        if is_metadata_label(label) {
            continue;
        }
        // Skip long labels (docstrings, descriptions, signatures) — they're not entities.
        if label.len() > 60 {
            continue;
        }

        let label_lower = label.to_lowercase();

        // Tokenize the label into words for exact boundary matching.
        let label_words: Vec<&str> = label_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty())
            .collect();

        let mut score: i32 = 0;

        for desc_word in &desc_words {
            // Exact word match in label (e.g., "vsa" matches label word "vsa")
            if label_words.iter().any(|lw| *lw == desc_word.as_str()) {
                score += 3; // Strong signal: exact word boundary match
            }
            // Whole label equals the word (e.g., label "Vsa" == word "vsa")
            else if label_lower == *desc_word {
                score += 5; // Strongest: the entire label IS the keyword
            }
        }

        if score == 0 {
            continue;
        }

        // Bonus for having code-structure triples (not just visibility annotations).
        let structural_triples: Vec<_> = engine
            .triples_from(sym.id)
            .iter()
            .filter(|t| {
                let pred = engine.resolve_label(t.predicate);
                pred == "code:contains-mod"
                    || pred == "code:defines-fn"
                    || pred == "code:defines-struct"
                    || pred == "code:defines-enum"
                    || pred == "code:depends-on"
                    || pred == "code:defined-in"
            })
            .cloned()
            .collect();

        if !structural_triples.is_empty() {
            score += 4;
            // Bonus for richer entities (more children = more to explore).
            let child_count = structural_triples.len();
            if child_count >= 3 {
                score += 2; // Rich module with many children
            }
        }

        // Prefer shorter, more specific labels (e.g., "Vsa" over "vsa::item_memory").
        if label.len() <= 10 {
            score += 2;
        } else if label.len() > 30 {
            score -= 1;
        }

        if score > best_score
            || (score == best_score
                && label.len() < best_label.as_ref().map_or(usize::MAX, |l| l.len()))
        {
            best_score = score;
            best_label = Some(label.clone());
        }
    }

    best_label
}

/// Evaluate whether a goal's success criteria are satisfied by the current KG state.
///
/// Searches the entire KG for data symbols (not agent metadata) matching
/// criteria keywords. Returns the match ratio in [0.0, 1.0].
fn evaluate_criteria_against_kg(goal: &Goal, engine: &crate::engine::Engine) -> f32 {
    let keywords = parse_criteria_keywords(&goal.success_criteria);
    if keywords.is_empty() {
        return 0.0;
    }

    let matched: usize = keywords
        .iter()
        .filter(|kw| {
            engine.all_symbols().iter().any(|meta| {
                let label = meta.label.to_lowercase();
                !is_metadata_label(&label) && label.contains(kw.as_str())
            })
        })
        .count();

    matched as f32 / keywords.len() as f32
}

/// Evaluate whether a tool output satisfies a goal's success criteria.
///
/// Uses VSA interference pattern matching:
/// 1. **Tool output signal**: encode output symbols as a hypervector, measure
///    constructive interference with the criteria vector.
/// 2. **KG state signal**: encode KG neighborhood as a hypervector, measure
///    interference with the criteria vector.
///
/// Constructive interference (similarity > completion_threshold) → Completed.
/// Partial interference → Advanced. No interference → NoChange.
///
/// Falls back to keyword matching if VSA encoding fails.
fn evaluate_goal_progress(
    goal: &Goal,
    tool_output: &ToolOutput,
    engine: &crate::engine::Engine,
) -> GoalProgress {
    if !tool_output.success {
        return GoalProgress::Failed {
            reason: tool_output.result.clone(),
        };
    }

    if goal.success_criteria.trim().is_empty() {
        if tool_output.symbols_involved.is_empty() {
            return GoalProgress::NoChange;
        }
        return GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols (no evaluable criteria).",
                tool_output.symbols_involved.len()
            ),
        };
    }

    let ops = engine.ops();
    let im = engine.item_memory();

    // ── Encode criteria as hypervector ──
    let criteria_vec = match encode_criteria(&goal.success_criteria, engine, ops, im) {
        Ok(v) => v,
        Err(_) => return evaluate_goal_progress_keyword_fallback(goal, tool_output, engine),
    };

    // ── Signal 1: Tool output interference ──
    // Bundle the output symbols into a state vector and measure similarity.
    let output_vecs: Vec<crate::vsa::HyperVec> = tool_output
        .symbols_involved
        .iter()
        .filter(|s| **s != goal.symbol_id)
        .filter_map(|s| im.get(*s))
        .collect();

    let tool_interference = if !output_vecs.is_empty() {
        let refs: Vec<&crate::vsa::HyperVec> = output_vecs.iter().collect();
        match ops.bundle(&refs) {
            Ok(state_vec) => ops.similarity(&criteria_vec, &state_vec).unwrap_or(0.5),
            Err(_) => 0.5,
        }
    } else {
        0.5 // neutral — no signal
    };

    // ── Signal 2: KG state interference ──
    // Bundle the goal's KG neighborhood into a state vector.
    let kg_triples = engine.triples_from(goal.symbol_id);
    let kg_vecs: Vec<crate::vsa::HyperVec> = kg_triples
        .iter()
        .flat_map(|t| [t.object, t.predicate])
        .filter_map(|s| im.get(s))
        .collect();

    let kg_interference = if !kg_vecs.is_empty() {
        let refs: Vec<&crate::vsa::HyperVec> = kg_vecs.iter().collect();
        match ops.bundle(&refs) {
            Ok(state_vec) => ops.similarity(&criteria_vec, &state_vec).unwrap_or(0.5),
            Err(_) => 0.5,
        }
    } else {
        0.5
    };

    // The best interference signal determines progress.
    let best_interference = tool_interference.max(kg_interference);

    // Thresholds calibrated for VSA similarity:
    // Random vectors have ~0.50 similarity (bipolar), but the majority-vote
    // tie-breaker introduces bias when bundling even numbers of vectors.
    // Require BOTH signals to indicate relevance for completion (constructive
    // interference in both tool output and KG state), or a very strong single
    // signal to avoid false positives from tie-breaker artifacts.
    let strong_threshold = 0.70;
    let completion_threshold = 0.60;
    let advancement_threshold = 0.53;

    // Require both signals for completion, or one very strong signal
    let both_above_threshold =
        tool_interference >= completion_threshold && kg_interference >= completion_threshold;
    let one_very_strong = best_interference >= strong_threshold;

    if both_above_threshold || one_very_strong {
        GoalProgress::Completed
    } else if best_interference >= advancement_threshold {
        GoalProgress::Advanced {
            detail: format!(
                "VSA interference: tool={tool_interference:.3}, KG={kg_interference:.3} (threshold={completion_threshold:.2})",
            ),
        }
    } else if !tool_output.symbols_involved.is_empty() {
        GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols, weak interference: {best_interference:.3}.",
                tool_output.symbols_involved.len()
            ),
        }
    } else {
        GoalProgress::NoChange
    }
}

/// Keyword-based fallback for goal progress evaluation.
///
/// Used when VSA encoding fails (e.g., empty item memory).
fn evaluate_goal_progress_keyword_fallback(
    goal: &Goal,
    tool_output: &ToolOutput,
    engine: &crate::engine::Engine,
) -> GoalProgress {
    let criteria_keywords = parse_criteria_keywords(&goal.success_criteria);

    if criteria_keywords.is_empty() {
        if tool_output.symbols_involved.is_empty() {
            return GoalProgress::NoChange;
        }
        return GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols (no evaluable criteria).",
                tool_output.symbols_involved.len()
            ),
        };
    }

    let output_labels: Vec<String> = tool_output
        .symbols_involved
        .iter()
        .filter(|s| **s != goal.symbol_id)
        .map(|s| engine.resolve_label(*s).to_lowercase())
        .filter(|label| !is_metadata_label(label))
        .collect();

    let result_lower: String = tool_output
        .result
        .lines()
        .filter(|line| {
            let l = line.trim().to_lowercase();
            !is_metadata_label(&l) && !l.contains("desc:") && !l.contains("criteria:")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    let tool_matched: usize = criteria_keywords
        .iter()
        .filter(|kw| {
            output_labels
                .iter()
                .any(|label| label.contains(kw.as_str()))
                || result_lower.contains(kw.as_str())
        })
        .count();

    let tool_ratio = tool_matched as f32 / criteria_keywords.len() as f32;
    let kg_ratio = evaluate_criteria_against_kg(goal, engine);
    let best_ratio = tool_ratio.max(kg_ratio);
    let best_matched = (best_ratio * criteria_keywords.len() as f32).round() as usize;

    if best_ratio >= 0.5 {
        GoalProgress::Completed
    } else if best_matched > 0 {
        GoalProgress::Advanced {
            detail: format!(
                "Matched {}/{} criteria keywords (tool: {:.0}%, KG: {:.0}%).",
                best_matched,
                criteria_keywords.len(),
                tool_ratio * 100.0,
                kg_ratio * 100.0,
            ),
        }
    } else if !tool_output.symbols_involved.is_empty() {
        GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols, no criteria keywords matched yet.",
                tool_output.symbols_involved.len()
            ),
        }
    } else {
        GoalProgress::NoChange
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;

    fn make_candidate(name: &str, base_score: f32) -> ToolCandidate {
        ToolCandidate::new(name, ToolInput::new(), base_score, "test".into())
    }

    #[test]
    fn detect_impasse_none_for_clear_winner() {
        let candidates = vec![make_candidate("kg_query", 0.8), make_candidate("reason", 0.3)];
        let impasse = detect_impasse(&candidates, SymbolId::new(1).unwrap());
        assert!(impasse.is_none());
    }

    #[test]
    fn detect_impasse_all_below_threshold() {
        let candidates = vec![make_candidate("kg_query", 0.10), make_candidate("reason", 0.05)];
        let impasse = detect_impasse(&candidates, SymbolId::new(1).unwrap());
        assert!(impasse.is_some());
        let imp = impasse.unwrap();
        assert!(matches!(imp.kind, ImpasseKind::AllBelowThreshold { .. }));
    }

    #[test]
    fn detect_impasse_tie() {
        let candidates = vec![
            make_candidate("kg_query", 0.50),
            make_candidate("reason", 0.495),
        ];
        let impasse = detect_impasse(&candidates, SymbolId::new(1).unwrap());
        assert!(impasse.is_some());
        let imp = impasse.unwrap();
        assert!(matches!(imp.kind, ImpasseKind::Tie { .. }));
    }

    #[test]
    fn detect_impasse_empty_candidates() {
        let candidates: Vec<ToolCandidate> = Vec::new();
        let impasse = detect_impasse(&candidates, SymbolId::new(1).unwrap());
        assert!(impasse.is_some());
    }
}
