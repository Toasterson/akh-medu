//! OODA loop: Observe → Orient → Decide → Act cycle.
//!
//! Each cycle gathers state, builds context, chooses an action, and executes it.
//! The agent's working memory is updated throughout the cycle.

use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::agent::Agent;
use super::error::{AgentError, AgentResult};
use super::goal::{self, Goal, GoalStatus};
use super::memory::{EpisodicEntry, WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind};
use super::tool::{ToolInput, ToolOutput};

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
        if let Ok(episodes) = super::memory::recall_episodes(
            &agent.engine,
            &active_goals,
            &agent.predicates,
            3,
        ) {
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
    // Get the top-priority active goal.
    let active = goal::active_goals(&agent.goals);
    let top_goal = active.first().ok_or(AgentError::NoGoals)?;
    let goal_id = top_goal.symbol_id;
    let goal_desc = &top_goal.description;

    // Increment reference counts on WM entries we're consulting for this decision.
    for entry_id in &observation.recent_entries {
        agent.working_memory.increment_reference(*entry_id);
    }

    // Rule-based strategy to select tool + build input.
    let (tool_name, tool_input, reasoning) = select_tool(
        top_goal,
        observation,
        orientation,
        &agent.engine,
        &agent.working_memory,
    );

    // Record decision in WM.
    let dec_content = format!(
        "Decide: tool={tool_name}, goal=\"{goal_desc}\", reason={reasoning}",
    );
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
        DerivationKind::AgentDecision { goal: goal_id, cycle },
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

/// Tool selection strategy.
///
/// Considers all 5 tools, observation context (recalled episodes), orientation
/// (knowledge, inferences, memory pressure), and recent WM history to avoid
/// repeating the same action.
fn select_tool(
    goal: &Goal,
    observation: &Observation,
    orientation: &Orientation,
    engine: &crate::engine::Engine,
    working_memory: &WorkingMemory,
) -> (String, ToolInput, String) {
    let goal_label = engine.resolve_label(goal.symbol_id);
    let has_knowledge = !orientation.relevant_knowledge.is_empty();
    let has_inferences = !orientation.inferences.is_empty();
    let has_episodes = !observation.recalled_episodes.is_empty();

    // Check what tool was used most recently to avoid immediate repetition.
    let last_tool = last_tool_used(working_memory);

    // ── Priority 1: If memory pressure is high and episodes exist, recall ──
    // Recalling past experience can inform better decisions under pressure.
    if orientation.memory_pressure > 0.7 && has_episodes && last_tool.as_deref() != Some("memory_recall") {
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
        return (
            "memory_recall".into(),
            ToolInput::new()
                .with_param("query_symbols", &query_str)
                .with_param("top_k", "3"),
            format!(
                "Memory pressure high ({:.0}%) with past episodes — recalling relevant experience.",
                orientation.memory_pressure * 100.0
            ),
        );
    }

    // ── Priority 2: No knowledge at all — explore via KG query ──
    if !has_knowledge {
        // If episodes exist but no KG knowledge, recall first.
        if has_episodes && last_tool.as_deref() != Some("memory_recall") {
            let query_str = observation
                .recalled_episodes
                .iter()
                .flat_map(|ep| ep.learnings.iter())
                .take(3)
                .map(|s| engine.resolve_label(*s))
                .collect::<Vec<_>>()
                .join(",");
            if !query_str.is_empty() {
                return (
                    "memory_recall".into(),
                    ToolInput::new()
                        .with_param("query_symbols", &query_str)
                        .with_param("top_k", "3"),
                    "No KG knowledge but have past episodes — recalling experience first.".into(),
                );
            }
        }

        return (
            "kg_query".into(),
            ToolInput::new()
                .with_param("symbol", &goal_label)
                .with_param("direction", "both"),
            "No relevant knowledge found — querying KG for goal context.".into(),
        );
    }

    // ── Priority 3: Knowledge + inferences → synthesize new knowledge ──
    // When we have both knowledge and inferences, we can create new triples.
    if has_knowledge && has_inferences && last_tool.as_deref() != Some("kg_mutate") {
        // Find a potential new connection: link the top inference to the goal
        // via a relation found in existing knowledge.
        if let Some(new_triple) = synthesize_triple(goal, orientation, engine) {
            return (
                "kg_mutate".into(),
                ToolInput::new()
                    .with_param("subject", &new_triple.0)
                    .with_param("predicate", &new_triple.1)
                    .with_param("object", &new_triple.2)
                    .with_param("confidence", "0.7"),
                format!(
                    "Have knowledge and inferences — synthesizing new triple: {} -> {} -> {}.",
                    new_triple.0, new_triple.1, new_triple.2
                ),
            );
        }
    }

    // ── Priority 4: Inferences available → explore similar symbols ──
    if has_inferences && last_tool.as_deref() != Some("similarity_search") {
        let top_sym = orientation.inferences[0].0;
        let top_label = engine.resolve_label(top_sym);
        return (
            "similarity_search".into(),
            ToolInput::new()
                .with_param("symbol", &top_label)
                .with_param("top_k", "5"),
            format!(
                "Found inferences — exploring similar symbols around \"{}\".",
                top_label
            ),
        );
    }

    // ── Priority 5: Knowledge but no inferences → symbolic reasoning ──
    if has_knowledge && !has_inferences && last_tool.as_deref() != Some("reason") {
        let t = &orientation.relevant_knowledge[0];
        let expr = format!(
            "(triple {} {} {})",
            engine.resolve_label(t.subject),
            engine.resolve_label(t.predicate),
            engine.resolve_label(t.object),
        );
        return (
            "reason".into(),
            ToolInput::new().with_param("expression", &expr),
            "Have knowledge but no inferences — attempting symbolic reasoning.".into(),
        );
    }

    // ── Fallback: KG query (always safe) ──
    (
        "kg_query".into(),
        ToolInput::new()
            .with_param("symbol", &goal_label)
            .with_param("direction", "both"),
        "Cycling tools — querying KG for fresh context.".into(),
    )
}

/// Check what tool was used in the most recent ToolResult WM entry.
fn last_tool_used(working_memory: &WorkingMemory) -> Option<String> {
    working_memory
        .by_kind(WorkingMemoryKind::ToolResult)
        .into_iter()
        .max_by_key(|e| e.id)
        .and_then(|e| {
            // Content format: "Tool result (tool_name):\n..."
            e.content
                .strip_prefix("Tool result (")
                .and_then(|s| s.find(')').map(|i| s[..i].to_string()))
        })
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
    // Get the set of symbols already connected to the goal.
    let connected: std::collections::HashSet<SymbolId> = orientation
        .relevant_knowledge
        .iter()
        .flat_map(|t| [t.subject, t.object])
        .collect();

    // Find the first inference symbol NOT already connected to the goal.
    let novel_inference = orientation
        .inferences
        .iter()
        .find(|(sym, _)| !connected.contains(sym) && *sym != goal.symbol_id)?;

    // Pick a predicate from existing knowledge.
    let predicate = orientation
        .relevant_knowledge
        .first()
        .map(|t| t.predicate)?;

    let subject = engine.resolve_label(goal.symbol_id);
    let pred_label = engine.resolve_label(predicate);
    let object = engine.resolve_label(novel_inference.0);

    Some((subject, pred_label, object))
}

/// Act: execute the selected tool and update goal status.
fn act(agent: &mut Agent, decision: &Decision, cycle: u64) -> AgentResult<ActionResult> {
    let tool_output = agent.tool_registry.execute(
        &decision.chosen_tool,
        decision.tool_input.clone(),
        &agent.engine,
    )?;

    // Record tool result in WM.
    let result_content = if tool_output.result.len() > 120 {
        format!("{}...", &tool_output.result[..120])
    } else {
        tool_output.result.clone()
    };
    let wm_id = agent
        .working_memory
        .push(WorkingMemoryEntry {
            id: 0,
            content: format!("Tool result ({}):\n{}", decision.chosen_tool, result_content),
            symbols: tool_output.symbols_involved.clone(),
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 0.6,
            source_cycle: cycle,
            reference_count: 0,
        })
        .ok();

    // Determine goal progress by evaluating success criteria.
    let goal_progress = if let Some(goal) = agent
        .goals
        .iter()
        .find(|g| g.symbol_id == decision.goal_id)
    {
        evaluate_goal_progress(goal, &tool_output, &agent.engine)
    } else {
        GoalProgress::NoChange
    };

    // Update goal status if completed or failed.
    if let Some(goal) = agent
        .goals
        .iter_mut()
        .find(|g| g.symbol_id == decision.goal_id)
    {
        match &goal_progress {
            GoalProgress::Completed => {
                let _ = goal::update_goal_status(
                    &agent.engine,
                    goal,
                    GoalStatus::Completed,
                    &agent.predicates,
                );
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
            _ => {}
        }
    }

    Ok(ActionResult {
        tool_output,
        goal_progress,
        new_wm_entries: wm_id.into_iter().collect(),
    })
}

/// Evaluate whether a tool output satisfies a goal's success criteria.
///
/// Compares the symbols returned by the tool against keywords in the goal's
/// success criteria. If enough criteria keywords match symbol labels in the
/// tool output, the goal is considered complete.
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

    if tool_output.symbols_involved.is_empty() {
        return GoalProgress::NoChange;
    }

    // Extract meaningful keywords from success criteria (words > 3 chars).
    let criteria_keywords: Vec<String> = goal
        .success_criteria
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    if criteria_keywords.is_empty() {
        // No parseable criteria — treat any result with symbols as progress.
        return GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols (no evaluable criteria).",
                tool_output.symbols_involved.len()
            ),
        };
    }

    // Collect symbol labels from the tool output (lowercased), excluding the
    // goal's own symbol and agent-metadata labels (desc:, status:, priority:,
    // criteria:, goal:) to prevent self-referential matching.
    let output_labels: Vec<String> = tool_output
        .symbols_involved
        .iter()
        .filter(|s| **s != goal.symbol_id)
        .map(|s| engine.resolve_label(*s).to_lowercase())
        .filter(|label| {
            !label.starts_with("desc:")
                && !label.starts_with("status:")
                && !label.starts_with("priority:")
                && !label.starts_with("criteria:")
                && !label.starts_with("goal:")
                && !label.starts_with("agent:")
        })
        .collect();

    // Also search the result text, but strip out agent-metadata lines.
    let result_lower: String = tool_output
        .result
        .lines()
        .filter(|line| {
            let l = line.trim().to_lowercase();
            !l.contains("desc:") && !l.contains("criteria:") && !l.contains("status:")
                && !l.contains("priority:") && !l.starts_with("agent:")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();

    // Count how many criteria keywords are satisfied.
    let matched: usize = criteria_keywords
        .iter()
        .filter(|kw| {
            output_labels.iter().any(|label| label.contains(kw.as_str()))
                || result_lower.contains(kw.as_str())
        })
        .count();

    let match_ratio = matched as f32 / criteria_keywords.len() as f32;

    if match_ratio >= 0.5 {
        GoalProgress::Completed
    } else if matched > 0 {
        GoalProgress::Advanced {
            detail: format!(
                "Matched {}/{} criteria keywords ({:.0}%).",
                matched,
                criteria_keywords.len(),
                match_ratio * 100.0
            ),
        }
    } else {
        GoalProgress::Advanced {
            detail: format!(
                "Tool produced {} symbols, no criteria keywords matched yet.",
                tool_output.symbols_involved.len()
            ),
        }
    }
}
