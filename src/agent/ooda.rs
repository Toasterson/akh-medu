//! OODA loop: Observe → Orient → Decide → Act cycle.
//!
//! Each cycle gathers state, builds context, chooses an action, and executes it.
//! The agent's working memory is updated throughout the cycle.

use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::agent::Agent;
use super::error::{AgentError, AgentResult};
use super::goal::{self, GoalStatus};
use super::memory::{WorkingMemoryEntry, WorkingMemoryKind};
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
    /// Recalled episodic memory symbol IDs.
    pub recalled_episodes: Vec<SymbolId>,
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
            recalled = episodes.iter().map(|e| e.symbol_id).collect();
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

    let memory_pressure = agent.working_memory.pressure();

    // Push orientation summary to WM if we found knowledge.
    if !relevant_knowledge.is_empty() || !inferences.is_empty() {
        let orient_content = format!(
            "Orient: {} relevant triples, {} inferences, pressure {:.2}",
            relevant_knowledge.len(),
            inferences.len(),
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
    _observation: &Observation,
    orientation: &Orientation,
    cycle: u64,
) -> AgentResult<Decision> {
    // Get the top-priority active goal.
    let active = goal::active_goals(&agent.goals);
    let top_goal = active.first().ok_or(AgentError::NoGoals)?;
    let goal_id = top_goal.symbol_id;
    let goal_desc = &top_goal.description;

    // Rule-based strategy to select tool + build input.
    let (tool_name, tool_input, reasoning) = select_tool(
        top_goal,
        orientation,
        &agent.engine,
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

/// Rule-based tool selection strategy.
fn select_tool(
    goal: &super::goal::Goal,
    orientation: &Orientation,
    engine: &crate::engine::Engine,
) -> (String, ToolInput, String) {
    let goal_label = engine.resolve_label(goal.symbol_id);

    // If there's very little relevant knowledge, do a KG query first.
    if orientation.relevant_knowledge.is_empty() {
        return (
            "kg_query".into(),
            ToolInput::new()
                .with_param("symbol", &goal_label)
                .with_param("direction", "both"),
            "No relevant knowledge found — querying KG for goal context.".into(),
        );
    }

    // If there are inferences, maybe do similarity search to explore further.
    if !orientation.inferences.is_empty() {
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

    // If we have knowledge but no inferences, try reasoning.
    if !orientation.relevant_knowledge.is_empty() && orientation.inferences.is_empty() {
        // Build a simple expression from the first triple.
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

    // Default: query the goal symbol.
    (
        "kg_query".into(),
        ToolInput::new()
            .with_param("symbol", &goal_label)
            .with_param("direction", "both"),
        "Default action — querying KG for goal symbol.".into(),
    )
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

    // Determine goal progress.
    let goal_progress = if tool_output.success && !tool_output.symbols_involved.is_empty() {
        GoalProgress::Advanced {
            detail: format!(
                "Tool {} produced {} symbols",
                decision.chosen_tool,
                tool_output.symbols_involved.len()
            ),
        }
    } else if !tool_output.success {
        GoalProgress::Failed {
            reason: tool_output.result.clone(),
        }
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
