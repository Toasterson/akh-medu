//! Agent struct: the main API for the autonomous agent layer.
//!
//! The `Agent` wraps an `Arc<Engine>` and composes all agent subsystems:
//! working memory, goals, tools, and the OODA loop.

use std::sync::Arc;

use crate::engine::Engine;
use crate::symbol::SymbolId;

// Re-used for session persistence serialization.
use bincode;

use super::error::{AgentError, AgentResult};
use super::goal::{self, Goal, GoalStatus, DEFAULT_STALL_THRESHOLD};
use super::memory::{
    consolidate, recall_episodes, ConsolidationConfig, ConsolidationResult, EpisodicEntry,
    WorkingMemory,
};
use super::ooda::{self, OodaCycleResult};
use super::tool::{ToolRegistry, ToolSignature};
use super::tools;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum working memory entries (default: 100).
    pub working_memory_capacity: usize,
    /// Consolidation settings.
    pub consolidation: ConsolidationConfig,
    /// Maximum OODA cycles before stopping (default: 1000).
    pub max_cycles: usize,
    /// Whether to auto-consolidate when WM nears capacity (default: true).
    pub auto_consolidate: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 100,
            consolidation: ConsolidationConfig::default(),
            max_cycles: 1000,
            auto_consolidate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds used by the agent for KG-based agent state.
///
/// These are created (or resolved) on agent initialization.
#[derive(Debug, Clone)]
pub struct AgentPredicates {
    pub has_status: SymbolId,
    pub has_priority: SymbolId,
    pub has_description: SymbolId,
    pub has_criteria: SymbolId,
    pub parent_goal: SymbolId,
    pub child_goal: SymbolId,
    pub has_summary: SymbolId,
    pub has_tag: SymbolId,
    pub learned: SymbolId,
    pub consolidation_reason: SymbolId,
    pub from_cycle: SymbolId,
    pub memory_type: SymbolId,
}

impl AgentPredicates {
    /// Resolve or create all well-known predicates in the engine.
    fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            has_status: engine.resolve_or_create_relation("agent:has_status")?,
            has_priority: engine.resolve_or_create_relation("agent:has_priority")?,
            has_description: engine.resolve_or_create_relation("agent:has_description")?,
            has_criteria: engine.resolve_or_create_relation("agent:has_criteria")?,
            parent_goal: engine.resolve_or_create_relation("agent:parent_goal")?,
            child_goal: engine.resolve_or_create_relation("agent:child_goal")?,
            has_summary: engine.resolve_or_create_relation("agent:has_summary")?,
            has_tag: engine.resolve_or_create_relation("agent:has_tag")?,
            learned: engine.resolve_or_create_relation("agent:learned")?,
            consolidation_reason: engine.resolve_or_create_relation("agent:consolidation_reason")?,
            from_cycle: engine.resolve_or_create_relation("agent:from_cycle")?,
            memory_type: engine.resolve_or_create_relation("agent:memory_type")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// The autonomous agent: wraps an `Arc<Engine>` with deliberate memory, goals,
/// tools, and an OODA loop.
pub struct Agent {
    pub(crate) engine: Arc<Engine>,
    pub(crate) config: AgentConfig,
    pub(crate) working_memory: WorkingMemory,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) goals: Vec<Goal>,
    pub(crate) predicates: AgentPredicates,
    pub(crate) cycle_count: u64,
}

impl Agent {
    /// Create a new agent wrapping the given engine.
    ///
    /// Initializes well-known predicates, registers built-in tools, and
    /// optionally restores goals from the KG.
    pub fn new(engine: Arc<Engine>, config: AgentConfig) -> AgentResult<Self> {
        let predicates = AgentPredicates::init(&engine)?;

        let mut tool_registry = ToolRegistry::new();
        // Register built-in tools.
        tool_registry.register(Box::new(tools::KgQueryTool));
        tool_registry.register(Box::new(tools::KgMutateTool));
        tool_registry.register(Box::new(tools::MemoryRecallTool::new(predicates.clone())));
        tool_registry.register(Box::new(tools::ReasonTool));
        tool_registry.register(Box::new(tools::SimilaritySearchTool));

        let working_memory = WorkingMemory::new(config.working_memory_capacity);

        // Try to restore goals from KG.
        let goals = goal::restore_goals(&engine, &predicates).unwrap_or_default();

        Ok(Self {
            engine,
            config,
            working_memory,
            tool_registry,
            goals,
            predicates,
            cycle_count: 0,
        })
    }

    /// Add a new goal. Returns the goal's symbol ID.
    pub fn add_goal(
        &mut self,
        description: &str,
        priority: u8,
        criteria: &str,
    ) -> AgentResult<SymbolId> {
        let g = goal::create_goal(&self.engine, description, priority, criteria, &self.predicates)?;
        let id = g.symbol_id;
        self.goals.push(g);
        Ok(id)
    }

    /// Run a single OODA cycle.
    pub fn run_cycle(&mut self) -> AgentResult<OodaCycleResult> {
        if goal::active_goals(&self.goals).is_empty() {
            return Err(AgentError::NoGoals);
        }

        let result = ooda::run_ooda_cycle(self)?;

        // Auto-decompose stalled goals.
        self.decompose_stalled_goals();

        // Auto-consolidate if enabled and WM pressure is high.
        if self.config.auto_consolidate
            && self.working_memory.len() >= self.config.consolidation.auto_consolidate_at
        {
            let _ = self.consolidate();
        }

        Ok(result)
    }

    /// Run cycles until all goals are complete or max_cycles is reached.
    pub fn run_until_complete(&mut self) -> AgentResult<Vec<OodaCycleResult>> {
        let mut results = Vec::new();
        let max = self.config.max_cycles;

        for _ in 0..max {
            let active = goal::active_goals(&self.goals);
            if active.is_empty() {
                break;
            }

            let result = ooda::run_ooda_cycle(self)?;
            results.push(result);

            // Auto-decompose stalled goals.
            self.decompose_stalled_goals();

            // Auto-consolidate.
            if self.config.auto_consolidate
                && self.working_memory.len() >= self.config.consolidation.auto_consolidate_at
            {
                let _ = self.consolidate();
            }
        }

        // Check if we hit the limit with active goals remaining.
        if !goal::active_goals(&self.goals).is_empty() && results.len() >= max {
            return Err(AgentError::MaxCyclesReached { max_cycles: max });
        }

        Ok(results)
    }

    /// Trigger memory consolidation.
    pub fn consolidate(&mut self) -> AgentResult<ConsolidationResult> {
        consolidate(
            &mut self.working_memory,
            &self.engine,
            &self.goals,
            &self.config.consolidation,
            &self.predicates,
        )
    }

    /// Recall episodic memories by query symbols.
    pub fn recall(
        &self,
        query: &[SymbolId],
        top_k: usize,
    ) -> AgentResult<Vec<EpisodicEntry>> {
        recall_episodes(&self.engine, query, &self.predicates, top_k)
    }

    /// Get current goals.
    pub fn goals(&self) -> &[Goal] {
        &self.goals
    }

    /// Get working memory (read-only).
    pub fn working_memory(&self) -> &WorkingMemory {
        &self.working_memory
    }

    /// Register a custom tool.
    pub fn register_tool(&mut self, tool: Box<dyn super::tool::Tool>) {
        self.tool_registry.register(tool);
    }

    /// List all registered tool signatures.
    pub fn list_tools(&self) -> Vec<ToolSignature> {
        self.tool_registry.list()
    }

    /// Get a reference to the wrapped engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the current cycle count.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Get the agent predicates.
    pub fn predicates(&self) -> &AgentPredicates {
        &self.predicates
    }

    /// Mark a goal as completed by its symbol ID.
    pub fn complete_goal(&mut self, goal_id: SymbolId) -> AgentResult<()> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        goal::update_goal_status(&self.engine, goal, GoalStatus::Completed, &self.predicates)
    }

    /// Suspend a goal (e.g., while sub-goals are being worked on).
    pub fn suspend_goal(&mut self, goal_id: SymbolId) -> AgentResult<()> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        goal::update_goal_status(&self.engine, goal, GoalStatus::Suspended, &self.predicates)
    }

    /// Mark a goal as failed with a reason.
    pub fn fail_goal(&mut self, goal_id: SymbolId, reason: &str) -> AgentResult<()> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        goal::update_goal_status(
            &self.engine,
            goal,
            GoalStatus::Failed {
                reason: reason.into(),
            },
            &self.predicates,
        )
    }

    /// Auto-decompose any active goals that have stalled.
    ///
    /// A goal is stalled if it has been worked on for `DEFAULT_STALL_THRESHOLD`
    /// cycles without making progress. Stalled leaf goals (no existing children)
    /// are decomposed into sub-goals.
    fn decompose_stalled_goals(&mut self) {
        let stalled_ids: Vec<SymbolId> = self
            .goals
            .iter()
            .filter(|g| {
                matches!(g.status, GoalStatus::Active)
                    && g.children.is_empty()
                    && g.is_stalled(self.cycle_count, DEFAULT_STALL_THRESHOLD)
            })
            .map(|g| g.symbol_id)
            .collect();

        for goal_id in stalled_ids {
            let _ = self.decompose_stalled_goal(goal_id);
        }
    }

    /// Decompose a stalled goal into sub-goals.
    ///
    /// Suspends the parent goal and creates active child goals derived from
    /// the parent's description. Returns the new sub-goal symbol IDs.
    pub fn decompose_stalled_goal(&mut self, goal_id: SymbolId) -> AgentResult<Vec<SymbolId>> {
        let (description, parent_idx) = {
            let (idx, goal) = self
                .goals
                .iter()
                .enumerate()
                .find(|(_, g)| g.symbol_id == goal_id)
                .ok_or(AgentError::GoalNotFound {
                    goal_id: goal_id.get(),
                })?;
            (goal.description.clone(), idx)
        };

        let sub_descs = goal::generate_sub_goal_descriptions(&description);
        let sub_tuples: Vec<(&str, u8, &str)> = sub_descs
            .iter()
            .map(|(d, p, c)| (d.as_str(), *p, c.as_str()))
            .collect();

        let parent = &mut self.goals[parent_idx];
        let children = goal::decompose_goal(&self.engine, parent, &sub_tuples, &self.predicates)?;

        let child_ids: Vec<SymbolId> = children.iter().map(|c| c.symbol_id).collect();

        // Suspend the parent.
        goal::update_goal_status(
            &self.engine,
            &mut self.goals[parent_idx],
            GoalStatus::Suspended,
            &self.predicates,
        )?;

        // Add children to the goals list.
        self.goals.extend(children);

        Ok(child_ids)
    }

    // -----------------------------------------------------------------------
    // Session persistence
    // -----------------------------------------------------------------------

    /// Persist session state (working memory + cycle count) to the engine's durable store.
    ///
    /// Call this before shutting down to enable `resume()` later.
    pub fn persist_session(&self) -> AgentResult<()> {
        let store = self.engine.store();

        // Serialize working memory entries.
        let (wm_next_id, wm_bytes) = self.working_memory.serialize()?;

        store
            .put_meta(b"agent:wm_entries", &wm_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist WM entries: {e}"),
            })?;

        // Persist WM next_id counter.
        let next_id_bytes =
            bincode::serialize(&wm_next_id).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize WM next_id: {e}"),
            })?;
        store
            .put_meta(b"agent:wm_next_id", &next_id_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist WM next_id: {e}"),
            })?;

        // Persist cycle count.
        let cycle_bytes = bincode::serialize(&self.cycle_count).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize cycle count: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:cycle_count", &cycle_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist cycle count: {e}"),
            })?;

        // Flush the engine's durable store.
        self.engine
            .persist()
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("engine persist failed: {e}"),
            })?;

        Ok(())
    }

    /// Resume an agent from a previously persisted session.
    ///
    /// Restores working memory and cycle count from the durable store, and
    /// rebuilds goals from the knowledge graph.
    pub fn resume(engine: Arc<Engine>, config: AgentConfig) -> AgentResult<Self> {
        let predicates = AgentPredicates::init(&engine)?;

        let mut tool_registry = ToolRegistry::new();
        tool_registry.register(Box::new(tools::KgQueryTool));
        tool_registry.register(Box::new(tools::KgMutateTool));
        tool_registry.register(Box::new(tools::MemoryRecallTool::new(predicates.clone())));
        tool_registry.register(Box::new(tools::ReasonTool));
        tool_registry.register(Box::new(tools::SimilaritySearchTool));

        let store = engine.store();

        // Restore working memory.
        let working_memory = match (
            store.get_meta(b"agent:wm_entries").ok().flatten(),
            store.get_meta(b"agent:wm_next_id").ok().flatten(),
        ) {
            (Some(wm_bytes), Some(next_id_bytes)) => {
                let next_id: u64 = bincode::deserialize(&next_id_bytes).unwrap_or(1);
                WorkingMemory::restore(config.working_memory_capacity, next_id, &wm_bytes)?
            }
            _ => WorkingMemory::new(config.working_memory_capacity),
        };

        // Restore cycle count.
        let cycle_count = store
            .get_meta(b"agent:cycle_count")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<u64>(&bytes).ok())
            .unwrap_or(0);

        // Restore goals from KG.
        let goals = goal::restore_goals(&engine, &predicates).unwrap_or_default();

        Ok(Self {
            engine,
            config,
            working_memory,
            tool_registry,
            goals,
            predicates,
            cycle_count,
        })
    }

    /// Check whether a persisted session exists in the durable store.
    pub fn has_persisted_session(engine: &Engine) -> bool {
        engine
            .store()
            .get_meta(b"agent:cycle_count")
            .ok()
            .flatten()
            .is_some()
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("goals", &self.goals.len())
            .field("working_memory", &self.working_memory)
            .field("tools", &self.tool_registry)
            .field("cycle_count", &self.cycle_count)
            .finish()
    }
}
