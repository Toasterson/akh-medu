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
    WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind,
};
use super::ooda::{self, OodaCycleResult};
use super::plan::{self, Plan, PlanStatus};
use super::reflect::{self, Adjustment, ReflectionConfig, ReflectionResult};
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
    /// Reflection settings.
    pub reflection: ReflectionConfig,
    /// Maximum backtrack attempts per goal before giving up (default: 3).
    pub max_backtrack_attempts: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            working_memory_capacity: 100,
            consolidation: ConsolidationConfig::default(),
            max_cycles: 1000,
            auto_consolidate: true,
            reflection: ReflectionConfig::default(),
            max_backtrack_attempts: 3,
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
    /// Active plans per goal (keyed by goal SymbolId).
    pub(crate) plans: std::collections::HashMap<u64, Plan>,
    /// Most recent reflection result.
    pub(crate) last_reflection: Option<ReflectionResult>,
    /// Structured message output sink.
    pub(crate) sink: Arc<dyn crate::message::MessageSink>,
    /// Optional Jungian psyche (loaded from the psyche compartment).
    pub(crate) psyche: Option<crate::compartment::psyche::Psyche>,
    /// Optional WASM tool runtime (only when `wasm-tools` feature is enabled).
    #[cfg(feature = "wasm-tools")]
    pub(crate) wasm_runtime: Option<super::wasm_runtime::WasmToolRuntime>,
}

impl Agent {
    /// Register all built-in tools (core + external).
    fn register_builtin_tools(
        registry: &mut ToolRegistry,
        predicates: &AgentPredicates,
        engine: &Engine,
    ) {
        // Core KG tools.
        registry.register(Box::new(tools::KgQueryTool));
        registry.register(Box::new(tools::KgMutateTool));
        registry.register(Box::new(tools::MemoryRecallTool::new(predicates.clone())));
        registry.register(Box::new(tools::ReasonTool));
        registry.register(Box::new(tools::SimilaritySearchTool));

        // External world-interaction tools.
        let scratch_dir = engine
            .config()
            .data_dir
            .as_ref()
            .map(|d| d.join("scratch"));
        registry.register(Box::new(tools::FileIoTool::new(scratch_dir)));
        registry.register(Box::new(tools::HttpFetchTool));
        registry.register(Box::new(tools::ShellExecTool));
        registry.register(Box::new(tools::UserInteractTool));

        // Autonomous reasoning tools.
        registry.register(Box::new(tools::InferRulesTool));
        registry.register(Box::new(tools::GapAnalysisTool));

        // Ingest tools.
        registry.register(Box::new(tools::CsvIngestTool));
        registry.register(Box::new(tools::TextIngestTool));
        registry.register(Box::new(tools::CodeIngestTool));

        // Documentation generation.
        registry.register(Box::new(tools::DocGenTool));
    }

    /// Load CLI and WASM tools from active skills.
    ///
    /// For each Hot skill, reads `cli_tools` paths as `CliToolManifest` JSONs
    /// and (if the `wasm-tools` feature is enabled) `wasm_tools` paths as WASM modules.
    fn load_skill_tools(&mut self) {
        let hot_skills: Vec<_> = self
            .engine
            .list_skills()
            .into_iter()
            .filter(|info| info.state == crate::skills::SkillState::Hot)
            .collect();

        if hot_skills.is_empty() {
            return;
        }

        let data_dir = self
            .engine
            .config()
            .data_dir
            .as_ref()
            .map(|d| d.join("skills"));

        for info in &hot_skills {
            let skill_dir = match &data_dir {
                Some(base) => base.join(&info.id),
                None => continue,
            };

            // Load CLI tools from manifest JSON files.
            if let Ok(manifest) = Self::read_skill_manifest(&skill_dir) {
                for cli_path in &manifest.cli_tools {
                    let full_path = skill_dir.join(cli_path);
                    match std::fs::read_to_string(&full_path) {
                        Ok(json) => {
                            match serde_json::from_str::<super::cli_tool::CliToolManifest>(&json) {
                                Ok(cli_manifest) => {
                                    let tool = super::cli_tool::CliTool::new(
                                        cli_manifest,
                                        Some(info.id.clone()),
                                    );
                                    self.tool_registry.register(Box::new(tool));
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        skill = %info.id,
                                        path = %full_path.display(),
                                        "Failed to parse CLI tool manifest: {e}"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                skill = %info.id,
                                path = %full_path.display(),
                                "Failed to read CLI tool manifest: {e}"
                            );
                        }
                    }
                }

                // Load WASM tools (feature-gated).
                #[cfg(feature = "wasm-tools")]
                if let Some(ref runtime) = self.wasm_runtime {
                    for wasm_path in &manifest.wasm_tools {
                        let full_path = skill_dir.join(wasm_path);
                        match runtime.load_tool(
                            &full_path.to_string_lossy(),
                            Arc::clone(&self.engine),
                            info.id.clone(),
                            manifest.tool_config.clone(),
                        ) {
                            Ok(wasm_tool) => {
                                self.tool_registry.register(Box::new(wasm_tool));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    skill = %info.id,
                                    path = %full_path.display(),
                                    "Failed to load WASM tool: {e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Read a skill's manifest JSON from its directory.
    fn read_skill_manifest(
        skill_dir: &std::path::Path,
    ) -> Result<crate::skills::SkillManifest, String> {
        let manifest_path = skill_dir.join("skill.json");
        let json = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("read skill.json: {e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("parse skill.json: {e}"))
    }

    /// Create a new agent wrapping the given engine.
    ///
    /// Initializes well-known predicates, registers built-in tools, and
    /// optionally restores goals from the KG.
    pub fn new(engine: Arc<Engine>, config: AgentConfig) -> AgentResult<Self> {
        let predicates = AgentPredicates::init(&engine)?;

        let mut tool_registry = ToolRegistry::new();
        Self::register_builtin_tools(&mut tool_registry, &predicates, &engine);

        let working_memory = WorkingMemory::new(config.working_memory_capacity);

        // Try to restore goals from KG.
        let goals = goal::restore_goals(&engine, &predicates).unwrap_or_default();

        // Load psyche from compartment manager if available.
        let psyche = engine
            .compartments()
            .and_then(|cm| cm.psyche());

        let mut agent = Self {
            engine,
            config,
            working_memory,
            tool_registry,
            goals,
            predicates,
            cycle_count: 0,
            plans: std::collections::HashMap::new(),
            last_reflection: None,
            sink: Arc::new(crate::message::StdoutSink),
            psyche,
            #[cfg(feature = "wasm-tools")]
            wasm_runtime: super::wasm_runtime::WasmToolRuntime::new().ok(),
        };

        // Load tools from active skills.
        agent.load_skill_tools();

        Ok(agent)
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
    ///
    /// Also triggers plan-aware execution (advances plan steps), periodic
    /// reflection, auto-decomposition of stalled goals, and auto-consolidation.
    pub fn run_cycle(&mut self) -> AgentResult<OodaCycleResult> {
        if goal::active_goals(&self.goals).is_empty() {
            return Err(AgentError::NoGoals);
        }

        // Before the OODA cycle, ensure the top-priority goal has a plan.
        let active = goal::active_goals(&self.goals);
        if let Some(top_goal) = active.first() {
            let gid = top_goal.symbol_id;
            let _ = self.plan_goal(gid);
        }

        let result = ooda::run_ooda_cycle(self)?;

        // Advance plan tracking based on the cycle result.
        let goal_id = result.decision.goal_id;
        let key = goal_id.get();
        let mut should_backtrack = false;
        if let Some(plan) = self.plans.get_mut(&key) {
            if let Some(idx) = plan.next_step_index() {
                match &result.action_result.goal_progress {
                    ooda::GoalProgress::Failed { reason } => {
                        plan.fail_step(idx, reason);
                        should_backtrack = true;
                    }
                    _ => {
                        plan.complete_step(idx);
                    }
                }
            }
        }
        if should_backtrack {
            let _ = self.backtrack_goal(goal_id);
        }

        // Periodic reflection.
        let reflect_interval = self.config.reflection.reflect_every_n_cycles;
        if reflect_interval > 0 && self.cycle_count % reflect_interval == 0 {
            if let Ok(reflection) = self.reflect() {
                // Auto-apply non-destructive adjustments (priority changes only).
                let safe_adjustments: Vec<Adjustment> = reflection
                    .adjustments
                    .iter()
                    .filter(|a| {
                        matches!(
                            a,
                            Adjustment::IncreasePriority { .. }
                                | Adjustment::DecreasePriority { .. }
                        )
                    })
                    .cloned()
                    .collect();
                let _ = self.apply_adjustments(&safe_adjustments);
            }
        }

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

    /// Set the message sink for structured output.
    pub fn set_sink(&mut self, sink: Arc<dyn crate::message::MessageSink>) {
        self.sink = sink;
    }

    /// Get a reference to the current message sink.
    pub fn sink(&self) -> &dyn crate::message::MessageSink {
        self.sink.as_ref()
    }

    /// Get the agent's psyche (read-only).
    pub fn psyche(&self) -> Option<&crate::compartment::psyche::Psyche> {
        self.psyche.as_ref()
    }

    /// Set or replace the agent's psyche.
    pub fn set_psyche(&mut self, psyche: crate::compartment::psyche::Psyche) {
        self.psyche = Some(psyche);
    }

    /// Synthesize human-readable narrative from the agent's working memory findings.
    pub fn synthesize_findings(&self, goal: &str) -> super::synthesize::NarrativeSummary {
        super::synthesize::synthesize(
            goal,
            self.working_memory.entries(),
            &self.engine,
        )
    }

    /// Synthesize findings using a specific grammar archetype.
    pub fn synthesize_findings_with_grammar(
        &self,
        goal: &str,
        grammar: &str,
    ) -> super::synthesize::NarrativeSummary {
        super::synthesize::synthesize_with_grammar(
            goal,
            self.working_memory.entries(),
            &self.engine,
            grammar,
        )
    }

    /// Clear all restored goals (for fresh-start mode).
    pub fn clear_goals(&mut self) {
        self.goals.clear();
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
    // Planning
    // -----------------------------------------------------------------------

    /// Get the active plan for a goal, if one exists.
    pub fn plan_for_goal(&self, goal_id: SymbolId) -> Option<&Plan> {
        self.plans.get(&goal_id.get())
    }

    /// Generate a plan for a goal. If a plan already exists and is active,
    /// returns it unchanged.
    pub fn plan_goal(&mut self, goal_id: SymbolId) -> AgentResult<&Plan> {
        let key = goal_id.get();

        // Return existing active plan.
        if self.plans.get(&key).is_some_and(|p| p.status == PlanStatus::Active) {
            return Ok(&self.plans[&key]);
        }

        let goal = self
            .goals
            .iter()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound { goal_id: key })?
            .clone();

        let attempt = self
            .plans
            .get(&key)
            .map(|p| p.attempt + 1)
            .unwrap_or(0);

        let new_plan = plan::generate_plan(
            &goal,
            &self.engine,
            &self.working_memory,
            attempt,
        )?;

        // Record plan creation in WM.
        let _ = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: format!(
                "Plan generated for \"{}\": {} steps, strategy: {}",
                goal.description,
                new_plan.total_steps(),
                new_plan.strategy,
            ),
            symbols: vec![goal_id],
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.7,
            source_cycle: self.cycle_count,
            reference_count: 0,
        });

        self.plans.insert(key, new_plan);
        Ok(&self.plans[&key])
    }

    /// Backtrack: discard the current plan for a goal and generate an alternative.
    ///
    /// Returns `None` if max backtrack attempts have been exhausted.
    pub fn backtrack_goal(&mut self, goal_id: SymbolId) -> AgentResult<Option<&Plan>> {
        let key = goal_id.get();
        let current = self.plans.get(&key);

        let attempt = current.map(|p| p.attempt + 1).unwrap_or(0);
        if attempt >= self.config.max_backtrack_attempts {
            return Ok(None);
        }

        // Mark old plan as superseded.
        if let Some(old) = self.plans.get_mut(&key) {
            old.status = PlanStatus::Superseded;
        }

        let goal = self
            .goals
            .iter()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound { goal_id: key })?
            .clone();

        let new_plan = plan::generate_plan(
            &goal,
            &self.engine,
            &self.working_memory,
            attempt,
        )?;

        let _ = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: format!(
                "Backtrack: new plan (attempt {}) for \"{}\": {}",
                attempt + 1,
                goal.description,
                new_plan.strategy,
            ),
            symbols: vec![goal_id],
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.7,
            source_cycle: self.cycle_count,
            reference_count: 0,
        });

        self.plans.insert(key, new_plan);
        Ok(Some(&self.plans[&key]))
    }

    // -----------------------------------------------------------------------
    // Reflection & meta-reasoning
    // -----------------------------------------------------------------------

    /// Run reflection: review working memory and strategy effectiveness.
    pub fn reflect(&mut self) -> AgentResult<ReflectionResult> {
        let result = reflect::reflect(
            &self.working_memory,
            &self.goals,
            self.cycle_count,
            &self.config.reflection,
            self.psyche.as_mut(),
        )?;

        // Record reflection in WM.
        let _ = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: result.summary.clone(),
            symbols: vec![],
            kind: WorkingMemoryKind::Inference,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: self.cycle_count,
            reference_count: 0,
        });

        self.last_reflection = Some(result.clone());
        Ok(result)
    }

    /// Apply meta-reasoning adjustments from a reflection result.
    ///
    /// Modifies goal priorities and creates suggested sub-goals. Returns
    /// the number of adjustments applied.
    pub fn apply_adjustments(&mut self, adjustments: &[Adjustment]) -> AgentResult<usize> {
        let mut applied = 0;

        for adj in adjustments {
            match adj {
                Adjustment::IncreasePriority {
                    goal_id, to, ..
                }
                | Adjustment::DecreasePriority {
                    goal_id, to, ..
                } => {
                    if let Some(g) = self.goals.iter_mut().find(|g| g.symbol_id == *goal_id) {
                        g.priority = *to;
                        applied += 1;
                    }
                }
                Adjustment::SuggestNewGoal {
                    description,
                    priority,
                    ..
                } => {
                    let _ = self.add_goal(description, *priority, description)?;
                    applied += 1;
                }
                Adjustment::SuggestAbandon { goal_id, reason } => {
                    let _ = self.fail_goal(*goal_id, reason);
                    applied += 1;
                }
            }
        }

        Ok(applied)
    }

    /// Get the most recent reflection result.
    pub fn last_reflection(&self) -> Option<&ReflectionResult> {
        self.last_reflection.as_ref()
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

        // Persist psyche if loaded.
        if let Some(ref psyche) = self.psyche {
            let psyche_bytes =
                bincode::serialize(psyche).map_err(|e| AgentError::ConsolidationFailed {
                    message: format!("failed to serialize psyche: {e}"),
                })?;
            store
                .put_meta(b"agent:psyche", &psyche_bytes)
                .map_err(|e| AgentError::ConsolidationFailed {
                    message: format!("failed to persist psyche: {e}"),
                })?;
        }

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
        Self::register_builtin_tools(&mut tool_registry, &predicates, &engine);

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

        // Restore psyche: prefer persisted state, fall back to compartment manager.
        let psyche = store
            .get_meta(b"agent:psyche")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<crate::compartment::psyche::Psyche>(&bytes).ok())
            .or_else(|| {
                engine
                    .compartments()
                    .and_then(|cm| cm.psyche())
            });

        let mut agent = Self {
            engine,
            config,
            working_memory,
            tool_registry,
            goals,
            predicates,
            cycle_count,
            plans: std::collections::HashMap::new(),
            last_reflection: None,
            sink: Arc::new(crate::message::StdoutSink),
            psyche,
            #[cfg(feature = "wasm-tools")]
            wasm_runtime: super::wasm_runtime::WasmToolRuntime::new().ok(),
        };

        // Load tools from active skills.
        agent.load_skill_tools();

        Ok(agent)
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
            .field("active_plans", &self.plans.len())
            .field("has_reflection", &self.last_reflection.is_some())
            .field("has_psyche", &self.psyche.is_some())
            .finish()
    }
}
