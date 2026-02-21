//! Agent struct: the main API for the autonomous agent layer.
//!
//! The `Agent` wraps an `Arc<Engine>` and composes all agent subsystems:
//! working memory, goals, tools, and the OODA loop.

use std::sync::Arc;

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// Re-used for session persistence serialization.
use bincode;

use super::channel::{ChannelRegistry, ChannelResult};
use super::channel_message::{ConstraintCheckStatus, InboundMessage};
use super::constraint_check::{CheckOutcome, ConstraintChecker, EmissionDecision, emission_decision};
use super::conversation::{ConversationState, ResponseDetail};
use super::operator_channel::InboundHandle;
use super::decomposition::{self, DecompositionOutput, MethodRegistry, MethodStats};
use super::drives::DriveSystem;
use super::error::{AgentError, AgentResult};
use super::goal::{self, DEFAULT_STALL_THRESHOLD, Goal, GoalStatus};
use super::goal_generation::{self, GoalGenerationConfig, GoalGenerationResult};
use super::metacognition::{self, CompetenceModel, FailureCase, FailureIndex, MetacognitionConfig};
use super::chunking::{self, ChunkingConfig, MethodIndex};
use super::resource::{self, EffortIndex, ImprovementHistory};
use super::watch::{self, Watch, WatchFiring};
use super::memory::{
    ConsolidationConfig, ConsolidationResult, EpisodicEntry, SessionSummary, WorkingMemory,
    WorkingMemoryEntry, WorkingMemoryKind, consolidate, generate_session_summary, recall_episodes,
    restore_session_summary,
};
use super::ooda::{self, DecisionImpasse, OodaCycleResult};
use super::plan::{self, Plan, PlanStatus};
use super::priority_reasoning::Audience;
use super::project::{
    self, Agenda, Project, ProjectPredicates, ProjectStatus, project_progress, update_project_status,
};
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
    /// Autonomous goal generation settings.
    pub goal_generation: GoalGenerationConfig,
    /// Metacognitive monitoring/control settings (Phase 11f).
    pub metacognition: MetacognitionConfig,
    /// Procedural learning (chunking) settings (Phase 11h).
    pub chunking: ChunkingConfig,
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
            goal_generation: GoalGenerationConfig::default(),
            metacognition: MetacognitionConfig::default(),
            chunking: ChunkingConfig::default(),
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
    /// HTN dependency: the object goal must complete before the subject goal.
    pub blocked_by: SymbolId,
}

impl AgentPredicates {
    /// Resolve or create all well-known predicates in the engine.
    pub(crate) fn init(engine: &Engine) -> AgentResult<Self> {
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
            consolidation_reason: engine
                .resolve_or_create_relation("agent:consolidation_reason")?,
            from_cycle: engine.resolve_or_create_relation("agent:from_cycle")?,
            memory_type: engine.resolve_or_create_relation("agent:memory_type")?,
            blocked_by: engine.resolve_or_create_relation("agent:blocked_by")?,
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
    /// Motivational drive system for autonomous goal generation.
    pub(crate) drives: DriveSystem,
    /// Value ordering for argumentation-based priority reasoning (Phase 11c).
    pub(crate) audience: Audience,
    /// HTN decomposition method registry.
    pub(crate) method_registry_htn: MethodRegistry,
    /// Most recent decision impasse (consumed by goal generation).
    pub(crate) last_impasse: Option<DecisionImpasse>,
    /// Active projects (groups of related goals backed by microtheories).
    pub(crate) projects: Vec<Project>,
    /// Project-related KG predicates.
    pub(crate) project_predicates: ProjectPredicates,
    /// Ordered project list persisted across sessions.
    pub(crate) agenda: Agenda,
    /// Cycle count at the start of this session (for session summary).
    pub(crate) session_start_cycle: u64,
    /// World-monitoring watches (Phase 11e).
    pub(crate) watches: Vec<Watch>,
    /// Most recent watch firings (consumed by goal generation).
    pub(crate) last_watch_firings: Vec<WatchFiring>,
    /// Cumulative competence tracker (Phase 11f).
    pub(crate) competence_model: CompetenceModel,
    /// HNSW-backed failure pattern index (Phase 11f).
    pub(crate) failure_index: FailureIndex,
    /// CBR-backed effort estimation index (Phase 11g).
    pub(crate) effort_index: EffortIndex,
    /// Per-goal improvement rate history (Phase 11g).
    pub(crate) improvement_history: ImprovementHistory,
    /// HNSW-backed learned method index (Phase 11h).
    pub(crate) method_index: MethodIndex,
    /// Procedural learning configuration (Phase 11h).
    pub(crate) chunking_config: ChunkingConfig,
    /// Communication channel registry (Phase 12a).
    pub(crate) channel_registry: ChannelRegistry,
    /// Conversation state for grounded dialogue (Phase 12b).
    pub(crate) conversation_state: ConversationState,
    /// Pre-communication constraint checker (Phase 12c).
    pub(crate) constraint_checker: ConstraintChecker,
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
        let scratch_dir = engine.config().data_dir.as_ref().map(|d| d.join("scratch"));
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
        registry.register(Box::new(tools::ContentIngestTool));

        // Library search.
        registry.register(Box::new(tools::LibrarySearchTool));

        // Documentation generation.
        registry.register(Box::new(tools::DocGenTool));

        // Code generation, validation, and pattern mining.
        registry.register(Box::new(tools::CodeGenTool));
        registry.register(Box::new(tools::CompileFeedbackTool));
        registry.register(Box::new(tools::PatternMineTool));

        // Agent management (multi-agent orchestration).
        registry.register(Box::new(tools::AgentListTool));
        registry.register(Box::new(tools::AgentSpawnTool));
        registry.register(Box::new(tools::AgentMessageTool));
        registry.register(Box::new(tools::AgentRetireTool));

        // Trigger management.
        registry.register(Box::new(tools::TriggerManageTool));
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
        let json =
            std::fs::read_to_string(&manifest_path).map_err(|e| format!("read skill.json: {e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("parse skill.json: {e}"))
    }

    /// Create a new agent wrapping the given engine.
    ///
    /// Initializes well-known predicates, registers built-in tools, and
    /// optionally restores goals from the KG.
    pub fn new(engine: Arc<Engine>, config: AgentConfig) -> AgentResult<Self> {
        let predicates = AgentPredicates::init(&engine)?;
        let project_predicates = ProjectPredicates::init(&engine)?;

        let mut tool_registry = ToolRegistry::new();
        Self::register_builtin_tools(&mut tool_registry, &predicates, &engine);

        let working_memory = WorkingMemory::new(config.working_memory_capacity);

        // Try to restore goals from KG.
        let goals = goal::restore_goals(&engine, &predicates).unwrap_or_default();

        // Load psyche from compartment manager if available.
        let psyche = engine.compartments().and_then(|cm| cm.psyche());

        let drives = DriveSystem::with_thresholds(config.goal_generation.drive_thresholds);
        let chunking_config = config.chunking.clone();

        let mut htn_registry = MethodRegistry::new();
        decomposition::register_builtin_methods(&mut htn_registry);

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
            drives,
            audience: Audience::exploration(),
            method_registry_htn: htn_registry,
            last_impasse: None,
            projects: Vec::new(),
            project_predicates,
            agenda: Agenda::new(),
            session_start_cycle: 0,
            watches: Vec::new(),
            last_watch_firings: Vec::new(),
            competence_model: CompetenceModel::default(),
            failure_index: FailureIndex::new(),
            effort_index: EffortIndex::new(),
            improvement_history: ImprovementHistory::default(),
            method_index: MethodIndex::new(),
            chunking_config,
            channel_registry: ChannelRegistry::new(),
            conversation_state: ConversationState::default(),
            constraint_checker: ConstraintChecker::new(),
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
        let g = goal::create_goal(
            &self.engine,
            description,
            priority,
            criteria,
            &self.predicates,
        )?;
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

        // ── Metacognition: update competence model ──────────────────────
        {
            let tool_name = &result.decision.chosen_tool;
            let tool_success = !matches!(
                result.action_result.goal_progress,
                ooda::GoalProgress::Failed { .. }
            );
            self.competence_model.update_tool(tool_name, tool_success);

            // Update goal category competence if goal completed/failed.
            let goal_completed = matches!(
                result.action_result.goal_progress,
                ooda::GoalProgress::Completed
            );
            let goal_failed = matches!(
                result.action_result.goal_progress,
                ooda::GoalProgress::Failed { .. }
            );
            if goal_completed || goal_failed {
                if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                    let category = metacognition::categorize_goal(g);
                    self.competence_model.update_category(&category, goal_completed);
                }
                // Mark resolved failures if a goal completed.
                if goal_completed {
                    if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                        self.failure_index
                            .mark_resolved(&g.description, tool_name);
                    }
                }
                // Record effort case for CBR (Phase 11g).
                if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                    let case = resource::create_effort_case(g, &self.working_memory, &self.engine);
                    self.effort_index.insert(case);
                }

                // Procedural learning: compile successful traces into methods (Phase 11h).
                if goal_completed {
                    if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                        let g_clone = g.clone();
                        if let Some(learned) = chunking::chunk_completed_goal(
                            &g_clone,
                            &self.working_memory,
                            &self.engine,
                            &self.chunking_config,
                            self.cycle_count,
                        ) {
                            // Register as HTN decomposition method.
                            let dm = chunking::to_decomposition_method(&learned);
                            self.method_registry_htn.register(dm);

                            // Record provenance.
                            let step_count = learned.steps.len();
                            let method_name = learned.name.clone();
                            let source_goal = learned.id;
                            let mut prov = ProvenanceRecord::new(
                                goal_id,
                                DerivationKind::ProceduralLearning {
                                    source_goal,
                                    method_name,
                                    step_count,
                                },
                            );
                            let _ = self.engine.store_provenance(&mut prov);

                            // Add to method index.
                            self.method_index.insert(learned);
                        }
                    }
                }
            }

            // Record improvement rate for all active goals (Phase 11g).
            for g in &self.goals {
                if matches!(g.status, GoalStatus::Active) {
                    let rate = if g.cycles_worked == 0 {
                        0.5
                    } else {
                        let cycles_since = self.cycle_count.saturating_sub(g.last_progress_cycle);
                        if cycles_since == 0 { 1.0 } else { (1.0 / cycles_since as f32).min(1.0) }
                    };
                    resource::record_improvement(
                        &mut self.improvement_history,
                        g.symbol_id,
                        self.cycle_count,
                        rate,
                        20,
                    );
                }
            }

            // Record failure in failure index.
            if !tool_success {
                if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                    let error_msg = match &result.action_result.goal_progress {
                        ooda::GoalProgress::Failed { reason } => reason.clone(),
                        _ => "unknown".to_string(),
                    };
                    let query = format!("{} {} {}", g.description, tool_name, error_msg);
                    let vector = super::metacognition::simple_text_hash(&query);
                    self.failure_index.insert(FailureCase {
                        goal_description: g.description.clone(),
                        tool_name: tool_name.clone(),
                        error_message: error_msg,
                        cycle: self.cycle_count,
                        resolved: false,
                        resolution_tool: None,
                        vector,
                    });
                }
            }
        }

        // Periodic reflection.
        let reflect_interval = self.config.reflection.reflect_every_n_cycles;
        if reflect_interval > 0 && self.cycle_count % reflect_interval == 0 {
            if let Ok(reflection) = self.reflect() {
                // Apply structural adjustments (abandon, decompose) from compute_adjustments().
                // Priority changes are now handled by argumentation below.
                let structural_adjustments: Vec<Adjustment> = reflection
                    .adjustments
                    .iter()
                    .filter(|a| {
                        matches!(
                            a,
                            Adjustment::SuggestAbandon { .. }
                                | Adjustment::SuggestNewGoal { .. }
                        )
                    })
                    .cloned()
                    .collect();
                let _ = self.apply_adjustments(&structural_adjustments);
            }

            // Argumentation-based reprioritization replaces raw ±30 adjustments.
            let stall_threshold = self.config.reflection.stagnation_threshold;
            let reprioritizations = reflect::reprioritize_goals(
                &self.goals,
                self.cycle_count,
                stall_threshold,
                &self.audience,
            );
            for (goal_id, old_priority, verdict) in reprioritizations {
                if let Some(g) = self.goals.iter_mut().find(|g| g.symbol_id == goal_id) {
                    let new_priority = verdict.computed_priority;
                    g.priority_rationale = Some(verdict);

                    // Record provenance for priority change.
                    let mut prov = ProvenanceRecord::new(
                        goal_id,
                        DerivationKind::PriorityArgumentation {
                            goal: goal_id,
                            old_priority,
                            new_priority,
                            audience: self.audience.name.clone(),
                            net_score: g
                                .priority_rationale
                                .as_ref()
                                .map(|v| v.net_score)
                                .unwrap_or(0.0),
                        },
                    )
                    .with_confidence(0.9);
                    let _ = self.engine.store_provenance(&mut prov);
                }
            }

            // ── Metacognitive goal relevance evaluation (Phase 11f) ──
            let meta_adjustments = reflect::evaluate_goal_relevance(
                &self.goals,
                &self.working_memory,
                &self.competence_model,
                &self.failure_index,
                &self.config.metacognition,
                self.cycle_count,
                &self.engine,
            );
            if !meta_adjustments.is_empty() {
                let _ = self.apply_adjustments(&meta_adjustments);
            }
        }

        // Periodic library learning (4x less frequent than reflection).
        let library_learn_interval = reflect_interval.saturating_mul(4).max(1);
        if library_learn_interval > 0 && self.cycle_count % library_learn_interval == 0 {
            let _ = self.run_library_learning();
        }

        // Evaluate watches against world delta (Phase 11e).
        {
            let snap_before = watch::take_snapshot(
                &self.engine,
                self.cycle_count.saturating_sub(1),
            );
            let snap_after = watch::take_snapshot(&self.engine, self.cycle_count);
            let firings = watch::evaluate_watches(
                &mut self.watches,
                &snap_before,
                &snap_after,
                &self.engine,
                self.cycle_count,
            );

            // Record watch firings as WM observations + provenance.
            for firing in &firings {
                let _ = self.working_memory.push(WorkingMemoryEntry {
                    id: 0,
                    content: format!(
                        "Watch '{}' fired: {}",
                        firing.watch_name, firing.condition_summary
                    ),
                    symbols: vec![],
                    kind: WorkingMemoryKind::Observation,
                    timestamp: 0,
                    relevance: 0.7,
                    source_cycle: self.cycle_count,
                    reference_count: 0,
                    access_timestamps: Vec::new(),
                });

                // Store provenance for the firing.
                let derived_id = self
                    .engine
                    .resolve_or_create_relation("agent:watch_event")
                    .unwrap_or(self.predicates.has_status);
                let mut prov = ProvenanceRecord::new(
                    derived_id,
                    DerivationKind::WatchFired {
                        watch_id: firing.watch_name.clone(),
                        condition_summary: firing.condition_summary.clone(),
                    },
                )
                .with_confidence(0.8);
                let _ = self.engine.store_provenance(&mut prov);
            }

            self.last_watch_firings = firings;
        }

        // Periodic autonomous goal generation.
        let gen_interval = self.config.goal_generation.generate_every_n_cycles;
        if gen_interval > 0 && self.cycle_count % gen_interval == 0 {
            let _ = self.generate_goals();
        }

        // Auto-decompose stalled goals.
        self.decompose_stalled_goals();

        // Consume a cycle from the active project's budget (Phase 11g).
        if let Some(active_pid) = self.agenda.active_project {
            if let Some(proj) = self.projects.iter_mut().find(|p| p.id == active_pid) {
                proj.consume_cycle();
            }
        }

        // Auto-project-completion: check if active project's goals are all done.
        if let Some(active_pid) = self.agenda.active_project {
            if let Some(proj) = self.projects.iter().find(|p| p.id == active_pid) {
                if !proj.goals.is_empty() && project_progress(proj, &self.goals) >= 1.0 {
                    // Mark project completed and switch to the next one.
                    if let Some(proj_mut) = self.projects.iter_mut().find(|p| p.id == active_pid)
                    {
                        let _ = update_project_status(
                            &self.engine,
                            proj_mut,
                            ProjectStatus::Completed,
                            &self.project_predicates,
                        );
                    }
                    self.agenda.select_active(&self.projects);

                    // Log context switch in WM.
                    let switch_msg = match self.agenda.active_project {
                        Some(new_pid) => {
                            let new_name = self
                                .projects
                                .iter()
                                .find(|p| p.id == new_pid)
                                .map(|p| p.name.as_str())
                                .unwrap_or("?");
                            format!("Project completed, switching to \"{new_name}\"")
                        }
                        None => "Project completed, no more projects in agenda".to_string(),
                    };
                    let _ = self.working_memory.push(WorkingMemoryEntry {
                        id: 0,
                        content: switch_msg,
                        symbols: vec![],
                        kind: WorkingMemoryKind::GoalUpdate,
                        timestamp: 0,
                        relevance: 0.9,
                        source_cycle: self.cycle_count,
                        reference_count: 0,
                        access_timestamps: Vec::new(),
                    });
                }
            }
        }

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
            self.cycle_count,
        )
    }

    /// Recall episodic memories by query symbols.
    pub fn recall(&self, query: &[SymbolId], top_k: usize) -> AgentResult<Vec<EpisodicEntry>> {
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

    // ── Channel registry (Phase 12a) ────────────────────────────────────

    /// Get a shared reference to the channel registry.
    pub fn channel_registry(&self) -> &ChannelRegistry {
        &self.channel_registry
    }

    /// Get a mutable reference to the channel registry.
    pub fn channel_registry_mut(&mut self) -> &mut ChannelRegistry {
        &mut self.channel_registry
    }

    /// Register a communication channel.
    pub fn register_channel(
        &mut self,
        channel: Box<dyn super::channel::CommChannel>,
    ) -> ChannelResult<()> {
        self.channel_registry.register(channel)
    }

    /// Create and register an operator channel wrapping the current sink,
    /// and return an `InboundHandle` for the UI event loop.
    pub fn setup_operator_channel(&mut self) -> InboundHandle {
        let op = super::operator_channel::OperatorChannel::new(Arc::clone(&self.sink));
        let handle = op.inbound_handle();
        // Registration cannot fail here: we are creating the first operator channel.
        let _ = self.channel_registry.register(Box::new(op));
        handle
    }

    /// Drain all pending inbound messages from all registered channels.
    pub fn drain_inbound(&mut self) -> Vec<InboundMessage> {
        self.channel_registry
            .drain_all()
            .into_iter()
            .map(|(_id, msg)| msg)
            .collect()
    }

    // ── Conversation state (Phase 12b) ─────────────────────────────────

    /// Get a shared reference to the conversation state.
    pub fn conversation_state(&self) -> &ConversationState {
        &self.conversation_state
    }

    /// Get a mutable reference to the conversation state.
    pub fn conversation_state_mut(&mut self) -> &mut ConversationState {
        &mut self.conversation_state
    }

    /// Set the response detail level for grounded dialogue.
    pub fn set_response_detail(&mut self, detail: ResponseDetail) {
        self.conversation_state.response_detail = detail;
    }

    // ── Constraint checking (Phase 12c) ────────────────────────────────

    /// Get a shared reference to the constraint checker.
    pub fn constraint_checker(&self) -> &ConstraintChecker {
        &self.constraint_checker
    }

    /// Get a mutable reference to the constraint checker.
    pub fn constraint_checker_mut(&mut self) -> &mut ConstraintChecker {
        &mut self.constraint_checker
    }

    /// Run constraint checks on a grounded response and produce an
    /// `OutboundMessage` with the check status populated.
    ///
    /// Returns `(OutboundMessage, EmissionDecision)` — the caller should
    /// respect the decision (emit or suppress) based on channel kind.
    pub fn check_and_wrap_grounded(
        &mut self,
        response: &super::conversation::GroundedResponse,
        channel_id: &str,
        channel_kind: super::channel::ChannelKind,
    ) -> (super::channel_message::OutboundMessage, EmissionDecision) {
        let detail = self.conversation_state.response_detail;
        let grammar = self.conversation_state.grammar.clone();

        // Run the constraint pipeline.
        let outcome = self.constraint_checker.check_grounded(
            response, channel_id, channel_kind, &self.engine,
        );

        let decision = emission_decision(channel_kind, &outcome);

        // Build the outbound message with constraint status.
        let mut msg = super::channel_message::OutboundMessage::grounded(response, detail, grammar);
        msg.constraint_check = ConstraintCheckStatus::from_outcome(&outcome);

        // Record emission in the budget.
        if decision == EmissionDecision::Emit {
            self.constraint_checker.record_emission(channel_id);
        }

        (msg, decision)
    }

    /// Synthesize human-readable narrative from the agent's working memory findings.
    pub fn synthesize_findings(&self, goal: &str) -> super::synthesize::NarrativeSummary {
        super::synthesize::synthesize(goal, self.working_memory.entries(), &self.engine)
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
                    && g.is_stalled(
                        self.cycle_count,
                        resource::dynamic_stall_threshold(
                            g.estimated_effort.as_ref(),
                        ),
                    )
            })
            .map(|g| g.symbol_id)
            .collect();

        for goal_id in stalled_ids {
            let _ = self.decompose_stalled_goal(goal_id);
        }
    }

    /// Decompose a stalled goal into sub-goals using HTN methods.
    ///
    /// Selects the best decomposition method from the registry, instantiates it
    /// to produce a TaskTree DAG, creates child goals with `blocked_by` relations,
    /// suspends the parent, and records `HtnDecomposition` provenance. Falls back
    /// to comma/and splitting if no HTN method matches.
    pub fn decompose_stalled_goal(&mut self, goal_id: SymbolId) -> AgentResult<Vec<SymbolId>> {
        let (goal_clone, parent_idx) = {
            let (idx, goal) = self
                .goals
                .iter()
                .enumerate()
                .find(|(_, g)| g.symbol_id == goal_id)
                .ok_or(AgentError::GoalNotFound {
                    goal_id: goal_id.get(),
                })?;
            (goal.clone(), idx)
        };

        // HTN decomposition: select method, build DAG, extract sub-goals + deps.
        let output: DecompositionOutput =
            decomposition::decompose_goal_htn(&goal_clone, &self.engine, &self.method_registry_htn);

        let sub_tuples: Vec<(&str, u8, &str)> = output
            .sub_goals
            .iter()
            .map(|(d, p, c)| (d.as_str(), *p, c.as_str()))
            .collect();

        let parent = &mut self.goals[parent_idx];
        let children = goal::decompose_goal(&self.engine, parent, &sub_tuples, &self.predicates)?;

        let child_ids: Vec<SymbolId> = children.iter().map(|c| c.symbol_id).collect();

        // Apply dependency edges: set `blocked_by` on child goals.
        for &(blocker_idx, blocked_idx) in &output.dependencies {
            if blocker_idx < child_ids.len() && blocked_idx < child_ids.len() {
                let blocker_id = child_ids[blocker_idx];
                let blocked_id = child_ids[blocked_idx];

                // Persist in KG.
                let _ = self.engine.add_triple(&crate::graph::Triple::new(
                    blocked_id,
                    self.predicates.blocked_by,
                    blocker_id,
                ));
            }
        }

        // Suspend the parent.
        goal::update_goal_status(
            &self.engine,
            &mut self.goals[parent_idx],
            GoalStatus::Suspended,
            &self.predicates,
        )?;

        // Add children to the goals list, attaching blocked_by info.
        for (i, mut child) in children.into_iter().enumerate() {
            // Gather blockers for this child from the dependency list.
            let blockers: Vec<SymbolId> = output
                .dependencies
                .iter()
                .filter(|&&(_, blocked)| blocked == i)
                .filter_map(|&(blocker, _)| child_ids.get(blocker).copied())
                .collect();
            child.blocked_by = blockers;
            self.goals.push(child);
        }

        // Record HTN provenance.
        let mut prov = ProvenanceRecord::new(
            goal_id,
            crate::provenance::DerivationKind::HtnDecomposition {
                method_name: output.tree.method_name.clone(),
                strategy: output.tree.strategy.to_string(),
                subtask_count: output.sub_goals.len(),
            },
        )
        .with_confidence(0.9);
        let _ = self.engine.store_provenance(&mut prov);

        // Update method stats.
        if let Some(method) = self
            .method_registry_htn
            .methods_mut()
            .iter_mut()
            .find(|m| m.name == output.tree.method_name)
        {
            method.usage_count += 1;
        }

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
        if self
            .plans
            .get(&key)
            .is_some_and(|p| p.status == PlanStatus::Active)
        {
            return Ok(&self.plans[&key]);
        }

        let goal = self
            .goals
            .iter()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound { goal_id: key })?
            .clone();

        let attempt = self.plans.get(&key).map(|p| p.attempt + 1).unwrap_or(0);

        let new_plan = plan::generate_plan(&goal, &self.engine, &self.working_memory, attempt)?;

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
            access_timestamps: Vec::new(),
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

        let new_plan = plan::generate_plan(&goal, &self.engine, &self.working_memory, attempt)?;

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
            access_timestamps: Vec::new(),
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
            access_timestamps: Vec::new(),
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
                Adjustment::IncreasePriority { goal_id, to, .. }
                | Adjustment::DecreasePriority { goal_id, to, .. } => {
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
                Adjustment::ReformulateGoal {
                    goal_id,
                    relaxed_criteria,
                    reason: _,
                } => {
                    // Find the goal, reformulate it, add the replacement.
                    if let Some(idx) = self.goals.iter().position(|g| g.symbol_id == *goal_id) {
                        let original = &mut self.goals[idx];
                        if let Ok(replacement) = goal::reformulate_goal(
                            &self.engine,
                            original,
                            relaxed_criteria,
                            &self.predicates,
                        ) {
                            let category = metacognition::categorize_goal(&replacement);
                            let competence = self.competence_model.category_competence(&category);

                            // Record metacognitive provenance.
                            let mut prov = ProvenanceRecord::new(
                                *goal_id,
                                DerivationKind::MetacognitiveEvaluation {
                                    goal: *goal_id,
                                    signal: "reformulated".to_string(),
                                    improvement_rate: 0.0,
                                    competence,
                                },
                            )
                            .with_confidence(0.8);
                            let _ = self.engine.store_provenance(&mut prov);

                            self.goals.push(replacement);
                            applied += 1;
                        }
                    }
                }
                Adjustment::SuspendGoal { goal_id, reason: _ } => {
                    if let Some(g) = self.goals.iter_mut().find(|g| g.symbol_id == *goal_id) {
                        let category = metacognition::categorize_goal(g);
                        let competence = self.competence_model.category_competence(&category);
                        let _ = goal::update_goal_status(
                            &self.engine,
                            g,
                            GoalStatus::Suspended,
                            &self.predicates,
                        );

                        // Record metacognitive provenance.
                        let mut prov = ProvenanceRecord::new(
                            *goal_id,
                            DerivationKind::MetacognitiveEvaluation {
                                goal: *goal_id,
                                signal: "suspended".to_string(),
                                improvement_rate: 0.0,
                                competence,
                            },
                        )
                        .with_confidence(0.8);
                        let _ = self.engine.store_provenance(&mut prov);

                        applied += 1;
                    }
                }
                Adjustment::ReviseBeliefs {
                    goal_id: _,
                    retract: _,
                    assert_syms: _,
                } => {
                    // Belief revision: retract/assert triples.
                    // The current KG does not support triple removal, so we
                    // just record the intent. Assertions are added as triples.
                    // Full retraction support will come with TMS integration.
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
    // Library learning
    // -----------------------------------------------------------------------

    /// Run a library learning cycle: discover reusable abstractions from recent code.
    ///
    /// Collects code entities from the KG, anti-unifies them, scores candidates,
    /// and stores the top abstractions as learned templates.
    pub fn run_library_learning(
        &self,
    ) -> AgentResult<super::library_learn::LibraryLearningResult> {
        let learner = super::library_learn::LibraryLearner::with_defaults();
        learner.run_cycle(&self.engine)
    }

    // -----------------------------------------------------------------------
    // Autonomous goal generation
    // -----------------------------------------------------------------------

    /// Run the autonomous goal generation pipeline.
    ///
    /// Updates drives from current engine/WM state, collects signals from
    /// multiple sources, deduplicates, and activates the top proposals.
    pub fn generate_goals(&mut self) -> AgentResult<GoalGenerationResult> {
        let goal_symbols: Vec<SymbolId> = goal::active_goals(&self.goals)
            .iter()
            .map(|g| g.symbol_id)
            .collect();

        // Update drive strengths.
        self.drives.update(
            &self.engine,
            &self.working_memory,
            &goal_symbols,
            self.cycle_count,
        );

        // Run the three-phase pipeline.
        let gen_result = goal_generation::generate_goals(
            &self.engine,
            &self.goals,
            &self.working_memory,
            &self.drives,
            &self.config.goal_generation,
            &self.predicates,
            self.cycle_count,
            self.last_impasse.as_ref(),
            self.last_reflection.as_ref(),
            &self.last_watch_firings,
        )?;

        // Activate generated goals by creating them in the goals list.
        for proposal in &gen_result.activated {
            let mut g = goal::create_goal(
                &self.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &self.predicates,
            )?;
            g.source = Some(proposal.source.clone());

            // Record provenance for autonomously generated goals.
            let (drive_name, drive_strength) =
                goal_generation::provenance_from_source(&proposal.source);
            let mut prov = ProvenanceRecord::new(
                g.symbol_id,
                DerivationKind::AutonomousGoalGeneration {
                    drive: drive_name,
                    strength: drive_strength,
                },
            )
            .with_confidence(proposal.feasibility);
            let _ = self.engine.store_provenance(&mut prov);

            self.goals.push(g);
        }

        // Store dormant proposals as goals with Dormant status.
        for proposal in &gen_result.dormant {
            let mut g = goal::create_goal(
                &self.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &self.predicates,
            )?;
            g.source = Some(proposal.source.clone());
            goal::update_goal_status(
                &self.engine,
                &mut g,
                GoalStatus::Dormant,
                &self.predicates,
            )?;
            self.goals.push(g);
        }

        // Log generation result to WM.
        let _ = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: format!(
                "Goal generation: {} activated, {} dormant, {} deduplicated, {} infeasible. Drives: curiosity={:.2} coherence={:.2} completeness={:.2} efficiency={:.2}",
                gen_result.activated.len(),
                gen_result.dormant.len(),
                gen_result.deduplicated,
                gen_result.infeasible,
                gen_result.drive_strengths[0],
                gen_result.drive_strengths[1],
                gen_result.drive_strengths[2],
                gen_result.drive_strengths[3],
            ),
            symbols: vec![],
            kind: WorkingMemoryKind::Inference,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: self.cycle_count,
            reference_count: 0,
            access_timestamps: Vec::new(),
        });

        // Clear impasse after it's been consumed.
        self.last_impasse = None;

        Ok(gen_result)
    }

    /// Get the current drive system state.
    pub fn drives(&self) -> &DriveSystem {
        &self.drives
    }

    /// Get the current argumentation audience.
    pub fn audience(&self) -> &Audience {
        &self.audience
    }

    /// Set the argumentation audience (operational mode).
    pub fn set_audience(&mut self, audience: Audience) {
        self.audience = audience;
    }

    // -----------------------------------------------------------------------
    // Watches (Phase 11e)
    // -----------------------------------------------------------------------

    /// Add a new world-monitoring watch. Returns the watch ID.
    pub fn add_watch(
        &mut self,
        name: &str,
        condition: super::watch::WatchCondition,
        action: super::watch::WatchAction,
        cooldown_cycles: u64,
    ) -> AgentResult<String> {
        let id = watch::generate_watch_id();
        self.watches.push(Watch {
            id: id.clone(),
            name: name.to_string(),
            condition,
            action,
            enabled: true,
            cooldown_cycles,
            last_fired_cycle: 0,
        });
        Ok(id)
    }

    /// Remove a watch by ID.
    pub fn remove_watch(&mut self, id: &str) -> AgentResult<()> {
        let idx = self
            .watches
            .iter()
            .position(|w| w.id == id)
            .ok_or_else(|| AgentError::WatchNotFound {
                watch_id: id.to_string(),
            })?;
        self.watches.remove(idx);
        Ok(())
    }

    /// Get all registered watches.
    pub fn watches(&self) -> &[Watch] {
        &self.watches
    }

    // -----------------------------------------------------------------------
    // Resource awareness (Phase 11g)
    // -----------------------------------------------------------------------

    /// Estimate effort for a goal using the CBR case base.
    pub fn estimate_goal_effort(&self, goal_id: SymbolId) -> AgentResult<resource::EffortEstimate> {
        let goal = self
            .goals
            .iter()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        resource::estimate_effort(goal, &self.effort_index, &self.engine, 3)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: e.to_string(),
            })
    }

    /// Prune dormant learned methods from the method index (Phase 11h).
    pub fn prune_methods(&mut self) -> Vec<SymbolId> {
        let session_cycles = self.cycle_count.saturating_sub(self.session_start_cycle).max(1);
        let mut methods: Vec<chunking::LearnedMethod> =
            self.method_index.methods().to_vec();
        let pruned = chunking::prune_dormant(
            &mut methods,
            self.cycle_count,
            session_cycles,
            self.chunking_config.dormant_sessions,
        );
        // Rebuild the index from remaining methods.
        if !pruned.is_empty() {
            self.method_index = MethodIndex::from_methods(methods);
        }
        pruned
    }

    /// Get a snapshot of all learned methods (Phase 11h).
    pub fn learned_methods(&self) -> &[chunking::LearnedMethod] {
        self.method_index.methods()
    }

    // -----------------------------------------------------------------------
    // Projects & Agenda
    // -----------------------------------------------------------------------

    /// Create a new project backed by a KG microtheory.
    ///
    /// Returns the project's symbol ID. The project is added to the agenda
    /// with the given priority.
    pub fn create_project(
        &mut self,
        name: &str,
        description: &str,
        scope_concepts: &[&str],
        priority: u8,
    ) -> AgentResult<SymbolId> {
        let proj = project::create_project(
            &self.engine,
            name,
            description,
            scope_concepts,
            &self.project_predicates,
            &self.predicates,
        )?;
        let id = proj.id;
        self.agenda.add_project(id, priority);
        self.projects.push(proj);
        Ok(id)
    }

    /// Get all projects (read-only).
    pub fn projects(&self) -> &[Project] {
        &self.projects
    }

    /// Get the agenda (read-only).
    pub fn agenda(&self) -> &Agenda {
        &self.agenda
    }

    /// Set the active project by ID.
    pub fn set_active_project(&mut self, project_id: SymbolId) -> AgentResult<()> {
        if !self.projects.iter().any(|p| p.id == project_id) {
            return Err(AgentError::ProjectNotFound {
                project_id: project_id.get(),
            });
        }
        self.agenda.active_project = Some(project_id);
        Ok(())
    }

    /// Add an existing goal to a project.
    pub fn assign_goal_to_project(
        &mut self,
        goal_id: SymbolId,
        project_id: SymbolId,
    ) -> AgentResult<()> {
        let proj = self
            .projects
            .iter_mut()
            .find(|p| p.id == project_id)
            .ok_or(AgentError::ProjectNotFound {
                project_id: project_id.get(),
            })?;
        project::add_goal_to_project(&self.engine, proj, goal_id, &self.project_predicates)
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
        let cycle_bytes =
            bincode::serialize(&self.cycle_count).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize cycle count: {e}"),
            })?;
        store
            .put_meta(b"agent:cycle_count", &cycle_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist cycle count: {e}"),
            })?;

        // Persist drive system.
        let drive_bytes =
            bincode::serialize(&self.drives).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize drive system: {e}"),
            })?;
        store
            .put_meta(b"agent:drive_system", &drive_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist drive system: {e}"),
            })?;

        // Persist HTN method stats.
        let method_stats = self.method_registry_htn.export_stats();
        let stats_bytes =
            bincode::serialize(&method_stats).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize method stats: {e}"),
            })?;
        store
            .put_meta(b"agent:method_stats", &stats_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist method stats: {e}"),
            })?;

        // Persist audience.
        let audience_bytes =
            bincode::serialize(&self.audience).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize audience: {e}"),
            })?;
        store
            .put_meta(b"agent:audience", &audience_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist audience: {e}"),
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

        // Generate and persist session summary.
        let active_project_name = self
            .agenda
            .active_project
            .and_then(|pid| self.projects.iter().find(|p| p.id == pid))
            .map(|p| p.name.as_str());
        let summary = generate_session_summary(
            &self.working_memory,
            &self.goals,
            active_project_name,
            self.session_start_cycle,
            self.cycle_count,
        );
        let summary_bytes =
            bincode::serialize(&summary).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize session summary: {e}"),
            })?;
        store
            .put_meta(b"agent:session_summary", &summary_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist session summary: {e}"),
            })?;

        // Persist agenda.
        let agenda_bytes =
            bincode::serialize(&self.agenda).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize agenda: {e}"),
            })?;
        store
            .put_meta(b"agent:agenda", &agenda_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist agenda: {e}"),
            })?;

        // Persist competence model (Phase 11f).
        let competence_bytes = bincode::serialize(&self.competence_model).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize competence model: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:competence_model", &competence_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist competence model: {e}"),
            })?;

        // Persist failure cases for HNSW rebuild (Phase 11f).
        let failure_bytes = bincode::serialize(self.failure_index.cases()).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize failure cases: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:failure_cases", &failure_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist failure cases: {e}"),
            })?;

        // Persist effort cases for HNSW rebuild (Phase 11g).
        let effort_bytes = bincode::serialize(self.effort_index.cases()).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize effort cases: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:effort_cases", &effort_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist effort cases: {e}"),
            })?;

        // Persist improvement history (Phase 11g).
        let history_bytes =
            bincode::serialize(&self.improvement_history).map_err(|e| {
                AgentError::ConsolidationFailed {
                    message: format!("failed to serialize improvement history: {e}"),
                }
            })?;
        store
            .put_meta(b"agent:improvement_history", &history_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist improvement history: {e}"),
            })?;

        // Persist learned methods for HNSW rebuild (Phase 11h).
        let methods_bytes =
            bincode::serialize(self.method_index.methods()).map_err(|e| {
                AgentError::ConsolidationFailed {
                    message: format!("failed to serialize learned methods: {e}"),
                }
            })?;
        store
            .put_meta(b"agent:learned_methods", &methods_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist learned methods: {e}"),
            })?;

        // Store competence as KG triples for SPARQL queryability.
        let success_rate_pred = self
            .engine
            .resolve_or_create_relation("agent:success_rate")
            .unwrap_or(self.predicates.has_status);
        let _ = self
            .competence_model
            .store_competence_triples(&self.engine, success_rate_pred);

        // Persist watches.
        watch::persist_watches(&self.engine, &self.watches)?;

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
        let project_predicates = ProjectPredicates::init(&engine)?;

        let mut tool_registry = ToolRegistry::new();
        Self::register_builtin_tools(&mut tool_registry, &predicates, &engine);

        let store = engine.store();

        // Restore working memory.
        let mut working_memory = match (
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
            .and_then(|bytes| {
                bincode::deserialize::<crate::compartment::psyche::Psyche>(&bytes).ok()
            })
            .or_else(|| engine.compartments().and_then(|cm| cm.psyche()));

        // Restore drive system: prefer persisted state, fall back to config defaults.
        let drives = store
            .get_meta(b"agent:drive_system")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<DriveSystem>(&bytes).ok())
            .unwrap_or_else(|| DriveSystem::with_thresholds(config.goal_generation.drive_thresholds));

        // Restore audience: prefer persisted state, fall back to exploration default.
        let audience = store
            .get_meta(b"agent:audience")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<Audience>(&bytes).ok())
            .unwrap_or_else(Audience::exploration);

        // Initialize HTN method registry and restore persisted stats.
        let mut htn_registry = MethodRegistry::new();
        decomposition::register_builtin_methods(&mut htn_registry);
        if let Some(stats_bytes) = store.get_meta(b"agent:method_stats").ok().flatten() {
            if let Ok(stats) = bincode::deserialize::<Vec<MethodStats>>(&stats_bytes) {
                htn_registry.import_stats(&stats);
            }
        }

        // Restore projects from KG.
        let projects =
            project::restore_projects(&engine, &project_predicates, &predicates).unwrap_or_default();

        // Restore agenda from durable store.
        let agenda = store
            .get_meta(b"agent:agenda")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<Agenda>(&bytes).ok())
            .unwrap_or_default();

        // Restore session summary and inject into WM for cross-session continuity.
        if let Some(summary_bytes) = store.get_meta(b"agent:session_summary").ok().flatten() {
            if let Ok(summary) = bincode::deserialize::<SessionSummary>(&summary_bytes) {
                let _ = restore_session_summary(&mut working_memory, &summary, cycle_count);
            }
        }

        // Restore competence model (Phase 11f).
        let competence_model = store
            .get_meta(b"agent:competence_model")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<CompetenceModel>(&bytes).ok())
            .unwrap_or_default();

        // Restore failure index from persisted cases (Phase 11f).
        let failure_index = store
            .get_meta(b"agent:failure_cases")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<Vec<FailureCase>>(&bytes).ok())
            .map(FailureIndex::from_cases)
            .unwrap_or_default();

        // Restore effort index from persisted cases (Phase 11g).
        let effort_index = store
            .get_meta(b"agent:effort_cases")
            .ok()
            .flatten()
            .and_then(|bytes| {
                bincode::deserialize::<Vec<resource::EffortCase>>(&bytes).ok()
            })
            .map(EffortIndex::from_cases)
            .unwrap_or_default();

        // Restore improvement history (Phase 11g).
        let improvement_history = store
            .get_meta(b"agent:improvement_history")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<ImprovementHistory>(&bytes).ok())
            .unwrap_or_default();

        // Restore learned methods from persisted cases (Phase 11h).
        let method_index = store
            .get_meta(b"agent:learned_methods")
            .ok()
            .flatten()
            .and_then(|bytes| {
                bincode::deserialize::<Vec<chunking::LearnedMethod>>(&bytes).ok()
            })
            .map(MethodIndex::from_methods)
            .unwrap_or_default();

        // Restore watches from durable store.
        let watches = watch::restore_watches(&engine).unwrap_or_default();

        let session_start_cycle = cycle_count;
        let chunking_config = config.chunking.clone();

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
            drives,
            audience,
            method_registry_htn: htn_registry,
            last_impasse: None,
            projects,
            project_predicates,
            agenda,
            session_start_cycle,
            watches,
            last_watch_firings: Vec::new(),
            competence_model,
            failure_index,
            effort_index,
            improvement_history,
            method_index,
            chunking_config,
            channel_registry: ChannelRegistry::new(),
            conversation_state: ConversationState::default(),
            constraint_checker: ConstraintChecker::new(),
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
            .field("has_impasse", &self.last_impasse.is_some())
            .field("projects", &self.projects.len())
            .field("active_project", &self.agenda.active_project)
            .field("watches", &self.watches.len())
            .field(
                "calibration_error",
                &self.competence_model.calibration_error(),
            )
            .field("failure_index_size", &self.failure_index.len())
            .field("learned_methods", &self.method_index.len())
            .field("channels", &self.channel_registry)
            .field("response_detail", &self.conversation_state.response_detail)
            .field("constraint_checker", &self.constraint_checker)
            .finish()
    }
}
