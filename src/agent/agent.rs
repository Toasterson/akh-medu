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
use super::constraint_check::{ConstraintChecker, EmissionDecision, emission_decision};
use super::conversation::{ConversationState, ResponseDetail};
use super::interlocutor::InterlocutorRegistry;
use super::operator_channel::InboundHandle;
use super::decomposition::{self, DecompositionOutput, MethodRegistry, MethodStats};
use super::drives::DriveSystem;
use super::error::{AgentError, AgentResult};
use super::goal::{self, Goal, GoalStatus};
use super::goal_generation::{self, GoalGenerationConfig, GoalGenerationResult};
use super::metacognition::{self, CompetenceModel, FailureCase, FailureIndex, MetacognitionConfig};
use super::chunking::{self, ChunkingConfig, MethodIndex};
use super::resource::{self, EffortIndex, ImprovementHistory};
use super::watch::{self, Watch, WatchFiring};
use super::memory::{
    ConsolidationConfig, ConsolidationResult, EpisodicEntry, SessionSummary, WorkingMemory,
    WorkingMemoryEntry, WorkingMemoryKind, consolidate, generate_session_summary, index_cycle_episode,
    recall_episodes, recall_recent_episodes, restore_session_summary,
};
use super::ooda::{self, DecisionImpasse, OodaCycleResult};
use super::vsa_decide::DecideMode;
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
    /// Sleep/consolidation cycle settings (Phase 11i).
    pub sleep: super::sleep::SleepConfig,
    /// Directed curiosity settings (Phase 11j).
    pub curiosity: super::curiosity::DirectedCuriosityConfig,
    /// Maximum consecutive OODA cycles with no goal progress before stopping early (default: 4).
    ///
    /// After this many back-to-back `NoChange` cycles the loop terminates without
    /// error, yielding the best findings so far.  This prevents the agent from
    /// spinning against a sparse knowledge graph.  Set to 0 to disable.
    pub max_empty_cycles: usize,
    /// Goal queue hygiene settings.
    pub hygiene: super::hygiene::HygieneConfig,
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
            sleep: super::sleep::SleepConfig::default(),
            curiosity: super::curiosity::DirectedCuriosityConfig::default(),
            max_empty_cycles: 4,
            hygiene: super::hygiene::HygieneConfig::default(),
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
    /// Affective valence at consolidation time — used for mood-congruent retrieval (Phase 23).
    pub has_affective_valence: SymbolId,
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
            has_affective_valence: engine
                .resolve_or_create_relation("agent:has_affective_valence")?,
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
    /// Non-blocking read-only snapshot of the current psyche.
    /// Shared via `Arc` so monitoring tools can read it without acquiring the
    /// agent mutex.  Kept in sync after every psyche mutation.
    pub(crate) psyche_snapshot: Arc<std::sync::RwLock<Option<crate::compartment::psyche::Psyche>>>,
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
    /// Interlocutor registry — social KG and theory of mind (Phase 12d).
    pub(crate) interlocutor_registry: InterlocutorRegistry,
    /// Capability token registry for multi-agent communication (Phase 12g).
    pub(crate) token_registry: super::multi_agent::TokenRegistry,
    /// Personal information manager (Phase 13e).
    pub(crate) pim_manager: super::pim::PimManager,
    /// Calendar & temporal reasoning manager (Phase 13f).
    pub(crate) calendar_manager: super::calendar::CalendarManager,
    /// Preference learning & proactive assistance manager (Phase 13g).
    pub(crate) preference_manager: super::preference::PreferenceManager,
    /// Causal world model manager (Phase 15a).
    pub(crate) causal_manager: super::causal::CausalManager,
    /// Event calculus engine — temporal fluent reasoning (Phase 15b).
    pub(crate) ec_engine: super::event_calculus::EventCalculusEngine,
    /// Prediction tracker — causal model refinement (Phase 15c).
    pub(crate) prediction_tracker: super::counterfactual::PredictionTracker,
    /// TD-learned value function for planning (Phase 16a).
    pub(crate) value_function: super::state_value::ValueFunction,
    /// Evidence manager — Dempster-Shafer belief intervals (Phase 17).
    pub(crate) evidence_manager: super::evidence::EvidenceManager,
    /// Source reliability manager — Admiralty ratings + Bayesian trust (Phase 18a).
    pub(crate) reliability_manager: super::source_reliability::SourceReliabilityManager,
    /// Epistemic state manager — what agents know/believe (Phase 19).
    pub(crate) epistemic_manager: super::epistemic::EpistemicStateManager,
    /// Active inference engine — prediction, surprise, free energy (Phase 20).
    pub(crate) active_inference: super::active_inference::ActiveInferenceEngine,
    /// Temporal experience — felt duration, boredom, urgency (Phase 15d).
    pub(crate) temporal_experience: super::temporal_experience::TemporalExperience,
    /// Affective system — mood, somatic markers, emotional salience (Phase 23).
    pub(crate) affective_system: super::affect::AffectiveSystem,
    /// Contact manager — unified people/identity resolution (Phase 25a).
    pub(crate) contact_manager: super::contact::ContactManager,
    /// Relationship graph — interpersonal relationships between contacts (Phase 25b).
    pub(crate) relationship_graph: super::contact_rel::RelationshipGraph,
    /// Per-person conversation memory index (Phase 25c).
    pub(crate) person_memory: super::contact_memory::PersonMemoryIndex,
    /// Communication style profile manager (Phase 25d).
    pub(crate) style_manager: super::contact_style::StyleManager,
    /// Dialogue state manager backed by KG microtheories.
    pub(crate) dialogue_manager: super::dialogue::DialogueManager,
    /// Sleep/consolidation cycle state (Phase 11i).
    pub(crate) sleep_cycle: super::sleep::SleepCycle,
    /// Directed curiosity configuration (Phase 11j).
    pub(crate) curiosity_config: super::curiosity::DirectedCuriosityConfig,
    /// Goal queue hygiene configuration.
    pub(crate) hygiene_config: super::hygiene::HygieneConfig,
    /// Optional WASM tool runtime (only when `wasm-tools` feature is enabled).
    #[cfg(feature = "wasm-tools")]
    pub(crate) wasm_runtime: Option<super::wasm_runtime::WasmToolRuntime>,
    /// Per-agent model manager for local Gemma inference via Ollama (optional).
    pub(crate) model_manager: Option<std::sync::Arc<super::model_manager::AgentModelManager>>,
    /// Alternative chat/agentic backing selected per-akh (ADR 070). When set,
    /// `run_agentic_turn` drives this instead of wrapping `model_manager`
    /// (Ollama). `None` ⇒ the Ollama path via `model_manager`.
    pub(crate) chat_backend: Option<std::sync::Arc<dyn super::chat_backend::ChatBackend>>,
    /// Working-context assembly strategy (Phase 44a, ADR 065). `None` ⇒ the
    /// byte-identical pass-through manager (current behaviour). When set,
    /// `run_agentic_turn` builds each prompt through it.
    pub(crate) context_manager: Option<Box<dyn super::context::ContextManager>>,
    /// Delegation manager for offloading LLM-class tasks (Phase 36).
    #[cfg(feature = "delegation")]
    pub(crate) delegation_manager: super::delegation::DelegationManager,
}

impl Agent {
    /// Register all built-in tools (core + external).
    fn register_builtin_tools(
        registry: &mut ToolRegistry,
        predicates: &AgentPredicates,
        engine: &Engine,
        #[cfg(feature = "delegation")]
        delegation_pending: std::sync::Arc<std::sync::Mutex<Vec<super::delegation::DelegationTask>>>,
    ) {
        // Core KG tools.
        registry.register(Box::new(tools::KgQueryTool));
        registry.register(Box::new(tools::KgMutateTool));
        registry.register(Box::new(tools::MemoryRecallTool::new(predicates.clone())));
        registry.register(Box::new(tools::ReasonTool));
        registry.register(Box::new(tools::SimilaritySearchTool));

        // External world-interaction tools.
        // Reads are sandboxed to the whole workspace data dir; writes stay
        // confined to a `scratch/` subdir. With no data dir (in-memory engine)
        // the filesystem fails closed.
        let fs: std::sync::Arc<dyn crate::fs::Filesystem> = match engine.config().data_dir.as_ref()
        {
            Some(dir) => std::sync::Arc::new(crate::fs::LocalFs::new(
                vec![dir.clone()],
                vec![dir.join("scratch")],
            )),
            None => std::sync::Arc::new(crate::fs::LocalFs::denied()),
        };
        registry.register(Box::new(tools::FileIoTool::new(fs.clone())));
        // Claude-Code-parity coding tools, sharing the same sandbox so reads
        // and writes stay confined to the akh's roots.
        registry.register(Box::new(tools::EditTool::new(fs.clone())));
        registry.register(Box::new(tools::GlobTool::new(fs.clone())));
        registry.register(Box::new(tools::GrepTool::new(fs.clone())));
        registry.register(Box::new(tools::HttpFetchTool));
        registry.register(Box::new(tools::WebSearchTool));
        registry.register(Box::new(tools::ShellExecTool));
        registry.register(Box::new(tools::UserInteractTool));
        // Structured task list (TodoWrite-style working state).
        registry.register(Box::new(tools::TaskListTool::new(std::sync::Arc::new(
            std::sync::Mutex::new(Vec::new()),
        ))));

        // Autonomous reasoning tools.
        registry.register(Box::new(tools::InferRulesTool));
        registry.register(Box::new(tools::GapAnalysisTool));

        // Ingest tools. The single-file ingesters share the agent's sandbox so
        // their reads stay within the workspace (code/journal ingesters still
        // read raw — sandboxing those walks folds into the ingestion rework).
        registry.register(Box::new(tools::CsvIngestTool::sandboxed(fs.clone())));
        registry.register(Box::new(tools::TextIngestTool::sandboxed(fs.clone())));
        registry.register(Box::new(tools::CodeIngestTool));
        registry.register(Box::new(tools::ContentIngestTool));
        registry.register(Box::new(tools::JournalIngestTool));

        // Library search.
        registry.register(Box::new(tools::LibrarySearchTool));

        // Documentation generation.
        registry.register(Box::new(tools::DocGenTool));

        // Code generation, validation, and pattern mining.
        registry.register(Box::new(tools::CodeGenTool));
        registry.register(Box::new(tools::CompileFeedbackTool));
        registry.register(Box::new(tools::PatternMineTool::sandboxed(fs.clone())));

        // Agent management (multi-agent orchestration).
        registry.register(Box::new(tools::AgentListTool));
        registry.register(Box::new(tools::AgentSpawnTool));
        registry.register(Box::new(tools::AgentMessageTool));
        registry.register(Box::new(tools::AgentRetireTool));

        // Trigger management.
        registry.register(Box::new(tools::TriggerManageTool));

        // Audit log.
        registry.register(Box::new(tools::AuditLogTool));

        // Tool-graph query (peer/federation discovery).
        registry.register(Box::new(tools::ToolQueryTool));

        // Delegation tool (Phase 36) — wired to pending task queue.
        #[cfg(feature = "delegation")]
        registry.register(Box::new(tools::DelegateTool::new(delegation_pending)));
        #[cfg(not(feature = "delegation"))]
        registry.register(Box::new(tools::DelegateTool::new_noop()));
    }

    /// Attach the audit ledger to the tool registry if persistence is configured.
    fn wire_audit_ledger(registry: &mut ToolRegistry, engine: &Engine) {
        if let Some(ledger) = engine.audit_ledger() {
            registry.set_audit_ledger(Arc::clone(ledger));
        }
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

    /// Wire procedural knowledge from all active (Hot) skills into the agent's
    /// reasoning subsystems — Phase 30b.
    ///
    /// For each Hot skill:
    /// - Converts `SkillAction` schemas → `ActionSchema` and registers with `causal_manager`.
    /// - Converts `SkillPlan` templates → `DecompositionMethod` and registers with
    ///   `method_registry_htn`.
    ///
    /// Named schemas/methods use the convention `skill/{skill_id}/{name}` so they can
    /// be cleanly removed via `remove_schemas_by_skill` / `remove_methods_by_skill`
    /// when the skill is deactivated.
    pub(crate) fn wire_skill_procedures(&mut self) {
        use super::causal::{ActionSchema, CausalEffect, CausalPattern, EffectKind, PatternElement};
        use super::decomposition::{DecompositionMethod, DecompositionStrategy, SubtaskTemplate};
        use crate::skills::{PlanMethod, SkillState};

        let hot_skills: Vec<String> = self
            .engine
            .list_skills()
            .into_iter()
            .filter(|info| info.state == SkillState::Hot)
            .map(|info| info.id)
            .collect();

        for skill_id in &hot_skills {
            // ── Actions → CausalManager ─────────────────────────────────────
            let actions = self.engine.skill_actions(skill_id);
            if !actions.is_empty() {
                // Resolve shared predicate + context entity once per skill.
                let Ok(has_state_pred) =
                    self.engine.resolve_or_create_relation("skill:has-state")
                else {
                    tracing::warn!(skill = %skill_id, "wire_skill_procedures: resolve skill:has-state failed");
                    continue;
                };
                let Ok(ctx_sym) = self.engine.resolve_or_create_entity("agent:context") else {
                    tracing::warn!(skill = %skill_id, "wire_skill_procedures: resolve agent:context failed");
                    continue;
                };

                for action in &actions {
                    let schema_name = format!("skill/{skill_id}/{}", action.action);
                    let Ok(action_sym) = self.engine.resolve_or_create_entity(&schema_name) else {
                        tracing::warn!(skill = %skill_id, action = %action.action,
                            "wire_skill_procedures: failed to resolve action symbol");
                        continue;
                    };

                    // Preconditions: each fluent → check `(?, skill:has-state, fluent_sym)`.
                    let mut preconditions = Vec::new();
                    for fluent in &action.preconditions {
                        if let Ok(fluent_sym) = self.engine.resolve_or_create_entity(fluent) {
                            preconditions.push(CausalPattern {
                                subject: PatternElement::Wildcard,
                                predicate: PatternElement::Concrete(has_state_pred),
                                object: PatternElement::Concrete(fluent_sym),
                                negated: false,
                            });
                        }
                    }

                    // Effects: initiates → Assert, terminates → Retract.
                    let mut effects = Vec::new();
                    for fluent in &action.effects.initiates {
                        if let Ok(fluent_sym) = self.engine.resolve_or_create_entity(fluent) {
                            effects.push(CausalEffect {
                                kind: EffectKind::Assert,
                                subject: PatternElement::Concrete(ctx_sym),
                                predicate: PatternElement::Concrete(has_state_pred),
                                object: PatternElement::Concrete(fluent_sym),
                                confidence: 0.9,
                            });
                        }
                    }
                    for fluent in &action.effects.terminates {
                        if let Ok(fluent_sym) = self.engine.resolve_or_create_entity(fluent) {
                            effects.push(CausalEffect {
                                kind: EffectKind::Retract,
                                subject: PatternElement::Concrete(ctx_sym),
                                predicate: PatternElement::Concrete(has_state_pred),
                                object: PatternElement::Concrete(fluent_sym),
                                confidence: 0.9,
                            });
                        }
                    }

                    self.causal_manager.register_action_schema(ActionSchema {
                        action_id: action_sym,
                        name: schema_name,
                        description: action.description.clone(),
                        parameters: Vec::new(),
                        preconditions,
                        effects,
                        success_rate: 1.0,
                        execution_count: 0,
                    });
                }

                tracing::debug!(skill = %skill_id, count = actions.len(),
                    "wire_skill_procedures: registered causal action schemas");
            }

            // ── Plans → MethodRegistry ───────────────────────────────────────
            let plans = self.engine.skill_plans(skill_id);
            for plan in &plans {
                let method_name = format!("skill/{skill_id}/{}", plan.goal);

                let strategy = match &plan.method {
                    PlanMethod::Sequential => DecompositionStrategy::Construction,
                    PlanMethod::Parallel => DecompositionStrategy::Research,
                    PlanMethod::Conditional { .. } => DecompositionStrategy::Investigation,
                };

                // Keyword hints from goal name + precondition words.
                let mut keyword_hints: Vec<String> = plan
                    .goal
                    .split('_')
                    .filter(|w| !w.is_empty())
                    .map(|w| w.to_string())
                    .collect();
                for pre in &plan.preconditions {
                    for word in pre.split('_') {
                        if !word.is_empty() && !keyword_hints.contains(&word.to_string()) {
                            keyword_hints.push(word.to_string());
                        }
                    }
                }

                // Each plan step → one SubtaskTemplate.
                let subtask_templates: Vec<SubtaskTemplate> = plan
                    .steps
                    .iter()
                    .map(|step| SubtaskTemplate {
                        description_template: format!("Execute {step} for {{goal}}"),
                        priority_offset: 0,
                        criteria_template: format!("{step} completed for {{goal}}"),
                    })
                    .collect();

                // Sequential ordering constraints (pairs of consecutive indices).
                let ordering: Vec<(usize, usize)> = match &plan.method {
                    PlanMethod::Sequential => (0..plan.steps.len().saturating_sub(1))
                        .map(|i| (i, i + 1))
                        .collect(),
                    _ => Vec::new(),
                };

                self.method_registry_htn.register(DecompositionMethod {
                    name: method_name,
                    strategy,
                    precondition_sparql: String::new(),
                    keyword_hints,
                    subtask_templates,
                    ordering,
                    semantic_vector: None,
                    usage_count: 0,
                    success_rate: 1.0,
                });
            }

            if !plans.is_empty() {
                tracing::debug!(skill = %skill_id, count = plans.len(),
                    "wire_skill_procedures: registered HTN decomposition methods");
            }
        }
    }

    /// Remove procedural knowledge for a deactivated skill — Phase 30b.
    ///
    /// Cleans up `CausalManager` schemas and `MethodRegistry` methods registered
    /// from `skill_id`.  Should be called after `engine.unload_skill(skill_id)`.
    pub fn unwire_skill_procedures(&mut self, skill_id: &str) {
        let removed_schemas = self.causal_manager.remove_schemas_by_skill(skill_id);
        let removed_methods = self.method_registry_htn.remove_methods_by_skill(skill_id);
        if removed_schemas > 0 || removed_methods > 0 {
            tracing::debug!(
                skill = skill_id,
                removed_schemas,
                removed_methods,
                "unwired procedural knowledge from deactivated skill"
            );
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
        let dialogue_predicates = super::dialogue::DialoguePredicates::init(&engine)?;
        let dialogue_manager = super::dialogue::DialogueManager::new(dialogue_predicates, "operator");

        // Create delegation manager early so we can pass the pending queue to DelegateTool.
        #[cfg(feature = "delegation")]
        let delegation_manager =
            super::delegation::DelegationManager::new(super::delegation::DelegationConfig::default());
        #[cfg(feature = "delegation")]
        let delegation_pending = delegation_manager.pending_queue();

        let mut tool_registry = ToolRegistry::new();
        Self::register_builtin_tools(
            &mut tool_registry,
            &predicates,
            &engine,
            #[cfg(feature = "delegation")]
            delegation_pending,
        );
        Self::wire_audit_ledger(&mut tool_registry, &engine);

        let working_memory = WorkingMemory::new(config.working_memory_capacity);

        // Try to restore goals: prefer durable-store blob (survives without
        // sparql-persist), fall back to KG-triple reconstruction.
        let mut goals = goal::restore_goals_from_store(&engine)
            .unwrap_or_else(|| goal::restore_goals(&engine, &predicates).unwrap_or_default());

        // Startup cap: prune excess non-terminal goals carried over from a previous
        // session.  Autonomous investigation goals can accumulate to 72+ in a degraded
        // session; capping here prevents them from blocking the OODA loop on restart.
        // We keep at most max_open_goals * 2 non-terminal goals, preferring goals with
        // higher priority (sorted descending).  Excess goals are marked Failed with an
        // "excess from previous session" reason so they remain auditable.
        {
            let max_restore = config.goal_generation.max_open_goals.saturating_mul(2);
            let non_terminal_ids: Vec<SymbolId> = {
                let mut nt: Vec<&Goal> = goals
                    .iter()
                    .filter(|g| {
                        !matches!(
                            g.status,
                            GoalStatus::Completed
                                | GoalStatus::Failed { .. }
                                | GoalStatus::Reformulated { .. }
                        )
                    })
                    .collect();
                // Prefer higher-priority and user-requested goals; discard the rest.
                nt.sort_by(|a, b| {
                    let ea = a.justification.as_ref().map(|j| j.entrenchment()).unwrap_or(0);
                    let eb = b.justification.as_ref().map(|j| j.entrenchment()).unwrap_or(0);
                    eb.cmp(&ea).then(b.computed_priority().cmp(&a.computed_priority()))
                });
                nt.iter().skip(max_restore).map(|g| g.symbol_id).collect()
            };
            for goal_id in non_terminal_ids {
                if let Some(g) = goals.iter_mut().find(|g| g.symbol_id == goal_id) {
                    tracing::debug!(goal = %goal_id, "startup cap: pruning excess open goal from previous session");
                    let _ = goal::update_goal_status(
                        &engine,
                        g,
                        GoalStatus::Failed {
                            reason: "pruned: excess open goals from previous session".to_string(),
                        },
                        &predicates,
                    );
                }
            }
        }

        // Load psyche from compartment manager if available.
        let psyche = engine.compartments().and_then(|cm| cm.psyche());

        let drives = DriveSystem::with_thresholds(config.goal_generation.drive_thresholds);
        let chunking_config = config.chunking.clone();
        let sleep_cycle = super::sleep::SleepCycle::new(config.sleep.clone());
        let curiosity_config = config.curiosity.clone();
        let hygiene_config = config.hygiene.clone();

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
            psyche_snapshot: Arc::new(std::sync::RwLock::new(psyche.clone())),
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
            interlocutor_registry: InterlocutorRegistry::new(),
            token_registry: super::multi_agent::TokenRegistry::new(),
            pim_manager: super::pim::PimManager::default(),
            calendar_manager: super::calendar::CalendarManager::default(),
            preference_manager: super::preference::PreferenceManager::default(),
            causal_manager: super::causal::CausalManager::default(),
            ec_engine: super::event_calculus::EventCalculusEngine::default(),
            prediction_tracker: super::counterfactual::PredictionTracker::default(),
            value_function: super::state_value::ValueFunction::default(),
            evidence_manager: super::evidence::EvidenceManager::new(),
            reliability_manager: super::source_reliability::SourceReliabilityManager::new(),
            epistemic_manager: super::epistemic::EpistemicStateManager::new(),
            active_inference: super::active_inference::ActiveInferenceEngine::default(),
            temporal_experience: super::temporal_experience::TemporalExperience::default(),
            affective_system: super::affect::AffectiveSystem::new(),
            contact_manager: super::contact::ContactManager::default(),
            relationship_graph: super::contact_rel::RelationshipGraph::default(),
            person_memory: super::contact_memory::PersonMemoryIndex::new(100),
            style_manager: super::contact_style::StyleManager::default(),
            dialogue_manager,
            sleep_cycle,
            curiosity_config,
            hygiene_config,
            #[cfg(feature = "wasm-tools")]
            wasm_runtime: super::wasm_runtime::WasmToolRuntime::new().ok(),
            model_manager: None,
            chat_backend: None,
            context_manager: None,
            #[cfg(feature = "delegation")]
            delegation_manager,
        };

        // Initialize interlocutor predicates.
        if let Err(e) = agent.interlocutor_registry.init_predicates(&agent.engine) {
            tracing::warn!("failed to init interlocutor predicates: {e}");
        }

        // Initialize PIM predicates.
        if let Err(e) = agent.pim_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init PIM predicates: {e}");
        }

        // Initialize calendar predicates (Phase 13f).
        if let Err(e) = agent.calendar_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init calendar predicates: {e}");
        }

        // Initialize preference predicates (Phase 13g).
        if let Err(e) = agent.preference_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init preference predicates: {e}");
        }

        // Initialize contact manager (Phase 25a).
        if let Err(e) = agent.contact_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init contact predicates: {e}");
        }

        // Initialize relationship graph (Phase 25b).
        if let Err(e) = agent.relationship_graph.ensure_init(&agent.engine) {
            tracing::warn!("failed to init relationship graph predicates: {e}");
        }

        // Initialize style manager (Phase 25d).
        if let Err(e) = agent.style_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init style manager predicates: {e}");
        }

        // Register multi-valued agent predicates so temporal conflict detection
        // doesn't flag natural one-to-many relations (e.g. episode learns many things).
        // Global predicates (child-of, doc:*, ingest:*) are registered in Engine::new()
        // and LibraryPredicates::init() / IngestionPredicates::init() respectively.
        {
            let mut mv = agent.engine.multi_valued_preds_mut();
            mv.declare_multi_valued(agent.predicates.learned);
            mv.declare_multi_valued(agent.predicates.has_tag);
            mv.declare_multi_valued(agent.predicates.child_goal);
            mv.declare_multi_valued(agent.predicates.has_criteria);
            mv.declare_multi_valued(agent.predicates.blocked_by);
            // tool:access is one-to-many: a tool may be granted to multiple agents.
            if let Ok(access_pred) =
                agent.engine.resolve_or_create_relation("tool:access")
            {
                mv.declare_multi_valued(access_pred);
            }
        }

        // Register all built-in tool manifests as KG triples for peer discovery.
        if let Ok(registrar) = super::tool_graph::ToolGraphRegistrar::init(&agent.engine) {
            let count = registrar.register_all(&agent.tool_registry, "akhomed", &["agent:any"]);
            tracing::debug!(count, "tool_graph: registered tool manifests in KG");
        } else {
            tracing::warn!("tool_graph: failed to initialise ToolGraphRegistrar");
        }

        // Initialize causal manager (Phase 15a): try restoring persisted
        // schemas first so a fresh Agent picks up bootstrapped schemas.
        if let Ok(restored) = super::causal::CausalManager::restore(&agent.engine) {
            if !restored.schemas.is_empty() {
                agent.causal_manager = restored;
            }
        }
        if let Err(e) = agent.causal_manager.ensure_init(&agent.engine) {
            tracing::warn!("failed to init causal predicates: {e}");
        }

        // Initialize event calculus engine (Phase 15b): restore or create.
        if let Ok(restored) = super::event_calculus::EventCalculusEngine::restore(&agent.engine) {
            agent.ec_engine = restored;
        } else if let Err(e) = agent.ec_engine.ensure_init(&agent.engine) {
            tracing::warn!("failed to init event calculus predicates: {e}");
        }

        // Restore KG dialogue state into ConversationState.
        if let Some(topic_id) = agent.dialogue_manager.query_active_topic(&agent.engine) {
            agent.conversation_state.active_topic = Some(topic_id);
        }

        // Load tools from active skills.
        agent.load_skill_tools();
        // Wire procedural knowledge (Phase 30b).
        agent.wire_skill_procedures();

        // Bootstrap causal world model from the full tool registry (Phase 15a / WI 019e1ab7).
        //
        // Runs after load_skill_tools() so skill-contributed tools are included.
        // Existing schemas (manually tuned or restored from disk) are preserved.
        {
            let signatures = agent.tool_registry.list();
            if !signatures.is_empty() {
                match agent
                    .causal_manager
                    .bootstrap_from_signatures(&signatures, &agent.engine)
                {
                    Ok(n) if n > 0 => {
                        tracing::info!(
                            new_schemas = n,
                            total = agent.causal_manager.schemas.len(),
                            "causal world model bootstrapped from tool registry",
                        );
                        if let Err(e) = agent.causal_manager.persist(&agent.engine) {
                            tracing::warn!("failed to persist bootstrapped causal schemas: {e}");
                        }
                    }
                    Ok(_) => {} // all schemas already present
                    Err(e) => {
                        tracing::warn!("failed to bootstrap causal schemas from tool registry: {e}");
                    }
                }
            }
        }

        Ok(agent)
    }

    // -----------------------------------------------------------------------
    // Delegation manager accessors (Phase 36)
    // -----------------------------------------------------------------------

    /// Immutable access to the delegation manager.
    #[cfg(feature = "delegation")]
    pub fn delegation_manager(&self) -> &super::delegation::DelegationManager {
        &self.delegation_manager
    }

    /// Mutable access to the delegation manager.
    #[cfg(feature = "delegation")]
    pub fn delegation_manager_mut(&mut self) -> &mut super::delegation::DelegationManager {
        &mut self.delegation_manager
    }

    /// Replace the delegation manager (e.g. after loading config from akh.toml).
    #[cfg(feature = "delegation")]
    pub fn set_delegation_manager(&mut self, manager: super::delegation::DelegationManager) {
        // Migrate the pending queue so any tasks queued before the swap are preserved.
        let pending = self.delegation_manager.drain_pending();
        self.delegation_manager = manager;
        for task in pending {
            let _ = self.delegation_manager.enqueue(task);
        }
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

    /// Run a single OODA cycle using the default (LLM-driven) decide mode.
    ///
    /// Also triggers plan-aware execution (advances plan steps), periodic
    /// reflection, auto-decomposition of stalled goals, and auto-consolidation.
    pub fn run_cycle(&mut self) -> AgentResult<OodaCycleResult> {
        self.run_cycle_with_mode(DecideMode::default())
    }

    /// Run a single OODA cycle with the specified [`DecideMode`].
    ///
    /// Use [`DecideMode::VsaNative`] for the fast VSA-native path (100-1000×
    /// faster than the LLM-driven path), or [`DecideMode::LlmDriven`] for the
    /// heuristic-enhanced legacy path.
    pub fn run_cycle_with_mode(&mut self, mode: DecideMode) -> AgentResult<OodaCycleResult> {
        if goal::active_goals(&self.goals).is_empty() {
            return Err(AgentError::NoGoals);
        }

        // Phase 11i: Check if the agent should enter a sleep cycle before OODA.
        let wm_pressure = self.working_memory.len() as f32
            / self.config.working_memory_capacity.max(1) as f32;
        if super::sleep::should_sleep(&self.sleep_cycle, self.cycle_count, wm_pressure)
            && let Err(e) = super::sleep::run_sleep_cycle(
                &self.engine,
                &self.working_memory,
                &mut self.sleep_cycle,
                self.cycle_count,
            )
        {
            tracing::debug!("sleep cycle failed: {e}");
        }

        // Before the OODA cycle, ensure the top-priority goal has a plan.
        let active = goal::active_goals(&self.goals);
        if let Some(top_goal) = active.first() {
            let gid = top_goal.symbol_id;
            if let Err(e) = self.plan_goal(gid) {
                tracing::debug!("failed to plan goal: {e}");
            }
        }

        let result = ooda::run_ooda_cycle_with_mode(self, mode)?;

        // Index the cycle immediately as a retrievable episode so agent_recall can
        // surface it without waiting for the WM auto-consolidation threshold.
        {
            let goal_id = result.decision.goal_id;
            let goal_label = self
                .goals
                .iter()
                .find(|g| g.symbol_id == goal_id)
                .map(|g| g.description.as_str())
                .unwrap_or("unknown goal");
            let progress = match &result.action_result.goal_progress {
                ooda::GoalProgress::Completed => "completed",
                ooda::GoalProgress::Advanced { .. } => "advanced",
                ooda::GoalProgress::Failed { .. } => "failed",
                ooda::GoalProgress::NoChange => "no_change",
            };
            if let Err(e) = index_cycle_episode(
                &self.engine,
                &self.predicates,
                self.cycle_count,
                goal_label,
                Some(goal_id),
                &result.decision.chosen_tool,
                progress,
            ) {
                tracing::debug!(cycle = self.cycle_count, error = %e, "failed to index cycle episode");
            }
        }

        // Advance plan tracking based on the cycle result.
        let goal_id = result.decision.goal_id;
        let key = goal_id.get();
        let mut should_backtrack = false;
        if let Some(plan) = self.plans.get_mut(&key)
            && let Some(idx) = plan.next_step_index()
        {
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
        if should_backtrack
            && let Err(e) = self.backtrack_goal(goal_id)
        {
            tracing::warn!("failed to backtrack goal: {e}");
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
                if goal_completed
                    && let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id)
                {
                    self.failure_index
                        .mark_resolved(&g.description, tool_name);
                }
                // Record effort case for CBR (Phase 11g).
                if let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id) {
                    let case = resource::create_effort_case(g, &self.working_memory, &self.engine);
                    self.effort_index.insert(case);
                }

                // Procedural learning: compile successful traces into methods (Phase 11h).
                if goal_completed
                    && let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id)
                {
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
                        if let Err(e) = self.engine.store_provenance(&mut prov) {
                            tracing::debug!("failed to store procedural learning provenance: {e}");
                        }

                        // Add to method index.
                        self.method_index.insert(learned);
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
            if !tool_success
                && let Some(g) = self.goals.iter().find(|g| g.symbol_id == goal_id)
            {
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

        // Periodic reflection.
        let reflect_interval = self.config.reflection.reflect_every_n_cycles;
        if reflect_interval > 0 && self.cycle_count.is_multiple_of(reflect_interval) {
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
                if let Err(e) = self.apply_adjustments(&structural_adjustments) {
                    tracing::warn!("failed to apply structural adjustments: {e}");
                }
            }
            // Psyche may have evolved during reflect() — keep snapshot in sync.
            self.sync_psyche_snapshot();

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
                    if let Err(e) = self.engine.store_provenance(&mut prov) {
                        tracing::debug!("failed to store priority argumentation provenance: {e}");
                    }
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
            if !meta_adjustments.is_empty()
                && let Err(e) = self.apply_adjustments(&meta_adjustments)
            {
                tracing::warn!("failed to apply metacognitive adjustments: {e}");
            }
        }

        // Periodic library learning (4x less frequent than reflection).
        let library_learn_interval = reflect_interval.saturating_mul(4).max(1);
        if library_learn_interval > 0
            && self.cycle_count.is_multiple_of(library_learn_interval)
            && let Err(e) = self.run_library_learning()
        {
            tracing::debug!("library learning cycle failed: {e}");
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
                if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
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
                }) {
                    tracing::debug!("failed to push watch firing to WM: {e}");
                }

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
                if let Err(e) = self.engine.store_provenance(&mut prov) {
                    tracing::debug!("failed to store watch firing provenance: {e}");
                }
            }

            self.last_watch_firings = firings;
        }

        // Periodic goal queue hygiene (dedup, staleness, depth cap).
        let hyg_interval = self.hygiene_config.run_every_n_cycles;
        if hyg_interval > 0 && self.cycle_count.is_multiple_of(hyg_interval) {
            let hr = super::hygiene::run_hygiene_pass(
                &mut self.goals,
                &self.engine,
                &self.predicates,
                self.cycle_count,
                self.session_start_cycle,
                &self.hygiene_config,
            );
            if hr.cancelled_stale + hr.cancelled_deep + hr.deduplicated > 0 {
                tracing::info!(
                    stale = hr.cancelled_stale,
                    deep = hr.cancelled_deep,
                    dedup = hr.deduplicated,
                    before = hr.total_before,
                    after = hr.total_after,
                    "goal hygiene pass complete"
                );
            }
        }

        // Periodic autonomous goal generation.
        let gen_interval = self.config.goal_generation.generate_every_n_cycles;
        if gen_interval > 0
            && self.cycle_count.is_multiple_of(gen_interval)
            && let Err(e) = self.generate_goals()
        {
            tracing::debug!("autonomous goal generation failed: {e}");
        }

        // Prune autonomous goals that have been stuck for too long.
        self.prune_abandoned_goals();

        // Auto-decompose stalled goals.
        self.decompose_stalled_goals();

        // Consume a cycle from the active project's budget (Phase 11g).
        if let Some(active_pid) = self.agenda.active_project
            && let Some(proj) = self.projects.iter_mut().find(|p| p.id == active_pid)
        {
            proj.consume_cycle();
        }

        // Auto-project-completion: check if active project's goals are all done.
        if let Some(active_pid) = self.agenda.active_project
            && let Some(proj) = self.projects.iter().find(|p| p.id == active_pid)
            && !proj.goals.is_empty()
            && project_progress(proj, &self.goals) >= 1.0
        {
            // Mark project completed and switch to the next one.
            if let Some(proj_mut) = self.projects.iter_mut().find(|p| p.id == active_pid)
                && let Err(e) = update_project_status(
                    &self.engine,
                    proj_mut,
                    ProjectStatus::Completed,
                    &self.project_predicates,
                )
            {
                tracing::warn!("failed to update project status to completed: {e}");
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
            if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
                id: 0,
                content: switch_msg,
                symbols: vec![],
                kind: WorkingMemoryKind::GoalUpdate,
                timestamp: 0,
                relevance: 0.9,
                source_cycle: self.cycle_count,
                reference_count: 0,
                access_timestamps: Vec::new(),
            }) {
                tracing::debug!("failed to push project switch to WM: {e}");
            }
        }

        // Auto-consolidate if enabled and WM pressure is high.
        if self.config.auto_consolidate
            && self.working_memory.len() >= self.config.consolidation.auto_consolidate_at
            && let Err(e) = self.consolidate()
        {
            tracing::warn!("auto-consolidation failed: {e}");
        }

        Ok(result)
    }

    /// Override the maximum number of OODA cycles for `run_until_complete`.
    pub fn set_max_cycles(&mut self, max: usize) {
        self.config.max_cycles = max;
    }

    /// Run cycles until all goals are complete or max_cycles is reached.
    ///
    /// Early-exit: if `config.max_empty_cycles > 0` and that many consecutive
    /// cycles all return `NoChange`, the loop stops without error so the caller
    /// can synthesize the best available findings rather than spinning against
    /// a sparse graph.
    pub fn run_until_complete(&mut self) -> AgentResult<Vec<OodaCycleResult>> {
        let mut results = Vec::new();
        let max = self.config.max_cycles;
        let max_empty = self.config.max_empty_cycles;
        let mut current_cycles: usize = 0;
        let mut consecutive_empty: usize = 0;

        loop {
            // Enforce the cycle cap before doing any work.
            if current_cycles >= max {
                tracing::warn!(
                    max_cycles = max,
                    "run_until_complete: cycle limit reached with active goals remaining"
                );
                break;
            }

            let active = goal::active_goals(&self.goals);
            if active.is_empty() {
                break;
            }

            // Use run_cycle (not ooda::run_ooda_cycle directly) so that episode
            // indexing, plan tracking, metacognition, and other per-cycle housekeeping
            // happen on every iteration — not just when the caller uses agent_run_cycle.
            let result = self.run_cycle()?;

            // Track consecutive no-progress cycles.
            match result.action_result.goal_progress {
                ooda::GoalProgress::NoChange => {
                    consecutive_empty += 1;
                }
                _ => {
                    consecutive_empty = 0;
                }
            }

            results.push(result);
            current_cycles += 1;

            // Early exit when the graph is yielding nothing new.
            if max_empty > 0 && consecutive_empty >= max_empty {
                tracing::debug!(
                    consecutive_empty,
                    "OODA early exit: {} consecutive no-progress cycles",
                    consecutive_empty
                );
                break;
            }

            // Auto-decompose stalled goals.
            self.decompose_stalled_goals();

            // Auto-consolidate.
            if self.config.auto_consolidate
                && self.working_memory.len() >= self.config.consolidation.auto_consolidate_at
                && let Err(e) = self.consolidate()
            {
                tracing::warn!("auto-consolidation failed: {e}");
            }
        }

        // Return a typed error so callers can detect and surface the cap.
        if !goal::active_goals(&self.goals).is_empty() && current_cycles >= max {
            return Err(AgentError::MaxCyclesReached { max_cycles: max });
        }

        Ok(results)
    }

    /// Trigger memory consolidation.
    ///
    /// After persisting episodic memories, runs VSA competence gap detection
    /// and auto-generates up to 10 `Proposed` investigation goals at priority 64
    /// for knowledge areas with coverage score < 0.70.  Duplicate goals (by VSA
    /// similarity) are silently skipped.  Human operators can promote, demote, or
    /// cancel auto-generated goals via normal goal management.
    pub fn consolidate(&mut self) -> AgentResult<ConsolidationResult> {
        let current_affect = *self.affective_system.current_affect();
        let mut result = consolidate(
            &mut self.working_memory,
            &self.engine,
            &self.goals,
            &self.config.consolidation,
            &self.predicates,
            self.cycle_count,
            Some(&current_affect),
        )?;

        // Post-consolidation: VSA competence gap → investigation goals.
        let gap_proposals = goal_generation::generate_competence_gap_goals(
            &self.engine,
            &self.goals,
            &self.config.goal_generation,
        );

        let mut vsa_goals_created = 0usize;
        for proposal in &gap_proposals {
            if vsa_goals_created >= goal_generation::MAX_COMPETENCE_GAP_GOALS {
                break;
            }
            match goal::create_goal(
                &self.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &self.predicates,
            ) {
                Ok(mut g) => {
                    g.source = Some(proposal.source.clone());
                    g.justification = Some(goal::GoalJustification::DefaultAssumption {
                        rationale: proposal.rationale.clone(),
                    });
                    // Auto-generated goals start as Proposed so humans can review.
                    if let Err(e) = goal::update_goal_status(
                        &self.engine,
                        &mut g,
                        GoalStatus::Proposed,
                        &self.predicates,
                    ) {
                        tracing::debug!("failed to set Proposed status for gap goal: {e}");
                    }
                    self.goals.push(g);
                    vsa_goals_created += 1;
                }
                Err(e) => {
                    tracing::debug!("failed to create competence gap goal: {e}");
                }
            }
        }

        result.vsa_gap_goals_created = vsa_goals_created;

        if vsa_goals_created > 0 {
            tracing::debug!(
                count = vsa_goals_created,
                "consolidation: auto-generated VSA competence gap goals"
            );
        }

        Ok(result)
    }

    /// Recall episodic memories by query symbols.
    pub fn recall(&self, query: &[SymbolId], top_k: usize) -> AgentResult<Vec<EpisodicEntry>> {
        let current_affect = *self.affective_system.current_affect();
        recall_episodes(&self.engine, query, &self.predicates, top_k, Some(&current_affect))
    }

    /// Recall the most recent episodic memories without requiring resolved symbols.
    ///
    /// Scans all episodes indexed under `episodic_memory` and returns the `top_k`
    /// most recent. Use this when query terms don't map to known symbols.
    pub fn recall_recent(&self, top_k: usize) -> AgentResult<Vec<EpisodicEntry>> {
        recall_recent_episodes(&self.engine, &self.predicates, top_k)
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

    /// Set the permission mode gating this agent's tool execution.
    pub fn set_permission_mode(&mut self, mode: crate::agent::permission::PermissionMode) {
        self.tool_registry.set_permission_mode(mode);
    }

    /// The active tool-permission mode.
    pub fn permission_mode(&self) -> crate::agent::permission::PermissionMode {
        self.tool_registry.permission_mode()
    }

    /// Attach an approver that resolves `Ask` permission decisions (e.g. an ACP
    /// `session/request_permission` bridge) for the duration of a turn.
    pub fn set_permission_asker(
        &mut self,
        asker: std::sync::Arc<dyn crate::agent::permission::PermissionAsker>,
    ) {
        self.tool_registry.set_permission_asker(asker);
    }

    /// Remove the permission approver (`Ask` then fails closed).
    pub fn clear_permission_asker(&mut self) {
        self.tool_registry.clear_permission_asker();
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

    /// Recall up to `top_k` past-episode summaries relevant to `prompt`, for the
    /// Phase 44g context section.
    ///
    /// Resolves the prompt's content words to seed symbols, **VSA-expands** them
    /// with their nearest neighbours (so episodes about *related* concepts surface
    /// too), recalls a wider candidate pool, drops any episode referencing a
    /// **TMS-retracted** (superseded) fact, then re-ranks by a blend of VSA
    /// relevance ⊕ recency ⊕ confidence. Empty when nothing resolves or matches.
    pub(crate) fn recall_for_prompt(&self, engine: &Engine, prompt: &str, top_k: usize) -> Vec<String> {
        use std::collections::HashMap;

        // 1. Resolve the prompt's content words to seed symbols.
        let seeds: Vec<crate::symbol::SymbolId> = prompt
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 2)
            .filter_map(|w| engine.resolve_symbol(w.as_str()).ok())
            .collect();
        if seeds.is_empty() {
            return Vec::new();
        }

        // 2. VSA expansion: each seed's nearest neighbours, keyed by similarity.
        //    Binary-VSA analogy is degenerate, so trust only strong matches.
        const SIM_FLOOR: f32 = 0.1;
        let mut query = seeds.clone();
        let mut sim_of: HashMap<crate::symbol::SymbolId, f32> =
            seeds.iter().map(|s| (*s, 1.0_f32)).collect();
        for s in &seeds {
            if let Ok(neighbours) = engine.search_similar_to(*s, 5) {
                for n in neighbours {
                    if n.similarity < SIM_FLOOR {
                        continue;
                    }
                    sim_of
                        .entry(n.symbol_id)
                        .and_modify(|v| *v = v.max(n.similarity))
                        .or_insert(n.similarity);
                    if !query.contains(&n.symbol_id) {
                        query.push(n.symbol_id);
                    }
                }
            }
        }

        // 3. Recall a wider candidate pool over the expanded query.
        let candidates = match super::memory::recall_episodes(
            engine,
            &query,
            self.predicates(),
            top_k.saturating_mul(4).max(top_k),
            None,
        ) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        // 4. Drop superseded episodes (any learning the TMS retracted), re-rank.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let tms = engine.tms();
        let mut scored: Vec<(f32, super::memory::EpisodicEntry)> = candidates
            .into_iter()
            .filter(|ep| !ep.learnings.iter().any(|l| tms.was_retracted(*l)))
            .map(|ep| {
                let vsa = ep
                    .learnings
                    .iter()
                    .filter_map(|l| sim_of.get(l).copied())
                    .fold(0.0_f32, f32::max);
                let age_days = now.saturating_sub(ep.timestamp) as f32 / 86_400.0;
                let recency = (-age_days / 30.0).exp();
                let conf = tms.effective_confidence(ep.symbol_id) as f32;
                let score = 0.5 * vsa + 0.3 * recency + 0.2 * ep.relevance_score.max(conf);
                (score, ep)
            })
            .collect();
        drop(tms);
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(_, e)| e.summary)
            .filter(|s| !s.trim().is_empty())
            .take(top_k)
            .collect()
    }

    /// A compact continuity seed for the working context (Phase 44e).
    ///
    /// Returns the opening operator ask of the current conversation so an akh
    /// keeps the thread even when middle turns are later trimmed under budget.
    /// `None` for a fresh (≤1-turn) conversation — there the first turn *is* the
    /// current prompt, so a seed would just duplicate it.
    pub(crate) fn session_seed(&self) -> Option<String> {
        let conv = self.conversation_state();
        if conv.turns.len() <= 1 {
            return None;
        }
        let opening = conv
            .turns
            .iter()
            .find(|t| matches!(t.speaker, crate::agent::conversation::Speaker::Operator))?;
        let ask = opening.text.trim();
        if ask.is_empty() {
            return None;
        }
        let ask: String = ask.chars().take(200).collect();
        Some(format!("This conversation began with: \"{ask}\""))
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

    /// Return a cloned handle to the non-blocking psyche snapshot.
    ///
    /// Callers can read the snapshot without holding the agent mutex,
    /// enabling concurrent monitoring during long OODA cycles.
    pub fn psyche_snapshot(
        &self,
    ) -> Arc<std::sync::RwLock<Option<crate::compartment::psyche::Psyche>>> {
        Arc::clone(&self.psyche_snapshot)
    }

    /// Sync `psyche_snapshot` to match the current `self.psyche`.
    ///
    /// Called after every mutation of `self.psyche` so monitoring reads
    /// always see a fresh copy without acquiring the agent mutex.
    /// Uses `try_write` so the OODA loop is never stalled by active readers.
    /// If the write lock is unavailable, the snapshot stays at the previous
    /// state until the next mutation (readers see data that is at most one
    /// cycle stale).
    pub(crate) fn sync_psyche_snapshot(&self) {
        if let Ok(mut g) = self.psyche_snapshot.try_write() {
            *g = self.psyche.clone();
        }
    }

    /// Set or replace the agent's psyche.
    ///
    /// Returns `PsycheImmutable` if the existing psyche is awakened.
    /// Use [`force_set_psyche`] for admin override.
    pub fn set_psyche(
        &mut self,
        psyche: crate::compartment::psyche::Psyche,
    ) -> super::error::AgentResult<()> {
        if let Some(ref existing) = self.psyche {
            if existing.is_awakened() {
                return Err(super::error::AgentError::PsycheImmutable);
            }
        }
        self.psyche = Some(psyche);
        self.sync_psyche_snapshot();
        Ok(())
    }

    /// Unconditionally replace the agent's psyche (admin override).
    pub fn force_set_psyche(&mut self, psyche: crate::compartment::psyche::Psyche) {
        self.psyche = Some(psyche);
        self.sync_psyche_snapshot();
    }

    // ── Model manager (per-agent Gemma via Ollama) ─────────────────────────

    /// Attach a model manager so this agent can run local Gemma inference.
    pub fn with_model_manager(
        mut self,
        manager: std::sync::Arc<super::model_manager::AgentModelManager>,
    ) -> Self {
        self.model_manager = Some(manager);
        self
    }

    /// Set or replace the model manager.
    pub fn set_model_manager(
        &mut self,
        manager: std::sync::Arc<super::model_manager::AgentModelManager>,
    ) {
        self.model_manager = Some(manager);
    }

    /// Get a reference to the model manager, if one is attached.
    pub fn model_manager(
        &self,
    ) -> Option<&super::model_manager::AgentModelManager> {
        self.model_manager.as_deref()
    }

    /// Set the akh's chat/agentic backing (ADR 070). When set, `run_agentic_turn`
    /// drives this backend instead of the default Ollama path.
    pub fn set_chat_backend(
        &mut self,
        backend: std::sync::Arc<dyn super::chat_backend::ChatBackend>,
    ) {
        self.chat_backend = Some(backend);
    }

    /// Attach a working-context assembly strategy (Phase 44a, ADR 065). `None`
    /// (the default) uses the byte-identical pass-through manager, so an akh with
    /// no manager set behaves exactly as before. Shipped managers should be
    /// wrapped in [`SafetyEnforced`](super::context::SafetyEnforced) by the caller.
    pub fn set_context_manager(&mut self, manager: Box<dyn super::context::ContextManager>) {
        self.context_manager = Some(manager);
    }

    /// Register this agent with its attached model manager using the current
    /// psyche (if any).
    ///
    /// The agent is identified by its engine workspace name, or by the provided
    /// `agent_id` string.  Idempotent — safe to call on every OODA cycle.
    ///
    /// Returns `true` if a model manager is attached; `false` otherwise.
    pub fn register_with_model_manager(&self, agent_id: &str, agent_name: &str) -> bool {
        let Some(ref mgr) = self.model_manager else {
            return false;
        };
        mgr.load_agent(agent_id, agent_name, self.psyche.as_ref());
        true
    }

    /// Generate text from the attached model manager using this agent's Psyche
    /// encoding.
    ///
    /// Returns `None` when no model manager is attached.
    /// Callers should handle `ModelManagerError` variants as follows:
    /// - `AgentNotLoaded` → call `register_with_model_manager()` first.
    /// - `NoBackendAvailable` → start Ollama or configure `fallback_api`.
    pub fn generate_with_model(
        &self,
        agent_id: &str,
        user_prompt: &str,
    ) -> Option<super::model_manager::ModelManagerResult<String>> {
        let mgr = self.model_manager.as_deref()?;
        Some(mgr.generate(agent_id, user_prompt))
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

    // ── Dialogue manager ─────────────────────────────────────────────

    /// Get a shared reference to the dialogue manager.
    pub fn dialogue_manager(&self) -> &super::dialogue::DialogueManager {
        &self.dialogue_manager
    }

    /// Get a mutable reference to the dialogue manager.
    pub fn dialogue_manager_mut(&mut self) -> &mut super::dialogue::DialogueManager {
        &mut self.dialogue_manager
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

    /// Get a reference to the interlocutor registry.
    pub fn interlocutor_registry(&self) -> &InterlocutorRegistry {
        &self.interlocutor_registry
    }

    /// Get a mutable reference to the interlocutor registry.
    pub fn interlocutor_registry_mut(&mut self) -> &mut InterlocutorRegistry {
        &mut self.interlocutor_registry
    }

    /// Get a reference to the capability token registry.
    pub fn token_registry(&self) -> &super::multi_agent::TokenRegistry {
        &self.token_registry
    }

    /// Get a mutable reference to the capability token registry.
    pub fn token_registry_mut(&mut self) -> &mut super::multi_agent::TokenRegistry {
        &mut self.token_registry
    }

    /// Get a reference to the PIM manager.
    pub fn pim_manager(&self) -> &super::pim::PimManager {
        &self.pim_manager
    }

    /// Get a mutable reference to the PIM manager.
    pub fn pim_manager_mut(&mut self) -> &mut super::pim::PimManager {
        &mut self.pim_manager
    }

    /// Get a reference to the calendar manager.
    pub fn calendar_manager(&self) -> &super::calendar::CalendarManager {
        &self.calendar_manager
    }

    /// Get a mutable reference to the calendar manager.
    pub fn calendar_manager_mut(&mut self) -> &mut super::calendar::CalendarManager {
        &mut self.calendar_manager
    }

    /// Get a reference to the preference manager.
    pub fn preference_manager(&self) -> &super::preference::PreferenceManager {
        &self.preference_manager
    }

    /// Get a mutable reference to the preference manager.
    pub fn preference_manager_mut(&mut self) -> &mut super::preference::PreferenceManager {
        &mut self.preference_manager
    }

    /// Get a reference to the causal manager.
    pub fn causal_manager(&self) -> &super::causal::CausalManager {
        &self.causal_manager
    }

    /// Get a mutable reference to the causal manager.
    pub fn causal_manager_mut(&mut self) -> &mut super::causal::CausalManager {
        &mut self.causal_manager
    }

    /// Get a reference to the event calculus engine.
    pub fn ec_engine(&self) -> &super::event_calculus::EventCalculusEngine {
        &self.ec_engine
    }

    /// Get a mutable reference to the event calculus engine.
    pub fn ec_engine_mut(&mut self) -> &mut super::event_calculus::EventCalculusEngine {
        &mut self.ec_engine
    }

    /// Get a reference to the prediction tracker.
    pub fn prediction_tracker(&self) -> &super::counterfactual::PredictionTracker {
        &self.prediction_tracker
    }

    /// Get a mutable reference to the prediction tracker.
    pub fn prediction_tracker_mut(&mut self) -> &mut super::counterfactual::PredictionTracker {
        &mut self.prediction_tracker
    }

    /// Get a reference to the value function.
    pub fn value_function(&self) -> &super::state_value::ValueFunction {
        &self.value_function
    }

    /// Get a mutable reference to the value function.
    pub fn value_function_mut(&mut self) -> &mut super::state_value::ValueFunction {
        &mut self.value_function
    }

    /// Get a reference to the evidence manager.
    pub fn evidence_manager(&self) -> &super::evidence::EvidenceManager {
        &self.evidence_manager
    }

    /// Get a mutable reference to the evidence manager.
    pub fn evidence_manager_mut(&mut self) -> &mut super::evidence::EvidenceManager {
        &mut self.evidence_manager
    }

    /// Get a reference to the source reliability manager.
    pub fn reliability_manager(&self) -> &super::source_reliability::SourceReliabilityManager {
        &self.reliability_manager
    }

    /// Get a mutable reference to the source reliability manager.
    pub fn reliability_manager_mut(&mut self) -> &mut super::source_reliability::SourceReliabilityManager {
        &mut self.reliability_manager
    }

    /// Ensure an interlocutor is registered, auto-creating on first interaction.
    ///
    /// Returns a reference to their profile. If the interlocutor already exists,
    /// updates their channel list and interaction count.
    pub fn ensure_interlocutor(
        &mut self,
        interlocutor_id: &super::channel_message::InterlocutorId,
        channel_id: &str,
        channel_kind: super::channel::ChannelKind,
    ) -> AgentResult<()> {
        self.interlocutor_registry
            .register(interlocutor_id, channel_id, channel_kind, &self.engine)?;
        Ok(())
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

    /// Resume a previously suspended goal.
    pub fn resume_goal(&mut self, goal_id: SymbolId) -> AgentResult<()> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        goal::update_goal_status(&self.engine, goal, GoalStatus::Active, &self.predicates)
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

    /// Permanently remove a goal by its symbol ID.
    ///
    /// Removes all KG triples for the goal so it is not restored on the next session,
    /// then drops it from the in-memory list.
    pub fn delete_goal(&mut self, goal_id: SymbolId) -> AgentResult<()> {
        let pos = self
            .goals
            .iter()
            .position(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;
        goal::delete_goal_from_kg(&self.engine, goal_id, &self.predicates)?;
        self.goals.remove(pos);
        Ok(())
    }

    /// Update mutable fields of a goal: priority, status, and/or success criteria.
    ///
    /// Only fields with `Some` values are changed. Status is parsed from a
    /// string label using [`GoalStatus::from_label`].
    pub fn update_goal(
        &mut self,
        goal_id: SymbolId,
        priority: Option<u8>,
        status: Option<&str>,
        criteria: Option<&str>,
    ) -> AgentResult<()> {
        // Find index first to avoid borrow issues when calling goal::* helpers.
        let idx = self
            .goals
            .iter()
            .position(|g| g.symbol_id == goal_id)
            .ok_or(AgentError::GoalNotFound {
                goal_id: goal_id.get(),
            })?;

        if let Some(p) = priority {
            goal::update_goal_priority(&self.engine, &mut self.goals[idx], p, &self.predicates)?;
        }
        if let Some(s) = status {
            let new_status = GoalStatus::from_label(s);
            goal::update_goal_status(
                &self.engine,
                &mut self.goals[idx],
                new_status,
                &self.predicates,
            )?;
        }
        if let Some(c) = criteria {
            goal::update_goal_criteria(&self.engine, &mut self.goals[idx], c, &self.predicates)?;
        }
        Ok(())
    }

    /// Mark long-stalled autonomous goals as `Failed` (abandoned).
    ///
    /// Goals whose `justification` entrenchment is < 2 (i.e. not `UserRequested`
    /// or `DecomposedFrom`) are considered autonomous investigations. If such a
    /// goal has been Active for at least `abandon_after_cycles` cycles without
    /// any progress (or has had `cycles_worked > 0` and no recent progress), it
    /// is failed with an "abandoned" reason so the slot can be reused.
    fn prune_abandoned_goals(&mut self) {
        let threshold = self.config.goal_generation.abandon_after_cycles;
        let cycle = self.cycle_count;

        // 1. Active goals with no recent progress.
        let active_to_abandon: Vec<SymbolId> = self
            .goals
            .iter()
            .filter(|g| matches!(g.status, GoalStatus::Active))
            // Exempt user-requested and decomposed-from goals (entrenchment >= 2).
            .filter(|g| {
                g.justification
                    .as_ref()
                    .map(|j| j.entrenchment())
                    .unwrap_or(0)
                    < 2
            })
            // Only abandon after meaningful work has been done (cycles_worked > 0)
            // and no progress for the configured threshold.
            .filter(|g| {
                g.cycles_worked > 0
                    && cycle.saturating_sub(g.last_progress_cycle) >= threshold
            })
            .map(|g| g.symbol_id)
            .collect();

        // 2. Pending/Dormant goals that have never been worked on for too long.
        //    These accumulate from goal generation cycles that produce goals which
        //    never get selected by the OODA loop (e.g., excess goals restored from a
        //    previous session, or low-priority dormant goals).
        //    Prune them after 3× the normal threshold, giving them a generous window.
        let pending_threshold = threshold.saturating_mul(3);
        let pending_to_abandon: Vec<SymbolId> = self
            .goals
            .iter()
            .filter(|g| matches!(g.status, GoalStatus::Pending | GoalStatus::Dormant))
            .filter(|g| {
                g.justification
                    .as_ref()
                    .map(|j| j.entrenchment())
                    .unwrap_or(0)
                    < 2
            })
            // Only prune if the agent has run long enough (cycle >= pending_threshold)
            // AND the goal has never been worked on.
            .filter(|g| g.cycles_worked == 0 && cycle >= pending_threshold)
            .map(|g| g.symbol_id)
            .collect();

        for goal_id in active_to_abandon.into_iter().chain(pending_to_abandon) {
            if let Some(g) = self.goals.iter_mut().find(|g| g.symbol_id == goal_id) {
                let reason = if matches!(g.status, GoalStatus::Active) {
                    format!("abandoned: no progress for {threshold} cycles")
                } else {
                    format!("abandoned: pending/dormant with no work for {pending_threshold} cycles")
                };
                tracing::debug!(
                    goal = %goal_id,
                    status = ?g.status,
                    cycles_worked = g.cycles_worked,
                    "Pruning abandoned investigation goal"
                );
                let _ = goal::update_goal_status(
                    &self.engine,
                    g,
                    GoalStatus::Failed { reason },
                    &self.predicates,
                );
            }
        }
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
            if let Err(e) = self.decompose_stalled_goal(goal_id) {
                tracing::debug!("failed to decompose stalled goal: {e}");
            }
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
                if let Err(e) = self.engine.add_triple(&crate::graph::Triple::new(
                    blocked_id,
                    self.predicates.blocked_by,
                    blocker_id,
                )) {
                    tracing::debug!("failed to add blocked_by triple: {e}");
                }
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
        if let Err(e) = self.engine.store_provenance(&mut prov) {
            tracing::debug!("failed to store HTN decomposition provenance: {e}");
        }

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
        if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
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
        }) {
            tracing::debug!("failed to push plan creation to WM: {e}");
        }

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

        if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
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
        }) {
            tracing::debug!("failed to push backtrack plan to WM: {e}");
        }

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
            Some(&self.pim_manager),
            Some(&self.projects),
            Some((&self.preference_manager, &self.engine)),
        )?;

        // Record reflection in WM.
        if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: result.summary.clone(),
            symbols: vec![],
            kind: WorkingMemoryKind::Inference,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: self.cycle_count,
            reference_count: 0,
            access_timestamps: Vec::new(),
        }) {
            tracing::debug!("failed to push reflection to WM: {e}");
        }

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
                    self.add_goal(description, *priority, description)?;
                    applied += 1;
                }
                Adjustment::SuggestAbandon { goal_id, reason } => {
                    if let Err(e) = self.fail_goal(*goal_id, reason) {
                        tracing::warn!("failed to abandon goal: {e}");
                    }
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
                            if let Err(e) = self.engine.store_provenance(&mut prov) {
                                tracing::debug!("failed to store reformulation provenance: {e}");
                            }

                            self.goals.push(replacement);
                            applied += 1;
                        }
                    }
                }
                Adjustment::SuspendGoal { goal_id, reason: _ } => {
                    if let Some(g) = self.goals.iter_mut().find(|g| g.symbol_id == *goal_id) {
                        let category = metacognition::categorize_goal(g);
                        let competence = self.competence_model.category_competence(&category);
                        if let Err(e) = goal::update_goal_status(
                            &self.engine,
                            g,
                            GoalStatus::Suspended,
                            &self.predicates,
                        ) {
                            tracing::warn!("failed to suspend goal: {e}");
                        }

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
                        if let Err(e) = self.engine.store_provenance(&mut prov) {
                            tracing::debug!("failed to store suspension provenance: {e}");
                        }

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

        // Update drive strengths with directed curiosity (Phase 11j).
        self.drives.update_with_curiosity(
            &self.engine,
            &self.working_memory,
            &goal_symbols,
            self.cycle_count,
            Some(&self.curiosity_config),
        );

        // Phase 11j: Generate curiosity proposals alongside drive signals.
        // Pass goal_symbols so curiosity is scoped to goal-reachable entities.
        let curiosity_proposals = super::curiosity::compute_directed_curiosity(
            &self.engine,
            &self.curiosity_config,
            &goal_symbols,
        )
        .map(|report| report.proposals)
        .unwrap_or_default();

        // Run the three-phase pipeline.
        let gen_result = goal_generation::generate_goals(
            &self.engine,
            &self.goals,
            &self.drives,
            &self.config.goal_generation,
            &self.predicates,
            self.cycle_count,
            self.last_impasse.as_ref(),
            self.last_reflection.as_ref(),
            &self.last_watch_firings,
        )?;

        // Merge curiosity proposals into the activated set (Phase 11j).
        let mut gen_result = gen_result;
        for cp in curiosity_proposals {
            // Don't exceed max activations.
            if gen_result.activated.len() < self.config.goal_generation.max_activations {
                gen_result.activated.push(cp);
            } else if gen_result.dormant.len() < self.config.goal_generation.max_dormant {
                gen_result.dormant.push(cp);
            }
        }

        // Count currently open (non-terminal) goals to enforce the hard cap.
        let max_open = self.config.goal_generation.max_open_goals;

        // Activate generated goals by creating them in the goals list,
        // respecting the max-open-goals cap.
        for proposal in &gen_result.activated {
            let current_open = self
                .goals
                .iter()
                .filter(|g| {
                    matches!(
                        g.status,
                        GoalStatus::Active
                            | GoalStatus::Pending
                            | GoalStatus::Proposed
                            | GoalStatus::Dormant
                            | GoalStatus::Suspended
                    )
                })
                .count();
            if current_open >= max_open {
                tracing::debug!(
                    max_open,
                    "max_open_goals reached — skipping remaining activations"
                );
                break;
            }

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
            if let Err(e) = self.engine.store_provenance(&mut prov) {
                tracing::debug!("failed to store goal generation provenance: {e}");
            }

            self.goals.push(g);
        }

        // Store dormant proposals as goals with Dormant status,
        // also respecting the max-open-goals cap.
        for proposal in &gen_result.dormant {
            let current_open = self
                .goals
                .iter()
                .filter(|g| {
                    matches!(
                        g.status,
                        GoalStatus::Active
                            | GoalStatus::Pending
                            | GoalStatus::Proposed
                            | GoalStatus::Dormant
                            | GoalStatus::Suspended
                    )
                })
                .count();
            if current_open >= max_open {
                break;
            }

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

        // Log generation result to WM (include open-goal count for observability).
        let current_open = self
            .goals
            .iter()
            .filter(|g| {
                matches!(
                    g.status,
                    GoalStatus::Active
                        | GoalStatus::Pending
                        | GoalStatus::Proposed
                        | GoalStatus::Dormant
                        | GoalStatus::Suspended
                )
            })
            .count();
        if let Err(e) = self.working_memory.push(WorkingMemoryEntry {
            id: 0,
            content: format!(
                "Goal generation: {} activated, {} dormant, {} deduplicated, {} infeasible (open={}/{} cap). Drives: curiosity={:.2} coherence={:.2} completeness={:.2} efficiency={:.2}",
                gen_result.activated.len(),
                gen_result.dormant.len(),
                gen_result.deduplicated,
                gen_result.infeasible,
                current_open,
                max_open,
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
        }) {
            tracing::debug!("failed to push goal generation result to WM: {e}");
        }

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
    ///
    /// `allowed_domains` sets the URL ingest allowlist for this project.
    /// Pass `None` to allow all sources (legacy open behaviour).
    pub fn create_project(
        &mut self,
        name: &str,
        description: &str,
        scope_concepts: &[&str],
        priority: u8,
        allowed_domains: Option<Vec<String>>,
    ) -> AgentResult<SymbolId> {
        let proj = project::create_project(
            &self.engine,
            name,
            description,
            scope_concepts,
            &self.project_predicates,
            &self.predicates,
            allowed_domains,
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
        if let Err(e) = self
            .competence_model
            .store_competence_triples(&self.engine, success_rate_pred)
        {
            tracing::warn!("failed to persist competence triples: {e}");
        }

        // Persist watches.
        watch::persist_watches(&self.engine, &self.watches)?;

        // Persist PIM manager (Phase 13e).
        let pim_bytes = bincode::serialize(&self.pim_manager).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize PIM manager: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:pim_manager", &pim_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist PIM manager: {e}"),
            })?;

        // Persist calendar manager (Phase 13f).
        let cal_bytes = bincode::serialize(&self.calendar_manager).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize calendar manager: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:calendar_manager", &cal_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist calendar manager: {e}"),
            })?;

        // Persist preference manager (Phase 13g).
        let pref_bytes = bincode::serialize(&self.preference_manager).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize preference manager: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:preference_manager", &pref_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist preference manager: {e}"),
            })?;

        // Persist causal manager (Phase 15a).
        // Only overwrite if this agent actually has schemas; an empty causal
        // manager should not clobber a bootstrapped one written by another
        // agent instance (e.g., the HTTP handler while the daemon is running).
        if !self.causal_manager.schemas.is_empty() {
            let causal_bytes = bincode::serialize(&self.causal_manager).map_err(|e| {
                AgentError::ConsolidationFailed {
                    message: format!("failed to serialize causal manager: {e}"),
                }
            })?;
            store
                .put_meta(b"agent:causal_manager", &causal_bytes)
                .map_err(|e| AgentError::ConsolidationFailed {
                    message: format!("failed to persist causal manager: {e}"),
                })?;
        }

        // Persist affective system (Phase 23).
        if !self.affective_system.markers.markers.is_empty()
            || !self.affective_system.mood.recent_appraisals.is_empty()
        {
            let _ = self.affective_system.persist(&self.engine);
        }

        // Persist active inference engine (Phase 20).
        if self.active_inference.cycle_count > 0 {
            let _ = self.active_inference.persist(&self.engine);
        }

        // Persist epistemic manager (Phase 19).
        if self.epistemic_manager.agent_count() > 0 {
            let _ = self.epistemic_manager.persist(&self.engine);
        }

        // Persist reliability manager (Phase 18a).
        if !self.reliability_manager.sources.is_empty() {
            let _ = self.reliability_manager.persist(&self.engine);
        }

        // Persist evidence manager (Phase 17).
        if self.evidence_manager.claim_count() > 0 {
            let _ = self.evidence_manager.persist(&self.engine);
        }

        // Persist value function (Phase 16a).
        if self.value_function.update_count > 0 {
            let _ = self.value_function.persist(&self.engine);
        }

        // Persist prediction tracker (Phase 15c).
        if self.prediction_tracker.predictions_made > 0 {
            let pt_bytes = bincode::serialize(&self.prediction_tracker).map_err(|e| {
                AgentError::ConsolidationFailed {
                    message: format!("failed to serialize prediction tracker: {e}"),
                }
            })?;
            store
                .put_meta(b"agent:prediction_tracker", &pt_bytes)
                .map_err(|e| AgentError::ConsolidationFailed {
                    message: format!("failed to persist prediction tracker: {e}"),
                })?;
        }

        // Persist event calculus engine (Phase 15b).
        if !self.ec_engine.events.is_empty() || !self.ec_engine.fluents.is_empty() {
            let ec_bytes = bincode::serialize(&self.ec_engine).map_err(|e| {
                AgentError::ConsolidationFailed {
                    message: format!("failed to serialize ec engine: {e}"),
                }
            })?;
            store
                .put_meta(b"agent:ec_engine", &ec_bytes)
                .map_err(|e| AgentError::ConsolidationFailed {
                    message: format!("failed to persist ec engine: {e}"),
                })?;
        }

        // Persist contact manager (Phase 25a).
        let contact_bytes = bincode::serialize(&self.contact_manager).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize contact manager: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:contact_manager", &contact_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist contact manager: {e}"),
            })?;

        // Persist relationship graph (Phase 25b).
        let rel_bytes = bincode::serialize(&self.relationship_graph).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize relationship graph: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:relationship_graph", &rel_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist relationship graph: {e}"),
            })?;

        // Persist person memory index (Phase 25c).
        let person_mem_bytes = bincode::serialize(&self.person_memory).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize person memory index: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:person_memory", &person_mem_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist person memory index: {e}"),
            })?;

        // Persist style manager (Phase 25d).
        let style_bytes = bincode::serialize(&self.style_manager).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize style manager: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:style_manager", &style_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist style manager: {e}"),
            })?;

        // Persist sleep cycle state (Phase 11i).
        let sleep_bytes =
            bincode::serialize(&self.sleep_cycle).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to serialize sleep cycle: {e}"),
            })?;
        store
            .put_meta(b"agent:sleep_cycle", &sleep_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist sleep cycle: {e}"),
            })?;

        // Persist curiosity config (Phase 11j).
        let curiosity_bytes = bincode::serialize(&self.curiosity_config).map_err(|e| {
            AgentError::ConsolidationFailed {
                message: format!("failed to serialize curiosity config: {e}"),
            }
        })?;
        store
            .put_meta(b"agent:curiosity_config", &curiosity_bytes)
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to persist curiosity config: {e}"),
            })?;

        // Persist goals to durable store (survives restart without sparql-persist).
        // Writes the `agent:goals` TieredStore snapshot so goals survive across
        // sessions even when the `sparql-persist` feature is not enabled.
        goal::persist_goals(&self.goals, &self.engine)?;

        // Flush the engine's durable store.
        self.engine
            .persist()
            .map_err(|e| AgentError::ConsolidationFailed {
                message: format!("engine persist failed: {e}"),
            })?;

        Ok(())
    }

    /// Lightweight persist: saves goals and flushes the engine's durable store.
    ///
    /// Use this instead of calling `engine.persist()` directly when you want
    /// goal state to survive a restart without a full `persist_session()`.
    pub fn persist(&self) -> AgentResult<()> {
        goal::persist_goals(&self.goals, &self.engine)?;
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
        let dialogue_predicates = super::dialogue::DialoguePredicates::init(&engine)?;
        let dialogue_manager = super::dialogue::DialogueManager::new(dialogue_predicates, "operator");

        // Create delegation manager (delegation state is not persisted between sessions
        // — active delegations are transient; the config reloads from defaults).
        #[cfg(feature = "delegation")]
        let delegation_manager =
            super::delegation::DelegationManager::new(super::delegation::DelegationConfig::default());
        #[cfg(feature = "delegation")]
        let delegation_pending = delegation_manager.pending_queue();

        let mut tool_registry = ToolRegistry::new();
        Self::register_builtin_tools(
            &mut tool_registry,
            &predicates,
            &engine,
            #[cfg(feature = "delegation")]
            delegation_pending,
        );
        Self::wire_audit_ledger(&mut tool_registry, &engine);

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

        // Restore goals: prefer the durable-store TieredStore snapshot (always
        // accurate, full struct) over KG-triple reconstruction (only accurate
        // when SPARQL persistence is enabled — without it, all goals default to
        // Pending status because KG triples are in-memory only and lost on restart).
        let goals = goal::restore_goals_from_store(&engine)
            .filter(|g| !g.is_empty())
            .unwrap_or_else(|| goal::restore_goals(&engine, &predicates).unwrap_or_default());

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
        if let Some(stats_bytes) = store.get_meta(b"agent:method_stats").ok().flatten()
            && let Ok(stats) = bincode::deserialize::<Vec<MethodStats>>(&stats_bytes)
        {
            htn_registry.import_stats(&stats);
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
        if let Some(summary_bytes) = store.get_meta(b"agent:session_summary").ok().flatten()
            && let Ok(summary) = bincode::deserialize::<SessionSummary>(&summary_bytes)
            && let Err(e) = restore_session_summary(&mut working_memory, &summary, cycle_count)
        {
            tracing::warn!("failed to restore session summary: {e}");
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

        // Restore PIM manager (Phase 13e).
        let pim_manager = super::pim::PimManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::pim::PimManager::new(&engine).unwrap_or_default()
            });

        // Restore calendar manager (Phase 13f).
        let calendar_manager = super::calendar::CalendarManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::calendar::CalendarManager::new(&engine).unwrap_or_default()
            });

        // Restore preference manager (Phase 13g).
        let preference_manager = super::preference::PreferenceManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::preference::PreferenceManager::new(&engine).unwrap_or_default()
            });

        // Restore causal manager (Phase 15a).
        let causal_manager = super::causal::CausalManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::causal::CausalManager::new(&engine).unwrap_or_default()
            });

        // Restore event calculus engine (Phase 15b).
        let ec_engine = super::event_calculus::EventCalculusEngine::restore(&engine)
            .unwrap_or_else(|_| {
                super::event_calculus::EventCalculusEngine::new(&engine).unwrap_or_default()
            });

        // Restore affective system (Phase 23).
        let affective_system = super::affect::AffectiveSystem::restore(&engine);

        // Restore active inference engine (Phase 20).
        let active_inference = super::active_inference::ActiveInferenceEngine::restore(&engine);

        // Restore epistemic manager (Phase 19).
        let epistemic_manager = super::epistemic::EpistemicStateManager::restore(&engine);

        // Restore reliability manager (Phase 18a).
        let reliability_manager = super::source_reliability::SourceReliabilityManager::restore(&engine);

        // Restore evidence manager (Phase 17).
        let evidence_manager = super::evidence::EvidenceManager::restore(&engine);

        // Restore value function (Phase 16a).
        let value_function = super::state_value::ValueFunction::restore(&engine);

        // Restore prediction tracker (Phase 15c).
        let prediction_tracker = engine
            .store()
            .get_meta(b"agent:prediction_tracker")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default();

        // Restore contact manager (Phase 25a).
        let contact_manager = super::contact::ContactManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::contact::ContactManager::new(&engine).unwrap_or_default()
            });

        // Restore relationship graph (Phase 25b).
        let relationship_graph = super::contact_rel::RelationshipGraph::restore(&engine)
            .unwrap_or_else(|_| {
                super::contact_rel::RelationshipGraph::new(&engine).unwrap_or_default()
            });

        // Restore person memory index (Phase 25c).
        let person_memory = store
            .get_meta(b"agent:person_memory")
            .ok()
            .flatten()
            .and_then(|bytes| {
                bincode::deserialize::<super::contact_memory::PersonMemoryIndex>(&bytes).ok()
            })
            .unwrap_or_else(|| super::contact_memory::PersonMemoryIndex::new(100));

        // Restore style manager (Phase 25d).
        let style_manager = super::contact_style::StyleManager::restore(&engine)
            .unwrap_or_else(|_| {
                super::contact_style::StyleManager::new(&engine).unwrap_or_default()
            });

        // Restore sleep cycle state (Phase 11i).
        let sleep_cycle = store
            .get_meta(b"agent:sleep_cycle")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<super::sleep::SleepCycle>(&bytes).ok())
            .unwrap_or_else(|| super::sleep::SleepCycle::new(config.sleep.clone()));

        // Restore curiosity config (Phase 11j).
        let curiosity_config = store
            .get_meta(b"agent:curiosity_config")
            .ok()
            .flatten()
            .and_then(|bytes| {
                bincode::deserialize::<super::curiosity::DirectedCuriosityConfig>(&bytes).ok()
            })
            .unwrap_or_else(|| config.curiosity.clone());

        let hygiene_config = config.hygiene.clone();
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
            psyche_snapshot: Arc::new(std::sync::RwLock::new(psyche.clone())),
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
            interlocutor_registry: InterlocutorRegistry::new(),
            token_registry: super::multi_agent::TokenRegistry::new(),
            pim_manager,
            calendar_manager,
            preference_manager,
            causal_manager,
            ec_engine,
            prediction_tracker,
            value_function,
            evidence_manager,
            reliability_manager,
            epistemic_manager,
            active_inference,
            temporal_experience: super::temporal_experience::TemporalExperience::default(),
            affective_system,
            contact_manager,
            relationship_graph,
            person_memory,
            style_manager,
            dialogue_manager,
            sleep_cycle,
            curiosity_config,
            hygiene_config,
            #[cfg(feature = "wasm-tools")]
            wasm_runtime: super::wasm_runtime::WasmToolRuntime::new().ok(),
            model_manager: None,
            chat_backend: None,
            context_manager: None,
            #[cfg(feature = "delegation")]
            delegation_manager,
        };

        // Initialize interlocutor predicates.
        if let Err(e) = agent.interlocutor_registry.init_predicates(&agent.engine) {
            tracing::warn!("failed to init interlocutor predicates on resume: {e}");
        }

        // Register multi-valued agent predicates (idempotent — safe even when
        // the set was already restored from the durable store). Global predicates
        // are handled by Engine::new() and subsystem Predicates::init() methods.
        {
            let mut mv = agent.engine.multi_valued_preds_mut();
            mv.declare_multi_valued(agent.predicates.learned);
            mv.declare_multi_valued(agent.predicates.has_tag);
            mv.declare_multi_valued(agent.predicates.child_goal);
            mv.declare_multi_valued(agent.predicates.has_criteria);
            mv.declare_multi_valued(agent.predicates.blocked_by);
        }

        // Restore KG dialogue state into ConversationState.
        if let Some(topic_id) = agent.dialogue_manager.query_active_topic(&agent.engine) {
            agent.conversation_state.active_topic = Some(topic_id);
        }

        // Load tools from active skills.
        agent.load_skill_tools();
        // Wire procedural knowledge (Phase 30b).
        agent.wire_skill_procedures();

        // Bootstrap causal world model from the full tool registry (Phase 15a / WI 019e1ab7).
        {
            let signatures = agent.tool_registry.list();
            if !signatures.is_empty() {
                match agent
                    .causal_manager
                    .bootstrap_from_signatures(&signatures, &agent.engine)
                {
                    Ok(n) if n > 0 => {
                        tracing::info!(
                            new_schemas = n,
                            total = agent.causal_manager.schemas.len(),
                            "causal world model bootstrapped from tool registry (resume)",
                        );
                        if let Err(e) = agent.causal_manager.persist(&agent.engine) {
                            tracing::warn!("failed to persist bootstrapped causal schemas: {e}");
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("failed to bootstrap causal schemas from tool registry: {e}");
                    }
                }
            }
        }

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
            .field("interlocutors", &self.interlocutor_registry.len())
            .finish()
    }
}
