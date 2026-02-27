//! Bootstrap orchestrator: meta-OODA loop over the 8-stage pipeline (Phase 14h).
//!
//! Chains all existing stage implementations (purpose parser → identity resolver →
//! ritual → expansion → prerequisite → resources → ingestion → assessment) with
//! looping (stages 4–7 repeat until target competence), session persistence across
//! restarts, personality-influenced exploration scheduling, and operator
//! interaction checkpoints.
//!
//! Pipeline: `Operator purpose statement → Orchestrator → Bootstrapped akh at target Dreyfus level`

use std::sync::Arc;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bootstrap::competence::{CompetenceAssessor, CompetenceConfig, CompetenceReport};
use crate::bootstrap::expand::{DomainExpander, ExpansionConfig, ExpansionResult};
use crate::bootstrap::identity::{self, CharacterKnowledge};
use crate::bootstrap::ingest::{CurriculumIngestor, IngestionConfig};
use crate::bootstrap::prerequisite::{PrerequisiteAnalyzer, PrerequisiteConfig};
use crate::bootstrap::purpose::{self, BootstrapIntent, DreyfusLevel};
use crate::bootstrap::resources::{
    self, ResourceDiscoverer, ResourceDiscoveryConfig, ResourceDiscoveryResult,
};
use crate::compartment::psyche::{ArchetypeWeights, Psyche};
use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from the bootstrap orchestrator.
#[derive(Debug, Error, Diagnostic)]
pub enum OrchestratorError {
    #[error("empty purpose statement — nothing to bootstrap")]
    #[diagnostic(
        code(akh::bootstrap::orchestrator::empty_purpose),
        help(
            "Provide a non-empty purpose statement, e.g.:\n  \
             akh awaken bootstrap \"You are the Architect based on Ptah, expert in systems\""
        )
    )]
    EmptyPurpose,

    #[error("stage \"{stage}\" failed: {reason}")]
    #[diagnostic(
        code(akh::bootstrap::orchestrator::stage_failed),
        help(
            "A bootstrap pipeline stage encountered an error. Check the inner \
             reason and retry, or use --resume to continue from the last checkpoint."
        )
    )]
    StageFailed { stage: String, reason: String },

    #[error("maximum learning cycles exhausted ({max_cycles}) without reaching target competence")]
    #[diagnostic(
        code(akh::bootstrap::orchestrator::max_cycles),
        help(
            "Increase --max-cycles or lower the target competence level in the \
             purpose statement. Current session can be resumed with --resume."
        )
    )]
    MaxCyclesExhausted { max_cycles: usize },

    #[error("no saved session to resume")]
    #[diagnostic(
        code(akh::bootstrap::orchestrator::no_session),
        help(
            "Start a new bootstrap with a purpose statement:\n  \
             akh awaken bootstrap \"Your purpose statement here\""
        )
    )]
    NoSessionToResume,

    #[error("{0}")]
    #[diagnostic(
        code(akh::bootstrap::orchestrator::engine),
        help("An engine-level error occurred during bootstrap orchestration.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for OrchestratorError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

impl From<crate::bootstrap::purpose::BootstrapError> for OrchestratorError {
    fn from(e: crate::bootstrap::purpose::BootstrapError) -> Self {
        Self::StageFailed {
            stage: "parse_purpose".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::identity::IdentityError> for OrchestratorError {
    fn from(e: crate::bootstrap::identity::IdentityError) -> Self {
        Self::StageFailed {
            stage: "identity".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::expand::DomainExpandError> for OrchestratorError {
    fn from(e: crate::bootstrap::expand::DomainExpandError) -> Self {
        Self::StageFailed {
            stage: "domain_expansion".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::prerequisite::PrerequisiteError> for OrchestratorError {
    fn from(e: crate::bootstrap::prerequisite::PrerequisiteError) -> Self {
        Self::StageFailed {
            stage: "prerequisite_discovery".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::resources::ResourceDiscoveryError> for OrchestratorError {
    fn from(e: crate::bootstrap::resources::ResourceDiscoveryError) -> Self {
        Self::StageFailed {
            stage: "resource_discovery".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::ingest::IngestionError> for OrchestratorError {
    fn from(e: crate::bootstrap::ingest::IngestionError) -> Self {
        Self::StageFailed {
            stage: "curriculum_ingestion".to_string(),
            reason: e.to_string(),
        }
    }
}

impl From<crate::bootstrap::competence::CompetenceError> for OrchestratorError {
    fn from(e: crate::bootstrap::competence::CompetenceError) -> Self {
        Self::StageFailed {
            stage: "competence_assessment".to_string(),
            reason: e.to_string(),
        }
    }
}

/// Convenience alias.
pub type OrchestratorResult<T> = std::result::Result<T, OrchestratorError>;

// ── Configuration ───────────────────────────────────────────────────────

/// Orchestrator-level configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum learning loop iterations (stages 4–7) before giving up.
    pub max_learning_cycles: usize,
    /// If true, only parse and show the plan — do not execute.
    pub plan_only: bool,
    /// Override for expansion config (None = default).
    pub expansion: Option<ExpansionConfig>,
    /// Override for prerequisite config (None = default).
    pub prerequisite: Option<PrerequisiteConfig>,
    /// Override for resource discovery config (None = default).
    pub resource_discovery: Option<ResourceDiscoveryConfig>,
    /// Override for ingestion config (None = default).
    pub ingestion: Option<IngestionConfig>,
    /// Override for competence config (None = default).
    pub competence: Option<CompetenceConfig>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_learning_cycles: 10,
            plan_only: false,
            expansion: None,
            prerequisite: None,
            resource_discovery: None,
            ingestion: None,
            competence: None,
        }
    }
}

// ── Bootstrap Stage ─────────────────────────────────────────────────────

/// Ordered stages in the bootstrap pipeline.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum BootstrapStage {
    ParsePurpose = 0,
    ResolveIdentity = 1,
    RitualOfAwakening = 2,
    DomainExpansion = 3,
    PrerequisiteDiscovery = 4,
    ResourceDiscovery = 5,
    CurriculumIngestion = 6,
    CompetenceAssessment = 7,
    Complete = 8,
}

impl BootstrapStage {
    /// Human-readable label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::ParsePurpose => "parse_purpose",
            Self::ResolveIdentity => "resolve_identity",
            Self::RitualOfAwakening => "ritual_of_awakening",
            Self::DomainExpansion => "domain_expansion",
            Self::PrerequisiteDiscovery => "prerequisite_discovery",
            Self::ResourceDiscovery => "resource_discovery",
            Self::CurriculumIngestion => "curriculum_ingestion",
            Self::CompetenceAssessment => "competence_assessment",
            Self::Complete => "complete",
        }
    }

    /// Parse from a label string.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "parse_purpose" => Some(Self::ParsePurpose),
            "resolve_identity" => Some(Self::ResolveIdentity),
            "ritual_of_awakening" => Some(Self::RitualOfAwakening),
            "domain_expansion" => Some(Self::DomainExpansion),
            "prerequisite_discovery" => Some(Self::PrerequisiteDiscovery),
            "resource_discovery" => Some(Self::ResourceDiscovery),
            "curriculum_ingestion" => Some(Self::CurriculumIngestion),
            "competence_assessment" => Some(Self::CompetenceAssessment),
            "complete" => Some(Self::Complete),
            _ => None,
        }
    }
}

impl std::fmt::Display for BootstrapStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

// ── Session Types ───────────────────────────────────────────────────────

/// Compact assessment snapshot stored in the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAssessment {
    pub overall_score: f32,
    pub overall_dreyfus: DreyfusLevel,
    pub focus_areas: Vec<String>,
    pub recommendation: String,
}

/// Persistent bootstrap session — serialized to durable store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapSession {
    pub current_stage: BootstrapStage,
    pub learning_cycle: usize,
    pub raw_purpose: String,
    pub intent: Option<BootstrapIntent>,
    pub chosen_name: Option<String>,
    pub psyche: Option<Psyche>,
    pub expansion_labels: Vec<String>,
    pub last_assessment: Option<SessionAssessment>,
    pub started_at: u64,
    pub last_updated: u64,
    pub exploration_rate: f32,
}

impl BootstrapSession {
    fn new(raw_purpose: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            current_stage: BootstrapStage::ParsePurpose,
            learning_cycle: 0,
            raw_purpose,
            intent: None,
            chosen_name: None,
            psyche: None,
            expansion_labels: Vec::new(),
            last_assessment: None,
            started_at: now,
            last_updated: now,
            exploration_rate: 0.8,
        }
    }

    fn touch(&mut self) {
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

// ── Checkpoint ──────────────────────────────────────────────────────────

/// Operator-visible checkpoints emitted during orchestration.
#[derive(Debug, Clone)]
pub enum Checkpoint {
    /// Purpose statement parsed into a structured intent.
    PurposeParsed {
        domain: String,
        competence_level: DreyfusLevel,
        seed_count: usize,
        has_identity: bool,
    },
    /// Identity resolved and Ritual of Awakening complete.
    IdentityConstructed {
        chosen_name: String,
    },
    /// Domain expansion + prerequisite analysis done; learning plan ready.
    LearningPlan {
        concept_count: usize,
        relation_count: usize,
    },
    /// Competence assessment complete for one learning cycle.
    AssessmentComplete {
        cycle: usize,
        overall_score: f32,
        overall_dreyfus: DreyfusLevel,
        recommendation: String,
    },
}

// ── Orchestration Result ────────────────────────────────────────────────

/// Final result of a bootstrap orchestration run.
#[derive(Debug)]
pub struct OrchestrationResult {
    pub intent: BootstrapIntent,
    pub chosen_name: Option<String>,
    pub psyche: Option<Psyche>,
    pub final_report: Option<CompetenceReport>,
    pub learning_cycles: usize,
    pub stages_completed: Vec<BootstrapStage>,
    pub target_reached: bool,
    pub provenance_ids: Vec<ProvenanceId>,
}

// ── Session persistence key ─────────────────────────────────────────────

const SESSION_KEY: &[u8] = b"bootstrap:session";

// ── BootstrapOrchestrator ───────────────────────────────────────────────

/// Manages the full 8-stage bootstrap pipeline.
pub struct BootstrapOrchestrator {
    config: OrchestratorConfig,
    session: BootstrapSession,
}

impl BootstrapOrchestrator {
    /// Start a fresh bootstrap from a purpose statement.
    pub fn new(purpose: &str, config: OrchestratorConfig) -> OrchestratorResult<Self> {
        let trimmed = purpose.trim();
        if trimmed.is_empty() {
            return Err(OrchestratorError::EmptyPurpose);
        }
        Ok(Self {
            config,
            session: BootstrapSession::new(trimmed.to_string()),
        })
    }

    /// Resume an interrupted bootstrap from a persisted session.
    pub fn resume(engine: &Engine, config: OrchestratorConfig) -> OrchestratorResult<Self> {
        let session = load_session(engine)?;
        Ok(Self { config, session })
    }

    /// Show current session status without running.
    pub fn status(engine: &Engine) -> OrchestratorResult<BootstrapSession> {
        load_session(engine)
    }

    /// Run the full orchestration pipeline, returning checkpoints along the way.
    pub fn run(
        &mut self,
        engine: &Arc<Engine>,
    ) -> OrchestratorResult<(OrchestrationResult, Vec<Checkpoint>)> {
        let mut checkpoints = Vec::new();
        let mut provenance_ids = Vec::new();
        let mut stages_completed = Vec::new();

        // ── Stage 1: Parse Purpose ──────────────────────────────────
        if self.session.current_stage <= BootstrapStage::ParsePurpose {
            let intent = purpose::parse_purpose(&self.session.raw_purpose)?;
            self.session.intent = Some(intent.clone());
            checkpoints.push(Checkpoint::PurposeParsed {
                domain: intent.purpose.domain.clone(),
                competence_level: intent.purpose.competence_level,
                seed_count: intent.purpose.seed_concepts.len(),
                has_identity: intent.identity.is_some(),
            });
            stages_completed.push(BootstrapStage::ParsePurpose);

            if self.config.plan_only {
                // Return early with just the parsed intent.
                return Ok((
                    OrchestrationResult {
                        intent,
                        chosen_name: None,
                        psyche: None,
                        final_report: None,
                        learning_cycles: 0,
                        stages_completed,
                        target_reached: false,
                        provenance_ids,
                    },
                    checkpoints,
                ));
            }

            self.session.current_stage = BootstrapStage::ResolveIdentity;
            self.session.touch();
            persist_session(&self.session, engine)?;
        }

        let intent = self
            .session
            .intent
            .clone()
            .expect("intent must be set after parse stage");

        // ── Stage 2a: Resolve Identity ──────────────────────────────
        let mut character: Option<CharacterKnowledge> = None;
        if self.session.current_stage <= BootstrapStage::ResolveIdentity {
            if let Some(ref identity_ref) = intent.identity {
                let ck = identity::resolve_identity(identity_ref, engine)?;
                character = Some(ck);
            }
            self.session.current_stage = BootstrapStage::RitualOfAwakening;
            self.session.touch();
            persist_session(&self.session, engine)?;
            stages_completed.push(BootstrapStage::ResolveIdentity);
        }

        // ── Stage 2b: Ritual of Awakening ───────────────────────────
        if self.session.current_stage <= BootstrapStage::RitualOfAwakening {
            if let Some(ref ck) = character {
                let ritual_result =
                    identity::ritual_of_awakening(ck, &intent.purpose, engine)?;
                self.session.chosen_name = Some(ritual_result.chosen_name.clone());
                self.session.psyche = Some(ritual_result.psyche.clone());
                provenance_ids.extend(ritual_result.provenance_ids);
                checkpoints.push(Checkpoint::IdentityConstructed {
                    chosen_name: ritual_result.chosen_name,
                });
            }
            self.session.current_stage = BootstrapStage::DomainExpansion;
            self.session.touch();
            persist_session(&self.session, engine)?;
            stages_completed.push(BootstrapStage::RitualOfAwakening);
        }

        // ── Stage 3: Domain Expansion ───────────────────────────────
        let mut expansion_result: Option<ExpansionResult> = None;
        if self.session.current_stage <= BootstrapStage::DomainExpansion {
            let exp_config = self.config.expansion.clone().unwrap_or_default();
            let mut expander = DomainExpander::new(engine, exp_config)?;
            let result = expander.expand(&intent.purpose, engine)?;

            self.session.expansion_labels = result.accepted_labels.clone();
            provenance_ids.extend(&result.provenance_ids);

            checkpoints.push(Checkpoint::LearningPlan {
                concept_count: result.concept_count,
                relation_count: result.relation_count,
            });

            expansion_result = Some(result);
            self.session.current_stage = BootstrapStage::PrerequisiteDiscovery;
            self.session.touch();
            persist_session(&self.session, engine)?;
            stages_completed.push(BootstrapStage::DomainExpansion);
        }

        // If resuming past stage 3, reconstruct expansion from labels.
        let expansion = expansion_result.unwrap_or_else(|| {
            reconstruct_expansion(&self.session.expansion_labels)
        });

        // ── Learning Loop (stages 4–7) ──────────────────────────────
        let mut final_report: Option<CompetenceReport> = None;
        let mut target_reached = false;

        while self.session.learning_cycle < self.config.max_learning_cycles {
            self.session.learning_cycle += 1;
            let cycle = self.session.learning_cycle;

            // Adjust exploration rate.
            self.session.exploration_rate = compute_exploration_rate(
                self.session.last_assessment.as_ref(),
                self.session.psyche.as_ref(),
            );

            // Apply personality bias to sub-stage configs.
            let (prereq_cfg, resource_cfg, ingest_cfg, competence_cfg) =
                self.personality_biased_configs();

            // ── Stage 4: Prerequisite Discovery ─────────────────────
            let prereq_result = match PrerequisiteAnalyzer::new(engine, prereq_cfg) {
                Ok(analyzer) => match analyzer.analyze(&expansion, engine) {
                    Ok(r) => r,
                    Err(_) => {
                        resources::synthetic_curriculum_from_expansion(&expansion, engine)
                    }
                },
                Err(_) => {
                    resources::synthetic_curriculum_from_expansion(&expansion, engine)
                }
            };
            provenance_ids.extend(&prereq_result.provenance_ids);

            // ── Stage 5: Resource Discovery ─────────────────────────
            let resource_result = match ResourceDiscoverer::new(engine, resource_cfg) {
                Ok(mut discoverer) => {
                    match discoverer.discover(
                        &prereq_result,
                        &expansion,
                        &intent.purpose.seed_concepts,
                        engine,
                    ) {
                        Ok(r) => {
                            provenance_ids.extend(&r.provenance_ids);
                            r
                        }
                        Err(_e) => {
                            // Resource discovery can fail (no proximal concepts etc.)
                            // Continue with empty result.
                            ResourceDiscoveryResult {
                                resources: Vec::new(),
                                api_calls_made: 0,
                                concepts_searched: 0,
                                provenance_ids: Vec::new(),
                            }
                        }
                    }
                }
                Err(_) => ResourceDiscoveryResult {
                    resources: Vec::new(),
                    api_calls_made: 0,
                    concepts_searched: 0,
                    provenance_ids: Vec::new(),
                },
            };

            // ── Stage 6: Curriculum Ingestion ───────────────────────
            match CurriculumIngestor::new(engine, ingest_cfg) {
                Ok(mut ingestor) => {
                    if let Ok(ingest_result) =
                        ingestor.ingest(&prereq_result, &resource_result, engine)
                    {
                        provenance_ids.extend(&ingest_result.provenance_ids);
                    }
                }
                Err(_) => { /* ingestion init failed; assessment will reflect sparse KG */ }
            }

            // ── Stage 7: Competence Assessment ──────────────────────
            let report = CompetenceAssessor::new(engine, competence_cfg)?
                .assess(&prereq_result, &intent.purpose, engine)?;
            provenance_ids.extend(&report.provenance_ids);

            let assessment = SessionAssessment {
                overall_score: report.overall_score,
                overall_dreyfus: report.overall_dreyfus,
                focus_areas: report.remaining_gaps.clone(),
                recommendation: format!("{}", report.recommendation),
            };

            checkpoints.push(Checkpoint::AssessmentComplete {
                cycle,
                overall_score: report.overall_score,
                overall_dreyfus: report.overall_dreyfus,
                recommendation: format!("{}", report.recommendation),
            });

            self.session.last_assessment = Some(assessment);
            self.session.touch();

            match &report.recommendation {
                crate::bootstrap::BootstrapRecommendation::Ready => {
                    target_reached = true;
                    final_report = Some(report);
                    self.session.current_stage = BootstrapStage::Complete;
                    persist_session(&self.session, engine)?;
                    break;
                }
                crate::bootstrap::BootstrapRecommendation::NeedsOperatorInput { .. } => {
                    // Persist and return partial result for operator review.
                    final_report = Some(report);
                    self.session.current_stage = BootstrapStage::PrerequisiteDiscovery;
                    persist_session(&self.session, engine)?;
                    break;
                }
                crate::bootstrap::BootstrapRecommendation::ContinueLearning { .. } => {
                    final_report = Some(report);
                    self.session.current_stage = BootstrapStage::PrerequisiteDiscovery;
                    persist_session(&self.session, engine)?;
                    // Continue loop.
                }
            }
        }

        if !target_reached && self.session.learning_cycle >= self.config.max_learning_cycles {
            // Still persist the session before returning the error.
            persist_session(&self.session, engine)?;

            // Check if the last assessment was NeedsOperatorInput — that's not a cycle failure.
            if let Some(ref report) = final_report {
                if matches!(
                    report.recommendation,
                    crate::bootstrap::BootstrapRecommendation::NeedsOperatorInput { .. }
                ) {
                    // Don't error — return partial result.
                } else {
                    return Err(OrchestratorError::MaxCyclesExhausted {
                        max_cycles: self.config.max_learning_cycles,
                    });
                }
            } else {
                return Err(OrchestratorError::MaxCyclesExhausted {
                    max_cycles: self.config.max_learning_cycles,
                });
            }
        }

        stages_completed.push(BootstrapStage::CompetenceAssessment);

        // ── Store orchestration provenance ───────────────────────────
        let prov_id = store_orchestration_provenance(engine, &self.session, &final_report)?;
        provenance_ids.push(prov_id);

        Ok((
            OrchestrationResult {
                intent,
                chosen_name: self.session.chosen_name.clone(),
                psyche: self.session.psyche.clone(),
                final_report,
                learning_cycles: self.session.learning_cycle,
                stages_completed,
                target_reached,
                provenance_ids,
            },
            checkpoints,
        ))
    }

    /// Build sub-stage configs with personality bias applied.
    fn personality_biased_configs(
        &self,
    ) -> (
        PrerequisiteConfig,
        ResourceDiscoveryConfig,
        IngestionConfig,
        CompetenceConfig,
    ) {
        let mut prereq = self.config.prerequisite.clone().unwrap_or_default();
        let mut resource = self.config.resource_discovery.clone().unwrap_or_default();
        let ingest = self.config.ingestion.clone().unwrap_or_default();
        let mut competence = self.config.competence.clone().unwrap_or_default();

        if let Some(ref psyche) = self.session.psyche {
            apply_personality_bias(
                &psyche.archetypes,
                &mut prereq,
                &mut resource,
                &mut competence,
            );
        }

        (prereq, resource, ingest, competence)
    }
}

// ── Personality Bias ────────────────────────────────────────────────────

/// Adjust sub-stage configs based on dominant archetype weights.
fn apply_personality_bias(
    archetypes: &ArchetypeWeights,
    prereq: &mut PrerequisiteConfig,
    resource: &mut ResourceDiscoveryConfig,
    competence: &mut CompetenceConfig,
) {
    // Explorer (>0.6): broader expansion, more API calls.
    if archetypes.explorer > 0.6 {
        resource.max_api_calls = (resource.max_api_calls as f32 * 1.5) as usize;
        resource.min_vsa_similarity *= 0.8;
    }

    // Sage (>0.6): prefer depth, increase bloom depth.
    if archetypes.sage > 0.6 {
        competence.bloom_max_depth = competence.bloom_max_depth.max(4);
        resource.min_quality += 0.1;
    }

    // Guardian (>0.6): conservative thresholds.
    if archetypes.guardian > 0.6 {
        prereq.known_similarity_threshold = (prereq.known_similarity_threshold + 0.05).min(0.9);
    }

    // Healer (>0.6): focus on gaps first, lower known threshold.
    if archetypes.healer > 0.6 {
        prereq.known_min_triples = prereq.known_min_triples.saturating_sub(1).max(1);
    }
}

/// Compute exploration rate from Dreyfus level and personality.
///
/// Higher rate = more breadth-first learning; lower = deeper, focused.
pub fn compute_exploration_rate(
    assessment: Option<&SessionAssessment>,
    psyche: Option<&Psyche>,
) -> f32 {
    let dreyfus_factor = match assessment.map(|a| a.overall_dreyfus) {
        Some(DreyfusLevel::Expert) | Some(DreyfusLevel::Proficient) => 0.2,
        Some(DreyfusLevel::Competent) => 0.5,
        Some(DreyfusLevel::AdvancedBeginner) | Some(DreyfusLevel::Novice) | None => 0.8,
    };

    let personality_factor = psyche
        .map(|p| (p.archetypes.explorer * 0.3).min(0.3))
        .unwrap_or(0.0);

    (dreyfus_factor + personality_factor).min(1.0)
}

// ── Session Persistence ─────────────────────────────────────────────────

fn persist_session(session: &BootstrapSession, engine: &Engine) -> OrchestratorResult<()> {
    let data = bincode::serialize(session).map_err(|e| OrchestratorError::StageFailed {
        stage: "session_persist".to_string(),
        reason: format!("serialization failed: {e}"),
    })?;
    engine
        .store()
        .put_meta(SESSION_KEY, &data)
        .map_err(|e| OrchestratorError::Engine(Box::new(e.into())))?;
    Ok(())
}

fn load_session(engine: &Engine) -> OrchestratorResult<BootstrapSession> {
    let data = engine
        .store()
        .get_meta(SESSION_KEY)
        .map_err(|e| OrchestratorError::Engine(Box::new(e.into())))?;
    match data {
        Some(bytes) => {
            bincode::deserialize(&bytes).map_err(|e| OrchestratorError::StageFailed {
                stage: "session_load".to_string(),
                reason: format!("deserialization failed: {e}"),
            })
        }
        None => Err(OrchestratorError::NoSessionToResume),
    }
}

/// Reconstruct a minimal `ExpansionResult` from stored labels (for resume).
fn reconstruct_expansion(labels: &[String]) -> ExpansionResult {
    ExpansionResult {
        concept_count: labels.len(),
        relation_count: 0,
        rejected_count: 0,
        api_calls: 0,
        domain_prototype_id: None,
        microtheory_id: None,
        provenance_ids: Vec::new(),
        accepted_labels: labels.to_vec(),
        boundary_rejects: Vec::new(),
    }
}

/// Store orchestration provenance record (tag 69).
fn store_orchestration_provenance(
    engine: &Engine,
    session: &BootstrapSession,
    report: &Option<CompetenceReport>,
) -> OrchestratorResult<ProvenanceId> {
    let target_dreyfus = session
        .intent
        .as_ref()
        .map(|i| i.purpose.competence_level.as_label().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let (current_dreyfus, current_score) = match &session.last_assessment {
        Some(a) => (a.overall_dreyfus.as_label().to_string(), a.overall_score),
        None => ("unknown".to_string(), 0.0),
    };

    let derived_id = session
        .intent
        .as_ref()
        .and_then(|i| {
            i.purpose
                .seed_concepts
                .first()
                .and_then(|_| SymbolId::new(1))
        })
        .unwrap_or_else(|| SymbolId::new(1).expect("1 is non-zero"));

    let mut record = ProvenanceRecord::new(
        derived_id,
        DerivationKind::BootstrapOrchestration {
            stage: session.current_stage.as_label().to_string(),
            learning_cycle: session.learning_cycle as u32,
            exploration_rate: session.exploration_rate,
            target_dreyfus,
            current_dreyfus,
            current_score,
        },
    )
    .with_confidence(report.as_ref().map(|r| r.overall_score).unwrap_or(0.0));

    engine
        .store_provenance(&mut record)
        .map_err(OrchestratorError::from)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Config defaults
    #[test]
    fn config_defaults() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.max_learning_cycles, 10);
        assert!(!config.plan_only);
        assert!(config.expansion.is_none());
        assert!(config.prerequisite.is_none());
        assert!(config.resource_discovery.is_none());
        assert!(config.ingestion.is_none());
        assert!(config.competence.is_none());
    }

    // 2. BootstrapStage ordering
    #[test]
    fn stage_ordering() {
        assert!(BootstrapStage::ParsePurpose < BootstrapStage::ResolveIdentity);
        assert!(BootstrapStage::ResolveIdentity < BootstrapStage::RitualOfAwakening);
        assert!(BootstrapStage::RitualOfAwakening < BootstrapStage::DomainExpansion);
        assert!(BootstrapStage::DomainExpansion < BootstrapStage::PrerequisiteDiscovery);
        assert!(BootstrapStage::PrerequisiteDiscovery < BootstrapStage::ResourceDiscovery);
        assert!(BootstrapStage::ResourceDiscovery < BootstrapStage::CurriculumIngestion);
        assert!(BootstrapStage::CurriculumIngestion < BootstrapStage::CompetenceAssessment);
        assert!(BootstrapStage::CompetenceAssessment < BootstrapStage::Complete);
    }

    // 3. BootstrapStage as_label/from_label roundtrip
    #[test]
    fn stage_label_roundtrip() {
        let stages = [
            BootstrapStage::ParsePurpose,
            BootstrapStage::ResolveIdentity,
            BootstrapStage::RitualOfAwakening,
            BootstrapStage::DomainExpansion,
            BootstrapStage::PrerequisiteDiscovery,
            BootstrapStage::ResourceDiscovery,
            BootstrapStage::CurriculumIngestion,
            BootstrapStage::CompetenceAssessment,
            BootstrapStage::Complete,
        ];
        for stage in &stages {
            let label = stage.as_label();
            let parsed = BootstrapStage::from_label(label);
            assert_eq!(parsed, Some(*stage), "roundtrip failed for {label}");
        }
    }

    // 4. Error enum formatting
    #[test]
    fn error_formatting() {
        let e1 = OrchestratorError::EmptyPurpose;
        assert!(format!("{e1}").contains("empty purpose"));

        let e2 = OrchestratorError::StageFailed {
            stage: "test".to_string(),
            reason: "bad".to_string(),
        };
        assert!(format!("{e2}").contains("test"));
        assert!(format!("{e2}").contains("bad"));

        let e3 = OrchestratorError::MaxCyclesExhausted { max_cycles: 42 };
        assert!(format!("{e3}").contains("42"));

        let e4 = OrchestratorError::NoSessionToResume;
        assert!(format!("{e4}").contains("no saved session"));
    }

    // 5. Error From<AkhError>
    #[test]
    fn error_from_akh_error() {
        let inner = crate::error::AkhError::Store(crate::error::StoreError::NotFound {
            key: "test".to_string(),
        });
        let err: OrchestratorError = inner.into();
        assert!(matches!(err, OrchestratorError::Engine(_)));
    }

    // 6. Error From<BootstrapError>
    #[test]
    fn error_from_bootstrap_error() {
        let inner = crate::bootstrap::purpose::BootstrapError::EmptyInput;
        let err: OrchestratorError = inner.into();
        assert!(matches!(err, OrchestratorError::StageFailed { stage, .. } if stage == "parse_purpose"));
    }

    // 7. Error From<CompetenceError>
    #[test]
    fn error_from_competence_error() {
        let inner = crate::bootstrap::competence::CompetenceError::NoConcepts;
        let err: OrchestratorError = inner.into();
        assert!(
            matches!(err, OrchestratorError::StageFailed { stage, .. } if stage == "competence_assessment")
        );
    }

    // 8. Session serialization roundtrip
    #[test]
    fn session_serialization_roundtrip() {
        let session = BootstrapSession::new("test purpose".to_string());
        let bytes = bincode::serialize(&session).unwrap();
        let decoded: BootstrapSession = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.raw_purpose, "test purpose");
        assert_eq!(decoded.current_stage, BootstrapStage::ParsePurpose);
        assert_eq!(decoded.learning_cycle, 0);
    }

    // 9. SessionAssessment serialization roundtrip
    #[test]
    fn session_assessment_serialization_roundtrip() {
        let assessment = SessionAssessment {
            overall_score: 0.65,
            overall_dreyfus: DreyfusLevel::Proficient,
            focus_areas: vec!["parsing".to_string(), "codegen".to_string()],
            recommendation: "continue".to_string(),
        };
        let bytes = bincode::serialize(&assessment).unwrap();
        let decoded: SessionAssessment = bincode::deserialize(&bytes).unwrap();
        assert!((decoded.overall_score - 0.65).abs() < f32::EPSILON);
        assert_eq!(decoded.overall_dreyfus, DreyfusLevel::Proficient);
        assert_eq!(decoded.focus_areas.len(), 2);
    }

    // 10. Checkpoint construction
    #[test]
    fn checkpoint_variants() {
        let cp1 = Checkpoint::PurposeParsed {
            domain: "compilers".to_string(),
            competence_level: DreyfusLevel::Expert,
            seed_count: 3,
            has_identity: true,
        };
        assert!(format!("{cp1:?}").contains("PurposeParsed"));

        let cp2 = Checkpoint::IdentityConstructed {
            chosen_name: "Akh-Ra".to_string(),
        };
        assert!(format!("{cp2:?}").contains("IdentityConstructed"));

        let cp3 = Checkpoint::LearningPlan {
            concept_count: 42,
            relation_count: 100,
        };
        assert!(format!("{cp3:?}").contains("LearningPlan"));

        let cp4 = Checkpoint::AssessmentComplete {
            cycle: 1,
            overall_score: 0.75,
            overall_dreyfus: DreyfusLevel::Proficient,
            recommendation: "ready".to_string(),
        };
        assert!(format!("{cp4:?}").contains("AssessmentComplete"));
    }

    // 11. Exploration rate: no assessment + no psyche → high (~0.8)
    #[test]
    fn exploration_rate_no_data() {
        let rate = compute_exploration_rate(None, None);
        assert!((rate - 0.8).abs() < f32::EPSILON);
    }

    // 12. Exploration rate: Expert + sage psyche → low (~0.2-0.35)
    #[test]
    fn exploration_rate_expert_sage() {
        let assessment = SessionAssessment {
            overall_score: 0.9,
            overall_dreyfus: DreyfusLevel::Expert,
            focus_areas: vec![],
            recommendation: "ready".to_string(),
        };
        let psyche = make_test_psyche(0.0, 0.8, 0.0, 0.2); // sage=0.8
        let rate = compute_exploration_rate(Some(&assessment), Some(&psyche));
        // dreyfus_factor=0.2, personality_factor = 0.2 * 0.3 = 0.06
        assert!(rate >= 0.2 && rate <= 0.35, "rate was {rate}");
    }

    // 13. Exploration rate: Novice + explorer psyche (0.9) → very high
    #[test]
    fn exploration_rate_novice_explorer() {
        let assessment = SessionAssessment {
            overall_score: 0.1,
            overall_dreyfus: DreyfusLevel::Novice,
            focus_areas: vec![],
            recommendation: "continue".to_string(),
        };
        let psyche = make_test_psyche(0.0, 0.1, 0.0, 0.9); // explorer=0.9
        let rate = compute_exploration_rate(Some(&assessment), Some(&psyche));
        // dreyfus_factor=0.8, personality_factor = 0.9 * 0.3 = 0.27
        assert!(rate >= 0.95, "rate was {rate}");
    }

    // 14. OrchestrationResult construction
    #[test]
    fn orchestration_result_construction() {
        use crate::bootstrap::purpose::PurposeModel;
        let intent = BootstrapIntent {
            purpose: PurposeModel {
                domain: "test".to_string(),
                competence_level: DreyfusLevel::Competent,
                seed_concepts: vec!["a".to_string()],
                description: "test".to_string(),
            },
            identity: None,
        };
        let result = OrchestrationResult {
            intent,
            chosen_name: Some("Akh-Test".to_string()),
            psyche: None,
            final_report: None,
            learning_cycles: 3,
            stages_completed: vec![
                BootstrapStage::ParsePurpose,
                BootstrapStage::DomainExpansion,
            ],
            target_reached: false,
            provenance_ids: vec![],
        };
        assert_eq!(result.learning_cycles, 3);
        assert!(!result.target_reached);
        assert_eq!(result.chosen_name.as_deref(), Some("Akh-Test"));
    }

    /// Helper to build a Psyche with specified archetype weights for testing.
    fn make_test_psyche(healer: f32, sage: f32, guardian: f32, explorer: f32) -> Psyche {
        use crate::compartment::psyche::{Persona, SelfIntegration, Shadow};
        Psyche {
            persona: Persona {
                name: "Test".to_string(),
                grammar_preference: "formal".to_string(),
                traits: Vec::new(),
                tone: Vec::new(),
            },
            shadow: Shadow {
                veto_patterns: Vec::new(),
                bias_patterns: Vec::new(),
            },
            archetypes: ArchetypeWeights {
                healer,
                sage,
                guardian,
                explorer,
            },
            self_integration: SelfIntegration {
                individuation_level: 0.5,
                last_evolution_cycle: 0,
                shadow_encounters: 0,
                rebalance_count: 0,
                dominant_archetype: "sage".to_string(),
            },
            awakened: false,
        }
    }
}
