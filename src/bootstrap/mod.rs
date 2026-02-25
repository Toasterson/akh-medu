//! Purpose-driven identity bootstrapping (Phase 14a-14f).
//!
//! Parses operator purpose statements, resolves cultural/historical/fictional
//! references, constructs a personalized Psyche, performs the Ritual of
//! Awakening, expands seed concepts into a skeleton ontology, discovers
//! prerequisite relationships, classifies concepts by Vygotsky ZPD zones,
//! discovers learning resources for ZPD-proximal concepts, and ingests
//! them in curriculum order with cross-validation and VSA grounding.

pub mod competence;
pub mod expand;
pub mod identity;
pub mod ingest;
pub mod orchestrator;
pub mod prerequisite;
pub mod purpose;
pub mod resources;

pub use competence::{
    BootstrapRecommendation, CompetenceAssessor, CompetenceConfig, CompetenceError,
    CompetenceReport, CompetenceResult, KnowledgeAreaAssessment,
};
pub use expand::{DomainExpandError, DomainExpander, ExpandResult, ExpansionConfig, ExpansionResult};
pub use identity::{
    CharacterKnowledge, CultureOrigin, IdentityError, IdentityResult, RitualResult,
};
pub use ingest::{
    CurriculumIngestor, IngestionConfig, IngestionError, IngestionResult, IngestionRunResult,
};
pub use prerequisite::{
    PrereqAnalysisResult, PrerequisiteAnalyzer, PrerequisiteConfig, PrerequisiteError,
    PrerequisiteResult,
};
pub use purpose::{
    BootstrapError, BootstrapIntent, BootstrapResult, DreyfusLevel, EntityType, IdentityRef,
    PurposeModel,
};
pub use orchestrator::{
    BootstrapOrchestrator, BootstrapSession, BootstrapStage, Checkpoint, OrchestratorConfig,
    OrchestratorError, OrchestratorResult, OrchestrationResult,
};
pub use resources::{
    ResourceDiscoverer, ResourceDiscoveryConfig, ResourceDiscoveryError, ResourceDiscoveryResult,
    ResourceResult,
};
