//! Purpose-driven identity bootstrapping (Phase 14a-14e).
//!
//! Parses operator purpose statements, resolves cultural/historical/fictional
//! references, constructs a personalized Psyche, performs the Ritual of
//! Awakening, expands seed concepts into a skeleton ontology, discovers
//! prerequisite relationships, classifies concepts by Vygotsky ZPD zones,
//! and discovers learning resources for ZPD-proximal concepts.

pub mod expand;
pub mod identity;
pub mod prerequisite;
pub mod purpose;
pub mod resources;

pub use expand::{DomainExpandError, DomainExpander, ExpandResult, ExpansionConfig, ExpansionResult};
pub use identity::{
    CharacterKnowledge, CultureOrigin, IdentityError, IdentityResult, RitualResult,
};
pub use prerequisite::{
    PrereqAnalysisResult, PrerequisiteAnalyzer, PrerequisiteConfig, PrerequisiteError,
    PrerequisiteResult,
};
pub use purpose::{
    BootstrapError, BootstrapIntent, BootstrapResult, DreyfusLevel, EntityType, IdentityRef,
    PurposeModel,
};
pub use resources::{
    ResourceDiscoverer, ResourceDiscoveryConfig, ResourceDiscoveryError, ResourceDiscoveryResult,
    ResourceResult,
};
