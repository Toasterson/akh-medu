//! Purpose-driven identity bootstrapping (Phase 14a-14d).
//!
//! Parses operator purpose statements, resolves cultural/historical/fictional
//! references, constructs a personalized Psyche, performs the Ritual of
//! Awakening, expands seed concepts into a skeleton ontology, discovers
//! prerequisite relationships, and classifies concepts by Vygotsky ZPD zones.

pub mod expand;
pub mod identity;
pub mod prerequisite;
pub mod purpose;

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
