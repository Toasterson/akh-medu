//! Purpose-driven identity bootstrapping (Phase 14a-14c).
//!
//! Parses operator purpose statements, resolves cultural/historical/fictional
//! references, constructs a personalized Psyche, performs the Ritual of
//! Awakening, and expands seed concepts into a skeleton ontology.

pub mod expand;
pub mod identity;
pub mod purpose;

pub use expand::{DomainExpandError, DomainExpander, ExpandResult, ExpansionConfig, ExpansionResult};
pub use identity::{
    CharacterKnowledge, CultureOrigin, IdentityError, IdentityResult, RitualResult,
};
pub use purpose::{
    BootstrapError, BootstrapIntent, BootstrapResult, DreyfusLevel, EntityType, IdentityRef,
    PurposeModel,
};
