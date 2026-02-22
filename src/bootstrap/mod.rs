//! Purpose-driven identity bootstrapping (Phase 14a+14b).
//!
//! Parses operator purpose statements, resolves cultural/historical/fictional
//! references, constructs a personalized Psyche, and performs the Ritual of
//! Awakening — self-naming via culture-specific morpheme composition.

pub mod identity;
pub mod purpose;

pub use identity::{
    CharacterKnowledge, CultureOrigin, IdentityError, IdentityResult, RitualResult,
};
pub use purpose::{
    BootstrapError, BootstrapIntent, BootstrapResult, DreyfusLevel, EntityType, IdentityRef,
    PurposeModel,
};
