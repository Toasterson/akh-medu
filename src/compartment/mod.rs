//! Knowledge compartmentalization: isolates knowledge by purpose.
//!
//! Compartments scope triples into named groups (skills, projects, core modules)
//! so they can be loaded, unloaded, and queried independently.
//!
//! ## Microtheories (Phase 9a)
//!
//! Compartments are promoted to first-class microtheories — reasoning contexts
//! with inheritance (`ctx:specializes`), domain assumptions (`ctx:assumes`),
//! disjointness (`ctx:disjoint`), and lifting rules for cross-context propagation.
//! See [`microtheory`] module for details.

pub mod cwa;
pub mod eliza;
pub mod error;
pub mod manager;
pub mod manifest;
pub mod microtheory;
pub mod prompt;
pub mod psyche;
pub mod psyche_compiler;

pub use eliza::{Decomposition, DefaultResponses, ElizaRule, ElizaScript, ScriptMeta};
pub use error::{CompartmentError, CompartmentResult};
pub use manager::CompartmentManager;
pub use manifest::{CompartmentKind, CompartmentManifest, CompartmentScope, CompartmentState};
pub use microtheory::{
    ContextAncestryCache, ContextDomain, ContextPredicates, LiftCondition, LiftingRule, Microtheory,
};
pub use prompt::{PsychePromptInput, PsychePromptSerializer, estimate_tokens};
pub use psyche::{OceanProfile, Psyche};
pub use psyche_compiler::{
    CompilerError, CompilerResult, ReflectionDepth, compile as compile_psyche_to_script,
    compile_to_toml as compile_psyche_to_toml,
};
