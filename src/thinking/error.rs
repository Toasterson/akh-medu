//! Diagnostic error types for the Thinking Engine (Phase 41).
//!
//! Follows the repo-wide miette pattern: every variant carries a
//! `#[diagnostic(code(...), help(...))]`, engine errors chain transparently,
//! and public APIs return [`ThinkingResult`].

use miette::Diagnostic;
use thiserror::Error;

use crate::error::AkhError;

/// Result alias for all Thinking Engine public APIs.
pub type ThinkingResult<T> = std::result::Result<T, ThinkingError>;

/// Errors raised by the unified thinking loop (Phase 41).
#[derive(Debug, Error, Diagnostic)]
pub enum ThinkingError {
    /// An underlying engine operation failed while running a primitive step.
    ///
    /// Boxed to break the recursive type cycle (`AkhError::Thinking` wraps
    /// `ThinkingError`, which wraps `AkhError`), mirroring `AgentError::Engine`.
    #[error("engine operation failed during deliberation: {source}")]
    #[diagnostic(
        code(akh::thinking::engine),
        help(
            "A reasoning primitive delegated to the engine and the engine call failed. \
             Inspect the inner error; the thinking loop itself is likely fine. \
             You can retry the deliberation with a smaller budget, or run the \
             failing primitive (see the trace's last ThoughtStep) in isolation."
        )
    )]
    Engine {
        #[source]
        source: Box<AkhError>,
    },

    /// The caller asked for a deliberation with an unusable budget.
    #[error("invalid thinking budget: {reason}")]
    #[diagnostic(
        code(akh::thinking::invalid_budget),
        help(
            "Provide a budget with max_steps >= 1. Use `ThinkingBudget::steps(n)` \
             for a simple step-bounded budget, or `ThinkingBudget::default()` for \
             a sensible interactive default."
        )
    )]
    InvalidBudget { reason: String },

    /// The scheduler was constructed with no primitives to rotate over.
    #[error("scheduler has no primitives registered")]
    #[diagnostic(
        code(akh::thinking::empty_rotation),
        help(
            "Register at least one primitive in the rotation. `Scheduler::round_robin()` \
             and `Scheduler::goal_shape(...)` provide the default full rotation."
        )
    )]
    EmptyRotation,

    /// A weighted scheduler was given weights that sum to zero (or are negative).
    #[error("scheduler weights are unusable: {reason}")]
    #[diagnostic(
        code(akh::thinking::bad_weights),
        help(
            "Weighted scheduling needs at least one strictly positive weight. \
             Fix the weight table or fall back to `Scheduler::round_robin()`."
        )
    )]
    BadWeights { reason: String },

    /// The goal symbol could not be grounded in the knowledge graph.
    #[error("goal symbol {goal} is not present in the knowledge graph")]
    #[diagnostic(
        code(akh::thinking::ungrounded_goal),
        help(
            "The deliberation goal must be a symbol that exists in the KG so the \
             loop has somewhere to start. Create it (engine.resolve_or_create_entity) \
             and assert at least one triple about it before deliberating."
        )
    )]
    UngroundedGoal { goal: u64 },

    /// Persistence backend rejected a flush (Phase 41d seam).
    #[error("thought persistence failed: {message}")]
    #[diagnostic(
        code(akh::thinking::persistence),
        help(
            "The configured ThoughtPersistence backend failed to record a step. \
             With the in-memory backend this indicates a bug; with a future \
             Postgres backend (Phase 41d) check connectivity and the \
             akhomed.thinking_traces schema."
        )
    )]
    Persistence { message: String },
}

impl From<AkhError> for ThinkingError {
    fn from(err: AkhError) -> Self {
        ThinkingError::Engine {
            source: Box::new(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::VsaError;

    #[test]
    fn akh_error_boxes_into_thinking_engine_variant() {
        let inner: AkhError = VsaError::EmptyBundle.into();
        let err: ThinkingError = inner.into();
        assert!(matches!(err, ThinkingError::Engine { .. }));
    }

    #[test]
    fn thinking_error_chains_into_akh_error() {
        let err = ThinkingError::EmptyRotation;
        let akh: AkhError = err.into();
        assert!(matches!(akh, AkhError::Thinking(_)));
    }

    #[test]
    fn diagnostics_have_codes() {
        let err = ThinkingError::InvalidBudget {
            reason: "max_steps = 0".into(),
        };
        let code = miette::Diagnostic::code(&err).expect("diagnostic code");
        assert_eq!(code.to_string(), "akh::thinking::invalid_budget");
    }
}
