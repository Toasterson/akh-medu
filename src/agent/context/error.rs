//! Errors for the context-assembly runtime (Phase 44, ADR 065).

use miette::Diagnostic;
use thiserror::Error;

/// Failures during working-context assembly.
#[derive(Debug, Error, Diagnostic)]
pub enum ContextError {
    /// A manager evicted a span that provenance marks as safety-critical. This is
    /// a *programming* error in the manager — the [`SafetyEnforced`] guard fails
    /// closed (44f), complementing the ADR-060 *execution* gate which guards
    /// dangerous tool *calls*.
    ///
    /// [`SafetyEnforced`]: super::safety::SafetyEnforced
    #[error("context assembly evicted a safety-pinned span in section {section}: {first:?}")]
    #[diagnostic(
        code(akh::context::safety_pin_violation),
        help(
            "A ContextManager must never drop PinnedSafety spans (anchored by an \
             AgentDecision / ShadowVeto / a Dangerous-or-higher tool execution in \
             the provenance ledger). Fix the manager's eviction order so safety \
             spans are kept, or raise the token budget so the span fits."
        )
    )]
    SafetyPinViolation { section: String, first: String },

    /// Assembly failed for another reason. The agentic loop surfaces this and
    /// falls back to a single-turn prompt rather than dropping the turn.
    #[error("context assembly failed: {message}")]
    #[diagnostic(code(akh::context::assemble))]
    Assemble { message: String },
}

/// Convenience alias for context-assembly operations.
pub type ContextResult<T> = std::result::Result<T, ContextError>;
