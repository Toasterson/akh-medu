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

    /// The pinned remainder plus the current turn do not fit the budget.
    ///
    /// Selection can drop history, but never standing instructions or the
    /// request being answered, so there is a floor below which a turn cannot be
    /// served. Failing here is deliberate: silently dropping the current turn
    /// would answer a question the user did not ask.
    #[error("context needs {needed} tokens against a {available} token budget")]
    #[diagnostic(
        code(akh::context::budget_exceeded),
        help(
            "Everything droppable was already dropped; what remains is pinned \
             system instructions plus the current turn. Either raise the akh's \
             context window, lower the output reservation so more of the window \
             is available for the prompt, or shorten the standing system \
             instructions - they are counted in every single turn."
        )
    )]
    BudgetExceeded { needed: usize, available: usize },

    /// A message-selecting strategy was given no message array to select from.
    #[error("the {strategy} strategy needs a message array")]
    #[diagnostic(
        code(akh::context::missing_messages),
        help(
            "This strategy selects over EpisodeState::messages, which the \
             caller left unset. Either attach the turn's messages with \
             EpisodeState::with_messages, or choose a strategy that assembles \
             from EpisodeState::conversation instead."
        )
    )]
    MissingMessages { strategy: String },

    /// Assembly failed for another reason. The agentic loop surfaces this and
    /// falls back to a single-turn prompt rather than dropping the turn.
    #[error("context assembly failed: {message}")]
    #[diagnostic(code(akh::context::assemble))]
    Assemble { message: String },
}

/// Convenience alias for context-assembly operations.
pub type ContextResult<T> = std::result::Result<T, ContextError>;
