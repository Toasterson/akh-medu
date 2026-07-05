//! Error types for `grug-train`.
//!
//! Follows the repo-wide convention: `miette` + `thiserror`, with rich
//! `#[diagnostic(help(...))]` text so a human (or agent) hitting this error
//! knows exactly what to do next, not just what broke.

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for `grug-train` public APIs.
pub type GrugResult<T> = Result<T, GrugError>;

/// Errors that can occur while building, training, or evaluating a
/// Grug-Brain beachhead model.
#[derive(Debug, Error, Diagnostic)]
pub enum GrugError {
    /// A gradient we expected to exist after `.backward()` was missing.
    ///
    /// This should only happen if a tensor was created without
    /// `.require_grad()`, or if it was detached somewhere in the forward
    /// pass before the loss was computed.
    #[error("expected gradient for parameter `{param}` but none was recorded")]
    #[diagnostic(
        code(grug_train::missing_gradient),
        help(
            "Check that every leaf tensor that should receive a gradient was \
             created with `.require_grad()` (or is a Burn `Param`, which is \
             tracked automatically), and that the loss tensor was produced by \
             an unbroken chain of autodiff-tracked ops back to it. Also verify \
             the backend type alias is `Autodiff<NdArray>` (or similar) and not \
             the plain inference backend -- `.backward()` on a non-Autodiff \
             backend silently does nothing."
        )
    )]
    MissingGradient { param: String },

    /// A gradient existed but was exactly zero across every element, which
    /// almost always means the computation graph did not actually connect
    /// the parameter to the loss (dead branch, wrong tensor cloned, etc).
    #[error("gradient for parameter `{param}` was present but all-zero")]
    #[diagnostic(
        code(grug_train::zero_gradient),
        help(
            "An all-zero gradient usually means the parameter did not \
             actually participate in the forward computation that produced \
             the loss -- e.g. a stale clone was used instead of the live \
             `Param`, or a `detach()` call cut the graph. Add a print of the \
             forward pass shapes/values, or bisect the module to find where \
             the connection to `param` was lost."
        )
    )]
    ZeroGradient { param: String },

    /// The tiny sanity-check training loop did not reduce the loss on its
    /// fixed target within the configured number of steps.
    #[error("training loop did not reduce loss: start={start_loss:.6}, end={end_loss:.6} over {steps} steps")]
    #[diagnostic(
        code(grug_train::loss_did_not_decrease),
        help(
            "This is the P0 de-risking gate for Phase 51 (see \
             docs/ai/decisions/087-grug-brain-baby-llm-training.md) -- it means \
             autodiff is wired but optimization is not actually improving the \
             model. Try: (1) increasing the learning rate, (2) checking the \
             optimizer step is applied to the *same* parameter set the forward \
             pass reads (no stale snapshot), (3) confirming the loss function \
             direction is correct (minimizing, not maximizing), (4) running more \
             steps -- a flat start on a randomly-initialized tiny net can need a \
             few extra iterations to leave a plateau."
        )
    )]
    LossDidNotDecrease {
        start_loss: f64,
        end_loss: f64,
        steps: usize,
    },
}
