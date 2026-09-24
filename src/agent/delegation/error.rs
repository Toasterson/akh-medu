//! Delegation error types with rich miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for delegation operations.
pub type DelegationResult<T> = Result<T, DelegationError>;

/// Errors specific to the delegation subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum DelegationError {
    #[error("delegation is disabled in configuration")]
    #[diagnostic(
        code(akh::delegation::disabled),
        help("Set `enabled = true` in the [delegation] section of akh.toml")
    )]
    Disabled,

    #[error("daily cost budget exceeded: spent {spent_cents}c of {budget_cents}c")]
    #[diagnostic(
        code(akh::delegation::budget_exceeded),
        help(
            "Increase daily_cost_budget_cents in akh.toml [delegation] section, \
             or wait for budget reset at midnight UTC"
        )
    )]
    BudgetExceeded { spent_cents: u64, budget_cents: u64 },

    #[error("no delegation target available for capabilities: {required:?}")]
    #[diagnostic(
        code(akh::delegation::no_target),
        help(
            "Enable additional targets in akh.toml: \
             [delegation.native], [delegation.remote], or [delegation.peers]"
        )
    )]
    NoTarget { required: Vec<String> },

    #[error("delegation at capacity: {active}/{max} active delegations")]
    #[diagnostic(
        code(akh::delegation::at_capacity),
        help(
            "Wait for active delegations to complete, or increase max_concurrent \
             in akh.toml [delegation] section"
        )
    )]
    AtCapacity { max: usize, active: usize },

    #[error("delegation timed out after {timeout_secs}s")]
    #[diagnostic(
        code(akh::delegation::timeout),
        help(
            "Increase timeout_seconds in akh.toml [delegation.native] section, \
             or break the task into smaller subtasks"
        )
    )]
    Timeout {
        timeout_secs: u64,
        delegation_id: u64,
    },

    #[error("delegation grant unavailable: {reason}")]
    #[diagnostic(
        code(akh::delegation::grant_unavailable),
        help(
            "Native-session delegation mints a per-delegation grant through the /v1 \
             core API. Start akhomed with AKH_CORE_CONFIG set, or pick another target \
             (`peer`, `remote`)."
        )
    )]
    GrantUnavailable { reason: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    NativeSession(#[from] super::native_session::NativeSessionError),

    #[error("failed to parse structured output from worker")]
    #[diagnostic(
        code(akh::delegation::parse_output),
        help(
            "The worker did not produce a valid ```akh-result``` block. \
             Check the task prompt instructs the worker to emit structured output."
        )
    )]
    ParseOutput,

    #[error("peer {endpoint} is unreachable")]
    #[diagnostic(
        code(akh::delegation::peer_unreachable),
        help(
            "Check network connectivity to {endpoint}, or remove it from \
             [delegation.peers] endpoints in akh.toml"
        )
    )]
    PeerUnreachable { endpoint: String },

    /// The peer accepted the ACP connection but the prompt turn itself
    /// failed (timeout, protocol error, or an empty response). Distinct from
    /// [`DelegationError::PeerUnreachable`] so a dead peer and a broken turn
    /// are diagnosable apart.
    #[error("peer ACP turn failed at {endpoint}: {reason}")]
    #[diagnostic(
        code(akh::delegation::peer_turn_failed),
        help(
            "The peer accepted the ACP connection but the prompt turn failed \
             (timeout, protocol error, or empty response). Check the peer \
             daemon's logs for the failing session."
        )
    )]
    PeerTurnFailed { endpoint: String, reason: String },

    #[error("remote trigger API is not yet available")]
    #[diagnostic(
        code(akh::delegation::remote_unavailable),
        help(
            "The claude.ai remote trigger API is not yet publicly available. \
             Use local delegation instead, or check for updates."
        )
    )]
    RemoteTriggerUnavailable,

    #[error("I/O error during delegation: {0}")]
    #[diagnostic(
        code(akh::delegation::io),
        help("Check file permissions and disk space")
    )]
    Io(#[from] std::io::Error),
}
