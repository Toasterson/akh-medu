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
             [delegation.local], [delegation.remote], or [delegation.peers]"
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
            "Increase timeout_seconds in akh.toml [delegation.local] section, \
             or break the task into smaller subtasks"
        )
    )]
    Timeout { timeout_secs: u64, delegation_id: u64 },

    #[error("worker process failed with exit code {exit_code}")]
    #[diagnostic(
        code(akh::delegation::worker_failed),
        help("Check the task prompt for clarity. Worker stderr: {stderr}")
    )]
    WorkerFailed { exit_code: i32, stderr: String },

    #[error("claude CLI not found on PATH")]
    #[diagnostic(
        code(akh::delegation::claude_not_found),
        help(
            "Install Claude Code CLI: https://docs.anthropic.com/en/docs/claude-code \
             or ensure 'claude' is on your PATH"
        )
    )]
    ClaudeNotFound,

    #[error("failed to generate MCP config: {message}")]
    #[diagnostic(
        code(akh::delegation::mcp_config),
        help("Check that the temp directory is writable and akhomed_url is configured")
    )]
    McpConfig { message: String },

    #[error("failed to parse structured output from worker")]
    #[diagnostic(
        code(akh::delegation::parse_output),
        help(
            "The worker did not produce a valid ```akh-result``` block. \
             Check the task prompt instructs the worker to emit structured output."
        )
    )]
    ParseOutput,

    #[error("I/O error during delegation: {0}")]
    #[diagnostic(
        code(akh::delegation::io),
        help("Check file permissions, disk space, and that the claude binary is accessible")
    )]
    Io(#[from] std::io::Error),
}
