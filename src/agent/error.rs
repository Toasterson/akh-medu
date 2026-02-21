//! Agent-specific error types with rich miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Errors specific to the agent layer (goals, tools, memory, OODA loop).
#[derive(Debug, Error, Diagnostic)]
pub enum AgentError {
    #[error("no active goals — the agent has nothing to work on")]
    #[diagnostic(
        code(akh::agent::no_goals),
        help("Add a goal with `agent.add_goal(...)` before running an OODA cycle.")
    )]
    NoGoals,

    #[error("tool not found: \"{name}\"")]
    #[diagnostic(
        code(akh::agent::tool_not_found),
        help("Register the tool first or check available tools with `agent.list_tools()`.")
    )]
    ToolNotFound { name: String },

    #[error("tool execution failed: {tool_name} — {message}")]
    #[diagnostic(
        code(akh::agent::tool_execution),
        help("The tool encountered an error. Check the inner cause for details.")
    )]
    ToolExecution { tool_name: String, message: String },

    #[error("working memory full: {capacity} entries")]
    #[diagnostic(
        code(akh::agent::memory_full),
        help(
            "Run consolidation to persist important entries, or increase working memory capacity."
        )
    )]
    WorkingMemoryFull { capacity: usize },

    #[error("goal not found: {goal_id}")]
    #[diagnostic(
        code(akh::agent::goal_not_found),
        help("The goal symbol does not exist. Check active goals with `agent.goals()`.")
    )]
    GoalNotFound { goal_id: u64 },

    #[error("consolidation failed: {message}")]
    #[diagnostic(
        code(akh::agent::consolidation),
        help("Memory consolidation could not complete. Check provenance store availability.")
    )]
    ConsolidationFailed { message: String },

    #[error("max cycles reached: {max_cycles}")]
    #[diagnostic(
        code(akh::agent::max_cycles),
        help(
            "The agent reached its cycle limit without completing all goals. \
             Increase max_cycles or simplify the goals."
        )
    )]
    MaxCyclesReached { max_cycles: usize },

    #[error("code ingestion failed for \"{path}\": {message}")]
    #[diagnostic(
        code(akh::agent::code_ingest),
        help(
            "Check that the path exists and contains valid Rust source files. \
             Ensure syn can parse the files (no syntax errors)."
        )
    )]
    CodeIngest { path: String, message: String },

    #[error("shadow veto: pattern \"{pattern_name}\" blocked tool \"{tool_name}\"")]
    #[diagnostic(
        code(akh::agent::shadow_veto),
        help(
            "The psyche's shadow system vetoed this tool based on danger metadata. \
             Check the tool's DangerLevel, capabilities, and action triggers."
        )
    )]
    ShadowVeto {
        tool_name: String,
        pattern_name: String,
        severity: f32,
        explanation: String,
    },

    #[error("CLI tool error: {tool_name} — {message}")]
    #[diagnostic(
        code(akh::agent::cli_tool),
        help("Check that the binary exists, is executable, and its CliToolManifest is valid.")
    )]
    CliToolError { tool_name: String, message: String },

    #[error("project not found: {project_id}")]
    #[diagnostic(
        code(akh::agent::project_not_found),
        help("The project symbol does not exist. Check active projects with `agent.projects()`.")
    )]
    ProjectNotFound { project_id: u64 },

    #[error("watch not found: \"{watch_id}\"")]
    #[diagnostic(
        code(akh::agent::watch_not_found),
        help("The watch ID does not exist. Check active watches with `agent.watches()`.")
    )]
    WatchNotFound { watch_id: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Channel(#[from] super::channel::ChannelError),

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::engine),
        help("An engine-level error occurred during an agent operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for AgentError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias for agent operations.
pub type AgentResult<T> = std::result::Result<T, AgentError>;
