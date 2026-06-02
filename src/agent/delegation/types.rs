//! Core delegation types: IDs, configuration, targets, tasks, and results.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::skills::LabelTriple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// DelegationId
// ---------------------------------------------------------------------------

/// Unique identifier for a delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DelegationId(pub u64);

impl std::fmt::Display for DelegationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "del:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Configuration (maps to akh.toml [delegation] section)
// ---------------------------------------------------------------------------

/// Top-level delegation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationConfig {
    /// Master switch for the delegation subsystem.
    pub enabled: bool,
    /// Maximum concurrent active delegations.
    pub max_concurrent: usize,
    /// Daily cost budget in cents (0 = local-only, no API costs).
    pub daily_cost_budget_cents: u64,
    /// Maximum cost per individual task in cents.
    pub per_task_max_cents: u64,
    /// URL of the running akhomed server (for MCP callback config).
    pub akhomed_url: String,
    /// Local Claude Code subprocess settings.
    pub local: LocalDelegationConfig,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_concurrent: 3,
            daily_cost_budget_cents: 0,
            per_task_max_cents: 10,
            akhomed_url: "http://localhost:3001".into(),
            local: LocalDelegationConfig::default(),
        }
    }
}

/// Configuration for local Claude Code subprocess delegation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDelegationConfig {
    /// Whether local delegation is enabled.
    pub enabled: bool,
    /// Model to use (e.g., "sonnet", "haiku", "opus").
    pub model: String,
    /// Tools the subprocess is allowed to use.
    pub allowed_tools: Vec<String>,
    /// Timeout in seconds for a single delegation.
    pub timeout_seconds: u64,
    /// Working directory for the subprocess (defaults to akhomed's cwd).
    pub working_dir: Option<PathBuf>,
}

impl Default for LocalDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "sonnet".into(),
            allowed_tools: vec![
                "Read".into(),
                "Write".into(),
                "Edit".into(),
                "Bash".into(),
                "Grep".into(),
                "Glob".into(),
            ],
            timeout_seconds: 120,
            working_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Target kinds
// ---------------------------------------------------------------------------

/// What kind of target a delegation is sent to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelegationTargetKind {
    /// Local Claude Code subprocess — uses `claude --print`.
    LocalClaude {
        model: String,
        allowed_tools: Vec<String>,
        working_dir: Option<PathBuf>,
    },
    // Future: RemoteTrigger, AkhMeduPeer, McpServer
}

impl std::fmt::Display for DelegationTargetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalClaude { model, .. } => write!(f, "local-claude:{model}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Task (input to a delegation)
// ---------------------------------------------------------------------------

/// A task to be delegated to an external worker.
#[derive(Debug, Clone)]
pub struct DelegationTask {
    /// Human-readable description of what needs to be done.
    pub description: String,
    /// Relevant knowledge graph triples for context.
    pub context_triples: Vec<LabelTriple>,
    /// What capabilities the target needs (for target selection).
    pub required_capabilities: Vec<String>,
    /// The goal this task is working toward.
    pub goal_id: u64,
    /// Priority (inherited from the goal).
    pub priority: f32,
}

// ---------------------------------------------------------------------------
// Active delegation tracking
// ---------------------------------------------------------------------------

/// A delegation that is currently in-flight.
#[derive(Debug, Clone)]
pub struct ActiveDelegation {
    pub id: DelegationId,
    pub goal_id: u64,
    pub target: DelegationTargetKind,
    pub task_description: String,
    pub started_at: u64,
    pub timeout_secs: u64,
    pub status: DelegationStatus,
}

/// Status of an active delegation.
#[derive(Debug, Clone)]
pub enum DelegationStatus {
    Pending,
    Running,
    Completed,
    Failed { error: String },
    TimedOut,
}

// ---------------------------------------------------------------------------
// Outcome (result from a completed delegation)
// ---------------------------------------------------------------------------

/// The result of a completed delegation.
#[derive(Debug, Clone)]
pub struct DelegationOutcome {
    pub id: DelegationId,
    pub stdout: String,
    pub structured: Option<DelegationOutput>,
    pub wall_time: Duration,
    pub tokens_consumed: Option<u64>,
    pub cost_usd: Option<f64>,
    pub exit_code: Option<i32>,
    pub goal_id: u64,
    pub error: Option<String>,
}

/// Structured output extracted from the worker's `akh-result` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationOutput {
    pub status: String,
    pub summary: String,
    #[serde(default)]
    pub triples: Vec<LabelTriple>,
    #[serde(default)]
    pub files_modified: Vec<String>,
    #[serde(default)]
    pub follow_up: Option<String>,
}
