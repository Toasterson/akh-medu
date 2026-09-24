//! Core delegation types: IDs, configuration, targets, tasks, and results.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::skills::LabelTriple;

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
///
/// Every field has a default, so `[delegation]` only needs the keys it
/// changes (Phase 53e).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DelegationConfig {
    /// Master switch for the delegation subsystem.
    pub enabled: bool,
    /// Maximum concurrent active delegations.
    pub max_concurrent: usize,
    /// Daily cost budget in cents (0 = local-only, no API costs).
    pub daily_cost_budget_cents: u64,
    /// Maximum cost per individual task in cents.
    pub per_task_max_cents: u64,
    /// URL of the running akhomed server (remote-trigger callbacks).
    pub akhomed_url: String,
    /// Native-runtime sessions brokered by akhsessd, each carrying a
    /// per-delegation grant (Phase 53e, ADR 113 decision 9).
    pub native: NativeDelegationConfig,
    /// Remote trigger settings (claude.ai API).
    pub remote: RemoteDelegationConfig,
    /// Peer akh-medu instance settings.
    pub peers: PeerDelegationConfig,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_concurrent: 3,
            daily_cost_budget_cents: 0,
            per_task_max_cents: 10,
            akhomed_url: "http://localhost:3001".into(),
            native: NativeDelegationConfig::default(),
            remote: RemoteDelegationConfig::default(),
            peers: PeerDelegationConfig::default(),
        }
    }
}

/// Which cognitive tools a delegation grant names (Phase 53e). Resolved
/// against the akh's live tool registry when the grant is minted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GrantTools {
    /// Every read-only cognitive tool the akh has, plus `memory_write`
    /// (the ADR 113 D9 default).
    #[default]
    ReadOnlyPlusMemoryWrite,
    /// Exactly these tools.
    Explicit {
        tools: std::collections::BTreeSet<String>,
    },
}

/// Delegation to a native runtime (Codex, Claude, ...) through an akhsessd
/// session (Phase 53e). The akh mints a grant per delegation and creates the
/// session with it; the runtime's cognitive tool calls then run under the
/// grant and are recorded as the delegate's provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeDelegationConfig {
    /// Whether native-session delegation is enabled.
    pub enabled: bool,
    /// Name of the session service (a label for targets and provenance).
    pub service: String,
    /// Base URL of akhsessd (e.g. `http://127.0.0.1:18431`).
    pub url: String,
    /// File holding akhsessd's bearer token for this akh (role `client`).
    pub token_file: Option<PathBuf>,
    /// akhsessd workspace profile the delegated session runs in.
    pub workspace_profile: String,
    /// The `/v1` core principal akhsessd presents the grant as.
    pub core_principal: String,
    /// Tools the grant names.
    pub grant_tools: GrantTools,
    /// How long one delegated turn may run.
    pub timeout_seconds: u64,
    /// Interval between session polls while the turn runs.
    pub poll_interval_ms: u64,
}

impl Default for NativeDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            service: "akhsessd".into(),
            url: "http://127.0.0.1:18431".into(),
            token_file: None,
            workspace_profile: "default".into(),
            core_principal: "sessions".into(),
            grant_tools: GrantTools::default(),
            timeout_seconds: 600,
            poll_interval_ms: 2_000,
        }
    }
}

/// Configuration for remote trigger delegation (claude.ai API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDelegationConfig {
    /// Whether remote delegation is enabled (requires API access and public URL).
    pub enabled: bool,
    /// Public URL of this akhomed instance (workers call back via MCP).
    pub akhomed_public_url: String,
    /// Model to use for remote triggers.
    pub model: String,
    /// Maximum cost per remote task in cents.
    pub max_cost_per_task_cents: u64,
}

impl Default for RemoteDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            akhomed_public_url: String::new(),
            model: "sonnet".into(),
            max_cost_per_task_cents: 10,
        }
    }
}

/// Configuration for peer akh-medu instance delegation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDelegationConfig {
    /// Whether peer delegation is enabled.
    pub enabled: bool,
    /// Explicit peer ACP addresses (`host:port` for a TCP peer, an absolute
    /// socket path or `unix:<path>` for a same-host peer).
    pub endpoints: Vec<String>,
    /// Whether to attempt auto-discovery via mDNS (future feature).
    pub auto_discover: bool,
}

impl Default for PeerDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: Vec::new(),
            auto_discover: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Target kinds
// ---------------------------------------------------------------------------

/// What kind of target a delegation is sent to.
///
/// Phase 53e removed `LocalClaude`: it spawned `claude --print` pointed at
/// akhomed's `/mcp`, which ADR 061 retired. Native runtimes are reached
/// through [`Self::NativeSession`] instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelegationTargetKind {
    /// A native-runtime session brokered by akhsessd, carrying a grant.
    NativeSession {
        service: String,
        workspace_profile: String,
    },
    /// Remote trigger via claude.ai API — survives restarts.
    RemoteTrigger {
        akhomed_public_url: String,
        model: String,
    },
    /// Another akh-medu instance — for KG-level operations.
    AkhMeduPeer { endpoint: String },
}

impl std::fmt::Display for DelegationTargetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NativeSession {
                service,
                workspace_profile,
            } => write!(f, "native-session:{service}/{workspace_profile}"),
            Self::RemoteTrigger {
                akhomed_public_url,
                model,
            } => {
                write!(f, "remote-trigger:{model}@{akhomed_public_url}")
            }
            Self::AkhMeduPeer { endpoint } => write!(f, "peer:{endpoint}"),
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
    /// The goal this task is working toward (0 = no specific goal).
    pub goal_id: u64,
    /// Priority (inherited from the goal).
    pub priority: f32,
    /// Optional explicit target hint ("local", "remote", "peer").
    pub target_hint: Option<String>,
    /// Target workspace name for peer delegation (default: same as caller).
    /// When None, the peer endpoint uses its default workspace.
    pub target_workspace: Option<String>,
}

impl DelegationTask {
    /// Create a minimal task with just a description.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            context_triples: Vec::new(),
            required_capabilities: Vec::new(),
            goal_id: 0,
            priority: 0.5,
            target_hint: None,
            target_workspace: None,
        }
    }
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
    /// Which target processed this delegation.
    pub target: Option<DelegationTargetKind>,
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
