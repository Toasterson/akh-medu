//! Global akhomed configuration, persisted as TOML.
//!
//! Stored at `$XDG_CONFIG_HOME/akh-medu/config.toml`. Controls which
//! workspaces auto-start daemons on boot and overrides for daemon task
//! intervals.
//!
//! # Example
//!
//! ```toml
//! [daemon]
//! auto_start = ["default", "research"]
//!
//! [daemon.intervals]
//! idle_cycle_secs = 30
//! goal_generation_secs = 300
//! ```

use std::path::Path;
use std::time::Duration;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::agent::DaemonConfig;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from global config operations.
#[derive(Debug, Error, Diagnostic)]
pub enum ConfigError {
    #[error("failed to read config: {path}")]
    #[diagnostic(
        code(akh::config::read),
        help("Ensure {path} exists and is readable. Run akhomed once to generate defaults.")
    )]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse config: {path}")]
    #[diagnostic(
        code(akh::config::parse),
        help("Check the TOML syntax. Error: {message}")
    )]
    Parse { path: String, message: String },

    #[error("failed to write config: {path}")]
    #[diagnostic(
        code(akh::config::write),
        help("Ensure you have write permissions to the config directory.")
    )]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

pub type ConfigResult<T> = std::result::Result<T, ConfigError>;

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

/// Top-level akhomed configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AkhomedConfig {
    /// `"deployed"` forbids in-process side effects (Phase 53 decision 6, see
    /// [`crate::deployment`]); `"development"` (the default) allows them.
    /// `AKH_DEPLOYMENT=deployed` has the same effect and cannot be relaxed by
    /// the config. A scalar, so it stays above every `[table]` when saved.
    #[serde(default, skip_serializing_if = "is_development")]
    pub deployment: crate::deployment::DeploymentMode,

    /// In deployed mode, admit the legacy `local-drop` / `remote` executors
    /// (with a warning) instead of refusing them. Ignored in development.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_legacy_exec: bool,

    /// Daemon-related settings.
    #[serde(default)]
    pub daemon: DaemonSection,

    /// Autonomous triple extraction settings.
    #[serde(default)]
    pub extraction: crate::extraction::ExtractionConfig,

    /// Upstream MCP service endpoints (anima, seshat).
    #[serde(default)]
    pub upstreams: UpstreamsSection,

    /// ACP (Agent Client Protocol) transport settings.
    #[serde(default)]
    pub acp: AcpSection,

    /// SeaORM backing store: `[database]` accepts PostgreSQL or an absolute SQLite URL.
    /// `[postgres]` remains a backwards-compatible section name. When absent, the daemon runs in legacy
    /// local-only mode (oxigraph + redb on disk, no WAL / relocation).
    #[serde(default)]
    #[serde(alias = "database")]
    pub postgres: Option<PostgresConfig>,

    /// Heka long-term-memory store (separate Postgres database). `None`
    /// (default) means `run_heka_curation` no-ops — matching every other
    /// store-backed background task's degrade-quietly-when-unconfigured
    /// convention.
    #[serde(default)]
    pub heka: Option<HekaConfig>,

    /// **Host-level** execution backend, applied to every workspace this
    /// daemon runs unless an akh's own config overrides it
    /// ([`crate::akh_config::AkhConfig::exec`]).
    ///
    /// This lives here — in the daemon's config, which never leaves the host —
    /// because an executor is a host property: which OS users exist, which
    /// workdir is on this disk, which build host is reachable. Akh configs
    /// travel with the akh on relocation (ADR 049), so an executor defined
    /// per-akh follows it to hosts where none of that holds. Concretely: a
    /// relocated akh lazily loaded on a root daemon came up `InProcess` — no
    /// privilege drop — because the drop lived in the other akhs' configs and
    /// nothing carried it. A host default cannot miss a workspace.
    ///
    /// ```toml
    /// [exec]
    /// kind = "local-drop"
    /// user = "toasty"
    /// workdir = "/Users/toasty/ws"
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec: Option<crate::akh_config::ExecBackendConfig>,
}

/// ACP transport configuration. Effective only when the daemon is built with the
/// `acp` feature. By default the Unix socket is served and TCP is disabled.
///
/// Example TOML:
/// ```toml
/// [acp]
/// unix = true
/// # socket_path = "/run/akh-medu/akhomed-acp.sock"  # default: {runtime_dir}/akhomed-acp.sock
/// tcp_bind = "127.0.0.1"
/// tcp_port = 8201
/// # workspace = "default"   # default: first auto_start workspace, else "default"
///
/// # Per-akh TCP principals (Phase 53, ADR 113). When any are configured the
/// # legacy AKHOMED_SHARED_SECRET is ignored for TCP ingress.
/// [[acp.principals]]
/// id = "t3-janus"
/// token_file = "/secrets/t3-janus.key"   # absolute; one token (24+ chars) per line
/// akhs = ["janus"]                       # ["*"] alone = every akh
///
/// [[acp.principals]]
/// id = "operator"
/// token_file = "/secrets/acp-operator.key"
/// akhs = ["*"]
/// control = true                         # server-wide `_akh/*` control plane
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcpSection {
    /// Serve ACP over a Unix domain socket (default `true`).
    #[serde(default = "default_acp_unix")]
    pub unix: bool,
    /// Override the Unix socket path. `None` ⇒ `{runtime_dir}/akhomed-acp.sock`.
    #[serde(default)]
    pub socket_path: Option<String>,
    /// TCP bind host for ACP, e.g. `"127.0.0.1"`. Requires `tcp_port`.
    #[serde(default)]
    pub tcp_bind: Option<String>,
    /// TCP port for ACP. `None` disables the TCP listener.
    #[serde(default)]
    pub tcp_port: Option<u16>,
    /// Workspace (akh) the ACP server binds to. `None` ⇒ first auto-start
    /// workspace, falling back to `"default"`.
    #[serde(default)]
    pub workspace: Option<String>,
    /// Per-akh TCP ingress principals. Empty (the default) keeps the legacy,
    /// unscoped `AKHOMED_SHARED_SECRET[_FILE]`; see [`crate::acp::auth`].
    /// Serialized last, as an array of tables below the scalars above.
    /// Only exists with the `acp` feature (without it there is no ACP
    /// listener to scope).
    #[cfg(feature = "acp")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub principals: Vec<crate::acp::auth::AcpPrincipalConfig>,
}

impl Default for AcpSection {
    fn default() -> Self {
        Self {
            unix: default_acp_unix(),
            socket_path: None,
            tcp_bind: None,
            tcp_port: None,
            workspace: None,
            #[cfg(feature = "acp")]
            principals: Vec::new(),
        }
    }
}

fn default_acp_unix() -> bool {
    true
}

impl AcpSection {
    /// Resolved TCP bind address (`host:port`) if TCP serving is enabled.
    /// Requires `tcp_port`; the host defaults to `127.0.0.1` when only the port
    /// is set.
    pub fn tcp_addr(&self) -> Option<String> {
        let port = self.tcp_port?;
        let host = self.tcp_bind.as_deref().unwrap_or("127.0.0.1");
        Some(format!("{host}:{port}"))
    }
}

// ---------------------------------------------------------------------------
// Postgres backing store (Phase 38)
// ---------------------------------------------------------------------------

/// PostgreSQL backing-store configuration.
///
/// Example TOML:
/// ```toml
/// [postgres]
/// url = "postgres://akh:akh@postgres-ha.cluster:5432/akhmedu"
/// pool_size = 10
/// connect_timeout_secs = 10
/// migrate_on_start = true
/// ```
///
/// `DATABASE_URL` in the environment overrides `url` at startup (useful for
/// Kubernetes secret injection), mirroring Seshat's behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    /// Connection URL, e.g. `postgres://user:pass@host:5432/akhmedu`.
    pub url: String,

    /// Maximum number of pooled connections.
    #[serde(default = "default_pg_pool_size")]
    pub pool_size: u32,

    /// Timeout, in seconds, for establishing a new connection.
    #[serde(default = "default_pg_connect_timeout_secs")]
    pub connect_timeout_secs: u64,

    /// Run pending migrations automatically on daemon start.
    #[serde(default)]
    pub migrate_on_start: bool,

    /// Phase 38b WAL: seconds between mutation-log flush attempts.
    #[serde(default = "default_wal_flush_interval_secs")]
    pub wal_flush_interval_secs: u64,

    /// Phase 38b WAL: maximum rows per flush transaction.
    #[serde(default = "default_wal_batch_size")]
    pub wal_batch_size: usize,

    /// Phase 38b WAL: in-memory buffer cap in mebibytes before the overflow
    /// drop policy applies.
    #[serde(default = "default_wal_max_buffer_mb")]
    pub wal_max_buffer_mb: usize,

    /// Phase 38c snapshot: seconds between automatic per-workspace snapshots of
    /// the akh's relational state to Postgres. `0` disables the periodic
    /// snapshot (a final snapshot on shutdown still runs). Default: 300 (5 min).
    #[serde(default = "default_snapshot_interval_secs")]
    pub snapshot_interval_secs: u64,
}

fn default_pg_pool_size() -> u32 {
    10
}

fn default_pg_connect_timeout_secs() -> u64 {
    10
}

fn default_wal_flush_interval_secs() -> u64 {
    2
}

fn default_wal_batch_size() -> usize {
    64
}

fn default_wal_max_buffer_mb() -> usize {
    16
}

fn default_snapshot_interval_secs() -> u64 {
    300
}

// ---------------------------------------------------------------------------
// Heka backing store (Phase 42 schema; v1 curation is surveillance-only)
// ---------------------------------------------------------------------------

/// Heka long-term-memory store configuration.
///
/// A **separate** Postgres database from `[postgres]` — Heka
/// (`crates/heka`) is its own evaluation store, not the akhomed backing
/// store. Only a connection URL today; `run_heka_curation` opens a fresh
/// connection per pass rather than holding a pool, since the default
/// interval is 24h and the query is two `COUNT(*)`s.
///
/// Example TOML:
/// ```toml
/// [heka]
/// url = "postgres://heka:heka@heka-pg.cluster:5432/heka"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HekaConfig {
    /// Connection URL, e.g. `postgres://user:pass@host:5432/heka`.
    pub url: String,
}

// ---------------------------------------------------------------------------
// Upstream config
// ---------------------------------------------------------------------------

/// Cross-MCP upstream service configuration.
///
/// Example TOML:
/// ```toml
/// [upstreams.anima]
/// url = "http://anima.local:4100"
/// auth_header = "env:ANIMA_API_KEY"
///
/// [upstreams.seshat]
/// url = "http://seshat.local:4200"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpstreamsSection {
    /// Anima project-management MCP endpoint.
    #[serde(default)]
    pub anima: Option<UpstreamEndpointConfig>,
    /// Seshat document-corpus MCP endpoint.
    #[serde(default)]
    pub seshat: Option<UpstreamEndpointConfig>,
}

/// Configuration for a single upstream MCP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamEndpointConfig {
    /// Base URL of the upstream service (e.g. `http://localhost:4100`).
    /// The MCP endpoint is expected at `{url}/mcp`.
    pub url: String,

    /// Optional HTTP Authorization header value.
    ///
    /// Supports two forms:
    /// - `"Bearer <token>"` — used verbatim.
    /// - `"env:VAR_NAME"` — reads the token from the named environment variable at call time.
    #[serde(default)]
    pub auth_header: Option<String>,
}

impl UpstreamEndpointConfig {
    /// Resolve the auth header, expanding `env:VAR` references.
    ///
    /// When `auth_header` is `"env:VAR"` and `VAR` is unset, this returns
    /// `None` (not the literal `"env:VAR"` string) and emits a
    /// `tracing::warn` naming the missing variable. Sending the literal
    /// indirection string as a bearer token used to fail silently as an
    /// unexplained 401 from the upstream; returning `None` instead makes the
    /// call site (which omits the `Authorization` header entirely when this
    /// is `None`) fail visibly as unauthenticated.
    pub fn resolved_auth(&self) -> Option<String> {
        self.auth_header.as_ref().and_then(|h| {
            if let Some(var) = h.strip_prefix("env:") {
                match std::env::var(var) {
                    Ok(v) => Some(v),
                    Err(_) => {
                        tracing::warn!(
                            target: "akh_config",
                            var,
                            "auth_header references env var {var:?} but it is unset; \
                             sending no Authorization header instead of the literal \
                             indirection string"
                        );
                        None
                    }
                }
            } else {
                Some(h.clone())
            }
        })
    }
}

/// Daemon configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonSection {
    /// Workspaces to auto-start daemons for on boot.
    ///
    /// Defaults to `[]` (P5: a fresh akh runs no background daemon). Add a
    /// workspace name to opt in, or override via `AKH_AUTO_START` / `--auto-start`.
    #[serde(default = "default_auto_start")]
    pub auto_start: Vec<String>,

    /// Maximum OODA cycles before stopping (0 = unlimited).
    #[serde(default)]
    pub max_cycles: usize,

    /// Override individual daemon task intervals (in seconds).
    #[serde(default)]
    pub intervals: IntervalOverrides,
}

impl Default for DaemonSection {
    fn default() -> Self {
        Self {
            auto_start: default_auto_start(),
            max_cycles: 0,
            intervals: IntervalOverrides::default(),
        }
    }
}

fn default_auto_start() -> Vec<String> {
    // P5: off by default — no workspace auto-starts a background daemon.
    Vec::new()
}

/// Optional overrides for daemon task intervals (all in seconds).
///
/// Any field set to `None` uses the built-in default from [`DaemonConfig`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntervalOverrides {
    pub equivalence_secs: Option<u64>,
    pub reflection_secs: Option<u64>,
    pub consolidation_secs: Option<u64>,
    pub schema_discovery_secs: Option<u64>,
    pub rule_inference_secs: Option<u64>,
    pub gap_analysis_secs: Option<u64>,
    pub persist_secs: Option<u64>,
    pub idle_cycle_secs: Option<u64>,
    pub continuous_learning_secs: Option<u64>,
    pub goal_generation_secs: Option<u64>,
    pub sleep_cycle_secs: Option<u64>,
    pub trigger_evaluation_secs: Option<u64>,
    pub reactor_evaluation_secs: Option<u64>,
    pub knowledge_extraction_secs: Option<u64>,
    /// Phase 32: LLM knowledge elicitation interval (default: 30 min).
    pub knowledge_elicitation_secs: Option<u64>,
    /// Seshat's Archive curation-pass interval (default: 6 h = 21600s).
    pub seshat_curation_secs: Option<u64>,
    /// Heka curation-pass interval (default: 24 h = 86400s).
    pub heka_curation_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

impl DaemonSection {
    /// Build a [`DaemonConfig`] by applying overrides on top of defaults.
    pub fn to_daemon_config(&self) -> DaemonConfig {
        let defaults = DaemonConfig::default();
        let ov = &self.intervals;

        DaemonConfig {
            equivalence_interval: ov
                .equivalence_secs
                .map_or(defaults.equivalence_interval, Duration::from_secs),
            reflection_interval: ov
                .reflection_secs
                .map_or(defaults.reflection_interval, Duration::from_secs),
            consolidation_interval: ov
                .consolidation_secs
                .map_or(defaults.consolidation_interval, Duration::from_secs),
            schema_discovery_interval: ov
                .schema_discovery_secs
                .map_or(defaults.schema_discovery_interval, Duration::from_secs),
            rule_inference_interval: ov
                .rule_inference_secs
                .map_or(defaults.rule_inference_interval, Duration::from_secs),
            gap_analysis_interval: ov
                .gap_analysis_secs
                .map_or(defaults.gap_analysis_interval, Duration::from_secs),
            persist_interval: ov
                .persist_secs
                .map_or(defaults.persist_interval, Duration::from_secs),
            idle_cycle_interval: ov
                .idle_cycle_secs
                .map_or(defaults.idle_cycle_interval, Duration::from_secs),
            continuous_learning_interval: ov
                .continuous_learning_secs
                .map_or(defaults.continuous_learning_interval, Duration::from_secs),
            goal_generation_interval: ov
                .goal_generation_secs
                .map_or(defaults.goal_generation_interval, Duration::from_secs),
            sleep_cycle_interval: ov
                .sleep_cycle_secs
                .map_or(defaults.sleep_cycle_interval, Duration::from_secs),
            trigger_evaluation_interval: ov
                .trigger_evaluation_secs
                .map_or(defaults.trigger_evaluation_interval, Duration::from_secs),
            reactor_evaluation_interval: ov
                .reactor_evaluation_secs
                .map_or(defaults.reactor_evaluation_interval, Duration::from_secs),
            knowledge_extraction_interval: ov
                .knowledge_extraction_secs
                .map_or(defaults.knowledge_extraction_interval, Duration::from_secs),
            bridge_training_interval: defaults.bridge_training_interval,
            openclaw_sync_interval: defaults.openclaw_sync_interval,
            knowledge_elicitation_interval: ov
                .knowledge_elicitation_secs
                .map_or(defaults.knowledge_elicitation_interval, Duration::from_secs),
            kg_compaction_interval: defaults.kg_compaction_interval,
            delegation_interval: defaults.delegation_interval,
            relation_enrichment_interval: defaults.relation_enrichment_interval,
            librarian_interval: defaults.librarian_interval,
            seshat_curation_interval: ov
                .seshat_curation_secs
                .map_or(defaults.seshat_curation_interval, Duration::from_secs),
            heka_curation_interval: ov
                .heka_curation_secs
                .map_or(defaults.heka_curation_interval, Duration::from_secs),
            max_cycles: self.max_cycles,
        }
    }
}

fn is_development(mode: &crate::deployment::DeploymentMode) -> bool {
    !mode.is_deployed()
}

// ---------------------------------------------------------------------------
// Deployment policy
// ---------------------------------------------------------------------------

impl AkhomedConfig {
    /// The deployment policy this config selects, with `AKH_DEPLOYMENT`
    /// applied (it can only make the process stricter).
    pub fn exec_policy(
        &self,
    ) -> Result<crate::deployment::ExecPolicy, crate::deployment::DeploymentPolicyError> {
        Ok(crate::deployment::ExecPolicy {
            mode: crate::deployment::DeploymentMode::resolve_from_env(self.deployment)?,
            allow_legacy_exec: self.allow_legacy_exec,
        })
    }

    /// Startup check for deployed mode: the host-level `[exec]` must be
    /// admissible on its own, so an akh without an override can never fall
    /// back to in-process execution. Per-akh overrides and chat backings are
    /// checked again when each akh loads.
    pub fn check_deployment(
        &self,
    ) -> Result<crate::deployment::ExecPolicy, crate::deployment::DeploymentPolicyError> {
        let policy = self.exec_policy()?;
        let host_exec = self.exec.clone().unwrap_or_default();
        let admission = policy.check_exec("host [exec]", &host_exec)?;
        crate::deployment::warn_on_legacy("host [exec]", admission);
        Ok(policy)
    }
}

// ---------------------------------------------------------------------------
// Load / Save
// ---------------------------------------------------------------------------

impl AkhomedConfig {
    /// Load from a TOML file. Returns defaults if the file doesn't exist.
    pub fn load(path: &Path) -> ConfigResult<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::Read {
            path: path.display().to_string(),
            source: e,
        })?;
        toml::from_str(&content).map_err(|e| ConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    /// Save to a TOML file, creating parent directories as needed.
    pub fn save(&self, path: &Path) -> ConfigResult<()> {
        let content = toml::to_string_pretty(self).map_err(|e| ConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ConfigError::Write {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(path, content).map_err(|e| ConfigError::Write {
            path: path.display().to_string(),
            source: e,
        })?;
        restrict_to_owner(path);
        Ok(())
    }

    /// Load from file, or create a default config file if none exists.
    pub fn load_or_create(path: &Path) -> ConfigResult<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            let config = Self::default();
            config.save(path)?;
            Ok(config)
        }
    }
}

/// Restrict a just-written config file to owner-only read/write (`0600`) on
/// Unix. `config.toml` can carry an inline Postgres URL with embedded
/// credentials or an `[upstreams.*] auth_header` literal, so a world-readable
/// default (`0644`) leaks it to every local user. Best-effort: a failure here
/// is logged, not propagated — the file is already written and usable. No-op
/// on non-Unix targets.
fn restrict_to_owner(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            tracing::warn!(
                target: "akh_config",
                path = %path.display(),
                error = %e,
                "failed to restrict config.toml permissions to 0600 (it may contain secrets)"
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

// ---------------------------------------------------------------------------
// AKH_* environment variable registry
// ---------------------------------------------------------------------------

/// One recognized `AKH_*` environment variable: its name, a short
/// description of what it controls, and (if any) its default/unset
/// behaviour, in the caller's own words at the read site.
///
/// This is a *catalog*, not a live reader — akhomed's ~45 behavior-controlling
/// env vars are still read ad hoc at their own call sites (that migration is
/// future work; this registry is the first step so they're at least
/// discoverable). Kept in sync manually; when adding a new `AKH_*` var, add
/// an entry here too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvVarSpec {
    /// The variable name, e.g. `"AKH_AGENT_MAX_ROUNDS"`.
    pub name: &'static str,
    /// What it controls.
    pub description: &'static str,
    /// Default / behaviour when unset, in human-readable form.
    pub default: &'static str,
}

/// The full registry of recognized `AKH_*` environment variables.
///
/// Enumerated by grepping `AKH_[A-Z_]+` across `src/` and describing each
/// from its usage site. A later CLI surfaces this (`akh config env`); this
/// function only builds the table.
pub fn env_var_registry() -> &'static [EnvVarSpec] {
    &[
        EnvVarSpec {
            name: "AKH_ACP_TCP_BIND",
            description: "ACP TCP listener bind host override.",
            default: "unset: uses [acp] config / 127.0.0.1",
        },
        EnvVarSpec {
            name: "AKH_ACP_TCP_PORT",
            description: "ACP TCP listener port override.",
            default: "unset: TCP listener disabled unless configured",
        },
        EnvVarSpec {
            name: "AKH_AGENT_DEBUG",
            description: "Verbose agentic-chat debug logging to stderr.",
            default: "\"1\" enables; unset/other = off",
        },
        EnvVarSpec {
            name: "AKH_AGENT_MAX_ROUNDS",
            description: "Hard cap on agentic tool-call rounds per turn.",
            default: "unset: scaled default (see agentic_chat.rs)",
        },
        EnvVarSpec {
            name: "AKH_AGENT_MAX_TOOLS",
            description: "Cap on distinct tools exposed per agentic turn.",
            default: "unset: no cap",
        },
        EnvVarSpec {
            name: "AKH_AGENT_PARAM_REPAIR",
            description: "Enable automatic tool-call parameter repair.",
            default: "on by default; \"0\" opts out",
        },
        EnvVarSpec {
            name: "AKH_AGENT_REPEAT_LIMIT",
            description: "Max identical consecutive tool calls before the loop aborts.",
            default: "unset: DEFAULT_REPEAT_LIMIT",
        },
        EnvVarSpec {
            name: "AKH_AGENT_TOOL_REPAIR",
            description: "Enable tool-name repair (fuzzy match on unknown tool names).",
            default: "on by default; \"0\" opts out",
        },
        EnvVarSpec {
            name: "AKH_AGENT_TOOL_RETRIEVAL",
            description: "Cap on tools surfaced via semantic tool retrieval.",
            default: "unset: retrieval cap default",
        },
        EnvVarSpec {
            name: "AKH_AGENT_TURN_TIMEOUT_SECS",
            description: "Wall-clock timeout for one agentic turn.",
            default: "unset: built-in default",
        },
        EnvVarSpec {
            name: "AKH_AGENT_VALUE_GROUND",
            description: "Enable value-grounding pass on agentic turns.",
            default: "test-toggled; see agentic_chat.rs",
        },
        EnvVarSpec {
            name: "AKH_AUDIO_SCALE",
            description: "Scale audio embeddings in the Candle NLU backend.",
            default: "\"1\" enables; unset = off",
        },
        EnvVarSpec {
            name: "AKH_AUTO_START",
            description: "Comma-separated workspaces to auto-start daemons for (CLI/service override of [daemon].auto_start).",
            default: "unset: uses config; empty = no auto-start (P5)",
        },
        EnvVarSpec {
            name: "AKH_COMM_INTEL",
            description: "Enable Phase-34 communicative-intelligence response shaping.",
            default: "on by default; \"0\" disables",
        },
        EnvVarSpec {
            name: "AKH_COMPILE_FEEDBACK_TIMEOUT_SECS",
            description: "Per-process default timeout for the compile_feedback tool's `cargo check`/`clippy`/`test` runs. Distinct from AKH_TURN_IDLE_TIMEOUT_SECS (the ACP idle watchdog) and from acp_adapter.py's --timeout (the client-side transport deadline); see src/agent/tools/compile_feedback.rs module docs for how the three reconcile.",
            default: "unset: built-in default (600s)",
        },
        EnvVarSpec {
            name: "AKH_CONTEXT_STRATEGY",
            description: "Which ContextManager strategy to use (Phase 44): compacting|passive|virtual|selecting. 'selecting' reads by selection over a message array, preserving tool traffic and an exact token prefix, and requires a caller that attaches EpisodeState::messages.",
            default: "\"compacting\"",
        },
        EnvVarSpec {
            name: "AKH_DEPLOYMENT",
            description: "\"deployed\" forbids in-process side effects (in-process exec, in-process Codex/Claude runtimes; legacy local-drop/remote need allow_legacy_exec). Can only make the config stricter.",
            default: "unset: follows config.toml `deployment` (default \"development\")",
        },
        EnvVarSpec {
            name: "AKH_ENRICH_MEMORY",
            description: "Enable memory enrichment during goal generation.",
            default: "on by default; \"0\" disables",
        },
        EnvVarSpec {
            name: "AKH_EXTRACTION_API_KEY",
            description: "API key for the LLM used by triple/knowledge extraction.",
            default: "unset: extraction falls back to configured key or is disabled",
        },
        EnvVarSpec {
            name: "AKH_FONT",
            description: "Override glyph catalog font path.",
            default: "unset: built-in catalog default",
        },
        EnvVarSpec {
            name: "AKH_GEMV_BATCH",
            description: "Enable batched GEMV verify path (Phase 47d) in the Candle backend.",
            default: "on by default; \"0\" disables",
        },
        EnvVarSpec {
            name: "AKH_HIP_Q8_GEMV",
            description: "Select the parity-gated Q8-activation/packed-Q4_K RDNA3 decode path.",
            default: "on by default; \"0\" restores f32",
        },
        EnvVarSpec {
            name: "AKH_GPU_MATMUL",
            description: "Route quantized matmul through the GPU backend.",
            default: "off by default; \"1\" enables",
        },
        EnvVarSpec {
            name: "AKH_GPU_QUANT",
            description: "Enable GPU-side quantization for quantized model backends.",
            default: "off by default; \"1\" enables",
        },
        EnvVarSpec {
            name: "AKH_GRAMMAR_AGGREGATE",
            description: "Enable Phase-33a grammar aggregation (N triples -> one fluent sentence).",
            default: "on by default; \"0\" disables",
        },
        EnvVarSpec {
            name: "AKH_JOURNAL_DIR",
            description: "Override the akh-medu journal directory.",
            default: "unset: platform default under paths::journal_dir",
        },
        EnvVarSpec {
            name: "AKH_KG_DELTA",
            description: "Include KG delta context in agentic turns (opt-in, Phase 44d).",
            default: "off by default; \"1\" enables",
        },
        EnvVarSpec {
            name: "AKH_KV_CTX",
            description: "Override KV-cache context length for quantized Gemma models.",
            default: "unset: model default",
        },
        EnvVarSpec {
            name: "AKH_LIBRARIAN",
            description: "Enable the librarian daemon task.",
            default: "off by default; \"1\" enables",
        },
        EnvVarSpec {
            name: "AKH_LIBRARIAN_MODEL",
            description: "Model tag the librarian task uses.",
            default: "unset: built-in default tag",
        },
        EnvVarSpec {
            name: "AKH_LIBRARIAN_OLLAMA_URL",
            description: "Ollama endpoint for the librarian task (e.g. https://ollama.com on hosts with no local Ollama).",
            default: "unset: local Ollama",
        },
        EnvVarSpec {
            name: "AKH_LIBRARIAN_OLLAMA_API_KEY",
            description: "Auth for the librarian's Ollama endpoint; supports env:VAR indirection (secret-looking, masked).",
            default: "unset: no auth header",
        },
        EnvVarSpec {
            name: "AKH_LLM_MODEL",
            description: "Model hint for the Candle NLU backend.",
            default: "unset: backend default",
        },
        EnvVarSpec {
            name: "AKH_LLM_VERBALIZE",
            description: "Verbalization strategy override for chat responses.",
            default: "unset: built-in default strategy",
        },
        EnvVarSpec {
            name: "AKH_NLU_LLM",
            description: "Enable LLM-backed NLU.",
            default: "on by default; \"0\" disables",
        },
        EnvVarSpec {
            name: "AKH_OLLAMA_TIMEOUT_SECS",
            description: "HTTP timeout for Ollama requests.",
            default: "unset: built-in default",
        },
        EnvVarSpec {
            name: "AKH_OPENROUTER_API_KEY",
            description: "API key for OpenRouter, used by the Phase 41g benchmark pattern.",
            default: "unset: benchmark pattern disabled",
        },
        EnvVarSpec {
            name: "AKH_PERMISSION_ASK_TIMEOUT_SECS",
            description: "Timeout waiting on an ACP client permission prompt before failing closed.",
            default: "unset: built-in default",
        },
        EnvVarSpec {
            name: "AKH_REMOTE",
            description: "Remote akhomed target for the CLI (host/URL) instead of the local daemon.",
            default: "unset: talk to local daemon",
        },
        EnvVarSpec {
            name: "AKH_SERVER_BIND",
            description: "HTTP server bind address.",
            default: "\"0.0.0.0\"",
        },
        EnvVarSpec {
            name: "AKH_SERVER_PORT",
            description: "HTTP server port.",
            default: "\"8200\"",
        },
        EnvVarSpec {
            name: "AKH_SPEC_DECODE",
            description: "Enable speculative (prompt-lookup) decode for the Candle backing (Phase 47/ADR 075), unless overridden per-akh by AkhConfig::spec_decode.",
            default: "off by default; \"1\" enables",
        },
        EnvVarSpec {
            name: "AKH_TRANSCRIPT_CAPTURE",
            description: "Override per-akh capture_transcripts at the emit site.",
            default: "unset: follows AkhConfig::capture_transcripts (on by default)",
        },
        EnvVarSpec {
            name: "AKH_TURN_IDLE_TIMEOUT_SECS",
            description: "ACP turn idle timeout before the session is considered stalled.",
            default: "unset: built-in default (300s)",
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 53 fix-up: `[[acp.principals]]` parses under `[acp]`, round-trips
    /// with the scalars kept above the array of tables, and rejects typos.
    #[cfg(feature = "acp")]
    #[test]
    fn acp_principals_parse_and_round_trip() {
        let src = r#"
            [acp]
            tcp_port = 8201

            [[acp.principals]]
            id = "t3-janus"
            token_file = "/secrets/t3-janus.key"
            akhs = ["janus"]

            [[acp.principals]]
            id = "operator"
            token_file = "/secrets/operator.key"
            akhs = ["*"]
            control = true
        "#;
        let cfg: AkhomedConfig = toml::from_str(src).expect("parse");
        assert_eq!(cfg.acp.tcp_port, Some(8201));
        assert_eq!(cfg.acp.principals.len(), 2);
        assert!(!cfg.acp.principals[0].control);
        assert!(cfg.acp.principals[1].control);
        let saved = toml::to_string(&cfg).expect("serialize");
        let reloaded: AkhomedConfig = toml::from_str(&saved).expect("reload");
        assert_eq!(reloaded.acp.principals, cfg.acp.principals);
        assert_eq!(reloaded.acp.tcp_port, Some(8201));
        assert!(
            toml::from_str::<AkhomedConfig>(
                "[[acp.principals]]\nid = \"x\"\ntoken_file = \"/x\"\nakh = [\"x\"]\n"
            )
            .is_err(),
            "a typo'd principal key is refused, not dropped"
        );
    }

    #[test]
    fn default_config_round_trips() {
        let config = AkhomedConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: AkhomedConfig = toml::from_str(&toml_str).unwrap();
        assert!(
            parsed.daemon.auto_start.is_empty(),
            "P5: auto_start is empty by default"
        );
        assert_eq!(parsed.daemon.max_cycles, 0);
    }

    #[test]
    fn partial_overrides_keep_defaults() {
        let toml_str = r#"
[daemon]
auto_start = ["research", "personal"]

[daemon.intervals]
idle_cycle_secs = 10
"#;
        let config: AkhomedConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.daemon.auto_start, vec!["research", "personal"]);

        let dc = config.daemon.to_daemon_config();
        assert_eq!(dc.idle_cycle_interval, Duration::from_secs(10));
        // Unset fields keep defaults.
        assert_eq!(dc.equivalence_interval, Duration::from_secs(300));
        assert_eq!(dc.goal_generation_interval, Duration::from_secs(300));
    }

    #[test]
    fn empty_auto_start_disables() {
        let toml_str = r#"
[daemon]
auto_start = []
"#;
        let config: AkhomedConfig = toml::from_str(toml_str).unwrap();
        assert!(config.daemon.auto_start.is_empty());
    }

    #[test]
    fn load_nonexistent_returns_defaults() {
        let config = AkhomedConfig::load(Path::new("/tmp/akh-does-not-exist.toml")).unwrap();
        assert!(
            config.daemon.auto_start.is_empty(),
            "P5: auto_start is empty by default"
        );
    }

    #[test]
    fn upstreams_section_parses() {
        let toml_str = r#"
[upstreams.anima]
url = "http://anima.local:4100"
auth_header = "env:ANIMA_TOKEN"

[upstreams.seshat]
url = "http://seshat.local:4200"
"#;
        let config: AkhomedConfig = toml::from_str(toml_str).unwrap();
        let anima = config.upstreams.anima.as_ref().unwrap();
        assert_eq!(anima.url, "http://anima.local:4100");
        assert_eq!(anima.auth_header.as_deref(), Some("env:ANIMA_TOKEN"));
        let seshat = config.upstreams.seshat.as_ref().unwrap();
        assert_eq!(seshat.url, "http://seshat.local:4200");
        assert!(seshat.auth_header.is_none());
    }

    #[test]
    fn upstreams_section_defaults_to_empty() {
        let config = AkhomedConfig::default();
        assert!(config.upstreams.anima.is_none());
        assert!(config.upstreams.seshat.is_none());
    }

    #[test]
    fn postgres_section_parses_with_defaults() {
        let toml_str = r#"
[postgres]
url = "postgres://akh:pw@db.local:5432/akhmedu"
"#;
        let config: AkhomedConfig = toml::from_str(toml_str).unwrap();
        let pg = config.postgres.as_ref().unwrap();
        assert_eq!(pg.url, "postgres://akh:pw@db.local:5432/akhmedu");
        assert_eq!(pg.pool_size, 10);
        assert_eq!(pg.connect_timeout_secs, 10);
        assert!(!pg.migrate_on_start);
        // Phase 38b WAL defaults.
        assert_eq!(pg.wal_flush_interval_secs, 2);
        assert_eq!(pg.wal_batch_size, 64);
        assert_eq!(pg.wal_max_buffer_mb, 16);
    }

    #[test]
    fn postgres_section_honours_overrides() {
        let toml_str = r#"
[postgres]
url = "postgres://db/akh"
pool_size = 25
connect_timeout_secs = 5
migrate_on_start = true
"#;
        let config: AkhomedConfig = toml::from_str(toml_str).unwrap();
        let pg = config.postgres.as_ref().unwrap();
        assert_eq!(pg.pool_size, 25);
        assert_eq!(pg.connect_timeout_secs, 5);
        assert!(pg.migrate_on_start);
    }

    #[test]
    fn postgres_section_absent_means_local_only() {
        let config = AkhomedConfig::default();
        assert!(config.postgres.is_none());
        // A config with only a [daemon] section must still parse with no [postgres].
        let config: AkhomedConfig = toml::from_str("[daemon]\nauto_start = []\n").unwrap();
        assert!(config.postgres.is_none());
    }

    #[test]
    fn upstream_env_auth_expansion() {
        unsafe { std::env::set_var("_AKH_TEST_TOKEN", "secret-token") };
        let endpoint = UpstreamEndpointConfig {
            url: "http://example.com".into(),
            auth_header: Some("env:_AKH_TEST_TOKEN".into()),
        };
        assert_eq!(endpoint.resolved_auth().as_deref(), Some("secret-token"));
        unsafe { std::env::remove_var("_AKH_TEST_TOKEN") };
    }

    #[test]
    fn upstream_literal_auth_passthrough() {
        let endpoint = UpstreamEndpointConfig {
            url: "http://example.com".into(),
            auth_header: Some("Bearer tok123".into()),
        };
        assert_eq!(endpoint.resolved_auth().as_deref(), Some("Bearer tok123"));
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut config = AkhomedConfig::default();
        config.daemon.auto_start = vec!["ws1".into(), "ws2".into()];
        config.daemon.intervals.idle_cycle_secs = Some(15);
        config.save(&path).unwrap();

        let loaded = AkhomedConfig::load(&path).unwrap();
        assert_eq!(loaded.daemon.auto_start, vec!["ws1", "ws2"]);
        assert_eq!(loaded.daemon.intervals.idle_cycle_secs, Some(15));
    }

    #[cfg(unix)]
    #[test]
    fn saved_config_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        AkhomedConfig::default().save(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "config.toml may contain secrets and must be 0600, got {mode:o}"
        );
    }

    #[test]
    fn unset_env_var_auth_resolves_to_none_not_literal() {
        // Ensure the var really is unset in this process.
        unsafe { std::env::remove_var("AKH_TEST_DEFINITELY_UNSET_XYZ") };
        let endpoint = UpstreamEndpointConfig {
            url: "http://example.com".into(),
            auth_header: Some("env:AKH_TEST_DEFINITELY_UNSET_XYZ".into()),
        };
        // Must NOT fall back to the literal "env:AKH_TEST_DEFINITELY_UNSET_XYZ"
        // string — that would send a bogus bearer token instead of failing
        // visibly as unauthenticated.
        assert_eq!(endpoint.resolved_auth(), None);
    }

    #[test]
    fn env_var_registry_is_nonempty_and_all_akh_prefixed() {
        let registry = env_var_registry();
        assert!(!registry.is_empty());
        for spec in registry {
            assert!(
                spec.name.starts_with("AKH_"),
                "{} must be AKH_-prefixed",
                spec.name
            );
            assert!(
                !spec.description.is_empty(),
                "{} needs a description",
                spec.name
            );
        }
        // No duplicate names.
        let mut names: Vec<&str> = registry.iter().map(|s| s.name).collect();
        names.sort_unstable();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped, "env_var_registry has duplicate entries");
    }
}

#[cfg(test)]
mod deployment_config_tests {
    use super::*;
    use crate::deployment::{DeploymentMode, DeploymentPolicyError};

    #[test]
    fn deployment_defaults_to_development_and_is_not_written_back() {
        let config = AkhomedConfig::default();
        assert_eq!(config.deployment, DeploymentMode::Development);
        assert!(!config.allow_legacy_exec);
        let saved = toml::to_string_pretty(&config).unwrap();
        assert!(!saved.contains("deployment"), "{saved}");
        assert!(!saved.contains("allow_legacy_exec"), "{saved}");
    }

    #[test]
    fn deployed_config_rejects_an_in_process_host_executor() {
        // No [exec] means in-process, which a deployed host must refuse.
        let config: AkhomedConfig = toml::from_str("deployment = \"deployed\"\n").unwrap();
        assert_eq!(config.deployment, DeploymentMode::Deployed);
        let error = config.check_deployment().unwrap_err();
        assert_eq!(
            error.code().unwrap().to_string(),
            "akh::config::in_process_forbidden"
        );
        assert!(matches!(
            error,
            DeploymentPolicyError::InProcessForbidden { .. }
        ));
    }

    #[test]
    fn deployed_config_admits_execd_and_gates_local_drop_on_the_opt_in() {
        let execd: AkhomedConfig = toml::from_str(
            "deployment = \"deployed\"\n[exec]\nkind = \"execd\"\n\
             sandbox_url = \"https://akhsbxd.lab\"\nsandbox_token_file = \"/run/akh/token\"\n\
             profile = \"rust\"\nprincipal = \"janus\"\n",
        )
        .unwrap();
        assert!(execd.check_deployment().unwrap().mode.is_deployed());

        let legacy = "deployment = \"deployed\"\n[exec]\nkind = \"local-drop\"\nuser = \"akh\"\n";
        let refused: AkhomedConfig = toml::from_str(legacy).unwrap();
        assert!(matches!(
            refused.check_deployment().unwrap_err(),
            DeploymentPolicyError::LegacyExecForbidden { .. }
        ));
        let opted_in: AkhomedConfig =
            toml::from_str(&format!("allow_legacy_exec = true\n{legacy}")).unwrap();
        opted_in.check_deployment().unwrap();
        // The opt-in survives a save/load round trip, above the [exec] table.
        let saved = toml::to_string_pretty(&opted_in).unwrap();
        let reloaded: AkhomedConfig = toml::from_str(&saved).unwrap();
        assert!(reloaded.allow_legacy_exec);
        assert_eq!(reloaded.deployment, DeploymentMode::Deployed);
    }
}
