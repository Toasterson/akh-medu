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
}

impl Default for AcpSection {
    fn default() -> Self {
        Self {
            unix: default_acp_unix(),
            socket_path: None,
            tcp_bind: None,
            tcp_port: None,
            workspace: None,
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
    pub fn resolved_auth(&self) -> Option<String> {
        self.auth_header.as_ref().map(|h| {
            if let Some(var) = h.strip_prefix("env:") {
                std::env::var(var).unwrap_or_else(|_| h.clone())
            } else {
                h.clone()
            }
        })
    }
}

/// Daemon configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonSection {
    /// Workspaces to auto-start daemons for on boot.
    ///
    /// Defaults to `["default"]`. Set to empty list to disable auto-start.
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
    vec!["default".into()]
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
    pub knowledge_extraction_secs: Option<u64>,
    /// Phase 32: LLM knowledge elicitation interval (default: 30 min).
    pub knowledge_elicitation_secs: Option<u64>,
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
            max_cycles: self.max_cycles,
        }
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
        })
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips() {
        let config = AkhomedConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: AkhomedConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.daemon.auto_start, vec!["default".to_string()]);
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
        assert_eq!(config.daemon.auto_start, vec!["default".to_string()]);
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
}
