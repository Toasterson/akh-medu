//! First-class akh configuration (`akh.toml`).
//!
//! An **akh** is the unit of agency in the pure-ACP harness. [`AkhConfig`] is the
//! persisted, file-backed identity of one akh: it embeds the existing
//! [`WorkspaceConfig`] (so today's per-workspace TOML loads unchanged) and adds
//! akh-level settings — a pinned model, the Ollama endpoint (local or
//! `ollama.com` cloud), an authoritative system prompt, related workspaces for
//! federation, automation gating, delegation, and the filesystem path mode.
//!
//! Design constraints (see ADR-053):
//! - **Back-compat is mandatory.** Every akh-level field is `#[serde(default)]`,
//!   and the workspace fields flatten in, so a legacy `{name}.toml` containing
//!   only [`WorkspaceConfig`] fields deserializes into an [`AkhConfig`] with all
//!   new fields defaulted. Absence of any akh config reproduces today's behaviour.
//! - **The explicit `system_prompt` is authoritative.** When set it *replaces*
//!   the psyche-derived prompt (locked decision #3 of the ACP pivot); psyche
//!   becomes optional flavour.
//! - **Several fields are parsed now, consumed later.** `related_workspaces`
//!   (P6 federation), `autonomy` (P5 automation gating), `delegation` (Phase 36),
//!   and `path_mode` (P4 filesystem) are stored by P1 but wired by their phases.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::agent::delegation::DelegationConfig;
use crate::agent::model_manager::ModelManagerConfig;
use crate::agent::ollama::OllamaConfig;
use crate::workspace::WorkspaceConfig;

/// Errors loading or saving an akh configuration (`akh.toml`).
///
/// Distinct from [`crate::workspace::WorkspaceError`] so diagnostics point the
/// operator at `akh.toml` specifically (per the project's miette conventions).
#[derive(Debug, Error, Diagnostic)]
pub enum AkhConfigError {
    #[error("failed to read akh config at {path}")]
    #[diagnostic(
        code(akh::akh_config::read),
        help("Ensure akh.toml exists and is readable, or omit it entirely to use name-only defaults.")
    )]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse akh config at {path}: {message}")]
    #[diagnostic(
        code(akh::akh_config::parse),
        help(
            "Check akh.toml for TOML syntax errors and that field types match — e.g. \
             `[ollama_endpoint]` needs `kind = \"local\"` or `kind = \"cloud\"`, and \
             `path_mode` is `\"local\"` or `\"remote\"`."
        )
    )]
    Parse { path: String, message: String },

    #[error("failed to write akh config to {path}")]
    #[diagnostic(
        code(akh::akh_config::write),
        help("Ensure the parent directory exists and is writable.")
    )]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Convenience alias for akh-config operations.
pub type AkhConfigResult<T> = std::result::Result<T, AkhConfigError>;

/// Where an akh's pinned model runs.
///
/// Mirrors the `env:VAR` secret indirection used by
/// [`crate::config::UpstreamEndpointConfig`]: a cloud `api_key` of the form
/// `"env:OLLAMA_API_KEY"` is resolved from the environment at call time (see
/// [`OllamaConfig::resolved_api_key`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum OllamaEndpoint {
    /// Operator-hosted Ollama reached over a plain (usually local) base URL.
    Local {
        #[serde(default = "default_local_base_url")]
        base_url: String,
    },
    /// `ollama.com` (or any Bearer-authenticated Ollama-compatible host).
    Cloud {
        #[serde(default = "default_cloud_base_url")]
        base_url: String,
        /// API key. Supports the `env:VAR` indirection (resolved at call time);
        /// a literal token is also accepted and used verbatim.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<String>,
    },
}

fn default_local_base_url() -> String {
    "http://localhost:11434".into()
}
fn default_cloud_base_url() -> String {
    "https://ollama.com".into()
}

impl Default for OllamaEndpoint {
    fn default() -> Self {
        OllamaEndpoint::Local {
            base_url: default_local_base_url(),
        }
    }
}

/// Per-akh background-automation gating (consumed by P5).
///
/// `enabled` is the master switch and defaults to **off** — a fresh akh runs no
/// background tasks. When enabled, `tasks` provides per-task overrides keyed by
/// the daemon task name (e.g. `"consolidation"`, `"reflection"`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutonomyConfig {
    /// Master switch. When false, no background automation runs (default).
    #[serde(default)]
    pub enabled: bool,
    /// Per-task enable/disable overrides keyed by daemon task name. An explicit
    /// override **always wins** over the master switch — so a single task can be
    /// turned on with the master off (`enabled=false` + `goal_generation=true`),
    /// or turned off with the master on.
    #[serde(default)]
    pub tasks: std::collections::BTreeMap<String, bool>,
}

impl AutonomyConfig {
    /// Whether a named daemon task should run. An explicit per-task override
    /// takes precedence; otherwise the task follows the master switch.
    pub fn task_enabled(&self, task: &str) -> bool {
        match self.tasks.get(task) {
            Some(&v) => v,
            None => self.enabled,
        }
    }
}

/// Filesystem access mode for an akh's tools (consumed by P4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PathMode {
    /// Native `std::fs` access, joined under the daemon `path_prefix` and
    /// sandboxed (the default; matches today's single-host behaviour).
    #[default]
    Local,
    /// Filesystem access delegated to the ACP client via `fs/*` methods.
    Remote,
}

fn default_true() -> bool {
    true
}

/// First-class akh configuration, persisted as `akh.toml` (the per-workspace
/// config file).
///
/// The embedded [`WorkspaceConfig`] is flattened, so its fields appear at the
/// top level of the TOML exactly as today — an existing workspace config file is
/// a strict subset of an `akh.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkhConfig {
    /// Embedded engine/workspace configuration (flattened: its fields stay at
    /// the top level for back-compat).
    #[serde(flatten)]
    pub workspace: WorkspaceConfig,

    // ── akh-level scalars / arrays (serialized before any table) ────────────
    /// Pinned model name. When set, this is the model the akh always answers
    /// with. `None` falls back to the endpoint/global default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Authoritative system prompt. When `Some`, it **replaces** the
    /// psyche-derived prompt; when `None`, the psyche-derived prompt is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Whether this is the akh's MAIN memory workspace (writes target MAIN).
    #[serde(default = "default_true")]
    pub is_main: bool,

    /// Names of related workspaces this akh federates reads across (P6).
    #[serde(default)]
    pub related_workspaces: Vec<String>,

    /// Filesystem access mode (P4).
    #[serde(default)]
    pub path_mode: PathMode,

    // ── tables (must be serialized after all values for TOML) ───────────────
    /// Where the pinned model runs (local Ollama or cloud).
    #[serde(default)]
    pub ollama_endpoint: OllamaEndpoint,

    /// Background-automation gating (P5). Defaults to fully disabled.
    #[serde(default)]
    pub autonomy: AutonomyConfig,

    /// Delegation / external scheduling (Phase 36). Parsed now; the
    /// `DelegationManager` is wired behind the `delegation` feature in its phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegation: Option<DelegationConfig>,
}

impl Default for AkhConfig {
    fn default() -> Self {
        Self {
            workspace: WorkspaceConfig::default(),
            model: None,
            system_prompt: None,
            is_main: true,
            related_workspaces: Vec::new(),
            path_mode: PathMode::default(),
            ollama_endpoint: OllamaEndpoint::default(),
            autonomy: AutonomyConfig::default(),
            delegation: None,
        }
    }
}

impl AkhConfig {
    /// Create an akh config with a specific name; all other fields default.
    pub fn with_name(name: &str) -> Self {
        Self {
            workspace: WorkspaceConfig::with_name(name),
            ..Default::default()
        }
    }

    /// Wrap an existing [`WorkspaceConfig`], leaving akh-level fields default.
    pub fn from_workspace(workspace: WorkspaceConfig) -> Self {
        Self {
            workspace,
            ..Default::default()
        }
    }

    /// The akh's name (delegates to the embedded workspace name).
    pub fn name(&self) -> &str {
        &self.workspace.name
    }

    /// Load from a TOML file. A legacy workspace-only TOML loads cleanly with
    /// all akh-level fields defaulted.
    pub fn load(path: &std::path::Path) -> AkhConfigResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| AkhConfigError::Read {
            path: path.display().to_string(),
            source: e,
        })?;
        toml::from_str(&content).map_err(|e| AkhConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    /// Save to a TOML file (pretty-printed), creating parent directories.
    pub fn save(&self, path: &std::path::Path) -> AkhConfigResult<()> {
        let content = toml::to_string_pretty(self).map_err(|e| AkhConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AkhConfigError::Write {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(path, content).map_err(|e| AkhConfigError::Write {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Build the [`ModelManagerConfig`] this akh's model manager should use:
    /// the endpoint (base URL + optional cloud key) and the pinned model, with
    /// the explicit system prompt carried through so it stays authoritative.
    pub fn to_model_manager_config(&self) -> ModelManagerConfig {
        let mut ollama = OllamaConfig::default();
        match &self.ollama_endpoint {
            OllamaEndpoint::Local { base_url } => {
                ollama.base_url = base_url.clone();
                ollama.api_key = None;
            }
            OllamaEndpoint::Cloud { base_url, api_key } => {
                ollama.base_url = base_url.clone();
                ollama.api_key = api_key.clone();
            }
        }
        if let Some(model) = &self.model {
            ollama.model = model.clone();
        }
        ModelManagerConfig {
            ollama,
            system_prompt: self.system_prompt.clone(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_main_and_local() {
        let cfg = AkhConfig::default();
        assert!(cfg.is_main, "akh defaults to MAIN");
        assert_eq!(cfg.path_mode, PathMode::Local);
        assert!(!cfg.autonomy.enabled, "automation is off by default");
        assert_eq!(cfg.ollama_endpoint, OllamaEndpoint::default());
        assert!(cfg.model.is_none());
        assert!(cfg.system_prompt.is_none());
    }

    #[test]
    fn autonomy_task_enabled_override_wins() {
        let mut a = AutonomyConfig::default();
        // Master off (default) → every task off.
        assert!(!a.enabled);
        assert!(!a.task_enabled("reflection"));
        // Explicit on with the master off → exactly that task runs.
        a.tasks.insert("goal_generation".into(), true);
        assert!(a.task_enabled("goal_generation"));
        assert!(!a.task_enabled("reflection"));
        // Master on → tasks follow it, except an explicit off.
        a.enabled = true;
        a.tasks.insert("reflection".into(), false);
        assert!(a.task_enabled("equivalence"), "unlisted follows master");
        assert!(!a.task_enabled("reflection"), "explicit off wins");
        assert!(a.task_enabled("goal_generation"), "explicit on still on");
    }

    #[test]
    fn autonomy_per_task_parses_from_toml() {
        // master off + a single task enabled — the canonical "run exactly one".
        let toml = r#"
            name = "thoth"
            [autonomy]
            enabled = false
            tasks = { goal_generation = true }
        "#;
        let cfg: AkhConfig = toml::from_str(toml).expect("parse");
        assert!(!cfg.autonomy.enabled);
        assert!(cfg.autonomy.task_enabled("goal_generation"));
        assert!(!cfg.autonomy.task_enabled("reflection"));
    }

    #[test]
    fn legacy_workspace_toml_loads_as_akh_config() {
        // A pre-pivot per-workspace TOML carries only WorkspaceConfig fields.
        let legacy = r#"
            name = "thoth"
            dimension = 8192
            seed_packs = ["identity", "ontology"]
        "#;
        let cfg: AkhConfig = toml::from_str(legacy).expect("legacy TOML must still parse");
        assert_eq!(cfg.name(), "thoth");
        assert_eq!(cfg.workspace.dimension, 8192);
        assert_eq!(cfg.workspace.seed_packs, vec!["identity", "ontology"]);
        // All akh-level fields fall back to defaults.
        assert!(cfg.is_main);
        assert!(cfg.model.is_none());
        assert!(cfg.system_prompt.is_none());
        assert!(cfg.related_workspaces.is_empty());
        assert_eq!(cfg.path_mode, PathMode::Local);
        assert!(!cfg.autonomy.enabled);
    }

    #[test]
    fn full_akh_toml_parses() {
        let toml_src = r#"
            name = "ptah"
            model = "qwen2.5:7b"
            system_prompt = "You are Ptah, architect of forms."
            is_main = true
            related_workspaces = ["egypt-history", "linguistics"]
            path_mode = "remote"

            [ollama_endpoint]
            kind = "cloud"
            base_url = "https://ollama.com"
            api_key = "env:OLLAMA_API_KEY"

            [autonomy]
            enabled = true
            tasks = { consolidation = true, dream = false }
        "#;
        let cfg: AkhConfig = toml::from_str(toml_src).expect("full akh.toml must parse");
        assert_eq!(cfg.name(), "ptah");
        assert_eq!(cfg.model.as_deref(), Some("qwen2.5:7b"));
        assert_eq!(
            cfg.system_prompt.as_deref(),
            Some("You are Ptah, architect of forms.")
        );
        assert_eq!(cfg.path_mode, PathMode::Remote);
        assert_eq!(cfg.related_workspaces, vec!["egypt-history", "linguistics"]);
        assert_eq!(
            cfg.ollama_endpoint,
            OllamaEndpoint::Cloud {
                base_url: "https://ollama.com".into(),
                api_key: Some("env:OLLAMA_API_KEY".into()),
            }
        );
        assert!(cfg.autonomy.enabled);
        assert_eq!(cfg.autonomy.tasks.get("consolidation"), Some(&true));
        assert_eq!(cfg.autonomy.tasks.get("dream"), Some(&false));
    }

    #[test]
    fn save_load_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");

        let mut cfg = AkhConfig::with_name("sekhmet");
        cfg.model = Some("llama3.2:3b".into());
        cfg.system_prompt = Some("You are Sekhmet.".into());
        cfg.related_workspaces = vec!["medicine".into()];
        cfg.path_mode = PathMode::Remote;
        cfg.ollama_endpoint = OllamaEndpoint::Cloud {
            base_url: "https://ollama.com".into(),
            api_key: Some("env:OLLAMA_API_KEY".into()),
        };
        cfg.autonomy.enabled = true;

        cfg.save(&path).expect("save akh.toml");
        let loaded = AkhConfig::load(&path).expect("load akh.toml");

        assert_eq!(loaded.name(), "sekhmet");
        assert_eq!(loaded.model.as_deref(), Some("llama3.2:3b"));
        assert_eq!(loaded.system_prompt.as_deref(), Some("You are Sekhmet."));
        assert_eq!(loaded.related_workspaces, vec!["medicine"]);
        assert_eq!(loaded.path_mode, PathMode::Remote);
        assert_eq!(loaded.ollama_endpoint, cfg.ollama_endpoint);
        assert!(loaded.autonomy.enabled);
    }

    #[test]
    fn to_model_manager_config_pins_model_and_prompt() {
        let mut cfg = AkhConfig::with_name("thoth");
        cfg.model = Some("qwen2.5:7b".into());
        cfg.system_prompt = Some("You are Thoth.".into());
        cfg.ollama_endpoint = OllamaEndpoint::Cloud {
            base_url: "https://ollama.com".into(),
            api_key: Some("env:OLLAMA_API_KEY".into()),
        };

        let mm = cfg.to_model_manager_config();
        assert_eq!(mm.ollama.model, "qwen2.5:7b");
        assert_eq!(mm.ollama.base_url, "https://ollama.com");
        assert_eq!(mm.ollama.api_key.as_deref(), Some("env:OLLAMA_API_KEY"));
        assert_eq!(mm.system_prompt.as_deref(), Some("You are Thoth."));
    }

    #[test]
    fn two_akhs_pin_distinct_models_and_prompts() {
        let mut a = AkhConfig::with_name("thoth");
        a.model = Some("qwen2.5:7b".into());
        a.system_prompt = Some("You are Thoth, keeper of records.".into());

        let mut b = AkhConfig::with_name("ptah");
        b.model = Some("llama3.2:3b".into());
        b.system_prompt = Some("You are Ptah, architect of forms.".into());

        let mm_a = a.to_model_manager_config();
        let mm_b = b.to_model_manager_config();

        assert_ne!(mm_a.ollama.model, mm_b.ollama.model);
        assert_ne!(mm_a.system_prompt, mm_b.system_prompt);
        assert_eq!(mm_a.ollama.model, "qwen2.5:7b");
        assert_eq!(mm_b.ollama.model, "llama3.2:3b");
    }
}
