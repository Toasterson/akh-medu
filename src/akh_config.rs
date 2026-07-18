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

/// One unrecognized key found while loading an `akh.toml`, surfaced so the
/// operator can see exactly what was silently dropped (see
/// [`AkhConfig::unknown_keys`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownConfigKey {
    /// Dotted path of the offending key, e.g. `"ollama_endpoint.kidn"` or the
    /// bare top-level key, e.g. `"modle"`.
    pub path: String,
    /// Set when the key exactly matches a *known top-level* field name but was
    /// found nested one level deeper than expected — the classic footgun of a
    /// scalar written under a preceding `[table]` header instead of above it.
    pub misplaced_under_table: Option<String>,
}

impl std::fmt::Display for UnknownConfigKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.misplaced_under_table {
            Some(table) => write!(
                f,
                "`{}` (nested under [{}] but that name matches a top-level akh.toml key — \
                 it was probably meant to be a top-level scalar written *above* the first \
                 [table] header, not inside one)",
                self.path, table
            ),
            None => write!(f, "`{}` (not a recognized akh.toml key)", self.path),
        }
    }
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
/// Which LLM backing drives this akh's chat/agentic turn (ADR 070).
///
/// `Ollama` is the common default (local or `ollama.com` cloud, native
/// function-calling). `Candle` runs an in-process GGUF model with
/// grammar-constrained tool calling — device-limited (GPU helps, but it runs
/// CPU-only too), for hosts that prefer fully local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatBacking {
    /// Ollama, local or cloud (the default).
    #[default]
    Ollama,
    /// In-process Candle GGUF model.
    Candle,
}

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

    /// Reasoning-effort hint for the akh's reasoning cloud model: `"low"`,
    /// `"medium"`, or `"high"`. `None` (the default) leaves the model's own
    /// default in effect. Carried into [`OllamaConfig::reasoning_effort`] by
    /// [`Self::to_model_manager_config`] and sent as `think` on the chat path.
    /// Serialized as a top-level scalar (before the `[ollama_endpoint]` table).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Authoritative system prompt. When `Some`, it **replaces** the
    /// psyche-derived prompt; when `None`, the psyche-derived prompt is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Whether this is the akh's MAIN memory workspace (writes target MAIN).
    #[serde(default = "default_true")]
    pub is_main: bool,

    /// Which chat/agentic backing this akh uses (ADR 070): `"ollama"` (default,
    /// local or cloud) or `"candle"` (in-process GGUF, grammar-constrained tool
    /// calling, device-limited). Serialized as a top-level scalar.
    #[serde(default)]
    pub backend: ChatBacking,

    /// Speculative (prompt-lookup) **greedy** decode for the Candle backing (Phase 47 / ADR 075).
    /// gemma4 only; ~2× on echo-heavy output (agentic / RAG / structured), lossless vs plain
    /// greedy, near-neutral otherwise. Off by default. Seeds the live `InferenceOptions` flag at
    /// startup; toggle at runtime with the `inference_config` tool. No effect on the Ollama
    /// backing. (The batched verify GEMV it relies on is on by default — Phase 47d.)
    #[serde(default)]
    pub spec_decode: bool,

    /// Names of related workspaces this akh federates reads across (P6).
    #[serde(default)]
    pub related_workspaces: Vec<String>,

    /// Filesystem access mode (P4).
    #[serde(default)]
    pub path_mode: PathMode,

    /// Opt-in: inject the OpenClaw self-import surface (ADR 076) into this akh's
    /// sessions — the `openclaw_discover` tool plus read-only access to
    /// `~/.openclaw`. Off by default, so normal akhs are unaffected. The single
    /// source of truth is this per-akh switch (no env override).
    #[serde(default)]
    pub openclaw_import: bool,

    /// Extra read-only filesystem roots to fold into the agent's `LocalFs` read
    /// roots (ADR 076 §4). Writes always stay confined to `scratch/`. Used
    /// alongside `openclaw_import`, which additionally implies `~/.openclaw`.
    #[serde(default)]
    pub extra_read_roots: Vec<std::path::PathBuf>,

    /// Whether completed agentic turns are captured to the WAL as raw
    /// transcript staging (ADR 091, `kind="turn"`). Default-on: capture is
    /// the akh's own memory, so on-by-default is right, but a sovereign akh
    /// must be able to refuse. `AKH_TRANSCRIPT_CAPTURE` ("0"/"1") overrides
    /// this at the emit site regardless of what is set here.
    ///
    /// Kept here, above the table fields below, so it serializes as a
    /// top-level scalar and precedes the first `[table]` header — a scalar
    /// declared after a table would otherwise be silently nested into that
    /// table by TOML (see the `AkhConfigError` diagnostics on this footgun).
    #[serde(default = "default_true")]
    pub capture_transcripts: bool,

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
            reasoning_effort: None,
            system_prompt: None,
            is_main: true,
            backend: ChatBacking::default(),
            spec_decode: false,
            related_workspaces: Vec::new(),
            path_mode: PathMode::default(),
            openclaw_import: false,
            extra_read_roots: Vec::new(),
            capture_transcripts: true,
            ollama_endpoint: OllamaEndpoint::default(),
            autonomy: AutonomyConfig::default(),
            delegation: None,
        }
    }
}

/// Expand a leading `~` / `~/` to `$HOME`. Paths without a leading tilde are
/// returned unchanged.
fn expand_home(path: &std::path::Path) -> std::path::PathBuf {
    let raw = path.to_string_lossy();
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest);
        }
    } else if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home);
        }
    }
    path.to_path_buf()
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

    /// The read-only filesystem roots this akh's agent may additionally read
    /// (ADR 076 §4): the explicit `extra_read_roots` (with `~` expanded), plus
    /// `~/.openclaw` when `openclaw_import` is on. Writes are unaffected — they
    /// always stay confined to `scratch/`. Empty when the switch is off and no
    /// extra roots are configured (normal akhs are unaffected).
    pub fn effective_read_roots(&self) -> Vec<std::path::PathBuf> {
        let mut roots: Vec<std::path::PathBuf> =
            self.extra_read_roots.iter().map(|p| expand_home(p)).collect();
        if self.openclaw_import {
            let openclaw = expand_home(std::path::Path::new("~/.openclaw"));
            if !roots.contains(&openclaw) {
                roots.push(openclaw);
            }
        }
        roots
    }

    /// Load from a TOML file. A legacy workspace-only TOML loads cleanly with
    /// all akh-level fields defaulted.
    ///
    /// Also validates the raw TOML against the known schema and emits a
    /// `tracing::warn` naming every unrecognized key (typo'd field, or a
    /// scalar misplaced under a `[table]` header) — `#[serde(flatten)]` on
    /// [`Self::workspace`] makes `deny_unknown_fields` impossible, so this is
    /// the only diagnostic an operator gets for a dropped key. Use
    /// [`Self::load_checked`] to also get the list programmatically.
    pub fn load(path: &std::path::Path) -> AkhConfigResult<Self> {
        let (cfg, unknown) = Self::load_checked(path)?;
        for key in &unknown {
            tracing::warn!(
                target: "akh_config",
                path = %path.display(),
                key = %key.path,
                "unrecognized key in akh.toml, value was dropped: {key}"
            );
        }
        Ok(cfg)
    }

    /// Like [`Self::load`], but also returns the list of unrecognized keys
    /// found in the raw TOML instead of only logging them.
    pub fn load_checked(
        path: &std::path::Path,
    ) -> AkhConfigResult<(Self, Vec<UnknownConfigKey>)> {
        let content = std::fs::read_to_string(path).map_err(|e| AkhConfigError::Read {
            path: path.display().to_string(),
            source: e,
        })?;
        let cfg: Self = toml::from_str(&content).map_err(|e| AkhConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let raw: toml::Value = toml::from_str(&content).map_err(|e| AkhConfigError::Parse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Ok((cfg, Self::unknown_keys(&raw)))
    }

    /// Top-level scalar/array keys accepted at the root of `akh.toml`
    /// (the [`Self::workspace`] flatten plus the akh-level fields declared
    /// above the `[table]` section of this struct).
    fn known_top_level_keys() -> &'static [&'static str] {
        &[
            // WorkspaceConfig (flattened)
            "name",
            "dimension",
            "encoding",
            "language",
            "max_memory_mb",
            "max_symbols",
            "max_triples",
            "seed_packs",
            "shared_partitions",
            // AkhConfig scalars
            "model",
            "reasoning_effort",
            "system_prompt",
            "is_main",
            "backend",
            "spec_decode",
            "related_workspaces",
            "path_mode",
            "openclaw_import",
            "extra_read_roots",
            "capture_transcripts",
        ]
    }

    /// Known `[table]` names at the root of `akh.toml`, each with its own
    /// known key set (used to flag typos *inside* a table too).
    fn known_tables() -> &'static [(&'static str, &'static [&'static str])] {
        &[
            ("ollama_endpoint", &["kind", "base_url", "api_key"]),
            ("autonomy", &["enabled", "tasks"]),
            (
                "delegation",
                &[
                    "enabled",
                    "max_concurrent",
                    "daily_cost_budget_cents",
                    "per_task_max_cents",
                    "akhomed_url",
                    "local",
                    "remote",
                    "peers",
                ],
            ),
        ]
    }

    /// Diff a raw parsed TOML document against the known `akh.toml` schema,
    /// returning every key that isn't recognized. Deliberately conservative:
    /// unrecognized keys *inside* an unrecognized table are not further
    /// expanded (the table itself is reported once).
    fn unknown_keys(raw: &toml::Value) -> Vec<UnknownConfigKey> {
        let mut found = Vec::new();
        let Some(table) = raw.as_table() else {
            return found;
        };
        let known_top = Self::known_top_level_keys();
        let known_tables = Self::known_tables();

        for (key, value) in table {
            if known_top.contains(&key.as_str()) {
                continue;
            }
            if let Some((_, table_keys)) = known_tables.iter().find(|(name, _)| name == key) {
                // Recognized table: check its own keys for typos.
                if let Some(inner) = value.as_table() {
                    for inner_key in inner.keys() {
                        if !table_keys.contains(&inner_key.as_str()) {
                            found.push(UnknownConfigKey {
                                path: format!("{key}.{inner_key}"),
                                // The classic footgun: a nested key whose name
                                // matches a *top-level* field means a scalar
                                // was accidentally written under this table
                                // header instead of above it.
                                misplaced_under_table: known_top
                                    .contains(&inner_key.as_str())
                                    .then(|| key.clone()),
                            });
                        }
                    }
                }
                continue;
            }
            // Entirely unrecognized top-level key. If it's a table and any of
            // its inner keys collide with known top-level names, call that
            // out explicitly — it's the same footgun under an unknown table.
            if let Some(inner) = value.as_table() {
                let mut any_misplaced = false;
                for inner_key in inner.keys() {
                    if known_top.contains(&inner_key.as_str()) {
                        any_misplaced = true;
                        found.push(UnknownConfigKey {
                            path: format!("{key}.{inner_key}"),
                            misplaced_under_table: Some(key.clone()),
                        });
                    }
                }
                if !any_misplaced {
                    found.push(UnknownConfigKey {
                        path: key.clone(),
                        misplaced_under_table: None,
                    });
                }
            } else {
                found.push(UnknownConfigKey {
                    path: key.clone(),
                    misplaced_under_table: None,
                });
            }
        }
        found
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
        })?;
        restrict_to_owner(path);
        Ok(())
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
        ollama.reasoning_effort = self.reasoning_effort.clone();
        ModelManagerConfig {
            ollama,
            system_prompt: self.system_prompt.clone(),
            ..Default::default()
        }
    }
}

/// Restrict a just-written config file to owner-only read/write (`0600`) on
/// Unix. `akh.toml` can carry an inline cloud `api_key`, so a world-readable
/// default (`0644`) leaks it to every local user. Best-effort: a failure here
/// is logged, not propagated — the file is already written and usable, and
/// surfacing a permission-fixup failure as a save error would be surprising.
/// No-op on non-Unix targets.
fn restrict_to_owner(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            tracing::warn!(
                target: "akh_config",
                path = %path.display(),
                error = %e,
                "failed to restrict akh.toml permissions to 0600 (it may contain secrets)"
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
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
        assert_eq!(cfg.backend, ChatBacking::Ollama, "Ollama is the default backing");
    }

    #[test]
    fn backend_selection_parses_and_defaults() {
        // Absent → Ollama (back-compat: existing akh.toml keeps its behaviour).
        let legacy: AkhConfig = toml::from_str("name = \"a\"").unwrap();
        assert_eq!(legacy.backend, ChatBacking::Ollama);
        // Explicit candle backing (ADR 070).
        let candle: AkhConfig =
            toml::from_str("name = \"a\"\nbackend = \"candle\"").unwrap();
        assert_eq!(candle.backend, ChatBacking::Candle);
        // Round-trips through serialization.
        let s = toml::to_string(&candle).unwrap();
        let back: AkhConfig = toml::from_str(&s).unwrap();
        assert_eq!(back.backend, ChatBacking::Candle);
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
    fn reasoning_effort_threads_into_model_manager_config() {
        let mut cfg = AkhConfig::with_name("thoth");
        cfg.reasoning_effort = Some("medium".into());
        let mm = cfg.to_model_manager_config();
        assert_eq!(mm.ollama.reasoning_effort.as_deref(), Some("medium"));

        // None ⇒ the OllamaConfig field stays None (model default).
        let bare = AkhConfig::with_name("ptah");
        assert!(bare.to_model_manager_config().ollama.reasoning_effort.is_none());
    }

    #[test]
    fn reasoning_effort_persists_as_top_level_scalar() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let mut cfg = AkhConfig::with_name("thoth");
        cfg.model = Some("gpt-oss:120b-cloud".into());
        cfg.reasoning_effort = Some("high".into());
        cfg.save(&path).expect("save");

        let content = std::fs::read_to_string(&path).expect("read");
        // The scalar must precede the [ollama_endpoint] table for valid TOML.
        let effort_at = content.find("reasoning_effort").expect("effort present");
        let table_at = content.find("[ollama_endpoint]").expect("table present");
        assert!(effort_at < table_at, "reasoning_effort must serialize before the table:\n{content}");

        let loaded = AkhConfig::load(&path).expect("load");
        assert_eq!(loaded.reasoning_effort.as_deref(), Some("high"));
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

    #[test]
    fn capture_transcripts_serializes_before_first_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let cfg = AkhConfig::with_name("thoth");
        cfg.save(&path).expect("save");
        let content = std::fs::read_to_string(&path).expect("read");
        let scalar_at = content.find("capture_transcripts").expect("scalar present");
        // Find the first *table header* line (starts with `[`, not an array
        // literal like `seed_packs = [...]`).
        let table_at = content
            .lines()
            .find(|line| line.starts_with('['))
            .and_then(|line| content.find(line))
            .expect("a table header present");
        assert!(
            scalar_at < table_at,
            "capture_transcripts must serialize before the first [table]:\n{content}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn saved_config_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let cfg = AkhConfig::with_name("thoth");
        cfg.save(&path).expect("save");
        let mode = std::fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "akh.toml may contain secrets and must be 0600, got {mode:o}");
    }

    #[test]
    fn unknown_top_level_key_is_reported() {
        let toml_src = "name = \"a\"\nmodle = \"typo\"\n";
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        let unknown = AkhConfig::unknown_keys(&raw);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "modle");
        assert!(unknown[0].misplaced_under_table.is_none());
    }

    #[test]
    fn unknown_key_inside_known_table_is_reported() {
        let toml_src = "name = \"a\"\n[ollama_endpoint]\nkidn = \"cloud\"\n";
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        let unknown = AkhConfig::unknown_keys(&raw);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "ollama_endpoint.kidn");
    }

    #[test]
    fn scalar_misplaced_under_table_is_flagged_specifically() {
        // The classic footgun: `model` belongs at the top level, but a table
        // header above it in the file causes TOML to nest it underneath.
        let toml_src = "name = \"a\"\n[ollama_endpoint]\nmodel = \"qwen2.5:7b\"\n";
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        let unknown = AkhConfig::unknown_keys(&raw);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "ollama_endpoint.model");
        assert_eq!(unknown[0].misplaced_under_table.as_deref(), Some("ollama_endpoint"));
        let msg = unknown[0].to_string();
        assert!(msg.contains("top-level"), "message should call out the footgun: {msg}");
    }

    #[test]
    fn no_unknown_keys_for_full_valid_config() {
        let toml_src = r#"
            name = "ptah"
            model = "qwen2.5:7b"
            capture_transcripts = false

            [ollama_endpoint]
            kind = "cloud"
            base_url = "https://ollama.com"
            api_key = "env:OLLAMA_API_KEY"

            [autonomy]
            enabled = true
            tasks = { consolidation = true }
        "#;
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        assert!(AkhConfig::unknown_keys(&raw).is_empty());
    }

    #[test]
    fn load_checked_surfaces_typo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        std::fs::write(&path, "name = \"a\"\nmodle = \"typo\"\n").expect("write");
        let (cfg, unknown) = AkhConfig::load_checked(&path).expect("load");
        assert_eq!(cfg.name(), "a");
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "modle");
    }
}
