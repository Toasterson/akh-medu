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
//! - **The explicit `system_prompt` composes (REVISED 2026-07-26).** When set it
//!   forms the **operator section** (persona/voice authority) that precedes the
//!   machine-owned sections (tool-operations + operating conventions). The
//!   explicit prompt no longer *replaces* the entire assembly — it composes,
//!   ensuring every akh receives tool documentation. Psyche remains the fallback
//!   when `system_prompt` is unset.
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
        help(
            "Ensure akh.toml exists and is readable, or omit it entirely to use name-only defaults."
        )
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
             `[ollama_endpoint]` needs `kind = \"local\"` or `kind = \"cloud\"`, \
             `path_mode` is `\"local\"` or `\"remote\"`, and `[autonomy.budget]` accepts \
             only `max_tokens_per_day` and `max_turns_per_day`."
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

    #[error("akh \"{akh}\" selects backend = \"openai\" but `{field}` is missing")]
    #[diagnostic(
        code(akh::akh_config::missing_openai_endpoint),
        help(
            "An OpenAI-backed akh needs both a pinned `model` and an \
             `[openai_endpoint]` table with at least `base_url` (version segment \
             included, e.g. \"https://inference.hetzner.com/api/v1\"). Two ways \
             forward: add the missing field, or switch the akh back to \
             `backend = \"ollama\"`. The daemon refuses rather than falling back, \
             because silently answering from a different model than the one \
             configured would invalidate any measurement taken against this akh."
        )
    )]
    MissingOpenAiEndpoint { akh: String, field: &'static str },

    #[error("akh \"{akh}\" selects backend = \"grug\" but `model` is missing")]
    #[diagnostic(
        code(akh::akh_config::missing_grug_model),
        help(
            "Set `model` to a BabyLLM bundle directory containing active-model.json. \
             Relative paths resolve below the shared Akh-Medu models directory; \
             absolute paths are accepted for development runs."
        )
    )]
    MissingGrugModel { akh: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    McpServers(#[from] crate::agent::tools::mcp_config::McpServersConfigError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Board(#[from] crate::agent::djehuty::BoardConfigError),
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

/// Per-akh MCP bridge policy (client-pushed servers at ACP `session/new`).
///
/// A client-pushed MCP server (e.g. a t3 GUI session's `t3-code` server) is
/// an OPTIONAL capability the host offers, not a hard dependency of the akh
/// itself — the akh still has all its own tools. Default policy is therefore
/// graceful degradation: a server that fails discovery is skipped (with a
/// visible warning in the turn), not fatal to the session.
///
/// Also carries the akh's own **static** MCP servers (`[[mcp.servers]]`,
/// Phase 53i / ADR 113 decision 9): external MCP servers whose tools are
/// registered into the akh's persistent tool registry at agent init as
/// `mcp__<server>__<tool>`. See [`crate::agent::tools::mcp_config`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpConfig {
    /// Opt-in "unsafe" stance: fail the whole turn if any client-pushed MCP
    /// server does not come back from discovery, instead of continuing with
    /// the servers that did. Off by default. Intended for dispatch contexts
    /// (e.g. an Anima-driven akh) that genuinely cannot proceed without a
    /// specific pushed tool. `AKH_REQUIRE_PUSHED_MCP` ("1"/"0") overrides
    /// this at the ACP layer regardless of what is set here.
    #[serde(default)]
    pub require_pushed_mcp: bool,

    /// Static per-akh MCP servers (`[[mcp.servers]]`). Empty by default.
    /// Declared after the scalar above so it serializes after it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<crate::agent::tools::mcp_config::McpServerSpec>,
}

/// Per-akh memory hygiene (`[memory]`, Phase 53j, ADR 113).
///
/// ```toml
/// [memory]
/// contradiction_policy = "warn"   # "warn" (default) | "replace" | "reject"
/// hygiene_batch = 1000            # max graph mutations per hygiene pass
/// ```
///
/// `contradiction_policy` sets what `add_triple` does with a detected
/// contradiction (warn and insert, replace the conflicting edge, or refuse
/// the write) and how the daemon's memory-hygiene pass repairs existing
/// contradictions on functional predicates. Predicates a schema pack
/// declares `single_valued` are never contradictions: the newest value
/// supersedes the older ones under every policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Contradiction policy for this akh's engine and hygiene repair.
    #[serde(default)]
    pub contradiction_policy: crate::engine::ContradictionPolicy,
    /// Upper bound on graph mutations per memory-hygiene pass. `0` makes the
    /// pass report-only.
    #[serde(default = "default_hygiene_batch")]
    pub hygiene_batch: usize,
}

fn default_hygiene_batch() -> usize {
    crate::schema::DEFAULT_HYGIENE_BATCH
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            contradiction_policy: crate::engine::ContradictionPolicy::default(),
            hygiene_batch: default_hygiene_batch(),
        }
    }
}

impl MemoryConfig {
    /// Whether every field is at its default (the table is then omitted
    /// when the config is saved).
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Per-akh background-automation gating (P5; defaults per ADR 113 decision 6).
///
/// `enabled` is the master switch and defaults to **on**: every akh runs its
/// background tasks, and LLM-cost tasks are bounded by `budget` (Phase 53b).
/// `tasks` provides per-task overrides keyed by the daemon task name
/// ([`DaemonTask::as_str`](crate::agent::autonomy::DaemonTask::as_str), for
/// example `"consolidation"` or `"reflection"`).
///
/// ```toml
/// [autonomy]
/// enabled = true
/// tasks = { knowledge_elicitation = false }
///
/// [autonomy.budget]
/// max_tokens_per_day = 250000
/// max_turns_per_day = 40
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutonomyConfig {
    /// Master switch. When false, only always-on infrastructure (session
    /// persistence, triggers, reactor schedules) and explicitly enabled tasks
    /// run. Defaults to true (ADR 113).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Per-task enable/disable overrides keyed by daemon task name. An explicit
    /// override **always wins** over the master switch — so a single task can be
    /// turned on with the master off (`enabled=false` + `goal_generation=true`),
    /// or turned off with the master on.
    #[serde(default)]
    pub tasks: std::collections::BTreeMap<String, bool>,
    /// Daily limits for LLM-cost background work (librarian, knowledge
    /// elicitation, knowledge extraction via an external API, reactor model
    /// turns). Defaults to 250 000 tokens and 40 turns per UTC day.
    #[serde(default)]
    pub budget: crate::agent::autonomy::AutonomyBudget,
    /// Which tools the idle OODA cycle may choose on its own. Defaults to
    /// `safe` (manifest `Safe`, not MCP-proxied). `all` takes effect only
    /// together with an explicit `tasks.idle_cycle = true`.
    #[serde(default)]
    pub idle_cycle_tools: crate::agent::autonomy::IdleCycleTools,
}

impl Default for AutonomyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tasks: std::collections::BTreeMap::new(),
            budget: crate::agent::autonomy::AutonomyBudget::default(),
            idle_cycle_tools: crate::agent::autonomy::IdleCycleTools::default(),
        }
    }
}

/// Which execution backend an akh's executor-facing tools
/// (`shell_exec`/`file_io`/`edit`/`glob`/`grep`) run through (ADR 082 / Phase
/// 48c wiring).
///
/// **Naming trap:** the existing top-level `backend` field is
/// [`ChatBacking`] — the *LLM* backing. This `exec` table is a different
/// concept (the *execution* [`crate::backend::Backend`] trait) and is named
/// `exec` precisely to avoid overloading `backend`.
///
/// Default is `InProcess` (today's behaviour — `/bin/sh -c` on the akhomed
/// host). `Remote` routes the same tools to a build/exec host over SSH via
/// [`crate::backend::remote::RemoteBackend`], which is what an arm64 akhomed
/// pod needs to run `cargo build` on a real compile host.
///
/// Example TOML:
/// ```toml
/// [exec]
/// kind = "remote"
/// host = "armini"
/// user = "toasty"
/// port = 22
/// key = "/etc/akh/ssh/id_ed25519"
/// known_hosts = "/etc/akh/ssh-hosts/known_hosts"
/// workdir = "/home/toasty/akh-workspaces"
/// ```
///
/// A missing `[exec]` section (or `kind = "in-process"`) reproduces today's
/// behaviour byte-for-byte — the workspace runs `InProcessBackend`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ExecBackendConfig {
    /// Default: run `shell_exec`/`file_io`/etc. in the akhomed process
    /// (`/bin/sh -c` on the daemon host). Byte-identical to pre-48c-wiring
    /// behaviour.
    InProcess,
    /// Run executor-facing tools on **this** host, but as a distinct, less
    /// privileged OS user ([`crate::backend::LocalDropBackend`], Phase 48b).
    ///
    /// This is the deployment for a system daemon: akhomed keeps the privilege
    /// it needs for its own job (binding ports, reading system-owned config and
    /// TLS material, surviving logout) while every command it executes has its
    /// uid, gid and supplementary groups replaced first. Without it, a
    /// root-owned akhomed hands root to every tool call.
    ///
    /// The user must already exist — akh-medu never provisions principals.
    ///
    /// ```toml
    /// [exec]
    /// kind = "local-drop"
    /// user = "toasty"
    /// workdir = "/Users/toasty/akh-workspaces"
    /// ```
    LocalDrop {
        /// Existing OS user every executed command runs as. Refused if it
        /// resolves to root, which would make the drop a no-op.
        user: String,
        /// Default cwd for executed commands and the advisory read+write root
        /// for the file verbs (ADR 082 §8: advisory only — the OS is the real
        /// boundary). Omit to inherit akhomed's cwd and filesystem sandbox.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workdir: Option<std::path::PathBuf>,
        /// Shell used to interpret command lines. Defaults to `/bin/sh`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shell: Option<std::path::PathBuf>,
        /// `PATH` for executed commands.
        ///
        /// Omit (the default) to derive one from the exec principal's home
        /// plus the platform's conventional prefixes — `~/.local/bin`,
        /// `~/.cargo/bin`, `~/go/bin`, Homebrew/pkgsrc, then the POSIX base.
        /// Set a literal value to pin it, or the sentinel `"inherit"` to hand
        /// down akhomed's own `PATH`, which under an init system is a minimal
        /// default with none of the exec user's toolchain on it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    /// Run executor-facing tools on a remote host over SSH
    /// (`crate::backend::remote::RemoteBackend`, Phase 48c / ADR 082).
    Remote {
        /// Target host (DNS name or address).
        host: String,
        /// Remote OS principal to authenticate as.
        user: String,
        /// Target port (default 22).
        #[serde(default = "default_remote_port")]
        port: u16,
        /// Path to an unencrypted OpenSSH-format private key (ADR 086 §1).
        key: std::path::PathBuf,
        /// Path to an OpenSSH `known_hosts` file for host-key verification
        /// (strict by default). Omit to fall back to the russh default
        /// (`~/.ssh/known_hosts`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        known_hosts: Option<std::path::PathBuf>,
        /// Where the remote workspace tree lives on the target host. Used as
        /// the advisory read+write root for the `Filesystem` verbs (ADR 082
        /// §8: advisory only — the OS is the sole security boundary).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workdir: Option<std::path::PathBuf>,
    },
    /// Register **no executor-facing tools at all** — not even in-process
    /// ones. `shell_exec`, `file_io`, `edit`, `glob`, `grep` simply do not
    /// exist in this akh's tool registry; the KG/memory/reasoning tools and
    /// the network tools (`http_fetch`, `web_search`) are unaffected.
    ///
    /// Intended for manager/backend-role akhs that only reason over the KG
    /// and delegate execution elsewhere (e.g. a cluster-resident thoth or
    /// gemma) — strictly stronger than an empty [`ExecBackendConfig::LocalDrop`],
    /// which still serves a shell under a dropped-privilege user. Container
    /// isolation is not tool absence: a sandboxed pod still hands out a full
    /// executor surface unless this kind says otherwise.
    ///
    /// ```toml
    /// [exec]
    /// kind = "none"
    /// ```
    None,
    /// Run executor-facing tools in a sandbox workspace that akhsbxd
    /// allocates and akhexecd operates
    /// (`crate::backend::execd::ExecdBackend`, Phase 53c / ADR 113 decision
    /// 3). This is the deployed-mode executor: no side effect runs in
    /// akhomed's address space.
    ///
    /// ```toml
    /// [exec]
    /// kind = "execd"
    /// sandbox_url = "https://akhsbxd.lab.internal"
    /// sandbox_token_file = "/run/akh/credentials/akhsbxd-token"
    /// profile = "rust-workspace"
    /// principal = "janus"
    /// # executor_url = "http://127.0.0.1:8413"      # default: akhsbxd gateway
    /// # executor_token_file = "/run/akh/credentials/akhexecd-token"
    /// # poll_wait_ms = 5000
    /// # allow_insecure_http = false  # plain http only to loopback unless true
    /// ```
    Execd {
        /// akhsbxd base URL.
        sandbox_url: String,
        /// Bearer token file for akhsbxd (principal with the `session` role).
        sandbox_token_file: std::path::PathBuf,
        /// akhexecd base URL for direct operations. Omit to use akhsbxd's
        /// `/v1/executors/{sandbox}/…` gateway (the default; the token then
        /// also needs the `executor_proxy` role at akhsbxd).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        executor_url: Option<String>,
        /// Bearer token file for akhexecd operations (role `session` there).
        /// Omit to reuse `sandbox_token_file`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        executor_token_file: Option<std::path::PathBuf>,
        /// akhsbxd workspace profile to allocate from.
        profile: String,
        /// Principal the allocation is made for.
        principal: String,
        /// Server-side wait per `exec_poll`, 1..=10000 ms.
        #[serde(default = "default_execd_poll_wait_ms")]
        poll_wait_ms: u64,
        /// Accept plain `http://` URLs to non-loopback hosts. The bearer
        /// tokens then cross the link in clear text; set it only for a
        /// trusted private link. Loopback (`localhost`, `127.0.0.1`, `[::1]`)
        /// never needs it.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        allow_insecure_http: bool,
    },
}

impl ExecBackendConfig {
    /// Translate `kind = "local-drop"` into the config
    /// [`crate::backend::LocalDropBackend::new`] takes. `None` for other kinds.
    /// Map the TOML `path` string onto the typed policy.
    ///
    /// One sentinel, `"inherit"`, selects akhomed's own `PATH`; anything else
    /// is taken literally. A literal `PATH` is a colon-joined list of absolute
    /// directories, so `"inherit"` can never collide with a real value.
    fn parse_exec_path(raw: &str) -> crate::backend::ExecPath {
        if raw.trim().eq_ignore_ascii_case("inherit") {
            crate::backend::ExecPath::Inherit
        } else {
            crate::backend::ExecPath::Explicit(raw.into())
        }
    }

    pub fn to_local_drop_config(&self) -> Option<crate::backend::LocalDropConfig> {
        let Self::LocalDrop {
            user,
            workdir,
            shell,
            path,
        } = self
        else {
            return None;
        };
        let mut cfg = crate::backend::LocalDropConfig::new(user.clone());
        if let Some(dir) = workdir {
            cfg = cfg.with_workdir(dir.clone());
        }
        if let Some(sh) = shell {
            cfg = cfg.with_shell(sh.clone());
        }
        if let Some(p) = path {
            cfg = cfg.with_path(Self::parse_exec_path(p));
        }
        Some(cfg)
    }

    /// Translate `kind = "remote"` into the config
    /// [`crate::backend::remote::RemoteBackend::connect`] takes. `None` for
    /// other kinds.
    ///
    /// `known_hosts = None` (the default) maps to `HostKeyPolicy::KnownHosts`
    /// with the russh default path (`~/.ssh/known_hosts`); a supplied path
    /// pins it. The remote `workdir` becomes the advisory read+write root
    /// (ADR 082 §8: advisory only — the OS is the real boundary).
    #[cfg(feature = "remote-executor")]
    pub fn to_remote_backend_config(&self) -> Option<crate::backend::remote::RemoteBackendConfig> {
        use crate::backend::remote::{HostKeyPolicy, RemoteBackendConfig};
        let Self::Remote {
            host,
            user,
            port,
            key,
            known_hosts,
            workdir,
        } = self
        else {
            return None;
        };
        let mut cfg = RemoteBackendConfig::new(host.clone(), user.clone(), key.clone())
            .with_port(*port)
            .with_host_key_policy(match known_hosts {
                Some(p) => HostKeyPolicy::KnownHosts {
                    path: Some(p.clone()),
                },
                None => HostKeyPolicy::KnownHosts { path: None },
            });
        if let Some(dir) = workdir {
            cfg = cfg.with_roots(vec![dir.clone()], vec![dir.clone()]);
        }
        Some(cfg)
    }

    /// Translate `kind = "execd"` into the config
    /// [`crate::backend::execd::ExecdBackend::connect`] takes, minting a fresh
    /// allocation session for `workspace`. `None` for other kinds.
    #[cfg(feature = "execd-executor")]
    pub fn to_execd_config(&self, workspace: &str) -> Option<crate::backend::execd::ExecdConfig> {
        use crate::backend::execd::{ExecdConfig, ExecutorRoute};
        let Self::Execd {
            sandbox_url,
            sandbox_token_file,
            executor_url,
            executor_token_file,
            profile,
            principal,
            poll_wait_ms,
            allow_insecure_http,
        } = self
        else {
            return None;
        };
        let token_file = executor_token_file
            .clone()
            .unwrap_or_else(|| sandbox_token_file.clone());
        let route = match executor_url {
            Some(url) => ExecutorRoute::Direct {
                url: url.clone(),
                token_file,
            },
            None => ExecutorRoute::ViaSandbox { token_file },
        };
        Some(
            ExecdConfig::new(
                sandbox_url.clone(),
                sandbox_token_file.clone(),
                profile.clone(),
                principal.clone(),
                workspace,
            )
            .with_executor(route)
            .with_poll_wait(std::time::Duration::from_millis(*poll_wait_ms))
            .with_insecure_http(*allow_insecure_http),
        )
    }

    /// Short, stable name of the variant for logs and diagnostics.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::InProcess => "in-process",
            Self::LocalDrop { .. } => "local-drop",
            Self::Remote { .. } => "remote",
            Self::None => "none",
            Self::Execd { .. } => "execd",
        }
    }
}

fn default_remote_port() -> u16 {
    22
}

fn default_execd_poll_wait_ms() -> u64 {
    5_000
}

impl Default for ExecBackendConfig {
    fn default() -> Self {
        Self::InProcess
    }
}

impl AutonomyConfig {
    /// Whether a named daemon task should run. An explicit per-task override
    /// takes precedence; otherwise the task follows the master switch.
    ///
    /// This is the raw config rule. The daemon gates with
    /// [`AutonomyConfig::gate`], which also applies the idle-cycle executor
    /// default and the always-on infrastructure tasks.
    pub fn task_enabled(&self, task: &str) -> bool {
        match self.tasks.get(task) {
            Some(&v) => v,
            None => self.enabled,
        }
    }

    /// Resolve whether `task` runs for an akh whose effective executor is
    /// `exec`.
    ///
    /// Order: always-on infrastructure runs; an explicit `tasks` override
    /// wins; a task that follows another one (harvest follows the sleep
    /// cycle) takes that task's gate; the master switch applies, except to
    /// tasks that were only ever env-gated (librarian, OpenClaw sync); the
    /// idle OODA cycle then defaults off while executor tools run in-process
    /// (ADR 113 decision 4).
    pub fn gate(
        &self,
        task: crate::agent::autonomy::DaemonTask,
        exec: crate::agent::autonomy::ExecPlacement,
    ) -> crate::agent::autonomy::TaskGate {
        use crate::agent::autonomy::{
            DaemonTask, DisabledReason, ExecPlacement, TaskGate, TaskGating,
        };
        let gating = task.gating();
        if gating == TaskGating::AlwaysOn {
            return TaskGate::Enabled;
        }
        match (self.tasks.get(task.as_str()), gating) {
            (Some(true), _) => TaskGate::Enabled,
            (Some(false), _) => TaskGate::Disabled(DisabledReason::ExplicitlyDisabled),
            (None, TaskGating::FollowsTask(leader)) => match self.gate(leader, exec) {
                TaskGate::Enabled => TaskGate::Enabled,
                TaskGate::Disabled(reason) => TaskGate::Disabled(DisabledReason::FollowsTask {
                    task: leader,
                    reason: Box::new(reason),
                }),
            },
            (None, TaskGating::IgnoresMasterSwitch) => TaskGate::Enabled,
            (None, TaskGating::OptIn) => TaskGate::Disabled(DisabledReason::OptIn),
            (None, _) if !self.enabled => TaskGate::Disabled(DisabledReason::MasterSwitchOff),
            (None, _) if task == DaemonTask::IdleCycle && exec == ExecPlacement::InProcess => {
                TaskGate::Disabled(DisabledReason::InProcessExecutor)
            }
            (None, _) => TaskGate::Enabled,
        }
    }

    /// The tool policy the idle OODA cycle runs with: safe-only unless
    /// `idle_cycle_tools = "all"` *and* `tasks.idle_cycle = true` are both
    /// set.
    pub fn idle_cycle_candidate_policy(&self) -> crate::agent::ooda::ToolCandidatePolicy {
        self.idle_cycle_tools
            .candidate_policy(self.tasks.get("idle_cycle") == Some(&true))
    }

    /// `tasks` keys that name no [`DaemonTask`](crate::agent::autonomy::DaemonTask).
    /// Such a key has no effect; the daemon warns about each one at start.
    pub fn unknown_task_names(&self) -> Vec<&str> {
        self.tasks
            .keys()
            .map(String::as_str)
            .filter(|name| name.parse::<crate::agent::autonomy::DaemonTask>().is_err())
            .collect()
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
/// function-calling). `Candle` runs an in-process GGUF model. `Grug` runs the
/// compact, from-scratch BabyLLM bundle trained against Akh's tool contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatBacking {
    /// Ollama, local or cloud (the default).
    #[default]
    Ollama,
    /// In-process Candle GGUF model.
    Candle,
    /// Any host serving the OpenAI `/chat/completions` contract — vLLM, SGLang,
    /// llama.cpp's server, OpenRouter, DeepSeek, Hetzner's inference experiment.
    /// Configured by the `[openai_endpoint]` table, which is **required** for
    /// this backing (an akh that selects it without one is a config error).
    #[serde(rename = "openai")]
    OpenAi,
    /// A local Codex app-server session, with its own coding-agent loop.
    Codex,
    /// A local Claude Agent SDK session, with its own coding-agent loop.
    Claude,
    /// Akh-Medu's own compact Burn transformer bundle (`active-model.json`).
    Grug,
}

/// An OpenAI-compatible inference endpoint (`backend = "openai"`).
///
/// Deliberately a separate table from [`OllamaEndpoint`] rather than a third
/// `kind` on it: the two speak different wire protocols, and folding a non-Ollama
/// provider into a field named `ollama_endpoint` would mislead every future
/// reader. `base_url` must include the version segment.
///
/// ```toml
/// backend = "openai"
/// model = "DeepSeek-V4-Flash-0731"
///
/// [openai_endpoint]
/// base_url = "https://inference.hetzner.com/api/v1"
/// api_key = "env:HETZNER_INFERENCE_KEY"
/// context_window = 512000
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiEndpoint {
    /// API root **including** the version segment (e.g. `.../api/v1`). The
    /// client appends `/chat/completions` and `/models`.
    pub base_url: String,

    /// Bearer token. Supports `env:VAR` (resolved at call time) so the secret
    /// need never be written into this file; a literal token also works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Context window in tokens. The OpenAI contract exposes no way to discover
    /// this (unlike Ollama's `/api/show`), so the Phase-44 context budget needs
    /// it stated here. `None` leaves the budget on its own default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,

    /// Per-request timeout. Agentic coding turns are long-running; defaults to
    /// 600 s.
    #[serde(default = "default_openai_timeout_secs")]
    pub timeout_secs: u64,

    /// Cap on generated tokens per turn. `None` ⇒ the server's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature. `None` ⇒ the server's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

fn default_openai_timeout_secs() -> u64 {
    600
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

    /// Per-workspace reasoning-effort hint: `"low"`, `"medium"`, `"high"`, or
    /// `"max"` (Ollama's `/api/chat` `think` string enum — see
    /// [`agent::ollama::VALID_THINK_LEVELS`](super::agent::ollama::VALID_THINK_LEVELS)).
    /// `None` (the default, and what every shipped workspace TOML — including
    /// ptah's — should stay on unless deliberately tuned) omits the `think`
    /// field entirely, leaving Ollama's own default in effect. Carried into
    /// [`OllamaConfig::reasoning_effort`](super::agent::ollama::OllamaConfig::reasoning_effort)
    /// by [`Self::to_model_manager_config`] and sent as `think` on the chat
    /// path; validated there
    /// ([`OllamaConfig::validated_think_level`](super::agent::ollama::OllamaConfig::validated_think_level))
    /// against the same enum, so a value that only makes sense under one
    /// renderer (e.g. Ollama's reasoning cloud models, or the dedicated
    /// Qwen3.8 renderer) never gets forwarded blindly. Only meaningful for
    /// models whose renderer actually implements a reasoning-effort concept
    /// (gpt-oss/glm's native support, or Qwen3.8's dedicated `RENDERER
    /// qwen3.8` — the generic `qwen3.5` renderer some Qwen3.8 GGUFs ship with
    /// by default has no reasoning-effort concept at all, so this is a no-op
    /// there). Serialized as a top-level scalar (before the
    /// `[ollama_endpoint]` table).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Authoritative system prompt. When `Some`, it **replaces** the
    /// psyche-derived prompt; when `None`, the psyche-derived prompt is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Whether this is the akh's MAIN memory workspace (writes target MAIN).
    #[serde(default = "default_true")]
    pub is_main: bool,

    /// Which chat/agentic backing this akh uses: `"ollama"` (default),
    /// `"candle"`, `"openai"`, or `"grug"` (Akh-Medu's compact BabyLLM).
    /// Serialized as a top-level scalar.
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

    /// Which Phase-44 context-assembly strategy this akh runs:
    /// `"compacting"` (the daemon default), `"passive"`, `"virtual"` (MemGPT
    /// tiers + self-edit tools), or `"passthrough"` (the A/B control).
    ///
    /// Per-akh because the strategies are meant to be *compared*: the daemon
    /// hosts several akhs in one process, so a single `AKH_CONTEXT_STRATEGY`
    /// env var could only ever move all of them together, leaving no control
    /// group. Precedence is: this field → `AKH_CONTEXT_STRATEGY` → `compacting`.
    ///
    /// Note this is also the **tier gate** (ADR 081 slice 7 decision 4): only
    /// akhs on `virtual` receive the memory self-edit tools, so setting it is a
    /// statement about the model's capability, not just its bookkeeping.
    ///
    /// Kept above the first `[table]` header so it serializes as a top-level
    /// scalar (see the TOML nesting footgun documented on `capture_transcripts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_strategy: Option<String>,

    // ── tables (must be serialized after all values for TOML) ───────────────
    /// Where the pinned model runs (local Ollama or cloud).
    #[serde(default)]
    pub ollama_endpoint: OllamaEndpoint,

    /// The OpenAI-compatible endpoint, required when `backend = "openai"` and
    /// ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_endpoint: Option<OpenAiEndpoint>,

    /// Local coding-agent runtime, required for `codex` and `claude`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_runtime: Option<crate::agent::runtime::RuntimeConfig>,

    /// Background-automation gating (P5). Defaults to enabled with a daily
    /// LLM budget (ADR 113 decision 6).
    #[serde(default)]
    pub autonomy: AutonomyConfig,

    /// MCP bridge policy for client-pushed servers (`session/new`). Defaults
    /// to graceful degradation (`require_pushed_mcp = false`).
    #[serde(default)]
    pub mcp: McpConfig,

    /// This akh's own Agent Board identity (Phase 53): URL + token source (+
    /// own agent id for inbox ingest). Reactor run reports, `DjehutyReply` and
    /// board ingress use it; `None` falls back to the process-wide
    /// `DJEHUTY_BOARD_*` environment (one identity for every akh on the host).
    ///
    /// ```toml
    /// [board]
    /// url = "https://board.lab.home"
    /// auth = { token_env = "ROMAN_BOARD_JANUS" }
    /// agent_id = "019d…"
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<crate::agent::djehuty::BoardConfig>,

    /// Memory hygiene: contradiction policy and hygiene batch (Phase 53j).
    #[serde(default, skip_serializing_if = "MemoryConfig::is_default")]
    pub memory: MemoryConfig,

    /// Provider-failure handling for the akh's own model turns (Phase 53g
    /// PR 4): the cooldown ladder and the exact structured error codes that
    /// name authentication, quota and rate-limit failures. Defaults: ladder
    /// 30 min / 1 h / 2 h / 4 h, empty code lists (classify by HTTP status).
    ///
    /// ```toml
    /// [provider]
    /// cooldown_ladder_secs = [1800, 3600, 7200, 14400]
    /// quota_error_codes = ["insufficient_quota"]
    /// auth_error_codes = ["invalid_api_key"]
    /// rate_limit_error_codes = ["rate_limit_exceeded"]
    /// ```
    #[serde(
        default,
        skip_serializing_if = "crate::agent::provider_failure::ProviderConfig::is_default"
    )]
    pub provider: crate::agent::provider_failure::ProviderConfig,

    /// Delegation / external scheduling (Phase 36). Parsed now; the
    /// `DelegationManager` is wired behind the `delegation` feature in its phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegation: Option<DelegationConfig>,

    /// Per-workspace execution backend **override** (ADR 082 / Phase 48c
    /// wiring). Named `exec` (not `backend`) to avoid the LLM-`backend`
    /// (`ChatBacking`) naming collision — see [`ExecBackendConfig`].
    ///
    /// `None` — the common and recommended case — defers to the **daemon's**
    /// host-level `[exec]` (`AkhomedConfig::exec`), falling back to
    /// `InProcess`. Executors are host properties: which OS users exist, which
    /// workdir is on this disk, which build host is reachable. An akh's config
    /// travels with the akh when it relocates (ADR 049), so an executor pinned
    /// here follows it to a machine where that user or path may not exist —
    /// or worse, silently loads with no drop at all on a host that needs one.
    /// Set this only when one akh on a host genuinely needs a different
    /// executor than the host default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec: Option<ExecBackendConfig>,
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
            context_strategy: None,
            ollama_endpoint: OllamaEndpoint::default(),
            openai_endpoint: None,
            agent_runtime: None,
            autonomy: AutonomyConfig::default(),
            mcp: McpConfig::default(),
            board: None,
            memory: MemoryConfig::default(),
            provider: crate::agent::provider_failure::ProviderConfig::default(),
            delegation: None,
            exec: None,
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
        let mut roots: Vec<std::path::PathBuf> = self
            .extra_read_roots
            .iter()
            .map(|p| expand_home(p))
            .collect();
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
    pub fn load_checked(path: &std::path::Path) -> AkhConfigResult<(Self, Vec<UnknownConfigKey>)> {
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
        crate::agent::tools::mcp_config::validate_mcp_servers(&cfg.mcp.servers)?;
        if let Some(board) = &cfg.board {
            board.validate(cfg.name())?;
        }
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
            // Wave R3 power-state switch (`WorkspaceConfig::enabled`); always
            // serialized, so omitting it here warned on every saved config.
            "enabled",
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
            "context_strategy",
        ]
    }

    /// Known `[table]` names at the root of `akh.toml`, each with its own
    /// known key set (used to flag typos *inside* a table too).
    fn known_tables() -> &'static [(&'static str, &'static [&'static str])] {
        &[
            ("ollama_endpoint", &["kind", "base_url", "api_key"]),
            // `[[mcp.servers]]` entries are checked by serde itself
            // (`deny_unknown_fields` on `McpServerSpec`).
            ("mcp", &["require_pushed_mcp", "servers"]),
            // `auth` is checked by serde (`deny_unknown_fields` on `BoardAuth`).
            ("board", &["url", "auth", "agent_id", "allow_insecure_http"]),
            (
                "agent_runtime",
                &[
                    "working_dir",
                    "binary",
                    "node_binary",
                    "bridge_script",
                    "timeout_secs",
                ],
            ),
            (
                "openai_endpoint",
                &[
                    "base_url",
                    "api_key",
                    "context_window",
                    "timeout_secs",
                    "max_tokens",
                    "temperature",
                ],
            ),
            (
                "autonomy",
                &["enabled", "tasks", "budget", "idle_cycle_tools"],
            ),
            ("memory", &["contradiction_policy", "hygiene_batch"]),
            (
                "provider",
                &[
                    "cooldown_ladder_secs",
                    "auth_error_codes",
                    "quota_error_codes",
                    "rate_limit_error_codes",
                ],
            ),
            (
                "delegation",
                &[
                    "enabled",
                    "max_concurrent",
                    "daily_cost_budget_cents",
                    "per_task_max_cents",
                    "akhomed_url",
                    "native",
                    "remote",
                    "peers",
                ],
            ),
            (
                "exec",
                &[
                    "kind",
                    "host",
                    "user",
                    "port",
                    "key",
                    "known_hosts",
                    "workdir",
                    "shell",
                    "path",
                    "sandbox_url",
                    "sandbox_token_file",
                    "executor_url",
                    "executor_token_file",
                    "profile",
                    "principal",
                    "poll_wait_ms",
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

    /// Whether this akh's `[exec]` config selects a remote SSH executor. The
    /// daemon uses this to decide whether to construct a `RemoteBackend` and
    /// pass it as the agent's `backend_override` (Phase 48c wiring).
    pub fn is_remote_exec(&self) -> bool {
        matches!(self.exec, Some(ExecBackendConfig::Remote { .. }))
    }

    /// Whether this akh's `[exec]` override selects the privilege-dropping
    /// local executor (Phase 48b).
    pub fn is_local_drop_exec(&self) -> bool {
        matches!(self.exec, Some(ExecBackendConfig::LocalDrop { .. }))
    }

    /// The executor this akh actually runs under on this host: the per-akh
    /// override when present, otherwise the daemon's host-level default,
    /// otherwise `InProcess`.
    ///
    /// The daemon resolves through this single choke point so the precedence
    /// cannot fork between the lazy-load path and the auto-start path — the
    /// lazy path once constructed an executor straight from the per-akh field
    /// and thereby loaded a relocated akh with no privilege drop on a root
    /// daemon.
    pub fn effective_exec(&self, host_default: Option<&ExecBackendConfig>) -> ExecBackendConfig {
        self.exec
            .clone()
            .or_else(|| host_default.cloned())
            .unwrap_or_default()
    }

    /// Translate a per-akh `[exec]` override into the `LocalDropConfig` the
    /// daemon hands to `LocalDropBackend::new`. Returns `None` unless this
    /// akh explicitly sets `kind = "local-drop"` — host defaults resolve via
    /// [`AkhConfig::effective_exec`] + [`ExecBackendConfig::to_local_drop_config`].
    pub fn to_local_drop_config(&self) -> Option<crate::backend::LocalDropConfig> {
        self.exec
            .as_ref()
            .and_then(ExecBackendConfig::to_local_drop_config)
    }

    /// Translate a per-akh `[exec]` override into the `RemoteBackendConfig`
    /// the daemon hands to `RemoteBackend::connect`. Returns `None` unless
    /// this akh explicitly sets `kind = "remote"`.
    #[cfg(feature = "remote-executor")]
    pub fn to_remote_backend_config(&self) -> Option<crate::backend::remote::RemoteBackendConfig> {
        self.exec
            .as_ref()
            .and_then(ExecBackendConfig::to_remote_backend_config)
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

    /// Build the [`OpenAiConfig`](crate::agent::openai_backend::OpenAiConfig)
    /// for an `backend = "openai"` akh.
    ///
    /// Returns `Err` with a diagnosable reason when the akh selects the OpenAI
    /// backing but is not actually configured for it, so the daemon can refuse
    /// loudly instead of silently falling back to a *different model than the
    /// operator asked for* — a fallback that would quietly invalidate any
    /// benchmark run against this akh.
    ///
    /// Callers that merely want to know whether this akh is OpenAI-backed
    /// should compare [`Self::backend`] against [`ChatBacking::OpenAi`]; this
    /// returns `Ok(None)` for every other backing.
    pub fn to_openai_config(
        &self,
    ) -> Result<Option<crate::agent::openai_backend::OpenAiConfig>, AkhConfigError> {
        if self.backend != ChatBacking::OpenAi {
            return Ok(None);
        }
        let endpoint =
            self.openai_endpoint
                .as_ref()
                .ok_or_else(|| AkhConfigError::MissingOpenAiEndpoint {
                    akh: self.workspace.name.clone(),
                    field: "openai_endpoint",
                })?;
        let model = self
            .model
            .clone()
            .ok_or_else(|| AkhConfigError::MissingOpenAiEndpoint {
                akh: self.workspace.name.clone(),
                field: "model",
            })?;

        Ok(Some(crate::agent::openai_backend::OpenAiConfig {
            base_url: endpoint.base_url.clone(),
            api_key: endpoint.api_key.clone(),
            model,
            timeout_secs: endpoint.timeout_secs,
            context_window: endpoint.context_window,
            max_tokens: endpoint.max_tokens,
            temperature: endpoint.temperature,
            reasoning_effort: self.reasoning_effort.clone(),
        }))
    }

    /// Resolve `model` for a `backend = "grug"` akh. Relative bundle paths
    /// live below the shared models directory; an explicit absolute path is
    /// useful for a run still under evaluation and is never rewritten.
    pub fn to_grug_model_path(
        &self,
        models_dir: &std::path::Path,
    ) -> Result<Option<std::path::PathBuf>, AkhConfigError> {
        if self.backend != ChatBacking::Grug {
            return Ok(None);
        }
        let configured = self
            .model
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| AkhConfigError::MissingGrugModel {
                akh: self.workspace.name.clone(),
            })?;
        let configured = std::path::PathBuf::from(configured);
        Ok(Some(if configured.is_absolute() {
            configured
        } else {
            models_dir.join(configured)
        }))
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
        assert!(
            cfg.autonomy.enabled,
            "automation is on by default (ADR 113 decision 6)"
        );
        assert_eq!(
            cfg.autonomy.budget,
            crate::agent::autonomy::AutonomyBudget {
                max_tokens_per_day: 250_000,
                max_turns_per_day: 40,
            }
        );
        assert_eq!(cfg.ollama_endpoint, OllamaEndpoint::default());
        assert!(cfg.model.is_none());
        assert!(cfg.system_prompt.is_none());
        assert_eq!(
            cfg.backend,
            ChatBacking::Ollama,
            "Ollama is the default backing"
        );
    }

    #[test]
    fn backend_selection_parses_and_defaults() {
        // Absent → Ollama (back-compat: existing akh.toml keeps its behaviour).
        let legacy: AkhConfig = toml::from_str("name = \"a\"").unwrap();
        assert_eq!(legacy.backend, ChatBacking::Ollama);
        // Explicit candle backing (ADR 070).
        let candle: AkhConfig = toml::from_str("name = \"a\"\nbackend = \"candle\"").unwrap();
        assert_eq!(candle.backend, ChatBacking::Candle);
        // Round-trips through serialization.
        let s = toml::to_string(&candle).unwrap();
        let back: AkhConfig = toml::from_str(&s).unwrap();
        assert_eq!(back.backend, ChatBacking::Candle);

        let grug: AkhConfig =
            toml::from_str("name = \"a\"\nbackend = \"grug\"\nmodel = \"grug/akh-tool-small\"")
                .unwrap();
        assert_eq!(grug.backend, ChatBacking::Grug);
        assert_eq!(
            grug.to_grug_model_path(std::path::Path::new("/models"))
                .unwrap()
                .unwrap(),
            std::path::PathBuf::from("/models/grug/akh-tool-small")
        );
    }

    #[test]
    fn grug_backend_requires_a_model_bundle() {
        let cfg: AkhConfig = toml::from_str("name = \"a\"\nbackend = \"grug\"").unwrap();
        assert!(matches!(
            cfg.to_grug_model_path(std::path::Path::new("/models")),
            Err(AkhConfigError::MissingGrugModel { .. })
        ));
    }

    #[test]
    fn autonomy_task_enabled_override_wins() {
        let mut a = AutonomyConfig::default();
        // Master on by default (ADR 113); switch it off to test overrides.
        assert!(a.enabled);
        assert!(a.task_enabled("reflection"));
        a.enabled = false;
        // Master off → every unlisted task off.
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

    /// kg_compaction deletes every `episode:*` triple, so it never runs by
    /// default, not even with autonomy on; only an explicit opt-in enables it.
    #[test]
    fn kg_compaction_is_opt_in() {
        use crate::agent::autonomy::{DaemonTask, DisabledReason, ExecPlacement, TaskGate};
        let mut a = AutonomyConfig::default();
        assert!(a.enabled);
        for exec in [
            ExecPlacement::InProcess,
            ExecPlacement::Contained,
            ExecPlacement::Disabled,
        ] {
            assert_eq!(
                a.gate(DaemonTask::KgCompaction, exec),
                TaskGate::Disabled(DisabledReason::OptIn)
            );
        }
        a.tasks.insert("kg_compaction".into(), true);
        assert_eq!(
            a.gate(DaemonTask::KgCompaction, ExecPlacement::Disabled),
            TaskGate::Enabled
        );
        a.tasks.insert("kg_compaction".into(), false);
        assert_eq!(
            a.gate(DaemonTask::KgCompaction, ExecPlacement::Disabled),
            TaskGate::Disabled(DisabledReason::ExplicitlyDisabled)
        );
    }

    #[test]
    fn autonomy_gate_applies_decision_4_idle_cycle_default() {
        use crate::agent::autonomy::{DaemonTask, DisabledReason, ExecPlacement, TaskGate};
        let mut a = AutonomyConfig::default();
        // In-process exec: idle OODA tool use is off, everything else on.
        assert_eq!(
            a.gate(DaemonTask::IdleCycle, ExecPlacement::InProcess),
            TaskGate::Disabled(DisabledReason::InProcessExecutor)
        );
        assert_eq!(
            a.gate(DaemonTask::Reflection, ExecPlacement::InProcess),
            TaskGate::Enabled
        );
        // Contained or disabled exec: idle cycle on.
        for exec in [ExecPlacement::Contained, ExecPlacement::Disabled] {
            assert_eq!(a.gate(DaemonTask::IdleCycle, exec), TaskGate::Enabled);
        }
        // An explicit override wins over the executor default.
        a.tasks.insert("idle_cycle".into(), true);
        assert_eq!(
            a.gate(DaemonTask::IdleCycle, ExecPlacement::InProcess),
            TaskGate::Enabled
        );
        // Master off: gated tasks stop, infrastructure keeps running.
        a.enabled = false;
        assert_eq!(
            a.gate(DaemonTask::Reflection, ExecPlacement::Contained),
            TaskGate::Disabled(DisabledReason::MasterSwitchOff)
        );
        for infra in [
            DaemonTask::Persist,
            DaemonTask::TriggerEvaluation,
            DaemonTask::Reactor,
        ] {
            assert_eq!(a.gate(infra, ExecPlacement::InProcess), TaskGate::Enabled);
        }
        a.tasks.insert("librarian".into(), false);
        a.enabled = true;
        assert_eq!(
            a.gate(DaemonTask::Librarian, ExecPlacement::Contained),
            TaskGate::Disabled(DisabledReason::ExplicitlyDisabled)
        );
    }

    #[test]
    fn board_table_parses_is_known_and_round_trips() {
        let src = r#"
            name = "janus"
            model = "glm-5.3"
            [board]
            url = "https://board.lab.home"
            auth = { token_env = "ROMAN_BOARD_JANUS" }
            agent_id = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
        "#;
        let cfg: AkhConfig = toml::from_str(src).expect("parse");
        let board = cfg.board.as_ref().expect("[board] parsed");
        assert_eq!(
            board.auth,
            crate::agent::djehuty::BoardAuth::TokenEnv("ROMAN_BOARD_JANUS".into())
        );
        assert!(board.own_agent_id().is_some());
        board.validate("janus").expect("valid");
        let raw: toml::Value = toml::from_str(src).unwrap();
        assert!(
            AkhConfig::unknown_keys(&raw).is_empty(),
            "[board] keys are known"
        );

        // Saved configs keep `model` above the tables and re-load unchanged.
        let saved = toml::to_string(&cfg).expect("serialize");
        let reloaded: AkhConfig = toml::from_str(&saved).expect("reload");
        assert_eq!(reloaded.board, cfg.board);
        assert_eq!(reloaded.model.as_deref(), Some("glm-5.3"));

        // A typo inside [board] and an unknown auth kind are refused.
        assert!(
            toml::from_str::<AkhConfig>(
                "name = \"j\"\n[board]\nurl = \"https://b\"\nauth = { token_envv = \"X\" }\n"
            )
            .is_err()
        );
    }

    #[test]
    fn board_table_rejects_plain_http_and_relative_token_files() {
        let insecure: AkhConfig = toml::from_str(
            "name = \"j\"\n[board]\nurl = \"http://board.lab.home\"\nauth = { token_env = \"X\" }\n",
        )
        .unwrap();
        let err = insecure.board.unwrap().validate("j").unwrap_err();
        assert_eq!(
            miette::Diagnostic::code(&err).unwrap().to_string(),
            "akh::board::config::insecure_http"
        );
        let relative: AkhConfig = toml::from_str(
            "name = \"j\"\n[board]\nurl = \"http://127.0.0.1:8080\"\nauth = { token_file = \"b.key\" }\n",
        )
        .unwrap();
        let err = relative.board.unwrap().validate("j").unwrap_err();
        assert_eq!(
            miette::Diagnostic::code(&err).unwrap().to_string(),
            "akh::board::config::token_file"
        );
    }

    #[test]
    fn master_switch_off_keeps_pre_53b_harvest_librarian_and_openclaw_behaviour() {
        use crate::agent::autonomy::{DaemonTask, DisabledReason, ExecPlacement, TaskGate};
        let exec = ExecPlacement::Contained;
        // A pre-53b TOML: `[autonomy] enabled = false`.
        let mut a = AutonomyConfig {
            enabled: false,
            ..AutonomyConfig::default()
        };
        // Harvest used to run inside the sleep cycle: it follows its gate.
        assert_eq!(
            a.gate(DaemonTask::Harvest, exec),
            TaskGate::Disabled(DisabledReason::FollowsTask {
                task: DaemonTask::SleepCycle,
                reason: Box::new(DisabledReason::MasterSwitchOff),
            })
        );
        a.tasks.insert("sleep_cycle".into(), true);
        assert_eq!(a.gate(DaemonTask::Harvest, exec), TaskGate::Enabled);
        // An explicit harvest override still wins.
        a.tasks.insert("harvest".into(), false);
        assert_eq!(
            a.gate(DaemonTask::Harvest, exec),
            TaskGate::Disabled(DisabledReason::ExplicitlyDisabled)
        );
        // Librarian and OpenClaw sync were only env-gated: the master switch
        // does not turn them off (their env gates apply in the daemon).
        assert_eq!(a.gate(DaemonTask::Librarian, exec), TaskGate::Enabled);
        assert_eq!(a.gate(DaemonTask::OpenclawSync, exec), TaskGate::Enabled);
        a.tasks.insert("openclaw_sync".into(), false);
        assert_eq!(
            a.gate(DaemonTask::OpenclawSync, exec),
            TaskGate::Disabled(DisabledReason::ExplicitlyDisabled)
        );
        // Master on: harvest follows the (enabled) sleep cycle, or its own
        // sleep-cycle opt-out.
        let mut on = AutonomyConfig::default();
        assert_eq!(on.gate(DaemonTask::Harvest, exec), TaskGate::Enabled);
        on.tasks.insert("sleep_cycle".into(), false);
        assert!(matches!(
            on.gate(DaemonTask::Harvest, exec),
            TaskGate::Disabled(DisabledReason::FollowsTask { .. })
        ));
    }

    #[test]
    fn idle_cycle_tools_default_safe_and_all_needs_an_explicit_idle_cycle() {
        use crate::agent::autonomy::IdleCycleTools;
        use crate::agent::ooda::ToolCandidatePolicy;
        let cfg: AkhConfig = toml::from_str("name = \"a\"").unwrap();
        assert_eq!(cfg.autonomy.idle_cycle_tools, IdleCycleTools::Safe);
        assert_eq!(
            cfg.autonomy.idle_cycle_candidate_policy(),
            ToolCandidatePolicy::SafeNonMcp
        );
        // `all` alone is not enough: the cycle stays safe-only.
        let src = r#"
            name = "a"
            [autonomy]
            idle_cycle_tools = "all"
        "#;
        let cfg: AkhConfig = toml::from_str(src).unwrap();
        assert_eq!(cfg.autonomy.idle_cycle_tools, IdleCycleTools::All);
        assert_eq!(
            cfg.autonomy.idle_cycle_candidate_policy(),
            ToolCandidatePolicy::SafeNonMcp
        );
        let raw: toml::Value = toml::from_str(src).unwrap();
        assert!(AkhConfig::unknown_keys(&raw).is_empty());
        // `all` + explicit `tasks.idle_cycle = true` opts in.
        let cfg: AkhConfig = toml::from_str(
            r#"
            name = "a"
            [autonomy]
            idle_cycle_tools = "all"
            tasks = { idle_cycle = true }
        "#,
        )
        .unwrap();
        assert_eq!(
            cfg.autonomy.idle_cycle_candidate_policy(),
            ToolCandidatePolicy::Unrestricted
        );
        assert!(
            toml::from_str::<AkhConfig>("name = \"a\"\n[autonomy]\nidle_cycle_tools = \"most\"")
                .is_err(),
            "an unknown policy is refused"
        );
    }

    #[test]
    fn autonomy_budget_typos_are_refused() {
        let src = r#"
            name = "janus"
            [autonomy.budget]
            max_token_per_day = 1000
        "#;
        let error = toml::from_str::<AkhConfig>(src).unwrap_err().to_string();
        assert!(
            error.contains("max_token_per_day") && error.contains("max_tokens_per_day"),
            "the error names the typo and the valid keys: {error}"
        );
    }

    #[test]
    fn autonomy_budget_parses_and_is_a_known_table_key() {
        let src = r#"
            name = "janus"
            [autonomy]
            enabled = true
            tasks = { idle_cycle = false, seshat_curaton = true }
            [autonomy.budget]
            max_turns_per_day = 12
        "#;
        let cfg: AkhConfig = toml::from_str(src).expect("parse");
        assert_eq!(cfg.autonomy.budget.max_turns_per_day, 12);
        assert_eq!(
            cfg.autonomy.budget.max_tokens_per_day,
            crate::agent::autonomy::AutonomyBudget::DEFAULT_MAX_TOKENS_PER_DAY,
            "an omitted limit keeps its default"
        );
        assert_eq!(cfg.autonomy.unknown_task_names(), vec!["seshat_curaton"]);
        let raw: toml::Value = toml::from_str(src).unwrap();
        assert!(
            AkhConfig::unknown_keys(&raw).is_empty(),
            "`budget` is a known [autonomy] key"
        );
        // Round-trips through serialization.
        let back: AkhConfig = toml::from_str(&toml::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back.autonomy, cfg.autonomy);
    }

    #[test]
    fn memory_table_sets_the_contradiction_policy_per_akh() {
        use crate::engine::ContradictionPolicy;

        // Default: unchanged engine behaviour (warn), default batch.
        let cfg: AkhConfig = toml::from_str(r#"name = "vesta""#).unwrap();
        assert_eq!(cfg.memory, MemoryConfig::default());
        assert_eq!(cfg.memory.contradiction_policy, ContradictionPolicy::Warn);
        assert_eq!(
            cfg.memory.hygiene_batch,
            crate::schema::DEFAULT_HYGIENE_BATCH
        );
        assert!(
            !toml::to_string(&cfg).unwrap().contains("[memory]"),
            "a default [memory] is not written back"
        );

        for (value, policy) in [
            ("warn", ContradictionPolicy::Warn),
            ("replace", ContradictionPolicy::Replace),
            ("reject", ContradictionPolicy::Reject),
        ] {
            let src = format!(
                "name = \"vesta\"\n[memory]\ncontradiction_policy = \"{value}\"\nhygiene_batch = 50\n"
            );
            let cfg: AkhConfig = toml::from_str(&src).expect("parse");
            assert_eq!(cfg.memory.contradiction_policy, policy);
            assert_eq!(cfg.memory.hygiene_batch, 50);
            let raw: toml::Value = toml::from_str(&src).unwrap();
            assert!(AkhConfig::unknown_keys(&raw).is_empty());
            let back: AkhConfig = toml::from_str(&toml::to_string(&cfg).unwrap()).unwrap();
            assert_eq!(back.memory, cfg.memory);
        }

        // An unknown policy is a parse error, not a silent default.
        assert!(
            toml::from_str::<AkhConfig>(
                "name = \"vesta\"\n[memory]\ncontradiction_policy = \"ignore\"\n"
            )
            .is_err()
        );
        // A typo inside [memory] is flagged; a top-level scalar misplaced
        // under it is called out as the TOML-ordering footgun.
        let raw: toml::Value = toml::from_str(
            "name = \"vesta\"\n[memory]\ncontradiction_polcy = \"replace\"\nmodel = \"x\"\n",
        )
        .unwrap();
        let unknown = AkhConfig::unknown_keys(&raw);
        assert!(
            unknown
                .iter()
                .any(|k| k.path == "memory.contradiction_polcy")
        );
        assert!(
            unknown.iter().any(|k| k.path == "memory.model"
                && k.misplaced_under_table.as_deref() == Some("memory"))
        );
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
        // No [autonomy] table → the ADR 113 default (on, budgeted).
        assert!(cfg.autonomy.enabled);
        assert_eq!(cfg.autonomy, AutonomyConfig::default());
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
        assert!(
            bare.to_model_manager_config()
                .ollama
                .reasoning_effort
                .is_none()
        );
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
        assert!(
            effort_at < table_at,
            "reasoning_effort must serialize before the table:\n{content}"
        );

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

    /// Every shipped example workspace config must load cleanly and carry no
    /// unrecognized keys.
    ///
    /// These files are what operators copy when standing up an akh, so a key
    /// that serde silently drops here propagates into real fleets — and the
    /// TOML nesting footgun (a scalar written below a `[table]` header) is
    /// invisible without exactly this check.
    #[test]
    fn shipped_example_workspace_configs_are_valid() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/three-akhs/workspaces");
        let entries = std::fs::read_dir(&dir).expect("example workspace dir exists");
        let mut checked = 0;
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let unknown = AkhConfig::load_checked(&path)
                .unwrap_or_else(|e| panic!("{} failed to load: {e}", path.display()))
                .1;
            assert!(
                unknown.is_empty(),
                "{} has unrecognized keys: {unknown:?}",
                path.display()
            );
            checked += 1;
        }
        assert!(
            checked >= 4,
            "expected the example akhs to be checked, saw {checked}"
        );
    }

    #[test]
    fn context_strategy_serializes_before_first_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let mut cfg = AkhConfig::with_name("deepseek");
        cfg.context_strategy = Some("virtual".into());
        cfg.save(&path).expect("save");
        let content = std::fs::read_to_string(&path).expect("read");
        let scalar_at = content.find("context_strategy").expect("scalar present");
        let table_at = content
            .lines()
            .find(|line| line.starts_with('['))
            .and_then(|line| content.find(line))
            .expect("a table header present");
        assert!(
            scalar_at < table_at,
            "context_strategy must serialize before the first [table]:\n{content}"
        );
    }

    #[test]
    fn openai_backend_parses_and_translates_to_client_config() {
        let toml_src = r#"
            name = "deepseek"
            backend = "openai"
            model = "DeepSeek-V4-Flash-0731"
            context_strategy = "virtual"

            [openai_endpoint]
            base_url = "https://inference.hetzner.com/api/v1"
            api_key = "env:HETZNER_INFERENCE_KEY"
            context_window = 131072
        "#;
        let cfg: AkhConfig = toml::from_str(toml_src).expect("parse");
        assert_eq!(cfg.backend, ChatBacking::OpenAi);
        assert_eq!(cfg.context_strategy.as_deref(), Some("virtual"));

        let openai = cfg
            .to_openai_config()
            .expect("valid config")
            .expect("openai-backed akh yields a config");
        assert_eq!(openai.base_url, "https://inference.hetzner.com/api/v1");
        assert_eq!(openai.model, "DeepSeek-V4-Flash-0731");
        assert_eq!(openai.context_window, Some(131072));
        // Unset default, not a hardcoded value.
        assert_eq!(openai.timeout_secs, 600);
        // No unknown keys: the new table is registered in the schema.
        let raw: toml::Value = toml::from_str(toml_src).expect("raw parse");
        assert!(
            AkhConfig::unknown_keys(&raw).is_empty(),
            "openai_endpoint keys must be recognized: {:?}",
            AkhConfig::unknown_keys(&raw)
        );
    }

    #[test]
    fn openai_backend_without_its_endpoint_is_refused_not_silently_downgraded() {
        let cfg: AkhConfig =
            toml::from_str("name = \"d\"\nbackend = \"openai\"\nmodel = \"m\"\n").expect("parse");
        let err = cfg.to_openai_config().expect_err("must refuse");
        assert!(
            matches!(
                err,
                AkhConfigError::MissingOpenAiEndpoint {
                    field: "openai_endpoint",
                    ..
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn openai_backend_without_a_pinned_model_is_refused() {
        let toml_src = "name = \"d\"\nbackend = \"openai\"\n\n[openai_endpoint]\nbase_url = \"https://x/v1\"\n";
        let cfg: AkhConfig = toml::from_str(toml_src).expect("parse");
        let err = cfg.to_openai_config().expect_err("must refuse");
        assert!(
            matches!(
                err,
                AkhConfigError::MissingOpenAiEndpoint { field: "model", .. }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn non_openai_akhs_yield_no_openai_config() {
        let cfg = AkhConfig::with_name("tecton");
        assert_eq!(cfg.backend, ChatBacking::Ollama);
        assert!(cfg.to_openai_config().expect("ok").is_none());
    }

    #[test]
    fn openai_config_round_trips_through_save_and_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let mut cfg = AkhConfig::with_name("deepseek");
        cfg.backend = ChatBacking::OpenAi;
        cfg.model = Some("DeepSeek-V4-Flash-0731".into());
        cfg.context_strategy = Some("virtual".into());
        cfg.openai_endpoint = Some(OpenAiEndpoint {
            base_url: "https://inference.hetzner.com/api/v1".into(),
            api_key: Some("env:HETZNER_INFERENCE_KEY".into()),
            context_window: Some(131072),
            timeout_secs: 600,
            max_tokens: None,
            temperature: None,
        });
        cfg.save(&path).expect("save");
        let loaded = AkhConfig::load(&path).expect("load");
        assert_eq!(loaded.backend, ChatBacking::OpenAi);
        assert_eq!(loaded.openai_endpoint, cfg.openai_endpoint);
        assert_eq!(loaded.context_strategy.as_deref(), Some("virtual"));
    }

    #[cfg(unix)]
    #[test]
    fn saved_config_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let cfg = AkhConfig::with_name("thoth");
        cfg.save(&path).expect("save");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o600,
            "akh.toml may contain secrets and must be 0600, got {mode:o}"
        );
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
        assert_eq!(
            unknown[0].misplaced_under_table.as_deref(),
            Some("ollama_endpoint")
        );
        let msg = unknown[0].to_string();
        assert!(
            msg.contains("top-level"),
            "message should call out the footgun: {msg}"
        );
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

    /// `[provider]` (Phase 53g PR 4): parses into the typed ladder and
    /// lexicon, is a known table (a typo inside it is reported), defaults
    /// when absent, round-trips through `save`, and rejects an empty ladder.
    #[test]
    fn provider_table_parses_is_known_and_round_trips() {
        let toml_src = r#"
            name = "janus"

            [provider]
            cooldown_ladder_secs = [60, 600]
            quota_error_codes = ["insufficient_quota"]
            auth_error_codes = ["invalid_api_key"]
            rate_limit_error_codes = ["rate_limit_exceeded"]
        "#;
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        assert!(AkhConfig::unknown_keys(&raw).is_empty());
        let cfg: AkhConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.provider.cooldown_ladder_secs.cooldown(1),
            std::time::Duration::from_secs(600)
        );
        assert!(
            cfg.provider
                .lexicon
                .auth_error_codes
                .contains("invalid_api_key")
        );

        let typo: toml::Value =
            toml::from_str("name = \"a\"\n[provider]\nquota_codes = [\"x\"]\n").unwrap();
        let unknown = AkhConfig::unknown_keys(&typo);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "provider.quota_codes");

        let absent: AkhConfig = toml::from_str("name = \"a\"\n").unwrap();
        assert!(absent.provider.is_default());
        assert!(
            !toml::to_string(&absent).unwrap().contains("[provider]"),
            "a default table is not written"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        cfg.save(&path).expect("save");
        let (loaded, unknown) = AkhConfig::load_checked(&path).expect("load");
        assert!(unknown.is_empty(), "{unknown:?}");
        assert_eq!(loaded.provider, cfg.provider);

        assert!(toml::from_str::<AkhConfig>("[provider]\ncooldown_ladder_secs = []\n").is_err());
    }

    /// Every key `save()` writes must be a known key: a saved config loads
    /// with no unknown-key warning (regression: the flattened
    /// `WorkspaceConfig::enabled` was written top-level but missing from
    /// the known list, so every load warned "value was dropped: enabled").
    #[test]
    fn saved_config_has_no_unknown_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");

        let plain = AkhConfig::with_name("janus");
        plain.save(&path).expect("save default akh.toml");
        let raw = std::fs::read_to_string(&path).expect("read");
        assert!(
            raw.lines().any(|line| line.starts_with("enabled")),
            "save writes the workspace power switch top-level:\n{raw}"
        );
        let (loaded, unknown) = AkhConfig::load_checked(&path).expect("load");
        assert!(
            unknown.is_empty(),
            "unknown keys in saved config: {unknown:?}"
        );
        assert!(loaded.workspace.enabled);

        let mut disabled = AkhConfig::with_name("vesta");
        disabled.workspace.enabled = false;
        disabled.model = Some("glm".into());
        disabled.context_strategy = Some("virtual".into());
        disabled.autonomy.enabled = false;
        disabled.save(&path).expect("save disabled akh.toml");
        let (loaded, unknown) = AkhConfig::load_checked(&path).expect("load");
        assert!(
            unknown.is_empty(),
            "unknown keys in saved config: {unknown:?}"
        );
        assert!(!loaded.workspace.enabled, "power switch round-trips");
        assert!(!loaded.autonomy.enabled);
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

    #[test]
    fn exec_section_defaults_to_no_override() {
        // Absent [exec] is None — "defer to the host's [exec] default" — not
        // an explicit InProcess pin. The distinction is what lets a daemon
        // apply its host policy to relocated akhs whose configs say nothing.
        let cfg = AkhConfig::default();
        assert!(cfg.exec.is_none());
        assert!(!cfg.is_remote_exec());
        // ...and with no host default either, execution is InProcess.
        assert!(matches!(
            cfg.effective_exec(None),
            ExecBackendConfig::InProcess
        ));
    }

    #[test]
    fn exec_section_absent_defers_to_host_default() {
        let cfg: AkhConfig = toml::from_str("name = \"a\"").unwrap();
        assert!(cfg.exec.is_none());
        // No host policy → byte-identical legacy behaviour.
        assert!(matches!(
            cfg.effective_exec(None),
            ExecBackendConfig::InProcess
        ));
        // A host policy reaches an akh that says nothing — the relocated-akh
        // case that once loaded with no privilege drop on a root daemon.
        let host = ExecBackendConfig::LocalDrop {
            user: "build".into(),
            workdir: None,
            shell: None,
            path: None,
        };
        assert!(matches!(
            cfg.effective_exec(Some(&host)),
            ExecBackendConfig::LocalDrop { .. }
        ));
    }

    #[test]
    fn explicit_per_akh_exec_beats_the_host_default() {
        let cfg: AkhConfig =
            toml::from_str("name = \"a\"\n[exec]\nkind = \"in-process\"\n").unwrap();
        let host = ExecBackendConfig::LocalDrop {
            user: "build".into(),
            workdir: None,
            shell: None,
            path: None,
        };
        // An akh that explicitly pins in-process keeps it even under a host
        // local-drop policy: overrides exist for the one-akh-differs case.
        assert!(matches!(
            cfg.effective_exec(Some(&host)),
            ExecBackendConfig::InProcess
        ));
    }

    #[test]
    fn exec_none_section_parses() {
        let toml_src = "name = \"thoth\"\n[exec]\nkind = \"none\"\n";
        let cfg: AkhConfig = toml::from_str(toml_src).unwrap();
        assert!(matches!(cfg.exec, Some(ExecBackendConfig::None)));
        assert!(matches!(cfg.effective_exec(None), ExecBackendConfig::None));
    }

    #[test]
    fn explicit_per_akh_exec_none_beats_host_local_drop_default() {
        // A manager/backend-role akh (kind = "none") must keep zero executor
        // tools even on a host whose default policy is local-drop — the
        // per-akh override always wins over the host default.
        let cfg: AkhConfig = toml::from_str("name = \"thoth\"\n[exec]\nkind = \"none\"\n").unwrap();
        let host = ExecBackendConfig::LocalDrop {
            user: "build".into(),
            workdir: None,
            shell: None,
            path: None,
        };
        assert!(matches!(
            cfg.effective_exec(Some(&host)),
            ExecBackendConfig::None
        ));
    }

    #[test]
    fn host_level_exec_none_applies_to_workspace_with_no_exec_section() {
        // A workspace that says nothing about [exec] defers to the host
        // default (the relocated-akh case) — including a host default of
        // "none" for a cluster of manager/backend-role akhs.
        let cfg: AkhConfig = toml::from_str("name = \"thoth\"").unwrap();
        assert!(cfg.exec.is_none());
        assert!(matches!(
            cfg.effective_exec(Some(&ExecBackendConfig::None)),
            ExecBackendConfig::None
        ));
    }

    #[test]
    fn exec_remote_section_parses() {
        let toml_src = r#"
            name = "thoth"
            [exec]
            kind = "remote"
            host = "armini"
            user = "toasty"
            port = 2222
            key = "/etc/akh/ssh/id_ed25519"
            known_hosts = "/etc/akh/ssh-hosts/known_hosts"
            workdir = "/home/toasty/akh-workspaces"
        "#;
        let cfg: AkhConfig = toml::from_str(toml_src).expect("remote exec config parses");
        let ExecBackendConfig::Remote {
            host,
            user,
            port,
            key,
            known_hosts,
            workdir,
        } = cfg.exec.as_ref().expect("[exec] present")
        else {
            panic!("expected Remote, got {:?}", cfg.exec);
        };
        assert_eq!(host, "armini");
        assert_eq!(user, "toasty");
        assert_eq!(*port, 2222);
        assert_eq!(key, std::path::Path::new("/etc/akh/ssh/id_ed25519"));
        assert_eq!(
            known_hosts.as_ref().map(std::path::Path::new),
            Some(std::path::Path::new("/etc/akh/ssh-hosts/known_hosts"))
        );
        assert_eq!(
            workdir.as_ref().map(std::path::Path::new),
            Some(std::path::Path::new("/home/toasty/akh-workspaces"))
        );
        assert!(cfg.is_remote_exec());
    }

    #[test]
    fn exec_remote_minimal_uses_default_port() {
        let toml_src = r#"
            name = "thoth"
            [exec]
            kind = "remote"
            host = "armini"
            user = "toasty"
            key = "/etc/akh/ssh/id_ed25519"
        "#;
        let cfg: AkhConfig = toml::from_str(toml_src).unwrap();
        let Some(ExecBackendConfig::Remote { port, .. }) = &cfg.exec else {
            panic!("expected Remote");
        };
        assert_eq!(*port, 22);
    }

    #[test]
    fn exec_in_process_parses_explicit() {
        let toml_src = "name = \"a\"\n[exec]\nkind = \"in-process\"\n";
        let cfg: AkhConfig = toml::from_str(toml_src).unwrap();
        assert!(matches!(cfg.exec, Some(ExecBackendConfig::InProcess)));
        assert!(!cfg.is_remote_exec());
    }

    #[test]
    fn exec_section_save_load_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let mut cfg = AkhConfig::with_name("thoth");
        cfg.exec = Some(ExecBackendConfig::Remote {
            host: "armini".into(),
            user: "toasty".into(),
            port: 22,
            key: "/etc/akh/ssh/id_ed25519".into(),
            known_hosts: Some("/etc/akh/ssh-hosts/known_hosts".into()),
            workdir: Some("/home/toasty/akh-workspaces".into()),
        });
        cfg.save(&path).expect("save");
        let loaded = AkhConfig::load(&path).expect("load");
        assert_eq!(loaded.exec, cfg.exec);
        assert!(loaded.is_remote_exec());
    }

    #[test]
    fn exec_section_no_unknown_keys() {
        let toml_src = r#"
            name = "thoth"
            [exec]
            kind = "remote"
            host = "armini"
            user = "toasty"
            key = "/etc/akh/ssh/id_ed25519"
        "#;
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        assert!(AkhConfig::unknown_keys(&raw).is_empty());
    }

    #[test]
    fn exec_section_unknown_key_is_flagged() {
        let toml_src = "name = \"a\"\n[exec]\nkind = \"remote\"\nhots = \"armini\"\n";
        let raw: toml::Value = toml::from_str(toml_src).unwrap();
        let unknown = AkhConfig::unknown_keys(&raw);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "exec.hots");
    }

    #[cfg(feature = "remote-executor")]
    #[test]
    fn to_remote_backend_config_translates_full_section() {
        use crate::backend::remote::{HostKeyPolicy, RemoteBackendConfig};
        let mut cfg = AkhConfig::with_name("thoth");
        cfg.exec = Some(ExecBackendConfig::Remote {
            host: "armini".into(),
            user: "toasty".into(),
            port: 2222,
            key: "/etc/akh/ssh/id_ed25519".into(),
            known_hosts: Some("/etc/akh/ssh-hosts/known_hosts".into()),
            workdir: Some("/home/toasty/wsd".into()),
        });
        let rbc = cfg.to_remote_backend_config().expect("Some");
        assert_eq!(rbc.host, "armini");
        assert_eq!(rbc.user, "toasty");
        assert_eq!(rbc.port, 2222);
        assert_eq!(
            rbc.private_key_path,
            std::path::Path::new("/etc/akh/ssh/id_ed25519")
        );
        assert!(matches!(
            rbc.host_key_policy,
            HostKeyPolicy::KnownHosts { path: Some(_) }
        ));
        assert_eq!(
            rbc.read_roots,
            vec![std::path::PathBuf::from("/home/toasty/wsd")]
        );
        assert_eq!(
            rbc.write_roots,
            vec![std::path::PathBuf::from("/home/toasty/wsd")]
        );
    }

    #[cfg(feature = "remote-executor")]
    #[test]
    fn to_remote_backend_config_none_for_in_process() {
        let cfg = AkhConfig::with_name("thoth");
        assert!(cfg.to_remote_backend_config().is_none());
    }

    /// The exact TOML shape an operator writes, parsed end to end.
    ///
    /// Worth a test of its own because the failure mode is silent: a
    /// mis-tagged `[exec]` section does not error, it falls back to
    /// `InProcess` — which under a root daemon means every command runs as
    /// root while the config says otherwise.
    #[test]
    fn local_drop_exec_section_parses_from_operator_toml() {
        let toml = r#"
name = "thoth-mac"

[exec]
kind = "local-drop"
user = "toasty"
workdir = "/Users/toasty/ws"
"#;
        let cfg: AkhConfig = toml::from_str(toml).expect("operator TOML must parse");
        assert!(
            cfg.is_local_drop_exec(),
            "`kind = \"local-drop\"` must select LocalDrop, not fall back to InProcess"
        );
        let drop = cfg
            .to_local_drop_config()
            .expect("is_local_drop_exec implies Some");
        assert_eq!(drop.user, "toasty");
        assert_eq!(
            drop.workdir.as_deref(),
            Some(std::path::Path::new("/Users/toasty/ws"))
        );
        // Unset `shell` must default rather than deserialize to empty.
        assert_eq!(drop.shell, std::path::Path::new("/bin/sh"));
    }

    #[test]
    fn exec_kind_is_kebab_case_on_the_wire() {
        // Guards the serde rename: `LocalDrop` must round-trip as `local-drop`,
        // which is what the docs and the installer script tell operators to write.
        let cfg = AkhConfig {
            exec: Some(ExecBackendConfig::LocalDrop {
                user: "build".into(),
                workdir: None,
                shell: None,
                path: None,
            }),
            ..AkhConfig::with_name("thoth")
        };
        let rendered = toml::to_string(&cfg).expect("config must serialize");
        assert!(
            rendered.contains(r#"kind = "local-drop""#),
            "expected kebab-case tag in:\n{rendered}"
        );
    }

    #[test]
    fn missing_exec_section_still_means_in_process() {
        // The default must not drift: an akh.toml with no [exec] is the
        // overwhelming majority, and must keep byte-identical behaviour.
        let cfg: AkhConfig = toml::from_str("name = \"thoth\"\n").expect("must parse");
        assert!(!cfg.is_local_drop_exec());
        assert!(!cfg.is_remote_exec());
        assert!(cfg.to_local_drop_config().is_none());
    }

    // ── Phase 53i: static per-akh MCP servers ────────────────────────────────

    const MCP_TOML: &str = r#"
name = "minerva"
model = "glm-5.3"

[ollama_endpoint]
kind = "local"

[mcp]
require_pushed_mcp = false

[[mcp.servers]]
name = "anima"
url = "https://anima.wegmueller.it/mcp"
auth = { token_env = "ROMAN_ANIMA_TOKEN" }

[[mcp.servers]]
name = "observer"
url = "http://127.0.0.1:18436/mcp"
tools = ["monitoring_snapshot"]
timeout_secs = 60
auth = { token_file = "/tmp/observer.key" }
"#;

    #[test]
    fn mcp_servers_parse_with_no_unknown_keys() {
        let cfg: AkhConfig = toml::from_str(MCP_TOML).expect("parse");
        assert_eq!(cfg.mcp.servers.len(), 2);
        assert_eq!(cfg.mcp.servers[0].name.as_str(), "anima");
        assert_eq!(
            cfg.mcp.servers[1].qualified_tool_name("monitoring_snapshot"),
            "mcp__observer__monitoring_snapshot"
        );
        assert_eq!(cfg.model.as_deref(), Some("glm-5.3"));
        let raw: toml::Value = toml::from_str(MCP_TOML).unwrap();
        assert!(
            AkhConfig::unknown_keys(&raw).is_empty(),
            "{:?}",
            AkhConfig::unknown_keys(&raw)
        );
    }

    #[test]
    fn mcp_servers_absent_means_none() {
        let cfg: AkhConfig = toml::from_str("name = \"a\"\n").expect("parse");
        assert!(cfg.mcp.servers.is_empty());
        assert!(!cfg.mcp.require_pushed_mcp);
    }

    #[test]
    fn mcp_unknown_table_key_is_reported() {
        let raw: toml::Value = toml::from_str("name = \"a\"\n[mcp]\nservrs = []\n").expect("parse");
        let unknown = AkhConfig::unknown_keys(&raw);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].path, "mcp.servrs");
    }

    #[test]
    fn mcp_duplicate_servers_fail_load_with_diagnostic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        std::fs::write(
            &path,
            "name = \"a\"\n\
             [[mcp.servers]]\nname = \"board\"\nurl = \"http://127.0.0.1:1/mcp\"\n\
             [[mcp.servers]]\nname = \"board\"\nurl = \"http://127.0.0.1:2/mcp\"\n",
        )
        .expect("write");
        let err = AkhConfig::load_checked(&path).unwrap_err();
        assert!(matches!(err, AkhConfigError::McpServers(_)), "{err:?}");
        use miette::Diagnostic;
        assert_eq!(
            err.code().map(|c| c.to_string()).as_deref(),
            Some("akh::mcp::config::invalid")
        );
        assert!(
            err.related()
                .is_some_and(|mut r| r.any(|d| d.to_string().contains("declared more than once")))
        );
    }

    #[test]
    fn mcp_bad_server_name_fails_parse() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        std::fs::write(
            &path,
            "name = \"a\"\n[[mcp.servers]]\nname = \"an ima\"\nurl = \"http://h/mcp\"\n",
        )
        .expect("write");
        let err = AkhConfig::load_checked(&path).unwrap_err();
        assert!(matches!(err, AkhConfigError::Parse { .. }), "{err:?}");
        assert!(err.to_string().contains("invalid MCP server name"), "{err}");
    }

    #[test]
    fn mcp_servers_round_trip_through_save_and_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("akh.toml");
        let cfg: AkhConfig = toml::from_str(MCP_TOML).expect("parse");
        cfg.save(&path).expect("save");
        let (loaded, unknown) = AkhConfig::load_checked(&path).expect("load");
        // Scoped to `mcp`: a saved config also carries the flattened workspace
        // `enabled` key, which the known-key list does not cover (pre-existing).
        assert!(
            !unknown.iter().any(|k| k.path.starts_with("mcp")),
            "{unknown:?}"
        );
        assert_eq!(loaded.mcp, cfg.mcp);
        assert_eq!(loaded.model, cfg.model);
    }
}

#[cfg(all(test, feature = "execd-executor"))]
mod execd_config_tests {
    use super::*;
    use crate::backend::execd::ExecutorRoute;

    #[test]
    fn execd_exec_table_parses_without_unknown_keys_and_translates() {
        let raw = r#"
name = "janus"
[exec]
kind = "execd"
sandbox_url = "https://akhsbxd.lab"
sandbox_token_file = "/run/akh/credentials/akhsbxd"
profile = "rust-workspace"
principal = "janus"
"#;
        let cfg: AkhConfig = toml::from_str(raw).unwrap();
        assert!(AkhConfig::unknown_keys(&toml::from_str(raw).unwrap()).is_empty());
        let exec = cfg.effective_exec(None);
        assert_eq!(exec.kind_name(), "execd");
        let execd = exec.to_execd_config("janus").unwrap();
        assert_eq!(execd.profile, "rust-workspace");
        assert_eq!(execd.poll_wait, std::time::Duration::from_millis(5_000));
        assert!(execd.session_id.starts_with("akhomed-janus-"));
        // Without executor_url/_token_file: the akhsbxd gateway with the sandbox token.
        assert_eq!(
            execd.executor,
            ExecutorRoute::ViaSandbox {
                token_file: "/run/akh/credentials/akhsbxd".into()
            }
        );
        // Each translation mints a fresh allocation session.
        assert_ne!(
            exec.to_execd_config("janus").unwrap().session_id,
            execd.session_id
        );

        let direct: AkhConfig = toml::from_str(&format!(
            "{raw}executor_url = \"http://127.0.0.1:8413\"\n\
             executor_token_file = \"/run/akh/credentials/akhexecd\"\npoll_wait_ms = 250\n"
        ))
        .unwrap();
        let direct = direct
            .effective_exec(None)
            .to_execd_config("janus")
            .unwrap();
        assert_eq!(
            direct.executor,
            ExecutorRoute::Direct {
                url: "http://127.0.0.1:8413".into(),
                token_file: "/run/akh/credentials/akhexecd".into()
            }
        );
        assert_eq!(direct.poll_wait, std::time::Duration::from_millis(250));
        assert!(
            ExecBackendConfig::InProcess
                .to_execd_config("janus")
                .is_none()
        );
    }
}
