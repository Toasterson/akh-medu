//! `sandbox_exec`: run a command in one of the workspace's additional execd
//! sandboxes (Phase 53g PR 8).
//!
//! A workspace with `[exec] kind = "execd"` gets one primary akhsbxd profile,
//! which serves `shell_exec`/`file_io`/`edit`/`glob`/`grep`. Some akhs need a
//! second, differently privileged sandbox on the same akhsbxd (for example one
//! whose pod runs under an RBAC-scoped Kubernetes ServiceAccount) while the
//! primary stays credential-free. Each `[[exec.sandboxes]]` entry is such a
//! sandbox, and this tool is the only way the model reaches it.
//!
//! ## Allocation
//!
//! Every sandbox connects lazily, on its first `sandbox_exec` call, so a
//! secondary sandbox that is down never stops the akh from loading. The
//! allocation is cached and reused; once it reports itself unusable
//! ([`crate::fs::Filesystem::is_writable`] is `false`, which for
//! [`ExecdBackend`] means the lease lapsed) the next call allocates a fresh
//! session. The connector is injectable ([`SandboxConnector`]), so tests run
//! the tool over an [`crate::backend::InProcessBackend`] instead of HTTP.
//!
//! ## stdin
//!
//! [`ExecRequest`] has no stdin channel. Text passed as `stdin` is staged as a
//! file under `<workspace root>/.akh-stdin/` through the backend's own
//! filesystem verbs, redirected into the command, and removed by the same
//! shell invocation once the command finishes. The command runs in a
//! subshell, so an `exit` inside it cannot skip that cleanup. A command killed
//! at its timeout leaves the staged file behind (best-effort cleanup only).
//!
//! ## Output
//!
//! Formatting and error mapping are shared with `shell_exec`
//! ([`render_exec_result`]), prefixed with the sandbox's name and profile.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use miette::Diagnostic;
use thiserror::Error;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::agent::tools::shell_exec::render_exec_result;
use crate::akh_config::{ExecBackendConfig, ExecdSandboxConfigError, SandboxName};
use crate::backend::execd::{ExecdBackend, ExecdError};
use crate::backend::{Backend, ExecLocus, ExecRequest};
use crate::engine::Engine;
use crate::fs::FsError;

/// Tool name the model calls.
pub const SANDBOX_EXEC_TOOL_NAME: &str = "sandbox_exec";

/// Default command timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Longest accepted command timeout in seconds. Longer values are refused,
/// not clamped, so the model learns the real limit instead of silently getting
/// a shorter run than it asked for.
pub const MAX_TIMEOUT_SECS: u64 = 600;

/// Largest accepted `stdin` payload, in bytes.
pub const MAX_STDIN_BYTES: usize = 1024 * 1024;

/// Directory (below the sandbox's first read root) where stdin is staged.
const STDIN_STAGING_DIR: &str = ".akh-stdin";

/// Opens a fresh allocation for one sandbox. Called on first use and again
/// whenever the cached backend reports itself unusable.
pub type SandboxConnector = Box<dyn Fn() -> Result<Arc<dyn Backend>, ExecdError> + Send + Sync>;

/// Why a `sandbox_exec` call could not run its command. Rendered into the
/// tool's error output together with its help (and the help of the wrapped
/// execd / filesystem diagnostic, which says whether a retry is safe).
#[derive(Debug, Error, Diagnostic)]
pub enum SandboxExecError {
    /// The requested sandbox is not configured on this akh.
    #[error("unknown sandbox `{requested}`; configured sandboxes: {configured}")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::unknown_sandbox),
        help(
            "Pass one of the configured names as `sandbox`. Workspace commands that need \
             no extra privileges belong in `shell_exec` instead."
        )
    )]
    UnknownSandbox {
        requested: String,
        configured: String,
    },

    /// `timeout` is not a whole number of seconds within range.
    #[error("invalid timeout `{value}`: expected whole seconds between 1 and {max}")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::timeout),
        help(
            "Pass `timeout` as whole seconds from 1 to {max}, or omit it for the default. \
             Split longer work into steps that each finish within the limit."
        )
    )]
    TimeoutOutOfRange { value: String, max: u64 },

    /// `stdin` exceeds [`MAX_STDIN_BYTES`].
    #[error("stdin is {len} bytes; at most {max} bytes are accepted")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::stdin_too_large),
        help(
            "Options: send the input in smaller pieces across several calls, or have the \
             command fetch the data itself inside the sandbox."
        )
    )]
    StdinTooLarge { len: usize, max: usize },

    /// The sandbox could not be allocated.
    #[error("sandbox `{name}` (profile {profile}) is unavailable: {source}")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::unavailable),
        help(
            "The primary workspace tools are unaffected. The allocation is retried on the \
             next `sandbox_exec` call; follow the guidance below before retrying."
        )
    )]
    Unavailable {
        name: SandboxName,
        profile: String,
        #[source]
        #[diagnostic_source]
        source: ExecdError,
    },

    /// The allocated sandbox reports no workspace root to stage stdin under.
    #[error("sandbox `{name}` reports no workspace root, so stdin cannot be staged")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::no_workspace_root),
        help(
            "Options: run the command without `stdin` (for example with a here-document \
             in `command`), or have an operator check the sandbox profile's binding root."
        )
    )]
    NoWorkspaceRoot { name: SandboxName },

    /// Writing the staged stdin file failed.
    #[error("sandbox `{name}`: cannot stage stdin at {path}: {source}")]
    #[diagnostic(
        code(akh::tools::sandbox_exec::stage_stdin),
        help(
            "The command did not run. Options: retry without `stdin` (a here-document in \
             `command` works too), or follow the filesystem guidance below."
        )
    )]
    StageStdin {
        name: SandboxName,
        path: String,
        #[source]
        #[diagnostic_source]
        source: FsError,
    },
}

impl SandboxExecError {
    /// Message, own help, then the wrapped diagnostic's help, as one block.
    fn render(&self) -> String {
        let mut text = self.to_string();
        if let Some(help) = self.help() {
            text.push_str(&format!("\nhelp: {help}"));
        }
        if let Some(inner_help) = self.diagnostic_source().and_then(|inner| inner.help()) {
            text.push_str(&format!("\nhelp: {inner_help}"));
        }
        text
    }

    fn into_output(self) -> ToolOutput {
        ToolOutput::err(self.render())
    }
}

/// One configured sandbox and its lazily opened allocation.
pub struct SandboxEntry {
    name: SandboxName,
    profile: String,
    description: Option<String>,
    connector: SandboxConnector,
    backend: Mutex<Option<Arc<dyn Backend>>>,
}

impl std::fmt::Debug for SandboxEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxEntry")
            .field("name", &self.name)
            .field("profile", &self.profile)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl SandboxEntry {
    /// A sandbox reached through `connector` on first use.
    pub fn new(
        name: SandboxName,
        profile: impl Into<String>,
        description: Option<String>,
        connector: SandboxConnector,
    ) -> Self {
        Self {
            name,
            profile: profile.into(),
            description,
            connector,
            backend: Mutex::new(None),
        }
    }

    /// The name the model selects this sandbox by.
    pub fn name(&self) -> &SandboxName {
        &self.name
    }

    /// The akhsbxd workspace profile this sandbox allocates from.
    pub fn profile(&self) -> &str {
        &self.profile
    }

    /// The operator-facing description, if configured.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// The cached allocation, connecting (or reconnecting after the lease
    /// lapsed) first. The lock is held across the connect so two concurrent
    /// calls never allocate twice.
    fn backend(&self) -> Result<Arc<dyn Backend>, SandboxExecError> {
        let mut slot = self.backend.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(backend) = slot.as_ref() {
            if backend.is_writable() {
                return Ok(Arc::clone(backend));
            }
            tracing::warn!(
                target: "akh::backend::execd",
                sandbox = %self.name,
                profile = %self.profile,
                "sandbox allocation is no longer usable (lease lapsed); allocating a fresh one"
            );
            // Dropping the stale allocation releases it (non-blocking).
            *slot = None;
        }
        let backend = (self.connector)().map_err(|source| SandboxExecError::Unavailable {
            name: self.name.clone(),
            profile: self.profile.clone(),
            source,
        })?;
        let locus = backend.locus();
        tracing::info!(
            target: "akh::backend::execd",
            sandbox = %self.name,
            profile = %self.profile,
            host = ?locus.host,
            session = ?locus.correlation_id,
            "sandbox_exec allocated its sandbox"
        );
        *slot = Some(Arc::clone(&backend));
        Ok(backend)
    }
}

/// Run a command in one of the workspace's additional execd sandboxes.
#[derive(Debug)]
pub struct SandboxExecTool {
    sandboxes: Vec<SandboxEntry>,
    /// Locus of the most recent execution, for the audit envelope.
    last_locus: Mutex<Option<ExecLocus>>,
}

impl SandboxExecTool {
    /// A tool over explicit sandbox entries.
    ///
    /// # Errors
    ///
    /// [`ExecdSandboxConfigError::DuplicateName`] when two entries share a name.
    pub fn new(sandboxes: Vec<SandboxEntry>) -> Result<Self, ExecdSandboxConfigError> {
        let mut seen = HashSet::new();
        for entry in &sandboxes {
            if !seen.insert(entry.name.clone()) {
                return Err(ExecdSandboxConfigError::DuplicateName {
                    name: entry.name.clone(),
                });
            }
        }
        Ok(Self {
            sandboxes,
            last_locus: Mutex::new(None),
        })
    }

    /// The tool for a workspace's `[exec]` table: one entry per
    /// `[[exec.sandboxes]]`, each allocating through [`ExecdBackend::connect`]
    /// with [`ExecBackendConfig::to_execd_sandbox_config`] (a fresh session
    /// per connect). Nothing is allocated here.
    ///
    /// Returns `Ok(None)` when the executor is not `execd` or configures no
    /// additional sandboxes.
    ///
    /// # Errors
    ///
    /// The first invalid entry, per
    /// [`ExecBackendConfig::validate_execd_sandboxes`].
    pub fn from_execd_config(
        exec: &ExecBackendConfig,
        workspace: &str,
    ) -> Result<Option<Self>, ExecdSandboxConfigError> {
        exec.validate_execd_sandboxes()?;
        if exec.execd_sandboxes().is_empty() {
            return Ok(None);
        }
        let entries = exec
            .execd_sandboxes()
            .iter()
            .map(|sandbox| {
                let exec = exec.clone();
                let workspace = workspace.to_owned();
                let spec = sandbox.clone();
                let connector: SandboxConnector = Box::new(move || {
                    let config =
                        exec.to_execd_sandbox_config(&workspace, &spec)
                            .ok_or_else(|| ExecdError::Config {
                                reason: format!(
                                    "sandbox `{}` needs `[exec] kind = \"execd\"`, found `{}`",
                                    spec.name,
                                    exec.kind_name()
                                ),
                            })?;
                    Ok(Arc::new(ExecdBackend::connect(config)?) as Arc<dyn Backend>)
                });
                SandboxEntry::new(
                    sandbox.name.clone(),
                    sandbox.profile.clone(),
                    sandbox.description.clone(),
                    connector,
                )
            })
            .collect();
        Self::new(entries).map(Some)
    }

    /// The configured sandboxes, in configuration order.
    pub fn sandboxes(&self) -> &[SandboxEntry] {
        &self.sandboxes
    }

    fn configured_names(&self) -> String {
        self.sandboxes
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn entry(&self, requested: &str) -> Result<&SandboxEntry, SandboxExecError> {
        self.sandboxes
            .iter()
            .find(|entry| entry.name.as_str() == requested)
            .ok_or_else(|| SandboxExecError::UnknownSandbox {
                requested: requested.to_owned(),
                configured: self.configured_names(),
            })
    }

    /// Stage `stdin` inside the sandbox and wrap `command` so it reads the
    /// staged file and removes it afterwards, preserving the exit status.
    fn stage_stdin(
        entry: &SandboxEntry,
        backend: &dyn Backend,
        command: &str,
        stdin: &str,
    ) -> Result<String, SandboxExecError> {
        let root =
            backend
                .read_roots()
                .first()
                .ok_or_else(|| SandboxExecError::NoWorkspaceRoot {
                    name: entry.name.clone(),
                })?;
        let path: PathBuf = root
            .join(STDIN_STAGING_DIR)
            .join(uuid::Uuid::now_v7().simple().to_string());
        backend
            .write(&path, stdin)
            .map_err(|source| SandboxExecError::StageStdin {
                name: entry.name.clone(),
                path: path.display().to_string(),
                source,
            })?;
        Ok(wrap_with_stdin(command, &path.to_string_lossy()))
    }

    fn run(&self, input: &ToolInput, command: &str) -> Result<ToolOutput, SandboxExecError> {
        let requested = input.get("sandbox").unwrap_or_default();
        let entry = self.entry(requested)?;
        let timeout = parse_timeout(input)?;
        let stdin = match input.get("stdin") {
            Some(text) if text.len() > MAX_STDIN_BYTES => {
                return Err(SandboxExecError::StdinTooLarge {
                    len: text.len(),
                    max: MAX_STDIN_BYTES,
                });
            }
            other => other,
        };

        let backend = entry.backend()?;
        let shell_command = match stdin {
            Some(text) => Self::stage_stdin(entry, backend.as_ref(), command, text)?,
            None => command.to_owned(),
        };
        // No ACP session cwd here: the client's project directory lives in the
        // primary workspace, not in this sandbox. Commands start at the
        // sandbox's own workspace root.
        let result = backend.exec(ExecRequest::shell(shell_command, timeout));
        *self.last_locus.lock().unwrap_or_else(|e| e.into_inner()) = Some(backend.locus());

        let mut output = render_exec_result(command, result);
        output.result = format!(
            "Sandbox: {} (profile {})\n{}",
            entry.name, entry.profile, output.result
        );
        Ok(output)
    }
}

/// `timeout` in whole seconds, 1..=[`MAX_TIMEOUT_SECS`]; out-of-range or
/// non-numeric values are refused (never clamped).
fn parse_timeout(input: &ToolInput) -> Result<Duration, SandboxExecError> {
    let Some(raw) = input.get_value("timeout") else {
        return Ok(Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    };
    let invalid = || SandboxExecError::TimeoutOutOfRange {
        value: match raw {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        },
        max: MAX_TIMEOUT_SECS,
    };
    match input.get_u64("timeout") {
        Some(secs) if (1..=MAX_TIMEOUT_SECS).contains(&secs) => Ok(Duration::from_secs(secs)),
        _ => Err(invalid()),
    }
}

/// POSIX single-quote `raw` for `/bin/sh`.
fn shell_single_quote(raw: &str) -> String {
    format!("'{}'", raw.replace('\'', r"'\''"))
}

/// Run `command` in a subshell reading `stdin_path`, then remove the file and
/// exit with the command's status. The subshell keeps an `exit` inside
/// `command` from skipping the cleanup; the newlines keep a leading `(` from
/// forming `((` and a trailing comment from swallowing the `)`.
fn wrap_with_stdin(command: &str, stdin_path: &str) -> String {
    let quoted = shell_single_quote(stdin_path);
    format!(
        "(\n{command}\n) < {quoted}\nakh_stdin_status=$?\nrm -f {quoted}\nexit \"$akh_stdin_status\""
    )
}

impl Tool for SandboxExecTool {
    fn signature(&self) -> ToolSignature {
        let mut description = format!(
            "Execute a shell command in a named additional sandbox (timeout default \
             {DEFAULT_TIMEOUT_SECS}s, max {MAX_TIMEOUT_SECS}s; output limit 64 KB). \
             Configured sandboxes:"
        );
        for entry in &self.sandboxes {
            description.push_str(&format!("\n- {} ({})", entry.name, entry.profile));
            if let Some(text) = entry.description() {
                description.push_str(&format!(": {text}"));
            }
        }
        ToolSignature {
            name: SANDBOX_EXEC_TOOL_NAME.into(),
            description,
            parameters: vec![
                ToolParam {
                    name: "sandbox".into(),
                    description: format!("Sandbox name: one of {}.", self.configured_names()),
                    required: true,
                },
                ToolParam {
                    name: "command".into(),
                    description: "Shell command to execute (via /bin/sh -c).".into(),
                    required: true,
                },
                ToolParam {
                    name: "timeout".into(),
                    description: format!(
                        "Timeout in seconds, 1 to {MAX_TIMEOUT_SECS} (default: \
                         {DEFAULT_TIMEOUT_SECS})."
                    ),
                    required: false,
                },
                ToolParam {
                    name: "stdin".into(),
                    description: "Text fed to the command's standard input (at most 1 MiB).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        input.require("sandbox", SANDBOX_EXEC_TOOL_NAME)?;
        let command = input.require("command", SANDBOX_EXEC_TOOL_NAME)?;
        Ok(self
            .run(&input, command)
            .unwrap_or_else(SandboxExecError::into_output))
    }

    fn manifest(&self) -> ToolManifest {
        let description = "Executes arbitrary shell commands in a separately privileged \
                           sandbox — full access to whatever that sandbox can reach.";
        ToolManifest {
            name: SANDBOX_EXEC_TOOL_NAME.into(),
            description: description.into(),
            parameters: vec![
                ToolParamSchema::required("sandbox", "Configured sandbox name."),
                ToolParamSchema::required("command", "Shell command to execute."),
                ToolParamSchema::optional("timeout", "Timeout in seconds, 1 to 600 (default: 60)."),
                ToolParamSchema::optional("stdin", "Text fed to the command's standard input."),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Critical,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description: description.into(),
                shadow_triggers: vec![
                    "exec".into(),
                    "shell".into(),
                    "command".into(),
                    "rm".into(),
                    "sudo".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }

    fn locus(&self) -> Option<ExecLocus> {
        self.last_locus
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InProcessBackend;
    use crate::engine::EngineConfig;
    use crate::fs::{Filesystem, LocalFs};
    use crate::vsa::Dimension;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn name(raw: &str) -> SandboxName {
        SandboxName::parse(raw).unwrap()
    }

    /// A connector over an in-process backend sandboxed to `root`, counting
    /// how often it is invoked.
    fn counting_connector(root: PathBuf, calls: Arc<AtomicUsize>) -> SandboxConnector {
        Box::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
            let fs: Arc<dyn Filesystem> = Arc::new(LocalFs::sandbox(&root));
            Ok(Arc::new(InProcessBackend::new(fs)) as Arc<dyn Backend>)
        })
    }

    fn ops_tool() -> (SandboxExecTool, tempfile::TempDir, Arc<AtomicUsize>) {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let entry = SandboxEntry::new(
            name("ops"),
            "vesta-ops",
            Some("cluster access".into()),
            counting_connector(dir.path().to_path_buf(), Arc::clone(&calls)),
        );
        (SandboxExecTool::new(vec![entry]).unwrap(), dir, calls)
    }

    #[test]
    fn sandbox_exec_runs_a_command_in_the_named_sandbox() {
        let (tool, _dir, _calls) = ops_tool();
        let output = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "echo hello-sandbox"),
            )
            .unwrap();
        assert!(output.success, "{}", output.result);
        assert!(
            output
                .result
                .starts_with("Sandbox: ops (profile vesta-ops)\nCommand: echo hello-sandbox\n"),
            "{}",
            output.result
        );
        assert!(output.result.contains("Exit code: 0"));
        assert!(output.result.contains("hello-sandbox"));
        assert_eq!(tool.locus().unwrap().backend, "in-process");
    }

    #[test]
    fn sandbox_exec_feeds_stdin_and_removes_the_staged_file() {
        let (tool, dir, _calls) = ops_tool();
        let output = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "cat; printf '|'; exit 3")
                    .with_param("stdin", "line one\nit's quoted\n"),
            )
            .unwrap();
        // `exit 3` inside the command still leaves the cleanup running and the
        // status preserved.
        assert!(!output.success);
        assert!(output.result.contains("Exit code: 3"), "{}", output.result);
        assert!(
            output.result.contains("line one\nit's quoted\n|"),
            "{}",
            output.result
        );
        let staging = dir.path().join(STDIN_STAGING_DIR);
        let leftovers: Vec<_> = std::fs::read_dir(&staging).unwrap().collect();
        assert!(
            leftovers.is_empty(),
            "staged stdin left behind: {leftovers:?}"
        );

        let counted = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "wc -c | tr -d ' '")
                    .with_param("stdin", "12345"),
            )
            .unwrap();
        assert!(counted.success, "{}", counted.result);
        assert!(
            counted.result.contains("stdout:\n5\n"),
            "{}",
            counted.result
        );
    }

    #[test]
    fn sandbox_exec_refuses_oversized_stdin() {
        let (tool, _dir, calls) = ops_tool();
        let output = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "cat")
                    .with_param("stdin", "x".repeat(MAX_STDIN_BYTES + 1)),
            )
            .unwrap();
        assert!(!output.success);
        assert!(output.result.contains("at most 1048576 bytes"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "no allocation for a refused call"
        );
    }

    #[test]
    fn sandbox_exec_unknown_sandbox_lists_the_valid_names() {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let entries = ["ops", "gitops"]
            .into_iter()
            .map(|n| {
                SandboxEntry::new(
                    name(n),
                    format!("vesta-{n}"),
                    None,
                    counting_connector(dir.path().to_path_buf(), Arc::clone(&calls)),
                )
            })
            .collect();
        let tool = SandboxExecTool::new(entries).unwrap();
        let output = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "prod")
                    .with_param("command", "true"),
            )
            .unwrap();
        assert!(!output.success);
        assert!(
            output
                .result
                .contains("unknown sandbox `prod`; configured sandboxes: ops, gitops"),
            "{}",
            output.result
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn sandbox_exec_surfaces_a_connector_error_with_its_help() {
        let entry = SandboxEntry::new(
            name("ops"),
            "vesta-ops",
            None,
            Box::new(|| {
                Err(ExecdError::UnknownOutcome {
                    operation: "allocate",
                    receipt: "r-1".into(),
                })
            }),
        );
        let tool = SandboxExecTool::new(vec![entry]).unwrap();
        let output = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "true"),
            )
            .unwrap();
        assert!(!output.success);
        assert!(
            output
                .result
                .contains("sandbox `ops` (profile vesta-ops) is unavailable"),
            "{}",
            output.result
        );
        // The execd diagnostic's help reaches the model: no blind retries.
        assert!(
            output.result.contains("Do not retry it blindly"),
            "{}",
            output.result
        );
        assert!(tool.locus().is_none());
    }

    #[test]
    fn sandbox_exec_connects_lazily_and_once() {
        let (tool, _dir, calls) = ops_tool();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "construction must not connect"
        );
        for _ in 0..2 {
            let output = tool
                .execute(
                    &test_engine(),
                    ToolInput::new()
                        .with_param("sandbox", "ops")
                        .with_param("command", "true"),
                )
                .unwrap();
            assert!(output.success, "{}", output.result);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn sandbox_exec_reconnects_when_the_backend_is_no_longer_writable() {
        // `LocalFs::denied()` reports `is_writable() == false`, the same
        // signal a lapsed execd lease gives, so every call reconnects.
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let entry = SandboxEntry::new(
            name("ops"),
            "vesta-ops",
            None,
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(InProcessBackend::exec_only()) as Arc<dyn Backend>)
            }),
        );
        let tool = SandboxExecTool::new(vec![entry]).unwrap();
        for _ in 0..2 {
            tool.execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "true"),
            )
            .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn sandbox_exec_refuses_timeouts_outside_the_range() {
        let (tool, _dir, calls) = ops_tool();
        for bad in [
            serde_json::json!(0),
            serde_json::json!(601),
            serde_json::json!(-5),
            serde_json::json!("soon"),
        ] {
            let output = tool
                .execute(
                    &test_engine(),
                    ToolInput::new()
                        .with_param("sandbox", "ops")
                        .with_param("command", "true")
                        .with_value("timeout", bad.clone()),
                )
                .unwrap();
            assert!(!output.success, "timeout {bad} must be refused");
            assert!(
                output.result.contains("between 1 and 600"),
                "{}",
                output.result
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let ok = tool
            .execute(
                &test_engine(),
                ToolInput::new()
                    .with_param("sandbox", "ops")
                    .with_param("command", "true")
                    .with_value("timeout", serde_json::json!(600)),
            )
            .unwrap();
        assert!(ok.success, "{}", ok.result);
    }

    #[test]
    fn sandbox_exec_rejects_duplicate_entries() {
        let dir = tempfile::tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let entries = (0..2)
            .map(|_| {
                SandboxEntry::new(
                    name("ops"),
                    "vesta-ops",
                    None,
                    counting_connector(dir.path().to_path_buf(), Arc::clone(&calls)),
                )
            })
            .collect();
        assert!(matches!(
            SandboxExecTool::new(entries),
            Err(ExecdSandboxConfigError::DuplicateName { .. })
        ));
    }

    #[test]
    fn sandbox_exec_signature_and_manifest_describe_each_sandbox() {
        let (tool, _dir, _calls) = ops_tool();
        let signature = tool.signature();
        assert_eq!(signature.name, "sandbox_exec");
        assert!(
            signature
                .description
                .contains("\n- ops (vesta-ops): cluster access"),
            "{}",
            signature.description
        );
        let required: Vec<_> = signature
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(required, ["sandbox", "command"]);
        let manifest = tool.manifest();
        assert_eq!(manifest.danger.level, DangerLevel::Critical);
        assert_eq!(manifest.danger.effect, ToolEffect::Mutating);
        assert!(
            manifest
                .danger
                .capabilities
                .contains(&Capability::ProcessExec)
        );
    }

    #[test]
    fn sandbox_exec_from_config_is_none_without_sandboxes_and_validates() {
        let primary_only: ExecBackendConfig = toml::from_str(
            r#"
kind = "execd"
sandbox_url = "https://akhsbxd.lab"
sandbox_token_file = "/run/akh/credentials/akhsbxd"
profile = "vesta-gitops"
principal = "vesta"
"#,
        )
        .unwrap();
        assert!(
            SandboxExecTool::from_execd_config(&primary_only, "vesta")
                .unwrap()
                .is_none()
        );
        assert!(
            SandboxExecTool::from_execd_config(&ExecBackendConfig::InProcess, "vesta")
                .unwrap()
                .is_none()
        );

        let with_ops: ExecBackendConfig = toml::from_str(
            r#"
kind = "execd"
sandbox_url = "https://akhsbxd.lab"
sandbox_token_file = "/run/akh/credentials/akhsbxd"
profile = "vesta-gitops"
principal = "vesta"

[[sandboxes]]
name = "ops"
profile = "vesta-ops"
"#,
        )
        .unwrap();
        let tool = SandboxExecTool::from_execd_config(&with_ops, "vesta")
            .unwrap()
            .expect("one sandbox configured");
        assert_eq!(tool.sandboxes().len(), 1);
        assert_eq!(tool.sandboxes()[0].profile(), "vesta-ops");

        let reused: ExecBackendConfig = toml::from_str(
            r#"
kind = "execd"
sandbox_url = "https://akhsbxd.lab"
sandbox_token_file = "/run/akh/credentials/akhsbxd"
profile = "vesta-gitops"
principal = "vesta"

[[sandboxes]]
name = "ops"
profile = "vesta-gitops"
"#,
        )
        .unwrap();
        assert!(matches!(
            SandboxExecTool::from_execd_config(&reused, "vesta"),
            Err(ExecdSandboxConfigError::ProfileIsPrimary { .. })
        ));
    }

    #[test]
    fn sandbox_exec_wrapper_quotes_the_staged_path() {
        assert_eq!(shell_single_quote("/a/it's"), r"'/a/it'\''s'");
        let wrapped = wrap_with_stdin("(echo hi) # trailing", "/w/.akh-stdin/x");
        assert!(wrapped.starts_with("(\n(echo hi) # trailing\n) < '/w/.akh-stdin/x'\n"));
        assert!(wrapped.ends_with("rm -f '/w/.akh-stdin/x'\nexit \"$akh_stdin_status\""));
    }
}
