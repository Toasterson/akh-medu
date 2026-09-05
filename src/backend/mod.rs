//! Unified execution/file backend — the Phase 48 `Backend` seam (ADR 082).
//!
//! A [`Backend`] is the single surface the executor-facing tools (`file_io`,
//! `edit`, `glob`, `grep`, `shell_exec`) operate through: the union of the
//! [`Filesystem`] verbs **plus** process execution ([`Backend::exec`]) and a
//! posture self-report ([`Backend::handshake`]). Binding files and exec to one
//! object guarantees a file write and a shell command land on the *same*
//! target (host, sandbox, principal) — the coherence property ADR 082 §10
//! requires.
//!
//! Phase 48a ships exactly one implementation, [`InProcessBackend`]: today's
//! sandboxed [`LocalFs`](crate::fs::LocalFs) file semantics plus the historic
//! in-process `/bin/sh -c` execution, byte-for-byte. It is honest about what
//! it is — [`Confinement::Unconfined`] — so `shell_exec` stays
//! `DangerLevel::Critical`. Later sub-phases plug in `LocalDrop` (48b,
//! privilege-dropped executor) and `Remote` (48c, russh + ACP wire) behind
//! this same trait without touching the tools.
//!
//! Decisions where ADR 082 / the phase plan are silent (kept minimal for 48a):
//!
//! - **`exec` blocks until exit and returns captured output.** The ADR's
//!   streamed-progress exec is deliberately *not* frozen into the trait yet:
//!   the streaming frame shape is an open question the 48c wire spike resolves
//!   (plan §"Open questions"), and the in-process tool never streamed. The
//!   signature takes a struct ([`ExecRequest`]) so a streaming sink can be
//!   added without rewiring every call site.
//! - **`glob`/`grep` stay tool-side walks over the file verbs.** ADR 082 §2
//!   lists them as backend verbs so a remote backend can run the walk
//!   server-side; promoting them is a 48c concern (there is no win in-process,
//!   and doing it now would fork the walk logic the tools already test).
//! - **[`Posture`] carries only the confinement claim.** The full self-report
//!   (euid, groups, SELinux/AppArmor domain, capability bounding set, writable
//!   roots) lands with the 48f handshake work; in-process can only honestly
//!   say "unconfined".

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use miette::Diagnostic;
use thiserror::Error;

use crate::fs::{Filesystem, FsResult, LocalFs};

pub mod local_drop;
pub mod process_supervisor;

#[cfg(feature = "remote-executor")]
pub mod remote;

pub use local_drop::{ExecPath, LocalDropBackend, LocalDropConfig, LocalDropError};

pub use process_supervisor::{
    DrainedOutput, ProgressTracker, SupervisedExecution, SupervisorError, SupervisorResult,
};

/// Errors from process execution through a [`Backend`].
///
/// File-verb errors keep the existing [`crate::fs::FsError`] taxonomy so the
/// tools' error mapping (and their user-facing messages) stay byte-identical;
/// this enum covers only the exec verb.
#[derive(Debug, Error, Diagnostic)]
pub enum ExecError {
    /// The command process could not be spawned at all.
    #[error("failed to spawn shell: {source}")]
    #[diagnostic(
        code(akh::backend::spawn),
        help(
            "The backend could not start `/bin/sh`. Check that a POSIX shell exists \
             on the execution target and that the akh's user may execute it."
        )
    )]
    Spawn {
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// The command outlived its deadline and was killed.
    #[error("command timed out after {}s: {command}", timeout.as_secs())]
    #[diagnostic(
        code(akh::backend::timeout),
        help(
            "The process was killed at the deadline. Re-run with a larger `timeout` \
             if the command legitimately needs longer, or break the work into \
             smaller steps."
        )
    )]
    Timeout {
        /// The command that was killed.
        command: String,
        /// The deadline that expired.
        timeout: Duration,
    },

    /// The command was killed by the progress-based supervisor because it
    /// appeared stuck (no output for the configured duration). Unlike
    /// [`Timeout`](Self::Timeout), this carries partial output and the
    /// stuck-vs-timeout distinction so callers can distinguish a true hang
    /// from a slow-but-healthy build that was cut off.
    #[error("command stuck after {elapsed_secs}s: {reason} (command: {command})")]
    #[diagnostic(
        code(akh::backend::stuck),
        help(
            "The process was killed because it produced no output for an extended \
             period. This typically indicates a hang (e.g. a build script waiting \
             for input, a deadlock, or a network fetch that never returns). \
             Check the partial output for the last thing the process did. \
             Increase AKH_TOOL_PROGRESS_TIMEOUT_SECS if the operation legitimately \
             needs longer without output."
        )
    )]
    Stuck {
        /// The command that was killed.
        command: String,
        /// Human-readable reason (e.g. "no output for 300 seconds").
        reason: String,
        /// Partial stdout captured before termination.
        partial_stdout: Vec<u8>,
        /// Partial stderr captured before termination.
        partial_stderr: Vec<u8>,
        /// How long the process ran before being killed.
        elapsed_secs: u64,
    },

    /// Waiting on the child process failed.
    #[error("failed to wait on command: {source}")]
    #[diagnostic(
        code(akh::backend::wait),
        help("The OS refused to report the child's status; the process state is unknown.")
    )]
    Wait {
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// A remote (SSH) exec failed. Kept transparent so [`remote::RemoteError`]'s
    /// rich diagnostic (host, auth, timeout detail) reaches the caller
    /// unmodified — only [`remote::RemoteBackend`] constructs this variant.
    #[cfg(feature = "remote-executor")]
    #[error(transparent)]
    #[diagnostic(transparent)]
    Remote(#[from] remote::RemoteError),
}

/// Result alias for exec operations.
pub type ExecResult<T> = Result<T, ExecError>;

/// A request to execute a shell command on the backend's bound target.
///
/// Mirrors the plan's `exec(cmd, timeout, cwd, env)` signature as one struct
/// so 48c can extend it (streaming sink, cancellation token) without breaking
/// every call site.
#[derive(Debug, Clone)]
pub struct ExecRequest {
    /// The shell command line, run via the target's `/bin/sh -c`.
    pub command: String,
    /// Deadline after which the process is killed and
    /// [`ExecError::Timeout`] returned.
    pub timeout: Duration,
    /// Working directory for the child; `None` inherits the backend's own.
    pub cwd: Option<PathBuf>,
    /// Environment variables overlaid on the inherited environment.
    pub env: Vec<(String, String)>,
}

impl ExecRequest {
    /// A command with the given deadline, inheriting cwd and environment.
    pub fn shell(command: impl Into<String>, timeout: Duration) -> Self {
        Self {
            command: command.into(),
            timeout,
            cwd: None,
            env: Vec::new(),
        }
    }

    /// Run in the given working directory.
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Overlay an environment variable onto the inherited environment.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

/// The captured result of a completed (exited) command.
///
/// Output is raw bytes; presentation (lossy UTF-8, truncation) is the tool's
/// concern, so the backend never mangles what the process actually wrote.
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    /// Exit code; `-1` when the process died without one (e.g. by signal),
    /// matching the historic `status.code().unwrap_or(-1)` behavior.
    pub exit_code: i32,
    /// Whether the process reported success (exit status zero).
    pub success: bool,
    /// Captured standard output.
    pub stdout: Vec<u8>,
    /// Captured standard error.
    pub stderr: Vec<u8>,
}

/// The confinement a backend honestly claims for its executor.
///
/// 48a knows only the unconfined in-process case; 48b/48f add confined
/// variants refined by the executor's handshake self-report, and
/// `effective_danger = worst(assertion, posture)` (ADR 082 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confinement {
    /// Nothing separates the executor from the host: it runs as the akhomed
    /// process/user with full access. Tools riding this posture keep their
    /// `Critical` danger gating.
    Unconfined,

    /// The executor runs on this host as a distinct, less-privileged, already-
    /// provisioned local OS principal ([`local_drop::LocalDropBackend`], Phase
    /// 48b): akhomed keeps whatever privilege it needs for its own job, and
    /// every executed command has had its uid, gid **and supplementary groups**
    /// replaced before `exec`.
    ///
    /// Strictly stronger than [`Confinement::Unconfined`] — a real OS
    /// credential boundary, verified in the child before the command runs — and
    /// strictly weaker than a jail, zone or container: the target is an ordinary
    /// user, so it keeps that user's filesystem reach and unrestricted network
    /// access. Policy should treat it as "a different principal", not as "a
    /// sandbox".
    PrivilegeDropped,

    /// The executor authenticated as a distinct, already-provisioned remote
    /// OS principal over SSH ([`remote::RemoteBackend`], Phase 48c) rather
    /// than running as akhomed's own process identity — a different claim
    /// than [`Confinement::Unconfined`], but not yet the full self-reported
    /// posture (euid, groups, MAC domain, capability bounding set) 48f's
    /// handshake adds. Policy treats this identically to `Unconfined` until
    /// then (ADR 082 §8: "absent assertion ⇒ fail-safe to
    /// unconfined/Critical") — it exists as its own variant purely so 48e/48f
    /// can later attach different friction rules to "no boundary at all" vs.
    /// "a real remote login, boundary unverified".
    #[cfg(feature = "remote-executor")]
    RemoteUnverified,
}

/// An executor's self-reported posture, returned by [`Backend::handshake`].
///
/// Deliberately minimal in 48a — the full report (euid, groups, MAC domain,
/// capability bounding set, writable roots) is 48f scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posture {
    /// The confinement the backend claims for itself.
    pub confinement: Confinement,
}

/// Where a backend's exec verbs land — the locus half of the ADR 082 §7
/// audit envelope ("the audit ledger is a correlation index, not a duplicate,
/// over the native OS audit"). `backend`/`host`/`os_user` are the fields a
/// human or the Phase-50 benchmark grader joins against the native audit
/// trail (illumos `auditreduce`, sshd logs, …) to answer "who/where actually
/// ran this"; `correlation_id` is the stable per-backend-instance handle that
/// makes the join precise instead of fuzzy timestamp matching.
///
/// Owned (not `Copy`) because `host`/`os_user`/`correlation_id` are runtime
/// strings, not `'static` data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecLocus {
    /// Which `Backend` impl produced this locus — `"in-process"` or
    /// `"remote"` today; a future `LocalDrop` (48b) or additional remote
    /// transport would add its own tag.
    pub backend: &'static str,
    /// The host the exec actually ran on. `None` means localhost/unknown
    /// (the in-process backend has no reliable way to name itself without a
    /// new dependency — see [`InProcessBackend::locus`]).
    pub host: Option<String>,
    /// The OS principal the exec ran as.
    pub os_user: Option<String>,
    /// A stable identifier for this backend *instance* (not this one exec
    /// call) that the native OS audit trail can be joined against — in
    /// production this is the value that would populate the SSH certificate
    /// key-id (ADR 082 §7), so a correlation-index query can walk straight
    /// from an akh-medu audit entry to the matching sshd/auditreduce record.
    pub correlation_id: Option<String>,
}

/// The unified file + exec surface the executor-facing tools operate through
/// (ADR 082).
///
/// The supertrait carries the file verbs unchanged — implementors enforce the
/// same containment contract [`Filesystem`] documents — and this trait adds
/// the execution verbs. One `Backend` == one bound target: every verb, file or
/// exec, lands on the same host/sandbox/principal.
pub trait Backend: Filesystem {
    /// Execute a shell command on the bound target, blocking until it exits
    /// (or the request's deadline kills it) and returning captured output.
    fn exec(&self, request: ExecRequest) -> ExecResult<ExecOutcome>;

    /// The backend's posture self-report (ADR 082 §8). Callers must treat an
    /// [`Confinement::Unconfined`] report as Critical-danger territory.
    fn handshake(&self) -> Posture;

    /// The identity/locus envelope for this backend instance (ADR 082 §7,
    /// Phase 48f) — who and where would run any exec issued through it. Cheap
    /// and connection-free: callers (the audit ledger, `ToolRegistry::execute`)
    /// may call this on every tool invocation without an extra round trip.
    fn locus(&self) -> ExecLocus;
}

/// The in-process backend: today's behavior, verbatim (ADR 082 §10 "default =
/// in-process, unconfined, honestly `Critical`/gated").
///
/// File verbs delegate to the wrapped [`Filesystem`] sandbox (normally a
/// [`LocalFs`]); [`Backend::exec`] runs `/bin/sh -c` directly in the akhomed
/// process as the akhomed OS user — exactly what `shell_exec` historically
/// did, including the 50 ms poll loop and kill-on-deadline.
pub struct InProcessBackend {
    fs: Arc<dyn Filesystem>,
}

impl InProcessBackend {
    /// Wrap an existing filesystem sandbox with in-process execution.
    pub fn new(fs: Arc<dyn Filesystem>) -> Self {
        Self { fs }
    }

    /// An exec-only backend whose file verbs fail closed (denied sandbox).
    ///
    /// Matches the historic `ShellExecTool` posture: it could run commands but
    /// had no file surface of its own.
    pub fn exec_only() -> Self {
        Self::new(Arc::new(LocalFs::denied()))
    }

    /// Build the exact command used by [`Backend::exec`]. Keeping execution
    /// context construction separate makes cwd/env precedence deterministic
    /// to test without depending on host child-process capture.
    fn command_for(request: &ExecRequest) -> Command {
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(&request.command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(cwd) = &request.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &request.env {
            command.env(key, value);
        }
        command
    }
}

impl Filesystem for InProcessBackend {
    fn read_to_string(&self, path: &Path) -> FsResult<String> {
        self.fs.read_to_string(path)
    }

    fn write(&self, path: &Path, contents: &str) -> FsResult<()> {
        self.fs.write(path, contents)
    }

    fn read_dir(&self, path: &Path) -> FsResult<Vec<PathBuf>> {
        self.fs.read_dir(path)
    }

    fn exists(&self, path: &Path) -> bool {
        self.fs.exists(path)
    }

    fn read_roots(&self) -> &[PathBuf] {
        self.fs.read_roots()
    }

    fn is_writable(&self) -> bool {
        self.fs.is_writable()
    }

    fn file_size(&self, path: &Path) -> FsResult<u64> {
        self.fs.file_size(path)
    }

    fn is_symlink(&self, path: &Path) -> bool {
        self.fs.is_symlink(path)
    }
}

impl Backend for InProcessBackend {
    fn exec(&self, request: ExecRequest) -> ExecResult<ExecOutcome> {
        let command = Self::command_for(&request);

        // Spawn under supervision (includes process group setup for clean kill)
        let mut exec = SupervisedExecution::spawn(command).map_err(|e| match e {
            SupervisorError::Spawn { source } => ExecError::Spawn { source },
            _ => ExecError::Spawn {
                source: std::io::Error::new(std::io::ErrorKind::Other, e),
            },
        })?;

        // Progress tracker: use the request timeout as the stuck threshold.
        // A process that produces no output for `request.timeout` is stuck;
        // a process that continuously outputs is never killed.
        let mut tracker = ProgressTracker::with_timeout(request.timeout);

        // Take the drain threads — this decouples pipe reads from the
        // supervision loop so is_stuck() is checked on a timer regardless
        // of whether the child is producing output. This is the fix for
        // the silent-hang blocker: a child that writes nothing and never
        // exits (e.g. `sleep 3600` with piped stdout) will be detected as
        // stuck and killed, instead of blocking forever on a read().
        let drain = exec.take_drain();

        // Wait with drain-based supervision
        match exec.wait_with_drain(&mut tracker, drain) {
            Ok((status, output)) => Ok(ExecOutcome {
                exit_code: status.code().unwrap_or(-1),
                success: status.success(),
                stdout: output.stdout,
                stderr: output.stderr,
            }),
            Err(SupervisorError::Killed {
                reason,
                partial_stdout,
                partial_stderr,
                elapsed,
            }) => Err(ExecError::Stuck {
                command: request.command.clone(),
                reason,
                partial_stdout,
                partial_stderr,
                elapsed_secs: elapsed.as_secs(),
            }),
            Err(SupervisorError::Wait { source }) => Err(ExecError::Wait { source }),
            Err(SupervisorError::Spawn { source }) => Err(ExecError::Spawn { source }),
        }
    }

    fn handshake(&self) -> Posture {
        Posture {
            confinement: Confinement::Unconfined,
        }
    }

    /// `backend = "in-process"`; `host`/`os_user` come from the environment
    /// (`HOSTNAME`/`/etc/hostname`, `USER`/`LOGNAME`) rather than a `libc`
    /// call — this crate takes no new dependency just to name the box it is
    /// already running on. `correlation_id` is process-stable
    /// (`inproc-<pid>`): every tool call this akhomed process makes
    /// in-process shares one id, which is all a correlation-index join needs
    /// since there is no separate remote session to distinguish.
    fn locus(&self) -> ExecLocus {
        ExecLocus {
            backend: "in-process",
            host: local_hostname(),
            os_user: std::env::var("USER")
                .or_else(|_| std::env::var("LOGNAME"))
                .ok(),
            correlation_id: Some(format!("inproc-{}", std::process::id())),
        }
    }
}

/// Best-effort local hostname: `HOSTNAME` env var first (cheap, no syscall),
/// falling back to reading `/etc/hostname` (Linux/illumos convention). `None`
/// if neither is available — callers must treat that as "unknown", not an
/// error; naming the box is a nicety for the audit envelope, not a
/// requirement for it to function.
fn local_hostname() -> Option<String> {
    if let Ok(name) = std::env::var("HOSTNAME") {
        if !name.is_empty() {
            return Some(name);
        }
    }
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::FsError;

    fn sandboxed(dir: &Path) -> InProcessBackend {
        InProcessBackend::new(Arc::new(LocalFs::sandbox(dir)))
    }

    #[test]
    fn exec_captures_stdout_stderr_and_exit_code() {
        let backend = InProcessBackend::exec_only();
        let out = backend
            .exec(ExecRequest::shell(
                "echo out_line; echo err_line >&2; exit 3",
                Duration::from_secs(10),
            ))
            .unwrap();
        assert_eq!(out.exit_code, 3);
        assert!(!out.success);
        assert_eq!(String::from_utf8_lossy(&out.stdout), "out_line\n");
        assert_eq!(String::from_utf8_lossy(&out.stderr), "err_line\n");
    }

    #[test]
    fn exec_success_reports_zero_exit() {
        let backend = InProcessBackend::exec_only();
        let out = backend
            .exec(ExecRequest::shell("true", Duration::from_secs(10)))
            .unwrap();
        assert!(out.success);
        assert_eq!(out.exit_code, 0);
    }

    #[test]
    fn exec_timeout_kills_the_process() {
        let backend = InProcessBackend::exec_only();
        let err = backend
            .exec(ExecRequest::shell("sleep 60", Duration::from_secs(1)))
            .unwrap_err();
        // With drain-based supervision, a silent process is killed as
        // Stuck (not Timeout) — the distinction matters: Stuck means the
        // progress-based supervisor detected no output, Timeout was the
        // old elapsed-time deadline.
        match err {
            ExecError::Stuck {
                command, reason, ..
            } => {
                assert_eq!(command, "sleep 60");
                assert!(
                    reason.contains("stuck"),
                    "reason must mention stuck: {reason}"
                );
            }
            other => panic!("expected Stuck, got: {other:?}"),
        }
    }

    #[test]
    fn exec_applies_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let request = ExecRequest::shell("pwd", Duration::from_secs(10)).with_cwd(&canonical);
        let command = InProcessBackend::command_for(&request);

        assert_eq!(command.get_current_dir(), Some(canonical.as_path()));
    }

    #[test]
    fn exec_overlays_env() {
        let request = ExecRequest::shell("ignored", Duration::from_secs(10))
            .with_env("AKH_BACKEND_TEST", "forty-two");
        let command = InProcessBackend::command_for(&request);
        let value = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("AKH_BACKEND_TEST"))
            .and_then(|(_, value)| value);

        assert_eq!(value, Some(std::ffi::OsStr::new("forty-two")));
    }

    #[test]
    fn file_verbs_delegate_to_the_sandbox() {
        let dir = tempfile::tempdir().unwrap();
        let backend = sandboxed(dir.path());

        // In-sandbox write + read round-trips.
        backend
            .write(Path::new("note.txt"), "through the seam")
            .unwrap();
        let read = backend
            .read_to_string(&dir.path().join("note.txt"))
            .unwrap();
        assert_eq!(read, "through the seam");

        // Outside the sandbox: denied by the wrapped LocalFs, not by std::fs.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        let denied = backend.read_to_string(&outside.path().join("secret.txt"));
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "read", .. })),
            "backend must preserve sandbox denial: {denied:?}"
        );
        let denied_write = backend.write(Path::new("/tmp/akh-backend-evil.txt"), "nope");
        assert!(
            matches!(
                denied_write,
                Err(FsError::OutsideSandbox { kind: "write", .. })
            ),
            "backend must preserve write containment: {denied_write:?}"
        );
    }

    #[test]
    fn exec_only_backend_denies_all_file_access() {
        let backend = InProcessBackend::exec_only();
        assert!(!backend.is_writable());
        assert!(backend.read_roots().is_empty());
        assert!(backend.read_to_string(Path::new("/etc/hostname")).is_err());
        assert!(matches!(
            backend.write(Path::new("x"), "y"),
            Err(FsError::WritesDisabled)
        ));
        // ... while exec still works (the historic ShellExecTool posture).
        let out = backend
            .exec(ExecRequest::shell("echo alive", Duration::from_secs(10)))
            .unwrap();
        assert!(out.success);
    }

    #[test]
    fn handshake_reports_unconfined() {
        let backend = InProcessBackend::exec_only();
        assert_eq!(
            backend.handshake(),
            Posture {
                confinement: Confinement::Unconfined
            }
        );
    }

    #[test]
    fn locus_reports_in_process_with_a_stable_correlation_id() {
        let backend = InProcessBackend::exec_only();
        let locus = backend.locus();
        assert_eq!(locus.backend, "in-process");
        assert!(
            locus
                .correlation_id
                .as_deref()
                .is_some_and(|id| id.starts_with("inproc-") && id != "inproc-"),
            "expected a non-empty inproc-<pid> id, got {:?}",
            locus.correlation_id
        );
        // Calling twice on the same process must agree — it's a process-
        // stable id, not a fresh one per call.
        assert_eq!(locus.correlation_id, backend.locus().correlation_id);
    }
}
