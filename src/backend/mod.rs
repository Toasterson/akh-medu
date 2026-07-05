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

#[cfg(feature = "remote-executor")]
pub mod remote;

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

        let mut child = command
            .spawn()
            .map_err(|source| ExecError::Spawn { source })?;

        // Poll for completion with a timeout loop (historic shell_exec
        // behavior, preserved verbatim: 50 ms poll cadence, kill at deadline,
        // no output returned on timeout).
        let deadline = std::time::Instant::now() + request.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(ExecError::Timeout {
                            command: request.command,
                            timeout: request.timeout,
                        });
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(source) => return Err(ExecError::Wait { source }),
            }
        };

        // Read stdout and stderr after exit (historic behavior).
        let stdout = child
            .stdout
            .take()
            .and_then(|mut s| {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut s, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();

        let stderr = child
            .stderr
            .take()
            .and_then(|mut s| {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut s, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();

        Ok(ExecOutcome {
            exit_code: status.code().unwrap_or(-1),
            success: status.success(),
            stdout,
            stderr,
        })
    }

    fn handshake(&self) -> Posture {
        Posture {
            confinement: Confinement::Unconfined,
        }
    }
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
        match err {
            ExecError::Timeout { command, timeout } => {
                assert_eq!(command, "sleep 60");
                assert_eq!(timeout, Duration::from_secs(1));
            }
            other => panic!("expected Timeout, got: {other:?}"),
        }
    }

    #[test]
    fn exec_applies_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let canonical = dir.path().canonicalize().unwrap();
        let backend = InProcessBackend::exec_only();
        let out = backend
            .exec(
                ExecRequest::shell("pwd", Duration::from_secs(10))
                    .with_cwd(&canonical),
            )
            .unwrap();
        assert!(out.success);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            canonical.to_string_lossy()
        );
    }

    #[test]
    fn exec_overlays_env() {
        let backend = InProcessBackend::exec_only();
        let out = backend
            .exec(
                ExecRequest::shell("printf '%s' \"$AKH_BACKEND_TEST\"", Duration::from_secs(10))
                    .with_env("AKH_BACKEND_TEST", "forty-two"),
            )
            .unwrap();
        assert!(out.success);
        assert_eq!(String::from_utf8_lossy(&out.stdout), "forty-two");
    }

    #[test]
    fn file_verbs_delegate_to_the_sandbox() {
        let dir = tempfile::tempdir().unwrap();
        let backend = sandboxed(dir.path());

        // In-sandbox write + read round-trips.
        backend.write(Path::new("note.txt"), "through the seam").unwrap();
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
            matches!(denied_write, Err(FsError::OutsideSandbox { kind: "write", .. })),
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
}
