//! Privilege-dropping local executor — the Phase 48b `LocalDrop` backend
//! (ADR 082 §10).
//!
//! [`InProcessBackend`](super::InProcessBackend) runs commands as whatever
//! principal akhomed itself is. That is fine when akhomed runs as an
//! unprivileged user, and unacceptable when it runs as a system daemon: a
//! root-owned akhomed would hand every tool call root on the box.
//!
//! `LocalDropBackend` is the answer for the daemon deployment. akhomed keeps
//! the privileges it needs for its own job (binding low ports, reading
//! system-owned config and TLS material, surviving user logout) while every
//! *executed* command runs as an ordinary, already-provisioned OS user.
//!
//! # The drop
//!
//! Resolution happens in the **parent**, before `fork(2)`:
//!
//! - [`getpwnam_r(3)`] resolves the target name to uid, gid and home
//! - [`getgrouplist(3)`] resolves the full supplementary group list
//!
//! This ordering is deliberate and load-bearing. Both of those calls consult
//! the platform name service — `nsswitch`/NIS on illumos and Linux, and
//! **Open Directory** on macOS — which allocates, takes locks and may perform
//! IPC. None of that is async-signal-safe, so doing it after `fork` in a
//! multi-threaded process (which akhomed always is — tokio) risks deadlocking
//! the child against a lock held by a thread that does not exist any more.
//!
//! The child's `pre_exec` hook therefore performs only the three raw syscalls,
//! all of which *are* async-signal-safe, in the only order that works:
//!
//! 1. [`setgroups(2)`] — supplementary groups first, while still privileged
//! 2. [`setgid(2)`] — real+effective gid
//! 3. [`setuid(2)`] — real+effective uid, **last**: after this the process can
//!    no longer change any of its credentials, so a `setgid` afterwards would
//!    silently fail and leave the child in the daemon's group
//!
//! Skipping step 1 is the classic privilege-drop bug: `setuid` alone leaves the
//! child holding the daemon's supplementary groups (on a root daemon, that is
//! `wheel`/`admin`), so the "unprivileged" command keeps group access to
//! everything those groups reach.
//!
//! The hook then re-reads its own credentials and fails the exec if they are
//! not exactly what was asked for, so a partial drop can never be mistaken for
//! a complete one.
//!
//! # What this is and is not
//!
//! This is a *privilege* boundary, not a *sandbox*. The dropped-to user is a
//! normal user: it can read anything that user can read and talk to the
//! network. [`Confinement::PrivilegeDropped`] claims exactly that and no more —
//! it is strictly better than [`Confinement::Unconfined`] and strictly weaker
//! than a jail/zone/container, and policy should treat it accordingly.
//!
//! [`getpwnam_r(3)`]: https://man7.org/linux/man-pages/man3/getpwnam_r.3.html
//! [`getgrouplist(3)`]: https://man7.org/linux/man-pages/man3/getgrouplist.3.html
//! [`setgroups(2)`]: https://man7.org/linux/man-pages/man2/setgroups.2.html
//! [`setgid(2)`]: https://man7.org/linux/man-pages/man2/setgid.2.html
//! [`setuid(2)`]: https://man7.org/linux/man-pages/man2/setuid.2.html

use std::ffi::{CStr, CString, OsStr, OsString};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use super::{
    Backend, Confinement, ExecError, ExecLocus, ExecOutcome, ExecRequest, ExecResult, Posture,
    ProgressTracker, SupervisedExecution, SupervisorError, local_hostname,
};
use crate::fs::{Filesystem, FsResult, LocalFs};

/// Failures setting up a [`LocalDropBackend`].
///
/// All of these are construction-time: they are raised once when the workspace
/// starts, not per exec, so a misconfigured drop stops the akh from coming up
/// instead of silently running commands with the wrong identity.
#[derive(Debug, Error, Diagnostic)]
pub enum LocalDropError {
    /// The configured user does not exist in the platform name service.
    #[error("exec user `{user}` does not exist on this host")]
    #[diagnostic(
        code(akh::backend::local_drop::unknown_user),
        help(
            "`[exec] kind = \"local-drop\"` requires an already-provisioned OS user. \
             Check the spelling, and verify with `id {user}`. akh-medu deliberately \
             does not create the account: provisioning principals is the operator's \
             job, not the agent's."
        )
    )]
    UnknownUser {
        /// The name that failed to resolve.
        user: String,
    },

    /// The name service lookup itself failed.
    #[error("could not resolve exec user `{user}`: {source}")]
    #[diagnostic(
        code(akh::backend::local_drop::lookup),
        help(
            "The platform name service returned an error rather than 'no such user'. \
             On macOS this usually means Open Directory is unavailable; on illumos or \
             Linux, check nsswitch.conf and that any directory service is reachable."
        )
    )]
    Lookup {
        /// The name being resolved.
        user: String,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// akhomed lacks the privilege to change credentials.
    #[error("cannot drop privileges to `{user}` (uid {target_uid}): akhomed runs as uid {euid}")]
    #[diagnostic(
        code(akh::backend::local_drop::not_privileged),
        help(
            "Dropping to a different user requires privilege akhomed does not have. \
             Either run akhomed as a system daemon (root) so it can drop, or set \
             `[exec] kind = \"in-process\"` and run akhomed as the target user \
             directly — those are the two coherent deployments. Running as an \
             unprivileged user while asking for a drop to a *different* user cannot \
             work and is refused here rather than at the first tool call."
        )
    )]
    NotPrivileged {
        /// The configured target.
        user: String,
        /// The uid that target resolves to.
        target_uid: u32,
        /// akhomed's own effective uid.
        euid: u32,
    },

    /// The configured target is root, which would make the drop a no-op.
    #[error("refusing to use root (uid 0) as the exec user")]
    #[diagnostic(
        code(akh::backend::local_drop::refuses_root),
        help(
            "`local-drop` exists to run commands as a *less* privileged principal. \
             Targeting root would keep full privilege while claiming a boundary in \
             the posture handshake, which is worse than being honestly unconfined. \
             Pick an ordinary user, or use `kind = \"in-process\"` if you genuinely \
             intend commands to run with akhomed's own privileges."
        )
    )]
    RefusesRoot,
}

/// Result alias for local-drop setup.
pub type LocalDropResult<T> = Result<T, LocalDropError>;

/// An OS principal resolved to the raw credentials needed post-`fork`.
///
/// Everything here is plain integers and an owned path: resolution touched the
/// name service in the parent so the child never has to.
#[derive(Debug, Clone)]
struct Principal {
    name: String,
    uid: libc::uid_t,
    gid: libc::gid_t,
    home: PathBuf,
    /// Full supplementary group list from `getgrouplist(3)`, including the
    /// primary gid (which is how `initgroups(3)` would have built it).
    groups: Vec<libc::gid_t>,
}

/// How the `PATH` handed to executed commands is decided.
///
/// This is a real decision, not a detail. akhomed as a LaunchDaemon (or an SMF
/// service, or a systemd unit) is started by an init system with a deliberately
/// minimal environment: launchd's default is `/usr/bin:/bin:/usr/sbin:/sbin`,
/// with nothing from any user's login shell. Letting that leak to the child is
/// the file-side mistake in reverse — the command runs *as* the target user,
/// with the target user's `HOME`, and then cannot find the target user's
/// toolchain. `~/.cargo/bin/cargo` is the canonical casualty: present, owned by
/// the exec principal, pointed at by `HOME`, and invisible.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ExecPath {
    /// Build a `PATH` from the exec principal's own home plus the platform's
    /// conventional prefixes. The default, and what an interactive login for
    /// that user would approximate.
    #[default]
    Derived,
    /// Use exactly this value. For deployments that pin a toolchain, or that
    /// want a deliberately narrow `PATH`.
    Explicit(OsString),
    /// Inherit akhomed's own `PATH`. This is pre-Phase-48b behaviour and is
    /// almost always wrong for a daemon deployment; it exists so an operator
    /// can reproduce the old semantics knowingly rather than by accident.
    Inherit,
}

impl ExecPath {
    /// Resolve to the value to set, or `None` to leave the child's `PATH`
    /// inherited from akhomed.
    ///
    /// `home` is the *exec principal's* home as resolved by `getpwnam_r(3)` —
    /// not akhomed's — so the derived value tracks whoever commands actually
    /// run as.
    pub fn resolve(&self, home: &Path) -> Option<OsString> {
        match self {
            Self::Inherit => None,
            Self::Explicit(value) => Some(value.clone()),
            Self::Derived => Some(Self::derive(home)),
        }
    }

    /// Per-user tool directories first (so a rustup/pipx/npm shim wins over a
    /// system copy, which is what the user's own shell would do), then the
    /// platform prefixes, then the POSIX base that is always present.
    fn derive(home: &Path) -> OsString {
        let mut parts: Vec<PathBuf> = vec![
            home.join(".local/bin"),
            home.join(".cargo/bin"),
            home.join("go/bin"),
        ];

        // Where third-party packages land differs per platform and getting it
        // wrong is silent: the entry simply never matches.
        #[cfg(target_os = "macos")]
        parts.extend([
            // Apple silicon Homebrew, then the Intel/MacPorts location.
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/opt/homebrew/sbin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/usr/local/sbin"),
        ]);
        #[cfg(any(target_os = "illumos", target_os = "solaris"))]
        parts.extend([
            PathBuf::from("/opt/local/bin"),
            PathBuf::from("/opt/local/sbin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/usr/local/sbin"),
        ]);
        #[cfg(not(any(target_os = "macos", target_os = "illumos", target_os = "solaris")))]
        parts.extend([
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/usr/local/sbin"),
        ]);

        parts.extend([
            PathBuf::from("/usr/bin"),
            PathBuf::from("/bin"),
            PathBuf::from("/usr/sbin"),
            PathBuf::from("/sbin"),
        ]);

        let joined: Vec<&OsStr> = parts.iter().map(|p| p.as_os_str()).collect();
        std::env::join_paths(joined)
            .expect("derived PATH components contain no ':' — they are fixed literals and a home dir")
    }
}

/// Configuration for a [`LocalDropBackend`].
#[derive(Debug, Clone)]
pub struct LocalDropConfig {
    /// The OS user every executed command runs as. Must already exist.
    pub user: String,
    /// Working directory for executed commands when the request does not
    /// name one. Also the advisory read+write root for the file verbs.
    pub workdir: Option<PathBuf>,
    /// Shell used to interpret command lines. Defaults to `/bin/sh`.
    pub shell: PathBuf,
    /// `PATH` policy for executed commands. Defaults to [`ExecPath::Derived`].
    pub path: ExecPath,
}

impl LocalDropConfig {
    /// A drop to `user` with platform defaults.
    pub fn new(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            workdir: None,
            shell: PathBuf::from("/bin/sh"),
            path: ExecPath::default(),
        }
    }

    /// Set the default working directory / advisory filesystem root.
    pub fn with_workdir(mut self, workdir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(workdir.into());
        self
    }

    /// Override the shell used to interpret command lines.
    pub fn with_shell(mut self, shell: impl Into<PathBuf>) -> Self {
        self.shell = shell.into();
        self
    }

    /// Override the `PATH` policy for executed commands.
    pub fn with_path(mut self, path: ExecPath) -> Self {
        self.path = path;
        self
    }
}

/// Executes commands as a distinct, less-privileged OS user (Phase 48b).
///
/// File verbs delegate to the wrapped [`Filesystem`]; see
/// [`LocalDropBackend::write`] for how ownership of written files is kept
/// consistent with the exec principal.
pub struct LocalDropBackend {
    fs: Arc<dyn Filesystem>,
    principal: Principal,
    config: LocalDropConfig,
    /// Whether an actual credential change is required before `exec`.
    ///
    /// False when akhomed already runs as the target user — the developer
    /// deployment. Changing credentials is privileged even when the new values
    /// equal the current ones, so attempting the drop in that case fails
    /// (`EPERM`) for no benefit.
    needs_drop: bool,
}

/// Names the principal and the shell, and deliberately omits the wrapped
/// filesystem (`dyn Filesystem` is not `Debug`). Hand-written rather than
/// derived so a backend can appear in assertion output and error context.
impl std::fmt::Debug for LocalDropBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalDropBackend")
            .field("user", &self.principal.name)
            .field("uid", &self.principal.uid)
            .field("gid", &self.principal.gid)
            .field("shell", &self.config.shell)
            .field("workdir", &self.config.workdir)
            .finish_non_exhaustive()
    }
}

impl LocalDropBackend {
    /// Resolve `config.user` and bind a backend to it.
    ///
    /// Fails fast (see [`LocalDropError`]) rather than degrading to in-process
    /// execution: silently running as root what an operator asked to run as
    /// `build` is the one outcome this type exists to prevent.
    pub fn new(config: LocalDropConfig, fs: Arc<dyn Filesystem>) -> LocalDropResult<Self> {
        let principal = resolve_principal(&config.user)?;

        if principal.uid == 0 {
            return Err(LocalDropError::RefusesRoot);
        }

        // SAFETY: `geteuid` is always safe; it takes no arguments and cannot fail.
        let euid = unsafe { libc::geteuid() };
        if euid != 0 && euid != principal.uid {
            return Err(LocalDropError::NotPrivileged {
                user: config.user.clone(),
                target_uid: principal.uid,
                euid,
            });
        }

        // SAFETY: argument-less credential read.
        let needs_drop = unsafe { libc::getuid() } != principal.uid || euid != principal.uid;

        Ok(Self {
            fs,
            principal,
            config,
            needs_drop,
        })
    }

    /// Whether executing a command will actually change credentials.
    ///
    /// False when akhomed already runs as the target user, in which case the
    /// [`Confinement::PrivilegeDropped`] claim describes an identity that was
    /// never elevated rather than one that was lowered.
    pub fn drops_privilege(&self) -> bool {
        self.needs_drop
    }

    /// Bind to `user` with a filesystem sandbox rooted at `workdir`.
    ///
    /// Convenience for the daemon's wiring path: the advisory read+write root
    /// and the default cwd are the same directory, which is what a build host
    /// wants.
    pub fn with_workspace_root(
        user: impl Into<String>,
        workdir: impl Into<PathBuf>,
    ) -> LocalDropResult<Self> {
        let workdir = workdir.into();
        let config = LocalDropConfig::new(user).with_workdir(&workdir);
        let fs = Arc::new(LocalFs::sandbox(workdir));
        Self::new(config, fs)
    }

    /// The uid commands run as.
    pub fn uid(&self) -> u32 {
        self.principal.uid
    }

    /// The gid commands run as.
    pub fn gid(&self) -> u32 {
        self.principal.gid
    }

    /// The user name commands run as.
    pub fn user(&self) -> &str {
        &self.principal.name
    }

    /// Give `path` to the exec principal, so a file akhomed wrote is one the
    /// executed commands can subsequently modify.
    ///
    /// Best-effort by design: a failure here is not an error the caller can act
    /// on (the write itself succeeded), and on a correctly-configured host it
    /// cannot happen. It is logged rather than propagated.
    fn chown_to_principal(&self, path: &Path) {
        let Ok(c_path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
            return;
        };
        // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
        // call; uid/gid are plain integers. `chown` reports failure by return
        // value and cannot violate memory safety.
        let rc = unsafe { libc::chown(c_path.as_ptr(), self.principal.uid, self.principal.gid) };
        if rc != 0 {
            tracing::debug!(
                target: "akh::backend::local_drop",
                path = %path.display(),
                user = %self.principal.name,
                error = %std::io::Error::last_os_error(),
                "could not chown written file to the exec principal"
            );
        }
    }
}

impl Filesystem for LocalDropBackend {
    fn read_to_string(&self, path: &Path) -> FsResult<String> {
        self.fs.read_to_string(path)
    }

    /// Writes through the sandbox, then hands the file to the exec principal.
    ///
    /// Without the `chown`, a root-daemon deployment produces root-owned files
    /// inside a workspace the dropped commands then cannot modify — the failure
    /// mode is a build that works once and fails on every incremental rerun.
    fn write(&self, path: &Path, contents: &str) -> FsResult<()> {
        self.fs.write(path, contents)?;
        self.chown_to_principal(path);
        Ok(())
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

impl Backend for LocalDropBackend {
    fn exec(&self, request: ExecRequest) -> ExecResult<ExecOutcome> {
        let mut command = Command::new(&self.config.shell);
        command
            .arg("-c")
            .arg(&request.command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Request cwd wins; otherwise the configured workdir; otherwise inherit.
        if let Some(cwd) = request.cwd.as_ref().or(self.config.workdir.as_ref()) {
            command.current_dir(cwd);
        }

        // The child is a different user, so the daemon's own HOME/USER would be
        // actively misleading — cargo, git and ssh all key off them and would
        // read (or try to write) the daemon's dotfiles. Set them to the target's
        // before applying caller overrides, so an explicit `env` still wins.
        command
            .env("HOME", &self.principal.home)
            .env("USER", &self.principal.name)
            .env("LOGNAME", &self.principal.name);
        // `PATH` belongs to the same set: akhomed's own is the init system's
        // minimal default (launchd hands down `/usr/bin:/bin:/usr/sbin:/sbin`
        // and nothing else), so inheriting it gives the child a `HOME` full of
        // tools it cannot resolve. Resolved from the *principal's* home.
        if let Some(path) = self.config.path.resolve(&self.principal.home) {
            command.env("PATH", path);
        }
        for (key, value) in &request.env {
            command.env(key, value);
        }

        // Everything the child needs, resolved here in the parent: the hook
        // below must not touch the name service (see the module docs).
        let uid = self.principal.uid;
        let gid = self.principal.gid;
        let groups = self.principal.groups.clone();

        // When akhomed already *is* the target user there is nothing to drop,
        // and attempting it would fail: changing credentials — even to the ones
        // already held — is privileged. Skipping keeps the unprivileged
        // deployment (akhomed running as the same user it executes as) working
        // through the identical code path, so behaviour does not diverge
        // between "daemon drops to build user" and "developer runs it as
        // themselves".
        if self.needs_drop {
            // SAFETY: `pre_exec` runs between fork and exec in the child, where
            // only async-signal-safe calls are permitted. This closure performs
            // exactly three raw syscalls plus three credential reads, allocates
            // nothing, and takes no locks: `groups` was built and truncated in
            // the parent and is only read through `as_ptr`/`len` here. The
            // ordering (setgroups, setgid, setuid) is required — setuid last,
            // because it is the one that makes the credentials immutable.
            unsafe {
                command.pre_exec(move || {
                    if libc::setgroups(groups.len() as _, groups.as_ptr()) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::setgid(gid) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::setuid(uid) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    // Refuse to exec on a partial drop. If any of the above
                    // succeeded in a way that did not actually change the
                    // credentials, running the command anyway would execute it
                    // with more privilege than promised.
                    if libc::getuid() != uid || libc::geteuid() != uid || libc::getgid() != gid {
                        return Err(std::io::Error::other(
                            "privilege drop did not take effect; refusing to exec",
                        ));
                    }
                    Ok(())
                });
            }
        }

        let mut exec = SupervisedExecution::spawn(command).map_err(|e| match e {
            SupervisorError::Spawn { source } => ExecError::Spawn { source },
            other => ExecError::Spawn {
                source: std::io::Error::other(other),
            },
        })?;

        let mut tracker = ProgressTracker::with_timeout(request.timeout);
        let drain = exec.take_drain();

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
            confinement: Confinement::PrivilegeDropped,
        }
    }

    /// `backend = "local-drop"`, running on this host as the dropped-to user.
    ///
    /// `correlation_id` is `localdrop-<pid>-<uid>`: the akhomed pid ties entries
    /// to the daemon instance, and the uid is what the platform's own audit
    /// trail (`auditreduce`, `execsnoop`, the macOS unified log) records for the
    /// child — so the join from an akh-medu audit entry to the OS record is on
    /// a field both sides actually have.
    fn locus(&self) -> ExecLocus {
        ExecLocus {
            backend: "local-drop",
            host: local_hostname(),
            os_user: Some(self.principal.name.clone()),
            correlation_id: Some(format!(
                "localdrop-{}-{}",
                std::process::id(),
                self.principal.uid
            )),
        }
    }
}

/// Resolve a user name to uid, gid, home and supplementary groups.
///
/// Uses the reentrant `getpwnam_r` rather than `getpwnam` because akhomed is
/// multi-threaded and the non-reentrant form returns a pointer into a shared
/// static that another thread can overwrite between the call and the read.
fn resolve_principal(user: &str) -> LocalDropResult<Principal> {
    let c_user = CString::new(user).map_err(|_| LocalDropError::UnknownUser {
        user: user.to_string(),
    })?;

    // `sysconf(_SC_GETPW_R_SIZE_MAX)` is a hint, not a bound: it returns -1 on
    // some platforms and can be too small on directory-backed systems (macOS
    // Open Directory, LDAP). Grow on ERANGE instead of trusting it once.
    let mut buf_len = match unsafe { libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) } {
        n if n > 0 => n as usize,
        _ => 1024,
    };

    let (uid, gid, home) = loop {
        let mut buf = vec![0i8; buf_len];
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::passwd = std::ptr::null_mut();

        // SAFETY: all pointers are to live local storage of the stated sizes,
        // and `getpwnam_r` writes only within them. `result` is set to either
        // null (not found) or `&mut pwd`.
        let rc = unsafe {
            libc::getpwnam_r(
                c_user.as_ptr(),
                &mut pwd,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf_len,
                &mut result,
            )
        };

        if rc == libc::ERANGE {
            buf_len = buf_len.saturating_mul(2);
            if buf_len > 1 << 20 {
                return Err(LocalDropError::Lookup {
                    user: user.to_string(),
                    source: std::io::Error::from_raw_os_error(libc::ERANGE),
                });
            }
            continue;
        }
        if rc != 0 {
            return Err(LocalDropError::Lookup {
                user: user.to_string(),
                source: std::io::Error::from_raw_os_error(rc),
            });
        }
        if result.is_null() {
            return Err(LocalDropError::UnknownUser {
                user: user.to_string(),
            });
        }

        // SAFETY: `result` is non-null, so `pwd` was populated and `pw_dir`
        // points into `buf`, which is still alive here. The bytes are copied
        // out before `buf` is dropped at the end of this iteration.
        let home = unsafe {
            if pwd.pw_dir.is_null() {
                PathBuf::from("/")
            } else {
                PathBuf::from(
                    String::from_utf8_lossy(CStr::from_ptr(pwd.pw_dir).to_bytes()).into_owned(),
                )
            }
        };
        break (pwd.pw_uid, pwd.pw_gid, home);
    };

    let groups = resolve_groups(&c_user, gid);

    Ok(Principal {
        name: user.to_string(),
        uid,
        gid,
        home,
        groups,
    })
}

/// Full supplementary group list for `user`, via `getgrouplist(3)`.
///
/// Falls back to just the primary gid if the lookup fails: a child with only
/// its primary group is *under*-privileged, which fails safe. Returning the
/// daemon's groups instead would fail open, so that is never done.
fn resolve_groups(c_user: &CStr, gid: libc::gid_t) -> Vec<libc::gid_t> {
    // macOS types this parameter as `i32` and Linux as `gid_t`; the local
    // binding lets the cast below satisfy both without a cfg.
    let mut count: libc::c_int = 32;
    loop {
        let mut buf: Vec<libc::gid_t> = vec![0; count.max(1) as usize];

        // SAFETY: `buf` has `count` elements and `getgrouplist` writes at most
        // that many, updating `count` with the number written (or the number
        // required, when it returns -1).
        let rc = unsafe {
            libc::getgrouplist(
                c_user.as_ptr(),
                gid as _,
                buf.as_mut_ptr() as *mut _,
                &mut count,
            )
        };

        if rc >= 0 {
            buf.truncate(count.max(0) as usize);
            if !buf.contains(&gid) {
                buf.push(gid);
            }
            return cap_to_ngroups_max(buf, gid);
        }

        // -1 means the buffer was too small; `count` now holds what is needed.
        // Guard against an implementation that does not update it.
        if count as usize <= buf.len() {
            return vec![gid];
        }
        if count > 1024 {
            return vec![gid];
        }
    }
}

/// Trim a group list to what `setgroups(2)` will actually accept, keeping the
/// primary gid.
///
/// **This is the macOS portability point.** `getgrouplist(3)` happily reports
/// every group a user belongs to, and on macOS that is routinely more than
/// `NGROUPS_MAX` (16) — a plain desktop account is typically in `staff`,
/// `everyone`, `localaccounts`, `admin`, `_appserverusr`, `_appserveradm`,
/// `_lpadmin`, `_developer`, `com.apple.access_ssh` and a dozen more. Passing
/// that full list to `setgroups` fails with **`EINVAL`**, and because the call
/// happens in `pre_exec` the failure surfaces as an opaque
/// `Spawn { InvalidInput }` on every single command.
///
/// Linux's limit is 65536 and illumos' is 16 by default but tunable, so the
/// limit is read from `sysconf(3)` rather than hardcoded per platform.
///
/// The primary gid is placed **first** so it survives truncation: it is the gid
/// that file ownership and most access checks actually key off, and losing it
/// to an arbitrary cut would be the one loss that breaks the deployment.
/// Dropping trailing supplementary groups leaves the child with *less* access,
/// which fails safe.
fn cap_to_ngroups_max(groups: Vec<libc::gid_t>, gid: libc::gid_t) -> Vec<libc::gid_t> {
    // SAFETY: `sysconf` takes an integer name and returns a long; no pointers.
    let max = match unsafe { libc::sysconf(libc::_SC_NGROUPS_MAX) } {
        n if n > 0 => n as usize,
        _ => 16, // conservative POSIX floor when the limit is indeterminate
    };
    if groups.len() <= max {
        return groups;
    }
    let mut capped = Vec::with_capacity(max);
    capped.push(gid);
    for g in groups {
        if capped.len() >= max {
            break;
        }
        if g != gid {
            capped.push(g);
        }
    }
    tracing::debug!(
        target: "akh::backend::local_drop",
        kept = capped.len(),
        limit = max,
        "supplementary group list truncated to the platform NGROUPS_MAX"
    );
    capped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_user() -> Option<String> {
        std::env::var("USER").ok().or_else(|| std::env::var("LOGNAME").ok())
    }

    /// Why an end-to-end exec test cannot run in this environment.
    ///
    /// Every variant is an environment fact, not a defect in the code under
    /// test. The one that matters is [`Self::ProbeSilent`]: a sandboxed test
    /// harness (and some CI runners) lets `LocalDropBackend::new` succeed and
    /// then returns `Ok` from `exec` with an empty capture, so an assertion on
    /// that capture fails for reasons entirely unrelated to the change being
    /// tested.
    #[derive(Debug)]
    enum ExecUnavailable {
        /// Neither `USER` nor `LOGNAME` is set — there is no principal to drop to.
        NoPrincipal,
        /// The backend refused to construct for this principal.
        BackendRejected(LocalDropError),
        /// The backend constructed, but the probe could not be spawned at all.
        ProbeFailed(ExecError),
        /// The probe ran and produced nothing: the environment swallowed it.
        ProbeSilent { exit_code: i32, success: bool },
    }

    impl std::fmt::Display for ExecUnavailable {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::NoPrincipal => {
                    write!(f, "neither USER nor LOGNAME is set, so there is no principal to drop to")
                }
                Self::BackendRejected(e) => {
                    write!(f, "the backend would not construct for this principal: {e}")
                }
                Self::ProbeFailed(e) => write!(f, "the probe command could not be spawned: {e}"),
                Self::ProbeSilent { exit_code, success } => write!(
                    f,
                    "the probe ran (exit_code={exit_code}, success={success}) but captured no \
                     stdout — this environment cannot observe a dropped child's output"
                ),
            }
        }
    }

    /// Construct a backend only when this environment can genuinely drop to the
    /// principal *and* observe the resulting child's output.
    ///
    /// Returns `None` **and prints why**. A bare `return` would make a skipped
    /// test indistinguishable from a passing one, which is exactly how
    /// `exec_puts_the_principals_bin_dirs_on_path_not_the_daemons` came to red
    /// the entire unfiltered `cargo test --features postgres --lib` suite — and
    /// with it every PR in the repo — without anyone noticing the cause was
    /// environmental rather than a real regression.
    ///
    /// This deliberately does **not** weaken coverage of the PATH policy: the
    /// derivation itself is exercised unconditionally, as a pure function over
    /// a fixture principal, by
    /// [`derived_path_leads_with_the_principals_own_tool_dirs`] and
    /// [`explicit_and_inherit_path_policies_are_distinguishable`]. Only the
    /// end-to-end confirmation is environment-gated.
    fn exec_capable_backend(test_name: &str) -> Option<LocalDropBackend> {
        match probe_exec_environment() {
            Ok(backend) => Some(backend),
            Err(reason) => {
                eprintln!("SKIP {test_name}: {reason}");
                None
            }
        }
    }

    /// Build a backend and prove it can run a trivial command that writes to stdout.
    fn probe_exec_environment() -> Result<LocalDropBackend, ExecUnavailable> {
        let user = current_user().ok_or(ExecUnavailable::NoPrincipal)?;
        let backend = LocalDropBackend::new(LocalDropConfig::new(&user), Arc::new(LocalFs::denied()))
            .map_err(ExecUnavailable::BackendRejected)?;
        let probe = backend
            .exec(ExecRequest::shell(
                "printf 'probe'",
                std::time::Duration::from_secs(30),
            ))
            .map_err(ExecUnavailable::ProbeFailed)?;
        if probe.stdout.is_empty() {
            return Err(ExecUnavailable::ProbeSilent {
                exit_code: probe.exit_code,
                success: probe.success,
            });
        }
        Ok(backend)
    }

    #[test]
    fn unknown_user_is_rejected_at_construction_not_at_exec() {
        let err = LocalDropBackend::new(
            LocalDropConfig::new("no-such-user-akh-medu-test-9f3a"),
            Arc::new(LocalFs::denied()),
        )
        .expect_err("a nonexistent user must not produce a working backend");
        assert!(matches!(err, LocalDropError::UnknownUser { .. }));
    }

    #[test]
    fn root_is_refused_as_a_drop_target() {
        // Targeting root would keep full privilege while claiming a boundary.
        let err = LocalDropBackend::new(
            LocalDropConfig::new("root"),
            Arc::new(LocalFs::denied()),
        )
        .expect_err("root must never be accepted as the drop target");
        assert!(matches!(err, LocalDropError::RefusesRoot));
    }

    #[test]
    fn resolving_the_current_user_yields_its_real_credentials() {
        let Some(user) = current_user() else {
            return; // no USER in this environment; nothing to assert against
        };
        let Ok(principal) = resolve_principal(&user) else {
            return; // container/CI images sometimes have no passwd entry
        };
        // SAFETY: argument-less credential reads.
        assert_eq!(principal.uid, unsafe { libc::getuid() });
        assert!(
            principal.groups.contains(&principal.gid),
            "group list must always include the primary gid"
        );
    }

    #[test]
    fn dropping_to_self_is_allowed_even_unprivileged() {
        // The euid == target case is the one deployment an unprivileged akhomed
        // can serve, and it must not be refused.
        let Some(user) = current_user() else { return };
        // SAFETY: argument-less credential read.
        if unsafe { libc::getuid() } == 0 {
            return; // running as root: this assertion is about the non-root path
        }
        let Ok(backend) =
            LocalDropBackend::new(LocalDropConfig::new(&user), Arc::new(LocalFs::denied()))
        else {
            return; // no passwd entry for USER in this environment
        };
        assert_eq!(backend.user(), user);
        assert!(matches!(
            backend.handshake().confinement,
            Confinement::PrivilegeDropped
        ));
    }

    #[test]
    fn dropping_to_another_user_without_privilege_is_refused_up_front() {
        // SAFETY: argument-less credential read.
        if unsafe { libc::geteuid() } == 0 {
            return; // as root the drop is permitted, so there is nothing to refuse
        }
        // `nobody` exists on macOS, Linux and illumos alike and is never us.
        match LocalDropBackend::new(
            LocalDropConfig::new("nobody"),
            Arc::new(LocalFs::denied()),
        ) {
            Err(LocalDropError::NotPrivileged { .. }) => {}
            Err(LocalDropError::UnknownUser { .. }) => {} // no `nobody` here
            Err(other) => panic!("expected NotPrivileged, got {other:?}"),
            Ok(_) => panic!("an unprivileged akhomed must not be able to drop to another user"),
        }
    }

    #[test]
    fn locus_names_the_backend_and_the_principal_it_runs_as() {
        let Some(user) = current_user() else { return };
        let Ok(backend) =
            LocalDropBackend::new(LocalDropConfig::new(&user), Arc::new(LocalFs::denied()))
        else {
            return;
        };
        let locus = backend.locus();
        assert_eq!(locus.backend, "local-drop");
        assert_eq!(locus.os_user.as_deref(), Some(user.as_str()));
        assert!(
            locus
                .correlation_id
                .as_deref()
                .is_some_and(|id| id.starts_with("localdrop-")),
            "correlation id must be joinable against the OS audit trail"
        );
    }

    #[test]
    fn exec_runs_as_the_target_user_and_reports_its_identity() {
        let Some(backend) =
            exec_capable_backend("exec_runs_as_the_target_user_and_reports_its_identity")
        else {
            return;
        };
        let outcome = backend
            .exec(ExecRequest::shell("id -u", std::time::Duration::from_secs(30)))
            .expect("`id -u` must run");
        assert!(outcome.success, "stderr: {}", String::from_utf8_lossy(&outcome.stderr));
        let reported: u32 = String::from_utf8_lossy(&outcome.stdout)
            .trim()
            .parse()
            .expect("`id -u` prints a number");
        assert_eq!(
            reported,
            backend.uid(),
            "the child must run as the configured principal, not as akhomed"
        );
    }

    #[test]
    fn exec_overrides_home_so_child_tools_do_not_read_the_daemons_dotfiles() {
        let Some(backend) = exec_capable_backend(
            "exec_overrides_home_so_child_tools_do_not_read_the_daemons_dotfiles",
        ) else {
            return;
        };
        let outcome = backend
            .exec(ExecRequest::shell(
                "printf '%s' \"$HOME\"",
                std::time::Duration::from_secs(30),
            ))
            .expect("printf must run");
        let home = String::from_utf8_lossy(&outcome.stdout).into_owned();
        assert_eq!(home, backend.principal.home.display().to_string());
    }

    #[test]
    fn explicit_env_still_wins_over_the_principal_defaults() {
        let Some(backend) =
            exec_capable_backend("explicit_env_still_wins_over_the_principal_defaults")
        else {
            return;
        };
        let outcome = backend
            .exec(
                ExecRequest::shell(
                    "printf '%s' \"$HOME\"",
                    std::time::Duration::from_secs(30),
                )
                .with_env("HOME", "/tmp/akh-explicit-home"),
            )
            .expect("printf must run");
        assert_eq!(
            String::from_utf8_lossy(&outcome.stdout),
            "/tmp/akh-explicit-home"
        );
    }

    #[test]
    fn derived_path_leads_with_the_principals_own_tool_dirs() {
        let path = ExecPath::Derived
            .resolve(Path::new("/home/build"))
            .expect("Derived always resolves to a value");
        let rendered = path.to_string_lossy().into_owned();
        let entries: Vec<&str> = rendered.split(':').collect();

        // The user's own toolchain must precede any system copy, which is what
        // that user's login shell would do — this is the whole point of the
        // policy: `cargo` lives in `~/.cargo/bin` and nowhere else.
        assert_eq!(entries[0], "/home/build/.local/bin");
        assert!(entries.contains(&"/home/build/.cargo/bin"));
        let cargo_at = entries.iter().position(|e| *e == "/home/build/.cargo/bin");
        let usr_bin_at = entries.iter().position(|e| *e == "/usr/bin");
        assert!(
            cargo_at < usr_bin_at,
            "principal tool dirs must precede the system base, got: {rendered}"
        );

        // The POSIX base is always reachable, or the child cannot run `sh`
        // builtins' external counterparts.
        for base in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"] {
            assert!(entries.contains(&base), "missing {base} in: {rendered}");
        }
    }

    #[test]
    fn explicit_and_inherit_path_policies_are_distinguishable() {
        let home = Path::new("/home/build");
        assert_eq!(
            ExecPath::Explicit("/opt/toolchain/bin:/usr/bin".into()).resolve(home),
            Some(OsString::from("/opt/toolchain/bin:/usr/bin")),
            "an explicit PATH must be passed through verbatim"
        );
        assert_eq!(
            ExecPath::Inherit.resolve(home),
            None,
            "Inherit must set nothing so the child keeps akhomed's PATH"
        );
    }

    #[test]
    fn exec_puts_the_principals_bin_dirs_on_path_not_the_daemons() {
        let Some(backend) =
            exec_capable_backend("exec_puts_the_principals_bin_dirs_on_path_not_the_daemons")
        else {
            return;
        };
        let outcome = backend
            .exec(ExecRequest::shell(
                "printf '%s' \"$PATH\"",
                std::time::Duration::from_secs(30),
            ))
            .expect("printf must run");
        let path = String::from_utf8_lossy(&outcome.stdout).into_owned();
        let expected_cargo = backend.principal.home.join(".cargo/bin");
        assert!(
            path.split(':').any(|e| Path::new(e) == expected_cargo),
            "the exec principal's ~/.cargo/bin must be on PATH — a daemon-inherited \
             PATH is why `cargo` was unfindable despite HOME being correct. got: {path}"
        );
    }

    #[test]
    fn explicit_env_path_still_wins_over_the_derived_default() {
        let Some(backend) =
            exec_capable_backend("explicit_env_path_still_wins_over_the_derived_default")
        else {
            return;
        };
        let outcome = backend
            .exec(
                ExecRequest::shell(
                    "printf '%s' \"$PATH\"",
                    std::time::Duration::from_secs(30),
                )
                .with_env("PATH", "/opt/pinned/bin"),
            )
            .expect("printf must run");
        assert_eq!(String::from_utf8_lossy(&outcome.stdout), "/opt/pinned/bin");
    }
}
