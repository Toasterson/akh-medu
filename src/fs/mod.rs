//! Filesystem abstraction with a sandboxed local backend.
//!
//! Tools that touch the filesystem go through a [`Filesystem`] rather than raw
//! `std::fs`, so path containment is enforced in one audited place. The
//! [`LocalFs`] backend confines reads and writes to explicit allow-lists of
//! roots with symlink-safe containment checks, closing two historical holes in
//! `FileIoTool`:
//!
//! 1. the read path applied **no** containment (it would `read_to_string` any
//!    caller-supplied path — `/etc/passwd`, `~/.ssh/id_rsa`, …), and
//! 2. the write path canonicalized only the **parent** and fell back to a
//!    non-canonical prefix compare, so a symlinked leaf or a `..` tail escaped.
//!
//! Read-roots and write-roots are **separate**: an akh can read across its
//! whole workspace while writes stay confined to a narrower scratch area, and
//! neither defaults to "anywhere". A future `RemoteFs` backend (P4.5) will
//! implement the same trait by delegating to the ACP client's
//! `fs/read_text_file` / `fs/write_text_file` for remote workspaces.

use std::cell::RefCell;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

thread_local! {
    /// The active ACP session's working directory for the current thread, if
    /// any. See [`set_session_cwd`].
    static SESSION_CWD: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// RAII handle from [`set_session_cwd`]. Restores the thread's previous
/// session cwd (`None` in the common case) on drop, so the scope that
/// activated a session cwd — normally one ACP prompt turn — can never leave
/// a stale value behind for whatever runs next on this pooled thread.
pub struct SessionCwdGuard {
    previous: Option<PathBuf>,
}

impl Drop for SessionCwdGuard {
    fn drop(&mut self) {
        SESSION_CWD.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

/// Set the *thread-local* ACP session working directory used to resolve
/// relative paths in [`LocalFs::resolve_read`]/[`resolve_write`], and read
/// directly by `shell_exec` for the executed command's `cwd`. Active until
/// the returned guard drops, which restores whatever was set before (`None`
/// in the common case).
///
/// # Why thread-local is safe here
///
/// Every ACP prompt turn runs entirely inside one
/// `tokio::task::spawn_blocking` closure, on one dedicated OS thread, for the
/// turn's whole duration (`AkhAcpAgent::run_turn`) — tokio's blocking pool
/// never interleaves two blocking tasks on the same thread, even though it
/// happily runs many such closures concurrently on *different* threads.
/// Concurrent ACP sessions on one workspace are real (the `Agent` lock is a
/// `RwLock`, and the agentic turn path takes only a read lock, so independent
/// sessions' turns proceed in parallel — see `run_turn`'s doc comment), but
/// each one gets its own OS thread for its own turn. Setting this once at the
/// top of that closure and dropping the guard before the closure returns (and
/// the thread goes back to the blocking pool) therefore isolates concurrent
/// sessions correctly: no thread ever observes another session's cwd.
pub fn set_session_cwd(cwd: Option<PathBuf>) -> SessionCwdGuard {
    let previous = SESSION_CWD.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), cwd));
    SessionCwdGuard { previous }
}

/// The current thread's active session cwd, set by [`set_session_cwd`].
/// `None` when no session cwd is active — the case for every caller that
/// predates ACP session-cwd threading (tests, the CLI, a non-ACP embed, or an
/// ACP client that never sent a real `cwd`), so those paths are entirely
/// unaffected by this mechanism.
pub fn session_cwd() -> Option<PathBuf> {
    SESSION_CWD.with(|cell| cell.borrow().clone())
}

/// Resolve a possibly-relative path against a base, in priority order:
///
/// 1. An absolute `path` passes through unchanged.
/// 2. Otherwise, the thread-local session cwd ([`session_cwd`]), if one is
///    active — this is what makes relative reads and writes share one base
///    once a session cwd is threaded in (fixing the historical read/write
///    asymmetry: read had no defined base at all, write always used
///    `write_roots[0]`).
/// 3. Otherwise, `default_base` — each call site's own historical default.
///    `resolve_read` passes `None`, which leaves `path` for
///    [`Path::canonicalize`] to resolve against the process's actual working
///    directory exactly as before this change; `resolve_write` passes
///    `write_roots[0]`, also exactly as before. This is what keeps the
///    no-session-cwd behavior of each verb byte-identical to today.
fn resolve_relative(path: &Path, default_base: Option<&Path>) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Some(base) = session_cwd() {
        return base.join(path);
    }
    match default_base {
        Some(base) => base.join(path),
        None => path.to_path_buf(),
    }
}

/// Errors from filesystem operations behind the sandbox.
#[derive(Debug, Error, Diagnostic)]
pub enum FsError {
    /// The resolved path lies outside every allowed root for the operation.
    #[error("path \"{path}\" is outside the allowed {kind} roots")]
    #[diagnostic(
        code(akh::fs::outside_sandbox),
        help(
            "Only paths within the akh's sandbox roots are permitted ({roots:?}). Use a path inside the workspace."
        )
    )]
    OutsideSandbox {
        /// The offending path (as requested).
        path: String,
        /// `"read"` or `"write"`.
        kind: &'static str,
        /// The roots the path was checked against.
        roots: Vec<PathBuf>,
    },

    /// A write path contained a `..` component (traversal attempt).
    #[error("path \"{path}\" contains a parent-directory (`..`) component")]
    #[diagnostic(
        code(akh::fs::traversal),
        help("Provide a path without `..` segments; sandbox writes reject traversal.")
    )]
    Traversal {
        /// The offending path.
        path: String,
    },

    /// The filesystem has no writable roots (e.g. an in-memory engine).
    #[error("writes are disabled for this akh's filesystem")]
    #[diagnostic(
        code(akh::fs::writes_disabled),
        help(
            "This filesystem has no writable roots configured. Run the akh with a data directory to enable writes."
        )
    )]
    WritesDisabled,

    /// An underlying `std::io` error (not found, permission, …).
    #[error("filesystem error on \"{path}\": {source}")]
    #[diagnostic(code(akh::fs::io))]
    Io {
        /// The path the operation targeted.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// An akhsbxd/akhexecd file verb failed (lease, transport, a definite
    /// executor failure or an unknown outcome). Transparent so the execd
    /// diagnostic reaches the caller; a missing file still maps to
    /// [`FsError::Io`] with `NotFound`, and a path outside the workspace to
    /// [`FsError::OutsideSandbox`].
    #[cfg(feature = "execd-executor")]
    #[error(transparent)]
    #[diagnostic(transparent)]
    Execd(#[from] crate::backend::execd::ExecdError),
}

/// Result alias for filesystem operations.
pub type FsResult<T> = Result<T, FsError>;

/// A filesystem the agent's tools operate through.
///
/// Implementors enforce containment; tools never call raw `std::fs`. All
/// methods take caller-supplied paths and are responsible for resolving and
/// validating them before touching disk.
pub trait Filesystem: Send + Sync {
    /// Read a file to a string (read-sandboxed).
    fn read_to_string(&self, path: &Path) -> FsResult<String>;
    /// Write a string to a file, creating parent dirs (write-sandboxed).
    fn write(&self, path: &Path, contents: &str) -> FsResult<()>;
    /// List the immediate entries of a directory (read-sandboxed).
    fn read_dir(&self, path: &Path) -> FsResult<Vec<PathBuf>>;
    /// Whether a path exists *and* is within the read sandbox.
    fn exists(&self, path: &Path) -> bool;
    /// The roots within which reads are permitted (for tools that scope a walk,
    /// e.g. Glob/Grep with no explicit path).
    fn read_roots(&self) -> &[PathBuf];
    /// Whether writes are possible at all (a writable root is configured).
    fn is_writable(&self) -> bool;
    /// On-disk byte length of a file (read-sandboxed). Lets directory-walking
    /// tools skip oversized files *before* materializing them.
    fn file_size(&self, path: &Path) -> FsResult<u64>;
    /// Whether the path is itself a symlink (does NOT follow it). Directory
    /// walks use this to avoid following symlinks — preventing cycles
    /// (stack-overflow) and symlink escapes, matching `find`/`ripgrep` defaults.
    fn is_symlink(&self, path: &Path) -> bool;
}

/// Read a file through an optional sandbox.
///
/// `Some(fs)` enforces read containment; `None` falls back to a raw read — for
/// operator-driven contexts (the CLI / daemon ingest endpoints) where the human
/// chose the path. Tools registered for the autonomous agent pass `Some` so the
/// akh cannot read outside its sandbox.
pub fn read_optional(fs: Option<&Arc<dyn Filesystem>>, path: &Path) -> FsResult<String> {
    match fs {
        Some(fs) => fs.read_to_string(path),
        None => std::fs::read_to_string(path).map_err(|source| FsError::Io {
            path: path.display().to_string(),
            source,
        }),
    }
}

/// Whether a path exists, through an optional sandbox. `Some(fs)` only reports
/// existence for in-sandbox paths; `None` is a raw existence check.
pub fn exists_optional(fs: Option<&Arc<dyn Filesystem>>, path: &Path) -> bool {
    match fs {
        Some(fs) => fs.exists(path),
        None => path.exists(),
    }
}

/// A sandboxed local filesystem: reads confined to `read_roots`, writes to
/// `write_roots`, with symlink-safe canonical containment.
#[derive(Debug, Clone)]
pub struct LocalFs {
    read_roots: Vec<PathBuf>,
    write_roots: Vec<PathBuf>,
}

impl LocalFs {
    /// Construct with explicit read and write roots. Write roots are created if
    /// absent (so they canonicalize consistently) and all roots are
    /// canonicalized up front so containment compares like-for-like.
    pub fn new(read_roots: Vec<PathBuf>, write_roots: Vec<PathBuf>) -> Self {
        for r in &write_roots {
            let _ = std::fs::create_dir_all(r);
        }
        Self {
            read_roots: canonicalize_roots(read_roots),
            write_roots: canonicalize_roots(write_roots),
        }
    }

    /// Read+write confined to a single directory (common for tests and simple
    /// single-workspace akhs).
    pub fn sandbox(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self::new(vec![root.clone()], vec![root])
    }

    /// Reads allowed within `read_roots`; writes disabled.
    pub fn read_only(read_roots: Vec<PathBuf>) -> Self {
        Self::new(read_roots, Vec::new())
    }

    /// A filesystem that denies everything (no roots) — used when an engine has
    /// no data directory, so file tools fail closed rather than reading the
    /// whole host.
    pub fn denied() -> Self {
        Self::new(Vec::new(), Vec::new())
    }

    /// Resolve a caller path for **reading** to a canonical, contained path.
    ///
    /// `canonicalize` resolves all symlinks (and `..`), so a symlink inside a
    /// root that points outside it fails containment. A *relative* `path` is
    /// joined against the active thread-local session cwd if one is set (see
    /// [`set_session_cwd`]) — the same base [`Self::resolve_write`] uses in
    /// that case, closing the historical read/write base asymmetry — and
    /// otherwise left for `canonicalize` to resolve against the process's
    /// own working directory, exactly as before session-cwd threading
    /// existed. Containment against `read_roots` is checked unconditionally
    /// either way: a session cwd only narrows where a relative path starts,
    /// never which root it must land inside.
    pub fn resolve_read(&self, path: &Path) -> FsResult<PathBuf> {
        let joined = resolve_relative(path, None);
        let canonical = joined.canonicalize().map_err(|source| FsError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if contained(&canonical, &self.read_roots) {
            Ok(canonical)
        } else {
            Err(FsError::OutsideSandbox {
                path: path.display().to_string(),
                kind: "read",
                roots: self.read_roots.clone(),
            })
        }
    }

    /// Resolve a caller path for **writing** to a canonical, contained path.
    ///
    /// Relative paths are resolved under the first write root. `..` components
    /// are rejected outright, and the deepest existing ancestor is canonicalized
    /// (resolving symlinks in the real prefix) before the containment check, so
    /// a symlinked directory in the path cannot redirect the write outside.
    pub fn resolve_write(&self, path: &Path) -> FsResult<PathBuf> {
        if self.write_roots.is_empty() {
            return Err(FsError::WritesDisabled);
        }
        if path.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(FsError::Traversal {
                path: path.display().to_string(),
            });
        }
        let joined = resolve_relative(path, self.write_roots.first().map(PathBuf::as_path));
        let canonical = canonicalize_existing_prefix(&joined);
        if contained(&canonical, &self.write_roots) {
            Ok(canonical)
        } else {
            Err(FsError::OutsideSandbox {
                path: joined.display().to_string(),
                kind: "write",
                roots: self.write_roots.clone(),
            })
        }
    }
}

impl Filesystem for LocalFs {
    fn read_to_string(&self, path: &Path) -> FsResult<String> {
        let resolved = self.resolve_read(path)?;
        std::fs::read_to_string(&resolved).map_err(|source| FsError::Io {
            path: resolved.display().to_string(),
            source,
        })
    }

    fn write(&self, path: &Path, contents: &str) -> FsResult<()> {
        let resolved = self.resolve_write(path)?;
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(|source| FsError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }
        std::fs::write(&resolved, contents).map_err(|source| FsError::Io {
            path: resolved.display().to_string(),
            source,
        })
    }

    fn read_dir(&self, path: &Path) -> FsResult<Vec<PathBuf>> {
        let resolved = self.resolve_read(path)?;
        let mut entries = Vec::new();
        let iter = std::fs::read_dir(&resolved).map_err(|source| FsError::Io {
            path: resolved.display().to_string(),
            source,
        })?;
        for entry in iter {
            let entry = entry.map_err(|source| FsError::Io {
                path: resolved.display().to_string(),
                source,
            })?;
            entries.push(entry.path());
        }
        Ok(entries)
    }

    fn exists(&self, path: &Path) -> bool {
        self.resolve_read(path).map(|p| p.exists()).unwrap_or(false)
    }

    fn read_roots(&self) -> &[PathBuf] {
        &self.read_roots
    }

    fn is_writable(&self) -> bool {
        !self.write_roots.is_empty()
    }

    fn file_size(&self, path: &Path) -> FsResult<u64> {
        let resolved = self.resolve_read(path)?;
        let meta = std::fs::metadata(&resolved).map_err(|source| FsError::Io {
            path: resolved.display().to_string(),
            source,
        })?;
        Ok(meta.len())
    }

    fn is_symlink(&self, path: &Path) -> bool {
        std::fs::symlink_metadata(path)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
    }
}

/// Canonicalize each root, keeping the original if it cannot be canonicalized
/// (e.g. it does not exist yet for a read root).
fn canonicalize_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    roots
        .into_iter()
        .map(|r| r.canonicalize().unwrap_or(r))
        .collect()
}

/// Whether `path` lies within any of `roots`.
fn contained(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path.starts_with(root))
}

/// Canonicalize the deepest existing ancestor of `path`, then re-append the
/// non-existent tail. This resolves symlinks in the real prefix before a
/// containment check — closing the parent-only-canonicalize bypass — while
/// still allowing writes to not-yet-existing files.
pub(crate) fn canonicalize_existing_prefix(path: &Path) -> PathBuf {
    let mut ancestor = path.to_path_buf();
    let mut tail: Vec<OsString> = Vec::new();
    loop {
        if let Ok(canonical) = ancestor.canonicalize() {
            let mut result = canonical;
            for component in tail.iter().rev() {
                result.push(component);
            }
            return result;
        }
        match (
            ancestor.file_name().map(|s| s.to_owned()),
            ancestor.parent().map(|p| p.to_path_buf()),
        ) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                ancestor = parent;
            }
            // Nothing in the chain canonicalizes; return as-is so the
            // containment check below fails closed.
            _ => return path.to_path_buf(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_inside_sandbox_ok_outside_denied() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("in.txt"), "inside").unwrap();

        // A file outside the root.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();

        let fs = LocalFs::read_only(vec![root.clone()]);
        assert_eq!(fs.read_to_string(&root.join("in.txt")).unwrap(), "inside");
        let denied = fs.read_to_string(&outside.path().join("secret.txt"));
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "read", .. })),
            "read outside the sandbox must be denied: {denied:?}"
        );
    }

    #[test]
    fn write_confined_to_write_roots() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let fs = LocalFs::sandbox(&root);

        // Relative path → joined under the write root.
        fs.write(Path::new("sub/out.txt"), "hello").unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join("sub/out.txt")).unwrap(),
            "hello"
        );

        // Absolute path outside the root → denied.
        let denied = fs.write(Path::new("/tmp/akh-evil-test.txt"), "nope");
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "write", .. })),
            "write outside the sandbox must be denied: {denied:?}"
        );
    }

    #[test]
    fn write_rejects_parent_dir_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFs::sandbox(dir.path());
        let denied = fs.write(Path::new("../escape.txt"), "x");
        assert!(
            matches!(denied, Err(FsError::Traversal { .. })),
            "`..` traversal must be rejected: {denied:?}"
        );
    }

    #[test]
    fn writes_disabled_when_no_write_root() {
        let dir = tempfile::tempdir().unwrap();
        let fs = LocalFs::read_only(vec![dir.path().to_path_buf()]);
        assert!(!fs.is_writable());
        assert!(matches!(
            fs.write(Path::new("x.txt"), "y"),
            Err(FsError::WritesDisabled)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_denied_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        // A secret outside the sandbox.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "top-secret").unwrap();
        // A symlink *inside* the root pointing at the outside secret.
        let link = root.join("link.txt");
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), &link).unwrap();

        let fs = LocalFs::read_only(vec![root.clone()]);
        let denied = fs.read_to_string(&link);
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "read", .. })),
            "reading through a symlink that escapes the sandbox must be denied: {denied:?}"
        );
    }

    #[test]
    fn denied_fs_reads_and_writes_fail() {
        let fs = LocalFs::denied();
        assert!(!fs.is_writable());
        assert!(fs.read_to_string(Path::new("/etc/hostname")).is_err());
        assert!(matches!(
            fs.write(Path::new("x"), "y"),
            Err(FsError::WritesDisabled)
        ));
    }

    #[test]
    fn file_size_is_sandboxed() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("f.txt"), "12345").unwrap();
        let fs = LocalFs::read_only(vec![root.clone()]);
        assert_eq!(fs.file_size(&root.join("f.txt")).unwrap(), 5);
        // Outside the sandbox → denied (not a size).
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("big.txt"), "xxxx").unwrap();
        assert!(fs.file_size(&outside.path().join("big.txt")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn is_symlink_detects_without_following() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("real.txt"), "x").unwrap();
        std::os::unix::fs::symlink(root.join("real.txt"), root.join("link.txt")).unwrap();
        let fs = LocalFs::read_only(vec![root.clone()]);
        assert!(fs.is_symlink(&root.join("link.txt")));
        assert!(!fs.is_symlink(&root.join("real.txt")));
    }

    // ── Session cwd threading (ACP `session/new` cwd → shell_exec/file verbs) ──

    #[test]
    fn relative_write_and_read_use_the_session_cwd_when_one_is_active() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let sub = root.join("project");
        std::fs::create_dir_all(&sub).unwrap();
        let fs = LocalFs::sandbox(&root);

        // No session cwd yet: relative write lands at the sandbox root, as
        // it always has.
        fs.write(Path::new("top.txt"), "top").unwrap();
        assert!(root.join("top.txt").exists());

        {
            let _guard = set_session_cwd(Some(sub.clone()));
            fs.write(Path::new("nested.txt"), "nested").unwrap();
            assert_eq!(
                std::fs::read_to_string(sub.join("nested.txt")).unwrap(),
                "nested",
                "a relative write must land under the active session cwd"
            );
            assert!(
                !root.join("nested.txt").exists(),
                "must not also land at the sandbox root"
            );

            // Read shares the same base as write once a session cwd is
            // active — a relative read for a file that only exists under
            // the session cwd must resolve there too (the read/write
            // asymmetry fix).
            let read_back = fs.read_to_string(Path::new("nested.txt")).unwrap();
            assert_eq!(read_back, "nested");
        }

        // Guard dropped: relative writes go back to the sandbox root exactly
        // as before — the no-session-cwd behavior is unchanged.
        fs.write(Path::new("after.txt"), "after").unwrap();
        assert!(root.join("after.txt").exists());
    }

    #[test]
    fn session_cwd_outside_the_sandbox_is_still_denied_on_write() {
        // The session cwd mechanism only changes which BASE a relative path
        // is joined against; the existing containment check against
        // `write_roots` still runs unconditionally afterward. Escaping the
        // configured root is not possible via a session cwd — belt-and-
        // suspenders alongside the containment clamp `acp::run_turn`
        // applies before ever setting this thread-local.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let outside = tempfile::tempdir().unwrap();
        let fs = LocalFs::sandbox(&root);

        let _guard = set_session_cwd(Some(outside.path().to_path_buf()));
        let denied = fs.write(Path::new("evil.txt"), "nope");
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "write", .. })),
            "a session cwd outside write_roots must not let a relative write escape: {denied:?}"
        );
    }

    #[test]
    fn session_cwd_outside_the_sandbox_is_still_denied_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "top-secret").unwrap();
        let fs = LocalFs::sandbox(&root);

        let _guard = set_session_cwd(Some(outside.path().to_path_buf()));
        let denied = fs.read_to_string(Path::new("secret.txt"));
        assert!(
            matches!(denied, Err(FsError::OutsideSandbox { kind: "read", .. })),
            "a session cwd outside read_roots must not let a relative read escape: {denied:?}"
        );
    }

    #[test]
    fn set_session_cwd_guard_restores_the_previous_value_on_drop() {
        assert_eq!(
            session_cwd(),
            None,
            "sanity: nothing active before the test"
        );
        let a = PathBuf::from("/tmp/akh-session-cwd-test-a");
        let b = PathBuf::from("/tmp/akh-session-cwd-test-b");
        {
            let _outer = set_session_cwd(Some(a.clone()));
            assert_eq!(session_cwd(), Some(a.clone()));
            {
                let _inner = set_session_cwd(Some(b.clone()));
                assert_eq!(session_cwd(), Some(b.clone()));
            }
            assert_eq!(
                session_cwd(),
                Some(a.clone()),
                "dropping the inner guard must restore the outer value, not clear it"
            );
        }
        assert_eq!(
            session_cwd(),
            None,
            "dropping the outer guard must restore None"
        );
    }
}
