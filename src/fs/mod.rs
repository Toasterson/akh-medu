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

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use miette::Diagnostic;
use thiserror::Error;

/// Errors from filesystem operations behind the sandbox.
#[derive(Debug, Error, Diagnostic)]
pub enum FsError {
    /// The resolved path lies outside every allowed root for the operation.
    #[error("path \"{path}\" is outside the allowed {kind} roots")]
    #[diagnostic(
        code(akh::fs::outside_sandbox),
        help("Only paths within the akh's sandbox roots are permitted ({roots:?}). Use a path inside the workspace.")
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
        help("This filesystem has no writable roots configured. Run the akh with a data directory to enable writes.")
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
    /// root that points outside it fails containment.
    pub fn resolve_read(&self, path: &Path) -> FsResult<PathBuf> {
        let canonical = path.canonicalize().map_err(|source| FsError::Io {
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
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.write_roots[0].join(path)
        };
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
fn canonicalize_existing_prefix(path: &Path) -> PathBuf {
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
}
