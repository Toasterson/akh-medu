//! XDG-compliant path resolution for akh-medu.
//!
//! Provides `AkhPaths` (global directories) and `WorkspacePaths` (per-workspace
//! directories) following the XDG Base Directory Specification.

use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from path resolution.
#[derive(Debug, Error, Diagnostic)]
pub enum PathError {
    #[error("cannot determine home directory")]
    #[diagnostic(
        code(akh::paths::no_home),
        help("Set the HOME environment variable or ensure a valid user profile exists.")
    )]
    NoHome,

    #[error("failed to create directory: {path}")]
    #[diagnostic(
        code(akh::paths::create_dir),
        help("Check that the parent directory exists and you have write permissions.")
    )]
    CreateDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("workspace not found: \"{name}\"")]
    #[diagnostic(
        code(akh::paths::workspace_not_found),
        help(
            "Create it with `akh workspace create {name}` or list existing workspaces with `akh workspace list`."
        )
    )]
    WorkspaceNotFound { name: String },
}

pub type PathResult<T> = std::result::Result<T, PathError>;

/// Global XDG-compliant directories for akh-medu.
#[derive(Debug, Clone)]
pub struct AkhPaths {
    /// `$XDG_CONFIG_HOME/akh-medu/`
    pub config_dir: PathBuf,
    /// `$XDG_DATA_HOME/akh-medu/`
    pub data_dir: PathBuf,
    /// `$XDG_STATE_HOME/akh-medu/`
    pub state_dir: PathBuf,
    /// `$XDG_RUNTIME_DIR/akh-medu/` (falls back to `state_dir/run/`)
    pub runtime_dir: PathBuf,
    /// `$XDG_CACHE_HOME/akh-medu/`
    pub cache_dir: PathBuf,
}

/// Environment variable selecting the system-daemon layout.
///
/// Opt-in and explicit rather than inferred from `euid == 0`: a developer who
/// happens to run under `sudo` must not silently get a different set of
/// directories than the one their config and workspaces live in.
pub const SYSTEM_PATHS_ENV: &str = "AKH_SYSTEM_PATHS";

impl AkhPaths {
    /// Resolve XDG directories from environment variables with standard fallbacks.
    ///
    /// When [`SYSTEM_PATHS_ENV`] is set to `1`, returns [`AkhPaths::system_daemon`]
    /// instead — the layout for an akhomed running as a system service, whose
    /// state must not live in a user's home directory.
    pub fn resolve() -> PathResult<Self> {
        if std::env::var(SYSTEM_PATHS_ENV).is_ok_and(|v| v == "1") {
            return Ok(Self::system_daemon());
        }

        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .map_err(|_| PathError::NoHome)?;

        let config_dir = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".config"))
            .join("akh-medu");

        let data_dir = std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".local/share"))
            .join("akh-medu");

        let state_dir = std::env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".local/state"))
            .join("akh-medu");

        let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
            .map(|d| PathBuf::from(d).join("akh-medu"))
            .unwrap_or_else(|_| state_dir.join("run"));

        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".cache"))
            .join("akh-medu");

        Ok(Self {
            config_dir,
            data_dir,
            state_dir,
            runtime_dir,
            cache_dir,
        })
    }

    /// Directories for an akhomed running as a **system service**, owned by the
    /// system rather than by a user.
    ///
    /// A daemon must not keep its state in someone's home directory. Doing so
    /// couples the service's lifetime to an account, leaves root-owned files
    /// scattered through a user's tree (where that user then cannot rewrite
    /// them), and on macOS drags the whole daemon inside per-user privacy
    /// (TCC) and FileVault scope.
    ///
    /// Each platform's own convention is used rather than one invented layout:
    ///
    /// **macOS** — Apple's File System Programming Guide puts system-wide
    /// support files in `/Library/Application Support/<name>`; caches belong in
    /// `/Library/Caches`; `/var/run` is BSD runtime state, cleared on boot,
    /// which is exactly a PID file's lifetime; `/var/db` is where BSD/macOS
    /// daemons keep mutable databases.
    ///
    /// ```text
    /// config   /Library/Application Support/akh-medu
    /// data     /Library/Application Support/akh-medu/data
    /// state    /var/db/akh-medu
    /// runtime  /var/run/akh-medu
    /// cache    /Library/Caches/akh-medu
    /// ```
    ///
    /// **Linux / illumos** — the FHS equivalents:
    ///
    /// ```text
    /// config   /etc/akh-medu
    /// data     /var/lib/akh-medu
    /// state    /var/lib/akh-medu/state
    /// runtime  /run/akh-medu
    /// cache    /var/cache/akh-medu
    /// ```
    #[cfg(target_os = "macos")]
    pub fn system_daemon() -> Self {
        let support = PathBuf::from("/Library/Application Support/akh-medu");
        Self {
            data_dir: support.join("data"),
            config_dir: support,
            state_dir: PathBuf::from("/var/db/akh-medu"),
            runtime_dir: Self::system_runtime_dir(),
            cache_dir: PathBuf::from("/Library/Caches/akh-medu"),
        }
    }

    /// System-service directories on Linux and illumos (FHS). See the macOS
    /// variant for the rationale.
    #[cfg(not(target_os = "macos"))]
    pub fn system_daemon() -> Self {
        let lib = PathBuf::from("/var/lib/akh-medu");
        Self {
            config_dir: PathBuf::from("/etc/akh-medu"),
            state_dir: lib.join("state"),
            data_dir: lib,
            runtime_dir: Self::system_runtime_dir(),
            cache_dir: PathBuf::from("/var/cache/akh-medu"),
        }
    }

    /// Where a system-service akhomed advertises itself (PID file, sockets).
    ///
    /// Exposed on its own because *clients* need it without adopting the whole
    /// system layout: an unprivileged `akh` looks in the user's runtime
    /// directory first and falls back here, so it can find a daemon that does
    /// not share its home.
    pub fn system_runtime_dir() -> PathBuf {
        if cfg!(target_os = "macos") {
            PathBuf::from("/var/run/akh-medu")
        } else {
            PathBuf::from("/run/akh-medu")
        }
    }

    /// Get workspace paths for a named workspace.
    pub fn workspace(&self, name: &str) -> WorkspacePaths {
        let root = self.data_dir.join("workspaces").join(name);
        WorkspacePaths {
            name: name.to_string(),
            root: root.clone(),
            kg_dir: root.join("kg"),
            skills_dir: root.join("skills"),
            compartments_dir: root.join("compartments"),
            notes_dir: root.join("notes"),
            scratch_dir: root.join("scratch"),
            session_file: self.state_dir.join("sessions").join(format!("{name}.bin")),
        }
    }

    /// List all existing workspace names.
    pub fn list_workspaces(&self) -> Vec<String> {
        let ws_dir = self.data_dir.join("workspaces");
        match std::fs::read_dir(&ws_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
                .filter_map(|e| e.file_name().into_string().ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Create all base directories. Idempotent.
    pub fn ensure_dirs(&self) -> PathResult<()> {
        let library_dir = self.library_dir();
        let library_inbox = self.library_inbox();
        let library_docs = self.library_documents_dir();
        let journals_dir = self.memory_journals_dir();
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.state_dir,
            &self.runtime_dir,
            &self.cache_dir,
            &self.data_dir.join("workspaces"),
            &self.data_dir.join("seeds"),
            &self.data_dir.join("skills"),
            &self.state_dir.join("sessions"),
            &self.state_dir.join("logs"),
            &self.config_dir.join("workspaces"),
            &library_dir,
            &library_inbox,
            &library_docs,
            &journals_dir,
        ] {
            std::fs::create_dir_all(dir).map_err(|e| PathError::CreateDir {
                path: dir.display().to_string(),
                source: e,
            })?;
        }
        Ok(())
    }

    /// Path to the global config file.
    pub fn global_config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    /// Path to a workspace's config file.
    pub fn workspace_config_file(&self, name: &str) -> PathBuf {
        self.config_dir
            .join("workspaces")
            .join(format!("{name}.toml"))
    }

    /// Path to the seeds directory.
    pub fn seeds_dir(&self) -> PathBuf {
        self.data_dir.join("seeds")
    }

    /// Shared NLU model directory: `$XDG_DATA_HOME/akh-medu/models/`.
    ///
    /// Models are large and workspace-independent, so they're stored in a
    /// shared location rather than per-workspace.
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// Path to the OpenClaw daily memory journal directory.
    ///
    /// Defaults to `$XDG_DATA_HOME/akh-medu/journals/`. Override with
    /// the `AKH_JOURNAL_DIR` environment variable.
    pub fn memory_journals_dir(&self) -> PathBuf {
        std::env::var("AKH_JOURNAL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| self.data_dir.join("journals"))
    }

    /// Path to the shared content library directory.
    pub fn library_dir(&self) -> PathBuf {
        self.data_dir.join("library")
    }

    /// Path to the library inbox directory (watched for new files).
    pub fn library_inbox(&self) -> PathBuf {
        self.data_dir.join("library").join("inbox")
    }

    /// Path to the library documents directory.
    pub fn library_documents_dir(&self) -> PathBuf {
        self.data_dir.join("library").join("documents")
    }

    /// Path to the akhomed PID/port file (`runtime_dir/akhomed.pid`).
    ///
    /// Contains JSON: `{ "pid": N, "port": P, "bind": "addr" }`.
    pub fn pid_file(&self) -> PathBuf {
        self.runtime_dir.join("akhomed.pid")
    }

    /// macOS log directory: `~/Library/Logs/akh-medu/`.
    ///
    /// Used by launchd for stdout/stderr capture. Returns `None` if HOME is
    /// not set.
    pub fn macos_log_dir() -> Option<PathBuf> {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Logs/akh-medu"))
    }

    /// macOS LaunchAgents directory: `~/Library/LaunchAgents/`.
    ///
    /// Where per-user launchd plist files are installed. Returns `None` if
    /// HOME is not set.
    pub fn macos_launch_agents_dir() -> Option<PathBuf> {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/LaunchAgents"))
    }
}

/// Per-workspace directory layout.
#[derive(Debug, Clone)]
pub struct WorkspacePaths {
    pub name: String,
    /// `data_dir/workspaces/{name}/`
    pub root: PathBuf,
    /// `root/kg/` — oxigraph, redb, hnsw data
    pub kg_dir: PathBuf,
    /// `root/skills/` — activated skill data
    pub skills_dir: PathBuf,
    /// `root/compartments/` — compartment data
    pub compartments_dir: PathBuf,
    /// `root/notes/` — append-only study-note heads and revision history
    pub notes_dir: PathBuf,
    /// `root/scratch/` — agent scratch space
    pub scratch_dir: PathBuf,
    /// `state_dir/sessions/{name}.bin` — agent session state
    pub session_file: PathBuf,
}

impl WorkspacePaths {
    /// Create all workspace directories. Idempotent.
    pub fn ensure_dirs(&self) -> PathResult<()> {
        for dir in [
            &self.root,
            &self.kg_dir,
            &self.skills_dir,
            &self.compartments_dir,
            &self.notes_dir,
            &self.scratch_dir,
        ] {
            std::fs::create_dir_all(dir).map_err(|e| PathError::CreateDir {
                path: dir.display().to_string(),
                source: e,
            })?;
        }
        // Ensure the sessions directory exists for the session file.
        if let Some(parent) = self.session_file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| PathError::CreateDir {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        Ok(())
    }

    /// Check if this workspace has been initialized (root directory exists).
    pub fn exists(&self) -> bool {
        self.root.is_dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_paths_use_xdg_layout() {
        // Test that workspace paths follow expected XDG structure
        // without mutating env vars (which is unsafe in edition 2024).
        let home = std::env::var("HOME").unwrap();
        let paths = AkhPaths::resolve().unwrap();

        // Should be under home-derived defaults or XDG overrides.
        assert!(
            paths.config_dir.to_string_lossy().contains("akh-medu"),
            "config_dir should contain 'akh-medu': {}",
            paths.config_dir.display()
        );
        assert!(
            paths.data_dir.to_string_lossy().contains("akh-medu"),
            "data_dir should contain 'akh-medu': {}",
            paths.data_dir.display()
        );

        // Workspace derivation should be consistent.
        let ws = paths.workspace("test-ws");
        assert!(ws.root.starts_with(&paths.data_dir));
        assert!(ws.kg_dir.starts_with(&ws.root));
        let _ = home; // suppress unused
    }

    #[test]
    fn workspace_paths_derive_from_name() {
        let paths = AkhPaths {
            config_dir: PathBuf::from("/cfg/akh-medu"),
            data_dir: PathBuf::from("/data/akh-medu"),
            state_dir: PathBuf::from("/state/akh-medu"),
            runtime_dir: PathBuf::from("/run/akh-medu"),
            cache_dir: PathBuf::from("/cache/akh-medu"),
        };

        let ws = paths.workspace("myproject");
        assert_eq!(ws.name, "myproject");
        assert_eq!(
            ws.root,
            PathBuf::from("/data/akh-medu/workspaces/myproject")
        );
        assert_eq!(
            ws.kg_dir,
            PathBuf::from("/data/akh-medu/workspaces/myproject/kg")
        );
        assert_eq!(
            ws.skills_dir,
            PathBuf::from("/data/akh-medu/workspaces/myproject/skills")
        );
        assert_eq!(
            ws.notes_dir,
            PathBuf::from("/data/akh-medu/workspaces/myproject/notes")
        );
        assert_eq!(
            ws.scratch_dir,
            PathBuf::from("/data/akh-medu/workspaces/myproject/scratch")
        );
        assert_eq!(
            ws.session_file,
            PathBuf::from("/state/akh-medu/sessions/myproject.bin")
        );
    }

    #[test]
    fn list_workspaces_empty_dir() {
        let paths = AkhPaths {
            config_dir: PathBuf::from("/nonexistent"),
            data_dir: PathBuf::from("/nonexistent"),
            state_dir: PathBuf::from("/nonexistent"),
            runtime_dir: PathBuf::from("/nonexistent"),
            cache_dir: PathBuf::from("/nonexistent"),
        };
        assert!(paths.list_workspaces().is_empty());
    }

    /// A system daemon must keep nothing in a user's home.
    ///
    /// This is the property that matters, independent of which platform's
    /// convention is in play: state under `/Users` or `/home` re-creates the
    /// coupling the system layout exists to remove.
    #[test]
    fn system_daemon_paths_escape_every_home_directory() {
        let p = AkhPaths::system_daemon();
        for dir in [
            &p.config_dir,
            &p.data_dir,
            &p.state_dir,
            &p.runtime_dir,
            &p.cache_dir,
        ] {
            let s = dir.display().to_string();
            assert!(
                !s.starts_with("/Users/") && !s.starts_with("/home/"),
                "system daemon path must not live in a user home: {s}"
            );
            assert!(
                dir.is_absolute(),
                "system daemon path must be absolute: {s}"
            );
        }
    }

    /// config and data must not collide.
    ///
    /// The obvious way to relocate — pointing several `XDG_*` variables at one
    /// parent — makes `config_dir` and `data_dir` the same directory, because
    /// each appends `akh-medu`. Workspaces would then write `<name>.toml` and
    /// `<name>/` side by side in one directory. Hence an explicit layout.
    #[test]
    fn system_daemon_config_and_data_are_distinct() {
        let p = AkhPaths::system_daemon();
        assert_ne!(p.config_dir, p.data_dir);
        assert_ne!(p.config_dir, p.state_dir);
        assert_ne!(p.data_dir, p.state_dir);
    }

    /// The runtime dir a client falls back to must be the one the daemon uses.
    ///
    /// If these ever drift, an unprivileged `akh` silently stops finding the
    /// system daemon — the failure looks like "no daemon running" rather than
    /// like a path bug.
    #[test]
    fn client_fallback_runtime_dir_matches_the_daemon_layout() {
        assert_eq!(
            AkhPaths::system_daemon().runtime_dir,
            AkhPaths::system_runtime_dir()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_system_layout_uses_apple_locations() {
        let p = AkhPaths::system_daemon();
        assert_eq!(
            p.config_dir,
            PathBuf::from("/Library/Application Support/akh-medu")
        );
        assert_eq!(p.state_dir, PathBuf::from("/var/db/akh-medu"));
        assert_eq!(p.runtime_dir, PathBuf::from("/var/run/akh-medu"));
        assert_eq!(p.cache_dir, PathBuf::from("/Library/Caches/akh-medu"));
        // The PID file lands where the installer and the client both expect.
        assert_eq!(p.pid_file(), PathBuf::from("/var/run/akh-medu/akhomed.pid"));
    }
}
