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
            "Create it with `akh-medu workspace create {name}` or list existing workspaces with `akh-medu workspace list`."
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

impl AkhPaths {
    /// Resolve XDG directories from environment variables with standard fallbacks.
    pub fn resolve() -> PathResult<Self> {
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

    /// Get workspace paths for a named workspace.
    pub fn workspace(&self, name: &str) -> WorkspacePaths {
        let root = self.data_dir.join("workspaces").join(name);
        WorkspacePaths {
            name: name.to_string(),
            root: root.clone(),
            kg_dir: root.join("kg"),
            skills_dir: root.join("skills"),
            compartments_dir: root.join("compartments"),
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
}
