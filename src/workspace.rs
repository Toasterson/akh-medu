//! Workspace management: named, isolated Engine instances.
//!
//! Each workspace has its own KG, skills, compartments, and scratch space.
//! Workspace configuration is persisted as TOML in `$XDG_CONFIG_HOME/akh-medu/workspaces/`.

use std::path::PathBuf;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::grammar::Language;
use crate::paths::{AkhPaths, WorkspacePaths};
use crate::vsa::{Dimension, Encoding};

/// Errors from workspace operations.
#[derive(Debug, Error, Diagnostic)]
pub enum WorkspaceError {
    #[error("workspace \"{name}\" already exists")]
    #[diagnostic(
        code(akh::workspace::already_exists),
        help("Use a different name or delete the existing workspace first.")
    )]
    AlreadyExists { name: String },

    #[error("workspace \"{name}\" not found")]
    #[diagnostic(
        code(akh::workspace::not_found),
        help("Create it with `akh-medu workspace create {name}` or list workspaces with `akh-medu workspace list`.")
    )]
    NotFound { name: String },

    #[error("failed to read workspace config: {path}")]
    #[diagnostic(
        code(akh::workspace::config_read),
        help("Ensure the config file exists and is valid TOML.")
    )]
    ConfigRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse workspace config: {path}")]
    #[diagnostic(
        code(akh::workspace::config_parse),
        help("Check the TOML syntax in the workspace config file.")
    )]
    ConfigParse { path: String, message: String },

    #[error("failed to write workspace config: {path}")]
    #[diagnostic(
        code(akh::workspace::config_write),
        help("Ensure you have write permissions to the config directory.")
    )]
    ConfigWrite {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to delete workspace \"{name}\": {message}")]
    #[diagnostic(
        code(akh::workspace::delete_failed),
        help("Ensure no processes are using the workspace data and you have write permissions.")
    )]
    DeleteFailed { name: String, message: String },
}

pub type WorkspaceResult<T> = std::result::Result<T, WorkspaceError>;

/// Per-workspace configuration, persisted as TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Workspace name.
    pub name: String,
    /// Hypervector dimension.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Encoding scheme name.
    #[serde(default = "default_encoding")]
    pub encoding: String,
    /// Default language.
    #[serde(default = "default_language")]
    pub language: String,
    /// Maximum memory budget in MB.
    #[serde(default = "default_max_memory_mb")]
    pub max_memory_mb: usize,
    /// Maximum expected symbols.
    #[serde(default = "default_max_symbols")]
    pub max_symbols: u64,
    /// Seed pack IDs to apply on first initialization.
    #[serde(default)]
    pub seed_packs: Vec<String>,
    /// Shared partition names to mount.
    #[serde(default)]
    pub shared_partitions: Vec<String>,
}

fn default_dimension() -> usize {
    10_000
}
fn default_encoding() -> String {
    "bipolar".into()
}
fn default_language() -> String {
    "auto".into()
}
fn default_max_memory_mb() -> usize {
    1024
}
fn default_max_symbols() -> u64 {
    1_000_000
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            dimension: default_dimension(),
            encoding: default_encoding(),
            language: default_language(),
            max_memory_mb: default_max_memory_mb(),
            max_symbols: default_max_symbols(),
            seed_packs: vec![
                "identity".into(),
                "ontology".into(),
                "common-sense".into(),
            ],
            shared_partitions: Vec::new(),
        }
    }
}

impl WorkspaceConfig {
    /// Create a config with a specific name (other fields default).
    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Convert to an EngineConfig using resolved workspace paths.
    pub fn to_engine_config(&self, ws_paths: &WorkspacePaths) -> crate::engine::EngineConfig {
        let language = match self.language.as_str() {
            "en" => Language::English,
            "ru" => Language::Russian,
            "ar" => Language::Arabic,
            "fr" => Language::French,
            "es" => Language::Spanish,
            _ => Language::Auto,
        };

        let encoding = match self.encoding.as_str() {
            // Currently only Bipolar is supported; future: FHRR, SSP.
            _ => Encoding::Bipolar,
        };

        crate::engine::EngineConfig {
            dimension: Dimension(self.dimension),
            encoding,
            data_dir: Some(ws_paths.kg_dir.clone()),
            max_memory_mb: self.max_memory_mb,
            max_symbols: self.max_symbols as usize,
            language,
        }
    }

    /// Load from a TOML file.
    pub fn load(path: &std::path::Path) -> WorkspaceResult<Self> {
        let content =
            std::fs::read_to_string(path).map_err(|e| WorkspaceError::ConfigRead {
                path: path.display().to_string(),
                source: e,
            })?;
        toml::from_str(&content).map_err(|e| WorkspaceError::ConfigParse {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    /// Save to a TOML file.
    pub fn save(&self, path: &std::path::Path) -> WorkspaceResult<()> {
        let content = toml::to_string_pretty(self).map_err(|e| WorkspaceError::ConfigParse {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| WorkspaceError::ConfigWrite {
                path: parent.display().to_string(),
                source: e,
            })?;
        }
        std::fs::write(path, content).map_err(|e| WorkspaceError::ConfigWrite {
            path: path.display().to_string(),
            source: e,
        })
    }
}

/// Workspace manager: create, list, delete workspaces.
pub struct WorkspaceManager {
    paths: AkhPaths,
}

impl WorkspaceManager {
    pub fn new(paths: AkhPaths) -> Self {
        Self { paths }
    }

    /// Create a new workspace with default or provided config.
    pub fn create(&self, config: WorkspaceConfig) -> WorkspaceResult<WorkspacePaths> {
        let ws_paths = self.paths.workspace(&config.name);

        if ws_paths.exists() {
            return Err(WorkspaceError::AlreadyExists {
                name: config.name.clone(),
            });
        }

        // Create directory structure.
        ws_paths.ensure_dirs().map_err(|e| WorkspaceError::ConfigWrite {
            path: ws_paths.root.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
        })?;

        // Save workspace config.
        let config_path = self.paths.workspace_config_file(&config.name);
        config.save(&config_path)?;

        Ok(ws_paths)
    }

    /// List all workspace names.
    pub fn list(&self) -> Vec<String> {
        self.paths.list_workspaces()
    }

    /// Delete a workspace (data + config).
    pub fn delete(&self, name: &str) -> WorkspaceResult<()> {
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.exists() {
            return Err(WorkspaceError::NotFound {
                name: name.to_string(),
            });
        }

        // Remove data directory.
        std::fs::remove_dir_all(&ws_paths.root).map_err(|e| WorkspaceError::DeleteFailed {
            name: name.to_string(),
            message: e.to_string(),
        })?;

        // Remove config file (may not exist).
        let config_path = self.paths.workspace_config_file(name);
        let _ = std::fs::remove_file(&config_path);

        // Remove session file (may not exist).
        let _ = std::fs::remove_file(&ws_paths.session_file);

        Ok(())
    }

    /// Get info about a workspace (load its config).
    pub fn info(&self, name: &str) -> WorkspaceResult<WorkspaceConfig> {
        let config_path = self.paths.workspace_config_file(name);
        if config_path.exists() {
            WorkspaceConfig::load(&config_path)
        } else {
            let ws_paths = self.paths.workspace(name);
            if ws_paths.exists() {
                // Workspace dir exists but no config — return defaults.
                Ok(WorkspaceConfig::with_name(name))
            } else {
                Err(WorkspaceError::NotFound {
                    name: name.to_string(),
                })
            }
        }
    }

    /// Resolve workspace paths, ensuring the workspace exists.
    pub fn resolve(&self, name: &str) -> WorkspaceResult<WorkspacePaths> {
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.exists() {
            return Err(WorkspaceError::NotFound {
                name: name.to_string(),
            });
        }
        Ok(ws_paths)
    }

    /// Get the underlying AkhPaths.
    pub fn paths(&self) -> &AkhPaths {
        &self.paths
    }
}

/// Check if a legacy `.akh-medu` directory exists in the current working directory.
pub fn detect_legacy_data_dir() -> Option<PathBuf> {
    let legacy = PathBuf::from(".akh-medu");
    if legacy.is_dir() {
        Some(legacy)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_seed_packs() {
        let cfg = WorkspaceConfig::default();
        assert_eq!(cfg.name, "default");
        assert!(!cfg.seed_packs.is_empty());
        assert!(cfg.seed_packs.contains(&"identity".to_string()));
    }

    #[test]
    fn config_roundtrip_toml() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("test.toml");

        let cfg = WorkspaceConfig {
            name: "test-ws".into(),
            dimension: 5000,
            ..Default::default()
        };
        cfg.save(&path).unwrap();

        let loaded = WorkspaceConfig::load(&path).unwrap();
        assert_eq!(loaded.name, "test-ws");
        assert_eq!(loaded.dimension, 5000);
    }

    #[test]
    fn workspace_manager_create_list_delete() {
        let tmp = tempfile::TempDir::new().unwrap();
        let paths = AkhPaths {
            config_dir: tmp.path().join("config"),
            data_dir: tmp.path().join("data"),
            state_dir: tmp.path().join("state"),
            runtime_dir: tmp.path().join("run"),
            cache_dir: tmp.path().join("cache"),
        };
        paths.ensure_dirs().unwrap();

        let mgr = WorkspaceManager::new(paths);

        // Create.
        let ws = mgr.create(WorkspaceConfig::with_name("alpha")).unwrap();
        assert!(ws.exists());
        assert!(ws.kg_dir.is_dir());

        // List.
        let names = mgr.list();
        assert!(names.contains(&"alpha".to_string()));

        // Duplicate create fails.
        assert!(mgr.create(WorkspaceConfig::with_name("alpha")).is_err());

        // Delete.
        mgr.delete("alpha").unwrap();
        assert!(!ws.exists());
    }
}
