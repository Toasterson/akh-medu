//! Workspace management: named, isolated Engine instances.
//!
//! Each workspace has its own KG, skills, compartments, and scratch space.
//! Workspace configuration is persisted as TOML in `$XDG_CONFIG_HOME/akh-medu/workspaces/`.

use std::path::PathBuf;
use std::sync::OnceLock;

use miette::Diagnostic;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::grammar::Language;
use crate::paths::{AkhPaths, WorkspacePaths};
use crate::vsa::{Dimension, Encoding};

/// Compiled workspace name validator. Names must be 1–63 lowercase alphanumeric
/// characters, hyphens, or underscores, and must start with a letter or digit.
static NAME_RE: OnceLock<Regex> = OnceLock::new();

fn name_re() -> &'static Regex {
    NAME_RE.get_or_init(|| Regex::new(r"^[a-z0-9][a-z0-9_-]{0,62}$").unwrap())
}

/// Validate a proposed workspace name.
///
/// Returns `Ok(())` if valid, or `Err(WorkspaceError::InvalidName)` with a
/// diagnostic explaining what is wrong.
pub fn validate_workspace_name(name: &str) -> WorkspaceResult<()> {
    if name_re().is_match(name) {
        return Ok(());
    }

    let detail = if name.is_empty() {
        "name is empty".to_string()
    } else if name.len() > 63 {
        format!("name is {} characters (max 63)", name.len())
    } else if name.contains(':') {
        "name contains a colon — workspace names must not contain query-like syntax".to_string()
    } else if name.contains(' ') {
        "name contains spaces".to_string()
    } else if name.chars().next().is_none_or(|c| !c.is_ascii_alphanumeric()) {
        "name must start with a lowercase letter or digit".to_string()
    } else {
        "name contains characters other than lowercase letters, digits, hyphens, and underscores"
            .to_string()
    };

    Err(WorkspaceError::InvalidName {
        name: name.to_string(),
        detail,
    })
}

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
        help(
            "Create it with `akh-medu workspace create {name}` or list workspaces with `akh-medu workspace list`."
        )
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

    #[error("role \"{current_role}\" already assigned to this workspace")]
    #[diagnostic(
        code(akh::workspace::role_already_assigned),
        help("Workspace role is write-once and cannot be changed after assignment.")
    )]
    RoleAlreadyAssigned { current_role: String },

    #[error("invalid workspace name \"{name}\": {detail}")]
    #[diagnostic(
        code(akh::workspace::invalid_name),
        help(
            "Workspace names must be 1–63 characters: lowercase letters (a-z), digits (0-9), \
             hyphens (-), or underscores (_), and must start with a letter or digit. \
             Names containing colons, spaces, or uppercase letters are rejected."
        )
    )]
    InvalidName { name: String, detail: String },
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
    #[serde(default = "default_seed_packs")]
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
fn default_seed_packs() -> Vec<String> {
    vec![
        "identity".into(),
        "ontology".into(),
        "common-sense".into(),
        "computing".into(),
        "architecture".into(),
    ]
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
                "computing".into(),
                "architecture".into(),
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

        // Currently only Bipolar is supported; future: FHRR, SSP.
        let encoding = Encoding::Bipolar;

        crate::engine::EngineConfig {
            dimension: Dimension(self.dimension),
            encoding,
            data_dir: Some(ws_paths.kg_dir.clone()),
            compartments_dir: Some(ws_paths.compartments_dir.clone()),
            max_memory_mb: self.max_memory_mb,
            max_symbols: self.max_symbols as usize,
            language,
        }
    }

    /// Load from a TOML file.
    pub fn load(path: &std::path::Path) -> WorkspaceResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| WorkspaceError::ConfigRead {
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
        validate_workspace_name(&config.name)?;

        let ws_paths = self.paths.workspace(&config.name);

        if ws_paths.exists() {
            return Err(WorkspaceError::AlreadyExists {
                name: config.name.clone(),
            });
        }

        // Create directory structure.
        ws_paths
            .ensure_dirs()
            .map_err(|e| WorkspaceError::ConfigWrite {
                path: ws_paths.root.display().to_string(),
                source: std::io::Error::other(e.to_string()),
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

    /// Return workspace names that fail the name validation rule.
    ///
    /// These are candidates for cleanup — they were typically created by agent
    /// OODA loops that passed query fragments or status strings as workspace names.
    pub fn list_invalid_names(&self) -> Vec<String> {
        self.list()
            .into_iter()
            .filter(|name| validate_workspace_name(name).is_err())
            .collect()
    }

    /// Delete all workspaces whose names fail validation.
    ///
    /// Returns the names of workspaces that were deleted and, separately, any
    /// that could not be deleted (with their error messages).
    pub fn delete_junk_workspaces(&self) -> (Vec<String>, Vec<(String, String)>) {
        let mut deleted = Vec::new();
        let mut errors = Vec::new();
        for name in self.list_invalid_names() {
            match self.delete(&name) {
                Ok(()) => deleted.push(name),
                Err(e) => errors.push((name, e.to_string())),
            }
        }
        (deleted, errors)
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
    if legacy.is_dir() { Some(legacy) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_workspace_names_accepted() {
        for name in &["default", "tecton", "my-ws", "ws_2", "a", "abc123", "x-y_z"] {
            assert!(
                validate_workspace_name(name).is_ok(),
                "expected \"{name}\" to be valid"
            );
        }
    }

    #[test]
    fn invalid_workspace_names_rejected() {
        let junk = [
            "status:active",
            "criteria:find star entities",
            "interlocutor:last-interaction",
            "docker build host",
            "episode:848",
            "",
            "Uppercase",
            "has space",
            "-leading-dash",
            &"a".repeat(64),
        ];
        for name in &junk {
            assert!(
                validate_workspace_name(name).is_err(),
                "expected \"{name}\" to be invalid"
            );
        }
    }

    #[test]
    fn create_rejects_invalid_name() {
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
        let err = mgr
            .create(WorkspaceConfig::with_name("status:active"))
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::InvalidName { .. }));
    }

    #[test]
    fn list_invalid_names_finds_junk() {
        let tmp = tempfile::TempDir::new().unwrap();
        let paths = AkhPaths {
            config_dir: tmp.path().join("config"),
            data_dir: tmp.path().join("data"),
            state_dir: tmp.path().join("state"),
            runtime_dir: tmp.path().join("run"),
            cache_dir: tmp.path().join("cache"),
        };
        paths.ensure_dirs().unwrap();
        let mgr = WorkspaceManager::new(paths.clone());

        // Create a valid workspace normally.
        mgr.create(WorkspaceConfig::with_name("good")).unwrap();

        // Simulate junk workspaces created by older agent runs: directly create
        // the directories that would have been made without name validation.
        let junk_names = ["status:active", "episode:848", "docker build host"];
        for junk in &junk_names {
            let junk_dir = paths.data_dir.join("workspaces").join(junk);
            std::fs::create_dir_all(&junk_dir).unwrap();
        }

        let invalid = mgr.list_invalid_names();
        for junk in &junk_names {
            assert!(invalid.contains(&junk.to_string()), "expected \"{junk}\" in invalid list");
        }
        assert!(!invalid.contains(&"good".to_string()));
    }

    #[test]
    fn delete_junk_workspaces_removes_them() {
        let tmp = tempfile::TempDir::new().unwrap();
        let paths = AkhPaths {
            config_dir: tmp.path().join("config"),
            data_dir: tmp.path().join("data"),
            state_dir: tmp.path().join("state"),
            runtime_dir: tmp.path().join("run"),
            cache_dir: tmp.path().join("cache"),
        };
        paths.ensure_dirs().unwrap();
        let mgr = WorkspaceManager::new(paths.clone());

        mgr.create(WorkspaceConfig::with_name("keep")).unwrap();
        for junk in &["status:active", "episode:848"] {
            std::fs::create_dir_all(paths.data_dir.join("workspaces").join(junk)).unwrap();
        }

        let (deleted, errors) = mgr.delete_junk_workspaces();
        assert_eq!(errors.len(), 0);
        assert_eq!(deleted.len(), 2);
        assert!(mgr.list().contains(&"keep".to_string()));
        assert!(!mgr.list().contains(&"status:active".to_string()));
    }

    #[test]
    fn default_config_has_seed_packs() {
        let cfg = WorkspaceConfig::default();
        assert_eq!(cfg.name, "default");
        assert_eq!(cfg.seed_packs.len(), 5);
        for pack in &["identity", "ontology", "common-sense", "computing", "architecture"] {
            assert!(
                cfg.seed_packs.contains(&pack.to_string()),
                "default seed_packs missing: {pack}"
            );
        }
    }

    #[test]
    fn legacy_config_without_seed_packs_gets_defaults() {
        // A workspace config saved before seed_packs existed (field absent from TOML)
        // must deserialize with the full default seed list so Tecton and other
        // pre-existing workspaces pick up seeds on next engine open.
        let toml = r#"
name = "tecton"
dimension = 10000
encoding = "bipolar"
language = "auto"
max_memory_mb = 1024
max_symbols = 1000000
"#;
        let cfg: WorkspaceConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.seed_packs.len(), 5, "missing seed_packs field must default to 5 packs");
        for pack in &["identity", "ontology", "common-sense", "computing", "architecture"] {
            assert!(
                cfg.seed_packs.contains(&pack.to_string()),
                "default seed_packs missing: {pack}"
            );
        }
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
