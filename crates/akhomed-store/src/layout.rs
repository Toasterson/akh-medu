//! Physical layout of the local SQLite backing store (ADR 120).
//!
//! PostgreSQL isolates writers per row, so one database serves every akh a
//! daemon hosts. SQLite has a single writer lock per *file*: with every akh's
//! periodic snapshot and WAL flush in one file, the akhs queue behind each
//! other and a slow snapshot surfaces as `database is locked (code: 5)` in an
//! unrelated akh. [`StoreLayout::PerAkh`] gives each akh its own file:
//!
//! * the **host** database (the configured `[database] url`) keeps the
//!   host-level tables — workspace registry, membership, active microtheories,
//!   the schema-pack registry and `seaql_migrations`;
//! * each **akh** database, `<host dir>/akh.<name>.sqlite3`, keeps the tables
//!   keyed by that akh ([`TABLE_HOMES`]), including `akhomed_akh`, whose
//!   `residency_epoch` fences the snapshot and WAL rows it sits beside.
//!
//! Every file carries the full schema (the one [`crate::Migrator`] runs per
//! file under its own `.migrate.lock`); a table outside a file's home set just
//! stays empty there. Residency lock files are anchored at the host path
//! (`<host>.akh-<uuid>.lock`) in both layouts, beside the file the deployment
//! configures. The shared layout's `akh.sqlite3.akh-*.lock` and a per-akh
//! host's `host.sqlite3.akh-*.lock` are different files, so a shared-layout
//! daemon and a per-akh daemon do NOT exclude each other: run exactly one core
//! per deployment (the Roman core has a single launchd label). A split writes
//! [`SplitMarker`] beside the new host; a per-akh daemon refuses to recreate a
//! listed akh's missing file. Under `PerAkh` each akh's file has exactly one connection in
//! the daemon, shared by its residency session (epoch, WAL) and its store
//! (snapshot, harvest, librarian, recall), so writers of one akh queue on that
//! connection instead of racing for SQLite's write lock.
//!
//! Files are addressed by akh **name**. Both store ids derive from it
//! ([`akh_id_for`], [`workspace_id_for`]), and an akh file holds rows keyed by
//! either id (`akhomed_akh` by akh id, `akhomed_facts` by workspace id), so
//! the name is the one identity that covers the whole file. Names are already
//! restricted to `[a-z0-9][a-z0-9_-]{0,62}` (the workspace-name rule), which
//! is safe as a file-name component, and an operator reading
//! `state/core/` sees which file belongs to which akh.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use miette::Diagnostic;
use sea_orm::sqlx::sqlite::SqliteConnectOptions;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, QuerySelect};
use thiserror::Error;
use uuid::Uuid;

use crate::entity::akh;
use crate::error::{StoreError, StoreResult};
use crate::pool::{PoolOptions, StablePgSession};

/// Namespace for deriving stable per-workspace UUIDs (UUIDv5). Fixed so the
/// same workspace name always maps to the same akh/workspace id across
/// restarts and hosts. The main crate's `wal::akh_id_for` delegates here.
pub const AKH_NAMESPACE: Uuid = Uuid::from_u128(0x616b_685f_6d65_6475_5f76_3500_0000_0001);

/// Stable workspace UUID for a workspace name.
pub fn workspace_id_for(name: &str) -> Uuid {
    Uuid::new_v5(&AKH_NAMESPACE, format!("workspace:{name}").as_bytes())
}

/// Stable akh UUID for a workspace name (one akh per workspace).
pub fn akh_id_for(name: &str) -> Uuid {
    Uuid::new_v5(&AKH_NAMESPACE, format!("akh:{name}").as_bytes())
}

/// How the backing store's tables map onto database files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StoreLayout {
    /// One database for every table (PostgreSQL, and SQLite before ADR 120).
    #[default]
    Shared,
    /// SQLite only: host-level tables in the configured file, each akh's
    /// tables in `<host dir>/akh.<name>.sqlite3`.
    PerAkh,
}

impl StoreLayout {
    /// The config spelling (`"shared"` / `"per_akh"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::PerAkh => "per_akh",
        }
    }
}

/// Errors in choosing or resolving a store layout.
#[derive(Debug, Error, Diagnostic)]
pub enum LayoutError {
    /// `per_akh` was requested for a non-SQLite URL.
    #[error("[database] layout = \"per_akh\" requires a sqlite: URL, got a {scheme}: URL")]
    #[diagnostic(
        code(akhomed_store::layout::requires_sqlite),
        help(
            "PostgreSQL already isolates writers per row, so it has no per-akh layout. \
             Either remove `layout` from [database] (or set it to \"shared\"), or point \
             `url` at an absolute sqlite:/// path."
        )
    )]
    RequiresSqlite { scheme: String },

    /// The URL is not an absolute on-disk SQLite path.
    #[error("per-akh layout needs an absolute on-disk SQLite path, got {url}")]
    #[diagnostic(
        code(akhomed_store::layout::path),
        help(
            "Use a URL like sqlite:///abs/dir/host.sqlite3?mode=rwc; the akh files are \
             created beside it as akh.<name>.sqlite3."
        )
    )]
    NotAbsolute { url: String },

    /// The akh name is not usable as a file-name component.
    #[error("akh name {name:?} cannot name a per-akh database file")]
    #[diagnostic(
        code(akhomed_store::layout::akh_name),
        help(
            "Per-akh files use the workspace-name rule: 1-63 characters from \
             [a-z0-9_-], starting with a lowercase letter or digit. Rename the workspace, \
             or keep this deployment on layout = \"shared\"."
        )
    )]
    InvalidAkhName { name: String },

    /// The host database would be the same file as an akh database.
    #[error("host database {path} collides with the per-akh file for akh {name:?}")]
    #[diagnostic(
        code(akhomed_store::layout::collision),
        help(
            "Rename the host database (for example host.sqlite3); names of the form \
             akh.<name>.sqlite3 are reserved for per-akh files."
        )
    )]
    HostCollidesWithAkh { path: PathBuf, name: String },

    /// The host database still holds this akh's rows, so per-akh mode would
    /// silently ignore (and later diverge from) them.
    #[error(
        "host database {host} still holds akh {name:?} (akhomed_akh row {akh_id}); \
         refusing to open {akh_file} under layout = \"per_akh\""
    )]
    #[diagnostic(
        code(akhomed_store::layout::unmigrated),
        help(
            "The configured [database] url is a shared-layout file. Options: (1) run \
             `akhomed store split-per-akh --source <this file> --host <new host file>` with \
             the daemon stopped, then point `url` at the new host file; (2) set \
             layout = \"shared\" to keep using this file as before. See \
             deploy/roman-mini/CUTOVER.md \"Per-akh databases\"."
        )
    )]
    UnmigratedAkh {
        name: String,
        akh_id: Uuid,
        host: PathBuf,
        akh_file: PathBuf,
    },

    /// A split created this akh's file and it is gone. Creating a fresh empty
    /// file would silently restart the akh from nothing.
    #[error(
        "the per-akh file {akh_file} of akh {name:?} is missing, but the split recorded in \
         {marker} created it; refusing to start the akh from an empty file"
    )]
    #[diagnostic(
        code(akhomed_store::layout::akh_file_missing),
        help(
            "Options: (1) restore akh.<name>.sqlite3 from a backup taken with the core stopped; \
             (2) roll back to the shared file (deploy/roman-mini/CUTOVER.md \"Per-akh \
             databases\", rollback); (3) if this akh should deliberately start empty, remove \
             its name from the marker's `akhs` list, then start the core again."
        )
    )]
    AkhFileMissing {
        name: String,
        akh_file: PathBuf,
        marker: PathBuf,
    },

    /// The split marker beside the host file cannot be read.
    #[error("split marker {path} is unreadable: {detail}")]
    #[diagnostic(
        code(akhomed_store::layout::marker),
        help(
            "The marker is written by `akhomed store split-per-akh` and lists the akhs the \
             split created. Restore it from the split's evidence directory or backup; do \
             not delete it to silence this error (it guards against a lost akh file)."
        )
    )]
    MarkerUnreadable { path: PathBuf, detail: String },
}

/// Schema of the split marker written beside the host database.
pub const SPLIT_MARKER_SCHEMA: &str = "akhomed.store.per_akh_split_marker.v1";

/// The split marker path for host database `host`: `<host>.per-akh-split.json`.
pub fn split_marker_path(host: &Path) -> PathBuf {
    PathBuf::from(format!("{}.per-akh-split.json", host.display()))
}

/// Written by `akhomed store split-per-akh` beside the new host file: the akhs
/// whose files the split created. A per-akh daemon refuses to create a fresh
/// file for a listed akh whose file is missing ([`LayoutError::AkhFileMissing`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SplitMarker {
    pub schema: String,
    pub akhs: Vec<String>,
    pub evidence: PathBuf,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl SplitMarker {
    /// Read the marker beside `host`, `Ok(None)` when there is none.
    pub fn read(host: &Path) -> Result<Option<Self>, LayoutError> {
        let path = split_marker_path(host);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(LayoutError::MarkerUnreadable {
                    path,
                    detail: error.to_string(),
                });
            }
        };
        let marker: Self =
            serde_json::from_str(&text).map_err(|error| LayoutError::MarkerUnreadable {
                path: path.clone(),
                detail: error.to_string(),
            })?;
        if marker.schema != SPLIT_MARKER_SCHEMA {
            return Err(LayoutError::MarkerUnreadable {
                path,
                detail: format!("schema {:?}, expected {SPLIT_MARKER_SCHEMA}", marker.schema),
            });
        }
        Ok(Some(marker))
    }
}

/// A validated akh name usable as a per-akh file-name component.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AkhStoreName(String);

impl AkhStoreName {
    /// Validate `name` against the workspace-name rule
    /// (`[a-z0-9][a-z0-9_-]{0,62}`).
    pub fn new(name: &str) -> Result<Self, LayoutError> {
        let mut chars = name.chars();
        let first_ok = chars
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
        let rest_ok =
            chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if first_ok && rest_ok && name.len() <= 63 {
            Ok(Self(name.to_owned()))
        } else {
            Err(LayoutError::InvalidAkhName {
                name: name.to_owned(),
            })
        }
    }

    /// The name as given.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The akh id derived from this name.
    pub fn akh_id(&self) -> Uuid {
        akh_id_for(&self.0)
    }

    /// The workspace id derived from this name.
    pub fn workspace_id(&self) -> Uuid {
        workspace_id_for(&self.0)
    }

    /// The per-akh database file name, `akh.<name>.sqlite3`.
    pub fn file_name(&self) -> String {
        format!("akh.{}.sqlite3", self.0)
    }
}

impl std::fmt::Display for AkhStoreName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// How an akh-homed table's rows are attributed to one akh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AkhKey {
    /// A column holding the akh id.
    AkhId(&'static str),
    /// A column holding the akh's own workspace id.
    WorkspaceId(&'static str),
    /// `akhomed_provenance_input`: rows whose `provenance_id` belongs to one
    /// of the akh's `akhomed_provenance` rows.
    ProvenanceOfAkh,
}

/// Which database file a table lives in under [`StoreLayout::PerAkh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableHome {
    /// The host database.
    Host,
    /// The akh's own database, with the rule attributing rows to it.
    Akh(AkhKey),
}

/// Every table the migrations create, with its per-akh home. The split tool
/// refuses a source holding a table missing here, so a new table cannot be
/// silently left behind (update this list with any new migration).
pub const TABLE_HOMES: &[(&str, TableHome)] = &[
    ("akhomed_akh", TableHome::Akh(AkhKey::AkhId("id"))),
    ("akhomed_akh_state", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    (
        "akhomed_akh_config",
        TableHome::Akh(AkhKey::WorkspaceId("workspace_id")),
    ),
    ("akhomed_symbols", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    (
        "akhomed_facts",
        TableHome::Akh(AkhKey::WorkspaceId("workspace_id")),
    ),
    (
        "akhomed_mutation_log",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    ("akhomed_episodes", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    (
        "akhomed_provenance",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_provenance_input",
        TableHome::Akh(AkhKey::ProvenanceOfAkh),
    ),
    (
        "akhomed_provenance_snapshot",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    ("akhomed_drives", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    ("akhomed_agenda", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    (
        "akhomed_curiosity_config",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    ("akhomed_goals", TableHome::Akh(AkhKey::AkhId("akh_id"))),
    (
        "akhomed_goal_hierarchy",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_goal_dependencies",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_failure_cases",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_effort_cases",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_learned_methods",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_learned_method_step",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_competence_tools",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_competence_categories",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_competence_calibration",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    (
        "akhomed_sleep_metrics",
        TableHome::Akh(AkhKey::AkhId("akh_id")),
    ),
    ("akhomed_schema_registry", TableHome::Host),
    ("workspaces_workspace", TableHome::Host),
    ("workspaces_workspace_seed_pack", TableHome::Host),
    ("workspaces_membership", TableHome::Host),
    ("workspaces_active_microtheories", TableHome::Host),
    // Migration bookkeeping: every file keeps its own copy, written by the
    // migrator, never by the split tool.
    ("seaql_migrations", TableHome::Host),
];

/// Look up a table's home. `None` for a table this build does not know.
pub fn table_home(table: &str) -> Option<TableHome> {
    TABLE_HOMES
        .iter()
        .find(|(name, _)| *name == table)
        .map(|(_, home)| *home)
}

/// A `sqlite:` URL split into its on-disk path and the pieces needed to
/// derive sibling URLs without re-encoding the directory.
#[derive(Debug, Clone)]
struct SqliteUrl {
    /// `sqlite://` or `sqlite:`.
    prefix: &'static str,
    /// The (still percent-encoded) directory part including the final `/`.
    dir: String,
    /// `?…` including the question mark, or empty.
    params: String,
    /// Decoded absolute file path.
    path: PathBuf,
}

impl SqliteUrl {
    fn parse(url: &str) -> Result<Self, LayoutError> {
        let prefix = if url.starts_with("sqlite://") {
            "sqlite://"
        } else if url.starts_with("sqlite:") {
            "sqlite:"
        } else {
            let scheme = url.split_once(':').map_or(url, |(scheme, _)| scheme);
            return Err(LayoutError::RequiresSqlite {
                scheme: scheme.to_owned(),
            });
        };
        let rest = &url[prefix.len()..];
        let (database, params) = match rest.split_once('?') {
            Some((database, params)) => (database, format!("?{params}")),
            None => (rest, String::new()),
        };
        let not_absolute = || LayoutError::NotAbsolute {
            url: url.to_owned(),
        };
        let options = SqliteConnectOptions::from_str(url).map_err(|_| not_absolute())?;
        let path = options.get_filename().to_path_buf();
        if !path.is_absolute() {
            return Err(not_absolute());
        }
        let slash = database.rfind('/').ok_or_else(not_absolute)?;
        Ok(Self {
            prefix,
            dir: database[..=slash].to_owned(),
            params,
            path,
        })
    }

    fn sibling(&self, file_name: &str) -> String {
        format!("{}{}{}{}", self.prefix, self.dir, file_name, self.params)
    }
}

/// The resolved backing-store topology: the host pool options plus the layout
/// that says where each akh's tables live.
#[derive(Debug, Clone)]
pub struct StoreTopology {
    host: PoolOptions,
    layout: StoreLayout,
    /// Parsed host URL; present exactly when `layout` is `PerAkh`.
    sqlite: Option<SqliteUrl>,
}

impl StoreTopology {
    /// Resolve a topology. `PerAkh` requires an absolute SQLite URL; `Shared`
    /// accepts anything [`crate::connect`] accepts and changes nothing.
    pub fn new(host: PoolOptions, layout: StoreLayout) -> Result<Self, LayoutError> {
        let sqlite = match layout {
            StoreLayout::Shared => None,
            StoreLayout::PerAkh => Some(SqliteUrl::parse(&host.url)?),
        };
        Ok(Self {
            host,
            layout,
            sqlite,
        })
    }

    /// The configured layout.
    pub fn layout(&self) -> StoreLayout {
        self.layout
    }

    /// Pool options of the host database (the configured URL).
    pub fn host(&self) -> &PoolOptions {
        &self.host
    }

    /// Absolute path of the host database, for SQLite per-akh layouts.
    pub fn host_path(&self) -> Option<&Path> {
        self.sqlite.as_ref().map(|url| url.path.as_path())
    }

    /// Absolute path of `name`'s database under the per-akh layout.
    pub fn akh_path(&self, name: &AkhStoreName) -> Option<PathBuf> {
        self.sqlite
            .as_ref()
            .and_then(|url| url.path.parent())
            .map(|dir| dir.join(name.file_name()))
    }

    /// Pool options for `name`'s tables: the host options under `Shared`,
    /// the sibling `akh.<name>.sqlite3` (same query parameters, one
    /// connection) under `PerAkh`.
    pub fn akh_options(&self, name: &AkhStoreName) -> Result<PoolOptions, LayoutError> {
        let Some(url) = &self.sqlite else {
            return Ok(self.host.clone());
        };
        let akh_url = url.sibling(&name.file_name());
        let parsed = SqliteUrl::parse(&akh_url)?;
        if parsed.path == url.path {
            return Err(LayoutError::HostCollidesWithAkh {
                path: url.path.clone(),
                name: name.as_str().to_owned(),
            });
        }
        Ok(PoolOptions {
            url: akh_url,
            max_connections: 1,
            connect_timeout: self.host.connect_timeout,
        })
    }

    /// Open `name`'s store under the per-akh layout: refuse when the host
    /// database still holds the akh's rows ([`LayoutError::UnmigratedAkh`]),
    /// run the migrations on the akh file under its own migration lock, then
    /// connect. Under `Shared` this returns `host` unchanged, so the caller's
    /// shared pool keeps serving every akh exactly as before.
    pub async fn open_akh_store(
        &self,
        name: &AkhStoreName,
        host: &DatabaseConnection,
    ) -> StoreResult<DatabaseConnection> {
        if self.layout == StoreLayout::Shared {
            return Ok(host.clone());
        }
        let options = self.akh_options(name)?;
        // A missing file is created by the migration below. That is right for
        // an akh born after the split, never for one the split created.
        if let (Some(host_path), Some(akh_file)) = (self.host_path(), self.akh_path(name))
            && !akh_file.exists()
            && let Some(marker) = SplitMarker::read(host_path)?
            && marker.akhs.iter().any(|akh| akh == name.as_str())
        {
            return Err(LayoutError::AkhFileMissing {
                name: name.as_str().to_owned(),
                akh_file,
                marker: split_marker_path(host_path),
            }
            .into());
        }
        let akh_id = name.akh_id();
        let legacy = akh::Entity::find()
            .select_only()
            .column(akh::Column::Id)
            .filter(akh::Column::Id.eq(akh_id))
            .into_tuple::<Uuid>()
            .one(host)
            .await?;
        if legacy.is_some() {
            return Err(LayoutError::UnmigratedAkh {
                name: name.as_str().to_owned(),
                akh_id,
                host: self.host_path().map(Path::to_path_buf).unwrap_or_default(),
                akh_file: self.akh_path(name).unwrap_or_default(),
            }
            .into());
        }
        crate::run_migrations_locked(&options).await?;
        crate::connect(&options).await
    }

    /// Open the residency session for `name`.
    ///
    /// Under `PerAkh` the session writes its fenced epoch and WAL rows through
    /// `akh_db` — the connection [`Self::open_akh_store`] returned — so every
    /// writer of the akh's file in this process shares one connection and
    /// queues on its pool (acquire timeout = `connect_timeout`). Separate
    /// connections would race for SQLite's write lock, and a deferred
    /// transaction that has already read fails with `SQLITE_BUSY` at once,
    /// without waiting for `busy_timeout`. The lock file stays anchored at
    /// the host path.
    ///
    /// Under `Shared` this is [`StablePgSession::connect`] on the host
    /// options, exactly as before ADR 120, and `akh_db` is not used.
    pub async fn connect_residency(
        &self,
        name: &AkhStoreName,
        akh_db: &DatabaseConnection,
    ) -> StoreResult<StablePgSession> {
        match self.host_path() {
            None => StablePgSession::connect(&self.host).await,
            Some(anchor) => {
                // Validates the name against the host path as well.
                self.akh_options(name)?;
                StablePgSession::sqlite_on_connection(akh_db.clone(), anchor)
            }
        }
    }
}

impl From<LayoutError> for StoreError {
    fn from(error: LayoutError) -> Self {
        StoreError::Layout(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(url: &str) -> StoreTopology {
        StoreTopology::new(PoolOptions::new(url), StoreLayout::PerAkh).expect("valid topology")
    }

    #[test]
    fn ids_match_the_historical_wal_derivation() {
        // The main crate's `wal::akh_id_for` produced these for years; the
        // store-side derivation must never drift from them.
        assert_eq!(
            akh_id_for("janus").to_string(),
            Uuid::new_v5(&AKH_NAMESPACE, b"akh:janus").to_string()
        );
        assert_ne!(akh_id_for("janus"), workspace_id_for("janus"));
    }

    #[test]
    fn akh_file_sits_beside_the_host_and_keeps_query_parameters() {
        let topo = topology("sqlite:///srv/a b/state/core/host.sqlite3?mode=rwc");
        let name = AkhStoreName::new("janus").unwrap();
        let opts = topo.akh_options(&name).unwrap();
        assert_eq!(
            opts.url,
            "sqlite:///srv/a b/state/core/akh.janus.sqlite3?mode=rwc"
        );
        assert_eq!(opts.max_connections, 1);
        assert_eq!(
            topo.akh_path(&name).unwrap(),
            PathBuf::from("/srv/a b/state/core/akh.janus.sqlite3")
        );
        assert_eq!(
            topo.host_path().unwrap(),
            Path::new("/srv/a b/state/core/host.sqlite3")
        );
    }

    #[test]
    fn shared_layout_keeps_the_host_options() {
        let host = PoolOptions::new("postgres://db.local/akh");
        let topo = StoreTopology::new(host.clone(), StoreLayout::Shared).unwrap();
        let opts = topo
            .akh_options(&AkhStoreName::new("janus").unwrap())
            .unwrap();
        assert_eq!(opts.url, host.url);
        assert_eq!(opts.max_connections, host.max_connections);
        assert!(topo.host_path().is_none());
    }

    #[test]
    fn per_akh_refuses_postgres_and_relative_paths() {
        let pg = StoreTopology::new(
            PoolOptions::new("postgres://db.local/akh"),
            StoreLayout::PerAkh,
        );
        assert!(matches!(pg, Err(LayoutError::RequiresSqlite { scheme }) if scheme == "postgres"));
        let relative = StoreTopology::new(
            PoolOptions::new("sqlite://relative.sqlite3"),
            StoreLayout::PerAkh,
        );
        assert!(matches!(relative, Err(LayoutError::NotAbsolute { .. })));
    }

    #[test]
    fn host_named_like_an_akh_file_is_refused() {
        let topo = topology("sqlite:///srv/akh.janus.sqlite3");
        let err = topo
            .akh_options(&AkhStoreName::new("janus").unwrap())
            .unwrap_err();
        assert!(matches!(err, LayoutError::HostCollidesWithAkh { .. }));
    }

    #[test]
    fn akh_names_follow_the_workspace_rule() {
        for ok in ["janus", "a", "lilith-2", "x_y", "0abc"] {
            assert!(AkhStoreName::new(ok).is_ok(), "{ok}");
        }
        let long = "a".repeat(64);
        for bad in [
            "",
            "Janus",
            "-x",
            "../etc",
            "a/b",
            "a.b",
            "a b",
            long.as_str(),
        ] {
            assert!(AkhStoreName::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn every_table_has_one_home() {
        let mut names: Vec<&str> = TABLE_HOMES.iter().map(|(name, _)| *name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate table in TABLE_HOMES");
        assert_eq!(
            table_home("akhomed_facts"),
            Some(TableHome::Akh(AkhKey::WorkspaceId("workspace_id")))
        );
        assert_eq!(table_home("workspaces_membership"), Some(TableHome::Host));
        assert_eq!(table_home("nope"), None);
    }
}
