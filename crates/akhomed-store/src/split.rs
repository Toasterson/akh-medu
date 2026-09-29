//! Offline split of a shared-layout SQLite store into the per-akh layout
//! (ADR 120).
//!
//! [`plan`] reads the source read-only and reports, per akh and per table, the
//! rows a split would copy (`--dry-run`). [`execute`] then:
//!
//! 1. refuses while any residency or migration lock file beside the source is
//!    held (the daemon is running), and keeps every one of them locked until
//!    it finishes, so a daemon cannot start mid-copy;
//! 2. refuses existing targets, unknown tables, schema versions this build
//!    does not know, akh rows whose id does not derive from their name, and
//!    rows it cannot attribute to an akh (unless told to leave those behind);
//! 3. creates the new host file and one `akh.<name>.sqlite3` per akh beside
//!    it, runs the migrations on each, and copies rows in **one transaction per
//!    file**, value-for-value (SQLite storage classes preserved);
//! 4. re-reads every target table and compares its row count and a SHA-256
//!    content digest with the source rows;
//! 5. hashes the source file (and its `-wal`) before and after, and fails if
//!    either changed;
//! 6. writes an evidence JSON (0600) describing all of the above.
//!
//! The source is opened read-only (`immutable` when it has no pending WAL)
//! and is never modified or deleted; the operator removes it after the
//! cutover. On any failure the target files this run created are deleted.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::time::Duration;

use miette::Diagnostic;
use sea_orm::sqlx::sqlite::{SqliteConnectOptions, SqliteConnection, SqliteRow};
use sea_orm::sqlx::{self, ConnectOptions as _, Connection as _, Row as _, TypeInfo as _, ValueRef as _};
use sea_orm_migration::MigratorTrait as _;
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::error::StoreError;
use crate::layout::{AkhKey, AkhStoreName, LayoutError, StoreLayout, StoreTopology, TableHome};
use crate::pool::PoolOptions;

/// Schema identifier of the evidence JSON.
pub const EVIDENCE_SCHEMA: &str = "akhomed.store.split_per_akh.v1";

/// Bind parameters per statement, below SQLite's default 32,766 ceiling.
const SQLITE_MAX_BINDS: usize = 32_000;

/// Errors from planning or executing a split.
#[derive(Debug, Error, Diagnostic)]
pub enum SplitError {
    /// A lock file beside the source is held: the daemon is running.
    #[error("source database is in use: {lock} is locked by another process")]
    #[diagnostic(
        code(akhomed_store::split::source_in_use),
        help(
            "Stop the akhomed that serves this file first (Roman core: \
             `sudo launchctl bootout system/<core label>`), then rerun. `--dry-run` \
             works while the daemon runs; the real split does not."
        )
    )]
    SourceInUse { lock: PathBuf },

    /// The source file does not exist.
    #[error("source database {path} does not exist")]
    #[diagnostic(
        code(akhomed_store::split::source_missing),
        help("Pass the shared-layout file named by the current `[database] url`.")
    )]
    SourceMissing { path: PathBuf },

    /// The new host file is the source file.
    #[error("the new host database {path} is the source database")]
    #[diagnostic(
        code(akhomed_store::split::same_file),
        help(
            "The split never writes into the source. Choose a new host file, for \
             example host.sqlite3 in the same directory."
        )
    )]
    HostIsSource { path: PathBuf },

    /// A target (or its evidence file) already exists.
    #[error("target {path} already exists")]
    #[diagnostic(
        code(akhomed_store::split::target_exists),
        help(
            "The split only creates new files. If it is left over from an earlier split, \
             confirm that split is not in use, then move it away and rerun; or choose \
             another --host path / --evidence path. Never remove the source this way."
        )
    )]
    TargetExists { path: PathBuf },

    /// The source holds a table this build does not know.
    #[error("source table {table} has no per-akh home in this build")]
    #[diagnostic(
        code(akhomed_store::split::unknown_table),
        help(
            "The source was written by a newer akhomed, or the table was created by hand. \
             Run the split with the akhomed build that created it (from origin/main), \
             or keep layout = \"shared\"."
        )
    )]
    UnknownTable { table: String },

    /// The source was migrated beyond this build.
    #[error("source migration {version} is unknown to this build")]
    #[diagnostic(
        code(akhomed_store::split::schema_ahead),
        help(
            "Use an akhomed built from the origin/main commit that migrated the source \
             (ADR 118: schema version is a deployment compatibility boundary)."
        )
    )]
    SchemaAhead { version: String },

    /// An `akhomed_akh` row cannot address a per-akh file.
    #[error("akhomed_akh row {akh_id} named {name:?} cannot be split: {reason}")]
    #[diagnostic(
        code(akhomed_store::split::identity),
        help(
            "Per-akh files are named after the akh and every id is derived from that name \
             (UUIDv5). A row whose id does not derive from its name was written by hand \
             or by a foreign tool; repair it through the engine first, or keep \
             layout = \"shared\"."
        )
    )]
    Identity {
        akh_id: Uuid,
        name: String,
        reason: String,
    },

    /// Rows that belong to no akh in `akhomed_akh`.
    #[error("{rows} row(s) in akh-scoped tables belong to no akh in akhomed_akh: {tables}")]
    #[diagnostic(
        code(akhomed_store::split::orphans),
        help(
            "These rows would not be copied anywhere. Options: (1) rerun with \
             --allow-orphans to leave them only in the source (the evidence lists them); \
             (2) investigate with `--dry-run --json` first."
        )
    )]
    Orphans { rows: u64, tables: String },

    /// A copied table does not match its source rows.
    #[error("verification failed for table {table} in {path}: {detail}")]
    #[diagnostic(
        code(akhomed_store::split::verification),
        help(
            "The run's target files were removed. This is a bug or a concurrent writer; \
             confirm the daemon is stopped and rerun. The source is untouched."
        )
    )]
    Verification {
        table: String,
        path: PathBuf,
        detail: String,
    },

    /// The source changed while the split ran.
    #[error("source file {path} changed during the split")]
    #[diagnostic(
        code(akhomed_store::split::source_changed),
        help(
            "Something wrote to the source while it was being copied (the daemon was \
             restarted, or another tool touched it). The run's targets were removed; stop \
             every writer and rerun."
        )
    )]
    SourceChanged { path: PathBuf },

    /// A SQLite statement failed.
    #[error("SQLite error while {context}: {source}")]
    #[diagnostic(
        code(akhomed_store::split::sqlite),
        help("See the underlying SQLite error. The source is opened read-only and never written.")
    )]
    Sqlite {
        context: String,
        #[source]
        source: sqlx::Error,
    },

    /// A filesystem operation failed.
    #[error("I/O error on {path}: {source}")]
    #[diagnostic(
        code(akhomed_store::split::io),
        help("Check that the directory exists, is writable by this user, and has free space.")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Layout resolution failed.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Layout(#[from] LayoutError),

    /// A store operation (migration, connect) failed.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] StoreError),
}

/// Result alias for split operations.
pub type SplitResult<T> = std::result::Result<T, SplitError>;

fn sqlite_err(context: impl Into<String>) -> impl FnOnce(sqlx::Error) -> SplitError {
    let context = context.into();
    move |source| SplitError::Sqlite { context, source }
}

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> SplitError + '_ {
    move |source| SplitError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// What to split and where.
#[derive(Debug, Clone)]
pub struct SplitRequest {
    /// The existing shared-layout database (read-only input).
    pub source: PathBuf,
    /// The new host database to create; akh files are created beside it.
    pub host: PathBuf,
    /// Leave rows that belong to no akh in the source instead of refusing.
    pub allow_orphans: bool,
    /// SQLite busy timeout for every connection the split opens.
    pub busy_timeout: Duration,
}

/// Row count of one table.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TableCount {
    pub table: String,
    pub rows: u64,
}

/// The rows a split would copy into one akh's file.
#[derive(Debug, Clone, Serialize)]
pub struct AkhPlan {
    pub name: String,
    pub akh_id: Uuid,
    pub workspace_id: Uuid,
    pub path: PathBuf,
    pub tables: Vec<TableCount>,
}

impl AkhPlan {
    /// Total rows across all tables.
    pub fn total_rows(&self) -> u64 {
        self.tables.iter().map(|t| t.rows).sum()
    }
}

/// Read-only description of a split, produced by [`plan`].
#[derive(Debug, Clone, Serialize)]
pub struct SplitPlan {
    pub source: PathBuf,
    pub host: PathBuf,
    /// Lock files beside the source currently held by another process. A
    /// dry-run reports them; [`execute`] refuses.
    pub in_use: Vec<PathBuf>,
    pub source_migrations: Vec<String>,
    pub host_tables: Vec<TableCount>,
    pub akhs: Vec<AkhPlan>,
    /// Rows in akh-scoped tables attributed to no akh.
    pub orphans: Vec<TableCount>,
}

/// Count and digest of one copied table, source vs target.
#[derive(Debug, Clone, Serialize)]
pub struct TableEvidence {
    pub table: String,
    pub source_rows: u64,
    pub source_sha256: String,
    pub target_rows: u64,
    pub target_sha256: String,
}

/// One target file's verified content.
#[derive(Debug, Clone, Serialize)]
pub struct TargetEvidence {
    pub path: PathBuf,
    pub tables: Vec<TableEvidence>,
}

/// One akh's verified file.
#[derive(Debug, Clone, Serialize)]
pub struct AkhEvidence {
    pub name: String,
    pub akh_id: Uuid,
    pub workspace_id: Uuid,
    #[serde(flatten)]
    pub target: TargetEvidence,
}

/// Source file identity before and after the split.
#[derive(Debug, Clone, Serialize)]
pub struct SourceEvidence {
    pub path: PathBuf,
    pub sha256_before: String,
    pub sha256_after: String,
    /// SHA-256 of `<source>-wal` (a missing file hashes as empty).
    pub wal_sha256_before: String,
    pub wal_sha256_after: String,
    pub opened: &'static str,
    pub migrations: Vec<String>,
}

/// The evidence JSON written by [`execute`].
#[derive(Debug, Clone, Serialize)]
pub struct SplitEvidence {
    pub schema: &'static str,
    pub tool: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: chrono::DateTime<chrono::Utc>,
    pub layout: &'static str,
    pub source: SourceEvidence,
    pub host: TargetEvidence,
    pub akhs: Vec<AkhEvidence>,
    pub orphans_left_in_source: Vec<TableCount>,
    pub verified: bool,
}

/// One SQLite value, storage class preserved.
#[derive(Debug, Clone)]
enum Cell {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Cell {
    fn decode(row: &SqliteRow, index: usize, context: &str) -> SplitResult<Self> {
        let raw = row
            .try_get_raw(index)
            .map_err(sqlite_err(format!("reading {context}")))?;
        if raw.is_null() {
            return Ok(Cell::Null);
        }
        let class = raw.type_info().name().to_owned();
        let get = |e| sqlite_err(format!("decoding {context} ({class})"))(e);
        Ok(match class.as_str() {
            "INTEGER" => Cell::Integer(row.try_get::<i64, _>(index).map_err(get)?),
            "REAL" => Cell::Real(row.try_get::<f64, _>(index).map_err(get)?),
            "TEXT" => Cell::Text(row.try_get::<String, _>(index).map_err(get)?),
            "BLOB" => Cell::Blob(row.try_get::<Vec<u8>, _>(index).map_err(get)?),
            other => {
                return Err(SplitError::Verification {
                    table: context.to_owned(),
                    path: PathBuf::new(),
                    detail: format!("column {index} has storage class {other}"),
                });
            }
        })
    }

    fn digest_into(&self, hasher: &mut Sha256) {
        match self {
            Cell::Null => hasher.update([0u8]),
            Cell::Integer(v) => {
                hasher.update([1u8]);
                hasher.update(v.to_le_bytes());
            }
            Cell::Real(v) => {
                hasher.update([2u8]);
                hasher.update(v.to_bits().to_le_bytes());
            }
            Cell::Text(v) => {
                hasher.update([3u8]);
                hasher.update((v.len() as u64).to_le_bytes());
                hasher.update(v.as_bytes());
            }
            Cell::Blob(v) => {
                hasher.update([4u8]);
                hasher.update((v.len() as u64).to_le_bytes());
                hasher.update(v);
            }
        }
    }
}

/// A table's columns and primary-key order.
#[derive(Debug, Clone)]
struct TableShape {
    name: String,
    columns: Vec<String>,
    order_by: String,
}

fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

impl TableShape {
    async fn read(conn: &mut SqliteConnection, table: &str) -> SplitResult<Self> {
        let rows = sqlx::query(&format!("PRAGMA table_info({})", quote_ident(table)))
            .fetch_all(&mut *conn)
            .await
            .map_err(sqlite_err(format!("reading the columns of {table}")))?;
        let mut columns = Vec::with_capacity(rows.len());
        let mut pk: Vec<(i64, String)> = Vec::new();
        for row in &rows {
            let name: String = row.try_get("name").map_err(sqlite_err("reading table_info"))?;
            let ordinal: i64 = row.try_get("pk").map_err(sqlite_err("reading table_info"))?;
            if ordinal > 0 {
                pk.push((ordinal, name.clone()));
            }
            columns.push(name);
        }
        pk.sort();
        let order_by = if pk.is_empty() {
            "rowid".to_owned()
        } else {
            pk.iter()
                .map(|(_, name)| quote_ident(name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        Ok(Self {
            name: table.to_owned(),
            columns,
            order_by,
        })
    }

    fn column_list(&self) -> String {
        self.columns
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Which rows of a table to read.
#[derive(Debug, Clone, Copy)]
enum RowFilter {
    All,
    Akh { key: AkhKey, akh_id: Uuid, workspace_id: Uuid },
}

impl RowFilter {
    fn clause(&self) -> String {
        match self {
            RowFilter::All => String::new(),
            RowFilter::Akh { key, .. } => match key {
                AkhKey::AkhId(column) | AkhKey::WorkspaceId(column) => {
                    format!(" WHERE {} = ?", quote_ident(column))
                }
                AkhKey::ProvenanceOfAkh => " WHERE \"provenance_id\" IN (SELECT \"provenance_id\" \
                     FROM \"akhomed_provenance\" WHERE \"akh_id\" = ?)"
                    .to_owned(),
            },
        }
    }

    fn bind(&self) -> Option<Vec<u8>> {
        match self {
            RowFilter::All => None,
            RowFilter::Akh {
                key,
                akh_id,
                workspace_id,
            } => Some(match key {
                AkhKey::WorkspaceId(_) => workspace_id.as_bytes().to_vec(),
                AkhKey::AkhId(_) | AkhKey::ProvenanceOfAkh => akh_id.as_bytes().to_vec(),
            }),
        }
    }
}

async fn count_rows(
    conn: &mut SqliteConnection,
    table: &str,
    filter: RowFilter,
) -> SplitResult<u64> {
    let sql = format!("SELECT COUNT(*) FROM {}{}", quote_ident(table), filter.clause());
    let mut query = sqlx::query_scalar::<_, i64>(&sql);
    if let Some(bytes) = filter.bind() {
        query = query.bind(bytes);
    }
    let count = query
        .fetch_one(&mut *conn)
        .await
        .map_err(sqlite_err(format!("counting {table}")))?;
    Ok(u64::try_from(count).unwrap_or(0))
}

/// Read a table's rows (filtered, primary-key order) and their digest.
async fn read_rows(
    conn: &mut SqliteConnection,
    shape: &TableShape,
    filter: RowFilter,
) -> SplitResult<(Vec<Vec<Cell>>, String)> {
    let sql = format!(
        "SELECT {} FROM {}{} ORDER BY {}",
        shape.column_list(),
        quote_ident(&shape.name),
        filter.clause(),
        shape.order_by
    );
    let mut query = sqlx::query(&sql);
    if let Some(bytes) = filter.bind() {
        query = query.bind(bytes);
    }
    let rows = query
        .fetch_all(&mut *conn)
        .await
        .map_err(sqlite_err(format!("reading {}", shape.name)))?;
    let mut hasher = Sha256::new();
    hasher.update(shape.name.as_bytes());
    for column in &shape.columns {
        hasher.update([0xffu8]);
        hasher.update(column.as_bytes());
    }
    let mut cells = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut values = Vec::with_capacity(shape.columns.len());
        for index in 0..shape.columns.len() {
            let cell = Cell::decode(row, index, &shape.name)?;
            cell.digest_into(&mut hasher);
            values.push(cell);
        }
        cells.push(values);
    }
    Ok((cells, hex(&hasher.finalize())))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

async fn insert_rows(
    conn: &mut SqliteConnection,
    shape: &TableShape,
    rows: &[Vec<Cell>],
) -> SplitResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let width = shape.columns.len();
    let per_statement = (SQLITE_MAX_BINDS / width.max(1)).max(1);
    let tuple = format!("({})", vec!["?"; width].join(", "));
    for chunk in rows.chunks(per_statement) {
        let sql = format!(
            "INSERT INTO {} ({}) VALUES {}",
            quote_ident(&shape.name),
            shape.column_list(),
            vec![tuple.as_str(); chunk.len()].join(", ")
        );
        let mut query = sqlx::query(&sql);
        for row in chunk {
            for cell in row {
                query = match cell {
                    Cell::Null => query.bind(Option::<i64>::None),
                    Cell::Integer(v) => query.bind(*v),
                    Cell::Real(v) => query.bind(*v),
                    Cell::Text(v) => query.bind(v.clone()),
                    Cell::Blob(v) => query.bind(v.clone()),
                };
            }
        }
        query
            .execute(&mut *conn)
            .await
            .map_err(sqlite_err(format!("inserting into {}", shape.name)))?;
    }
    Ok(())
}

/// The source opened read-only, with its tables and akhs discovered.
struct Source {
    conn: SqliteConnection,
    opened: &'static str,
    tables: Vec<String>,
    migrations: Vec<String>,
    akhs: Vec<AkhStoreName>,
}

fn wal_path(db: &Path) -> PathBuf {
    PathBuf::from(format!("{}-wal", db.display()))
}

fn has_pending_wal(db: &Path) -> bool {
    std::fs::metadata(wal_path(db)).is_ok_and(|m| m.len() > 0)
}

impl Source {
    async fn open(path: &Path, busy_timeout: Duration) -> SplitResult<Self> {
        if !path.is_file() {
            return Err(SplitError::SourceMissing {
                path: path.to_path_buf(),
            });
        }
        // Without a pending WAL the file is complete on its own: open it
        // `immutable`, so SQLite takes no locks and creates no -shm/-wal.
        // With one, a plain read-only open reads the WAL too.
        let immutable = !has_pending_wal(path);
        let options = SqliteConnectOptions::new()
            .filename(path)
            .read_only(true)
            .immutable(immutable)
            .create_if_missing(false)
            .busy_timeout(busy_timeout)
            .disable_statement_logging();
        let mut conn = options
            .connect()
            .await
            .map_err(sqlite_err(format!("opening {} read-only", path.display())))?;

        let tables: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master WHERE type = 'table' \
             AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .fetch_all(&mut conn)
        .await
        .map_err(sqlite_err("listing source tables"))?;
        for table in &tables {
            if crate::layout::table_home(table).is_none() {
                return Err(SplitError::UnknownTable {
                    table: table.clone(),
                });
            }
        }

        let migrations: Vec<String> = if tables.iter().any(|t| t == "seaql_migrations") {
            sqlx::query_scalar("SELECT version FROM seaql_migrations ORDER BY version")
                .fetch_all(&mut conn)
                .await
                .map_err(sqlite_err("reading seaql_migrations"))?
        } else {
            Vec::new()
        };
        let known: BTreeSet<String> = crate::Migrator::migrations()
            .iter()
            .map(|m| m.name().to_owned())
            .collect();
        if let Some(version) = migrations.iter().find(|v| !known.contains(*v)) {
            return Err(SplitError::SchemaAhead {
                version: version.clone(),
            });
        }

        let mut akhs = Vec::new();
        if tables.iter().any(|t| t == "akhomed_akh") {
            let rows = sqlx::query("SELECT id, name FROM akhomed_akh ORDER BY name")
                .fetch_all(&mut conn)
                .await
                .map_err(sqlite_err("reading akhomed_akh"))?;
            let mut seen = BTreeSet::new();
            for row in rows {
                let id_bytes: Vec<u8> = row.try_get("id").map_err(sqlite_err("reading akh id"))?;
                let name: String = row.try_get("name").map_err(sqlite_err("reading akh name"))?;
                let akh_id = Uuid::from_slice(&id_bytes).map_err(|_| SplitError::Identity {
                    akh_id: Uuid::nil(),
                    name: name.clone(),
                    reason: format!("id is {} bytes, not a 16-byte UUID", id_bytes.len()),
                })?;
                let store_name =
                    AkhStoreName::new(&name).map_err(|error| SplitError::Identity {
                        akh_id,
                        name: name.clone(),
                        reason: error.to_string(),
                    })?;
                if store_name.akh_id() != akh_id {
                    return Err(SplitError::Identity {
                        akh_id,
                        name,
                        reason: format!("the id derived from the name is {}", store_name.akh_id()),
                    });
                }
                if !seen.insert(name.clone()) {
                    return Err(SplitError::Identity {
                        akh_id,
                        name,
                        reason: "two akhomed_akh rows share this name".into(),
                    });
                }
                akhs.push(store_name);
            }
        }

        Ok(Self {
            conn,
            opened: if immutable { "read_only_immutable" } else { "read_only" },
            tables,
            migrations,
            akhs,
        })
    }

    /// Akh-scoped tables present in the source, with their key.
    fn akh_tables(&self) -> Vec<(String, AkhKey)> {
        self.tables
            .iter()
            .filter_map(|t| match crate::layout::table_home(t) {
                Some(TableHome::Akh(key)) => Some((t.clone(), key)),
                _ => None,
            })
            .collect()
    }

    /// Host tables the split copies (everything host-homed except the
    /// migrator's own bookkeeping, which each target writes itself).
    fn host_tables(&self) -> Vec<String> {
        self.tables
            .iter()
            .filter(|t| {
                *t != "seaql_migrations"
                    && crate::layout::table_home(t) == Some(TableHome::Host)
            })
            .cloned()
            .collect()
    }
}

fn filter_for(name: &AkhStoreName, key: AkhKey) -> RowFilter {
    RowFilter::Akh {
        key,
        akh_id: name.akh_id(),
        workspace_id: name.workspace_id(),
    }
}

fn topology_for(req: &SplitRequest) -> SplitResult<StoreTopology> {
    let url = format!("sqlite://{}?mode=rwc", req.host.display());
    let host = PoolOptions {
        url,
        max_connections: 1,
        connect_timeout: req.busy_timeout,
    };
    Ok(StoreTopology::new(host, StoreLayout::PerAkh)?)
}

fn absolute(path: &Path) -> SplitResult<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir().map_err(io_err(path))?;
    Ok(cwd.join(path))
}

/// Lock files beside `db` (`<db>.akh-*.lock`, `<db>.migrate.lock`).
fn lock_files_beside(db: &Path) -> SplitResult<Vec<PathBuf>> {
    let Some(dir) = db.parent() else {
        return Ok(Vec::new());
    };
    let Some(file_name) = db.file_name().and_then(|n| n.to_str()) else {
        return Ok(Vec::new());
    };
    let residency_prefix = format!("{file_name}.akh-");
    let migrate = format!("{file_name}.migrate.lock");
    let mut locks = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(io_err(dir))? {
        let entry = entry.map_err(io_err(dir))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if (name.starts_with(&residency_prefix) && name.ends_with(".lock")) || name == migrate {
            locks.push(entry.path());
        }
    }
    locks.sort();
    Ok(locks)
}

/// Try to lock every existing lock file beside the source. Returns the held
/// handles and the lock files another process holds.
fn try_lock_all(locks: &[PathBuf]) -> SplitResult<(Vec<File>, Vec<PathBuf>)> {
    let mut held = Vec::new();
    let mut busy = Vec::new();
    for lock in locks {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock)
            .map_err(io_err(lock))?;
        match file.try_lock() {
            Ok(()) => held.push(file),
            Err(std::fs::TryLockError::WouldBlock) => busy.push(lock.clone()),
            Err(std::fs::TryLockError::Error(source)) => {
                return Err(SplitError::Io {
                    path: lock.clone(),
                    source,
                });
            }
        }
    }
    Ok((held, busy))
}

/// Plan a split: read-only, creates nothing. Works while the daemon runs;
/// held lock files are reported in [`SplitPlan::in_use`].
pub async fn plan(req: &SplitRequest) -> SplitResult<SplitPlan> {
    let source_path = absolute(&req.source)?;
    let host_path = absolute(&req.host)?;
    let topology = topology_for(&SplitRequest {
        host: host_path.clone(),
        ..req.clone()
    })?;
    let (held, in_use) = try_lock_all(&lock_files_beside(&source_path)?)?;
    drop(held);
    let mut source = Source::open(&source_path, req.busy_timeout).await?;
    let plan = build_plan(&mut source, source_path, host_path, &topology, in_use).await;
    let _ = source.conn.close().await;
    plan
}

async fn build_plan(
    source: &mut Source,
    source_path: PathBuf,
    host_path: PathBuf,
    topology: &StoreTopology,
    in_use: Vec<PathBuf>,
) -> SplitResult<SplitPlan> {
    let mut host_tables = Vec::new();
    for table in source.host_tables() {
        let rows = count_rows(&mut source.conn, &table, RowFilter::All).await?;
        host_tables.push(TableCount { table, rows });
    }

    let akh_tables = source.akh_tables();
    let mut attributed: BTreeMap<String, u64> = BTreeMap::new();
    let mut akhs = Vec::new();
    for name in source.akhs.clone() {
        let mut tables = Vec::new();
        for (table, key) in &akh_tables {
            let rows = count_rows(&mut source.conn, table, filter_for(&name, *key)).await?;
            *attributed.entry(table.clone()).or_default() += rows;
            tables.push(TableCount {
                table: table.clone(),
                rows,
            });
        }
        let path = topology
            .akh_path(&name)
            .expect("per-akh topology resolves akh paths");
        topology.akh_options(&name)?;
        akhs.push(AkhPlan {
            name: name.as_str().to_owned(),
            akh_id: name.akh_id(),
            workspace_id: name.workspace_id(),
            path,
            tables,
        });
    }

    let mut orphans = Vec::new();
    for (table, _) in &akh_tables {
        let total = count_rows(&mut source.conn, table, RowFilter::All).await?;
        let copied = attributed.get(table).copied().unwrap_or(0);
        if total > copied {
            orphans.push(TableCount {
                table: table.clone(),
                rows: total - copied,
            });
        }
    }

    Ok(SplitPlan {
        source: source_path,
        host: host_path,
        in_use,
        source_migrations: source.migrations.clone(),
        host_tables,
        akhs,
        orphans,
    })
}

fn sha256_file(path: &Path) -> SplitResult<String> {
    let mut hasher = Sha256::new();
    match File::open(path) {
        Ok(mut file) => {
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let read = file.read(&mut buffer).map_err(io_err(path))?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
        }
        // A missing -wal hashes like an empty one.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(SplitError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    }
    Ok(hex(&hasher.finalize()))
}

/// Files this run created; removed on drop unless [`Created::keep`] ran.
struct Created {
    paths: Vec<PathBuf>,
    keep: bool,
}

impl Created {
    fn track_db(&mut self, db: &Path) {
        self.paths.push(db.to_path_buf());
        for suffix in ["-wal", "-shm", ".migrate.lock"] {
            self.paths.push(PathBuf::from(format!("{}{suffix}", db.display())));
        }
    }
}

impl Drop for Created {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        for path in &self.paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn refuse_existing(path: &Path) -> SplitResult<()> {
    let wal = wal_path(path);
    for candidate in [path.to_path_buf(), wal] {
        if candidate.exists() {
            return Err(SplitError::TargetExists { path: candidate });
        }
    }
    Ok(())
}

/// Copy one target file's tables from the source in one transaction, then
/// verify every table by re-reading it.
async fn copy_target(
    source: &mut SqliteConnection,
    options: &PoolOptions,
    path: &Path,
    tables: &[(String, RowFilter)],
) -> SplitResult<TargetEvidence> {
    crate::run_migrations_locked(options).await?;
    let db = crate::connect(options).await?;
    let pool = db.get_sqlite_connection_pool();
    let mut conn = pool
        .acquire()
        .await
        .map_err(sqlite_err(format!("opening {}", path.display())))?;

    let mut staged = Vec::with_capacity(tables.len());
    let mut transaction = conn
        .begin()
        .await
        .map_err(sqlite_err(format!("starting the copy into {}", path.display())))?;
    for (table, filter) in tables {
        let shape = TableShape::read(source, table).await?;
        let target_shape = TableShape::read(&mut *transaction, table).await?;
        if let Some(missing) = shape
            .columns
            .iter()
            .find(|c| !target_shape.columns.contains(c))
        {
            return Err(SplitError::Verification {
                table: table.clone(),
                path: path.to_path_buf(),
                detail: format!("target has no column {missing}"),
            });
        }
        let (rows, digest) = read_rows(source, &shape, *filter).await?;
        insert_rows(&mut *transaction, &shape, &rows).await?;
        staged.push((shape, rows.len() as u64, digest));
    }
    transaction
        .commit()
        .await
        .map_err(sqlite_err(format!("committing {}", path.display())))?;

    let mut evidence = Vec::with_capacity(staged.len());
    for (shape, source_rows, source_sha256) in staged {
        let (rows, target_sha256) = read_rows(&mut *conn, &shape, RowFilter::All).await?;
        let target_rows = rows.len() as u64;
        if target_rows != source_rows || target_sha256 != source_sha256 {
            return Err(SplitError::Verification {
                table: shape.name.clone(),
                path: path.to_path_buf(),
                detail: format!(
                    "source {source_rows} row(s) sha256 {source_sha256}, \
                     target {target_rows} row(s) sha256 {target_sha256}"
                ),
            });
        }
        evidence.push(TableEvidence {
            table: shape.name,
            source_rows,
            source_sha256,
            target_rows,
            target_sha256,
        });
    }
    // Fold the WAL into the file so each target is self-contained.
    sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(&mut *conn)
        .await
        .map_err(sqlite_err(format!("checkpointing {}", path.display())))?;
    drop(conn);
    let _ = db.close().await;
    Ok(TargetEvidence {
        path: path.to_path_buf(),
        tables: evidence,
    })
}

/// Execute a split and write `evidence` (which must not exist). See the
/// module docs for the guarantees.
pub async fn execute(req: &SplitRequest, evidence_path: &Path) -> SplitResult<SplitEvidence> {
    let started_at = chrono::Utc::now();
    let source_path = absolute(&req.source)?;
    let host_path = absolute(&req.host)?;
    let evidence_path = absolute(evidence_path)?;
    if !source_path.is_file() {
        return Err(SplitError::SourceMissing { path: source_path });
    }
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.parent().map(Path::canonicalize)) {
        (Ok(a), Some(Ok(dir))) => b.file_name().is_some_and(|n| dir.join(n) == a),
        _ => false,
    };
    if same(&source_path, &host_path) {
        return Err(SplitError::HostIsSource { path: host_path });
    }
    let req = SplitRequest {
        source: source_path.clone(),
        host: host_path.clone(),
        ..req.clone()
    };
    let topology = topology_for(&req)?;

    // Lock every lock file beside the source and keep them locked to the end.
    let (_source_locks, busy) = try_lock_all(&lock_files_beside(&source_path)?)?;
    if let Some(lock) = busy.into_iter().next() {
        return Err(SplitError::SourceInUse { lock });
    }

    let sha256_before = sha256_file(&source_path)?;
    let wal_sha256_before = sha256_file(&wal_path(&source_path))?;
    let mut source = Source::open(&source_path, req.busy_timeout).await?;

    let plan = build_plan(
        &mut source,
        source_path.clone(),
        host_path.clone(),
        &topology,
        Vec::new(),
    )
    .await?;
    if !plan.orphans.is_empty() && !req.allow_orphans {
        return Err(SplitError::Orphans {
            rows: plan.orphans.iter().map(|o| o.rows).sum(),
            tables: plan
                .orphans
                .iter()
                .map(|o| format!("{} ({})", o.table, o.rows))
                .collect::<Vec<_>>()
                .join(", "),
        });
    }

    refuse_existing(&host_path)?;
    for akh in &plan.akhs {
        refuse_existing(&akh.path)?;
    }
    if evidence_path.exists() {
        return Err(SplitError::TargetExists {
            path: evidence_path,
        });
    }

    let mut created = Created {
        paths: Vec::new(),
        keep: false,
    };

    // Hold the new host's residency lock files too, so a per-akh daemon
    // started against the new host cannot claim an akh mid-copy.
    let mut target_locks = Vec::new();
    for name in &source.akhs {
        let lock = PathBuf::from(format!("{}.akh-{}.lock", host_path.display(), name.akh_id()));
        let existed = lock.exists();
        let mut open = std::fs::OpenOptions::new();
        open.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open.mode(0o600);
        }
        let file = open.open(&lock).map_err(io_err(&lock))?;
        if !existed {
            created.paths.push(lock.clone());
        }
        match file.try_lock() {
            Ok(()) => target_locks.push(file),
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(SplitError::SourceInUse { lock });
            }
            Err(std::fs::TryLockError::Error(source)) => {
                return Err(SplitError::Io { path: lock, source });
            }
        }
    }

    created.track_db(&host_path);
    let host_tables: Vec<(String, RowFilter)> = source
        .host_tables()
        .into_iter()
        .map(|t| (t, RowFilter::All))
        .collect();
    let host = copy_target(&mut source.conn, topology.host(), &host_path, &host_tables).await?;

    let akh_tables = source.akh_tables();
    let mut akhs = Vec::with_capacity(source.akhs.len());
    for name in source.akhs.clone() {
        let options = topology.akh_options(&name)?;
        let path = topology
            .akh_path(&name)
            .expect("per-akh topology resolves akh paths");
        created.track_db(&path);
        let tables: Vec<(String, RowFilter)> = akh_tables
            .iter()
            .map(|(t, key)| (t.clone(), filter_for(&name, *key)))
            .collect();
        let target = copy_target(&mut source.conn, &options, &path, &tables).await?;
        akhs.push(AkhEvidence {
            name: name.as_str().to_owned(),
            akh_id: name.akh_id(),
            workspace_id: name.workspace_id(),
            target,
        });
    }

    let opened = source.opened;
    let migrations = source.migrations.clone();
    let _ = source.conn.close().await;

    let sha256_after = sha256_file(&source_path)?;
    let wal_sha256_after = sha256_file(&wal_path(&source_path))?;
    if sha256_after != sha256_before || wal_sha256_after != wal_sha256_before {
        return Err(SplitError::SourceChanged { path: source_path });
    }

    for path in std::iter::once(&host_path).chain(akhs.iter().map(|a| &a.target.path)) {
        restrict(path)?;
    }

    let evidence = SplitEvidence {
        schema: EVIDENCE_SCHEMA,
        tool: format!("akhomed store split-per-akh (akhomed-store {})", env!("CARGO_PKG_VERSION")),
        started_at,
        finished_at: chrono::Utc::now(),
        layout: StoreLayout::PerAkh.as_str(),
        source: SourceEvidence {
            path: source_path,
            sha256_before,
            sha256_after,
            wal_sha256_before,
            wal_sha256_after,
            opened,
            migrations,
        },
        host,
        akhs,
        orphans_left_in_source: plan.orphans,
        verified: true,
    };
    write_evidence(&evidence_path, &evidence)?;
    created.keep = true;
    drop(target_locks);
    Ok(evidence)
}

fn restrict(path: &Path) -> SplitResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(io_err(path))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn write_evidence(path: &Path, evidence: &SplitEvidence) -> SplitResult<()> {
    let text = serde_json::to_string_pretty(evidence).map_err(|error| SplitError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(error),
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io_err(parent))?;
    }
    let mut open = std::fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    let mut file = open.open(path).map_err(io_err(path))?;
    use std::io::Write as _;
    file.write_all(text.as_bytes()).map_err(io_err(path))?;
    file.write_all(b"\n").map_err(io_err(path))?;
    file.sync_all().map_err(io_err(path))?;
    Ok(())
}

/// Parse a `sqlite:` URL (as configured in `[database] url`) into its file
/// path, for callers that accept either form on the command line.
pub fn sqlite_url_path(url: &str) -> Option<PathBuf> {
    if !url.starts_with("sqlite:") {
        return None;
    }
    SqliteConnectOptions::from_str(url)
        .ok()
        .map(|o| o.get_filename().to_path_buf())
}
