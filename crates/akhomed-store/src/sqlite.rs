//! Local SeaORM backing store. Residency is held by an OS file lock, paired
//! with the same durable epochs used by PostgreSQL. Lock files are never unlinked.
use crate::{
    MutationRow, PoolOptions, StoreError, StoreResult,
    entity::{akh, mutation_log},
};
use sea_orm::sqlx::ConnectOptions as _;
use sea_orm::sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous,
};
use sea_orm::{
    ActiveModelTrait,
    ActiveValue::{NotSet, Set},
    DatabaseConnection, EntityTrait, TransactionTrait,
};
use std::{
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};
use uuid::Uuid;

fn failure(error: impl std::fmt::Display) -> StoreError {
    StoreError::Conflict {
        message: format!("SQLite backing store: {error}"),
    }
}

pub(crate) async fn migration_guard(opts: &PoolOptions) -> StoreResult<File> {
    let options = SqliteConnectOptions::from_str(&opts.url).map_err(failure)?;
    let filename = options.get_filename();
    if !filename.is_absolute() {
        return Err(failure("requires an absolute on-disk database path"));
    }
    if let Some(parent) = filename.parent() {
        std::fs::create_dir_all(parent).map_err(failure)?;
    }
    let mut create = std::fs::OpenOptions::new();
    create.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        create.mode(0o600);
    }
    create.open(filename).map_err(failure)?;
    let path = filename.canonicalize().map_err(failure)?;
    let mut open = std::fs::OpenOptions::new();
    open.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    let file = open
        .open(format!("{}.migrate.lock", path.display()))
        .map_err(failure)?;
    let deadline = tokio::time::Instant::now() + opts.connect_timeout;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Err(error) => return Err(failure(error)),
        }
    }
}

pub(crate) async fn connect(opts: &PoolOptions) -> StoreResult<DatabaseConnection> {
    let options = SqliteConnectOptions::from_str(&opts.url).map_err(failure)?;
    if !options.get_filename().is_absolute() {
        return Err(failure("requires an absolute on-disk database path"));
    }
    let path = options.get_filename();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(failure)?;
    }
    let mut create = std::fs::OpenOptions::new();
    create.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        create.mode(0o600);
    }
    create.open(path).map_err(failure)?;
    let options = options
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Full)
        .busy_timeout(opts.connect_timeout)
        .disable_statement_logging();
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(opts.connect_timeout)
        .connect_with(options)
        .await
        .map_err(failure)?;
    Ok(sea_orm::SqlxSqliteConnector::from_sqlx_sqlite_pool(pool))
}

/// The residency lock file for akh `id` anchored at database `anchor`:
/// `<anchor>.akh-<id>.lock`. Lock files are never unlinked.
pub(crate) fn residency_lock_path(anchor: &Path, id: Uuid) -> PathBuf {
    PathBuf::from(format!("{}.akh-{id}.lock", anchor.display()))
}

pub(crate) struct LocalSession {
    db: DatabaseConnection,
    path: PathBuf,
    held: Option<(Uuid, File)>,
}

impl LocalSession {
    pub(crate) async fn open(opts: &PoolOptions) -> StoreResult<Self> {
        let db = connect(opts).await?;
        let options = SqliteConnectOptions::from_str(&opts.url).map_err(failure)?;
        let path = options.get_filename().canonicalize().map_err(failure)?;
        Ok(Self {
            db,
            path,
            held: None,
        })
    }

    /// A session whose data (epoch, WAL rows) goes through `db` — the akh's
    /// own store connection under the per-akh layout (ADR 120), so the WAL
    /// and the snapshot share one connection and queue on its pool instead of
    /// racing for the file's write lock — while its residency lock file is
    /// anchored at `anchor` (the host database): `<host>.akh-<uuid>.lock`.
    pub(crate) fn with_connection(db: DatabaseConnection, anchor: &Path) -> StoreResult<Self> {
        let parent = anchor
            .parent()
            .ok_or_else(|| failure("residency lock anchor has no parent directory"))?;
        let file_name = anchor
            .file_name()
            .ok_or_else(|| failure("residency lock anchor has no file name"))?;
        let path = parent.canonicalize().map_err(failure)?.join(file_name);
        Ok(Self {
            db,
            path,
            held: None,
        })
    }

    pub(crate) fn acquire(&mut self, id: Uuid) -> StoreResult<bool> {
        if let Some((owner, _)) = self.held {
            return Ok(owner == id);
        }
        let lock = residency_lock_path(&self.path, id);
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(lock).map_err(failure)?;
        match file.try_lock() {
            Ok(()) => {
                self.held = Some((id, file));
                Ok(true)
            }
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(error) => Err(failure(error)),
        }
    }
    fn check(&self, id: Uuid) -> StoreResult<()> {
        if self.held.as_ref().is_some_and(|(owner, _)| *owner == id) {
            Ok(())
        } else {
            Err(failure("session does not own the akh residency lock"))
        }
    }
    pub(crate) fn release(&mut self, id: Uuid) -> StoreResult<bool> {
        if self.check(id).is_err() {
            return Ok(false);
        }
        self.held.take();
        Ok(true)
    }
    pub(crate) async fn claim(&mut self, id: Uuid) -> StoreResult<i64> {
        self.check(id)?;
        let txn = self.db.begin().await?;
        let owner = akh::Entity::find_by_id(id)
            .one(&txn)
            .await?
            .ok_or_else(|| failure("unknown akh"))?;
        let next = owner
            .residency_epoch
            .checked_add(1)
            .ok_or_else(|| failure("epoch exhausted"))?;
        let mut owner: akh::ActiveModel = owner.into();
        owner.residency_epoch = Set(next);
        owner.update(&txn).await?;
        txn.commit().await?;
        Ok(next)
    }
    pub(crate) async fn append(
        &mut self,
        id: Uuid,
        workspace: Uuid,
        epoch: i64,
        rows: &[MutationRow],
    ) -> StoreResult<u64> {
        self.check(id)?;
        let txn = self.db.begin().await?;
        let owner = akh::Entity::find_by_id(id)
            .one(&txn)
            .await?
            .ok_or_else(|| failure("unknown akh"))?;
        if owner.residency_epoch != epoch {
            return Err(failure("stale WAL residency epoch"));
        }
        for row in rows {
            mutation_log::ActiveModel {
                seq: NotSet,
                akh_id: Set(id),
                workspace_id: Set(workspace),
                kind: Set(row.kind.clone()),
                payload: Set(row.payload.clone()),
                applied_at: Set(chrono::Utc::now()),
            }
            .insert(&txn)
            .await?;
        }
        txn.commit().await?;
        Ok(rows.len() as u64)
    }
}
