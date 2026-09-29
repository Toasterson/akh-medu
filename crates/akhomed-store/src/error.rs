//! Error type for `akhomed-store`, following the project's miette + thiserror
//! convention: every variant carries a diagnostic `code` and actionable `help`.

use miette::Diagnostic;
use thiserror::Error;

/// Errors raised while connecting to, migrating, or querying the akhomed
/// Postgres backing store.
#[derive(Debug, Error, Diagnostic)]
pub enum StoreError {
    /// The connection pool could not be opened.
    #[error("failed to connect to PostgreSQL at {url}: {source}")]
    #[diagnostic(
        code(akhomed_store::connect),
        help(
            "Check the [postgres] url in akhomed's config.toml (or DATABASE_URL) \
             and confirm the server is reachable and accepting connections."
        )
    )]
    Connect {
        /// Credential-redacted connection URL.
        url: String,
        #[source]
        source: sea_orm::DbErr,
    },

    /// A schema migration failed.
    #[error("akhomed-store migration failed: {source}")]
    #[diagnostic(
        code(akhomed_store::migrate),
        help(
            "Migrations run transactionally, so the schema is unchanged. Inspect \
             the underlying error, then re-run `akhomed-migrate` once resolved."
        )
    )]
    Migrate {
        #[source]
        source: sea_orm::DbErr,
    },

    /// A query or mutation against the store failed.
    #[error("akhomed-store database operation failed: {source}")]
    #[diagnostic(
        code(akhomed_store::db),
        help("This is an unexpected database error; see the source for details.")
    )]
    Db {
        #[source]
        source: sea_orm::DbErr,
    },

    /// SQLite refused a statement because another connection held the file's
    /// lock past `busy_timeout` (`SQLITE_BUSY` / `SQLITE_LOCKED`). This is
    /// write-lock contention between connections to one file, not a
    /// residency conflict: SQLite residency is a host-local file lock.
    #[error("SQLite lock contention (another connection held the database lock): {source}")]
    #[diagnostic(
        code(akhomed_store::sqlite_busy),
        help(
            "Several connections write one SQLite file. busy_timeout only helps a writer \
             that has not read yet: a transaction that already read and then writes fails \
             at once while another connection holds the write lock. This is contention, \
             not a split-brain (SQLite residency is a host-local lock file). Options: \
             (1) give each akh its own file and one connection with \
             `[database] layout = \"per_akh\"` (split the existing file first with \
             `akhomed store split-per-akh`; see deploy/roman-mini/CUTOVER.md \
             \"Per-akh databases\"); (2) raise `[database] connect_timeout_secs` (the \
             SQLite busy_timeout and pool acquire timeout) for writers that wait at BEGIN; \
             (3) lengthen `snapshot_interval_secs` so fewer snapshots overlap. The failed \
             write is retried on the next tick; nothing was committed."
        )
    )]
    LockContention {
        #[source]
        source: sea_orm::DbErr,
    },

    /// The configured store layout cannot be used as asked.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Layout(Box<crate::layout::LayoutError>),

    /// A write was refused because it would silently clobber existing state
    /// (e.g. re-admitting an already-admitted schema-pack `(name, version)`).
    #[error("{message}")]
    #[diagnostic(
        code(akhomed_store::conflict),
        help(
            "This is a duplicate-write guard, not a transient failure: check the existing \
             row before retrying under a new identity (e.g. a bumped pack version)."
        )
    )]
    Conflict { message: String },
}

/// Convenience result alias for store operations.
pub type StoreResult<T> = std::result::Result<T, StoreError>;

impl From<sea_orm::DbErr> for StoreError {
    /// SQLite busy/locked errors become [`StoreError::LockContention`]; every
    /// other error (and every PostgreSQL error) stays [`StoreError::Db`].
    fn from(source: sea_orm::DbErr) -> Self {
        if is_sqlite_lock_contention(&source) {
            StoreError::LockContention { source }
        } else {
            StoreError::Db { source }
        }
    }
}

impl StoreError {
    /// True for SQLite write-lock contention ([`StoreError::LockContention`]).
    pub fn is_lock_contention(&self) -> bool {
        matches!(self, StoreError::LockContention { .. })
    }

    /// True when the caller waited past the pool's acquire timeout for a
    /// connection (in-process contention for a small pool). Kept as
    /// [`StoreError::Db`] so PostgreSQL errors read exactly as before; callers
    /// that know they run on SQLite can report it as contention.
    pub fn is_pool_timeout(&self) -> bool {
        matches!(
            self,
            StoreError::Db {
                source: sea_orm::DbErr::ConnectionAcquire(sea_orm::ConnAcquireErr::Timeout)
            }
        )
    }
}

/// SQLite primary result codes for "another connection holds the lock".
const SQLITE_BUSY: i32 = 5;
const SQLITE_LOCKED: i32 = 6;

/// Whether `error` is a SQLite `SQLITE_BUSY` / `SQLITE_LOCKED` failure
/// (including extended codes such as `SQLITE_BUSY_SNAPSHOT`). Classified by
/// the driver's typed error and numeric code, never by message text.
pub fn is_sqlite_lock_contention(error: &sea_orm::DbErr) -> bool {
    use sea_orm::{DbErr, RuntimeErr};
    let sqlx_error = match error {
        DbErr::Conn(RuntimeErr::SqlxError(e))
        | DbErr::Exec(RuntimeErr::SqlxError(e))
        | DbErr::Query(RuntimeErr::SqlxError(e)) => e,
        _ => return false,
    };
    is_sqlx_lock_contention(sqlx_error)
}

/// [`is_sqlite_lock_contention`] for a bare sqlx error.
pub fn is_sqlx_lock_contention(error: &sea_orm::sqlx::Error) -> bool {
    let sea_orm::sqlx::Error::Database(database) = error else {
        return false;
    };
    let Some(sqlite) = database.try_downcast_ref::<sea_orm::sqlx::sqlite::SqliteError>() else {
        return false;
    };
    use sea_orm::sqlx::error::DatabaseError as _;
    sqlite
        .code()
        .and_then(|code| code.parse::<i32>().ok())
        .is_some_and(|code| matches!(code & 0xff, SQLITE_BUSY | SQLITE_LOCKED))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_error_display_includes_source() {
        let inner = sea_orm::DbErr::Custom("unique constraint \"akhomed_snapshot_pkey\"".into());
        let err = StoreError::from(inner);
        let text = err.to_string();
        assert!(
            text.contains("unique constraint"),
            "display should surface the source error, got: {text}"
        );
    }

    #[test]
    fn db_error_debug_keeps_source_chain() {
        let inner = sea_orm::DbErr::Custom("sqlstate 23505".into());
        let err = StoreError::from(inner);
        let debug = format!("{err:?}");
        assert!(
            debug.contains("sqlstate 23505"),
            "debug should retain source chain, got: {debug}"
        );
    }
}
