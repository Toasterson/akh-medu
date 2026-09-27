//! Connection-pool construction and migration runner for the `harvest-pg`
//! store (ADR 119).
//!
//! Kept deliberately small: `connect` takes a bare URL rather than a config
//! struct, because config precedence (env var vs. flag vs. file, and how each
//! of `akh-harvest` / `akh-derive` / `akh-transcripts-mcp` resolves its own
//! credential) is a decision for the binaries built in later increments, not
//! for this leaf crate.

use std::time::Duration;

use sea_orm::{
    ConnectOptions, ConnectionTrait, Database, DatabaseBackend, DatabaseConnection, Statement,
};
use sea_orm_migration::MigratorTrait;

use crate::error::{TranscriptError, TranscriptResult};
use crate::migration::Migrator;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Open a pooled connection to the `harvest-pg` Postgres store.
///
/// On failure returns [`TranscriptError::Connect`] carrying a
/// **credential-redacted** URL, safe to log.
pub async fn connect(url: &str) -> TranscriptResult<DatabaseConnection> {
    let mut opts = ConnectOptions::new(url.to_owned());
    opts.connect_timeout(CONNECT_TIMEOUT)
        // Statement logging would leak query text (including harvested
        // transcript content passed as bind parameters) into operator logs.
        .sqlx_logging(false);

    Database::connect(opts)
        .await
        .map_err(|source| TranscriptError::Connect {
            url: redact_url(url),
            source,
        })
}

/// Advisory-lock key serializing concurrent migrators. More than one
/// `akh-harvest` instance (one per machine) or the singleton `akh-derive`
/// could plausibly race a fresh `harvest-pg` database. Distinct from
/// `akhomed_store::MIGRATION_LOCK_KEY` so the two schemas never contend for
/// the same lock if a deployment ever shares a cluster between them.
/// Mnemonic ASCII "AKTRNSCR" (Akh Transcripts).
const MIGRATION_LOCK_KEY: i64 = 0x414B_5452_4E53_4352_u64 as i64;

/// Run all pending migrations against an already-open connection, with no
/// locking.
///
/// **Concurrency caveat**, same as `akhomed_store::run_migrations`: a shared
/// pool's statements may each land on a different pooled connection, so this
/// is not safe to run from several processes against the same fresh database
/// simultaneously (they race on creating `seaql_migrations`). Use
/// [`run_migrations_locked`] for that case. This variant is for callers that
/// already hold a connection and guarantee a single migrator, e.g. serial
/// tests.
pub async fn run_migrations(db: &DatabaseConnection) -> TranscriptResult<()> {
    if db.get_database_backend() == DatabaseBackend::Postgres && !pgvector_available(db).await? {
        return Err(TranscriptError::PgvectorMissing);
    }
    Migrator::up(db, None)
        .await
        .map_err(|source| TranscriptError::Migrate { source })
}

/// Run all pending migrations under a Postgres session **advisory lock**,
/// serializing concurrent migrators across processes and hosts.
///
/// Mirrors `akhomed_store::run_migrations_locked`: this opens a dedicated
/// **single-connection** pool so every statement `Migrator::up` issues shares
/// the one connection that holds the lock, and the lock auto-releases when
/// that connection closes (success, error, or crash). Other migrators block
/// on the lock until the first finishes, then observe the migrations already
/// applied and no-op. On non-Postgres backends there is nothing to serialize
/// and this behaves like [`run_migrations`].
///
/// **Pre-flight pgvector check, before the lock and before the migrator.**
/// This queries `pg_available_extensions` directly and returns
/// [`TranscriptError::PgvectorMissing`] immediately if `vector` is absent —
/// the loud, specific diagnostic the store-schema ticket requires (pgvector
/// absence must fail loudly, unlike TimescaleDB's warn-and-continue), rather
/// than surfacing it indirectly through a wrapped migration error.
/// `m20260919_000001_create_raw_records` repeats a version of this check as a
/// safety net for callers that invoke [`MigratorTrait::up`] directly without
/// going through this function.
pub async fn run_migrations_locked(url: &str) -> TranscriptResult<()> {
    // Dedicated single connection so the advisory lock and every migration
    // statement share one backend session.
    let mut opts = ConnectOptions::new(url.to_owned());
    opts.max_connections(1)
        .connect_timeout(CONNECT_TIMEOUT)
        .sqlx_logging(false);
    let db = Database::connect(opts)
        .await
        .map_err(|source| TranscriptError::Connect {
            url: redact_url(url),
            source,
        })?;

    let backend = db.get_database_backend();
    if backend == DatabaseBackend::Postgres {
        if !pgvector_available(&db).await? {
            return Err(TranscriptError::PgvectorMissing);
        }
        db.execute(Statement::from_string(
            backend,
            format!("SELECT pg_advisory_lock({MIGRATION_LOCK_KEY})"),
        ))
        .await
        .map_err(|source| TranscriptError::AdvisoryLock {
            action: "acquire",
            source,
        })?;
    }

    let result = Migrator::up(&db, None)
        .await
        .map_err(|source| TranscriptError::Migrate { source });

    // Explicit unlock so the lock frees promptly; dropping `db` also releases
    // it when the connection closes. Best-effort — the migration result is
    // what gets reported to the caller.
    if backend == DatabaseBackend::Postgres {
        let _ = db
            .execute(Statement::from_string(
                backend,
                format!("SELECT pg_advisory_unlock({MIGRATION_LOCK_KEY})"),
            ))
            .await;
    }
    result
}

/// `SELECT 1 FROM pg_available_extensions WHERE name = 'vector'`.
async fn pgvector_available(db: &DatabaseConnection) -> TranscriptResult<bool> {
    let row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT 1 FROM pg_available_extensions WHERE name = $1",
            ["vector".into()],
        ))
        .await
        .map_err(TranscriptError::from)?;
    Ok(row.is_some())
}

/// Strip the `user:password@` userinfo component from a connection URL so it
/// is safe to surface in error messages and logs.
fn redact_url(url: &str) -> String {
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = &url[scheme_end + 3..];
        if let Some(at) = after_scheme.find('@') {
            return format!("{}://***@{}", &url[..scheme_end], &after_scheme[at + 1..]);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_credentials() {
        assert_eq!(
            redact_url("postgres://alice:s3cret@db.local:5432/transcripts"),
            "postgres://***@db.local:5432/transcripts"
        );
    }

    #[test]
    fn passes_through_credential_free_url() {
        assert_eq!(
            redact_url("postgres://db.local:5432/transcripts"),
            "postgres://db.local:5432/transcripts"
        );
    }

    #[test]
    fn migration_lock_key_is_distinct_from_akhomed_store() {
        // Mnemonic ASCII "AKMDMIGR" from akhomed-store's lib.rs, kept as a
        // literal here (this crate does not depend on akhomed-store) so a
        // future accidental collision is caught by inspection.
        const AKHOMED_STORE_KEY: i64 = 0x414B_4D44_4D49_4752_u64 as i64;
        assert_ne!(MIGRATION_LOCK_KEY, AKHOMED_STORE_KEY);
    }
}
