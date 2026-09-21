//! SeaORM PostgreSQL/SQLite pools and backend-specific residency ownership.

use std::time::Duration;

use sea_orm::sqlx::postgres::PgConnection;
use sea_orm::sqlx::{Connection, Row};
use sea_orm::{ConnectOptions, Database, DatabaseConnection};
use tokio::sync::{Mutex, MutexGuard};
use uuid::Uuid;

use crate::error::{StoreError, StoreResult};

/// Options for opening the akhomed Postgres pool.
///
/// Kept as a plain struct (rather than reaching into the daemon's config types)
/// so this crate stays a leaf dependency. The daemon maps its `[postgres]`
/// config section onto these fields.
#[derive(Debug, Clone)]
pub struct PoolOptions {
    /// PostgreSQL connection URL, e.g. `postgres://user:pass@host:5432/akhmedu`.
    pub url: String,
    /// Maximum number of pooled connections.
    pub max_connections: u32,
    /// Timeout for establishing a new connection.
    pub connect_timeout: Duration,
}

/// A pinned PostgreSQL session or an OS-locked local SQLite residency.
///
/// The historical type name is retained for API compatibility. SQLite ownership
/// and fenced writes are implemented in `sqlite`, not PostgreSQL emulation.
///
/// A one-connection *pool* is not sufficient for a session advisory lock: the
/// pool may recycle or reconnect its sole socket while the Rust pool handle
/// remains alive.  This type owns the [`PgConnection`] itself.  A disconnect
/// therefore makes subsequent fenced writes fail instead of silently running
/// on a replacement session that never acquired the lock.
pub struct StablePgSession {
    pub(crate) sqlite: Option<Mutex<crate::sqlite::LocalSession>>,
    pub(crate) state: Mutex<StablePgSessionState>,
}

pub(crate) struct StablePgSessionState {
    pub(crate) connection: Option<PgConnection>,
    pub(crate) locked_akh: Option<Uuid>,
}

impl StablePgSession {
    /// Open a physically pinned PostgreSQL connection.
    pub async fn connect(opts: &PoolOptions) -> StoreResult<Self> {
        if opts.url.starts_with("sqlite:") {
            return Ok(Self {
                sqlite: Some(Mutex::new(crate::sqlite::LocalSession::open(opts).await?)),
                state: Mutex::new(StablePgSessionState {
                    connection: None,
                    locked_akh: None,
                }),
            });
        }
        let connect = PgConnection::connect(&opts.url);
        let connection = tokio::time::timeout(opts.connect_timeout, connect)
            .await
            .map_err(|_| StoreError::Connect {
                url: redact_url(&opts.url),
                source: sea_orm::DbErr::Conn(sea_orm::RuntimeErr::Internal(format!(
                    "stable PostgreSQL session timed out after {:?}",
                    opts.connect_timeout
                ))),
            })?
            .map_err(|source| StoreError::Connect {
                url: redact_url(&opts.url),
                source: sea_orm::DbErr::Conn(sea_orm::RuntimeErr::SqlxError(source)),
            })?;
        Ok(Self {
            sqlite: None,
            state: Mutex::new(StablePgSessionState {
                connection: Some(connection),
                locked_akh: None,
            }),
        })
    }

    /// Acquire the session-scoped residency lock on this exact connection.
    /// Re-checks for the same Akh are local no-ops so PostgreSQL's re-entrant
    /// advisory-lock counter is never incremented twice.
    pub async fn try_acquire_akh_lock(&self, akh_id: Uuid) -> StoreResult<bool> {
        if let Some(local) = &self.sqlite {
            return local.lock().await.acquire(akh_id);
        }
        let mut state = self.state.lock().await;
        if state.locked_akh == Some(akh_id) {
            return Ok(true);
        }
        if state.locked_akh.is_some() {
            return Ok(false);
        }
        let connection = state
            .connection
            .as_mut()
            .ok_or_else(|| StoreError::Conflict {
                message: "stable PostgreSQL residency session is closed".into(),
            })?;
        let locked: bool =
            sea_orm::sqlx::query("SELECT pg_try_advisory_lock(hashtext($1)) AS locked")
                .bind(format!("akh:{akh_id}"))
                .fetch_one(connection)
                .await
                .map_err(query_error)?
                .try_get("locked")
                .map_err(query_error)?;
        if locked {
            state.locked_akh = Some(akh_id);
        }
        Ok(locked)
    }

    /// Advance and return the durable residency fencing epoch while holding
    /// the advisory lock on this exact physical PostgreSQL session.
    ///
    /// Keeping the ownership check, snapshot-publication fence, and epoch
    /// update under `state` means a dropped/reconnected socket cannot advance
    /// an epoch after the session advisory lock has disappeared. In
    /// particular, this must never be implemented through a connection pool:
    /// the pooled transaction would not be protected by this session's lock.
    pub async fn claim_akh_residency(&self, akh_id: Uuid, workspace_id: Uuid) -> StoreResult<i64> {
        if let Some(local) = &self.sqlite {
            return local.lock().await.claim(akh_id).await;
        }
        let mut state = self.state.lock().await;
        if state.locked_akh != Some(akh_id) {
            return Err(StoreError::Conflict {
                message: format!(
                    "cannot claim residency for akh {akh_id}: the pinned session does not own its advisory lock"
                ),
            });
        }
        let connection = state
            .connection
            .as_mut()
            .ok_or_else(|| StoreError::Conflict {
                message: format!(
                    "cannot claim residency for akh {akh_id}: pinned PostgreSQL session is closed"
                ),
            })?;
        let mut transaction = connection.begin().await.map_err(query_error)?;
        sea_orm::sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(format!("snapshot:{workspace_id}"))
            .execute(&mut *transaction)
            .await
            .map_err(query_error)?;

        let current: Option<i64> = sea_orm::sqlx::query_scalar(
            "SELECT residency_epoch FROM akhomed_akh WHERE id = $1 FOR UPDATE",
        )
        .bind(akh_id)
        .fetch_optional(&mut *transaction)
        .await
        .map_err(query_error)?;
        let Some(current) = current else {
            transaction.rollback().await.map_err(query_error)?;
            return Err(StoreError::Conflict {
                message: format!("cannot claim residency for unknown akh {akh_id}"),
            });
        };
        let next = current.checked_add(1).ok_or_else(|| StoreError::Conflict {
            message: format!("residency epoch exhausted for akh {akh_id}"),
        })?;
        sea_orm::sqlx::query("UPDATE akhomed_akh SET residency_epoch = $2 WHERE id = $1")
            .bind(akh_id)
            .bind(next)
            .execute(&mut *transaction)
            .await
            .map_err(query_error)?;
        transaction.commit().await.map_err(query_error)?;
        Ok(next)
    }

    /// PostgreSQL backend PID for live ownership/failure tests and diagnostics.
    /// The query runs on the same pinned connection that owns the lock.
    pub async fn backend_process_id(&self) -> StoreResult<i32> {
        let mut state = self.state.lock().await;
        let connection = state
            .connection
            .as_mut()
            .ok_or_else(|| StoreError::Conflict {
                message: "stable PostgreSQL residency session is closed".into(),
            })?;
        sea_orm::sqlx::query_scalar("SELECT pg_backend_pid()")
            .fetch_one(connection)
            .await
            .map_err(query_error)
    }

    /// Release this session's residency lock.  The local ownership marker is
    /// cleared before the query so a failed/unreachable unlock still fences
    /// every retained WAL handle immediately.  Dropping the connection is the
    /// final PostgreSQL-side release guarantee.
    pub async fn release_akh_lock(&self, akh_id: Uuid) -> StoreResult<bool> {
        if let Some(local) = &self.sqlite {
            return local.lock().await.release(akh_id);
        }
        let mut state = self.state.lock().await;
        if state.locked_akh != Some(akh_id) {
            return Ok(false);
        }
        state.locked_akh = None;
        let Some(connection) = state.connection.as_mut() else {
            return Ok(false);
        };
        let result = sea_orm::sqlx::query("SELECT pg_advisory_unlock(hashtext($1)) AS unlocked")
            .bind(format!("akh:{akh_id}"))
            .fetch_one(connection)
            .await
            .and_then(|row| row.try_get("unlocked"));
        match result {
            Ok(unlocked) => Ok(unlocked),
            Err(error) => {
                // Do not leave a possibly-live server-side lock attached to a
                // session retained by an old Engine Arc. Taking and closing
                // the physical connection makes release deterministic.
                if let Some(connection) = state.connection.take() {
                    let _ = connection.close().await;
                }
                Err(query_error(error))
            }
        }
    }

    pub(crate) async fn lock_state(&self) -> MutexGuard<'_, StablePgSessionState> {
        self.state.lock().await
    }
}

fn query_error(source: sea_orm::sqlx::Error) -> StoreError {
    StoreError::Db {
        source: sea_orm::DbErr::Query(sea_orm::RuntimeErr::SqlxError(source)),
    }
}

impl PoolOptions {
    /// Build options for `url` with conservative defaults (10 connections,
    /// 10 s connect timeout).
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_connections: 10,
            connect_timeout: Duration::from_secs(10),
        }
    }
}

/// Open a pooled connection to PostgreSQL.
///
/// On failure returns [`StoreError::Connect`] carrying a **credential-redacted**
/// URL so the diagnostic can be logged safely.
pub async fn connect(opts: &PoolOptions) -> StoreResult<DatabaseConnection> {
    if opts.url.starts_with("sqlite:") {
        return crate::sqlite::connect(opts).await;
    }
    let mut connect_opts = ConnectOptions::new(opts.url.clone());
    connect_opts
        .max_connections(opts.max_connections)
        .connect_timeout(opts.connect_timeout)
        // The daemon does its own structured logging; sqlx statement logging is
        // noisy and would leak query text into operator logs.
        .sqlx_logging(false);

    Database::connect(connect_opts)
        .await
        .map_err(|source| StoreError::Connect {
            url: redact_url(&opts.url),
            source,
        })
}

/// Strip the `user:password@` userinfo component from a connection URL so it is
/// safe to surface in error messages and logs.
fn redact_url(url: &str) -> String {
    // Find the scheme separator and the host separator; replace anything in
    // between (the userinfo) with a placeholder. Falls back to the raw scheme
    // when the URL doesn't parse like `scheme://userinfo@host/...`.
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
            redact_url("postgres://alice:s3cret@db.local:5432/akh"),
            "postgres://***@db.local:5432/akh"
        );
    }

    #[test]
    fn passes_through_credential_free_url() {
        assert_eq!(
            redact_url("postgres://db.local:5432/akh"),
            "postgres://db.local:5432/akh"
        );
    }

    #[test]
    fn default_options() {
        let opts = PoolOptions::new("postgres://localhost/akh");
        assert_eq!(opts.max_connections, 10);
        assert_eq!(opts.connect_timeout, Duration::from_secs(10));
    }
}
