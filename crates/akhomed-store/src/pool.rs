//! Connection-pool construction for the akhomed Postgres backing store.

use std::time::Duration;

use sea_orm::{ConnectOptions, Database, DatabaseConnection};

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
