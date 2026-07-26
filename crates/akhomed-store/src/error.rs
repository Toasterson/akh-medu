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
    #[error("akhomed-store database operation failed")]
    #[diagnostic(
        code(akhomed_store::db),
        help("This is an unexpected database error; see the source for details.")
    )]
    Db {
        #[from]
        source: sea_orm::DbErr,
    },

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
