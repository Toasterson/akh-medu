//! Error type for `akh-transcripts`, following the project's miette + thiserror
//! convention: every variant carries a diagnostic `code` and actionable `help`.

use miette::Diagnostic;
use thiserror::Error;

pub type TranscriptResult<T> = Result<T, TranscriptError>;

/// Errors raised while connecting to, migrating, or querying the `harvest-pg`
/// store, or while reading a source transcript on disk.
#[derive(Debug, Error, Diagnostic)]
pub enum TranscriptError {
    /// The connection pool could not be opened.
    #[error("failed to connect to PostgreSQL at {url}: {source}")]
    #[diagnostic(
        code(akh_transcripts::connect),
        help(
            "Check the connection URL passed to `akh_transcripts::pool::connect` (or your \
             binary's config/env override) and confirm the `harvest-pg` cluster is reachable \
             and accepting connections."
        )
    )]
    Connect {
        /// Credential-redacted connection URL.
        url: String,
        #[source]
        source: sea_orm::DbErr,
    },

    /// A schema migration failed.
    #[error("akh-transcripts migration failed: {source}")]
    #[diagnostic(
        code(akh_transcripts::migrate),
        help(
            "Migrations run transactionally per step, so the schema is unchanged on failure. \
             Inspect the underlying error and re-run once resolved. If the failure mentions a \
             missing extension, see the `pgvector_missing` / `timescaledb_unavailable` \
             diagnostics for the specific remedy."
        )
    )]
    Migrate {
        #[source]
        source: sea_orm::DbErr,
    },

    /// `pgvector` is not installed in the target Postgres. Unlike TimescaleDB
    /// (warn-and-continue, per the store-schema decision), this is a hard,
    /// loud failure: the MCP read surface advertises semantic search, and a
    /// silent degrade there would be worse than refusing to start.
    #[error("the `vector` (pgvector) extension is not available on this Postgres server")]
    #[diagnostic(
        code(akh_transcripts::pgvector_missing),
        help(
            "The transcript store requires pgvector for semantic search over harvested prose. \
             Point at a Postgres image that bundles it — the project's wrapper image is \
             `code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1` (CNPG 18 base + \
             TimescaleDB 2.25.1 + pgvector). If you only need exact-addressing reads, that is \
             not yet a supported degrade for this store; open an issue against \
             docs/wayfinder/transcript-harvest/tickets/store-schema.md instead of patching \
             around the check."
        )
    )]
    PgvectorMissing,

    /// A query or mutation against the store failed.
    #[error("akh-transcripts database operation failed: {source}")]
    #[diagnostic(
        code(akh_transcripts::db),
        help("This is an unexpected database error; see the source for details.")
    )]
    Db {
        #[from]
        #[source]
        source: sea_orm::DbErr,
    },

    /// Failed to acquire (or release) the Postgres advisory lock guarding
    /// concurrent migrators — see `run_migrations_locked`.
    #[error("failed to {action} the akh-transcripts migration advisory lock: {source}")]
    #[diagnostic(
        code(akh_transcripts::advisory_lock),
        help(
            "Another process may be holding the lock, or the connection dropped mid-migration. \
             Retry once any concurrent migrator finishes; if the lock appears stuck, \
             `SELECT pg_advisory_unlock_all()` on a fresh session against the same database."
        )
    )]
    AdvisoryLock {
        action: &'static str,
        #[source]
        source: sea_orm::DbErr,
    },

    /// A source file or row could not be read or parsed by an adapter.
    #[error("failed to read transcript source at {path}: {reason}")]
    #[diagnostic(
        code(akh_transcripts::source_read),
        help(
            "Confirm the path exists and is readable, and that its format has not drifted from \
             what the adapter expects — see docs/wayfinder/transcript-harvest/tickets/\
             source-file-lifecycle.md for known drift traps."
        )
    )]
    SourceRead { path: String, reason: String },
}
