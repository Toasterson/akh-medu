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

    // adapter (increment 2 — source adapters; owned by
    // `crates/akh-transcripts/src/adapter/`). Append new variants here only;
    // this section is edited concurrently with the top of the file by
    // another increment.
    // -------------------------------------------------------------------
    #[error("failed to read transcript stream at {path}")]
    #[diagnostic(
        code(akh_transcripts::adapter::io),
        help(
            "Check that the file exists and is readable. If it was rotated to a \
             `.jsonl.zst` sibling this is a genuine I/O failure on both paths, not \
             the expected rotation fallback (source-file-lifecycle.md)."
        )
    )]
    AdapterIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to decompress rotated stream at {path}")]
    #[diagnostic(
        code(akh_transcripts::adapter::decompress),
        help(
            "The `.jsonl.zst` sibling exists but did not decode as zstd. Verify it \
             with `zstd -t {path}`; treat a truncated rotation as data loss, not \
             something to retry blindly."
        )
    )]
    AdapterDecompress {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("could not open OpenCode database at {path}")]
    #[diagnostic(
        code(akh_transcripts::adapter::opencode_open),
        help(
            "Confirm `opencode.db` exists at this path and is not held under an \
             exclusive lock by another writer. This adapter opens it read-only and \
             never writes to it."
        )
    )]
    AdapterOpencodeOpen {
        path: String,
        #[source]
        source: rusqlite::Error,
    },

    #[error("OpenCode query failed: {detail}")]
    #[diagnostic(
        code(akh_transcripts::adapter::opencode_query),
        help(
            "This adapter reads only the `event` table (`aggregate_id, seq, type, \
             data`). `message`/`part` are stale per source-format-stability.md, and \
             `account`/`credential` must never be named in a query — they hold live \
             OAuth tokens."
        )
    )]
    AdapterOpencodeQuery {
        detail: String,
        #[source]
        source: rusqlite::Error,
    },

    #[error("transcript root {path} does not exist or is not readable")]
    #[diagnostic(
        code(akh_transcripts::adapter::root_missing),
        help(
            "Pass the real session-store root: `~/.claude/projects` for Claude \
             Code, `~/.codex/sessions` for Codex, or the `opencode.db` file path \
             for OpenCode."
        )
    )]
    AdapterRootMissing { path: String },
}

// adapter — convenience constructors for `crates/akh-transcripts/src/adapter/*`.
impl TranscriptError {
    pub(crate) fn adapter_io(path: &std::path::Path, source: std::io::Error) -> Self {
        TranscriptError::AdapterIo {
            path: path.display().to_string(),
            source,
        }
    }

    pub(crate) fn adapter_decompress(path: &std::path::Path, source: std::io::Error) -> Self {
        TranscriptError::AdapterDecompress {
            path: path.display().to_string(),
            source,
        }
    }

    pub(crate) fn adapter_opencode_open(path: &std::path::Path, source: rusqlite::Error) -> Self {
        TranscriptError::AdapterOpencodeOpen {
            path: path.display().to_string(),
            source,
        }
    }

    pub(crate) fn adapter_opencode_query(
        detail: impl Into<String>,
        source: rusqlite::Error,
    ) -> Self {
        TranscriptError::AdapterOpencodeQuery {
            detail: detail.into(),
            source,
        }
    }

    pub(crate) fn adapter_root_missing(path: &std::path::Path) -> Self {
        TranscriptError::AdapterRootMissing {
            path: path.display().to_string(),
        }
    }
}
