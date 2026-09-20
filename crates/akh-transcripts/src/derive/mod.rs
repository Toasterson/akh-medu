//! The `akh-derive` worker: anti-join discovery, per-record normalization,
//! session rollups, turn-boundary chunking and embedding — everything the
//! `akh-derive` binary (`src/bin/akh-derive.rs`) loops over.
//!
//! Design: [derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md)
//! (trigger, watermark-by-absence, re-derivation, poison records, runtime
//! shape) and [embedding-pipeline ticket](../../../docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md)
//! (what gets embedded, chunk granularity, re-embedding). Plan increment 5:
//! `docs/ai/plans/2026-09-19-transcript-harvest.md`.
//!
//! Split into two submodules along the same "discovery vs. embedding" line
//! the tickets draw:
//! - [`pass`] — anti-join stream discovery, per-record normalization
//!   (including the poison-record `catch_unwind` boundary), session rollups,
//!   and turn-boundary chunking.
//! - [`embed`] — the embedding phase, run after every pass.

pub mod embed;
pub mod pass;

use sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement};

use crate::error::{TranscriptError, TranscriptResult};
use akh_embed::SharedEmbedder;

/// The derivation version this build of `akh-derive` produces.
///
/// Pending is defined as "no `th_events` row at this version"
/// (derivation-worker ticket, §1), so **bumping this constant is the
/// documented trigger for re-deriving the whole corpus** — no deletes, no
/// window during which the corpus has a hole, and old rows become simply
/// unreferenced once their replacement lands. Bump it whenever the
/// normalizer, render, or chunking logic changes in a way that should be
/// reflected in already-derived data.
pub const DERIVATION_VERSION: i32 = 1;

/// Default number of pending *streams* — not raw records — per pass. Streams
/// are the unit of work because turn-boundary chunking needs a stream's
/// prose events in order, and streams are small (median 156 records, max
/// 11,762 per the plan's measured corpus).
pub const DEFAULT_BATCH_STREAMS: u64 = 200;

/// Default chunk size for the embedding phase's own pending-lookup —
/// distinct from `DEFAULT_BATCH_STREAMS`, since embedding batches chunks,
/// not streams. Mirrors `crates/seshat/src/embed_worker.rs`'s batch shape.
pub const DEFAULT_EMBED_BATCH: u64 = 64;

/// One pass's configuration. Kept separate from environment parsing (owned
/// by the binary) so this module has no `std::env` dependency.
#[derive(Debug, Clone, Copy)]
pub struct DeriveConfig {
    pub batch_streams: u64,
    /// `None` disables the embedding phase entirely (`--no-embed`).
    pub embed_batch: Option<u64>,
}

impl Default for DeriveConfig {
    fn default() -> Self {
        DeriveConfig {
            batch_streams: DEFAULT_BATCH_STREAMS,
            embed_batch: Some(DEFAULT_EMBED_BATCH),
        }
    }
}

/// Totals from one pass, in the exact order the worker logs them:
/// `derive pass: streams=<n> records=<n> events=<n> noise=<n>
/// undetermined=<n> failed=<n> chunks=<n> embedded=<n>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassSummary {
    pub streams: usize,
    pub records: usize,
    pub events: usize,
    pub noise: usize,
    pub undetermined: usize,
    pub failed: usize,
    pub chunks: usize,
    pub embedded: usize,
}

impl std::fmt::Display for PassSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "derive pass: streams={} records={} events={} noise={} undetermined={} failed={} chunks={} embedded={}",
            self.streams,
            self.records,
            self.events,
            self.noise,
            self.undetermined,
            self.failed,
            self.chunks,
            self.embedded
        )
    }
}

impl PassSummary {
    fn add_stream(&mut self, stream: &pass::StreamStats) {
        self.records += stream.records;
        self.events += stream.events;
        self.noise += stream.noise;
        self.undetermined += stream.undetermined;
        self.failed += stream.failed;
        self.chunks += stream.chunks;
    }
}

/// Run one full derivation pass: discover pending streams (§1 of the
/// derivation-worker ticket), normalize each into `th_events` (with the
/// poison-record `catch_unwind` boundary — §2), roll up `th_sessions`,
/// chunk `prompt`/`reply` runs into `th_chunks` (embedding-pipeline ticket
/// §2), and — unless `config.embed_batch` is `None` — embed everything
/// still pending.
///
/// Streams, not raw records, are the unit of work: turn-boundary chunking
/// needs a stream's prose events in seq order, so batching anything finer
/// would mean re-reading the same stream's already-derived events on every
/// record.
pub async fn run_pass(
    db: &DatabaseConnection,
    config: &DeriveConfig,
    embedder: Option<&SharedEmbedder>,
) -> TranscriptResult<PassSummary> {
    let mut summary = PassSummary::default();

    let pending = pass::find_pending_streams(db, DERIVATION_VERSION, config.batch_streams).await?;
    summary.streams = pending.len();

    // A session can span several streams (main + subagents); avoid
    // recomputing its rollup once per stream when more than one of its
    // streams is pending in the same pass.
    let mut sessions_upserted = std::collections::HashSet::new();

    for (source, session_id, stream_id) in &pending {
        let stream_stats =
            pass::process_stream(db, DERIVATION_VERSION, source, session_id, stream_id).await?;
        summary.add_stream(&stream_stats);

        if sessions_upserted.insert((source.clone(), session_id.clone())) {
            pass::upsert_session(db, DERIVATION_VERSION, source, session_id).await?;
        }
    }

    if let (Some(embedder), Some(embed_batch)) = (embedder, config.embed_batch) {
        summary.embedded = embed::embed_pending(db, embedder, embed_batch).await?;
    }

    Ok(summary)
}

/// Advisory-lock key for the `akh-derive` singleton, held for the process's
/// entire lifetime rather than released after one operation like
/// [`crate::pool::run_migrations_locked`]'s migration lock. Distinct from
/// that lock's key (`crates/akh-transcripts/src/pool.rs`, hardcoded here
/// rather than imported since that constant is private to its module — the
/// same "kept as a literal so a future collision is caught by inspection"
/// choice `pool.rs`'s own test makes against `akhomed-store`'s key) and from
/// `akhomed_store::MIGRATION_LOCK_KEY`. Mnemonic ASCII "AKDERIVE".
const SINGLETON_LOCK_KEY: i64 = 0x414B_4445_5249_5645_u64 as i64;

/// A held Postgres session advisory lock, released when this value (and the
/// dedicated single connection behind it) is dropped — i.e. for the rest of
/// the process's life once acquired at startup. Not `Clone`; there is
/// exactly one per process by construction.
pub struct SingletonGuard {
    _db: DatabaseConnection,
}

/// Try to become the one running `akh-derive` for this database.
///
/// Uses `pg_try_advisory_lock` (non-blocking) rather than
/// `pg_advisory_lock`: a second instance starting up should fail fast with a
/// clear diagnostic, per the derivation-worker ticket's "singleton via an
/// advisory lock" runtime decision — not queue up waiting for the first one
/// to exit. Returns [`TranscriptError::DeriveAlreadyRunning`] if another
/// process already holds it.
///
/// Opens a **dedicated single-connection** pool, the same reason
/// [`crate::pool::run_migrations_locked`] does: a session advisory lock is
/// tied to the backend connection that took it, so every check of "am I
/// still the singleton" — implicitly, simply staying alive — must share
/// that one connection for the lock to mean anything.
pub async fn acquire_singleton(url: &str) -> TranscriptResult<SingletonGuard> {
    let mut opts = sea_orm::ConnectOptions::new(url.to_owned());
    opts.max_connections(1).sqlx_logging(false);
    let db = sea_orm::Database::connect(opts)
        .await
        .map_err(|source| TranscriptError::Connect {
            url: redact_url(url),
            source,
        })?;

    if db.get_database_backend() != DatabaseBackend::Postgres {
        // Non-Postgres backends have no advisory locks; every caller in
        // practice points this at harvest-pg, so this is a test-only escape
        // hatch, not a supported production path.
        return Ok(SingletonGuard { _db: db });
    }

    let row = db
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            format!("SELECT pg_try_advisory_lock({SINGLETON_LOCK_KEY}) AS acquired"),
        ))
        .await
        .map_err(|source| TranscriptError::AdvisoryLock {
            action: "acquire",
            source,
        })?
        .ok_or_else(|| TranscriptError::AdvisoryLock {
            action: "acquire",
            source: sea_orm::DbErr::Custom("pg_try_advisory_lock returned no row".to_string()),
        })?;
    let acquired: bool = row.try_get("", "acquired").map_err(TranscriptError::from)?;

    if !acquired {
        return Err(TranscriptError::DeriveAlreadyRunning);
    }

    Ok(SingletonGuard { _db: db })
}

/// Strip credentials the same way `pool::redact_url` does. Duplicated
/// (rather than made `pub(crate)` there, which is out of this increment's
/// scope) because it is a one-line, purely cosmetic concern used only in an
/// error path here.
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
    fn singleton_lock_key_is_distinct_from_migration_lock_key() {
        // Literal copy of `pool::MIGRATION_LOCK_KEY`, which is private to
        // its module — see the comment on `SINGLETON_LOCK_KEY`.
        const MIGRATION_LOCK_KEY: i64 = 0x414B_5452_4E53_4352_u64 as i64;
        assert_ne!(SINGLETON_LOCK_KEY, MIGRATION_LOCK_KEY);
    }

    #[test]
    fn pass_summary_display_matches_the_documented_log_line_shape() {
        let summary = PassSummary {
            streams: 1,
            records: 2,
            events: 3,
            noise: 4,
            undetermined: 5,
            failed: 6,
            chunks: 7,
            embedded: 8,
        };
        assert_eq!(
            summary.to_string(),
            "derive pass: streams=1 records=2 events=3 noise=4 undetermined=5 failed=6 chunks=7 embedded=8"
        );
    }
}
