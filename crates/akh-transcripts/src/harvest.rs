//! Library layer for the `akh-harvest` binary (increment 3 of
//! `docs/ai/plans/2026-09-19-transcript-harvest.md`).
//!
//! Composes the layers that already exist: [`crate::adapter`] (read-only
//! source readers) and [`crate::entity::raw_record`] / [`crate::pool`] (the
//! `harvest-pg` store). This module owns none of those — only the per-tick
//! orchestration: resolve config from the environment, load a per-`(source,
//! stream)` watermark, read new records, resolve `observed_at` per the
//! carry-forward contract (`store-schema.md` §2, frozen), and batch-insert.
//!
//! Design notes worth keeping in view while reading this file:
//!
//! - **Watermarks are files, not database rows.** The insert-only PG role
//!   cannot `SELECT` its own writes back (`ingest-transport.md`), so resume
//!   state lives under [`HarvestConfig::state_dir`], one JSON file per
//!   [`Source`]. Losing that directory is harmless: every stream re-reads
//!   from zero and collides into `ON CONFLICT DO NOTHING`
//!   (`source-file-lifecycle.md` §2).
//! - **OpenCode's watermark key cannot be the bare stream path.** Every
//!   OpenCode [`StreamRef`] shares one `path` (`opencode.db`) — the adapter
//!   multiplexes streams by `session_id`, not by file. [`watermark_key`]
//!   therefore keys OpenCode entries on `path#session_id`; the JSONL sources
//!   key on `path` alone, since it is already unique per stream there.
//! - **The watermark only advances after its batch commits.** A failed
//!   insert leaves the in-memory watermark map untouched for that stream, so
//!   the next tick re-reads from the old offset — re-inserting already-committed
//!   rows from an earlier chunk in the same failed batch is a harmless
//!   `ON CONFLICT DO NOTHING` (`ingest-transport.md` §2).
//! - **`forward_fill_observed_at` needs a seed to survive a resumed tail
//!   read.** The adapter's version (`adapter::scan`) only ever sees the
//!   records in one call, so a stream resumed mid-run would otherwise lose
//!   its preceding timestamp and quarantine records that a full re-read
//!   would have resolved. [`resolve_observed_at`] mirrors that function's
//!   exact carry-forward algorithm with an added seed, rather than importing
//!   it and re-wrapping records — [`tests::resolve_observed_at_matches_unseeded_forward_fill_when_seed_is_none`]
//!   pins the two to the same behaviour so they cannot drift apart silently.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use miette::Diagnostic;
use sea_orm::sea_query::OnConflict;
use sea_orm::{ActiveValue::Set, DatabaseConnection, EntityTrait};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::adapter::{self, FileId, ObservedAt, StreamRef};
use crate::entity::raw_record;
use crate::error::TranscriptError;
use crate::model::{Source, StreamKind, VerbatimRecord};
use crate::pool;

/// Every source `akh-harvest` knows about, in the order a tick processes
/// them. Not [`Source`]'s own concern — that enum is the adapter/store
/// vocabulary, this ordering is a harvester scheduling choice.
const ALL_SOURCES: [Source; 3] = [Source::ClaudeCode, Source::Codex, Source::Opencode];

/// Records are inserted in chunks of this size (contract point 5). Each
/// chunk is one `INSERT ... ON CONFLICT DO NOTHING` statement; a failure
/// partway through a stream's batch still leaves earlier chunks committed
/// (harmlessly re-insertable) and stops the watermark from advancing.
const INSERT_BATCH_SIZE: usize = 500;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

pub type HarvestResult<T> = Result<T, HarvestError>;

/// Errors that abort a tick with a non-zero exit code: an unusable state
/// directory, a missing required environment variable, a corrupt watermark
/// file, or (via [`TranscriptError::Connect`]) an unreachable database. A
/// single stream failing to read or insert is deliberately *not* one of
/// these — see [`run`]'s module docs on that split.
#[derive(Debug, Error, Diagnostic)]
pub enum HarvestError {
    #[error("environment variable {var} is required but was not set")]
    #[diagnostic(
        code(akh_harvest::env_missing),
        help(
            "Set {var} in the environment, or source the env file this machine's runbook \
             describes (docs/ops/transcript-harvest.md): \
             `set -a; . ~/.config/akh-transcripts/harvest.env; set +a`."
        )
    )]
    EnvMissing { var: &'static str },

    #[error("state directory {path} is not usable: {reason}")]
    #[diagnostic(
        code(akh_harvest::state_dir),
        help(
            "Confirm the directory exists (or can be created) and is writable by this user. \
             Default is $XDG_STATE_HOME/akh-transcripts, else ~/.local/state/akh-transcripts; \
             override with AKH_TRANSCRIPTS_STATE_DIR."
        )
    )]
    StateDir { path: String, reason: String },

    #[error("failed to read watermark file {path}: {source}")]
    #[diagnostic(
        code(akh_harvest::watermark_read),
        help(
            "If the file is corrupt beyond repair, delete it: a full re-scan of that source \
             is harmless, every record collides into ON CONFLICT DO NOTHING."
        )
    )]
    WatermarkRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse watermark file {path} as JSON: {source}")]
    #[diagnostic(
        code(akh_harvest::watermark_parse),
        help(
            "Delete the file to force a full re-scan of that source; it is harmless (ON \
             CONFLICT DO NOTHING dedupes every row that was already stored)."
        )
    )]
    WatermarkParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to write watermark file {path}: {source}")]
    #[diagnostic(
        code(akh_harvest::watermark_write),
        help("Check available disk space and write permissions on the state directory.")
    )]
    WatermarkWrite {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Covers, among others, [`TranscriptError::Connect`] — the database
    /// unreachable case this binary's contract calls out by name.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Transcript(#[from] TranscriptError),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved environment configuration for one `akh-harvest` run. See the
/// binary's module docs (`src/bin/akh-harvest.rs`) for the full variable
/// list; this struct is the one place their defaults are computed.
#[derive(Clone, Debug)]
pub struct HarvestConfig {
    pub database_url: String,
    pub machine: String,
    pub state_dir: PathBuf,
    pub claude_root: PathBuf,
    pub codex_root: PathBuf,
    pub opencode_db: PathBuf,
}

impl HarvestConfig {
    /// Resolve every setting from the environment. `AKH_TRANSCRIPTS_DATABASE_URL`
    /// is the only required variable; everything else has a documented default.
    pub fn from_env() -> HarvestResult<Self> {
        let database_url = required_env("AKH_TRANSCRIPTS_DATABASE_URL")?;
        let machine = optional_env("AKH_TRANSCRIPTS_MACHINE").unwrap_or_else(default_hostname);
        let home = home_dir()?;

        let state_dir = optional_env("AKH_TRANSCRIPTS_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_state_dir(&home));
        let claude_root = optional_env("AKH_TRANSCRIPTS_CLAUDE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".claude").join("projects"));
        let codex_root = optional_env("AKH_TRANSCRIPTS_CODEX_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".codex").join("sessions"));
        let opencode_db = optional_env("AKH_TRANSCRIPTS_OPENCODE_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                home.join(".local")
                    .join("share")
                    .join("opencode")
                    .join("opencode.db")
            });

        Ok(HarvestConfig {
            database_url,
            machine,
            state_dir,
            claude_root,
            codex_root,
            opencode_db,
        })
    }

    fn root_for(&self, source: Source) -> &Path {
        match source {
            Source::ClaudeCode => &self.claude_root,
            Source::Codex => &self.codex_root,
            Source::Opencode => &self.opencode_db,
        }
    }
}

fn required_env(var: &'static str) -> HarvestResult<String> {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .ok_or(HarvestError::EnvMissing { var })
}

fn optional_env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

fn default_hostname() -> String {
    hostname::get()
        .ok()
        .map(|h| h.to_string_lossy().into_owned())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown-machine".to_string())
}

fn home_dir() -> HarvestResult<PathBuf> {
    required_env("HOME").map(PathBuf::from)
}

fn default_state_dir(home: &Path) -> PathBuf {
    default_state_dir_with_xdg(home, optional_env("XDG_STATE_HOME"))
}

/// Pure core of [`default_state_dir`], taking the `XDG_STATE_HOME` value as
/// a parameter rather than reading the process environment directly — so
/// tests can exercise both branches without mutating global state that
/// other tests in this same binary might read concurrently.
fn default_state_dir_with_xdg(home: &Path, xdg_state_home: Option<String>) -> PathBuf {
    if let Some(xdg) = xdg_state_home {
        return PathBuf::from(xdg).join("akh-transcripts");
    }
    home.join(".local").join("state").join("akh-transcripts")
}

// ---------------------------------------------------------------------------
// Run options + summary
// ---------------------------------------------------------------------------

/// Per-invocation behaviour, set from CLI flags by `src/bin/akh-harvest.rs`.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Discover, read and forward-fill only; write nothing, touch no
    /// watermark, require no database connection.
    pub dry_run: bool,
    /// Restrict the tick to these sources. Empty means "every source".
    pub sources: Vec<Source>,
}

impl RunOptions {
    fn wants(&self, source: Source) -> bool {
        self.sources.is_empty() || self.sources.contains(&source)
    }
}

/// Aggregate counts for one tick, across every source and stream. The final
/// log line (contract point 6) is [`Summary::log_line`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub streams: usize,
    pub records_read: usize,
    pub inserted: usize,
    pub deduped: usize,
    pub quarantined: usize,
    pub failed_streams: usize,
}

impl Summary {
    pub fn log_line(&self) -> String {
        format!(
            "harvest complete: streams={} records_read={} inserted={} deduped={} quarantined={} failed_streams={}",
            self.streams, self.records_read, self.inserted, self.deduped, self.quarantined, self.failed_streams
        )
    }
}

// ---------------------------------------------------------------------------
// Watermarks
// ---------------------------------------------------------------------------

/// Per-stream resume state, one JSON file per [`Source`] under
/// [`HarvestConfig::state_dir`], keyed by [`watermark_key`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct StreamWatermark {
    /// `None` for OpenCode, which has no per-stream backing file (see
    /// module docs). `Some` for the JSONL sources, and the basis of
    /// [`resume_state_for`]'s inode-change invalidation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    inode: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dev: Option<u64>,
    /// Byte offset for the JSONL sources; next-unseen `seq` for OpenCode —
    /// exactly the unit [`adapter::read_stream`] takes and returns.
    offset: u64,
    /// The highest `seq` seen so far, for operator visibility only —
    /// `offset` is what actually drives resume.
    last_seq: i64,
    /// Seed for [`resolve_observed_at`] on the next tick's resumed read.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    last_timestamp: Option<DateTime<Utc>>,
    /// Codex's `cli_version`, threaded across ticks because the adapter only
    /// threads it within one `read_stream` call (`codex.rs`'s own doc
    /// comment). `None` for every other source.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    codex_cli_version: Option<String>,
}

/// Key a watermark map by. `path` alone is unique per stream for the JSONL
/// sources; OpenCode's streams all share one `path` (`opencode.db`), so its
/// key adds `session_id` to disambiguate.
fn watermark_key(stream: &StreamRef) -> String {
    match stream.source {
        Source::Opencode => format!("{}#{}", stream.path.display(), stream.session_id),
        Source::ClaudeCode | Source::Codex => stream.path.display().to_string(),
    }
}

fn watermark_path(state_dir: &Path, source: Source) -> PathBuf {
    state_dir.join(format!("{}.json", source.as_str()))
}

fn load_watermarks(
    state_dir: &Path,
    source: Source,
) -> HarvestResult<BTreeMap<String, StreamWatermark>> {
    let path = watermark_path(state_dir, source);
    match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).map_err(|source| {
            HarvestError::WatermarkParse {
                path: path.display().to_string(),
                source,
            }
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(source) => Err(HarvestError::WatermarkRead {
            path: path.display().to_string(),
            source,
        }),
    }
}

/// Atomic write: write to a sibling `.tmp` file, then rename over the real
/// path, so a crash mid-write never leaves a half-written watermark file
/// behind for the next tick to fail on.
fn save_watermarks(
    state_dir: &Path,
    source: Source,
    watermarks: &BTreeMap<String, StreamWatermark>,
) -> HarvestResult<()> {
    let path = watermark_path(state_dir, source);
    let tmp_path = state_dir.join(format!("{}.json.tmp", source.as_str()));
    let json = serde_json::to_string_pretty(watermarks).map_err(|e| {
        HarvestError::WatermarkWrite {
            path: path.display().to_string(),
            source: std::io::Error::other(e),
        }
    })?;
    std::fs::write(&tmp_path, json).map_err(|source| HarvestError::WatermarkWrite {
        path: tmp_path.display().to_string(),
        source,
    })?;
    std::fs::rename(&tmp_path, &path).map_err(|source| HarvestError::WatermarkWrite {
        path: path.display().to_string(),
        source,
    })
}

fn ensure_state_dir(path: &Path) -> HarvestResult<()> {
    std::fs::create_dir_all(path).map_err(|e| HarvestError::StateDir {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    // `create_dir_all` succeeds on a read-only parent when the directory
    // already exists; probe with a real write so an unusable state dir is
    // caught here rather than at the first watermark save of the tick.
    let probe = path.join(".akh-harvest-write-check");
    std::fs::write(&probe, b"ok").map_err(|e| HarvestError::StateDir {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

/// Resolved resume state for one stream, after applying the invalidation
/// rule (`source-file-lifecycle.md` §2): inode/dev change, or the file
/// shrinking below the stored offset, means re-read from zero.
struct ResumeState {
    offset: u64,
    last_seq: i64,
    last_timestamp: Option<DateTime<Utc>>,
    codex_cli_version: Option<String>,
}

impl ResumeState {
    fn fresh() -> Self {
        ResumeState {
            offset: 0,
            last_seq: -1,
            last_timestamp: None,
            codex_cli_version: None,
        }
    }
}

/// `stat`s `stream.path` itself for the shrink check — deliberately not
/// relying solely on [`adapter::scan::read_complete_lines`]'s own internal
/// "offset past EOF -> read from zero" fallback, since that fallback only
/// fires on the exact "past current length" case and this rule is a
/// documented, independently-testable contract of the harvester, not an
/// adapter implementation detail to lean on silently.
fn resume_state_for(stream: &StreamRef, watermark: Option<&StreamWatermark>) -> ResumeState {
    let Some(wm) = watermark else {
        return ResumeState::fresh();
    };
    match stream.file_id {
        Some(FileId { dev, ino }) => {
            let same_identity = wm.dev == Some(dev) && wm.inode == Some(ino);
            let shrank = std::fs::metadata(&stream.path)
                .map(|m| m.len() < wm.offset)
                .unwrap_or(false);
            if same_identity && !shrank {
                ResumeState {
                    offset: wm.offset,
                    last_seq: wm.last_seq,
                    last_timestamp: wm.last_timestamp,
                    codex_cli_version: wm.codex_cli_version.clone(),
                }
            } else {
                ResumeState::fresh()
            }
        }
        // OpenCode: no per-stream backing file, so no identity to check —
        // the offset (next-unseen seq) is trusted as-is.
        None => ResumeState {
            offset: wm.offset,
            last_seq: wm.last_seq,
            last_timestamp: wm.last_timestamp,
            codex_cli_version: wm.codex_cli_version.clone(),
        },
    }
}

// ---------------------------------------------------------------------------
// observed_at resolution (seeded carry-forward)
// ---------------------------------------------------------------------------

/// [`adapter::forward_fill_observed_at`]'s exact carry-forward algorithm,
/// seeded with the watermark's `last_timestamp` so a resumed tail read still
/// has its preceding timestamp available — that seed is the entire reason
/// the watermark stores it (`store-schema.md` §2). Returns the resolved
/// per-record outcome plus the final carried timestamp, which becomes the
/// next watermark's seed.
///
/// Deliberately re-implements rather than calls the adapter function: doing
/// so by prepending a sentinel record would require cloning the whole batch
/// (raw bytes and parsed JSON payload included) just to smuggle one
/// timestamp through, for logic that is eight lines of pure per-record
/// state. [`tests::resolve_observed_at_matches_unseeded_forward_fill_when_seed_is_none`]
/// keeps the two from drifting apart.
fn resolve_observed_at(
    records: &[VerbatimRecord],
    seed: Option<DateTime<Utc>>,
) -> (Vec<ObservedAt>, Option<DateTime<Utc>>) {
    let mut last = seed;
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        match record.timestamp {
            Some(ts) => {
                last = Some(ts);
                out.push(ObservedAt::Own(ts));
            }
            None => match last {
                Some(ts) => out.push(ObservedAt::ForwardFilled(ts)),
                None => out.push(ObservedAt::Quarantined),
            },
        }
    }
    (out, last)
}

// ---------------------------------------------------------------------------
// OpenCode timestamp derivation
// ---------------------------------------------------------------------------

/// OpenCode's `event` table carries no timestamp column
/// (`adapter::opencode`'s own doc comment); derive one from the event's
/// `data` JSON, first present of `time.created`, `time.updated`,
/// `timestamp`, `created_at`, accepting either epoch milliseconds or
/// RFC 3339. Absent from all four -> `None`, which becomes
/// [`ObservedAt::Quarantined`] downstream unless forward-fill covers it.
fn fill_opencode_timestamps(records: &mut [VerbatimRecord]) {
    for record in records.iter_mut() {
        if record.timestamp.is_some() {
            continue;
        }
        record.timestamp = extract_opencode_timestamp(record);
    }
}

fn extract_opencode_timestamp(record: &VerbatimRecord) -> Option<DateTime<Utc>> {
    let data = record.payload.as_ref()?.get("data")?;
    [
        // Measured against the live database (2026-09-20): `session.*` and
        // `message.updated` events nest their clock under `info.time`, not
        // at the top level the spec's shorthand names suggest — `info` is
        // the session/message snapshot the event carries, and it wraps the
        // same `{created, updated}` shape.
        data.pointer("/info/time/created"),
        data.pointer("/info/time/updated"),
        // Kept for event shapes that do carry `time` at the top level
        // (covers the general case named in the spec, even though no
        // sampled event type used it directly).
        data.pointer("/time/created"),
        data.pointer("/time/updated"),
        // `message.part.updated` carries `time` as a bare epoch-millis
        // number, not an object — also measured, not assumed.
        data.get("time"),
        data.get("timestamp"),
        data.get("created_at"),
    ]
    .into_iter()
    .flatten()
    .find_map(parse_flexible_timestamp)
}

/// Accepts a JSON number (epoch milliseconds), a numeric string (same), or
/// an RFC 3339 string.
fn parse_flexible_timestamp(value: &Value) -> Option<DateTime<Utc>> {
    if let Some(millis) = value.as_i64() {
        return epoch_millis(millis);
    }
    if let Some(s) = value.as_str() {
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Some(dt.with_timezone(&Utc));
        }
        if let Ok(millis) = s.parse::<i64>() {
            return epoch_millis(millis);
        }
    }
    None
}

fn epoch_millis(millis: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_millis_opt(millis).single()
}

// ---------------------------------------------------------------------------
// Codex cli_version threading
// ---------------------------------------------------------------------------

/// Threads Codex's `cli_version` (carried as `schema_marker`) across ticks.
/// The adapter already threads it within one `read_stream` call
/// (`adapter::codex`'s doc comment); this extends that across a resumed tail
/// read that starts after the file's last `session_meta`, using the
/// watermark's persisted value as the seed. Returns the final known version,
/// which becomes the next watermark's seed.
fn thread_codex_cli_version(
    records: &mut [VerbatimRecord],
    seed: Option<String>,
) -> Option<String> {
    let mut current = seed;
    for record in records.iter_mut() {
        match &record.origin.schema_marker {
            Some(v) => current = Some(v.clone()),
            None => record.origin.schema_marker = current.clone(),
        }
    }
    current
}

// ---------------------------------------------------------------------------
// Insertion
// ---------------------------------------------------------------------------

fn stream_kind_str(kind: StreamKind) -> &'static str {
    match kind {
        StreamKind::Main => "main",
        StreamKind::Subagent => "subagent",
        StreamKind::WorkflowJournal => "workflow_journal",
        StreamKind::EventLog => "event_log",
    }
}

/// Postgres's `text` and `jsonb` types reject the NUL byte (`\0`)
/// outright — a hard constraint of Postgres's own storage format, not a
/// choice made here. Measured live on 2026-09-20: a Claude Code session
/// whose transcript contained a stray NUL failed its **entire** insert
/// chunk with `unsupported Unicode escape sequence`, taking up to
/// [`INSERT_BATCH_SIZE`] unrelated, otherwise-healthy records down with it.
/// Stripping is scoped to exactly this byte — every other byte in
/// `raw`/`payload` still round-trips untouched, so "verbatim" only bends at
/// the one point where Postgres itself cannot represent the input at all.
fn strip_nul_bytes(s: &str) -> String {
    if s.contains('\0') {
        s.replace('\0', "")
    } else {
        s.to_string()
    }
}

fn sanitize_json_value(value: &mut Value) {
    match value {
        Value::String(s) => {
            if s.contains('\0') {
                *s = s.replace('\0', "");
            }
        }
        Value::Array(items) => items.iter_mut().for_each(sanitize_json_value),
        Value::Object(map) => map.values_mut().for_each(sanitize_json_value),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn to_active_model(
    record: &VerbatimRecord,
    observed_at: DateTime<Utc>,
    synthetic: bool,
    harvested_at: DateTime<Utc>,
) -> raw_record::ActiveModel {
    let mut payload = record.payload.clone();
    if let Some(p) = payload.as_mut() {
        sanitize_json_value(p);
    }
    raw_record::ActiveModel {
        observed_at: Set(observed_at),
        source: Set(record.key.source.as_str().to_string()),
        session_id: Set(record.key.session_id.clone()),
        stream_id: Set(record.key.stream_id.clone()),
        seq: Set(record.key.seq),
        stream_kind: Set(stream_kind_str(record.stream_kind).to_string()),
        machine: Set(record.origin.machine.clone()),
        raw: Set(strip_nul_bytes(&record.raw)),
        payload: Set(payload),
        record_type: Set(record.origin.record_type.clone()),
        schema_marker: Set(record.origin.schema_marker.clone()),
        origin_path: Set(record.origin.path.clone()),
        observed_at_synthetic: Set(synthetic),
        harvested_at: Set(harvested_at),
    }
}

/// Insert `models` in chunks of [`INSERT_BATCH_SIZE`], `ON CONFLICT DO
/// NOTHING`. Returns `(rows actually inserted, rows attempted)` — the
/// difference is the dedupe count. Stops at the first chunk that errors;
/// earlier chunks in the same call remain committed (harmless, per module
/// docs), and the caller must not advance the watermark when this returns
/// `Err`.
async fn insert_batches(
    db: &DatabaseConnection,
    models: Vec<raw_record::ActiveModel>,
) -> Result<(usize, usize), TranscriptError> {
    let attempted = models.len();
    let mut inserted_total: u64 = 0;
    for chunk in models.chunks(INSERT_BATCH_SIZE) {
        let rows_affected = raw_record::Entity::insert_many(chunk.to_vec())
            .on_conflict(OnConflict::new().do_nothing().to_owned())
            .exec_without_returning(db)
            .await
            .map_err(TranscriptError::from)?;
        inserted_total += rows_affected;
    }
    Ok((inserted_total as usize, attempted))
}

// ---------------------------------------------------------------------------
// The tick
// ---------------------------------------------------------------------------

/// Run one harvest tick: discover every configured source, read new records
/// from each stream, resolve `observed_at`, and (unless `opts.dry_run`)
/// batch-insert and advance watermarks.
///
/// Returns `Err` only for the two conditions the binary's contract treats as
/// fatal: the database is unreachable (surfaces as
/// [`TranscriptError::Connect`] via [`HarvestError::Transcript`]) or the
/// state directory is unusable ([`HarvestError::StateDir`]). Everything
/// else — a source root that does not exist, a stream that fails to parse,
/// a batch that fails to insert — is logged and folded into
/// [`Summary::failed_streams`] so the tick keeps making progress on the
/// streams that *are* healthy; the calling systemd timer retries on its own
/// schedule regardless.
pub async fn run(cfg: &HarvestConfig, opts: &RunOptions) -> HarvestResult<Summary> {
    if !opts.dry_run {
        ensure_state_dir(&cfg.state_dir)?;
    }

    let db = if opts.dry_run {
        None
    } else {
        Some(pool::connect(&cfg.database_url).await?)
    };

    let harvested_at = Utc::now();
    let mut summary = Summary::default();

    for source in ALL_SOURCES {
        if !opts.wants(source) {
            continue;
        }
        let root = cfg.root_for(source);
        if !root.exists() {
            tracing::warn!(
                source = source.as_str(),
                path = %root.display(),
                "transcript root does not exist, skipping this source for this tick"
            );
            continue;
        }

        let streams = match adapter::discover(source, root) {
            Ok(streams) => streams,
            Err(error) => {
                tracing::warn!(source = source.as_str(), %error, "discovery failed, skipping this source for this tick");
                continue;
            }
        };
        // The under-discovery alarm the runbook and source-file-lifecycle.md
        // both call out by name: a harvester silently seeing fewer streams
        // than it should looks exactly like a healthy, quiet one.
        tracing::info!(source = source.as_str(), streams = streams.len(), "discovered streams");

        let watermarks_before = load_watermarks(&cfg.state_dir, source)?;
        let mut watermarks = watermarks_before.clone();

        for stream in &streams {
            summary.streams += 1;
            let key = watermark_key(stream);
            let resume = resume_state_for(stream, watermarks_before.get(&key));

            let (mut records, new_offset) =
                match adapter::read_stream(stream, resume.offset, &cfg.machine) {
                    Ok(v) => v,
                    Err(error) => {
                        tracing::warn!(
                            source = source.as_str(),
                            session_id = %stream.session_id,
                            stream_id = %stream.stream_id,
                            path = %stream.path.display(),
                            %error,
                            "failed to read stream, watermark unchanged; will retry next tick"
                        );
                        summary.failed_streams += 1;
                        continue;
                    }
                };

            summary.records_read += records.len();

            if source == Source::Opencode {
                fill_opencode_timestamps(&mut records);
            }
            let codex_cli_version = if source == Source::Codex {
                thread_codex_cli_version(&mut records, resume.codex_cli_version.clone())
            } else {
                None
            };

            let (resolved, last_timestamp) = resolve_observed_at(&records, resume.last_timestamp);

            let mut to_insert = Vec::with_capacity(records.len());
            let mut quarantined_here = 0usize;
            for (record, observed) in records.iter().zip(resolved.iter()) {
                match observed {
                    ObservedAt::Own(ts) => to_insert.push(to_active_model(record, *ts, false, harvested_at)),
                    ObservedAt::ForwardFilled(ts) => {
                        to_insert.push(to_active_model(record, *ts, true, harvested_at))
                    }
                    ObservedAt::Quarantined => quarantined_here += 1,
                }
            }
            if quarantined_here > 0 {
                tracing::warn!(
                    source = source.as_str(),
                    session_id = %stream.session_id,
                    stream_id = %stream.stream_id,
                    count = quarantined_here,
                    "quarantined records with no derivable observed_at (no own timestamp, none preceding in stream)"
                );
            }
            summary.quarantined += quarantined_here;

            let last_seq = records
                .last()
                .map(|r| r.key.seq)
                .unwrap_or(resume.last_seq);

            if opts.dry_run {
                tracing::info!(
                    source = source.as_str(),
                    session_id = %stream.session_id,
                    stream_id = %stream.stream_id,
                    read = records.len(),
                    would_insert = to_insert.len(),
                    quarantined = quarantined_here,
                    "dry run: stream summary (nothing written)"
                );
                continue;
            }

            let db = db.as_ref().expect("connected whenever opts.dry_run is false");
            match insert_batches(db, to_insert).await {
                Ok((inserted, attempted)) => {
                    let deduped = attempted - inserted;
                    summary.inserted += inserted;
                    summary.deduped += deduped;
                    tracing::info!(
                        source = source.as_str(),
                        session_id = %stream.session_id,
                        stream_id = %stream.stream_id,
                        read = records.len(),
                        inserted,
                        deduped,
                        quarantined = quarantined_here,
                        "stream committed"
                    );
                    watermarks.insert(
                        key,
                        StreamWatermark {
                            inode: stream.file_id.map(|f| f.ino),
                            dev: stream.file_id.map(|f| f.dev),
                            offset: new_offset,
                            last_seq,
                            last_timestamp: last_timestamp.or(resume.last_timestamp),
                            codex_cli_version: codex_cli_version.or(resume.codex_cli_version),
                        },
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        source = source.as_str(),
                        session_id = %stream.session_id,
                        stream_id = %stream.stream_id,
                        %error,
                        "failed to insert batch, watermark unchanged; will retry next tick"
                    );
                    summary.failed_streams += 1;
                }
            }
        }

        if !opts.dry_run {
            save_watermarks(&cfg.state_dir, source, &watermarks)?;
        }
    }

    tracing::info!("{}", summary.log_line());
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{RecordKey, RecordOrigin, Source, StreamKind};
    use tempfile::tempdir;

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn record(seq: i64, timestamp: Option<DateTime<Utc>>) -> VerbatimRecord {
        VerbatimRecord {
            key: RecordKey {
                source: Source::ClaudeCode,
                session_id: "s".to_string(),
                stream_id: "main".to_string(),
                seq,
            },
            stream_kind: StreamKind::Main,
            raw: "{}".to_string(),
            payload: None,
            timestamp,
            origin: RecordOrigin {
                machine: "m".to_string(),
                path: "p".to_string(),
                record_type: None,
                schema_marker: None,
            },
        }
    }

    // -- watermark invalidation ------------------------------------------

    #[test]
    fn resume_state_reads_stored_offset_when_identity_matches_and_no_shrink() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sess.jsonl");
        std::fs::write(&path, "0123456789").unwrap();
        let file_id = adapter::scan::file_id(&path).expect("file_id");

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "s".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: path.clone(),
            file_id: Some(file_id),
        };
        let wm = StreamWatermark {
            inode: Some(file_id.ino),
            dev: Some(file_id.dev),
            offset: 5,
            last_seq: 2,
            last_timestamp: Some(ts("2026-07-05T11:08:16Z")),
            codex_cli_version: None,
        };

        let resume = resume_state_for(&stream, Some(&wm));
        assert_eq!(resume.offset, 5);
        assert_eq!(resume.last_seq, 2);
        assert_eq!(resume.last_timestamp, Some(ts("2026-07-05T11:08:16Z")));
    }

    #[test]
    fn resume_state_resets_on_inode_change() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sess.jsonl");
        std::fs::write(&path, "0123456789").unwrap();
        let real_file_id = adapter::scan::file_id(&path).expect("file_id");

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "s".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: path.clone(),
            file_id: Some(real_file_id),
        };
        // Watermark claims a different inode than the file actually has —
        // simulating a replaced file at the same path.
        let wm = StreamWatermark {
            inode: Some(real_file_id.ino + 1),
            dev: Some(real_file_id.dev),
            offset: 5,
            last_seq: 2,
            last_timestamp: Some(ts("2026-07-05T11:08:16Z")),
            codex_cli_version: None,
        };

        let resume = resume_state_for(&stream, Some(&wm));
        assert_eq!(resume.offset, 0);
        assert_eq!(resume.last_seq, -1);
        assert_eq!(resume.last_timestamp, None);
    }

    #[test]
    fn resume_state_resets_when_file_shrank_below_stored_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sess.jsonl");
        std::fs::write(&path, "short").unwrap(); // 5 bytes
        let file_id = adapter::scan::file_id(&path).expect("file_id");

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "s".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: path.clone(),
            file_id: Some(file_id),
        };
        // Watermark thinks it read past where the (now-shrunk) file ends.
        let wm = StreamWatermark {
            inode: Some(file_id.ino),
            dev: Some(file_id.dev),
            offset: 1000,
            last_seq: 9,
            last_timestamp: Some(ts("2026-07-05T11:08:16Z")),
            codex_cli_version: None,
        };

        let resume = resume_state_for(&stream, Some(&wm));
        assert_eq!(resume.offset, 0);
    }

    #[test]
    fn resume_state_with_no_watermark_starts_fresh() {
        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "s".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: PathBuf::from("/does/not/matter.jsonl"),
            file_id: None,
        };
        let resume = resume_state_for(&stream, None);
        assert_eq!(resume.offset, 0);
        assert_eq!(resume.last_timestamp, None);
    }

    #[test]
    fn opencode_streams_trust_the_stored_offset_with_no_identity_check() {
        // OpenCode's StreamRef never carries a file_id (module docs) — the
        // watermark's offset (next-unseen seq) is trusted outright.
        let stream = StreamRef {
            source: Source::Opencode,
            session_id: "ses_abc".to_string(),
            stream_id: "events".to_string(),
            stream_kind: StreamKind::EventLog,
            path: PathBuf::from("/wherever/opencode.db"),
            file_id: None,
        };
        let wm = StreamWatermark {
            inode: None,
            dev: None,
            offset: 42,
            last_seq: 41,
            last_timestamp: None,
            codex_cli_version: None,
        };
        let resume = resume_state_for(&stream, Some(&wm));
        assert_eq!(resume.offset, 42);
    }

    // -- watermark_key ------------------------------------------------------

    #[test]
    fn watermark_key_disambiguates_opencode_sessions_sharing_one_path() {
        let base = StreamRef {
            source: Source::Opencode,
            session_id: "ses_1".to_string(),
            stream_id: "events".to_string(),
            stream_kind: StreamKind::EventLog,
            path: PathBuf::from("/x/opencode.db"),
            file_id: None,
        };
        let other = StreamRef {
            session_id: "ses_2".to_string(),
            ..base.clone()
        };
        assert_ne!(watermark_key(&base), watermark_key(&other));
    }

    #[test]
    fn watermark_key_is_path_alone_for_jsonl_sources() {
        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "s".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: PathBuf::from("/x/sess.jsonl"),
            file_id: None,
        };
        assert_eq!(watermark_key(&stream), "/x/sess.jsonl");
    }

    // -- OpenCode timestamp extraction --------------------------------------

    fn opencode_record(data: Value) -> VerbatimRecord {
        VerbatimRecord {
            key: RecordKey {
                source: Source::Opencode,
                session_id: "ses_1".to_string(),
                stream_id: "events".to_string(),
                seq: 0,
            },
            stream_kind: StreamKind::EventLog,
            raw: "{}".to_string(),
            payload: Some(serde_json::json!({ "data": data })),
            timestamp: None,
            origin: RecordOrigin {
                machine: "m".to_string(),
                path: "opencode.db".to_string(),
                record_type: Some("message.updated".to_string()),
                schema_marker: Some("1".to_string()),
            },
        }
    }

    #[test]
    fn extracts_epoch_millis_from_time_created() {
        let record = opencode_record(serde_json::json!({ "time": { "created": 1_772_000_000_000i64 } }));
        let extracted = extract_opencode_timestamp(&record).unwrap();
        assert_eq!(extracted, Utc.timestamp_millis_opt(1_772_000_000_000).unwrap());
    }

    #[test]
    fn extracts_rfc3339_from_timestamp_field() {
        let record = opencode_record(serde_json::json!({ "timestamp": "2026-07-05T11:08:16Z" }));
        let extracted = extract_opencode_timestamp(&record).unwrap();
        assert_eq!(extracted, ts("2026-07-05T11:08:16Z"));
    }

    #[test]
    fn extracts_created_at_as_lowest_priority_fallback() {
        let record = opencode_record(serde_json::json!({ "created_at": "2026-07-05T11:08:16Z" }));
        assert_eq!(extract_opencode_timestamp(&record).unwrap(), ts("2026-07-05T11:08:16Z"));
    }

    #[test]
    fn prefers_time_created_over_time_updated_and_timestamp() {
        let record = opencode_record(serde_json::json!({
            "time": { "created": "2026-01-01T00:00:00Z", "updated": "2026-02-02T00:00:00Z" },
            "timestamp": "2026-03-03T00:00:00Z",
        }));
        assert_eq!(extract_opencode_timestamp(&record).unwrap(), ts("2026-01-01T00:00:00Z"));
    }

    /// Measured against the live `opencode.db` (2026-09-20): `session.*` and
    /// `message.updated` events wrap their clock as `info.time.{created,updated}`,
    /// not at the top level — this was the real cause of every OpenCode
    /// record quarantining on the first live dry run.
    #[test]
    fn extracts_nested_info_time_created_matching_the_real_session_updated_shape() {
        let record = opencode_record(serde_json::json!({
            "sessionID": "ses_x",
            "info": {
                "id": "ses_x",
                "time": { "created": 1_787_065_786_856i64, "updated": 1_787_065_787_339i64 },
            },
        }));
        let extracted = extract_opencode_timestamp(&record).unwrap();
        assert_eq!(extracted, Utc.timestamp_millis_opt(1_787_065_786_856).unwrap());
    }

    /// Measured shape of `message.part.updated`: `time` is a bare
    /// epoch-millis number at the top level of `data`, not an object.
    #[test]
    fn extracts_bare_top_level_time_number_matching_the_real_message_part_updated_shape() {
        let record = opencode_record(serde_json::json!({
            "sessionID": "ses_x",
            "part": { "id": "prt_x" },
            "time": 1_787_065_787_335i64,
        }));
        let extracted = extract_opencode_timestamp(&record).unwrap();
        assert_eq!(extracted, Utc.timestamp_millis_opt(1_787_065_787_335).unwrap());
    }

    #[test]
    fn absent_timestamp_fields_yield_none() {
        let record = opencode_record(serde_json::json!({ "unrelated": true }));
        assert_eq!(extract_opencode_timestamp(&record), None);
    }

    #[test]
    fn fill_opencode_timestamps_never_overwrites_an_existing_timestamp() {
        let mut record = opencode_record(serde_json::json!({ "timestamp": "2026-01-01T00:00:00Z" }));
        record.timestamp = Some(ts("2025-01-01T00:00:00Z"));
        let mut records = [record];
        fill_opencode_timestamps(&mut records);
        assert_eq!(records[0].timestamp, Some(ts("2025-01-01T00:00:00Z")));
    }

    // -- resolve_observed_at (seeded carry-forward) -------------------------

    #[test]
    fn seed_is_used_when_the_first_record_has_no_own_timestamp() {
        let seed = ts("2026-07-05T11:00:00Z");
        let records = [record(10, None), record(11, None)];
        let (resolved, last) = resolve_observed_at(&records, Some(seed));
        assert_eq!(resolved, vec![ObservedAt::ForwardFilled(seed), ObservedAt::ForwardFilled(seed)]);
        assert_eq!(last, Some(seed));
    }

    #[test]
    fn a_records_own_timestamp_overrides_and_replaces_the_seed() {
        let seed = ts("2026-01-01T00:00:00Z");
        let own = ts("2026-07-05T11:00:00Z");
        let records = [record(10, Some(own)), record(11, None)];
        let (resolved, last) = resolve_observed_at(&records, Some(seed));
        assert_eq!(resolved, vec![ObservedAt::Own(own), ObservedAt::ForwardFilled(own)]);
        assert_eq!(last, Some(own));
    }

    #[test]
    fn no_seed_and_no_own_timestamp_quarantines_exactly_like_the_unseeded_case() {
        let records = [record(10, None)];
        let (resolved, last) = resolve_observed_at(&records, None);
        assert_eq!(resolved, vec![ObservedAt::Quarantined]);
        assert_eq!(last, None);
    }

    /// Pins [`resolve_observed_at`] to [`adapter::forward_fill_observed_at`]'s
    /// behaviour whenever there is no seed, so the two cannot silently drift
    /// apart — see this function's module docs.
    #[test]
    fn resolve_observed_at_matches_unseeded_forward_fill_when_seed_is_none() {
        let ts0 = ts("2026-07-05T11:08:16Z");
        let ts2 = ts("2026-07-05T11:09:00Z");
        let records = [
            record(0, Some(ts0)),
            record(1, None),
            record(2, Some(ts2)),
            record(3, None),
        ];
        let (seeded, _) = resolve_observed_at(&records, None);
        let unseeded = adapter::forward_fill_observed_at(&records);
        assert_eq!(seeded, unseeded);
    }

    // -- Codex cli_version threading -----------------------------------------

    fn codex_record(seq: i64, schema_marker: Option<&str>) -> VerbatimRecord {
        VerbatimRecord {
            key: RecordKey {
                source: Source::Codex,
                session_id: "s".to_string(),
                stream_id: "rollout".to_string(),
                seq,
            },
            stream_kind: StreamKind::Subagent,
            raw: "{}".to_string(),
            payload: None,
            timestamp: None,
            origin: RecordOrigin {
                machine: "m".to_string(),
                path: "p".to_string(),
                record_type: None,
                schema_marker: schema_marker.map(str::to_string),
            },
        }
    }

    #[test]
    fn seed_version_fills_records_before_any_session_meta_in_this_batch() {
        let mut records = [codex_record(0, None), codex_record(1, None)];
        let last = thread_codex_cli_version(&mut records, Some("0.153.4".to_string()));
        assert_eq!(records[0].origin.schema_marker.as_deref(), Some("0.153.4"));
        assert_eq!(records[1].origin.schema_marker.as_deref(), Some("0.153.4"));
        assert_eq!(last.as_deref(), Some("0.153.4"));
    }

    #[test]
    fn a_new_session_meta_in_this_batch_overrides_the_seed_for_subsequent_records() {
        let mut records = [
            codex_record(0, None),
            codex_record(1, Some("0.160.0")),
            codex_record(2, None),
        ];
        let last = thread_codex_cli_version(&mut records, Some("0.153.4".to_string()));
        assert_eq!(records[0].origin.schema_marker.as_deref(), Some("0.153.4"));
        assert_eq!(records[1].origin.schema_marker.as_deref(), Some("0.160.0"));
        assert_eq!(records[2].origin.schema_marker.as_deref(), Some("0.160.0"));
        assert_eq!(last.as_deref(), Some("0.160.0"));
    }

    #[test]
    fn no_seed_and_no_session_meta_leaves_schema_marker_none() {
        let mut records = [codex_record(0, None)];
        let last = thread_codex_cli_version(&mut records, None);
        assert_eq!(records[0].origin.schema_marker, None);
        assert_eq!(last, None);
    }

    // -- NUL sanitization (measured live, 2026-09-20) ------------------------

    #[test]
    fn strip_nul_bytes_removes_only_nul_and_leaves_everything_else() {
        let input = "before\0after\0\0end";
        assert_eq!(strip_nul_bytes(input), "beforeafterend");
    }

    #[test]
    fn strip_nul_bytes_is_a_no_op_when_there_is_nothing_to_strip() {
        let input = "perfectly ordinary transcript line";
        assert_eq!(strip_nul_bytes(input), input);
    }

    #[test]
    fn sanitize_json_value_strips_nul_from_nested_strings_only() {
        let mut value = serde_json::json!({
            "message": {"content": "hello\0world"},
            "count": 3,
            "tags": ["a\0", "b"],
            "ok": true,
            "nothing": null,
        });
        sanitize_json_value(&mut value);
        assert_eq!(value["message"]["content"], "helloworld");
        assert_eq!(value["tags"][0], "a");
        assert_eq!(value["tags"][1], "b");
        assert_eq!(value["count"], 3);
        assert_eq!(value["ok"], true);
        assert!(value["nothing"].is_null());
    }

    #[test]
    fn to_active_model_sanitizes_both_raw_and_payload() {
        let mut record = record(0, Some(ts("2026-09-20T00:00:00Z")));
        record.raw = "line with\0a nul".to_string();
        record.payload = Some(serde_json::json!({"content": "text\0here"}));

        let model = to_active_model(&record, ts("2026-09-20T00:00:00Z"), false, Utc::now());
        let raw = match model.raw {
            sea_orm::ActiveValue::Set(v) => v,
            _ => panic!("expected Set"),
        };
        let payload = match model.payload {
            sea_orm::ActiveValue::Set(v) => v,
            _ => panic!("expected Set"),
        };
        assert_eq!(raw, "line witha nul");
        assert_eq!(payload.unwrap()["content"], "texthere");
    }

    // -- Summary --------------------------------------------------------------

    #[test]
    fn summary_log_line_matches_the_contract_format() {
        let summary = Summary {
            streams: 86,
            records_read: 66_084,
            inserted: 66_000,
            deduped: 50,
            quarantined: 34,
            failed_streams: 2,
        };
        assert_eq!(
            summary.log_line(),
            "harvest complete: streams=86 records_read=66084 inserted=66000 deduped=50 quarantined=34 failed_streams=2"
        );
    }

    // -- config ---------------------------------------------------------------

    // Both exercise the pure `_with_xdg` core directly rather than mutating
    // the process environment, which `cargo test`'s parallel-by-default
    // execution makes unsafe to share across tests in this binary.

    #[test]
    fn default_state_dir_prefers_xdg_state_home_when_set() {
        let home = PathBuf::from("/home/x");
        let result = default_state_dir_with_xdg(&home, Some("/custom/state".to_string()));
        assert_eq!(result, PathBuf::from("/custom/state/akh-transcripts"));
    }

    #[test]
    fn default_state_dir_falls_back_to_dot_local_state_under_home() {
        let home = PathBuf::from("/home/example");
        let result = default_state_dir_with_xdg(&home, None);
        assert_eq!(result, PathBuf::from("/home/example/.local/state/akh-transcripts"));
    }
}
