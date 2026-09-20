//! Anti-join stream discovery, per-record normalization (including the
//! poison-record `catch_unwind` boundary), session rollups, and
//! turn-boundary chunking — everything one derivation pass does except the
//! embedding phase ([`super::embed`]).
//!
//! Discovery and re-derivation vocabulary: [derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md).
//! Disposition/`subseq` schema: `m20260920_000001_events_subseq_disposition`
//! and `crate::entity::event`.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use chrono::{DateTime, Utc};
use sea_orm::sea_query::OnConflict;
use sea_orm::{
    ActiveValue::{NotSet, Set},
    ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection, DbErr, EntityTrait,
    QueryFilter, QueryOrder, QuerySelect, Statement,
};

use crate::chunk::chunk_turns;
use crate::entity::{chunk as chunk_entity, event, raw_record, session};
use crate::error::{TranscriptError, TranscriptResult};
use crate::model::{EventKind, RecordKey, RecordOrigin, Source, StreamKind, VerbatimRecord};
use crate::normalize::{self, NormalizedEvent, Outcome, SecretAction, SecretFlag};
use crate::render;

/// `th_events.disposition` values — the closed set the migration's `CHECK`
/// constraint enforces.
pub const DISPOSITION_EVENT: &str = "event";
pub const DISPOSITION_NOISE: &str = "noise";
pub const DISPOSITION_UNDETERMINED: &str = "undetermined";
pub const DISPOSITION_FAILED: &str = "failed";

/// Per-stream totals, folded into [`super::PassSummary`] by the caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    pub records: usize,
    pub events: usize,
    pub noise: usize,
    pub undetermined: usize,
    pub failed: usize,
    pub chunks: usize,
}

// --- discovery ---------------------------------------------------------

/// `SELECT DISTINCT source, session_id, stream_id FROM th_raw_records r
/// WHERE NOT EXISTS (a matching th_events row at subseq=0)`. Pushed
/// entirely into SQL, including `LIMIT` — the derivation-worker ticket's
/// explicit warning against `embed_pending_batch_impl`'s `.all(db).take(n)`
/// mistake, which loads every pending row before truncating.
///
/// `e.subseq = 0` is enough to test "this raw record has been derived at
/// all", regardless of how many events it produced: every record writes at
/// least one row (`subseq = 0`, whether that's its first event or its sole
/// noise/undetermined/failed placeholder), so the anti-join never needs to
/// enumerate every possible `subseq`.
fn pending_streams_sql() -> &'static str {
    "SELECT DISTINCT r.source, r.session_id, r.stream_id \
     FROM th_raw_records r \
     WHERE NOT EXISTS ( \
         SELECT 1 FROM th_events e \
         WHERE e.derivation_version = $1 \
           AND e.source = r.source \
           AND e.session_id = r.session_id \
           AND e.stream_id = r.stream_id \
           AND e.seq = r.seq \
           AND e.subseq = 0 \
     ) \
     LIMIT $2"
}

/// Run [`pending_streams_sql`] and return up to `limit` `(source,
/// session_id, stream_id)` triples with at least one un-derived raw record.
pub async fn find_pending_streams(
    db: &DatabaseConnection,
    version: i32,
    limit: u64,
) -> TranscriptResult<Vec<(String, String, String)>> {
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        pending_streams_sql(),
        [version.into(), (limit as i64).into()],
    );
    let rows = db.query_all(stmt).await.map_err(TranscriptError::from)?;
    rows.into_iter()
        .map(|row| {
            let source: String = row.try_get("", "source").map_err(TranscriptError::from)?;
            let session_id: String = row
                .try_get("", "session_id")
                .map_err(TranscriptError::from)?;
            let stream_id: String = row
                .try_get("", "stream_id")
                .map_err(TranscriptError::from)?;
            Ok((source, session_id, stream_id))
        })
        .collect()
}

// --- per-stream processing ----------------------------------------------

/// Load a stream's raw rows, normalize whichever ones have no `th_events`
/// row yet at `version`, write every outcome (never dropping a record —
/// derivation-worker ticket §2), bump drift counters, and re-chunk the
/// stream's prose events.
pub async fn process_stream(
    db: &DatabaseConnection,
    version: i32,
    source: &str,
    session_id: &str,
    stream_id: &str,
) -> TranscriptResult<StreamStats> {
    let mut stats = StreamStats::default();

    let raw_rows = raw_record::Entity::find()
        .filter(raw_record::Column::Source.eq(source))
        .filter(raw_record::Column::SessionId.eq(session_id))
        .filter(raw_record::Column::StreamId.eq(stream_id))
        .order_by_asc(raw_record::Column::Seq)
        .all(db)
        .await
        .map_err(TranscriptError::from)?;

    let existing_seqs: HashSet<i64> = event::Entity::find()
        .select_only()
        .column(event::Column::Seq)
        .filter(event::Column::DerivationVersion.eq(version))
        .filter(event::Column::Source.eq(source))
        .filter(event::Column::SessionId.eq(session_id))
        .filter(event::Column::StreamId.eq(stream_id))
        .filter(event::Column::Subseq.eq(0))
        .into_tuple::<i64>()
        .all(db)
        .await
        .map_err(TranscriptError::from)?
        .into_iter()
        .collect();

    for row in &raw_rows {
        if existing_seqs.contains(&row.seq) {
            continue; // already derived at this version
        }
        stats.records += 1;

        let derived = derive_record(version, row);
        insert_event_rows(db, derived.rows).await?;

        match derived.stat {
            RecordStat::Events { count, degraded } => {
                stats.events += count;
                if degraded {
                    bump_drift_counter(db, row, DriftKind::Degraded).await?;
                }
            }
            // Noise is a deliberate classification, not drift — nothing to
            // count here per the normal-form ticket's distinction between
            // "known to be noise" and "undetermined".
            RecordStat::Noise => stats.noise += 1,
            RecordStat::Undetermined => {
                stats.undetermined += 1;
                bump_drift_counter(db, row, DriftKind::Unknown).await?;
            }
            RecordStat::Failed => {
                stats.failed += 1;
                bump_drift_counter(db, row, DriftKind::Degraded).await?;
            }
        }
    }

    stats.chunks = chunk_stream(db, version, source, session_id, stream_id).await?;

    Ok(stats)
}

async fn insert_event_rows(
    db: &DatabaseConnection,
    rows: Vec<event::ActiveModel>,
) -> TranscriptResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let result = event::Entity::insert_many(rows)
        .on_conflict(
            OnConflict::columns([
                event::Column::DerivationVersion,
                event::Column::Source,
                event::Column::SessionId,
                event::Column::StreamId,
                event::Column::Seq,
                event::Column::Subseq,
            ])
            .do_nothing()
            .to_owned(),
        )
        .exec(db)
        .await;
    match result {
        Ok(_) => Ok(()),
        // ON CONFLICT DO NOTHING colliding with every row in the batch
        // reports zero affected rows this way (same shape as
        // `tests/live_pg.rs`'s single-row case) — that IS the no-op
        // success case, e.g. re-running a pass that already wrote these
        // rows.
        Err(DbErr::RecordNotInserted) => Ok(()),
        Err(e) => Err(TranscriptError::from(e)),
    }
}

// --- per-record derivation ------------------------------------------------

struct DerivedRecord {
    rows: Vec<event::ActiveModel>,
    stat: RecordStat,
}

enum RecordStat {
    Events { count: usize, degraded: bool },
    Noise,
    Undetermined,
    Failed,
}

enum DriftKind {
    Unknown,
    Degraded,
}

/// Turn one raw row into the `th_events` row(s) it produces, and the stat
/// that row should count toward. Never returns zero rows: every arm of
/// [`crate::normalize::Outcome`], and a normalizer panic caught via
/// `catch_unwind`, writes at least a placeholder row — the derivation
/// worker's fix for the poison-record failure mode (a record with no row
/// satisfying the anti-join re-pends on every tick, forever, silently).
fn derive_record(version: i32, row: &raw_record::Model) -> DerivedRecord {
    let Some(source) = source_from_str(&row.source) else {
        let reason = format!(
            "th_raw_records.source {:?} is not one of claude_code|codex|opencode \
             (a value this worker itself never writes; check for a schema or \
             harvester regression)",
            row.source
        );
        return DerivedRecord {
            rows: vec![placeholder_row(
                version,
                row,
                DISPOSITION_FAILED,
                Some(reason),
            )],
            stat: RecordStat::Failed,
        };
    };

    let record = verbatim_record_from_raw(row, source);
    let observed_at = row.observed_at;

    // `normalize` is documented as never surfacing a hard error — shape
    // problems fold into `Outcome::Degraded` — but a poison record must not
    // be allowed to take the whole pass down if that contract is ever
    // violated by a bug, so the call itself is caught.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        normalize::normalize(&record, observed_at)
    }));

    match outcome {
        Ok(Outcome::Event(ev)) | Ok(Outcome::Degraded(ev)) => {
            let degraded = ev.degraded;
            let subseq = ev.subseq.unwrap_or(0) as i32;
            DerivedRecord {
                rows: vec![event_active_model(version, &ev, subseq, DISPOSITION_EVENT)],
                stat: RecordStat::Events { count: 1, degraded },
            }
        }
        Ok(Outcome::Events(evs)) => {
            let count = evs.len();
            let degraded = evs.iter().any(|ev| ev.degraded);
            let rows = evs
                .iter()
                .enumerate()
                .map(|(idx, ev)| {
                    let subseq = ev.subseq.unwrap_or(idx as u32) as i32;
                    event_active_model(version, ev, subseq, DISPOSITION_EVENT)
                })
                .collect();
            DerivedRecord {
                rows,
                stat: RecordStat::Events { count, degraded },
            }
        }
        Ok(Outcome::Noise { reason }) => DerivedRecord {
            rows: vec![placeholder_row(
                version,
                row,
                DISPOSITION_NOISE,
                Some(reason),
            )],
            stat: RecordStat::Noise,
        },
        Ok(Outcome::Undetermined { record_type }) => DerivedRecord {
            rows: vec![placeholder_row(
                version,
                row,
                DISPOSITION_UNDETERMINED,
                Some(format!("record_type={record_type}")),
            )],
            stat: RecordStat::Undetermined,
        },
        Err(payload) => {
            let reason = describe_panic(payload);
            DerivedRecord {
                rows: vec![placeholder_row(
                    version,
                    row,
                    DISPOSITION_FAILED,
                    Some(reason),
                )],
                stat: RecordStat::Failed,
            }
        }
    }
}

/// Render a `catch_unwind` panic payload as a stored reason. Prefers a
/// [`crate::normalize::DerivationError`] if the panic happened to carry one
/// (via [`TranscriptError`]'s `From` impl, so the message is the same
/// miette-formatted text a caller reading `TranscriptError::Derive` would
/// see), else the conventional `&str`/`String` panic message shapes, else a
/// generic fallback — this is inherently best-effort since panic payloads
/// are `Any`.
fn describe_panic(payload: Box<dyn std::any::Any + Send>) -> String {
    let payload = match payload.downcast::<normalize::DerivationError>() {
        Ok(err) => return TranscriptError::from(*err).to_string(),
        Err(payload) => payload,
    };
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "normalizer panicked with a non-string, non-DerivationError payload".to_string()
}

fn verbatim_record_from_raw(row: &raw_record::Model, source: Source) -> VerbatimRecord {
    VerbatimRecord {
        key: RecordKey {
            source,
            session_id: row.session_id.clone(),
            stream_id: row.stream_id.clone(),
            seq: row.seq,
        },
        stream_kind: stream_kind_from_str(&row.stream_kind),
        raw: row.raw.clone(),
        payload: row.payload.clone(),
        // `observed_at` is forward-filled by the harvester; only surface it
        // as the record's "own" timestamp when it was not synthesised —
        // `VerbatimRecord::timestamp`'s contract is "never synthesised".
        // `normalize` itself is passed `observed_at` separately and does
        // not read this field today, but a faithful reconstruction should
        // not claim more than the raw row actually recorded.
        timestamp: if row.observed_at_synthetic {
            None
        } else {
            Some(row.observed_at)
        },
        origin: RecordOrigin {
            machine: row.machine.clone(),
            path: row.origin_path.clone(),
            record_type: row.record_type.clone(),
            schema_marker: row.schema_marker.clone(),
        },
    }
}

fn source_from_str(s: &str) -> Option<Source> {
    match s {
        "claude_code" => Some(Source::ClaudeCode),
        "codex" => Some(Source::Codex),
        "opencode" => Some(Source::Opencode),
        _ => None,
    }
}

/// Inverse of the harvester's `StreamKind` serialization. Falls back to
/// `Main` for anything unrecognized, logging loudly — this is our own
/// column drifting from our own contract, which `normalize` does not
/// actually consume (see `verbatim_record_from_raw`'s comment), so a
/// fallback rather than a hard failure is appropriate.
fn stream_kind_from_str(s: &str) -> StreamKind {
    match s {
        "main" => StreamKind::Main,
        "subagent" => StreamKind::Subagent,
        "workflow_journal" => StreamKind::WorkflowJournal,
        "event_log" => StreamKind::EventLog,
        other => {
            tracing::warn!(
                stream_kind = other,
                "unrecognized th_raw_records.stream_kind; defaulting to Main \
                 (normalize does not read this field, so this only affects \
                 provenance)"
            );
            StreamKind::Main
        }
    }
}

/// The DB string for one of the ten closed event kinds — the `th_events
/// .kind` column's values. Kept as its own exhaustive match, rather than
/// leaning on `EventKind`'s `#[serde(rename_all = "snake_case")]` derive
/// via `serde_json`, so a new variant added to `crate::model::EventKind`
/// fails this file to compile instead of silently serializing to whatever
/// serde happens to produce.
fn kind_str(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Prompt => "prompt",
        EventKind::Reply => "reply",
        EventKind::Thought => "thought",
        EventKind::ToolCall => "tool_call",
        EventKind::ToolResult => "tool_result",
        EventKind::Context => "context",
        EventKind::SessionMeta => "session_meta",
        EventKind::Lifecycle => "lifecycle",
        EventKind::Error => "error",
        EventKind::Telemetry => "telemetry",
    }
}

fn kind_from_str(s: &str) -> Option<EventKind> {
    Some(match s {
        "prompt" => EventKind::Prompt,
        "reply" => EventKind::Reply,
        "thought" => EventKind::Thought,
        "tool_call" => EventKind::ToolCall,
        "tool_result" => EventKind::ToolResult,
        "context" => EventKind::Context,
        "session_meta" => EventKind::SessionMeta,
        "lifecycle" => EventKind::Lifecycle,
        "error" => EventKind::Error,
        "telemetry" => EventKind::Telemetry,
        _ => return None,
    })
}

fn secret_flags_json(flags: &[SecretFlag]) -> serde_json::Value {
    serde_json::Value::Array(
        flags
            .iter()
            .map(|flag| {
                serde_json::json!({
                    "pattern": flag.pattern,
                    "start": flag.start,
                    "end": flag.end,
                    "action": match flag.action {
                        SecretAction::Mask => "mask",
                        SecretAction::Withhold => "withhold",
                    },
                })
            })
            .collect(),
    )
}

/// `text` is stored **unmasked** here — the read surface is the outward
/// boundary that masks, per the plan's increment-5 brief — while
/// `secret_flags` records what `redact::scan` found so the read surface
/// does not have to re-scan on every read.
fn event_active_model(
    version: i32,
    event: &NormalizedEvent,
    subseq: i32,
    disposition: &str,
) -> event::ActiveModel {
    let secret_flags = event.text.as_deref().and_then(|text| {
        let flags = crate::redact::scan(text);
        if flags.is_empty() {
            None
        } else {
            Some(secret_flags_json(&flags))
        }
    });
    let render_text = render::render(event).to_string();

    event::ActiveModel {
        derivation_version: Set(version),
        source: Set(event.key.source.as_str().to_string()),
        session_id: Set(event.key.session_id.clone()),
        stream_id: Set(event.key.stream_id.clone()),
        seq: Set(event.key.seq),
        subseq: Set(subseq),
        observed_at: Set(event.observed_at),
        kind: Set(Some(kind_str(event.kind).to_string())),
        disposition: Set(disposition.to_string()),
        subkind: Set(event.subkind.clone()),
        role: Set(event.role.clone()),
        text: Set(event.text.clone()),
        tool_name: Set(event.tool_name.clone()),
        tool_call_id: Set(event.tool_call_id.clone()),
        agent_id: Set(event.agent_id.clone()),
        parent_agent_id: Set(event.parent_agent_id.clone()),
        render: Set(Some(render_text)),
        degraded: Set(event.degraded),
        degraded_reason: Set(event.degraded_reason.clone()),
        secret_flags: Set(secret_flags),
        repo: Set(event.repo.clone()),
        workspace: Set(event.workspace.clone()),
        branch: Set(event.branch.clone()),
        derived_at: Set(Utc::now()),
    }
}

/// A `th_events` row for a record that produced no event at all
/// (`disposition` is `noise`, `undetermined`, or `failed`) or that could not
/// even be classified enough to attempt normalization (`failed`, source
/// column drift). `reason` is stored in `degraded_reason`, repurposed
/// across every non-`event` disposition — see `entity::event`'s doc comment
/// on that column.
fn placeholder_row(
    version: i32,
    row: &raw_record::Model,
    disposition: &str,
    reason: Option<String>,
) -> event::ActiveModel {
    event::ActiveModel {
        derivation_version: Set(version),
        source: Set(row.source.clone()),
        session_id: Set(row.session_id.clone()),
        stream_id: Set(row.stream_id.clone()),
        seq: Set(row.seq),
        subseq: Set(0),
        observed_at: Set(row.observed_at),
        kind: Set(None),
        disposition: Set(disposition.to_string()),
        subkind: Set(None),
        role: Set(None),
        text: Set(None),
        tool_name: Set(None),
        tool_call_id: Set(None),
        agent_id: Set(None),
        parent_agent_id: Set(None),
        render: Set(None),
        degraded: Set(disposition == DISPOSITION_FAILED),
        degraded_reason: Set(reason),
        secret_flags: Set(None),
        repo: Set(None),
        workspace: Set(None),
        branch: Set(None),
        derived_at: Set(Utc::now()),
    }
}

async fn bump_drift_counter(
    db: &DatabaseConnection,
    row: &raw_record::Model,
    kind: DriftKind,
) -> TranscriptResult<()> {
    let record_type = row
        .record_type
        .clone()
        .unwrap_or_else(|| "<unknown>".to_string());
    let schema_marker = row.schema_marker.clone().unwrap_or_default();
    let (unknown_inc, degraded_inc): (i64, i64) = match kind {
        DriftKind::Unknown => (1, 0),
        DriftKind::Degraded => (0, 1),
    };

    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO th_drift_counters \
             (source, record_type, schema_marker, unknown_count, degraded_count, last_seen) \
         VALUES ($1, $2, $3, $4, $5, $6) \
         ON CONFLICT (source, record_type, schema_marker) DO UPDATE SET \
             unknown_count = th_drift_counters.unknown_count + EXCLUDED.unknown_count, \
             degraded_count = th_drift_counters.degraded_count + EXCLUDED.degraded_count, \
             last_seen = EXCLUDED.last_seen",
        [
            row.source.clone().into(),
            record_type.into(),
            schema_marker.into(),
            unknown_inc.into(),
            degraded_inc.into(),
            row.observed_at.into(),
        ],
    );
    db.execute(stmt).await.map_err(TranscriptError::from)?;
    Ok(())
}

// --- chunking -------------------------------------------------------------

/// Re-chunk the stream's *entire* `prompt`/`reply` history (not just this
/// pass's newly-derived events) into `th_chunks`, so a stream that receives
/// new raw records across several passes still gets correct turn-boundary
/// grouping rather than one truncated chunk per pass. `ON CONFLICT DO
/// NOTHING` on `idx_th_chunks_identity` makes re-chunking unchanged spans
/// free.
async fn chunk_stream(
    db: &DatabaseConnection,
    version: i32,
    source: &str,
    session_id: &str,
    stream_id: &str,
) -> TranscriptResult<usize> {
    let Some(source_enum) = source_from_str(source) else {
        // Already recorded as a `disposition = 'failed'` row per-record
        // above; nothing left here to chunk.
        return Ok(0);
    };

    let rows = event::Entity::find()
        .filter(event::Column::DerivationVersion.eq(version))
        .filter(event::Column::Source.eq(source))
        .filter(event::Column::SessionId.eq(session_id))
        .filter(event::Column::StreamId.eq(stream_id))
        .filter(event::Column::Disposition.eq(DISPOSITION_EVENT))
        .order_by_asc(event::Column::Seq)
        .order_by_asc(event::Column::Subseq)
        .all(db)
        .await
        .map_err(TranscriptError::from)?;

    let events: Vec<NormalizedEvent> = rows
        .iter()
        .filter_map(|r| event_row_to_normalized(r, source_enum))
        .collect();

    let chunks = chunk_turns(&events, 0);
    if chunks.is_empty() {
        return Ok(0);
    }

    let mut inserted = 0usize;
    for chunk in chunks {
        let model = chunk_entity::ActiveModel {
            id: NotSet,
            derivation_version: Set(version),
            source: Set(chunk.source.as_str().to_string()),
            session_id: Set(chunk.session_id.clone()),
            stream_id: Set(chunk.stream_id.clone()),
            seq_from: Set(chunk.seq_from),
            seq_to: Set(chunk.seq_to),
            text: Set(chunk.text.clone()),
            // `NotSet`, not `Set(None)`: SeaORM sends an explicit `Set(None)`
            // as a typed NULL bind parameter (text), which Postgres rejects
            // for a `vector` column with "column is of type vector but
            // expression is of type text" even though the value is NULL —
            // the same pgvector/sqlx mismatch `entity::chunk`'s own doc
            // comment describes for a *populated* embedding. Omitting the
            // column from the INSERT entirely lets it default to SQL NULL,
            // which has no type to conflict with.
            embedding: NotSet,
            embedded_at: Set(None),
            created_at: Set(Utc::now()),
        };
        let result = chunk_entity::Entity::insert(model)
            .on_conflict(
                OnConflict::columns([
                    chunk_entity::Column::DerivationVersion,
                    chunk_entity::Column::Source,
                    chunk_entity::Column::SessionId,
                    chunk_entity::Column::StreamId,
                    chunk_entity::Column::SeqFrom,
                ])
                .do_nothing()
                .to_owned(),
            )
            .exec(db)
            .await;
        match result {
            Ok(_) => inserted += 1,
            Err(DbErr::RecordNotInserted) => {}
            Err(e) => return Err(TranscriptError::from(e)),
        }
    }
    Ok(inserted)
}

fn event_row_to_normalized(row: &event::Model, source: Source) -> Option<NormalizedEvent> {
    let kind = kind_from_str(row.kind.as_deref()?)?;
    let mut ev = NormalizedEvent::new(
        RecordKey {
            source,
            session_id: row.session_id.clone(),
            stream_id: row.stream_id.clone(),
            seq: row.seq,
        },
        row.observed_at,
        kind,
    );
    ev.text = row.text.clone();
    Some(ev)
}

// --- session rollup ---------------------------------------------------

/// Upsert `th_sessions` for `(source, session_id)`, rolled up across *every*
/// stream of that session (a session can have a main stream plus several
/// subagent streams) — not just the one stream this pass just derived.
pub async fn upsert_session(
    db: &DatabaseConnection,
    version: i32,
    source: &str,
    session_id: &str,
) -> TranscriptResult<()> {
    let title_row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT text FROM th_events \
             WHERE derivation_version = $1 AND source = $2 AND session_id = $3 \
               AND disposition = 'event' AND kind = 'session_meta' \
               AND subkind = 'ai-title' AND text IS NOT NULL \
             ORDER BY observed_at DESC LIMIT 1",
            [version.into(), source.into(), session_id.into()],
        ))
        .await
        .map_err(TranscriptError::from)?;
    let title: Option<String> = match &title_row {
        Some(r) => r.try_get("", "text").map_err(TranscriptError::from)?,
        None => None,
    };

    let ctx_row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT repo, workspace, branch FROM th_events \
             WHERE derivation_version = $1 AND source = $2 AND session_id = $3 \
               AND disposition = 'event' AND repo IS NOT NULL \
             ORDER BY observed_at ASC LIMIT 1",
            [version.into(), source.into(), session_id.into()],
        ))
        .await
        .map_err(TranscriptError::from)?;
    let (repo, workspace, branch): (Option<String>, Option<String>, Option<String>) = match &ctx_row
    {
        Some(r) => (
            r.try_get("", "repo").map_err(TranscriptError::from)?,
            r.try_get("", "workspace").map_err(TranscriptError::from)?,
            r.try_get("", "branch").map_err(TranscriptError::from)?,
        ),
        None => (None, None, None),
    };

    let counts_row = db
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT count(DISTINCT stream_id)::int AS stream_count, \
                    count(*)::bigint AS record_count, \
                    min(observed_at) AS first_seen, max(observed_at) AS last_seen \
             FROM th_raw_records WHERE source = $1 AND session_id = $2",
            [source.into(), session_id.into()],
        ))
        .await
        .map_err(TranscriptError::from)?
        .ok_or_else(|| TranscriptError::Db {
            source: DbErr::Custom(
                "th_raw_records aggregate for a session that owns pending raw rows returned no \
                 row — should be unreachable"
                    .to_string(),
            ),
        })?;

    let stream_count: i32 = counts_row
        .try_get("", "stream_count")
        .map_err(TranscriptError::from)?;
    let record_count: i64 = counts_row
        .try_get("", "record_count")
        .map_err(TranscriptError::from)?;
    let first_seen: DateTime<Utc> = counts_row
        .try_get("", "first_seen")
        .map_err(TranscriptError::from)?;
    let last_seen: DateTime<Utc> = counts_row
        .try_get("", "last_seen")
        .map_err(TranscriptError::from)?;

    let model = session::ActiveModel {
        source: Set(source.to_string()),
        session_id: Set(session_id.to_string()),
        title: Set(title),
        repo: Set(repo),
        workspace: Set(workspace),
        branch: Set(branch),
        first_seen: Set(first_seen),
        last_seen: Set(last_seen),
        stream_count: Set(stream_count),
        record_count: Set(record_count),
        derivation_version: Set(version),
    };

    session::Entity::insert(model)
        .on_conflict(
            OnConflict::columns([session::Column::Source, session::Column::SessionId])
                .update_columns([
                    session::Column::Title,
                    session::Column::Repo,
                    session::Column::Workspace,
                    session::Column::Branch,
                    session::Column::FirstSeen,
                    session::Column::LastSeen,
                    session::Column::StreamCount,
                    session::Column::RecordCount,
                    session::Column::DerivationVersion,
                ])
                .to_owned(),
        )
        .exec(db)
        .await
        .map_err(TranscriptError::from)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    fn raw_row(
        source: &str,
        record_type: &str,
        payload: serde_json::Value,
        seq: i64,
    ) -> raw_record::Model {
        let now = Utc::now();
        raw_record::Model {
            observed_at: now,
            source: source.to_string(),
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            seq,
            stream_kind: "main".to_string(),
            machine: "test-host".to_string(),
            raw: payload.to_string(),
            payload: Some(payload),
            record_type: Some(record_type.to_string()),
            schema_marker: None,
            origin_path: "test.jsonl".to_string(),
            observed_at_synthetic: false,
            harvested_at: now,
        }
    }

    // --- the anti-join SQL --------------------------------------------

    #[test]
    fn pending_streams_sql_is_an_anti_join_with_limit_pushed_into_sql() {
        let sql = pending_streams_sql();
        assert!(sql.contains("NOT EXISTS"), "must be an anti-join: {sql}");
        assert!(
            sql.contains("LIMIT $2"),
            "the batch size must be a bind parameter inside the SQL, not a \
             `.take(n)` in Rust after loading everything: {sql}"
        );
        assert!(sql.contains("e.derivation_version = $1"));
        assert!(sql.contains("e.subseq = 0"));
        assert!(sql.contains("DISTINCT"));
    }

    // --- disposition rules ----------------------------------------------

    #[test]
    fn a_recognised_event_record_derives_to_disposition_event() {
        let row = raw_row(
            "claude_code",
            "user",
            json!({"type": "user", "message": {"role": "user", "content": "hello"}}),
            1,
        );
        let derived = derive_record(1, &row);
        assert!(matches!(
            derived.stat,
            RecordStat::Events {
                count: 1,
                degraded: false
            }
        ));
        assert_eq!(derived.rows.len(), 1);
    }

    #[test]
    fn noise_record_derives_to_disposition_noise_and_is_not_a_drift_signal() {
        let row = raw_row(
            "claude_code",
            "queue-operation",
            json!({"type": "queue-operation"}),
            1,
        );
        let derived = derive_record(1, &row);
        assert!(matches!(derived.stat, RecordStat::Noise));
        assert_eq!(derived.rows.len(), 1);
    }

    #[test]
    fn an_unparseable_record_type_derives_to_disposition_undetermined() {
        let row = raw_row(
            "codex",
            "world_state",
            json!({"type": "world_state", "opaque": true}),
            1,
        );
        let derived = derive_record(1, &row);
        assert!(matches!(derived.stat, RecordStat::Undetermined));
    }

    #[test]
    fn a_row_with_an_unrecognised_source_column_fails_rather_than_panicking() {
        let row = raw_row("not-a-real-source", "user", json!({}), 1);
        let derived = derive_record(1, &row);
        assert!(matches!(derived.stat, RecordStat::Failed));
        assert_eq!(derived.rows.len(), 1);
    }

    // --- codec round-trips used by the write and chunk paths -------------

    #[test]
    fn every_event_kind_round_trips_through_its_db_string() {
        for kind in [
            EventKind::Prompt,
            EventKind::Reply,
            EventKind::Thought,
            EventKind::ToolCall,
            EventKind::ToolResult,
            EventKind::Context,
            EventKind::SessionMeta,
            EventKind::Lifecycle,
            EventKind::Error,
            EventKind::Telemetry,
        ] {
            assert_eq!(kind_from_str(kind_str(kind)), Some(kind));
        }
    }

    #[test]
    fn unrecognised_kind_string_is_none_not_a_guess() {
        assert_eq!(kind_from_str("not-a-kind"), None);
    }

    #[test]
    fn source_from_str_matches_source_as_str_for_every_variant() {
        for source in [Source::ClaudeCode, Source::Codex, Source::Opencode] {
            assert_eq!(source_from_str(source.as_str()), Some(source));
        }
        assert_eq!(source_from_str("not-a-source"), None);
    }

    #[test]
    fn secret_flags_json_carries_pattern_span_and_action() {
        let flags = vec![SecretFlag {
            pattern: "aws_access_key".to_string(),
            start: 3,
            end: 7,
            action: SecretAction::Mask,
        }];
        let value = secret_flags_json(&flags);
        assert_eq!(value[0]["pattern"], "aws_access_key");
        assert_eq!(value[0]["start"], 3);
        assert_eq!(value[0]["end"], 7);
        assert_eq!(value[0]["action"], "mask");
    }

    #[test]
    fn describe_panic_prefers_a_derivation_error_payload() {
        let err = normalize::DerivationError::MissingField {
            record_type: "user".to_string(),
            field: "message",
        };
        let expected = TranscriptError::from(normalize::DerivationError::MissingField {
            record_type: "user".to_string(),
            field: "message",
        })
        .to_string();
        let payload: Box<dyn std::any::Any + Send> = Box::new(err);
        assert_eq!(describe_panic(payload), expected);
    }

    #[test]
    fn describe_panic_falls_back_to_a_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("boom".to_string());
        assert_eq!(describe_panic(payload), "boom");
    }
}
