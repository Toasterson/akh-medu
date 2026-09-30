//! OpenCode adapter: reads `opencode.db`'s **`event` table only**.
//!
//! `source-format-stability.md` measured why: 38 migrations touch this
//! database in about 146 days, ~24 of them reshaping `session`/`message`/
//! `part` directly, while exactly **one** has ever touched `event` — which
//! upstream is already migrating *toward* (`session_v2`/`session_message`
//! are superseding the tables this adapter deliberately avoids). And
//! `account`/`credential` hold live OAuth tokens: this adapter must never
//! name them in a query — a test asserts every query constant below against
//! that forbidden-table list.
//!
//! Identity, per `session-identity.md`:
//!
//! | | value |
//! |---|---|
//! | `session_id` | `event.aggregate_id` |
//! | `stream_id` | `"events"` — one stream per session |
//! | `seq` | `event.seq` (already dense and monotonic per aggregate; no synthetic seq needed) |
//!
//! The "offset" `read_stream` takes and returns is the **next unseen** `seq`
//! for that aggregate (0 for a fresh stream) — deliberately the same
//! convention as the JSONL sources' byte offset (0 = nothing consumed yet,
//! the value returned is where the next read resumes), rather than "the last
//! seq seen", which would collide with the legitimate `seq = 0` on a fresh
//! stream and silently drop its first record. OpenCode's own
//! `event_sequence` table already holds an equivalent high-water mark, which
//! is why no extra synthetic watermark unit is needed here.

use std::path::Path;

use rusqlite::{Connection, OpenFlags};
use serde_json::{Value, json};

use super::StreamRef;
use crate::error::{TranscriptError, TranscriptResult};
use crate::model::{RecordKey, RecordOrigin, Source, StreamKind, VerbatimRecord};

const DISCOVER_QUERY: &str = "SELECT DISTINCT aggregate_id FROM event";
// `aggregate_id` is not selected: it is always `stream.session_id`, already
// pinned by the WHERE clause, so re-reading it per row would only be an
// extra argument for `build_record` to carry with no new information.
const READ_QUERY: &str =
    "SELECT id, seq, type, data FROM event WHERE aggregate_id = ?1 AND seq >= ?2 ORDER BY seq ASC";

fn open_read_only(db_path: &Path) -> TranscriptResult<Connection> {
    Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| TranscriptError::adapter_opencode_open(db_path, e))
}

/// Discover every session (`aggregate_id`) present in the `event` table.
/// `root` is the path to `opencode.db`.
pub fn discover(root: &Path) -> TranscriptResult<Vec<StreamRef>> {
    if !root.exists() {
        return Err(TranscriptError::adapter_root_missing(root));
    }
    let conn = open_read_only(root)?;
    let mut stmt = conn
        .prepare(DISCOVER_QUERY)
        .map_err(|e| TranscriptError::adapter_opencode_query(DISCOVER_QUERY, e))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| TranscriptError::adapter_opencode_query(DISCOVER_QUERY, e))?;

    let mut out = Vec::new();
    for row in rows {
        let aggregate_id =
            row.map_err(|e| TranscriptError::adapter_opencode_query(DISCOVER_QUERY, e))?;
        out.push(StreamRef {
            source: Source::Opencode,
            session_id: aggregate_id,
            stream_id: "events".to_string(),
            stream_kind: StreamKind::EventLog,
            path: root.to_path_buf(),
            file_id: None,
        });
    }
    Ok(out)
}

/// Read new events for `stream`'s aggregate, starting at `from_offset` (the
/// next unseen `seq`). Returns the offset to resume from next time —
/// `from_offset` unchanged if nothing new landed.
pub fn read_stream(
    stream: &StreamRef,
    from_offset: u64,
    machine: &str,
) -> TranscriptResult<(Vec<VerbatimRecord>, u64)> {
    let conn = open_read_only(&stream.path)?;
    let mut stmt = conn
        .prepare(READ_QUERY)
        .map_err(|e| TranscriptError::adapter_opencode_query(READ_QUERY, e))?;

    let from_seq = i64::try_from(from_offset).unwrap_or(i64::MAX);
    let path_str = stream.path.display().to_string();

    let rows = stmt
        .query_map(
            rusqlite::params![stream.session_id, from_seq],
            |row| -> rusqlite::Result<(String, i64, String, String)> {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            },
        )
        .map_err(|e| TranscriptError::adapter_opencode_query(READ_QUERY, e))?;

    let mut records = Vec::new();
    let mut next_offset = from_offset;
    for row in rows {
        let (id, seq, event_type, data) =
            row.map_err(|e| TranscriptError::adapter_opencode_query(READ_QUERY, e))?;
        records.push(build_record(
            stream,
            seq,
            &id,
            &event_type,
            &data,
            machine,
            &path_str,
        ));
        // Next unseen seq is one past the highest we've read.
        next_offset = next_offset.max(u64::try_from(seq).unwrap_or(0) + 1);
    }

    Ok((records, next_offset))
}

/// `event.type` embeds a schema-version suffix (`message.updated.1`). Split
/// it into a base type family and the numeric marker so drift detection can
/// track "highest known suffix per base type" per
/// `source-format-stability.md`'s alarm design.
fn split_type_and_marker(event_type: &str) -> (String, Option<String>) {
    if let Some((base, suffix)) = event_type.rsplit_once('.')
        && !suffix.is_empty()
        && suffix.chars().all(|c| c.is_ascii_digit())
    {
        return (base.to_string(), Some(suffix.to_string()));
    }
    (event_type.to_string(), None)
}

fn build_record(
    stream: &StreamRef,
    seq: i64,
    id: &str,
    event_type: &str,
    data: &str,
    machine: &str,
    path: &str,
) -> VerbatimRecord {
    let data_value: Value =
        serde_json::from_str(data).unwrap_or_else(|_| Value::String(data.to_string()));
    let row = json!({
        "id": id,
        "aggregate_id": stream.session_id,
        "seq": seq,
        "type": event_type,
        "data": data_value,
    });
    let raw = row.to_string();
    let (record_type, schema_marker) = split_type_and_marker(event_type);

    VerbatimRecord {
        key: RecordKey {
            source: Source::Opencode,
            session_id: stream.session_id.clone(),
            stream_id: stream.stream_id.clone(),
            seq,
        },
        stream_kind: stream.stream_kind,
        raw,
        payload: Some(row),
        // The `event` table carries no timestamp column of its own; the
        // per-type embedded times inside `data` (e.g. `time.created`) are
        // not uniform across event types, so extracting one is a
        // normalization concern, not an adapter one. Every OpenCode record
        // therefore relies entirely on forward-fill from a neighbouring
        // source, or is quarantined — see `forward_fill_observed_at`.
        timestamp: None,
        origin: RecordOrigin {
            machine: machine.to_string(),
            path: path.to_string(),
            record_type: Some(record_type),
            schema_marker,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::tempdir;

    fn seed_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE event (id TEXT, aggregate_id TEXT, seq INTEGER, type TEXT, data TEXT);
             CREATE TABLE message (id TEXT);
             CREATE TABLE part (id TEXT);
             CREATE TABLE account (id TEXT, token TEXT);
             CREATE TABLE credential (id TEXT, secret TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO event (id, aggregate_id, seq, type, data) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                "evt_1",
                "ses_abc",
                0,
                "session.created.1",
                "{\"sessionID\":\"ses_abc\"}"
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO event (id, aggregate_id, seq, type, data) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                "evt_2",
                "ses_abc",
                1,
                "message.updated.1",
                "{\"sessionID\":\"ses_abc\"}"
            ],
        )
        .unwrap();
        // A poison row in a forbidden table — discovery/read must never
        // touch it, which this test would otherwise never notice.
        conn.execute(
            "INSERT INTO account (id, token) VALUES ('acc_1', 'sk-should-never-be-read')",
            [],
        )
        .unwrap();
    }

    #[test]
    fn discover_finds_one_stream_per_aggregate() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("opencode.db");
        seed_db(&db_path);

        let streams = discover(&db_path).unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].session_id, "ses_abc");
        assert_eq!(streams[0].stream_id, "events");
        assert_eq!(streams[0].stream_kind, StreamKind::EventLog);
    }

    #[test]
    fn read_stream_splits_type_and_schema_marker() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("opencode.db");
        seed_db(&db_path);

        let streams = discover(&db_path).unwrap();
        let (records, next_offset) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].origin.record_type.as_deref(),
            Some("session.created")
        );
        assert_eq!(records[0].origin.schema_marker.as_deref(), Some("1"));
        assert_eq!(
            records[1].origin.record_type.as_deref(),
            Some("message.updated")
        );
        assert_eq!(next_offset, 2);
    }

    #[test]
    fn resume_from_seq_yields_only_new_events() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("opencode.db");
        seed_db(&db_path);

        let streams = discover(&db_path).unwrap();
        let (first_pass, next_offset) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(first_pass.len(), 2);
        assert_eq!(next_offset, 2);

        let (second_pass, next_offset_again) = read_stream(&streams[0], next_offset, "m").unwrap();
        assert!(second_pass.is_empty());
        assert_eq!(next_offset_again, next_offset);
    }

    /// The sharpest guardrail in this adapter: it must never be possible for
    /// a query to name `account` or `credential` (both hold live OAuth
    /// tokens), nor the stale `message`/`part` tables.
    #[test]
    fn queries_never_name_forbidden_tables() {
        // `account`/`credential` hold live OAuth tokens; `message`/`part` are
        // the stale tables `source-format-stability.md` found upstream is
        // already migrating away from. This adapter must never name any of
        // them in a query.
        const FORBIDDEN_TABLE_NAMES: &[&str] = &["account", "credential", "message", "part"];
        for query in [DISCOVER_QUERY, READ_QUERY] {
            let lowered = query.to_lowercase();
            for forbidden in FORBIDDEN_TABLE_NAMES {
                assert!(
                    !lowered.contains(forbidden),
                    "query {query:?} must never name table {forbidden:?}"
                );
            }
        }
    }
}
