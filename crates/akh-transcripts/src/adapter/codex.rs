//! Codex adapter: `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`.
//!
//! Identity, per `session-identity.md`:
//!
//! | | value |
//! |---|---|
//! | `session_id` | `session_meta.payload.session_id`, from the **first** `session_meta` record in the file |
//! | `stream_id` | the rollout file's own uuid, from the filename |
//! | `seq` | 0-based line index — **never** the `ordinal` field, which is absent in 22 of 33 local files |
//!
//! `session_meta` re-emits mid-file on forked or resumed sessions, each time
//! embedding the parent's full header inline — measured in 18 of 33 local
//! files, and the filename uuid differs from `session_meta.session_id` in 27
//! of 33. So the filename identifies a *rollout*; the payload identifies the
//! *conversation*. This adapter reads the file once at discovery time to
//! resolve the real session id and never trusts the filename for it.
//!
//! `stream_kind` is documented as always [`StreamKind::Subagent`]: telling a
//! session's original rollout apart from a forked/resumed one is not
//! reliably derivable per-file, since a forked file re-embeds its parent's
//! complete `session_meta` header inline. Rather than guess which rollout is
//! "first", every Codex stream gets the same, honestly-approximate kind —
//! exactly the escape hatch `session-identity.md` allows.

use std::path::Path;
use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use regex::Regex;
use serde_json::Value;

use super::StreamRef;
use super::scan::{self, seq_lines};
use crate::error::{TranscriptError, TranscriptResult};
use crate::model::{RecordKey, RecordOrigin, Source, StreamKind, VerbatimRecord};

fn rollout_uuid_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$",
        )
        .expect("static regex is valid")
    })
}

/// Recursively discover every Codex rollout under `root`
/// (`~/.codex/sessions`).
pub fn discover(root: &Path) -> TranscriptResult<Vec<StreamRef>> {
    if !root.exists() {
        return Err(TranscriptError::adapter_root_missing(root));
    }
    let mut out = Vec::new();
    for path in scan::discover_jsonl_paths(root) {
        let Some(stream_id) = extract_rollout_uuid(&path) else {
            tracing::warn!(
                path = %path.display(),
                "codex: filename did not carry a rollout uuid, skipping"
            );
            continue;
        };
        match find_first_session_id(&path) {
            Ok(Some(session_id)) => {
                out.push(StreamRef {
                    source: Source::Codex,
                    session_id,
                    stream_id,
                    stream_kind: StreamKind::Subagent,
                    file_id: scan::file_id_or_zst(&path),
                    path,
                });
            }
            Ok(None) => {
                tracing::warn!(
                    path = %path.display(),
                    "codex: no session_meta record found, skipping rather than \
                     guessing identity"
                );
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "codex: failed to read file during discovery, skipping"
                );
            }
        }
    }
    Ok(out)
}

fn extract_rollout_uuid(path: &Path) -> Option<String> {
    let filename = path.file_name()?.to_str()?;
    rollout_uuid_pattern()
        .captures(filename)
        .map(|c| c[1].to_string())
}

/// Read the whole file looking for the **first** `session_meta` record and
/// return its `payload.session_id`. Subsequent `session_meta` re-emissions
/// (fork/resume) are deliberately ignored here — the first one governs.
fn find_first_session_id(path: &Path) -> TranscriptResult<Option<String>> {
    let content = scan::read_complete_lines(path, 0)?;
    for line in &content.lines {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) != Some("session_meta") {
            continue;
        }
        if let Some(session_id) = value
            .get("payload")
            .and_then(|p| p.get("session_id"))
            .and_then(Value::as_str)
        {
            return Ok(Some(session_id.to_string()));
        }
    }
    Ok(None)
}

/// For `response_item` records the top-level `type` is uninformative (every
/// model-I/O record shares it); the meaningful kind lives at
/// `payload.type` (`message`, `reasoning`, `function_call`, ...).
fn record_type_of(payload: &Value) -> Option<String> {
    let top = payload.get("type").and_then(Value::as_str)?;
    if top == "response_item" {
        payload
            .get("payload")
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| Some(top.to_string()))
    } else {
        Some(top.to_string())
    }
}

/// Read new, complete lines from `stream`, starting at the byte `from_offset`.
///
/// `cli_version` (the schema marker) is threaded forward from whatever
/// `session_meta` records appear *within this call's batch* — a resumed tail
/// read that starts after the file's last `session_meta` will not see it and
/// leaves `schema_marker` unset for those records. Threading it across
/// separate calls would require persisting the value with the watermark,
/// which is out of scope for a read-only, database-free adapter; increment 3
/// (`akh-harvest`) or the normalizer can carry it further if needed.
pub fn read_stream(
    stream: &StreamRef,
    from_offset: u64,
    machine: &str,
) -> TranscriptResult<(Vec<VerbatimRecord>, u64)> {
    let content = scan::read_complete_lines(&stream.path, from_offset)?;
    let path_str = stream.path.display().to_string();

    let mut cli_version: Option<String> = None;
    let mut records = Vec::with_capacity(content.lines.len());
    for (seq, line) in seq_lines(&content) {
        let payload: Option<Value> = serde_json::from_str(line).ok();
        if let Some(value) = payload.as_ref()
            && value.get("type").and_then(Value::as_str) == Some("session_meta")
            && let Some(cv) = value
                .get("payload")
                .and_then(|p| p.get("cli_version"))
                .and_then(Value::as_str)
        {
            cli_version = Some(cv.to_string());
        }
        records.push(build_record(
            stream,
            seq,
            line,
            payload,
            machine,
            &path_str,
            cli_version.clone(),
        ));
    }

    Ok((records, content.consumed_through))
}

fn build_record(
    stream: &StreamRef,
    seq: i64,
    raw_line: &str,
    payload: Option<Value>,
    machine: &str,
    path: &str,
    schema_marker: Option<String>,
) -> VerbatimRecord {
    let record_type = payload.as_ref().and_then(record_type_of);
    let timestamp = payload
        .as_ref()
        .and_then(|v| v.get("timestamp"))
        .and_then(Value::as_str)
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    VerbatimRecord {
        key: RecordKey {
            source: Source::Codex,
            session_id: stream.session_id.clone(),
            stream_id: stream.stream_id.clone(),
            seq,
        },
        stream_kind: stream.stream_kind,
        raw: raw_line.to_string(),
        payload,
        timestamp,
        origin: RecordOrigin {
            machine: machine.to_string(),
            path: path.to_string(),
            record_type,
            schema_marker,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn rollout_path(root: &Path, filename: &str) -> std::path::PathBuf {
        root.join("2026").join("09").join("15").join(filename)
    }

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// The filename's uuid is a *rollout* id, not the conversation's session
    /// id — measured to differ in 27 of 33 real files.
    #[test]
    fn session_id_comes_from_session_meta_not_filename() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let path = rollout_path(root, filename);
        write(
            &path,
            "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"01a09e99-5594-7a61-aa97-6cf242b8d102\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\",\"cli_version\":\"0.153.4\"}}\n",
        );

        let streams = discover(root).unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].stream_id, "01a0a5dd-a9f6-7091-9926-d286bae5623e");
        assert_eq!(
            streams[0].session_id,
            "01a09e99-5594-7a61-aa97-6cf242b8d102"
        );
        assert_ne!(streams[0].stream_id, streams[0].session_id);
    }

    /// A forked/resumed rollout re-embeds its parent's `session_meta` header
    /// mid-file; the adapter must keep the **first** session_id seen, not the
    /// last.
    #[test]
    fn second_session_meta_does_not_override_the_first_session_id() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let path = rollout_path(root, filename);
        write(
            &path,
            "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"01a09e99-5594-7a61-aa97-6cf242b8d102\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\",\"cli_version\":\"0.153.4\"}}\n\
             {\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"01a09e99-5594-7a61-aa97-6cf242b8d102\",\"id\":\"01a09e99-5594-7a61-aa97-6cf242b8d102\",\"cli_version\":\"0.153.4\"}}\n",
        );

        let streams = discover(root).unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(
            streams[0].session_id,
            "01a09e99-5594-7a61-aa97-6cf242b8d102"
        );

        let (records, _) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .all(|r| r.key.session_id == "01a09e99-5594-7a61-aa97-6cf242b8d102")
        );
    }

    /// Unknown top-level types (e.g. the undetermined `world_state`) are not
    /// an error — they come back as records with a parsed payload so the
    /// caller can count them.
    #[test]
    fn unknown_record_types_are_not_errors() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let path = rollout_path(root, filename);
        write(
            &path,
            "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"s1\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\"}}\n\
             {\"timestamp\":\"2026-09-15T16:19:23.000Z\",\"type\":\"world_state\",\"payload\":{\"opaque\":true}}\n",
        );

        let streams = discover(root).unwrap();
        let (records, _) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[1].origin.record_type.as_deref(),
            Some("world_state")
        );
        assert!(records[1].payload.is_some());
    }

    /// `response_item`'s meaningful type lives at `payload.type`, not the
    /// generic top-level `type` — this is what keeps `message`/`reasoning`/
    /// `function_call` distinguishable for the normalizer.
    #[test]
    fn response_item_record_type_comes_from_nested_payload_type() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let path = rollout_path(root, filename);
        write(
            &path,
            "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"s1\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\"}}\n\
             {\"timestamp\":\"2026-09-15T16:19:23.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\"}}\n",
        );

        let streams = discover(root).unwrap();
        let (records, _) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(records[1].origin.record_type.as_deref(), Some("message"));
    }

    /// Resume-from-offset behaves the same as Claude Code's: only genuinely
    /// new records come back, at the correct continuing seq.
    #[test]
    fn resume_from_offset_yields_only_new_records() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let path = rollout_path(root, filename);
        let line1 = "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"s1\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\"}}\n";
        write(&path, line1);

        let streams = discover(root).unwrap();
        let (first_pass, offset) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(first_pass.len(), 1);

        let line2 = "{\"timestamp\":\"2026-09-15T16:19:23.000Z\",\"type\":\"event_msg\",\"payload\":{\"type\":\"task_started\"}}\n";
        let mut appended = line1.to_string();
        appended.push_str(line2);
        fs::write(&path, appended).unwrap();

        let (second_pass, _) = read_stream(&streams[0], offset, "m").unwrap();
        assert_eq!(second_pass.len(), 1);
        assert_eq!(second_pass[0].key.seq, 1);
    }

    /// `.jsonl.zst` fallback, same contract as Claude Code's.
    #[test]
    fn zst_fallback_is_read_when_plain_file_is_absent() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let filename = "rollout-2026-09-15T18-19-22-01a0a5dd-a9f6-7091-9926-d286bae5623e.jsonl";
        let logical_path = rollout_path(root, filename);
        let zst_path = {
            let mut p = logical_path.clone().into_os_string();
            p.push(".zst");
            std::path::PathBuf::from(p)
        };
        let content = "{\"timestamp\":\"2026-09-15T16:19:22.749Z\",\"type\":\"session_meta\",\"payload\":{\"session_id\":\"s1\",\"id\":\"01a0a5dd-a9f6-7091-9926-d286bae5623e\"}}\n";
        fs::create_dir_all(zst_path.parent().unwrap()).unwrap();
        let compressed = zstd::stream::encode_all(content.as_bytes(), 0).unwrap();
        fs::write(&zst_path, compressed).unwrap();

        let streams = discover(root).unwrap();
        assert_eq!(streams.len(), 1);
        let (records, offset) = read_stream(&streams[0], 0, "m").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(offset, content.len() as u64);
    }
}
