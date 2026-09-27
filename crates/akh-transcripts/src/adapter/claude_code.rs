//! Claude Code adapter: `~/.claude/projects/<slug>/<sessionId>.jsonl`, plus
//! `<slug>/<sessionId>/subagents/agent-<agentId>.jsonl`, plus deeper nesting
//! under `subagents/workflows/wf_<id>/` (`journal.jsonl` and further
//! `agent-<agentId>.jsonl` files spawned by that workflow).
//!
//! Identity, per `session-identity.md`:
//!
//! | | value |
//! |---|---|
//! | `session_id` | the record's own `sessionId`, for every stream kind except the workflow journal |
//! | `stream_id` | `"main"` for the top-level file; `agent-<agentId>` for a subagent file; `workflow-<wf_id>` for a journal |
//! | `seq` | 0-based line index |
//!
//! `journal.jsonl` carries **no** `sessionId` and **no** timestamps in its
//! content (`started`/`result` records only) — its identity is derived
//! entirely from its path, which is why classification happens once at
//! [`discover`] time rather than being re-derived per record.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde_json::Value;

use super::StreamRef;
use super::scan::{self, seq_lines};
use crate::error::{TranscriptError, TranscriptResult};
use crate::model::{RecordKey, RecordOrigin, Source, StreamKind, VerbatimRecord};

/// Recursively discover every Claude Code stream under `root`
/// (`~/.claude/projects`). A naive `<slug>/*.jsonl` glob finds only the main
/// files — see `source-file-lifecycle.md`.
pub fn discover(root: &Path) -> TranscriptResult<Vec<StreamRef>> {
    if !root.exists() {
        return Err(TranscriptError::adapter_root_missing(root));
    }
    let mut out = Vec::new();
    for path in scan::discover_jsonl_paths(root) {
        match classify(root, &path) {
            Some((session_id, stream_id, stream_kind)) => {
                out.push(StreamRef {
                    source: Source::ClaudeCode,
                    session_id,
                    stream_id,
                    stream_kind,
                    file_id: scan::file_id_or_zst(&path),
                    path,
                });
            }
            None => {
                tracing::warn!(
                    path = %path.display(),
                    "claude_code: path did not match any known layout, skipping \
                     rather than guessing at identity"
                );
            }
        }
    }
    Ok(out)
}

/// Classify one discovered path into `(session_id, stream_id, stream_kind)`,
/// or `None` if it matches no known Claude Code layout.
fn classify(root: &Path, path: &Path) -> Option<(String, String, StreamKind)> {
    let rel = path.strip_prefix(root).ok()?;
    let components: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();

    // <slug>/<sessionId>.jsonl
    if components.len() == 2 {
        let session_id = components[1].strip_suffix(".jsonl")?.to_string();
        return Some((session_id, "main".to_string(), StreamKind::Main));
    }

    // Anything deeper is <slug>/<sessionId>/... — the session id is always
    // the path component right after the slug, no matter how much deeper
    // `subagents/` nests (the ticket flags "and possibly deeper").
    if components.len() < 3 {
        return None;
    }
    let session_id = components[1].clone();
    let filename = components.last()?.clone();

    if filename == "journal.jsonl" {
        // .../subagents/workflows/wf_<id>/journal.jsonl
        let wf_component = components
            .iter()
            .rev()
            .find(|c| c.starts_with("wf_"))?
            .clone();
        let wf_id = wf_component.strip_prefix("wf_").unwrap_or(&wf_component);
        return Some((
            session_id,
            format!("workflow-{wf_id}"),
            StreamKind::WorkflowJournal,
        ));
    }

    if filename.starts_with("agent-") && filename.ends_with(".jsonl") {
        let stream_id = filename.strip_suffix(".jsonl")?.to_string();
        return Some((session_id, stream_id, StreamKind::Subagent));
    }

    None
}

/// Read new, complete lines from `stream`, starting at the byte `from_offset`.
pub fn read_stream(
    stream: &StreamRef,
    from_offset: u64,
    machine: &str,
) -> TranscriptResult<(Vec<VerbatimRecord>, u64)> {
    let content = scan::read_complete_lines(&stream.path, from_offset)?;
    let path_str = stream.path.display().to_string();

    let records = seq_lines(&content)
        .map(|(seq, line)| build_record(stream, seq, line, machine, &path_str))
        .collect();

    Ok((records, content.consumed_through))
}

fn build_record(
    stream: &StreamRef,
    seq: i64,
    raw_line: &str,
    machine: &str,
    path: &str,
) -> VerbatimRecord {
    let payload: Option<Value> = serde_json::from_str(raw_line).ok();

    let record_type = payload
        .as_ref()
        .and_then(|v| v.get("type"))
        .and_then(Value::as_str)
        .map(str::to_string);

    // Claude Code's own version dispatch is unreliable (16.7% of lines carry
    // no `version` at all — source-format-stability.md) so this is genuinely
    // optional, not a parsing bug when absent.
    let schema_marker = payload
        .as_ref()
        .and_then(|v| v.get("version"))
        .and_then(Value::as_str)
        .map(str::to_string);

    // `journal.jsonl` carries no timestamp field at all; every other stream
    // kind does on most (88.9%) of its records. Never synthesised here —
    // that is `forward_fill_observed_at`'s job, downstream.
    let timestamp = payload
        .as_ref()
        .and_then(|v| v.get("timestamp"))
        .and_then(Value::as_str)
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    VerbatimRecord {
        key: RecordKey {
            source: Source::ClaudeCode,
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

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    /// Recursive discovery must find the main session file, a subagent
    /// stream, and a workflow journal stream nested under it, and classify
    /// each correctly. This is the exact failure mode `source-file-lifecycle`
    /// calls out: a naive `<slug>/*.jsonl` glob would find only the first.
    #[test]
    fn discovery_is_recursive_and_classifies_every_stream_kind() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let slug = "-home-toasty-ws-agi-akh-medu";
        let session_id = "50cb29a0-72ff-4e1f-9a2c-0032c07acbf7";

        write(
            &root.join(slug).join(format!("{session_id}.jsonl")),
            "{\"type\":\"user\",\"sessionId\":\"50cb29a0-72ff-4e1f-9a2c-0032c07acbf7\",\"timestamp\":\"2026-07-05T11:08:16.676Z\"}\n",
        );
        write(
            &root
                .join(slug)
                .join(session_id)
                .join("subagents")
                .join("agent-af42a51231e6a2fd8.jsonl"),
            "{\"type\":\"user\",\"sessionId\":\"50cb29a0-72ff-4e1f-9a2c-0032c07acbf7\",\"agentId\":\"af42a51231e6a2fd8\",\"isSidechain\":true,\"timestamp\":\"2026-07-05T18:19:27.280Z\"}\n",
        );
        write(
            &root
                .join(slug)
                .join(session_id)
                .join("subagents")
                .join("workflows")
                .join("wf_713f56df-22a")
                .join("journal.jsonl"),
            "{\"type\":\"started\",\"key\":\"v2:abc\",\"agentId\":\"ac08b8a2f0884cdf8\"}\n",
        );
        write(
            &root
                .join(slug)
                .join(session_id)
                .join("subagents")
                .join("workflows")
                .join("wf_713f56df-22a")
                .join("agent-ac08b8a2f0884cdf8.jsonl"),
            "{\"type\":\"user\",\"sessionId\":\"50cb29a0-72ff-4e1f-9a2c-0032c07acbf7\",\"agentId\":\"ac08b8a2f0884cdf8\",\"timestamp\":\"2026-07-05T18:19:27.280Z\"}\n",
        );

        let streams = discover(root).unwrap();
        assert_eq!(
            streams.len(),
            4,
            "expected all four nested streams: {streams:?}"
        );

        let main = streams
            .iter()
            .find(|s| s.stream_kind == StreamKind::Main)
            .expect("main stream");
        assert_eq!(main.session_id, session_id);
        assert_eq!(main.stream_id, "main");

        let subagent = streams
            .iter()
            .find(|s| s.stream_id == "agent-af42a51231e6a2fd8")
            .expect("top-level subagent stream");
        assert_eq!(subagent.session_id, session_id);
        assert_eq!(subagent.stream_kind, StreamKind::Subagent);

        let nested_subagent = streams
            .iter()
            .find(|s| s.stream_id == "agent-ac08b8a2f0884cdf8")
            .expect("workflow-nested subagent stream");
        assert_eq!(nested_subagent.session_id, session_id);
        assert_eq!(nested_subagent.stream_kind, StreamKind::Subagent);

        let journal = streams
            .iter()
            .find(|s| s.stream_kind == StreamKind::WorkflowJournal)
            .expect("workflow journal stream");
        assert_eq!(journal.session_id, session_id);
        assert_eq!(journal.stream_id, "workflow-713f56df-22a");
    }

    /// `journal.jsonl` carries no `sessionId` in its content at all — its
    /// identity must come entirely from the path.
    #[test]
    fn journal_identity_is_path_derived() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let path = root
            .join("slug")
            .join("sess-1")
            .join("subagents")
            .join("workflows")
            .join("wf_abc")
            .join("journal.jsonl");
        write(
            &path,
            "{\"type\":\"started\",\"key\":\"v2:x\",\"agentId\":\"a1\"}\n",
        );

        let streams = discover(root).unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].session_id, "sess-1");
        assert_eq!(streams[0].stream_id, "workflow-abc");
        assert_eq!(streams[0].stream_kind, StreamKind::WorkflowJournal);

        let (records, _) = read_stream(&streams[0], 0, "test-machine").unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].timestamp.is_none());
        assert_eq!(records[0].key.session_id, "sess-1");
    }

    /// A trailing line without a newline is not consumed, and the returned
    /// offset never advances past it.
    #[test]
    fn partial_trailing_line_is_not_consumed() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let path = root.join("slug").join("sess-1.jsonl");
        let complete = "{\"type\":\"user\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-07-05T11:08:16.676Z\"}\n";
        let partial = "{\"type\":\"user\",\"sessionId\":\"sess-1\"";
        write(&path, &format!("{complete}{partial}"));

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: path.clone(),
            file_id: None,
        };

        let (records, offset) = read_stream(&stream, 0, "m").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(offset, complete.len() as u64);

        // A second read from the returned offset yields nothing new until
        // the partial line is completed.
        let (records2, offset2) = read_stream(&stream, offset, "m").unwrap();
        assert!(records2.is_empty());
        assert_eq!(offset2, offset);
    }

    /// Resuming from a non-zero offset yields only the records appended
    /// after it — the core of "re-running over an unchanged file is a
    /// no-op, over a grown file inserts only the tail".
    #[test]
    fn resume_from_offset_yields_only_new_records() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let path = root.join("slug").join("sess-1.jsonl");
        let line1 = "{\"type\":\"user\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-07-05T11:08:16.676Z\"}\n";
        write(&path, line1);

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: path.clone(),
            file_id: None,
        };

        let (first_pass, offset) = read_stream(&stream, 0, "m").unwrap();
        assert_eq!(first_pass.len(), 1);
        assert_eq!(first_pass[0].key.seq, 0);

        // Unchanged file: re-running from the returned offset is a no-op.
        let (noop_pass, offset_again) = read_stream(&stream, offset, "m").unwrap();
        assert!(noop_pass.is_empty());
        assert_eq!(offset_again, offset);

        // File grows: only the new line comes back, at seq 1.
        let line2 = "{\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-07-05T11:08:17.000Z\"}\n";
        let mut appended = line1.to_string();
        appended.push_str(line2);
        fs::write(&path, appended).unwrap();

        let (second_pass, _offset2) = read_stream(&stream, offset, "m").unwrap();
        assert_eq!(second_pass.len(), 1);
        assert_eq!(second_pass[0].key.seq, 1);
    }

    /// `.jsonl.zst` fallback: never observed locally, must still work.
    #[test]
    fn zst_fallback_is_read_when_plain_file_is_absent() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let logical_path = root.join("slug").join("sess-1.jsonl");
        let zst_path = root.join("slug").join("sess-1.jsonl.zst");

        let content = "{\"type\":\"user\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-07-05T11:08:16.676Z\"}\n\
                        {\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-07-05T11:08:17.000Z\"}\n";
        fs::create_dir_all(zst_path.parent().unwrap()).unwrap();
        let compressed = zstd::stream::encode_all(content.as_bytes(), 0).unwrap();
        fs::write(&zst_path, compressed).unwrap();
        // Deliberately no plain file at `logical_path`.

        let stream = StreamRef {
            source: Source::ClaudeCode,
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            stream_kind: StreamKind::Main,
            path: logical_path,
            file_id: None,
        };

        let (records, offset) = read_stream(&stream, 0, "m").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(offset, content.len() as u64);
    }

    /// Discovery also finds a zst-only stream (no `.jsonl`, only
    /// `.jsonl.zst` on disk), normalizing its path to the logical name.
    #[test]
    fn discovery_finds_zst_only_streams() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let zst_path = root.join("slug").join("sess-1.jsonl.zst");
        fs::create_dir_all(zst_path.parent().unwrap()).unwrap();
        let compressed = zstd::stream::encode_all(&b"{}\n"[..], 0).unwrap();
        fs::write(&zst_path, compressed).unwrap();

        let streams = discover(root).unwrap();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].path, root.join("slug").join("sess-1.jsonl"));
    }
}
