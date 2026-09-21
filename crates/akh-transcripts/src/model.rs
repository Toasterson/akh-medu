//! Core domain types shared by every layer. Vocabulary per
//! `docs/wayfinder/transcript-harvest/CONTEXT.md` — do not invent synonyms.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One agent CLI whose transcripts we harvest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    ClaudeCode,
    Codex,
    Opencode,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::ClaudeCode => "claude_code",
            Source::Codex => "codex",
            Source::Opencode => "opencode",
        }
    }
}

/// What a stream *is*. Not every file is a transcript (Claude's workflow
/// `journal.jsonl` holds only `started`/`result` lifecycle records).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamKind {
    /// Main conversation transcript.
    Main,
    /// A subagent's transcript (Claude `agent-<id>.jsonl`; Codex resume rollout).
    Subagent,
    /// Claude workflow journal — spawn links only, identity is path-derived.
    WorkflowJournal,
    /// OpenCode's single event stream per session.
    EventLog,
}

/// Identity of one record: `(source, session_id, stream_id, seq)`.
/// The harvesting machine is provenance, deliberately *not* part of identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecordKey {
    pub source: Source,
    pub session_id: String,
    pub stream_id: String,
    pub seq: i64,
}

/// A source record, byte-exact, plus what the adapter could determine about it
/// without interpreting its meaning. This is the system-of-record row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerbatimRecord {
    pub key: RecordKey,
    pub stream_kind: StreamKind,
    /// The exact source bytes (one JSONL line, or a serialised SQLite row).
    pub raw: String,
    /// Parsed form when the raw is JSON; `None` if it did not parse.
    pub payload: Option<serde_json::Value>,
    /// The record's own timestamp, if it carries one. Forward-fill happens
    /// later and is recorded separately — this field is never synthesised.
    pub timestamp: Option<DateTime<Utc>>,
    /// Where it came from, for provenance and drift diagnostics.
    pub origin: RecordOrigin,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordOrigin {
    pub machine: String,
    /// File path for JSONL sources; the SQLite path for OpenCode.
    pub path: String,
    /// The source's own top-level type marker, for `(source, type, schema_marker)`
    /// dispatch and drift counting.
    pub record_type: Option<String>,
    pub schema_marker: Option<String>,
}

/// The ten closed event kinds (normal-form ticket, decision 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Prompt,
    Reply,
    Thought,
    ToolCall,
    ToolResult,
    Context,
    SessionMeta,
    Lifecycle,
    Error,
    Telemetry,
}
