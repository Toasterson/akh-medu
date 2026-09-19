//! Source adapters: read-only readers that turn Claude Code, Codex and
//! OpenCode session stores into [`VerbatimRecord`]s. No database — this is
//! increment 2 of `docs/ai/plans/2026-09-19-transcript-harvest.md`.
//!
//! Identity, scanning and normal-form decisions are settled in
//! `docs/wayfinder/transcript-harvest/tickets/`:
//! `session-identity.md`, `source-file-lifecycle.md`,
//! `source-format-stability.md`, `event-normal-form.md`, `store-schema.md`.
//!
//! Each source module ([`claude_code`], [`codex`], [`opencode`]) exposes the
//! same two-function contract:
//!
//! - `discover(root) -> TranscriptResult<Vec<StreamRef>>` — recursive,
//!   classifies stream kind, safe to call repeatedly.
//! - `read_stream(stream, from_offset, machine) -> TranscriptResult<(Vec<VerbatimRecord>, u64)>`
//!   — complete lines/rows only; the returned offset is what the caller
//!   persists as the new watermark.
//!
//! [`scan`] holds the filesystem plumbing shared by the two JSONL sources
//! (Claude Code, Codex); OpenCode reads SQLite directly and does not use it.

pub mod claude_code;
pub mod codex;
pub mod opencode;
pub mod scan;

pub use scan::{ObservedAt, forward_fill_observed_at};

use std::path::{Path, PathBuf};

use crate::error::TranscriptResult;
use crate::model::{Source, StreamKind, VerbatimRecord};

/// Identity of a physical file instance (device + inode), so a caller can
/// tell a replaced or truncated file apart from the one whose watermark it
/// stored. `None` for OpenCode, which has no per-stream backing file — it
/// addresses by `seq` against a shared SQLite database instead.
pub type FileId = scan::FileId;

/// One discovered append-only stream, addressable by [`read_stream`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamRef {
    pub source: Source,
    pub session_id: String,
    pub stream_id: String,
    pub stream_kind: StreamKind,
    /// Canonical logical path. For the JSONL sources this is always the
    /// `.jsonl` name, even when only a `.jsonl.zst` sibling exists on disk —
    /// `read_stream` resolves the rotation fallback internally. For
    /// OpenCode, the path to `opencode.db` (informational: OpenCode
    /// addresses by `seq`, not by byte offset into this path).
    pub path: PathBuf,
    pub file_id: Option<FileId>,
}

/// Discover every stream `source` exposes under `root`.
///
/// - Claude Code / Codex: `root` is the sessions directory; discovery walks
///   it recursively — a non-recursive `<slug>/*.jsonl` glob misses most of
///   Claude Code's real layout (`source-file-lifecycle.md`).
/// - OpenCode: `root` is the path to `opencode.db`; discovery is one query
///   against the `event` table for distinct `aggregate_id`s. `message`,
///   `part`, `account` and `credential` are never touched.
pub fn discover(source: Source, root: &Path) -> TranscriptResult<Vec<StreamRef>> {
    match source {
        Source::ClaudeCode => claude_code::discover(root),
        Source::Codex => codex::discover(root),
        Source::Opencode => opencode::discover(root),
    }
}

/// Read new, complete records from `stream`, starting at `from_offset` (a
/// byte offset for the JSONL sources, the last-seen `seq` for OpenCode).
/// Returns the records in stream order and the offset to resume from on the
/// next call — persisting it verbatim is what makes a re-run over unchanged
/// input a no-op.
///
/// `machine` is the harvesting host's identity, recorded on
/// [`crate::model::RecordOrigin`] as provenance (never as part of a record's
/// key — see `session-identity.md`'s "machine as attribute" resolution).
/// This adapter layer does not resolve its own hostname: that choice belongs
/// to the harvester (increment 3), which may want to override it for testing
/// or multi-tenant deployments.
pub fn read_stream(
    stream: &StreamRef,
    from_offset: u64,
    machine: &str,
) -> TranscriptResult<(Vec<VerbatimRecord>, u64)> {
    match stream.source {
        Source::ClaudeCode => claude_code::read_stream(stream, from_offset, machine),
        Source::Codex => codex::read_stream(stream, from_offset, machine),
        Source::Opencode => opencode::read_stream(stream, from_offset, machine),
    }
}
