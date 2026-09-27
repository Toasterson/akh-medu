//! Shared filesystem plumbing for the JSONL-based adapters ([`super::claude_code`]
//! and [`super::codex`]). OpenCode does not use this module — it reads SQLite.
//!
//! Two contracts live here, both settled by
//! `docs/wayfinder/transcript-harvest/tickets/source-file-lifecycle.md`:
//!
//! - **Recursive discovery.** A non-recursive `<slug>/*.jsonl` glob misses most
//!   of Claude Code's real layout (subagent and workflow-journal streams live
//!   several directories deeper). [`discover_jsonl_paths`] always walks.
//! - **Complete-lines-only reading, with the `.jsonl.zst` rotation fallback.**
//!   [`read_complete_lines`] never returns a trailing partial line, and if the
//!   plain file is gone but a same-named `.jsonl.zst` sibling exists, it
//!   decompresses and reads that from byte zero — crude but correct, per the
//!   ticket's reasoning: re-inserting an already-seen record is a harmless
//!   `ON CONFLICT DO NOTHING` at the store, so a full re-read costs bandwidth
//!   and nothing else. This path has never been observed locally and is
//!   covered only by synthetic tests.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::TranscriptError;
use crate::model::VerbatimRecord;

/// Identity of a physical file instance, so a caller can tell a truncated or
/// replaced file apart from the one whose watermark it stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileId {
    pub dev: u64,
    pub ino: u64,
}

#[cfg(unix)]
pub(crate) fn file_id(path: &Path) -> Option<FileId> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).ok().map(|m| FileId {
        dev: m.dev(),
        ino: m.ino(),
    })
}

#[cfg(not(unix))]
pub(crate) fn file_id(_path: &Path) -> Option<FileId> {
    None
}

/// [`file_id`] for a logical `.jsonl` path, falling back to its `.jsonl.zst`
/// sibling when the plain file does not exist (rotation).
pub(crate) fn file_id_or_zst(path: &Path) -> Option<FileId> {
    file_id(path).or_else(|| file_id(&append_zst_extension(path)))
}

/// Recursively find every logical `.jsonl` stream under `root`: every real
/// `.jsonl` file, plus every `.jsonl.zst` file for which no `.jsonl` sibling
/// exists, normalized to its logical `.jsonl` path. Order is deterministic
/// (lexicographic) so callers get stable results across runs.
pub(crate) fn discover_jsonl_paths(root: &Path) -> Vec<PathBuf> {
    let mut plain = BTreeSet::new();
    let mut zst_only = BTreeSet::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        match path.extension().and_then(|e| e.to_str()) {
            Some("jsonl") => {
                plain.insert(path.to_path_buf());
            }
            Some("zst") => {
                // `foo.jsonl.zst` -> stem is `foo.jsonl`; strip the `.zst`
                // extension to get the logical path back.
                let looks_like_jsonl_zst = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".jsonl"))
                    .unwrap_or(false);
                if looks_like_jsonl_zst {
                    zst_only.insert(path.with_extension(""));
                }
            }
            _ => {}
        }
    }

    let mut out: Vec<PathBuf> = plain.iter().cloned().collect();
    for logical in zst_only {
        if !plain.contains(&logical) {
            out.push(logical);
        }
    }
    out.sort();
    out
}

/// The result of one `read_complete_lines` call: complete lines in stream
/// order, the 0-based seq of the first line in `lines`, and the byte offset
/// the caller should persist as the new watermark. File-identity change
/// detection (rotation, replacement) is [`discover`]'s job via
/// [`file_id_or_zst`] on a fresh scan, not this per-call result — a `.zst`
/// fallback read has nowhere to report a *new* identity through this shape,
/// and re-discovering periodically is the documented way a caller learns
/// about it.
pub(crate) struct StreamContent {
    pub lines: Vec<String>,
    pub start_seq: i64,
    pub consumed_through: u64,
}

/// Read complete lines from the logical stream at `path`, starting at
/// `from_offset`. `path` is always the `.jsonl` name; if it does not exist,
/// falls back to `path` + `.zst` per the rotation rule (always from byte
/// zero — see module docs).
pub(crate) fn read_complete_lines(
    path: &Path,
    from_offset: u64,
) -> Result<StreamContent, TranscriptError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(slice_complete_lines(&bytes, from_offset)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let zst_path = append_zst_extension(path);
            match std::fs::read(&zst_path) {
                Ok(compressed) => {
                    let decompressed = zstd::stream::decode_all(&compressed[..])
                        .map_err(|e2| TranscriptError::adapter_decompress(&zst_path, e2))?;
                    // Rotation rule: always re-read from zero, regardless of
                    // whatever offset the caller last stored against the
                    // plain file.
                    Ok(slice_complete_lines(&decompressed, 0))
                }
                Err(e2) if e2.kind() == std::io::ErrorKind::NotFound => {
                    // Neither the plain file nor its `.zst` sibling exist:
                    // report the original path's error, it's the one the
                    // caller actually asked to read.
                    Err(TranscriptError::adapter_io(path, e))
                }
                Err(e2) => Err(TranscriptError::adapter_io(&zst_path, e2)),
            }
        }
        Err(e) => Err(TranscriptError::adapter_io(path, e)),
    }
}

fn append_zst_extension(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".zst");
    PathBuf::from(s)
}

fn slice_complete_lines(bytes: &[u8], from_offset: u64) -> StreamContent {
    let requested_start = usize::try_from(from_offset).unwrap_or(usize::MAX);
    // source-file-lifecycle.md §2: "inode changed, or size < offset -> re-read
    // from zero". A stale offset past the end of a (possibly replaced) file
    // is exactly that condition applied defensively at the byte level.
    let start = if requested_start > bytes.len() {
        0
    } else {
        requested_start
    };

    let tail = &bytes[start..];
    // Only bytes up to and including the last newline are "complete";
    // anything after is a partial line still being written.
    let complete_len = match tail.iter().rposition(|&b| b == b'\n') {
        Some(pos) => pos + 1,
        None => 0,
    };
    let complete = &tail[..complete_len];

    let start_seq = bytes[..start].iter().filter(|&&b| b == b'\n').count() as i64;

    let lines = complete
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| String::from_utf8_lossy(l).into_owned())
        .collect();

    StreamContent {
        lines,
        start_seq,
        consumed_through: (start + complete_len) as u64,
    }
}

/// Split raw lines into `(seq, line)` pairs starting at `start_seq`. A tiny
/// helper so every adapter's `read_stream` builds [`VerbatimRecord`]s the
/// same way.
pub(crate) fn seq_lines(content: &StreamContent) -> impl Iterator<Item = (i64, &str)> {
    content
        .lines
        .iter()
        .enumerate()
        .map(move |(i, l)| (content.start_seq + i as i64, l.as_str()))
}

/// Forward-fill resolution for one record's `observed_at`
/// (`store-schema.md` §2, a frozen contract):
///
/// 1. The record's own timestamp, when present.
/// 2. Otherwise the nearest **preceding** timestamped record in the same
///    stream.
/// 3. Never a *following* record — backward-fill would change a record's
///    `observed_at`, and therefore its primary key, as the stream grows,
///    silently duplicating it.
/// 4. No preceding timestamp at all -> [`ObservedAt::Quarantined`], never
///    guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservedAt {
    /// The record carried its own timestamp.
    Own(chrono::DateTime<chrono::Utc>),
    /// Carried forward from the nearest preceding timestamped record in the
    /// same stream (`observed_at_synthetic = true` downstream).
    ForwardFilled(chrono::DateTime<chrono::Utc>),
    /// No timestamp on the record, and none preceding it in the stream.
    Quarantined,
}

/// Pure function implementing the carry-forward contract above.
/// `records` must already be in stream (seq) order — this function does not
/// sort, since sorting is a stream-ordering decision this crate makes
/// exactly once, at read time.
pub fn forward_fill_observed_at(records: &[VerbatimRecord]) -> Vec<ObservedAt> {
    let mut last: Option<chrono::DateTime<chrono::Utc>> = None;
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{RecordKey, RecordOrigin, Source, StreamKind};
    use chrono::TimeZone;

    fn record(seq: i64, timestamp: Option<chrono::DateTime<chrono::Utc>>) -> VerbatimRecord {
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

    #[test]
    fn own_timestamp_is_used_verbatim() {
        let ts = chrono::Utc.with_ymd_and_hms(2026, 7, 5, 11, 8, 16).unwrap();
        let records = [record(0, Some(ts))];
        assert_eq!(
            forward_fill_observed_at(&records),
            vec![ObservedAt::Own(ts)]
        );
    }

    #[test]
    fn missing_timestamp_carries_forward_from_the_preceding_record_only() {
        let ts0 = chrono::Utc.with_ymd_and_hms(2026, 7, 5, 11, 8, 16).unwrap();
        let ts2 = chrono::Utc.with_ymd_and_hms(2026, 7, 5, 11, 9, 0).unwrap();
        let records = [
            record(0, Some(ts0)),
            record(1, None),
            record(2, Some(ts2)),
            record(3, None),
        ];
        let resolved = forward_fill_observed_at(&records);
        assert_eq!(
            resolved,
            vec![
                ObservedAt::Own(ts0),
                ObservedAt::ForwardFilled(ts0),
                ObservedAt::Own(ts2),
                ObservedAt::ForwardFilled(ts2),
            ]
        );
    }

    /// The frozen contract's sharpest rule: a record with no timestamp and no
    /// *preceding* timestamped record must never borrow one from later in the
    /// stream, even when one exists a few records ahead. Getting this wrong
    /// would change `observed_at` — and therefore the primary key — as the
    /// stream grows.
    #[test]
    fn never_looks_ahead_for_a_later_timestamp() {
        let ts1 = chrono::Utc.with_ymd_and_hms(2026, 7, 5, 11, 9, 0).unwrap();
        let records = [record(0, None), record(1, None), record(2, Some(ts1))];
        let resolved = forward_fill_observed_at(&records);
        assert_eq!(
            resolved,
            vec![
                ObservedAt::Quarantined,
                ObservedAt::Quarantined,
                ObservedAt::Own(ts1),
            ]
        );
    }
}
