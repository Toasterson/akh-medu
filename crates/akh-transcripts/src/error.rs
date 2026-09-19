//! Error type for `akh-transcripts`, following the project's miette + thiserror
//! convention: every variant carries a diagnostic `code` and actionable `help`.

use miette::Diagnostic;
use thiserror::Error;

pub type TranscriptResult<T> = Result<T, TranscriptError>;

#[derive(Debug, Error, Diagnostic)]
pub enum TranscriptError {
    // Filled in by the schema/pool and adapter increments. Keep variants
    // specific: one per failure a caller can act on differently.
    #[error("placeholder — replace during the first increment")]
    #[diagnostic(code(akh_transcripts::placeholder))]
    Placeholder,

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
