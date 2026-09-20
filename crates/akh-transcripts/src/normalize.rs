//! normalize — turn verbatim records into normalized events.
//!
//! Vocabulary: `docs/wayfinder/transcript-harvest/CONTEXT.md`. Mapping table
//! and the two addenda that correct it (`compacted` is `context`; `thought`
//! carries almost no content): `tickets/event-normal-form.md`. Poison-record
//! handling (why a malformed-but-known shape becomes `Degraded` rather than
//! being dropped): `tickets/derivation-worker.md` §2.
//!
//! This module owns the shared derivation vocabulary — `DerivationError`,
//! `NormalizedEvent`, `SecretFlag`/`SecretAction`, `Render` and `Chunk` — even
//! though the functions that *produce* `Render` and `Chunk` values live in
//! sibling modules (`render`, `chunk`). The types are small, mutually
//! referential, and `DerivationError` needs one home the other modules can
//! import without a cycle; putting the whole vocabulary here keeps that home
//! obvious.

use chrono::{DateTime, Utc};
use miette::Diagnostic;
use thiserror::Error;

use crate::model::{EventKind, RecordKey, Source, VerbatimRecord};

mod claude_code;
mod codex;
mod opencode;

#[cfg(test)]
mod tests;

/// Derivation-time failures. Every variant means: a record's *type* was
/// recognised, but its *shape* under that type was not what the mapping
/// table expects. Per the derivation-worker's poison-record fix, a value of
/// this kind is never surfaced as a hard error out of [`normalize`] — it is
/// folded into a [`Outcome::Degraded`] event, because dropping the record
/// would make the anti-join re-pend it on every tick, forever, silently.
#[derive(Debug, Error, Diagnostic)]
pub enum DerivationError {
    #[error("`{record_type}` record is missing the `{field}` field")]
    #[diagnostic(
        code(akh_transcripts::derive::missing_field),
        help(
            "The record matched a known type but a field this normalizer \
             depends on for that type was absent. The verbatim row is \
             retained; this record has been normalized as a degraded event \
             instead of being silently dropped. If this recurs often, the \
             source's shape has probably drifted and the mapping table in \
             tickets/event-normal-form.md needs revisiting."
        )
    )]
    MissingField {
        record_type: String,
        field: &'static str,
    },

    #[error("`{record_type}` record's `{field}` was expected to be {expected}, found {found}")]
    #[diagnostic(
        code(akh_transcripts::derive::unexpected_shape),
        help(
            "The field existed but its JSON shape did not match what the \
             mapping table assumes for this record type. The verbatim row \
             is retained; this record has been normalized as a degraded \
             event instead of being silently dropped."
        )
    )]
    UnexpectedShape {
        record_type: String,
        field: &'static str,
        expected: &'static str,
        found: String,
    },
}

/// What deriving one verbatim record produces. Every arm must be handled by
/// the derivation worker — nothing here silently disappears. Vocabulary:
/// `CONTEXT.md` (`Noise`, `Undetermined`, `Degraded`).
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// One record produced exactly one normalized event.
    Event(NormalizedEvent),
    /// One record produced several normalized events — e.g. a Claude `user`
    /// record with three `tool_result` blocks. They share `key`, and are
    /// distinguished from each other by `subseq`.
    Events(Vec<NormalizedEvent>),
    /// Retained verbatim; carries no transcript content, or duplicates
    /// another record. Never deleted — this is a statement about derivation,
    /// not about value.
    Noise { reason: String },
    /// The record's type could not be established from the data. Retained,
    /// counted, never guessed.
    Undetermined { record_type: String },
    /// A known type whose shape was only partially recognised. The event
    /// carries `degraded = true` and a `degraded_reason`.
    Degraded(NormalizedEvent),
}

/// A cross-source normalized event: the ten closed kinds in
/// [`crate::model::EventKind`], keyed back to the verbatim row by `key`
/// (+ `subseq` when one record yields more than one event).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedEvent {
    pub key: RecordKey,
    /// Distinguishes multiple events derived from one record (e.g. the block
    /// index inside a `user` record's content array). `None` when the record
    /// maps to at most one event.
    pub subseq: Option<u32>,
    pub observed_at: DateTime<Utc>,
    pub kind: EventKind,
    pub subkind: Option<String>,
    pub role: Option<String>,
    pub text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub agent_id: Option<String>,
    pub parent_agent_id: Option<String>,
    pub degraded: bool,
    pub degraded_reason: Option<String>,
    pub repo: Option<String>,
    pub workspace: Option<String>,
    pub branch: Option<String>,
}

impl NormalizedEvent {
    /// A minimal event with every optional field empty. Per-source mapping
    /// code fills in only the fields its record type actually carries,
    /// rather than repeating a dozen `None`s per arm.
    pub(crate) fn new(key: RecordKey, observed_at: DateTime<Utc>, kind: EventKind) -> Self {
        NormalizedEvent {
            key,
            subseq: None,
            observed_at,
            kind,
            subkind: None,
            role: None,
            text: None,
            tool_name: None,
            tool_call_id: None,
            agent_id: None,
            parent_agent_id: None,
            degraded: false,
            degraded_reason: None,
            repo: None,
            workspace: None,
            branch: None,
        }
    }

    /// Mark this event degraded with a human-readable reason. Chainable so
    /// mapping code can write `NormalizedEvent::new(...).degrade(err)`.
    pub(crate) fn degrade(mut self, reason: impl Into<String>) -> Self {
        self.degraded = true;
        self.degraded_reason = Some(reason.into());
        self
    }
}

/// Fold a single derived event into the right [`Outcome`] arm: degraded
/// events surface as [`Outcome::Degraded`], everything else as
/// [`Outcome::Event`]. Centralising this means per-source mapping code only
/// has to build the `NormalizedEvent` correctly — it never has to remember
/// to pick the right `Outcome` variant by hand.
pub(crate) fn finish_single(event: NormalizedEvent) -> Outcome {
    if event.degraded {
        Outcome::Degraded(event)
    } else {
        Outcome::Event(event)
    }
}

/// How a [`SecretFlag`] should be applied by `redact::apply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretAction {
    /// Replace just the matched span with `[REDACTED:<pattern>]`.
    Mask,
    /// The match is the unboundable start of a multi-line secret (a PEM
    /// header, which the pattern cannot bound) — replace the whole text.
    Withhold,
}

/// One credential-shaped match inside a normalized event's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretFlag {
    pub pattern: String,
    /// Byte offsets into the original text passed to `redact::scan`.
    pub start: usize,
    pub end: usize,
    pub action: SecretAction,
}

/// The compact one-line render of a [`NormalizedEvent`]
/// (`<seq> <kind> | <field> <value> | ...`). A thin newtype rather than a
/// bare `String`, so a render can't be confused with raw event text at the
/// type level — nothing else produces this exact wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Render(pub String);

impl std::fmt::Display for Render {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Render {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<Render> for String {
    fn from(render: Render) -> String {
        render.0
    }
}

/// A prose chunk spanning a contiguous run of `prompt`/`reply` events within
/// one stream, ready for embedding. `seq_from..=seq_to` is the pointer the
/// read surface widens into `get_lines`. Text is already redacted — this is
/// the outward boundary the embedder and the read surface both see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub source: Source,
    pub session_id: String,
    pub stream_id: String,
    pub seq_from: i64,
    pub seq_to: i64,
    pub text: String,
}

/// Turn one verbatim record into zero or more normalized events.
///
/// Dispatches on `(record.key.source, record.origin.record_type)` per the
/// normal-form ticket's decision 7. Never panics: a payload that didn't
/// parse as JSON, or a JSON shape this normalizer has never seen at all,
/// becomes [`Outcome::Undetermined`]; a shape that is recognised but
/// malformed becomes an [`Outcome::Degraded`] event. The verbatim row is
/// always retained regardless of which arm fires.
pub fn normalize(record: &VerbatimRecord, observed_at: DateTime<Utc>) -> Outcome {
    match record.key.source {
        Source::ClaudeCode => claude_code::normalize(record, observed_at),
        Source::Codex => codex::normalize(record, observed_at),
        Source::Opencode => opencode::normalize(record, observed_at),
    }
}

// --- helpers shared by the per-source submodules ---

/// The source's own top-level type marker, or a placeholder if the adapter
/// could not determine one — the latter is itself grounds for
/// `Undetermined`, never a guess.
pub(crate) fn record_type_of(record: &VerbatimRecord) -> String {
    record
        .origin
        .record_type
        .clone()
        .unwrap_or_else(|| "<unknown>".to_string())
}

pub(crate) fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}
