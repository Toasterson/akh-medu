//! OpenCode mapping. The local corpus is thin (18 events, one session), so
//! this adapter is deliberately conservative: every branch below is a
//! best-effort reading of the `event.data` JSON shape implied by the census
//! (`docs/wayfinder/transcript-harvest/assets/source-format-stability.md`
//! §2-3) and the ticket's dispatch rules, not a shape confirmed against a
//! wide corpus.
//!
//! `VerbatimRecord.payload` for OpenCode is the *whole* wrapped row the
//! adapter builds — `{"id", "aggregate_id", "seq", "type", "data"}`
//! (`adapter::opencode::build_record`) — not the `data` blob directly. The
//! actual event body this module maps from is always `payload["data"]`.
//! `origin.record_type` also arrives pre-stripped of the `.N` schema-version
//! suffix by the adapter's own `split_type_and_marker`; this module strips
//! it again defensively (a no-op against already-stripped input) so
//! dispatch does not silently depend on that adapter behaviour.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::model::{EventKind, VerbatimRecord};

use super::{finish_single, record_type_of, DerivationError, NormalizedEvent, Outcome};

pub(super) fn normalize(record: &VerbatimRecord, observed_at: DateTime<Utc>) -> Outcome {
    let record_type = record_type_of(record);
    let Some(row) = record.payload.as_ref() else {
        return Outcome::Undetermined { record_type };
    };
    // Fall back to the row itself if `data` is absent, rather than treating
    // an unexpected wrapper shape as fatal — the field-lookups below simply
    // find nothing and degrade or fall through to Undetermined.
    let doc = row.get("data").unwrap_or(row);

    let (base, _version) = strip_version_suffix(&record_type);
    match base {
        "session.created" | "session.updated" => {
            normalize_session_meta(record, doc, observed_at, base)
        }
        "message.updated" => normalize_message_updated(record, doc, observed_at),
        "message.part.updated" => normalize_part_updated(record, doc, observed_at),
        _ => Outcome::Undetermined { record_type },
    }
}

/// OpenCode's `event.type` carries a self-versioned suffix (`message.updated.1`).
/// Strip a single trailing `.<digits>` group; the base string is what the
/// mapping table dispatches on.
fn strip_version_suffix(event_type: &str) -> (&str, Option<&str>) {
    match event_type.rsplit_once('.') {
        Some((base, suffix)) if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) => {
            (base, Some(suffix))
        }
        _ => (event_type, None),
    }
}

fn normalize_session_meta(
    record: &VerbatimRecord,
    doc: &Value,
    observed_at: DateTime<Utc>,
    subkind: &str,
) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::SessionMeta);
    ev.subkind = Some(subkind.to_string());
    ev.text = doc
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some(doc.to_string()));
    finish_single(ev)
}

fn normalize_message_updated(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let role = doc
        .get("role")
        .and_then(Value::as_str)
        .or_else(|| doc.get("info").and_then(|i| i.get("role")).and_then(Value::as_str));

    let kind = match role {
        Some("user") => EventKind::Prompt,
        Some("assistant") => EventKind::Reply,
        Some(other) => {
            return Outcome::Undetermined {
                record_type: format!("message.updated:role={other}"),
            }
        }
        None => {
            let ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Prompt).degrade(
                DerivationError::MissingField {
                    record_type: "message.updated".to_string(),
                    field: "role",
                }
                .to_string(),
            );
            return finish_single(ev);
        }
    };

    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, kind);
    ev.role = role.map(str::to_string);
    ev.text = doc.get("text").and_then(Value::as_str).map(str::to_string);
    finish_single(ev)
}

fn normalize_part_updated(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let part = match doc.get("part") {
        Some(p) => p,
        None => {
            let ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Context).degrade(
                DerivationError::MissingField {
                    record_type: "message.part.updated".to_string(),
                    field: "part",
                }
                .to_string(),
            );
            return finish_single(ev);
        }
    };
    let role = doc.get("role").and_then(Value::as_str);
    let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");

    match part_type {
        "text" => {
            // Conservative default: a streamed text part with no legible
            // role is treated as the (far more common) assistant case
            // rather than guessed as a user prompt.
            let kind = match role {
                Some("user") => EventKind::Prompt,
                _ => EventKind::Reply,
            };
            let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, kind);
            ev.role = role.map(str::to_string);
            ev.text = part.get("text").and_then(Value::as_str).map(str::to_string);
            finish_single(ev)
        }
        "tool" => {
            let has_output = part.get("output").is_some() || part.get("result").is_some();
            let kind = if has_output {
                EventKind::ToolResult
            } else {
                EventKind::ToolCall
            };
            let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, kind);
            ev.tool_name = part.get("tool").and_then(Value::as_str).map(str::to_string);
            ev.tool_call_id = part
                .get("callID")
                .and_then(Value::as_str)
                .or_else(|| part.get("id").and_then(Value::as_str))
                .map(str::to_string);
            ev.text = if has_output {
                part.get("output")
                    .or_else(|| part.get("result"))
                    .map(|v| v.to_string())
            } else {
                part.get("input").or_else(|| part.get("state")).map(|v| v.to_string())
            };
            finish_single(ev)
        }
        other => Outcome::Undetermined {
            record_type: format!("message.part.updated:part={other}"),
        },
    }
}
