//! Codex mapping. `response_item` is canonical for model I/O; `event_msg` is
//! mined only for lifecycle narration `response_item` lacks. Ticket table +
//! addenda: `docs/wayfinder/transcript-harvest/tickets/event-normal-form.md`.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::model::{EventKind, RecordKey, VerbatimRecord};

use super::{finish_single, json_type_name, record_type_of, DerivationError, NormalizedEvent, Outcome};

pub(super) fn normalize(record: &VerbatimRecord, observed_at: DateTime<Utc>) -> Outcome {
    let Some(doc) = record.payload.as_ref() else {
        return Outcome::Undetermined {
            record_type: record_type_of(record),
        };
    };

    // Dispatch on the record's own top-level `type` field, not
    // `origin.record_type`: the adapter deliberately flattens a
    // `response_item`'s `origin.record_type` to its *nested*
    // `payload.type` (`message`, `reasoning`, `function_call`, ...) because
    // that is the meaningful drift-counting key — see
    // `adapter::codex::record_type_of`. Dispatch here needs the outer
    // envelope kind instead, which `doc["type"]` always carries verbatim.
    let top_type = doc.get("type").and_then(Value::as_str).unwrap_or("");

    match top_type {
        "response_item" => normalize_response_item(record, doc, observed_at),
        "event_msg" => normalize_event_msg(record, doc, observed_at),
        "session_meta" => normalize_session_meta(record, doc, observed_at),
        "token_usage_record" => normalize_telemetry(record, doc, observed_at),
        "compacted" => normalize_context(record, doc, observed_at, "compaction"),
        "turn_context" => normalize_context(record, doc, observed_at, "turn_context"),
        // `world_state` is explicitly Undetermined per the normal-form
        // ticket's decision 5 — opaque payload, not classified as noise.
        // Everything else not enumerated in the mapping table (including
        // `inter_agent_communication_metadata`) falls to the same catch-all:
        // counted, never guessed.
        _ => Outcome::Undetermined {
            record_type: if top_type.is_empty() {
                record_type_of(record)
            } else {
                top_type.to_string()
            },
        },
    }
}

fn normalize_response_item(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let key = record.key.clone();
    let payload = match doc.get("payload") {
        Some(p) => p,
        None => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Error).degrade(
                DerivationError::MissingField {
                    record_type: "response_item".to_string(),
                    field: "payload",
                }
                .to_string(),
            );
            return finish_single(ev);
        }
    };
    let payload_type = payload.get("type").and_then(Value::as_str).unwrap_or("");

    match payload_type {
        "message" => normalize_message_payload(key, payload, observed_at),
        "agent_message" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Reply);
            match payload.get("message").and_then(Value::as_str) {
                Some(s) => ev.text = Some(s.to_string()),
                None => {
                    ev = ev.degrade(
                        DerivationError::MissingField {
                            record_type: "response_item:agent_message".to_string(),
                            field: "message",
                        }
                        .to_string(),
                    )
                }
            }
            finish_single(ev)
        }
        "reasoning" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Thought);
            ev.subkind = Some("encrypted".to_string());
            // CORRECTION (render-format ticket): Codex reasoning is measured
            // at 100% `summary: []` + `encrypted_content`. `text` stays
            // `None` unconditionally — never emit the ciphertext, and never
            // trust an as-yet-unobserved non-empty `summary` either, since
            // the posture here is "marker, not body."
            finish_single(ev)
        }
        "custom_tool_call" | "function_call" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::ToolCall);
            ev.tool_name = payload.get("name").and_then(Value::as_str).map(str::to_string);
            ev.tool_call_id = payload
                .get("call_id")
                .and_then(Value::as_str)
                .or_else(|| payload.get("id").and_then(Value::as_str))
                .map(str::to_string);
            match payload.get("input").or_else(|| payload.get("arguments")) {
                Some(v) => ev.text = Some(v.to_string()),
                None => {
                    ev = ev.degrade(
                        DerivationError::MissingField {
                            record_type: format!("response_item:{payload_type}"),
                            field: "input/arguments",
                        }
                        .to_string(),
                    )
                }
            }
            finish_single(ev)
        }
        "custom_tool_call_output" | "function_call_output" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::ToolResult);
            // Not explicitly named in the mapping table for the output side,
            // but call_id/id is how the ticket ties tool_call to tool_result
            // on the call side; mirrored here so the pair is joinable.
            ev.tool_call_id = payload
                .get("call_id")
                .and_then(Value::as_str)
                .or_else(|| payload.get("id").and_then(Value::as_str))
                .map(str::to_string);
            match payload.get("output") {
                Some(v) => ev.text = Some(v.to_string()),
                None => {
                    ev = ev.degrade(
                        DerivationError::MissingField {
                            record_type: format!("response_item:{payload_type}"),
                            field: "output",
                        }
                        .to_string(),
                    )
                }
            }
            finish_single(ev)
        }
        other => Outcome::Undetermined {
            record_type: format!("response_item:{other}"),
        },
    }
}

fn normalize_message_payload(key: RecordKey, payload: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let role = payload.get("role").and_then(Value::as_str).unwrap_or("");
    let kind = match role {
        "user" => EventKind::Prompt,
        "assistant" => EventKind::Reply,
        "developer" => EventKind::Context,
        _ => {
            return Outcome::Undetermined {
                record_type: format!("response_item:message:role={role}"),
            }
        }
    };
    let mut ev = NormalizedEvent::new(key, observed_at, kind);
    ev.role = Some(role.to_string());
    match payload.get("content") {
        Some(Value::Array(items)) => {
            let text = items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("");
            ev.text = if text.is_empty() { None } else { Some(text) };
        }
        Some(other) => {
            ev = ev.degrade(
                DerivationError::UnexpectedShape {
                    record_type: "response_item:message".to_string(),
                    field: "content",
                    expected: "array of content items",
                    found: json_type_name(other).to_string(),
                }
                .to_string(),
            );
        }
        None => {
            ev = ev.degrade(
                DerivationError::MissingField {
                    record_type: "response_item:message".to_string(),
                    field: "content",
                }
                .to_string(),
            );
        }
    }
    finish_single(ev)
}

/// `event_msg` is mined *only* for lifecycle narration `response_item`
/// lacks. Everything else — including `token_count`, the single largest
/// record type in the whole corpus — is noise: a UI/telemetry narration of
/// turns `response_item` already carries canonically.
fn normalize_event_msg(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let payload = match doc.get("payload") {
        Some(p) => p,
        None => {
            return Outcome::Noise {
                reason: "event_msg without a payload".to_string(),
            }
        }
    };
    let payload_type = payload.get("type").and_then(Value::as_str).unwrap_or("");
    match payload_type {
        "task_started" | "task_complete" | "sub_agent_activity" => {
            normalize_lifecycle(record.key.clone(), payload, observed_at, payload_type)
        }
        other => Outcome::Noise {
            reason: format!(
                "event_msg/{other} is a UI/telemetry narration superseded by response_item"
            ),
        },
    }
}

fn normalize_lifecycle(
    key: RecordKey,
    payload: &Value,
    observed_at: DateTime<Utc>,
    subkind: &str,
) -> Outcome {
    let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Lifecycle);
    ev.subkind = Some(subkind.to_string());
    ev.agent_id = payload
        .get("agent_id")
        .and_then(Value::as_str)
        .or_else(|| payload.get("agentId").and_then(Value::as_str))
        .map(str::to_string);
    ev.text = Some(payload.to_string());
    finish_single(ev)
}

fn normalize_session_meta(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::SessionMeta);
    // No specific field is named for Codex's top-level `session_meta` in the
    // mapping table; the whole record is retained as text.
    ev.text = Some(doc.to_string());
    finish_single(ev)
}

fn normalize_telemetry(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Telemetry);
    ev.text = Some(doc.to_string());
    finish_single(ev)
}

fn normalize_context(
    record: &VerbatimRecord,
    doc: &Value,
    observed_at: DateTime<Utc>,
    subkind: &str,
) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Context);
    ev.subkind = Some(subkind.to_string());
    ev.text = Some(doc.to_string());
    finish_single(ev)
}
