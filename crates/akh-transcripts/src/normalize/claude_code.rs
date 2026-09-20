//! Claude Code mapping. Ticket table + addenda:
//! `docs/wayfinder/transcript-harvest/tickets/event-normal-form.md`.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::model::{EventKind, RecordKey, VerbatimRecord};

use super::{finish_single, json_type_name, record_type_of, DerivationError, NormalizedEvent, Outcome};

pub(super) fn normalize(record: &VerbatimRecord, observed_at: DateTime<Utc>) -> Outcome {
    let record_type = record_type_of(record);
    let Some(doc) = record.payload.as_ref() else {
        return Outcome::Undetermined { record_type };
    };

    match record_type.as_str() {
        "user" => normalize_user(record, doc, observed_at),
        "assistant" => normalize_assistant(record, doc, observed_at),
        "attachment" => normalize_attachment(record, doc, observed_at),
        "system" => normalize_system(record, doc, observed_at),
        "ai-title" => {
            let text = doc.get("aiTitle").and_then(Value::as_str).map(str::to_string);
            normalize_session_meta(record, doc, observed_at, "ai-title", text)
        }
        "mode" => {
            let text = doc.get("mode").and_then(Value::as_str).map(str::to_string);
            normalize_session_meta(record, doc, observed_at, "mode", text)
        }
        "started" => normalize_lifecycle(record, doc, observed_at, "subagent_started", false),
        "result" => normalize_lifecycle(record, doc, observed_at, "subagent_result", true),
        "queue-operation" => Outcome::Noise {
            reason: "duplicates the prompt text already carried by the `user`/`prompt` event"
                .to_string(),
        },
        "last-prompt" => Outcome::Noise {
            reason: "caches the most recent prompt text; redundant with the conversation DAG"
                .to_string(),
        },
        "atis-latch" => Outcome::Noise {
            reason: "opaque content-free latch token, constant within a session".to_string(),
        },
        _ => Outcome::Undetermined { record_type },
    }
}

/// `cwd` -> workspace; `gitBranch` -> branch; repo = `cwd` with a trailing
/// `/.claude/worktrees/<name>` suffix stripped (subagent worktrees), or `cwd`
/// itself otherwise. `isSidechain` is deliberately not read here — the spawn
/// link comes from `started`/`result` lifecycle records, not from this flag.
fn apply_context_fields(event: &mut NormalizedEvent, doc: &Value) {
    let cwd = doc.get("cwd").and_then(Value::as_str).map(str::to_string);
    event.branch = doc.get("gitBranch").and_then(Value::as_str).map(str::to_string);
    event.repo = cwd.as_deref().map(strip_worktree_suffix);
    event.workspace = cwd;
}

fn strip_worktree_suffix(cwd: &str) -> String {
    match cwd.find("/.claude/worktrees/") {
        Some(idx) => cwd[..idx].to_string(),
        None => cwd.to_string(),
    }
}

fn normalize_user(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let key = record.key.clone();
    let message = match doc.get("message") {
        Some(m) => m,
        None => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Prompt).degrade(
                DerivationError::MissingField {
                    record_type: "user".to_string(),
                    field: "message",
                }
                .to_string(),
            );
            return finish_single(with_context(ev, doc));
        }
    };

    match message.get("content") {
        Some(Value::String(text)) => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Prompt);
            ev.text = Some(text.clone());
            finish_single(with_context(ev, doc))
        }
        Some(Value::Array(blocks)) => normalize_user_blocks(&key, doc, blocks, observed_at),
        Some(other) => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Prompt).degrade(
                DerivationError::UnexpectedShape {
                    record_type: "user".to_string(),
                    field: "message.content",
                    expected: "string or array",
                    found: json_type_name(other).to_string(),
                }
                .to_string(),
            );
            finish_single(with_context(ev, doc))
        }
        None => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Prompt).degrade(
                DerivationError::MissingField {
                    record_type: "user".to_string(),
                    field: "message.content",
                }
                .to_string(),
            );
            finish_single(with_context(ev, doc))
        }
    }
}

fn normalize_user_blocks(
    key: &RecordKey,
    doc: &Value,
    blocks: &[Value],
    observed_at: DateTime<Utc>,
) -> Outcome {
    if blocks.is_empty() {
        return Outcome::Noise {
            reason: "user record with an empty content array".to_string(),
        };
    }
    let mut events: Vec<NormalizedEvent> = blocks
        .iter()
        .map(|block| with_context(normalize_user_block(key.clone(), observed_at, block), doc))
        .collect();
    if events.len() == 1 {
        return finish_single(events.pop().expect("checked len == 1"));
    }
    for (i, ev) in events.iter_mut().enumerate() {
        ev.subseq = Some(i as u32);
    }
    Outcome::Events(events)
}

fn normalize_user_block(key: RecordKey, observed_at: DateTime<Utc>, block: &Value) -> NormalizedEvent {
    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
    match block_type {
        "text" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Prompt);
            ev.text = block.get("text").and_then(Value::as_str).map(str::to_string);
            ev
        }
        "tool_result" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::ToolResult);
            ev.tool_call_id = block
                .get("tool_use_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            ev.text = block.get("content").map(|c| c.to_string());
            ev
        }
        other => NormalizedEvent::new(key, observed_at, EventKind::Prompt).degrade(
            DerivationError::UnexpectedShape {
                record_type: "user".to_string(),
                field: "message.content[].type",
                expected: "\"text\" or \"tool_result\"",
                found: if other.is_empty() {
                    "missing `type` field".to_string()
                } else {
                    format!("block type `{other}`")
                },
            }
            .to_string(),
        ),
    }
}

fn normalize_assistant(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let key = record.key.clone();
    let message = match doc.get("message") {
        Some(m) => m,
        None => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Reply).degrade(
                DerivationError::MissingField {
                    record_type: "assistant".to_string(),
                    field: "message",
                }
                .to_string(),
            );
            return finish_single(with_context(ev, doc));
        }
    };

    match message.get("content") {
        Some(Value::Array(blocks)) => normalize_assistant_blocks(&key, doc, blocks, observed_at),
        Some(other) => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Reply).degrade(
                DerivationError::UnexpectedShape {
                    record_type: "assistant".to_string(),
                    field: "message.content",
                    expected: "array of content blocks",
                    found: json_type_name(other).to_string(),
                }
                .to_string(),
            );
            finish_single(with_context(ev, doc))
        }
        None => {
            let ev = NormalizedEvent::new(key, observed_at, EventKind::Reply).degrade(
                DerivationError::MissingField {
                    record_type: "assistant".to_string(),
                    field: "message.content",
                }
                .to_string(),
            );
            finish_single(with_context(ev, doc))
        }
    }
}

fn normalize_assistant_blocks(
    key: &RecordKey,
    doc: &Value,
    blocks: &[Value],
    observed_at: DateTime<Utc>,
) -> Outcome {
    if blocks.is_empty() {
        return Outcome::Noise {
            reason: "assistant record with an empty content array".to_string(),
        };
    }
    let mut events: Vec<NormalizedEvent> = blocks
        .iter()
        .map(|block| with_context(normalize_assistant_block(key.clone(), observed_at, block), doc))
        .collect();
    if events.len() == 1 {
        return finish_single(events.pop().expect("checked len == 1"));
    }
    for (i, ev) in events.iter_mut().enumerate() {
        ev.subseq = Some(i as u32);
    }
    Outcome::Events(events)
}

fn normalize_assistant_block(
    key: RecordKey,
    observed_at: DateTime<Utc>,
    block: &Value,
) -> NormalizedEvent {
    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
    match block_type {
        "text" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Reply);
            ev.text = block.get("text").and_then(Value::as_str).map(str::to_string);
            ev
        }
        "thinking" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Thought);
            // MEASURED (render-format ticket correction): 91.8% of thinking
            // blocks are an empty string plus an opaque `signature`. Emit
            // `text: None` rather than `Some("")` — a marker, not a body.
            let raw = block.get("thinking").and_then(Value::as_str).unwrap_or("");
            ev.text = if raw.is_empty() { None } else { Some(raw.to_string()) };
            ev
        }
        "tool_use" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::ToolCall);
            ev.tool_name = block.get("name").and_then(Value::as_str).map(str::to_string);
            ev.tool_call_id = block.get("id").and_then(Value::as_str).map(str::to_string);
            ev.text = block.get("input").map(|v| v.to_string());
            ev
        }
        "fallback" => {
            let mut ev = NormalizedEvent::new(key, observed_at, EventKind::Error);
            ev.subkind = Some("model_fallback".to_string());
            // No specific field is named for a fallback block's payload in
            // the mapping table; the whole block is retained as text so
            // nothing about the fallback is lost.
            ev.text = Some(block.to_string());
            ev
        }
        other => NormalizedEvent::new(key, observed_at, EventKind::Reply).degrade(
            DerivationError::UnexpectedShape {
                record_type: "assistant".to_string(),
                field: "message.content[].type",
                expected: "\"text\", \"thinking\", \"tool_use\" or \"fallback\"",
                found: if other.is_empty() {
                    "missing `type` field".to_string()
                } else {
                    format!("block type `{other}`")
                },
            }
            .to_string(),
        ),
    }
}

fn normalize_attachment(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Context);
    match doc.get("content") {
        Some(Value::String(s)) => ev.text = Some(s.clone()),
        Some(other) => ev.text = Some(other.to_string()),
        None => {
            ev = ev.degrade(
                DerivationError::MissingField {
                    record_type: "attachment".to_string(),
                    field: "content",
                }
                .to_string(),
            );
        }
    }
    finish_single(with_context(ev, doc))
}

/// Only `subtype: "model_refusal_fallback"` is specified by the mapping
/// table. Other `system` subtypes (hook output, compaction bookkeeping —
/// see the census) are not covered by any ticket, so they are retained as
/// `Undetermined` rather than guessed at. Flagged in the report for the
/// integrator to ratify or extend.
fn normalize_system(record: &VerbatimRecord, doc: &Value, observed_at: DateTime<Utc>) -> Outcome {
    let subtype = doc.get("subtype").and_then(Value::as_str).unwrap_or("");
    if subtype != "model_refusal_fallback" {
        return Outcome::Undetermined {
            record_type: format!("system:{subtype}"),
        };
    }
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Error);
    ev.subkind = Some(subtype.to_string());
    ev.text = doc.get("content").map(|c| match c {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    finish_single(with_context(ev, doc))
}

fn normalize_session_meta(
    record: &VerbatimRecord,
    doc: &Value,
    observed_at: DateTime<Utc>,
    subkind: &str,
    text: Option<String>,
) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::SessionMeta);
    ev.subkind = Some(subkind.to_string());
    ev.text = text;
    finish_single(with_context(ev, doc))
}

fn normalize_lifecycle(
    record: &VerbatimRecord,
    doc: &Value,
    observed_at: DateTime<Utc>,
    subkind: &str,
    is_result: bool,
) -> Outcome {
    let mut ev = NormalizedEvent::new(record.key.clone(), observed_at, EventKind::Lifecycle);
    ev.subkind = Some(subkind.to_string());
    ev.agent_id = doc.get("agentId").and_then(Value::as_str).map(str::to_string);
    if is_result {
        ev.text = doc.get("result").map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });
    }
    finish_single(with_context(ev, doc))
}

fn with_context(mut event: NormalizedEvent, doc: &Value) -> NormalizedEvent {
    apply_context_fields(&mut event, doc);
    event
}
