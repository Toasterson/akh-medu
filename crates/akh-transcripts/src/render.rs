//! render — the compact, lossy one-line render of a normalized event
//! (High-Grug family, ADR 088). Grammar, tier caps, and the measurement that
//! reasoning is 0.06% of the corpus:
//! `docs/wayfinder/transcript-harvest/tickets/render-format.md`. Ported from
//! `assets/scripts/render_prototype.py`, whose numbers were validated
//! against the real corpus.
//!
//! What the render is allowed to lose (stated per the ticket, so a reader
//! knows when to page the verbatim row instead):
//! - Tier 2 (`tool_call`, `tool_result`, `context`, `session_meta`,
//!   `lifecycle`, `error`, `telemetry`) is truncated head-and-tail with an
//!   explicit `…[N elided]…` marker. Truncation is never silent.
//! - `thought` never expands its text at all, even the rare (8.2% of Claude
//!   `thinking` blocks) cases that carry real content — those are served
//!   verbatim only on an explicit request for that record.
//! - Field names are abbreviated (`t`, `n`, `a`, `o`, `sub`, `enc`).

use crate::model::EventKind;
use crate::normalize::{NormalizedEvent, Render};

/// Byte-budget tier a `NormalizedEvent`'s text falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    /// Never truncated — the decisions and plans a reader is nearly always
    /// after (`prompt`, `reply`).
    Prose,
    /// Truncated hard, head-and-tail, at the given byte cap.
    Bounded(usize),
    /// Never expanded at all, regardless of cap — `thought`.
    Marker,
}

fn tier_for(kind: EventKind) -> Tier {
    match kind {
        EventKind::Prompt | EventKind::Reply => Tier::Prose,
        EventKind::Thought => Tier::Marker,
        EventKind::ToolCall => Tier::Bounded(400),
        EventKind::ToolResult => Tier::Bounded(600),
        EventKind::Context => Tier::Bounded(200),
        EventKind::SessionMeta => Tier::Bounded(200),
        EventKind::Lifecycle => Tier::Bounded(200),
        EventKind::Error => Tier::Bounded(400),
        EventKind::Telemetry => Tier::Bounded(80),
    }
}

fn kind_label(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Prompt => "prompt",
        EventKind::Reply => "reply",
        EventKind::Thought => "thought",
        EventKind::ToolCall => "tool_call",
        EventKind::ToolResult => "tool_result",
        EventKind::Context => "context",
        EventKind::SessionMeta => "session_meta",
        EventKind::Lifecycle => "lifecycle",
        EventKind::Error => "error",
        EventKind::Telemetry => "telemetry",
    }
}

/// Collapse all whitespace (including newlines) to single spaces, so a
/// field's value can never break the one-line grammar.
fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Escape the ` | ` field separator if it survives whitespace collapsing.
fn escape_separator(s: &str) -> String {
    s.replace(" | ", " \\| ")
}

fn clamp_to_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Head-and-tail truncate whitespace-collapsed `s` to (approximately) `cap`
/// bytes, with an explicit `…[N elided]…` marker. Mirrors
/// `render_prototype.py`'s `trunc`.
fn truncate(s: &str, cap: usize) -> String {
    let collapsed = collapse_whitespace(s);
    if collapsed.len() <= cap {
        return collapsed;
    }
    let head_len = clamp_to_char_boundary(&collapsed, cap / 2);
    let tail_start = {
        let raw = collapsed.len().saturating_sub(cap - cap / 2);
        let mut idx = raw;
        while idx < collapsed.len() && !collapsed.is_char_boundary(idx) {
            idx += 1;
        }
        idx.max(head_len)
    };
    let head = &collapsed[..head_len];
    let tail = &collapsed[tail_start..];
    let elided = collapsed.len() - head.len() - tail.len();
    format!("{head}…[{elided} elided]…{tail}")
}

/// Push one `<key> <value>` field segment, escaped, skipped entirely when
/// the value is empty (mirrors the prototype, which drops falsy fields).
fn field(segments: &mut Vec<String>, key: &str, value: Option<String>) {
    if let Some(v) = value
        && !v.is_empty()
    {
        segments.push(format!("{key} {}", escape_separator(&v)));
    }
}

/// Render one normalized event as a compact, single, lossy line:
/// `<seq> <kind> | <field> <value> | ...`.
pub fn render(event: &NormalizedEvent) -> Render {
    let mut segments = vec![format!("{} {}", event.key.seq, kind_label(event.kind))];

    match tier_for(event.kind) {
        Tier::Marker => {
            // Never expand thought content, even when it exists. `enc 1`
            // marks a sealed payload (Codex: always; Claude: the 91.8% with
            // an empty field and an opaque signature).
            if event.text.is_none() {
                field(&mut segments, "enc", Some("1".to_string()));
            }
        }
        Tier::Prose => {
            field(&mut segments, "t", event.text.as_deref().map(collapse_whitespace));
        }
        Tier::Bounded(cap) => match event.kind {
            EventKind::ToolCall => {
                field(&mut segments, "n", event.tool_name.clone());
                field(&mut segments, "a", event.text.as_deref().map(|t| truncate(t, cap)));
            }
            EventKind::ToolResult => {
                field(&mut segments, "o", event.text.as_deref().map(|t| truncate(t, cap)));
            }
            _ => {
                field(&mut segments, "t", event.text.as_deref().map(|t| truncate(t, cap)));
            }
        },
    }

    field(&mut segments, "sub", event.subkind.clone());

    Render(segments.join(" | "))
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::model::{EventKind, RecordKey, Source};
    use crate::normalize::NormalizedEvent;

    use super::render;

    fn key(seq: i64) -> RecordKey {
        RecordKey {
            source: Source::ClaudeCode,
            session_id: "s1".to_string(),
            stream_id: "main".to_string(),
            seq,
        }
    }

    fn event(seq: i64, kind: EventKind) -> NormalizedEvent {
        NormalizedEvent::new(key(seq), Utc::now(), kind)
    }

    #[test]
    fn prompt_is_never_truncated() {
        let mut ev = event(412, EventKind::Prompt);
        let long = "a".repeat(500_000);
        ev.text = Some(long.clone());
        let rendered = render(&ev).0;
        assert!(rendered.contains(&long), "prose tier must not truncate");
        assert_eq!(rendered, format!("412 prompt | t {long}"));
    }

    #[test]
    fn reply_is_never_truncated() {
        let mut ev = event(1, EventKind::Reply);
        ev.text = Some("x".repeat(10_000));
        let rendered = render(&ev).0;
        assert!(!rendered.contains("elided"));
    }

    #[test]
    fn thought_never_expands_even_with_text() {
        let mut ev = event(413, EventKind::Thought);
        ev.text = Some("the model actually thought something here".to_string());
        let rendered = render(&ev).0;
        assert_eq!(rendered, "413 thought");
    }

    #[test]
    fn sealed_thought_shows_marker() {
        let ev = event(413, EventKind::Thought);
        let rendered = render(&ev).0;
        assert_eq!(rendered, "413 thought | enc 1");
    }

    #[test]
    fn sealed_thought_with_subkind_shows_both() {
        let mut ev = event(413, EventKind::Thought);
        ev.subkind = Some("encrypted".to_string());
        let rendered = render(&ev).0;
        assert_eq!(rendered, "413 thought | enc 1 | sub encrypted");
    }

    #[test]
    fn tool_call_truncates_args_at_400_with_elision_marker() {
        let mut ev = event(414, EventKind::ToolCall);
        ev.tool_name = Some("Bash".to_string());
        ev.text = Some(format!(r#"{{"command":"{}"}}"#, "x".repeat(1000)));
        let rendered = render(&ev).0;
        assert!(rendered.starts_with("414 tool_call | n Bash | a "));
        assert!(rendered.contains("…["));
        assert!(rendered.contains("elided]…"));
    }

    #[test]
    fn tool_result_truncates_output_at_600() {
        let mut ev = event(415, EventKind::ToolResult);
        ev.text = Some("y".repeat(5000));
        let rendered = render(&ev).0;
        assert!(rendered.starts_with("415 tool_result | o "));
        assert!(rendered.contains("elided]…"));
    }

    #[test]
    fn context_truncates_at_200() {
        let mut ev = event(1, EventKind::Context);
        ev.text = Some("z".repeat(1000));
        assert!(render(&ev).0.contains("elided]…"));
    }

    #[test]
    fn session_meta_truncates_at_200() {
        let mut ev = event(1, EventKind::SessionMeta);
        ev.text = Some("z".repeat(1000));
        assert!(render(&ev).0.contains("elided]…"));
    }

    #[test]
    fn lifecycle_truncates_at_200_and_renders_via_t_field() {
        let mut ev = event(416, EventKind::Lifecycle);
        ev.text = Some("agent-a44585373b94cca44".to_string());
        let rendered = render(&ev).0;
        assert_eq!(rendered, "416 lifecycle | t agent-a44585373b94cca44");
    }

    #[test]
    fn error_truncates_at_400() {
        let mut ev = event(1, EventKind::Error);
        ev.text = Some("e".repeat(1000));
        assert!(render(&ev).0.contains("elided]…"));
    }

    #[test]
    fn telemetry_truncates_at_80() {
        let mut ev = event(1, EventKind::Telemetry);
        ev.text = Some("t".repeat(1000));
        assert!(render(&ev).0.contains("elided]…"));
    }

    #[test]
    fn newlines_are_collapsed_in_values() {
        let mut ev = event(1, EventKind::Prompt);
        ev.text = Some("line one\nline two\r\n\tline three".to_string());
        let rendered = render(&ev).0;
        assert!(!rendered.contains('\n'));
        assert!(!rendered.contains('\t'));
        assert_eq!(rendered, "1 prompt | t line one line two line three");
    }

    #[test]
    fn pipe_separator_in_value_is_escaped() {
        let mut ev = event(1, EventKind::Prompt);
        ev.text = Some("a | b".to_string());
        let rendered = render(&ev).0;
        assert_eq!(rendered, "1 prompt | t a \\| b");
    }

    #[test]
    fn empty_text_field_is_omitted() {
        let ev = event(1, EventKind::Prompt);
        assert_eq!(render(&ev).0, "1 prompt");
    }

    #[test]
    fn grammar_splits_once_on_first_space_per_field() {
        let mut ev = event(414, EventKind::ToolCall);
        ev.tool_name = Some("Bash".to_string());
        ev.text = Some(r#"{"command":"cargo test"}"#.to_string());
        let rendered = render(&ev).0;
        assert_eq!(rendered, r#"414 tool_call | n Bash | a {"command":"cargo test"}"#);
    }
}
