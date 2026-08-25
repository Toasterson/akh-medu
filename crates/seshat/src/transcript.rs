//! Agent-session transcript extraction.
//!
//! Anima uploads completed coding-agent sessions as line-delimited JSON.  The
//! event envelope is useful for replay, but embedding that envelope verbatim
//! makes IDs, timestamps, tool arguments, and protocol keys dominate search.
//! This module extracts only the durable knowledge-bearing parts: assistant
//! text and tool-result text.

use serde_json::Value;

use crate::error::{SeshatError, SeshatResult};
use crate::ingest::{ChunkDraft, chunk_paragraph_aware};

/// Format value used by Anima's transcript producer.
pub const AGENT_SESSION_TRANSCRIPT_FORMAT: &str = "agent_session_transcript";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    Assistant,
    ToolResult,
}

impl SegmentKind {
    fn heading(self) -> &'static str {
        match self {
            Self::Assistant => "Assistant",
            Self::ToolResult => "Tool result",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptSegment {
    kind: SegmentKind,
    text: String,
}

/// Extract and chunk a completed Anima agent-session transcript.
///
/// `metadata` and `tags` are not modified; ingestion stores them verbatim on
/// the document.  A compact copy of the searchable identity fields is added
/// to each chunk so an ordinary semantic query can find a session by work
/// item, executor, outcome, or tag.  Structured search filters use the
/// original JSONB metadata instead (see `query::SearchFilters`).
pub fn chunk_agent_session_transcript(
    body: &str,
    target_chars: usize,
    tags: &[String],
    metadata: &Value,
) -> SeshatResult<Vec<ChunkDraft>> {
    let segments = extract_segments(body)?;
    if segments.is_empty() {
        return Err(SeshatError::Config(
            "ingest: agent_session_transcript contains no assistant or tool-result text"
                .to_string(),
        ));
    }

    let context = searchable_context(tags, metadata);
    let context_chars = context.chars().count();
    let content_target = target_chars.saturating_sub(context_chars.saturating_add(2));
    let mut chunks = Vec::new();

    for segment in segments {
        for mut chunk in chunk_paragraph_aware(&segment.text, content_target, "text") {
            if !context.is_empty() {
                chunk.text = format!("{context}\n\n{}", chunk.text);
                chunk.word_count = chunk.text.split_whitespace().count() as i32;
            }
            chunk.heading = Some(segment.kind.heading().to_string());
            chunks.push(chunk);
        }
    }

    Ok(chunks)
}

fn extract_segments(body: &str) -> SeshatResult<Vec<TranscriptSegment>> {
    let mut segments = Vec::new();

    for (line_index, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(line).map_err(|error| {
            SeshatError::Config(format!(
                "ingest: invalid agent_session_transcript JSONL at line {}: {error}",
                line_index + 1
            ))
        })?;

        let event_type = event.get("type").and_then(Value::as_str);
        let message = event.get("message").filter(|value| value.is_object());
        let role = message
            .and_then(|value| value.get("role"))
            .or_else(|| event.get("role"))
            .and_then(Value::as_str);
        let content = message
            .and_then(|value| value.get("content"))
            .or_else(|| event.get("content"));

        if event_type == Some("assistant") || role == Some("assistant") {
            if let Some(text) = assistant_text(content) {
                push_segment(&mut segments, SegmentKind::Assistant, text);
            }
        }

        // In Claude stream JSONL, tool results are content blocks on a user
        // message.  Direct tool/tool_result rows are also accepted so the
        // format remains useful for other ACP executors without admitting
        // arbitrary user-message text.
        if let Some(Value::Array(blocks)) = content {
            for block in blocks {
                if matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("tool_result" | "tool-result")
                ) && let Some(text) = content_text(block.get("content"))
                {
                    push_segment(&mut segments, SegmentKind::ToolResult, text);
                }
            }
        }

        if matches!(event_type, Some("tool_result" | "tool-result"))
            || matches!(role, Some("tool" | "tool_result"))
        {
            if let Some(text) = content_text(content) {
                push_segment(&mut segments, SegmentKind::ToolResult, text);
            }
        }
    }

    Ok(segments)
}

fn push_segment(segments: &mut Vec<TranscriptSegment>, kind: SegmentKind, text: String) {
    let text = text.trim();
    if !text.is_empty() {
        segments.push(TranscriptSegment {
            kind,
            text: text.to_string(),
        });
    }
}

fn assistant_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => useful_text(text),
        Value::Array(blocks) => joined_text(blocks.iter().filter_map(|block| {
            if matches!(
                block.get("type").and_then(Value::as_str),
                Some("text" | "output_text")
            ) {
                block.get("text").and_then(Value::as_str)
            } else {
                None
            }
        })),
        _ => None,
    }
}

fn content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => useful_text(text),
        Value::Array(blocks) => joined_text(blocks.iter().filter_map(|block| match block {
            Value::String(text) => Some(text.as_str()),
            Value::Object(_) => block.get("text").and_then(Value::as_str).filter(|_| {
                block
                    .get("type")
                    .and_then(Value::as_str)
                    .is_none_or(|kind| matches!(kind, "text" | "output_text"))
            }),
            _ => None,
        })),
        _ => None,
    }
}

fn joined_text<'a>(parts: impl Iterator<Item = &'a str>) -> Option<String> {
    useful_text(&parts.collect::<Vec<_>>().join("\n"))
}

fn useful_text(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn searchable_context(tags: &[String], metadata: &Value) -> String {
    let mut lines = Vec::new();
    for (label, key) in [
        ("Work item", "work_item_id"),
        ("Executor", "executor"),
        ("Outcome", "outcome"),
    ] {
        if let Some(value) = metadata_scalar(metadata, key) {
            lines.push(format!("{label}: {value}"));
        }
    }
    let tags = tags
        .iter()
        .map(|tag| tag.trim())
        .filter(|tag| !tag.is_empty())
        .collect::<Vec<_>>();
    if !tags.is_empty() {
        lines.push(format!("Tags: {}", tags.join(", ")));
    }
    lines.join("\n")
}

fn metadata_scalar(metadata: &Value, key: &str) -> Option<String> {
    match metadata.get(key)? {
        Value::String(value) => useful_text(value),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPLETED_SESSION: &str = include_str!("../tests/fixtures/completed_agent_session.jsonl");

    fn metadata() -> Value {
        serde_json::json!({
            "work_item_id": "01a03511-6678-7e93-89d7-64fdb0033c44",
            "executor": "codex",
            "outcome": "completed",
            "anima_private_field": { "preserved": true }
        })
    }

    #[test]
    fn ingestion_extracts_knowledge_without_json_event_scaffolding() {
        let tags = vec!["WI-341".to_string(), "agent-session".to_string()];
        let chunks = chunk_agent_session_transcript(COMPLETED_SESSION, 1500, &tags, &metadata())
            .expect("representative completed transcript should ingest");
        let text = chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("Implemented metadata-preserving transcript extraction."));
        assert!(text.contains("test result: 17 passed; 0 failed"));
        assert!(text.contains("The targeted ingestion and search tests pass."));
        assert!(!text.contains("Please implement the transcript indexer"));
        assert!(!text.contains("private chain of thought"));
        assert!(!text.contains("cargo test -p seshat"));
        assert!(!text.contains("session_id"));
        assert!(!text.contains("duration_ms"));
        assert!(!text.contains("tool_use_id"));
        assert!(
            chunks
                .iter()
                .all(|chunk| matches!(chunk.heading.as_deref(), Some("Assistant" | "Tool result")))
        );
    }

    #[test]
    fn indexed_chunks_are_searchable_by_anima_identity_fields_and_tags() {
        let tags = vec!["WI-341".to_string(), "agent-session".to_string()];
        let chunks = chunk_agent_session_transcript(COMPLETED_SESSION, 1500, &tags, &metadata())
            .expect("representative completed transcript should ingest");

        for term in [
            "01a03511-6678-7e93-89d7-64fdb0033c44",
            "codex",
            "completed",
            "WI-341",
        ] {
            assert!(
                chunks.iter().any(|chunk| chunk.text.contains(term)),
                "expected indexed text to contain {term}"
            );
        }
    }

    #[test]
    fn rejects_invalid_jsonl_instead_of_indexing_it_as_opaque_text() {
        let error = chunk_agent_session_transcript(
            "{\"type\":\"assistant\",not-json}",
            1500,
            &[],
            &serde_json::json!({}),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("line 1"), "{error}");
    }

    #[test]
    fn accepts_direct_tool_result_rows() {
        let body = concat!(
            r#"{"type":"tool_result","content":[{"type":"text","text":"direct output"}]}"#,
            "\n"
        );
        let chunks = chunk_agent_session_transcript(body, 1500, &[], &serde_json::json!({}))
            .expect("direct tool result should ingest");
        assert_eq!(chunks[0].text, "direct output");
        assert_eq!(chunks[0].heading.as_deref(), Some("Tool result"));
    }
}
