//! Claude Code Stop-hook payload and transcript parsing.
//!
//! Claude Code sends metadata on stdin and stores the conversation itself in
//! the JSONL file named by `transcript_path`. The transcript can lag the final
//! response, so `last_assistant_message` is used as a lossless fallback for the
//! just-completed assistant turn.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::{HekaError, HekaResult};

/// A normalized turn ready for drawer ingestion.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct CaptureTurn {
    pub speaker: Option<String>,
    pub body: Option<String>,
    /// Per-turn contact override.
    pub contact: Option<String>,
    /// Per-turn topics.
    #[serde(default)]
    pub topics: Option<Vec<String>>,
}

/// Normalized capture input, independent of the hook transport format.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CaptureBatch {
    pub session_id: Option<String>,
    pub contact: Option<String>,
    pub turns: Vec<CaptureTurn>,
}

#[derive(Debug, Deserialize, Default)]
struct StopHookPayload {
    session_id: Option<String>,
    transcript_path: Option<PathBuf>,
    last_assistant_message: Option<String>,
    /// Pre-Phase-42c inline format, retained for compatibility.
    transcript: Option<Vec<CaptureTurn>>,
    /// Older alias for the inline format.
    turns: Option<Vec<CaptureTurn>>,
    contact: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TranscriptEntry {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default, rename = "isSidechain")]
    is_sidechain: bool,
    message: Option<TranscriptMessage>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<Value>,
}

/// Parse stdin accepted by `heka-capture`.
///
/// JSON objects are treated as Claude Stop-hook payloads. Non-JSON input is
/// preserved as the historical pipe-friendly single system turn.
pub fn parse_capture_input(input: &str) -> HekaResult<CaptureBatch> {
    if input.trim().is_empty() {
        return Ok(CaptureBatch::default());
    }

    if !input.trim_start().starts_with('{') {
        return Ok(CaptureBatch {
            turns: vec![CaptureTurn {
                speaker: Some("system".to_string()),
                body: Some(input.to_string()),
                ..Default::default()
            }],
            ..Default::default()
        });
    }

    let payload: StopHookPayload = serde_json::from_str(input)?;
    let mut turns = if let Some(turns) = payload.transcript.or(payload.turns) {
        turns
    } else if let Some(path) = payload.transcript_path.as_deref() {
        match std::fs::read_to_string(expand_tilde(path)) {
            Ok(transcript) => parse_transcript_jsonl(&transcript)?,
            Err(error) if useful_text(payload.last_assistant_message.as_deref()).is_some() => {
                tracing::warn!(path = %path.display(), %error, "could not read Claude transcript; using last_assistant_message");
                Vec::new()
            }
            Err(error) => return Err(HekaError::Io(error)),
        }
    } else {
        Vec::new()
    };

    // Stop-hook transcripts are written asynchronously and can omit the final
    // assistant response. Append the dedicated field unless the transcript
    // already ends with exactly that response.
    if let Some(last_message) = useful_text(payload.last_assistant_message.as_deref()) {
        let already_present = turns
            .iter()
            .rev()
            .find(|turn| turn.speaker.as_deref() == Some("assistant"))
            .is_some_and(|turn| turn.body.as_deref().map(str::trim) == Some(last_message));
        if !already_present {
            turns.push(CaptureTurn {
                speaker: Some("assistant".to_string()),
                body: Some(last_message.to_string()),
                ..Default::default()
            });
        }
    }

    turns.retain(|turn| useful_text(turn.body.as_deref()).is_some());

    Ok(CaptureBatch {
        session_id: payload.session_id,
        contact: payload.contact,
        turns,
    })
}

/// Parse Claude Code's line-delimited transcript format into conversational
/// user and assistant turns. Tool results, thinking blocks, progress events,
/// sidechains, and other bookkeeping records are deliberately excluded.
pub fn parse_transcript_jsonl(transcript: &str) -> HekaResult<Vec<CaptureTurn>> {
    let mut turns = Vec::new();
    let mut lines = transcript
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .peekable();

    while let Some((line_index, line)) = lines.next() {
        let entry: TranscriptEntry = match serde_json::from_str(line) {
            Ok(entry) => entry,
            // Claude writes transcripts asynchronously. A final partial line
            // must not discard all complete turns that preceded it.
            Err(error) if lines.peek().is_none() => {
                tracing::warn!(
                    line = line_index + 1,
                    %error,
                    "ignoring incomplete final Claude transcript line"
                );
                break;
            }
            Err(error) => {
                return Err(HekaError::Config(format!(
                    "invalid Claude transcript JSONL at line {}: {error}",
                    line_index + 1
                )));
            }
        };

        if entry.is_sidechain {
            continue;
        }

        let Some(message) = entry.message else {
            continue;
        };
        let role = message.role.or(entry.kind);
        let Some(role) = role.filter(|role| role == "user" || role == "assistant") else {
            continue;
        };
        let Some(body) = message.content.as_ref().and_then(content_text) else {
            continue;
        };

        turns.push(CaptureTurn {
            speaker: Some(role),
            body: Some(body),
            ..Default::default()
        });
    }

    Ok(turns)
}

fn content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => useful_text(Some(text)).map(ToOwned::to_owned),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        block.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            useful_text(Some(&text)).map(ToOwned::to_owned)
        }
        _ => None,
    }
}

fn useful_text(text: Option<&str>) -> Option<&str> {
    text.map(str::trim).filter(|text| !text.is_empty())
}

fn expand_tilde(path: &Path) -> PathBuf {
    let Some(path_text) = path.to_str() else {
        return path.to_path_buf();
    };
    if path_text == "~" {
        return std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(remainder) = path_text.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(remainder);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_stop_payload_reads_jsonl_and_appends_lagging_final_message() {
        let path = std::env::temp_dir().join(format!(
            "heka-claude-transcript-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let transcript = concat!(
            r#"{"type":"user","isSidechain":false,"message":{"role":"user","content":"Remember the launch code."}}"#,
            "\n",
            r#"{"type":"assistant","isSidechain":false,"message":{"role":"assistant","content":[{"type":"thinking","thinking":"private"},{"type":"text","text":"I will remember it."}]}}"#,
            "\n",
            r#"{"type":"user","isSidechain":false,"message":{"role":"user","content":[{"type":"tool_result","content":"secret tool output"}]}}"#,
            "\n",
        );
        std::fs::write(&path, transcript).unwrap();

        let input = serde_json::json!({
            "session_id": "abc123",
            "transcript_path": path.to_string_lossy(),
            "cwd": "/tmp/project",
            "permission_mode": "default",
            "hook_event_name": "Stop",
            "stop_hook_active": false,
            "last_assistant_message": "Final answer not flushed to JSONL yet.",
            "background_tasks": [],
            "session_crons": []
        })
        .to_string();

        let batch = parse_capture_input(&input).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(batch.session_id.as_deref(), Some("abc123"));
        assert_eq!(batch.turns.len(), 3);
        assert_eq!(batch.turns[0].speaker.as_deref(), Some("user"));
        assert_eq!(
            batch.turns[0].body.as_deref(),
            Some("Remember the launch code.")
        );
        assert_eq!(batch.turns[1].body.as_deref(), Some("I will remember it."));
        assert_eq!(
            batch.turns[2].body.as_deref(),
            Some("Final answer not flushed to JSONL yet.")
        );
        assert!(batch.turns.iter().all(|turn| {
            turn.body.as_deref().is_none_or(|body| {
                !body.contains("secret tool output") && !body.contains("private")
            })
        }));
    }

    #[test]
    fn final_message_is_not_duplicated_when_transcript_has_it() {
        let input = serde_json::json!({
            "session_id": "abc123",
            "transcript": [{"speaker": "assistant", "body": "done"}],
            "last_assistant_message": " done "
        })
        .to_string();

        let batch = parse_capture_input(&input).unwrap();
        assert_eq!(batch.turns.len(), 1);
    }

    #[test]
    fn repeated_final_text_is_kept_when_it_is_not_the_latest_assistant_turn() {
        let input = serde_json::json!({
            "session_id": "abc123",
            "transcript": [
                {"speaker": "assistant", "body": "same answer"},
                {"speaker": "user", "body": "please repeat it"},
                {"speaker": "assistant", "body": "intermediate text"}
            ],
            "last_assistant_message": "same answer"
        })
        .to_string();

        let batch = parse_capture_input(&input).unwrap();
        assert_eq!(batch.turns.len(), 4);
        assert_eq!(batch.turns[3].body.as_deref(), Some("same answer"));
    }

    #[test]
    fn unreadable_transcript_still_captures_stop_final_message() {
        let input = serde_json::json!({
            "session_id": "abc123",
            "transcript_path": std::env::temp_dir().join(format!(
                "missing-heka-transcript-{}.jsonl",
                uuid::Uuid::new_v4()
            )),
            "hook_event_name": "Stop",
            "last_assistant_message": "The final response survives."
        })
        .to_string();

        let batch = parse_capture_input(&input).unwrap();
        assert_eq!(batch.turns.len(), 1);
        assert_eq!(
            batch.turns[0].body.as_deref(),
            Some("The final response survives.")
        );
    }

    #[test]
    fn parses_two_hundred_realistic_conversation_rows() {
        let transcript = (0..200)
            .map(|index| {
                let role = if index % 2 == 0 { "user" } else { "assistant" };
                serde_json::json!({
                    "type": role,
                    "isSidechain": false,
                    "message": {
                        "role": role,
                        "content": format!("unique turn {index}")
                    }
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        let turns = parse_transcript_jsonl(&transcript).unwrap();
        assert_eq!(turns.len(), 200);
        assert_eq!(turns[199].body.as_deref(), Some("unique turn 199"));
    }

    #[test]
    fn preserves_inline_and_raw_pipe_compatibility() {
        let inline = parse_capture_input(
            r#"{"session_id":"legacy","turns":[{"speaker":"user","body":"hello"}]}"#,
        )
        .unwrap();
        assert_eq!(inline.turns.len(), 1);

        let raw = parse_capture_input("plain pipe input").unwrap();
        assert_eq!(raw.turns[0].speaker.as_deref(), Some("system"));
        assert_eq!(raw.turns[0].body.as_deref(), Some("plain pipe input"));
    }

    #[test]
    fn incomplete_final_jsonl_line_preserves_complete_turns() {
        let transcript = concat!(
            r#"{"type":"user","message":{"role":"user","content":"complete"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":"#,
        );
        let turns = parse_transcript_jsonl(transcript).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].body.as_deref(), Some("complete"));
    }
}
