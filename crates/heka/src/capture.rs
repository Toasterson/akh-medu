//! Claude Code Stop-hook payload and transcript parsing.
//!
//! Claude Code sends metadata on stdin and stores the conversation itself in
//! the JSONL file named by `transcript_path`. The transcript can lag the final
//! response, so `last_assistant_message` is used as a lossless fallback for the
//! just-completed assistant turn.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::{HekaError, HekaResult};

/// Stop-hook JSON is metadata, not the transcript itself. Keep stdin small so
/// a misconfigured hook cannot make the capture process allocate without bound.
pub const MAX_HOOK_STDIN_BYTES: usize = 1024 * 1024;
/// Claude transcripts can be substantial, but capture must still have a hard
/// memory and I/O ceiling.
pub const MAX_TRANSCRIPT_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_TRANSCRIPT_LINES: usize = 100_000;
pub const MAX_CAPTURE_TURNS: usize = 50_000;

/// Filesystem and resource policy for reading a Claude transcript.
#[derive(Debug, Clone)]
pub struct CapturePolicy {
    allowed_transcript_root: PathBuf,
    max_transcript_bytes: usize,
    max_transcript_lines: usize,
    max_turns: usize,
}

impl CapturePolicy {
    /// Default Claude Code transcript root for the current user.
    pub fn for_current_user() -> HekaResult<Self> {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            HekaError::Config(
                "HOME is not set; cannot locate the Claude transcript root".to_string(),
            )
        })?;
        Ok(Self::new(PathBuf::from(home).join(".claude/projects")))
    }

    /// Construct a policy for a trusted transcript root. This is primarily
    /// useful for a deliberately isolated test or deployment root.
    pub fn new(allowed_transcript_root: PathBuf) -> Self {
        Self {
            allowed_transcript_root,
            max_transcript_bytes: MAX_TRANSCRIPT_BYTES,
            max_transcript_lines: MAX_TRANSCRIPT_LINES,
            max_turns: MAX_CAPTURE_TURNS,
        }
    }

    #[cfg(test)]
    fn with_limits(mut self, max_bytes: usize, max_lines: usize, max_turns: usize) -> Self {
        self.max_transcript_bytes = max_bytes;
        self.max_transcript_lines = max_lines;
        self.max_turns = max_turns;
        self
    }
}

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
    cwd: Option<PathBuf>,
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
    #[serde(default, rename = "isMeta")]
    is_meta: bool,
    #[serde(default, rename = "isCompactSummary")]
    is_compact_summary: bool,
    #[serde(rename = "sessionId")]
    session_id: Option<String>,
    cwd: Option<PathBuf>,
    message: Option<TranscriptMessage>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<Value>,
}

/// Read UTF-8 hook input through a hard byte ceiling.
pub fn read_hook_input(reader: impl Read) -> HekaResult<String> {
    read_utf8_bounded(reader, MAX_HOOK_STDIN_BYTES, "hook stdin")
}

/// Parse stdin accepted by `heka-capture` using the current user's canonical
/// Claude transcript root.
pub fn parse_capture_input(input: &str) -> HekaResult<CaptureBatch> {
    let policy = CapturePolicy::for_current_user()?;
    parse_capture_input_with_policy(input, &policy)
}

/// Parse hook input with an explicit trusted transcript policy.
///
/// JSON objects are treated as Claude Stop-hook payloads. Non-JSON input is
/// preserved as the historical pipe-friendly single system turn.
pub fn parse_capture_input_with_policy(
    input: &str,
    policy: &CapturePolicy,
) -> HekaResult<CaptureBatch> {
    if input.len() > MAX_HOOK_STDIN_BYTES {
        return Err(resource_limit_error("hook stdin", MAX_HOOK_STDIN_BYTES));
    }
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
        if turns.len() > policy.max_turns {
            return Err(resource_limit_error(
                "inline capture turns",
                policy.max_turns,
            ));
        }
        turns
    } else if let Some(path) = payload.transcript_path.as_deref() {
        let session_id = validate_hook_session_id(payload.session_id.as_deref())?;
        let cwd = payload.cwd.as_deref().ok_or_else(|| {
            HekaError::Config("Claude Stop payload with transcript_path is missing cwd".to_string())
        })?;
        let (transcript, canonical_cwd) = read_validated_transcript(path, session_id, cwd, policy)?;
        parse_transcript_jsonl_for_session(
            &transcript,
            session_id,
            &canonical_cwd,
            policy.max_transcript_lines,
            policy.max_turns,
        )?
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

    if turns.len() > policy.max_turns {
        return Err(resource_limit_error("captured turns", policy.max_turns));
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
    parse_transcript_jsonl_inner(
        transcript,
        None,
        None,
        MAX_TRANSCRIPT_LINES,
        MAX_CAPTURE_TURNS,
    )
}

fn parse_transcript_jsonl_for_session(
    transcript: &str,
    expected_session_id: &str,
    expected_cwd: &Path,
    max_lines: usize,
    max_turns: usize,
) -> HekaResult<Vec<CaptureTurn>> {
    parse_transcript_jsonl_inner(
        transcript,
        Some(expected_session_id),
        Some(expected_cwd),
        max_lines,
        max_turns,
    )
}

fn parse_transcript_jsonl_inner(
    transcript: &str,
    expected_session_id: Option<&str>,
    expected_cwd: Option<&Path>,
    max_lines: usize,
    max_turns: usize,
) -> HekaResult<Vec<CaptureTurn>> {
    let mut turns = Vec::new();
    let mut lines = transcript.lines().enumerate().peekable();

    while let Some((line_index, line)) = lines.next() {
        if line_index >= max_lines {
            return Err(resource_limit_error("Claude transcript lines", max_lines));
        }
        if line.trim().is_empty() {
            continue;
        }
        let entry: TranscriptEntry = match serde_json::from_str(line) {
            Ok(entry) => entry,
            // Claude writes transcripts asynchronously. A final partial line
            // must not discard all complete turns that preceded it.
            Err(error) if lines.peek().is_none() && error.is_eof() => {
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

        if entry.is_sidechain || entry.is_meta || entry.is_compact_summary {
            continue;
        }

        let Some(message) = entry.message else {
            continue;
        };
        let role = message.role.or(entry.kind);
        let Some(role) = role.filter(|role| role == "user" || role == "assistant") else {
            continue;
        };
        if let Some(expected) = expected_session_id {
            match entry.session_id.as_deref() {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(HekaError::Config(format!(
                        "Claude transcript session mismatch at line {}: expected `{expected}`, got `{actual}`",
                        line_index + 1
                    )));
                }
                None => {
                    return Err(HekaError::Config(format!(
                        "Claude conversational entry at line {} has no sessionId",
                        line_index + 1
                    )));
                }
            }
        }
        if let Some(expected) = expected_cwd {
            validate_transcript_entry_cwd(entry.cwd.as_deref(), expected, line_index + 1)?;
        }
        let Some(body) = message.content.as_ref().and_then(content_text) else {
            continue;
        };

        turns.push(CaptureTurn {
            speaker: Some(role),
            body: Some(body),
            ..Default::default()
        });
        if turns.len() > max_turns {
            return Err(resource_limit_error("captured turns", max_turns));
        }
    }

    Ok(turns)
}

fn read_validated_transcript(
    path: &Path,
    session_id: &str,
    cwd: &Path,
    policy: &CapturePolicy,
) -> HekaResult<(String, PathBuf)> {
    let raw_root = expand_tilde(&policy.allowed_transcript_root);
    let raw_path = expand_tilde(path);
    let relative = raw_path.strip_prefix(&raw_root).map_err(|_| {
        HekaError::Config(format!(
            "Claude transcript must be beneath the allowed root {}",
            raw_root.display()
        ))
    })?;
    if relative.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(HekaError::Config(
            "Claude transcript path contains a disallowed path component".to_string(),
        ));
    }

    let mut cursor = raw_root.clone();
    for component in relative.components() {
        cursor.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&cursor)?;
        if metadata.file_type().is_symlink() {
            return Err(HekaError::Config(format!(
                "Claude transcript path contains a symlink: {}",
                cursor.display()
            )));
        }
    }

    let root = raw_root.canonicalize()?;
    let transcript_path = raw_path.canonicalize()?;
    let cwd = expand_tilde(cwd).canonicalize()?;
    if !cwd.is_dir() {
        return Err(HekaError::Config(
            "Claude Stop payload cwd is not a directory".to_string(),
        ));
    }

    let expected_project = root.join(claude_project_dir_name(&cwd));
    if transcript_path.parent() != Some(expected_project.as_path()) {
        return Err(HekaError::Config(format!(
            "Claude transcript is not bound to the Stop payload cwd under {}",
            root.display()
        )));
    }
    let expected_filename = format!("{session_id}.jsonl");
    if transcript_path.file_name().and_then(|name| name.to_str()) != Some(&expected_filename) {
        return Err(HekaError::Config(
            "Claude transcript filename does not match the Stop payload session_id".to_string(),
        ));
    }

    let metadata = std::fs::metadata(&transcript_path)?;
    if !metadata.is_file() {
        return Err(HekaError::Config(
            "Claude transcript_path is not a regular file".to_string(),
        ));
    }
    if metadata.len() > policy.max_transcript_bytes as u64 {
        return Err(resource_limit_error(
            "Claude transcript bytes",
            policy.max_transcript_bytes,
        ));
    }

    let file = open_transcript_without_following(&transcript_path)?;
    let opened_metadata = file.metadata()?;
    if !opened_metadata.is_file() {
        return Err(HekaError::Config(
            "Claude transcript_path changed and is no longer a regular file".to_string(),
        ));
    }
    if !same_file_identity(&metadata, &opened_metadata) {
        return Err(HekaError::Config(
            "Claude transcript_path changed while it was being opened".to_string(),
        ));
    }
    let transcript =
        read_utf8_bounded(file, policy.max_transcript_bytes, "Claude transcript bytes")?;
    Ok((transcript, cwd))
}

fn validate_transcript_entry_cwd(
    entry_cwd: Option<&Path>,
    expected_cwd: &Path,
    line_number: usize,
) -> HekaResult<()> {
    let entry_cwd = entry_cwd.ok_or_else(|| {
        HekaError::Config(format!(
            "Claude transcript entry at line {line_number} has no cwd"
        ))
    })?;
    let canonical_entry_cwd = expand_tilde(entry_cwd).canonicalize().map_err(|error| {
        HekaError::Config(format!(
            "Claude transcript cwd at line {line_number} cannot be canonicalized: {error}"
        ))
    })?;
    if canonical_entry_cwd != expected_cwd {
        return Err(HekaError::Config(format!(
            "Claude transcript cwd mismatch at line {line_number}"
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn open_transcript_without_following(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        // Non-blocking prevents a path swapped to a FIFO from hanging the Stop
        // hook. O_NOFOLLOW closes the final-component symlink race; the inode
        // comparison below covers parent-component replacement.
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

#[cfg(not(unix))]
fn open_transcript_without_following(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

#[cfg(unix)]
fn same_file_identity(before: &std::fs::Metadata, opened: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    before.dev() == opened.dev() && before.ino() == opened.ino()
}

#[cfg(not(unix))]
fn same_file_identity(_before: &std::fs::Metadata, _opened: &std::fs::Metadata) -> bool {
    true
}

fn validate_hook_session_id(session_id: Option<&str>) -> HekaResult<&str> {
    let session_id = session_id
        .filter(|session_id| {
            !session_id.is_empty()
                && session_id.len() <= 128
                && session_id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        })
        .ok_or_else(|| {
            HekaError::Config(
                "Claude Stop payload has an invalid or missing session_id".to_string(),
            )
        })?;
    Ok(session_id)
}

fn claude_project_dir_name(cwd: &Path) -> String {
    cwd.to_string_lossy().replace(['/', '\\'], "-")
}

fn read_utf8_bounded(
    mut reader: impl Read,
    max_bytes: usize,
    resource: &str,
) -> HekaResult<String> {
    let mut bytes = Vec::with_capacity(max_bytes.min(64 * 1024));
    reader
        .by_ref()
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(resource_limit_error(resource, max_bytes));
    }
    String::from_utf8(bytes)
        .map_err(|_| HekaError::Config(format!("{resource} is not valid UTF-8")))
}

fn resource_limit_error(resource: &str, limit: usize) -> HekaError {
    HekaError::Config(format!(
        "{resource} exceeds the configured limit of {limit}"
    ))
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

    struct HookLayout {
        base: PathBuf,
        root: PathBuf,
        cwd: PathBuf,
        session_id: String,
        transcript: PathBuf,
    }

    impl HookLayout {
        fn new() -> Self {
            let base = std::env::temp_dir()
                .join(format!("heka-capture-policy-test-{}", uuid::Uuid::new_v4()));
            let root = base.join("claude-projects");
            let cwd = base.join("workspace");
            std::fs::create_dir_all(&root).unwrap();
            std::fs::create_dir_all(&cwd).unwrap();
            let cwd = cwd.canonicalize().unwrap();
            let project = root.join(claude_project_dir_name(&cwd));
            std::fs::create_dir_all(&project).unwrap();
            let session_id = uuid::Uuid::new_v4().to_string();
            let transcript = project.join(format!("{session_id}.jsonl"));
            Self {
                base,
                root,
                cwd,
                session_id,
                transcript,
            }
        }

        fn policy(&self) -> CapturePolicy {
            CapturePolicy::new(self.root.clone())
        }

        fn payload(&self, last_assistant_message: Option<&str>) -> String {
            serde_json::json!({
                "session_id": self.session_id,
                "transcript_path": self.transcript,
                "cwd": self.cwd,
                "hook_event_name": "Stop",
                "last_assistant_message": last_assistant_message,
            })
            .to_string()
        }
    }

    impl Drop for HookLayout {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    fn inline_policy() -> CapturePolicy {
        CapturePolicy::new(std::env::temp_dir().join("unused-claude-projects"))
    }

    #[test]
    fn real_stop_payload_reads_jsonl_and_appends_lagging_final_message() {
        let layout = HookLayout::new();
        let records = [
            serde_json::json!({
                "type": "user", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isSidechain": false,
                "message": {"role": "user", "content": "Remember the launch code."}
            }),
            serde_json::json!({
                "type": "assistant", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isSidechain": false,
                "message": {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "I will remember it."}
                ]}
            }),
            serde_json::json!({
                "type": "user", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isMeta": true,
                "message": {"role": "user", "content": "meta injection"}
            }),
            serde_json::json!({
                "type": "assistant", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isCompactSummary": true,
                "message": {"role": "assistant", "content": "compaction summary"}
            }),
            serde_json::json!({
                "type": "user", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isSidechain": false,
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "content": "secret tool output"}
                ]}
            }),
            // Real Claude transcripts contain bookkeeping rows without a
            // sessionId/cwd. They carry no conversational content and must be
            // ignored rather than making an otherwise valid capture fail.
            serde_json::json!({
                "type": "file-history-snapshot",
                "snapshot": {"trackedFileBackups": {}}
            }),
        ];
        let transcript = records
            .into_iter()
            .map(|record| record.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&layout.transcript, transcript).unwrap();

        let input = layout.payload(Some("Final answer not flushed to JSONL yet."));
        let batch = parse_capture_input_with_policy(&input, &layout.policy()).unwrap();

        assert_eq!(
            batch.session_id.as_deref(),
            Some(layout.session_id.as_str())
        );
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
                !body.contains("secret tool output")
                    && !body.contains("private")
                    && !body.contains("meta injection")
                    && !body.contains("compaction summary")
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

        let batch = parse_capture_input_with_policy(&input, &inline_policy()).unwrap();
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

        let batch = parse_capture_input_with_policy(&input, &inline_policy()).unwrap();
        assert_eq!(batch.turns.len(), 4);
        assert_eq!(batch.turns[3].body.as_deref(), Some("same answer"));
    }

    #[test]
    fn arbitrary_unreadable_path_is_rejected_even_with_a_final_message() {
        let layout = HookLayout::new();
        let input = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": std::env::temp_dir().join(format!(
                "missing-heka-transcript-{}.jsonl",
                uuid::Uuid::new_v4()
            )),
            "cwd": layout.cwd,
            "hook_event_name": "Stop",
            "last_assistant_message": "The final response survives."
        })
        .to_string();

        assert!(parse_capture_input_with_policy(&input, &layout.policy()).is_err());
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
        let inline = parse_capture_input_with_policy(
            r#"{"session_id":"legacy","turns":[{"speaker":"user","body":"hello"}]}"#,
            &inline_policy(),
        )
        .unwrap();
        assert_eq!(inline.turns.len(), 1);

        let raw = parse_capture_input_with_policy("plain pipe input", &inline_policy()).unwrap();
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

    #[test]
    fn malformed_complete_final_jsonl_line_fails_closed() {
        let transcript = concat!(
            r#"{"type":"user","message":{"role":"user","content":"complete"}}"#,
            "\n",
            r#"{"type":"assistant","message":42}"#,
        );
        assert!(parse_transcript_jsonl(transcript).is_err());
    }

    #[test]
    fn mismatched_entry_session_fails_closed() {
        let layout = HookLayout::new();
        let record = serde_json::json!({
            "type": "user",
            "sessionId": "other-session",
            "cwd": layout.cwd,
            "message": {"role": "user", "content": "poison"},
        });
        std::fs::write(&layout.transcript, record.to_string()).unwrap();

        let error = parse_capture_input_with_policy(&layout.payload(None), &layout.policy())
            .unwrap_err()
            .to_string();
        assert!(error.contains("session mismatch"));
    }

    #[test]
    fn mismatched_entry_cwd_fails_closed_even_when_encoded_project_names_collide() {
        let layout = HookLayout::new();
        let colliding_cwd = layout.base.join("work-space");
        let expected_cwd = layout.base.join("work/space");
        std::fs::create_dir_all(&colliding_cwd).unwrap();
        std::fs::create_dir_all(&expected_cwd).unwrap();
        let colliding_cwd = colliding_cwd.canonicalize().unwrap();
        let expected_cwd = expected_cwd.canonicalize().unwrap();
        assert_eq!(
            claude_project_dir_name(&colliding_cwd),
            claude_project_dir_name(&expected_cwd)
        );

        let project = layout.root.join(claude_project_dir_name(&expected_cwd));
        std::fs::create_dir_all(&project).unwrap();
        let transcript = project.join(format!("{}.jsonl", layout.session_id));
        let record = serde_json::json!({
            "type": "user",
            "sessionId": layout.session_id,
            "cwd": colliding_cwd,
            "message": {"role": "user", "content": "wrong project"},
        });
        std::fs::write(&transcript, record.to_string()).unwrap();
        let payload = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": transcript,
            "cwd": expected_cwd,
        })
        .to_string();

        let error = parse_capture_input_with_policy(&payload, &layout.policy())
            .unwrap_err()
            .to_string();
        assert!(error.contains("cwd mismatch"));
    }

    #[test]
    fn project_and_filename_binding_fail_closed() {
        let layout = HookLayout::new();
        let other_project = layout.root.join("other-project");
        std::fs::create_dir_all(&other_project).unwrap();
        let wrong_path = other_project.join(format!("{}.jsonl", layout.session_id));
        std::fs::write(&wrong_path, "{}").unwrap();
        let wrong_project = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": wrong_path,
            "cwd": layout.cwd,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&wrong_project, &layout.policy()).is_err());

        std::fs::write(&layout.transcript, "{}").unwrap();
        let wrong_session = serde_json::json!({
            "session_id": "different-session",
            "transcript_path": layout.transcript,
            "cwd": layout.cwd,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&wrong_session, &layout.policy()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_and_fifos_are_rejected_without_opening_them() {
        use std::os::unix::fs::symlink;

        let layout = HookLayout::new();
        let target = layout.transcript.with_extension("target");
        std::fs::write(&target, "{}").unwrap();
        symlink(&target, &layout.transcript).unwrap();
        assert!(parse_capture_input_with_policy(&layout.payload(None), &layout.policy()).is_err());

        std::fs::remove_file(&layout.transcript).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(&layout.transcript)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(parse_capture_input_with_policy(&layout.payload(None), &layout.policy()).is_err());
    }

    #[test]
    fn stdin_transcript_line_and_turn_limits_are_enforced() {
        let oversized = vec![b'x'; MAX_HOOK_STDIN_BYTES + 1];
        assert!(read_hook_input(std::io::Cursor::new(oversized)).is_err());

        let layout = HookLayout::new();
        let first = serde_json::json!({
            "type": "user", "sessionId": layout.session_id,
            "cwd": layout.cwd,
            "message": {"role": "user", "content": "first"}
        });
        let second = serde_json::json!({
            "type": "assistant", "sessionId": layout.session_id,
            "cwd": layout.cwd,
            "message": {"role": "assistant", "content": "second"}
        });
        std::fs::write(&layout.transcript, format!("{}\n{}", first, second)).unwrap();

        let line_policy = layout.policy().with_limits(1024, 1, 10);
        assert!(parse_capture_input_with_policy(&layout.payload(None), &line_policy).is_err());

        let turn_policy = layout.policy().with_limits(1024, 10, 1);
        assert!(parse_capture_input_with_policy(&layout.payload(None), &turn_policy).is_err());

        let byte_policy = layout.policy().with_limits(4, 10, 10);
        assert!(parse_capture_input_with_policy(&layout.payload(None), &byte_policy).is_err());

        // Empty rows still consume parsing work and count toward the physical
        // line ceiling; an all-whitespace file cannot bypass the CPU guard.
        assert!(parse_transcript_jsonl_inner("\n\n", None, None, 1, 10).is_err());
    }
}
