//! Claude Code Stop-hook payload and transcript parsing.
//!
//! Claude Code sends metadata on stdin and stores the conversation itself in
//! the JSONL file named by `transcript_path`. The transcript can lag the final
//! response, so `last_assistant_message` is used as a lossless fallback for the
//! just-completed assistant turn.

use std::ffi::{OsStr, OsString};
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
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
    // Real Claude transcripts can move between a project root, nested
    // directories, and alternate worktrees within one stable session. The
    // first exact-session conversational cwd binds the opened project; every
    // later exact-session cwd is eligible evidence for the Stop cwd without
    // filtering otherwise valid turns.
    cwd: Option<PathBuf>,
    message: Option<TranscriptMessage>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<Value>,
}

#[derive(Debug)]
struct ValidatedTranscript {
    contents: String,
    project_component: OsString,
    stop_cwd: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct TranscriptAuthorization<'a> {
    project_component: &'a OsStr,
    stop_cwd: &'a Path,
}

/// Read UTF-8 hook input through a hard byte ceiling.
pub fn read_hook_input(reader: impl Read) -> HekaResult<String> {
    read_utf8_bounded(reader, MAX_HOOK_STDIN_BYTES, "hook stdin")
}

/// Parse stdin accepted by `heka-capture` using the current user's canonical
/// Claude transcript root.
pub fn parse_capture_input(input: &str) -> HekaResult<CaptureBatch> {
    parse_capture_input_with_policy_factory(input, None, CapturePolicy::for_current_user)
}

/// Parse hook input with an explicit trusted transcript policy.
///
/// JSON objects are treated as Claude Stop-hook payloads. Non-JSON input is
/// preserved as the historical pipe-friendly single system turn.
pub fn parse_capture_input_with_policy(
    input: &str,
    policy: &CapturePolicy,
) -> HekaResult<CaptureBatch> {
    parse_capture_input_with_policy_factory(input, Some(policy), || {
        unreachable!("an explicit capture policy was already supplied")
    })
}

fn parse_capture_input_with_policy_factory<F>(
    input: &str,
    explicit_policy: Option<&CapturePolicy>,
    default_policy: F,
) -> HekaResult<CaptureBatch>
where
    F: FnOnce() -> HekaResult<CapturePolicy>,
{
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
    let max_turns = explicit_policy.map_or(MAX_CAPTURE_TURNS, |policy| policy.max_turns);
    let mut turns = if let Some(turns) = payload.transcript.or(payload.turns) {
        if turns.len() > max_turns {
            return Err(resource_limit_error("inline capture turns", max_turns));
        }
        turns
    } else if let Some(path) = payload.transcript_path.as_deref() {
        let owned_policy;
        let policy = if let Some(policy) = explicit_policy {
            policy
        } else {
            owned_policy = default_policy()?;
            &owned_policy
        };
        let session_id = validate_hook_session_id(payload.session_id.as_deref())?;
        let cwd = payload.cwd.as_deref().ok_or_else(|| {
            HekaError::Config(
                "Claude Stop payload with transcript_path is missing project cwd".to_string(),
            )
        })?;
        let transcript = read_validated_transcript(path, session_id, cwd, policy)?;
        parse_transcript_jsonl_for_session(
            &transcript.contents,
            session_id,
            TranscriptAuthorization {
                project_component: &transcript.project_component,
                stop_cwd: &transcript.stop_cwd,
            },
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
        let already_present = turns.last().is_some_and(|turn| {
            turn.speaker.as_deref() == Some("assistant")
                && turn.body.as_deref().map(str::trim) == Some(last_message)
        });
        if !already_present {
            turns.push(CaptureTurn {
                speaker: Some("assistant".to_string()),
                body: Some(last_message.to_string()),
                ..Default::default()
            });
        }
    }

    if turns.len() > max_turns {
        return Err(resource_limit_error("captured turns", max_turns));
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
    authorization: TranscriptAuthorization<'_>,
    max_lines: usize,
    max_turns: usize,
) -> HekaResult<Vec<CaptureTurn>> {
    parse_transcript_jsonl_inner(
        transcript,
        Some(expected_session_id),
        Some(authorization),
        max_lines,
        max_turns,
    )
}

fn parse_transcript_jsonl_inner(
    transcript: &str,
    expected_session_id: Option<&str>,
    authorization: Option<TranscriptAuthorization<'_>>,
    max_lines: usize,
    max_turns: usize,
) -> HekaResult<Vec<CaptureTurn>> {
    let mut turns = Vec::new();
    let mut saw_conversational_row = false;
    let mut stop_cwd_recorded = false;
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
                Some(_) => {
                    return Err(HekaError::Config(format!(
                        "Claude transcript session mismatch at line {}",
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

        if let Some(authorization) = authorization {
            let canonical_row_cwd = match entry.cwd.as_deref() {
                Some(row_cwd) => Some(canonical_transcript_cwd(row_cwd, line_index + 1)?),
                None if !saw_conversational_row => {
                    return Err(HekaError::Config(format!(
                        "first Claude conversational entry at line {} has no cwd",
                        line_index + 1
                    )));
                }
                None => None,
            };

            if !saw_conversational_row {
                let first_cwd = canonical_row_cwd.as_deref().ok_or_else(|| {
                    HekaError::Config(
                        "first Claude conversational entry has no canonical cwd".to_string(),
                    )
                })?;
                if authorization.project_component
                    != OsStr::new(&claude_project_dir_name(first_cwd))
                {
                    return Err(HekaError::Config(
                        "Claude transcript project does not match its first conversational cwd"
                            .to_string(),
                    ));
                }
            }
            if canonical_row_cwd.as_deref() == Some(authorization.stop_cwd) {
                stop_cwd_recorded = true;
            }
        }
        saw_conversational_row = true;

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

    if authorization.is_some() {
        if !saw_conversational_row {
            return Err(HekaError::Config(
                "Claude transcript contains no conversational project evidence".to_string(),
            ));
        }
        if !stop_cwd_recorded {
            return Err(HekaError::Config(
                "Claude Stop project cwd is not recorded by the selected session".to_string(),
            ));
        }
    }

    Ok(turns)
}

fn canonical_transcript_cwd(cwd: &Path, line: usize) -> HekaResult<PathBuf> {
    let raw_cwd = expand_tilde(cwd);
    if !raw_cwd.is_absolute() {
        return Err(HekaError::Config(format!(
            "Claude conversational entry at line {line} has a non-absolute cwd"
        )));
    }
    let canonical = raw_cwd.canonicalize().map_err(|error| {
        HekaError::Config(format!(
            "Claude conversational entry cwd at line {line} cannot be canonicalized: {error}"
        ))
    })?;
    if !canonical.is_dir() {
        return Err(HekaError::Config(format!(
            "Claude conversational entry cwd at line {line} is not a directory"
        )));
    }
    Ok(canonical)
}

fn read_validated_transcript(
    path: &Path,
    session_id: &str,
    cwd: &Path,
    policy: &CapturePolicy,
) -> HekaResult<ValidatedTranscript> {
    let raw_root = expand_tilde(&policy.allowed_transcript_root);
    let raw_path = expand_tilde(path);
    let raw_cwd = expand_tilde(cwd);
    if !raw_root.is_absolute() || !raw_path.is_absolute() || !raw_cwd.is_absolute() {
        return Err(HekaError::Config(
            "Claude transcript root, transcript_path, and Stop project cwd must be absolute"
                .to_string(),
        ));
    }
    let relative = raw_path.strip_prefix(&raw_root).map_err(|_| {
        HekaError::Config(format!(
            "Claude transcript must be beneath the allowed root {}",
            raw_root.display()
        ))
    })?;
    let relative_components = relative.components().collect::<Vec<_>>();
    if relative_components.len() != 2
        || relative_components
            .iter()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(HekaError::Config(
            "Claude transcript path must be one project directory and one session file beneath the allowed root"
                .to_string(),
        ));
    }

    let root = raw_root.canonicalize()?;
    let cwd = raw_cwd.canonicalize().map_err(|error| {
        HekaError::Config(format!(
            "Claude Stop project cwd cannot be canonicalized: {error}"
        ))
    })?;
    if !cwd.is_dir() {
        return Err(HekaError::Config(
            "Claude Stop project cwd is not a directory".to_string(),
        ));
    }
    let project_component = relative_components[0].as_os_str();
    let filename_component = relative_components[1].as_os_str();
    let expected_filename = format!("{session_id}.jsonl");
    if filename_component.to_str() != Some(&expected_filename) {
        return Err(HekaError::Config(
            "Claude transcript filename does not match the Stop payload session_id".to_string(),
        ));
    }

    let file = open_transcript_beneath_root(&root, project_component, filename_component)?;
    let opened_metadata = file.metadata()?;
    if !opened_metadata.is_file() {
        return Err(HekaError::Config(
            "Claude transcript_path is not a regular file".to_string(),
        ));
    }
    if opened_metadata.len() > policy.max_transcript_bytes as u64 {
        return Err(resource_limit_error(
            "Claude transcript bytes",
            policy.max_transcript_bytes,
        ));
    }
    let transcript =
        read_utf8_bounded(file, policy.max_transcript_bytes, "Claude transcript bytes")?;
    Ok(ValidatedTranscript {
        contents: transcript,
        project_component: project_component.to_os_string(),
        stop_cwd: cwd,
    })
}

#[cfg(unix)]
fn open_transcript_beneath_root(
    root: &Path,
    project_component: &std::ffi::OsStr,
    filename_component: &std::ffi::OsStr,
) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;

    fn component_name(component: &std::ffi::OsStr) -> std::io::Result<CString> {
        CString::new(component.as_bytes()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "transcript path component contains NUL",
            )
        })
    }

    let root_dir = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(root)?;
    let project_name = component_name(project_component)?;
    // SAFETY: project_name is NUL-terminated, root_dir is a live directory
    // descriptor, and the returned descriptor is immediately owned by File.
    let project_fd = unsafe {
        libc::openat(
            root_dir.as_raw_fd(),
            project_name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW,
        )
    };
    if project_fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: openat returned a new owned descriptor on success.
    let project_dir = unsafe { File::from_raw_fd(project_fd) };

    let filename = component_name(filename_component)?;
    // SAFETY: filename is NUL-terminated, project_dir is live, and the
    // returned descriptor is immediately owned by File. O_NOFOLLOW anchors
    // every untrusted component, while O_NONBLOCK prevents FIFO hangs.
    let transcript_fd = unsafe {
        libc::openat(
            project_dir.as_raw_fd(),
            filename.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if transcript_fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: openat returned a new owned descriptor on success.
    Ok(unsafe { File::from_raw_fd(transcript_fd) })
}

#[cfg(not(unix))]
fn open_transcript_beneath_root(
    _root: &Path,
    _project_component: &std::ffi::OsStr,
    _filename_component: &std::ffi::OsStr,
) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "secure Claude transcript capture requires Unix openat semantics; inline and raw capture remain available",
    ))
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
            self.payload_with_cwd(last_assistant_message, &self.cwd)
        }

        fn payload_with_cwd(&self, last_assistant_message: Option<&str>, cwd: &Path) -> String {
            serde_json::json!({
                "session_id": self.session_id,
                "transcript_path": self.transcript,
                "cwd": cwd,
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
        let nested_cwd = layout.cwd.join("nested/source");
        let alternate_cwd = layout.base.join("alternate-worktree");
        std::fs::create_dir_all(&nested_cwd).unwrap();
        std::fs::create_dir_all(&alternate_cwd).unwrap();
        let records = [
            serde_json::json!({
                "type": "user", "sessionId": layout.session_id,
                "cwd": layout.cwd,
                "isSidechain": false,
                "message": {"role": "user", "content": "Remember the launch code."}
            }),
            serde_json::json!({
                "type": "assistant", "sessionId": layout.session_id,
                "cwd": nested_cwd,
                "isSidechain": false,
                "message": {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "I will remember it."}
                ]}
            }),
            serde_json::json!({
                "type": "user", "sessionId": layout.session_id,
                "cwd": alternate_cwd,
                "isSidechain": false,
                "message": {"role": "user", "content": "Continue from the alternate worktree."}
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

        let input =
            layout.payload_with_cwd(Some("Final answer not flushed to JSONL yet."), &nested_cwd);
        let batch = parse_capture_input_with_policy(&input, &layout.policy()).unwrap();

        assert_eq!(
            batch.session_id.as_deref(),
            Some(layout.session_id.as_str())
        );
        assert_eq!(batch.turns.len(), 4);
        assert_eq!(batch.turns[0].speaker.as_deref(), Some("user"));
        assert_eq!(
            batch.turns[0].body.as_deref(),
            Some("Remember the launch code.")
        );
        assert_eq!(batch.turns[1].body.as_deref(), Some("I will remember it."));
        assert_eq!(
            batch.turns[2].body.as_deref(),
            Some("Continue from the alternate worktree.")
        );
        assert_eq!(
            batch.turns[3].body.as_deref(),
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

        // A sibling worktree is authorized only because this exact session
        // records its canonical cwd. All useful turns remain available; cwd
        // is authorization evidence, not a per-turn filter.
        let alternate_batch = parse_capture_input_with_policy(
            &layout.payload_with_cwd(None, &alternate_cwd),
            &layout.policy(),
        )
        .unwrap();
        assert_eq!(alternate_batch.turns.len(), 3);
        assert_eq!(alternate_batch.turns.as_slice(), &batch.turns[..3]);
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
    fn repeated_final_text_is_kept_when_transcript_ends_with_user() {
        let input = serde_json::json!({
            "session_id": "abc123",
            "transcript": [
                {"speaker": "assistant", "body": "same answer"},
                {"speaker": "user", "body": "please repeat it"}
            ],
            "last_assistant_message": "same answer"
        })
        .to_string();

        let batch = parse_capture_input_with_policy(&input, &inline_policy()).unwrap();
        assert_eq!(batch.turns.len(), 3);
        assert_eq!(batch.turns[2].body.as_deref(), Some("same answer"));
    }

    #[test]
    fn arbitrary_readable_path_is_rejected_even_with_a_final_message() {
        let layout = HookLayout::new();
        let outside = layout.base.join(format!("{}.jsonl", layout.session_id));
        std::fs::write(&outside, "{}").unwrap();
        let input = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": outside,
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
    fn inline_and_raw_inputs_do_not_require_a_home_transcript_policy() {
        let unavailable_policy = || -> HekaResult<CapturePolicy> {
            Err(HekaError::Config(
                "HOME is not set; sentinel transcript policy failure".to_string(),
            ))
        };
        let inline = parse_capture_input_with_policy_factory(
            r#"{"session_id":"legacy","turns":[{"speaker":"user","body":"hello"}]}"#,
            None,
            unavailable_policy,
        )
        .unwrap();
        assert_eq!(inline.turns.len(), 1);

        let raw =
            parse_capture_input_with_policy_factory("plain pipe input", None, unavailable_policy)
                .unwrap();
        assert_eq!(raw.turns.len(), 1);
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
        assert!(!error.contains("other-session"));
    }

    #[test]
    fn transcript_path_shape_and_filename_fail_closed() {
        let layout = HookLayout::new();
        let nested_project = layout.root.join("project/nested");
        std::fs::create_dir_all(&nested_project).unwrap();
        let nested_path = nested_project.join(format!("{}.jsonl", layout.session_id));
        std::fs::write(&nested_path, "{}").unwrap();
        let nested_payload = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": nested_path,
            "cwd": layout.cwd,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&nested_payload, &layout.policy()).is_err());

        std::fs::write(&layout.transcript, "{}").unwrap();
        let wrong_session = serde_json::json!({
            "session_id": "different-session",
            "transcript_path": layout.transcript,
            "cwd": layout.cwd,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&wrong_session, &layout.policy()).is_err());
    }

    #[test]
    fn stop_project_authorization_rejects_cross_project_selection() {
        let layout = HookLayout::new();
        let other_cwd = layout.base.join("other-workspace");
        std::fs::create_dir_all(&other_cwd).unwrap();
        let other_cwd = other_cwd.canonicalize().unwrap();
        let other_project = layout.root.join(claude_project_dir_name(&other_cwd));
        std::fs::create_dir_all(&other_project).unwrap();
        let other_transcript = other_project.join(format!("{}.jsonl", layout.session_id));
        std::fs::write(&other_transcript, "{}").unwrap();
        let cross_project = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": other_transcript,
            "cwd": layout.cwd,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&cross_project, &layout.policy()).is_err());
    }

    #[test]
    fn opened_project_component_must_match_first_session_cwd() {
        let layout = HookLayout::new();
        let other_cwd = layout.base.join("other-first-row-project");
        std::fs::create_dir_all(&other_cwd).unwrap();
        let other_cwd = other_cwd.canonicalize().unwrap();
        let record = serde_json::json!({
            "type": "user",
            "sessionId": layout.session_id,
            "cwd": other_cwd,
            "message": {"role": "user", "content": "wrong opened project"},
        });
        std::fs::write(&layout.transcript, record.to_string()).unwrap();

        // Exact Stop/session metadata alone is insufficient when the opened
        // project component does not derive from the transcript's first row.
        let payload = layout.payload_with_cwd(None, &other_cwd);
        assert!(parse_capture_input_with_policy(&payload, &layout.policy()).is_err());
    }

    #[test]
    fn lossy_project_key_collision_cannot_authorize_an_unrecorded_cwd() {
        let layout = HookLayout::new();
        let stop_cwd = layout.base.join("collision/a-b/c");
        let transcript_cwd = layout.base.join("collision/a/b-c");
        std::fs::create_dir_all(&stop_cwd).unwrap();
        std::fs::create_dir_all(&transcript_cwd).unwrap();
        let stop_cwd = stop_cwd.canonicalize().unwrap();
        let transcript_cwd = transcript_cwd.canonicalize().unwrap();
        assert_ne!(stop_cwd, transcript_cwd);
        assert_eq!(
            claude_project_dir_name(&stop_cwd),
            claude_project_dir_name(&transcript_cwd)
        );

        let project = layout.root.join(claude_project_dir_name(&transcript_cwd));
        std::fs::create_dir_all(&project).unwrap();
        let transcript = project.join(format!("{}.jsonl", layout.session_id));
        let record = serde_json::json!({
            "type": "user",
            "sessionId": layout.session_id,
            "cwd": transcript_cwd,
            "message": {"role": "user", "content": "belongs to the other project"},
        });
        std::fs::write(&transcript, record.to_string()).unwrap();
        let payload = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": transcript,
            "cwd": stop_cwd,
        })
        .to_string();

        assert!(parse_capture_input_with_policy(&payload, &layout.policy()).is_err());
    }

    #[test]
    fn unrecorded_descendant_cwd_is_not_authorized_by_an_ancestor_key() {
        let layout = HookLayout::new();
        let unrecorded_nested = layout.cwd.join("unrecorded/nested");
        std::fs::create_dir_all(&unrecorded_nested).unwrap();
        let record = serde_json::json!({
            "type": "user",
            "sessionId": layout.session_id,
            "cwd": layout.cwd,
            "message": {"role": "user", "content": "root-only session"},
        });
        std::fs::write(&layout.transcript, record.to_string()).unwrap();

        let payload = layout.payload_with_cwd(None, &unrecorded_nested);
        assert!(parse_capture_input_with_policy(&payload, &layout.policy()).is_err());
    }

    #[test]
    fn transcript_capture_requires_valid_top_level_project_metadata() {
        let layout = HookLayout::new();
        std::fs::write(&layout.transcript, "{}").unwrap();
        let missing_cwd = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": layout.transcript,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&missing_cwd, &layout.policy()).is_err());

        for invalid_cwd in ["", "."] {
            let invalid = serde_json::json!({
                "session_id": layout.session_id,
                "transcript_path": layout.transcript,
                "cwd": invalid_cwd,
            })
            .to_string();
            assert!(parse_capture_input_with_policy(&invalid, &layout.policy()).is_err());
        }

        let missing_project = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": layout.transcript,
            "cwd": layout.base.join("does-not-exist"),
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&missing_project, &layout.policy()).is_err());

        let file_cwd = serde_json::json!({
            "session_id": layout.session_id,
            "transcript_path": layout.transcript,
            "cwd": layout.transcript,
        })
        .to_string();
        assert!(parse_capture_input_with_policy(&file_cwd, &layout.policy()).is_err());
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
        let project_layout = HookLayout::new();
        let expected_project = project_layout.transcript.parent().unwrap();
        std::fs::remove_dir(expected_project).unwrap();
        let project_target = project_layout.base.join("project-symlink-target");
        std::fs::create_dir(&project_target).unwrap();
        std::fs::write(
            project_target.join(format!("{}.jsonl", project_layout.session_id)),
            "{}",
        )
        .unwrap();
        symlink(&project_target, expected_project).unwrap();
        assert!(
            parse_capture_input_with_policy(
                &project_layout.payload(None),
                &project_layout.policy()
            )
            .is_err()
        );

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
