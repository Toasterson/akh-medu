//! Synthetic-fixture tests for every row of the mapping table, plus
//! noise/undetermined/degraded cases. No real transcript content — shapes
//! only, built from the census in
//! `docs/wayfinder/transcript-harvest/assets/source-format-stability.md`.

use chrono::Utc;
use serde_json::{Value, json};

use crate::model::{RecordKey, RecordOrigin, Source, StreamKind, VerbatimRecord};

use super::{EventKind, Outcome, normalize};

fn record(source: Source, record_type: &str, payload: Value) -> VerbatimRecord {
    VerbatimRecord {
        key: RecordKey {
            source,
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            seq: 1,
        },
        stream_kind: StreamKind::Main,
        raw: payload.to_string(),
        payload: Some(payload),
        timestamp: None,
        origin: RecordOrigin {
            machine: "test-host".to_string(),
            path: "test.jsonl".to_string(),
            record_type: Some(record_type.to_string()),
            schema_marker: None,
        },
    }
}

fn unparsed(source: Source, record_type: &str) -> VerbatimRecord {
    VerbatimRecord {
        key: RecordKey {
            source,
            session_id: "sess-1".to_string(),
            stream_id: "main".to_string(),
            seq: 1,
        },
        stream_kind: StreamKind::Main,
        raw: "not json".to_string(),
        payload: None,
        timestamp: None,
        origin: RecordOrigin {
            machine: "test-host".to_string(),
            path: "test.jsonl".to_string(),
            record_type: Some(record_type.to_string()),
            schema_marker: None,
        },
    }
}

fn as_event(outcome: Outcome) -> super::NormalizedEvent {
    match outcome {
        Outcome::Event(ev) => ev,
        other => panic!("expected Outcome::Event, got {other:?}"),
    }
}

fn as_degraded(outcome: Outcome) -> super::NormalizedEvent {
    match outcome {
        Outcome::Degraded(ev) => ev,
        other => panic!("expected Outcome::Degraded, got {other:?}"),
    }
}

fn as_events(outcome: Outcome) -> Vec<super::NormalizedEvent> {
    match outcome {
        Outcome::Events(evs) => evs,
        other => panic!("expected Outcome::Events, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Unparsed payload -> Undetermined, for every source.
// ---------------------------------------------------------------------

#[test]
fn unparsed_payload_is_undetermined_for_every_source() {
    for source in [Source::ClaudeCode, Source::Codex, Source::Opencode] {
        let rec = unparsed(source, "whatever");
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => assert_eq!(record_type, "whatever"),
            other => panic!("expected Undetermined for {source:?}, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// Claude Code
// ---------------------------------------------------------------------

mod claude_code {
    use super::*;

    #[test]
    fn user_string_content_is_prompt() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": "hello there"}, "cwd": "/home/x/proj", "gitBranch": "main"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert_eq!(ev.text.as_deref(), Some("hello there"));
        assert_eq!(ev.workspace.as_deref(), Some("/home/x/proj"));
        assert_eq!(ev.repo.as_deref(), Some("/home/x/proj"));
        assert_eq!(ev.branch.as_deref(), Some("main"));
        assert!(!ev.degraded);
    }

    #[test]
    fn repo_strips_worktree_suffix_from_cwd() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({
                "message": {"content": "hi"},
                "cwd": "/home/x/proj/.claude/worktrees/cool-carson-b63ef4"
            }),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.repo.as_deref(), Some("/home/x/proj"));
        assert_eq!(
            ev.workspace.as_deref(),
            Some("/home/x/proj/.claude/worktrees/cool-carson-b63ef4")
        );
    }

    #[test]
    fn user_single_text_block_is_prompt_no_subseq() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": [{"type": "text", "text": "one block"}]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert_eq!(ev.text.as_deref(), Some("one block"));
        assert_eq!(ev.subseq, None);
    }

    #[test]
    fn user_three_tool_result_blocks_yields_three_events_sharing_key() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "out1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "out2"},
                {"type": "tool_result", "tool_use_id": "t3", "content": "out3"},
            ]}}),
        );
        let events = as_events(normalize(&rec, Utc::now()));
        assert_eq!(events.len(), 3);
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.kind, EventKind::ToolResult);
            assert_eq!(ev.key, rec.key);
            assert_eq!(ev.subseq, Some(i as u32));
        }
        assert_eq!(events[0].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(events[1].tool_call_id.as_deref(), Some("t2"));
    }

    #[test]
    fn user_empty_content_array_is_noise() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": []}}),
        );
        assert!(matches!(normalize(&rec, Utc::now()), Outcome::Noise { .. }));
    }

    #[test]
    fn user_missing_message_is_degraded() {
        let rec = record(Source::ClaudeCode, "user", json!({}));
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert!(ev.degraded);
        assert!(ev.degraded_reason.as_deref().unwrap().contains("message"));
    }

    #[test]
    fn user_content_wrong_type_is_degraded() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": 42}}),
        );
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert!(ev.degraded);
    }

    #[test]
    fn user_unrecognized_block_type_is_degraded_within_a_single_block() {
        let rec = record(
            Source::ClaudeCode,
            "user",
            json!({"message": {"content": [{"type": "mystery"}]}}),
        );
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert!(ev.degraded);
        assert!(ev.degraded_reason.as_deref().unwrap().contains("mystery"));
    }

    #[test]
    fn assistant_text_block_is_reply() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": [{"type": "text", "text": "sure, here goes"}]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
        assert_eq!(ev.text.as_deref(), Some("sure, here goes"));
    }

    #[test]
    fn assistant_thinking_block_with_text_is_thought_with_text() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": [{"type": "thinking", "thinking": "let me consider"}]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Thought);
        assert_eq!(ev.text.as_deref(), Some("let me consider"));
    }

    #[test]
    fn assistant_thinking_block_empty_is_thought_with_no_text() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": [{"type": "thinking", "thinking": "", "signature": "abc123"}]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Thought);
        assert_eq!(ev.text, None);
    }

    #[test]
    fn assistant_tool_use_block_is_tool_call() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": [
                {"type": "tool_use", "id": "call-1", "name": "Bash", "input": {"command": "ls"}}
            ]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::ToolCall);
        assert_eq!(ev.tool_name.as_deref(), Some("Bash"));
        assert_eq!(ev.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(ev.text.as_deref(), Some(r#"{"command":"ls"}"#));
    }

    #[test]
    fn assistant_fallback_block_is_error_model_fallback() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": [{"type": "fallback", "reason": "overloaded"}]}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Error);
        assert_eq!(ev.subkind.as_deref(), Some("model_fallback"));
    }

    #[test]
    fn assistant_content_not_array_is_degraded() {
        let rec = record(
            Source::ClaudeCode,
            "assistant",
            json!({"message": {"content": 7}}),
        );
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
        assert!(ev.degraded);
        assert!(ev.degraded_reason.as_deref().unwrap().contains("number"));
    }

    #[test]
    fn attachment_is_context() {
        let rec = record(
            Source::ClaudeCode,
            "attachment",
            json!({"content": "a file"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Context);
        assert_eq!(ev.text.as_deref(), Some("a file"));
    }

    #[test]
    fn system_model_refusal_fallback_is_error() {
        let rec = record(
            Source::ClaudeCode,
            "system",
            json!({"subtype": "model_refusal_fallback", "content": "refused"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Error);
        assert_eq!(ev.subkind.as_deref(), Some("model_refusal_fallback"));
    }

    #[test]
    fn system_other_subtype_is_undetermined() {
        let rec = record(
            Source::ClaudeCode,
            "system",
            json!({"subtype": "hook_output"}),
        );
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => assert_eq!(record_type, "system:hook_output"),
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }

    #[test]
    fn ai_title_is_session_meta() {
        let rec = record(
            Source::ClaudeCode,
            "ai-title",
            json!({"aiTitle": "Fix the bug"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::SessionMeta);
        assert_eq!(ev.subkind.as_deref(), Some("ai-title"));
        assert_eq!(ev.text.as_deref(), Some("Fix the bug"));
    }

    #[test]
    fn mode_is_session_meta() {
        let rec = record(Source::ClaudeCode, "mode", json!({"mode": "plan"}));
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::SessionMeta);
        assert_eq!(ev.subkind.as_deref(), Some("mode"));
        assert_eq!(ev.text.as_deref(), Some("plan"));
    }

    #[test]
    fn started_is_lifecycle_subagent_started() {
        let rec = record(
            Source::ClaudeCode,
            "started",
            json!({"agentId": "agent-abc"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Lifecycle);
        assert_eq!(ev.subkind.as_deref(), Some("subagent_started"));
        assert_eq!(ev.agent_id.as_deref(), Some("agent-abc"));
        assert_eq!(ev.text, None);
    }

    #[test]
    fn result_is_lifecycle_subagent_result_with_text() {
        let rec = record(
            Source::ClaudeCode,
            "result",
            json!({"agentId": "agent-abc", "result": {"report": "done"}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Lifecycle);
        assert_eq!(ev.subkind.as_deref(), Some("subagent_result"));
        assert_eq!(ev.agent_id.as_deref(), Some("agent-abc"));
        assert_eq!(ev.text.as_deref(), Some(r#"{"report":"done"}"#));
    }

    #[test]
    fn queue_operation_last_prompt_atis_latch_are_noise() {
        for t in ["queue-operation", "last-prompt", "atis-latch"] {
            let rec = record(Source::ClaudeCode, t, json!({}));
            assert!(
                matches!(normalize(&rec, Utc::now()), Outcome::Noise { .. }),
                "{t} should be Noise"
            );
        }
    }

    #[test]
    fn unknown_type_is_undetermined() {
        let rec = record(Source::ClaudeCode, "some-new-type", json!({}));
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => assert_eq!(record_type, "some-new-type"),
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// Codex
// ---------------------------------------------------------------------

mod codex {
    use super::*;

    fn response_item(payload: Value) -> Value {
        json!({"type": "response_item", "payload": payload})
    }

    #[test]
    fn message_user_is_prompt() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert_eq!(ev.text.as_deref(), Some("hi"));
        assert_eq!(ev.role.as_deref(), Some("user"));
    }

    #[test]
    fn message_assistant_is_reply() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "sure"}]}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
        assert_eq!(ev.text.as_deref(), Some("sure"));
    }

    #[test]
    fn message_developer_is_context() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "system prompt bits"}]}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Context);
    }

    #[test]
    fn message_unknown_role_is_undetermined() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(json!({"type": "message", "role": "tool", "content": []})),
        );
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => {
                assert_eq!(record_type, "response_item:message:role=tool")
            }
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }

    #[test]
    fn message_content_not_array_is_degraded() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(json!({"type": "message", "role": "user", "content": "not an array"})),
        );
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert!(ev.degraded);
    }

    #[test]
    fn agent_message_is_reply() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(json!({"type": "agent_message", "message": "the answer"})),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
        assert_eq!(ev.text.as_deref(), Some("the answer"));
    }

    #[test]
    fn reasoning_is_thought_encrypted_never_emits_text() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "reasoning", "summary": [], "encrypted_content": "gAAAAA-ciphertext-blob"}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Thought);
        assert_eq!(ev.subkind.as_deref(), Some("encrypted"));
        assert_eq!(
            ev.text, None,
            "must never surface encrypted_content as text"
        );
    }

    #[test]
    fn reasoning_with_nonempty_summary_still_never_emits_text() {
        // Not observed in the corpus (100% were summary: []), but the
        // posture is "never emit the ciphertext" unconditionally.
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "reasoning", "summary": ["some summary text"], "encrypted_content": "blob"}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.text, None);
    }

    #[test]
    fn custom_tool_call_and_function_call_are_tool_call() {
        for (t, id_field) in [
            ("custom_tool_call", "call_id"),
            ("function_call", "call_id"),
        ] {
            let rec = record(
                Source::Codex,
                "response_item",
                response_item(
                    json!({"type": t, "name": "shell", id_field: "c1", "input": {"cmd": "ls"}}),
                ),
            );
            let ev = as_event(normalize(&rec, Utc::now()));
            assert_eq!(ev.kind, EventKind::ToolCall, "{t}");
            assert_eq!(ev.tool_name.as_deref(), Some("shell"));
            assert_eq!(ev.tool_call_id.as_deref(), Some("c1"));
        }
    }

    #[test]
    fn function_call_arguments_field_is_used_when_input_absent() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(
                json!({"type": "function_call", "name": "shell", "id": "c2", "arguments": {"cmd": "pwd"}}),
            ),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.tool_call_id.as_deref(), Some("c2"));
        assert_eq!(ev.text.as_deref(), Some(r#"{"cmd":"pwd"}"#));
    }

    #[test]
    fn custom_tool_call_output_and_function_call_output_are_tool_result() {
        for t in ["custom_tool_call_output", "function_call_output"] {
            let rec = record(
                Source::Codex,
                "response_item",
                response_item(json!({"type": t, "call_id": "c1", "output": "done"})),
            );
            let ev = as_event(normalize(&rec, Utc::now()));
            assert_eq!(ev.kind, EventKind::ToolResult, "{t}");
            assert_eq!(ev.tool_call_id.as_deref(), Some("c1"));
        }
    }

    #[test]
    fn response_item_missing_payload_is_degraded() {
        let rec = record(
            Source::Codex,
            "response_item",
            json!({"type": "response_item"}),
        );
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert!(ev.degraded);
    }

    #[test]
    fn response_item_unknown_payload_type_is_undetermined() {
        let rec = record(
            Source::Codex,
            "response_item",
            response_item(json!({"type": "mystery_payload"})),
        );
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => {
                assert_eq!(record_type, "response_item:mystery_payload")
            }
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }

    #[test]
    fn event_msg_lifecycle_subtypes_are_lifecycle() {
        for t in ["task_started", "task_complete", "sub_agent_activity"] {
            let rec = record(
                Source::Codex,
                "event_msg",
                json!({"type": "event_msg", "payload": {"type": t, "agent_id": "a1"}}),
            );
            let ev = as_event(normalize(&rec, Utc::now()));
            assert_eq!(ev.kind, EventKind::Lifecycle, "{t}");
            assert_eq!(ev.subkind.as_deref(), Some(t));
            assert_eq!(ev.agent_id.as_deref(), Some("a1"));
        }
    }

    #[test]
    fn event_msg_token_count_is_noise() {
        let rec = record(
            Source::Codex,
            "event_msg",
            json!({"type": "event_msg", "payload": {"type": "token_count", "count": 17311}}),
        );
        assert!(matches!(normalize(&rec, Utc::now()), Outcome::Noise { .. }));
    }

    #[test]
    fn event_msg_other_payload_types_are_noise() {
        let rec = record(
            Source::Codex,
            "event_msg",
            json!({"type": "event_msg", "payload": {"type": "web_search_end"}}),
        );
        assert!(matches!(normalize(&rec, Utc::now()), Outcome::Noise { .. }));
    }

    #[test]
    fn top_level_session_meta_is_session_meta() {
        let rec = record(
            Source::Codex,
            "session_meta",
            json!({"type": "session_meta", "session_id": "abc", "cli_version": "0.153.4"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::SessionMeta);
    }

    #[test]
    fn token_usage_record_is_telemetry() {
        let rec = record(
            Source::Codex,
            "token_usage_record",
            json!({"type": "token_usage_record", "input_tokens": 10, "output_tokens": 20}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Telemetry);
    }

    #[test]
    fn compacted_is_context_with_compaction_subkind() {
        let rec = record(
            Source::Codex,
            "compacted",
            json!({"type": "compacted", "summary": "big snapshot"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Context);
        assert_eq!(ev.subkind.as_deref(), Some("compaction"));
    }

    #[test]
    fn turn_context_is_context() {
        let rec = record(
            Source::Codex,
            "turn_context",
            json!({"type": "turn_context", "cwd": "/repo", "model": "gpt"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Context);
        assert_eq!(ev.subkind.as_deref(), Some("turn_context"));
    }

    #[test]
    fn world_state_is_explicitly_undetermined() {
        let rec = record(Source::Codex, "world_state", json!({"type": "world_state"}));
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => assert_eq!(record_type, "world_state"),
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }

    #[test]
    fn inter_agent_communication_metadata_falls_to_the_others_catch_all() {
        // Not enumerated by name in the mapping table; resolved via the
        // "others -> Undetermined" catch-all. See report for this decision.
        let rec = record(
            Source::Codex,
            "inter_agent_communication_metadata",
            json!({"type": "inter_agent_communication_metadata"}),
        );
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => {
                assert_eq!(record_type, "inter_agent_communication_metadata")
            }
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// OpenCode — thin corpus, conservative synthetic shapes.
// ---------------------------------------------------------------------

mod opencode {
    use super::*;

    /// Mirrors `adapter::opencode::build_record`'s real wrapped-row shape:
    /// `{"id", "aggregate_id", "seq", "type", "data"}`, with `origin.record_type`
    /// already stripped of its `.N` schema-version suffix — exactly what
    /// `split_type_and_marker` does upstream. This is the shape `normalize`
    /// actually receives in production, not a bare `data` blob.
    fn wrapped(event_type_with_suffix: &str, data: Value) -> VerbatimRecord {
        let (base, marker) = event_type_with_suffix
            .rsplit_once('.')
            .filter(|(_, suffix)| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
            .map(|(base, suffix)| (base.to_string(), Some(suffix.to_string())))
            .unwrap_or_else(|| (event_type_with_suffix.to_string(), None));
        let row = json!({
            "id": "evt_1",
            "aggregate_id": "sess-1",
            "seq": 1,
            "type": event_type_with_suffix,
            "data": data,
        });
        VerbatimRecord {
            key: RecordKey {
                source: Source::Opencode,
                session_id: "sess-1".to_string(),
                stream_id: "events".to_string(),
                seq: 1,
            },
            stream_kind: StreamKind::EventLog,
            raw: row.to_string(),
            payload: Some(row),
            timestamp: None,
            origin: RecordOrigin {
                machine: "test-host".to_string(),
                path: "opencode.db".to_string(),
                record_type: Some(base),
                schema_marker: marker,
            },
        }
    }

    #[test]
    fn session_created_and_updated_are_session_meta() {
        for t in ["session.created.1", "session.updated.1"] {
            let rec = wrapped(t, json!({"title": "My session"}));
            let ev = as_event(normalize(&rec, Utc::now()));
            assert_eq!(ev.kind, EventKind::SessionMeta, "{t}");
            assert_eq!(ev.text.as_deref(), Some("My session"));
        }
    }

    #[test]
    fn message_updated_user_is_prompt() {
        let rec = wrapped(
            "message.updated.1",
            json!({"role": "user", "text": "hello"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert_eq!(ev.text.as_deref(), Some("hello"));
    }

    #[test]
    fn message_updated_assistant_is_reply() {
        let rec = wrapped(
            "message.updated.1",
            json!({"role": "assistant", "text": "hi back"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
    }

    #[test]
    fn message_updated_missing_role_is_degraded() {
        let rec = wrapped("message.updated.1", json!({"text": "no role here"}));
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert!(ev.degraded);
    }

    #[test]
    fn message_part_updated_text_part_is_prose() {
        let rec = wrapped(
            "message.part.updated.1",
            json!({"role": "assistant", "part": {"type": "text", "text": "streamed chunk"}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Reply);
        assert_eq!(ev.text.as_deref(), Some("streamed chunk"));
    }

    #[test]
    fn message_part_updated_tool_part_without_output_is_tool_call() {
        let rec = wrapped(
            "message.part.updated.1",
            json!({"part": {"type": "tool", "tool": "bash", "callID": "c1", "input": {"cmd": "ls"}}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::ToolCall);
        assert_eq!(ev.tool_name.as_deref(), Some("bash"));
    }

    #[test]
    fn message_part_updated_tool_part_with_output_is_tool_result() {
        let rec = wrapped(
            "message.part.updated.1",
            json!({"part": {"type": "tool", "tool": "bash", "callID": "c1", "output": "file1\nfile2"}}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::ToolResult);
    }

    #[test]
    fn message_part_updated_missing_part_is_degraded() {
        let rec = wrapped("message.part.updated.1", json!({}));
        let ev = as_degraded(normalize(&rec, Utc::now()));
        assert!(ev.degraded);
    }

    #[test]
    fn unknown_event_type_is_undetermined() {
        let rec = wrapped("some.new.event.1", json!({}));
        match normalize(&rec, Utc::now()) {
            Outcome::Undetermined { record_type } => assert_eq!(record_type, "some.new.event"),
            other => panic!("expected Undetermined, got {other:?}"),
        }
    }

    #[test]
    fn version_suffix_is_stripped_for_dispatch() {
        // Same base type, different version suffix, must dispatch the same.
        for t in [
            "session.created.1",
            "session.created.2",
            "session.created.42",
        ] {
            let rec = wrapped(t, json!({"title": "t"}));
            let ev = as_event(normalize(&rec, Utc::now()));
            assert_eq!(ev.kind, EventKind::SessionMeta, "{t}");
        }
    }

    #[test]
    fn payload_without_a_data_wrapper_falls_back_to_the_row_itself() {
        // Defensive fallback if the wrapper shape ever changes upstream:
        // normalize should not panic, it should just find nothing useful
        // and degrade or fall through, never crash.
        let rec = record(
            Source::Opencode,
            "message.updated",
            json!({"role": "user", "text": "hi"}),
        );
        let ev = as_event(normalize(&rec, Utc::now()));
        assert_eq!(ev.kind, EventKind::Prompt);
        assert_eq!(ev.text.as_deref(), Some("hi"));
    }
}
