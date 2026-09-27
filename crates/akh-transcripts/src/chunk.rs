//! chunk — turn-boundary prose chunks for embedding. Scope (prose only,
//! `prompt`/`reply` — not `thought`) and turn-boundary rationale:
//! `docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md`
//! (corrected from the original `prompt`/`reply`/`thought` decision once the
//! render prototype measured reasoning content at 0.06% of the corpus, 100%
//! ciphertext on the Codex side).

use crate::model::{EventKind, Source};
use crate::normalize::{Chunk, NormalizedEvent};
use crate::redact;

/// ~1500 chars, comfortably inside MiniLM's 512-token limit
/// (`embedder.rs:15` per the ticket). Treated as a byte cap here, which is
/// an approximation for non-ASCII text — acceptable given the corpus is
/// overwhelmingly English prose and code.
const DEFAULT_TARGET_CHARS: usize = 1500;

/// `prompt` and `reply` only. `thought` is excluded — the embedding-pipeline
/// ticket's correction: Codex reasoning is 100% encrypted and 91.8% of
/// Claude thinking blocks are empty signatures, so there is next to nothing
/// there to embed, and embedding a sealed marker would risk embedding
/// ciphertext if the exclusion ever slipped.
fn is_prose(kind: EventKind) -> bool {
    matches!(kind, EventKind::Prompt | EventKind::Reply)
}

/// Group consecutive prose events into turn-bounded chunks of roughly
/// `target_chars` (falls back to the 1500-char default if `0`), splitting on
/// sentence/newline boundaries and never mid-word. A turn boundary is a
/// change of role or any non-prose event between two prose events — encoded
/// here as a change of `EventKind`, since `Prompt` and `Reply` already are
/// the role signal for every source by construction of the mapping table.
/// Each chunk's text is redacted before it is returned: chunks cross the
/// outward boundary to the embedder and the read surface.
pub fn chunk_turns(events: &[NormalizedEvent], target_chars: usize) -> Vec<Chunk> {
    let target_chars = if target_chars == 0 {
        DEFAULT_TARGET_CHARS
    } else {
        target_chars
    };

    let mut chunks = Vec::new();
    let mut i = 0;
    while i < events.len() {
        if !is_prose(events[i].kind) {
            i += 1;
            continue;
        }
        let start = i;
        let kind = events[i].kind;
        let mut j = i;
        while j < events.len() && events[j].kind == kind {
            j += 1;
        }
        chunks.extend(chunk_one_turn(&events[start..j], target_chars));
        i = j;
    }
    chunks
}

fn chunk_one_turn(turn: &[NormalizedEvent], target_chars: usize) -> Vec<Chunk> {
    let Some(first) = turn.first() else {
        return Vec::new();
    };
    let source = first.key.source;
    let session_id = first.key.session_id.clone();
    let stream_id = first.key.stream_id.clone();

    let mut chunks = Vec::new();
    let mut buf = String::new();
    let mut seq_from: Option<i64> = None;
    let mut seq_to: Option<i64> = None;

    for event in turn {
        let Some(text) = event.text.as_deref() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }

        if text.len() > target_chars {
            // This single event's own text needs splitting regardless of
            // buffer state — flush whatever is pending first, then emit one
            // chunk per piece, all attributed to this event's seq.
            flush(
                &mut chunks,
                source,
                &session_id,
                &stream_id,
                &mut buf,
                &mut seq_from,
                &mut seq_to,
            );
            for piece in split_at_boundary(text, target_chars) {
                buf.push_str(&piece);
                seq_from = Some(event.key.seq);
                seq_to = Some(event.key.seq);
                flush(
                    &mut chunks,
                    source,
                    &session_id,
                    &stream_id,
                    &mut buf,
                    &mut seq_from,
                    &mut seq_to,
                );
            }
            continue;
        }

        if !buf.is_empty() && buf.len() + 1 + text.len() > target_chars {
            flush(
                &mut chunks,
                source,
                &session_id,
                &stream_id,
                &mut buf,
                &mut seq_from,
                &mut seq_to,
            );
        }
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(text);
        seq_from.get_or_insert(event.key.seq);
        seq_to = Some(event.key.seq);
    }
    flush(
        &mut chunks,
        source,
        &session_id,
        &stream_id,
        &mut buf,
        &mut seq_from,
        &mut seq_to,
    );
    chunks
}

#[allow(clippy::too_many_arguments)]
fn flush(
    chunks: &mut Vec<Chunk>,
    source: Source,
    session_id: &str,
    stream_id: &str,
    buf: &mut String,
    seq_from: &mut Option<i64>,
    seq_to: &mut Option<i64>,
) {
    if buf.is_empty() {
        return;
    }
    if let (Some(from), Some(to)) = (*seq_from, *seq_to) {
        let flags = redact::scan(buf);
        let text = redact::apply(buf, &flags);
        chunks.push(Chunk {
            source,
            session_id: session_id.to_string(),
            stream_id: stream_id.to_string(),
            seq_from: from,
            seq_to: to,
            text,
        });
    }
    buf.clear();
    *seq_from = None;
    *seq_to = None;
}

fn clamp_to_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Find the best split point inside `head` (a prefix already clamped to
/// `cap`): the last sentence end, else the last newline, else the last
/// whitespace, else fall through to a hard cut. Never mid-word except in
/// that last, pathological, no-whitespace-at-all case.
fn find_boundary(head: &str) -> Option<usize> {
    for pat in [". ", "! ", "? ", ".\n", "!\n", "?\n"] {
        if let Some(pos) = head.rfind(pat) {
            return Some(pos + pat.len());
        }
    }
    if let Some(pos) = head.rfind('\n') {
        return Some(pos + 1);
    }
    head.char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map(|(pos, ch)| pos + ch.len_utf8())
}

/// Split `text` into pieces of at most `cap` bytes each, breaking at a
/// sentence or whitespace boundary and never mid-word (barring the
/// pathological single-token-longer-than-`cap` case, where a hard cut at a
/// char boundary is the only option left).
fn split_at_boundary(text: &str, cap: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        if rest.len() <= cap {
            pieces.push(rest.to_string());
            break;
        }
        let limit = clamp_to_char_boundary(rest, cap).max(1);
        let cut = find_boundary(&rest[..limit])
            .filter(|&c| c > 0)
            .unwrap_or(limit);
        let (piece, remainder) = rest.split_at(cut);
        pieces.push(piece.trim_end().to_string());
        rest = remainder.trim_start();
    }
    pieces
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::model::{EventKind, RecordKey, Source};
    use crate::normalize::NormalizedEvent;

    use super::chunk_turns;

    fn key(seq: i64) -> RecordKey {
        RecordKey {
            source: Source::ClaudeCode,
            session_id: "s1".to_string(),
            stream_id: "main".to_string(),
            seq,
        }
    }

    fn prose(seq: i64, kind: EventKind, text: &str) -> NormalizedEvent {
        let mut ev = NormalizedEvent::new(key(seq), Utc::now(), kind);
        ev.text = Some(text.to_string());
        ev
    }

    #[test]
    fn adjacent_prompt_and_reply_never_merge_across_the_turn_boundary() {
        let events = vec![
            prose(1, EventKind::Prompt, "hello"),
            prose(2, EventKind::Reply, "hi there"),
        ];
        let chunks = chunk_turns(&events, 1500);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "hello");
        assert_eq!(chunks[0].seq_from, 1);
        assert_eq!(chunks[0].seq_to, 1);
        assert_eq!(chunks[1].text, "hi there");
        assert_eq!(chunks[1].seq_from, 2);
        assert_eq!(chunks[1].seq_to, 2);
    }

    #[test]
    fn consecutive_same_role_prose_groups_into_one_chunk() {
        let events = vec![
            prose(1, EventKind::Prompt, "part one."),
            prose(2, EventKind::Prompt, "part two."),
        ];
        let chunks = chunk_turns(&events, 1500);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("part one."));
        assert!(chunks[0].text.contains("part two."));
        assert_eq!(chunks[0].seq_from, 1);
        assert_eq!(chunks[0].seq_to, 2);
    }

    #[test]
    fn a_non_prose_event_between_two_prompts_breaks_the_turn() {
        let mut tool_call = NormalizedEvent::new(key(2), Utc::now(), EventKind::ToolCall);
        tool_call.text = Some("{}".to_string());
        let events = vec![
            prose(1, EventKind::Prompt, "before the tool call"),
            tool_call,
            prose(3, EventKind::Prompt, "after the tool call"),
        ];
        let chunks = chunk_turns(&events, 1500);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].seq_to, 1);
        assert_eq!(chunks[1].seq_from, 3);
    }

    #[test]
    fn thought_events_are_never_chunked() {
        let mut thought = NormalizedEvent::new(key(1), Utc::now(), EventKind::Thought);
        thought.text = Some("real reasoning text".to_string());
        let events = vec![thought];
        assert!(chunk_turns(&events, 1500).is_empty());
    }

    #[test]
    fn long_text_splits_on_sentence_boundary_never_mid_word() {
        let sentence = "This is one complete sentence about the transcript harvester. ";
        let long_text = sentence.repeat(50); // well over 1500 chars
        let events = vec![prose(1, EventKind::Reply, &long_text)];
        let chunks = chunk_turns(&events, 1500);
        assert!(
            chunks.len() > 1,
            "expected the long turn to split into multiple chunks"
        );
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= 1500,
                "chunk exceeded the 1500-char cap: {}",
                chunk.text.len()
            );
            let trimmed = chunk.text.trim();
            assert!(
                trimmed.ends_with('.') || chunk == chunks.last().unwrap(),
                "split did not land on a sentence boundary: {trimmed:?}"
            );
            // Never mid-word: every chunk boundary lines up with whitespace
            // on at least one side once trimmed.
            assert!(!trimmed.is_empty());
        }
    }

    #[test]
    fn chunk_never_exceeds_the_512_token_minilm_hard_cap_approximation() {
        let long_text = "word ".repeat(2000); // 10,000 bytes, no punctuation
        let events = vec![prose(1, EventKind::Prompt, &long_text)];
        let chunks = chunk_turns(&events, 1500);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.text.len() <= 1500);
        }
    }

    #[test]
    fn chunk_text_is_redacted() {
        let text = "here is a key AKIAIOSFODNN7EXAMPLE in the prompt";
        let events = vec![prose(1, EventKind::Prompt, text)];
        let chunks = chunk_turns(&events, 1500);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("[REDACTED:aws_access_key]"));
        assert!(!chunks[0].text.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn empty_text_events_are_skipped() {
        let events = vec![
            prose(1, EventKind::Prompt, ""),
            prose(2, EventKind::Prompt, "actual content"),
        ];
        let chunks = chunk_turns(&events, 1500);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "actual content");
    }
}
