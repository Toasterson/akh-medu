//! `get_session`'s reading-order algorithms — event-normal-form.md §7:
//! "the canonical linearization is depth-first: a subagent stream is
//! inserted into its parent at the `tool_call` [and this crate's
//! `lifecycle`] that spawned it."
//!
//! The spawn link is only *proven* for Claude Code (`started`/`result`
//! lifecycle records carry `agentId`, which is exactly the spawned stream's
//! `stream_id`). For Codex/OpenCode the ticket's residual risk says the link
//! is unverified and reading order there should "fall back to stream-order
//! concatenation — a degradation, not a failure". [`spliced_order`] is
//! written to degrade exactly that way on its own, without special-casing a
//! source: a stream that no lifecycle event ever names as `agent_id` is
//! simply never spliced in as a child, so it surfaces as a top-level stream
//! instead — which for Codex's sequential resume-rollouts (no cross-stream
//! `agent_id` link at all, per the residual-risk note) is exactly "stream
//! order", just applied to top-level streams ordered by first `observed_at`
//! rather than by `stream_id` (see the ordering note below on why).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::normalize::NormalizedEvent;

/// Convenience: caller already grouped `Vec<NormalizedEvent>` by stream.
pub type ByStream = BTreeMap<String, Vec<NormalizedEvent>>;

/// Group a flat list of one session's events by `stream_id`, sorting each
/// stream's events by `(seq, subseq)`.
pub fn group_by_stream(events: Vec<NormalizedEvent>) -> ByStream {
    let mut by_stream: ByStream = BTreeMap::new();
    for event in events {
        by_stream
            .entry(event.key.stream_id.clone())
            .or_default()
            .push(event);
    }
    for events in by_stream.values_mut() {
        events.sort_by_key(|e| (e.key.seq, e.subseq.unwrap_or(0)));
    }
    by_stream
}

/// Plain `(stream_id, seq, subseq)` order — `get_session(order = "stream")`.
/// `BTreeMap` iteration is already lexicographic on `stream_id`, so this is
/// just a flatten.
pub fn stream_order(by_stream: &ByStream) -> Vec<NormalizedEvent> {
    by_stream.values().flatten().cloned().collect()
}

/// Depth-first spliced order — `get_session(order = "spliced")`, the
/// default. A subagent stream is inserted immediately after the `lifecycle`
/// event (any subkind) whose `agent_id` equals that stream's `stream_id`,
/// recursively (nested subagents splice into their own parent the same way).
/// Any stream no lifecycle event ever names is a top-level stream; top-level
/// streams are concatenated with `main` (if present) first, then by first
/// `observed_at` ascending — the natural "session order" for sources with no
/// spawn-link at all (Codex's sequential resumes, OpenCode's single stream).
/// A stream that is never reached at all (a cycle, or a dangling spawn link
/// with no matching top-level ancestor) is appended at the end in
/// `stream_id` order as a safety net — every input event appears in the
/// output exactly once, unconditionally.
pub fn spliced_order(by_stream: &ByStream) -> Vec<NormalizedEvent> {
    // child_stream_id -> (parent_stream_id, parent_seq, parent_subseq).
    // `or_insert` keeps the *first* spawn record seen (a stream's `started`
    // rather than a later `result` naming the same agent_id), which matches
    // "inserted at the tool_call/lifecycle that spawned it".
    let mut spawn_of: HashMap<String, (String, i64, u32)> = HashMap::new();
    for (stream_id, events) in by_stream {
        for event in events {
            if event.kind != crate::model::EventKind::Lifecycle {
                continue;
            }
            let Some(agent_id) = &event.agent_id else {
                continue;
            };
            if agent_id == stream_id || !by_stream.contains_key(agent_id) {
                continue;
            }
            spawn_of.entry(agent_id.clone()).or_insert((
                stream_id.clone(),
                event.key.seq,
                event.subseq.unwrap_or(0),
            ));
        }
    }

    let mut top_level: Vec<&String> = by_stream
        .keys()
        .filter(|s| !spawn_of.contains_key(s.as_str()))
        .collect();
    top_level.sort_by_key(|stream_id| {
        let is_main = stream_id.as_str() != "main";
        let first_observed = by_stream
            .get(*stream_id)
            .and_then(|events| events.first())
            .map(|e| e.observed_at);
        (is_main, first_observed, (*stream_id).clone())
    });

    let mut consumed: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for stream_id in top_level {
        splice_stream(stream_id, by_stream, &spawn_of, &mut consumed, &mut out);
    }

    // Safety net: a stream that spawn_of names as a child but whose parent
    // chain never reached a top-level root (a cycle, or a parent stream
    // missing from `by_stream`) still must not silently disappear.
    let mut leftover: Vec<&String> = by_stream
        .keys()
        .filter(|s| !consumed.contains(s.as_str()))
        .collect();
    leftover.sort();
    for stream_id in leftover {
        if let Some(events) = by_stream.get(stream_id) {
            out.extend(events.iter().cloned());
        }
    }

    out
}

fn splice_stream(
    stream_id: &str,
    by_stream: &ByStream,
    spawn_of: &HashMap<String, (String, i64, u32)>,
    consumed: &mut HashSet<String>,
    out: &mut Vec<NormalizedEvent>,
) {
    if !consumed.insert(stream_id.to_string()) {
        return; // already spliced in — cycle guard.
    }
    let Some(events) = by_stream.get(stream_id) else {
        return;
    };

    let mut children_at: HashMap<(i64, u32), Vec<String>> = HashMap::new();
    for (child, (parent, seq, subseq)) in spawn_of {
        if parent == stream_id {
            children_at
                .entry((*seq, *subseq))
                .or_default()
                .push(child.clone());
        }
    }

    for event in events {
        out.push(event.clone());
        if let Some(children) = children_at.get(&(event.key.seq, event.subseq.unwrap_or(0))) {
            let mut children = children.clone();
            children.sort();
            for child in children {
                splice_stream(&child, by_stream, spawn_of, consumed, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone, Utc};

    use crate::model::{EventKind, RecordKey, Source};
    use crate::normalize::NormalizedEvent;

    use super::*;

    fn base_time() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 19, 0, 0, 0).unwrap()
    }

    fn ev(stream_id: &str, seq: i64, kind: EventKind, offset_secs: i64) -> NormalizedEvent {
        NormalizedEvent {
            key: RecordKey {
                source: Source::ClaudeCode,
                session_id: "s1".to_string(),
                stream_id: stream_id.to_string(),
                seq,
            },
            subseq: None,
            observed_at: base_time() + Duration::seconds(offset_secs),
            kind,
            subkind: None,
            role: None,
            text: None,
            tool_name: None,
            tool_call_id: None,
            agent_id: None,
            parent_agent_id: None,
            degraded: false,
            degraded_reason: None,
            repo: None,
            workspace: None,
            branch: None,
        }
    }

    fn lifecycle_spawn(
        stream_id: &str,
        seq: i64,
        agent_id: &str,
        offset_secs: i64,
    ) -> NormalizedEvent {
        let mut e = ev(stream_id, seq, EventKind::Lifecycle, offset_secs);
        e.subkind = Some("subagent_started".to_string());
        e.agent_id = Some(agent_id.to_string());
        e
    }

    #[test]
    fn stream_order_is_plain_stream_then_seq() {
        let events = vec![
            ev("main", 2, EventKind::Reply, 2),
            ev("main", 1, EventKind::Prompt, 1),
            ev("agent-a", 1, EventKind::Prompt, 0),
        ];
        let by_stream = group_by_stream(events);
        let ordered = stream_order(&by_stream);
        let ids: Vec<(&str, i64)> = ordered
            .iter()
            .map(|e| (e.key.stream_id.as_str(), e.key.seq))
            .collect();
        assert_eq!(ids, vec![("agent-a", 1), ("main", 1), ("main", 2)]);
    }

    #[test]
    fn splices_one_subagent_at_its_spawn_point() {
        let events = vec![
            ev("main", 1, EventKind::Prompt, 0),
            lifecycle_spawn("main", 2, "agent-a", 1),
            ev("main", 3, EventKind::Reply, 5),
            ev("agent-a", 1, EventKind::Prompt, 2),
            ev("agent-a", 2, EventKind::Reply, 3),
        ];
        let by_stream = group_by_stream(events);
        let ordered = spliced_order(&by_stream);
        let ids: Vec<(&str, i64)> = ordered
            .iter()
            .map(|e| (e.key.stream_id.as_str(), e.key.seq))
            .collect();
        assert_eq!(
            ids,
            vec![
                ("main", 1),
                ("main", 2),
                ("agent-a", 1),
                ("agent-a", 2),
                ("main", 3),
            ]
        );
    }

    #[test]
    fn splices_a_nested_subagent_two_levels_deep() {
        let events = vec![
            ev("main", 1, EventKind::Prompt, 0),
            lifecycle_spawn("main", 2, "agent-a", 1),
            ev("main", 3, EventKind::Reply, 10),
            ev("agent-a", 1, EventKind::Prompt, 2),
            lifecycle_spawn("agent-a", 2, "agent-b", 3),
            ev("agent-a", 3, EventKind::Reply, 6),
            ev("agent-b", 1, EventKind::Prompt, 4),
            ev("agent-b", 2, EventKind::Reply, 5),
        ];
        let by_stream = group_by_stream(events);
        let ordered = spliced_order(&by_stream);
        let ids: Vec<(&str, i64)> = ordered
            .iter()
            .map(|e| (e.key.stream_id.as_str(), e.key.seq))
            .collect();
        assert_eq!(
            ids,
            vec![
                ("main", 1),
                ("main", 2),
                ("agent-a", 1),
                ("agent-a", 2),
                ("agent-b", 1),
                ("agent-b", 2),
                ("agent-a", 3),
                ("main", 3),
            ]
        );
    }

    #[test]
    fn every_event_appears_exactly_once_across_both_orders() {
        let events = vec![
            ev("main", 1, EventKind::Prompt, 0),
            lifecycle_spawn("main", 2, "agent-a", 1),
            ev("main", 3, EventKind::Reply, 10),
            ev("agent-a", 1, EventKind::Prompt, 2),
            ev("agent-a", 2, EventKind::Reply, 3),
        ];
        let total = events.len();
        let by_stream = group_by_stream(events);
        for ordered in [spliced_order(&by_stream), stream_order(&by_stream)] {
            assert_eq!(ordered.len(), total);
            let mut seen = HashSet::new();
            for e in &ordered {
                assert!(
                    seen.insert((e.key.stream_id.clone(), e.key.seq)),
                    "duplicate event in order"
                );
            }
        }
    }

    #[test]
    fn streams_with_no_spawn_link_fall_back_to_observed_at_order() {
        // Codex-shaped: two sibling streams, no lifecycle event links them —
        // "stream-order concatenation" degrades to arrival order rather than
        // lexicographic stream_id order, since Codex stream ids are rollout
        // UUIDs with no inherent sequence.
        let events = vec![
            ev(
                "11111111-0000-0000-0000-000000000000",
                1,
                EventKind::Prompt,
                10,
            ),
            ev(
                "00000000-0000-0000-0000-000000000000",
                1,
                EventKind::Prompt,
                0,
            ),
        ];
        let by_stream = group_by_stream(events);
        let ordered = spliced_order(&by_stream);
        assert_eq!(
            ordered[0].key.stream_id,
            "00000000-0000-0000-0000-000000000000"
        );
        assert_eq!(
            ordered[1].key.stream_id,
            "11111111-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn main_stream_always_leads_top_level_order() {
        let events = vec![
            ev("zzz-early-stream", 1, EventKind::Prompt, 0),
            ev("main", 1, EventKind::Prompt, 100),
        ];
        let by_stream = group_by_stream(events);
        let ordered = spliced_order(&by_stream);
        assert_eq!(ordered[0].key.stream_id, "main");
    }

    #[test]
    fn a_stream_never_named_by_any_lifecycle_is_not_lost() {
        let events = vec![
            ev("main", 1, EventKind::Prompt, 0),
            ev("orphan", 1, EventKind::Prompt, 1),
        ];
        let by_stream = group_by_stream(events);
        let ordered = spliced_order(&by_stream);
        assert_eq!(ordered.len(), 2);
        assert!(ordered.iter().any(|e| e.key.stream_id == "orphan"));
    }
}
