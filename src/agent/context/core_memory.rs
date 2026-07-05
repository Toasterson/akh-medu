//! KG-backed **core memory** store for the [`VirtualContextManager`] (Phase 44i).
//!
//! Core memory is the MemGPT "core" tier: a small, always-in-context, model-
//! editable scratchpad (persona facts, the user's name, the standing task…). It
//! lives in the KG — not on the `Agent` — because the self-edit tools only get
//! `&Engine`, so the KG is the one store both the manager (read) and the tools
//! (write) can reach. Each agent has an anchor entity `core_memory:<agent_id>`;
//! every line is a sequence-tagged entity (`core_mem:<agent_id>:<seq>:<text>`)
//! linked by the well-known `agent:core_memory_line` predicate. Sequence tags
//! give a stable display order without a separate ordering structure.

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

/// Well-known predicate linking an agent's core-memory anchor to its lines.
const LINE_PREDICATE: &str = "agent:core_memory_line";

/// Hard cap on core-memory lines — the tier is meant to stay small and always in
/// context. When full, [`append`] evicts the oldest line (a rolling buffer).
pub const MAX_LINES: usize = 16;

fn anchor(engine: &Engine, agent_id: &str) -> AkhResult<SymbolId> {
    engine.resolve_or_create_entity(&format!("core_memory:{agent_id}"))
}

fn line_predicate(engine: &Engine) -> AkhResult<SymbolId> {
    engine.resolve_or_create_entity(LINE_PREDICATE)
}

/// Parse a line entity label `core_mem:<agent>:<seq>:<text>` → `(seq, text)`.
fn parse_line(label: &str) -> Option<(u64, String)> {
    let mut parts = label.splitn(4, ':');
    if parts.next()? != "core_mem" {
        return None;
    }
    let _agent = parts.next()?;
    let seq: u64 = parts.next()?.parse().ok()?;
    let text = parts.next()?.to_string();
    Some((seq, text))
}

/// All `(line_entity, seq, text)` for an agent, unsorted.
fn lines_with_seq(engine: &Engine, agent_id: &str) -> Vec<(SymbolId, u64, String)> {
    let (Ok(a), Ok(p)) = (anchor(engine, agent_id), line_predicate(engine)) else {
        return Vec::new();
    };
    engine
        .triples_from(a)
        .into_iter()
        .filter(|t| t.predicate == p)
        .filter_map(|t| parse_line(&engine.resolve_label(t.object)).map(|(s, txt)| (t.object, s, txt)))
        .collect()
}

/// Read the agent's core-memory lines in sequence order.
pub fn read(engine: &Engine, agent_id: &str) -> Vec<String> {
    let mut lines = lines_with_seq(engine, agent_id);
    lines.sort_by_key(|(_, seq, _)| *seq);
    lines.into_iter().map(|(_, _, text)| text).collect()
}

/// Append a line to core memory (idempotent on identical text). When the tier is
/// full, the oldest line is evicted first so the buffer stays bounded. Returns
/// the new line count.
pub fn append(engine: &Engine, agent_id: &str, text: &str) -> AkhResult<usize> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(read(engine, agent_id).len());
    }
    let mut existing = lines_with_seq(engine, agent_id);
    if existing.iter().any(|(_, _, t)| t == text) {
        return Ok(existing.len()); // already present — idempotent
    }

    let a = anchor(engine, agent_id)?;
    let p = line_predicate(engine)?;

    // Rolling eviction: drop the oldest line(s) until there's room.
    existing.sort_by_key(|(_, seq, _)| *seq);
    while existing.len() >= MAX_LINES {
        let (oldest, _, _) = existing.remove(0);
        let _ = engine.remove_triple(a, p, oldest);
    }

    let next_seq = existing.iter().map(|(_, seq, _)| *seq).max().map(|m| m + 1).unwrap_or(0);
    let line = engine.resolve_or_create_entity(&format!("core_mem:{agent_id}:{next_seq}:{text}"))?;
    engine.add_triple(&Triple::new(a, p, line))?;
    Ok(existing.len() + 1)
}

/// Replace the first core-memory line whose text equals `old` with `new`,
/// preserving its position. Returns `true` if a matching line was found.
pub fn replace(engine: &Engine, agent_id: &str, old: &str, new: &str) -> AkhResult<bool> {
    let (old, new) = (old.trim(), new.trim());
    let a = anchor(engine, agent_id)?;
    let p = line_predicate(engine)?;
    let target = lines_with_seq(engine, agent_id)
        .into_iter()
        .find(|(_, _, text)| text == old);
    let Some((line_entity, seq, _)) = target else {
        return Ok(false);
    };
    engine.remove_triple(a, p, line_entity)?;
    if !new.is_empty() {
        let replacement =
            engine.resolve_or_create_entity(&format!("core_mem:{agent_id}:{seq}:{new}"))?;
        engine.add_triple(&Triple::new(a, p, replacement))?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let e = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (e, tmp)
    }

    #[test]
    fn append_read_replace_roundtrip() {
        let (e, _t) = engine();
        assert!(read(&e, "alice").is_empty());

        append(&e, "alice", "user's name is Toni").unwrap();
        append(&e, "alice", "prefers concise answers").unwrap();
        assert_eq!(read(&e, "alice"), vec!["user's name is Toni", "prefers concise answers"]);

        // Idempotent.
        append(&e, "alice", "user's name is Toni").unwrap();
        assert_eq!(read(&e, "alice").len(), 2);

        // Replace preserves order.
        assert!(replace(&e, "alice", "prefers concise answers", "prefers detailed answers").unwrap());
        assert_eq!(read(&e, "alice"), vec!["user's name is Toni", "prefers detailed answers"]);

        // Replace of a missing line is a no-op false.
        assert!(!replace(&e, "alice", "nope", "x").unwrap());
    }

    #[test]
    fn per_agent_isolation() {
        let (e, _t) = engine();
        append(&e, "a", "fact A").unwrap();
        append(&e, "b", "fact B").unwrap();
        assert_eq!(read(&e, "a"), vec!["fact A"]);
        assert_eq!(read(&e, "b"), vec!["fact B"]);
    }

    #[test]
    fn rolling_eviction_keeps_bound() {
        let (e, _t) = engine();
        for i in 0..(MAX_LINES + 4) {
            append(&e, "a", &format!("line {i}")).unwrap();
        }
        let lines = read(&e, "a");
        assert_eq!(lines.len(), MAX_LINES, "bounded to MAX_LINES");
        // The oldest were evicted; the newest survives.
        assert_eq!(lines.last().unwrap(), &format!("line {}", MAX_LINES + 3));
        assert!(!lines.contains(&"line 0".to_string()), "oldest evicted");
    }
}
