//! Sleep-cycle harvest phase (ADR 091 increment 2): projects raw `kind="turn"`
//! WAL rows into lightweight episode-index symbols in the KG.
//!
//! The capture hook (`agentic_chat.rs::emit_turn_wal`, Wave 1) appends one
//! `kind='turn'` row per completed agentic turn to `akhomed_mutation_log` —
//! undistilled, staging only. This module is the other end of the pipe: it
//! reads the un-harvested tail (since the per-akh `harvest_watermark` in
//! `akh_state`), mints one `episode:turn:{seq}` entity per turn (keyword→symbol
//! links, tagged, typed as episodic memory, summarized, provenance-tracked),
//! and advances the watermark past everything it processed.
//!
//! Distillation stays crude on purpose (ADR 091 decision 1/8): the gpt-oss
//! Librarian (ADR 081 slice 4) later replaces the keyword extraction with
//! typed triples over the same raw turns; this phase only has to make the
//! turns *findable* by `recall_episodes` (`memory.rs`), not well-understood.

use akhomed_store::DatabaseConnection;
use akhomed_store::uuid::Uuid;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolId, SymbolKind};

use super::agent::AgentPredicates;
use super::error::AgentResult;

/// Batch size cap for a single [`harvest_daily_log`] call — bounds how many
/// turns one sleep-cycle tick will mint, so a very chatty akh degrades
/// gracefully (subsequent sleep ticks pick up where this one left off) rather
/// than blocking a tick on an unbounded backlog.
const HARVEST_BATCH_LIMIT: u64 = 256;

/// Outcome of one [`harvest_daily_log`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HarvestMetrics {
    /// Turn rows read from the WAL tail (includes skipped/malformed ones).
    pub turns_harvested: usize,
    /// Episode-index symbols actually minted (excludes skipped/malformed turns).
    pub episodes_minted: usize,
    /// The watermark after this call (unchanged from the input watermark if
    /// nothing new was read).
    pub last_seq: i64,
}

/// Read un-harvested `kind='turn'` rows for `akh_id` and mint an episode index
/// per turn, advancing the per-akh harvest watermark.
///
/// **Sync/async boundary:** this function is deliberately **synchronous**. It
/// is called from `AgentDaemon::run_sleep_cycle` while the daemon holds the
/// `std::sync::Mutex<Agent>` inside `tokio::task::block_in_place` — ADR 027
/// forbids holding that lock across an `.await`, so the three store calls
/// (watermark read → turn-tail read → watermark write) run via
/// `tokio::runtime::Handle::current().block_on(...)` internally instead of
/// this function being `async fn`. Callers must invoke it from within a tokio
/// runtime context (true for every daemon/TUI/MCP surface that runs the sleep
/// cycle today); `block_on` inside `block_in_place` is exactly the pattern
/// `tokio::task::block_in_place`'s own docs recommend for this situation.
///
/// A malformed turn payload (missing/empty `user_prompt` or `agent_answer`) is
/// logged and skipped — it still counts toward `turns_harvested` and the
/// watermark still advances past it, so one poison row cannot wedge harvest
/// forever. A turn that fails to mint for another reason (e.g. a transient
/// engine error) is also skipped-with-warning for the same reason; harvest is
/// best-effort progress, not all-or-nothing.
pub fn harvest_daily_log(
    db: &DatabaseConnection,
    engine: &Engine,
    predicates: &AgentPredicates,
    akh_id: Uuid,
) -> AgentResult<HarvestMetrics> {
    let handle = tokio::runtime::Handle::current();

    let watermark = handle.block_on(akhomed_store::get_harvest_watermark(db, akh_id))?;
    let turns = handle.block_on(akhomed_store::load_turns_since(
        db,
        akh_id,
        watermark,
        HARVEST_BATCH_LIMIT,
    ))?;

    let mut metrics = HarvestMetrics {
        turns_harvested: 0,
        episodes_minted: 0,
        last_seq: watermark,
    };

    for turn in &turns {
        metrics.turns_harvested += 1;
        metrics.last_seq = turn.seq;

        match mint_turn_episode(engine, predicates, turn.seq, &turn.payload) {
            Ok(Some(_episode_id)) => metrics.episodes_minted += 1,
            Ok(None) => {
                tracing::warn!(
                    seq = turn.seq,
                    "harvest: skipped malformed turn payload (missing/empty user_prompt or agent_answer)"
                );
            }
            Err(e) => {
                tracing::warn!(
                    seq = turn.seq,
                    error = %e,
                    "harvest: failed to mint episode for turn, skipping"
                );
            }
        }
    }

    if metrics.last_seq > watermark {
        handle.block_on(akhomed_store::set_harvest_watermark(
            db,
            akh_id,
            metrics.last_seq,
        ))?;
    }

    Ok(metrics)
}

/// Mint a single `episode:turn:{seq}` index symbol from one turn's payload.
///
/// Factored out of [`harvest_daily_log`] so the minting logic is testable
/// against a plain in-memory `Engine` without a database — the caller supplies
/// the already-loaded `payload` JSON (the locked schema from
/// `agentic_chat.rs::emit_turn_wal`: `{v, ts, workspace, model, user_prompt,
/// agent_answer, tools}`).
///
/// Returns `Ok(None)` (not an error) when the payload is malformed — missing
/// or empty `user_prompt`/`agent_answer` — so the caller can log-and-skip
/// without treating a poison row as fatal.
pub fn mint_turn_episode(
    engine: &Engine,
    predicates: &AgentPredicates,
    seq: i64,
    payload: &serde_json::Value,
) -> AgentResult<Option<SymbolId>> {
    let user_prompt = payload
        .get("user_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let agent_answer = payload
        .get("agent_answer")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if user_prompt.trim().is_empty() && agent_answer.trim().is_empty() {
        return Ok(None);
    }

    // 1. Keywords from the combined turn text (reuses the same extraction the
    //    rest of KG grounding uses — no new NLP for this crude first slice).
    let text = format!("{user_prompt} {agent_answer}");
    let keywords = crate::nlu::kg_context::extract_keywords(&text);

    // 2. Resolve/create a symbol per keyword.
    let mut syms: Vec<SymbolId> = Vec::with_capacity(keywords.len());
    for keyword in &keywords {
        syms.push(engine.resolve_or_create_entity(keyword)?);
    }

    // 3. Mint the episode index. The label carries only the seq (the body
    //    stays in the WAL, paged by `seq` at recall time — ADR 091 decision 5);
    //    unlike `persist_episode`'s `episode:{id}:{excerpt}` convention, no
    //    transcript text is baked into the label.
    let episode_label = format!("episode:turn:{seq}");
    let episode_id = engine.create_symbol(SymbolKind::Entity, &episode_label)?.id;

    // 4. Links, following the `persist_episode` (memory.rs) template.
    let mut triples = syms
        .iter()
        .map(|symbol| Triple::new(episode_id, predicates.learned, *symbol))
        .collect::<Vec<_>>();

    let conversation_tag = engine.resolve_or_create_entity("tag:conversation")?;
    triples.push(Triple::new(
        episode_id,
        predicates.has_tag,
        conversation_tag,
    ));

    // REQUIRED: `reconstruct_episode` (memory.rs) rejects episodes without a
    // `memory_type` link — without this, recall never sees these episodes.
    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
    triples.push(Triple::new(
        episode_id,
        predicates.memory_type,
        mem_type_sym,
    ));

    // Excerpt entity: first ~80 chars of `user_prompt`, capped at a safe UTF-8
    // boundary (this is recall's display content until body-paging lands).
    let excerpt = char_boundary_truncate(user_prompt, 80);
    let summary_label = format!("summary:{excerpt}");
    let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
    triples.push(Triple::new(
        episode_id,
        predicates.has_summary,
        summary_sym,
    ));

    // 5. Provenance: reason string carries the source seq as the Librarian's
    // future re-projection key (ADR 091: no new DerivationKind variant).
    let mut prov = ProvenanceRecord::new(
        episode_id,
        DerivationKind::AgentConsolidation {
            reason: format!("transcript_harvest:{seq}"),
            relevance_score: 0.5,
        },
    )
    .with_sources(syms)
    .with_confidence(0.5);
    engine.add_triples_with_provenance(&mut triples, &mut prov)?;

    Ok(Some(episode_id))
}

/// Truncate `s` to at most `max_bytes` bytes at a safe UTF-8 char boundary
/// (byte-slicing at a fixed offset panics on multi-byte sequences).
fn char_boundary_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let end = s
        .char_indices()
        .take_while(|(i, _)| *i <= max_bytes)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    /// A persistent (temp-dir-backed) engine — required for `provenance_of` to
    /// work at all (`Engine::provenance_of` errors with `NoPersistence` on a
    /// pure in-memory engine, by design; see `engine::tests::provenance_requires_persistence`).
    /// The `TempDir` must outlive the engine's use in the test, so it is
    /// returned alongside it.
    fn test_engine() -> (tempfile::TempDir, Engine, AgentPredicates) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        let predicates = AgentPredicates::init(&engine).unwrap();
        (dir, engine, predicates)
    }

    fn turn_payload(user_prompt: &str, agent_answer: &str) -> serde_json::Value {
        serde_json::json!({
            "v": 1,
            "ts": "2026-07-10T00:00:00.000000000Z",
            "workspace": "test-ws",
            "model": "test-model",
            "user_prompt": user_prompt,
            "agent_answer": agent_answer,
            "tools": [],
        })
    }

    #[test]
    fn mints_episode_with_required_links_and_provenance() {
        let (_dir, engine, predicates) = test_engine();
        let payload = turn_payload(
            "What is the capital of France?",
            "The capital of France is Paris.",
        );

        let episode_id = mint_turn_episode(&engine, &predicates, 42, &payload)
            .unwrap()
            .expect("well-formed payload should mint an episode");

        assert_eq!(engine.resolve_label(episode_id), "episode:turn:42");

        let triples = engine.triples_from(episode_id);

        // memory_type link — REQUIRED by reconstruct_episode.
        assert!(
            triples
                .iter()
                .any(|t| t.predicate == predicates.memory_type),
            "episode must carry a memory_type link"
        );
        // has_tag tag:conversation.
        let has_conversation_tag = triples.iter().any(|t| {
            t.predicate == predicates.has_tag
                && engine.resolve_label(t.object) == "tag:conversation"
        });
        assert!(
            has_conversation_tag,
            "episode must be tagged tag:conversation"
        );
        // has_summary excerpt of user_prompt.
        let has_summary = triples
            .iter()
            .any(|t| t.predicate == predicates.has_summary);
        assert!(has_summary, "episode must carry a has_summary link");
        // learned links to at least one keyword symbol.
        let learned_count = triples
            .iter()
            .filter(|t| t.predicate == predicates.learned)
            .count();
        assert!(
            learned_count > 0,
            "episode must learn at least one keyword symbol"
        );

        // Provenance recorded with the seq-carrying reason string.
        let prov = engine.provenance_of(episode_id).unwrap();
        assert!(
            prov.iter().any(|p| matches!(
                &p.kind,
                DerivationKind::AgentConsolidation { reason, .. } if reason == "transcript_harvest:42"
            )),
            "provenance must carry the transcript_harvest:{{seq}} reason"
        );
    }

    #[test]
    fn malformed_payload_is_skipped_without_error() {
        let (_dir, engine, predicates) = test_engine();

        // Missing both fields entirely.
        let payload = serde_json::json!({ "v": 1 });
        let result = mint_turn_episode(&engine, &predicates, 1, &payload).unwrap();
        assert!(result.is_none(), "missing fields should skip, not error");

        // Present but empty.
        let payload = turn_payload("", "");
        let result = mint_turn_episode(&engine, &predicates, 2, &payload).unwrap();
        assert!(result.is_none(), "empty fields should skip, not error");

        // No episode symbols should have been created.
        assert!(engine.resolve_symbol("episode:turn:1").is_err());
        assert!(engine.resolve_symbol("episode:turn:2").is_err());
    }

    #[test]
    fn one_sided_payload_still_mints() {
        let (_dir, engine, predicates) = test_engine();
        // agent_answer empty but user_prompt present is still "some content".
        let payload = turn_payload("Tell me about Rust ownership.", "");
        let result = mint_turn_episode(&engine, &predicates, 7, &payload).unwrap();
        assert!(
            result.is_some(),
            "a turn with content on one side should still mint"
        );
    }
}
