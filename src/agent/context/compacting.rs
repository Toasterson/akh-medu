//! The default curating context manager (Phase 44c — first increment).
//!
//! Measure-first Q1 showed conversation history demonstrably improves NL quality
//! (a follow-up that needs prior context fails without it), and the agentic path
//! shipped *no* history. This manager fixes that: it renders the recent
//! conversation turns into the prompt, newest-first up to the token budget. The
//! windows are large (128K–1M, read from the model — see the measure-first
//! report), so in practice all recent turns fit; the budget check is the cap for
//! the rare long transcript.
//!
//! Graduated fidelity (full → summary → reference-handle → dropped) and the KG /
//! recalled-episode sections are later increments (44c-full / 44d / 44e / 44g);
//! this increment is verbatim recent history, the highest-value piece.

use std::collections::BTreeMap;

use serde_json::json;

use crate::agent::conversation::Speaker;
use crate::library::note::{FrontmatterValue, Note};

use super::budget::BudgetAllocator;
use super::error::ContextResult;
use super::manager::{ContextManager, TokenCounter};
use super::types::{
    AssembledContext, ContextSection, DroppedSpan, EpisodeState, StrategyId, TokenBudget,
};

/// Render a `key: value` frontmatter block for each note (title-labelled),
/// **never** touching [`Note::body`] — this is the mechanical enforcement of
/// the "bodies never resident" invariant (ADR 081 slice 7 decision 11):
/// however a caller populated `active_notes` / `recent_notes` (even with
/// bodies attached), this function has no way to render one. Notes with no
/// frontmatter contribute nothing (there is nothing to show in a
/// frontmatter-only tier).
fn render_notes_frontmatter(notes: &[Note]) -> String {
    let mut out = String::new();
    for note in notes {
        if note.frontmatter.is_empty() {
            continue;
        }
        let title = match note.frontmatter.get("title") {
            Some(FrontmatterValue::Text(t)) => t.clone(),
            _ => note.document.title.clone(),
        };
        out.push_str(&format!("### {title} (note:{})\n", note.id.as_str()));
        out.push_str(&note.frontmatter.to_block());
        out.push('\n');
    }
    out
}

/// Assembles recent conversation history + the current turn within budget.
///
/// Expects the caller to have recorded the current user turn into the
/// [`ConversationState`](crate::agent::conversation::ConversationState) before
/// assembly, so the conversation's last turn *is* the current prompt — this
/// manager renders the turns as-is (no separate append, no duplication).
#[derive(Debug, Default, Clone, Copy)]
pub struct CompactingManager;

impl ContextManager for CompactingManager {
    fn id(&self) -> StrategyId {
        StrategyId::new("compacting")
    }

    fn assemble(
        &self,
        episode: &EpisodeState<'_>,
        budget: &TokenBudget,
        counter: &dyn TokenCounter,
    ) -> ContextResult<AssembledContext> {
        assemble_with_cap(episode, budget, counter, None)
    }
}

/// Shared section assembly for the curating managers (44b/44e).
///
/// `history_cap` bounds how many of the most-recent turns are eligible (older
/// turns are recorded as dropped): `None` for [`CompactingManager`] (all turns
/// within budget), `Some(tail)` for the
/// [`PassiveManager`](super::passive::PassiveManager), which leans on background
/// consolidation + recall to carry the longer-term thread instead of verbatim
/// history.
pub(super) fn assemble_with_cap(
    episode: &EpisodeState<'_>,
    budget: &TokenBudget,
    counter: &dyn TokenCounter,
    history_cap: Option<usize>,
) -> ContextResult<AssembledContext> {
        let available = budget.available();

        // Keep-first priority (what survives budget pressure): continuity seed →
        // schema digest → active-note frontmatter (the core tier, ADR 081 slice
        // 7) → the question → grounding facts → recently-touched-note
        // frontmatter → recalled episodes (the recall tier) → recent history →
        // tools (trimmed first). The *render* order differs (positional): blocks
        // lead, history + current prompt trail, closest to the generation point.
        let allocator = BudgetAllocator::new(vec![
            ContextSection::System,
            ContextSection::Schema,
            ContextSection::ActiveNotes,
            ContextSection::UserPrompt,
            ContextSection::KnowledgeGraph,
            ContextSection::RecentNotes,
            ContextSection::RecalledEpisodes,
            ContextSection::History,
            ContextSection::Tools,
        ]);

        // ── Candidate atomic blocks (each → one leading `system` message) ──
        let make_block = |section: ContextSection,
                          header: &str,
                          body: &str|
         -> Option<(ContextSection, String, usize)> {
            let body = body.trim();
            if body.is_empty() {
                return None;
            }
            let content = format!("{header}\n{body}");
            let tokens = counter.count(&content);
            Some((section, content, tokens))
        };
        let mut blocks: Vec<(ContextSection, String, usize)> = Vec::new();
        if let Some(seed) = episode.session_seed
            && let Some(b) = make_block(ContextSection::System, "Session so far:", seed)
        {
            blocks.push(b);
        }
        // Core tier (ADR 081 slice 7): the in-force schema digest, so the model
        // always knows the installed types/relations — degrades like every
        // other atomic section under budget pressure (an engine with no schema
        // pack installed yields an empty digest, so this is silently absent).
        if let Ok(digest) = crate::schema::digest(episode.engine)
            && !digest.is_empty()
            && let Some(b) = make_block(
                ContextSection::Schema,
                "In-force schema (get_graph_schema digest):",
                &digest.render_compact(),
            )
        {
            blocks.push(b);
        }
        // Core tier: active-note frontmatter only — never bodies (see
        // `render_notes_frontmatter`).
        if let Some(b) = make_block(
            ContextSection::ActiveNotes,
            "Active notes (frontmatter only):",
            &render_notes_frontmatter(episode.active_notes),
        ) {
            blocks.push(b);
        }
        if let Some(kg) = episode.kg_context
            && let Some(b) =
                make_block(ContextSection::KnowledgeGraph, "Relevant knowledge from your memory:", kg)
        {
            blocks.push(b);
        }
        // Recall tier: recently-touched-note frontmatter only — never bodies.
        if let Some(b) = make_block(
            ContextSection::RecentNotes,
            "Recently touched notes (frontmatter only):",
            &render_notes_frontmatter(episode.recent_notes),
        ) {
            blocks.push(b);
        }
        if !episode.recalled.is_empty() {
            let body = format!("- {}", episode.recalled.join("\n- "));
            if let Some(b) = make_block(ContextSection::RecalledEpisodes, "Relevant past episodes:", &body)
            {
                blocks.push(b);
            }
        }

        // ── Divisible sections: history turns + the tool catalog ──
        let history: Vec<(&'static str, &str, usize)> = episode
            .conversation
            .map(|c| {
                c.turns
                    .iter()
                    .map(|t| {
                        let role = match t.speaker {
                            Speaker::Operator => "user",
                            Speaker::Agent => "assistant",
                        };
                        (role, t.text.as_str(), counter.count(&t.text))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let history_total: usize = history.iter().map(|(_, _, t)| *t).sum();

        let tool_costs: Vec<usize> = episode
            .tool_catalog
            .iter()
            .map(|t| counter.count(&t.to_string()))
            .collect();
        let tools_total: usize = tool_costs.iter().sum();

        // ── Budget division (one allocator, not three independent caps) ──
        let mut requested: BTreeMap<ContextSection, usize> = BTreeMap::new();
        for (section, _, tokens) in &blocks {
            *requested.entry(*section).or_insert(0) += *tokens;
        }
        if history_total > 0 {
            requested.insert(ContextSection::History, history_total);
        } else {
            requested.insert(ContextSection::UserPrompt, counter.count(episode.user_prompt));
        }
        if tools_total > 0 {
            requested.insert(ContextSection::Tools, tools_total);
        }
        // Atomic sections render whole-or-not (no partial block without a summary —
        // that's 44c-full); history + tools degrade smoothly within their grant.
        let atomic = [
            ContextSection::System,
            ContextSection::Schema,
            ContextSection::ActiveNotes,
            ContextSection::KnowledgeGraph,
            ContextSection::RecentNotes,
            ContextSection::RecalledEpisodes,
            ContextSection::UserPrompt,
        ];
        let granted = allocator.allocate_keep_first(available, &requested, &atomic);

        // ── Render ──
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let mut section_tokens: BTreeMap<ContextSection, usize> = BTreeMap::new();
        let mut dropped: Vec<DroppedSpan> = Vec::new();

        // Leading atomic blocks, in render order (seed → kg → recalled). A block
        // that didn't win its grant is recorded as dropped (safety can veto).
        for (section, content, tokens) in &blocks {
            let grant = granted.get(section).copied().unwrap_or(0);
            if *tokens > 0 && grant >= *tokens {
                messages.push(json!({ "role": "system", "content": content }));
                *section_tokens.entry(*section).or_insert(0) += *tokens;
            } else {
                dropped.push(DroppedSpan::new(*section, content.clone(), None));
            }
        }

        // History within its grant: 44e discourse selection chooses *which* turns
        // (recency, or graph-proximity when the conversation carries entities),
        // always keeping the most recent turn (= the current prompt), then renders
        // the kept turns chronologically.
        if history_total > 0 {
            let conv = episode
                .conversation
                .expect("history_total > 0 implies a conversation is present");
            let grant = granted.get(&ContextSection::History).copied().unwrap_or(0);
            let proximity = super::discourse::has_proximity_signal(conv);
            // PassiveManager caps eligibility to the last `tail` turns (the
            // immediate exchange); the rest is carried by recall, not verbatim.
            let considered: Option<std::collections::HashSet<usize>> = history_cap.map(|n| {
                let start = history.len().saturating_sub(n.max(1));
                (start..history.len()).collect()
            });
            let mut used = 0usize;
            let mut keep: Vec<usize> = Vec::new();
            for i in super::discourse::rank_turns_keep_first(conv) {
                if let Some(set) = &considered
                    && !set.contains(&i)
                {
                    continue;
                }
                let t = history[i].2;
                if keep.is_empty() {
                    used += t;
                    keep.push(i);
                    continue;
                }
                if used + t > grant {
                    // Recency keeps a contiguous recent tail (stop); proximity may
                    // still fit a smaller, more-relevant earlier turn (skip on).
                    if proximity {
                        continue;
                    } else {
                        break;
                    }
                }
                used += t;
                keep.push(i);
            }
            keep.sort_unstable(); // chronological render order
            for &i in &keep {
                let (role, text, _) = history[i];
                messages.push(json!({ "role": role, "content": text }));
            }
            section_tokens.insert(ContextSection::History, used);
            for (i, (_, text, _)) in history.iter().enumerate() {
                if !keep.contains(&i) {
                    dropped.push(DroppedSpan::new(
                        ContextSection::History,
                        (*text).to_string(),
                        None,
                    ));
                }
            }
        } else {
            // No conversation ⇒ the current prompt as a single user turn (never
            // dropped, even under an impossibly tight budget).
            messages.push(json!({ "role": "user", "content": episode.user_prompt }));
            section_tokens.insert(ContextSection::UserPrompt, counter.count(episode.user_prompt));
        }

        // Tools within their grant: keep whole tools in catalog order until full
        // (the lowest-priority section — measure-first Q2's #1 lever for small
        // windows). With a large window the whole catalog passes unchanged.
        let mut tools: Vec<serde_json::Value> = Vec::new();
        if tools_total > 0 {
            let grant = granted.get(&ContextSection::Tools).copied().unwrap_or(0);
            let mut used = 0usize;
            for (tool, cost) in episode.tool_catalog.iter().zip(&tool_costs) {
                if used + *cost <= grant {
                    used += *cost;
                    tools.push(tool.clone());
                } else {
                    dropped.push(DroppedSpan::new(ContextSection::Tools, "(tool)", None));
                }
            }
            if used > 0 {
                section_tokens.insert(ContextSection::Tools, used);
            }
        }

        Ok(AssembledContext {
            messages,
            tools,
            section_tokens,
            dropped,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::conversation::ConversationState;
    use crate::agent::context::manager::HeuristicTokenCounter;
    use crate::engine::{Engine, EngineConfig};

    fn test_engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, tmp)
    }

    #[test]
    fn renders_recorded_history_as_roles() {
        let (engine, _tmp) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        conv.record_operator_turn("I'm thinking of the Eiffel Tower.");
        conv.record_agent_turn("Noted.");
        conv.record_operator_turn("What city is it in?"); // the current turn

        let episode =
            EpisodeState::new("a", "What city is it in?", &engine).with_conversation(Some(&conv));
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");

        // All three turns rendered, in order, with the right roles.
        assert_eq!(out.messages.len(), 3);
        assert_eq!(out.messages[0]["role"], "user");
        assert_eq!(out.messages[0]["content"], "I'm thinking of the Eiffel Tower.");
        assert_eq!(out.messages[1]["role"], "assistant");
        assert_eq!(out.messages[2]["role"], "user");
        assert_eq!(out.messages[2]["content"], "What city is it in?");
    }

    #[test]
    fn renders_kg_and_recalled_context_blocks() {
        let (engine, _tmp) = test_engine();
        let recalled = vec!["learned that Paris is the capital of France".to_string()];
        let episode = EpisodeState::new("a", "tell me about France", &engine).with_context(
            Some("France capital-is Paris"),
            &recalled,
            Some("Earlier you discussed travel plans."),
        );
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        // Three leading `system` context blocks (seed → kg → recalled), then the
        // current prompt as the final user turn.
        let systems = out.messages.iter().filter(|m| m["role"] == "system").count();
        assert_eq!(systems, 3, "seed + kg + recalled blocks: {:?}", out.messages);
        assert!(
            out.messages.iter().any(|m| m["content"].as_str().unwrap_or("").contains("Paris")),
            "KG/recalled content present"
        );
        assert_eq!(out.messages.last().unwrap()["role"], "user");
        assert!(out.section_tokens.contains_key(&ContextSection::KnowledgeGraph));
        assert!(out.section_tokens.contains_key(&ContextSection::RecalledEpisodes));
    }

    #[test]
    fn empty_conversation_is_passthrough_equivalent() {
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState::new("a", "hello", &engine);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert_eq!(out.messages, vec![json!({"role":"user","content":"hello"})]);
    }

    #[test]
    fn tools_pass_through_under_a_large_budget() {
        let (engine, _tmp) = test_engine();
        let tools = vec![
            json!({"type":"function","function":{"name":"a","description":"tool a"}}),
            json!({"type":"function","function":{"name":"b","description":"tool b"}}),
        ];
        let episode = EpisodeState::new("a", "hi", &engine).with_tools(&tools);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert_eq!(out.tools.len(), 2, "large budget keeps every tool");
        assert!(out.dropped.iter().all(|d| d.section != ContextSection::Tools));
    }

    #[test]
    fn tools_section_trimmed_under_pressure() {
        let (engine, _tmp) = test_engine();
        // Three ~10-token tools; a budget that fits the prompt + ~1 tool forces a trim.
        let tools: Vec<_> = (0..3)
            .map(|i| json!({"type":"function","function":{"name":format!("tool{i}"),"description":"xxxxxxxxxxxxxxxxxxxx"}}))
            .collect();
        let episode = EpisodeState::new("a", "hi", &engine).with_tools(&tools);
        // available = 30 - 6 = 24; the prompt is tiny, so a couple of tools fit, not all.
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::new(30, 6), &HeuristicTokenCounter)
            .expect("assemble");
        assert!(out.tools.len() < 3, "tool catalog trimmed to the budget: {}", out.tools.len());
        assert!(
            out.dropped.iter().any(|d| d.section == ContextSection::Tools),
            "trimmed tools recorded as dropped"
        );
    }

    // -- ADR 081 slice 7: schema digest in core --------------------------------

    const SEED_PACK: &str = r#"{
        "pack": {"name": "compacting-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "email", "value_type": "text"}]}
        ]
    }"#;

    fn install_seed(engine: &Engine) {
        let pack = crate::schema::compile::compile(SEED_PACK).unwrap();
        crate::schema::install::install_pack(engine, &pack).unwrap();
    }

    #[test]
    fn schema_digest_absent_when_no_packs_installed() {
        let (engine, _tmp) = test_engine();
        let episode = EpisodeState::new("a", "hello", &engine);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert!(
            !out.section_tokens.contains_key(&ContextSection::Schema),
            "no digest block without an installed schema pack"
        );
    }

    #[test]
    fn schema_digest_appears_in_core_when_packs_installed() {
        let (engine, _tmp) = test_engine();
        install_seed(&engine);
        let episode = EpisodeState::new("a", "hello", &engine);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert!(out.section_tokens.contains_key(&ContextSection::Schema));
        assert!(
            out.messages
                .iter()
                .any(|m| m["content"].as_str().unwrap_or("").contains("Person")),
            "the digest names the installed type: {:?}",
            out.messages
        );
    }

    #[test]
    fn schema_digest_dropped_under_tight_budget() {
        let (engine, _tmp) = test_engine();
        install_seed(&engine);
        let episode = EpisodeState::new("a", "hi", &engine);
        // A budget too tight to fit the (atomic) digest block at all: it must
        // degrade to absent, not truncated garbage, and the turn must still
        // succeed (the current prompt is never dropped).
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::new(6, 2), &HeuristicTokenCounter)
            .expect("assemble");
        assert!(
            !out.section_tokens.contains_key(&ContextSection::Schema),
            "digest didn't fit whole ⇒ dropped, not partially rendered"
        );
        assert!(
            out.dropped.iter().any(|d| d.section == ContextSection::Schema),
            "the drop is recorded for telemetry/safety"
        );
    }

    // -- ADR 081 slice 7: notes render frontmatter, never bodies ---------------

    fn note_with_body(title: &str, body: &str) -> crate::library::note::Note {
        use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
        use crate::library::note::{Frontmatter, FrontmatterValue};

        let doc = DocumentRecord {
            id: title.to_lowercase(),
            title: title.to_string(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        };
        let mut fm = Frontmatter::new();
        fm.insert("title", FrontmatterValue::Text(title.to_string()));
        crate::library::note::Note::new(doc, body).with_frontmatter(fm)
    }

    #[test]
    fn active_note_frontmatter_present_body_absent() {
        let (engine, _tmp) = test_engine();
        let note = note_with_body("Paris", "SECRET BODY TEXT that must never be resident.");
        let notes = vec![note];
        let episode =
            EpisodeState::new("a", "tell me about the city", &engine).with_notes(&notes, &[]);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");

        assert!(out.section_tokens.contains_key(&ContextSection::ActiveNotes));
        let rendered: String = out
            .messages
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("Paris"), "frontmatter title present: {rendered}");
        assert!(
            !rendered.contains("SECRET BODY TEXT"),
            "body must never be resident: {rendered}"
        );
    }

    #[test]
    fn recent_note_frontmatter_present_body_absent() {
        let (engine, _tmp) = test_engine();
        let note = note_with_body("Berlin", "SECRET BODY TEXT that must never be resident.");
        let notes = vec![note];
        let episode =
            EpisodeState::new("a", "what did I touch recently?", &engine).with_notes(&[], &notes);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");

        assert!(out.section_tokens.contains_key(&ContextSection::RecentNotes));
        let rendered: String = out
            .messages
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("Berlin"), "frontmatter title present: {rendered}");
        assert!(
            !rendered.contains("SECRET BODY TEXT"),
            "body must never be resident: {rendered}"
        );
    }

    #[test]
    fn frontmatter_less_notes_contribute_nothing() {
        let (engine, _tmp) = test_engine();
        let note = crate::library::note::Note::new(
            crate::library::model::DocumentRecord {
                id: "plain".into(),
                title: "Plain".into(),
                source: crate::library::model::DocumentSource::Inline,
                format: crate::library::model::ContentFormat::PlainText,
                tags: vec![],
                chunk_count: 0,
                triple_count: 0,
                ingested_at: 0,
            },
            "some body text",
        );
        let notes = vec![note];
        let episode = EpisodeState::new("a", "hi", &engine).with_notes(&notes, &notes);
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::default(), &HeuristicTokenCounter)
            .expect("assemble");
        assert!(!out.section_tokens.contains_key(&ContextSection::ActiveNotes));
        assert!(!out.section_tokens.contains_key(&ContextSection::RecentNotes));
    }

    #[test]
    fn budget_keeps_most_recent_turns() {
        let (engine, _tmp) = test_engine();
        let mut conv = ConversationState::new("acp", "default");
        // 20 turns of ~4 tokens each; a tiny budget keeps only the newest few.
        for i in 0..20 {
            conv.record_operator_turn(&format!("turn number {i} here"));
        }
        let episode = EpisodeState::new("a", "turn number 19 here", &engine)
            .with_conversation(Some(&conv));
        // available = window - reserved; use a tiny window to force trimming.
        let out = CompactingManager
            .assemble(&episode, &TokenBudget::new(40, 16), &HeuristicTokenCounter)
            .expect("assemble");
        // Fewer than all 20 turns, and the LAST rendered is the most recent.
        assert!(out.messages.len() < 20, "history trimmed to budget");
        assert_eq!(out.messages.last().unwrap()["content"], "turn number 19 here");
    }
}
