//! Memory sync: ingest daily OpenClaw journals into Tecton episodic memory.
//!
//! Each call fetches all journals created since the last successful sync,
//! skips any already recorded in the KG (idempotency), converts the
//! remainder to episodic memory entries, and persists them with provenance.
//!
//! Idempotency key: a well-known KG entity `openclaw:synced:{journal_id}` is
//! created for each ingested journal.  A subsequent run checks for this symbol
//! with `engine.resolve_symbol(...)` and skips journals that already exist.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolKind;

use super::agent::AgentPredicates;
use super::error::AgentResult;
use super::memory::{WorkingMemoryEntry, WorkingMemoryKind};
use super::openclaw::{OpenClawClient, OpenClawConfig, OpenClawJournal};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the OpenClaw → episodic-memory sync pipeline.
#[derive(Debug, Clone)]
pub struct MemorySyncConfig {
    /// Maximum number of journals to ingest in a single sync run.
    pub max_per_run: usize,
    /// Relevance score to assign to externally-imported episodes (0.0–1.0).
    ///
    /// OpenClaw journals are pre-curated daily summaries so we default to a
    /// high value, ensuring they survive the consolidation threshold.
    pub import_relevance: f32,
}

impl Default for MemorySyncConfig {
    fn default() -> Self {
        Self {
            max_per_run: 30,
            import_relevance: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Outcome of one sync run.
#[derive(Debug, Clone, Default)]
pub struct MemorySyncResult {
    /// Number of journals returned by the API.
    pub journals_fetched: usize,
    /// Journals successfully ingested into episodic memory.
    pub ingested: usize,
    /// Journals skipped because they were already recorded.
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Core sync logic
// ---------------------------------------------------------------------------

/// KG label used to record that a journal has been imported (idempotency).
fn synced_label(journal_id: &str) -> String {
    format!("openclaw:synced:{journal_id}")
}

/// Prefix for the KG entity storing the last sync timestamp.
/// The full label is `openclaw:last_sync_at:{unix_secs}`.
const LAST_SYNC_PREFIX: &str = "openclaw:last_sync_at:";

/// Load the Unix timestamp stored as the last successful sync, or `0`.
fn load_last_sync(engine: &Engine) -> u64 {
    engine
        .item_memory()
        .all_symbols()
        .into_iter()
        .filter_map(|sym| {
            let label = engine.resolve_label(sym);
            if label.starts_with(LAST_SYNC_PREFIX) {
                label[LAST_SYNC_PREFIX.len()..].parse::<u64>().ok()
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
}

/// Persist the current Unix timestamp as the last-sync marker in the KG.
fn save_last_sync(engine: &Engine, now: u64) {
    let label = format!("{LAST_SYNC_PREFIX}{now}");
    let _ = engine.resolve_or_create_entity(&label);
}

/// Check whether a journal has already been imported.
fn already_imported(engine: &Engine, journal_id: &str) -> bool {
    engine.resolve_symbol(&synced_label(journal_id)).is_ok()
}

/// Mark a journal as imported (idempotency record).
fn mark_imported(engine: &Engine, journal_id: &str) {
    let _ = engine.resolve_or_create_entity(&synced_label(journal_id));
}

/// Convert one OpenClaw journal entry into a synthetic `WorkingMemoryEntry`
/// ready for episodic persistence.
fn journal_to_wm_entry(journal: &OpenClawJournal, entry_text: &str, cycle: u64) -> WorkingMemoryEntry {
    let content = format!(
        "[OpenClaw {}] {}: {}",
        journal.date, journal.title, entry_text
    );
    WorkingMemoryEntry {
        id: 0, // assigned by WorkingMemory::push / ignored in direct persist
        content,
        symbols: Vec::new(),
        kind: WorkingMemoryKind::Observation,
        timestamp: journal.created_at,
        relevance: 0.8,
        source_cycle: cycle,
        reference_count: 0,
        access_timestamps: Vec::new(),
    }
}

/// Persist a synthetic WM entry directly as an episodic memory in the KG.
///
/// This mirrors the private `persist_episode` path in `memory.rs` but is
/// called here without routing through the working memory buffer (the journals
/// are pre-curated and high-relevance by definition).
fn persist_journal_episode(
    engine: &Engine,
    journal: &OpenClawJournal,
    entry_text: &str,
    relevance: f32,
    predicates: &AgentPredicates,
    cycle: u64,
) -> AgentResult<()> {
    let wm_entry = journal_to_wm_entry(journal, entry_text, cycle);

    // Create the episode entity.
    let episode_label = format!(
        "episode:openclaw:{}:{}",
        journal.id,
        entry_text.chars().take(40).collect::<String>().replace(' ', "_")
    );
    let episode = engine.create_symbol(SymbolKind::Entity, &episode_label)?;

    // Summary triple.
    let summary_label = format!(
        "summary:{}",
        wm_entry.content.chars().take(80).collect::<String>()
    );
    let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
    let mut triples = vec![Triple::new(
        episode.id,
        predicates.has_summary,
        summary_sym,
    )];

    // Tag with "openclaw" source and the journal's own tags.
    let source_tag = engine.resolve_or_create_entity("tag:openclaw")?;
    triples.push(Triple::new(episode.id, predicates.has_tag, source_tag));

    for tag_str in &journal.tags {
        let tag_label = format!("tag:{tag_str}");
        if let Ok(tag_sym) = engine.resolve_or_create_entity(&tag_label) {
            triples.push(Triple::new(episode.id, predicates.has_tag, tag_sym));
        }
    }

    // Mark as episodic memory.
    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
    triples.push(Triple::new(
        episode.id,
        predicates.memory_type,
        mem_type_sym,
    ));

    // Store provenance.
    let reason = format!(
        "Imported from OpenClaw journal '{}' ({}): {}",
        journal.title,
        journal.date,
        entry_text.chars().take(80).collect::<String>(),
    );
    let mut prov = ProvenanceRecord::new(
        episode.id,
        DerivationKind::AgentConsolidation {
            reason,
            relevance_score: relevance,
        },
    )
    .with_confidence(relevance);
    engine.add_triples_with_provenance(&mut triples, &mut prov)?;

    Ok(())
}

/// Ingest all new OpenClaw journals into Tecton episodic memory.
///
/// Returns `Ok(MemorySyncResult)` even when the API is unreachable (the error
/// is logged but the daemon keeps running). Returns `Err` only for KG write
/// failures that indicate persistent storage issues.
pub fn sync_openclaw_journals(
    engine: &Engine,
    openclaw_cfg: &OpenClawConfig,
    sync_cfg: &MemorySyncConfig,
    predicates: &AgentPredicates,
    cycle: u64,
) -> AgentResult<MemorySyncResult> {
    let mut result = MemorySyncResult::default();

    // Determine fetch window: everything since the last successful sync.
    let since = load_last_sync(engine);

    // Fetch journals from the API.
    let client = OpenClawClient::new(openclaw_cfg.clone());
    let journals = match client.fetch_since(since) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!(error = %e, "openclaw: API fetch failed, skipping sync");
            return Ok(result);
        }
    };

    result.journals_fetched = journals.len();

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut ingested = 0usize;

    'outer: for journal in &journals {
        // Idempotency: skip if the journal's anchor symbol already exists.
        if already_imported(engine, &journal.id) {
            result.skipped += 1;
            continue;
        }

        // Persist each entry in the journal as a separate episode.
        for entry_text in &journal.entries {
            if ingested >= sync_cfg.max_per_run {
                break 'outer;
            }
            persist_journal_episode(
                engine,
                journal,
                entry_text,
                sync_cfg.import_relevance,
                predicates,
                cycle,
            )
            .map_err(|e| {
                tracing::warn!(
                    journal_id = %journal.id,
                    error = %e,
                    "openclaw: failed to persist episode"
                );
                e
            })?;
            ingested += 1;
        }

        // Only mark as imported once *all* entries for this journal are done.
        mark_imported(engine, &journal.id);
        result.ingested += 1;
    }

    // Advance the sync cursor on a clean run.
    if result.ingested > 0 || result.skipped == result.journals_fetched {
        save_last_sync(engine, now_secs);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Status query
// ---------------------------------------------------------------------------

/// Return the Unix timestamp of the last successful sync, or `None`.
pub fn last_sync_timestamp(engine: &Engine) -> Option<u64> {
    let ts = load_last_sync(engine);
    if ts == 0 { None } else { Some(ts) }
}

/// Prefix for the KG entity marking an already-imported journal.
const SYNCED_PREFIX: &str = "openclaw:synced:";

/// Return the number of journals already imported (idempotency records).
pub fn imported_journal_count(engine: &Engine) -> usize {
    engine
        .item_memory()
        .all_symbols()
        .into_iter()
        .filter(|sym| engine.resolve_label(*sym).starts_with(SYNCED_PREFIX))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synced_label_format() {
        assert_eq!(synced_label("2026-05-19-abc"), "openclaw:synced:2026-05-19-abc");
    }

    #[test]
    fn journal_to_wm_entry_content() {
        let journal = OpenClawJournal {
            id: "j1".into(),
            date: "2026-05-19".into(),
            title: "Test Journal".into(),
            entries: vec!["Hello world".into()],
            tags: vec![],
            created_at: 1_000_000,
        };
        let entry = journal_to_wm_entry(&journal, "Hello world", 42);
        assert!(entry.content.contains("[OpenClaw 2026-05-19]"));
        assert!(entry.content.contains("Hello world"));
        assert_eq!(entry.kind, WorkingMemoryKind::Observation);
        assert_eq!(entry.source_cycle, 42);
    }
}
