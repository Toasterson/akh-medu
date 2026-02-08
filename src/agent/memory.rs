//! Deliberate memory system: working memory (ephemeral) and episodic memory (persistent).
//!
//! The agent explicitly decides what knowledge is worth persisting to long-term
//! episodic memory, with provenance tracking WHY it was consolidated.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolId, SymbolKind};

use super::agent::AgentPredicates;
use super::error::{AgentError, AgentResult};
use super::goal::Goal;

// ---------------------------------------------------------------------------
// Working Memory
// ---------------------------------------------------------------------------

/// Classification of a working memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkingMemoryKind {
    /// Produced during the Observe phase.
    Observation,
    /// Produced during the Decide phase.
    Decision,
    /// A goal status change.
    GoalUpdate,
    /// Result of inference or reasoning.
    Inference,
    /// Output from a tool execution.
    ToolResult,
}

/// A single entry in working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryEntry {
    /// Auto-incremented ID within the session.
    pub id: u64,
    /// Human-readable description of what was observed/decided/inferred.
    pub content: String,
    /// Related symbols for KG linking.
    pub symbols: Vec<SymbolId>,
    /// What phase produced this entry.
    pub kind: WorkingMemoryKind,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Dynamic relevance score in [0.0, 1.0].
    pub relevance: f32,
    /// Which OODA cycle produced this entry.
    pub source_cycle: u64,
    /// How many times this entry was referenced during Decide phases.
    pub reference_count: u32,
}

/// Ephemeral per-session scratch memory.
pub struct WorkingMemory {
    entries: Vec<WorkingMemoryEntry>,
    capacity: usize,
    next_id: AtomicU64,
}

impl WorkingMemory {
    /// Create a new working memory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            next_id: AtomicU64::new(1),
        }
    }

    /// Push a new entry. Returns the assigned ID, or error if full.
    pub fn push(&mut self, mut entry: WorkingMemoryEntry) -> AgentResult<u64> {
        if self.entries.len() >= self.capacity {
            return Err(AgentError::WorkingMemoryFull {
                capacity: self.capacity,
            });
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        entry.id = id;
        entry.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.entries.push(entry);
        Ok(id)
    }

    /// Get an entry by ID.
    pub fn get(&self, id: u64) -> Option<&WorkingMemoryEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Get a mutable entry by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut WorkingMemoryEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// Return the N most recent entries (by ID, highest first).
    pub fn recent(&self, n: usize) -> Vec<&WorkingMemoryEntry> {
        let mut sorted: Vec<&WorkingMemoryEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.id.cmp(&a.id));
        sorted.truncate(n);
        sorted
    }

    /// Filter entries by kind.
    pub fn by_kind(&self, kind: WorkingMemoryKind) -> Vec<&WorkingMemoryEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    /// Filter entries that reference a given symbol.
    pub fn by_symbol(&self, symbol: SymbolId) -> Vec<&WorkingMemoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.symbols.contains(&symbol))
            .collect()
    }

    /// Increment the reference count for an entry (called when Decide consults it).
    pub fn increment_reference(&mut self, id: u64) {
        if let Some(entry) = self.get_mut(id) {
            entry.reference_count += 1;
        }
    }

    /// Update the relevance score for an entry.
    pub fn update_relevance(&mut self, id: u64, score: f32) {
        if let Some(entry) = self.get_mut(id) {
            entry.relevance = score.clamp(0.0, 1.0);
        }
    }

    /// Evict all entries with relevance below the threshold.
    /// Returns the number of evicted entries.
    pub fn evict_below(&mut self, threshold: f32) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.relevance >= threshold);
        before - self.entries.len()
    }

    /// Remove specific entries by their IDs.
    pub fn remove_ids(&mut self, ids: &[u64]) {
        self.entries.retain(|e| !ids.contains(&e.id));
    }

    /// Number of entries currently in working memory.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether working memory is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether working memory has reached capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Memory pressure: ratio of current size to capacity, in [0.0, 1.0].
    pub fn pressure(&self) -> f32 {
        if self.capacity == 0 {
            return 1.0;
        }
        self.entries.len() as f32 / self.capacity as f32
    }

    /// Current capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// All entries (for iteration during consolidation).
    pub fn entries(&self) -> &[WorkingMemoryEntry] {
        &self.entries
    }

    /// Serialize working memory state for persistence.
    ///
    /// Returns `(next_id, entries_bytes)` — the next ID counter and bincode-encoded entries.
    pub fn serialize(&self) -> AgentResult<(u64, Vec<u8>)> {
        let next_id = self.next_id.load(Ordering::Relaxed);
        let bytes = bincode::serialize(&self.entries).map_err(|e| AgentError::ConsolidationFailed {
            message: format!("failed to serialize working memory: {e}"),
        })?;
        Ok((next_id, bytes))
    }

    /// Restore working memory from serialized state.
    pub fn restore(capacity: usize, next_id: u64, bytes: &[u8]) -> AgentResult<Self> {
        let entries: Vec<WorkingMemoryEntry> =
            bincode::deserialize(bytes).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to deserialize working memory: {e}"),
            })?;
        Ok(Self {
            entries,
            capacity,
            next_id: AtomicU64::new(next_id),
        })
    }
}

impl std::fmt::Debug for WorkingMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkingMemory")
            .field("entries", &self.entries.len())
            .field("capacity", &self.capacity)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Episodic Memory (persistent, in-KG)
// ---------------------------------------------------------------------------

/// A consolidated episode stored in the knowledge graph.
#[derive(Debug, Clone)]
pub struct EpisodicEntry {
    /// The episode's Entity symbol in the KG.
    pub symbol_id: SymbolId,
    /// Human-readable summary of what happened.
    pub summary: String,
    /// Symbols of triples or concepts learned.
    pub learnings: Vec<SymbolId>,
    /// Categorization tags.
    pub tags: Vec<SymbolId>,
    /// WHY the agent chose to remember this.
    pub consolidation_reason: String,
    /// Relevance score at consolidation time.
    pub relevance_score: f32,
    /// When this was consolidated (seconds since UNIX epoch).
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Consolidation
// ---------------------------------------------------------------------------

/// Configuration for the consolidation process.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum relevance threshold to persist (default: 0.5).
    pub min_relevance: f32,
    /// Weight for goal-relevance scoring (default: 0.4).
    pub goal_relevance_weight: f32,
    /// Weight for novelty scoring (default: 0.3).
    pub novelty_weight: f32,
    /// Weight for utility scoring (default: 0.3).
    pub utility_weight: f32,
    /// Trigger auto-consolidation when WM reaches this count (default: 50).
    pub auto_consolidate_at: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            min_relevance: 0.5,
            goal_relevance_weight: 0.4,
            novelty_weight: 0.3,
            utility_weight: 0.3,
            auto_consolidate_at: 50,
        }
    }
}

/// Result of a consolidation step.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Total entries scored.
    pub entries_scored: usize,
    /// Entries persisted to episodic memory.
    pub entries_persisted: usize,
    /// Entries evicted from working memory.
    pub entries_evicted: usize,
    /// SymbolIds of newly created episodes.
    pub episodes_created: Vec<SymbolId>,
}

/// Score a working memory entry for consolidation.
fn score_entry(
    entry: &WorkingMemoryEntry,
    goals: &[Goal],
    engine: &Engine,
    config: &ConsolidationConfig,
) -> f32 {
    // Goal relevance: does this entry's symbols overlap with any active goal's symbols?
    let goal_relevance = {
        let mut max_rel = 0.0f32;
        for goal in goals {
            // Check if any of the entry's symbols appear in triples reachable from the goal symbol.
            for sym in &entry.symbols {
                let triples = engine.triples_from(goal.symbol_id);
                if triples.iter().any(|t| t.object == *sym || t.subject == *sym) {
                    max_rel = max_rel.max(0.8);
                }
                // Also check if the goal itself is in the entry's symbols.
                if *sym == goal.symbol_id {
                    max_rel = 1.0;
                }
            }
        }
        max_rel
    };

    // Novelty: how much new information is in this entry?
    // Simple heuristic: entries with more unique symbols that don't have many existing triples are more novel.
    let novelty = {
        if entry.symbols.is_empty() {
            0.5 // neutral if no symbols
        } else {
            let total_existing: usize = entry
                .symbols
                .iter()
                .map(|s| engine.triples_from(*s).len() + engine.triples_to(*s).len())
                .sum();
            // Fewer existing triples = more novel
            let avg_triples = total_existing as f32 / entry.symbols.len() as f32;
            (1.0 / (1.0 + avg_triples * 0.2)).clamp(0.0, 1.0)
        }
    };

    // Utility: how many times was this entry referenced during Decide phases?
    let utility = (entry.reference_count as f32 * 0.25).clamp(0.0, 1.0);

    config.goal_relevance_weight * goal_relevance
        + config.novelty_weight * novelty
        + config.utility_weight * utility
}

/// Run the deliberate consolidation process.
///
/// Scores each working memory entry, persists qualifying ones as episodic
/// memories in the knowledge graph, records provenance, and evicts processed entries.
pub fn consolidate(
    working_memory: &mut WorkingMemory,
    engine: &Engine,
    goals: &[Goal],
    config: &ConsolidationConfig,
    predicates: &AgentPredicates,
) -> AgentResult<ConsolidationResult> {
    let entries_scored = working_memory.len();
    let mut entries_to_persist = Vec::new();
    let mut entries_to_evict = Vec::new();

    // Score every entry.
    for entry in working_memory.entries() {
        let score = score_entry(entry, goals, engine, config);
        if score >= config.min_relevance {
            entries_to_persist.push((entry.clone(), score));
        } else {
            entries_to_evict.push(entry.id);
        }
    }

    let mut episodes_created = Vec::new();

    // Persist qualifying entries as episodic memories.
    for (entry, score) in &entries_to_persist {
        match persist_episode(engine, entry, *score, predicates) {
            Ok(episode_id) => {
                episodes_created.push(episode_id);
                entries_to_evict.push(entry.id);
            }
            Err(e) => {
                tracing::warn!(
                    entry_id = entry.id,
                    error = %e,
                    "failed to persist episode, keeping in working memory"
                );
            }
        }
    }

    let entries_persisted = episodes_created.len();

    // Evict processed entries.
    working_memory.remove_ids(&entries_to_evict);
    let entries_evicted = entries_to_evict.len();

    Ok(ConsolidationResult {
        entries_scored,
        entries_persisted,
        entries_evicted,
        episodes_created,
    })
}

/// Persist a single working memory entry as an episodic memory in the KG.
fn persist_episode(
    engine: &Engine,
    entry: &WorkingMemoryEntry,
    relevance_score: f32,
    predicates: &AgentPredicates,
) -> AgentResult<SymbolId> {
    // Create the episode entity.
    let episode_label = format!("episode:{}", entry.id);
    let episode = engine
        .create_symbol(SymbolKind::Entity, &episode_label)?;

    // Link the episode to its summary via has_summary predicate.
    // We store the summary as the episode's own label (already done above),
    // but also create a dedicated summary entity.
    let summary_label = if entry.content.len() > 80 {
        format!("summary:{}", &entry.content[..80])
    } else {
        format!("summary:{}", entry.content)
    };
    let summary_sym = engine
        .resolve_or_create_entity(&summary_label)?;
    let _ = engine.add_triple(&Triple::new(episode.id, predicates.has_summary, summary_sym));

    // Link to learned symbols.
    for sym in &entry.symbols {
        let _ = engine.add_triple(&Triple::new(episode.id, predicates.learned, *sym));
    }

    // Create a tag for the entry kind.
    let kind_tag_label = match entry.kind {
        WorkingMemoryKind::Observation => "tag:observation",
        WorkingMemoryKind::Decision => "tag:decision",
        WorkingMemoryKind::GoalUpdate => "tag:goal_update",
        WorkingMemoryKind::Inference => "tag:inference",
        WorkingMemoryKind::ToolResult => "tag:tool_result",
    };
    let tag_sym = engine
        .resolve_or_create_entity(kind_tag_label)?;
    let _ = engine.add_triple(&Triple::new(episode.id, predicates.has_tag, tag_sym));

    // Mark as episodic memory type.
    let mem_type_sym = engine
        .resolve_or_create_entity("episodic_memory")?;
    let _ = engine.add_triple(&Triple::new(episode.id, predicates.memory_type, mem_type_sym));

    // Store provenance.
    let reason = format!(
        "Consolidated from WM entry {} ({}): {}",
        entry.id,
        kind_tag_label.trim_start_matches("tag:"),
        if entry.content.len() > 60 {
            format!("{}...", &entry.content[..60])
        } else {
            entry.content.clone()
        }
    );

    let mut prov = ProvenanceRecord::new(
        episode.id,
        DerivationKind::AgentConsolidation {
            reason: reason.clone(),
            relevance_score,
        },
    )
    .with_sources(entry.symbols.clone())
    .with_confidence(relevance_score);

    let _ = engine.store_provenance(&mut prov);

    Ok(episode.id)
}

/// Recall episodic memories from the KG by querying related symbols.
pub fn recall_episodes(
    engine: &Engine,
    query_symbols: &[SymbolId],
    predicates: &AgentPredicates,
    top_k: usize,
) -> AgentResult<Vec<EpisodicEntry>> {
    let mut episodes: Vec<EpisodicEntry> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // For each query symbol, find episodes that `learned` it.
    for sym in query_symbols {
        let incoming = engine.triples_to(*sym);
        for triple in &incoming {
            if triple.predicate == predicates.learned && !seen.contains(&triple.subject) {
                seen.insert(triple.subject);
                if let Some(ep) = reconstruct_episode(engine, triple.subject, predicates) {
                    episodes.push(ep);
                }
            }
        }

        // Also check episodes tagged with the symbol.
        for triple in &incoming {
            if triple.predicate == predicates.has_tag && !seen.contains(&triple.subject) {
                seen.insert(triple.subject);
                if let Some(ep) = reconstruct_episode(engine, triple.subject, predicates) {
                    episodes.push(ep);
                }
            }
        }
    }

    // Sort by timestamp (most recent first), then relevance.
    episodes.sort_by(|a, b| {
        b.timestamp
            .cmp(&a.timestamp)
            .then(b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal))
    });
    episodes.truncate(top_k);

    Ok(episodes)
}

/// Reconstruct an EpisodicEntry from its KG triples.
fn reconstruct_episode(
    engine: &Engine,
    episode_id: SymbolId,
    predicates: &AgentPredicates,
) -> Option<EpisodicEntry> {
    let triples = engine.triples_from(episode_id);
    if triples.is_empty() {
        return None;
    }

    // Check if this is actually an episode (has memory_type link).
    let is_episode = triples.iter().any(|t| t.predicate == predicates.memory_type);
    if !is_episode {
        return None;
    }

    let mut summary = String::new();
    let mut learnings = Vec::new();
    let mut tags = Vec::new();

    for triple in &triples {
        if triple.predicate == predicates.has_summary {
            summary = engine.resolve_label(triple.object);
        } else if triple.predicate == predicates.learned {
            learnings.push(triple.object);
        } else if triple.predicate == predicates.has_tag {
            tags.push(triple.object);
        }
    }

    // Get the episode's creation timestamp from its symbol metadata.
    let timestamp = engine
        .get_symbol_meta(episode_id)
        .map(|m| m.created_at)
        .unwrap_or(0);

    Some(EpisodicEntry {
        symbol_id: episode_id,
        summary,
        learnings,
        tags,
        consolidation_reason: String::new(), // would need provenance lookup for full detail
        relevance_score: 0.0, // would need provenance lookup
        timestamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(kind: WorkingMemoryKind, content: &str, cycle: u64) -> WorkingMemoryEntry {
        WorkingMemoryEntry {
            id: 0, // will be assigned
            content: content.into(),
            symbols: Vec::new(),
            kind,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: cycle,
            reference_count: 0,
        }
    }

    #[test]
    fn push_and_retrieve() {
        let mut wm = WorkingMemory::new(10);
        let id = wm
            .push(make_entry(WorkingMemoryKind::Observation, "saw something", 1))
            .unwrap();
        assert_eq!(wm.len(), 1);
        let entry = wm.get(id).unwrap();
        assert_eq!(entry.content, "saw something");
        assert_eq!(entry.kind, WorkingMemoryKind::Observation);
    }

    #[test]
    fn capacity_enforced() {
        let mut wm = WorkingMemory::new(2);
        wm.push(make_entry(WorkingMemoryKind::Observation, "a", 1))
            .unwrap();
        wm.push(make_entry(WorkingMemoryKind::Decision, "b", 1))
            .unwrap();
        let result = wm.push(make_entry(WorkingMemoryKind::Inference, "c", 1));
        assert!(result.is_err());
        assert!(wm.is_full());
    }

    #[test]
    fn filter_by_kind() {
        let mut wm = WorkingMemory::new(10);
        wm.push(make_entry(WorkingMemoryKind::Observation, "obs1", 1))
            .unwrap();
        wm.push(make_entry(WorkingMemoryKind::Decision, "dec1", 1))
            .unwrap();
        wm.push(make_entry(WorkingMemoryKind::Observation, "obs2", 2))
            .unwrap();

        let obs = wm.by_kind(WorkingMemoryKind::Observation);
        assert_eq!(obs.len(), 2);
        let decs = wm.by_kind(WorkingMemoryKind::Decision);
        assert_eq!(decs.len(), 1);
    }

    #[test]
    fn evict_below_threshold() {
        let mut wm = WorkingMemory::new(10);
        let id1 = wm
            .push(make_entry(WorkingMemoryKind::Observation, "high", 1))
            .unwrap();
        let _id2 = wm
            .push(make_entry(WorkingMemoryKind::Observation, "low", 1))
            .unwrap();

        wm.update_relevance(id1, 0.8);
        // id2 stays at default 0.5

        let evicted = wm.evict_below(0.6);
        assert_eq!(evicted, 1);
        assert_eq!(wm.len(), 1);
        assert!(wm.get(id1).is_some());
    }

    #[test]
    fn recent_returns_newest_first() {
        let mut wm = WorkingMemory::new(10);
        wm.push(make_entry(WorkingMemoryKind::Observation, "first", 1))
            .unwrap();
        wm.push(make_entry(WorkingMemoryKind::Observation, "second", 2))
            .unwrap();
        wm.push(make_entry(WorkingMemoryKind::Observation, "third", 3))
            .unwrap();

        let recent = wm.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "third");
        assert_eq!(recent[1].content, "second");
    }

    #[test]
    fn pressure_calculation() {
        let mut wm = WorkingMemory::new(4);
        assert_eq!(wm.pressure(), 0.0);
        wm.push(make_entry(WorkingMemoryKind::Observation, "a", 1))
            .unwrap();
        assert_eq!(wm.pressure(), 0.25);
        wm.push(make_entry(WorkingMemoryKind::Observation, "b", 1))
            .unwrap();
        assert_eq!(wm.pressure(), 0.5);
    }
}
