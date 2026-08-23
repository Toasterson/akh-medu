//! Deliberate memory system: working memory (ephemeral) and episodic memory (persistent).
//!
//! The agent explicitly decides what knowledge is worth persisting to long-term
//! episodic memory, with provenance tracking WHY it was consolidated.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::affect::{AffectiveState, emotional_salience, mood_congruence_bias};
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
    /// Cycle numbers at which this entry was referenced (for ACT-R activation).
    #[serde(default)]
    pub access_timestamps: Vec<u64>,
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

    /// Push a high-priority entry, evicting lowest-relevance entries if needed.
    ///
    /// This is used for critical entries (e.g., session summaries) that should
    /// take precedence over restored working memory entries. If working memory
    /// is at capacity, this evicts the lowest-relevance entry to make room.
    pub fn push_priority(&mut self, mut entry: WorkingMemoryEntry) -> AgentResult<u64> {
        // Make room if needed by evicting the lowest-relevance entry.
        if self.entries.len() >= self.capacity {
            if let Some((idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.relevance.partial_cmp(&b.relevance).unwrap_or(std::cmp::Ordering::Equal))
            {
                let evicted = self.entries.remove(idx);
                tracing::debug!(
                    evicted_id = evicted.id,
                    evicted_relevance = evicted.relevance,
                    new_relevance = entry.relevance,
                    "evicted low-relevance entry to make room for priority entry"
                );
            }
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
    ///
    /// Also records the cycle number for ACT-R activation computation.
    pub fn increment_reference(&mut self, id: u64, current_cycle: u64) {
        if let Some(entry) = self.get_mut(id) {
            entry.reference_count += 1;
            entry.access_timestamps.push(current_cycle);
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
        let bytes =
            bincode::serialize(&self.entries).map_err(|e| AgentError::ConsolidationFailed {
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
    /// Affective valence (mood) at consolidation time — used for mood-congruent retrieval (Phase 23).
    pub affective_valence: f32,
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
    /// Multiplier for emotional salience boost on consolidation scores (Phase 23, default: 0.2).
    ///
    /// When non-zero, emotionally significant moments (high |valence| × arousal) boost
    /// the consolidation score, making those memories more likely to be persisted.
    pub affective_salience_weight: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            min_relevance: 0.5,
            goal_relevance_weight: 0.4,
            novelty_weight: 0.3,
            utility_weight: 0.3,
            auto_consolidate_at: 50,
            affective_salience_weight: 0.2,
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
    /// Goals auto-generated from VSA competence gap detection.
    pub vsa_gap_goals_created: usize,
}

/// ACT-R base-level activation for a working memory entry.
///
/// Implements `B_i = ln(Σ t_j^{-d})` where `t_j` is the time (in cycles) since
/// each access and `d = 0.5` (decay parameter). The result is normalized to
/// `[0.0, 1.0]` via a sigmoid.
///
/// Falls back to creation-time-based activation if there are no access timestamps.
pub fn actr_activation(entry: &WorkingMemoryEntry, current_cycle: u64) -> f32 {
    const DECAY: f32 = 0.5;

    if entry.access_timestamps.is_empty() {
        // Fallback: use reference_count and age for a rough activation.
        let age = current_cycle.saturating_sub(entry.source_cycle).max(1) as f32;
        let freq_component = (entry.reference_count as f32 + 1.0).ln();
        let recency_component = -(age.ln() * DECAY);
        let raw = freq_component + recency_component;
        // Sigmoid normalization to [0, 1].
        return 1.0 / (1.0 + (-raw).exp());
    }

    let sum: f32 = entry
        .access_timestamps
        .iter()
        .map(|&t| {
            let elapsed = current_cycle.saturating_sub(t).max(1) as f32;
            elapsed.powf(-DECAY)
        })
        .sum();

    let raw = if sum > 0.0 { sum.ln() } else { -1.0 };

    // Sigmoid normalization to [0, 1].
    1.0 / (1.0 + (-raw).exp())
}

/// Score a working memory entry for consolidation.
///
/// When `affect` is provided, emotionally significant moments receive a salience
/// boost — high |valence| × arousal increases the consolidation priority (Phase 23).
fn score_entry(
    entry: &WorkingMemoryEntry,
    goals: &[Goal],
    engine: &Engine,
    config: &ConsolidationConfig,
    current_cycle: u64,
    affect: Option<&AffectiveState>,
) -> f32 {
    // Goal relevance: does this entry's symbols overlap with any active goal's symbols?
    let goal_relevance = {
        let mut max_rel = 0.0f32;
        for goal in goals {
            // Check if any of the entry's symbols appear in triples reachable from the goal symbol.
            for sym in &entry.symbols {
                let triples = engine.triples_from(goal.symbol_id);
                if triples
                    .iter()
                    .any(|t| t.object == *sym || t.subject == *sym)
                {
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

    // Utility: ACT-R activation-based (replaces raw reference_count * 0.25).
    let utility = actr_activation(entry, current_cycle);

    let base_score = config.goal_relevance_weight * goal_relevance
        + config.novelty_weight * novelty
        + config.utility_weight * utility;

    // Affective salience boost (Phase 23): emotionally intense moments are more memorable.
    if config.affective_salience_weight > 0.0
        && let Some(state) = affect
    {
        let salience = emotional_salience(state); // 1.0–2.0
        base_score * (1.0 + (salience - 1.0) * config.affective_salience_weight)
    } else {
        base_score
    }
}

/// Run the deliberate consolidation process.
///
/// Scores each working memory entry, persists qualifying ones as episodic
/// memories in the knowledge graph, records provenance, and evicts processed entries.
///
/// When `affect` is `Some`, emotionally significant episodes receive a salience
/// boost and the current mood valence is stored for future mood-congruent recall (Phase 23).
pub fn consolidate(
    working_memory: &mut WorkingMemory,
    engine: &Engine,
    goals: &[Goal],
    config: &ConsolidationConfig,
    predicates: &AgentPredicates,
    current_cycle: u64,
    affect: Option<&AffectiveState>,
) -> AgentResult<ConsolidationResult> {
    let entries_scored = working_memory.len();
    let mut entries_to_persist = Vec::new();
    let mut entries_to_evict = Vec::new();

    // Score every entry.
    for entry in working_memory.entries() {
        // Observation entries (OODA cycle-state snapshots) and Decision entries
        // (which tool was chosen and why) are transient operational logs — they
        // carry no semantic knowledge worth persisting to the KG.  Evict them
        // immediately without ever scoring or consolidating.
        if matches!(
            entry.kind,
            WorkingMemoryKind::Observation | WorkingMemoryKind::Decision
        ) {
            entries_to_evict.push(entry.id);
            continue;
        }

        let score = score_entry(entry, goals, engine, config, current_cycle, affect);
        if score >= config.min_relevance {
            entries_to_persist.push((entry.clone(), score));
        } else {
            entries_to_evict.push(entry.id);
        }
    }

    let mut episodes_created = Vec::new();
    let affective_valence = affect.map(|s| s.valence).unwrap_or(0.0);

    // Persist qualifying entries as episodic memories.
    for (entry, score) in &entries_to_persist {
        match persist_episode(engine, entry, *score, affective_valence, predicates) {
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
        vsa_gap_goals_created: 0,
    })
}

/// Persist a single working memory entry as an episodic memory in the KG.
fn persist_episode(
    engine: &Engine,
    entry: &WorkingMemoryEntry,
    relevance_score: f32,
    affective_valence: f32,
    predicates: &AgentPredicates,
) -> AgentResult<SymbolId> {
    // Embed a truncated content excerpt in the episode entity's label so the
    // summary is recoverable without creating a separate KG entity.  Storing
    // the summary as a distinct `summary:*` entity was the original source of
    // workspace KG pollution (WI-019e1061).
    let excerpt = if entry.content.len() > 80 {
        &entry.content[..80]
    } else {
        entry.content.as_str()
    };
    let episode_label = format!("episode:{}:{}", entry.id, excerpt);
    let episode_id = engine
        .create_symbol(crate::symbol::SymbolKind::Entity, &episode_label)?
        .id;

    // Create a tag for the entry kind.
    let kind_tag_label = match entry.kind {
        WorkingMemoryKind::Observation => "tag:observation",
        WorkingMemoryKind::Decision => "tag:decision",
        WorkingMemoryKind::GoalUpdate => "tag:goal_update",
        WorkingMemoryKind::Inference => "tag:inference",
        WorkingMemoryKind::ToolResult => "tag:tool_result",
    };
    let tag_sym = engine.resolve_or_create_entity(kind_tag_label)?;

    // Mark as episodic memory type.
    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;

    // Store affective valence for mood-congruent retrieval (Phase 23).
    let valence_label = format!("affect_valence:{:.3}", affective_valence);
    let valence_sym = engine.resolve_or_create_entity(&valence_label)?;

    // Store provenance. Use char_indices to find a safe UTF-8 truncation boundary;
    // byte-slicing at a fixed offset panics on multi-byte sequences.
    let reason = {
        let end = entry
            .content
            .char_indices()
            .nth(60)
            .map(|(i, _)| i)
            .unwrap_or(entry.content.len());
        let truncated = &entry.content[..end];
        let suffix = if end < entry.content.len() { "..." } else { "" };
        format!(
            "Consolidated from WM entry {} ({}): {truncated}{suffix}",
            entry.id,
            kind_tag_label.trim_start_matches("tag:"),
        )
    };

    let mut prov = ProvenanceRecord::new(
        episode_id,
        DerivationKind::AgentConsolidation {
            reason: reason.clone(),
            relevance_score,
        },
    )
    .with_sources(entry.symbols.clone())
    .with_confidence(relevance_score);

    let mut triples = entry
        .symbols
        .iter()
        .map(|symbol| Triple::new(episode_id, predicates.learned, *symbol))
        .collect::<Vec<_>>();
    triples.extend([
        Triple::new(episode_id, predicates.has_tag, tag_sym),
        Triple::new(episode_id, predicates.memory_type, mem_type_sym),
        Triple::new(
            episode_id,
            predicates.has_affective_valence,
            valence_sym,
        ),
    ]);
    engine.add_triples_with_provenance(&mut triples, &mut prov)?;

    Ok(episode_id)
}

/// Recall episodic memories from the KG by querying related symbols.
///
/// When `affect` is `Some`, mood-congruent retrieval is applied: memories with
/// valence matching the current mood are ranked higher (Phase 23).
pub fn recall_episodes(
    engine: &Engine,
    query_symbols: &[SymbolId],
    predicates: &AgentPredicates,
    top_k: usize,
    affect: Option<&AffectiveState>,
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

    // Sort with mood-congruent bias (Phase 23) or plain recency + relevance.
    if let Some(mood) = affect {
        episodes.sort_by(|a, b| {
            let bias_a = mood_congruence_bias(mood, a.affective_valence);
            let bias_b = mood_congruence_bias(mood, b.affective_valence);
            let score_a = a.relevance_score * bias_a;
            let score_b = b.relevance_score * bias_b;
            // Primary: biased score; secondary: recency.
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.timestamp.cmp(&a.timestamp))
        });
    } else {
        episodes.sort_by(|a, b| {
            b.timestamp.cmp(&a.timestamp).then(
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
    }
    episodes.truncate(top_k);

    Ok(episodes)
}

/// Recall the most recent episodic memories from the KG without requiring query symbols.
///
/// Scans all episodes by following reverse links from the well-known `episodic_memory`
/// marker entity and returns the `top_k` most recent, sorted by symbol creation timestamp.
/// Returns an empty vec (not an error) when no episodes exist yet.
pub fn recall_recent_episodes(
    engine: &Engine,
    predicates: &AgentPredicates,
    top_k: usize,
) -> AgentResult<Vec<EpisodicEntry>> {
    // The "episodic_memory" label is the type marker written by persist_episode and
    // index_cycle_episode. If it has never been created, no episodes exist yet.
    let mem_type_id = match engine.resolve_symbol("episodic_memory") {
        Ok(id) => id,
        Err(_) => return Ok(Vec::new()),
    };

    // All episodes have a triple: episode -[memory_type]-> episodic_memory.
    let incoming = engine.triples_to(mem_type_id);
    let mut episodes: Vec<EpisodicEntry> = incoming
        .iter()
        .filter(|t| t.predicate == predicates.memory_type)
        .filter_map(|t| reconstruct_episode(engine, t.subject, predicates))
        .collect();

    // Most recent first: primary sort by creation timestamp, secondary by SymbolId
    // (monotonically increasing) so episodes within the same second are deterministic.
    episodes.sort_by(|a, b| {
        b.timestamp
            .cmp(&a.timestamp)
            .then(b.symbol_id.get().cmp(&a.symbol_id.get()))
    });
    episodes.truncate(top_k);

    Ok(episodes)
}

/// Create and index an episodic memory entry directly from an OODA cycle result.
///
/// This bypasses the consolidation threshold so that every OODA cycle immediately
/// leaves a retrievable trace. Episodes created here are identical in structure to
/// consolidated ones and are found by both `recall_episodes` and `recall_recent_episodes`.
pub fn index_cycle_episode(
    engine: &Engine,
    predicates: &AgentPredicates,
    cycle_count: u64,
    goal_label: &str,
    goal_id: Option<SymbolId>,
    chosen_tool: &str,
    progress: &str,
) -> AgentResult<()> {
    // Resolve or create the episode entity. A deterministic label means crash-recovery
    // or session restarts with overlapping cycle numbers reuse the existing symbol and
    // add triples idempotently rather than failing with DuplicateLabel.
    let episode_label = format!("cycle_episode:{cycle_count}");
    let episode_id = engine.resolve_or_create_entity(&episode_label)?;

    // Summary — truncated to 80 chars so it fits comfortably in recall output.
    let summary_text = format!(
        "Cycle {cycle_count}: tool={chosen_tool} goal=\"{goal_label}\" progress={progress}"
    );
    let summary_label = if summary_text.len() > 80 {
        format!("summary:{}", &summary_text[..80])
    } else {
        format!("summary:{summary_text}")
    };
    let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
    let mut triples = vec![Triple::new(
        episode_id,
        predicates.has_summary,
        summary_sym,
    )];

    // Tag: marks this as a direct cycle episode (as opposed to a consolidated one).
    let cycle_tag = engine.resolve_or_create_entity("tag:cycle_episode")?;
    triples.push(Triple::new(episode_id, predicates.has_tag, cycle_tag));

    // Tag: progress outcome for targeted queries.
    let progress_tag_label = format!("tag:progress_{progress}");
    let progress_tag = engine.resolve_or_create_entity(&progress_tag_label)?;
    triples.push(Triple::new(
        episode_id,
        predicates.has_tag,
        progress_tag,
    ));

    // Link to the goal so symbol-based recall can find this episode.
    if let Some(gid) = goal_id {
        triples.push(Triple::new(episode_id, predicates.learned, gid));
    }

    // Mark as episodic memory — this is how recall_recent_episodes discovers all episodes.
    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
    triples.push(Triple::new(
        episode_id,
        predicates.memory_type,
        mem_type_sym,
    ));

    // from_cycle: link to a cycle-number entity for temporal ordering queries.
    let cycle_entity_label = format!("cycle:{cycle_count}");
    let cycle_sym = engine.resolve_or_create_entity(&cycle_entity_label)?;
    triples.push(Triple::new(
        episode_id,
        predicates.from_cycle,
        cycle_sym,
    ));

    // Provenance — record at full confidence since this is a direct agent action.
    let reason = format!(
        "Direct episode from OODA cycle {cycle_count}: tool={chosen_tool} progress={progress}"
    );
    let mut prov = ProvenanceRecord::new(
        episode_id,
        DerivationKind::AgentConsolidation {
            reason,
            relevance_score: 1.0,
        },
    )
    .with_confidence(1.0);
    if let Some(gid) = goal_id {
        prov = prov.with_sources(vec![gid]);
    }
    engine.add_triples_with_provenance(&mut triples, &mut prov)?;

    Ok(())
}

/// Cap on the paged-in turn body surfaced as an episode's summary/content
/// (ADR 091 decision 5). The context manager budgets the recalled section
/// further; this cap just bounds a single entry's contribution.
const TURN_BODY_SUMMARY_CAP: usize = 1500;

/// Truncate `s` to at most `max_chars` **characters** (not bytes), returning
/// the original string unchanged if it is already shorter. Uses
/// `char_indices` to find a safe UTF-8 boundary — a fixed byte-offset slice
/// panics on multi-byte sequences.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => s[..byte_idx].to_string(),
        None => s.to_string(),
    }
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
    let is_episode = triples
        .iter()
        .any(|t| t.predicate == predicates.memory_type);
    if !is_episode {
        return None;
    }

    let mut summary = String::new();
    let mut learnings = Vec::new();
    let mut tags = Vec::new();
    let mut affective_valence = 0.0f32;

    // Cycle episodes (persist_episode) embed the summary in the entity label's
    // 3rd segment ("episode:{id}:{content}"). The typed `memory_write` path
    // instead puts a hash/timestamp *discriminator* there (for idempotent labels)
    // and stores the real summary via `has_summary`. So the label segment is only
    // a FALLBACK — prefer `has_summary` whenever present, else recall surfaces the
    // discriminator hash as the "summary".
    let episode_label = engine.resolve_label(episode_id);
    let label_summary = episode_label
        .splitn(3, ':')
        .nth(2)
        .filter(|c| !c.is_empty())
        .unwrap_or("")
        .to_string();

    for triple in &triples {
        if triple.predicate == predicates.has_summary {
            let raw = engine.resolve_label(triple.object);
            summary = raw.strip_prefix("summary:").unwrap_or(&raw).to_string();
        } else if triple.predicate == predicates.learned {
            learnings.push(triple.object);
        } else if triple.predicate == predicates.has_tag {
            tags.push(triple.object);
        } else if triple.predicate == predicates.has_affective_valence {
            let label = engine.resolve_label(triple.object);
            if let Some(val_str) = label.strip_prefix("affect_valence:") {
                affective_valence = val_str.parse().unwrap_or(0.0);
            }
        }
    }

    // No explicit summary → fall back to the label-embedded excerpt (cycle path).
    if summary.is_empty() {
        summary = label_summary;
    }

    // ADR 091 decision 5: body-paging. Harvested transcript episodes carry the
    // label convention "episode:turn:{seq}" (Wave 2A); when it matches, and an
    // `Engine::turn_body` reader is installed, page the raw turn text out of
    // the WAL and prefer it over the has_summary/label fallback above — the
    // point of paging is to surface the *actual conversation content* at
    // recall time, not the 80-char excerpt. If no reader is installed (e.g.
    // in-memory engines) or the row is gone, `turn_body` returns `None` and
    // the existing fallback stands untouched.
    if let Some(seq_str) = episode_label.strip_prefix("episode:turn:") {
        if let Ok(seq) = seq_str.parse::<i64>() {
            if let Some(body) = engine.turn_body(seq) {
                summary = truncate_chars(&body, TURN_BODY_SUMMARY_CAP);
            }
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
        relevance_score: 0.0,                // would need provenance lookup
        timestamp,
        affective_valence,
    })
}

// ---------------------------------------------------------------------------
// Session Summary
// ---------------------------------------------------------------------------

/// Compressed summary of a session for cross-session continuity.
///
/// Generated at session end and injected as high-relevance WM entries on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Descriptions of goals completed during this session.
    pub accomplished: Vec<String>,
    /// Descriptions of goals still pending at session end.
    pub pending: Vec<String>,
    /// Brief KG change summary (e.g., "added 42 triples").
    pub kg_changes: String,
    /// Suggested next actions.
    pub next_steps: Vec<String>,
    /// Name of the active project at session end, if any.
    pub active_project: Option<String>,
    /// (start_cycle, end_cycle) range for this session.
    pub cycle_range: (u64, u64),
}

/// Generate a session summary from current state.
pub fn generate_session_summary(
    wm: &WorkingMemory,
    goals: &[Goal],
    active_project: Option<&str>,
    cycle_start: u64,
    cycle_end: u64,
) -> SessionSummary {
    let accomplished: Vec<String> = goals
        .iter()
        .filter(|g| matches!(g.status, super::goal::GoalStatus::Completed))
        .map(|g| g.description.clone())
        .collect();

    let pending: Vec<String> = goals
        .iter()
        .filter(|g| {
            matches!(
                g.status,
                super::goal::GoalStatus::Active | super::goal::GoalStatus::Pending
            )
        })
        .map(|g| g.description.clone())
        .collect();

    // Derive next steps from recent Decision entries.
    let next_steps: Vec<String> = wm
        .by_kind(WorkingMemoryKind::Decision)
        .iter()
        .rev()
        .take(3)
        .map(|e| {
            // Extract tool and goal from decision content.
            if let Some(goal_part) = e.content.split("goal=\"").nth(1) {
                let goal_name = goal_part.split('"').next().unwrap_or("?");
                format!("Continue work on: {goal_name}")
            } else {
                format!("Review: {}", &e.content[..e.content.len().min(60)])
            }
        })
        .collect();

    let kg_changes = format!("{} WM entries at session end", wm.len());

    SessionSummary {
        accomplished,
        pending,
        kg_changes,
        next_steps,
        active_project: active_project.map(String::from),
        cycle_range: (cycle_start, cycle_end),
    }
}

/// Restore session context by injecting summary as high-relevance WM entries.
///
/// Returns the IDs of the injected entries.
pub fn restore_session_summary(
    wm: &mut WorkingMemory,
    summary: &SessionSummary,
    current_cycle: u64,
) -> AgentResult<Vec<u64>> {
    let mut ids = Vec::new();

    // Session overview entry.
    let overview = format!(
        "Previous session (cycles {}–{}): completed {} goals",
        summary.cycle_range.0,
        summary.cycle_range.1,
        summary.accomplished.len()
    );
    let id = wm.push_priority(WorkingMemoryEntry {
        id: 0,
        content: overview,
        symbols: Vec::new(),
        kind: WorkingMemoryKind::Observation,
        timestamp: 0,
        relevance: 0.9,
        source_cycle: current_cycle,
        reference_count: 0,
        access_timestamps: Vec::new(),
    })?;
    ids.push(id);

    // Pending goals entry (if any).
    if !summary.pending.is_empty() {
        let pending_desc = summary.pending.join("; ");
        let content = format!("Pending from last session: {pending_desc}");
        let id = wm.push_priority(WorkingMemoryEntry {
            id: 0,
            content,
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.85,
            source_cycle: current_cycle,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })?;
        ids.push(id);
    }

    // Next steps entry (if any).
    if !summary.next_steps.is_empty() {
        let steps_desc = summary.next_steps.join("; ");
        let content = format!("Next steps: {steps_desc}");
        let id = wm.push_priority(WorkingMemoryEntry {
            id: 0,
            content,
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.85,
            source_cycle: current_cycle,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })?;
        ids.push(id);
    }

    // Active project entry (if any).
    if let Some(ref project_name) = summary.active_project {
        let content = format!("Active project: {project_name}");
        let id = wm.push_priority(WorkingMemoryEntry {
            id: 0,
            content,
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: current_cycle,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })?;
        ids.push(id);
    }

    Ok(ids)
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
            access_timestamps: Vec::new(),
        }
    }

    #[test]
    fn push_and_retrieve() {
        let mut wm = WorkingMemory::new(10);
        let id = wm
            .push(make_entry(
                WorkingMemoryKind::Observation,
                "saw something",
                1,
            ))
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

    #[test]
    fn push_priority_evicts_low_relevance() {
        let mut wm = WorkingMemory::new(3);

        // Fill to capacity with entries of varying relevance.
        let id1 = wm.push(make_entry(WorkingMemoryKind::Observation, "low", 1)).unwrap();
        wm.update_relevance(id1, 0.3);

        let id2 = wm.push(make_entry(WorkingMemoryKind::Observation, "medium", 1)).unwrap();
        wm.update_relevance(id2, 0.6);

        let id3 = wm.push(make_entry(WorkingMemoryKind::Observation, "high", 1)).unwrap();
        wm.update_relevance(id3, 0.9);

        assert_eq!(wm.len(), 3);
        assert!(wm.is_full());

        // Push a high-priority entry. Should evict the lowest-relevance entry (id1).
        let mut priority_entry = make_entry(WorkingMemoryKind::Observation, "priority", 2);
        priority_entry.relevance = 0.95;
        let id4 = wm.push_priority(priority_entry).unwrap();

        // Still at capacity, but the low-relevance entry was evicted.
        assert_eq!(wm.len(), 3);
        assert!(wm.get(id1).is_none(), "low-relevance entry should be evicted");
        assert!(wm.get(id2).is_some(), "medium-relevance entry should remain");
        assert!(wm.get(id3).is_some(), "high-relevance entry should remain");
        assert!(wm.get(id4).is_some(), "priority entry should be added");
    }

    #[test]
    fn push_priority_succeeds_when_not_full() {
        let mut wm = WorkingMemory::new(5);
        wm.push(make_entry(WorkingMemoryKind::Observation, "existing", 1)).unwrap();

        let mut priority_entry = make_entry(WorkingMemoryKind::Observation, "priority", 2);
        priority_entry.relevance = 0.95;
        let result = wm.push_priority(priority_entry);

        assert!(result.is_ok());
        assert_eq!(wm.len(), 2);
    }

    // -- ACT-R activation tests --

    #[test]
    fn actr_single_access() {
        let entry = WorkingMemoryEntry {
            id: 1,
            content: "test".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 1,
            access_timestamps: vec![5],
        };

        // At cycle 10, single access at cycle 5 → elapsed = 5.
        let activation = actr_activation(&entry, 10);
        assert!(activation > 0.0 && activation < 1.0);
    }

    #[test]
    fn actr_recency_bias() {
        // Recent access should have higher activation than old access.
        let recent = WorkingMemoryEntry {
            id: 1,
            content: "recent".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 1,
            access_timestamps: vec![9],
        };
        let old = WorkingMemoryEntry {
            id: 2,
            content: "old".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 1,
            access_timestamps: vec![1],
        };

        let act_recent = actr_activation(&recent, 10);
        let act_old = actr_activation(&old, 10);
        assert!(
            act_recent > act_old,
            "recent {act_recent} should > old {act_old}"
        );
    }

    #[test]
    fn actr_frequency_boost() {
        // More accesses should increase activation.
        let single = WorkingMemoryEntry {
            id: 1,
            content: "single".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 1,
            access_timestamps: vec![5],
        };
        let multi = WorkingMemoryEntry {
            id: 2,
            content: "multi".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 3,
            access_timestamps: vec![3, 5, 7],
        };

        let act_single = actr_activation(&single, 10);
        let act_multi = actr_activation(&multi, 10);
        assert!(
            act_multi > act_single,
            "multi {act_multi} should > single {act_single}"
        );
    }

    #[test]
    fn actr_no_accesses() {
        // No access timestamps → fallback path.
        let entry = WorkingMemoryEntry {
            id: 1,
            content: "never accessed".into(),
            symbols: Vec::new(),
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.5,
            source_cycle: 1,
            reference_count: 0,
            access_timestamps: Vec::new(),
        };
        let activation = actr_activation(&entry, 10);
        assert!(activation > 0.0 && activation < 1.0);
    }

    // -- Session summary tests --

    #[test]
    fn session_summary_completed() {
        use super::super::goal::{Goal, GoalStatus};

        let wm = WorkingMemory::new(10);
        let goals = vec![
            Goal {
                symbol_id: crate::symbol::SymbolId::new(1).unwrap(),
                description: "Learn Rust".into(),
                status: GoalStatus::Completed,
                priority: 100,
                success_criteria: "done".into(),
                parent: None,
                children: vec![],
                created_at: 0,
                cycles_worked: 5,
                last_progress_cycle: 5,
                source: None,
                blocked_by: vec![],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
                estimated_effort: None,
            },
            Goal {
                symbol_id: crate::symbol::SymbolId::new(2).unwrap(),
                description: "Build project".into(),
                status: GoalStatus::Active,
                priority: 80,
                success_criteria: "working".into(),
                parent: None,
                children: vec![],
                created_at: 0,
                cycles_worked: 2,
                last_progress_cycle: 2,
                source: None,
                blocked_by: vec![],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
                estimated_effort: None,
            },
        ];

        let summary = generate_session_summary(&wm, &goals, Some("my-project"), 1, 10);
        assert_eq!(summary.accomplished.len(), 1);
        assert_eq!(summary.accomplished[0], "Learn Rust");
        assert_eq!(summary.pending.len(), 1);
        assert_eq!(summary.pending[0], "Build project");
        assert_eq!(summary.active_project, Some("my-project".into()));
        assert_eq!(summary.cycle_range, (1, 10));
    }

    #[test]
    fn session_summary_pending() {
        let wm = WorkingMemory::new(10);
        let summary = generate_session_summary(&wm, &[], None, 0, 0);
        assert!(summary.accomplished.is_empty());
        assert!(summary.pending.is_empty());
        assert!(summary.active_project.is_none());
    }

    #[test]
    fn restore_injects_entries() {
        let mut wm = WorkingMemory::new(20);
        let summary = SessionSummary {
            accomplished: vec!["done goal".into()],
            pending: vec!["pending goal".into()],
            kg_changes: "5 triples added".into(),
            next_steps: vec!["continue building".into()],
            active_project: Some("proj1".into()),
            cycle_range: (1, 10),
        };

        let ids = restore_session_summary(&mut wm, &summary, 11).unwrap();
        // Should inject 4 entries: overview + pending + next_steps + active_project.
        assert_eq!(ids.len(), 4);
        assert_eq!(wm.len(), 4);

        // First entry is overview.
        let overview = wm.get(ids[0]).unwrap();
        assert!(overview.content.contains("completed 1 goals"));
        assert_eq!(overview.relevance, 0.9);
    }

    #[test]
    fn increment_reference_records_cycle() {
        let mut wm = WorkingMemory::new(10);
        let id = wm
            .push(make_entry(WorkingMemoryKind::Observation, "test", 1))
            .unwrap();

        wm.increment_reference(id, 5);
        wm.increment_reference(id, 8);

        let entry = wm.get(id).unwrap();
        assert_eq!(entry.reference_count, 2);
        assert_eq!(entry.access_timestamps, vec![5, 8]);
    }

    // -- Runtime artifact prevention tests --

    /// consolidate() must evict Observation and Decision entries without ever
    /// persisting them to the KG (they are operational logs, not knowledge).
    #[test]
    fn consolidate_skips_observation_and_decision_entries() {
        use super::super::goal::{Goal, GoalStatus};
        use crate::engine::{Engine, EngineConfig};
        use crate::symbol::SymbolId;

        let engine = Engine::new(EngineConfig::default()).unwrap();
        let predicates = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let mut wm = WorkingMemory::new(20);

        // Observation entry — operational log, must NOT be consolidated.
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "Cycle 1: 1 active goals, 0 WM entries, 0 recalled episodes".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.9,
            source_cycle: 1,
            reference_count: 5,
            access_timestamps: vec![1, 2, 3, 4, 5],
        })
        .unwrap();

        // Decision entry — operational log, must NOT be consolidated.
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "Decide: tool=kg_query, goal=\"test goal\", reason=explore".into(),
            symbols: vec![SymbolId::new(1).unwrap()],
            kind: WorkingMemoryKind::Decision,
            timestamp: 0,
            relevance: 0.9,
            source_cycle: 1,
            reference_count: 5,
            access_timestamps: vec![1, 2, 3, 4, 5],
        })
        .unwrap();

        let goals: Vec<Goal> = vec![];
        let config = ConsolidationConfig {
            min_relevance: 0.0, // accept everything that passes the kind filter
            ..ConsolidationConfig::default()
        };

        let result = consolidate(&mut wm, &engine, &goals, &config, &predicates, 5, None).unwrap();

        // Both entries evicted, neither persisted as episodes.
        assert_eq!(result.entries_persisted, 0);
        assert_eq!(result.episodes_created.len(), 0);
        assert_eq!(wm.len(), 0, "WM should be empty after eviction");

        // No episode:* entities created in the KG.
        let episode_entities: Vec<_> = engine
            .all_symbols()
            .into_iter()
            .filter(|m| m.label.starts_with("episode:"))
            .collect();
        assert!(
            episode_entities.is_empty(),
            "no episode entities should be created for Observation/Decision entries"
        );
    }

    /// consolidate() must NOT create any `summary:*` entities for ToolResult
    /// or GoalUpdate entries — the content lives in the episode label itself.
    #[test]
    fn consolidate_no_summary_entities_for_tool_results() {
        use super::super::goal::{Goal, GoalStatus};
        use crate::engine::{Engine, EngineConfig};
        use crate::symbol::SymbolId;

        let engine = Engine::new(EngineConfig::default()).unwrap();
        let predicates = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let mut wm = WorkingMemory::new(20);

        // ToolResult entry with a real symbol — eligible for consolidation.
        let sym = engine
            .resolve_or_create_entity("some:concept")
            .unwrap();
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "Tool result (kg_query):\nfound concept".into(),
            symbols: vec![sym],
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 0.9,
            source_cycle: 1,
            reference_count: 3,
            access_timestamps: vec![1, 2, 3],
        })
        .unwrap();

        let goals: Vec<Goal> = vec![];
        let config = ConsolidationConfig {
            min_relevance: 0.0,
            ..ConsolidationConfig::default()
        };

        let result = consolidate(&mut wm, &engine, &goals, &config, &predicates, 5, None).unwrap();

        assert_eq!(result.entries_persisted, 1);
        assert_eq!(result.episodes_created.len(), 1);

        // Verify NO `summary:*` entities were created.
        let summary_entities: Vec<_> = engine
            .all_symbols()
            .into_iter()
            .filter(|m| m.label.starts_with("summary:"))
            .collect();
        assert!(
            summary_entities.is_empty(),
            "summary:* entities must not be created (got: {:?})",
            summary_entities.iter().map(|m| &m.label).collect::<Vec<_>>()
        );

        // Episode entity label should embed the content excerpt.
        let episode_id = result.episodes_created[0];
        let episode_label = engine.resolve_label(episode_id);
        assert!(
            episode_label.starts_with("episode:"),
            "episode label should start with 'episode:': {episode_label}"
        );
        // The label includes the content excerpt after the second colon.
        assert!(
            episode_label.contains("Tool result"),
            "episode label should embed content excerpt: {episode_label}"
        );
    }

    // -- KG-backed episodic memory tests --

    fn test_predicates(engine: &crate::engine::Engine) -> super::super::agent::AgentPredicates {
        super::super::agent::AgentPredicates::init(engine).unwrap()
    }

    #[test]
    fn index_and_recall_recent_cycle_episode() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        let goal_sym = engine
            .resolve_or_create_entity("goal:learn_rust")
            .unwrap();

        index_cycle_episode(&engine, &predicates, 1, "learn rust", Some(goal_sym), "nlu_parse", "advanced").unwrap();

        let episodes = recall_recent_episodes(&engine, &predicates, 10).unwrap();
        assert_eq!(episodes.len(), 1);
        assert!(
            episodes[0].summary.contains("learn rust"),
            "summary should contain goal text: got {:?}",
            episodes[0].summary
        );
        assert!(
            !episodes[0].summary.starts_with("summary:"),
            "summary should not have 'summary:' prefix: got {:?}",
            episodes[0].summary
        );
    }

    #[test]
    fn multiple_cycle_episodes_ordered_by_recency() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        index_cycle_episode(&engine, &predicates, 1, "goal A", None, "tool_a", "advanced").unwrap();
        index_cycle_episode(&engine, &predicates, 2, "goal B", None, "tool_b", "completed").unwrap();
        index_cycle_episode(&engine, &predicates, 3, "goal C", None, "tool_c", "no_change").unwrap();

        let episodes = recall_recent_episodes(&engine, &predicates, 2).unwrap();
        assert_eq!(episodes.len(), 2);
        // Most recent (highest symbol id / created latest) should come first.
        // Cycle 3 was created last.
        assert!(
            episodes[0].summary.contains("goal C") || episodes[0].summary.contains('3'),
            "Expected most recent episode first: {:?}",
            episodes[0].summary
        );
    }

    #[test]
    fn recall_episodes_by_symbol_finds_goal_link() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        let goal_sym = engine.resolve_or_create_entity("goal:my_goal").unwrap();
        index_cycle_episode(&engine, &predicates, 5, "my goal", Some(goal_sym), "search", "advanced").unwrap();

        let episodes = recall_episodes(&engine, &[goal_sym], &predicates, 5, None).unwrap();
        assert_eq!(episodes.len(), 1);
        assert!(episodes[0].summary.contains("my goal") || episodes[0].summary.contains('5'));
    }

    #[test]
    fn index_cycle_episode_idempotent_on_same_label() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        // Indexing the same cycle twice should not error (resolve_or_create_entity is idempotent).
        index_cycle_episode(&engine, &predicates, 7, "goal X", None, "tool_x", "advanced").unwrap();
        index_cycle_episode(&engine, &predicates, 7, "goal X", None, "tool_x", "advanced").unwrap();

        // Still exactly one episode (triples are deduplicated).
        let episodes = recall_recent_episodes(&engine, &predicates, 10).unwrap();
        assert_eq!(episodes.len(), 1);
    }

    #[test]
    fn recall_recent_empty_when_no_episodes() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        let episodes = recall_recent_episodes(&engine, &predicates, 5).unwrap();
        assert!(episodes.is_empty());
    }

    // -- ADR 091 decision 5: recall body-paging tests --

    /// Mint an `episode:turn:{seq}` index the way harvest does (Wave 2A
    /// convention): `learned`-linked to `syms`, `has_tag` → `tag:conversation`,
    /// `memory_type` → `episodic_memory`, `has_summary` → an 80-char excerpt.
    /// The body itself is *not* stored in the KG (ADR 091 decision 5) — it
    /// lives only behind the installed `TurnBodyReader`.
    fn mint_turn_episode(
        engine: &crate::engine::Engine,
        predicates: &AgentPredicates,
        seq: i64,
        syms: &[SymbolId],
        excerpt: &str,
    ) -> SymbolId {
        let label = format!("episode:turn:{seq}");
        let episode_id = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, &label)
            .unwrap()
            .id;
        for sym in syms {
            engine
                .add_triple(&Triple::new(episode_id, predicates.learned, *sym))
                .unwrap();
        }
        let tag_sym = engine.resolve_or_create_entity("tag:conversation").unwrap();
        engine
            .add_triple(&Triple::new(episode_id, predicates.has_tag, tag_sym))
            .unwrap();
        let mem_type_sym = engine.resolve_or_create_entity("episodic_memory").unwrap();
        engine
            .add_triple(&Triple::new(episode_id, predicates.memory_type, mem_type_sym))
            .unwrap();
        let summary_label = format!("summary:{excerpt}");
        let summary_sym = engine.resolve_or_create_entity(&summary_label).unwrap();
        engine
            .add_triple(&Triple::new(episode_id, predicates.has_summary, summary_sym))
            .unwrap();
        episode_id
    }

    /// A `TurnBodyReader` stub returning a fixed body for one known `seq`.
    struct StubTurnBodyReader {
        seq: i64,
        body: String,
    }

    impl crate::mutation::TurnBodyReader for StubTurnBodyReader {
        fn turn_body(&self, seq: i64) -> Option<String> {
            if seq == self.seq {
                Some(self.body.clone())
            } else {
                None
            }
        }
    }

    #[test]
    fn recall_pages_body_from_installed_reader() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        engine
            .set_turn_body_reader(std::sync::Arc::new(StubTurnBodyReader {
                seq: 42,
                body: "User: what is rust?\nAkh: a systems language.".to_string(),
            }))
            .map_err(|_| ())
            .expect("first reader install should succeed");

        let sym = engine.resolve_or_create_entity("rust").unwrap();
        mint_turn_episode(&engine, &predicates, 42, &[sym], "what is rust? excerpt");

        let episodes = recall_episodes(&engine, &[sym], &predicates, 5, None).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(
            episodes[0].summary,
            "User: what is rust?\nAkh: a systems language."
        );
    }

    #[test]
    fn recall_falls_back_to_summary_when_no_reader_installed() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        // No `set_turn_body_reader` call — `Engine::turn_body` must return
        // `None` and `reconstruct_episode` must keep the has_summary excerpt.
        let sym = engine.resolve_or_create_entity("golang").unwrap();
        mint_turn_episode(&engine, &predicates, 99, &[sym], "what is golang? excerpt");

        let episodes = recall_episodes(&engine, &[sym], &predicates, 5, None).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].summary, "what is golang? excerpt");
    }

    #[test]
    fn recall_truncates_oversized_body_at_char_boundary() {
        let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
        let predicates = test_predicates(&engine);

        // Build a body well over the cap using multi-byte UTF-8 characters
        // (each "é" is 2 bytes) so a byte-offset slice would panic on a
        // non-char boundary; `truncate_chars` must not.
        let long_body: String = std::iter::repeat_n('é', TURN_BODY_SUMMARY_CAP + 500).collect();
        engine
            .set_turn_body_reader(std::sync::Arc::new(StubTurnBodyReader {
                seq: 7,
                body: long_body,
            }))
            .map_err(|_| ())
            .expect("reader install should succeed");

        let sym = engine.resolve_or_create_entity("unicode_topic").unwrap();
        mint_turn_episode(&engine, &predicates, 7, &[sym], "excerpt");

        let episodes = recall_episodes(&engine, &[sym], &predicates, 5, None).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].summary.chars().count(), TURN_BODY_SUMMARY_CAP);
        assert!(episodes[0].summary.chars().all(|c| c == 'é'));
    }
}
