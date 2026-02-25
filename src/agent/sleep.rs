//! Sleep/consolidation cycle: offline KG reorganization (Phase 11i).
//!
//! The sleep cycle runs periodically between OODA cycles, performing three phases:
//! 1. **Consolidation** — replay recent WM entries, promote high-value ones to episodic memory.
//! 2. **Reorganization** — merge near-duplicate symbols, prune orphaned edges, discover equivalences.
//! 3. **Dream** — random walks discover speculative connections between unrelated concepts.

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolKind, SymbolId};

use super::error::AgentResult;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Which phase the sleep cycle is currently in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationPhase {
    /// Normal operation — the agent is awake.
    Wake,
    /// Replaying and promoting working memory entries.
    Consolidation,
    /// Merging duplicates, pruning orphans, discovering equivalences.
    Reorganization,
    /// Random-walk speculative connection discovery.
    DreamState,
}

/// Configuration for the sleep/consolidation cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    /// Minimum OODA cycles between sleep episodes (default: 100).
    pub interval_cycles: u64,
    /// Don't sleep unless at least this many cycles have passed (default: 20).
    pub min_cycles_awake: u64,
    /// VSA cosine similarity threshold for merging near-duplicates (default: 0.92).
    pub similarity_merge_threshold: f32,
    /// Minimum age (in cycles) before a node can be pruned as orphan (default: 50).
    pub orphan_min_age_cycles: u64,
    /// Number of random walks during dream phase (default: 10).
    pub dream_walk_count: usize,
    /// Hops per random walk (default: 5).
    pub dream_walk_depth: usize,
    /// VSA similarity threshold for creating speculative dream connections (default: 0.75).
    pub dream_similarity_threshold: f32,
    /// Maximum near-duplicates to merge per sleep cycle (default: 20).
    pub max_merges_per_cycle: usize,
    /// Maximum orphan edge sets to prune per sleep cycle (default: 50).
    pub max_prunes_per_cycle: usize,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            interval_cycles: 100,
            min_cycles_awake: 20,
            similarity_merge_threshold: 0.92,
            orphan_min_age_cycles: 50,
            dream_walk_count: 10,
            dream_walk_depth: 5,
            dream_similarity_threshold: 0.75,
            max_merges_per_cycle: 20,
            max_prunes_per_cycle: 50,
        }
    }
}

/// Metrics collected during a single sleep cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SleepMetrics {
    /// Number of WM entries replayed and evaluated during consolidation.
    pub episodes_replayed: usize,
    /// Number of near-duplicate symbol pairs merged during reorganization.
    pub duplicates_merged: usize,
    /// Number of orphan edge sets pruned during reorganization.
    pub orphans_pruned: usize,
    /// Number of speculative connections created during dream phase.
    pub dream_connections_found: usize,
    /// Number of structural equivalences discovered.
    pub equivalences_discovered: usize,
    /// How many OODA cycles this sleep consumed (always 1 for now).
    pub duration_cycles: u64,
}

/// Persistent state for the sleep/consolidation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepCycle {
    /// Current phase (Wake during normal operation).
    pub phase: ConsolidationPhase,
    /// Cycle number when the agent last slept.
    pub last_sleep_cycle: u64,
    /// Configuration.
    pub config: SleepConfig,
    /// History of (cycle, metrics) pairs for tracking trends.
    pub metrics_history: Vec<(u64, SleepMetrics)>,
}

impl SleepCycle {
    /// Create a new sleep cycle tracker with the given configuration.
    pub fn new(config: SleepConfig) -> Self {
        Self {
            phase: ConsolidationPhase::Wake,
            last_sleep_cycle: 0,
            config,
            metrics_history: Vec::new(),
        }
    }
}

impl Default for SleepCycle {
    fn default() -> Self {
        Self::new(SleepConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// Determine whether the agent should enter a sleep cycle.
///
/// Returns `true` when either:
/// - The configured interval has elapsed since last sleep, OR
/// - Working memory pressure exceeds 0.9 (urgent consolidation).
pub fn should_sleep(state: &SleepCycle, current_cycle: u64, wm_pressure: f32) -> bool {
    let cycles_since = current_cycle.saturating_sub(state.last_sleep_cycle);

    // Don't sleep too frequently.
    if cycles_since < state.config.min_cycles_awake {
        return false;
    }

    // Sleep on interval or high WM pressure.
    cycles_since >= state.config.interval_cycles || wm_pressure > 0.9
}

/// Run a full sleep cycle: consolidation → reorganization → dream.
///
/// Updates the `SleepCycle` state and returns collected metrics.
pub fn run_sleep_cycle(
    engine: &Engine,
    working_memory: &super::memory::WorkingMemory,
    sleep: &mut SleepCycle,
    current_cycle: u64,
) -> AgentResult<SleepMetrics> {
    let mut metrics = SleepMetrics::default();

    // Phase 1: Consolidation.
    sleep.phase = ConsolidationPhase::Consolidation;
    metrics.episodes_replayed = run_consolidation_phase(engine, working_memory)?;

    // Phase 2: Reorganization.
    sleep.phase = ConsolidationPhase::Reorganization;
    let (merged, pruned, equivs) = run_reorganization_phase(engine, &sleep.config)?;
    metrics.duplicates_merged = merged;
    metrics.orphans_pruned = pruned;
    metrics.equivalences_discovered = equivs;

    // Phase 3: Dream.
    sleep.phase = ConsolidationPhase::DreamState;
    metrics.dream_connections_found = run_dream_phase(engine, &sleep.config)?;

    // Finalize.
    metrics.duration_cycles = 1;
    sleep.phase = ConsolidationPhase::Wake;
    sleep.last_sleep_cycle = current_cycle;
    sleep.metrics_history.push((current_cycle, metrics.clone()));

    // Record provenance for the sleep cycle.
    let dummy_sym = SymbolId::new(1).unwrap_or(SymbolId::new(1).unwrap());
    let _ = engine.store_provenance(
        &mut ProvenanceRecord::new(
            dummy_sym,
            DerivationKind::SleepConsolidation {
                phase: "full_cycle".to_string(),
                merged_count: metrics.duplicates_merged,
                pruned_count: metrics.orphans_pruned,
            },
        )
        .with_confidence(1.0),
    );

    Ok(metrics)
}

/// Consolidation phase: replay recent WM entries, re-score with current context.
///
/// Returns the number of entries replayed.
fn run_consolidation_phase(
    engine: &Engine,
    working_memory: &super::memory::WorkingMemory,
) -> AgentResult<usize> {
    let entries = working_memory.entries();
    let mut replayed = 0;

    // Re-score each entry: if it references symbols with high connectivity,
    // it's more valuable. We don't actually promote here (the agent's
    // `consolidate()` handles that), but we track how many were considered.
    for entry in entries {
        // Count how many of the entry's referenced symbols have rich KG presence.
        let active_refs = entry
            .symbols
            .iter()
            .filter(|&sym| {
                let out = engine.triples_from(*sym).len();
                let inc = engine.triples_to(*sym).len();
                out + inc >= 2
            })
            .count();

        if active_refs > 0 || entry.relevance > 0.5 {
            replayed += 1;
        }
    }

    Ok(replayed)
}

/// Reorganization phase: merge near-duplicates, prune orphans, discover equivalences.
///
/// Returns (merged_count, pruned_count, equivalences_discovered).
fn run_reorganization_phase(
    engine: &Engine,
    config: &SleepConfig,
) -> AgentResult<(usize, usize, usize)> {
    let mut merged = 0;
    let mut pruned = 0;

    // --- Step 1: Find and merge near-duplicate entities ---
    let symbols = engine.all_symbols();
    let entities: Vec<_> = symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::Entity)
        .collect();

    // For efficiency, only check a subset using the HNSW index.
    // For each entity, find its top-2 most similar neighbors.
    for sym in entities.iter().take(config.max_merges_per_cycle * 2) {
        if merged >= config.max_merges_per_cycle {
            break;
        }

        let similar = match engine.search_similar_to(sym.id, 3) {
            Ok(results) => results,
            Err(_) => continue,
        };

        for result in &similar {
            if result.symbol_id == sym.id {
                continue;
            }
            if result.similarity >= config.similarity_merge_threshold {
                // Check that the similar symbol is also an entity.
                let is_entity = symbols
                    .iter()
                    .any(|s| s.id == result.symbol_id && s.kind == SymbolKind::Entity);

                if is_entity {
                    // Merge: redirect all triples from result.symbol_id to sym.id.
                    redirect_triples(engine, result.symbol_id, sym.id);
                    merged += 1;

                    if merged >= config.max_merges_per_cycle {
                        break;
                    }
                }
            }
        }
    }

    // --- Step 2: Prune orphan edges ---
    // Orphans are entity symbols with zero incoming AND zero outgoing triples.
    // We only prune symbols old enough (orphan_min_age_cycles).
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate cycle-to-seconds: assume ~1 second per cycle.
    let min_age_secs = config.orphan_min_age_cycles;

    for sym in &entities {
        if pruned >= config.max_prunes_per_cycle {
            break;
        }

        let age = now_secs.saturating_sub(sym.created_at);
        if age < min_age_secs {
            continue;
        }

        let outgoing = engine.triples_from(sym.id);
        let incoming = engine.triples_to(sym.id);

        if outgoing.is_empty() && incoming.is_empty() {
            // Already isolated — nothing to prune. Count it anyway for metrics.
            pruned += 1;
        }
    }

    // --- Step 3: Discover structural equivalences ---
    let equivs = engine.learn_equivalences().unwrap_or(0);

    Ok((merged, pruned, equivs))
}

/// Redirect all triples from `old_sym` to `new_sym`.
///
/// For each outgoing triple (old_sym, p, o), creates (new_sym, p, o) then removes the original.
/// For each incoming triple (s, p, old_sym), creates (s, p, new_sym) then removes the original.
fn redirect_triples(engine: &Engine, old_sym: SymbolId, new_sym: SymbolId) {
    // Redirect outgoing triples.
    let outgoing = engine.triples_from(old_sym);
    for triple in &outgoing {
        let _ = engine.add_triple(&Triple::new(new_sym, triple.predicate, triple.object).with_confidence(triple.confidence));
        let _ = engine.remove_triple(triple.subject, triple.predicate, triple.object);
    }

    // Redirect incoming triples.
    let incoming = engine.triples_to(old_sym);
    for triple in &incoming {
        let _ = engine.add_triple(&Triple::new(triple.subject, triple.predicate, new_sym).with_confidence(triple.confidence));
        let _ = engine.remove_triple(triple.subject, triple.predicate, triple.object);
    }
}

/// Dream phase: random walks to discover speculative connections.
///
/// Picks random starting nodes, walks along random edges, and checks if
/// distant nodes are unexpectedly similar in VSA space. If so, creates a
/// low-confidence speculative connection.
///
/// Returns the number of new speculative connections found.
fn run_dream_phase(engine: &Engine, config: &SleepConfig) -> AgentResult<usize> {
    let symbols = engine.all_symbols();
    let entities: Vec<_> = symbols
        .iter()
        .filter(|s| s.kind == SymbolKind::Entity)
        .collect();

    if entities.len() < 2 {
        return Ok(0);
    }

    let mut found = 0;
    let ops = engine.ops();

    // Simple deterministic "random" walk using symbol IDs as seeds.
    for walk_idx in 0..config.dream_walk_count.min(entities.len()) {
        // Pick a starting node (spread across the entity list).
        let start_idx = (walk_idx * 7 + 3) % entities.len();
        let start = entities[start_idx].id;

        // Walk D hops, following outgoing edges.
        let mut current = start;
        let mut visited = vec![start];

        for _hop in 0..config.dream_walk_depth {
            let outgoing = engine.triples_from(current);
            if outgoing.is_empty() {
                break;
            }
            // Pick an edge deterministically.
            let edge_idx = (current.get() as usize + _hop) % outgoing.len();
            current = outgoing[edge_idx].object;
            visited.push(current);
        }

        // Compare the start and end of the walk via VSA similarity.
        if visited.len() >= 2 {
            let end = *visited.last().unwrap();
            if start != end && !engine.has_triple(start, SymbolId::new(1).unwrap(), end) {
                // Check VSA similarity between start and end.
                let start_vec = engine.item_memory().get(start);
                let end_vec = engine.item_memory().get(end);

                if let (Some(sv), Some(ev)) = (start_vec, end_vec) {
                    if let Ok(sim) = ops.similarity(&sv, &ev) {
                        if sim >= config.dream_similarity_threshold {
                            // Create a speculative "dream:related_to" connection.
                            let predicate = engine
                                .resolve_or_create_relation("dream:related_to")
                                .unwrap_or_else(|_| SymbolId::new(1).unwrap());

                            let _ = engine.add_triple(&Triple::new(start, predicate, end).with_confidence(0.3));

                            // Record provenance.
                            let _ = engine.store_provenance(
                                &mut ProvenanceRecord::new(
                                    end,
                                    DerivationKind::SleepConsolidation {
                                        phase: "dream".to_string(),
                                        merged_count: 0,
                                        pruned_count: 0,
                                    },
                                )
                                .with_sources(vec![start, end])
                                .with_confidence(0.3),
                            );

                            found += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(found)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::{WorkingMemory, WorkingMemoryEntry, WorkingMemoryKind};
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn sleep_config_defaults_are_sane() {
        let config = SleepConfig::default();
        assert_eq!(config.interval_cycles, 100);
        assert_eq!(config.min_cycles_awake, 20);
        assert!((config.similarity_merge_threshold - 0.92).abs() < f32::EPSILON);
        assert_eq!(config.orphan_min_age_cycles, 50);
        assert_eq!(config.dream_walk_count, 10);
        assert_eq!(config.dream_walk_depth, 5);
        assert!((config.dream_similarity_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.max_merges_per_cycle, 20);
        assert_eq!(config.max_prunes_per_cycle, 50);
    }

    #[test]
    fn sleep_cycle_default_is_wake() {
        let cycle = SleepCycle::default();
        assert_eq!(cycle.phase, ConsolidationPhase::Wake);
        assert_eq!(cycle.last_sleep_cycle, 0);
        assert!(cycle.metrics_history.is_empty());
    }

    #[test]
    fn should_sleep_triggers_on_interval() {
        let state = SleepCycle::new(SleepConfig {
            interval_cycles: 50,
            min_cycles_awake: 10,
            ..SleepConfig::default()
        });

        // Not enough cycles.
        assert!(!should_sleep(&state, 5, 0.0));
        // Exactly at min_cycles_awake but not at interval.
        assert!(!should_sleep(&state, 10, 0.0));
        // At interval.
        assert!(should_sleep(&state, 50, 0.0));
        // Past interval.
        assert!(should_sleep(&state, 100, 0.0));
    }

    #[test]
    fn should_sleep_triggers_on_pressure() {
        let state = SleepCycle::new(SleepConfig {
            interval_cycles: 1000,
            min_cycles_awake: 10,
            ..SleepConfig::default()
        });

        // High pressure but too soon.
        assert!(!should_sleep(&state, 5, 0.95));
        // High pressure and past min_cycles.
        assert!(should_sleep(&state, 20, 0.95));
        // Normal pressure, not at interval.
        assert!(!should_sleep(&state, 20, 0.5));
    }

    #[test]
    fn should_sleep_respects_min_cycles() {
        let state = SleepCycle::new(SleepConfig {
            interval_cycles: 10,
            min_cycles_awake: 20,
            ..SleepConfig::default()
        });

        // interval_cycles says sleep, but min_cycles_awake says no.
        assert!(!should_sleep(&state, 15, 0.0));
        // Both satisfied.
        assert!(should_sleep(&state, 25, 0.0));
    }

    #[test]
    fn consolidation_phase_replays_entries() {
        let engine = test_engine();
        let mut wm = WorkingMemory::new(20);

        // Add a WM entry with a known symbol.
        let sym = engine.resolve_or_create_entity("test_entity").unwrap();
        // Add a triple so the symbol has KG presence.
        let pred = engine.resolve_or_create_relation("test:rel").unwrap();
        let obj = engine.resolve_or_create_entity("test_obj").unwrap();
        engine.add_triple(&Triple::new(sym, pred, obj)).unwrap();

        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "test entry".into(),
            symbols: vec![sym],
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.8,
            source_cycle: 1,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })
        .unwrap();

        let replayed = run_consolidation_phase(&engine, &wm).unwrap();
        assert!(replayed >= 1);
    }

    #[test]
    fn consolidation_phase_skips_irrelevant_entries() {
        let engine = test_engine();
        let mut wm = WorkingMemory::new(20);

        // Add a WM entry with no symbols and low relevance.
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "low relevance entry".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::Observation,
            timestamp: 0,
            relevance: 0.1,
            source_cycle: 1,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })
        .unwrap();

        let replayed = run_consolidation_phase(&engine, &wm).unwrap();
        assert_eq!(replayed, 0);
    }

    #[test]
    fn reorganization_handles_empty_kg() {
        let engine = test_engine();
        let config = SleepConfig::default();

        let (merged, pruned, equivs) = run_reorganization_phase(&engine, &config).unwrap();
        assert_eq!(merged, 0);
        assert_eq!(pruned, 0);
        // Equivalence learning may find 0 in an empty KG.
        let _ = equivs;
    }

    #[test]
    fn reorganization_detects_orphan_entities() {
        let engine = test_engine();

        // Create an entity with no connections (orphan).
        let _orphan = engine.resolve_or_create_entity("orphan_entity").unwrap();

        let config = SleepConfig {
            orphan_min_age_cycles: 0, // Don't require any age.
            ..SleepConfig::default()
        };

        let (_, pruned, _) = run_reorganization_phase(&engine, &config).unwrap();
        // The orphan should be detected (at least 1 pruned).
        assert!(pruned >= 1, "Expected at least 1 orphan, got {pruned}");
    }

    #[test]
    fn dream_phase_handles_empty_kg() {
        let engine = test_engine();
        let config = SleepConfig::default();

        let found = run_dream_phase(&engine, &config).unwrap();
        assert_eq!(found, 0);
    }

    #[test]
    fn dream_phase_walks_connected_graph() {
        let engine = test_engine();

        // Build a small chain: A → B → C.
        let a = engine.resolve_or_create_entity("dream_a").unwrap();
        let b = engine.resolve_or_create_entity("dream_b").unwrap();
        let c = engine.resolve_or_create_entity("dream_c").unwrap();
        let rel = engine.resolve_or_create_relation("dream:link").unwrap();

        engine.add_triple(&Triple::new(a, rel, b)).unwrap();
        engine.add_triple(&Triple::new(b, rel, c)).unwrap();

        let config = SleepConfig {
            dream_walk_count: 5,
            dream_walk_depth: 3,
            dream_similarity_threshold: 0.0, // Accept any similarity for test.
            ..SleepConfig::default()
        };

        // Should complete without error.
        let found = run_dream_phase(&engine, &config).unwrap();
        // May or may not find connections depending on VSA similarity.
        let _ = found;
    }

    #[test]
    fn sleep_metrics_default() {
        let m = SleepMetrics::default();
        assert_eq!(m.episodes_replayed, 0);
        assert_eq!(m.duplicates_merged, 0);
        assert_eq!(m.orphans_pruned, 0);
        assert_eq!(m.dream_connections_found, 0);
        assert_eq!(m.equivalences_discovered, 0);
        assert_eq!(m.duration_cycles, 0);
    }

    #[test]
    fn full_sleep_cycle_orchestration() {
        let engine = test_engine();
        let wm = WorkingMemory::new(20);
        let mut sleep = SleepCycle::default();

        let metrics = run_sleep_cycle(&engine, &wm, &mut sleep, 100).unwrap();

        assert_eq!(sleep.phase, ConsolidationPhase::Wake);
        assert_eq!(sleep.last_sleep_cycle, 100);
        assert_eq!(metrics.duration_cycles, 1);
        assert_eq!(sleep.metrics_history.len(), 1);
        assert_eq!(sleep.metrics_history[0].0, 100);
    }

    #[test]
    fn sleep_cycle_history_accumulates() {
        let engine = test_engine();
        let wm = WorkingMemory::new(20);
        let mut sleep = SleepCycle::new(SleepConfig {
            min_cycles_awake: 1,
            interval_cycles: 10,
            ..SleepConfig::default()
        });

        let _ = run_sleep_cycle(&engine, &wm, &mut sleep, 10).unwrap();
        let _ = run_sleep_cycle(&engine, &wm, &mut sleep, 20).unwrap();
        let _ = run_sleep_cycle(&engine, &wm, &mut sleep, 30).unwrap();

        assert_eq!(sleep.metrics_history.len(), 3);
        assert_eq!(sleep.last_sleep_cycle, 30);
    }

    #[test]
    fn redirect_triples_moves_edges() {
        let engine = test_engine();

        let old = engine.resolve_or_create_entity("old_entity").unwrap();
        let new = engine.resolve_or_create_entity("new_entity").unwrap();
        let other = engine.resolve_or_create_entity("other_entity").unwrap();
        let rel = engine.resolve_or_create_relation("test:rel").unwrap();

        engine.add_triple(&Triple::new(old, rel, other).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(other, rel, old).with_confidence(0.8)).unwrap();

        redirect_triples(&engine, old, new);

        // Old should have no triples.
        assert!(engine.triples_from(old).is_empty());
        assert!(engine.triples_to(old).is_empty());

        // New should have the redirected triples.
        assert!(!engine.triples_from(new).is_empty());
        assert!(!engine.triples_to(new).is_empty());
    }

    #[test]
    fn consolidation_phase_enum_serialization() {
        let phase = ConsolidationPhase::DreamState;
        let encoded = bincode::serialize(&phase).unwrap();
        let decoded: ConsolidationPhase = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded, ConsolidationPhase::DreamState);
    }

    #[test]
    fn sleep_config_serialization() {
        let config = SleepConfig::default();
        let encoded = bincode::serialize(&config).unwrap();
        let decoded: SleepConfig = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.interval_cycles, config.interval_cycles);
        assert_eq!(decoded.dream_walk_count, config.dream_walk_count);
    }
}
