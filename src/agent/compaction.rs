//! KG compaction: classify and remove episode/agent scaffolding triples.
//!
//! The knowledge graph accumulates significant noise from episode tracking.
//! Triples of the form `episode:N → agent:learned → X` and
//! `X → * → episode:N` account for 50–95% of all triples in practice yet
//! carry no domain knowledge. This module classifies them via label heuristics
//! and removes them in configurable batches during daemon idle time.

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::error::AgentResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the periodic KG compaction task.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Maximum scaffolding triples to remove per run (default: 500).
    ///
    /// Caps per-cycle cost; the daemon fires compaction repeatedly until the
    /// graph is clean enough.
    pub max_triples_per_run: usize,
    /// Skip compaction when scaffolding fraction is below this (default: 0.2).
    ///
    /// Prevents unnecessary churn on an already-clean graph.
    pub scaffolding_fraction_threshold: f32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_triples_per_run: 500,
            scaffolding_fraction_threshold: 0.2,
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Metrics from one compaction run.
#[derive(Debug, Clone, Default)]
pub struct CompactionMetrics {
    /// Total triples in the KG before this run.
    pub total_before: usize,
    /// Scaffolding triples identified by the heuristic.
    pub scaffolding_found: usize,
    /// Scaffolding triples actually removed (may be less than `scaffolding_found`
    /// when `max_triples_per_run` caps the batch).
    pub scaffolding_removed: usize,
    /// Knowledge density after this run: knowledge triples / remaining triples.
    ///
    /// 1.0 = fully clean, 0.0 = entirely scaffolding.
    pub knowledge_density: f32,
}

// ---------------------------------------------------------------------------
// Classification heuristic
// ---------------------------------------------------------------------------

/// Return `true` when a triple is pure episodic scaffolding rather than
/// domain knowledge.
///
/// Scaffolding triples are:
///
/// 1. **Episode subjects** — any triple where the subject label starts with
///    `episode:` (covers `episode:N → agent:learned → X` and siblings).
/// 2. **Episode objects** — any triple pointing *back* to an episode node
///    (covers `X → child-of → episode:N` reverse links).
/// 3. **Pure bookkeeping predicates** — `agent:from_cycle`,
///    `agent:memory_type`, `agent:consolidation_reason` carry no domain
///    content regardless of subject/object.
pub fn is_scaffolding(subject_label: &str, predicate_label: &str, object_label: &str) -> bool {
    if subject_label.starts_with("episode:") {
        return true;
    }
    if object_label.starts_with("episode:") {
        return true;
    }
    matches!(
        predicate_label,
        "agent:from_cycle" | "agent:memory_type" | "agent:consolidation_reason"
    )
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// Run one compaction pass.
///
/// Scans all triples currently in the KG, removes scaffolding triples up to
/// `config.max_triples_per_run`, and returns [`CompactionMetrics`] with the
/// knowledge density after the pass.
///
/// Returns immediately (metrics with `scaffolding_removed = 0`) when:
/// - The KG is empty, or
/// - The scaffolding fraction is below `config.scaffolding_fraction_threshold`.
pub fn run_kg_compaction(
    engine: &Engine,
    config: &CompactionConfig,
) -> AgentResult<CompactionMetrics> {
    let all_triples = engine.all_triples();
    let total_before = all_triples.len();

    if total_before == 0 {
        return Ok(CompactionMetrics {
            knowledge_density: 1.0,
            ..Default::default()
        });
    }

    // Classify every triple and collect the scaffolding candidates.
    let scaffolding_candidates: Vec<(SymbolId, SymbolId, SymbolId)> = all_triples
        .iter()
        .filter(|t| {
            let sub = engine.resolve_label(t.subject);
            let pred = engine.resolve_label(t.predicate);
            let obj = engine.resolve_label(t.object);
            is_scaffolding(&sub, &pred, &obj)
        })
        .map(|t| (t.subject, t.predicate, t.object))
        .collect();

    let scaffolding_found = scaffolding_candidates.len();

    // Skip when the graph is already clean enough.
    let frac = scaffolding_found as f32 / total_before as f32;
    if frac < config.scaffolding_fraction_threshold {
        return Ok(CompactionMetrics {
            total_before,
            scaffolding_found,
            scaffolding_removed: 0,
            knowledge_density: 1.0 - frac,
        });
    }

    // Remove scaffolding in a bounded batch.
    let mut scaffolding_removed = 0usize;
    for (subject, predicate, object) in scaffolding_candidates
        .into_iter()
        .take(config.max_triples_per_run)
    {
        match engine.remove_triple(subject, predicate, object) {
            Ok(_) => scaffolding_removed += 1,
            Err(e) => {
                tracing::debug!(error = %e, "compaction: remove_triple failed, continuing");
            }
        }
    }

    // Recompute knowledge density over remaining triples.
    let remaining = total_before.saturating_sub(scaffolding_removed);
    let remaining_scaffolding = scaffolding_found.saturating_sub(scaffolding_removed);
    let knowledge_density = if remaining == 0 {
        1.0_f32
    } else {
        remaining.saturating_sub(remaining_scaffolding) as f32 / remaining as f32
    };

    // Record provenance (sentinel sym 1 — compaction affects no single symbol).
    let sentinel = SymbolId::new(1).unwrap();
    if let Err(e) = engine.store_provenance(
        &mut ProvenanceRecord::new(
            sentinel,
            DerivationKind::KgCompaction {
                scaffolding_removed,
                total_before,
                knowledge_density,
            },
        )
        .with_confidence(1.0),
    ) {
        tracing::debug!("compaction: failed to store provenance: {e}");
    }

    Ok(CompactionMetrics {
        total_before,
        scaffolding_found,
        scaffolding_removed,
        knowledge_density,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::graph::Triple;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    // --- is_scaffolding ---

    #[test]
    fn episode_subject_is_scaffolding() {
        assert!(is_scaffolding("episode:42", "agent:learned", "concept:foo"));
    }

    #[test]
    fn episode_object_is_scaffolding() {
        assert!(is_scaffolding("concept:foo", "child-of", "episode:42"));
    }

    #[test]
    fn from_cycle_predicate_is_scaffolding() {
        assert!(is_scaffolding("concept:foo", "agent:from_cycle", "cycle:1"));
    }

    #[test]
    fn memory_type_predicate_is_scaffolding() {
        assert!(is_scaffolding(
            "concept:foo",
            "agent:memory_type",
            "semantic"
        ));
    }

    #[test]
    fn consolidation_reason_predicate_is_scaffolding() {
        assert!(is_scaffolding(
            "concept:bar",
            "agent:consolidation_reason",
            "high_relevance"
        ));
    }

    #[test]
    fn domain_triple_is_not_scaffolding() {
        assert!(!is_scaffolding("dog", "is-a", "animal"));
        assert!(!is_scaffolding("Paris", "capital-of", "France"));
        assert!(!is_scaffolding(
            "concept:foo",
            "agent:learned",
            "concept:bar"
        ));
    }

    #[test]
    fn agent_learned_without_episode_is_knowledge() {
        // agent:learned between two domain concepts is signal, not scaffolding.
        assert!(!is_scaffolding("concept:a", "agent:learned", "concept:b"));
    }

    // --- run_kg_compaction ---

    #[test]
    fn compaction_on_empty_engine_returns_clean() {
        let engine = test_engine();
        let config = CompactionConfig::default();
        let metrics = run_kg_compaction(&engine, &config).unwrap();
        assert_eq!(metrics.total_before, 0);
        assert_eq!(metrics.scaffolding_removed, 0);
        assert!((metrics.knowledge_density - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compaction_removes_episode_subject_triples() {
        let engine = test_engine();

        // Domain triple — should survive.
        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();
        let animal = engine.resolve_or_create_entity("animal").unwrap();
        engine.add_triple(&Triple::new(dog, is_a, animal)).unwrap();

        // Episode scaffolding — should be removed.
        let episode = engine.resolve_or_create_entity("episode:1").unwrap();
        let learned = engine.resolve_or_create_relation("agent:learned").unwrap();
        engine
            .add_triple(&Triple::new(episode, learned, dog))
            .unwrap();

        let config = CompactionConfig {
            scaffolding_fraction_threshold: 0.0, // always compact
            ..Default::default()
        };
        let metrics = run_kg_compaction(&engine, &config).unwrap();

        assert_eq!(metrics.total_before, 2);
        assert_eq!(metrics.scaffolding_found, 1);
        assert_eq!(metrics.scaffolding_removed, 1);

        // Domain triple still present.
        assert!(!engine.triples_from(dog).is_empty());
        // Episode triple gone.
        assert!(engine.triples_from(episode).is_empty());
    }

    #[test]
    fn compaction_removes_episode_object_triples() {
        let engine = test_engine();

        let concept = engine.resolve_or_create_entity("concept:foo").unwrap();
        let child_of = engine.resolve_or_create_relation("child-of").unwrap();
        let episode = engine.resolve_or_create_entity("episode:5").unwrap();
        engine
            .add_triple(&Triple::new(concept, child_of, episode))
            .unwrap();

        let config = CompactionConfig {
            scaffolding_fraction_threshold: 0.0,
            ..Default::default()
        };
        let metrics = run_kg_compaction(&engine, &config).unwrap();
        assert_eq!(metrics.scaffolding_removed, 1);
    }

    #[test]
    fn compaction_skips_when_below_threshold() {
        let engine = test_engine();

        // 9 domain triples, 1 scaffolding = 10% scaffolding.
        for i in 0..9u64 {
            let s = engine
                .resolve_or_create_entity(&format!("entity:{i}"))
                .unwrap();
            let p = engine.resolve_or_create_relation("rel:x").unwrap();
            let o = engine
                .resolve_or_create_entity(&format!("other:{i}"))
                .unwrap();
            engine.add_triple(&Triple::new(s, p, o)).unwrap();
        }
        let ep = engine.resolve_or_create_entity("episode:1").unwrap();
        let lrn = engine.resolve_or_create_relation("agent:learned").unwrap();
        let tgt = engine.resolve_or_create_entity("entity:0").unwrap();
        engine.add_triple(&Triple::new(ep, lrn, tgt)).unwrap();

        // Threshold = 0.2 → 10% < 20%, so compaction should skip.
        let config = CompactionConfig {
            scaffolding_fraction_threshold: 0.2,
            ..Default::default()
        };
        let metrics = run_kg_compaction(&engine, &config).unwrap();
        assert_eq!(metrics.scaffolding_removed, 0);
    }

    #[test]
    fn compaction_respects_max_triples_per_run() {
        let engine = test_engine();

        // Add 5 episode scaffolding triples.
        for i in 0..5u64 {
            let ep = engine
                .resolve_or_create_entity(&format!("episode:{i}"))
                .unwrap();
            let lrn = engine.resolve_or_create_relation("agent:learned").unwrap();
            let tgt = engine
                .resolve_or_create_entity(&format!("concept:{i}"))
                .unwrap();
            engine.add_triple(&Triple::new(ep, lrn, tgt)).unwrap();
        }

        let config = CompactionConfig {
            max_triples_per_run: 3,
            scaffolding_fraction_threshold: 0.0,
        };
        let metrics = run_kg_compaction(&engine, &config).unwrap();
        // At most 3 removed in one pass.
        assert!(metrics.scaffolding_removed <= 3);
    }

    #[test]
    fn knowledge_density_is_normalized() {
        let engine = test_engine();

        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();
        let animal = engine.resolve_or_create_entity("animal").unwrap();
        engine.add_triple(&Triple::new(dog, is_a, animal)).unwrap();

        let ep = engine.resolve_or_create_entity("episode:99").unwrap();
        let lrn = engine.resolve_or_create_relation("agent:learned").unwrap();
        engine.add_triple(&Triple::new(ep, lrn, dog)).unwrap();

        let config = CompactionConfig {
            scaffolding_fraction_threshold: 0.0,
            ..Default::default()
        };
        let metrics = run_kg_compaction(&engine, &config).unwrap();
        // After removing 1 of 2 triples, 1 remains and it's knowledge → density 1.0.
        assert!(metrics.knowledge_density >= 0.0);
        assert!(metrics.knowledge_density <= 1.0);
    }
}
