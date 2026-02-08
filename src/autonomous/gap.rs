//! Knowledge gap analysis: identifies missing knowledge in the KG.
//!
//! Three detection strategies:
//! 1. Dead-end detection — entities with very few connections
//! 2. Missing predicate detection — entities missing predicates that similar entities have
//! 3. Incomplete type detection — entities missing typical predicates for their type

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::error::{AutonomousError, AutonomousResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// What kind of knowledge gap was detected.
#[derive(Debug, Clone)]
pub enum GapKind {
    /// Entity with few connections.
    DeadEnd {
        in_degree: usize,
        out_degree: usize,
    },
    /// Entity missing a predicate that similar entities have.
    MissingPredicate {
        expected_predicate: SymbolId,
        exemplar_entities: Vec<SymbolId>,
    },
    /// Entity has a type but is missing typical predicates for that type.
    IncompleteType {
        entity_type: SymbolId,
        missing_predicates: Vec<SymbolId>,
    },
}

/// A single identified knowledge gap.
#[derive(Debug, Clone)]
pub struct KnowledgeGap {
    pub entity: SymbolId,
    pub kind: GapKind,
    /// Severity from 0.0 to 1.0 (higher = more critical).
    pub severity: f32,
    pub description: String,
    pub suggested_predicates: Vec<SymbolId>,
}

/// Configuration for gap analysis.
#[derive(Debug, Clone)]
pub struct GapAnalysisConfig {
    /// Minimum total degree (in + out) before flagging as dead end (default: 2).
    pub min_degree: usize,
    /// Number of similar entities to consider for missing-predicate detection (default: 5).
    pub cluster_size: usize,
    /// Minimum similarity for clustering (default: 0.6).
    pub min_cluster_similarity: f32,
    /// Maximum gaps to report (default: 20).
    pub max_gaps: usize,
}

impl Default for GapAnalysisConfig {
    fn default() -> Self {
        Self {
            min_degree: 2,
            cluster_size: 5,
            min_cluster_similarity: 0.6,
            max_gaps: 20,
        }
    }
}

/// Result of gap analysis.
#[derive(Debug, Clone)]
pub struct GapAnalysisResult {
    /// Gaps sorted by severity (highest first).
    pub gaps: Vec<KnowledgeGap>,
    /// Total entities analyzed.
    pub entities_analyzed: usize,
    /// Count of dead-end entities found.
    pub dead_ends: usize,
    /// Coverage score: well-connected entities / total entities.
    pub coverage_score: f32,
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Analyze knowledge gaps around the given goal symbols.
pub fn analyze_gaps(
    engine: &Engine,
    goal_symbols: &[SymbolId],
    config: &GapAnalysisConfig,
) -> AutonomousResult<GapAnalysisResult> {
    if goal_symbols.is_empty() {
        return Err(AutonomousError::NoGoalsForGap);
    }

    let mut gaps = Vec::new();
    let mut entities_analyzed = 0usize;
    let mut dead_ends = 0usize;
    let mut well_connected = 0usize;

    // Collect entities reachable from goal symbols (BFS, depth 2).
    let reachable = collect_reachable(engine, goal_symbols, 2);
    entities_analyzed = reachable.len();

    // 1. Dead-end detection.
    for &entity in &reachable {
        let out_degree = engine.triples_from(entity).len();
        let in_degree = engine.triples_to(entity).len();
        let total = in_degree + out_degree;

        if total < config.min_degree {
            dead_ends += 1;
            let depth = min_depth_from_goals(engine, entity, goal_symbols);
            // Higher severity for entities closer to goals.
            let severity = if depth <= 1 { 0.9 } else { 0.7 / depth as f32 };

            gaps.push(KnowledgeGap {
                entity,
                kind: GapKind::DeadEnd {
                    in_degree,
                    out_degree,
                },
                severity,
                description: format!(
                    "{} has only {} connection(s) (in={}, out={})",
                    engine.resolve_label(entity),
                    total,
                    in_degree,
                    out_degree
                ),
                suggested_predicates: Vec::new(),
            });
        } else {
            well_connected += 1;
        }
    }

    // 2. Missing predicate detection via VSA similarity.
    for &entity in &reachable {
        let entity_predicates = outgoing_predicates(engine, entity);
        if entity_predicates.is_empty() {
            continue;
        }

        if let Ok(similar) = engine.search_similar_to(entity, config.cluster_size) {
            let mut pred_counts: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();

            for sr in &similar {
                if sr.symbol_id == entity {
                    continue;
                }
                for pred in outgoing_predicates(engine, sr.symbol_id) {
                    pred_counts
                        .entry(pred)
                        .or_default()
                        .push(sr.symbol_id);
                }
            }

            let similar_count = similar.len().max(1);
            for (pred, exemplars) in &pred_counts {
                // Predicate present in >50% of similar entities but absent from this one.
                let coverage = exemplars.len() as f32 / similar_count as f32;
                if coverage > 0.5 && !entity_predicates.contains(pred) {
                    let severity = coverage * 0.8;
                    gaps.push(KnowledgeGap {
                        entity,
                        kind: GapKind::MissingPredicate {
                            expected_predicate: *pred,
                            exemplar_entities: exemplars.clone(),
                        },
                        severity,
                        description: format!(
                            "{} is missing predicate '{}' (present in {:.0}% of similar entities)",
                            engine.resolve_label(entity),
                            engine.resolve_label(*pred),
                            coverage * 100.0
                        ),
                        suggested_predicates: vec![*pred],
                    });
                }
            }
        }
    }

    // 3. Incomplete type detection.
    if let Ok(isa_id) = engine.lookup_symbol("is-a") {
        // Group entities by type.
        let mut type_members: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        for &entity in &reachable {
            let types = engine.knowledge_graph().objects_of(entity, isa_id);
            for type_id in types {
                type_members.entry(type_id).or_default().push(entity);
            }
        }

        for (type_id, members) in &type_members {
            if members.len() < 2 {
                continue;
            }

            // Compute typical predicate set (>60% coverage).
            let mut pred_counts: HashMap<SymbolId, usize> = HashMap::new();
            for &member in members {
                for pred in outgoing_predicates(engine, member) {
                    *pred_counts.entry(pred).or_insert(0) += 1;
                }
            }

            let typical_preds: Vec<SymbolId> = pred_counts
                .iter()
                .filter(|&(_, &count)| (count as f32 / members.len() as f32) > 0.6)
                .map(|(pred, _)| *pred)
                .collect();

            for &member in members {
                let member_preds = outgoing_predicates(engine, member);
                let missing: Vec<SymbolId> = typical_preds
                    .iter()
                    .filter(|p| !member_preds.contains(p))
                    .copied()
                    .collect();

                if !missing.is_empty() {
                    let severity = missing.len() as f32 / typical_preds.len().max(1) as f32;
                    gaps.push(KnowledgeGap {
                        entity: member,
                        kind: GapKind::IncompleteType {
                            entity_type: *type_id,
                            missing_predicates: missing.clone(),
                        },
                        severity: severity.min(1.0) * 0.75,
                        description: format!(
                            "{} (type {}) is missing {} typical predicate(s)",
                            engine.resolve_label(member),
                            engine.resolve_label(*type_id),
                            missing.len()
                        ),
                        suggested_predicates: missing,
                    });
                }
            }
        }
    }

    // Sort by severity descending.
    gaps.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
    gaps.truncate(config.max_gaps);

    let coverage_score = if entities_analyzed > 0 {
        well_connected as f32 / entities_analyzed as f32
    } else {
        0.0
    };

    Ok(GapAnalysisResult {
        gaps,
        entities_analyzed,
        dead_ends,
        coverage_score,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// BFS from seeds to collect all reachable entities within depth.
fn collect_reachable(engine: &Engine, seeds: &[SymbolId], max_depth: usize) -> Vec<SymbolId> {
    let mut visited: HashSet<SymbolId> = HashSet::new();
    let mut frontier: Vec<SymbolId> = seeds.to_vec();
    visited.extend(seeds);

    for _ in 0..max_depth {
        let mut next_frontier = Vec::new();
        for &entity in &frontier {
            for triple in engine.triples_from(entity) {
                if visited.insert(triple.object) {
                    next_frontier.push(triple.object);
                }
            }
            for triple in engine.triples_to(entity) {
                if visited.insert(triple.subject) {
                    next_frontier.push(triple.subject);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    visited.into_iter().collect()
}

/// Minimum hop distance from any goal symbol to the entity.
fn min_depth_from_goals(engine: &Engine, entity: SymbolId, goals: &[SymbolId]) -> usize {
    for goal in goals {
        if let Ok(Some(path)) = engine.shortest_path(*goal, entity) {
            return path.len().saturating_sub(1);
        }
    }
    // Fallback: unreachable gets max depth.
    usize::MAX
}

/// Collect the set of outgoing predicate IDs for an entity.
fn outgoing_predicates(engine: &Engine, entity: SymbolId) -> HashSet<SymbolId> {
    engine
        .triples_from(entity)
        .iter()
        .map(|t| t.predicate)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn no_goals_returns_error() {
        let engine = test_engine();
        let result = analyze_gaps(&engine, &[], &GapAnalysisConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn dead_end_detected() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("A".into(), "knows".into(), "B".into(), 1.0)])
            .unwrap();

        let b = engine.lookup_symbol("B").unwrap();
        let result = analyze_gaps(&engine, &[b], &GapAnalysisConfig::default()).unwrap();

        // B has only 1 incoming edge — should be flagged as dead end.
        assert!(result.dead_ends > 0);
        assert!(result.gaps.iter().any(|g| matches!(g.kind, GapKind::DeadEnd { .. })));
    }

    #[test]
    fn well_connected_entity_not_flagged() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "knows".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("B".into(), "has".into(), "D".into(), 1.0),
            ])
            .unwrap();

        let b = engine.lookup_symbol("B").unwrap();
        let config = GapAnalysisConfig {
            min_degree: 2,
            ..Default::default()
        };
        let result = analyze_gaps(&engine, &[b], &config).unwrap();

        // B has 3 connections (1 in, 2 out), should not be a dead end.
        let b_dead = result.gaps.iter().any(|g| {
            g.entity == b && matches!(g.kind, GapKind::DeadEnd { .. })
        });
        assert!(!b_dead);
    }

    #[test]
    fn gaps_sorted_by_severity() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "rel".into(), "B".into(), 1.0),
                ("A".into(), "rel".into(), "C".into(), 1.0),
            ])
            .unwrap();

        let a = engine.lookup_symbol("A").unwrap();
        let result = analyze_gaps(&engine, &[a], &GapAnalysisConfig::default()).unwrap();

        // Verify severity is sorted descending.
        for window in result.gaps.windows(2) {
            assert!(window[0].severity >= window[1].severity);
        }
    }

    #[test]
    fn empty_graph_no_crash() {
        let engine = test_engine();
        let sym = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "lonely")
            .unwrap();
        let result = analyze_gaps(&engine, &[sym.id], &GapAnalysisConfig::default()).unwrap();
        assert_eq!(result.entities_analyzed, 1);
    }

    #[test]
    fn incomplete_type_detected() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("Dog1".into(), "is-a".into(), "Animal".into(), 1.0),
                ("Dog1".into(), "has-legs".into(), "4".into(), 1.0),
                ("Dog1".into(), "has-name".into(), "Fido".into(), 1.0),
                ("Dog2".into(), "is-a".into(), "Animal".into(), 1.0),
                ("Dog2".into(), "has-legs".into(), "4".into(), 1.0),
                ("Dog2".into(), "has-name".into(), "Rex".into(), 1.0),
                ("Dog3".into(), "is-a".into(), "Animal".into(), 1.0),
                // Dog3 is missing has-legs and has-name
            ])
            .unwrap();

        let animal = engine.lookup_symbol("Animal").unwrap();
        let result = analyze_gaps(&engine, &[animal], &GapAnalysisConfig::default()).unwrap();

        let dog3 = engine.lookup_symbol("Dog3").unwrap();
        assert!(result.gaps.iter().any(|g| {
            g.entity == dog3 && matches!(g.kind, GapKind::IncompleteType { .. })
        }));
    }
}
