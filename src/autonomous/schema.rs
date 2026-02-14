//! Schema discovery from predicate patterns.
//!
//! Analyzes the KG to discover implicit type structures, co-occurring
//! predicate patterns, and relation hierarchies.

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::error::{AutonomousError, AutonomousResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A predicate pattern: how many entities use this predicate and its coverage.
#[derive(Debug, Clone)]
pub struct PredicatePattern {
    pub predicate: SymbolId,
    pub entity_count: usize,
    pub coverage: f32,
}

/// A discovered entity type based on shared predicate signatures.
#[derive(Debug, Clone)]
pub struct DiscoveredType {
    /// Representative entity of this type cluster.
    pub exemplar: SymbolId,
    /// Existing type symbol if entities share an `is-a` target.
    pub type_symbol: Option<SymbolId>,
    /// All entities in this type cluster.
    pub members: Vec<SymbolId>,
    /// Predicates typical for this type.
    pub typical_predicates: Vec<PredicatePattern>,
}

/// A discovered implication between predicates.
#[derive(Debug, Clone)]
pub struct RelationHierarchy {
    /// The more specific predicate.
    pub specific: SymbolId,
    /// The more general predicate.
    pub general: SymbolId,
    /// Strength of the implication (0.0–1.0).
    pub implication_strength: f32,
}

/// Result of schema discovery.
#[derive(Debug, Clone)]
pub struct SchemaDiscoveryResult {
    pub types: Vec<DiscoveredType>,
    /// Predicate pairs that frequently co-occur: (P1, P2, co-occurrence score).
    pub co_occurring_predicates: Vec<(SymbolId, SymbolId, f32)>,
    /// Discovered implication hierarchies.
    pub relation_hierarchies: Vec<RelationHierarchy>,
}

/// Configuration for schema discovery.
#[derive(Debug, Clone)]
pub struct SchemaDiscoveryConfig {
    /// Minimum members in a type cluster (default: 3).
    pub min_type_members: usize,
    /// Minimum co-occurrence ratio to report (default: 0.5).
    pub min_co_occurrence: f32,
    /// Minimum implication strength to report (default: 0.7).
    pub min_implication_strength: f32,
}

impl Default for SchemaDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_type_members: 3,
            min_co_occurrence: 0.5,
            min_implication_strength: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover schema patterns from the knowledge graph.
pub fn discover_schema(
    engine: &Engine,
    config: &SchemaDiscoveryConfig,
) -> AutonomousResult<SchemaDiscoveryResult> {
    let all_triples = engine.all_triples();
    if all_triples.len() < 3 {
        return Err(AutonomousError::InsufficientData {
            min_triples: 3,
            actual: all_triples.len(),
        });
    }

    // Collect all entities (subjects) and their predicate signatures.
    let mut entity_predicates: HashMap<SymbolId, HashSet<SymbolId>> = HashMap::new();
    for triple in &all_triples {
        entity_predicates
            .entry(triple.subject)
            .or_default()
            .insert(triple.predicate);
    }

    // 1. Discover types by grouping entities with similar predicate signatures.
    let types = discover_types(&entity_predicates, engine, config);

    // 2. Discover co-occurring predicates.
    let co_occurring = discover_co_occurrences(&entity_predicates, config);

    // 3. Discover relation hierarchies (P1 implies P2).
    let hierarchies = discover_hierarchies(&entity_predicates, config);

    Ok(SchemaDiscoveryResult {
        types,
        co_occurring_predicates: co_occurring,
        relation_hierarchies: hierarchies,
    })
}

/// Group entities by predicate signature similarity (Jaccard > 0.5).
fn discover_types(
    entity_predicates: &HashMap<SymbolId, HashSet<SymbolId>>,
    engine: &Engine,
    config: &SchemaDiscoveryConfig,
) -> Vec<DiscoveredType> {
    let entities: Vec<SymbolId> = entity_predicates.keys().copied().collect();
    let mut assigned: HashSet<SymbolId> = HashSet::new();
    let mut types = Vec::new();

    for &entity in &entities {
        if assigned.contains(&entity) {
            continue;
        }
        let sig = match entity_predicates.get(&entity) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };

        // Find all entities with Jaccard similarity > 0.5 to this entity's signature.
        let mut cluster = vec![entity];
        for &other in &entities {
            if other == entity || assigned.contains(&other) {
                continue;
            }
            if let Some(other_sig) = entity_predicates.get(&other) {
                let jaccard = jaccard_index(sig, other_sig);
                if jaccard > 0.5 {
                    cluster.push(other);
                }
            }
        }

        if cluster.len() >= config.min_type_members {
            for &member in &cluster {
                assigned.insert(member);
            }

            // Check if members share a common is-a type.
            let type_symbol = find_common_type(&cluster, engine);

            // Compute typical predicates (>50% coverage in cluster).
            let mut pred_counts: HashMap<SymbolId, usize> = HashMap::new();
            for &member in &cluster {
                if let Some(preds) = entity_predicates.get(&member) {
                    for &pred in preds {
                        *pred_counts.entry(pred).or_insert(0) += 1;
                    }
                }
            }

            let typical: Vec<PredicatePattern> = pred_counts
                .iter()
                .filter_map(|(&pred, &count)| {
                    let coverage = count as f32 / cluster.len() as f32;
                    if coverage > 0.5 {
                        Some(PredicatePattern {
                            predicate: pred,
                            entity_count: count,
                            coverage,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            types.push(DiscoveredType {
                exemplar: cluster[0],
                type_symbol,
                members: cluster,
                typical_predicates: typical,
            });
        }
    }

    types
}

/// Find co-occurring predicate pairs.
fn discover_co_occurrences(
    entity_predicates: &HashMap<SymbolId, HashSet<SymbolId>>,
    config: &SchemaDiscoveryConfig,
) -> Vec<(SymbolId, SymbolId, f32)> {
    // Collect all unique predicates.
    let all_preds: HashSet<SymbolId> = entity_predicates
        .values()
        .flat_map(|s| s.iter())
        .copied()
        .collect();
    let pred_list: Vec<SymbolId> = all_preds.into_iter().collect();

    // Count entities per predicate.
    let mut pred_entity_count: HashMap<SymbolId, usize> = HashMap::new();
    for preds in entity_predicates.values() {
        for &pred in preds {
            *pred_entity_count.entry(pred).or_insert(0) += 1;
        }
    }

    let mut co_occurring = Vec::new();

    for i in 0..pred_list.len() {
        for j in (i + 1)..pred_list.len() {
            let p1 = pred_list[i];
            let p2 = pred_list[j];

            // Count entities that have both.
            let both_count = entity_predicates
                .values()
                .filter(|preds| preds.contains(&p1) && preds.contains(&p2))
                .count();

            let p1_count = pred_entity_count.get(&p1).copied().unwrap_or(0);
            if p1_count == 0 {
                continue;
            }

            let co_occ = both_count as f32 / p1_count as f32;
            if co_occ >= config.min_co_occurrence {
                co_occurring.push((p1, p2, co_occ));
            }
        }
    }

    co_occurring.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    co_occurring
}

/// Discover predicate implication hierarchies:
/// P1 implies P2 if `|entities with both| / |entities with P1| > threshold`.
fn discover_hierarchies(
    entity_predicates: &HashMap<SymbolId, HashSet<SymbolId>>,
    config: &SchemaDiscoveryConfig,
) -> Vec<RelationHierarchy> {
    let all_preds: HashSet<SymbolId> = entity_predicates
        .values()
        .flat_map(|s| s.iter())
        .copied()
        .collect();
    let pred_list: Vec<SymbolId> = all_preds.into_iter().collect();

    let mut pred_entity_count: HashMap<SymbolId, usize> = HashMap::new();
    for preds in entity_predicates.values() {
        for &pred in preds {
            *pred_entity_count.entry(pred).or_insert(0) += 1;
        }
    }

    let mut hierarchies = Vec::new();

    for &p1 in &pred_list {
        let p1_count = pred_entity_count.get(&p1).copied().unwrap_or(0);
        if p1_count == 0 {
            continue;
        }

        for &p2 in &pred_list {
            if p1 == p2 {
                continue;
            }

            let both_count = entity_predicates
                .values()
                .filter(|preds| preds.contains(&p1) && preds.contains(&p2))
                .count();

            let strength = both_count as f32 / p1_count as f32;
            if strength >= config.min_implication_strength {
                hierarchies.push(RelationHierarchy {
                    specific: p1,
                    general: p2,
                    implication_strength: strength,
                });
            }
        }
    }

    hierarchies.sort_by(|a, b| {
        b.implication_strength
            .partial_cmp(&a.implication_strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hierarchies
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Jaccard similarity index between two sets.
fn jaccard_index(a: &HashSet<SymbolId>, b: &HashSet<SymbolId>) -> f32 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Find a common `is-a` type shared by all members.
fn find_common_type(members: &[SymbolId], engine: &Engine) -> Option<SymbolId> {
    let isa_id = engine.lookup_symbol("is-a").ok()?;

    let mut type_counts: HashMap<SymbolId, usize> = HashMap::new();
    for &member in members {
        let types = engine.knowledge_graph().objects_of(member, isa_id);
        for t in types {
            *type_counts.entry(t).or_insert(0) += 1;
        }
    }

    // Return the type shared by the most members.
    type_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .filter(|(_, count)| *count > members.len() / 2)
        .map(|(t, _)| t)
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
    fn insufficient_data_error() {
        let engine = test_engine();
        let result = discover_schema(&engine, &SchemaDiscoveryConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn discovers_entity_types() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("Dog1".into(), "has-legs".into(), "4".into(), 1.0),
                ("Dog1".into(), "has-name".into(), "Fido".into(), 1.0),
                ("Dog1".into(), "is-a".into(), "Animal".into(), 1.0),
                ("Dog2".into(), "has-legs".into(), "4".into(), 1.0),
                ("Dog2".into(), "has-name".into(), "Rex".into(), 1.0),
                ("Dog2".into(), "is-a".into(), "Animal".into(), 1.0),
                ("Dog3".into(), "has-legs".into(), "4".into(), 1.0),
                ("Dog3".into(), "has-name".into(), "Bud".into(), 1.0),
                ("Dog3".into(), "is-a".into(), "Animal".into(), 1.0),
            ])
            .unwrap();

        let result = discover_schema(&engine, &SchemaDiscoveryConfig::default()).unwrap();

        // Should find a type cluster for the dogs.
        assert!(!result.types.is_empty());
        let dog_type = &result.types[0];
        assert!(dog_type.members.len() >= 3);
    }

    #[test]
    fn co_occurring_predicates_found() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "pred1".into(), "X".into(), 1.0),
                ("A".into(), "pred2".into(), "Y".into(), 1.0),
                ("B".into(), "pred1".into(), "X".into(), 1.0),
                ("B".into(), "pred2".into(), "Y".into(), 1.0),
                ("C".into(), "pred1".into(), "X".into(), 1.0),
                ("C".into(), "pred2".into(), "Y".into(), 1.0),
            ])
            .unwrap();

        let result = discover_schema(&engine, &SchemaDiscoveryConfig::default()).unwrap();

        // pred1 and pred2 always co-occur.
        assert!(!result.co_occurring_predicates.is_empty());
    }

    #[test]
    fn relation_hierarchy_found() {
        let engine = test_engine();
        // Every entity with pred1 also has pred2 (but not vice versa).
        engine
            .ingest_label_triples(&[
                ("A".into(), "specific".into(), "X".into(), 1.0),
                ("A".into(), "general".into(), "Y".into(), 1.0),
                ("B".into(), "specific".into(), "X".into(), 1.0),
                ("B".into(), "general".into(), "Y".into(), 1.0),
                ("C".into(), "specific".into(), "X".into(), 1.0),
                ("C".into(), "general".into(), "Y".into(), 1.0),
                ("D".into(), "general".into(), "Y".into(), 1.0),
            ])
            .unwrap();

        let config = SchemaDiscoveryConfig {
            min_implication_strength: 0.7,
            ..Default::default()
        };
        let result = discover_schema(&engine, &config).unwrap();

        // "specific" implies "general" (3/3 = 1.0).
        let specific = engine.lookup_symbol("specific").unwrap();
        let general = engine.lookup_symbol("general").unwrap();
        assert!(
            result
                .relation_hierarchies
                .iter()
                .any(|h| { h.specific == specific && h.general == general })
        );
    }

    #[test]
    fn min_type_members_enforced() {
        let engine = test_engine();
        // Only 2 similar entities — below default threshold of 3.
        engine
            .ingest_label_triples(&[
                ("A".into(), "p1".into(), "X".into(), 1.0),
                ("A".into(), "p2".into(), "Y".into(), 1.0),
                ("B".into(), "p1".into(), "X".into(), 1.0),
                ("B".into(), "p2".into(), "Y".into(), 1.0),
                ("C".into(), "p3".into(), "Z".into(), 1.0),
            ])
            .unwrap();

        let result = discover_schema(
            &engine,
            &SchemaDiscoveryConfig {
                min_type_members: 3,
                ..Default::default()
            },
        )
        .unwrap();

        // No type cluster should have < 3 members.
        for t in &result.types {
            assert!(t.members.len() >= 3);
        }
    }
}
