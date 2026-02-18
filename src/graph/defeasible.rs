//! Defeasible reasoning: specificity-based override resolution.
//!
//! Inspired by Cyc's defeasibility model. Almost all knowledge is default-true.
//! When triples conflict, specificity-based override applies: more specific rules
//! trump more general ones (`penguin.flies=false` overrides `bird.flies=true`
//! because penguin is-a bird). Monotonic rules always preferred over defaults.
//!
//! Override resolution order:
//! 1. Monotonic > default (monotonic assertions never overridden)
//! 2. Specific > general (deeper in `is-a` chain wins)
//! 3. Exception > default (explicitly registered exceptions win)
//! 4. Recency (newer triples break ties)
//! 5. Confidence (higher confidence as final tiebreaker)

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::predicate_hierarchy::PredicateHierarchy;

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known predicates for the defeasible reasoning system (prefixed `defeasible:`).
#[derive(Debug, Clone)]
pub struct DefeasiblePredicates {
    /// `defeasible:monotonic` — marks a triple as monotonically true (never overridden).
    pub monotonic: SymbolId,
    /// `defeasible:except` — registers an exception: `(general_rule defeasible:except specific_override)`.
    pub except: SymbolId,
    /// `defeasible:overrides` — explicit override declaration.
    pub overrides: SymbolId,
    /// `is-a` — used for specificity computation via type hierarchy.
    pub is_a: SymbolId,
}

impl DefeasiblePredicates {
    /// Resolve all `defeasible:` predicates from the engine's registry, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            monotonic: engine.resolve_or_create_relation("defeasible:monotonic")?,
            except: engine.resolve_or_create_relation("defeasible:except")?,
            overrides: engine.resolve_or_create_relation("defeasible:overrides")?,
            is_a: engine.resolve_or_create_relation("is-a")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Override reason
// ---------------------------------------------------------------------------

/// Why one triple overrides another in defeasible conflict resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OverrideReason {
    /// The winner is monotonically asserted (never defeatable).
    Monotonic,
    /// The winner's subject is more specific in the `is-a` hierarchy.
    Specificity {
        /// How much deeper the winner's type is in the `is-a` chain.
        depth_advantage: usize,
    },
    /// The winner is an explicitly registered exception to the loser.
    Exception,
    /// The winner was asserted more recently.
    Recency,
    /// The winner has higher confidence (final tiebreaker).
    Confidence,
}

impl std::fmt::Display for OverrideReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Monotonic => write!(f, "monotonic"),
            Self::Specificity { depth_advantage } => {
                write!(f, "specificity (depth advantage: {})", depth_advantage)
            }
            Self::Exception => write!(f, "exception"),
            Self::Recency => write!(f, "recency"),
            Self::Confidence => write!(f, "confidence"),
        }
    }
}

// ---------------------------------------------------------------------------
// Defeasible result
// ---------------------------------------------------------------------------

/// Result of defeasible conflict resolution for a set of competing triples.
#[derive(Debug, Clone)]
pub struct DefeasibleResult {
    /// The winning triple.
    pub winner: Triple,
    /// The losing triples, in the order they were defeated.
    pub losers: Vec<Triple>,
    /// Why the winner won.
    pub reason: OverrideReason,
}

// ---------------------------------------------------------------------------
// Specificity computation
// ---------------------------------------------------------------------------

/// Compute the `is-a` chain depth of a symbol: how many hops from the root
/// of the type hierarchy. Deeper = more specific.
///
/// Uses BFS up the `is-a` chain and returns the longest path length.
pub fn type_depth(engine: &Engine, symbol: SymbolId, is_a_pred: SymbolId) -> usize {
    let kg = engine.knowledge_graph();
    let mut max_depth = 0;
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(SymbolId, usize)> = VecDeque::new();

    // Start from the symbol's types
    for parent in kg.objects_of(symbol, is_a_pred) {
        if visited.insert(parent) {
            queue.push_back((parent, 1));
        }
    }

    while let Some((current, depth)) = queue.pop_front() {
        max_depth = max_depth.max(depth);
        for grandparent in kg.objects_of(current, is_a_pred) {
            if visited.insert(grandparent) {
                queue.push_back((grandparent, depth + 1));
            }
        }
    }

    max_depth
}

/// Check if `specific_type` is a (transitive) subtype of `general_type` via `is-a` chains.
fn is_subtype_of(
    engine: &Engine,
    specific_type: SymbolId,
    general_type: SymbolId,
    is_a_pred: SymbolId,
) -> bool {
    if specific_type == general_type {
        return true;
    }

    let kg = engine.knowledge_graph();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    for parent in kg.objects_of(specific_type, is_a_pred) {
        if visited.insert(parent) {
            queue.push_back(parent);
        }
    }

    while let Some(current) = queue.pop_front() {
        if current == general_type {
            return true;
        }
        for grandparent in kg.objects_of(current, is_a_pred) {
            if visited.insert(grandparent) {
                queue.push_back(grandparent);
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Conflict resolution
// ---------------------------------------------------------------------------

/// Check if a triple is marked as monotonic.
fn is_monotonic(engine: &Engine, triple: &Triple, preds: &DefeasiblePredicates) -> bool {
    let kg = engine.knowledge_graph();
    // Check if this triple's subject has `defeasible:monotonic` for this predicate
    kg.objects_of(triple.subject, preds.monotonic)
        .contains(&triple.predicate)
}

/// Check if triple `a` is an explicit exception to triple `b`.
fn is_exception_of(
    engine: &Engine,
    exception: &Triple,
    general: &Triple,
    preds: &DefeasiblePredicates,
) -> bool {
    let kg = engine.knowledge_graph();
    // Check: general.subject defeasible:except exception.subject
    kg.objects_of(general.subject, preds.except)
        .contains(&exception.subject)
        || kg
            .objects_of(general.predicate, preds.except)
            .contains(&exception.predicate)
}

/// Resolve a conflict between competing triples for the same query.
///
/// Given multiple triples that provide conflicting answers for the same
/// (subject-or-type, predicate) pair, determines which triple wins using
/// the defeasibility override order:
///
/// 1. Monotonic triples always win over default triples
/// 2. More specific types win over more general types
/// 3. Explicit exceptions win over the rules they except
/// 4. More recent triples win over older ones
/// 5. Higher confidence is the final tiebreaker
///
/// Returns `None` if fewer than 2 candidates are provided.
pub fn resolve_conflict(
    engine: &Engine,
    candidates: &[Triple],
    preds: &DefeasiblePredicates,
) -> Option<DefeasibleResult> {
    if candidates.len() < 2 {
        return candidates.first().map(|t| DefeasibleResult {
            winner: t.clone(),
            losers: vec![],
            reason: OverrideReason::Confidence,
        });
    }

    // Tag each candidate with monotonicity and type depth
    let mut tagged: Vec<(Triple, bool, usize)> = candidates
        .iter()
        .map(|t| {
            let mono = is_monotonic(engine, t, preds);
            let depth = type_depth(engine, t.subject, preds.is_a);
            (t.clone(), mono, depth)
        })
        .collect();

    // Sort by override priority (stable sort preserves insertion order for ties):
    // 1. Monotonic first (true > false when reversed)
    // 2. Deeper type first (more specific)
    // 3. More recent first (higher timestamp)
    // 4. Higher confidence first
    tagged.sort_by(|a, b| {
        // Monotonic: true > false
        b.1.cmp(&a.1)
            // Depth: deeper > shallower
            .then(b.2.cmp(&a.2))
            // Timestamp: newer > older
            .then(b.0.timestamp.cmp(&a.0.timestamp))
            // Confidence: higher > lower
            .then(
                b.0.confidence
                    .partial_cmp(&a.0.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let winner = tagged[0].0.clone();
    let winner_mono = tagged[0].1;
    let winner_depth = tagged[0].2;
    let losers: Vec<Triple> = tagged[1..].iter().map(|(t, _, _)| t.clone()).collect();

    // Determine the override reason by comparing winner to the best loser
    let best_loser_mono = tagged[1].1;
    let best_loser_depth = tagged[1].2;

    // Check for exception relationship
    let is_except = is_exception_of(engine, &winner, &tagged[1].0, preds);

    let reason = if winner_mono && !best_loser_mono {
        OverrideReason::Monotonic
    } else if is_except {
        OverrideReason::Exception
    } else if winner_depth > best_loser_depth {
        OverrideReason::Specificity {
            depth_advantage: winner_depth - best_loser_depth,
        }
    } else if winner.timestamp > tagged[1].0.timestamp {
        OverrideReason::Recency
    } else {
        OverrideReason::Confidence
    };

    Some(DefeasibleResult {
        winner,
        losers,
        reason,
    })
}

/// Find conflicting triples for a (subject, predicate) pair.
///
/// Two triples conflict when they share the same subject and predicate
/// but have different objects. This is the basic conflict pattern for
/// functional predicates (one-to-one relations).
pub fn find_conflicts(engine: &Engine, subject: SymbolId, predicate: SymbolId) -> Vec<Triple> {
    let kg = engine.knowledge_graph();
    let triples: Vec<Triple> = kg
        .triples_from(subject)
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .collect();

    if triples.len() > 1 {
        triples
    } else {
        vec![]
    }
}

/// Find conflicting triples across type hierarchy.
///
/// For a given entity and predicate, collects triples from the entity itself
/// and from all its types/supertypes via `is-a`. This catches the classic
/// "penguin.flies=false overrides bird.flies=true" pattern.
pub fn find_hierarchy_conflicts(
    engine: &Engine,
    entity: SymbolId,
    predicate: SymbolId,
    preds: &DefeasiblePredicates,
) -> Vec<Triple> {
    let kg = engine.knowledge_graph();
    let mut candidates = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Start with the entity itself
    queue.push_back(entity);
    visited.insert(entity);

    while let Some(current) = queue.pop_front() {
        // Collect triples for (current, predicate, ?)
        for triple in kg.triples_from(current) {
            if triple.predicate == predicate {
                candidates.push(triple);
            }
        }

        // Walk up the is-a hierarchy
        for parent_type in kg.objects_of(current, preds.is_a) {
            if visited.insert(parent_type) {
                queue.push_back(parent_type);
            }
        }
    }

    if candidates.len() > 1 {
        candidates
    } else {
        vec![]
    }
}

/// Resolve a query with defeasible reasoning: find the best answer considering
/// specificity, monotonicity, exceptions, recency, and confidence.
///
/// Returns the winning triple (if any) after conflict resolution, or the
/// single answer if there's no conflict.
pub fn query_defeasible(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    preds: &DefeasiblePredicates,
) -> Option<DefeasibleResult> {
    let conflicts = find_hierarchy_conflicts(engine, subject, predicate, preds);
    if conflicts.is_empty() {
        // No conflicts — check for a single answer
        let kg = engine.knowledge_graph();
        let triples: Vec<Triple> = kg
            .triples_from(subject)
            .into_iter()
            .filter(|t| t.predicate == predicate)
            .collect();
        if triples.len() == 1 {
            return Some(DefeasibleResult {
                winner: triples.into_iter().next().unwrap(),
                losers: vec![],
                reason: OverrideReason::Confidence,
            });
        }
        return None;
    }
    resolve_conflict(engine, &conflicts, preds)
}

/// Mark a subject's predicate as monotonically true (never overridden by defaults).
///
/// Stores: `subject defeasible:monotonic predicate`
pub fn mark_monotonic(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    preds: &DefeasiblePredicates,
) -> AkhResult<()> {
    engine.add_triple(&Triple::new(subject, preds.monotonic, predicate))
}

/// Register an exception: `general_subject defeasible:except specific_subject`.
///
/// This declares that `specific_subject`'s assertions should override
/// `general_subject`'s for conflicting predicates.
pub fn register_exception(
    engine: &Engine,
    general_subject: SymbolId,
    specific_subject: SymbolId,
    preds: &DefeasiblePredicates,
) -> AkhResult<()> {
    engine.add_triple(&Triple::new(general_subject, preds.except, specific_subject))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn no_conflict_single_answer() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let flies = engine.resolve_or_create_relation("flies").unwrap();
        let yes = engine.resolve_or_create_entity("yes").unwrap();

        engine.add_triple(&Triple::new(dog, flies, yes)).unwrap();

        let result = query_defeasible(&engine, dog, flies, &preds);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.winner.object, yes);
        assert!(r.losers.is_empty());
    }

    #[test]
    fn specificity_override() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        // Type hierarchy: penguin is-a bird
        let penguin = engine.resolve_or_create_entity("penguin").unwrap();
        let bird = engine.resolve_or_create_entity("bird").unwrap();
        engine
            .add_triple(&Triple::new(penguin, preds.is_a, bird))
            .unwrap();

        let flies = engine.resolve_or_create_relation("flies").unwrap();
        let yes = engine.resolve_or_create_entity("yes").unwrap();
        let no = engine.resolve_or_create_entity("no").unwrap();

        // bird flies yes (general rule)
        engine.add_triple(&Triple::new(bird, flies, yes)).unwrap();
        // penguin flies no (specific override)
        engine.add_triple(&Triple::new(penguin, flies, no)).unwrap();

        // Query: does penguin fly? Should find "no" because penguin is more specific
        let conflicts = find_hierarchy_conflicts(&engine, penguin, flies, &preds);
        assert_eq!(conflicts.len(), 2);

        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();
        assert_eq!(result.winner.object, no);
        assert_eq!(result.winner.subject, penguin);
        assert!(matches!(
            result.reason,
            OverrideReason::Specificity { depth_advantage: 1 }
        ));
    }

    #[test]
    fn monotonic_beats_default() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let animal = engine.resolve_or_create_entity("animal").unwrap();
        let cat = engine.resolve_or_create_entity("cat").unwrap();
        engine
            .add_triple(&Triple::new(cat, preds.is_a, animal))
            .unwrap();

        let breathes = engine.resolve_or_create_relation("breathes").unwrap();
        let yes = engine.resolve_or_create_entity("yes").unwrap();
        let no = engine.resolve_or_create_entity("no").unwrap();

        // animal breathes yes (monotonic)
        engine
            .add_triple(&Triple::new(animal, breathes, yes))
            .unwrap();
        mark_monotonic(&engine, animal, breathes, &preds).unwrap();

        // cat breathes no (default — should NOT override)
        engine
            .add_triple(&Triple::new(cat, breathes, no))
            .unwrap();

        let conflicts = find_hierarchy_conflicts(&engine, cat, breathes, &preds);
        assert_eq!(conflicts.len(), 2);

        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();
        // Monotonic wins even though cat is more specific
        assert_eq!(result.winner.subject, animal);
        assert_eq!(result.winner.object, yes);
        assert_eq!(result.reason, OverrideReason::Monotonic);
    }

    #[test]
    fn exception_registration() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let mammal = engine.resolve_or_create_entity("mammal").unwrap();
        let platypus = engine.resolve_or_create_entity("platypus").unwrap();
        engine
            .add_triple(&Triple::new(platypus, preds.is_a, mammal))
            .unwrap();

        let lays_eggs = engine.resolve_or_create_relation("lays-eggs").unwrap();
        let no = engine.resolve_or_create_entity("no").unwrap();
        let yes = engine.resolve_or_create_entity("yes").unwrap();

        engine
            .add_triple(&Triple::new(mammal, lays_eggs, no))
            .unwrap();
        engine
            .add_triple(&Triple::new(platypus, lays_eggs, yes))
            .unwrap();

        // Register platypus as exception to mammal
        register_exception(&engine, mammal, platypus, &preds).unwrap();

        let conflicts = find_hierarchy_conflicts(&engine, platypus, lays_eggs, &preds);
        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();

        // platypus wins by specificity (it's both more specific AND an exception)
        assert_eq!(result.winner.subject, platypus);
        assert_eq!(result.winner.object, yes);
    }

    #[test]
    fn recency_tiebreaker() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let city = engine.resolve_or_create_entity("city").unwrap();
        let mayor = engine.resolve_or_create_relation("mayor").unwrap();
        let alice = engine.resolve_or_create_entity("alice").unwrap();
        let bob = engine.resolve_or_create_entity("bob").unwrap();

        // Old: city mayor alice (timestamp 1000)
        let old_triple = Triple {
            subject: city,
            predicate: mayor,
            object: alice,
            confidence: 0.9,
            timestamp: 1000,
            provenance_id: None,
            compartment_id: None,
        };
        engine.add_triple(&old_triple).unwrap();

        // New: city mayor bob (timestamp 2000)
        let new_triple = Triple {
            subject: city,
            predicate: mayor,
            object: bob,
            confidence: 0.9,
            timestamp: 2000,
            provenance_id: None,
            compartment_id: None,
        };
        engine.add_triple(&new_triple).unwrap();

        let conflicts = find_conflicts(&engine, city, mayor);
        assert_eq!(conflicts.len(), 2);

        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();
        assert_eq!(result.winner.object, bob);
        assert_eq!(result.reason, OverrideReason::Recency);
    }

    #[test]
    fn confidence_final_tiebreaker() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let item = engine.resolve_or_create_entity("item").unwrap();
        let color = engine.resolve_or_create_relation("color").unwrap();
        let red = engine.resolve_or_create_entity("red").unwrap();
        let blue = engine.resolve_or_create_entity("blue").unwrap();

        // Same timestamp, same type depth, same monotonicity — confidence decides
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        engine
            .add_triple(&Triple {
                subject: item,
                predicate: color,
                object: red,
                confidence: 0.7,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })
            .unwrap();

        engine
            .add_triple(&Triple {
                subject: item,
                predicate: color,
                object: blue,
                confidence: 0.95,
                timestamp: now,
                provenance_id: None,
                compartment_id: None,
            })
            .unwrap();

        let conflicts = find_conflicts(&engine, item, color);
        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();
        assert_eq!(result.winner.object, blue);
        assert_eq!(result.reason, OverrideReason::Confidence);
    }

    #[test]
    fn no_conflict_returns_none() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        let entity = engine.resolve_or_create_entity("entity").unwrap();
        let pred = engine.resolve_or_create_relation("pred").unwrap();

        // No triples at all
        let result = query_defeasible(&engine, entity, pred, &preds);
        assert!(result.is_none());
    }

    #[test]
    fn multi_level_hierarchy_specificity() {
        let engine = test_engine();
        let preds = DefeasiblePredicates::resolve(&engine).unwrap();

        // animal → mammal → dog → golden_retriever
        let animal = engine.resolve_or_create_entity("animal").unwrap();
        let mammal = engine.resolve_or_create_entity("mammal").unwrap();
        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let golden = engine
            .resolve_or_create_entity("golden-retriever")
            .unwrap();

        engine
            .add_triple(&Triple::new(mammal, preds.is_a, animal))
            .unwrap();
        engine
            .add_triple(&Triple::new(dog, preds.is_a, mammal))
            .unwrap();
        engine
            .add_triple(&Triple::new(golden, preds.is_a, dog))
            .unwrap();

        let friendly = engine.resolve_or_create_relation("friendly").unwrap();
        let somewhat = engine.resolve_or_create_entity("somewhat").unwrap();
        let very = engine.resolve_or_create_entity("very").unwrap();

        // animal friendly somewhat
        engine
            .add_triple(&Triple::new(animal, friendly, somewhat))
            .unwrap();
        // golden-retriever friendly very
        engine
            .add_triple(&Triple::new(golden, friendly, very))
            .unwrap();

        let conflicts = find_hierarchy_conflicts(&engine, golden, friendly, &preds);
        assert_eq!(conflicts.len(), 2);

        let result = resolve_conflict(&engine, &conflicts, &preds).unwrap();
        assert_eq!(result.winner.subject, golden);
        assert_eq!(result.winner.object, very);
        assert!(matches!(result.reason, OverrideReason::Specificity { .. }));
    }

    #[test]
    fn type_depth_computation() {
        let engine = test_engine();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();

        let thing = engine.resolve_or_create_entity("thing").unwrap();
        let animal = engine.resolve_or_create_entity("animal").unwrap();
        let mammal = engine.resolve_or_create_entity("mammal").unwrap();
        let dog = engine.resolve_or_create_entity("dog").unwrap();

        engine
            .add_triple(&Triple::new(animal, is_a, thing))
            .unwrap();
        engine
            .add_triple(&Triple::new(mammal, is_a, animal))
            .unwrap();
        engine
            .add_triple(&Triple::new(dog, is_a, mammal))
            .unwrap();

        assert_eq!(type_depth(&engine, thing, is_a), 0);
        assert_eq!(type_depth(&engine, animal, is_a), 1);
        assert_eq!(type_depth(&engine, mammal, is_a), 2);
        assert_eq!(type_depth(&engine, dog, is_a), 3);
    }

    #[test]
    fn is_subtype_check() {
        let engine = test_engine();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();

        let animal = engine.resolve_or_create_entity("animal").unwrap();
        let mammal = engine.resolve_or_create_entity("mammal").unwrap();
        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let fish = engine.resolve_or_create_entity("fish").unwrap();

        engine
            .add_triple(&Triple::new(mammal, is_a, animal))
            .unwrap();
        engine
            .add_triple(&Triple::new(dog, is_a, mammal))
            .unwrap();
        engine
            .add_triple(&Triple::new(fish, is_a, animal))
            .unwrap();

        assert!(is_subtype_of(&engine, dog, animal, is_a));
        assert!(is_subtype_of(&engine, dog, mammal, is_a));
        assert!(!is_subtype_of(&engine, dog, fish, is_a));
        assert!(!is_subtype_of(&engine, animal, dog, is_a));
        assert!(is_subtype_of(&engine, dog, dog, is_a)); // reflexive
    }

    #[test]
    fn find_conflicts_empty_when_no_conflict() {
        let engine = test_engine();

        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let color = engine.resolve_or_create_relation("color").unwrap();
        let brown = engine.resolve_or_create_entity("brown").unwrap();

        engine
            .add_triple(&Triple::new(dog, color, brown))
            .unwrap();

        let conflicts = find_conflicts(&engine, dog, color);
        assert!(conflicts.is_empty()); // single triple = no conflict
    }

    #[test]
    fn override_reason_display() {
        assert_eq!(OverrideReason::Monotonic.to_string(), "monotonic");
        assert_eq!(
            OverrideReason::Specificity {
                depth_advantage: 2
            }
            .to_string(),
            "specificity (depth advantage: 2)"
        );
        assert_eq!(OverrideReason::Exception.to_string(), "exception");
        assert_eq!(OverrideReason::Recency.to_string(), "recency");
        assert_eq!(OverrideReason::Confidence.to_string(), "confidence");
    }
}
