//! Predicate hierarchy: `genlPreds` subsumption and `genlInverse` inference.
//!
//! Inspired by Cyc's `genlPreds` (predicate generalization) and `genlInverse`
//! (argument-swapped equivalence). When `biologicalMother` generalizes to `parent`
//! via `rel:generalizes`, a query for `parent(X, Y)` will also find
//! `biologicalMother(X, Y)`. When `parent` has inverse `child`, a query for
//! `child(Y, X)` checks `parent(X, Y)`.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for the predicate hierarchy (prefixed `rel:`).
#[derive(Debug, Clone)]
pub struct HierarchyPredicates {
    /// `rel:generalizes` — predicate subsumption (genlPreds).
    /// `(specific rel:generalizes general)` means specific is a specialization of general.
    pub generalizes: SymbolId,
    /// `rel:inverse` — argument-swapped equivalence (genlInverse).
    /// `(P rel:inverse Q)` means P(X, Y) ↔ Q(Y, X).
    pub inverse: SymbolId,
}

impl HierarchyPredicates {
    /// Resolve all `rel:` hierarchy predicates, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            generalizes: engine.resolve_or_create_relation("rel:generalizes")?,
            inverse: engine.resolve_or_create_relation("rel:inverse")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Predicate hierarchy
// ---------------------------------------------------------------------------

/// Cached predicate subsumption lattice and inverse mappings.
///
/// Built from `rel:generalizes` and `rel:inverse` triples in the KG.
/// The `generalizes` map stores the transitive closure: for predicate P,
/// `generalizes[P]` contains all predicates more specific than P (whose
/// triples should be included when querying P).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredicateHierarchy {
    /// For predicate P, the set of more specific predicates Q where
    /// `genlPreds(Q, P)` holds (transitively). Querying P should also
    /// search all of these.
    specializations: HashMap<SymbolId, Vec<SymbolId>>,
    /// For predicate P, the set of more general predicates G where
    /// `genlPreds(P, G)` holds (transitively). Useful for subsumption checks.
    generalizations: HashMap<SymbolId, Vec<SymbolId>>,
    /// Inverse pairs: P ↔ Q means P(X,Y) ↔ Q(Y,X).
    inverses: HashMap<SymbolId, SymbolId>,
}

impl PredicateHierarchy {
    /// Create a new empty hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the hierarchy from the knowledge graph's `rel:generalizes` and `rel:inverse` triples.
    pub fn build(engine: &Engine) -> AkhResult<Self> {
        let preds = HierarchyPredicates::resolve(engine)?;
        let kg = engine.knowledge_graph();

        // Collect direct generalization edges: (specific, general)
        let direct_edges: Vec<(SymbolId, SymbolId)> =
            kg.triples_for_predicate(preds.generalizes);

        // Build adjacency lists for both directions
        let mut children_of: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new(); // general → specifics
        let mut parents_of: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new(); // specific → generals

        for &(specific, general) in &direct_edges {
            children_of.entry(general).or_default().push(specific);
            parents_of.entry(specific).or_default().push(general);
        }

        // Compute transitive closure of specializations (general → all specifics below)
        let mut specializations: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
        let all_generals: HashSet<SymbolId> = children_of.keys().copied().collect();
        let all_specifics: HashSet<SymbolId> = parents_of.keys().copied().collect();
        let all_preds: HashSet<SymbolId> = all_generals.union(&all_specifics).copied().collect();

        for &pred in &all_preds {
            let mut specs = Vec::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            // BFS down from pred to find all transitive specializations
            if let Some(children) = children_of.get(&pred) {
                for &child in children {
                    if visited.insert(child) {
                        queue.push_back(child);
                    }
                }
            }
            while let Some(current) = queue.pop_front() {
                specs.push(current);
                if let Some(children) = children_of.get(&current) {
                    for &child in children {
                        if visited.insert(child) {
                            queue.push_back(child);
                        }
                    }
                }
            }

            if !specs.is_empty() {
                specializations.insert(pred, specs);
            }
        }

        // Compute transitive closure of generalizations (specific → all generals above)
        let mut generalizations: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();

        for &pred in &all_preds {
            let mut gens = Vec::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            if let Some(parents) = parents_of.get(&pred) {
                for &parent in parents {
                    if visited.insert(parent) {
                        queue.push_back(parent);
                    }
                }
            }
            while let Some(current) = queue.pop_front() {
                gens.push(current);
                if let Some(parents) = parents_of.get(&current) {
                    for &parent in parents {
                        if visited.insert(parent) {
                            queue.push_back(parent);
                        }
                    }
                }
            }

            if !gens.is_empty() {
                generalizations.insert(pred, gens);
            }
        }

        // Collect inverse pairs
        let inverse_edges: Vec<(SymbolId, SymbolId)> = kg.triples_for_predicate(preds.inverse);
        let mut inverses = HashMap::new();
        for (p, q) in inverse_edges {
            inverses.insert(p, q);
            inverses.insert(q, p); // symmetric
        }

        Ok(Self {
            specializations,
            generalizations,
            inverses,
        })
    }

    /// Get all predicates more specific than the given predicate (transitive).
    ///
    /// When querying for triples with predicate P, also search for triples
    /// with any of these predicates (they are all specializations of P).
    pub fn specializations_of(&self, predicate: SymbolId) -> &[SymbolId] {
        self.specializations
            .get(&predicate)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all predicates more general than the given predicate (transitive).
    pub fn generalizations_of(&self, predicate: SymbolId) -> &[SymbolId] {
        self.generalizations
            .get(&predicate)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the inverse predicate, if declared.
    pub fn inverse_of(&self, predicate: SymbolId) -> Option<SymbolId> {
        self.inverses.get(&predicate).copied()
    }

    /// Check if `specific` is a (transitive) specialization of `general`.
    pub fn is_specialization_of(&self, specific: SymbolId, general: SymbolId) -> bool {
        self.specializations_of(general).contains(&specific)
    }

    /// Check if there is any hierarchy data at all.
    pub fn is_empty(&self) -> bool {
        self.specializations.is_empty() && self.inverses.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Hierarchy-aware queries
// ---------------------------------------------------------------------------

/// Query objects using predicate hierarchy: returns results for the given
/// predicate AND all of its specializations.
///
/// Example: querying `parent(dog, ?)` will also return results from
/// `biologicalMother(dog, ?)` if `biologicalMother` generalizes to `parent`.
pub fn objects_with_hierarchy(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    hierarchy: &PredicateHierarchy,
) -> Vec<(SymbolId, SymbolId)> {
    let kg = engine.knowledge_graph();
    let mut results: Vec<(SymbolId, SymbolId)> = Vec::new();

    // Direct results for the queried predicate
    for obj in kg.objects_of(subject, predicate) {
        results.push((predicate, obj));
    }

    // Results from more specific predicates
    for &spec_pred in hierarchy.specializations_of(predicate) {
        for obj in kg.objects_of(subject, spec_pred) {
            results.push((spec_pred, obj));
        }
    }

    results
}

/// Query objects using predicate hierarchy AND inverse inference.
///
/// If `parent` has inverse `child`, querying `parent(X, ?)` also checks
/// `child(?, X)` — returning subjects of the inverse predicate as objects.
pub fn objects_with_hierarchy_and_inverse(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    hierarchy: &PredicateHierarchy,
) -> Vec<(SymbolId, SymbolId)> {
    let kg = engine.knowledge_graph();
    let mut results = objects_with_hierarchy(engine, subject, predicate, hierarchy);

    // Check inverse: if P has inverse Q, then P(X, Y) ↔ Q(Y, X)
    // So querying P(subject, ?) also returns Y where Q(Y, subject) exists.
    if let Some(inv_pred) = hierarchy.inverse_of(predicate) {
        for subj in kg.subjects_of(inv_pred, subject) {
            results.push((inv_pred, subj));
        }
        // Also check inverses of specializations
        for &spec_pred in hierarchy.specializations_of(predicate) {
            if let Some(spec_inv) = hierarchy.inverse_of(spec_pred) {
                for subj in kg.subjects_of(spec_inv, subject) {
                    results.push((spec_inv, subj));
                }
            }
        }
    }

    results
}

/// Full hierarchy-aware triple query returning complete Triple structs.
///
/// Searches the queried predicate, all specializations, and inverse predicates.
/// Each result carries `DerivationKind::PredicateGeneralization` or
/// `DerivationKind::PredicateInverse` provenance metadata (via the returned
/// `HierarchyMatch` wrapper).
pub fn triples_with_hierarchy(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    hierarchy: &PredicateHierarchy,
) -> Vec<HierarchyMatch> {
    let kg = engine.knowledge_graph();
    let mut results = Vec::new();

    // Direct matches
    for triple in kg.triples_from(subject) {
        if triple.predicate == predicate {
            results.push(HierarchyMatch {
                triple,
                via: MatchVia::Direct,
            });
        }
    }

    // Specialization matches
    for &spec_pred in hierarchy.specializations_of(predicate) {
        for triple in kg.triples_from(subject) {
            if triple.predicate == spec_pred {
                results.push(HierarchyMatch {
                    triple,
                    via: MatchVia::Specialization {
                        actual_predicate: spec_pred,
                        queried_predicate: predicate,
                    },
                });
            }
        }
    }

    // Inverse matches
    if let Some(inv_pred) = hierarchy.inverse_of(predicate) {
        for triple in kg.triples_to(subject) {
            if triple.predicate == inv_pred {
                // Swap subject/object for the result since we're using the inverse
                results.push(HierarchyMatch {
                    triple: Triple::new(subject, predicate, triple.subject)
                        .with_confidence(triple.confidence),
                    via: MatchVia::Inverse {
                        actual_predicate: inv_pred,
                        queried_predicate: predicate,
                    },
                });
            }
        }
    }

    results
}

/// A triple match annotated with how it was found through the hierarchy.
#[derive(Debug, Clone)]
pub struct HierarchyMatch {
    /// The matched triple (may have swapped subject/object for inverse matches).
    pub triple: Triple,
    /// How this match was found.
    pub via: MatchVia,
}

/// How a triple was matched through the predicate hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchVia {
    /// Direct match on the queried predicate.
    Direct,
    /// Matched via a more specific predicate.
    Specialization {
        actual_predicate: SymbolId,
        queried_predicate: SymbolId,
    },
    /// Matched via an inverse predicate (subject/object swapped).
    Inverse {
        actual_predicate: SymbolId,
        queried_predicate: SymbolId,
    },
}

// ---------------------------------------------------------------------------
// Provenance integration
// ---------------------------------------------------------------------------

/// New derivation kinds for predicate hierarchy inference.
/// These are added to `DerivationKind` in provenance.rs.
///
/// - `PredicateGeneralization { specific, general }` — triple inferred via genlPreds
/// - `PredicateInverse { predicate, inverse }` — triple inferred via genlInverse

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
    fn empty_hierarchy() {
        let engine = test_engine();
        let hierarchy = PredicateHierarchy::build(&engine).unwrap();
        assert!(hierarchy.is_empty());
    }

    #[test]
    fn linear_generalization_chain() {
        let engine = test_engine();
        let rel_gen = engine
            .resolve_or_create_relation("rel:generalizes")
            .unwrap();

        // biologicalMother → mother → parent
        let bio_mother = engine
            .resolve_or_create_relation("biologicalMother")
            .unwrap();
        let mother = engine.resolve_or_create_relation("mother").unwrap();
        let parent = engine.resolve_or_create_relation("parent").unwrap();

        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, mother))
            .unwrap();
        engine
            .add_triple(&Triple::new(mother, rel_gen, parent))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        // parent has specializations: mother, biologicalMother
        let specs = hierarchy.specializations_of(parent);
        assert_eq!(specs.len(), 2);
        assert!(specs.contains(&mother));
        assert!(specs.contains(&bio_mother));

        // mother has specialization: biologicalMother
        let specs = hierarchy.specializations_of(mother);
        assert_eq!(specs.len(), 1);
        assert!(specs.contains(&bio_mother));

        // biologicalMother has no specializations
        assert!(hierarchy.specializations_of(bio_mother).is_empty());

        // Check generalizations
        let gens = hierarchy.generalizations_of(bio_mother);
        assert_eq!(gens.len(), 2);
        assert!(gens.contains(&mother));
        assert!(gens.contains(&parent));

        // Subsumption check
        assert!(hierarchy.is_specialization_of(bio_mother, parent));
        assert!(hierarchy.is_specialization_of(mother, parent));
        assert!(!hierarchy.is_specialization_of(parent, bio_mother));
    }

    #[test]
    fn inverse_predicates() {
        let engine = test_engine();
        let rel_inv = engine.resolve_or_create_relation("rel:inverse").unwrap();

        let parent = engine.resolve_or_create_relation("parent-of").unwrap();
        let child = engine.resolve_or_create_relation("child-of").unwrap();

        engine
            .add_triple(&Triple::new(parent, rel_inv, child))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        assert_eq!(hierarchy.inverse_of(parent), Some(child));
        assert_eq!(hierarchy.inverse_of(child), Some(parent)); // symmetric
    }

    #[test]
    fn hierarchy_aware_query() {
        let engine = test_engine();
        let rel_gen = engine
            .resolve_or_create_relation("rel:generalizes")
            .unwrap();

        let bio_mother = engine
            .resolve_or_create_relation("biologicalMother")
            .unwrap();
        let parent = engine.resolve_or_create_relation("parent").unwrap();
        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, parent))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        // Assert: alice biologicalMother bob
        let alice = engine.resolve_or_create_entity("alice").unwrap();
        let bob = engine.resolve_or_create_entity("bob").unwrap();
        engine
            .add_triple(&Triple::new(alice, bio_mother, bob))
            .unwrap();

        // Query: parent(alice, ?) should find bob via hierarchy
        let results = objects_with_hierarchy(&engine, alice, parent, &hierarchy);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (bio_mother, bob));
    }

    #[test]
    fn inverse_query() {
        let engine = test_engine();
        let rel_inv = engine.resolve_or_create_relation("rel:inverse").unwrap();

        let parent = engine.resolve_or_create_relation("parent-of").unwrap();
        let child = engine.resolve_or_create_relation("child-of").unwrap();
        engine
            .add_triple(&Triple::new(parent, rel_inv, child))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        // Assert: alice parent-of bob
        let alice = engine.resolve_or_create_entity("alice").unwrap();
        let bob = engine.resolve_or_create_entity("bob").unwrap();
        engine
            .add_triple(&Triple::new(alice, parent, bob))
            .unwrap();

        // Query: child-of(bob, ?) should find alice via inverse
        let results =
            objects_with_hierarchy_and_inverse(&engine, bob, child, &hierarchy);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (parent, alice));
    }

    #[test]
    fn hierarchy_match_types() {
        let engine = test_engine();
        let rel_gen = engine
            .resolve_or_create_relation("rel:generalizes")
            .unwrap();
        let rel_inv = engine.resolve_or_create_relation("rel:inverse").unwrap();

        let bio_mother = engine
            .resolve_or_create_relation("biologicalMother")
            .unwrap();
        let parent = engine.resolve_or_create_relation("parent").unwrap();
        let child = engine.resolve_or_create_relation("child").unwrap();

        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, parent))
            .unwrap();
        engine
            .add_triple(&Triple::new(parent, rel_inv, child))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        let alice = engine.resolve_or_create_entity("alice").unwrap();
        let bob = engine.resolve_or_create_entity("bob").unwrap();

        // Direct: alice parent bob
        engine
            .add_triple(&Triple::new(alice, parent, bob))
            .unwrap();
        // Specialization: alice biologicalMother bob
        engine
            .add_triple(&Triple::new(alice, bio_mother, bob))
            .unwrap();

        let matches = triples_with_hierarchy(&engine, alice, parent, &hierarchy);

        // Should have: 1 direct + 1 specialization + 0 inverse (no child(?, alice) triples)
        let direct_count = matches.iter().filter(|m| m.via == MatchVia::Direct).count();
        let spec_count = matches
            .iter()
            .filter(|m| matches!(m.via, MatchVia::Specialization { .. }))
            .count();

        assert_eq!(direct_count, 1);
        assert_eq!(spec_count, 1);
    }

    #[test]
    fn diamond_hierarchy() {
        let engine = test_engine();
        let rel_gen = engine
            .resolve_or_create_relation("rel:generalizes")
            .unwrap();

        // Diamond: bio_mother → mother, bio_mother → female_parent, mother → parent, female_parent → parent
        let bio_mother = engine
            .resolve_or_create_relation("bio-mother")
            .unwrap();
        let mother = engine.resolve_or_create_relation("mother").unwrap();
        let female_parent = engine
            .resolve_or_create_relation("female-parent")
            .unwrap();
        let parent = engine.resolve_or_create_relation("parent").unwrap();

        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, mother))
            .unwrap();
        engine
            .add_triple(&Triple::new(bio_mother, rel_gen, female_parent))
            .unwrap();
        engine
            .add_triple(&Triple::new(mother, rel_gen, parent))
            .unwrap();
        engine
            .add_triple(&Triple::new(female_parent, rel_gen, parent))
            .unwrap();

        let hierarchy = PredicateHierarchy::build(&engine).unwrap();

        // parent should have 3 specializations: mother, female_parent, bio_mother
        let specs = hierarchy.specializations_of(parent);
        assert_eq!(specs.len(), 3);
        assert!(specs.contains(&mother));
        assert!(specs.contains(&female_parent));
        assert!(specs.contains(&bio_mother));
    }
}
