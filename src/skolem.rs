//! Skolem functions: existential witness symbols.
//!
//! When the engine encounters an existential pattern (e.g., from a `relationAllExists`
//! rule macro), it creates a **Skolem symbol** — a placeholder entity representing
//! "the thing that exists." Skolem symbols are first-class `Entity` symbols that can
//! be queried and reified.
//!
//! When a concrete entity later satisfies the existential, the Skolem symbol is
//! **grounded** — linked to the real entity via provenance.
//!
//! ## Example
//!
//! Given `relationAllExists(owns, Person, Car)` and `Alice is-a Person`:
//! 1. Engine creates Skolem: `SK_owns_Alice` (a Car that Alice owns)
//! 2. Later, if we assert `Alice owns MyCar`, the Skolem is grounded to `MyCar`

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to Skolem function operations.
#[derive(Debug, Error, Diagnostic)]
pub enum SkolemError {
    #[error("Skolem symbol not found: {id}")]
    #[diagnostic(
        code(akh::skolem::not_found),
        help("No Skolem symbol with this ID is registered. Check the Skolem registry.")
    )]
    NotFound { id: u64 },

    #[error("Skolem symbol {id} is already grounded to {grounded_to}")]
    #[diagnostic(
        code(akh::skolem::already_grounded),
        help(
            "This Skolem symbol has already been resolved to a concrete entity. \
             If you need to change the grounding, unground it first."
        )
    )]
    AlreadyGrounded { id: u64, grounded_to: u64 },

    #[error("cannot create Skolem for non-existent existential relation {relation_id}")]
    #[diagnostic(
        code(akh::skolem::invalid_existential),
        help("The relation symbol used for the existential must exist in the knowledge graph.")
    )]
    InvalidExistential { relation_id: u64 },
}

/// Result type for Skolem operations.
pub type SkolemResult<T> = std::result::Result<T, SkolemError>;

// ---------------------------------------------------------------------------
// Skolem symbol
// ---------------------------------------------------------------------------

/// A Skolem symbol: a witness for an existential assertion.
///
/// Created when a rule macro (e.g., `relationAllExists`) implies existence
/// of an entity that hasn't been concretely identified yet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkolemSymbol {
    /// The SymbolId of this Skolem entity in the knowledge graph.
    pub id: SymbolId,
    /// The existential relation this Skolem witnesses (e.g., "owns").
    pub existential_relation: SymbolId,
    /// The bound variables — entities this Skolem is existentially tied to.
    /// For `relationAllExists(owns, Person, Car)` with `Alice is-a Person`,
    /// bound_vars = [Alice].
    pub bound_vars: Vec<SymbolId>,
    /// The expected type of the Skolem (e.g., Car).
    pub expected_type: Option<SymbolId>,
    /// If grounded, the concrete entity this Skolem was resolved to.
    pub grounded_to: Option<SymbolId>,
    /// Human-readable label (e.g., "SK_owns_Alice").
    pub label: String,
}

impl SkolemSymbol {
    /// Check if this Skolem has been grounded to a concrete entity.
    pub fn is_grounded(&self) -> bool {
        self.grounded_to.is_some()
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of Skolem symbols.
///
/// Tracks all active Skolem symbols and provides grounding operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkolemRegistry {
    /// Skolem SymbolId → SkolemSymbol data.
    skolems: HashMap<SymbolId, SkolemSymbol>,
    /// Index: (relation, bound_var) → Skolem SymbolId for fast lookup.
    /// Used to check if a Skolem already exists for a given existential.
    existential_index: HashMap<(SymbolId, SymbolId), SymbolId>,
}

impl SkolemRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a Skolem symbol for an existential.
    ///
    /// If a Skolem already exists for the same (relation, bound_var), returns it.
    pub fn create_skolem(
        &mut self,
        engine: &Engine,
        existential_relation: SymbolId,
        bound_var: SymbolId,
        expected_type: Option<SymbolId>,
    ) -> AkhResult<SkolemSymbol> {
        // Check if one already exists
        let key = (existential_relation, bound_var);
        if let Some(&existing_id) = self.existential_index.get(&key) {
            if let Some(existing) = self.skolems.get(&existing_id) {
                return Ok(existing.clone());
            }
        }

        // Create a new Skolem entity
        let rel_label = engine.resolve_label(existential_relation);
        let var_label = engine.resolve_label(bound_var);
        let label = format!("SK_{rel_label}_{var_label}");

        let meta = engine.create_symbol(crate::symbol::SymbolKind::Entity, &label)?;

        let skolem = SkolemSymbol {
            id: meta.id,
            existential_relation,
            bound_vars: vec![bound_var],
            expected_type,
            grounded_to: None,
            label,
        };

        self.skolems.insert(meta.id, skolem.clone());
        self.existential_index.insert(key, meta.id);

        Ok(skolem)
    }

    /// Ground a Skolem symbol to a concrete entity.
    ///
    /// This marks the Skolem as resolved — the concrete entity satisfies
    /// the existential that the Skolem witnessed.
    pub fn ground(
        &mut self,
        skolem_id: SymbolId,
        concrete_entity: SymbolId,
    ) -> SkolemResult<&SkolemSymbol> {
        let skolem = self.skolems.get_mut(&skolem_id).ok_or(SkolemError::NotFound {
            id: skolem_id.get(),
        })?;

        if let Some(existing) = skolem.grounded_to {
            return Err(SkolemError::AlreadyGrounded {
                id: skolem_id.get(),
                grounded_to: existing.get(),
            });
        }

        skolem.grounded_to = Some(concrete_entity);
        Ok(skolem)
    }

    /// Unground a Skolem (remove its concrete resolution).
    pub fn unground(&mut self, skolem_id: SymbolId) -> SkolemResult<()> {
        let skolem = self.skolems.get_mut(&skolem_id).ok_or(SkolemError::NotFound {
            id: skolem_id.get(),
        })?;
        skolem.grounded_to = None;
        Ok(())
    }

    /// Look up a Skolem by its ID.
    pub fn get(&self, skolem_id: SymbolId) -> Option<&SkolemSymbol> {
        self.skolems.get(&skolem_id)
    }

    /// Find the Skolem for a given (relation, bound_var) pair.
    pub fn find_for_existential(
        &self,
        relation: SymbolId,
        bound_var: SymbolId,
    ) -> Option<&SkolemSymbol> {
        let key = (relation, bound_var);
        self.existential_index
            .get(&key)
            .and_then(|id| self.skolems.get(id))
    }

    /// Check grounding: given a relation and bound variable, check if any
    /// concrete triple satisfies the Skolem's existential.
    ///
    /// Returns the concrete entity if found, or None.
    pub fn check_grounding(
        &self,
        engine: &Engine,
        skolem_id: SymbolId,
    ) -> Option<SymbolId> {
        let skolem = self.skolems.get(&skolem_id)?;
        if skolem.is_grounded() {
            return skolem.grounded_to;
        }

        let kg = engine.knowledge_graph();
        // Check if any of the bound vars have the existential relation
        for &bound_var in &skolem.bound_vars {
            let triples = kg.triples_from(bound_var);
            for t in triples {
                if t.predicate == skolem.existential_relation {
                    return Some(t.object);
                }
            }
        }

        None
    }

    /// Auto-ground all ungrounded Skolems by checking the KG.
    ///
    /// Returns the number of newly grounded Skolems.
    pub fn auto_ground(&mut self, engine: &Engine) -> usize {
        let mut to_ground: Vec<(SymbolId, SymbolId)> = Vec::new();

        for (id, skolem) in &self.skolems {
            if skolem.is_grounded() {
                continue;
            }
            let kg = engine.knowledge_graph();
            for &bound_var in &skolem.bound_vars {
                let triples = kg.triples_from(bound_var);
                for t in triples {
                    if t.predicate == skolem.existential_relation {
                        to_ground.push((*id, t.object));
                        break;
                    }
                }
            }
        }

        let mut count = 0;
        for (skolem_id, entity) in to_ground {
            if self.ground(skolem_id, entity).is_ok() {
                count += 1;
            }
        }
        count
    }

    /// List all Skolem symbols.
    pub fn all_skolems(&self) -> Vec<&SkolemSymbol> {
        self.skolems.values().collect()
    }

    /// List all ungrounded Skolems.
    pub fn ungrounded(&self) -> Vec<&SkolemSymbol> {
        self.skolems
            .values()
            .filter(|s| !s.is_grounded())
            .collect()
    }

    /// Number of registered Skolems.
    pub fn len(&self) -> usize {
        self.skolems.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.skolems.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::symbol::SymbolKind;

    #[test]
    fn create_skolem_basic() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let car = engine.create_symbol(SymbolKind::Entity, "Car").unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, Some(car.id))
            .unwrap();

        assert!(!skolem.is_grounded());
        assert_eq!(skolem.existential_relation, owns.id);
        assert_eq!(skolem.bound_vars, vec![alice.id]);
        assert_eq!(skolem.expected_type, Some(car.id));
        assert!(skolem.label.contains("owns"));
        assert!(skolem.label.contains("Alice"));
    }

    #[test]
    fn create_skolem_deduplication() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();

        let mut registry = SkolemRegistry::new();
        let s1 = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();
        let s2 = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();

        assert_eq!(s1.id, s2.id);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn ground_skolem() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let my_car = engine.create_symbol(SymbolKind::Entity, "MyCar").unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();
        let skolem_id = skolem.id;

        registry.ground(skolem_id, my_car.id).unwrap();

        let grounded = registry.get(skolem_id).unwrap();
        assert!(grounded.is_grounded());
        assert_eq!(grounded.grounded_to, Some(my_car.id));
    }

    #[test]
    fn double_grounding_fails() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let car1 = engine.create_symbol(SymbolKind::Entity, "Car1").unwrap();
        let car2 = engine.create_symbol(SymbolKind::Entity, "Car2").unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();
        let skolem_id = skolem.id;

        registry.ground(skolem_id, car1.id).unwrap();
        let err = registry.ground(skolem_id, car2.id).unwrap_err();
        assert!(matches!(err, SkolemError::AlreadyGrounded { .. }));
    }

    #[test]
    fn unground_and_reground() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let car1 = engine.create_symbol(SymbolKind::Entity, "Car1").unwrap();
        let car2 = engine.create_symbol(SymbolKind::Entity, "Car2").unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();
        let skolem_id = skolem.id;

        registry.ground(skolem_id, car1.id).unwrap();
        registry.unground(skolem_id).unwrap();
        registry.ground(skolem_id, car2.id).unwrap();

        assert_eq!(
            registry.get(skolem_id).unwrap().grounded_to,
            Some(car2.id)
        );
    }

    #[test]
    fn check_grounding_from_kg() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let my_car = engine.create_symbol(SymbolKind::Entity, "MyCar").unwrap();

        // Add the triple that satisfies the existential
        engine
            .add_triple(&Triple::new(alice.id, owns.id, my_car.id))
            .unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();

        let found = registry.check_grounding(&engine, skolem.id);
        assert_eq!(found, Some(my_car.id));
    }

    #[test]
    fn auto_ground_resolves_skolems() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();
        let my_car = engine.create_symbol(SymbolKind::Entity, "MyCar").unwrap();

        let mut registry = SkolemRegistry::new();
        registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();

        assert_eq!(registry.ungrounded().len(), 1);

        // Add the satisfying triple
        engine
            .add_triple(&Triple::new(alice.id, owns.id, my_car.id))
            .unwrap();

        let grounded = registry.auto_ground(&engine);
        assert_eq!(grounded, 1);
        assert_eq!(registry.ungrounded().len(), 0);
    }

    #[test]
    fn find_for_existential() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let owns = engine.create_symbol(SymbolKind::Relation, "owns").unwrap();
        let alice = engine.create_symbol(SymbolKind::Entity, "Alice").unwrap();

        let mut registry = SkolemRegistry::new();
        let skolem = registry
            .create_skolem(&engine, owns.id, alice.id, None)
            .unwrap();

        let found = registry.find_for_existential(owns.id, alice.id).unwrap();
        assert_eq!(found.id, skolem.id);

        assert!(registry
            .find_for_existential(alice.id, owns.id)
            .is_none());
    }

    #[test]
    fn not_found_error() {
        let mut registry = SkolemRegistry::new();
        let fake_id = SymbolId::new(9999).unwrap();
        let err = registry.ground(fake_id, fake_id).unwrap_err();
        assert!(matches!(err, SkolemError::NotFound { .. }));
    }
}
