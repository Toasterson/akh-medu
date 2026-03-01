//! Rule macro predicates: compact meta-predicates that expand to quantified patterns.
//!
//! Inspired by Cyc's rule macros (`relationAllExists`, `relationExistsAll`, `genls`).
//! A rule macro is a compact assertion like `relationAllExists(owns, Person, Car)`
//! meaning "every Person owns at least one Car." Instead of materializing all
//! individual triples, the macro registers a virtual reasoner that can answer
//! queries matching its pattern without expansion.
//!
//! ## Built-in macros
//!
//! - **Genls** — `genls(C1, C2)`: every instance of C1 is also an instance of C2
//! - **RelationAllExists** — every domain instance has at least one range-linked instance
//! - **RelationExistsAll** — some domain instance is range-linked to every range instance

use std::collections::HashMap;
use std::fmt;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to rule macro operations.
#[derive(Debug, Error, Diagnostic)]
pub enum RuleMacroError {
    #[error("rule macro not found: {name}")]
    #[diagnostic(
        code(akh::rule_macro::not_found),
        help("No rule macro with this name is registered. Check available macros with the registry.")
    )]
    NotFound { name: String },

    #[error("rule macro expansion failed for {macro_name}: {reason}")]
    #[diagnostic(
        code(akh::rule_macro::expansion_failed),
        help("The rule macro could not expand the given triple. Check that the triple matches the macro's pattern.")
    )]
    ExpansionFailed { macro_name: String, reason: String },

    #[error("duplicate rule macro registration: {name}")]
    #[diagnostic(
        code(akh::rule_macro::duplicate),
        help("A rule macro with this name is already registered. Use a different name or remove the existing one first.")
    )]
    Duplicate { name: String },
}

/// Result type for rule macro operations.
pub type RuleMacroResult<T> = std::result::Result<T, RuleMacroError>;

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for rule macros (prefixed `macro:`).
#[derive(Debug, Clone)]
pub struct MacroPredicates {
    /// `macro:genls` — every instance of subject is also instance of object.
    pub genls: SymbolId,
    /// `macro:relationAllExists` — every domain instance has a relation-linked range instance.
    pub relation_all_exists: SymbolId,
    /// `macro:relationExistsAll` — some domain instance is relation-linked to every range instance.
    pub relation_exists_all: SymbolId,
    /// `is-a` — instance-of relation used by macros.
    pub is_a: SymbolId,
}

impl MacroPredicates {
    /// Resolve all `macro:` predicates, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            genls: engine.resolve_or_create_relation("macro:genls")?,
            relation_all_exists: engine
                .resolve_or_create_relation("macro:relationAllExists")?,
            relation_exists_all: engine
                .resolve_or_create_relation("macro:relationExistsAll")?,
            is_a: engine.resolve_or_create_relation("is-a")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// A rule macro that can expand compact assertions into patterns and answer queries.
pub trait RuleMacro: Send + Sync + fmt::Debug {
    /// Unique name of this macro kind.
    fn name(&self) -> &str;

    /// Check if a triple is a macro invocation (uses this macro's predicate).
    fn matches(&self, triple: &Triple) -> bool;

    /// Expand the macro into concrete triples that would need to hold.
    ///
    /// For `genls(Dog, Animal)` this produces individual `(X, is-a, Animal)` triples
    /// for every known Dog instance.
    fn expand(&self, engine: &Engine) -> RuleMacroResult<Vec<Triple>>;

    /// Check if this macro can answer a query `(subject, predicate, ?)`.
    fn can_answer(&self, subject: SymbolId, predicate: SymbolId) -> bool;

    /// Answer a query without materializing full expansion.
    ///
    /// Returns triples that the macro can virtually derive for the query.
    fn answer(
        &self,
        engine: &Engine,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> RuleMacroResult<Vec<Triple>>;
}

// ---------------------------------------------------------------------------
// Macro invocation record
// ---------------------------------------------------------------------------

/// A registered macro invocation: a specific assertion that uses a macro.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInvocation {
    /// The macro kind name (e.g., "genls", "relationAllExists").
    pub macro_name: String,
    /// The predicate symbol used by this invocation.
    pub macro_predicate: SymbolId,
    /// Subject of the macro triple.
    pub subject: SymbolId,
    /// Object of the macro triple.
    pub object: SymbolId,
    /// Optional: the relation parameter (for RelationAllExists/RelationExistsAll).
    pub relation: Option<SymbolId>,
}

// ---------------------------------------------------------------------------
// Built-in: Genls
// ---------------------------------------------------------------------------

/// `genls(C1, C2)` — every instance of C1 is also an instance of C2.
///
/// Stored as `(C1, macro:genls, C2)`. When queried for `(X, is-a, C2)`,
/// checks if X is an instance of C1 (or any sub-collection of C1).
#[derive(Debug)]
pub struct GenlsMacro {
    /// The child collection.
    pub child: SymbolId,
    /// The parent collection.
    pub parent: SymbolId,
    /// The `macro:genls` predicate.
    pub genls_pred: SymbolId,
    /// The `is-a` predicate for expansion.
    pub is_a: SymbolId,
}

impl RuleMacro for GenlsMacro {
    fn name(&self) -> &str {
        "genls"
    }

    fn matches(&self, triple: &Triple) -> bool {
        triple.predicate == self.genls_pred
            && triple.subject == self.child
            && triple.object == self.parent
    }

    fn expand(&self, engine: &Engine) -> RuleMacroResult<Vec<Triple>> {
        // Find all instances of child: (X, is-a, child)
        let kg = engine.knowledge_graph();
        let instances = kg.triples_to(self.child)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>();
        // For each child instance, produce (X, is-a, parent)
        Ok(instances
            .into_iter()
            .map(|t| Triple::new(t.subject, self.is_a, self.parent))
            .collect())
    }

    fn can_answer(&self, _subject: SymbolId, predicate: SymbolId) -> bool {
        predicate == self.is_a
    }

    fn answer(
        &self,
        engine: &Engine,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> RuleMacroResult<Vec<Triple>> {
        if predicate != self.is_a {
            return Ok(vec![]);
        }
        // Check: is subject an instance of child? If so, it's also an instance of parent.
        let kg = engine.knowledge_graph();
        let is_child_instance = kg
            .triples_to(self.child)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>()
            .iter()
            .any(|t| t.subject == subject);
        if is_child_instance {
            Ok(vec![Triple::new(subject, self.is_a, self.parent)])
        } else {
            Ok(vec![])
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in: RelationAllExists
// ---------------------------------------------------------------------------

/// `relationAllExists(R, Domain, Range)` — every Domain instance has at least one
/// R-linked Range instance.
///
/// Stored as `(Domain, macro:relationAllExists, Range)` with metadata relation R.
/// When queried for `(X, R, ?)` where X is a Domain instance, checks if
/// X already has an R-linked range instance.
#[derive(Debug)]
pub struct RelationAllExistsMacro {
    /// The relation that all domain instances must have.
    pub relation: SymbolId,
    /// The domain collection.
    pub domain: SymbolId,
    /// The range collection.
    pub range: SymbolId,
    /// The `macro:relationAllExists` predicate.
    pub macro_pred: SymbolId,
    /// The `is-a` predicate.
    pub is_a: SymbolId,
}

impl RuleMacro for RelationAllExistsMacro {
    fn name(&self) -> &str {
        "relationAllExists"
    }

    fn matches(&self, triple: &Triple) -> bool {
        triple.predicate == self.macro_pred
            && triple.subject == self.domain
            && triple.object == self.range
    }

    fn expand(&self, engine: &Engine) -> RuleMacroResult<Vec<Triple>> {
        let kg = engine.knowledge_graph();
        // Find all domain instances
        let domain_instances = kg.triples_to(self.domain)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>();
        let mut result = Vec::new();
        for inst in &domain_instances {
            // Check if this instance already has the relation
            let existing: Vec<_> = kg
                .triples_from(inst.subject)
                .into_iter()
                .filter(|t| t.predicate == self.relation)
                .collect();
            if existing.is_empty() {
                // Flag: this instance lacks the required relation — but we don't
                // create concrete triples since we don't know the specific range
                // entity. This would be where Skolem functions (Phase 9h) step in.
                // For now, produce a placeholder triple with the range *collection*.
                result.push(Triple::new(inst.subject, self.relation, self.range));
            }
        }
        Ok(result)
    }

    fn can_answer(&self, _subject: SymbolId, predicate: SymbolId) -> bool {
        predicate == self.relation
    }

    fn answer(
        &self,
        engine: &Engine,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> RuleMacroResult<Vec<Triple>> {
        if predicate != self.relation {
            return Ok(vec![]);
        }
        // Check: is subject an instance of domain?
        let kg = engine.knowledge_graph();
        let is_domain_instance = kg
            .triples_to(self.domain)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>()
            .iter()
            .any(|t| t.subject == subject);
        if !is_domain_instance {
            return Ok(vec![]);
        }
        // Find existing relation triples for this subject
        let existing: Vec<_> = kg
            .triples_from(subject)
            .into_iter()
            .filter(|t| t.predicate == self.relation)
            .collect();
        if existing.is_empty() {
            // The macro asserts this should exist but no concrete triple does.
            Ok(vec![Triple::new(subject, self.relation, self.range)])
        } else {
            Ok(existing)
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in: RelationExistsAll
// ---------------------------------------------------------------------------

/// `relationExistsAll(R, Domain, Range)` — there exists a Domain instance that is
/// R-linked to every Range instance.
///
/// This is the dual of `relationAllExists`: at least one domain entity must be
/// linked to *every* range entity.
#[derive(Debug)]
pub struct RelationExistsAllMacro {
    /// The relation.
    pub relation: SymbolId,
    /// The domain collection.
    pub domain: SymbolId,
    /// The range collection.
    pub range: SymbolId,
    /// The `macro:relationExistsAll` predicate.
    pub macro_pred: SymbolId,
    /// The `is-a` predicate.
    pub is_a: SymbolId,
}

impl RuleMacro for RelationExistsAllMacro {
    fn name(&self) -> &str {
        "relationExistsAll"
    }

    fn matches(&self, triple: &Triple) -> bool {
        triple.predicate == self.macro_pred
            && triple.subject == self.domain
            && triple.object == self.range
    }

    fn expand(&self, engine: &Engine) -> RuleMacroResult<Vec<Triple>> {
        let kg = engine.knowledge_graph();
        // Find all range instances
        let range_instances = kg.triples_to(self.range)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>();
        // Find a domain instance linked to all of them
        let domain_instances = kg.triples_to(self.domain)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>();
        let mut best_domain: Option<(SymbolId, usize)> = None;

        for inst in &domain_instances {
            let count = range_instances
                .iter()
                .filter(|r| {
                    kg.triples_from(inst.subject)
                        .iter()
                        .any(|t| t.predicate == self.relation && t.object == r.subject)
                })
                .count();
            if let Some((_, best_count)) = best_domain {
                if count > best_count {
                    best_domain = Some((inst.subject, count));
                }
            } else {
                best_domain = Some((inst.subject, count));
            }
        }

        // Produce virtual triples for the best-covering domain instance
        if let Some((domain_sym, _)) = best_domain {
            Ok(range_instances
                .into_iter()
                .map(|r| Triple::new(domain_sym, self.relation, r.subject))
                .collect())
        } else {
            Ok(vec![])
        }
    }

    fn can_answer(&self, _subject: SymbolId, predicate: SymbolId) -> bool {
        predicate == self.relation
    }

    fn answer(
        &self,
        engine: &Engine,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> RuleMacroResult<Vec<Triple>> {
        if predicate != self.relation {
            return Ok(vec![]);
        }
        // Check: is subject a domain instance?
        let kg = engine.knowledge_graph();
        let is_domain_instance = kg
            .triples_to(self.domain)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>()
            .iter()
            .any(|t| t.subject == subject);
        if !is_domain_instance {
            return Ok(vec![]);
        }
        // Return all range instances as relation targets
        let range_instances = kg.triples_to(self.range)
            .into_iter()
            .filter(|t| t.predicate == self.is_a)
            .collect::<Vec<_>>();
        Ok(range_instances
            .into_iter()
            .map(|r| Triple::new(subject, self.relation, r.subject))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of active rule macros.
///
/// Holds all macro instances and provides query answering across all macros.
#[derive(Debug, Default)]
pub struct RuleMacroRegistry {
    macros: Vec<Box<dyn RuleMacro>>,
    /// Index: macro name → position in the macros vec.
    name_index: HashMap<String, usize>,
}

impl RuleMacroRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new rule macro.
    pub fn register(&mut self, rule_macro: Box<dyn RuleMacro>) -> RuleMacroResult<()> {
        let name = rule_macro.name().to_string();
        if self.name_index.contains_key(&name) {
            return Err(RuleMacroError::Duplicate { name });
        }
        let idx = self.macros.len();
        self.name_index.insert(name, idx);
        self.macros.push(rule_macro);
        Ok(())
    }

    /// Number of registered macros.
    pub fn len(&self) -> usize {
        self.macros.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.macros.is_empty()
    }

    /// Get a macro by name.
    pub fn get(&self, name: &str) -> Option<&dyn RuleMacro> {
        self.name_index
            .get(name)
            .map(|&idx| self.macros[idx].as_ref())
    }

    /// Expand all macros that match the given triple.
    pub fn expand_matching(
        &self,
        triple: &Triple,
        engine: &Engine,
    ) -> RuleMacroResult<Vec<Triple>> {
        let mut result = Vec::new();
        for m in &self.macros {
            if m.matches(triple) {
                result.extend(m.expand(engine)?);
            }
        }
        Ok(result)
    }

    /// Expand all registered macros.
    pub fn expand_all(&self, engine: &Engine) -> RuleMacroResult<Vec<Triple>> {
        let mut result = Vec::new();
        for m in &self.macros {
            result.extend(m.expand(engine)?);
        }
        Ok(result)
    }

    /// Answer a query across all macros.
    ///
    /// Returns all virtual triples that any macro can produce for the given
    /// subject and predicate.
    pub fn answer_query(
        &self,
        engine: &Engine,
        subject: SymbolId,
        predicate: SymbolId,
    ) -> RuleMacroResult<Vec<Triple>> {
        let mut result = Vec::new();
        for m in &self.macros {
            if m.can_answer(subject, predicate) {
                result.extend(m.answer(engine, subject, predicate)?);
            }
        }
        Ok(result)
    }

    /// List all registered macro names.
    pub fn macro_names(&self) -> Vec<&str> {
        self.macros.iter().map(|m| m.name()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;

    fn setup_genls_scenario() -> (Engine, MacroPredicates) {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = MacroPredicates::resolve(&engine).unwrap();

        // Create collections: Dog, Animal
        let dog = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Dog")
            .unwrap();
        let animal = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Animal")
            .unwrap();
        // genls(Dog, Animal)
        engine
            .add_triple(&Triple::new(dog.id, preds.genls, animal.id))
            .unwrap();
        // Create instances: (fido, is-a, Dog)
        let fido = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Fido")
            .unwrap();
        engine
            .add_triple(&Triple::new(fido.id, preds.is_a, dog.id))
            .unwrap();

        (engine, preds)
    }

    #[test]
    fn genls_macro_matches() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let m = GenlsMacro {
            child: dog,
            parent: animal,
            genls_pred: preds.genls,
            is_a: preds.is_a,
        };
        let triple = Triple::new(dog, preds.genls, animal);
        assert!(m.matches(&triple));
    }

    #[test]
    fn genls_expand_produces_parent_triples() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();
        let fido = engine.lookup_symbol("Fido").unwrap();

        let m = GenlsMacro {
            child: dog,
            parent: animal,
            genls_pred: preds.genls,
            is_a: preds.is_a,
        };

        let expanded = m.expand(&engine).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].subject, fido);
        assert_eq!(expanded[0].predicate, preds.is_a);
        assert_eq!(expanded[0].object, animal);
    }

    #[test]
    fn genls_answer_derives_parent_membership() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();
        let fido = engine.lookup_symbol("Fido").unwrap();

        let m = GenlsMacro {
            child: dog,
            parent: animal,
            genls_pred: preds.genls,
            is_a: preds.is_a,
        };

        let results = m.answer(&engine, fido, preds.is_a).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, animal);
    }

    #[test]
    fn genls_answer_non_member_returns_empty() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let cat = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Cat")
            .unwrap();

        let m = GenlsMacro {
            child: dog,
            parent: animal,
            genls_pred: preds.genls,
            is_a: preds.is_a,
        };

        let results = m.answer(&engine, cat.id, preds.is_a).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn registry_register_and_query() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();
        let fido = engine.lookup_symbol("Fido").unwrap();

        let mut registry = RuleMacroRegistry::new();
        registry
            .register(Box::new(GenlsMacro {
                child: dog,
                parent: animal,
                genls_pred: preds.genls,
                is_a: preds.is_a,
            }))
            .unwrap();

        assert_eq!(registry.len(), 1);
        let results = registry.answer_query(&engine, fido, preds.is_a).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn registry_duplicate_name_fails() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mut registry = RuleMacroRegistry::new();
        registry
            .register(Box::new(GenlsMacro {
                child: dog,
                parent: animal,
                genls_pred: preds.genls,
                is_a: preds.is_a,
            }))
            .unwrap();
        // Registering another genls should fail
        let err = registry
            .register(Box::new(GenlsMacro {
                child: dog,
                parent: animal,
                genls_pred: preds.genls,
                is_a: preds.is_a,
            }))
            .unwrap_err();
        assert!(matches!(err, RuleMacroError::Duplicate { .. }));
    }

    #[test]
    fn relation_all_exists_flags_missing() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = MacroPredicates::resolve(&engine).unwrap();

        let person = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Person")
            .unwrap();
        let car = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Car")
            .unwrap();
        let owns = engine
            .create_symbol(crate::symbol::SymbolKind::Relation, "owns")
            .unwrap();
        let alice = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Alice")
            .unwrap();

        // Alice is-a Person
        engine
            .add_triple(&Triple::new(alice.id, preds.is_a, person.id))
            .unwrap();

        let m = RelationAllExistsMacro {
            relation: owns.id,
            domain: person.id,
            range: car.id,
            macro_pred: preds.relation_all_exists,
            is_a: preds.is_a,
        };

        // Alice has no car, so expand should produce a virtual triple
        let expanded = m.expand(&engine).unwrap();
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].subject, alice.id);
        assert_eq!(expanded[0].predicate, owns.id);
    }

    #[test]
    fn relation_all_exists_answer_existing() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = MacroPredicates::resolve(&engine).unwrap();

        let person = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Person")
            .unwrap();
        let car = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Car")
            .unwrap();
        let owns = engine
            .create_symbol(crate::symbol::SymbolKind::Relation, "owns")
            .unwrap();
        let alice = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Alice")
            .unwrap();
        let my_car = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "MyCar")
            .unwrap();

        engine
            .add_triple(&Triple::new(alice.id, preds.is_a, person.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(alice.id, owns.id, my_car.id))
            .unwrap();

        let m = RelationAllExistsMacro {
            relation: owns.id,
            domain: person.id,
            range: car.id,
            macro_pred: preds.relation_all_exists,
            is_a: preds.is_a,
        };

        // Alice already has a car, answer should return the existing triple
        let results = m.answer(&engine, alice.id, owns.id).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, my_car.id);
    }

    #[test]
    fn relation_exists_all_answer() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = MacroPredicates::resolve(&engine).unwrap();

        let teacher = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Teacher")
            .unwrap();
        let course = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Course")
            .unwrap();
        let teaches = engine
            .create_symbol(crate::symbol::SymbolKind::Relation, "teaches")
            .unwrap();
        let mr_smith = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "MrSmith")
            .unwrap();
        let math = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Math")
            .unwrap();
        let physics = engine
            .create_symbol(crate::symbol::SymbolKind::Entity, "Physics")
            .unwrap();

        // MrSmith is-a Teacher, Math and Physics are Courses
        engine
            .add_triple(&Triple::new(mr_smith.id, preds.is_a, teacher.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(math.id, preds.is_a, course.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(physics.id, preds.is_a, course.id))
            .unwrap();

        let m = RelationExistsAllMacro {
            relation: teaches.id,
            domain: teacher.id,
            range: course.id,
            macro_pred: preds.relation_exists_all,
            is_a: preds.is_a,
        };

        // MrSmith should be virtually linked to all courses
        let results = m.answer(&engine, mr_smith.id, teaches.id).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn registry_expand_all() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mut registry = RuleMacroRegistry::new();
        registry
            .register(Box::new(GenlsMacro {
                child: dog,
                parent: animal,
                genls_pred: preds.genls,
                is_a: preds.is_a,
            }))
            .unwrap();

        let all = registry.expand_all(&engine).unwrap();
        assert_eq!(all.len(), 1); // Fido is-a Animal
    }

    #[test]
    fn registry_macro_names() {
        let (engine, preds) = setup_genls_scenario();
        let dog = engine.lookup_symbol("Dog").unwrap();
        let animal = engine.lookup_symbol("Animal").unwrap();

        let mut registry = RuleMacroRegistry::new();
        registry
            .register(Box::new(GenlsMacro {
                child: dog,
                parent: animal,
                genls_pred: preds.genls,
                is_a: preds.is_a,
            }))
            .unwrap();

        let names = registry.macro_names();
        assert_eq!(names, vec!["genls"]);
    }

    #[test]
    fn empty_registry_answer_returns_empty() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let preds = MacroPredicates::resolve(&engine).unwrap();
        let registry = RuleMacroRegistry::new();
        let results = registry
            .answer_query(&engine, preds.is_a, preds.is_a)
            .unwrap();
        assert!(results.is_empty());
    }
}
