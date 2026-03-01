//! Circumscription and Closed World Assumption (CWA) for microtheories.
//!
//! Provides per-context reasoning assumptions inspired by Cyc's configurable
//! closed-world reasoning:
//!
//! - **Closed World Assumption (CWA)**: failure to find a triple is negation, not unknown
//! - **Unique Names Assumption (UNA)**: different SymbolIds refer to different entities
//! - **Circumscription**: the only instances of a collection are those explicitly known
//!
//! These are stored as per-microtheory settings queryable via well-known predicates.

use std::collections::{HashMap, HashSet};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::{SymbolId, SymbolKind};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to CWA / circumscription operations.
#[derive(Debug, Error, Diagnostic)]
pub enum CwaError {
    #[error("context {context_id} not found")]
    #[diagnostic(
        code(akh::cwa::context_not_found),
        help("The specified context does not exist. Create it with `create_context()` first.")
    )]
    ContextNotFound { context_id: u64 },

    #[error("circumscribed collection {collection_id} not found in context {context_id}")]
    #[diagnostic(
        code(akh::cwa::collection_not_found),
        help(
            "The collection is not registered as circumscribed in this context. \
             Use `add_circumscribed_collection()` first."
        )
    )]
    CollectionNotFound {
        collection_id: u64,
        context_id: u64,
    },

    #[error("negation-as-failure query requires CWA to be active in context {context_id}")]
    #[diagnostic(
        code(akh::cwa::cwa_not_active),
        help("Enable CWA for this context with `set_cwa(context_id, true)` first.")
    )]
    CwaNotActive { context_id: u64 },
}

/// Result type for CWA operations.
pub type CwaResult<T> = std::result::Result<T, CwaError>;

// ---------------------------------------------------------------------------
// Context assumptions
// ---------------------------------------------------------------------------

/// Per-context reasoning assumptions.
///
/// Each microtheory can configure whether it uses closed-world reasoning,
/// unique names, and which collections are circumscribed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextAssumptions {
    /// When true, failure to find a triple is treated as negation.
    pub cwa: bool,
    /// When true, different SymbolIds are assumed to refer to different entities
    /// unless explicitly linked via equivalence.
    pub una: bool,
    /// Collections (types) that are circumscribed: the only instances are those
    /// explicitly known or derivable.
    pub circumscribed_collections: HashSet<SymbolId>,
}

impl ContextAssumptions {
    /// Create default assumptions (all off).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create assumptions with CWA active.
    pub fn with_cwa() -> Self {
        Self {
            cwa: true,
            ..Default::default()
        }
    }

    /// Create assumptions with both CWA and UNA active.
    pub fn with_cwa_and_una() -> Self {
        Self {
            cwa: true,
            una: true,
            ..Default::default()
        }
    }

    /// Add a circumscribed collection.
    pub fn circumscribe(&mut self, collection: SymbolId) {
        self.circumscribed_collections.insert(collection);
    }

    /// Check if a collection is circumscribed.
    pub fn is_circumscribed(&self, collection: SymbolId) -> bool {
        self.circumscribed_collections.contains(&collection)
    }
}

// ---------------------------------------------------------------------------
// Well-known predicates
// ---------------------------------------------------------------------------

/// Well-known predicates for CWA settings.
pub struct CwaPredicates {
    /// `ctx:cwa` — marks a context as using closed-world assumption.
    pub cwa: SymbolId,
    /// `ctx:una` — marks a context as using unique names assumption.
    pub una: SymbolId,
    /// `ctx:circumscribes` — links a context to a circumscribed collection.
    pub circumscribes: SymbolId,
    /// `is-a` relation for type checking.
    pub is_a: SymbolId,
}

impl CwaPredicates {
    /// Resolve well-known CWA predicates, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        let cwa = engine
            .create_symbol(SymbolKind::Relation, "ctx:cwa")
            .or_else(|_| engine.lookup_symbol("ctx:cwa").map(|id| crate::symbol::SymbolMeta::new(id, SymbolKind::Relation, "ctx:cwa")))?;
        let una = engine
            .create_symbol(SymbolKind::Relation, "ctx:una")
            .or_else(|_| engine.lookup_symbol("ctx:una").map(|id| crate::symbol::SymbolMeta::new(id, SymbolKind::Relation, "ctx:una")))?;
        let circumscribes = engine
            .create_symbol(SymbolKind::Relation, "ctx:circumscribes")
            .or_else(|_| engine.lookup_symbol("ctx:circumscribes").map(|id| crate::symbol::SymbolMeta::new(id, SymbolKind::Relation, "ctx:circumscribes")))?;
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .or_else(|_| engine.lookup_symbol("is-a").map(|id| crate::symbol::SymbolMeta::new(id, SymbolKind::Relation, "is-a")))?;

        Ok(Self {
            cwa: cwa.id,
            una: una.id,
            circumscribes: circumscribes.id,
            is_a: is_a.id,
        })
    }
}

// ---------------------------------------------------------------------------
// Assumptions registry
// ---------------------------------------------------------------------------

/// Registry mapping contexts to their reasoning assumptions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssumptionRegistry {
    /// Context SymbolId → assumptions.
    assumptions: HashMap<SymbolId, ContextAssumptions>,
}

impl AssumptionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set assumptions for a context.
    pub fn set_assumptions(&mut self, context: SymbolId, assumptions: ContextAssumptions) {
        self.assumptions.insert(context, assumptions);
    }

    /// Get assumptions for a context.
    pub fn get_assumptions(&self, context: SymbolId) -> Option<&ContextAssumptions> {
        self.assumptions.get(&context)
    }

    /// Check if CWA is active for a context.
    pub fn is_cwa_active(&self, context: SymbolId) -> bool {
        self.assumptions
            .get(&context)
            .map(|a| a.cwa)
            .unwrap_or(false)
    }

    /// Check if UNA is active for a context.
    pub fn is_una_active(&self, context: SymbolId) -> bool {
        self.assumptions
            .get(&context)
            .map(|a| a.una)
            .unwrap_or(false)
    }

    /// Add a circumscribed collection to a context.
    pub fn circumscribe(&mut self, context: SymbolId, collection: SymbolId) {
        self.assumptions
            .entry(context)
            .or_default()
            .circumscribe(collection);
    }

    /// Number of contexts with assumptions.
    pub fn len(&self) -> usize {
        self.assumptions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.assumptions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Negation-as-failure query
// ---------------------------------------------------------------------------

/// Result of a negation-as-failure query.
#[derive(Debug, Clone, PartialEq)]
pub enum NafResult {
    /// Triple found — positive result.
    Found(Vec<Triple>),
    /// Triple not found and CWA active — negation.
    NegatedByAbsence,
    /// Triple not found but CWA not active — unknown.
    Unknown,
}

/// Query with negation-as-failure: if CWA is active in the given context
/// and no matching triples are found, returns `NegatedByAbsence` instead
/// of an empty result.
pub fn query_naf(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    context: Option<SymbolId>,
    registry: &AssumptionRegistry,
) -> NafResult {
    let kg = engine.knowledge_graph();
    let triples: Vec<Triple> = kg
        .triples_from(subject)
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .filter(|t| match context {
            Some(ctx) => t.compartment_id.as_deref() == Some(&format!("{}", ctx.get())),
            None => true,
        })
        .collect();

    if !triples.is_empty() {
        return NafResult::Found(triples);
    }

    // No triples found — check CWA
    match context {
        Some(ctx) if registry.is_cwa_active(ctx) => NafResult::NegatedByAbsence,
        _ => NafResult::Unknown,
    }
}

/// Query circumscribed instances: returns all known instances of a collection
/// in a context where that collection is circumscribed.
///
/// Returns `None` if the collection is not circumscribed (open-world).
pub fn circumscribed_instances(
    engine: &Engine,
    collection: SymbolId,
    context: SymbolId,
    registry: &AssumptionRegistry,
    predicates: &CwaPredicates,
) -> Option<Vec<SymbolId>> {
    let assumptions = registry.get_assumptions(context)?;
    if !assumptions.is_circumscribed(collection) {
        return None;
    }

    let kg = engine.knowledge_graph();
    // Find all entities that are instances of this collection
    let instances: Vec<SymbolId> = kg
        .triples_to(collection)
        .into_iter()
        .filter(|t| t.predicate == predicates.is_a)
        .filter(|t| match &t.compartment_id {
            Some(cid) => cid == &format!("{}", context.get()),
            None => true, // Global triples visible to all contexts
        })
        .map(|t| t.subject)
        .collect();

    Some(instances)
}

/// Check UNA: under the unique names assumption, two different SymbolIds
/// are considered distinct entities unless explicitly linked.
///
/// Returns `true` if the entities are considered the same (either same ID
/// or explicitly linked), `false` if UNA says they're different.
pub fn una_same_entity(
    engine: &Engine,
    a: SymbolId,
    b: SymbolId,
    context: SymbolId,
    registry: &AssumptionRegistry,
) -> bool {
    if a == b {
        return true;
    }

    if !registry.is_una_active(context) {
        // UNA not active — can't determine, default to structural equality
        return a == b;
    }

    // Under UNA, different IDs means different entities
    // unless an explicit equivalence triple exists
    let kg = engine.knowledge_graph();
    let triples = kg.triples_from(a);
    for t in triples {
        if t.object == b {
            // Check if the predicate is an equivalence relation
            let label = engine.resolve_label(t.predicate);
            if label == "owl:sameAs" || label == "same-as" || label == "equivalent-to" {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;

    #[test]
    fn context_assumptions_defaults() {
        let assumptions = ContextAssumptions::new();
        assert!(!assumptions.cwa);
        assert!(!assumptions.una);
        assert!(assumptions.circumscribed_collections.is_empty());
    }

    #[test]
    fn context_assumptions_cwa() {
        let assumptions = ContextAssumptions::with_cwa();
        assert!(assumptions.cwa);
        assert!(!assumptions.una);
    }

    #[test]
    fn context_assumptions_cwa_and_una() {
        let assumptions = ContextAssumptions::with_cwa_and_una();
        assert!(assumptions.cwa);
        assert!(assumptions.una);
    }

    #[test]
    fn circumscribe_collection() {
        let mut assumptions = ContextAssumptions::new();
        let collection = SymbolId::new(42).unwrap();
        assert!(!assumptions.is_circumscribed(collection));
        assumptions.circumscribe(collection);
        assert!(assumptions.is_circumscribed(collection));
    }

    #[test]
    fn registry_set_and_get() {
        let mut reg = AssumptionRegistry::new();
        let ctx = SymbolId::new(1).unwrap();
        assert!(!reg.is_cwa_active(ctx));

        reg.set_assumptions(ctx, ContextAssumptions::with_cwa());
        assert!(reg.is_cwa_active(ctx));
        assert!(!reg.is_una_active(ctx));
    }

    #[test]
    fn registry_circumscribe() {
        let mut reg = AssumptionRegistry::new();
        let ctx = SymbolId::new(1).unwrap();
        let coll = SymbolId::new(2).unwrap();

        reg.circumscribe(ctx, coll);
        let assumptions = reg.get_assumptions(ctx).unwrap();
        assert!(assumptions.is_circumscribed(coll));
    }

    #[test]
    fn naf_query_found() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap();
        let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
        engine
            .add_triple(&Triple::new(sun.id, is_a.id, star.id))
            .unwrap();

        let reg = AssumptionRegistry::new();
        let result = query_naf(&engine, sun.id, is_a.id, None, &reg);
        assert!(matches!(result, NafResult::Found(_)));
    }

    #[test]
    fn naf_query_unknown_without_cwa() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let likes = engine
            .create_symbol(SymbolKind::Relation, "likes")
            .unwrap();

        let reg = AssumptionRegistry::new();
        let result = query_naf(&engine, sun.id, likes.id, None, &reg);
        assert!(matches!(result, NafResult::Unknown));
    }

    #[test]
    fn naf_query_negated_with_cwa() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let ctx_sym = engine
            .create_symbol(SymbolKind::Entity, "TestContext")
            .unwrap();
        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let likes = engine
            .create_symbol(SymbolKind::Relation, "likes")
            .unwrap();

        let mut reg = AssumptionRegistry::new();
        reg.set_assumptions(ctx_sym.id, ContextAssumptions::with_cwa());

        let result = query_naf(&engine, sun.id, likes.id, Some(ctx_sym.id), &reg);
        assert!(matches!(result, NafResult::NegatedByAbsence));
    }

    #[test]
    fn circumscribed_instances_returns_known() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let ctx_sym = engine
            .create_symbol(SymbolKind::Entity, "ScienceCtx")
            .unwrap();
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap();
        let planet = engine
            .create_symbol(SymbolKind::Entity, "Planet")
            .unwrap();
        let earth = engine
            .create_symbol(SymbolKind::Entity, "Earth")
            .unwrap();
        let mars = engine.create_symbol(SymbolKind::Entity, "Mars").unwrap();

        engine
            .add_triple(&Triple::new(earth.id, is_a.id, planet.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(mars.id, is_a.id, planet.id))
            .unwrap();

        let predicates = CwaPredicates {
            cwa: SymbolId::new(100).unwrap(),
            una: SymbolId::new(101).unwrap(),
            circumscribes: SymbolId::new(102).unwrap(),
            is_a: is_a.id,
        };

        let mut reg = AssumptionRegistry::new();
        reg.circumscribe(ctx_sym.id, planet.id);

        let instances =
            circumscribed_instances(&engine, planet.id, ctx_sym.id, &reg, &predicates).unwrap();
        assert_eq!(instances.len(), 2);
        assert!(instances.contains(&earth.id));
        assert!(instances.contains(&mars.id));
    }

    #[test]
    fn circumscribed_instances_none_when_not_circumscribed() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let ctx_sym = engine
            .create_symbol(SymbolKind::Entity, "OpenCtx")
            .unwrap();
        let planet = engine
            .create_symbol(SymbolKind::Entity, "Planet")
            .unwrap();

        let predicates = CwaPredicates {
            cwa: SymbolId::new(100).unwrap(),
            una: SymbolId::new(101).unwrap(),
            circumscribes: SymbolId::new(102).unwrap(),
            is_a: SymbolId::new(103).unwrap(),
        };

        let reg = AssumptionRegistry::new();
        let result = circumscribed_instances(&engine, planet.id, ctx_sym.id, &reg, &predicates);
        assert!(result.is_none());
    }

    #[test]
    fn una_same_entity_identical() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let ctx = engine
            .create_symbol(SymbolKind::Entity, "UnaCtx")
            .unwrap();
        let entity = engine.create_symbol(SymbolKind::Entity, "A").unwrap();

        let mut reg = AssumptionRegistry::new();
        reg.set_assumptions(ctx.id, ContextAssumptions::with_cwa_and_una());

        assert!(una_same_entity(&engine, entity.id, entity.id, ctx.id, &reg));
    }

    #[test]
    fn una_different_entities() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let ctx = engine
            .create_symbol(SymbolKind::Entity, "UnaCtx")
            .unwrap();
        let a = engine.create_symbol(SymbolKind::Entity, "A").unwrap();
        let b = engine.create_symbol(SymbolKind::Entity, "B").unwrap();

        let mut reg = AssumptionRegistry::new();
        reg.set_assumptions(ctx.id, ContextAssumptions::with_cwa_and_una());

        assert!(!una_same_entity(&engine, a.id, b.id, ctx.id, &reg));
    }
}
