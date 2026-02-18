//! Microtheory system: first-class reasoning contexts with inheritance.
//!
//! Inspired by Cyc's `ist(context, sentence)` modal operator, `genlMt` inheritance,
//! and lifting rules. Microtheories promote compartments from opaque containers to
//! first-class reasoning objects that can inherit from each other, carry domain
//! assumptions, and scope truth to specific contexts.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::{SymbolId, SymbolKind};

// ---------------------------------------------------------------------------
// Context domain
// ---------------------------------------------------------------------------

/// The kind of domain a microtheory covers.
///
/// Determines default behaviors like temporal decay, closed-world assumptions,
/// and lifting rule preferences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextDomain {
    /// Time-scoped context (e.g., "knowledge as of 2024").
    Temporal,
    /// Cultural context (e.g., "Egyptian mythology").
    Cultural,
    /// Belief context (e.g., "what Alice believes").
    Belief,
    /// Fictional context (e.g., "Star Wars universe").
    Fictional,
    /// Task-scoped context (e.g., "current project goals").
    Task,
    /// General-purpose context.
    General,
}

impl std::fmt::Display for ContextDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Temporal => write!(f, "temporal"),
            Self::Cultural => write!(f, "cultural"),
            Self::Belief => write!(f, "belief"),
            Self::Fictional => write!(f, "fictional"),
            Self::Task => write!(f, "task"),
            Self::General => write!(f, "general"),
        }
    }
}

impl ContextDomain {
    /// Parse a domain from a string label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "temporal" => Some(Self::Temporal),
            "cultural" => Some(Self::Cultural),
            "belief" => Some(Self::Belief),
            "fictional" => Some(Self::Fictional),
            "task" => Some(Self::Task),
            "general" => Some(Self::General),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifting condition
// ---------------------------------------------------------------------------

/// Under what conditions entailments propagate between contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LiftCondition {
    /// Always propagate entailments from source to target.
    Always,
    /// Propagate only if the entailment is consistent with the target context.
    IfConsistent,
    /// Propagate unless the target context has an explicit override.
    IfNotOverridden,
}

impl std::fmt::Display for LiftCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Always => write!(f, "always"),
            Self::IfConsistent => write!(f, "if-consistent"),
            Self::IfNotOverridden => write!(f, "if-not-overridden"),
        }
    }
}

impl LiftCondition {
    /// Parse a lift condition from a string label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "always" => Some(Self::Always),
            "if-consistent" => Some(Self::IfConsistent),
            "if-not-overridden" => Some(Self::IfNotOverridden),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Lifting rule
// ---------------------------------------------------------------------------

/// A rule governing when entailments propagate between two contexts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiftingRule {
    /// Source context (entailments propagate FROM here).
    pub from: SymbolId,
    /// Target context (entailments propagate TO here).
    pub to: SymbolId,
    /// Condition under which propagation occurs.
    pub condition: LiftCondition,
}

// ---------------------------------------------------------------------------
// Microtheory
// ---------------------------------------------------------------------------

/// A first-class reasoning context with inheritance and domain assumptions.
///
/// A microtheory is represented as an Entity symbol in the KG, with `ctx:`
/// predicates describing its properties. The cached `ancestors` field holds
/// the transitive closure of `ctx:specializes` for fast context-scoped queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Microtheory {
    /// The Entity symbol representing this microtheory.
    pub id: SymbolId,
    /// What kind of domain this context covers.
    pub domain: ContextDomain,
    /// Cached ancestor contexts via `ctx:specializes` transitive closure.
    /// Ordered from most specific (immediate parents) to most general.
    pub ancestors: Vec<SymbolId>,
}

// ---------------------------------------------------------------------------
// Context predicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for the microtheory system (prefixed `ctx:`).
///
/// Resolved lazily on first use via the engine's symbol registry.
#[derive(Debug, Clone)]
pub struct ContextPredicates {
    /// `ctx:specializes` — inheritance link (genlMt). Child specializes parent.
    pub specializes: SymbolId,
    /// `ctx:assumes` — domain assumption triples factored out of in-context assertions.
    pub assumes: SymbolId,
    /// `ctx:domain` — the kind of context (temporal, cultural, belief, etc.).
    pub domain: SymbolId,
    /// `ctx:disjoint` — two contexts cannot both apply simultaneously.
    pub disjoint: SymbolId,
    /// `ctx:lifting-condition` — lifting rule condition between contexts.
    pub lifting_condition: SymbolId,
    /// `ctx:lifts-to` — target context for a lifting rule.
    pub lifts_to: SymbolId,
}

impl ContextPredicates {
    /// Resolve all `ctx:` predicates from the engine's registry, creating them if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            specializes: engine.resolve_or_create_relation("ctx:specializes")?,
            assumes: engine.resolve_or_create_relation("ctx:assumes")?,
            domain: engine.resolve_or_create_relation("ctx:domain")?,
            disjoint: engine.resolve_or_create_relation("ctx:disjoint")?,
            lifting_condition: engine.resolve_or_create_relation("ctx:lifting-condition")?,
            lifts_to: engine.resolve_or_create_relation("ctx:lifts-to")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Ancestor resolution
// ---------------------------------------------------------------------------

/// Compute the transitive closure of `ctx:specializes` ancestors for a context.
///
/// Returns ancestor SymbolIds in BFS order (immediate parents first, most general last).
/// Handles cycles gracefully by tracking visited nodes.
pub fn resolve_ancestors(
    engine: &Engine,
    context_id: SymbolId,
    specializes_pred: SymbolId,
) -> Vec<SymbolId> {
    let kg = engine.knowledge_graph();
    let mut ancestors = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Seed with direct parents: context_id ctx:specializes parent
    for parent in kg.objects_of(context_id, specializes_pred) {
        if visited.insert(parent) {
            queue.push_back(parent);
        }
    }

    // BFS up the inheritance chain
    while let Some(current) = queue.pop_front() {
        ancestors.push(current);
        for grandparent in kg.objects_of(current, specializes_pred) {
            if visited.insert(grandparent) {
                queue.push_back(grandparent);
            }
        }
    }

    ancestors
}

// ---------------------------------------------------------------------------
// Context-scoped query
// ---------------------------------------------------------------------------

/// Query triples visible in a given context, including inherited ancestor contexts.
///
/// Returns triples from the current context and all ancestor contexts, filtering
/// by `compartment_id` matching. Triples from more specific contexts appear first.
pub fn triples_in_context(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    context_id: SymbolId,
    specializes_pred: SymbolId,
) -> Vec<Triple> {
    let kg = engine.knowledge_graph();
    let context_label = engine.resolve_label(context_id);
    let ancestors = resolve_ancestors(engine, context_id, specializes_pred);

    // Build the set of context labels to search (current + ancestors)
    let mut context_labels = Vec::with_capacity(1 + ancestors.len());
    context_labels.push(context_label);
    for ancestor in &ancestors {
        context_labels.push(engine.resolve_label(*ancestor));
    }

    // Get all triples from subject with predicate, then filter by context
    let all_triples = kg.triples_from(subject);
    let mut result: Vec<Triple> = all_triples
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .filter(|t| {
            match &t.compartment_id {
                Some(cid) => context_labels.iter().any(|cl| cl == cid),
                // Triples without a compartment_id are in the "global" context,
                // visible everywhere.
                None => true,
            }
        })
        .collect();

    // Sort so that triples from the most specific context come first.
    // Lower index in context_labels = more specific.
    result.sort_by_key(|t| {
        t.compartment_id
            .as_ref()
            .and_then(|cid| context_labels.iter().position(|cl| cl == cid))
            .unwrap_or(context_labels.len()) // global triples last
    });

    result
}

/// Query all triples visible in a context for a given subject (all predicates).
pub fn all_triples_in_context(
    engine: &Engine,
    subject: SymbolId,
    context_id: SymbolId,
    specializes_pred: SymbolId,
) -> Vec<Triple> {
    let kg = engine.knowledge_graph();
    let context_label = engine.resolve_label(context_id);
    let ancestors = resolve_ancestors(engine, context_id, specializes_pred);

    let mut context_labels = Vec::with_capacity(1 + ancestors.len());
    context_labels.push(context_label);
    for ancestor in &ancestors {
        context_labels.push(engine.resolve_label(*ancestor));
    }

    let all_triples = kg.triples_from(subject);
    all_triples
        .into_iter()
        .filter(|t| match &t.compartment_id {
            Some(cid) => context_labels.iter().any(|cl| cl == cid),
            None => true,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Context-scoped object query
// ---------------------------------------------------------------------------

/// Get objects of a subject+predicate visible in a given context (including ancestors).
pub fn objects_in_context(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    context_id: SymbolId,
    specializes_pred: SymbolId,
) -> Vec<SymbolId> {
    triples_in_context(engine, subject, predicate, context_id, specializes_pred)
        .into_iter()
        .map(|t| t.object)
        .collect()
}

// ---------------------------------------------------------------------------
// Disjointness checking
// ---------------------------------------------------------------------------

/// Check if two contexts are declared disjoint.
pub fn contexts_are_disjoint(
    engine: &Engine,
    ctx_a: SymbolId,
    ctx_b: SymbolId,
    disjoint_pred: SymbolId,
) -> bool {
    let kg = engine.knowledge_graph();
    kg.objects_of(ctx_a, disjoint_pred).contains(&ctx_b)
        || kg.objects_of(ctx_b, disjoint_pred).contains(&ctx_a)
}

// ---------------------------------------------------------------------------
// Lifting
// ---------------------------------------------------------------------------

/// Collect all lifting rules originating from a context.
pub fn lifting_rules_from(
    engine: &Engine,
    from_context: SymbolId,
    preds: &ContextPredicates,
) -> Vec<LiftingRule> {
    let kg = engine.knowledge_graph();
    let mut rules = Vec::new();

    // Each lifting rule is represented as:
    //   from_context ctx:lifts-to target_context
    //   from_context ctx:lifting-condition condition_entity
    // The condition entity's label maps to a LiftCondition variant.
    let targets = kg.objects_of(from_context, preds.lifts_to);
    let conditions = kg.objects_of(from_context, preds.lifting_condition);

    // Pair targets with conditions (positional pairing, or default to Always)
    for (i, target) in targets.iter().enumerate() {
        let condition = conditions
            .get(i)
            .and_then(|cid| {
                let label = engine.resolve_label(*cid);
                LiftCondition::from_label(&label)
            })
            .unwrap_or(LiftCondition::Always);

        rules.push(LiftingRule {
            from: from_context,
            to: *target,
            condition,
        });
    }

    rules
}

/// Apply a lifting rule: propagate triples from source to target context.
///
/// Returns the triples that should be propagated (caller decides whether to insert them).
/// Respects the lift condition:
/// - `Always`: propagate all matching triples
/// - `IfNotOverridden`: skip triples whose subject+predicate already exist in target
/// - `IfConsistent`: same as IfNotOverridden (consistency checking deferred to Phase 9l)
pub fn apply_lifting_rule(
    engine: &Engine,
    rule: &LiftingRule,
    specializes_pred: SymbolId,
) -> Vec<Triple> {
    let target_label = engine.resolve_label(rule.to);
    let source_label = engine.resolve_label(rule.from);

    // Get all triples in the source context
    let kg = engine.knowledge_graph();
    let source_triples: Vec<Triple> = kg
        .all_triples()
        .into_iter()
        .filter(|t| t.compartment_id.as_deref() == Some(&source_label))
        .collect();

    match rule.condition {
        LiftCondition::Always => {
            // Propagate all, rewriting compartment_id to target
            source_triples
                .into_iter()
                .map(|mut t| {
                    t.compartment_id = Some(target_label.clone());
                    t
                })
                .collect()
        }
        LiftCondition::IfNotOverridden | LiftCondition::IfConsistent => {
            // Check what already exists in the target context
            let target_triples: Vec<Triple> = kg
                .all_triples()
                .into_iter()
                .filter(|t| t.compartment_id.as_deref() == Some(&target_label))
                .collect();

            let existing_pairs: HashSet<(SymbolId, SymbolId)> = target_triples
                .iter()
                .map(|t| (t.subject, t.predicate))
                .collect();

            source_triples
                .into_iter()
                .filter(|t| !existing_pairs.contains(&(t.subject, t.predicate)))
                .map(|mut t| {
                    t.compartment_id = Some(target_label.clone());
                    t
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Context ancestry cache
// ---------------------------------------------------------------------------

/// A cache of context ancestry chains for fast repeated lookups.
#[derive(Debug, Clone, Default)]
pub struct ContextAncestryCache {
    /// context_id → ordered list of ancestor ids (BFS order, immediate parents first).
    cache: HashMap<SymbolId, Vec<SymbolId>>,
}

impl ContextAncestryCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or compute the ancestors for a context.
    pub fn ancestors(
        &mut self,
        engine: &Engine,
        context_id: SymbolId,
        specializes_pred: SymbolId,
    ) -> &[SymbolId] {
        if !self.cache.contains_key(&context_id) {
            let ancestors = resolve_ancestors(engine, context_id, specializes_pred);
            self.cache.insert(context_id, ancestors);
        }
        &self.cache[&context_id]
    }

    /// Invalidate the entire cache (call after hierarchy changes).
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    /// Invalidate a specific context's cached ancestry.
    pub fn invalidate_context(&mut self, context_id: SymbolId) {
        self.cache.remove(&context_id);
        // Also invalidate any context that had this context as an ancestor
        let to_invalidate: Vec<SymbolId> = self
            .cache
            .iter()
            .filter(|(_, ancestors)| ancestors.contains(&context_id))
            .map(|(k, _)| *k)
            .collect();
        for k in to_invalidate {
            self.cache.remove(&k);
        }
    }
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
    fn context_domain_roundtrip() {
        for domain in [
            ContextDomain::Temporal,
            ContextDomain::Cultural,
            ContextDomain::Belief,
            ContextDomain::Fictional,
            ContextDomain::Task,
            ContextDomain::General,
        ] {
            let label = domain.to_string();
            assert_eq!(ContextDomain::from_label(&label), Some(domain));
        }
    }

    #[test]
    fn lift_condition_roundtrip() {
        for cond in [
            LiftCondition::Always,
            LiftCondition::IfConsistent,
            LiftCondition::IfNotOverridden,
        ] {
            let label = cond.to_string();
            assert_eq!(LiftCondition::from_label(&label), Some(cond));
        }
    }

    #[test]
    fn resolve_empty_ancestors() {
        let engine = test_engine();
        let ctx = engine
            .create_symbol(SymbolKind::Entity, "test-ctx")
            .unwrap();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();
        let ancestors = resolve_ancestors(&engine, ctx.id, spec);
        assert!(ancestors.is_empty());
    }

    #[test]
    fn resolve_linear_ancestors() {
        let engine = test_engine();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();

        let general = engine
            .create_symbol(SymbolKind::Entity, "general-ctx")
            .unwrap();
        let science = engine
            .create_symbol(SymbolKind::Entity, "science-ctx")
            .unwrap();
        let physics = engine
            .create_symbol(SymbolKind::Entity, "physics-ctx")
            .unwrap();

        // physics specializes science, science specializes general
        engine
            .add_triple(&Triple::new(physics.id, spec, science.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(science.id, spec, general.id))
            .unwrap();

        let ancestors = resolve_ancestors(&engine, physics.id, spec);
        assert_eq!(ancestors, vec![science.id, general.id]);
    }

    #[test]
    fn resolve_diamond_ancestors() {
        let engine = test_engine();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();

        let root = engine
            .create_symbol(SymbolKind::Entity, "root-ctx")
            .unwrap();
        let left = engine
            .create_symbol(SymbolKind::Entity, "left-ctx")
            .unwrap();
        let right = engine
            .create_symbol(SymbolKind::Entity, "right-ctx")
            .unwrap();
        let child = engine
            .create_symbol(SymbolKind::Entity, "child-ctx")
            .unwrap();

        // Diamond: child → left, child → right, left → root, right → root
        engine
            .add_triple(&Triple::new(child.id, spec, left.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(child.id, spec, right.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(left.id, spec, root.id))
            .unwrap();
        engine
            .add_triple(&Triple::new(right.id, spec, root.id))
            .unwrap();

        let ancestors = resolve_ancestors(&engine, child.id, spec);
        // BFS: left, right, root (root only once despite diamond)
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&left.id));
        assert!(ancestors.contains(&right.id));
        assert!(ancestors.contains(&root.id));
    }

    #[test]
    fn context_scoped_query() {
        let engine = test_engine();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();

        // Create two contexts: child specializes parent
        let parent_ctx = engine
            .create_symbol(SymbolKind::Entity, "parent-ctx")
            .unwrap();
        let child_ctx = engine
            .create_symbol(SymbolKind::Entity, "child-ctx")
            .unwrap();
        engine
            .add_triple(&Triple::new(child_ctx.id, spec, parent_ctx.id))
            .unwrap();

        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let mammal = engine.resolve_or_create_entity("mammal").unwrap();
        let pet = engine.resolve_or_create_entity("pet").unwrap();

        // Parent context: dog is-a mammal
        engine
            .add_triple(
                &Triple::new(dog, is_a, mammal).with_compartment("parent-ctx".into()),
            )
            .unwrap();

        // Child context: dog is-a pet
        engine
            .add_triple(
                &Triple::new(dog, is_a, pet).with_compartment("child-ctx".into()),
            )
            .unwrap();

        // Query in child context should see both triples
        let results = triples_in_context(&engine, dog, is_a, child_ctx.id, spec);
        assert_eq!(results.len(), 2);

        // First result should be from child context (more specific)
        assert_eq!(results[0].object, pet);
        assert_eq!(results[1].object, mammal);

        // Query in parent context should only see parent triple
        let results = triples_in_context(&engine, dog, is_a, parent_ctx.id, spec);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, mammal);
    }

    #[test]
    fn global_triples_visible_everywhere() {
        let engine = test_engine();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();

        let ctx = engine
            .create_symbol(SymbolKind::Entity, "some-ctx")
            .unwrap();
        let dog = engine.resolve_or_create_entity("dog").unwrap();
        let animal = engine.resolve_or_create_entity("animal").unwrap();

        // Global triple (no compartment_id)
        engine.add_triple(&Triple::new(dog, is_a, animal)).unwrap();

        // Should be visible in the context
        let results = triples_in_context(&engine, dog, is_a, ctx.id, spec);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, animal);
    }

    #[test]
    fn disjoint_contexts() {
        let engine = test_engine();
        let preds = ContextPredicates::resolve(&engine).unwrap();

        let star_wars = engine
            .create_symbol(SymbolKind::Entity, "star-wars")
            .unwrap();
        let real_world = engine
            .create_symbol(SymbolKind::Entity, "real-world")
            .unwrap();

        engine
            .add_triple(&Triple::new(star_wars.id, preds.disjoint, real_world.id))
            .unwrap();

        assert!(contexts_are_disjoint(
            &engine,
            star_wars.id,
            real_world.id,
            preds.disjoint,
        ));
        // Symmetric
        assert!(contexts_are_disjoint(
            &engine,
            real_world.id,
            star_wars.id,
            preds.disjoint,
        ));
    }

    #[test]
    fn ancestry_cache() {
        let engine = test_engine();
        let spec = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();

        let root = engine
            .create_symbol(SymbolKind::Entity, "root")
            .unwrap();
        let child = engine
            .create_symbol(SymbolKind::Entity, "child")
            .unwrap();
        engine
            .add_triple(&Triple::new(child.id, spec, root.id))
            .unwrap();

        let mut cache = ContextAncestryCache::new();

        // First call computes
        let ancestors = cache.ancestors(&engine, child.id, spec);
        assert_eq!(ancestors, &[root.id]);

        // Second call uses cache
        let ancestors = cache.ancestors(&engine, child.id, spec);
        assert_eq!(ancestors, &[root.id]);

        // Invalidate and recompute
        cache.invalidate_context(root.id);
        // child should be invalidated because root was in its ancestors
        assert!(!cache.cache.contains_key(&child.id));
    }
}
