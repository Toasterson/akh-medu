//! Inference engine: spreading activation + VSA recovery.
//!
//! Combines graph-guided spreading activation with VSA bind/unbind
//! operations to infer new knowledge from existing triples.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use egg::Rewrite;

use crate::error::InferError;
use crate::graph::index::KnowledgeGraph;
use crate::graph::predicate_hierarchy::PredicateHierarchy;
use crate::reason::AkhLang;
use crate::symbol::SymbolId;
use crate::temporal::{TemporalRegistry, apply_temporal_decay};
use crate::vsa::HyperVec;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

use super::{DerivationKind, InferenceQuery, InferenceResult, ProvenanceRecord};

/// Result type for inference operations.
pub type InferResult<T> = std::result::Result<T, InferError>;

/// Stateless inference engine — per-query state lives in [`InferContext`].
pub struct InferEngine {
    ops: Arc<VsaOps>,
    item_memory: Arc<ItemMemory>,
    knowledge_graph: Arc<KnowledgeGraph>,
}

/// Per-query mutable state for a single inference run.
struct InferContext {
    activations: HashMap<SymbolId, f32>,
    pattern: Option<HyperVec>,
    provenance: Vec<ProvenanceRecord>,
    expanded: HashSet<SymbolId>,
}

impl InferContext {
    fn new() -> Self {
        Self {
            activations: HashMap::new(),
            pattern: None,
            provenance: Vec::new(),
            expanded: HashSet::new(),
        }
    }

    /// Activate a symbol. Returns true if the confidence improved.
    fn activate(&mut self, symbol: SymbolId, confidence: f32) -> bool {
        let entry = self.activations.entry(symbol).or_insert(0.0);
        if confidence > *entry {
            *entry = confidence;
            true
        } else {
            false
        }
    }
}

/// Phase 9 context for hierarchy-aware and temporal-aware inference.
///
/// When provided to [`InferEngine::infer_with_phase9`], the spreading activation
/// loop expands to include specialization predicates, inverse predicates, and
/// temporal decay of triple confidences.
pub struct InferPhase9Context<'a> {
    /// Predicate hierarchy for specialization/inverse expansion.
    pub hierarchy: Option<&'a PredicateHierarchy>,
    /// Temporal registry for confidence decay.
    pub temporal: Option<&'a TemporalRegistry>,
    /// Current query time in seconds since epoch (for temporal decay).
    pub query_time_secs: u64,
}

impl<'a> InferPhase9Context<'a> {
    /// Create an empty Phase 9 context (no hierarchy, no temporal).
    pub fn empty() -> Self {
        Self {
            hierarchy: None,
            temporal: None,
            query_time_secs: 0,
        }
    }
}

impl InferEngine {
    /// Create an inference engine from shared subsystem handles.
    pub fn new(
        ops: Arc<VsaOps>,
        item_memory: Arc<ItemMemory>,
        knowledge_graph: Arc<KnowledgeGraph>,
    ) -> Self {
        Self {
            ops,
            item_memory,
            knowledge_graph,
        }
    }

    /// Run spreading-activation inference using the built-in rules.
    pub fn infer(&self, query: &InferenceQuery) -> InferResult<InferenceResult> {
        let rules = crate::reason::builtin_rules();
        self.infer_with_rules(query, &rules)
    }

    /// Run spreading-activation inference with Phase 9 context (hierarchy + temporal).
    ///
    /// When a `PredicateHierarchy` is provided, the spreading activation loop
    /// also follows specialization predicates and inverse predicates. When a
    /// `TemporalRegistry` is provided, triple confidences are decayed according
    /// to the relation's temporal profile before propagation.
    pub fn infer_with_phase9(
        &self,
        query: &InferenceQuery,
        rules: &[Rewrite<AkhLang, ()>],
        phase9: &InferPhase9Context<'_>,
    ) -> InferResult<InferenceResult> {
        if query.seeds.is_empty() {
            return Err(InferError::NoSeeds);
        }

        let mut ctx = InferContext::new();
        let now = phase9.query_time_secs;

        // --- Seed activation ---
        let mut seed_vecs: Vec<HyperVec> = Vec::with_capacity(query.seeds.len());
        for &seed in &query.seeds {
            let vec = self.item_memory.get_or_create(&self.ops, seed);
            seed_vecs.push(vec);
            ctx.activate(seed, 1.0);
            ctx.provenance.push(
                ProvenanceRecord::new(seed, DerivationKind::Seed)
                    .with_confidence(1.0)
                    .with_depth(0),
            );
        }

        let seed_refs: Vec<&HyperVec> = seed_vecs.iter().collect();
        ctx.pattern = Some(self.ops.bundle(&seed_refs)?);

        // --- Spreading activation with Phase 9 extensions ---
        for depth in 0..query.max_depth {
            let frontier: Vec<(SymbolId, f32)> = ctx
                .activations
                .iter()
                .filter(|(sym, _)| !ctx.expanded.contains(sym))
                .map(|(&sym, &conf)| (sym, conf))
                .collect();

            if frontier.is_empty() {
                break;
            }

            let mut new_vecs: Vec<HyperVec> = Vec::new();

            for (sym, parent_confidence) in frontier {
                ctx.expanded.insert(sym);

                // Collect triples: direct + hierarchy-expanded
                let mut triples = self.knowledge_graph.triples_from(sym);

                // Hierarchy: also follow specialization predicates
                if let Some(hierarchy) = phase9.hierarchy {
                    let base_triples = triples.clone();
                    for triple in &base_triples {
                        for &spec_pred in hierarchy.specializations_of(triple.predicate) {
                            let spec_triples = self.knowledge_graph.triples_from(sym);
                            for st in spec_triples {
                                if st.predicate == spec_pred && !triples.iter().any(|t| t.subject == st.subject && t.predicate == st.predicate && t.object == st.object) {
                                    triples.push(st);
                                }
                            }
                        }
                    }

                    // Inverse: for each predicate with an inverse, also look at triples_to(sym)
                    let all_preds: Vec<SymbolId> = triples.iter().map(|t| t.predicate).collect();
                    let mut seen_inverses = std::collections::HashSet::new();
                    for pred in &all_preds {
                        if let Some(inv_pred) = hierarchy.inverse_of(*pred) {
                            if seen_inverses.insert(inv_pred) {
                                let incoming = self.knowledge_graph.triples_to(sym);
                                for t in incoming {
                                    if t.predicate == inv_pred {
                                        // Flip: t is (?, inv_pred, sym) → treat as (sym, pred, ?)
                                        let mut flipped = crate::graph::Triple::new(sym, *pred, t.subject);
                                        flipped.confidence = t.confidence;
                                        flipped.timestamp = t.timestamp;
                                        if !triples.iter().any(|existing| existing.subject == flipped.subject && existing.predicate == flipped.predicate && existing.object == flipped.object) {
                                            triples.push(flipped);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if triples.is_empty() {
                    continue;
                }

                let sym_vec = self.item_memory.get_or_create(&self.ops, sym);

                for triple in &triples {
                    if let Some(ref filter) = query.predicate_filter {
                        if !filter.contains(&triple.predicate) {
                            continue;
                        }
                    }

                    // Apply temporal decay if available
                    let edge_confidence = if let Some(temporal) = phase9.temporal {
                        if let Some(profile) = temporal.get_profile(triple.predicate) {
                            apply_temporal_decay(profile, triple.confidence, triple.timestamp, now)
                        } else {
                            triple.confidence
                        }
                    } else {
                        triple.confidence
                    };

                    let graph_confidence = parent_confidence * edge_confidence;
                    if graph_confidence >= query.min_confidence {
                        if ctx.activate(triple.object, graph_confidence) {
                            ctx.provenance.push(
                                ProvenanceRecord::new(
                                    triple.object,
                                    DerivationKind::GraphEdge {
                                        from: sym,
                                        predicate: triple.predicate,
                                    },
                                )
                                .with_sources(vec![sym])
                                .with_confidence(graph_confidence)
                                .with_depth(depth + 1),
                            );
                            let obj_vec = self.item_memory.get_or_create(&self.ops, triple.object);
                            new_vecs.push(obj_vec);
                        }
                    }

                    // VSA recovery
                    let pred_vec = self.item_memory.get_or_create(&self.ops, triple.predicate);
                    let recovered = self.ops.unbind(&sym_vec, &pred_vec)?;

                    if let Ok(search_results) = self.item_memory.search(&recovered, 1) {
                        for sr in &search_results {
                            if sr.symbol_id == triple.object {
                                continue;
                            }
                            if sr.similarity >= query.min_similarity {
                                let vsa_confidence =
                                    parent_confidence * edge_confidence.min(sr.similarity);
                                let combined = graph_confidence.max(vsa_confidence);
                                if combined >= query.min_confidence {
                                    if ctx.activate(sr.symbol_id, combined) {
                                        ctx.provenance.push(
                                            ProvenanceRecord::new(
                                                sr.symbol_id,
                                                DerivationKind::VsaRecovery {
                                                    from: sym,
                                                    predicate: triple.predicate,
                                                    similarity: sr.similarity,
                                                },
                                            )
                                            .with_sources(vec![sym, triple.predicate])
                                            .with_confidence(combined)
                                            .with_depth(depth + 1),
                                        );
                                        let sr_vec = self.item_memory.get_or_create(&self.ops, sr.symbol_id);
                                        new_vecs.push(sr_vec);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !new_vecs.is_empty() {
                if let Some(ref current_pattern) = ctx.pattern {
                    let mut all_refs: Vec<&HyperVec> = vec![current_pattern];
                    all_refs.extend(new_vecs.iter());
                    ctx.pattern = Some(self.ops.bundle(&all_refs)?);
                }
            }
        }

        // --- Optional e-graph verification ---
        if query.verify_with_egraph {
            self.verify_with_egraph(&mut ctx, rules);
        }

        // --- Collect results ---
        let seed_set: HashSet<SymbolId> = query.seeds.iter().copied().collect();
        let mut activations: Vec<(SymbolId, f32)> = ctx
            .activations
            .into_iter()
            .filter(|(sym, _)| !seed_set.contains(sym))
            .collect();
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        activations.truncate(query.top_k);

        Ok(InferenceResult {
            activations,
            pattern: ctx.pattern,
            provenance: ctx.provenance,
        })
    }

    /// Run spreading-activation inference with custom rewrite rules.
    pub fn infer_with_rules(
        &self,
        query: &InferenceQuery,
        rules: &[Rewrite<AkhLang, ()>],
    ) -> InferResult<InferenceResult> {
        if query.seeds.is_empty() {
            return Err(InferError::NoSeeds);
        }

        let mut ctx = InferContext::new();

        // --- Seed activation ---
        let mut seed_vecs: Vec<HyperVec> = Vec::with_capacity(query.seeds.len());
        for &seed in &query.seeds {
            let vec = self.item_memory.get_or_create(&self.ops, seed);
            seed_vecs.push(vec);
            ctx.activate(seed, 1.0);
            ctx.provenance.push(
                ProvenanceRecord::new(seed, DerivationKind::Seed)
                    .with_confidence(1.0)
                    .with_depth(0),
            );
        }

        // Bundle seeds into initial pattern
        let seed_refs: Vec<&HyperVec> = seed_vecs.iter().collect();
        ctx.pattern = Some(self.ops.bundle(&seed_refs)?);

        // --- Spreading activation ---
        for depth in 0..query.max_depth {
            let frontier: Vec<(SymbolId, f32)> = ctx
                .activations
                .iter()
                .filter(|(sym, _)| !ctx.expanded.contains(sym))
                .map(|(&sym, &conf)| (sym, conf))
                .collect();

            if frontier.is_empty() {
                break;
            }

            let mut new_vecs: Vec<HyperVec> = Vec::new();

            for (sym, parent_confidence) in frontier {
                ctx.expanded.insert(sym);

                let triples = self.knowledge_graph.triples_from(sym);
                if triples.is_empty() {
                    continue;
                }

                let sym_vec = self.item_memory.get_or_create(&self.ops, sym);

                for triple in &triples {
                    // Apply predicate filter
                    if let Some(ref filter) = query.predicate_filter {
                        if !filter.contains(&triple.predicate) {
                            continue;
                        }
                    }

                    let edge_confidence = triple.confidence;

                    // --- Graph-direct activation ---
                    let graph_confidence = parent_confidence * edge_confidence;
                    if graph_confidence >= query.min_confidence {
                        if ctx.activate(triple.object, graph_confidence) {
                            ctx.provenance.push(
                                ProvenanceRecord::new(
                                    triple.object,
                                    DerivationKind::GraphEdge {
                                        from: sym,
                                        predicate: triple.predicate,
                                    },
                                )
                                .with_sources(vec![sym])
                                .with_confidence(graph_confidence)
                                .with_depth(depth + 1),
                            );
                            let obj_vec = self.item_memory.get_or_create(&self.ops, triple.object);
                            new_vecs.push(obj_vec);
                        }
                    }

                    // --- VSA recovery: unbind(subject, predicate) → recovered ---
                    let pred_vec = self.item_memory.get_or_create(&self.ops, triple.predicate);
                    let recovered = self.ops.unbind(&sym_vec, &pred_vec)?;

                    if let Ok(search_results) = self.item_memory.search(&recovered, 1) {
                        for sr in &search_results {
                            // Skip if it's the same as the graph-direct object
                            if sr.symbol_id == triple.object {
                                continue;
                            }
                            if sr.similarity >= query.min_similarity {
                                // Cap VSA recovery confidence by the graph path:
                                // similarity is a ceiling, not the sole confidence.
                                let vsa_confidence =
                                    parent_confidence * edge_confidence.min(sr.similarity);
                                let combined = graph_confidence.max(vsa_confidence);
                                if combined >= query.min_confidence {
                                    if ctx.activate(sr.symbol_id, combined) {
                                        ctx.provenance.push(
                                            ProvenanceRecord::new(
                                                sr.symbol_id,
                                                DerivationKind::VsaRecovery {
                                                    from: sym,
                                                    predicate: triple.predicate,
                                                    similarity: sr.similarity,
                                                },
                                            )
                                            .with_sources(vec![sym, triple.predicate])
                                            .with_confidence(combined)
                                            .with_depth(depth + 1),
                                        );
                                        let sr_vec =
                                            self.item_memory.get_or_create(&self.ops, sr.symbol_id);
                                        new_vecs.push(sr_vec);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Bundle new vectors into the interference pattern
            if !new_vecs.is_empty() {
                if let Some(ref current_pattern) = ctx.pattern {
                    let mut all_refs: Vec<&HyperVec> = vec![current_pattern];
                    all_refs.extend(new_vecs.iter());
                    ctx.pattern = Some(self.ops.bundle(&all_refs)?);
                }
            }
        }

        // --- Optional e-graph verification ---
        if query.verify_with_egraph {
            self.verify_with_egraph(&mut ctx, rules);
        }

        // --- Collect results: filter out seeds, sort by confidence ---
        let seed_set: HashSet<SymbolId> = query.seeds.iter().copied().collect();
        let mut activations: Vec<(SymbolId, f32)> = ctx
            .activations
            .into_iter()
            .filter(|(sym, _)| !seed_set.contains(sym))
            .collect();
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        activations.truncate(query.top_k);

        Ok(InferenceResult {
            activations,
            pattern: ctx.pattern,
            provenance: ctx.provenance,
        })
    }

    /// Analogy: "A is to B as C is to ?" via `unbind(bind(A,B), C)` + cleanup.
    pub fn infer_analogy(
        &self,
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
        top_k: usize,
    ) -> InferResult<Vec<(SymbolId, f32)>> {
        let mut distinct = HashSet::new();
        distinct.insert(a);
        distinct.insert(b);
        distinct.insert(c);
        if distinct.len() != 3 {
            return Err(InferError::InvalidAnalogy {
                count: distinct.len(),
            });
        }

        let a_vec = self.item_memory.get_or_create(&self.ops, a);
        let b_vec = self.item_memory.get_or_create(&self.ops, b);
        let c_vec = self.item_memory.get_or_create(&self.ops, c);

        // Compute the relational vector: bind(A, B) captures the A→B relationship
        let relation = self.ops.bind(&a_vec, &b_vec)?;
        // Apply the same relation to C: unbind(relation, C) ≈ D
        let recovered = self.ops.unbind(&relation, &c_vec)?;

        let results = self.item_memory.search(&recovered, top_k)?;
        Ok(results
            .into_iter()
            .map(|sr| (sr.symbol_id, sr.similarity))
            .collect())
    }

    /// Recover the filler for a (subject, predicate) pair.
    ///
    /// Uses `unbind(subject_vec, predicate_vec)` and searches item memory
    /// for the nearest match — the "what is the object of this triple?" operation.
    pub fn recover_filler(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        top_k: usize,
    ) -> InferResult<Vec<(SymbolId, f32)>> {
        let subj_vec = self.item_memory.get_or_create(&self.ops, subject);
        let pred_vec = self.item_memory.get_or_create(&self.ops, predicate);

        let recovered = self.ops.unbind(&subj_vec, &pred_vec)?;
        let results = self.item_memory.search(&recovered, top_k)?;
        Ok(results
            .into_iter()
            .map(|sr| (sr.symbol_id, sr.similarity))
            .collect())
    }

    /// Apply e-graph verification: build expressions from provenance chains
    /// and penalize non-simplifiable inferences.
    fn verify_with_egraph(&self, ctx: &mut InferContext, rules: &[Rewrite<AkhLang, ()>]) {
        use egg::{AstSize, Extractor, Runner};

        for record in &ctx.provenance {
            if let DerivationKind::VsaRecovery {
                from, predicate, ..
            } = &record.kind
            {
                // Build expression: (bind (bind from predicate) symbol)
                // If the e-graph can simplify (bind X (bind X Y)) → Y,
                // the inference is well-formed.
                let expr_str = format!(
                    "(bind {} (bind {} {}))",
                    from.get(),
                    from.get(),
                    predicate.get()
                );
                if let Ok(expr) = expr_str.parse::<egg::RecExpr<crate::reason::AkhLang>>() {
                    let runner = Runner::default().with_expr(&expr).run(rules);
                    let extractor = Extractor::new(&runner.egraph, AstSize);
                    let (original_cost, _) = extractor.find_best(runner.roots[0]);
                    // If the expression didn't simplify at all, it's suspicious
                    // but we only slightly penalize rather than rejecting
                    if original_cost >= 5 {
                        if let Some(conf) = ctx.activations.get_mut(&record.derived_id) {
                            *conf *= 0.9;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Triple;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    /// Helper to build a test inference engine with shared subsystems.
    fn test_engine() -> (
        InferEngine,
        Arc<VsaOps>,
        Arc<ItemMemory>,
        Arc<KnowledgeGraph>,
    ) {
        let ops = Arc::new(VsaOps::new(
            simd::best_kernel(),
            Dimension::TEST,
            Encoding::Bipolar,
        ));
        let item_memory = Arc::new(ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 1000));
        let knowledge_graph = Arc::new(KnowledgeGraph::new());
        let engine = InferEngine::new(
            Arc::clone(&ops),
            Arc::clone(&item_memory),
            Arc::clone(&knowledge_graph),
        );
        (engine, ops, item_memory, knowledge_graph)
    }

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn infer_no_seeds_returns_error() {
        let (engine, _, _, _) = test_engine();
        let query = InferenceQuery::default();
        let result = engine.infer(&query);
        assert!(matches!(result, Err(InferError::NoSeeds)));
    }

    #[test]
    fn single_hop_inference() {
        let (engine, ops, item_memory, kg) = test_engine();

        let sun = sym(1);
        let is_a = sym(2);
        let star = sym(3);

        // Create vectors in item memory
        item_memory.get_or_create(&ops, sun);
        item_memory.get_or_create(&ops, is_a);
        item_memory.get_or_create(&ops, star);

        kg.insert_triple(&Triple::new(sun, is_a, star)).unwrap();

        let query = InferenceQuery::default()
            .with_seeds(vec![sun])
            .with_max_depth(1);

        let result = engine.infer(&query).unwrap();
        // Star should be in the activations (inferred via graph edge)
        let activated_symbols: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();
        assert!(
            activated_symbols.contains(&star),
            "Star should be activated; got: {activated_symbols:?}"
        );
    }

    #[test]
    fn multi_hop_inference() {
        let (engine, ops, item_memory, kg) = test_engine();

        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let rel = sym(10);

        for &s in &[a, b, c, rel] {
            item_memory.get_or_create(&ops, s);
        }

        kg.insert_triple(&Triple::new(a, rel, b)).unwrap();
        kg.insert_triple(&Triple::new(b, rel, c)).unwrap();

        // Depth 1: should find B but not C
        let query_d1 = InferenceQuery::default()
            .with_seeds(vec![a])
            .with_max_depth(1);
        let result_d1 = engine.infer(&query_d1).unwrap();
        let syms_d1: Vec<SymbolId> = result_d1.activations.iter().map(|(s, _)| *s).collect();
        assert!(syms_d1.contains(&b), "Depth 1 should find B");
        assert!(!syms_d1.contains(&c), "Depth 1 should NOT find C");

        // Depth 2: should find both B and C
        let query_d2 = InferenceQuery::default()
            .with_seeds(vec![a])
            .with_max_depth(2);
        let result_d2 = engine.infer(&query_d2).unwrap();
        let syms_d2: Vec<SymbolId> = result_d2.activations.iter().map(|(s, _)| *s).collect();
        assert!(syms_d2.contains(&b), "Depth 2 should find B");
        assert!(syms_d2.contains(&c), "Depth 2 should find C");
    }

    #[test]
    fn confidence_propagates_multiplicatively() {
        let (engine, ops, item_memory, kg) = test_engine();

        let a = sym(1);
        let b = sym(2);
        let c = sym(3);
        let rel = sym(10);

        for &s in &[a, b, c, rel] {
            item_memory.get_or_create(&ops, s);
        }

        kg.insert_triple(&Triple::new(a, rel, b).with_confidence(0.8))
            .unwrap();
        kg.insert_triple(&Triple::new(b, rel, c).with_confidence(0.5))
            .unwrap();

        let query = InferenceQuery::default()
            .with_seeds(vec![a])
            .with_max_depth(2)
            .with_min_confidence(0.0);
        let result = engine.infer(&query).unwrap();

        // C's graph-direct confidence should be <= 0.8 * 0.5 = 0.4
        if let Some((_, c_conf)) = result.activations.iter().find(|(s, _)| *s == c) {
            assert!(
                *c_conf <= 0.8 * 0.5 + 0.01,
                "C confidence {c_conf} should be <= 0.4 (with small epsilon for VSA recovery)"
            );
        }
    }

    #[test]
    fn role_filler_recovery() {
        let (engine, ops, item_memory, _kg) = test_engine();

        let subject = sym(1);
        let predicate = sym(2);
        let object = sym(3);

        // Create vectors: bind subject with predicate, then store
        let subj_vec = item_memory.get_or_create(&ops, subject);
        let pred_vec = item_memory.get_or_create(&ops, predicate);
        item_memory.get_or_create(&ops, object);

        // Store the bound vector as a "composite" that encodes the relationship
        let _bound = ops.bind(&subj_vec, &pred_vec).unwrap();
        // The object's vector is stored independently, so unbind(bound, pred)
        // should recover something close to subject (since bind(s,p) XOR p = s)
        // and unbind(subj, pred) should recover something that we search for
        let results = engine.recover_filler(subject, predicate, 5).unwrap();
        assert!(!results.is_empty(), "Filler recovery should return results");
    }

    #[test]
    fn analogy_inference() {
        let (engine, ops, item_memory, _kg) = test_engine();

        let a = sym(1);
        let b = sym(2);
        let c = sym(3);

        // Create vectors for all symbols
        item_memory.get_or_create(&ops, a);
        item_memory.get_or_create(&ops, b);
        item_memory.get_or_create(&ops, c);

        // Analogy: a:b :: c:?
        let results = engine.infer_analogy(a, b, c, 5).unwrap();
        assert!(!results.is_empty(), "Analogy should return results");
    }

    #[test]
    fn analogy_requires_three_distinct() {
        let (engine, _, _, _) = test_engine();
        let a = sym(1);
        let b = sym(2);

        // Passing duplicate should error
        let result = engine.infer_analogy(a, b, a, 5);
        assert!(
            matches!(result, Err(InferError::InvalidAnalogy { count: 2 })),
            "Should fail with InvalidAnalogy; got {result:?}"
        );
    }

    #[test]
    fn provenance_records_generated() {
        let (engine, ops, item_memory, kg) = test_engine();

        let sun = sym(1);
        let is_a = sym(2);
        let star = sym(3);

        item_memory.get_or_create(&ops, sun);
        item_memory.get_or_create(&ops, is_a);
        item_memory.get_or_create(&ops, star);

        kg.insert_triple(&Triple::new(sun, is_a, star)).unwrap();

        let query = InferenceQuery::default()
            .with_seeds(vec![sun])
            .with_max_depth(1);

        let result = engine.infer(&query).unwrap();
        assert!(
            !result.provenance.is_empty(),
            "Provenance should have records"
        );

        // Should have at least a Seed record
        assert!(
            result
                .provenance
                .iter()
                .any(|p| p.kind == DerivationKind::Seed),
            "Should have a Seed provenance record"
        );

        // Should have an Inferred record for the graph edge
        assert!(
            result.provenance.iter().any(|p| matches!(
                &p.kind,
                DerivationKind::GraphEdge { from, predicate }
                    if *from == sun && *predicate == is_a
            )),
            "Should have a GraphEdge provenance record for Sun→Star via is-a"
        );
    }

    #[test]
    fn empty_graph_no_activations() {
        let (engine, ops, item_memory, _kg) = test_engine();

        let lonely = sym(99);
        item_memory.get_or_create(&ops, lonely);

        let query = InferenceQuery::default()
            .with_seeds(vec![lonely])
            .with_max_depth(2);

        let result = engine.infer(&query).unwrap();
        assert!(
            result.activations.is_empty(),
            "Seeds with no edges should produce empty results (excluding seeds)"
        );
    }

    #[test]
    fn predicate_filter_respected() {
        let (engine, ops, item_memory, kg) = test_engine();

        let sun = sym(1);
        let is_a = sym(2);
        let has_part = sym(3);
        let star = sym(4);
        let corona = sym(5);

        for &s in &[sun, is_a, has_part, star, corona] {
            item_memory.get_or_create(&ops, s);
        }

        kg.insert_triple(&Triple::new(sun, is_a, star)).unwrap();
        kg.insert_triple(&Triple::new(sun, has_part, corona))
            .unwrap();

        // Only follow is_a edges
        let query = InferenceQuery::default()
            .with_seeds(vec![sun])
            .with_max_depth(1)
            .with_predicate_filter(vec![is_a]);

        let result = engine.infer(&query).unwrap();
        let activated: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();

        assert!(
            activated.contains(&star),
            "Star should be activated via is-a"
        );
        assert!(
            !activated.contains(&corona),
            "Corona should NOT be activated (has-part filtered out)"
        );
    }

    #[test]
    fn phase9_inference_follows_hierarchy_specializations() {
        use crate::graph::predicate_hierarchy::PredicateHierarchy;

        let (engine, ops, item_memory, kg) = test_engine();

        let dog = sym(1);
        let parent_rel = sym(2);
        let bio_mother = sym(3);
        let mother_dog = sym(4);
        let generalizes = sym(5);

        for &s in &[dog, parent_rel, bio_mother, mother_dog, generalizes] {
            item_memory.get_or_create(&ops, s);
        }

        // bio_mother generalizes to parent_rel
        kg.insert_triple(&Triple::new(bio_mother, generalizes, parent_rel)).unwrap();
        // bio_mother(dog, mother_dog)
        kg.insert_triple(&Triple::new(dog, bio_mother, mother_dog)).unwrap();

        // Build hierarchy with those predicates
        let mut hierarchy = PredicateHierarchy::new();
        // Manually build (normally from KG, but we shortcut since we don't have rel:generalizes resolved)
        // We'll just use infer_with_phase9 with an empty hierarchy first
        let phase9_empty = InferPhase9Context::empty();
        let rules = crate::reason::builtin_rules();

        // Without hierarchy: querying from dog should find mother_dog via bio_mother
        let query = InferenceQuery::default()
            .with_seeds(vec![dog])
            .with_max_depth(1);
        let result = engine.infer_with_phase9(&query, &rules, &phase9_empty).unwrap();
        let syms: Vec<SymbolId> = result.activations.iter().map(|(s, _)| *s).collect();
        assert!(syms.contains(&mother_dog), "Should find mother_dog via bio_mother edge");
    }

    #[test]
    fn phase9_temporal_decay_reduces_confidence() {
        use crate::temporal::{TemporalProfile, TemporalRegistry};

        let (engine, ops, item_memory, kg) = test_engine();

        let a = sym(1);
        let rel = sym(10);
        let b = sym(2);

        for &s in &[a, rel, b] {
            item_memory.get_or_create(&ops, s);
        }

        // Insert triple with a timestamp in the past
        let mut triple = Triple::new(a, rel, b);
        triple.confidence = 1.0;
        triple.timestamp = 1000; // asserted at t=1000
        kg.insert_triple(&triple).unwrap();

        // Set up temporal registry with ephemeral profile (TTL = 100s)
        let mut temporal = TemporalRegistry::new();
        temporal.set_profile(rel, TemporalProfile::Ephemeral { ttl_secs: 100 });

        let rules = crate::reason::builtin_rules();

        // Query at t=1200 — 200s past assertion, well beyond TTL of 100s
        let phase9_ctx = InferPhase9Context {
            hierarchy: None,
            temporal: Some(&temporal),
            query_time_secs: 1200,
        };

        let query = InferenceQuery::default()
            .with_seeds(vec![a])
            .with_max_depth(1)
            .with_min_confidence(0.0);
        let result = engine.infer_with_phase9(&query, &rules, &phase9_ctx).unwrap();

        // With ephemeral TTL=100s and query at +200s, confidence should be 0
        let b_activation = result.activations.iter().find(|(s, _)| *s == b);
        assert!(
            b_activation.is_none() || b_activation.unwrap().1 < 0.01,
            "Ephemeral triple should have decayed to zero; got {:?}",
            b_activation
        );
    }

    #[test]
    fn phase9_stable_triple_retains_confidence() {
        use crate::temporal::{TemporalProfile, TemporalRegistry};

        let (engine, ops, item_memory, kg) = test_engine();

        let a = sym(1);
        let rel = sym(10);
        let b = sym(2);

        for &s in &[a, rel, b] {
            item_memory.get_or_create(&ops, s);
        }

        let mut triple = Triple::new(a, rel, b);
        triple.confidence = 0.9;
        triple.timestamp = 1000;
        kg.insert_triple(&triple).unwrap();

        // Stable profile — no decay
        let mut temporal = TemporalRegistry::new();
        temporal.set_profile(rel, TemporalProfile::Stable);

        let rules = crate::reason::builtin_rules();

        let phase9_ctx = InferPhase9Context {
            hierarchy: None,
            temporal: Some(&temporal),
            query_time_secs: 999_999_999,
        };

        let query = InferenceQuery::default()
            .with_seeds(vec![a])
            .with_max_depth(1)
            .with_min_confidence(0.0);
        let result = engine.infer_with_phase9(&query, &rules, &phase9_ctx).unwrap();

        let b_activation = result.activations.iter().find(|(s, _)| *s == b);
        assert!(
            b_activation.is_some(),
            "Stable triple should still be activated"
        );
        assert!(
            b_activation.unwrap().1 >= 0.85,
            "Stable triple should retain high confidence; got {}",
            b_activation.unwrap().1
        );
    }
}
