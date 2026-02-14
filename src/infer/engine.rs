//! Inference engine: spreading activation + VSA recovery.
//!
//! Combines graph-guided spreading activation with VSA bind/unbind
//! operations to infer new knowledge from existing triples.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use egg::Rewrite;

use crate::error::InferError;
use crate::graph::index::KnowledgeGraph;
use crate::reason::AkhLang;
use crate::symbol::SymbolId;
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
}
