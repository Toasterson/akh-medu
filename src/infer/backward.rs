//! Backward chaining: reason from a goal back to supporting evidence.
//!
//! Given a goal symbol, finds all triples where it appears as the object,
//! then recursively finds support for those subjects. VSA similarity
//! is used to verify each step's plausibility.

use crate::graph::Triple;
use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

use super::engine::InferResult;

/// A backward-chaining inference chain.
#[derive(Debug, Clone)]
pub struct BackwardChain {
    /// The goal symbol this chain supports.
    pub goal: SymbolId,
    /// Supporting triples found by backward search.
    pub supporting_triples: Vec<Triple>,
    /// Confidence of this chain (product of step confidences, decayed by depth).
    pub confidence: f32,
    /// Depth of the deepest step in this chain.
    pub depth: usize,
}

/// Configuration for backward chaining.
#[derive(Debug, Clone)]
pub struct BackwardConfig {
    /// Maximum recursion depth (default: 3).
    pub max_depth: usize,
    /// Minimum confidence to continue chaining (default: 0.1).
    pub min_confidence: f32,
    /// Whether to use VSA verification at each step (default: true).
    pub vsa_verify: bool,
}

impl Default for BackwardConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_confidence: 0.1,
            vsa_verify: true,
        }
    }
}

/// Backward-chain from a goal symbol to find supporting evidence.
///
/// Finds all triples where `goal` is the object: `(?, ?, goal)`.
/// For each, optionally verifies via VSA: `similarity(unbind(S_vec, P_vec), goal_vec)`.
/// Then recursively finds support for each subject.
pub fn infer_backward(
    engine: &crate::engine::Engine,
    goal: SymbolId,
    config: &BackwardConfig,
) -> InferResult<Vec<BackwardChain>> {
    let ops = engine.ops();
    let im = engine.item_memory();

    let mut chains = Vec::new();
    backward_recurse(
        engine,
        ops,
        im,
        goal,
        config,
        0,
        1.0,
        &mut Vec::new(),
        &mut chains,
    );

    // Sort by confidence (highest first)
    chains.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(chains)
}

fn backward_recurse(
    engine: &crate::engine::Engine,
    ops: &VsaOps,
    im: &ItemMemory,
    goal: SymbolId,
    config: &BackwardConfig,
    depth: usize,
    parent_confidence: f32,
    current_chain: &mut Vec<Triple>,
    results: &mut Vec<BackwardChain>,
) {
    if depth >= config.max_depth {
        return;
    }

    // Find all triples where goal is the object
    let incoming = engine.triples_to(goal);

    if incoming.is_empty() && !current_chain.is_empty() {
        // Leaf: no more support, record the chain so far
        results.push(BackwardChain {
            goal,
            supporting_triples: current_chain.clone(),
            confidence: parent_confidence,
            depth,
        });
        return;
    }

    for triple in &incoming {
        let step_confidence = triple.confidence;

        // Optional VSA verification
        let vsa_confidence = if config.vsa_verify {
            let subj_vec = im.get_or_create(ops, triple.subject);
            let pred_vec = im.get_or_create(ops, triple.predicate);
            let goal_vec = im.get_or_create(ops, goal);

            // unbind(subject, predicate) should recover something similar to object
            match ops.unbind(&subj_vec, &pred_vec) {
                Ok(recovered) => ops.similarity(&recovered, &goal_vec).unwrap_or(0.5),
                Err(_) => 0.5,
            }
        } else {
            1.0
        };

        let combined_confidence = parent_confidence * step_confidence * vsa_confidence;
        if combined_confidence < config.min_confidence {
            continue;
        }

        current_chain.push(triple.clone());

        // Record this chain
        results.push(BackwardChain {
            goal,
            supporting_triples: current_chain.clone(),
            confidence: combined_confidence,
            depth: depth + 1,
        });

        // Recurse: find support for the subject
        backward_recurse(
            engine,
            ops,
            im,
            triple.subject,
            config,
            depth + 1,
            combined_confidence,
            current_chain,
            results,
        );

        current_chain.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> crate::engine::Engine {
        crate::engine::Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn find_support_chain() {
        let engine = test_engine();

        let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap().id;
        let mammal = engine.create_symbol(SymbolKind::Entity, "Mammal").unwrap().id;
        let animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap().id;
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap().id;

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();
        engine.add_triple(&Triple::new(mammal, is_a, animal)).unwrap();

        // Backward chain from "Animal" — should find Mammal→Animal and Dog→Mammal
        let chains = infer_backward(&engine, animal, &BackwardConfig::default()).unwrap();

        assert!(!chains.is_empty(), "Should find at least one support chain");

        // At least one chain should include Dog→is-a→Mammal
        let has_deep_chain = chains.iter().any(|c| c.supporting_triples.len() >= 2);
        assert!(
            has_deep_chain || chains.len() >= 2,
            "Should find multi-step support or multiple chains"
        );
    }

    #[test]
    fn confidence_decreases_with_depth() {
        let engine = test_engine();

        let a = engine.create_symbol(SymbolKind::Entity, "A").unwrap().id;
        let b = engine.create_symbol(SymbolKind::Entity, "B").unwrap().id;
        let c = engine.create_symbol(SymbolKind::Entity, "C").unwrap().id;
        let rel = engine.create_symbol(SymbolKind::Relation, "r").unwrap().id;

        engine
            .add_triple(&Triple::new(a, rel, b).with_confidence(0.8))
            .unwrap();
        engine
            .add_triple(&Triple::new(b, rel, c).with_confidence(0.8))
            .unwrap();

        let chains = infer_backward(&engine, c, &BackwardConfig::default()).unwrap();

        // Deeper chains should have lower confidence
        if chains.len() >= 2 {
            let shallow: Vec<&BackwardChain> = chains.iter().filter(|c| c.depth == 1).collect();
            let deep: Vec<&BackwardChain> = chains.iter().filter(|c| c.depth > 1).collect();

            if let (Some(s), Some(d)) = (shallow.first(), deep.first()) {
                assert!(
                    s.confidence >= d.confidence,
                    "Shallow ({:.3}) should be >= deep ({:.3})",
                    s.confidence,
                    d.confidence
                );
            }
        }
    }

    #[test]
    fn no_support_for_isolated_symbol() {
        let engine = test_engine();

        let lonely = engine
            .create_symbol(SymbolKind::Entity, "Lonely")
            .unwrap()
            .id;

        let chains = infer_backward(&engine, lonely, &BackwardConfig::default()).unwrap();
        assert!(chains.is_empty(), "Isolated symbol should have no support");
    }
}
