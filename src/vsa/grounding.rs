//! Semantic symbol grounding via graph-derived embeddings.
//!
//! Recomputes symbol hypervectors from their KG neighborhood so that
//! related symbols (e.g., Dog and Cat, both `is-a Mammal`) end up with
//! correlated representations instead of random ~0.5 similarity.

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::HyperVec;
use super::item_memory::ItemMemory;
use super::ops::{VsaOps, VsaResult};

/// Configuration for the grounding process.
#[derive(Debug, Clone)]
pub struct GroundingConfig {
    /// Number of neighborhood aggregation rounds (default: 3).
    /// After k rounds, symbols k hops apart share representation.
    pub rounds: usize,
    /// Blend ratio for neighbor influence (default: 0.3).
    /// `new_vec = bundle(old_vec, permute(neighborhood, 1))` weighted by this.
    pub neighbor_weight: f32,
    /// Minimum edge confidence to include in neighborhood (default: 0.5).
    pub min_confidence: f32,
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            rounds: 3,
            neighbor_weight: 0.3,
            min_confidence: 0.5,
        }
    }
}

/// Result of a grounding operation.
#[derive(Debug)]
pub struct GroundingResult {
    /// Number of symbols that were updated.
    pub symbols_updated: usize,
    /// Number of rounds performed.
    pub rounds_completed: usize,
}

/// Recompute a symbol's vector from its graph neighborhood.
///
/// Gathers all neighbors (subjects/objects of triples involving the symbol),
/// bundles their vectors, and blends the result with the symbol's current vector.
/// Also includes role-aware grounding: for `(S, P, O)`, creates `bind(P_vec, O_vec)`
/// and includes it in S's neighborhood bundle.
pub fn ground_symbol(
    symbol: SymbolId,
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
    config: &GroundingConfig,
) -> VsaResult<HyperVec> {
    let current_vec = item_memory.get_or_create(ops, symbol);

    // Gather neighbor vectors from outgoing triples
    let from_triples = engine.triples_from(symbol);
    let to_triples = engine.triples_to(symbol);

    let mut neighbor_vecs: Vec<HyperVec> = Vec::new();

    for triple in &from_triples {
        if triple.confidence < config.min_confidence {
            continue;
        }
        // Direct neighbor: the object
        let obj_vec = item_memory.get_or_create(ops, triple.object);
        neighbor_vecs.push(obj_vec);

        // Role-aware: bind(predicate, object) captures the relationship type
        let pred_vec = item_memory.get_or_create(ops, triple.predicate);
        let obj_vec2 = item_memory.get_or_create(ops, triple.object);
        if let Ok(role_filler) = ops.bind(&pred_vec, &obj_vec2) {
            neighbor_vecs.push(role_filler);
        }
    }

    for triple in &to_triples {
        if triple.confidence < config.min_confidence {
            continue;
        }
        // Direct neighbor: the subject
        let subj_vec = item_memory.get_or_create(ops, triple.subject);
        neighbor_vecs.push(subj_vec);

        // Role-aware: bind(predicate, subject) for incoming edges
        let pred_vec = item_memory.get_or_create(ops, triple.predicate);
        let subj_vec2 = item_memory.get_or_create(ops, triple.subject);
        if let Ok(role_filler) = ops.bind(&pred_vec, &subj_vec2) {
            neighbor_vecs.push(role_filler);
        }
    }

    if neighbor_vecs.is_empty() {
        return Ok(current_vec);
    }

    // Bundle all neighbor vectors
    let refs: Vec<&HyperVec> = neighbor_vecs.iter().collect();
    let neighborhood = ops.bundle(&refs)?;

    // Permute neighborhood to avoid self-reinforcement
    let shifted_neighborhood = ops.permute(&neighborhood, 1);

    // Blend via probabilistic bit selection.
    // For each bit position, with probability `neighbor_weight` take the
    // neighborhood bit, otherwise keep the self bit. This is implemented
    // deterministically using a seeded RNG based on the symbol to ensure
    // reproducibility.
    blend_vectors(
        &current_vec,
        &shifted_neighborhood,
        config.neighbor_weight,
        symbol,
    )
}

/// Probabilistic bit-level blending for binary hypervectors.
///
/// For each bit position, takes the `other` bit with probability `weight`,
/// otherwise keeps the `base` bit. Uses a deterministic PRNG seeded from the
/// symbol ID so the result is reproducible.
fn blend_vectors(
    base: &HyperVec,
    other: &HyperVec,
    weight: f32,
    symbol: SymbolId,
) -> VsaResult<HyperVec> {
    let base_data = base.data();
    let other_data = other.data();

    // Use the same dimension check as VsaOps — if mismatch, bundle will fail anyway
    if base_data.len() != other_data.len() {
        return Err(crate::error::VsaError::DimensionMismatch {
            expected: base.dim().0,
            actual: other.dim().0,
        });
    }

    let mut result = base_data.to_vec();

    // Deterministic PRNG seeded from symbol ID (xorshift64)
    let mut state = u64::from(symbol.get()).wrapping_mul(0x517cc1b727220a95) | 1;
    let threshold = (weight * 256.0) as u64;

    for (i, byte) in result.iter_mut().enumerate() {
        let other_byte = other_data[i];
        // Generate a mask where each bit is 1 with probability `weight`
        let mut mask: u8 = 0;
        for bit in 0..8u8 {
            // xorshift64 step
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if (state & 0xFF) < threshold {
                mask |= 1 << bit;
            }
        }
        // Where mask is 1, take from other; where 0, keep base
        *byte = (*byte & !mask) | (other_byte & mask);
    }

    Ok(HyperVec::from_raw(result, base.dim(), base.encoding()))
}

/// Ground all symbols using iterative neighborhood aggregation.
///
/// Each round, every symbol's vector is updated based on its graph neighbors.
/// After k rounds, symbols k hops apart share representation.
pub fn ground_all(
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
    config: &GroundingConfig,
) -> VsaResult<GroundingResult> {
    let all_symbols = engine.all_symbols();
    let symbol_ids: Vec<SymbolId> = all_symbols.iter().map(|m| m.id).collect();

    // Ensure all symbols have initial vectors
    for &sym in &symbol_ids {
        item_memory.get_or_create(ops, sym);
    }

    let mut total_updated = 0usize;

    for _round in 0..config.rounds {
        let mut round_updated = 0usize;

        for &sym in &symbol_ids {
            match ground_symbol(sym, engine, ops, item_memory, config) {
                Ok(new_vec) => {
                    item_memory.insert(sym, new_vec);
                    round_updated += 1;
                }
                Err(_) => continue,
            }
        }

        total_updated = total_updated.max(round_updated);

        // Early termination: if no symbols have neighbors, no point continuing
        if round_updated == 0 {
            return Ok(GroundingResult {
                symbols_updated: 0,
                rounds_completed: _round + 1,
            });
        }
    }

    Ok(GroundingResult {
        symbols_updated: total_updated,
        rounds_completed: config.rounds,
    })
}

/// Encode a text description as a hypervector by looking up or creating
/// symbols for each word and bundling them.
///
/// Used by tool semantics and goal encoding.
pub fn encode_text_as_vector(
    text: &str,
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
) -> VsaResult<HyperVec> {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();

    if words.is_empty() {
        // Return a random vector for empty text
        let mut rng = rand::thread_rng();
        return Ok(ops.random(&mut rng));
    }

    let mut vecs: Vec<HyperVec> = Vec::new();
    for word in &words {
        let lower = word.to_lowercase();
        // Look up existing symbols only — don't create new KG entities as
        // a side effect of encoding text. Words without a matching symbol
        // get a deterministic vector from the encoding function.
        if let Ok(sym_id) = engine.lookup_symbol(&lower) {
            vecs.push(item_memory.get_or_create(ops, sym_id));
        } else {
            // Create a deterministic vector from the word hash, without
            // polluting the KG with temporary entities.
            use crate::vsa::encode::encode_symbol;
            let word_hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                lower.hash(&mut hasher);
                hasher.finish()
            };
            // Create a synthetic SymbolId from hash (high bits to avoid collisions)
            let synthetic_id = crate::symbol::SymbolId::new(word_hash | (1u64 << 63))
                .expect("non-zero hash with high bit set");
            vecs.push(encode_symbol(ops, synthetic_id));
        }
    }

    if vecs.is_empty() {
        let mut rng = rand::thread_rng();
        return Ok(ops.random(&mut rng));
    }

    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs)
}

/// Bundle a list of concept labels into a single hypervector.
///
/// Resolves each label to a symbol (creating if needed) and bundles
/// their grounded vectors.
pub fn bundle_symbols(
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
    labels: &[&str],
) -> VsaResult<HyperVec> {
    let mut vecs: Vec<HyperVec> = Vec::new();
    for label in labels {
        if let Ok(sym_id) = engine.resolve_or_create_entity(label) {
            vecs.push(item_memory.get_or_create(ops, sym_id));
        }
    }

    if vecs.is_empty() {
        let mut rng = rand::thread_rng();
        return Ok(ops.random(&mut rng));
    }

    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
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
    fn grounding_increases_related_similarity() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap().id;
        let cat = engine.create_symbol(SymbolKind::Entity, "Cat").unwrap().id;
        let mammal = engine
            .create_symbol(SymbolKind::Entity, "Mammal")
            .unwrap()
            .id;
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap()
            .id;

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();
        engine.add_triple(&Triple::new(cat, is_a, mammal)).unwrap();

        // Before grounding: Dog and Cat are random (~0.5)
        let dog_vec_before = im.get_or_create(ops, dog);
        let cat_vec_before = im.get_or_create(ops, cat);
        let sim_before = ops.similarity(&dog_vec_before, &cat_vec_before).unwrap();

        // After grounding: Dog and Cat should be more similar
        let config = GroundingConfig::default();
        let result = ground_all(&engine, ops, im, &config).unwrap();
        assert!(result.symbols_updated > 0);

        let dog_vec_after = im.get(dog).unwrap();
        let cat_vec_after = im.get(cat).unwrap();
        let sim_after = ops.similarity(&dog_vec_after, &cat_vec_after).unwrap();

        assert!(
            sim_after > sim_before,
            "Grounding should increase similarity: before={sim_before:.3}, after={sim_after:.3}"
        );
    }

    #[test]
    fn unrelated_symbols_stay_dissimilar() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap().id;
        let paris = engine
            .create_symbol(SymbolKind::Entity, "Paris")
            .unwrap()
            .id;
        let mammal = engine
            .create_symbol(SymbolKind::Entity, "Mammal")
            .unwrap()
            .id;
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap()
            .id;
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap()
            .id;
        let capital_of = engine
            .create_symbol(SymbolKind::Relation, "capital-of")
            .unwrap()
            .id;

        engine.add_triple(&Triple::new(dog, is_a, mammal)).unwrap();
        engine
            .add_triple(&Triple::new(paris, capital_of, france))
            .unwrap();

        let config = GroundingConfig::default();
        ground_all(&engine, ops, im, &config).unwrap();

        let dog_vec = im.get(dog).unwrap();
        let paris_vec = im.get(paris).unwrap();
        let sim = ops.similarity(&dog_vec, &paris_vec).unwrap();

        // Unrelated symbols should stay near random (< 0.6)
        assert!(
            sim < 0.65,
            "Unrelated symbols should stay dissimilar: sim={sim:.3}"
        );
    }

    #[test]
    fn multi_round_grounding_converges() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        let a = engine.create_symbol(SymbolKind::Entity, "A").unwrap().id;
        let b = engine.create_symbol(SymbolKind::Entity, "B").unwrap().id;
        let c = engine.create_symbol(SymbolKind::Entity, "C").unwrap().id;
        let rel = engine.create_symbol(SymbolKind::Relation, "r").unwrap().id;

        engine.add_triple(&Triple::new(a, rel, b)).unwrap();
        engine.add_triple(&Triple::new(b, rel, c)).unwrap();

        let config = GroundingConfig {
            rounds: 3,
            ..Default::default()
        };
        let result = ground_all(&engine, ops, im, &config).unwrap();

        assert_eq!(result.rounds_completed, 3);
        assert!(result.symbols_updated > 0);
    }

    #[test]
    fn encode_text_produces_vector() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        let vec = encode_text_as_vector("search knowledge triples", &engine, ops, im).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn bundle_symbols_produces_vector() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        let vec = bundle_symbols(&engine, ops, im, &["query", "search", "knowledge"]).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }
}
