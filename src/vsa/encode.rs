//! Symbol-to-vector encoding.
//!
//! Maps symbols into hypervector space using deterministic seeded random
//! generation, ensuring the same symbol always maps to the same vector.

use rand::SeedableRng;

use crate::symbol::SymbolId;

use super::ops::VsaOps;
use super::HyperVec;

/// Encode a symbol ID into a hypervector using deterministic seeded randomness.
///
/// The same `SymbolId` always produces the same `HyperVec`, regardless of
/// when or where it's computed. The seed is derived from the symbol's raw ID.
pub fn encode_symbol(ops: &VsaOps, symbol: SymbolId) -> HyperVec {
    let mut rng = rand::rngs::StdRng::seed_from_u64(symbol.get());
    ops.random(&mut rng)
}

/// Encode a sequence of symbols as a permutation-bound chain.
///
/// This captures order information: `encode_sequence([A, B, C])` produces
/// `permute(A, 2) ⊕ permute(B, 1) ⊕ C` — each element is permuted by its
/// distance from the end of the sequence.
pub fn encode_sequence(ops: &VsaOps, symbols: &[SymbolId]) -> Option<HyperVec> {
    if symbols.is_empty() {
        return None;
    }
    let n = symbols.len();
    let vecs: Vec<HyperVec> = symbols
        .iter()
        .enumerate()
        .map(|(i, &sym)| {
            let base = encode_symbol(ops, sym);
            let shift = n - 1 - i;
            if shift > 0 {
                ops.permute(&base, shift)
            } else {
                base
            }
        })
        .collect();

    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs).ok()
}

/// Encode a role-filler pair: `bind(role_vec, filler_vec)`.
///
/// This is the standard VSA pattern for structured knowledge:
/// "color" ⊗ "blue" represents "the color is blue".
pub fn encode_role_filler(
    ops: &VsaOps,
    role: SymbolId,
    filler: SymbolId,
) -> Result<HyperVec, crate::error::VsaError> {
    let role_vec = encode_symbol(ops, role);
    let filler_vec = encode_symbol(ops, filler);
    ops.bind(&role_vec, &filler_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    #[test]
    fn deterministic_encoding() {
        let ops = test_ops();
        let id = SymbolId::new(42).unwrap();
        let v1 = encode_symbol(&ops, id);
        let v2 = encode_symbol(&ops, id);
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_symbols_different_vectors() {
        let ops = test_ops();
        let a = encode_symbol(&ops, SymbolId::new(1).unwrap());
        let b = encode_symbol(&ops, SymbolId::new(2).unwrap());
        let sim = ops.similarity(&a, &b).unwrap();
        assert!(sim < 0.6, "different symbols should be dissimilar: sim={sim}");
    }

    #[test]
    fn role_filler_recoverable() {
        let ops = test_ops();
        let role = SymbolId::new(10).unwrap();
        let filler = SymbolId::new(20).unwrap();

        let bound = encode_role_filler(&ops, role, filler).unwrap();
        let role_vec = encode_symbol(&ops, role);

        // Unbind with role to recover filler
        let recovered = ops.unbind(&bound, &role_vec).unwrap();
        let filler_vec = encode_symbol(&ops, filler);
        let sim = ops.similarity(&recovered, &filler_vec).unwrap();
        assert!((sim - 1.0).abs() < 0.001, "recovered filler sim={sim}");
    }

    #[test]
    fn sequence_encoding_order_matters() {
        let ops = test_ops();
        let a = SymbolId::new(1).unwrap();
        let b = SymbolId::new(2).unwrap();
        let c = SymbolId::new(3).unwrap();

        let seq_abc = encode_sequence(&ops, &[a, b, c]).unwrap();
        let seq_cba = encode_sequence(&ops, &[c, b, a]).unwrap();

        let sim = ops.similarity(&seq_abc, &seq_cba).unwrap();
        // Different orderings should produce different vectors
        assert!(sim < 0.7, "different orders should differ: sim={sim}");
    }

    #[test]
    fn sequence_empty_is_none() {
        let ops = test_ops();
        assert!(encode_sequence(&ops, &[]).is_none());
    }
}
