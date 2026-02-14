//! VSA operations: bind, bundle, permute, similarity.
//!
//! These are the core algebraic operations of the Vector Symbolic Architecture.
//! Each operation is dispatched through the SIMD kernel for maximum performance.

use crate::error::VsaError;
use crate::simd::VsaKernel;

use super::{Dimension, Encoding, HyperVec};

/// Result type for VSA operations.
pub type VsaResult<T> = std::result::Result<T, VsaError>;

/// Ensure two hypervectors have matching dimensions and encoding.
fn check_compatible(a: &HyperVec, b: &HyperVec) -> VsaResult<()> {
    if a.dim() != b.dim() {
        return Err(VsaError::DimensionMismatch {
            expected: a.dim().0,
            actual: b.dim().0,
        });
    }
    if a.encoding() != b.encoding() {
        return Err(VsaError::UnsupportedEncoding {
            encoding: format!("mixed encodings: {} and {}", a.encoding(), b.encoding()),
        });
    }
    Ok(())
}

/// VSA operations backed by a SIMD kernel.
pub struct VsaOps {
    kernel: Box<dyn VsaKernel>,
    dim: Dimension,
    encoding: Encoding,
}

impl VsaOps {
    /// Create VSA operations with the given kernel, dimension, and encoding.
    pub fn new(kernel: Box<dyn VsaKernel>, dim: Dimension, encoding: Encoding) -> Self {
        Self {
            kernel,
            dim,
            encoding,
        }
    }

    /// The dimension these ops work with.
    pub fn dim(&self) -> Dimension {
        self.dim
    }

    /// The encoding these ops work with.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// The SIMD instruction set level this kernel uses.
    pub fn isa_level(&self) -> crate::simd::IsaLevel {
        self.kernel.isa_level()
    }

    /// Generate a random hypervector using the given RNG.
    pub fn random(&self, rng: &mut impl rand::Rng) -> HyperVec {
        let byte_len = self.dim.binary_byte_len();
        let mut data = vec![0u8; byte_len];
        rng.fill_bytes(&mut data);
        // Mask out unused trailing bits
        let used_bits = self.dim.0 % 8;
        if used_bits != 0 {
            if let Some(last) = data.last_mut() {
                *last &= (1u8 << used_bits) - 1;
            }
        }
        HyperVec::from_raw(data, self.dim, self.encoding)
    }

    /// Bind two hypervectors (XOR for bipolar).
    ///
    /// Binding creates a representation that is dissimilar to both inputs
    /// — it's the VSA equivalent of variable binding or role-filler pairing.
    pub fn bind(&self, a: &HyperVec, b: &HyperVec) -> VsaResult<HyperVec> {
        check_compatible(a, b)?;
        let mut out = vec![0u8; a.byte_len()];
        self.kernel.xor_bind(a.data(), b.data(), &mut out);
        Ok(HyperVec::from_raw(out, a.dim(), a.encoding()))
    }

    /// Unbind (same as bind for XOR — XOR is its own inverse).
    pub fn unbind(&self, bound: &HyperVec, key: &HyperVec) -> VsaResult<HyperVec> {
        self.bind(bound, key)
    }

    /// Bundle multiple hypervectors (majority vote for bipolar).
    ///
    /// The result is similar to all inputs — it represents a set/superposition.
    /// Uses an i8 accumulator for counting, then thresholds at 0.
    pub fn bundle(&self, vectors: &[&HyperVec]) -> VsaResult<HyperVec> {
        if vectors.is_empty() {
            return Err(VsaError::EmptyBundle);
        }
        let dim = vectors[0].dim();
        let encoding = vectors[0].encoding();
        for v in &vectors[1..] {
            if v.dim() != dim {
                return Err(VsaError::DimensionMismatch {
                    expected: dim.0,
                    actual: v.dim().0,
                });
            }
        }

        // Accumulate in i16 to avoid saturation for large bundles
        let n_components = dim.0;
        let mut acc = vec![0i16; n_components];

        for &v in vectors {
            for i in 0..n_components {
                let bit = v.get_bit(i);
                acc[i] += if bit { 1 } else { -1 };
            }
        }

        // Threshold: positive → 1, negative → 0, tie → random (break with bit position parity)
        let mut result = HyperVec::zero(dim, encoding);
        for i in 0..n_components {
            let val = acc[i] > 0 || (acc[i] == 0 && i % 2 == 0);
            result.set_bit(i, val);
        }

        Ok(result)
    }

    /// Permute a hypervector by a cyclic shift.
    ///
    /// Permutation creates a representation that is dissimilar to the input
    /// — useful for encoding sequence/order information.
    pub fn permute(&self, v: &HyperVec, shift: usize) -> HyperVec {
        let mut out = vec![0u8; v.byte_len()];
        self.kernel.permute(v.data(), shift, &mut out);
        HyperVec::from_raw(out, v.dim(), v.encoding())
    }

    /// Compute similarity between two hypervectors.
    ///
    /// For bipolar encoding, returns normalized Hamming similarity in `[0.0, 1.0]`
    /// where 1.0 means identical and 0.5 means uncorrelated (random).
    pub fn similarity(&self, a: &HyperVec, b: &HyperVec) -> VsaResult<f32> {
        check_compatible(a, b)?;
        let hamming = self.kernel.hamming_distance(a.data(), b.data());
        let total_bits = a.dim().0 as f32;
        Ok(1.0 - (hamming as f32 / total_bits))
    }

    /// Compute cosine similarity using i8 interpretation.
    ///
    /// Interprets each bit as +1 or -1 and computes the cosine.
    pub fn cosine_similarity(&self, a: &HyperVec, b: &HyperVec) -> VsaResult<f32> {
        check_compatible(a, b)?;
        // Convert to i8 bipolar representation for cosine
        let n = a.dim().0;
        let a_i8: Vec<i8> = (0..n)
            .map(|i| if a.get_bit(i) { 1i8 } else { -1i8 })
            .collect();
        let b_i8: Vec<i8> = (0..n)
            .map(|i| if b.get_bit(i) { 1i8 } else { -1i8 })
            .collect();
        Ok(self.kernel.cosine_similarity_i8(&a_i8, &b_i8))
    }
}

impl std::fmt::Debug for VsaOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VsaOps")
            .field("dim", &self.dim)
            .field("encoding", &self.encoding)
            .field("isa", &self.kernel.isa_level())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use rand::SeedableRng;

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn random_vectors_are_roughly_uncorrelated() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let b = ops.random(&mut rng);
        let sim = ops.similarity(&a, &b).unwrap();
        // Random vectors should have similarity ~0.5 ± noise
        assert!(
            sim > 0.4 && sim < 0.6,
            "similarity was {sim}, expected ~0.5"
        );
    }

    #[test]
    fn self_similarity_is_one() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let sim = ops.similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn bind_is_dissimilar_to_inputs() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let b = ops.random(&mut rng);
        let bound = ops.bind(&a, &b).unwrap();

        let sim_a = ops.similarity(&bound, &a).unwrap();
        let sim_b = ops.similarity(&bound, &b).unwrap();
        // Bound vector should be roughly uncorrelated with both inputs
        assert!(sim_a > 0.4 && sim_a < 0.6, "sim_a={sim_a}");
        assert!(sim_b > 0.4 && sim_b < 0.6, "sim_b={sim_b}");
    }

    #[test]
    fn bind_is_invertible() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let b = ops.random(&mut rng);
        let bound = ops.bind(&a, &b).unwrap();
        let recovered = ops.unbind(&bound, &a).unwrap();

        let sim = ops.similarity(&recovered, &b).unwrap();
        assert!((sim - 1.0).abs() < 0.001, "sim={sim}, expected 1.0");
    }

    #[test]
    fn bundle_is_similar_to_inputs() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let b = ops.random(&mut rng);
        let c = ops.random(&mut rng);
        let bundled = ops.bundle(&[&a, &b, &c]).unwrap();

        let sim_a = ops.similarity(&bundled, &a).unwrap();
        let sim_b = ops.similarity(&bundled, &b).unwrap();
        let sim_c = ops.similarity(&bundled, &c).unwrap();
        // Bundle should be more similar to inputs than random
        assert!(sim_a > 0.55, "sim_a={sim_a}");
        assert!(sim_b > 0.55, "sim_b={sim_b}");
        assert!(sim_c > 0.55, "sim_c={sim_c}");
    }

    #[test]
    fn bundle_empty_is_error() {
        let ops = test_ops();
        let result = ops.bundle(&[]);
        assert!(matches!(result, Err(VsaError::EmptyBundle)));
    }

    #[test]
    fn permute_is_dissimilar() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let permuted = ops.permute(&a, 1);
        let sim = ops.similarity(&a, &permuted).unwrap();
        // Permuted vector should be roughly uncorrelated
        assert!(sim > 0.4 && sim < 0.6, "sim={sim}");
    }

    #[test]
    fn dimension_mismatch_detected() {
        let ops = test_ops();
        let a = HyperVec::zero(Dimension(100), Encoding::Bipolar);
        let b = HyperVec::zero(Dimension(200), Encoding::Bipolar);
        let result = ops.bind(&a, &b);
        assert!(matches!(result, Err(VsaError::DimensionMismatch { .. })));
    }

    #[test]
    fn cosine_similarity_identical() {
        let ops = test_ops();
        let mut rng = seeded_rng();
        let a = ops.random(&mut rng);
        let sim = ops.cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 0.001, "cosine self-sim={sim}");
    }
}
