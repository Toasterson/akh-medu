//! Version-one neural-span representation: pooled residual means → 10000 bits.
//!
//! The recipe is deliberately simple and fully described by two pinned
//! artifacts so that stored vectors, queries and future re-indexing agree on
//! one immutable representation identity:
//!
//! - **Capture policy** (`policy_sha256`): which post-block residual streams are
//!   pooled (five layers), how (count-weighted mean over the newly forwarded
//!   user-input span, f32 arithmetic), and in which prompt context (the durable
//!   history without any recalled block).
//! - **Projector** (`projector_sha256`): per layer, L2-normalise the 5120-wide
//!   mean, zero-pad to 8192, apply a seeded randomized Hadamard rotation, keep
//!   the first 2000 coefficients and binarise by sign. Five layers give exactly
//!   10000 bits, packed most-significant-bit first.
//!
//! Pins identify the recipe; they are not attestation that a vector was
//! produced by it. Same-space Hamming hits are *neural span similarity*, not
//! grounded symbolic concept matches (see the VSA-space assessment).

use crate::vsa::polar_quant::randomized_hadamard;
use sha2::{Digest, Sha256};

/// Bits in one projected vector.
pub const VECTOR_BITS: usize = 10_000;
/// Bytes in one packed projected vector.
pub const VECTOR_BYTES: usize = VECTOR_BITS / 8;
/// Pooled layers per capture.
pub const LAYERS_PER_CAPTURE: usize = 5;
/// Bits taken from each layer.
pub const BITS_PER_LAYER: usize = VECTOR_BITS / LAYERS_PER_CAPTURE;
/// Hadamard block size (next power of two above the 5120 residual width).
pub const HADAMARD_WIDTH: usize = 8192;
/// Residual width of the target model.
pub const TARGET_HIDDEN: usize = 5120;
/// Target-model block indices (64 blocks): 5 + 14k.
pub const TARGET_LAYERS: [usize; LAYERS_PER_CAPTURE] = [5, 19, 33, 47, 61];
/// Base seed for the projector; per-layer seeds derive from it.
pub const PROJECTOR_SEED: u64 = 0x5074_6168_5370_616e; // "PtahSpan"

/// Which blocks are tapped for a model with `block_count` blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturePolicy {
    /// Total transformer blocks in the model (excludes the MTP block).
    pub block_count: usize,
    /// Residual width.
    pub hidden: usize,
    /// Tapped block indices, ascending.
    pub layers: [usize; LAYERS_PER_CAPTURE],
}

impl CapturePolicy {
    /// Select five tapped blocks. The target model (64 blocks, 5120 wide) gets
    /// exactly [5, 19, 33, 47, 61]; smaller models spread five taps evenly.
    pub fn for_model(block_count: usize, hidden: usize) -> Option<Self> {
        if block_count < LAYERS_PER_CAPTURE || hidden == 0 || hidden > HADAMARD_WIDTH {
            return None;
        }
        let layers = if block_count == 64 {
            TARGET_LAYERS
        } else {
            let mut layers = [0usize; LAYERS_PER_CAPTURE];
            // Evenly spaced, last tap on the final block, first strictly inside.
            for (i, slot) in layers.iter_mut().enumerate() {
                *slot = ((i + 1) * block_count) / LAYERS_PER_CAPTURE - 1;
            }
            layers
        };
        Some(Self {
            block_count,
            hidden,
            layers,
        })
    }

    /// Canonical description hashed into `policy_sha256`.
    pub fn canonical(&self) -> String {
        format!(
            "akh-neural-span-policy/1;blocks={};hidden={};layers={};pool=count_weighted_mean_post_block_residual;\
             arithmetic=f32_activations;span=user_input_text_leading_bytes_le_8192;\
             context=durable_history_without_recalled_block;source=prefill_only",
            self.block_count,
            self.hidden,
            self.layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Lowercase SHA-256 of the canonical policy description.
    pub fn sha256(&self) -> String {
        hex(&Sha256::digest(self.canonical().as_bytes()))
    }
}

/// The seeded rotation + subsample + sign recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projector {
    pub seed: u64,
}

impl Default for Projector {
    fn default() -> Self {
        Self {
            seed: PROJECTOR_SEED,
        }
    }
}

/// One pooled layer mean as read back from the device.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerMean {
    pub layer: usize,
    pub mean: Vec<f32>,
}

/// Why a set of pooled means could not be projected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    /// Policy dimensions or layer indices are invalid, including caller-built policies.
    InvalidPolicy,
    /// Layer set differs from the policy.
    LayerMismatch,
    /// A mean has the wrong width.
    WidthMismatch { expected: usize, actual: usize },
    /// A mean contains NaN/Inf or is entirely zero.
    DegenerateMean { layer: usize },
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => f.write_str("invalid capture policy dimensions or layers"),
            Self::LayerMismatch => f.write_str("pooled layers do not match the capture policy"),
            Self::WidthMismatch { expected, actual } => {
                write!(
                    f,
                    "pooled mean width {actual} differs from policy width {expected}"
                )
            }
            Self::DegenerateMean { layer } => {
                write!(f, "pooled mean for layer {layer} is non-finite or zero")
            }
        }
    }
}

impl std::error::Error for ProjectionError {}

impl Projector {
    /// Canonical description hashed into `projector_sha256`.
    pub fn canonical(&self) -> String {
        format!(
            "akh-neural-span-projector/1;seed={:#018x};per_layer=l2norm,zero_pad_{},randomized_hadamard_xorshift64_signs_normalized,\
             first_{}_coefficients,sign_bit(nonnegative=1);layer_seed=seed^layer*0x9e3779b97f4a7c15;bits={};pack=msb_first",
            self.seed, HADAMARD_WIDTH, BITS_PER_LAYER, VECTOR_BITS
        )
    }

    /// Lowercase SHA-256 of the canonical projector description.
    pub fn sha256(&self) -> String {
        hex(&Sha256::digest(self.canonical().as_bytes()))
    }

    fn layer_seed(&self, layer: usize) -> u64 {
        // Distinct rotations per layer; never zero (xorshift would stall).
        let seed = self.seed ^ (layer as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        }
    }

    /// Project pooled means for the policy's layers into 1250 packed bytes.
    pub fn project(
        &self,
        policy: &CapturePolicy,
        means: &[LayerMean],
    ) -> Result<Vec<u8>, ProjectionError> {
        // The public fields permit caller-built policies: never rely solely on
        // for_model() to prevent truncation or invalid representation identities.
        if policy.hidden == 0
            || policy.hidden > HADAMARD_WIDTH
            || policy.block_count < LAYERS_PER_CAPTURE
            || policy
                .layers
                .iter()
                .any(|&layer| layer >= policy.block_count)
            || policy
                .layers
                .windows(2)
                .any(|layers| layers[0] >= layers[1])
        {
            return Err(ProjectionError::InvalidPolicy);
        }
        if means.len() != LAYERS_PER_CAPTURE {
            return Err(ProjectionError::LayerMismatch);
        }
        let mut bits = Vec::with_capacity(VECTOR_BITS);
        for (expected_layer, mean) in policy.layers.iter().zip(means) {
            if mean.layer != *expected_layer {
                return Err(ProjectionError::LayerMismatch);
            }
            if mean.mean.len() != policy.hidden {
                return Err(ProjectionError::WidthMismatch {
                    expected: policy.hidden,
                    actual: mean.mean.len(),
                });
            }
            let norm: f64 = mean
                .mean
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm == 0.0 || mean.mean.iter().any(|v| !v.is_finite()) {
                return Err(ProjectionError::DegenerateMean { layer: mean.layer });
            }
            let mut block = vec![0f32; HADAMARD_WIDTH];
            for (dst, src) in block.iter_mut().zip(&mean.mean) {
                *dst = (*src as f64 / norm) as f32;
            }
            randomized_hadamard(&mut block, self.layer_seed(mean.layer));
            bits.extend(block[..BITS_PER_LAYER].iter().map(|v| *v >= 0.0));
        }
        debug_assert_eq!(bits.len(), VECTOR_BITS);
        Ok(pack_bits(&bits))
    }
}

/// Pack booleans MSB-first into bytes.
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            bytes[i / 8] |= 0x80 >> (i % 8);
        }
    }
    bytes
}

/// Hamming distance between two packed vectors of equal length.
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_means(policy: &CapturePolicy, seed: u64) -> Vec<LayerMean> {
        let mut state = seed | 1;
        policy
            .layers
            .iter()
            .map(|&layer| LayerMean {
                layer,
                mean: (0..policy.hidden)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn target_policy_pins_the_agreed_layers_and_hash_is_stable() {
        let policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        assert_eq!(policy.layers, [5, 19, 33, 47, 61]);
        assert_eq!(policy.sha256().len(), 64);
        assert_eq!(
            policy.sha256(),
            CapturePolicy::for_model(64, TARGET_HIDDEN)
                .unwrap()
                .sha256()
        );
        assert_ne!(
            policy.sha256(),
            CapturePolicy::for_model(48, TARGET_HIDDEN)
                .unwrap()
                .sha256()
        );
        assert_ne!(policy.sha256(), Projector::default().sha256());
        assert!(CapturePolicy::for_model(4, 8).is_none());
        assert!(CapturePolicy::for_model(64, HADAMARD_WIDTH + 1).is_none());
    }

    #[test]
    fn small_models_get_five_distinct_valid_layers() {
        for blocks in [5usize, 6, 7, 8, 12, 31] {
            let policy = CapturePolicy::for_model(blocks, 8).unwrap();
            assert!(
                policy.layers.windows(2).all(|w| w[0] < w[1]),
                "{blocks}: {:?}",
                policy.layers
            );
            assert_eq!(policy.layers[LAYERS_PER_CAPTURE - 1], blocks - 1);
            assert!(policy.layers[0] < blocks);
        }
        assert_eq!(
            CapturePolicy::for_model(5, 8).unwrap().layers,
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn projection_is_deterministic_and_exactly_ten_thousand_bits() {
        let policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        let projector = Projector::default();
        let means = seeded_means(&policy, 7);
        let a = projector.project(&policy, &means).unwrap();
        let b = projector.project(&policy, &means).unwrap();
        assert_eq!(a.len(), VECTOR_BYTES);
        assert_eq!(a, b);
        let ones: u32 = a.iter().map(|b| b.count_ones()).sum();
        // Balanced bits for isotropic input.
        assert!((4000..=6000).contains(&ones), "{ones}");
    }

    #[test]
    fn scale_invariant_and_layer_rotations_differ() {
        let policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        let projector = Projector::default();
        let means = seeded_means(&policy, 11);
        let scaled: Vec<LayerMean> = means
            .iter()
            .map(|m| LayerMean {
                layer: m.layer,
                mean: m.mean.iter().map(|v| v * 37.5).collect(),
            })
            .collect();
        assert_eq!(
            projector.project(&policy, &means).unwrap(),
            projector.project(&policy, &scaled).unwrap()
        );
        // Same activations in two different layers must not yield identical bits.
        let mut swapped = means.clone();
        swapped[0].mean = means[1].mean.clone();
        swapped[1].mean = means[0].mean.clone();
        assert_ne!(
            projector.project(&policy, &means).unwrap(),
            projector.project(&policy, &swapped).unwrap()
        );
    }

    #[test]
    fn nearby_inputs_are_near_and_unrelated_inputs_are_far() {
        let policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        let projector = Projector::default();
        let base = seeded_means(&policy, 3);
        let mut nearby = base.clone();
        for m in &mut nearby {
            for (i, v) in m.mean.iter_mut().enumerate() {
                if i % 50 == 0 {
                    *v += 0.05;
                }
            }
        }
        let other = seeded_means(&policy, 4);
        let pb = projector.project(&policy, &base).unwrap();
        let pn = projector.project(&policy, &nearby).unwrap();
        let po = projector.project(&policy, &other).unwrap();
        let near = hamming(&pb, &pn);
        let far = hamming(&pb, &po);
        assert!(near < 1500, "near distance {near}");
        assert!((4000..=6000).contains(&far), "far distance {far}");
        assert!(near < far);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        let projector = Projector::default();
        let means = seeded_means(&policy, 5);
        assert_eq!(
            projector.project(&policy, &means[..4]).unwrap_err(),
            ProjectionError::LayerMismatch
        );
        let mut wrong_layer = means.clone();
        wrong_layer[2].layer = 34;
        assert_eq!(
            projector.project(&policy, &wrong_layer).unwrap_err(),
            ProjectionError::LayerMismatch
        );
        let mut short = means.clone();
        short[0].mean.pop();
        assert!(matches!(
            projector.project(&policy, &short).unwrap_err(),
            ProjectionError::WidthMismatch {
                expected: 5120,
                actual: 5119
            }
        ));
        let mut zero = means.clone();
        zero[4].mean.iter_mut().for_each(|v| *v = 0.0);
        assert_eq!(
            projector.project(&policy, &zero).unwrap_err(),
            ProjectionError::DegenerateMean { layer: 61 }
        );
        let mut nan = means;
        nan[1].mean[3] = f32::NAN;
        assert_eq!(
            projector.project(&policy, &nan).unwrap_err(),
            ProjectionError::DegenerateMean { layer: 19 }
        );
    }

    #[test]
    fn caller_constructed_invalid_policies_are_rejected() {
        let projector = Projector::default();
        let mut policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        policy.hidden = HADAMARD_WIDTH + 1;
        assert!(
            projector
                .project(&policy, &seeded_means(&policy, 5))
                .is_err(),
            "oversized means must not be silently truncated"
        );
        let mut policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        policy.layers[1] = policy.layers[0];
        assert!(
            projector
                .project(&policy, &seeded_means(&policy, 5))
                .is_err(),
            "duplicate layers must not masquerade as five independent taps"
        );
        let mut policy = CapturePolicy::for_model(64, TARGET_HIDDEN).unwrap();
        policy.layers[4] = policy.block_count;
        assert!(
            projector
                .project(&policy, &seeded_means(&policy, 5))
                .is_err(),
            "out-of-model layers are invalid"
        );
    }

    #[test]
    fn packing_is_msb_first_and_hamming_counts_bits() {
        assert_eq!(
            pack_bits(&[true, false, false, false, false, false, false, true, true]),
            vec![0x81, 0x80]
        );
        assert_eq!(hamming(&[0xff, 0x00], &[0x0f, 0x01]), 5);
    }
}
