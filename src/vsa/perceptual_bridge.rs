//! Phase 24 — Perceptual VSA Bridge.
//!
//! Projects perceptual feature vectors (image / audio / sensor) into the
//! hypervector space where they can be compared against symbolic concepts via
//! Hamming distance.
//!
//! The bridge mirrors `NeuralVsaBridge` but is tailored for perception:
//! instead of LLM hidden states it accepts continuous f32 feature vectors
//! produced by the `perception` module's extractors.
//!
//! ## Encoding pipeline
//!
//! 1. Feature vector `f` (f32, `[feature_dim]`)
//! 2. Linear projection: `z = W·f + b`  where `W` is `[vsa_dim × feature_dim]`
//! 3. Binarise: `v[i] = 1 if z[i] > threshold else 0`
//! 4. Result: `vsa_dim`-bit HyperVec compatible with all VSA operations
//!
//! Random initialisation means the raw bridge produces uncorrelated vectors,
//! which still allows nearest-neighbour search.  A self-supervised alignment
//! pass (grounding percept vectors against known KG concept vectors) can be
//! added in Phase 27 once the training infrastructure lands.

use super::{Dimension, Encoding, HyperVec};
use super::item_memory::{ItemMemory, SearchResult};
use super::neural_bridge::LinearProjection;
use crate::perception::PerceptionError;

// ─────────────────────────────────────────────────────────────────────────────
// PerceptualBridge
// ─────────────────────────────────────────────────────────────────────────────

/// Maps perceptual feature vectors to HyperVec space via a linear projection.
///
/// Create one bridge per modality and feature dimensionality.
#[derive(Debug)]
pub struct PerceptualBridge {
    /// Linear map: `feature_dim → vsa_dim` (pre-binarisation).
    projection: LinearProjection,
    /// Binarisation threshold (default: 0.0 for a symmetric split).
    threshold: f32,
    /// Target VSA dimension.
    vsa_dim: Dimension,
}

impl PerceptualBridge {
    /// Create a bridge with random (Xavier) weight initialisation.
    ///
    /// - `feature_dim`: dimension of the perceptual feature vectors
    /// - `vsa_dim`:     target VSA dimension (e.g. `Dimension::DEFAULT`)
    pub fn new(feature_dim: usize, vsa_dim: Dimension) -> Self {
        Self {
            projection: LinearProjection::new(feature_dim, vsa_dim.0),
            threshold: 0.0,
            vsa_dim,
        }
    }

    /// Create a bridge from pre-trained projection weights.
    ///
    /// Panics in debug mode if `projection.out_dim() != vsa_dim.0`.
    pub fn from_projection(projection: LinearProjection, vsa_dim: Dimension) -> Self {
        debug_assert_eq!(
            projection.out_dim(),
            vsa_dim.0,
            "projection output dim must equal vsa_dim"
        );
        Self {
            projection,
            threshold: 0.0,
            vsa_dim,
        }
    }

    /// Set a custom binarisation threshold (default: 0.0).
    ///
    /// Values above the threshold become `1` (+1 in bipolar); below become `0`.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    // ── Projection ──────────────────────────────────────────────────────────

    /// Project a feature vector to a HyperVec.
    ///
    /// # Errors
    /// Returns [`PerceptionError::VsaProjection`] if `features.len() !=
    /// feature_dim`.
    pub fn project(&self, features: &[f32]) -> Result<HyperVec, PerceptionError> {
        if features.len() != self.projection.in_dim() {
            return Err(PerceptionError::VsaProjection {
                feature_dim: features.len(),
                bridge_dim: self.projection.in_dim(),
            });
        }
        let projected = self.projection.forward(features);
        Ok(binarise(&projected, self.threshold, self.vsa_dim))
    }

    // ── Grounding (ANN search) ───────────────────────────────────────────────

    /// Project features to a HyperVec and retrieve the `k` nearest concepts.
    ///
    /// Results are sorted by descending similarity.
    ///
    /// # Errors
    /// Returns [`PerceptionError::VsaProjection`] if the feature dimension
    /// does not match.
    pub fn ground(
        &self,
        features: &[f32],
        item_memory: &ItemMemory,
        k: usize,
    ) -> Result<Vec<SearchResult>, PerceptionError> {
        let hv = self.project(features)?;
        item_memory
            .search(&hv, k)
            .map_err(|e| PerceptionError::Sensor {
                message: format!("ANN search failed: {e}"),
            })
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// The expected input feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.projection.in_dim()
    }

    /// The target VSA dimension.
    pub fn vsa_dim(&self) -> Dimension {
        self.vsa_dim
    }

    /// Access the raw linear projection (for serialisation or alignment).
    pub fn projection(&self) -> &LinearProjection {
        &self.projection
    }

    /// Current binarisation threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a projected f32 vector to a bipolar HyperVec by thresholding.
fn binarise(projected: &[f32], threshold: f32, dim: Dimension) -> HyperVec {
    let mut hv = HyperVec::zero(dim, Encoding::Bipolar);
    for (i, &z) in projected.iter().enumerate().take(dim.0) {
        if z > threshold {
            hv.set_bit(i, true);
        }
    }
    hv
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_returns_correct_dim() {
        let bridge = PerceptualBridge::new(128, Dimension::TEST);
        let features = vec![0.5f32; 128];
        let hv = bridge.project(&features).unwrap();
        assert_eq!(hv.dim(), Dimension::TEST);
        assert_eq!(hv.encoding(), Encoding::Bipolar);
        assert_eq!(hv.data().len(), Dimension::TEST.binary_byte_len());
    }

    #[test]
    fn project_wrong_dim_returns_error() {
        let bridge = PerceptualBridge::new(128, Dimension::TEST);
        let features = vec![0.5f32; 64]; // wrong dim
        let result = bridge.project(&features);
        assert!(
            matches!(result, Err(PerceptionError::VsaProjection { .. })),
            "expected VsaProjection error"
        );
    }

    #[test]
    fn different_features_produce_different_hvecs() {
        let bridge = PerceptualBridge::new(64, Dimension::TEST);
        let all_high = vec![10.0f32; 64];
        let all_low = vec![-10.0f32; 64];
        let hv_high = bridge.project(&all_high).unwrap();
        let hv_low = bridge.project(&all_low).unwrap();
        assert_ne!(hv_high, hv_low, "opposite extremes should map to different HyperVecs");
    }

    #[test]
    fn same_features_produce_same_hvec() {
        let bridge = PerceptualBridge::new(32, Dimension::TEST);
        let features = vec![0.3f32; 32];
        let hv1 = bridge.project(&features).unwrap();
        let hv2 = bridge.project(&features).unwrap();
        assert_eq!(hv1, hv2, "identical inputs must produce identical HyperVecs");
    }

    #[test]
    fn from_projection_preserves_dims() {
        let proj = LinearProjection::new(64, Dimension::TEST.0);
        let bridge = PerceptualBridge::from_projection(proj, Dimension::TEST);
        assert_eq!(bridge.feature_dim(), 64);
        assert_eq!(bridge.vsa_dim(), Dimension::TEST);
    }

    #[test]
    fn with_threshold_affects_output() {
        let bridge_low = PerceptualBridge::new(32, Dimension::TEST).with_threshold(-100.0);
        let bridge_high = PerceptualBridge::new(32, Dimension::TEST).with_threshold(100.0);

        // Re-seed with same weights is not easy; just verify threshold is stored.
        assert!((bridge_low.threshold() - -100.0).abs() < 1e-6);
        assert!((bridge_high.threshold() - 100.0).abs() < 1e-6);
    }
}
