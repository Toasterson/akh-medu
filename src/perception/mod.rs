//! Phase 24 — Sensory Grounding.
//!
//! Connects symbolic reasoning to perceptual input from the physical world.
//! The agent can process images, audio, and structured sensor data, grounding
//! them in VSA hypervectors and the knowledge graph via `DerivationKind::Perceived`.
//!
//! ## Architecture
//!
//! ```text
//! Raw sensory input (image bytes / audio samples / sensor readings)
//!   ↓
//! Perceptual Frontend (vision / audio / structured)
//!   ↓  f32 feature vectors
//! PerceptualBridge  (vsa/perceptual_bridge.rs)
//!   ↓  HyperVec
//! ItemMemory nearest-neighbour search → GroundedConcepts
//!   ↓
//! KG triples  +  DerivationKind::Perceived  +  affective appraisal
//! ```
//!
//! **Sub-phases**: 24a (vision), 24b (audio), 24c (structured sensors).
//! All sub-phases run on-device with no cloud APIs.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::symbol::SymbolId;

pub mod audio;
pub mod structured;
pub mod vision;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from the sensory grounding subsystem (Phase 24).
#[derive(Debug, Error, Diagnostic)]
pub enum PerceptionError {
    #[error("empty {modality} input — no data to process")]
    #[diagnostic(
        code(akh::perception::empty_input),
        help(
            "Provide non-empty input: RGB24 bytes for images, \
             f32 PCM mono samples for audio, or at least one sensor reading."
        )
    )]
    EmptyInput { modality: String },

    #[error("unsupported format: {format}")]
    #[diagnostic(
        code(akh::perception::unsupported_format),
        help(
            "Supported formats: raw RGB24 (images), raw f32 PCM mono (audio). \
             Convert or re-encode the input before calling the perception pipeline."
        )
    )]
    UnsupportedFormat { format: String },

    #[error("feature extraction failed for {modality}: {reason}")]
    #[diagnostic(
        code(akh::perception::feature_extraction),
        help(
            "The perceptual frontend could not extract features. \
             For images, verify pixels.len() == width × height × 3. \
             For audio, ensure samples are f32 PCM mono."
        )
    )]
    FeatureExtraction { modality: String, reason: String },

    #[error(
        "VSA projection mismatch: feature vector is {feature_dim}-dim, \
         bridge expects {bridge_dim}-dim"
    )]
    #[diagnostic(
        code(akh::perception::vsa_projection),
        help(
            "Rebuild PerceptualBridge with a matching feature dimension, or ensure \
             the extractor produces vectors of the configured length."
        )
    )]
    VsaProjection {
        feature_dim: usize,
        bridge_dim: usize,
    },

    #[error("sensor error: {message}")]
    #[diagnostic(
        code(akh::perception::sensor),
        help("Check sensor values, units, and schema configuration.")
    )]
    Sensor { message: String },
}

pub type PerceptionResult<T> = Result<T, PerceptionError>;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// Sensory modality tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    /// RGB image: `width × height` pixels, each 3 bytes (R, G, B).
    Image { width: u32, height: u32 },
    /// Mono PCM audio: f32 samples at a known sample rate.
    Audio { sample_rate: u32, duration_ms: u32 },
    /// Structured key-value sensor readings.
    Structured { schema: String },
}

impl Modality {
    pub fn name(&self) -> &'static str {
        match self {
            Modality::Image { .. } => "image",
            Modality::Audio { .. } => "audio",
            Modality::Structured { .. } => "structured",
        }
    }
}

/// A symbolic concept grounded from a perceptual feature vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedConcept {
    /// Symbol in the knowledge graph.
    pub symbol: SymbolId,
    /// Human-readable label (for display and debugging).
    pub label: String,
    /// Similarity score in [0, 1] — higher means a closer match.
    pub similarity: f32,
}

/// Affective significance computed from a perceptual event.
///
/// Used to trigger somatic markers and mood updates via the Phase 23 system.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct PerceptualAffect {
    /// Valence: -1.0 (unpleasant) to 1.0 (pleasant).
    pub valence: f32,
    /// Arousal: 0.0 (calm) to 1.0 (highly activated).
    pub arousal: f32,
    /// Novelty: 0.0 (very familiar) to 1.0 (completely new).
    pub novelty: f32,
}

impl PerceptualAffect {
    pub fn new(valence: f32, arousal: f32, novelty: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            novelty: novelty.clamp(0.0, 1.0),
        }
    }
}

/// Extracted perceptual features ready for VSA grounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceivedFeatures {
    /// The modality this percept came from.
    pub modality: Modality,
    /// Normalized feature vector for VSA projection.
    ///
    /// Length equals the configured `{image,audio}_feature_dim`.
    pub features: Vec<f32>,
    /// KG concepts grounded from the feature vector via ANN search.
    pub concepts: Vec<GroundedConcept>,
    /// Overall extraction confidence in [0, 1].
    pub confidence: f32,
    /// Affective significance of this percept.
    pub affect: PerceptualAffect,
    /// Unix timestamp (seconds since epoch) when the percept was captured.
    pub timestamp: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the perceptual grounding pipeline.
#[derive(Debug, Clone)]
pub struct PerceptionConfig {
    /// Feature vector length produced by the image extractor (default: 128).
    pub image_feature_dim: usize,
    /// Feature vector length produced by the audio extractor (default: 128).
    pub audio_feature_dim: usize,
    /// Number of mel frequency bins for audio (default: 32).
    ///
    /// The extractor computes `mel_bins × 4` stats (mean/std/min/max), so
    /// `audio_feature_dim` should equal `mel_bins × 4`.
    pub mel_bins: usize,
    /// Maximum KG concepts to match per percept (default: 5).
    pub max_concepts: usize,
    /// Minimum similarity to accept a concept match (default: 0.55).
    pub min_similarity: f32,
    /// Similarity below which a percept counts as novel (default: 0.65).
    ///
    /// When the best-matching concept has similarity below this threshold,
    /// the `novelty` field on `PerceptualAffect` is set to 1.0.
    pub novelty_threshold: f32,
    /// Maximum audio frames to analyse (default: 200).
    ///
    /// Limits CPU usage for long audio clips.  Each frame is 512 samples
    /// with 256-sample hop, so 200 frames ≈ 3.2 s at 16 kHz.
    pub max_audio_frames: usize,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            image_feature_dim: 128,
            audio_feature_dim: 128,
            mel_bins: 32,
            max_concepts: 5,
            min_similarity: 0.55,
            novelty_threshold: 0.65,
            max_audio_frames: 200,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute sample mean and standard deviation in a single pass.
pub(crate) fn mean_and_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Return the current Unix timestamp in seconds.
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Pad or truncate a feature vector to exactly `target_len`.
pub fn resize_features(mut features: Vec<f32>, target_len: usize) -> Vec<f32> {
    features.resize(target_len, 0.0);
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perception_config_defaults_are_consistent() {
        let cfg = PerceptionConfig::default();
        // audio_feature_dim should be mel_bins × 4
        assert_eq!(cfg.audio_feature_dim, cfg.mel_bins * 4);
    }

    #[test]
    fn perceptual_affect_clamps() {
        let pa = PerceptualAffect::new(2.0, -1.0, 5.0);
        assert_eq!(pa.valence, 1.0);
        assert_eq!(pa.arousal, 0.0);
        assert_eq!(pa.novelty, 1.0);
    }

    #[test]
    fn mean_and_std_empty() {
        let (m, s) = mean_and_std(&[]);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn mean_and_std_uniform() {
        let (m, s) = mean_and_std(&[3.0, 3.0, 3.0]);
        assert!((m - 3.0).abs() < 1e-6);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn resize_features_pads_and_truncates() {
        let v = resize_features(vec![1.0, 2.0, 3.0], 5);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 0.0, 0.0]);

        let v = resize_features(vec![1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }
}
