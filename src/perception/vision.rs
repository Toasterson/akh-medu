//! Phase 24a — Image perception.
//!
//! Pure-Rust feature extraction from raw RGB24 images.
//! No ML models are required; the extracted features are statistical
//! descriptors that capture colour distribution, intensity, and edges.
//! These features are projected into HyperVec space via `PerceptualBridge`.
//!
//! ## Feature layout (padded/truncated to `image_feature_dim`, default 128)
//!
//! | Range     | Count | Description                              |
//! |-----------|-------|------------------------------------------|
//! | [0, 96)   | 96    | Colour histograms: 32 bins × 3 channels  |
//! | [96, 102) | 6     | Intensity stats: mean + std per channel  |
//! | [102, 104)| 2     | Luminance: mean + std                    |
//! | [104, 108)| 4     | Edge density: horiz/vert mean + std      |
//! | [108, 128)| 20    | Zero padding                             |

use super::{PerceptionConfig, PerceptionError, PerceptionResult, PerceptualAffect, mean_and_std};

/// Extract a feature vector from a raw RGB24 image.
///
/// # Arguments
/// - `pixels`  — row-major RGB24 data: `[R, G, B, R, G, B, …]`
/// - `width`   — image width in pixels
/// - `height`  — image height in pixels
/// - `config`  — controls output `image_feature_dim`
///
/// Returns the feature vector and a heuristic affective estimate.
///
/// # Errors
/// - [`PerceptionError::EmptyInput`] — `pixels` is empty
/// - [`PerceptionError::FeatureExtraction`] — dimension mismatch
pub fn extract_image_features(
    pixels: &[u8],
    width: u32,
    height: u32,
    config: &PerceptionConfig,
) -> PerceptionResult<(Vec<f32>, PerceptualAffect)> {
    if pixels.is_empty() {
        return Err(PerceptionError::EmptyInput {
            modality: "image".to_string(),
        });
    }

    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() != expected {
        return Err(PerceptionError::FeatureExtraction {
            modality: "image".to_string(),
            reason: format!(
                "expected {expected} bytes ({width}×{height}×3), got {}",
                pixels.len()
            ),
        });
    }

    let n_pixels = (width as usize) * (height as usize);

    // Decompose pixels into per-channel f32 vectors in [0, 1].
    let mut ch_r = Vec::with_capacity(n_pixels);
    let mut ch_g = Vec::with_capacity(n_pixels);
    let mut ch_b = Vec::with_capacity(n_pixels);
    let mut luminance = Vec::with_capacity(n_pixels);

    for chunk in pixels.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        ch_r.push(r);
        ch_g.push(g);
        ch_b.push(b);
        luminance.push(0.299 * r + 0.587 * g + 0.114 * b);
    }

    let mut features = Vec::with_capacity(config.image_feature_dim);

    // 1. Colour histograms: 32 bins per channel × 3 = 96 features.
    let n_bins = 32usize;
    for channel in [&ch_r, &ch_g, &ch_b] {
        let mut hist = vec![0u32; n_bins];
        for &v in channel {
            let bin = ((v * n_bins as f32) as usize).min(n_bins - 1);
            hist[bin] += 1;
        }
        for &count in &hist {
            features.push(count as f32 / n_pixels as f32);
        }
    }

    // 2. Intensity stats: (mean, std) per channel × 3 = 6 features.
    for channel in [&ch_r, &ch_g, &ch_b] {
        let (m, s) = mean_and_std(channel);
        features.push(m);
        features.push(s);
    }

    // 3. Luminance stats: (mean, std) = 2 features.
    let (lum_mean, lum_std) = mean_and_std(&luminance);
    features.push(lum_mean);
    features.push(lum_std);

    // 4. Edge statistics: 4 features.
    let edge = compute_edge_stats(&luminance, width as usize, height as usize);
    features.extend_from_slice(&edge);

    // Pad/truncate to configured dimension.
    features.resize(config.image_feature_dim, 0.0);

    let affect = image_affect(lum_mean, mean_and_std(&ch_r).1, &edge);
    Ok((features, affect))
}

/// Compute horizontal and vertical edge statistics.
///
/// Returns `[h_mean, h_std, v_mean, v_std]` where values are in [0, 1].
fn compute_edge_stats(luminance: &[f32], width: usize, height: usize) -> Vec<f32> {
    if width < 2 || height < 2 {
        return vec![0.0; 4];
    }

    let mut h_grads = Vec::with_capacity((width - 1) * height);
    let mut v_grads = Vec::with_capacity(width * (height - 1));

    for y in 0..height {
        for x in 1..width {
            h_grads.push((luminance[y * width + x] - luminance[y * width + x - 1]).abs());
        }
    }
    for y in 1..height {
        for x in 0..width {
            v_grads.push((luminance[y * width + x] - luminance[(y - 1) * width + x]).abs());
        }
    }

    let (hm, hs) = mean_and_std(&h_grads);
    let (vm, vs) = mean_and_std(&v_grads);
    vec![hm, hs, vm, vs]
}

/// Heuristic affective state for an image.
///
/// - Brightness → valence (bright ↑ / dark ↓)
/// - Colour variation → positive valence
/// - Edge density → arousal
fn image_affect(lum_mean: f32, sat_std: f32, edge_stats: &[f32]) -> PerceptualAffect {
    let brightness_valence = (lum_mean - 0.5) * 0.6;
    let color_valence = sat_std * 0.4;
    let edge_density = edge_stats.first().copied().unwrap_or(0.0);
    let arousal = (edge_density * 5.0).clamp(0.0, 1.0);
    let novelty = (arousal * 0.5 + (lum_mean - 0.5).abs() * 0.3).clamp(0.0, 1.0);

    PerceptualAffect::new(
        (brightness_valence + color_valence).clamp(-1.0, 1.0),
        arousal,
        novelty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut pixels = Vec::with_capacity(n * 3);
        for _ in 0..n {
            pixels.extend_from_slice(&[r, g, b]);
        }
        pixels
    }

    #[test]
    fn white_image_returns_correct_dim() {
        let cfg = PerceptionConfig::default();
        let pixels = solid(255, 255, 255, 8, 8);
        let (features, _) = extract_image_features(&pixels, 8, 8, &cfg).unwrap();
        assert_eq!(features.len(), cfg.image_feature_dim);
    }

    #[test]
    fn black_image_returns_correct_dim() {
        let cfg = PerceptionConfig::default();
        let pixels = solid(0, 0, 0, 4, 4);
        let (features, _) = extract_image_features(&pixels, 4, 4, &cfg).unwrap();
        assert_eq!(features.len(), cfg.image_feature_dim);
    }

    #[test]
    fn white_image_has_positive_valence() {
        let cfg = PerceptionConfig::default();
        let pixels = solid(255, 255, 255, 8, 8);
        let (_, affect) = extract_image_features(&pixels, 8, 8, &cfg).unwrap();
        assert!(affect.valence > 0.0, "bright image → positive valence");
    }

    #[test]
    fn feature_dim_matches_config() {
        let cfg = PerceptionConfig {
            image_feature_dim: 64,
            ..Default::default()
        };
        let pixels = solid(128, 64, 32, 4, 4);
        let (features, _) = extract_image_features(&pixels, 4, 4, &cfg).unwrap();
        assert_eq!(features.len(), 64);
    }

    #[test]
    fn empty_input_error() {
        let cfg = PerceptionConfig::default();
        let result = extract_image_features(&[], 8, 8, &cfg);
        assert!(matches!(result, Err(PerceptionError::EmptyInput { .. })));
    }

    #[test]
    fn wrong_size_error() {
        let cfg = PerceptionConfig::default();
        // 10 bytes is not 8×8×3 = 192
        let result = extract_image_features(&vec![0u8; 10], 8, 8, &cfg);
        assert!(matches!(result, Err(PerceptionError::FeatureExtraction { .. })));
    }

    #[test]
    fn gradient_image_has_nonzero_edges() {
        let cfg = PerceptionConfig::default();
        // Build a 4×4 gradient image (left=black, right=white)
        let mut pixels = Vec::new();
        for _y in 0..4 {
            for x in 0..4u8 {
                let v = x * 85; // 0, 85, 170, 255
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        let (features, affect) = extract_image_features(&pixels, 4, 4, &cfg).unwrap();
        assert_eq!(features.len(), cfg.image_feature_dim);
        // Gradient image should have measurable arousal
        assert!(affect.arousal > 0.0, "gradient image → some arousal");
    }

    #[test]
    fn colour_histogram_sums_to_one_per_channel() {
        let cfg = PerceptionConfig::default();
        let pixels = solid(200, 100, 50, 8, 8);
        let (features, _) = extract_image_features(&pixels, 8, 8, &cfg).unwrap();
        // Each channel's 32 histogram bins should sum to 1.0
        for ch in 0..3 {
            let sum: f32 = features[ch * 32..(ch + 1) * 32].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "channel {ch} histogram sums to {sum}, expected 1.0"
            );
        }
    }
}
