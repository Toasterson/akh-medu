//! Phase 24b — Audio perception.
//!
//! Mel spectrogram extraction from raw f32 PCM mono audio.
//! No ML models are required; features are computed via a pure-Rust DFT
//! with precomputed twiddle tables and mel triangular filterbanks.
//!
//! ## Feature layout (`mel_bins × 4`, default 128)
//!
//! For each of the `mel_bins` (default 32) mel frequency bins:
//! - mean log energy over time
//! - standard deviation of log energy over time
//! - minimum log energy
//! - maximum log energy
//!
//! All values are sigmoid-normalised to [0, 1] before output.

use super::{PerceptionConfig, PerceptionError, PerceptionResult, PerceptualAffect, mean_and_std};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a mel-spectrogram feature vector from raw f32 PCM mono audio.
///
/// # Arguments
/// - `samples`     — f32 PCM mono samples (any amplitude scale)
/// - `sample_rate` — samples per second (e.g. 16 000)
/// - `config`      — controls `mel_bins` and `audio_feature_dim`
///
/// Returns the feature vector and a heuristic affective estimate.
///
/// # Errors
/// - [`PerceptionError::EmptyInput`] — `samples` is empty
/// - [`PerceptionError::FeatureExtraction`] — no frames could be extracted
pub fn extract_audio_features(
    samples: &[f32],
    sample_rate: u32,
    config: &PerceptionConfig,
) -> PerceptionResult<(Vec<f32>, PerceptualAffect)> {
    if samples.is_empty() {
        return Err(PerceptionError::EmptyInput {
            modality: "audio".to_string(),
        });
    }

    let n_fft = 512usize;
    let hop = n_fft / 2;
    let n_mels = config.mel_bins;
    let max_frames = config.max_audio_frames;

    let twiddle = TwiddleTables::new(n_fft);
    let window = hann_window(n_fft);
    let filterbank = mel_filterbank(n_mels, n_fft, sample_rate);

    let mut mel_frames: Vec<Vec<f32>> = Vec::new();
    let mut start = 0;

    while start < samples.len() && mel_frames.len() < max_frames {
        // Build a zero-padded frame.
        let mut frame = vec![0.0f32; n_fft];
        let avail = samples.len().saturating_sub(start).min(n_fft);
        frame[..avail].copy_from_slice(&samples[start..start + avail]);

        // Apply Hann window.
        for (f, &w) in frame.iter_mut().zip(window.iter()) {
            *f *= w;
        }

        // Compute DFT magnitudes.
        let mag = twiddle.magnitude(&frame);

        // Apply mel filterbank → log mel energies.
        let mel_energies: Vec<f32> = filterbank
            .iter()
            .map(|filter| {
                let energy: f32 = filter.iter().zip(mag.iter()).map(|(w, m)| w * m).sum();
                (energy + 1e-9f32).ln()
            })
            .collect();

        mel_frames.push(mel_energies);
        start += hop;
    }

    if mel_frames.is_empty() {
        return Err(PerceptionError::FeatureExtraction {
            modality: "audio".to_string(),
            reason: "no frames extracted from input".to_string(),
        });
    }

    // Compute per-mel-bin statistics across frames.
    let mut features = Vec::with_capacity(n_mels * 4);
    for bin in 0..n_mels {
        let values: Vec<f32> = mel_frames.iter().map(|f| f[bin]).collect();
        let (mean, std) = mean_and_std(&values);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        features.push(sigmoid(mean));
        features.push(sigmoid(std));
        features.push(sigmoid(if min.is_finite() { min } else { 0.0 }));
        features.push(sigmoid(if max.is_finite() { max } else { 0.0 }));
    }

    // Pad / truncate to configured dimension.
    features.resize(config.audio_feature_dim, 0.0);

    let affect = audio_affect(&mel_frames, n_mels);
    Ok((features, affect))
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed twiddle factors for the DFT.
///
/// Stores `cos(2π·m/N)` and `-sin(2π·m/N)` for `m = 0..N`.
/// The inner loop indexes via `(k * n) % N`, avoiding repeated trig calls.
struct TwiddleTables {
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    n: usize,
}

impl TwiddleTables {
    fn new(n: usize) -> Self {
        let mut cos_table = Vec::with_capacity(n);
        let mut sin_table = Vec::with_capacity(n);
        for m in 0..n {
            let angle = 2.0 * std::f32::consts::PI * m as f32 / n as f32;
            cos_table.push(angle.cos());
            sin_table.push(-angle.sin()); // negative: e^{-j 2π k n / N}
        }
        Self { cos_table, sin_table, n }
    }

    /// Compute positive-frequency DFT magnitudes for a real-valued frame.
    ///
    /// Returns `N/2 + 1` values (k = 0 … N/2).
    fn magnitude(&self, frame: &[f32]) -> Vec<f32> {
        let n = self.n;
        let n_pos = n / 2 + 1;
        let mut magnitudes = Vec::with_capacity(n_pos);

        for k in 0..n_pos {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (i, &x) in frame.iter().take(n).enumerate() {
                let tw = (k * i) % n;
                real += x * self.cos_table[tw];
                imag += x * self.sin_table[tw];
            }
            magnitudes.push((real * real + imag * imag).sqrt());
        }

        magnitudes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hann window
// ─────────────────────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let r = i as f32 / (n.max(2) - 1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * r).cos())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank
// ─────────────────────────────────────────────────────────────────────────────

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Build a mel triangular filterbank matrix `[n_mels × (n_fft/2+1)]`.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate as f32 / 2.0;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    // `n_mels + 2` equally spaced mel-scale centre frequencies.
    let centers: Vec<f32> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert centre frequencies from mel to FFT bin index.
    let bin_centers: Vec<f32> = centers
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            (n_fft + 1) as f32 * hz / sample_rate as f32
        })
        .collect();

    // Build triangular filters.
    let mut filters = vec![vec![0.0f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let (f_lo, f_mid, f_hi) = (bin_centers[m], bin_centers[m + 1], bin_centers[m + 2]);
        for k in 0..n_bins {
            let k_f = k as f32;
            let rise_width = f_mid - f_lo;
            let fall_width = f_hi - f_mid;
            if k_f >= f_lo && k_f <= f_mid && rise_width > 1e-9 {
                filters[m][k] = (k_f - f_lo) / rise_width;
            } else if k_f > f_mid && k_f <= f_hi && fall_width > 1e-9 {
                filters[m][k] = (f_hi - k_f) / fall_width;
            }
        }
    }

    filters
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Heuristic affective state from mel spectrogram frames.
fn audio_affect(mel_frames: &[Vec<f32>], n_mels: usize) -> PerceptualAffect {
    if mel_frames.is_empty() || n_mels == 0 {
        return PerceptualAffect::default();
    }

    let n_frames = mel_frames.len() as f32;

    // Overall log energy level → arousal.
    let total_energy: f32 = mel_frames
        .iter()
        .flat_map(|f| f.iter())
        .sum::<f32>()
        / (n_frames * n_mels as f32);

    let arousal = sigmoid(total_energy * 0.5).clamp(0.0, 1.0);

    // High-frequency vs low-frequency balance.
    // Higher frequencies (brighter sound) → slightly more positive valence.
    let low_bins = (n_mels / 4).max(1);
    let high_bins = low_bins;
    let low_energy: f32 = mel_frames
        .iter()
        .map(|f| f[..low_bins].iter().sum::<f32>() / low_bins as f32)
        .sum::<f32>()
        / n_frames;
    let hi_energy: f32 = mel_frames
        .iter()
        .map(|f| {
            let start = n_mels.saturating_sub(high_bins);
            f[start..].iter().sum::<f32>() / high_bins as f32
        })
        .sum::<f32>()
        / n_frames;

    let valence = ((hi_energy - low_energy) * 0.1).clamp(-1.0, 1.0);
    let novelty = (arousal * 0.5 + valence.abs() * 0.2).clamp(0.0, 1.0);

    PerceptualAffect::new(valence, arousal, novelty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_error() {
        let cfg = PerceptionConfig::default();
        let result = extract_audio_features(&[], 16000, &cfg);
        assert!(matches!(result, Err(PerceptionError::EmptyInput { .. })));
    }

    #[test]
    fn silence_returns_correct_dim() {
        let cfg = PerceptionConfig::default();
        let silence = vec![0.0f32; 16000]; // 1 s at 16 kHz
        let (features, _) = extract_audio_features(&silence, 16000, &cfg).unwrap();
        assert_eq!(features.len(), cfg.audio_feature_dim);
    }

    #[test]
    fn short_clip_produces_features() {
        let cfg = PerceptionConfig::default();
        let short = vec![0.5f32; 256]; // shorter than one FFT frame
        let (features, _) = extract_audio_features(&short, 16000, &cfg).unwrap();
        assert_eq!(features.len(), cfg.audio_feature_dim);
    }

    #[test]
    fn all_features_in_unit_range() {
        let cfg = PerceptionConfig::default();
        // 440 Hz sine wave at 16 kHz for 0.1 s
        let samples: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let (features, _) = extract_audio_features(&samples, 16000, &cfg).unwrap();
        for &v in &features {
            assert!(
                v >= 0.0 && v <= 1.0,
                "feature out of range [0,1]: {v}"
            );
        }
    }

    #[test]
    fn sine_wave_arousal_higher_than_silence() {
        let cfg = PerceptionConfig::default();
        let silence = vec![0.0f32; 8000];
        let sine: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let (_, silence_affect) = extract_audio_features(&silence, 16000, &cfg).unwrap();
        let (_, sine_affect) = extract_audio_features(&sine, 16000, &cfg).unwrap();
        assert!(
            sine_affect.arousal >= silence_affect.arousal,
            "sine arousal {:.3} should be >= silence arousal {:.3}",
            sine_affect.arousal,
            silence_affect.arousal,
        );
    }

    #[test]
    fn hann_window_tapers_to_zero() {
        let w = hann_window(512);
        assert_eq!(w.len(), 512);
        assert!(w[0].abs() < 1e-6, "Hann window should start near 0");
        assert!(w[511].abs() < 1e-6, "Hann window should end near 0");
        // Centre should be near 1.0
        assert!(w[256] > 0.99);
    }

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filterbank(32, 512, 16000);
        assert_eq!(filters.len(), 32);
        for f in &filters {
            assert_eq!(f.len(), 257); // n_fft/2 + 1
            // All weights should be in [0, 1]
            for &w in f {
                assert!(w >= 0.0 && w <= 1.0, "filter weight out of [0,1]: {w}");
            }
        }
    }
}
