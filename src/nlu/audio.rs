//! Native audio input for Gemma 4 (ADR 054).
//!
//! Gemma 4 is **encoder-free**: raw 16 kHz mono PCM is cut into 40 ms windows
//! (`640` samples each), and every window is mapped by a **single linear
//! projection** (`640 → 3840`, no bias) directly into the text-embedding space.
//! The projector lives in the `…mmproj.gguf` sidecar as the f16 tensor
//! `mm.a.input_projection.weight` (GGUF dims `[640, 3840]`, which candle reads
//! reversed → shape `[3840, 640]` = `[out, in]`, the standard `Linear`
//! orientation).
//!
//! The projected frames are spliced into the prompt at the audio-placeholder
//! positions; there is no separate audio encoder, attention stack or pooling.
//!
//! This module owns three concerns:
//! 1. [`AudioProjector`] — load the projector and run [`AudioProjector::project_audio`].
//! 2. WAV decoding + resampling to 16 kHz mono ([`decode_wav_to_16k_mono`]).
//! 3. A bounded HTTP fetch for audio URLs ([`fetch_media_bytes`]).

use std::path::Path;

use super::error::{NluError, NluResult};

/// Samples per 40 ms frame at 16 kHz (`16_000 * 0.040`).
pub const FRAME_SAMPLES: usize = 640;

/// Gemma 4 audio-projection output width (= text embedding length).
pub const AUDIO_PROJECTION_DIM: usize = 3840;

/// Target sample rate for Gemma 4 audio input.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Upper bound for the bounded media fetch (25 MiB) — large enough for a few
/// minutes of 16 kHz mono WAV, small enough to stay honest about memory.
pub const MAX_MEDIA_BYTES: usize = 25 * 1024 * 1024;

/// The Gemma 4 audio linear projector (`640 → 3840`), held on the inference
/// device as a dense f32 weight in `[out, in]` orientation.
#[cfg(feature = "candle-backend")]
pub struct AudioProjector {
    /// Projection weight, shape `[AUDIO_PROJECTION_DIM, FRAME_SAMPLES]` (`[out, in]`).
    weight: candle_core::Tensor,
    device: candle_core::Device,
}

#[cfg(feature = "candle-backend")]
impl AudioProjector {
    /// Load `mm.a.input_projection.weight` from the mmproj GGUF sidecar and
    /// dequantize it to f32 on `device`.
    ///
    /// Shape-checks that the projector maps `FRAME_SAMPLES → AUDIO_PROJECTION_DIM`
    /// so an unexpected orientation fails loudly instead of producing garbage.
    pub fn load(mmproj_path: &Path, device: &candle_core::Device) -> NluResult<Self> {
        use candle_core::quantized::gguf_file;

        if !mmproj_path.exists() {
            return Err(NluError::ModelNotFound {
                path: mmproj_path.to_path_buf(),
            });
        }

        let mut file = std::fs::File::open(mmproj_path).map_err(|e| NluError::ModelLoadFailed {
            reason: format!("open mmproj GGUF {}: {e}", mmproj_path.display()),
        })?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| NluError::ModelLoadFailed {
                reason: format!("read mmproj GGUF: {e}"),
            })?;

        const TENSOR: &str = "mm.a.input_projection.weight";
        let qtensor =
            content
                .tensor(&mut file, TENSOR, device)
                .map_err(|e| NluError::ModelLoadFailed {
                    reason: format!(
                        "mmproj is missing the audio projector tensor `{TENSOR}` \
                     (is this a Gemma 4 audio mmproj?): {e}"
                    ),
                })?;
        let weight = qtensor
            .dequantize(device)
            .map_err(|e| NluError::ModelLoadFailed {
                reason: format!("dequantize `{TENSOR}`: {e}"),
            })?;

        // candle reverses GGUF dims, so [640, 3840] → [3840, 640] = [out, in].
        let dims = weight.dims();
        if dims != [AUDIO_PROJECTION_DIM, FRAME_SAMPLES] {
            return Err(NluError::ModelLoadFailed {
                reason: format!(
                    "audio projector `{TENSOR}` has shape {dims:?}, expected \
                     [{AUDIO_PROJECTION_DIM}, {FRAME_SAMPLES}] ([out, in]). \
                     The GGUF dim orientation may have changed — refusing to \
                     splice audio with a mismatched projector."
                ),
            });
        }

        Ok(Self {
            weight,
            device: device.clone(),
        })
    }

    /// Project 16 kHz mono PCM into the text-embedding space.
    ///
    /// Pads (with zeros) or truncates `pcm` to a whole multiple of
    /// [`FRAME_SAMPLES`], reshapes to `[n_frames, 640]`, and applies the linear
    /// projection: `frames @ Wᵀ` with `W` shape `[3840, 640]` → `[n_frames, 3840]`.
    ///
    /// `n_frames = ceil(pcm.len() / 640)` (at least 1). The returned rows are
    /// the audio-frame embeddings to splice into the prompt — they are **not**
    /// `sqrt(embedding_length)`-scaled.
    pub fn project_audio(&self, pcm_16k_mono: &[f32]) -> NluResult<candle_core::Tensor> {
        use candle_core::Tensor;

        // Pad up to a whole frame; never produce zero frames.
        let n_frames = (pcm_16k_mono.len() + FRAME_SAMPLES - 1) / FRAME_SAMPLES;
        let n_frames = n_frames.max(1);
        let padded_len = n_frames * FRAME_SAMPLES;
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend_from_slice(pcm_16k_mono);
        padded.resize(padded_len, 0.0);

        let frames =
            Tensor::from_vec(padded, (n_frames, FRAME_SAMPLES), &self.device).map_err(|e| {
                NluError::LlmGenerationFailed {
                    reason: format!("audio frame tensor: {e}"),
                }
            })?;

        // [n_frames, 640] @ [640, 3840] = [n_frames, 3840].
        let w_t = self.weight.t().and_then(|w| w.contiguous()).map_err(|e| {
            NluError::LlmGenerationFailed {
                reason: format!("transpose projector: {e}"),
            }
        })?;
        let projected = frames
            .matmul(&w_t)
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("audio projection matmul: {e}"),
            })?;

        // Shape sanity: [n_frames, 3840].
        debug_assert_eq!(projected.dims(), [n_frames, AUDIO_PROJECTION_DIM]);
        Ok(projected)
    }
}

/// Decode a WAV byte buffer into 16 kHz mono `f32` PCM in `[-1, 1]`.
///
/// Channels are averaged to mono; integer samples are normalized by their
/// full-scale magnitude; the signal is then linearly resampled to 16 kHz.
pub fn decode_wav_bytes_to_16k_mono(bytes: &[u8]) -> NluResult<Vec<f32>> {
    use hound::SampleFormat;

    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor).map_err(|e| NluError::LlmGenerationFailed {
        reason: format!("parse WAV: {e}"),
    })?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let sample_rate = spec.sample_rate;

    // Read interleaved samples as f32 in [-1, 1].
    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| NluError::LlmGenerationFailed {
                reason: format!("read f32 WAV samples: {e}"),
            })?,
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| NluError::LlmGenerationFailed {
                    reason: format!("read int WAV samples: {e}"),
                })?
        }
    };

    if interleaved.is_empty() {
        return Err(NluError::LlmGenerationFailed {
            reason: "WAV contained no samples".to_string(),
        });
    }

    // Downmix to mono by averaging channels.
    let mono: Vec<f32> = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(resample_linear(&mono, sample_rate, TARGET_SAMPLE_RATE))
}

/// Convenience: read a WAV file from disk and decode to 16 kHz mono.
pub fn decode_wav_to_16k_mono(path: &Path) -> NluResult<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| NluError::ModelLoadFailed {
        reason: format!("read WAV {}: {e}", path.display()),
    })?;
    decode_wav_bytes_to_16k_mono(&bytes)
}

/// Linear resampler from `src_rate` to `dst_rate`. Adequate for speech LLM
/// input (the projector is robust to mild resampling artefacts); a higher-order
/// filter is unnecessary here.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.len() < 2 {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // Position in the source signal for output sample i.
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Download a media URL into a bounded byte buffer (≤ [`MAX_MEDIA_BYTES`]).
///
/// The agent's [`crate::agent::tools::HttpFetchTool`] caps responses at 256 KB,
/// which is far too small for audio; this is a dedicated bounded reader that
/// streams the body and aborts if it exceeds the limit.
pub fn fetch_media_bytes(url: &str, timeout_secs: u64) -> NluResult<Vec<u8>> {
    use std::io::Read;

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(NluError::LlmGenerationFailed {
            reason: format!("invalid media URL `{url}`: must start with http(s)://"),
        });
    }

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| NluError::LlmGenerationFailed {
            reason: format!("fetch media `{url}`: {e}"),
        })?;

    // Read at most MAX_MEDIA_BYTES + 1 to detect overflow.
    let mut reader = response.into_reader().take(MAX_MEDIA_BYTES as u64 + 1);
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| NluError::LlmGenerationFailed {
            reason: format!("read media body: {e}"),
        })?;
    if buf.len() > MAX_MEDIA_BYTES {
        return Err(NluError::LlmGenerationFailed {
            reason: format!(
                "media `{url}` exceeds {} MiB cap — refusing to load",
                MAX_MEDIA_BYTES / (1024 * 1024)
            ),
        });
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity_when_rates_match() {
        let s = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&s, 16_000, 16_000), s);
    }

    #[test]
    fn resample_downsamples_length() {
        let s: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let out = resample_linear(&s, 48_000, 16_000);
        // 48k -> 16k is a 3x reduction.
        assert!(
            (out.len() as i64 - 16_000).abs() <= 1,
            "len = {}",
            out.len()
        );
    }

    #[test]
    fn frame_count_math() {
        // 1000 samples -> 2 frames of 640 (640 + 360 padded).
        assert_eq!((1000 + FRAME_SAMPLES - 1) / FRAME_SAMPLES, 2);
        // exactly one frame.
        assert_eq!((640 + FRAME_SAMPLES - 1) / FRAME_SAMPLES, 1);
    }
}
