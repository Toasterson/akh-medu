//! Bounded audio for the native ACP/ChatBackend path. Originals remain at the
//! caller; this first input contract accepts complete 16 kHz mono PCM16 WAVs.

use sha2::{Digest, Sha256};

pub const MAX_AUDIO_BYTES: usize = 4 * 1024 * 1024;
const MAX_AUDIO_SAMPLES: u32 = 120 * 16_000;

#[derive(Clone, PartialEq)]
pub struct ChatPromptAudio {
    bytes: Vec<u8>,
    sha256: String,
    samples: u32,
}

impl std::fmt::Debug for ChatPromptAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatPromptAudio")
            .field("bytes", &self.bytes.len())
            .field("sha256", &self.sha256)
            .field("duration_seconds", &self.duration_seconds())
            .finish()
    }
}

impl ChatPromptAudio {
    pub fn from_wav(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_AUDIO_BYTES {
            return Err(format!("audio must contain 1..={MAX_AUDIO_BYTES} bytes"));
        }
        let mut wav = hound::WavReader::new(std::io::Cursor::new(&bytes))
            .map_err(|error| format!("invalid audio WAV: {error}"))?;
        let spec = wav.spec();
        if spec.channels != 1
            || spec.sample_rate != 16_000
            || spec.bits_per_sample != 16
            || spec.sample_format != hound::SampleFormat::Int
        {
            return Err(
                "audio must be 16 kHz mono PCM16 WAV; resample a copy of the original".into(),
            );
        }
        let samples = wav.duration();
        if samples == 0 || samples > MAX_AUDIO_SAMPLES {
            return Err("audio must contain more than zero and at most 120 seconds".into());
        }
        // Validate the complete data, not merely a plausible header. A truncated
        // attachment must not reach inference as a silently shortened recording.
        let actual = wav
            .samples::<i16>()
            .try_fold(0u32, |n, sample| sample.map(|_| n + 1))
            .map_err(|error| format!("incomplete audio WAV: {error}"))?;
        if actual != samples {
            return Err("audio WAV data does not match its declared duration".into());
        }
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        Ok(Self {
            bytes,
            sha256,
            samples,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn duration_seconds(&self) -> f64 {
        f64::from(self.samples) / 16_000.0
    }
}

#[cfg(test)]
pub(crate) fn fixture_wav() -> Vec<u8> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut wav = hound::WavWriter::new(
            &mut cursor,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..160 {
            wav.write_sample(100i16).unwrap();
        }
        wav.finalize().unwrap();
    }
    cursor.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_preserves_bytes_and_checks_complete_wav() {
        let bytes = fixture_wav();
        let audio = ChatPromptAudio::from_wav(bytes.clone()).unwrap();
        assert_eq!(audio.bytes(), bytes);
        assert_eq!(audio.duration_seconds(), 0.01);
        assert_eq!(audio.sha256(), format!("{:x}", Sha256::digest(&bytes)));
        assert!(ChatPromptAudio::from_wav(bytes[..bytes.len() - 10].to_vec()).is_err());
        assert!(ChatPromptAudio::from_wav(b"not a WAV".to_vec()).is_err());
        assert!(ChatPromptAudio::from_wav(vec![0; MAX_AUDIO_BYTES + 1]).is_err());
        let mut slow_rate = bytes;
        slow_rate[24..28].copy_from_slice(&1u32.to_le_bytes());
        assert!(ChatPromptAudio::from_wav(slow_rate).is_err());
    }
}
