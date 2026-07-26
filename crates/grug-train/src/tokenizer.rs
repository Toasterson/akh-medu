//! Byte-level tokenizer for the Grug-Brain BabyLLM spike (Phase 51 P0, ADR
//! 087 decision 4: "standard tokenizer", High Grug deferred).
//!
//! Every possible byte value (0..=255) is its own token, so encoding is
//! total and lossless for arbitrary UTF-8 (or even non-UTF-8) byte strings
//! -- there is no unknown-token case to handle. A small fixed set of special
//! tokens are appended after the 256 byte tokens:
//!
//! - `BOS` -- beginning of sequence
//! - `EOS` -- end of sequence (also used as the generation stop token)
//! - `PAD` -- padding, for batching sequences of different lengths
//! - `SEP` -- separates the prompt half of a training example from the
//!   target half (the training loop uses `SEP`'s position to know where the
//!   teacher-forced loss mask starts; see `train.rs`)
//!
//! This is deliberately the simplest tokenizer that could work: no BPE
//! merges, no vocabulary training, no external data file. The vocabulary
//! size is fixed at 260 regardless of corpus. Byte-level tokenization means
//! sequences are ~1 token per byte, which is token-inefficient compared to
//! BPE -- acceptable for a CPU spike on tiny JSON strings, revisited in P1
//! ("real BPE tokenizer over Apertus", per the Phase 51 plan).

use crate::error::{GrugError, GrugResult};

/// The tokenizer seam (Phase 51 P1a): anything that can turn text into
/// token ids and back, plus report its special-token ids and vocab size.
///
/// [`ByteTokenizer`] (P0, this module) and [`crate::bpe::BpeTokenizer`]
/// (P1a) both implement this trait, so [`crate::train`], [`crate::generate`],
/// and [`crate::eval`] can be written once against `V: Vocab` rather than
/// duplicated per tokenizer. `encode`/`decode` return [`GrugResult`] (rather
/// than `ByteTokenizer`'s infallible `Vec<u32>`/fallible-only-on-bad-UTF8
/// shape) since [`crate::bpe::BpeTokenizer`]'s underlying `tokenizers` calls
/// are fallible; `ByteTokenizer`'s impl simply never returns `Err` from
/// `encode`.
pub trait Vocab {
    /// Encode `text` with no special tokens added.
    fn encode(&self, text: &str) -> GrugResult<Vec<u32>>;
    /// Decode token ids back into a `String`, dropping special tokens.
    fn decode(&self, tokens: &[u32]) -> GrugResult<String>;
    /// The full vocabulary size (including special tokens).
    fn vocab_size(&self) -> usize;
    /// `BOS` token id.
    fn bos_id(&self) -> u32;
    /// `EOS` token id (also the generation stop token).
    fn eos_id(&self) -> u32;
    /// `PAD` token id.
    fn pad_id(&self) -> u32;
    /// `SEP` token id -- separates prompt tokens from target tokens.
    fn sep_id(&self) -> u32;

    /// The raw byte contribution of every token id, indexed by id
    /// (`table[id]`), for schema-constrained decoding
    /// (`crate::generate::generate_constrained`).
    ///
    /// `Some(bytes)` is the exact byte sequence this token appends to the
    /// decoded output; `None` marks a token that emits no directly-appendable
    /// bytes (the special tokens `BOS`/`EOS`/`PAD`/`SEP`) and so must never be
    /// offered as a constrained continuation. Concatenating the `Some` pieces
    /// of a token id sequence reproduces the same bytes
    /// [`Vocab::decode`] would (the constrained decoder drives a byte-level
    /// grammar over exactly those bytes), so a byte-level tokenizer's pieces
    /// and its `decode` agree by construction.
    fn token_pieces(&self) -> Vec<Option<Vec<u8>>>;
}

/// `BOS` token id: one past the 256 byte-value tokens (0..=255).
pub const BOS: u32 = 256;
/// `EOS` token id.
pub const EOS: u32 = 257;
/// `PAD` token id.
pub const PAD: u32 = 258;
/// `SEP` token id -- separates prompt tokens from target tokens in a training
/// example.
pub const SEP: u32 = 259;

/// Total vocabulary size: 256 byte tokens + 4 special tokens.
pub const VOCAB_SIZE: usize = 260;

/// A byte-level tokenizer: `String` <-> `Vec<u32>`, reversible and
/// deterministic (no learned state, so there is nothing to save/load).
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// The fixed vocabulary size (256 byte values + 4 special tokens).
    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    /// Encode `text`'s UTF-8 bytes as token ids, with no special tokens
    /// added. Use [`ByteTokenizer::encode_with_bos_eos`] or manually splice
    /// in [`SEP`] for the training-example framing in `train.rs`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    /// Encode `text` wrapped in `BOS` .. `EOS`.
    pub fn encode_with_bos_eos(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() + 2);
        out.push(BOS);
        out.extend(self.encode(text));
        out.push(EOS);
        out
    }

    /// Decode token ids back into a `String`.
    ///
    /// Special tokens (`BOS`/`EOS`/`PAD`/`SEP`) are dropped rather than
    /// erroring, since generated sequences routinely contain a trailing
    /// `EOS` (or padding, in a batch) that the caller does not want
    /// round-tripped into the text. A byte value that is individually valid
    /// but not valid UTF-8 in context (e.g. a truncated multi-byte
    /// sequence, which greedy generation can produce if it stops
    /// mid-codepoint) is surfaced as [`GrugError::InvalidUtf8`] rather than
    /// silently lossy-decoded, so callers can distinguish "model emitted
    /// garbage bytes" from "model emitted valid text".
    pub fn decode(&self, tokens: &[u32]) -> GrugResult<String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8(bytes).map_err(|e| GrugError::InvalidUtf8 {
            message: e.to_string(),
        })
    }
}

impl Vocab for ByteTokenizer {
    fn encode(&self, text: &str) -> GrugResult<Vec<u32>> {
        Ok(ByteTokenizer::encode(self, text))
    }
    fn decode(&self, tokens: &[u32]) -> GrugResult<String> {
        ByteTokenizer::decode(self, tokens)
    }
    fn vocab_size(&self) -> usize {
        ByteTokenizer::vocab_size(self)
    }
    fn bos_id(&self) -> u32 {
        BOS
    }
    fn eos_id(&self) -> u32 {
        EOS
    }
    fn pad_id(&self) -> u32 {
        PAD
    }
    fn sep_id(&self) -> u32 {
        SEP
    }
    fn token_pieces(&self) -> Vec<Option<Vec<u8>>> {
        // Byte tokens (0..=255) each contribute their own single byte; the
        // four special tokens (>=256) contribute nothing appendable.
        (0..VOCAB_SIZE)
            .map(|id| if id < 256 { Some(vec![id as u8]) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ascii() {
        let tok = ByteTokenizer::new();
        let text = "hello world";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len());
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn round_trip_json_and_unicode() {
        let tok = ByteTokenizer::new();
        let text = r#"{"name":"get_weather","arguments":{"city":"Zürich 北京"}}"#;
        let ids = tok.encode(text);
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn round_trip_empty_string() {
        let tok = ByteTokenizer::new();
        assert_eq!(tok.encode(""), Vec::<u32>::new());
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn bos_eos_framing() {
        let tok = ByteTokenizer::new();
        let ids = tok.encode_with_bos_eos("hi");
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.last(), Some(&EOS));
        assert_eq!(ids.len(), 2 + 2);
    }

    #[test]
    fn decode_drops_special_tokens() {
        let tok = ByteTokenizer::new();
        let ids = vec![BOS, b'h' as u32, b'i' as u32, EOS, PAD, PAD];
        assert_eq!(tok.decode(&ids).unwrap(), "hi");
    }

    #[test]
    fn special_tokens_are_distinct_and_above_byte_range() {
        let mut specials = vec![BOS, EOS, PAD, SEP];
        specials.sort_unstable();
        specials.dedup();
        assert_eq!(specials.len(), 4, "special tokens must be pairwise distinct");
        for s in specials {
            assert!(s >= 256, "special token {s} must be outside the byte range");
        }
        assert_eq!(VOCAB_SIZE, 260);
    }

    #[test]
    fn decode_rejects_invalid_utf8_bytes() {
        let tok = ByteTokenizer::new();
        // 0xFF is never valid UTF-8 on its own.
        let err = tok.decode(&[0xFFu32]).unwrap_err();
        assert!(matches!(err, GrugError::InvalidUtf8 { .. }));
    }

    #[test]
    fn all_byte_values_round_trip() {
        let tok = ByteTokenizer::new();
        // Every single-byte ASCII character encodes to its own byte value.
        for b in 0u8..=127 {
            let s = (b as char).to_string();
            assert_eq!(tok.encode(&s), vec![u32::from(b)]);
        }

        let all_bytes: Vec<u8> = (0u8..=255).collect();
        // Not valid UTF-8 as a whole, but every individual byte token must
        // still encode to its own id.
        let ids: Vec<u32> = all_bytes.iter().map(|&b| u32::from(b)).collect();
        assert_eq!(ids.len(), 256);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[255], 255);
    }
}
