//! BPE tokenizer for the Phase 51 P1a scale-up (ADR 087, Phase 51 plan
//! "Tokenizer" bullet under P1a).
//!
//! [`ByteTokenizer`](crate::tokenizer::ByteTokenizer) (P0) is one token per
//! byte -- simple and total, but token-inefficient: every JSON brace, quote,
//! and digit costs a full sequence position, which both wastes context and
//! makes the next-token-prediction task harder than it needs to be for a
//! tiny model. [`BpeTokenizer`] trains a real (throwaway, ~2-4K vocab) BPE
//! merge table over the actual training corpus using the workspace
//! `tokenizers` 0.22 crate -- the same crate this workspace already uses for
//! `nlu-ml`/`candle-backend` (see root `Cargo.toml`), so no new dependency
//! enters the graph.
//!
//! # Design
//!
//! - **Byte-level pre-tokenization + decoder** (GPT-2 style): every possible
//!   input byte maps into the BPE alphabet via a reversible byte<->unicode
//!   remapping, so encoding is total over arbitrary UTF-8 (or non-UTF-8)
//!   byte strings -- there is no "unknown token" case, exactly matching
//!   [`ByteTokenizer`]'s totality guarantee, just with far fewer tokens per
//!   example once merges have been learned.
//! - **Same four special tokens** as the byte tokenizer (`<bos>`, `<eos>`,
//!   `<pad>`, `<sep>`), added as genuine special/added tokens so they never
//!   get merged into or split out of ordinary BPE subwords, and so
//!   `decode(..., skip_special_tokens: true)` drops them automatically --
//!   the same "special tokens don't round-trip into decoded text" contract
//!   the byte tokenizer's `decode` has.
//! - **Save/load to a single JSON file** via `tokenizers::Tokenizer::save` /
//!   `Tokenizer::from_file`, so a trained vocabulary survives a training run
//!   (written under the run's artifacts dir) and can be reloaded for
//!   `--resume` or standalone generation later.
//! - This module does *not* replace [`crate::tokenizer::ByteTokenizer`] --
//!   both exist behind the same conceptual `encode`/`decode` shape (see
//!   `train.rs`'s and `eval.rs`'s use of whichever tokenizer is passed in),
//!   per the plan's "keep the existing tokenizer seam" instruction.

use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::{AddedToken, Tokenizer};

use crate::error::{GrugError, GrugResult};

/// `<bos>` special token string.
pub const BOS_TOKEN: &str = "<bos>";
/// `<eos>` special token string (also the generation stop token).
pub const EOS_TOKEN: &str = "<eos>";
/// `<pad>` special token string.
pub const PAD_TOKEN: &str = "<pad>";
/// `<sep>` special token string -- separates prompt tokens from target
/// tokens in a training example, mirroring [`crate::tokenizer::SEP`].
pub const SEP_TOKEN: &str = "<sep>";

/// The four special tokens, in the fixed order their ids are assigned when
/// training a fresh vocabulary: `[BOS, EOS, PAD, SEP]`. Kept as a single slice
/// so training and any future re-derivation agree on order.
const SPECIAL_TOKENS: [&str; 4] = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEP_TOKEN];

/// A trained byte-level BPE tokenizer, wrapping `tokenizers::Tokenizer`.
///
/// Unlike [`crate::tokenizer::ByteTokenizer`] (which has no learned state),
/// this type must be trained (or loaded from a previously-trained file)
/// before use -- there is no `Default`/`new()` that produces a usable empty
/// tokenizer, by design: an empty BPE vocabulary cannot encode anything
/// meaningful.
#[derive(Debug)]
pub struct BpeTokenizer {
    inner: Tokenizer,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
    sep_id: u32,
}

impl BpeTokenizer {
    /// Train a fresh BPE vocabulary of (approximately) `vocab_size` merges +
    /// base alphabet + specials, over `corpus` (one training string per
    /// element -- callers typically pass formatted prompts + target JSON
    /// strings drawn from `grug_data`).
    ///
    /// `vocab_size` is a target ceiling, not a guarantee: the trainer stops
    /// early if the corpus doesn't support that many merges (e.g. a tiny or
    /// low-diversity corpus). The four special tokens in [`SPECIAL_TOKENS`]
    /// are always reserved regardless of corpus content.
    pub fn train<I, S>(corpus: I, vocab_size: usize) -> GrugResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let bpe = BPE::default();
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::new(
            /* add_prefix_space */ false, /* trim_offsets */ true,
            /* use_regex */ true,
        )));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(true, true, true)));

        let special_tokens: Vec<AddedToken> = SPECIAL_TOKENS
            .iter()
            .map(|t| AddedToken::from(t.to_string(), true))
            .collect();

        // `initial_alphabet` wants a `std::collections::HashSet<char>`, while
        // `ByteLevel::alphabet()` returns an `ahash::AHashSet<char>` (a
        // faster-hashing set used internally by `tokenizers`) -- collect
        // through the iterator to convert.
        let initial_alphabet: std::collections::HashSet<char> =
            ByteLevelPreTokenizer::alphabet().into_iter().collect();

        let trainer: BpeTrainerBuilder = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(1)
            .special_tokens(special_tokens)
            .show_progress(false)
            // The byte-level alphabet (256 remapped byte-unicode symbols) is
            // the base vocabulary every BPE merge builds on -- without this,
            // `train` would only learn merges over whatever characters
            // happen to appear in `corpus`, breaking totality on unseen
            // bytes at inference time.
            .initial_alphabet(initial_alphabet);
        let mut trainer = TrainerWrapper::BpeTrainer(trainer.build());

        let corpus: Vec<String> = corpus.into_iter().map(|s| s.as_ref().to_string()).collect();
        if corpus.is_empty() {
            return Err(GrugError::EmptyTrainingCorpus);
        }

        tokenizer
            .train(&mut trainer, corpus.into_iter())
            .map_err(|e| GrugError::TokenizerTrainingFailed {
                message: e.to_string(),
            })?;

        Self::from_tokenizer(tokenizer)
    }

    /// Load a previously trained tokenizer from a JSON file written by
    /// [`BpeTokenizer::save`].
    pub fn load(path: impl AsRef<Path>) -> GrugResult<Self> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path).map_err(|e| GrugError::TokenizerLoadFailed {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Self::from_tokenizer(tokenizer)
    }

    /// Save this tokenizer to a single JSON file (creating parent
    /// directories as needed).
    pub fn save(&self, path: impl AsRef<Path>) -> GrugResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| GrugError::TokenizerSaveFailed {
                path: path.display().to_string(),
                message: format!("creating parent dir: {e}"),
            })?;
        }
        self.inner
            .save(path, /* pretty */ false)
            .map_err(|e| GrugError::TokenizerSaveFailed {
                path: path.display().to_string(),
                message: e.to_string(),
            })
    }

    /// Resolve the four special-token ids from an already-built/loaded
    /// `Tokenizer`, and wrap it as a [`BpeTokenizer`].
    fn from_tokenizer(inner: Tokenizer) -> GrugResult<Self> {
        let id_of = |name: &str| {
            inner
                .token_to_id(name)
                .ok_or_else(|| GrugError::TokenizerMissingSpecialToken {
                    token: name.to_string(),
                })
        };
        Ok(Self {
            bos_id: id_of(BOS_TOKEN)?,
            eos_id: id_of(EOS_TOKEN)?,
            pad_id: id_of(PAD_TOKEN)?,
            sep_id: id_of(SEP_TOKEN)?,
            inner,
        })
    }

    /// `<bos>` token id (assigned at training time; not assumed fixed).
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
    /// `<eos>` token id.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    /// `<pad>` token id.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
    /// `<sep>` token id.
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }

    /// The full vocabulary size, including the byte-level base alphabet,
    /// learned merges, and the four special tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(/* with_added_tokens */ true)
    }

    /// Encode `text` with no special tokens added -- the counterpart to
    /// [`crate::tokenizer::ByteTokenizer::encode`].
    pub fn encode(&self, text: &str) -> GrugResult<Vec<u32>> {
        let encoding =
            self.inner
                .encode(text, false)
                .map_err(|e| GrugError::TokenizerEncodeFailed {
                    message: e.to_string(),
                })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Per-id byte contribution for schema-constrained decoding (see
    /// [`crate::tokenizer::Vocab::token_pieces`]).
    ///
    /// Byte-level BPE stores token surfaces in the GPT-2 byte<->unicode
    /// *remapped* alphabet (a space is the char `Ġ`, etc.), so
    /// `id_to_token(id)` returns those remapped chars, not raw bytes. We invert
    /// the GPT-2 map ([`gpt2_char_to_byte`]) char-by-char to recover the exact
    /// bytes the token appends to the decoded output. The four special tokens
    /// map to `None` (they contribute no appendable bytes); any token whose
    /// surface contains a char outside the byte-level alphabet also maps to
    /// `None` (should not occur for a byte-level BPE, but fail closed rather
    /// than guess a byte).
    fn token_pieces_impl(&self) -> Vec<Option<Vec<u8>>> {
        let char_to_byte = gpt2_char_to_byte();
        let vocab_size = self.vocab_size();
        let specials = [self.bos_id, self.eos_id, self.pad_id, self.sep_id];
        (0..vocab_size as u32)
            .map(|id| {
                if specials.contains(&id) {
                    return None;
                }
                let surface = self.inner.id_to_token(id)?;
                let mut bytes = Vec::with_capacity(surface.len());
                for ch in surface.chars() {
                    bytes.push(*char_to_byte.get(&ch)?);
                }
                Some(bytes)
            })
            .collect()
    }

    /// Decode token ids back into a `String`, dropping special tokens
    /// (`<bos>`/`<eos>`/`<pad>`/`<sep>`) rather than round-tripping them into
    /// the output -- same contract as
    /// [`crate::tokenizer::ByteTokenizer::decode`].
    pub fn decode(&self, tokens: &[u32]) -> GrugResult<String> {
        self.inner
            .decode(tokens, /* skip_special_tokens */ true)
            .map_err(|e| GrugError::TokenizerDecodeFailed {
                message: e.to_string(),
            })
    }
}

impl crate::tokenizer::Vocab for BpeTokenizer {
    fn encode(&self, text: &str) -> GrugResult<Vec<u32>> {
        BpeTokenizer::encode(self, text)
    }
    fn decode(&self, tokens: &[u32]) -> GrugResult<String> {
        BpeTokenizer::decode(self, tokens)
    }
    fn vocab_size(&self) -> usize {
        BpeTokenizer::vocab_size(self)
    }
    fn bos_id(&self) -> u32 {
        self.bos_id
    }
    fn eos_id(&self) -> u32 {
        self.eos_id
    }
    fn pad_id(&self) -> u32 {
        self.pad_id
    }
    fn sep_id(&self) -> u32 {
        self.sep_id
    }
    fn token_pieces(&self) -> Vec<Option<Vec<u8>>> {
        self.token_pieces_impl()
    }
}

/// The inverse of GPT-2's `bytes_to_unicode` bijection: maps each remapped
/// alphabet char back to its raw byte. This is the standard byte-level BPE
/// alphabet (`tokenizers`' `ByteLevel::alphabet()` builds the forward map from
/// the same construction), reproduced here so [`BpeTokenizer::token_pieces`]
/// can turn a token's remapped-char surface back into the bytes it emits,
/// without depending on a private helper in the `tokenizers` crate.
fn gpt2_char_to_byte() -> std::collections::HashMap<char, u8> {
    // Bytes that map to themselves (printable ASCII + two Latin-1 ranges).
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs: Vec<u32> = bs.clone();
    // The remaining bytes map to fresh code points at 256+n, in byte order.
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .filter_map(|(&b, &c)| char::from_u32(c).map(|ch| (ch, b as u8)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_corpus() -> Vec<String> {
        vec![
            r#"{"name":"get_weather","arguments":{"city":"Zurich"}}"#.to_string(),
            r#"{"name":"get_weather","arguments":{"city":"Berlin"}}"#.to_string(),
            r#"{"name":"send_email","arguments":{"to":"a@b.com","subject":"hi"}}"#.to_string(),
            "what is the weather like in zurich right now".to_string(),
            "please send an email to a@b.com with subject hi".to_string(),
            "book a table for four people tonight at 8pm".to_string(),
            r#"{"name":"book_table","arguments":{"party_size":4,"time":"20:00"}}"#.to_string(),
        ]
        .into_iter()
        .cycle()
        .take(200)
        .collect()
    }

    #[test]
    fn trains_and_reports_a_bounded_vocab_size() {
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        // At least the byte-level base alphabet (256) + 4 specials; capped
        // at (approximately) the requested ceiling.
        assert!(
            tok.vocab_size() >= 260,
            "vocab suspiciously small: {}",
            tok.vocab_size()
        );
        assert!(
            tok.vocab_size() <= 512 + 16,
            "vocab exceeded requested ceiling: {}",
            tok.vocab_size()
        );
    }

    #[test]
    fn round_trip_ascii_and_json() {
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        for text in [
            "hello world",
            r#"{"name":"get_weather","arguments":{"city":"Zurich"}}"#,
            "",
        ] {
            let ids = tok.encode(text).unwrap();
            let back = tok.decode(&ids).unwrap();
            assert_eq!(back, text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn round_trip_unicode_never_seen_in_training_corpus() {
        // Byte-level pre-tokenization must stay total even over bytes/glyphs
        // absent from the training corpus (the whole point of byte-level
        // BPE over a plain word/subword-level BPE).
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        let text = "Zürich 北京 emoji test \u{1F600}";
        let ids = tok.encode(text).unwrap();
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn special_tokens_are_distinct_and_excluded_from_decode() {
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        // Encode "hi there" as one string (not two independently-encoded
        // fragments spliced together) so byte-level BPE's leading-space
        // convention (GPT-2 style: a space is encoded as part of the
        // *following* token, e.g. "Ġthere") round-trips correctly -- gluing
        // together separately-encoded "hi" and "there" would legitimately
        // lose the space between them, which is a fragment-splicing
        // artifact, not a tokenizer bug.
        let mut ids = vec![tok.bos_id()];
        ids.extend(tok.encode("hi there").unwrap());
        ids.push(tok.sep_id());
        ids.extend(tok.encode("ok").unwrap());
        ids.push(tok.eos_id());
        ids.push(tok.pad_id());

        let mut specials = vec![tok.bos_id(), tok.eos_id(), tok.pad_id(), tok.sep_id()];
        specials.sort_unstable();
        specials.dedup();
        assert_eq!(
            specials.len(),
            4,
            "special tokens must be pairwise distinct"
        );

        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "hi thereok");
    }

    #[test]
    fn save_and_load_round_trip_preserves_vocab_and_specials() {
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        let dir = std::env::temp_dir().join(format!("grug_train_bpe_test_{}", std::process::id()));
        let path = dir.join("tokenizer.json");
        let _ = std::fs::remove_file(&path);

        tok.save(&path).unwrap();
        let loaded = BpeTokenizer::load(&path).unwrap();

        assert_eq!(loaded.vocab_size(), tok.vocab_size());
        assert_eq!(loaded.bos_id(), tok.bos_id());
        assert_eq!(loaded.eos_id(), tok.eos_id());
        assert_eq!(loaded.pad_id(), tok.pad_id());
        assert_eq!(loaded.sep_id(), tok.sep_id());

        let text = r#"{"name":"send_email","arguments":{"to":"a@b.com","subject":"hi"}}"#;
        assert_eq!(loaded.encode(text).unwrap(), tok.encode(text).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_corpus_is_rejected() {
        let err = BpeTokenizer::train(Vec::<String>::new(), 512).unwrap_err();
        assert!(matches!(err, GrugError::EmptyTrainingCorpus));
    }

    /// `token_pieces` must reconstruct exactly the bytes `decode` produces:
    /// concatenating the per-token byte pieces of an encoded sequence yields
    /// the original text's bytes. This validates the GPT-2 byte<->unicode
    /// inverse map against the tokenizer's own decoder over real JSON.
    #[test]
    fn token_pieces_concatenate_to_the_decoded_bytes() {
        use crate::tokenizer::Vocab;
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        let pieces = Vocab::token_pieces(&tok);
        assert_eq!(pieces.len(), tok.vocab_size());

        for text in [
            r#"{"arguments":{"city":"Zurich"},"name":"get_weather"}"#,
            r#"{"arguments":{"party_size":4,"time":"20:00"},"name":"book_table"}"#,
            "please send an email to a@b.com with subject hi",
            "Zürich 北京",
        ] {
            let ids = Vocab::encode(&tok, text).unwrap();
            let mut reconstructed: Vec<u8> = Vec::new();
            for &id in &ids {
                let piece = pieces[id as usize]
                    .as_ref()
                    .unwrap_or_else(|| panic!("ordinary token {id} has no byte piece"));
                reconstructed.extend_from_slice(piece);
            }
            assert_eq!(
                reconstructed,
                text.as_bytes(),
                "token_pieces did not reconstruct {text:?}"
            );
        }
    }

    /// The four special tokens contribute no appendable bytes.
    #[test]
    fn token_pieces_are_none_for_special_tokens() {
        use crate::tokenizer::Vocab;
        let tok = BpeTokenizer::train(sample_corpus(), 512).unwrap();
        let pieces = Vocab::token_pieces(&tok);
        for id in [tok.bos_id(), tok.eos_id(), tok.pad_id(), tok.sep_id()] {
            assert!(
                pieces[id as usize].is_none(),
                "special token {id} should have no byte piece"
            );
        }
    }
}
