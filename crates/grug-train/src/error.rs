//! Error types for `grug-train`.
//!
//! Follows the repo-wide convention: `miette` + `thiserror`, with rich
//! `#[diagnostic(help(...))]` text so a human (or agent) hitting this error
//! knows exactly what to do next, not just what broke.

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for `grug-train` public APIs.
pub type GrugResult<T> = Result<T, GrugError>;

/// Errors that can occur while building, training, or evaluating a
/// Grug-Brain beachhead model.
#[derive(Debug, Error, Diagnostic)]
pub enum GrugError {
    /// A gradient we expected to exist after `.backward()` was missing.
    ///
    /// This should only happen if a tensor was created without
    /// `.require_grad()`, or if it was detached somewhere in the forward
    /// pass before the loss was computed.
    #[error("expected gradient for parameter `{param}` but none was recorded")]
    #[diagnostic(
        code(grug_train::missing_gradient),
        help(
            "Check that every leaf tensor that should receive a gradient was \
             created with `.require_grad()` (or is a Burn `Param`, which is \
             tracked automatically), and that the loss tensor was produced by \
             an unbroken chain of autodiff-tracked ops back to it. Also verify \
             the backend type alias is `Autodiff<NdArray>` (or similar) and not \
             the plain inference backend -- `.backward()` on a non-Autodiff \
             backend silently does nothing."
        )
    )]
    MissingGradient { param: String },

    /// A gradient existed but was exactly zero across every element, which
    /// almost always means the computation graph did not actually connect
    /// the parameter to the loss (dead branch, wrong tensor cloned, etc).
    #[error("gradient for parameter `{param}` was present but all-zero")]
    #[diagnostic(
        code(grug_train::zero_gradient),
        help(
            "An all-zero gradient usually means the parameter did not \
             actually participate in the forward computation that produced \
             the loss -- e.g. a stale clone was used instead of the live \
             `Param`, or a `detach()` call cut the graph. Add a print of the \
             forward pass shapes/values, or bisect the module to find where \
             the connection to `param` was lost."
        )
    )]
    ZeroGradient { param: String },

    /// The tiny sanity-check training loop did not reduce the loss on its
    /// fixed target within the configured number of steps.
    #[error(
        "training loop did not reduce loss: start={start_loss:.6}, end={end_loss:.6} over {steps} steps"
    )]
    #[diagnostic(
        code(grug_train::loss_did_not_decrease),
        help(
            "This is the P0 de-risking gate for Phase 51 (see \
             docs/ai/decisions/087-grug-brain-baby-llm-training.md) -- it means \
             autodiff is wired but optimization is not actually improving the \
             model. Try: (1) increasing the learning rate, (2) checking the \
             optimizer step is applied to the *same* parameter set the forward \
             pass reads (no stale snapshot), (3) confirming the loss function \
             direction is correct (minimizing, not maximizing), (4) running more \
             steps -- a flat start on a randomly-initialized tiny net can need a \
             few extra iterations to leave a plateau."
        )
    )]
    LossDidNotDecrease {
        start_loss: f64,
        end_loss: f64,
        steps: usize,
    },

    /// A decoded token sequence was not valid UTF-8.
    ///
    /// Greedy generation can, in principle, stop mid-codepoint (e.g. if
    /// `EOS` or `max_len` is hit right after the first byte of a multi-byte
    /// UTF-8 sequence). Surfacing this distinctly lets callers tell "the
    /// model emitted well-formed but wrong text" apart from "the model
    /// emitted byte garbage".
    #[error("decoded token sequence was not valid UTF-8: {message}")]
    #[diagnostic(
        code(grug_train::invalid_utf8),
        help(
            "This can happen if generation stops (EOS or max_len) in the \
             middle of a multi-byte UTF-8 codepoint. Options: (1) treat this \
             example as a scoring failure rather than panicking -- the eval \
             harness should count it as invalid output, not crash; (2) if it \
             happens often, consider decoding lossily with \
             `String::from_utf8_lossy` for display purposes while still \
             scoring the strict version as invalid."
        )
    )]
    InvalidUtf8 { message: String },

    /// Writing a JSONL eval time-series row to disk failed.
    #[error("failed to write eval time-series row: {message}")]
    #[diagnostic(
        code(grug_train::eval_log_write_failed),
        help(
            "Check that the parent directory of the configured eval log path \
             exists and is writable. The eval harness appends one JSON line \
             per report; consider running the example with a path under a \
             directory you know is writable (e.g. `/tmp` or a scratch dir) if \
             this keeps failing."
        )
    )]
    EvalLogWriteFailed { message: String },

    /// Serializing an eval time-series row to JSON failed.
    #[error("failed to serialize eval time-series row: {message}")]
    #[diagnostic(
        code(grug_train::eval_log_serialize_failed),
        help(
            "This indicates a bug in the `EvalLogRow` type (e.g. a NaN/Inf \
             float, which `serde_json` cannot represent) rather than an \
             environment problem. Check whether any rate field could be NaN \
             (e.g. an empty batch producing a 0/0 division) before this \
             point."
        )
    )]
    EvalLogSerializeFailed { message: String },

    /// The underlying `grug-data` crate returned an error while generating
    /// examples or building the dataset split.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Data(#[from] grug_data::GrugError),

    /// [`crate::bpe::BpeTokenizer::train`] was given an empty corpus.
    #[error("cannot train a BPE tokenizer on an empty corpus")]
    #[diagnostic(
        code(grug_train::empty_training_corpus),
        help(
            "Pass at least one training string. Typically this comes from \
             `grug_data::Dataset::train_examples` -- check the requested \
             example count and dataset seed produced a non-empty `Vec`."
        )
    )]
    EmptyTrainingCorpus,

    /// The `tokenizers` crate's BPE trainer returned an error.
    #[error("BPE tokenizer training failed: {message}")]
    #[diagnostic(
        code(grug_train::tokenizer_training_failed),
        help(
            "This is an error from the `tokenizers` crate's `Tokenizer::train` \
             call. Common causes: a corpus so small/repetitive that the \
             byte-level pre-tokenizer produces no learnable merges, or an \
             internal panic caught as an error. Try a larger/more varied \
             corpus, or a smaller requested `vocab_size`."
        )
    )]
    TokenizerTrainingFailed { message: String },

    /// Loading a previously-saved BPE tokenizer file failed.
    #[error("failed to load BPE tokenizer from {path}: {message}")]
    #[diagnostic(
        code(grug_train::tokenizer_load_failed),
        help(
            "Check the path exists and was written by `BpeTokenizer::save` \
             (a `tokenizers` JSON file) -- not a byte-tokenizer artifact or a \
             checkpoint file, which use different formats. If resuming a \
             run, confirm the `--artifacts-dir` matches the original run's."
        )
    )]
    TokenizerLoadFailed { path: String, message: String },

    /// Saving a trained BPE tokenizer to disk failed.
    #[error("failed to save BPE tokenizer to {path}: {message}")]
    #[diagnostic(
        code(grug_train::tokenizer_save_failed),
        help(
            "Check that the target directory is writable and there is \
             enough disk space. The tokenizer is saved as a single JSON \
             file; a partially-written file from a previous failed run at \
             the same path may also need to be removed first."
        )
    )]
    TokenizerSaveFailed { path: String, message: String },

    /// A trained/loaded BPE tokenizer is missing one of the four required
    /// special tokens (`<bos>`/`<eos>`/`<pad>`/`<sep>`).
    #[error("BPE tokenizer is missing required special token `{token}`")]
    #[diagnostic(
        code(grug_train::tokenizer_missing_special_token),
        help(
            "Every `BpeTokenizer` must be trained via `BpeTokenizer::train` \
             (which always reserves `<bos>`/`<eos>`/`<pad>`/`<sep>`) or \
             loaded from a file that was itself saved from such a tokenizer. \
             This error usually means a tokenizer JSON file from an \
             unrelated/incompatible training run was loaded by mistake."
        )
    )]
    TokenizerMissingSpecialToken { token: String },

    /// Encoding text with a [`crate::bpe::BpeTokenizer`] failed.
    #[error("BPE tokenizer encode failed: {message}")]
    #[diagnostic(
        code(grug_train::tokenizer_encode_failed),
        help(
            "This is an error surfaced from the `tokenizers` crate's \
             `Tokenizer::encode`. Since encoding is byte-level and total, \
             this should be rare -- check for a regression in pre-tokenizer \
             configuration if it appears consistently."
        )
    )]
    TokenizerEncodeFailed { message: String },

    /// Decoding token ids with a [`crate::bpe::BpeTokenizer`] failed.
    #[error("BPE tokenizer decode failed: {message}")]
    #[diagnostic(
        code(grug_train::tokenizer_decode_failed),
        help(
            "This is an error surfaced from the `tokenizers` crate's \
             `Tokenizer::decode`. Check that the token ids being decoded are \
             within this tokenizer's vocabulary range (e.g. not ids produced \
             by a *different* trained tokenizer)."
        )
    )]
    TokenizerDecodeFailed { message: String },

    /// Writing a training checkpoint (model + optimizer state + metadata)
    /// failed.
    #[error("failed to write checkpoint to {path}: {message}")]
    #[diagnostic(
        code(grug_train::checkpoint_write_failed),
        help(
            "Check that the checkpoint directory is writable and there is \
             enough disk space for both the model and optimizer state files \
             (the optimizer state is roughly 2x the model size for Adam's \
             first/second moment buffers). If this is a detached long run, \
             check `df -h` on the artifacts volume."
        )
    )]
    CheckpointWriteFailed { path: String, message: String },

    /// Loading a training checkpoint failed.
    #[error("failed to load checkpoint from {path}: {message}")]
    #[diagnostic(
        code(grug_train::checkpoint_read_failed),
        help(
            "Common causes: (1) the directory does not exist or is missing \
             `model`/`optim`/`meta.json` (a partially-written checkpoint from \
             a killed run -- prefer the second-to-last checkpoint directory \
             in that case); (2) the model/optimizer config passed to \
             `load_checkpoint` does not match the config the checkpoint was \
             saved with (different `d_model`/`n_layers`/vocab size -- shapes \
             must match exactly); (3) a `schema_version` mismatch from a \
             checkpoint written by an older/incompatible build of this crate."
        )
    )]
    CheckpointReadFailed { path: String, message: String },

    /// [`crate::transformer::TinyTransformer::with_serving_length`] was
    /// called on a model using
    /// [`crate::transformer::PositionalEncoding::Learned`].
    #[error(
        "cannot serve max_seq_len {requested} beyond the trained {trained}: this model uses \
         learned absolute position embeddings, which have no way to represent an untrained \
         position"
    )]
    #[diagnostic(
        code(grug_train::learned_positions_cannot_extend),
        help(
            "Learned position embeddings are a fixed [max_seq_len, d_model] table with one \
             row per trained position -- there is no principled value to put in a new row for \
             an untrained position. Retrain (or continue training) this model with \
             `PositionalEncoding::Rope` if you need to serve past its trained context length \
             -- e.g. the `akh_tool_small_rope4k`, `akh_tool_small_8k`, or `akh_tool_350m_8k` \
             presets on `TinyTransformerConfig`; `TinyTransformer::with_serving_length` (and \
             the `--extend-context` knowledge-SFT warm start built on it) only supports Rope \
             models, since RoPE has no position table to resize -- it derives rotation angles \
             from `theta`/`scale` at call time instead."
        )
    )]
    LearnedPositionsCannotExtend { trained: usize, requested: usize },

    /// [`crate::knowledge_train::KnowledgeModelShape::context_extension_scale`]
    /// was asked to context-extend a foundation shape into a target SFT
    /// shape (ADR 104 D3, `train_knowledge --extend-context`) that differs
    /// in more than `max_seq_len`/`positional_encoding.scale`.
    #[error(
        "foundation bundle shape is not a valid context extension of the knowledge SFT target \
         shape: {reason}"
    )]
    #[diagnostic(
        code(grug_train::context_extension_shape_mismatch),
        help(
            "`--extend-context` only tolerates a foundation shape that differs from the SFT \
             target in `max_seq_len` (which must strictly grow) and, under \
             `PositionalEncoding::Rope`, `scale` -- every other field (vocab_size, d_model, \
             n_heads, n_layers, d_ff, rope theta, attention_chunk) must match exactly. Either \
             retrain the foundation at the SFT target's exact non-context architecture, or \
             pick an SFT `--model` whose shape (other than context length) matches this \
             foundation bundle."
        )
    )]
    ContextExtensionShapeMismatch { reason: String },

    /// [`crate::remediation::validate_remediation_share`] was given a
    /// `--remediation-max-share-ppm` outside the valid parts-per-million
    /// range.
    #[error(
        "remediation share ceiling must be in 1..=1_000_000 parts-per-million, got {ceiling_ppm}"
    )]
    #[diagnostic(
        code(grug_train::remediation_invalid_ceiling),
        help(
            "Pass --remediation-max-share-ppm as an integer between 1 and 1_000_000 \
             (parts per million). The default, 50_000, matches the adaptive loop's \
             steering ceiling (5%)."
        )
    )]
    RemediationInvalidCeiling { ceiling_ppm: u64 },

    /// `train_knowledge --remediate-from-bundle` was given a base or
    /// supplemental corpus with zero records.
    #[error(
        "remediation requires non-empty base ({base_records}) and supplemental \
         ({supplemental_records}) corpora"
    )]
    #[diagnostic(
        code(grug_train::remediation_empty_mixture),
        help(
            "Pass --train-jsonl with at least one record and at least one non-empty \
             --supplemental-jsonl file. Remediation continues a candidate's Adam state \
             on a real base-plus-sibling mixture, not on either corpus alone."
        )
    )]
    RemediationEmptyMixture {
        base_records: usize,
        supplemental_records: usize,
    },

    /// The combined base + supplemental record count overflowed while
    /// computing a remediation segment's supplemental share.
    #[error("remediation record count overflow while computing the supplemental share")]
    #[diagnostic(
        code(grug_train::remediation_record_count_overflow),
        help(
            "The combined base + supplemental record count overflowed a 64-bit integer. \
             Split the remediation segment's corpus across multiple smaller runs."
        )
    )]
    RemediationRecordCountOverflow,

    /// A remediation segment's `--supplemental-jsonl` mixture exceeds
    /// `--remediation-max-share-ppm` (default 50_000 ppm / 5%, matching the
    /// adaptive loop's steering ceiling).
    #[error(
        "remediation supplemental share {actual_ppm} ppm ({supplemental_records} supplemental \
         of {base_records} base records) exceeds the configured ceiling {ceiling_ppm} ppm"
    )]
    #[diagnostic(
        code(grug_train::remediation_share_exceeded),
        help(
            "Reduce --supplemental-jsonl relative to --train-jsonl, split the remediation \
             segment across multiple runs, or raise --remediation-max-share-ppm only with \
             explicit operator sign-off -- it mirrors the adaptive loop's steering ceiling \
             and exists so an engine-generated sibling corpus cannot dominate a retraining \
             segment."
        )
    )]
    RemediationShareExceeded {
        actual_ppm: u64,
        ceiling_ppm: u64,
        base_records: usize,
        supplemental_records: usize,
    },
}
