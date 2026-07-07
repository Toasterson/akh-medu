//! Greedy autoregressive generation from a trained [`TinyTransformer`].
//!
//! Runs on a plain (non-autodiff) backend -- callers pass the result of
//! `model.valid()` (see [`crate::train::to_eval_model`]) so generation does
//! not build a computation graph it will never call `.backward()` on.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use grug_data::constrain::{GrammarState, ToolCallGrammar};

use crate::tokenizer::Vocab;
use crate::transformer::TinyTransformer;

/// Generation options.
#[derive(Debug, Clone, Copy)]
pub struct GenerateConfig {
    /// Hard stop on total generated tokens (beyond the prompt), even if
    /// `EOS` is never produced -- guards against runaway generation from an
    /// undertrained model.
    pub max_new_tokens: usize,
    /// Must match (or be <=) the model's configured `max_seq_len`; sequences
    /// longer than this are truncated from the left (oldest tokens dropped)
    /// before each forward pass.
    pub max_seq_len: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            max_seq_len: 256,
        }
    }
}

/// Greedily generate a tool-call string from `prompt`, using the same
/// `BOS <prompt> SEP` framing [`crate::train::encode_example`] trains on.
///
/// Stops at `EOS`, at `max_new_tokens`, or (defensively) if the running
/// sequence would exceed `max_seq_len` even after left-truncation of prompt
/// tokens -- returns whatever was generated so far in every case. Decoding
/// goes through `tokenizer.decode`, which strips special tokens; if decoding
/// fails outright (e.g. [`crate::tokenizer::ByteTokenizer`] hitting a
/// mid-codepoint stop), the raw generated ids are decoded lossily as UTF-8
/// bytes instead of erroring, since the eval harness needs a string back
/// either way -- an invalid-JSON string still scores as "invalid json",
/// which is a real and useful eval outcome.
pub fn generate<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let mut ids: Vec<u32> = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt).unwrap_or_default());
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let logits = model.forward(input); // [1, seq_len, vocab]
        let vocab_size = logits.dims()[2];
        let last_logits = logits
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        let next_id = argmax(last_logits);

        if next_id == tokenizer.eos_id() {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        // Fallback for tokenizers (e.g. the byte tokenizer) whose `decode`
        // can fail on a truncated multi-byte codepoint -- lossily decode the
        // byte-range tokens directly rather than surfacing an error from a
        // generation call, per this function's documented contract.
        let bytes: Vec<u8> = generated.iter().copied().filter(|&t| t < 256).map(|t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Schema-constrained greedy generation (Phase 51 P1a-4 Step A).
///
/// Identical framing to [`generate`], but at every step the next-token choice
/// is restricted to tokens whose byte surface keeps the running output on a
/// path to a call accepted by `grammar` (a byte-level acceptor over the
/// offered tools' schemas — see [`grug_data::constrain`]). Disallowed tokens'
/// logits are masked to `-inf` before the argmax, so the model still picks its
/// highest-scoring *legal* continuation. `EOS` is offered only once a complete
/// call has been formed (`grammar.accepts_eos`), which prevents the model
/// stopping mid-call.
///
/// `pieces` is the tokenizer's per-id byte table ([`Vocab::token_pieces`]),
/// built once by the caller and shared across examples. Tokens with no
/// appendable bytes (the special tokens) are never offered as a continuation.
///
/// This constrains *structure and parameter names*, not values (see the module
/// grammar docs): it can force valid, schema-consistent keys but cannot supply
/// correct argument *values*, which must be copied from the request — exactly
/// the isolation Step A measures.
pub fn generate_constrained<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    grammar: &ToolCallGrammar,
    pieces: &[Option<Vec<u8>>],
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let eos_id = tokenizer.eos_id();

    let mut ids: Vec<u32> = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt).unwrap_or_default());
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();
    let mut frontier: GrammarState = grammar.start();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input = Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let logits = model.forward(input); // [1, seq_len, vocab]
        let vocab_size = logits.dims()[2];
        let last_logits = logits
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        // Build an additive mask: 0.0 for tokens the grammar allows as the
        // next continuation, -inf otherwise. Precompute the per-byte
        // legality of the current frontier so most tokens reject on their
        // first byte without cloning grammar state.
        let mut mask = vec![f32::NEG_INFINITY; vocab_size];
        let mut any_allowed = false;
        for (id, piece) in pieces.iter().enumerate().take(vocab_size) {
            if let Some(bytes) = piece {
                if !bytes.is_empty() && grammar.advance(&frontier, bytes).is_some() {
                    mask[id] = 0.0;
                    any_allowed = true;
                }
            }
        }
        // Offer EOS only when a complete call has been formed.
        let eos_ok = grammar.accepts_eos(&frontier);
        if eos_ok && (eos_id as usize) < vocab_size {
            mask[eos_id as usize] = 0.0;
            any_allowed = true;
        }
        if !any_allowed {
            // Dead end (should not happen: every non-`End` state has a legal
            // continuation, and `End` offers EOS) — stop rather than emit an
            // out-of-grammar token.
            break;
        }

        let mask_tensor = Tensor::<B, 1>::from_data(TensorData::new(mask, [vocab_size]), device);
        let masked = last_logits + mask_tensor;
        let next_id = argmax(masked);

        if next_id == eos_id {
            break;
        }

        // Commit: advance the frontier by the chosen token's bytes. It was in
        // the allowed set, so `advance` is `Some`; fall back defensively.
        if let Some(bytes) = pieces.get(next_id as usize).and_then(|p| p.as_ref()) {
            match grammar.advance(&frontier, bytes) {
                Some(next_state) => frontier = next_state,
                None => break,
            }
        } else {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated.iter().copied().filter(|&t| t < 256).map(|t| t as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Argmax over a 1-D tensor, returned as `u32` (token id).
///
/// Uses `into_scalar().elem()` (the same backend-agnostic conversion path
/// [`crate::train::train_step`] uses for the loss scalar) rather than a
/// hardcoded `as_slice::<i64>()`: Burn's `Int` element type is
/// backend-defined (`burn-ndarray`'s CPU backend uses `i64`, but CubeCL
/// backends like `Rocm`/`Wgpu` default their `Int` type to `i32`), so
/// assuming `i64` panics with a `TypeMismatch` on GPU backends. `.elem::<i64>()`
/// converts whatever the backend's actual element type is into `i64` first,
/// so this works for any backend's `Int` representation.
fn argmax<B: Backend>(logits: Tensor<B, 1>) -> u32 {
    let idx: Tensor<B, 1, Int> = logits.argmax(0);
    let value: i64 = idx.into_scalar().elem();
    value as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteTokenizer;
    use crate::transformer::TinyTransformerConfig;
    use crate::CpuBackend;

    #[test]
    fn generate_stops_at_max_new_tokens_on_untrained_model() {
        let device = Default::default();
        let model_config = TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tok = ByteTokenizer::new();

        let config = GenerateConfig {
            max_new_tokens: 5,
            max_seq_len: 64,
        };
        let out = generate(&model, &tok, "hello", config, &device);
        // No correctness claim on an untrained model -- just that it
        // terminates without panicking. Note the bound is on *tokens*
        // generated (<= max_new_tokens = 5), not on the resulting string's
        // byte length: a lossily-decoded replacement character (U+FFFD, an
        // expected outcome from an untrained model emitting arbitrary byte
        // tokens) is 3 UTF-8 bytes per single generated token.
        assert!(
            out.chars().count() <= 5 * 3,
            "expected at most 5 generated tokens' worth of (possibly replacement-char-inflated) output, got {out:?}"
        );
    }
}
