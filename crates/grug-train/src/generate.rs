//! Greedy autoregressive generation from a trained [`TinyTransformer`].
//!
//! Runs on a plain (non-autodiff) backend -- callers pass the result of
//! `model.valid()` (see [`crate::train::to_eval_model`]) so generation does
//! not build a computation graph it will never call `.backward()` on.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

use crate::tokenizer::{ByteTokenizer, BOS, EOS, SEP};
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
/// tokens -- returns whatever was generated so far in every case, decoded
/// lossily (`String::from_utf8_lossy`) since a partially-generated sequence
/// is exactly the case where a stray split multi-byte codepoint is
/// expected, and the eval harness needs a string back either way (an
/// invalid-JSON string still scores as "invalid json", which is a real and
/// useful eval outcome -- an error here would just be a worse way to convey
/// the same information).
pub fn generate<B: Backend>(
    model: &TinyTransformer<B>,
    tokenizer: &ByteTokenizer,
    prompt: &str,
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let mut ids: Vec<u32> = Vec::new();
    ids.push(BOS);
    ids.extend(tokenizer.encode(prompt));
    ids.push(SEP);

    let target_start = ids.len();
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

        if next_id == EOS {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    let _ = target_start; // kept for readability/symmetry with `encode_example`; not needed to slice `generated`.
    String::from_utf8_lossy(&tokenizer_bytes(&generated)).into_owned()
}

/// Extract only the byte-range tokens from `ids` as raw bytes (drops any
/// special tokens that might appear, though generation should only ever
/// append byte tokens or `EOS`, and `EOS` is never pushed into `generated`).
fn tokenizer_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().copied().filter(|&t| t < 256).map(|t| t as u8).collect()
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
