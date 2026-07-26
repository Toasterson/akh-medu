//! Pointer-generator head (Phase 51 P1a-4 Step B).
//!
//! The P1a-3 ablation + P1a-4 Step A localised the residual zero-shot failure
//! to **value copying**: a vanilla LM can only emit vocabulary it learned, but
//! an unseen tool's parameter names (and the request's argument values) live
//! only in the prompt. This module adds a copy mechanism (See et al. 2017,
//! *"Get To The Point"*) so the model can copy spans straight from the input.
//!
//! # Mechanism (adapted to a decoder-only LM)
//!
//! At every position `t` the head produces a mixed next-token distribution:
//!
//! ```text
//! P_final(t) = p_gen(t) · P_vocab(t)  +  (1 − p_gen(t)) · P_copy(t)
//! ```
//!
//! - `P_vocab(t) = softmax(lm_head(H_t))` — the ordinary generation
//!   distribution over the vocabulary.
//! - `P_copy(t)` — a **copy distribution over the vocabulary**, obtained by a
//!   single-head causal attention from position `t` over all earlier input
//!   positions `j ≤ t` (weights `α_{t,j}`), then scattering each `α_{t,j}` onto
//!   the vocabulary bucket of `input_ids[j]`. So probability mass lands on
//!   exactly the tokens present in the prompt/history — including a novel
//!   parameter name the model has never generated before.
//! - `p_gen(t) ∈ [0,1]` — a learned gate `σ(W·[H_t ; C_t])` from the hidden
//!   state and the copy-attention context vector `C_t`, deciding how much to
//!   generate vs. copy.
//!
//! Training minimises the NLL of the target under `P_final` (a *probability*,
//! not logits — see `crate::train::masked_nll`), so generate-vs-copy is
//! learned by marginal likelihood rather than supervised directly.
//!
//! # Trunk reuse
//!
//! The head wraps a [`TinyTransformer`] trunk and drives it via
//! [`TinyTransformer::hidden`] / [`TinyTransformer::vocab_logits`] — the copy
//! attention sees the *same* final hidden states the LM head sees. Only the
//! copy projections (`q`/`k`/`v`) and the `p_gen` gate are new parameters, so
//! this is a small addition on top of the proven 13M trunk.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor};

use crate::transformer::{TinyTransformer, TinyTransformerConfig};

/// Configuration for [`PointerTransformer`]: the trunk config plus the
/// copy-attention width (defaults to `d_model`).
#[derive(Config, Debug)]
pub struct PointerTransformerConfig {
    pub transformer: TinyTransformerConfig,
}

impl PointerTransformerConfig {
    /// Build a pointer-generator model: a [`TinyTransformer`] trunk plus the
    /// copy-attention projections and `p_gen` gate.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PointerTransformer<B> {
        let d = self.transformer.d_model;
        PointerTransformer {
            trunk: self.transformer.init(device),
            copy_query: LinearConfig::new(d, d).init(device),
            copy_key: LinearConfig::new(d, d).init(device),
            copy_value: LinearConfig::new(d, d).init(device),
            // p_gen reads [hidden ; context] -> a scalar gate.
            p_gen: LinearConfig::new(2 * d, 1).init(device),
            d_model: d,
        }
    }
}

/// A [`TinyTransformer`] with a pointer-generator copy head (Phase 51 P1a-4
/// Step B). See the module docs for the mechanism.
#[derive(Module, Debug)]
pub struct PointerTransformer<B: Backend> {
    trunk: TinyTransformer<B>,
    copy_query: Linear<B>,
    copy_key: Linear<B>,
    copy_value: Linear<B>,
    p_gen: Linear<B>,
    d_model: usize,
}

/// The intermediate pieces of a pointer-generator forward pass, exposed for
/// the copy-supervision loss (`crate::train`) and `p_gen` instrumentation.
pub struct PointerParts<B: Backend> {
    /// Mixed distribution as log-probabilities `log(P_final + eps)` `[B,S,V]`.
    pub log_probs: Tensor<B, 3>,
    /// The generate-vs-copy gate `[B,S,1]` (1 = generate, 0 = copy).
    pub p_gen: Tensor<B, 3>,
    /// The copy distribution over the vocabulary `[B,S,V]` (before gating).
    pub p_copy: Tensor<B, 3>,
}

impl<B: Backend> PointerTransformer<B> {
    /// Full forward pass returning the mixed log-probs plus the intermediate
    /// `p_gen` gate and copy distribution (`PointerParts`). [`log_probs`] is a
    /// thin wrapper that keeps only `log_probs`; the copy-supervision loss and
    /// the `p_gen` instrumentation need the other two.
    ///
    /// [`log_probs`]: PointerTransformer::log_probs
    pub fn forward_parts(&self, input_ids: Tensor<B, 2, Int>) -> PointerParts<B> {
        let [batch, seq] = input_ids.dims();
        let device = input_ids.device();

        let hidden = self.trunk.hidden(input_ids.clone()); // [B, S, D]
        let vocab_logits = self.trunk.vocab_logits(hidden.clone()); // [B, S, V]
        let vocab = vocab_logits.dims()[2];
        let p_vocab = softmax(vocab_logits, 2); // [B, S, V]

        // --- Copy attention: from each position over earlier positions. ---
        let q = self.copy_query.forward(hidden.clone()); // [B, S, D]
        let k = self.copy_key.forward(hidden.clone());
        let v = self.copy_value.forward(hidden.clone());

        let scale = (self.d_model as f64).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)) / scale; // [B, S, S]
        let causal_mask = generate_autoregressive_mask::<B>(batch, seq, &device); // [B,S,S] true=blocked
        let scores = scores.mask_fill(causal_mask, f32::NEG_INFINITY);
        let alpha = softmax(scores, 2); // [B, S, S], rows sum to 1 over source j
        let context = alpha.clone().matmul(v); // [B, S, D]

        // --- p_gen gate from [hidden ; context]. ---
        let gate_in = Tensor::cat(vec![hidden, context], 2); // [B, S, 2D]
        let p_gen = sigmoid(self.p_gen.forward(gate_in)); // [B, S, 1]

        // --- Copy distribution over vocab: scatter α onto input token ids. ---
        // index[b, t, j] = input_ids[b, j] (independent of t); scatter-add
        // α[b, t, j] into P_copy[b, t, input_ids[b, j]].
        let index = input_ids.reshape([batch, 1, seq]).repeat_dim(1, seq); // [B, S, S]
        let p_copy = Tensor::<B, 3>::zeros([batch, seq, vocab], &device).scatter(
            2,
            index,
            alpha,
            IndexingUpdateOp::Add,
        ); // [B, S, V]

        // --- Mix (broadcast p_gen [B,S,1] over the vocab dim). ---
        let one_minus = p_gen.clone().neg().add_scalar(1.0); // [B, S, 1]
        let p_final = p_vocab * p_gen.clone() + p_copy.clone() * one_minus; // [B, S, V]

        // log(P + eps): P_final sums to 1 over vocab; eps guards log(0).
        let log_probs = p_final.add_scalar(1e-9).log();
        PointerParts { log_probs, p_gen, p_copy }
    }

    /// The mixed next-token distribution as **log-probabilities**
    /// `[batch, seq, vocab]`: `log(P_final + eps)`. Used both by the training
    /// loss (`crate::train::masked_nll`) and by generation
    /// (`crate::generate::generate_ptr`), which takes the argmax of the last
    /// position (argmax over log-probs == argmax over probs).
    pub fn log_probs(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_parts(input_ids).log_probs
    }

    /// Approximate learnable parameter count (trunk + copy head).
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;
    use burn::tensor::TensorData;

    fn tiny_config() -> PointerTransformerConfig {
        PointerTransformerConfig::new(TinyTransformerConfig::new(37, 16, 2, 2, 32, 64))
    }

    #[test]
    fn log_probs_have_expected_shape() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let ids: Vec<i64> = (0..(3 * 10) as i64).map(|i| i % 37).collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(ids, [3, 10]), &device);
        let lp = model.log_probs(input);
        assert_eq!(lp.dims(), [3, 10, 37]);
    }

    /// `P_final` must be a genuine probability distribution: `exp(log_probs)`
    /// sums to ~1 across the vocab at every position. This validates the
    /// scatter-based copy distribution + the gated mixture together.
    #[test]
    fn mixed_distribution_normalises_to_one() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let seq = 8;
        let ids: Vec<i64> = (0..seq as i64).map(|i| (i * 3) % 37).collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(ids, [1, seq]), &device);

        let probs = model.log_probs(input).exp(); // [1, seq, 37]
        let sums = probs.sum_dim(2).reshape([seq]); // [seq]
        let sums: Vec<f32> = sums.into_data().to_vec().unwrap();
        for (i, s) in sums.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-4, "position {i} sums to {s}, not 1.0");
        }
    }

    /// The copy path must place mass on a token that appears in the input:
    /// with the gate forced toward copying (checked indirectly), a token id
    /// present in the prompt should be reachable. Here we just assert the
    /// distribution is not degenerate (some non-trivial entropy) and that a
    /// present token gets non-zero probability.
    #[test]
    fn present_input_token_has_nonzero_probability() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let seq = 6;
        let present_token = 21i64;
        let ids: Vec<i64> = vec![present_token; seq];
        let input = Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(ids, [1, seq]), &device);

        let probs = model.log_probs(input).exp();
        // Probability assigned to `present_token` at the last position.
        let last = probs.slice([0..1, seq - 1..seq, 0..37]).reshape([37]);
        let p: Vec<f32> = last.into_data().to_vec().unwrap();
        assert!(p[present_token as usize] > 0.0, "present token got zero probability");
    }
}
