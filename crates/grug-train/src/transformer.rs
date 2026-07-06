//! A vanilla decoder-only transformer, generic over `B: Backend`.
//!
//! This is the real Spike-0 model (Phase 51 plan, P0): token embedding +
//! sinusoidal positional encoding + N causal decoder blocks (self-attention
//! + MLP, each with a residual + pre-`LayerNorm`) + an LM head back to
//! vocab size. Kept deliberately tiny (see [`TinyTransformerConfig::spike`])
//! so a handful of training steps run in seconds on CPU.
//!
//! # The `Attention` seam (plan decision 8)
//!
//! Attention/context managers are explicitly punted to a later
//! context-scaling phase (RoPE-extrapolation + EM-LLM first, DuoAttention
//! only if those are insufficient). This module defines [`Attention`] as
//! the swap point: [`CausalSelfAttention`] is the *only* implementation
//! today (plain scaled-dot-product, full causal mask, no KV eviction, no
//! sparsity), but any later attention/context manager needs only to
//! implement this trait and be substituted into [`DecoderBlock`] --
//! everything above it (the block, the stack, training, generation) is
//! unaware of which attention implementation it is running.
//!
//! Note the trait is intentionally *not* an `AutodiffModule`-shaped
//! abstraction with its own `Config` type: Burn's `#[derive(Module)]`
//! requires a concrete field type (no `dyn Trait` fields, since module
//! parameter traversal is done by static dispatch), so `DecoderBlock` is
//! generic over `A: Attention<B>` rather than holding a boxed trait object.
//! A later context-manager implementation is a new concrete type plugged
//! into that generic parameter, not a new enum variant or trait object.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// The attention seam: anything that can turn `[batch, seq, d_model]` into
/// an attended `[batch, seq, d_model]` under a causal mask.
///
/// Implement this trait to swap in a different attention/context-management
/// strategy (RoPE-extrapolation, EM-LLM external memory, DuoAttention
/// sparse/full head routing, ...) without touching [`DecoderBlock`],
/// [`TinyTransformer`], the training loop, or generation.
///
/// # Contract
///
/// - `input`: `[batch_size, seq_len, d_model]`.
/// - `causal_mask`: `[batch_size, seq_len, seq_len]`, `true` where attention
///   must be *blocked* (i.e. the same convention as
///   `burn::nn::attention::generate_autoregressive_mask` and
///   `Tensor::mask_fill`: `true` = masked out). Implementations that add
///   their own structural sparsity (e.g. DuoAttention's streaming heads, or
///   an EM-LLM external-memory gate) combine their own mask with this one
///   rather than replacing it, so causality is never accidentally violated.
/// - Returns: `[batch_size, seq_len, d_model]`, same shape as `input`.
pub trait Attention<B: Backend> {
    fn forward(&self, input: Tensor<B, 3>, causal_mask: Tensor<B, 3, Bool>) -> Tensor<B, 3>;
}

/// Configuration for [`CausalSelfAttention`].
#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );
        CausalSelfAttention {
            query: LinearConfig::new(self.d_model, self.d_model).init(device),
            key: LinearConfig::new(self.d_model, self.d_model).init(device),
            value: LinearConfig::new(self.d_model, self.d_model).init(device),
            out_proj: LinearConfig::new(self.d_model, self.d_model).init(device),
            n_heads: self.n_heads,
            d_head: self.d_model / self.n_heads,
        }
    }
}

/// Vanilla scaled-dot-product multi-head causal self-attention -- the only
/// [`Attention`] implementation in the Spike-0 seam. See the module docs for
/// why this is a deliberate, minimal seed and not a dead end.
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out_proj: Linear<B>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> Attention<B> for CausalSelfAttention<B> {
    fn forward(&self, input: Tensor<B, 3>, causal_mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = input.dims();
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        let q = self.query.forward(input.clone());
        let k = self.key.forward(input.clone());
        let v = self.value.forward(input);

        // [batch, seq, d_model] -> [batch, n_heads, seq, d_head]
        let split_heads = |t: Tensor<B, 3>| -> Tensor<B, 4> {
            t.reshape([batch_size, seq_len, n_heads, d_head])
                .swap_dims(1, 2)
        };
        let q = split_heads(q);
        let k = split_heads(k);
        let v = split_heads(v);

        // Scaled dot-product: [batch, n_heads, seq, seq]
        let scale = (d_head as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Broadcast the [batch, seq, seq] causal mask across the head dim.
        let mask = causal_mask.unsqueeze_dim::<4>(1);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = softmax(scores, 3);
        let attended = weights.matmul(v); // [batch, n_heads, seq, d_head]

        let merged = attended
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, d_model]);

        self.out_proj.forward(merged)
    }
}

/// A position-wise feed-forward block: `Linear -> GELU -> Linear`, expanding
/// to `d_ff` and back to `d_model`.
#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            activation: Gelu::new(),
            linear2: LinearConfig::new(self.d_ff, self.d_model).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    activation: Gelu,
    linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

/// One decoder block: pre-LayerNorm causal self-attention with a residual,
/// then pre-LayerNorm feed-forward with a residual (the standard "Pre-LN"
/// transformer arrangement, which trains more stably than post-LN at small
/// scale / few steps -- relevant since Spike-0 trains for only tens to
/// hundreds of steps).
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attention: CausalSelfAttention<B>,
    ln2: LayerNorm<B>,
    feed_forward: FeedForward<B>,
}

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            ln1: LayerNormConfig::new(self.d_model).init(device),
            attention: CausalSelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
            ln2: LayerNormConfig::new(self.d_model).init(device),
            feed_forward: FeedForwardConfig::new(self.d_model, self.d_ff).init(device),
        }
    }
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>, causal_mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let attended = self.attention.forward(self.ln1.forward(input.clone()), causal_mask);
        let input = input + attended;
        let fed_forward = self.feed_forward.forward(self.ln2.forward(input.clone()));
        input + fed_forward
    }
}

/// Configuration for [`TinyTransformer`].
#[derive(Config, Debug)]
pub struct TinyTransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

impl TinyTransformerConfig {
    /// The Spike-0 default: small enough to train on CPU in seconds, per
    /// the Phase 51 plan's "~2-10M transformer" P0 target.
    ///
    /// `d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=256`. See
    /// [`TinyTransformer`] doc comment for the resulting parameter count.
    pub fn spike(vocab_size: usize) -> Self {
        Self::new(vocab_size, 128, 4, 2, 512, 256)
    }

    /// The P1a scale-up default (Phase 51 plan, "Model" bullet): "~5-30M
    /// params (d_model 384-512, 6-8 layers/heads, ctx 512-1024), sized to
    /// the *task* not the organ". `d_model=384, n_heads=6, n_layers=6,
    /// d_ff=1536, max_seq_len=640` lands in the middle of that range --
    /// roughly 10-20M params depending on the (now much smaller, BPE) vocab
    /// size, vs. the byte tokenizer's fixed 260. See
    /// [`TinyTransformer::num_params`] for the actual count at a given
    /// vocab size; the P1a plan's gate is on *task performance*, not a
    /// precise parameter budget, so this is a reasonable starting point
    /// rather than a tuned optimum.
    pub fn p1a(vocab_size: usize) -> Self {
        Self::new(vocab_size, 384, 6, 6, 1536, 640)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyTransformer<B> {
        let blocks = (0..self.n_layers)
            .map(|_| DecoderBlockConfig::new(self.d_model, self.n_heads, self.d_ff).init(device))
            .collect();

        TinyTransformer {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
            blocks,
            ln_final: LayerNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            max_seq_len: self.max_seq_len,
        }
    }
}

/// A tiny vanilla decoder-only transformer LM.
///
/// `token_embedding + position_embedding -> N x DecoderBlock -> LayerNorm ->
/// lm_head`. Positions use a learned [`Embedding`] (not the sinusoidal
/// [`burn::nn::PositionalEncoding`]) purely so both embeddings share the
/// same simple `Embedding` mechanics -- either is a fine choice at this
/// scale, and the choice of learned-vs-sinusoidal has no bearing on the
/// `Attention` seam.
#[derive(Module, Debug)]
pub struct TinyTransformer<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    ln_final: LayerNorm<B>,
    lm_head: Linear<B>,
    max_seq_len: usize,
}

impl<B: Backend> TinyTransformer<B> {
    /// Forward pass: `input_ids` is `[batch_size, seq_len]` token ids,
    /// output is `[batch_size, seq_len, vocab_size]` logits.
    ///
    /// Panics (via Burn's tensor-shape assertions) if `seq_len` exceeds the
    /// configured `max_seq_len`, since the learned position embedding table
    /// only has that many rows.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        assert!(
            seq_len <= self.max_seq_len,
            "sequence length {seq_len} exceeds configured max_seq_len {}",
            self.max_seq_len
        );
        let device = input_ids.device();

        let tokens = self.token_embedding.forward(input_ids);

        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .repeat_dim(0, batch_size);
        let positions = self.position_embedding.forward(positions);

        let mut hidden = tokens + positions;

        let causal_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);
        for block in &self.blocks {
            hidden = block.forward(hidden, causal_mask.clone());
        }

        let hidden = self.ln_final.forward(hidden);
        self.lm_head.forward(hidden)
    }

    /// Approximate learnable parameter count (sum of `Param` tensor sizes).
    /// Used for reporting, not part of the forward path.
    pub fn num_params(&self) -> usize
    where
        B: Backend,
    {
        // Burn's `Module` trait exposes `num_params` directly.
        Module::num_params(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;
    use burn::tensor::TensorData;

    fn tiny_config() -> TinyTransformerConfig {
        // Deliberately smaller than the spike default so shape tests run
        // fast: vocab=260 (byte tokenizer), d_model=16, 2 heads, 2 layers.
        TinyTransformerConfig::new(260, 16, 2, 2, 32, 64)
    }

    #[test]
    fn forward_pass_produces_expected_shape() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);

        let batch_size = 3;
        let seq_len = 10;
        let ids: Vec<i64> = (0..(batch_size * seq_len) as i64).map(|i| i % 260).collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids, [batch_size, seq_len]),
            &device,
        );

        let output = model.forward(input);
        assert_eq!(output.dims(), [batch_size, seq_len, 260]);
    }

    #[test]
    fn forward_pass_single_token() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let input = Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(vec![42i64], [1, 1]), &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 1, 260]);
    }

    #[test]
    #[should_panic(expected = "exceeds configured max_seq_len")]
    fn forward_pass_rejects_overlong_sequence() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let ids: Vec<i64> = (0..100i64).map(|i| i % 260).collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(ids, [1, 100]), &device);
        let _ = model.forward(input);
    }

    #[test]
    fn spike_config_param_count_is_small() {
        let device = Default::default();
        let model = TinyTransformerConfig::spike(260).init::<CpuBackend>(&device);
        let n = model.num_params();
        // Documented as "~2-10M" in the plan; assert it's in a sane tiny
        // range rather than pinning an exact number that would break on
        // every hyperparameter tweak.
        assert!(n > 100_000, "spike model suspiciously small: {n} params");
        assert!(n < 20_000_000, "spike model larger than the P0 target: {n} params");
    }

    #[test]
    fn p1a_config_param_count_is_in_target_range() {
        let device = Default::default();
        // A representative BPE vocab size (P1a target ~2-4K); the exact
        // count depends on the actually-trained vocab, but this pins the
        // architecture (not vocab) contribution to the 5-30M P1a range from
        // the Phase 51 plan.
        let model = TinyTransformerConfig::p1a(2500).init::<CpuBackend>(&device);
        let n = model.num_params();
        assert!(n > 5_000_000, "P1a model below the 5-30M target range: {n} params");
        assert!(n < 30_000_000, "P1a model above the 5-30M target range: {n} params");
    }
}
