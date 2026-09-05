//! A vanilla decoder-only transformer, generic over `B: Backend`.
//!
//! This is the real Spike-0 model (Phase 51 plan, P0): token embedding +
//! sinusoidal positional encoding + N causal decoder blocks (self-attention
//! + MLP, each with a residual + pre-`LayerNorm`) + an LM head back to
//!   vocab size. Kept deliberately tiny (see [`TinyTransformerConfig::spike`])
//!   so a handful of training steps run in seconds on CPU.
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
use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
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
            self.d_model.is_multiple_of(self.n_heads),
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

/// Per-layer projected keys and values retained during autoregressive
/// decoding. This is runtime state only: it is not a module parameter and
/// therefore does not change checkpoint compatibility.
#[derive(Debug)]
pub struct CausalSelfAttentionCache<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
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

impl<B: Backend> CausalSelfAttention<B> {
    /// Attention for a prompt prefill or one incremental token. Prefill
    /// supplies the ordinary causal mask and no cache; an incremental call
    /// supplies one query token plus all prior projected keys and values.
    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        causal_mask: Option<Tensor<B, 3, Bool>>,
        cache: Option<CausalSelfAttentionCache<B>>,
    ) -> (Tensor<B, 3>, CausalSelfAttentionCache<B>) {
        let [batch_size, query_len, d_model] = input.dims();
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        let q = self.query.forward(input.clone());
        let new_key = self.key.forward(input.clone());
        let new_value = self.value.forward(input);
        let split_heads = |tensor: Tensor<B, 3>| -> Tensor<B, 4> {
            tensor
                .reshape([batch_size, query_len, n_heads, d_head])
                .swap_dims(1, 2)
        };
        let q = split_heads(q);
        let new_key = split_heads(new_key);
        let new_value = split_heads(new_value);

        let (key, value) = match cache {
            Some(cache) => {
                let [cached_batch, cached_heads, _, cached_width] = cache.key.dims();
                assert_eq!(cached_batch, batch_size, "attention cache batch changed");
                assert_eq!(cached_heads, n_heads, "attention cache head count changed");
                assert_eq!(cached_width, d_head, "attention cache head width changed");
                assert_eq!(
                    cache.value.dims(),
                    cache.key.dims(),
                    "attention key/value cache shapes differ"
                );
                assert_eq!(
                    query_len, 1,
                    "incremental attention accepts exactly one new token"
                );
                (
                    Tensor::cat(vec![cache.key, new_key], 2),
                    Tensor::cat(vec![cache.value, new_value], 2),
                )
            }
            None => (new_key, new_value),
        };
        let total_len = key.dims()[2];
        let scale = (d_head as f64).sqrt();
        let scores = q.matmul(key.clone().swap_dims(2, 3)) / scale;
        let scores = match causal_mask {
            Some(mask) => {
                assert_eq!(
                    mask.dims(),
                    [batch_size, query_len, total_len],
                    "cached attention mask shape differs from query/key lengths"
                );
                scores.mask_fill(mask.unsqueeze_dim::<4>(1), f32::NEG_INFINITY)
            }
            None => {
                assert_eq!(
                    query_len, 1,
                    "multi-token cached attention requires a causal mask"
                );
                scores
            }
        };
        let attended = softmax(scores, 3).matmul(value.clone());
        let merged = attended
            .swap_dims(1, 2)
            .reshape([batch_size, query_len, d_model]);
        (
            self.out_proj.forward(merged),
            CausalSelfAttentionCache { key, value },
        )
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

#[derive(Debug)]
pub struct DecoderBlockCache<B: Backend> {
    attention: CausalSelfAttentionCache<B>,
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
        let attended = self
            .attention
            .forward(self.ln1.forward(input.clone()), causal_mask);
        let input = input + attended;
        let fed_forward = self.feed_forward.forward(self.ln2.forward(input.clone()));
        input + fed_forward
    }

    fn forward_cached(
        &self,
        input: Tensor<B, 3>,
        causal_mask: Option<Tensor<B, 3, Bool>>,
        cache: Option<DecoderBlockCache<B>>,
    ) -> (Tensor<B, 3>, DecoderBlockCache<B>) {
        let (attended, attention) = self.attention.forward_cached(
            self.ln1.forward(input.clone()),
            causal_mask,
            cache.map(|cache| cache.attention),
        );
        let input = input + attended;
        let fed_forward = self.feed_forward.forward(self.ln2.forward(input.clone()));
        (input + fed_forward, DecoderBlockCache { attention })
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

    /// Phase 51 P1a-3 capacity test: ~30M params (d_model 512, 8 layers, 8 heads,
    /// d_ff 2048), ~2.3x the p1a 13M, to test whether the zero-shot / arg-correct
    /// ceilings are capacity-bound rather than data-bound.
    pub fn p1a_large(vocab_size: usize) -> Self {
        Self::new(vocab_size, 512, 8, 8, 2048, 640)
    }

    /// Phase 51 P4 capacity-ladder rung ~50M (d_model 640, 10 layers, 10 heads,
    /// d_ff 2560) — the middle rung of the 11M→50M→100M ladder that reads the
    /// `chain`-component *slope* (Q5): a positive slope says "scale it", flat
    /// says "broken corpus, not capacity".
    pub fn p1a_50m(vocab_size: usize) -> Self {
        Self::new(vocab_size, 640, 10, 10, 2560, 640)
    }

    /// Phase 51 P4 capacity-ladder rung ~100M (d_model 768, 13 layers, 12
    /// heads, d_ff 3072) — the top rung of the ladder.
    pub fn p1a_100m(vocab_size: usize) -> Self {
        Self::new(vocab_size, 768, 13, 12, 3072, 640)
    }

    /// **The Phase 51 P4 target: the 350M foundation model** — 347.3M params
    /// (d_model 1024, 24 layers, 16 heads, d_ff 4864, ctx 1024), verified by
    /// `Module::num_params` at vocab 3000 — the scale at which the
    /// kludge-brain bet is meant to hold (cf. phi-1-small@350M, the one
    /// primary datapoint at this size; see the Phi memo).
    ///
    /// Context defaults to 1024 (not the ladder's 640): the deductive corpus's
    /// depth-6 sequences reach ~800 tokens under the saturated BPE, and 640
    /// silently truncates them.
    ///
    /// **Do not train this with the 11M recipe.** `lr 3e-4` converged 11M but
    /// left 50M undertrained with depressed trace-validity — 350M needs a
    /// lower LR (~1e-4) plus warmup. See `training/configs/350m.env`.
    pub fn p1a_350m(vocab_size: usize) -> Self {
        Self::new(vocab_size, 1024, 16, 24, 4864, 1024)
    }

    /// Fast iteration model for Akh tool-use and knowledge-workflow curricula.
    /// At a 3k BPE vocabulary this is roughly 30M parameters. The 4096 context
    /// matches the measured v2 knowledge-workflow envelope (longest episode
    /// ~3.4k tokens): at 2048 the long note-revision families were compacted
    /// during training, which corrupted their supervision and produced 100%
    /// catastrophic sentinel failures (adaptive-loop-v2, 2026-09-05).
    pub fn akh_tool_small(vocab_size: usize) -> Self {
        Self::new(vocab_size, 512, 8, 8, 2048, 4096)
    }

    /// Full BabyLLM tool-capable rung: the existing ~350M capacity with a
    /// 4096-token context so textbook excerpts, exact nested tool schemas, and
    /// multi-round memory traces fit without discarding supervision.
    pub fn akh_tool_350m(vocab_size: usize) -> Self {
        Self::new(vocab_size, 1024, 16, 24, 4864, 4096)
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

/// Runtime-only decoder state. `next_position` is the absolute learned
/// position assigned to the next token; callers must pre-compact the prompt
/// so the full completion fits instead of silently rebasing cached states.
#[derive(Debug)]
pub struct TinyTransformerCache<B: Backend> {
    layers: Vec<DecoderBlockCache<B>>,
    next_position: usize,
    batch_size: usize,
}

impl<B: Backend> TinyTransformer<B> {
    /// Forward pass: `input_ids` is `[batch_size, seq_len]` token ids,
    /// output is `[batch_size, seq_len, vocab_size]` logits.
    ///
    /// Panics (via Burn's tensor-shape assertions) if `seq_len` exceeds the
    /// configured `max_seq_len`, since the learned position embedding table
    /// only has that many rows.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.vocab_logits(self.hidden(input_ids))
    }

    /// Encode a complete prompt once and retain each layer's projected keys
    /// and values for incremental generation.
    pub(crate) fn prefill_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, TinyTransformerCache<B>) {
        let [batch_size, seq_len] = input_ids.dims();
        assert!(seq_len > 0, "cached prefill requires at least one token");
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
        let mut hidden = tokens + self.position_embedding.forward(positions);
        let causal_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, cache) = block.forward_cached(hidden, Some(causal_mask.clone()), None);
            hidden = next;
            layers.push(cache);
        }
        let logits = self.vocab_logits(self.ln_final.forward(hidden));
        (
            logits,
            TinyTransformerCache {
                layers,
                next_position: seq_len,
                batch_size,
            },
        )
    }

    /// Encode a continuous prefix and the complete textual prompt once, then
    /// retain every layer's projected keys and values for incremental decode.
    ///
    /// This is the cached counterpart to [`Self::forward_with_prefix_embeddings`].
    /// Prefix positions occupy `0..prefix_len`; textual positions follow them,
    /// preserving the exact learned-position ABI used by the uncached path.
    pub(crate) fn prefill_cached_with_prefix_embeddings(
        &self,
        prefix: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, TinyTransformerCache<B>) {
        let [prefix_batch, prefix_len, prefix_width] = prefix.dims();
        let [text_batch, text_len] = input_ids.dims();
        assert!(
            prefix_len > 0 || text_len > 0,
            "cached prefix prefill requires at least one position"
        );
        assert_eq!(
            prefix_batch, text_batch,
            "prefix batch {prefix_batch} differs from text batch {text_batch}"
        );
        let tokens = self.token_embedding.forward(input_ids);
        let token_width = tokens.dims()[2];
        assert_eq!(
            prefix_width, token_width,
            "prefix width {prefix_width} differs from model width {token_width}"
        );
        let seq_len = prefix_len + text_len;
        assert!(
            seq_len <= self.max_seq_len,
            "combined prefix+text sequence length {seq_len} exceeds configured max_seq_len {}",
            self.max_seq_len
        );
        let device = prefix.device();
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .repeat_dim(0, text_batch);
        let mut hidden =
            Tensor::cat(vec![prefix, tokens], 1) + self.position_embedding.forward(positions);
        let causal_mask = generate_autoregressive_mask::<B>(text_batch, seq_len, &device);
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, cache) = block.forward_cached(hidden, Some(causal_mask.clone()), None);
            hidden = next;
            layers.push(cache);
        }
        let logits = self.vocab_logits(self.ln_final.forward(hidden));
        (
            logits,
            TinyTransformerCache {
                layers,
                next_position: seq_len,
                batch_size: text_batch,
            },
        )
    }

    /// Decode exactly one token using a cache returned by
    /// [`Self::prefill_cached`] or an earlier incremental call.
    pub(crate) fn decode_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: TinyTransformerCache<B>,
    ) -> (Tensor<B, 3>, TinyTransformerCache<B>) {
        let [batch_size, seq_len] = input_ids.dims();
        assert_eq!(seq_len, 1, "cached decode accepts exactly one token");
        assert_eq!(
            batch_size, cache.batch_size,
            "cached decode batch differs from prefill batch"
        );
        assert_eq!(
            cache.layers.len(),
            self.blocks.len(),
            "decoder cache layer count differs from model"
        );
        assert!(
            cache.next_position < self.max_seq_len,
            "cached decode position {} exceeds configured max_seq_len {}",
            cache.next_position,
            self.max_seq_len
        );
        let device = input_ids.device();
        let tokens = self.token_embedding.forward(input_ids);
        let positions = Tensor::<B, 1, Int>::arange(
            cache.next_position as i64..cache.next_position as i64 + 1,
            &device,
        )
        .unsqueeze::<2>()
        .repeat_dim(0, batch_size);
        let mut hidden = tokens + self.position_embedding.forward(positions);
        let mut layers = Vec::with_capacity(self.blocks.len());
        for (block, prior) in self.blocks.iter().zip(cache.layers) {
            let (next, updated) = block.forward_cached(hidden, None, Some(prior));
            hidden = next;
            layers.push(updated);
        }
        let logits = self.vocab_logits(self.ln_final.forward(hidden));
        (
            logits,
            TinyTransformerCache {
                layers,
                next_position: cache.next_position + 1,
                batch_size,
            },
        )
    }

    /// Forward with continuous prefix embeddings before the textual token
    /// embeddings. This is the modality seam used by the native vision stem:
    /// image patches and text share this decoder stack rather than routing
    /// through an external OCR model.
    ///
    /// The prefix is `[batch, prefix_len, d_model]`; `input_ids` remains
    /// `[batch, text_len]`. Output covers the combined sequence in prefix/text
    /// order. Existing text-only methods are intentionally untouched so old
    /// checkpoints and tests retain their exact path.
    pub fn forward_with_prefix_embeddings(
        &self,
        prefix: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.vocab_logits(self.hidden_with_prefix_embeddings(prefix, input_ids))
    }

    /// Hidden-state counterpart to [`Self::forward_with_prefix_embeddings`].
    pub fn hidden_with_prefix_embeddings(
        &self,
        prefix: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [prefix_batch, prefix_len, prefix_width] = prefix.dims();
        let [text_batch, text_len] = input_ids.dims();
        assert_eq!(
            prefix_batch, text_batch,
            "prefix batch {prefix_batch} differs from text batch {text_batch}"
        );
        let tokens = self.token_embedding.forward(input_ids);
        let token_width = tokens.dims()[2];
        assert_eq!(
            prefix_width, token_width,
            "prefix width {prefix_width} differs from model width {token_width}"
        );
        let seq_len = prefix_len + text_len;
        assert!(
            seq_len <= self.max_seq_len,
            "combined prefix+text sequence length {seq_len} exceeds configured max_seq_len {}",
            self.max_seq_len
        );
        let device = prefix.device();
        let tokens = Tensor::cat(vec![prefix, tokens], 1);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .repeat_dim(0, text_batch);
        let positions = self.position_embedding.forward(positions);
        let mut hidden = tokens + positions;
        let causal_mask = generate_autoregressive_mask::<B>(text_batch, seq_len, &device);
        for block in &self.blocks {
            hidden = block.forward(hidden, causal_mask.clone());
        }
        self.ln_final.forward(hidden)
    }

    /// The final hidden states `[batch_size, seq_len, d_model]` (post the
    /// stack and final `LayerNorm`, pre `lm_head`).
    ///
    /// Split out of [`TinyTransformer::forward`] so a pointer-generator head
    /// (Phase 51 P1a-4 Step B, `crate::pointer`) can consume the same trunk
    /// representation the LM head sees — the copy-attention + `p_gen` gate are
    /// computed from these hidden states. `forward` is exactly
    /// `vocab_logits(hidden(input_ids))`, so its behaviour is unchanged.
    pub fn hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
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

        self.ln_final.forward(hidden)
    }

    /// Project hidden states `[batch, seq, d_model]` to vocabulary logits
    /// `[batch, seq, vocab_size]` via the LM head. See [`TinyTransformer::hidden`].
    pub fn vocab_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.lm_head.forward(hidden)
    }

    /// Every intermediate hidden state the trunk produces, pre-`ln_final`:
    /// `n_layers + 1` entries -- `layers[0]` is the token+position embedding
    /// sum (before any [`DecoderBlock`] runs), and `layers[i]` for `i in
    /// 1..=n_layers` is the residual stream immediately after block `i - 1`.
    /// None of these have `ln_final` applied (that is [`TinyTransformer::lens_logits`]'s
    /// job) -- this is the raw per-layer residual stream the GWT logit-lens
    /// probe (`docs/ai/research/2026-07-09-global-workspace-language-models.md`
    /// §7.1) reads.
    ///
    /// This duplicates [`TinyTransformer::hidden`]'s embedding + block-loop
    /// logic rather than calling it, precisely so `hidden`'s behavior (and its
    /// callers' compiled code) stays byte-identical -- this method is a pure
    /// addition, not a refactor of the forward path.
    pub fn hidden_layers(&self, input_ids: Tensor<B, 2, Int>) -> Vec<Tensor<B, 3>> {
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

        let mut layers = Vec::with_capacity(self.blocks.len() + 1);
        layers.push(hidden.clone());

        let causal_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);
        for block in &self.blocks {
            hidden = block.forward(hidden, causal_mask.clone());
            layers.push(hidden.clone());
        }

        layers
    }

    /// The logit lens (GWT probe §7.1): apply the model's *own* final norm and
    /// unembedding to an intermediate hidden state `h` (e.g. one entry of
    /// [`TinyTransformer::hidden_layers`]), rather than only ever to the final
    /// post-block state. `lens_logits(h) == lm_head(ln_final(h))`; calling it on
    /// `hidden_layers(x)[n_layers]` (the last, post-final-block entry)
    /// reproduces [`TinyTransformer::forward`] exactly, since `forward` is
    /// `vocab_logits(hidden(x))` and `hidden` is `ln_final` applied to that same
    /// state.
    pub fn lens_logits(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        self.vocab_logits(self.ln_final.forward(h))
    }

    /// Resume the trunk from an (possibly patched) intermediate hidden state
    /// `h` -- run `blocks[from_block..]` on it, then `ln_final`, matching
    /// exactly the tail of [`TinyTransformer::hidden`]'s block loop that would
    /// have executed after block `from_block - 1`. The causal mask is
    /// regenerated from `h`'s own `[batch, seq, _]` dims (not carried over from
    /// wherever `h` came from), since a caller may have spliced/patched rows
    /// without changing shape.
    ///
    /// # Invariant
    /// For every `k in 0..=n_layers`: `hidden_from(hidden_layers(x)[k], k) ==
    /// hidden(x)`. At `k == 0` this replays the full block stack (identical to
    /// `hidden`); at `k == n_layers` `blocks[n_layers..]` is empty, so this is
    /// just `ln_final(h)` -- exactly what `hidden` applies to the post-block
    /// state.
    pub fn hidden_from(&self, h: Tensor<B, 3>, from_block: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = h.dims();
        let device = h.device();
        let causal_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);

        let mut hidden = h;
        for block in self.blocks.iter().skip(from_block) {
            hidden = block.forward(hidden, causal_mask.clone());
        }
        self.ln_final.forward(hidden)
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
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    use crate::{GpuBackend, gpu_device};
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
        let ids: Vec<i64> = (0..(batch_size * seq_len) as i64)
            .map(|i| i % 260)
            .collect();
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
        let input =
            Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(vec![42i64], [1, 1]), &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, 1, 260]);
    }

    #[test]
    fn cached_prefill_and_decode_match_full_forward_logits() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let ids = vec![3i64, 5, 7, 11, 13];
        let full_input = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids.clone(), [1, ids.len()]),
            &device,
        );
        let full = model.forward(full_input);

        let prefix = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids[..3].to_vec(), [1, 3]),
            &device,
        );
        let (prefill, mut cache) = model.prefill_cached(prefix);
        assert_close(
            &tensor_to_vec(prefill),
            &tensor_to_vec(full.clone().slice([0..1, 0..3, 0..260])),
            1e-4,
            "cached prefill",
        );

        for (position, token) in ids.into_iter().enumerate().skip(3) {
            let input = Tensor::<CpuBackend, 2, Int>::from_data(
                TensorData::new(vec![token], [1, 1]),
                &device,
            );
            let (step, next_cache) = model.decode_cached(input, cache);
            cache = next_cache;
            assert_close(
                &tensor_to_vec(step),
                &tensor_to_vec(full.clone().slice([0..1, position..position + 1, 0..260])),
                1e-4,
                "cached decode",
            );
        }
        assert_eq!(cache.next_position, 5);
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn cached_decode_matches_full_forward_on_the_gpu_backend() {
        let device = gpu_device();
        let model = tiny_config().init::<GpuBackend>(&device);
        let ids = vec![17i64, 19, 23, 29];
        let full = model.forward(Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(ids.clone(), [1, ids.len()]),
            &device,
        ));
        let (_, cache) = model.prefill_cached(Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(ids[..3].to_vec(), [1, 3]),
            &device,
        ));
        let (step, _) = model.decode_cached(
            Tensor::<GpuBackend, 2, Int>::from_data(TensorData::new(vec![ids[3]], [1, 1]), &device),
            cache,
        );
        let cached = step.into_data().to_vec::<f32>().unwrap();
        let expected = full
            .slice([0..1, 3..4, 0..260])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_close(&cached, &expected, 1e-3, "GPU cached decode");
    }

    #[test]
    #[should_panic(expected = "exceeds configured max_seq_len")]
    fn forward_pass_rejects_overlong_sequence() {
        let device = Default::default();
        let model = tiny_config().init::<CpuBackend>(&device);
        let ids: Vec<i64> = (0..100i64).map(|i| i % 260).collect();
        let input =
            Tensor::<CpuBackend, 2, Int>::from_data(TensorData::new(ids, [1, 100]), &device);
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
        assert!(
            n < 20_000_000,
            "spike model larger than the P0 target: {n} params"
        );
    }

    /// GWT probe hook (§7.1): `hidden_layers` returns exactly `n_layers + 1`
    /// entries -- the pre-block embedding sum plus one per decoder block.
    #[test]
    fn hidden_layers_has_n_layers_plus_one_entries() {
        let device = Default::default();
        let config = tiny_config();
        let model = config.init::<CpuBackend>(&device);

        let batch_size = 2;
        let seq_len = 6;
        let ids: Vec<i64> = (0..(batch_size * seq_len) as i64)
            .map(|i| i % 260)
            .collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids, [batch_size, seq_len]),
            &device,
        );

        let layers = model.hidden_layers(input);
        assert_eq!(layers.len(), config.n_layers + 1);
        for h in &layers {
            assert_eq!(h.dims(), [batch_size, seq_len, config.d_model]);
        }
    }

    fn tensor_to_vec<const D: usize>(t: Tensor<CpuBackend, D>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{msg}: element {i} differs: {x} vs {y} (tol {tol})"
            );
        }
    }

    /// GWT probe hook (§7.1a): the logit lens applied to the *last*
    /// `hidden_layers` entry (post-final-block, pre-`ln_final`) must reproduce
    /// `forward()` exactly, since `forward` is defined as
    /// `vocab_logits(hidden(x))` and `hidden` is `ln_final` applied to that same
    /// post-block state.
    #[test]
    fn lens_logits_on_last_layer_matches_forward() {
        let device = Default::default();
        let config = tiny_config();
        let model = config.init::<CpuBackend>(&device);

        let batch_size = 2;
        let seq_len = 5;
        let ids: Vec<i64> = (0..(batch_size * seq_len) as i64)
            .map(|i| (i * 7) % 260)
            .collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids.clone(), [batch_size, seq_len]),
            &device,
        );

        let forward_out = model.forward(input.clone());
        let layers = model.hidden_layers(input);
        let last = layers.last().cloned().expect("at least one layer");
        let lens_out = model.lens_logits(last);

        assert_eq!(forward_out.dims(), lens_out.dims());
        assert_close(
            &tensor_to_vec(forward_out),
            &tensor_to_vec(lens_out),
            1e-5,
            "lens_logits(hidden_layers(x)[n_layers]) vs forward(x)",
        );
    }

    /// GWT probe hook (§7.1b): `hidden_from(hidden_layers(x)[k], k) ==
    /// hidden(x)` for `k == 0` (replay the whole stack) and a middle `k`
    /// (resume partway through) -- the round-trip invariant the probe examples
    /// depend on.
    #[test]
    fn hidden_from_reproduces_hidden_at_every_layer() {
        let device = Default::default();
        let config = tiny_config(); // 2 layers
        let model = config.init::<CpuBackend>(&device);

        let batch_size = 2;
        let seq_len = 7;
        let ids: Vec<i64> = (0..(batch_size * seq_len) as i64)
            .map(|i| (i * 3 + 1) % 260)
            .collect();
        let input = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(ids.clone(), [batch_size, seq_len]),
            &device,
        );

        let expected = tensor_to_vec(model.hidden(input.clone()));
        let layers = model.hidden_layers(input);
        assert_eq!(layers.len(), config.n_layers + 1);

        for (k, h_k) in layers.into_iter().enumerate() {
            let resumed = model.hidden_from(h_k, k);
            assert_close(
                &tensor_to_vec(resumed),
                &expected,
                1e-5,
                &format!("hidden_from(hidden_layers(x)[{k}], {k}) vs hidden(x)"),
            );
        }
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
        assert!(
            n > 5_000_000,
            "P1a model below the 5-30M target range: {n} params"
        );
        assert!(
            n < 30_000_000,
            "P1a model above the 5-30M target range: {n} params"
        );
    }

    #[test]
    fn akh_tool_small_stays_in_compact_weight_class() {
        let device = Default::default();
        let model = TinyTransformerConfig::akh_tool_small(3_000).init::<crate::CpuBackend>(&device);
        let parameters = model.num_params();
        assert!(
            (25_000_000..=45_000_000).contains(&parameters),
            "unexpected compact tool model size: {parameters}"
        );
    }
}
