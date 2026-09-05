//! Gemma 4 model implementation with quantization support.
//!
//! Gemma 4 is a family of language models developed by Google. This vendored
//! loader/decoder targets the GGUF architecture tag `general.architecture =
//! gemma4` and is modelled directly on the sibling `quantized_gemma3` module so
//! the two share their ROCm GPU wiring, KV cache and hidden-state extraction.
//!
//! Key characteristics that distinguish Gemma 4 from Gemma 3 (per the llama.cpp
//! reference, commit 379ac66):
//! - Per-layer alternation between **sliding-window** layers (n_kv=8,
//!   head_dim=256, RoPE base 1e4, full-width rotation) and **full-attention**
//!   layers (every 6th: n_kv=1, head_dim=512, RoPE base 1e6, only the first 128
//!   dims rotate).
//! - **Plain** RMSNorm everywhere (no `+1` on the weight, unlike Gemma 3).
//! - Per-head Q/K RMSNorm applied **before** RoPE, plus a **bare** (weightless)
//!   RMSNorm on V (no RoPE on V).
//! - Attention score scale of **1.0** (no `1/sqrt(head_dim)`), no attention
//!   softcapping, but a final-logit softcap of 30.0.
//! - Feed-forward uses **GELU** (tanh approximation), not SiLU.
//! - A per-block `layer_output_scale` scalar applied after both residuals.
//! - Tied embeddings: `token_embd.weight` doubles as the lm_head (no
//!   `output.weight`).
//!
//! References:
//! - [Gemma](https://blog.google/technology/developers/gemma/)
//!

use candle_core as candle;

use std::sync::Arc;

use candle::quantized::QTensor;
use candle::quantized::gguf_file;
use candle::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;

pub const MAX_SEQ_LEN: usize = 131072; // Gemma 4 long-context window.

// --- HIP-device op routing (Phase 45f / ADR 073) ----------------------------
// candle-nn's `rms_norm` / `softmax_last_dim` / `rope` are custom ops whose GPU path
// is only compiled when candle-nn is built with its `cuda` feature. We drive candle's
// cuda slot through `candle-core/hip` (a fork; candle-nn's cuda feature is off), so on
// a HIP device those custom ops have no `cuda_fwd` and would error. Route to candle-nn's
// public on-device "slow" decompositions (pure primitive ops, all HIP-parity-gated) when
// the tensor lives on the GPU; keep the fast custom ops byte-for-byte on CPU. Mirrors
// `quantized_gemma3`.

/// RMSNorm matching `candle_transformers::quantized_nn::RmsNorm` (weight applied
/// directly, computed in f32), device-aware: `rms_norm_slow` on HIP, fast op on CPU.
#[derive(Debug, Clone)]
struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let eps = self.eps as f32;
        if x.device().is_cuda() {
            // Fused single-kernel RMSNorm on the HIP device (Phase 45h).
            #[cfg(feature = "hip")]
            return candle::cuda::rms_norm_fused(x, &self.weight, eps);
            #[cfg(not(feature = "hip"))]
            return candle_nn::ops::rms_norm_slow(x, &self.weight, eps);
        }
        candle_nn::ops::rms_norm(x, &self.weight, eps)
    }
}

/// Last-dim softmax: generic on-device form on HIP, fast custom op on CPU.
/// (45h: fused-softmax built+parity-passed but no decode gain — reverted; see gemma3.)
fn softmax_last_dim_dev(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        candle_nn::ops::softmax(x, D::Minus1)
    } else {
        candle_nn::ops::softmax_last_dim(x)
    }
}

/// Rotary embedding: `rope_slow` on HIP, fast custom op on CPU.
fn rope_dev(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        candle_nn::rotary_emb::rope_slow(x, cos, sin)
    } else {
        candle_nn::rotary_emb::rope(x, cos, sin)
    }
}

// --- architecture constants (overridable from GGUF metadata) --------------------------------
pub const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6; // every 6th layer is full-attention.
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.; // full-attention layers.
pub const DEFAULT_ROPE_FREQUENCY_SWA: f32 = 10_000.; // sliding-window layers.
pub const DEFAULT_SLIDING_WINDOW: usize = 1024;
pub const DEFAULT_FINAL_LOGIT_SOFTCAP: f64 = 30.0;

/// Plain RMSNorm over the last dim **without any weight**:
/// `x / sqrt(mean(x^2) + eps)`. Used on V (the reference applies a weightless
/// RMSNorm to V before attention).
fn rmsnorm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    x_norm.to_dtype(in_dtype)
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Build from a shared `Arc<QTensor>` so the tied lm_head reuses the one resident
    /// copy of `token_embd` (Phase 46 VRAM) instead of a second load.
    fn from_arc(qtensor: Arc<QTensor>) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_arc(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// Gemma 4 MLP: `down(gelu(gate(x)) * up(x))` with **GELU** (tanh approx), not SiLU.
///
/// The GPU path deliberately does **not** use [`crate::nlu::hip::GpuMlp`] — that
/// type fuses a hardcoded SiLU activation which is wrong for Gemma 4. Instead it
/// routes gate/up/down through individual [`crate::nlu::hip::GpuLinear`] matmuls
/// and does `gelu(gate) * up` in Candle (option (a) from ADR 068).
#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul, // ffn_gate in GGUF
    feed_forward_up: QMatMul,   // ffn_up in GGUF
    feed_forward_down: QMatMul, // ffn_down in GGUF
    /// When present (`AKH_GPU_MATMUL=1`), the gate/up/down GEMMs run on the AMD GPU
    /// as three separate `GpuLinear`s (NOT the silu-fused `GpuMlp`).
    #[cfg(feature = "rocm")]
    gpu_gate: Option<crate::nlu::hip::GpuLinear>,
    #[cfg(feature = "rocm")]
    gpu_up: Option<crate::nlu::hip::GpuLinear>,
    #[cfg(feature = "rocm")]
    gpu_down: Option<crate::nlu::hip::GpuLinear>,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if let (Some(gate), Some(up), Some(down)) = (&self.gpu_gate, &self.gpu_up, &self.gpu_down) {
            let g = gate
                .forward(xs)
                .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_gate: {e}")))?;
            let u = up
                .forward(xs)
                .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_up: {e}")))?;
            // GELU (tanh approx) on the gate, NOT SiLU.
            let gated = (g.gelu()? * u)?;
            return down
                .forward(&gated)
                .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_down: {e}")));
        }
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let gelu = gate.gelu()?; // tanh approximation (gelu_pytorch_tanh).
        let gated = (gelu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

/// NEOX rotary embedding table. For full-attention layers only the first
/// `rotary_pairs` frequency pairs are non-zero (the remaining pairs become
/// identity rotations because their inv_freq is 0 → cos=1, sin=0).
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Build a rope table for a head of `head_dim` dims using `rope_frequency` as
    /// the base. Only the first `rotary_pairs` of the `head_dim/2` frequency pairs
    /// actually rotate; the rest are forced to identity (inv_freq = 0).
    fn new(
        head_dim: usize,
        rotary_pairs: usize,
        rope_frequency: f32,
        device: &Device,
    ) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .enumerate()
            .map(|(pair, i)| {
                if pair < rotary_pairs {
                    1f32 / rope_frequency.powf(i as f32 / head_dim as f32)
                } else {
                    0f32 // identity rotation for non-rotated pairs.
                }
            })
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = rope_dev(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rope_dev(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    // Attention components
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    /// When present (`AKH_GPU_MATMUL=1`), the q/k/v/o projections run on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu_attn: Option<crate::nlu::hip::GpuAttnProj>,

    // Specialized normalization for Q and K (per-head, over head_dim), plain (no +1).
    attention_q_norm: GemmaRmsNorm,
    attention_k_norm: GemmaRmsNorm,

    // Layer normalization (all plain RMSNorm, no +1)
    attention_norm: GemmaRmsNorm,      // Applied before attention
    post_attention_norm: GemmaRmsNorm, // Applied after attention
    ffn_norm: GemmaRmsNorm,            // Applied before feedforward
    post_ffn_norm: GemmaRmsNorm,       // Applied after feedforward

    // Feed-forward network
    mlp: Mlp,

    /// The [1] `layer_output_scale` scalar broadcast over the whole layer output.
    layer_output_scale: Tensor,

    // Attention parameters
    n_head: usize,    // Number of query heads (16 always)
    n_kv_head: usize, // Number of key-value heads (8 sliding, 1 full)
    head_dim: usize,  // Dimension of each head (256 sliding, 512 full)
    q_dim: usize,     // Total dimension for queries (n_head * head_dim)

    rms_norm_eps: f64,

    sliding_window_size: Option<usize>,

    rotary_embedding: Arc<RotaryEmbedding>,
    neg_inf: Tensor,

    // Cache
    kv_cache: Option<(Tensor, Tensor)>, // CPU path: dynamic grow-by-cat (byte-identical)
    // Phase 46a: static pre-allocated KV cache for the HIP device (see quantized_gemma3).
    kv_buf: Option<(Tensor, Tensor)>,
    kv_ctx: usize,

    // Tracing
    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let total = seq_len + index_pos;
        let rows = Self::causal_mask_rows(seq_len, index_pos, self.sliding_window_size);
        Tensor::from_slice(&rows, (seq_len, total), device)?
            .expand((b_sz, 1, seq_len, total))?
            .to_dtype(dtype)
    }

    /// Row-major causal attention-mask values `[seq_len * (index_pos + seq_len)]` (`1` = keep,
    /// `0` = mask — the convention [`forward_attn`]'s `mask.eq(0)?.where_cond(neg_inf, …)`
    /// consumes). Query `i` sits at GLOBAL position `index_pos + i` and keeps every key
    /// `j ≤ index_pos + i` (the whole cached prefix `[0, index_pos)` plus the causal new block);
    /// a sliding `window` (when `Some`) additionally drops keys older than the window
    /// (`j + window < index_pos + i`). Built host-side so the device construction is one
    /// `from_slice` — the HIP backend has no `ones`/`const_set`. For `index_pos == 0` this is
    /// byte-identical to the original prefill mask (lower-triangular, windowed where applicable).
    fn causal_mask_rows(seq_len: usize, index_pos: usize, window: Option<usize>) -> Vec<u32> {
        let total = seq_len + index_pos;
        (0..seq_len)
            .flat_map(|i| {
                let qg = index_pos + i; // global query position
                (0..total).map(move |j| {
                    let future = j > qg;
                    let out_of_window = window.is_some_and(|w| j + w < qg);
                    if future || out_of_window { 0u32 } else { 1u32 }
                })
            })
            .collect()
    }

    /// Phase 47 speculative-decode **verify** mask: pure-causal continuation, `[b, 1, seq,
    /// index_pos+seq]`, `1 = keep / 0 = mask` (the convention [`forward_attn`] consumes). Query
    /// `index_pos + i` keeps every key `j ≤ index_pos + i` — the whole cached prefix plus the
    /// causal new block, and **no sliding-window restriction** so it matches `seq_len==1` decode
    /// (which passes `mask = None` and thus ignores the window); that makes the verify forward
    /// byte-exact to incremental decode. Built once per forward (layer-independent without the
    /// window) and host-side via `from_slice` (no device `const_set`/`ones`).
    fn verify_mask(
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let total = seq_len + index_pos;
        // The no-window continuation case of `causal_mask_rows` — explicitly `None` so a
        // sliding-window layer is NOT windowed here (matching `seq==1` decode's `mask=None`).
        let rows = Self::causal_mask_rows(seq_len, index_pos, None);
        Tensor::from_slice(&rows, (seq_len, total), device)?
            .expand((1, 1, seq_len, total))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        #[cfg(feature = "rocm")]
        let (q, k, v) = match &self.gpu_attn {
            // Share one activation upload across q/k/v on the GPU.
            Some(g) if g.supports_fused() => g
                .project_qkv(x)
                .map_err(|e| candle_core::Error::Msg(format!("gpu attn qkv: {e}")))?,
            Some(g) => {
                let gpu = |t: std::result::Result<Tensor, crate::nlu::error::NluError>| {
                    t.map_err(|e| candle_core::Error::Msg(format!("gpu attn proj: {e}")))
                };
                (
                    gpu(g.wq.forward(x))?,
                    gpu(g.wk.forward(x))?,
                    gpu(g.wv.forward(x))?,
                )
            }
            None => (
                self.attention_wq.forward(x)?,
                self.attention_wk.forward(x)?,
                self.attention_wv.forward(x)?,
            ),
        };
        #[cfg(not(feature = "rocm"))]
        let q = self.attention_wq.forward(x)?;
        #[cfg(not(feature = "rocm"))]
        let k = self.attention_wk.forward(x)?;
        #[cfg(not(feature = "rocm"))]
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head Q/K RMSNorm over head_dim (plain, no +1), applied BEFORE RoPE.
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        // Bare (weightless) RMSNorm on V over head_dim; NO RoPE on V.
        let v = rmsnorm_no_weight(&v.contiguous()?, self.rms_norm_eps)?;

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = if k.device().is_cuda() {
            // Phase 46a: static pre-allocated KV cache (see quantized_gemma3 for detail).
            let len = index_pos + seq_len;
            if len > self.kv_ctx {
                return Err(candle_core::Error::Msg(format!(
                    "static KV cache overflow: position {len} > kv_ctx {} (raise AKH_KV_CTX)",
                    self.kv_ctx
                )));
            }
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            if self.kv_buf.is_none() {
                let shape = (b_sz, self.n_kv_head, self.kv_ctx, self.head_dim);
                // f16 KV cache (Phase 46 VRAM): halves the static cache (~3.2 -> ~1.6 GB
                // at kv_ctx=4096). K/V cast to f16 on store; read prefix back to f32 for
                // the f32 attention matmul (prefix is `len` positions, transient is small).
                self.kv_buf = Some((
                    Tensor::zeros(shape, DType::F16, k.device())?,
                    Tensor::zeros(shape, DType::F16, v.device())?,
                ));
            }
            let (kbuf, vbuf) = self.kv_buf.as_ref().unwrap();
            kbuf.slice_set(&k.to_dtype(DType::F16)?, 2, index_pos)?;
            vbuf.slice_set(&v.to_dtype(DType::F16)?, 2, index_pos)?;
            (
                kbuf.narrow(2, 0, len)?.contiguous()?.to_dtype(DType::F32)?,
                vbuf.narrow(2, 0, len)?.contiguous()?.to_dtype(DType::F32)?,
            )
        } else {
            // CPU path: dynamic grow-by-cat (byte-identical to the original).
            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[k_cache, &k], 2)?; // concat on seq dim
                        let v = Tensor::cat(&[v_cache, &v], 2)?;
                        (k, v)
                    }
                }
            };
            self.kv_cache = Some((k.clone(), v.clone())); // update cache
            (k, v)
        };

        // Scaled Dot-Product Attention. Gemma 4 uses scale = 1.0 (q is pre-scaled; NO
        // 1/sqrt(head_dim)). Phase 46b: decode fuses the attention into one flash kernel.
        #[cfg(feature = "hip")]
        let flash = seq_len == 1 && q.device().is_cuda();
        #[cfg(not(feature = "hip"))]
        let flash = false;
        let attn_output = if flash {
            #[cfg(feature = "hip")]
            {
                // `k`/`v` are the f32 static-KV prefix (already narrowed to `len` and cast
                // from the f16 store above) — the flash kernel is f32-only and does GQA +
                // the [0..len) read itself, so feed it the cast prefix, not the raw f16 buffer.
                let len = index_pos + seq_len;
                candle::cuda::flash_attn_decode(&q.contiguous()?, &k, &v, len, 1.0f32)?
            }
            #[cfg(not(feature = "hip"))]
            unreachable!()
        } else {
            // Repeat KV for GQA/MQA (n_kv -> n_head).
            let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;
            let mut attn_weights = q.matmul(&k.transpose(2, 3)?)?; // scale = 1.0 (no-op)
            if let Some(mask) = mask {
                let mask = mask.broadcast_as(attn_weights.shape())?;
                let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
                attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
            }
            // softmax in f32 (no attention softcapping).
            let attn_weights = softmax_last_dim_dev(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;

        #[cfg(feature = "rocm")]
        if let Some(g) = &self.gpu_attn {
            return g
                .wo
                .forward(&attn_output)
                .map_err(|e| candle_core::Error::Msg(format!("gpu attn wo: {e}")));
        }
        self.attention_wo.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding table kept QUANTIZED (Phase 46 VRAM) — `get_rows` dequantizes
    /// only looked-up rows (Q6_K ~0.83 GB vs 4 GB f32). `Arc`-shared with the tied lm_head.
    tok_embeddings: Arc<QTensor>,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: GemmaRmsNorm,
    output: QMatMul,
    /// When present (`AKH_GPU_MATMUL=1`), the lm_head projection runs on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu_output: Option<crate::nlu::hip::GpuLinear>,
    final_logit_softcap: f64,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Architecture prefix. Gemma 4 tags itself `gemma4`; probe for a known key.
        let prefix = ["gemma4", "gemma3"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma4");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };
        // Tolerant integer read: this GGUF (unsloth conversion) stores several hparams as
        // SIGNED i32 (e.g. `head_count_kv` elements), so a bare `to_u32()` fails with
        // "not a u32 I32(..)". Accept any integer width/signedness.
        let vusize = |v: &gguf_file::Value| -> Result<usize> {
            v.to_u32()
                .map(|x| x as usize)
                .or_else(|_| v.to_i32().map(|x| x as usize))
                .or_else(|_| v.to_u64().map(|x| x as usize))
                .or_else(|_| v.to_i64().map(|x| x as usize))
                .or_else(|_| v.to_u16().map(|x| x as usize))
                .or_else(|_| v.to_i16().map(|x| x as usize))
        };
        let mdu = |s: &str| -> Result<usize> { vusize(md_get(s)?) };

        let head_count = mdu("attention.head_count")?;
        let block_count = mdu("block_count")?;
        let embedding_length = mdu("embedding_length")?;

        // Per-layer head_count_kv: an ARRAY of `block_count` ints (full layers = 1,
        // sliding layers = 8). Fall back to a scalar, then to the derived pattern.
        let head_count_kv_per_layer: Vec<usize> = match md_get("attention.head_count_kv") {
            Ok(v) => match v.to_vec() {
                Ok(arr) => arr.iter().map(&vusize).collect::<Result<Vec<_>>>()?,
                // Scalar fallback: replicate across all layers.
                Err(_) => vec![vusize(v)?; block_count],
            },
            Err(_) => Vec::new(), // derived below.
        };

        // Per-layer sliding pattern: ARRAY of `block_count` bools (true = sliding).
        // Fall back to deriving it: layer `il` is FULL iff (il + 1) % 6 == 0.
        let sliding_window_pattern_arr: Option<Vec<bool>> =
            match md_get("attention.sliding_window_pattern") {
                Ok(v) => v.to_vec().ok().map(|arr| {
                    arr.iter()
                        .map(|e| e.to_bool().unwrap_or(false))
                        .collect::<Vec<_>>()
                }),
                Err(_) => None,
            };

        // Key/value lengths differ per layer-type; read all four with sane defaults.
        let key_length_full = mdu("attention.key_length").unwrap_or(512);
        let value_length_full = mdu("attention.value_length").unwrap_or(512);
        let key_length_swa = mdu("attention.key_length_swa").unwrap_or(256);
        let _value_length_swa = mdu("attention.value_length_swa").unwrap_or(256);
        let _ = value_length_full;

        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")
            .and_then(|m| Ok(m.to_f32()? as f64))
            .unwrap_or(1e-6);
        let sliding_window_size = mdu("attention.sliding_window").unwrap_or(DEFAULT_SLIDING_WINDOW);

        let rope_freq_base_full = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_swa = md_get("rope.freq_base_swa")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SWA);

        // Number of dims that rope actually rotates. SWA rotates all of head_dim;
        // FULL rotates only `rope.dimension_count_swa` (= 256 → 128 effective per the
        // reference: only the first 128 dims = 64 pairs of the 512-dim head rotate).
        let rope_dim_full = mdu("rope.dimension_count").unwrap_or(512);
        let rope_dim_swa = mdu("rope.dimension_count_swa").unwrap_or(256);
        let _ = rope_dim_full;

        let final_logit_softcap = md_get("final_logit_softcapping")
            .and_then(|m| Ok(m.to_f32()? as f64))
            .unwrap_or(DEFAULT_FINAL_LOGIT_SOFTCAP);

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Optional AMD GPU path (ADR 068): AKH_GPU_MATMUL=1 uploads linear weights to the GPU
        // and runs their GEMMs via rocBLAS; AKH_GPU_QUANT=1 keeps them int4 (fits 12B/27B).
        #[cfg(feature = "rocm")]
        let gpu_blas: Option<std::sync::Arc<crate::nlu::hip::RocmBlas>> =
            if std::env::var("AKH_GPU_MATMUL").ok().as_deref() == Some("1") {
                match crate::nlu::hip::RocmBlas::load() {
                    Ok(b) => {
                        tracing::info!(
                            "AKH_GPU_MATMUL=1: routing Gemma 4 linear matmuls through the AMD GPU"
                        );
                        Some(std::sync::Arc::new(b))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "AKH_GPU_MATMUL=1 but ROCm unavailable — CPU");
                        None
                    }
                }
            } else {
                None
            };
        #[cfg(feature = "rocm")]
        let gpu_quant = std::env::var("AKH_GPU_QUANT").ok().as_deref() == Some("1");

        // Token embeddings kept QUANTIZED (Phase 46 VRAM): `get_rows` dequantizes only
        // the looked-up rows, skipping the ~4 GB f32 expansion of the whole table.
        let tok_embeddings = Arc::new(ct.tensor(reader, "token_embd.weight", device)?);
        let norm = GemmaRmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        // Tied lm_head (no output.weight in Gemma 4): share the one resident quantized
        // copy of token_embd via `Arc` (saves a 2nd ~0.83 GB load).
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => Arc::new(tensor),
            Err(_) => tok_embeddings.clone(),
        };
        #[cfg(feature = "rocm")]
        let gpu_output: Option<crate::nlu::hip::GpuLinear> = match &gpu_blas {
            Some(blas) => Some(
                crate::nlu::hip::GpuLinear::from_qtensor(blas, &output, device, gpu_quant)
                    .map_err(|e| candle_core::Error::Msg(format!("gpu lm_head: {e}")))?,
            ),
            None => None,
        };

        let mut layers = Vec::with_capacity(block_count);
        // Share one RotaryEmbedding per distinct (head_dim, rotary_pairs, freq) instead of
        // rebuilding a full [MAX_SEQ_LEN x pairs] cos/sin cache per layer: gemma4 has 2
        // layer types (SWA + full), so this drops ~46 duplicate caches (~6 GB VRAM),
        // byte-identical (freq keyed by its f32 bit pattern).
        let mut rope_cache: std::collections::HashMap<(usize, usize, u32), Arc<RotaryEmbedding>> =
            std::collections::HashMap::new();
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            // Layer-type resolution: prefer the GGUF arrays, else derive (every 6th
            // layer — (il+1) % 6 == 0 — is FULL-attention, the rest are sliding).
            let is_sliding = match &sliding_window_pattern_arr {
                Some(arr) if layer_idx < arr.len() => arr[layer_idx],
                _ => (layer_idx + 1) % DEFAULT_SLIDING_WINDOW_PATTERN != 0,
            };

            let n_kv_head = match head_count_kv_per_layer.get(layer_idx) {
                Some(&n) => n,
                None => {
                    if is_sliding {
                        8
                    } else {
                        1
                    }
                }
            };

            let head_dim = if is_sliding {
                key_length_swa
            } else {
                key_length_full
            };
            let q_dim = head_count * head_dim;

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            // Full-attention layers (every 6th) omit a separate V projection and tie V to the
            // K projection (reference: "if no wv tensor, V = K"); they then get a bare V-norm
            // while K gets k_norm. Re-read attn_k as the V weight when attn_v is absent.
            let attention_wv = match ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device) {
                Ok(t) => t,
                Err(_) => ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
            };
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_k_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attention_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;
            let ffn_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = GemmaRmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            // Per-block [1] layer_output_scale scalar.
            let layer_output_scale = ct
                .tensor(
                    reader,
                    &format!("{prefix}.layer_output_scale.weight"),
                    device,
                )?
                .dequantize(device)?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            // GPU MLP: three independent GpuLinear matmuls (NOT GpuMlp's silu-fused path,
            // which would apply the wrong activation for Gemma 4). gelu(gate)*up done in
            // Candle inside Mlp::forward.
            #[cfg(feature = "rocm")]
            let (gpu_gate, gpu_up, gpu_down) = match &gpu_blas {
                Some(blas) => {
                    let g = crate::nlu::hip::GpuLinear::from_qtensor(
                        blas,
                        &feed_forward_gate,
                        device,
                        gpu_quant,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_gate build: {e}")))?;
                    let u = crate::nlu::hip::GpuLinear::from_qtensor(
                        blas,
                        &feed_forward_up,
                        device,
                        gpu_quant,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_up build: {e}")))?;
                    let d = crate::nlu::hip::GpuLinear::from_qtensor(
                        blas,
                        &feed_forward_down,
                        device,
                        gpu_quant,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("gpu ffn_down build: {e}")))?;
                    (Some(g), Some(u), Some(d))
                }
                None => (None, None, None),
            };

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
                #[cfg(feature = "rocm")]
                gpu_gate,
                #[cfg(feature = "rocm")]
                gpu_up,
                #[cfg(feature = "rocm")]
                gpu_down,
            };

            // GPU attention projections (plain matmuls — safe to use GpuAttnProj).
            #[cfg(feature = "rocm")]
            let gpu_attn = match &gpu_blas {
                Some(blas) => Some(
                    crate::nlu::hip::GpuAttnProj::from_qtensors(
                        blas,
                        &attention_wq,
                        &attention_wk,
                        &attention_wv,
                        &attention_wo,
                        device,
                        gpu_quant,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("gpu attn build: {e}")))?,
                ),
                None => None,
            };

            let sliding_window_size = is_sliding.then_some(sliding_window_size);

            // RoPE table: SWA rotates all head_dim dims (base 1e4); FULL rotates only
            // the first rope_dim_swa/2 (=128 dims → 64 pairs) with base 1e6.
            let (layer_rope_frequency, rotary_pairs) = if is_sliding {
                (rope_freq_base_swa, head_dim / 2)
            } else {
                // Only the first `rope_dim_swa` dims (=256/2=128 → but per reference only
                // 128 dims = 64 pairs) rotate on full layers. rope_dim_swa is 256, so
                // rotary_pairs = rope_dim_swa / 2 = 128… the reference rotates 128 dims =
                // 64 pairs, i.e. rope_dim_swa / 4. We follow the spec: 64 pairs.
                (rope_freq_base_full, (rope_dim_swa / 2) / 2)
            };
            let _ = rope_dim_swa; // (consumed above)

            let rope_key = (head_dim, rotary_pairs, layer_rope_frequency.to_bits());
            let rotary_embedding = match rope_cache.get(&rope_key) {
                Some(r) => r.clone(),
                None => {
                    let r = Arc::new(RotaryEmbedding::new(
                        head_dim,
                        rotary_pairs,
                        layer_rope_frequency,
                        device,
                    )?);
                    rope_cache.insert(rope_key, r.clone());
                    r
                }
            };

            // Tracing spans
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                #[cfg(feature = "rocm")]
                gpu_attn,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                layer_output_scale,
                n_head: head_count,
                n_kv_head,
                head_dim,
                q_dim,
                rms_norm_eps,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                kv_buf: None,
                kv_ctx: std::env::var("AKH_KV_CTX")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(4096),
                span_attn,
                span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings,
            #[cfg(feature = "rocm")]
            gpu_output,
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_arc(output)?,
            final_logit_softcap,
            span,
            span_output,
        })
    }

    /// Embedding length (`= 3840` for Gemma 4 12B). Exposed so callers can build
    /// spliced input-embedding tensors (e.g. audio frames) at the right width.
    /// (akh-medu vendored extension.)
    pub fn embedding_length(&self) -> usize {
        self.embedding_length
    }

    /// Static KV-cache capacity (env `AKH_KV_CTX`, default 4096) — the upper bound for a
    /// speculative-decode verify span / decode position. `0` before any layer is built.
    pub fn kv_ctx(&self) -> usize {
        self.layers.first().map(|l| l.kv_ctx).unwrap_or(0)
    }

    /// Look up the (un-scaled) token embeddings for `ids` (shape `[batch, seq]`),
    /// returning `[batch, seq, embedding_length]`. The caller is responsible for
    /// any `sqrt(embedding_length)` scaling (the decoder applies that to *text*
    /// tokens — see [`hidden_states`]/[`hidden_states_from_embeds`]). Used to
    /// build the base sequence for multimodal splicing (e.g. native audio).
    /// (akh-medu vendored extension.)
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.get_rows(ids)
    }

    /// `sqrt(embedding_length)` — the scale Gemma 4 applies to *text*-token
    /// embeddings. Audio-frame embeddings, already projected into embedding
    /// space, are NOT scaled by this. (akh-medu vendored extension.)
    pub fn embedding_scale(&self) -> f64 {
        (self.embedding_length as f64).sqrt()
    }

    /// Run the transformer stack and return the post-norm hidden state of the
    /// **last position** (shape `[batch, hidden_dim]`), before the output
    /// projection — for the Neural→VSA bridge. (akh-medu vendored extension.)
    pub fn hidden_states(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.get_rows(x)?;
        // Scale embeddings by sqrt(embedding_length) (≈61.9677 for 3840).
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
        let (b_sz, seq_len) = x.dims2()?;
        self.run_layers(layer_in, b_sz, seq_len, index_pos, true)
    }

    /// Like [`hidden_states`] but starts from a pre-built input-embedding tensor
    /// `[batch, seq, embedding_length]` instead of token ids. **No scaling is
    /// applied here** — the caller must have already scaled text-token rows by
    /// `sqrt(embedding_length)` and left directly-projected rows (e.g. audio
    /// frames) un-scaled. (akh-medu vendored extension.)
    pub fn hidden_states_from_embeds(
        &mut self,
        embeds: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = embeds.dims3()?;
        self.run_layers(embeds.clone(), b_sz, seq_len, index_pos, true)
    }

    /// Shared decoder body: run all layers over a pre-scaled input-embedding tensor and return
    /// the post-norm hidden state. With `last_only` it returns only the **last position**
    /// (`[b, hidden]`, the decode/Neural-VSA path); otherwise it returns **every position**
    /// (`[b, seq, hidden]`) for the Phase 47 speculative-decode verify forward. `last_only` is
    /// byte-exact to the all-position result sliced at `seq_len-1` (RMSNorm is per-position).
    fn run_layers(
        &mut self,
        mut layer_in: Tensor,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        last_only: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dtype = layer_in.dtype();
        let device = layer_in.device().clone();
        // Phase 47 verify forward (`last_only == false`, `seq_len > 1`): one pure-causal
        // continuation mask shared by every layer (no sliding window → matches `seq==1` decode →
        // byte-exact to incremental decode). The normal path keeps the per-layer windowed
        // prefill mask / `None` for single-token decode.
        let verify_mask = if !last_only && seq_len > 1 {
            Some(LayerWeights::verify_mask(
                seq_len, index_pos, dtype, &device,
            )?)
        } else {
            None
        };
        for layer in self.layers.iter_mut() {
            let attention_mask = if let Some(m) = &verify_mask {
                Some(m.clone())
            } else if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, dtype, &device)?)
            };

            // Attention block (all norms plain RMSNorm).
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block.
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let h = layer.ffn_norm.forward(&x)?;
            let h = layer.mlp.forward(&h)?;
            let h = layer.post_ffn_norm.forward(&h)?;
            let h = (h + residual)?;
            drop(_enter);

            // Per-block layer_output_scale applied to the whole residual stream
            // after both residual adds (matches llama.cpp gemma4 `out_scale`).
            layer_in = h.broadcast_mul(&layer.layer_output_scale)?;
        }

        if last_only {
            let x = layer_in.i((.., seq_len - 1, ..))?;
            self.norm.forward(&x)
        } else {
            // All-position hidden `[b, seq, hidden]` — RMSNorm normalizes each position
            // independently, so this is byte-exact to the last-position path at `seq_len-1`.
            self.norm.forward(&layer_in)
        }
    }

    /// Apply the tied lm_head and the final-logit softcap: `30 * tanh(logits/30)`.
    fn project_and_softcap(&self, h: &Tensor) -> Result<Tensor> {
        let _enter = self.span_output.enter();
        #[cfg(feature = "rocm")]
        let logits = match &self.gpu_output {
            Some(gpu) => gpu
                .forward(h)
                .map_err(|e| candle_core::Error::Msg(format!("gpu lm_head: {e}")))?,
            None => self.output.forward(h)?,
        };
        #[cfg(not(feature = "rocm"))]
        let logits = self.output.forward(h)?;

        // Final logit softcap: logits = cap * tanh(logits / cap).
        let cap = self.final_logit_softcap;
        if cap > 0.0 {
            let scaled = (logits / cap)?;
            scaled.tanh()? * cap
        } else {
            Ok(logits)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let h = self.hidden_states(x, index_pos)?;
        self.project_and_softcap(&h)
    }

    /// Phase 47 speculative-decode **verify forward**: logits at EVERY position
    /// (`[b, seq, vocab]`) for the token ids `x` (`[b, seq]`) fed as a continuation at
    /// `index_pos`. The target scores all `seq` candidate positions in one forward; position
    /// `index_pos + i` attends the cached prefix `[0, index_pos)` plus the causal `[0, i]` of the
    /// new block (see [`LayerWeights::mask`]). Byte-identical to running `seq` single-token
    /// [`forward`](Self::forward)s incrementally (same K/V writes, same per-position rope/mask) —
    /// the last position equals a plain `forward` of the same token. K/V for all `seq` positions
    /// are written into the static cache at `[index_pos, index_pos+seq)`; on partial acceptance
    /// the caller rewinds the write index (Phase 47c).
    pub fn forward_verify(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.get_rows(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;
        let (b_sz, seq_len) = x.dims2()?;
        let h = self.run_layers(layer_in, b_sz, seq_len, index_pos, false)?;
        self.project_and_softcap(&h)
    }

    /// Like [`forward`](Self::forward) but also returns the last-position hidden
    /// state. Returns `(logits, hidden)`. (akh-medu vendored extension.)
    pub fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.hidden_states(x, index_pos)?;
        let logits = self.project_and_softcap(&h)?;
        Ok((logits, h))
    }

    /// Run the decoder from a pre-built input-embedding tensor
    /// `[1, seq, embedding_length]` instead of token ids, and return the
    /// last-position logits. Used for multimodal prefill where some embedding
    /// rows are projected directly into embedding space (e.g. native audio
    /// frames) and must NOT be `sqrt(embedding_length)`-scaled — the caller owns
    /// the per-row scaling. (akh-medu vendored extension.)
    pub fn forward_embeds(&mut self, embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let h = self.hidden_states_from_embeds(embeds, index_pos)?;
        self.project_and_softcap(&h)
    }
}

#[cfg(test)]
mod mask_tests {
    use super::LayerWeights;

    // index_pos == 0 is the live prefill path: must stay plain lower-triangular causal
    // (byte-identical to the original mask before the continuation fix).
    #[test]
    fn causal_mask_rows_prefill_is_lower_triangular() {
        assert_eq!(
            LayerWeights::causal_mask_rows(3, 0, None),
            vec![
                1, 0, 0, //
                1, 1, 0, //
                1, 1, 1, //
            ]
        );
    }

    // Continuation (index_pos > 0): the previously dead+broken branch. Query `i` is at global
    // position `index_pos + i` and must keep the WHOLE cached prefix plus the causal new block.
    // This is also exactly what `verify_mask` (the Phase 47a forward) builds.
    #[test]
    fn causal_mask_rows_continuation_keeps_prefix() {
        // seq_len=2 appended at index_pos=3 (global queries 3, 4); total = 5.
        assert_eq!(
            LayerWeights::causal_mask_rows(2, 3, None),
            vec![
                1, 1, 1, 1, 0, // q=3: keep keys 0..=3
                1, 1, 1, 1, 1, // q=4: keep keys 0..=4
            ]
        );
    }

    // Sliding window applied in GLOBAL coordinates: drop keys older than the window.
    #[test]
    fn causal_mask_rows_continuation_sliding_window() {
        assert_eq!(
            LayerWeights::causal_mask_rows(2, 3, Some(2)),
            vec![
                0, 1, 1, 1, 0, // q=3: keep 1,2,3
                0, 0, 1, 1, 1, // q=4: keep 2,3,4
            ]
        );
    }
}
