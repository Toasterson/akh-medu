//! Gemma 3 model implementation with quantization support.
//!
//! Gemma 3 is a family of multimodal language models developed by Google.
//! This implementation provides quantization for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group-Query Attention (GQA) with specialized key-value heads
//! - RMSNorm for layer normalization
//! - Specialized attention patterns with separate normalization for Q/K/V
//! - Feed-forward network with SwiGLU activation
//! - Support for 2/3/4/8-bit quantization
//!
//! References:
//! - [Gemma 3 Models](https://blog.google/technology/developers/gemma-3/)
//!

use candle_core as candle;

use std::sync::Arc;

use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::D;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;

// --- HIP-device op routing (Phase 45f / ADR 073) ----------------------------
// candle-nn's `rms_norm` / `softmax_last_dim` / `rope` are custom ops whose GPU path
// is only compiled when candle-nn itself is built with its `cuda` feature. Here we
// drive candle's cuda slot through `candle-core/hip` (a *fork*, candle-nn's cuda
// feature is off), so on a HIP device those custom ops have no `cuda_fwd` and would
// error. candle-nn ships public on-device "slow" decompositions built from primitive
// ops (sqr/sum/exp/broadcast/narrow/cat/…) — every one already parity-gated in the
// HIP backend — so we route to those when the tensor lives on the GPU. On CPU we keep
// the fast custom ops byte-for-byte (zero change to the existing path).

/// RMSNorm matching `candle_transformers::quantized_nn::RmsNorm` (weight applied
/// directly, computed in f32), but device-aware: `rms_norm_slow` on the HIP device,
/// the fast custom op on CPU.
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
            // Fused single-kernel RMSNorm on the HIP device (Phase 45h) — one launch
            // instead of the ~7 of the slow decomposition (the biggest decode kernel
            // source, profiled at ~2000 launches/token).
            #[cfg(feature = "hip")]
            return candle::cuda::rms_norm_fused(x, &self.weight, eps);
            #[cfg(not(feature = "hip"))]
            return candle_nn::ops::rms_norm_slow(x, &self.weight, eps);
        }
        candle_nn::ops::rms_norm(x, &self.weight, eps)
    }
}

/// Last-dim softmax: generic on-device form on the HIP device, fast custom op on CPU.
/// (45h: a fused single-kernel softmax was built + parity-passed but gave no decode
/// speedup — outside GEMV the decode is launch/dispatch-bound on tiny kernels, so
/// collapsing softmax's 5 launches → 1 just redistributes the idle. Reverted.)
fn softmax_last_dim_dev(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        candle_nn::ops::softmax(x, D::Minus1)
    } else {
        candle_nn::ops::softmax_last_dim(x)
    }
}

/// Rotary embedding: `rope_slow` (narrow/cat/broadcast-mul) on the HIP device, fast
/// custom op on CPU. Same interleaving as `candle_nn::rotary_emb::rope`.
fn rope_dev(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        candle_nn::rotary_emb::rope_slow(x, cos, sin)
    } else {
        candle_nn::rotary_emb::rope(x, cos, sin)
    }
}

pub const MAX_SEQ_LEN: usize = 131072; // Gemma 3 supports 128K context window
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

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

    /// Build from a shared `Arc<QTensor>` so a tied lm_head can reuse the one resident
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

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul, // ffn_gate in GGUF
    feed_forward_up: QMatMul,   // ffn_up in GGUF
    feed_forward_down: QMatMul, // ffn_down in GGUF
    /// When present (`AKH_GPU_MATMUL=1`), the gate/up/down GEMMs run on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu: Option<crate::nlu::hip::GpuMlp>,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "rocm")]
        if let Some(gpu) = &self.gpu {
            return gpu
                .forward(xs)
                .map_err(|e| candle_core::Error::Msg(format!("gpu mlp: {e}")));
        }
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
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

    // Specialized normalization for Q and K
    attention_q_norm: GemmaRmsNorm,
    attention_k_norm: GemmaRmsNorm,

    // Layer normalization
    attention_norm: GemmaRmsNorm,      // Applied before attention
    post_attention_norm: GemmaRmsNorm, // Applied after attention
    ffn_norm: GemmaRmsNorm,            // Applied before feedforward
    post_ffn_norm: GemmaRmsNorm,       // Applied after feedforward

    // Feed-forward network
    mlp: Mlp,

    // Attention parameters
    n_head: usize,    // Number of query heads
    n_kv_head: usize, // Number of key-value heads
    head_dim: usize,  // Dimension of each head
    q_dim: usize,     // Total dimension for queries

    sliding_window_size: Option<usize>,

    rotary_embedding: Arc<RotaryEmbedding>,
    neg_inf: Tensor,

    // Cache
    kv_cache: Option<(Tensor, Tensor)>, // CPU path: dynamic grow-by-cat (byte-identical)
    // Phase 46a: static pre-allocated KV cache for the HIP device. `[b, n_kv_head,
    // kv_ctx, head_dim]` allocated once; each token `slice_set`s the new position in
    // place (no `cat`, no realloc) and attention reads the `[0..pos+seq]` prefix view.
    // Fixed buffer addresses → also the prerequisite for HIP-graph capture (46c).
    kv_buf: Option<(Tensor, Tensor)>,
    kv_ctx: usize, // static KV capacity (env `AKH_KV_CTX`, default 4096)

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
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
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
                (gpu(g.wq.forward(x))?, gpu(g.wk.forward(x))?, gpu(g.wv.forward(x))?)
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

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = if k.device().is_cuda() {
            // Phase 46a: static pre-allocated KV cache. Allocate `[b, n_kv_head, kv_ctx,
            // head_dim]` once, `slice_set` the `seq_len` new positions in place at
            // `index_pos` (one `copy2d`, not a whole-cache `cat`), and read the
            // `[0..index_pos+seq_len]` prefix as a view. `index_pos == 0` starts a new
            // sequence by overwriting from offset 0 (the stale tail is never read).
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
                // f16 KV cache (Phase 46 VRAM): halves the static cache (kv_ctx=4096:
                // ~3.2 -> ~1.6 GB). llama.cpp's default; the f16 round-trip is a small
                // precision tradeoff. K/V are cast to f16 on store; the read prefix is
                // cast back to f32 for the f32 attention matmul (the prefix is `len`
                // positions, not the whole kv_ctx, so the transient stays small).
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

        // Scaled Dot-Product Attention.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Phase 46b: decode (seq_len==1, no mask) fuses q·kᵀ→online-softmax→·v over the
        // static KV in one HIP kernel — no scores tensor, no softmax swarm. Prefill
        // (seq_len>1) + CPU keep the matmul/softmax path.
        #[cfg(feature = "hip")]
        let flash = seq_len == 1 && q.device().is_cuda();
        #[cfg(not(feature = "hip"))]
        let flash = false;
        let attn_output = if flash {
            #[cfg(feature = "hip")]
            {
                let len = index_pos + seq_len;
                let (kbuf, vbuf) = self.kv_buf.as_ref().unwrap();
                candle::cuda::flash_attn_decode(&q.contiguous()?, kbuf, vbuf, len, scale as f32)?
            }
            #[cfg(not(feature = "hip"))]
            unreachable!()
        } else {
            // Repeat KV for GQA.
            let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;
            let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(mask) = mask {
                let mask = mask.broadcast_as(attn_weights.shape())?;
                let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
                attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
            }
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
    /// Token embedding table, kept QUANTIZED (Phase 46 VRAM): the lookup dequantizes
    /// only the rows it needs via `QTensor::get_rows`, so the ~1 B-row table stays
    /// Q6_K (~0.83 GB) instead of a 4 GB f32 expansion. Shared (`Arc`) with the tied
    /// lm_head `output` when there is no separate `output.weight`.
    tok_embeddings: Arc<QTensor>,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: GemmaRmsNorm,
    output: QMatMul,
    /// When present (`AKH_GPU_MATMUL=1`), the lm_head projection runs on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu_output: Option<crate::nlu::hip::GpuLinear>,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Detect architecture prefix by probing which keys exist in metadata.
        // This supports gemma3, gemma2, gemma, gemma-embedding, and future variants.
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        // Unused in Llama.cpp so we aren't using it here.
        let _rope_freq_scaling_factor = md_get("rope.scaling.factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR);

        // Compute the dimensions for queries, keys, and values
        // These are the total dimensions when projected across all heads
        let q_dim = head_count * key_length;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Optional AMD GPU path (ADR 068): AKH_GPU_MATMUL=1 uploads linear weights to the GPU
        // and runs their GEMMs via rocBLAS; AKH_GPU_QUANT=1 keeps them int4 (fits 12B/27B).
        #[cfg(feature = "rocm")]
        let gpu_blas: Option<std::sync::Arc<crate::nlu::hip::RocmBlas>> =
            if std::env::var("AKH_GPU_MATMUL").ok().as_deref() == Some("1") {
                match crate::nlu::hip::RocmBlas::load() {
                    Ok(b) => {
                        tracing::info!("AKH_GPU_MATMUL=1: routing Gemma linear matmuls through the AMD GPU");
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
        // the looked-up rows, so we skip the ~4 GB f32 expansion of the whole table.
        let tok_embeddings = Arc::new(ct.tensor(reader, "token_embd.weight", device)?);
        let norm = GemmaRmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        // lm_head: a distinct `output.weight` if present, else tied to token_embd —
        // share the single resident quantized copy via `Arc` (saves a 2nd ~0.83 GB load).
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
        // Share one RotaryEmbedding per distinct rope frequency instead of rebuilding a
        // full [MAX_SEQ_LEN x head_dim] cos/sin cache for every layer: gemma3 has only 2
        // frequencies (global + sliding), so this drops 46 duplicate ~134 MB caches
        // (~6.2 GB VRAM) with byte-identical results (keyed by the f32 freq bit pattern).
        let mut rope_cache: std::collections::HashMap<u32, Arc<RotaryEmbedding>> =
            std::collections::HashMap::new();
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
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

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            // GPU MLP (borrows the qtensors before they move into QMatMul). Prefers native Q4_K.
            #[cfg(feature = "rocm")]
            let gpu_mlp = match &gpu_blas {
                Some(blas) => Some(
                    crate::nlu::hip::GpuMlp::from_qtensors(
                        blas,
                        &feed_forward_gate,
                        &feed_forward_up,
                        &feed_forward_down,
                        device,
                        gpu_quant,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("gpu mlp build: {e}")))?,
                ),
                None => None,
            };
            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
                #[cfg(feature = "rocm")]
                gpu: gpu_mlp,
            };

            // GPU attention projections. Prefers native Q4_K.
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

            // Sliding window pattern hardcoded to 6 because it's not explicitly defined
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let rope_key = layer_rope_frequency.to_bits();
            let rotary_embedding = match rope_cache.get(&rope_key) {
                Some(r) => r.clone(),
                None => {
                    let r = Arc::new(RotaryEmbedding::new(key_length, layer_rope_frequency, device)?);
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
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
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
            span,
            span_output,
        })
    }

    /// Run the transformer stack and return the post-norm hidden state of the
    /// **last position** (shape `[batch, hidden_dim]`), before the output
    /// projection — for the Neural→VSA bridge. (akh-medu vendored extension.)
    pub fn hidden_states(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let _enter = self.span.enter();

        let mut layer_in = self.tok_embeddings.get_rows(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;
            drop(_enter);

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?;
        self.norm.forward(&x)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let h = self.hidden_states(x, index_pos)?;
        let _enter = self.span_output.enter();
        #[cfg(feature = "rocm")]
        if let Some(gpu) = &self.gpu_output {
            return gpu
                .forward(&h)
                .map_err(|e| candle_core::Error::Msg(format!("gpu lm_head: {e}")));
        }
        self.output.forward(&h)
    }

    /// Like [`forward`](Self::forward) but also returns the last-position hidden
    /// state. Returns `(logits, hidden)`. (akh-medu vendored extension.)
    pub fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.hidden_states(x, index_pos)?;
        let logits = {
            let _enter = self.span_output.enter();
            #[cfg(feature = "rocm")]
            let l = match &self.gpu_output {
                Some(gpu) => gpu
                    .forward(&h)
                    .map_err(|e| candle_core::Error::Msg(format!("gpu lm_head: {e}")))?,
                None => self.output.forward(&h)?,
            };
            #[cfg(not(feature = "rocm"))]
            let l = self.output.forward(&h)?;
            l
        };
        Ok((logits, h))
    }
}
