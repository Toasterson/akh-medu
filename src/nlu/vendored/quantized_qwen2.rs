//! Qwen2 model implementation with quantization support.
//!
//! Qwen2 is a chat-optimized language model that supports 8-bit quantization
//! for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group Query Attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - [Model Card](https://huggingface.co/Qwen/Qwen2)
//!

use candle_core as candle;

use candle_transformers::utils::repeat_kv;
use candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    D, DType, Device, IndexOp, Result, Tensor,
};
use candle_nn::{Embedding, Module};
use std::collections::HashMap;

// --- HIP-device op routing (Phase 45f / ADR 073) ----------------------------
// candle-transformers' `RmsNorm::forward` calls candle-nn's `rms_norm`, a custom
// op whose GPU path is only compiled when candle-nn itself is built with its
// `cuda` feature. Here we drive candle's cuda slot through `candle-core/hip` (a
// *fork*; candle-nn's cuda feature is off), so on a HIP device that custom op has
// no `cuda_fwd` and would error. The same idiom as `GemmaRmsNorm` in
// quantized_gemma3: on a CUDA/HIP device use candle's fused single-kernel
// `rms_norm_fused` (or the public `rms_norm_slow` decomposition when not built
// with the `hip` feature), and on CPU keep the fast custom op byte-for-byte.
/// RMSNorm matching `candle_transformers::quantized_nn::RmsNorm` (dequantized
/// weight applied directly, computed in f32), but device-aware on the HIP device.
#[derive(Debug, Clone)]
struct Qwen2RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen2RmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for Qwen2RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let eps = self.eps as f32;
        if x.device().is_cuda() {
            // Fused single-kernel RMSNorm on the HIP device (mirrors GemmaRmsNorm).
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

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    /// When present (built at load with `AKH_GPU_MATMUL=1`), the three projections run on the
    /// AMD GPU via rocBLAS (ADR 068) instead of the quantized CPU path.
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
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    /// When present (`AKH_GPU_MATMUL=1`), the q/k/v/o projections run on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu_attn: Option<crate::nlu::hip::GpuAttnProj>,
    attention_norm: Qwen2RmsNorm,
    mlp: Mlp,
    ffn_norm: Qwen2RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        rope_dev(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

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

        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // let (q, k) = self
        //     .rotary_embedding
        //     .apply_rotary_emb_qkv(&q, &k, index_pos)?;
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for MQA, useful for 70B models and mistral.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = softmax_last_dim_dev(&att)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        #[cfg(feature = "rocm")]
        let y = match &self.gpu_attn {
            Some(g) => g
                .wo
                .forward(&y)
                .map_err(|e| candle_core::Error::Msg(format!("gpu attn wo: {e}")))?,
            None => self.attention_wo.forward(&y)?,
        };
        #[cfg(not(feature = "rocm"))]
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: Qwen2RmsNorm,
    output: QMatMul,
    /// When present (`AKH_GPU_MATMUL=1`), the lm_head projection runs on the AMD GPU.
    #[cfg(feature = "rocm")]
    gpu_output: Option<crate::nlu::hip::GpuLinear>,
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("qwen2.context_length")?.to_u32()? as usize;
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Optional AMD GPU path (ADR 068): when `AKH_GPU_MATMUL=1`, linear-layer weights are
        // dequantized and uploaded to the GPU; their GEMMs run on the gfx1100 via rocBLAS.
        #[cfg(feature = "rocm")]
        let gpu_blas: Option<std::sync::Arc<crate::nlu::hip::RocmBlas>> =
            if std::env::var("AKH_GPU_MATMUL").ok().as_deref() == Some("1") {
                match crate::nlu::hip::RocmBlas::load() {
                    Ok(b) => {
                        tracing::info!(
                            "AKH_GPU_MATMUL=1: routing linear-layer matmuls through the AMD GPU"
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
        // int4 quantized resident weights (AKH_GPU_QUANT=1) — required to fit 12B/27B in 24 GB.
        #[cfg(feature = "rocm")]
        let gpu_quant = std::env::var("AKH_GPU_QUANT").ok().as_deref() == Some("1");

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = Qwen2RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output_qt = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => v,
            // tie_word_embeddings: reuse the token embedding matrix as the lm_head.
            _ => ct.tensor(reader, "token_embd.weight", device)?,
        };
        #[cfg(feature = "rocm")]
        let gpu_output: Option<crate::nlu::hip::GpuLinear> = match &gpu_blas {
            Some(blas) => Some(
                crate::nlu::hip::GpuLinear::from_qtensor(blas, &output_qt, device, gpu_quant)
                    .map_err(|e| candle_core::Error::Msg(format!("gpu lm_head: {e}")))?,
            ),
            None => None,
        };
        let output = QMatMul::from_qtensor(output_qt)?;

        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp = {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                // Build the GPU MLP first (borrows the qtensors) when enabled. Prefers the native
                // Q4_K path (weights used as shipped); falls back to int4/f32 per projection.
                #[cfg(feature = "rocm")]
                let gpu = match &gpu_blas {
                    Some(blas) => Some(
                        crate::nlu::hip::GpuMlp::from_qtensors(
                            blas,
                            &feed_forward_w1,
                            &feed_forward_w3,
                            &feed_forward_w2,
                            device,
                            gpu_quant,
                        )
                        .map_err(|e| candle_core::Error::Msg(format!("gpu mlp build: {e}")))?,
                    ),
                    None => None,
                };
                Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    #[cfg(feature = "rocm")]
                    gpu,
                }
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            // Build GPU attention projections (borrows the qtensors) when enabled. Prefers the
            // native Q4_K path; falls back to int4/f32 per projection.
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

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                #[cfg(feature = "rocm")]
                gpu_attn,
                attention_norm: Qwen2RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mlp,
                ffn_norm: Qwen2RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            #[cfg(feature = "rocm")]
            gpu_output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    /// Run the transformer stack and return the post-norm hidden state of the
    /// **last position** (shape `[batch, hidden_dim]`), before the output
    /// (lm_head) projection. This is the representation the Neural→VSA bridge
    /// consumes. (akh-medu vendored extension.)
    pub fn hidden_states(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        x.i((.., seq_len - 1, ..))
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
