//! Reference Qwen3.5/Qwen3.8 hybrid decoder for quantized GGUF artifacts.
//!
//! This module deliberately starts at the artifact seam: metadata is converted
//! into a strongly typed configuration and the complete target tensor schema is
//! validated before any multi-gigabyte weight is read.

use std::collections::HashMap;
use std::sync::Arc;

use candle::{
    D, DType, Device, IndexOp, Module, Result, Tensor,
    quantized::{QMatMul, QTensor, gguf_file},
};
use candle_core as candle;
use candle_transformers::utils::repeat_kv;

/// One already-normalized Gated DeltaNet recurrence step.
///
/// Shapes are `[batch, value_heads, width]` for `q`, `k`, and `v`,
/// `[batch, value_heads, 1]` for `decay` and `beta`, and
/// `[batch, value_heads, value_width, key_width]` for `state`.
fn gated_delta_step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "hip")]
    if state.device().is_cuda() && std::env::var("AKH_HIP_QWEN35_FUSED_DELTA").as_deref() == Ok("1")
    {
        return candle_core::cuda::gated_delta_step(
            &state.contiguous()?,
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &decay.contiguous()?,
            &beta.contiguous()?,
        );
    }
    gated_delta_step_reference(state, q, k, v, decay, beta)
}

fn gated_delta_step_reference(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let key_width = k.dim(candle_core::D::Minus1)?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let state = state
        .broadcast_mul(&decay.unsqueeze(candle_core::D::Minus1)?)?
        .contiguous()?;
    let prediction = state
        .matmul(&k.unsqueeze(candle_core::D::Minus1)?)?
        .squeeze(candle_core::D::Minus1)?;
    let update = beta.broadcast_mul(&(v - prediction)?)?;
    let state = (state
        + update
            .unsqueeze(candle_core::D::Minus1)?
            .matmul(&k.unsqueeze(candle_core::D::Minus2)?)?)?;
    let output = (state
        .matmul(&q.unsqueeze(candle_core::D::Minus1)?)?
        .squeeze(candle_core::D::Minus1)?
        / (key_width as f64).sqrt())?;
    Ok((output, state))
}

fn l2_normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
}

fn repeat_tiled_heads(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    // Qwen35 GGUF conversion reorders every V-side projection from grouped
    // `[G0v0, G0v1, G1v0, G1v1]` to tiled `[G0v0, G1v0, G0v1, G1v1]`.
    // Match llama.cpp's tiled ggml_repeat for Q/K rather than HF's original
    // repeat_interleave ordering.
    Tensor::cat(&vec![x; repeats], 2)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // The exact checkpoint's alpha projection stays in a range where this
    // direct definition is stable. Keeping it decomposed also makes the CPU
    // reference independent of a backend-specific fused activation.
    (x.exp()? + 1.0)?.log()
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    // `candle_nn::ops::sigmoid` uses a custom op without a HIP implementation.
    // The primitive identity stays on-device and is the same decomposition
    // Candle's own backward pass uses.
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// Lossy KV settings, read once from the environment.
///
/// `AKH_QWEN35_KV_QUANT_BITS` enables it at that bit-width;
/// `AKH_QWEN35_KV_QUANT_RECENT` keeps that many newest positions exact, since a
/// deployed compressor would page out only older context. Absent, nothing
/// changes and the bit-exact path is untouched.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvQuantization {
    bits: u8,
    recent: usize,
}

fn kv_quantization() -> Option<KvQuantization> {
    use std::sync::OnceLock;
    static SETTINGS: OnceLock<Option<KvQuantization>> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let bits: u8 = std::env::var("AKH_QWEN35_KV_QUANT_BITS")
            .ok()?
            .parse()
            .ok()
            .filter(|bits| (1..=8).contains(bits))?;
        let recent = std::env::var("AKH_QWEN35_KV_QUANT_RECENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        tracing::warn!(bits, recent, "lossy KV quantization is enabled");
        Some(KvQuantization { bits, recent })
    })
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if x.device().is_vulkan() {
        // Reference Vulkan path is entirely device-resident primitives. The
        // upstream candle-nn custom op has no Vulkan implementation.
        return candle_nn::ops::softmax(x, D::Minus1);
    }
    if x.device().is_cuda() {
        candle_nn::ops::softmax(x, D::Minus1)
    } else {
        candle_nn::ops::softmax_last_dim(x)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let eps = self.eps as f32;
        #[cfg(feature = "vulkan")]
        if x.device().is_vulkan() {
            if std::env::var("AKH_VULKAN_QWEN35_FUSED_RMS").as_deref() != Ok("0") {
                return candle_core::vulkan_backend::rms_norm_fused(x, &self.weight, eps);
            }
            return candle_nn::ops::rms_norm_slow(x, &self.weight, eps);
        }
        if x.device().is_cuda() {
            #[cfg(feature = "hip")]
            return candle::cuda::rms_norm_fused(x, &self.weight, eps);
            #[cfg(not(feature = "hip"))]
            return candle_nn::ops::rms_norm_slow(x, &self.weight, eps);
        }
        candle_nn::ops::rms_norm(x, &self.weight, eps)
    }
}

#[derive(Debug)]
enum TokenEmbedding {
    Dense(candle_nn::Embedding),
    Quantized(QTensor),
}

impl TokenEmbedding {
    fn from_qtensor(weight: QTensor, hidden: usize, device: &Device) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        let quantized_rows = device.is_cuda() || device.is_vulkan();
        #[cfg(not(feature = "vulkan"))]
        let quantized_rows = device.is_cuda();
        if quantized_rows {
            Ok(Self::Quantized(weight))
        } else {
            Ok(Self::Dense(candle_nn::Embedding::new(
                weight.dequantize(device)?,
                hidden,
            )))
        }
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => embedding.forward(tokens),
            Self::Quantized(weight) => weight.get_rows(tokens),
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct PartialRope {
    cos: Tensor,
    sin: Tensor,
    width: usize,
}

impl PartialRope {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..config.rope_dim)
            .step_by(2)
            .map(|i| {
                1f32 / config
                    .rope_freq_base
                    .powf(i as f32 / config.rope_dim as f32)
            })
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let positions = Tensor::arange(0, config.context_length as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.context_length, 1))?;
        let angles = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
            width: config.rope_dim,
        })
    }

    fn apply(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let head_dim = x.dim(3)?;
        let rotated = x.narrow(3, 0, self.width)?.contiguous()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        #[cfg(feature = "vulkan")]
        let primitive_rope = x.device().is_cuda() || x.device().is_vulkan();
        #[cfg(not(feature = "vulkan"))]
        let primitive_rope = x.device().is_cuda();
        let rotated = if primitive_rope {
            candle_nn::rotary_emb::rope_slow(&rotated, &cos, &sin)?
        } else {
            candle_nn::rotary_emb::rope(&rotated, &cos, &sin)?
        };
        if self.width == head_dim {
            Ok(rotated)
        } else {
            Tensor::cat(
                &[&rotated, &x.narrow(3, self.width, head_dim - self.width)?],
                3,
            )
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionState {
    /// Newest positions, exact.
    key: Tensor,
    value: Tensor,
    /// Older positions, compressed. One store per (batch, head) slot so each
    /// slot's positions stay in order across evictions.
    pages: Option<KvPages>,
}

/// Compressed older positions for one attention layer.
#[derive(Debug, Clone)]
struct KvPages {
    key: Vec<crate::nlu::turbo_quant::KvPageStore>,
    value: Vec<crate::nlu::turbo_quant::KvPageStore>,
    heads: usize,
    dim: usize,
}

impl KvPages {
    fn new(policy: KvQuantization, slots: usize, heads: usize, dim: usize) -> Self {
        // One prepared quantizer shared by every slot, and by both tensors.
        let quantizer = std::sync::Arc::new(crate::nlu::turbo_quant::TurboQuant::new(
            dim,
            policy.bits,
            KV_PAGE_SEED,
        ));
        let make = || {
            (0..slots)
                .map(|_| {
                    crate::nlu::turbo_quant::KvPageStore::with_quantizer(quantizer.clone())
                })
                .collect()
        };
        Self {
            key: make(),
            value: make(),
            heads,
            dim,
        }
    }

    fn positions(&self) -> usize {
        self.key.first().map(|store| store.positions()).unwrap_or(0)
    }

    /// Bytes held against the f32 equivalent, summed over every slot.
    fn stored_and_uncompressed_bytes(&self) -> (usize, usize) {
        self.key
            .iter()
            .chain(&self.value)
            .fold((0, 0), |(stored, plain), store| {
                (
                    stored + store.stored_bytes(),
                    plain + store.uncompressed_bytes(),
                )
            })
    }

    /// Append the oldest `count` positions of `tensor` into the given stores.
    fn absorb(
        stores: &mut [crate::nlu::turbo_quant::KvPageStore],
        tensor: &Tensor,
        count: usize,
    ) -> Result<()> {
        let (batch, heads, positions, dim) = tensor.dims4()?;
        let values: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for b in 0..batch {
            for head in 0..heads {
                let slot = b * heads + head;
                let base = (b * heads + head) * positions * dim;
                let rows = &values[base..base + count * dim];
                stores[slot]
                    .push_rows(rows)
                    .map_err(candle_core::Error::msg)?;
            }
        }
        Ok(())
    }

    /// Rebuild `(batch, heads, positions, dim)` from the stores.
    fn decode(
        stores: &[crate::nlu::turbo_quant::KvPageStore],
        heads: usize,
        dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let positions = stores[0].positions();
        let batch = stores.len() / heads;
        let mut values = Vec::with_capacity(batch * heads * positions * dim);
        for store in stores {
            values.extend_from_slice(&store.decode());
        }
        Tensor::from_vec(values, (batch, heads, positions, dim), device)
    }
}

/// Fixed seed so a rotation is identical across every layer and restart.
const KV_PAGE_SEED: u64 = 0x7EED_5EED;

#[derive(Debug, Clone)]
struct FullAttention {
    query_gate: QMatMul,
    key: QMatMul,
    value: QMatMul,
    output: QMatMul,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    rope: PartialRope,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    neg_inf: Tensor,
}

impl FullAttention {
    fn forward(
        &self,
        state: &mut Option<AttentionState>,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let query_gate = self.query_gate.forward(x)?.reshape((
            batch,
            seq_len,
            self.head_count,
            2,
            self.head_dim,
        ))?;
        let query = query_gate.narrow(3, 0, 1)?.squeeze(3)?.contiguous()?;
        let gate = query_gate.narrow(3, 1, 1)?.squeeze(3)?.contiguous()?;
        let query = self
            .query_norm
            .forward(&query)?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .key_norm
            .forward(&self.key.forward(x)?.reshape((
                batch,
                seq_len,
                self.kv_head_count,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self
            .value
            .forward(x)?
            .reshape((batch, seq_len, self.kv_head_count, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = self.rope.apply(&query, index_pos)?.contiguous()?;
        let key = self.rope.apply(&key, index_pos)?.contiguous()?;

        let (key, value) = match state.as_ref() {
            Some(state) if index_pos > 0 => {
                if state.key.dim(2)? != index_pos {
                    candle_core::bail!(
                        "Qwen35 attention state is at position {}, cannot append at {index_pos}",
                        state.key.dim(2)?
                    )
                }
                (
                    Tensor::cat(&[&state.key, &key], 2)?,
                    Tensor::cat(&[&state.value, &value], 2)?,
                )
            }
            None if index_pos > 0 => candle_core::bail!(
                "Qwen35 attention has no state to append at position {index_pos}"
            ),
            _ => (key, value),
        };
        // Optional compressed KV, off unless configured. Positions older than
        // the exact window move into bit-packed pages; the recent window and
        // the current token stay f32, so a decode step never compresses what it
        // just wrote.
        let (key, value) = match kv_quantization() {
            None => {
                *state = Some(AttentionState {
                    key: key.clone(),
                    value: value.clone(),
                    pages: None,
                });
                (key, value)
            }
            Some(policy) => {
                let (batch, heads, positions, dim) = key.dims4()?;
                let mut pages = state
                    .take()
                    .and_then(|previous| previous.pages)
                    .unwrap_or_else(|| KvPages::new(policy, batch * heads, heads, dim));
                let overflow = positions.saturating_sub(policy.recent);
                if overflow > 0 {
                    KvPages::absorb(&mut pages.key, &key, overflow)?;
                    KvPages::absorb(&mut pages.value, &value, overflow)?;
                }
                let recent_key = key.narrow(2, overflow, positions - overflow)?.contiguous()?;
                let recent_value =
                    value.narrow(2, overflow, positions - overflow)?.contiguous()?;
                let device = key.device().clone();
                let full = if pages.positions() > 0 {
                    let old_key = KvPages::decode(&pages.key, heads, dim, &device)?
                        .to_dtype(key.dtype())?;
                    let old_value = KvPages::decode(&pages.value, heads, dim, &device)?
                        .to_dtype(value.dtype())?;
                    (
                        Tensor::cat(&[&old_key, &recent_key], 2)?,
                        Tensor::cat(&[&old_value, &recent_value], 2)?,
                    )
                } else {
                    (recent_key.clone(), recent_value.clone())
                };
                *state = Some(AttentionState {
                    key: recent_key,
                    value: recent_value,
                    pages: Some(pages),
                });
                full
            }
        };

        let key = repeat_kv(key, self.head_count / self.kv_head_count)?;
        let value = repeat_kv(value, self.head_count / self.kv_head_count)?;
        let scores = (query.matmul(&key.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = match mask {
            Some(mask) => {
                let mask = mask.broadcast_as(scores.shape())?;
                masked_fill(&scores, &mask, &self.neg_inf)?
            }
            None => scores,
        };
        let attended = softmax_last_dim(&scores)?.matmul(&value.contiguous()?)?;
        let attended = attended.transpose(1, 2)?;
        let attended = attended.broadcast_mul(&sigmoid(&gate)?)?;
        self.output
            .forward(&attended.reshape((batch, seq_len, self.head_count * self.head_dim))?)
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaState {
    conv_history: Tensor,
    recurrent: Tensor,
    position: usize,
}

#[derive(Debug, Clone)]
struct GatedDelta {
    qkv: QMatMul,
    z: QMatMul,
    alpha: QMatMul,
    beta: QMatMul,
    conv: Tensor,
    dt_bias: Tensor,
    a: Tensor,
    norm: RmsNorm,
    output: QMatMul,
    config: Config,
}

fn causal_depthwise_conv(
    input: &Tensor,
    kernel: &Tensor,
    history: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "vulkan")]
    if input.device().is_vulkan()
        && std::env::var("AKH_VULKAN_QWEN35_FUSED_CONV").as_deref() == Ok("1")
    {
        return candle_core::vulkan_backend::causal_conv_fused(input, kernel, history);
    }
    let (batch, seq_len, channels) = input.dims3()?;
    let (kernel_channels, kernel_width) = kernel.dims2()?;
    if channels != kernel_channels || kernel_width == 0 {
        candle_core::bail!(
            "Qwen35 causal-conv shape mismatch: input channels {channels}, kernel {:?}",
            kernel.shape()
        )
    }
    let history = match history {
        Some(history) => history.clone(),
        None => Tensor::zeros(
            (batch, kernel_width - 1, channels),
            DType::F32,
            input.device(),
        )?,
    };
    let extended = Tensor::cat(&[&history, input], 1)?;
    let mut output: Option<Tensor> = None;
    for offset in 0..kernel_width {
        let window = extended.narrow(1, offset, seq_len)?;
        let weight = kernel.narrow(1, offset, 1)?.squeeze(1)?;
        let term = window.broadcast_mul(&weight)?;
        output = Some(match output {
            Some(sum) => (sum + term)?,
            None => term,
        });
    }
    let next_history = extended.narrow(1, seq_len, kernel_width - 1)?;
    Ok((output.expect("non-empty kernel width"), next_history))
}

impl GatedDelta {
    fn forward(
        &self,
        state: &mut Option<GatedDeltaState>,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        if index_pos > 0 && state.is_none() {
            candle_core::bail!(
                "Qwen35 recurrent mixer has no state to append at position {index_pos}"
            )
        }
        if let Some(state) = state.as_ref() {
            if index_pos > 0 && state.position != index_pos {
                candle_core::bail!(
                    "Qwen35 recurrent state is at position {}, cannot append at {index_pos}",
                    state.position
                )
            }
        }
        let prior = if index_pos == 0 { None } else { state.as_ref() };
        let qkv = self.qkv.forward(x)?;
        let (qkv, conv_history) =
            causal_depthwise_conv(&qkv, &self.conv, prior.map(|state| &state.conv_history))?;
        let qkv = candle_nn::ops::silu(&qkv)?;
        let key_width = self.config.group_count * self.config.state_size;
        let value_width = self.config.inner_size;
        let q = qkv.narrow(2, 0, key_width)?.reshape((
            batch,
            seq_len,
            self.config.group_count,
            self.config.state_size,
        ))?;
        let k = qkv.narrow(2, key_width, key_width)?.reshape((
            batch,
            seq_len,
            self.config.group_count,
            self.config.state_size,
        ))?;
        let v = qkv.narrow(2, key_width * 2, value_width)?.reshape((
            batch,
            seq_len,
            self.config.value_head_count,
            self.config.state_size,
        ))?;
        let repeats = self.config.value_head_count / self.config.group_count;
        let q = repeat_tiled_heads(&l2_normalize(&q, self.config.rms_norm_eps)?, repeats)?;
        let k = repeat_tiled_heads(&l2_normalize(&k, self.config.rms_norm_eps)?, repeats)?;

        let alpha = self.alpha.forward(x)?.broadcast_add(&self.dt_bias)?;
        let decay = softplus(&alpha)?.broadcast_mul(&self.a)?.exp()?;
        let beta = sigmoid(&self.beta.forward(x)?)?;
        let mut recurrent = match prior {
            Some(state) => state.recurrent.clone(),
            None => Tensor::zeros(
                (
                    batch,
                    self.config.value_head_count,
                    self.config.state_size,
                    self.config.state_size,
                ),
                DType::F32,
                x.device(),
            )?,
        };
        let mut outputs = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let q = q.narrow(1, token, 1)?.squeeze(1)?;
            let k = k.narrow(1, token, 1)?.squeeze(1)?;
            let v = v.narrow(1, token, 1)?.squeeze(1)?;
            let decay = decay.narrow(1, token, 1)?.squeeze(1)?.unsqueeze(2)?;
            let beta = beta.narrow(1, token, 1)?.squeeze(1)?.unsqueeze(2)?;
            let (output, next) = gated_delta_step(&recurrent, &q, &k, &v, &decay, &beta)?;
            outputs.push(output.unsqueeze(1)?);
            recurrent = next;
        }
        *state = Some(GatedDeltaState {
            conv_history,
            recurrent,
            position: index_pos + seq_len,
        });

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?;
        let z = self.z.forward(x)?.reshape((
            batch,
            seq_len,
            self.config.value_head_count,
            self.config.state_size,
        ))?;
        let output = self
            .norm
            .forward(&output)?
            .broadcast_mul(&candle_nn::ops::silu(&z)?)?
            .reshape((batch, seq_len, self.config.inner_size))?;
        self.output.forward(&output)
    }
}

#[derive(Debug, Clone)]
enum Mixer {
    Attention(FullAttention),
    GatedDelta(GatedDelta),
}

#[derive(Debug, Clone)]
enum MixerState {
    Attention(Option<AttentionState>),
    GatedDelta(Option<GatedDeltaState>),
}

impl Mixer {
    fn forward(
        &self,
        state: &mut MixerState,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        match (self, state) {
            (Self::Attention(attention), MixerState::Attention(state)) => {
                attention.forward(state, x, mask, index_pos)
            }
            (Self::GatedDelta(gated_delta), MixerState::GatedDelta(state)) => {
                gated_delta.forward(state, x, index_pos)
            }
            _ => candle_core::bail!("Qwen35 session layer kind does not match weights"),
        }
    }

    fn empty_state(&self) -> MixerState {
        match self {
            Self::Attention(_) => MixerState::Attention(None),
            Self::GatedDelta(_) => MixerState::GatedDelta(None),
        }
    }
}

#[derive(Debug, Clone)]
struct Layer {
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    mixer: Mixer,
    mlp: Mlp,
}

impl Layer {
    fn forward(
        &self,
        state: &mut MixerState,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let normalized = self.input_norm.forward(x)?;
        let mixed = self.mixer.forward(state, &normalized, mask, index_pos)?;
        let x = (residual + mixed)?;
        let mlp = self.mlp.forward(&self.post_attention_norm.forward(&x)?)?;
        x + mlp
    }
}

/// Snapshot of one session's mutable target-model state. Immutable weights are
/// deliberately absent. Sibling sessions cannot restore each other's snapshots.
#[derive(Debug, Clone)]
pub(crate) struct Checkpoint {
    owner: Arc<()>,
    layers: Vec<MixerState>,
    text: Option<TextState>,
}

#[derive(Debug, Clone)]
struct TextState {
    /// Only tokens that have reached forward, never a merely sampled token.
    tokens: Vec<u32>,
    logits: Tensor,
}

/// Device-resident mean of a selected span's post-block residual stream.
pub(crate) struct PooledTap {
    pub(crate) layer: usize,
    pub(crate) mean: Tensor,
}

pub(crate) struct PromptCapture {
    pub(crate) logits: Tensor,
    pub(crate) reused_tokens: usize,
    /// None means the requested span was not completely newly forwarded.
    pub(crate) taps: Option<Vec<PooledTap>>,
}

/// Operation-local accumulation: errors drop every partial device vector.
struct TapCollector {
    span: std::ops::Range<usize>,
    layers: Vec<(usize, Option<Tensor>, usize)>,
}

impl TapCollector {
    fn observe(&mut self, layer: usize, hidden: &Tensor, position: usize) -> Result<()> {
        let Some((_, sum, count)) = self.layers.iter_mut().find(|entry| entry.0 == layer) else {
            return Ok(());
        };
        let (batch, length, _) = hidden.dims3()?;
        if batch != 1 {
            candle_core::bail!("Qwen35 span capture requires one session row");
        }
        let start = self.span.start.max(position);
        let end = self.span.end.min(position + length);
        if start >= end {
            return Ok(());
        }
        let chunk = hidden
            .narrow(1, start - position, end - start)?
            .to_dtype(DType::F32)?
            .sum(1)?
            .squeeze(0)?;
        *sum = Some(match sum.as_ref() {
            Some(previous) => (previous + &chunk)?,
            None => chunk,
        });
        *count += end - start;
        Ok(())
    }

    fn finish(self) -> Result<Option<Vec<PooledTap>>> {
        let expected = self.span.end - self.span.start;
        if self.layers.iter().any(|(_, _, count)| *count != expected) {
            return Ok(None);
        }
        self.layers
            .into_iter()
            .map(|(layer, sum, count)| {
                Ok(PooledTap {
                    layer,
                    mean: (sum.expect("complete nonempty span") / count as f64)?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
}

struct Weights {
    embedding: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    context_length: usize,
    /// Residual stream width; pooled taps have exactly this many values.
    hidden: usize,
}

/// One execution path with immutable shared weights and exclusively owned
/// position-dependent state. Sessions still require the backend's compute lane;
/// sharing weights does not authorize concurrent HIP execution.
pub(crate) struct InferenceSession {
    weights: Arc<Weights>,
    states: Vec<MixerState>,
    mask_cache: Option<((usize, usize), Tensor)>,
    checkpoint_owner: Arc<()>,
    text: Option<TextState>,
}

/// Compatibility name used by Candle's family dispatch; not a second runner.
pub(crate) type ModelWeights = InferenceSession;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Config {
    pub(crate) total_block_count: usize,
    pub(crate) block_count: usize,
    pub(crate) nextn_block_count: usize,
    pub(crate) context_length: usize,
    pub(crate) embedding_length: usize,
    pub(crate) feed_forward_length: usize,
    pub(crate) head_count: usize,
    pub(crate) head_count_kv: usize,
    pub(crate) head_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) rope_freq_base: f32,
    pub(crate) rms_norm_eps: f64,
    pub(crate) full_attention_interval: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) state_size: usize,
    pub(crate) group_count: usize,
    pub(crate) value_head_count: usize,
    pub(crate) inner_size: usize,
}

impl Config {
    pub(crate) fn from_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Result<Self> {
        let value = |key: &str| {
            metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata `{key}`")))
        };
        let usize_value = |key: &str| -> Result<usize> {
            value(key)?.to_u64().map(|v| v as usize).map_err(|error| {
                candle_core::Error::Msg(format!("invalid integer GGUF metadata `{key}`: {error}"))
            })
        };
        let f32_value = |key: &str| -> Result<f32> {
            value(key)?.to_f32().map_err(|error| {
                candle_core::Error::Msg(format!("invalid float GGUF metadata `{key}`: {error}"))
            })
        };

        let total_block_count = usize_value("qwen35.block_count")?;
        let nextn_block_count = usize_value("qwen35.nextn_predict_layers")?;
        let block_count = total_block_count.checked_sub(nextn_block_count).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "qwen35.nextn_predict_layers ({nextn_block_count}) exceeds block_count ({total_block_count})"
            ))
        })?;
        let head_dim = usize_value("qwen35.attention.key_length")?;
        let value_dim = usize_value("qwen35.attention.value_length")?;
        let rope_dim = usize_value("qwen35.rope.dimension_count")?;
        let full_attention_interval = usize_value("qwen35.full_attention_interval")?;
        let state_size = usize_value("qwen35.ssm.state_size")?;
        let value_head_count = usize_value("qwen35.ssm.time_step_rank")?;
        let inner_size = usize_value("qwen35.ssm.inner_size")?;
        let head_count = usize_value("qwen35.attention.head_count")?;
        let head_count_kv = usize_value("qwen35.attention.head_count_kv")?;
        let group_count = usize_value("qwen35.ssm.group_count")?;

        if block_count == 0 || nextn_block_count > 1 {
            candle_core::bail!(
                "unsupported Qwen35 target/MTP split: {block_count} target blocks and {nextn_block_count} MTP blocks"
            )
        }
        if full_attention_interval == 0 || block_count % full_attention_interval != 0 {
            candle_core::bail!(
                "invalid Qwen35 full-attention interval {full_attention_interval} for {block_count} target blocks"
            )
        }
        if head_count_kv == 0 || head_count % head_count_kv != 0 {
            candle_core::bail!(
                "Qwen35 query head count {head_count} is not divisible by KV head count {head_count_kv}"
            )
        }
        if head_dim != value_dim {
            candle_core::bail!("Qwen35 key/value head widths differ ({head_dim} vs {value_dim})")
        }
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) || rope_dim > head_dim {
            candle_core::bail!(
                "invalid Qwen35 rotary width {rope_dim} for attention head width {head_dim}"
            )
        }
        if value_head_count == 0
            || group_count == 0
            || !value_head_count.is_multiple_of(group_count)
            || !inner_size.is_multiple_of(value_head_count)
            || inner_size / value_head_count != state_size
        {
            candle_core::bail!(
                "inconsistent Qwen35 Gated DeltaNet geometry: inner={inner_size}, value_heads={value_head_count}, groups={group_count}, state={state_size}"
            )
        }

        Ok(Self {
            total_block_count,
            block_count,
            nextn_block_count,
            context_length: usize_value("qwen35.context_length")?,
            embedding_length: usize_value("qwen35.embedding_length")?,
            feed_forward_length: usize_value("qwen35.feed_forward_length")?,
            head_count,
            head_count_kv,
            head_dim,
            rope_dim,
            rope_freq_base: f32_value("qwen35.rope.freq_base")?,
            rms_norm_eps: f32_value("qwen35.attention.layer_norm_rms_epsilon")? as f64,
            full_attention_interval,
            conv_kernel: usize_value("qwen35.ssm.conv_kernel")?,
            state_size,
            group_count,
            value_head_count,
            inner_size,
        })
    }

    pub(crate) fn is_recurrent(&self, layer: usize) -> bool {
        layer < self.block_count && (layer + 1) % self.full_attention_interval != 0
    }
}

fn validate_tensor_schema(content: &gguf_file::Content, config: &Config) -> Result<usize> {
    let tensor_shape = |name: &str| -> Result<&[usize]> {
        content
            .tensor_infos
            .get(name)
            .map(|info| info.shape.dims())
            .ok_or_else(|| candle_core::Error::Msg(format!("missing Qwen35 tensor `{name}`")))
    };
    let expect = |name: &str, expected: &[usize]| -> Result<()> {
        let actual = tensor_shape(name)?;
        if actual != expected {
            candle_core::bail!("Qwen35 tensor `{name}` has shape {actual:?}, expected {expected:?}")
        }
        Ok(())
    };

    let embedding = tensor_shape("token_embd.weight")?;
    if embedding.len() != 2 || embedding[1] != config.embedding_length {
        candle_core::bail!(
            "Qwen35 tensor `token_embd.weight` has shape {embedding:?}, expected [vocab, {}]",
            config.embedding_length
        )
    }
    let vocabulary = embedding[0];
    expect("output_norm.weight", &[config.embedding_length])?;
    if content.tensor_infos.contains_key("output.weight") {
        expect("output.weight", &[vocabulary, config.embedding_length])?;
    }

    let q_width = config.head_count * config.head_dim * 2;
    let kv_width = config.head_count_kv * config.head_dim;
    let attention_width = config.head_count * config.head_dim;
    let recurrent_qk_width = config.group_count * config.state_size;
    let recurrent_width = recurrent_qk_width * 2 + config.inner_size;
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        expect(
            &format!("{prefix}.attn_norm.weight"),
            &[config.embedding_length],
        )?;
        expect(
            &format!("{prefix}.post_attention_norm.weight"),
            &[config.embedding_length],
        )?;
        expect(
            &format!("{prefix}.ffn_gate.weight"),
            &[config.feed_forward_length, config.embedding_length],
        )?;
        expect(
            &format!("{prefix}.ffn_up.weight"),
            &[config.feed_forward_length, config.embedding_length],
        )?;
        expect(
            &format!("{prefix}.ffn_down.weight"),
            &[config.embedding_length, config.feed_forward_length],
        )?;
        if config.is_recurrent(layer) {
            expect(
                &format!("{prefix}.attn_qkv.weight"),
                &[recurrent_width, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.attn_gate.weight"),
                &[config.inner_size, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.ssm_alpha.weight"),
                &[config.value_head_count, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.ssm_beta.weight"),
                &[config.value_head_count, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.ssm_conv1d.weight"),
                &[recurrent_width, config.conv_kernel],
            )?;
            expect(&format!("{prefix}.ssm_dt.bias"), &[config.value_head_count])?;
            expect(&format!("{prefix}.ssm_a"), &[config.value_head_count])?;
            expect(&format!("{prefix}.ssm_norm.weight"), &[config.state_size])?;
            expect(
                &format!("{prefix}.ssm_out.weight"),
                &[config.embedding_length, config.inner_size],
            )?;
        } else {
            expect(
                &format!("{prefix}.attn_q.weight"),
                &[q_width, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.attn_k.weight"),
                &[kv_width, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.attn_v.weight"),
                &[kv_width, config.embedding_length],
            )?;
            expect(
                &format!("{prefix}.attn_output.weight"),
                &[config.embedding_length, attention_width],
            )?;
            expect(&format!("{prefix}.attn_q_norm.weight"), &[config.head_dim])?;
            expect(&format!("{prefix}.attn_k_norm.weight"), &[config.head_dim])?;
        }
    }

    // MTP is deliberately not executed in the target-only reference path, but
    // its defining tensors are part of the exact artifact contract and are
    // checked before weight allocation so the later speculative slice can rely
    // on a known companion head rather than silently substituting one.
    if config.nextn_block_count == 1 {
        let prefix = format!("blk.{}", config.block_count);
        for (suffix, shape) in [
            ("attn_norm.weight", vec![config.embedding_length]),
            ("post_attention_norm.weight", vec![config.embedding_length]),
            ("attn_q.weight", vec![q_width, config.embedding_length]),
            ("attn_k.weight", vec![kv_width, config.embedding_length]),
            ("attn_v.weight", vec![kv_width, config.embedding_length]),
            (
                "attn_output.weight",
                vec![config.embedding_length, attention_width],
            ),
            ("attn_q_norm.weight", vec![config.head_dim]),
            ("attn_k_norm.weight", vec![config.head_dim]),
            (
                "ffn_gate.weight",
                vec![config.feed_forward_length, config.embedding_length],
            ),
            (
                "ffn_up.weight",
                vec![config.feed_forward_length, config.embedding_length],
            ),
            (
                "ffn_down.weight",
                vec![config.embedding_length, config.feed_forward_length],
            ),
            (
                "nextn.eh_proj.weight",
                vec![config.embedding_length, config.embedding_length * 2],
            ),
            ("nextn.enorm.weight", vec![config.embedding_length]),
            ("nextn.hnorm.weight", vec![config.embedding_length]),
        ] {
            expect(&format!("{prefix}.{suffix}"), &shape)?;
        }
    }
    Ok(vocabulary)
}

fn load_layer<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    config: &Config,
    rope: &PartialRope,
    neg_inf: &Tensor,
    layer_index: usize,
    device: &Device,
) -> Result<Layer> {
    let prefix = format!("blk.{layer_index}");
    let input_norm = RmsNorm::from_qtensor(
        content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
        config.rms_norm_eps,
    )?;
    let post_attention_norm = RmsNorm::from_qtensor(
        content.tensor(
            reader,
            &format!("{prefix}.post_attention_norm.weight"),
            device,
        )?,
        config.rms_norm_eps,
    )?;
    let mlp = Mlp {
        gate: QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{prefix}.ffn_gate.weight"),
            device,
        )?)?,
        up: QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{prefix}.ffn_up.weight"),
            device,
        )?)?,
        down: QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{prefix}.ffn_down.weight"),
            device,
        )?)?,
    };
    let mixer = if config.is_recurrent(layer_index) {
        Mixer::GatedDelta(GatedDelta {
            qkv: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_qkv.weight"),
                device,
            )?)?,
            z: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_gate.weight"),
                device,
            )?)?,
            alpha: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.ssm_alpha.weight"),
                device,
            )?)?,
            beta: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.ssm_beta.weight"),
                device,
            )?)?,
            conv: content
                .tensor(reader, &format!("{prefix}.ssm_conv1d.weight"), device)?
                .dequantize(device)?,
            dt_bias: content
                .tensor(reader, &format!("{prefix}.ssm_dt.bias"), device)?
                .dequantize(device)?,
            a: content
                .tensor(reader, &format!("{prefix}.ssm_a"), device)?
                .dequantize(device)?,
            norm: RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.ssm_norm.weight"), device)?,
                config.rms_norm_eps,
            )?,
            output: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.ssm_out.weight"),
                device,
            )?)?,
            config: config.clone(),
        })
    } else {
        Mixer::Attention(FullAttention {
            query_gate: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_q.weight"),
                device,
            )?)?,
            key: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_k.weight"),
                device,
            )?)?,
            value: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_v.weight"),
                device,
            )?)?,
            output: QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?)?,
            query_norm: RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                config.rms_norm_eps,
            )?,
            key_norm: RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                config.rms_norm_eps,
            )?,
            rope: rope.clone(),
            head_count: config.head_count,
            kv_head_count: config.head_count_kv,
            head_dim: config.head_dim,
            neg_inf: neg_inf.clone(),
        })
    };
    Ok(Layer {
        input_norm,
        post_attention_norm,
        mixer,
        mlp,
    })
}

impl InferenceSession {
    /// Resident attention key/value caches, by layer index.
    ///
    /// Read-only borrow for measurement: quantization quality has to be judged
    /// against this model's real cache contents, not synthetic vectors. Layers
    /// with recurrent mixers, and attention layers that have not run yet, are
    /// absent. Key tensors are post-RoPE; values are not. Both are shaped
    /// `(batch, kv_heads, position, head_dim)`.
    pub(crate) fn attention_states(&self) -> Vec<(usize, &Tensor, &Tensor)> {
        self.states
            .iter()
            .enumerate()
            .filter_map(|(layer, state)| match state {
                MixerState::Attention(Some(state)) => Some((layer, &state.key, &state.value)),
                _ => None,
            })
            .collect()
    }

    /// Capture a span during this operation only, without replaying cached
    /// tokens. Returned values are not archival commits or persistent snapshots.
    pub(crate) fn resume_prompt_with_taps(
        &mut self,
        tokens: &[u32],
        span: std::ops::Range<usize>,
        layers: &[usize],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PromptCapture> {
        if span.start >= span.end
            || span.end > tokens.len()
            || layers.is_empty()
            || layers.len() > 5
            || layers
                .iter()
                .any(|layer| *layer >= self.weights.layers.len())
            || layers
                .iter()
                .enumerate()
                .any(|(index, layer)| layers[..index].contains(layer))
        {
            candle_core::bail!("Qwen35 invalid tap span or layer selection");
        }
        let mut collector = TapCollector {
            span,
            layers: layers.iter().map(|layer| (*layer, None, 0)).collect(),
        };
        let result = (|| {
            let (logits, reused_tokens) =
                self.resume_prompt_collecting(tokens, check, Some(&mut collector))?;
            let taps = collector.finish()?;
            check()?;
            Ok(PromptCapture {
                logits,
                reused_tokens,
                taps,
            })
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Forward a sampled token before adding it to this session's exact prefix.
    pub(crate) fn forward_token(
        &mut self,
        token: u32,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Tensor> {
        let Some(mut text) = self.text.take() else {
            candle_core::bail!("Qwen35 tracked decode requires a prepared text prompt");
        };
        let result = (|| {
            check()?;
            let input = Tensor::new(&[[token]], self.weights.norm.weight.device())?;
            let logits = self.decode_step(&input, text.tokens.len())?;
            check()?;
            Ok(logits)
        })();
        match result {
            Ok(logits) => {
                text.tokens.push(token);
                text.logits = logits.clone();
                self.text = Some(text);
                Ok(logits)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    /// Prepare a complete authoritative text prompt, reusing only an exact
    /// prefix of tokens previously forwarded by this session.
    pub(crate) fn resume_prompt(
        &mut self,
        tokens: &[u32],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(Tensor, usize)> {
        self.resume_prompt_collecting(tokens, check, None)
    }

    fn resume_prompt_collecting(
        &mut self,
        tokens: &[u32],
        check: &mut dyn FnMut() -> Result<()>,
        collector: Option<&mut TapCollector>,
    ) -> Result<(Tensor, usize)> {
        if tokens.is_empty() || tokens.len() > self.weights.context_length {
            candle_core::bail!("Qwen35 invalid text prompt length {}", tokens.len());
        }
        if let Err(error) = check() {
            self.reset();
            return Err(error);
        }
        let reused = self
            .text
            .as_ref()
            .filter(|state| tokens.starts_with(&state.tokens))
            .map_or(0, |state| state.tokens.len());
        if reused == tokens.len() {
            return Ok((self.text.as_ref().unwrap().logits.clone(), reused));
        }
        let input =
            Tensor::new(&tokens[reused..], self.weights.norm.weight.device())?.unsqueeze(0)?;
        let collector = collector.filter(|collector| collector.span.start >= reused);
        let (logits, _) = self.prefill_at_position(&input, reused, 128, check, collector)?;
        self.text = Some(TextState {
            tokens: tokens.to_vec(),
            logits: logits.clone(),
        });
        Ok((logits, reused))
    }

    /// Start an independent empty session sharing this model's immutable weights.
    pub(crate) fn new_session(&self) -> Self {
        Self::with_weights(self.weights.clone())
    }

    /// Number of target transformer blocks (excludes the MTP block).
    /// Resident KV bytes added per additional context token.
    ///
    /// Only the attention blocks grow: `is_recurrent` puts a fixed-size state in
    /// every layer whose index is not a multiple of `full_attention_interval`,
    /// so the per-token cost scales with the attention blocks alone. Keys and
    /// values are both `head_count_kv x head_dim` f32 per token per such block.
    /// Recurrent and MTP state are constant in context length and excluded.
    pub(crate) fn kv_bytes_per_token(&self) -> usize {
        // Read the real layers rather than recomputing from metadata, so the
        // figure follows whatever this session actually loaded.
        self.weights
            .layers
            .iter()
            .filter_map(|layer| match &layer.mixer {
                Mixer::Attention(attention) => Some(attention),
                Mixer::GatedDelta(_) => None,
            })
            .map(|attention| {
                2 * attention.kv_head_count * attention.head_dim * std::mem::size_of::<f32>()
            })
            .sum()
    }

    pub(crate) fn block_count(&self) -> usize {
        self.weights.layers.len()
    }

    /// Residual stream width of every pooled tap.
    pub(crate) fn hidden_size(&self) -> usize {
        self.weights.hidden
    }

    fn with_weights(weights: Arc<Weights>) -> Self {
        let states = weights
            .layers
            .iter()
            .map(|layer| layer.mixer.empty_state())
            .collect();
        Self {
            weights,
            states,
            mask_cache: None,
            checkpoint_owner: Arc::new(()),
            text: None,
        }
    }

    /// Read and validate only the GGUF header and tensor directory.
    ///
    /// This intentionally performs no tensor allocation, so operators can
    /// verify a production artifact while another process owns the GPU.
    pub(crate) fn validate_gguf<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
    ) -> Result<Config> {
        let content = gguf_file::Content::read(reader)?;
        let config = Config::from_metadata(&content.metadata)?;
        validate_tensor_schema(&content, &config)?;
        Ok(config)
    }

    pub(crate) fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let config = Config::from_metadata(&content.metadata)?;
        validate_tensor_schema(&content, &config)?;
        let rope = PartialRope::new(&config, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let embedding_q = content.tensor(reader, "token_embd.weight", device)?;
        let embedding = TokenEmbedding::from_qtensor(embedding_q, config.embedding_length, device)?;
        let norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            config.rms_norm_eps,
        )?;
        let output = if content.tensor_infos.contains_key("output.weight") {
            QMatMul::from_qtensor(content.tensor(reader, "output.weight", device)?)?
        } else {
            QMatMul::from_qtensor(content.tensor(reader, "token_embd.weight", device)?)?
        };

        let mut layers = Vec::with_capacity(config.block_count);
        for layer_index in 0..config.block_count {
            layers.push(load_layer(
                &content,
                reader,
                &config,
                &rope,
                &neg_inf,
                layer_index,
                device,
            )?);
        }
        Ok(Self::with_weights(Arc::new(Weights {
            embedding,
            layers,
            norm,
            output,
            context_length: config.context_length,
            hidden: config.embedding_length,
        })))
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some((shape, mask)) = &self.mask_cache {
            if *shape == (seq_len, kv_len) {
                return Ok(mask.clone());
            }
        }
        let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
        self.mask_cache = Some(((seq_len, kv_len), mask.clone()));
        Ok(mask)
    }

    fn reset(&mut self) {
        self.mask_cache = None;
        self.text = None;
        for state in &mut self.states {
            match state {
                MixerState::Attention(state) => *state = None,
                MixerState::GatedDelta(state) => *state = None,
            }
        }
    }

    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            owner: self.checkpoint_owner.clone(),
            layers: self.states.clone(),
            text: self.text.clone(),
        }
    }

    pub(crate) fn rewind(&mut self, checkpoint: Checkpoint) -> Result<()> {
        if !Arc::ptr_eq(&self.checkpoint_owner, &checkpoint.owner) {
            candle_core::bail!("Qwen35 checkpoint belongs to a different inference session")
        }
        if checkpoint.layers.len() != self.weights.layers.len() {
            candle_core::bail!(
                "Qwen35 checkpoint has {} layers, model has {}",
                checkpoint.layers.len(),
                self.weights.layers.len()
            )
        }
        for (layer, state) in self.weights.layers.iter().zip(&checkpoint.layers) {
            if !matches!(
                (&layer.mixer, state),
                (Mixer::Attention(_), MixerState::Attention(_))
                    | (Mixer::GatedDelta(_), MixerState::GatedDelta(_))
            ) {
                candle_core::bail!("Qwen35 checkpoint layer kind does not match weights");
            }
        }
        self.states = checkpoint.layers;
        self.text = checkpoint.text;
        self.mask_cache = None;
        Ok(())
    }

    fn run_hidden_states(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_hidden_states_collecting(tokens, index_pos, None)
    }

    fn run_hidden_states_collecting(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        mut collector: Option<&mut TapCollector>,
    ) -> Result<Tensor> {
        let (_, seq_len) = tokens.dims2()?;
        // Tensor-based legacy/probe calls cannot claim a known host-token prefix.
        // Text entry points reinstall their log only after a successful forward.
        self.text = None;
        if index_pos == 0 {
            self.reset();
        }
        let mask = if seq_len > 1 {
            Some(self.mask(seq_len, index_pos, tokens.device())?)
        } else {
            None
        };
        let mut hidden = self.weights.embedding.forward(tokens)?;
        for (index, (layer, state)) in self.weights.layers.iter().zip(&mut self.states).enumerate()
        {
            hidden = layer.forward(state, &hidden, mask.as_ref(), index_pos)?;
            if let Some(collector) = collector.as_deref_mut() {
                collector.observe(index, &hidden, index_pos)?;
            }
        }
        self.weights.norm.forward(&hidden)?.i((.., seq_len - 1, ..))
    }

    fn run_with_hidden(&mut self, tokens: &Tensor, index_pos: usize) -> Result<(Tensor, Tensor)> {
        let hidden = self.run_hidden_states(tokens, index_pos)?;
        let logits = self.weights.output.forward(&hidden)?;
        Ok((logits, hidden))
    }

    pub(crate) fn prefill(&mut self, tokens: &Tensor) -> Result<Tensor> {
        self.prefill_with_hidden(tokens).map(|(logits, _)| logits)
    }

    pub(crate) fn decode_step(&mut self, token: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.decode_step_with_hidden(token, index_pos)
            .map(|(logits, _)| logits)
    }

    fn prefill_with_hidden(&mut self, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
        self.prefill_controlled_with_hidden(tokens, 128, &mut || Ok(()))
    }

    /// Prefill bounded query chunks, retaining exact uncompressed KV and recurrent
    /// state. The callback runs before every chunk and before the final projection.
    /// Invalid dimensions leave existing state untouched; execution errors clear
    /// partial state. Checkpoints from this loaded model remain valid for rewind.
    pub(crate) fn prefill_controlled(
        &mut self,
        tokens: &Tensor,
        chunk_size: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Tensor> {
        self.prefill_controlled_with_hidden(tokens, chunk_size, check)
            .map(|(logits, _)| logits)
    }

    fn prefill_controlled_with_hidden(
        &mut self,
        tokens: &Tensor,
        chunk_size: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(Tensor, Tensor)> {
        self.prefill_at_position(tokens, 0, chunk_size, check, None)
    }

    fn prefill_at_position(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        chunk_size: usize,
        check: &mut dyn FnMut() -> Result<()>,
        mut collector: Option<&mut TapCollector>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len) = tokens.dims2()?;
        let end = index_pos.checked_add(seq_len);
        if batch == 0
            || seq_len == 0
            || end.is_none_or(|end| end > self.weights.context_length)
            || chunk_size == 0
        {
            candle_core::bail!(
                "Qwen35 invalid prefill: batch={batch}, tokens={seq_len}, context={}, chunk={chunk_size}",
                self.weights.context_length
            )
        }
        let result = (|| {
            let mut final_hidden = None;
            for position in (0..seq_len).step_by(chunk_size) {
                check()?;
                let length = chunk_size.min(seq_len - position);
                let chunk = tokens.narrow(1, position, length)?.contiguous()?;
                final_hidden = Some(self.run_hidden_states_collecting(
                    &chunk,
                    index_pos + position,
                    collector.as_deref_mut(),
                )?);
            }
            check()?;
            let hidden = final_hidden.expect("non-empty prefill validated above");
            Ok((self.weights.output.forward(&hidden)?, hidden))
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn decode_step_with_hidden(
        &mut self,
        token: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len) = token.dims2()?;
        if seq_len != 1 || index_pos == 0 || index_pos >= self.weights.context_length {
            candle_core::bail!(
                "Qwen35 decode_step requires one token at a non-zero position, got {seq_len} tokens at {index_pos}"
            )
        }
        self.run_with_hidden(token, index_pos)
    }

    pub(crate) fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        if index_pos == 0 {
            return self.prefill(tokens);
        }
        self.decode_step(tokens, index_pos)
    }

    pub(crate) fn forward_with_hidden(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        if index_pos == 0 {
            return self.prefill_with_hidden(tokens);
        }
        self.decode_step_with_hidden(tokens, index_pos)
    }
}


/// Deliberately tiny, complete Qwen3.5 artifacts for CPU tests in this and
/// dependent crates. Mixer and MLP matrices are zero, so every post-block
/// residual equals the token embedding (token 1 is all ones, every other row
/// is zero) and the tied output head makes the argmax land on token 0.
/// This is test scaffolding, never a model.
pub mod test_fixture {
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use candle_core::{Device, Shape, Tensor};
    use std::io::Cursor;

    /// A `tokenizers` WordLevel/Whitespace tokenizer whose `<|im_end|>` is
    /// token 0 (the fixture's constant argmax) and whose unknown token is 1
    /// (the only nonzero embedding). "zero" maps to an all-zero embedding.
    pub const TINY_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<|im_end|>":0,"[UNK]":1,"zero":2},"unk_token":"[UNK]"}}"#;

    /// Tokenizer for [`tiny_gguf_replying`]: the rendered chat prompt ends in
    /// `</think>`, whose final `>` is token 4, so the model answers `ok`
    /// (token 5) and then stops with `<|im_end|>` (token 0).
    /// The chat markers are added special tokens, as in the real tokenizer,
    /// so they never merge with adjacent punctuation of the user text.
    pub const TINY_REPLYING_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":6,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<|im_end|>":0,"[UNK]":1,"zero":2,">":4,"ok":5,"<|im_start|>":6},"unk_token":"[UNK]"}}"#;

    pub fn tiny_metadata() -> Vec<(String, gguf_file::Value)> {
        use gguf_file::Value::{F32, String as GgufString, U32};

        vec![
            ("general.architecture".into(), GgufString("qwen35".into())),
            ("qwen35.block_count".into(), U32(5)),
            ("qwen35.nextn_predict_layers".into(), U32(1)),
            ("qwen35.context_length".into(), U32(16)),
            ("qwen35.embedding_length".into(), U32(8)),
            ("qwen35.feed_forward_length".into(), U32(16)),
            ("qwen35.attention.head_count".into(), U32(2)),
            ("qwen35.attention.head_count_kv".into(), U32(1)),
            ("qwen35.attention.key_length".into(), U32(4)),
            ("qwen35.attention.value_length".into(), U32(4)),
            ("qwen35.rope.dimension_count".into(), U32(4)),
            ("qwen35.rope.freq_base".into(), F32(10_000.0)),
            ("qwen35.attention.layer_norm_rms_epsilon".into(), F32(1e-6)),
            ("qwen35.full_attention_interval".into(), U32(4)),
            ("qwen35.ssm.conv_kernel".into(), U32(3)),
            ("qwen35.ssm.state_size".into(), U32(2)),
            ("qwen35.ssm.group_count".into(), U32(1)),
            ("qwen35.ssm.time_step_rank".into(), U32(2)),
            ("qwen35.ssm.inner_size".into(), U32(4)),
        ]
    }

    pub fn quantized(values: Vec<f32>, shape: &[usize]) -> QTensor {
        let tensor = Tensor::from_vec(values, Shape::from_dims(shape), &Device::Cpu).unwrap();
        QTensor::quantize(&tensor, GgmlDType::F32).unwrap()
    }

    pub fn push_tensor(
        tensors: &mut Vec<(String, QTensor)>,
        name: impl Into<String>,
        shape: &[usize],
        value: f32,
    ) {
        tensors.push((
            name.into(),
            quantized(vec![value; shape.iter().product()], shape),
        ));
    }

    /// A complete, deliberately tiny Qwen35 artifact. Mixer and MLP matrices
    /// are zero, making the independent expected logits the final RMS-normalized
    /// embedding while still exercising every loader and state path.
    pub fn tiny_gguf() -> Vec<u8> {
        tiny_gguf_with_context(16)
    }

    pub fn tiny_gguf_with_context(context_length: u32) -> Vec<u8> {
        tiny_gguf_with_shape(context_length, 4)
    }

    /// Tiny artifact with `blocks` target blocks (plus the MTP block). Every
    /// fourth block is full attention, matching the target interval.
    pub fn tiny_gguf_with_shape(context_length: u32, blocks: usize) -> Vec<u8> {
        tiny_gguf_custom(context_length, blocks, &[])
    }

    /// Like [`tiny_gguf_with_shape`] with two one-hot rows: token 4 predicts
    /// token 5 and token 5 predicts token 0. With the tied identity output
    /// head, a prompt ending in token 4 yields exactly one reply token before
    /// the stop token, which is enough for loops that reject empty answers.
    pub fn tiny_gguf_replying(context_length: u32, blocks: usize) -> Vec<u8> {
        tiny_gguf_custom(context_length, blocks, &[(4, 5), (5, 0)])
    }

    /// `one_hot_rows` sets `embedding[token][dim] = 1` for extra rows.
    pub fn tiny_gguf_custom(
        context_length: u32,
        blocks: usize,
        one_hot_rows: &[(usize, usize)],
    ) -> Vec<u8> {
        let mut tensors = Vec::new();
        let mut embeddings = vec![0.0f32; 16 * 8];
        embeddings[8..16].fill(1.0);
        for (token, dim) in one_hot_rows {
            embeddings[token * 8 + dim] = 1.0;
        }
        tensors.push(("token_embd.weight".into(), quantized(embeddings, &[16, 8])));
        push_tensor(&mut tensors, "output_norm.weight", &[8], 1.0);
        let mut output = vec![0.0f32; 16 * 8];
        for index in 0..8 {
            output[index * 8 + index] = 1.0;
        }
        tensors.push(("output.weight".into(), quantized(output, &[16, 8])));

        for layer in 0..blocks {
            let prefix = format!("blk.{layer}");
            push_tensor(
                &mut tensors,
                format!("{prefix}.attn_norm.weight"),
                &[8],
                1.0,
            );
            push_tensor(
                &mut tensors,
                format!("{prefix}.post_attention_norm.weight"),
                &[8],
                1.0,
            );
            push_tensor(
                &mut tensors,
                format!("{prefix}.ffn_gate.weight"),
                &[16, 8],
                0.0,
            );
            push_tensor(
                &mut tensors,
                format!("{prefix}.ffn_up.weight"),
                &[16, 8],
                0.0,
            );
            push_tensor(
                &mut tensors,
                format!("{prefix}.ffn_down.weight"),
                &[8, 16],
                0.0,
            );
            if (layer + 1) % 4 != 0 {
                for (suffix, shape) in [
                    ("attn_qkv.weight", vec![8, 8]),
                    ("attn_gate.weight", vec![4, 8]),
                    ("ssm_alpha.weight", vec![2, 8]),
                    ("ssm_beta.weight", vec![2, 8]),
                    ("ssm_conv1d.weight", vec![8, 3]),
                    ("ssm_dt.bias", vec![2]),
                    ("ssm_a", vec![2]),
                    ("ssm_norm.weight", vec![2]),
                    ("ssm_out.weight", vec![8, 4]),
                ] {
                    push_tensor(
                        &mut tensors,
                        format!("{prefix}.{suffix}"),
                        &shape,
                        if suffix == "ssm_norm.weight" {
                            1.0
                        } else {
                            0.0
                        },
                    );
                }
            } else {
                for (suffix, shape, value) in [
                    ("attn_q.weight", vec![16, 8], 0.0),
                    ("attn_k.weight", vec![4, 8], 0.0),
                    ("attn_v.weight", vec![4, 8], 0.0),
                    ("attn_output.weight", vec![8, 8], 0.0),
                    ("attn_q_norm.weight", vec![4], 1.0),
                    ("attn_k_norm.weight", vec![4], 1.0),
                ] {
                    push_tensor(&mut tensors, format!("{prefix}.{suffix}"), &shape, value);
                }
            }
        }

        let prefix = format!("blk.{blocks}");
        for (suffix, shape, value) in [
            ("attn_norm.weight", vec![8], 1.0),
            ("post_attention_norm.weight", vec![8], 1.0),
            ("attn_q.weight", vec![16, 8], 0.0),
            ("attn_k.weight", vec![4, 8], 0.0),
            ("attn_v.weight", vec![4, 8], 0.0),
            ("attn_output.weight", vec![8, 8], 0.0),
            ("attn_q_norm.weight", vec![4], 1.0),
            ("attn_k_norm.weight", vec![4], 1.0),
            ("ffn_gate.weight", vec![16, 8], 0.0),
            ("ffn_up.weight", vec![16, 8], 0.0),
            ("ffn_down.weight", vec![8, 16], 0.0),
            ("nextn.eh_proj.weight", vec![8, 16], 0.0),
            ("nextn.enorm.weight", vec![8], 1.0),
            ("nextn.hnorm.weight", vec![8], 1.0),
        ] {
            push_tensor(&mut tensors, format!("{prefix}.{suffix}"), &shape, value);
        }

        let mut metadata = tiny_metadata();
        metadata
            .iter_mut()
            .find(|(key, _)| key == "qwen35.context_length")
            .unwrap()
            .1 = gguf_file::Value::U32(context_length);
        metadata
            .iter_mut()
            .find(|(key, _)| key == "qwen35.block_count")
            .unwrap()
            .1 = gguf_file::Value::U32(blocks as u32 + 1);
        let metadata_refs = metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let tensor_refs = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let mut writer = Cursor::new(Vec::new());
        gguf_file::write(&mut writer, &metadata_refs, &tensor_refs).unwrap();
        writer.into_inner()
    }

}

#[cfg(test)]
pub(crate) mod tests {
    use super::test_fixture::*;
    use super::*;
    use candle_core::{Shape, quantized::GgmlDType};
    use sha2::{Digest, Sha256};
    use std::ffi::OsString;
    use std::io::{BufReader, Cursor, Read};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn kv_pages_absorb_the_oldest_positions_and_decode_them_back_in_order() {
        let device = Device::Cpu;
        let (heads, positions, dim) = (2usize, 8usize, 64usize);
        let values: Vec<f32> = (0..heads * positions * dim)
            .map(|i| ((i % 97) as f32 / 48.0) - 1.0)
            .collect();
        let tensor = Tensor::from_vec(values.clone(), (1, heads, positions, dim), &device).unwrap();

        let policy = KvQuantization { bits: 4, recent: 3 };
        let mut pages = KvPages::new(policy, heads, heads, dim);
        let overflow = positions - policy.recent;
        KvPages::absorb(&mut pages.key, &tensor, overflow).unwrap();
        assert_eq!(pages.positions(), overflow);

        let decoded = KvPages::decode(&pages.key, heads, dim, &device).unwrap();
        assert_eq!(decoded.dims(), &[1, heads, overflow, dim]);
        let got: Vec<f32> = decoded.flatten_all().unwrap().to_vec1().unwrap();

        // Each head's oldest positions come back in order, close but not exact.
        for head in 0..heads {
            for position in 0..overflow {
                let original_at = (head * positions + position) * dim;
                let decoded_at = (head * overflow + position) * dim;
                let a = &values[original_at..original_at + dim];
                let b = &got[decoded_at..decoded_at + dim];
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    dot / (na * nb) > 0.97,
                    "head {head} position {position} cosine {}",
                    dot / (na * nb)
                );
                assert!(a != b, "a compressed page should not be bit-identical");
            }
        }

        // Compression is real, not an unpacked index array.
        let (stored, plain) = pages.stored_and_uncompressed_bytes();
        assert!(
            plain as f64 / stored as f64 > 5.0,
            "ratio {}",
            plain as f64 / stored as f64
        );
    }

    static Q8_ROUTE_ENV: Mutex<()> = Mutex::new(());
    static VERIFIED_PTAH_MODEL: OnceLock<PathBuf> = OnceLock::new();
    const PTAH_MODEL_SHA256: &str =
        "3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec";

    struct EnvOverride {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvOverride {
        fn new(key: &'static str, value: &str) -> Self {
            let original = std::env::var_os(key);
            // SAFETY: every test mutating this process-wide selector holds
            // `Q8_ROUTE_ENV`; production never mutates it after startup.
            unsafe { std::env::set_var(key, value) };
            Self { key, original }
        }
    }

    impl Drop for EnvOverride {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn assert_ptah_model_fingerprint(path: &Path) {
        let verified = VERIFIED_PTAH_MODEL.get_or_init(|| {
            let file = std::fs::File::open(path).expect("open Ptah model for fingerprinting");
            let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 8 * 1024 * 1024];
            loop {
                let read = reader
                    .read(&mut buffer)
                    .expect("read Ptah model while fingerprinting");
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            let actual = format!("{:x}", hasher.finalize());
            assert_eq!(
                actual, PTAH_MODEL_SHA256,
                "hardware acceptance must run against Ptah's exact custom Qwen3.8 artifact"
            );
            path.to_owned()
        });
        assert_eq!(
            path, verified,
            "all exact-model gates in one test process must use the same Ptah artifact"
        );
    }

    fn exact_shape_metadata() -> HashMap<String, gguf_file::Value> {
        use gguf_file::Value::{F32, U32};

        HashMap::from([
            ("qwen35.block_count".into(), U32(65)),
            ("qwen35.nextn_predict_layers".into(), U32(1)),
            ("qwen35.context_length".into(), U32(262_144)),
            ("qwen35.embedding_length".into(), U32(5_120)),
            ("qwen35.feed_forward_length".into(), U32(17_408)),
            ("qwen35.attention.head_count".into(), U32(24)),
            ("qwen35.attention.head_count_kv".into(), U32(4)),
            ("qwen35.attention.key_length".into(), U32(256)),
            ("qwen35.attention.value_length".into(), U32(256)),
            ("qwen35.rope.dimension_count".into(), U32(64)),
            ("qwen35.rope.freq_base".into(), F32(10_000_000.0)),
            ("qwen35.attention.layer_norm_rms_epsilon".into(), F32(1e-6)),
            ("qwen35.full_attention_interval".into(), U32(4)),
            ("qwen35.ssm.conv_kernel".into(), U32(4)),
            ("qwen35.ssm.state_size".into(), U32(128)),
            ("qwen35.ssm.group_count".into(), U32(16)),
            ("qwen35.ssm.time_step_rank".into(), U32(48)),
            ("qwen35.ssm.inner_size".into(), U32(6_144)),
        ])
    }

    fn load_tiny_model(bytes: &[u8]) -> ModelWeights {
        let mut reader = Cursor::new(bytes);
        let content = gguf_file::Content::read(&mut reader).unwrap();
        ModelWeights::from_gguf(content, &mut reader, &Device::Cpu).unwrap()
    }

    #[test]
    fn prompt_taps_pool_only_the_selected_post_block_span() {
        let mut session = load_tiny_model(&tiny_gguf());
        let mut reference = session.new_session();
        // Zero mixers/MLPs make each block's residual the input embedding:
        // token 1 is eight ones; token 2 is eight zeros. Exclude the first one.
        let tokens = [1, 2, 1, 1];
        let result = session
            .resume_prompt_with_taps(&tokens, 1..4, &[0, 3], &mut || Ok(()))
            .unwrap();
        assert_eq!(result.reused_tokens, 0);
        let taps = result.taps.unwrap();
        assert_eq!(taps.len(), 2);
        for (tap, layer) in taps.iter().zip([0, 3]) {
            assert_eq!(tap.layer, layer);
            assert_eq!(tap.mean.dims(), &[8]);
            for value in tap.mean.to_vec1::<f32>().unwrap() {
                assert!((value - 2.0 / 3.0).abs() < 1e-6);
            }
        }
        let expected = reference.resume_prompt(&tokens, &mut || Ok(())).unwrap().0;
        assert_tensor_close(&result.logits, &expected, "tap capture preserves logits");
    }

    #[test]
    fn prompt_taps_weight_tokens_across_unequal_prefill_chunks() {
        let mut session = load_tiny_model(&tiny_gguf_with_context(256));
        let mut tokens = vec![2; 130];
        tokens[127] = 1;
        tokens[128] = 1;
        let result = session
            .resume_prompt_with_taps(&tokens, 127..130, &[0, 3], &mut || Ok(()))
            .unwrap();
        for tap in result.taps.unwrap() {
            // One selected token in chunk 1, two in chunk 2. Averaging the
            // chunk means would incorrectly produce 0.75 instead of 2/3.
            for value in tap.mean.to_vec1::<f32>().unwrap() {
                assert!((value - 2.0 / 3.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn prompt_taps_report_cached_spans_as_misses_and_keep_sessions_private() {
        let mut session = load_tiny_model(&tiny_gguf());
        let mut sibling = session.new_session();
        let prompt = [1, 2, 1];
        session.resume_prompt(&prompt, &mut || Ok(())).unwrap();
        let checkpoint = session.checkpoint();
        let cached = session
            .resume_prompt_with_taps(&prompt, 1..3, &[0], &mut || Ok(()))
            .unwrap();
        assert_eq!(cached.reused_tokens, 3);
        assert!(cached.taps.is_none());
        let extended = [1, 2, 1, 2, 1];
        let partial = session
            .resume_prompt_with_taps(&extended, 2..5, &[0], &mut || Ok(()))
            .unwrap();
        assert_eq!(partial.reused_tokens, 3);
        assert!(
            partial.taps.is_none(),
            "a partial span must not masquerade as complete"
        );
        session.rewind(checkpoint).unwrap();
        let suffix = session
            .resume_prompt_with_taps(&extended, 3..5, &[0], &mut || Ok(()))
            .unwrap();
        assert_eq!(suffix.reused_tokens, 3);
        assert_eq!(
            suffix.taps.unwrap()[0].mean.to_vec1::<f32>().unwrap(),
            vec![0.5; 8]
        );
        let independent = sibling
            .resume_prompt_with_taps(&extended, 3..5, &[0], &mut || Ok(()))
            .unwrap();
        assert_eq!(independent.reused_tokens, 0);
        assert_eq!(
            independent.taps.unwrap()[0].mean.to_vec1::<f32>().unwrap(),
            vec![0.5; 8]
        );
    }

    #[test]
    fn prompt_taps_discard_cancelled_capture_and_reject_invalid_requests() {
        let mut session = load_tiny_model(&tiny_gguf());
        session.resume_prompt(&[1, 2], &mut || Ok(())).unwrap();
        for (span, layers) in [
            (0..0, vec![0]),
            (0..4, vec![0]),
            (0..3, vec![]),
            (0..3, vec![4]),
            (0..3, vec![0, 0]),
        ] {
            assert!(
                session
                    .resume_prompt_with_taps(&[1, 2, 1], span, &layers, &mut || Ok(()))
                    .is_err()
            );
        }
        assert_eq!(session.resume_prompt(&[1, 2], &mut || Ok(())).unwrap().1, 2);
        let checkpoint = session.checkpoint();
        let mut checks = 0;
        assert!(
            session
                .resume_prompt_with_taps(&[1, 2, 1], 2..3, &[0], &mut || {
                    checks += 1;
                    if checks == 3 {
                        candle_core::bail!("cancel after collected suffix");
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(checks, 3);
        let recovered = session
            .resume_prompt_with_taps(&[1, 2, 1], 2..3, &[0], &mut || Ok(()))
            .unwrap();
        assert_eq!(recovered.reused_tokens, 0);
        assert_eq!(
            recovered.taps.unwrap()[0].mean.to_vec1::<f32>().unwrap(),
            vec![1.0; 8]
        );
        session.rewind(checkpoint).unwrap();
        assert_eq!(session.resume_prompt(&[1, 2], &mut || Ok(())).unwrap().1, 2);
    }

    fn make_recurrent_state_observable(model: &mut ModelWeights) {
        for (layer_index, layer) in Arc::get_mut(&mut model.weights)
            .unwrap()
            .layers
            .iter_mut()
            .enumerate()
        {
            let Mixer::GatedDelta(gated_delta) = &mut layer.mixer else {
                continue;
            };
            let scale = 0.01 * (layer_index + 1) as f32;
            gated_delta.qkv =
                QMatMul::from_qtensor(quantized(vec![scale; 8 * 8], &[8, 8])).unwrap();
            gated_delta.alpha =
                QMatMul::from_qtensor(quantized(vec![scale; 2 * 8], &[2, 8])).unwrap();
            gated_delta.beta =
                QMatMul::from_qtensor(quantized(vec![scale; 2 * 8], &[2, 8])).unwrap();
            gated_delta.conv = Tensor::from_vec(
                (0..8)
                    .flat_map(|channel| {
                        let channel = channel as f32 + 1.0;
                        [0.1 * channel, 0.2 * channel, 0.3 * channel]
                    })
                    .collect::<Vec<_>>(),
                (8, 3),
                &Device::Cpu,
            )
            .unwrap();
            gated_delta.dt_bias = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
            gated_delta.a = Tensor::new(&[-0.5f32, -0.25], &Device::Cpu).unwrap();
        }
    }

    fn assert_tensor_close(left: &Tensor, right: &Tensor, label: &str) {
        assert_eq!(left.shape(), right.shape(), "{label} shape");
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (left, right)) in left.iter().zip(&right).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "{label}[{index}]: {left} != {right}"
            );
        }
    }

    fn assert_checkpoint_close(left: &Checkpoint, right: &Checkpoint) {
        assert_eq!(left.layers.len(), right.layers.len());
        for (layer, (left, right)) in left.layers.iter().zip(&right.layers).enumerate() {
            match (left, right) {
                (MixerState::Attention(Some(left)), MixerState::Attention(Some(right))) => {
                    assert_tensor_close(&left.key, &right.key, &format!("layer {layer} key"));
                    assert_tensor_close(&left.value, &right.value, &format!("layer {layer} value"));
                }
                (MixerState::GatedDelta(Some(left)), MixerState::GatedDelta(Some(right))) => {
                    assert_eq!(left.position, right.position, "layer {layer} position");
                    assert_tensor_close(
                        &left.conv_history,
                        &right.conv_history,
                        &format!("layer {layer} conv history"),
                    );
                    assert_tensor_close(
                        &left.recurrent,
                        &right.recurrent,
                        &format!("layer {layer} recurrent"),
                    );
                }
                _ => panic!("layer {layer} state kinds differ or are absent"),
            }
        }
    }

    fn streaming_cpu_logits(path: &std::path::Path, tokens: &[u32]) -> Vec<Vec<f32>> {
        let device = Device::Cpu;
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let config = Config::from_metadata(&content.metadata).unwrap();
        validate_tensor_schema(&content, &config).unwrap();
        let rope = PartialRope::new(&config, &device).unwrap();
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device).unwrap();

        let embedding = TokenEmbedding::from_qtensor(
            content
                .tensor(&mut reader, "token_embd.weight", &device)
                .unwrap(),
            config.embedding_length,
            &device,
        )
        .unwrap();
        let tokens = Tensor::from_slice(tokens, (1, tokens.len()), &device).unwrap();
        let mut hidden = embedding.forward(&tokens).unwrap();
        drop(embedding);
        let mask =
            candle_transformers::utils::build_causal_mask(tokens.dim(1).unwrap(), 0, &device)
                .unwrap();

        for layer_index in 0..config.block_count {
            let layer = load_layer(
                &content,
                &mut reader,
                &config,
                &rope,
                &neg_inf,
                layer_index,
                &device,
            )
            .unwrap();
            hidden = layer
                .forward(&mut layer.mixer.empty_state(), &hidden, Some(&mask), 0)
                .unwrap();
        }
        let norm = RmsNorm::from_qtensor(
            content
                .tensor(&mut reader, "output_norm.weight", &device)
                .unwrap(),
            config.rms_norm_eps,
        )
        .unwrap();
        let hidden = norm.forward(&hidden).unwrap();
        let output = QMatMul::from_qtensor(
            content
                .tensor(&mut reader, "output.weight", &device)
                .unwrap(),
        )
        .unwrap();
        output
            .forward(&hidden)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }

    fn log_probability(logits: &[f32], token: usize) -> f32 {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max
            + logits
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f32>()
                .ln();
        logits[token] - log_sum_exp
    }

    #[test]
    fn exact_qwen38_metadata_becomes_a_typed_hybrid_schedule() {
        let config = Config::from_metadata(&exact_shape_metadata()).unwrap();

        assert_eq!(config.total_block_count, 65);
        assert_eq!(config.block_count, 64);
        assert_eq!(config.nextn_block_count, 1);
        assert_eq!(config.context_length, 262_144);
        assert_eq!(config.embedding_length, 5_120);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.rope_dim, 64);
        assert_eq!(config.inner_size / config.value_head_count, 128);
        assert!(config.is_recurrent(0));
        assert!(config.is_recurrent(1));
        assert!(config.is_recurrent(2));
        assert!(!config.is_recurrent(3));
        assert!(!config.is_recurrent(63));
        assert!(
            !config.is_recurrent(64),
            "the appended MTP block is not target state"
        );
    }

    #[test]
    fn gated_delta_step_updates_state_before_reading_the_query() {
        let device = candle_core::Device::Cpu;
        let state = Tensor::zeros((1, 1, 2, 2), candle_core::DType::F32, &device).unwrap();
        let q = Tensor::new(&[[[1.0f32, 0.0]]], &device).unwrap();
        let k = Tensor::new(&[[[1.0f32, 0.0]]], &device).unwrap();
        let v = Tensor::new(&[[[2.0f32, 3.0]]], &device).unwrap();
        let decay = Tensor::new(&[[[0.5f32]]], &device).unwrap();
        let beta = Tensor::new(&[[[0.25f32]]], &device).unwrap();

        let (output, state) = gated_delta_step(&state, &q, &k, &v, &decay, &beta).unwrap();

        assert_eq!(
            state.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.5, 0.0, 0.75, 0.0]
        );
        let output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((output[0] - 0.353_553_38).abs() < 1e-6, "{output:?}");
        assert!((output[1] - 0.530_330_06).abs() < 1e-6, "{output:?}");
    }

    #[test]
    fn causal_conv_chunk_matches_token_at_a_time() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[[1.0f32, 10.0], [2.0, 20.0], [3.0, 30.0]]], &device).unwrap();
        let kernel = Tensor::new(&[[1.0f32, 2.0, 3.0], [0.5, 0.0, -0.5]], &device).unwrap();

        let (chunked, chunk_history) = causal_depthwise_conv(&input, &kernel, None).unwrap();
        assert_eq!(
            chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, -5.0, 8.0, -10.0, 14.0, -10.0]
        );

        let mut history = None;
        let mut steps = Vec::new();
        for offset in 0..3 {
            let token = input.narrow(1, offset, 1).unwrap();
            let (output, next_history) =
                causal_depthwise_conv(&token, &kernel, history.as_ref()).unwrap();
            steps.push(output);
            history = Some(next_history);
        }
        let stepwise = Tensor::cat(&steps.iter().collect::<Vec<_>>(), 1).unwrap();

        assert_eq!(
            chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            stepwise.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            chunk_history
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            history
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn gguf_reordered_value_heads_use_tiled_qk_expansion() {
        let heads = Tensor::new(&[[[[1.0f32], [2.0]]]], &Device::Cpu).unwrap();
        let tiled = repeat_tiled_heads(&heads, 3).unwrap();

        assert_eq!(
            tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn complete_fixture_loads_and_matches_independent_logits() {
        let bytes = tiny_gguf();
        let mut header = Cursor::new(&bytes);
        let config = ModelWeights::validate_gguf(&mut header).unwrap();
        assert_eq!(config.block_count, 4);

        let mut model = load_tiny_model(&bytes);
        let tokens = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
        let logits = model
            .forward(&tokens, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let expected_active = 1.0 / (1.0f32 + 1e-6).sqrt();
        for (index, value) in logits.iter().enumerate() {
            let expected = if index < 8 { expected_active } else { 0.0 };
            assert!((value - expected).abs() < 1e-5, "logit {index}: {value}");
        }
    }

    #[test]
    fn bounded_prefill_matches_dense_state_logits_and_decode() {
        let bytes = tiny_gguf();
        let tokens = Tensor::new(&[[1u32, 2, 1, 3, 1, 2, 1]], &Device::Cpu).unwrap();
        for chunk_size in [1, 2, 3, 7, 128] {
            let mut dense = load_tiny_model(&bytes);
            let mut bounded = load_tiny_model(&bytes);
            make_recurrent_state_observable(&mut dense);
            make_recurrent_state_observable(&mut bounded);
            let (dense_logits, dense_hidden) = dense.run_with_hidden(&tokens, 0).unwrap();
            let mut calls = 0;
            let (logits, hidden) = bounded
                .prefill_controlled_with_hidden(&tokens, chunk_size, &mut || {
                    calls += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(calls, 7usize.div_ceil(chunk_size) + 1);
            assert_tensor_close(&dense_logits, &logits, "chunked logits");
            assert_tensor_close(&dense_hidden, &hidden, "chunked hidden");
            assert_checkpoint_close(&dense.checkpoint(), &bounded.checkpoint());
            let checkpoint = bounded.checkpoint();
            let token = Tensor::new(&[[2u32]], &Device::Cpu).unwrap();
            let expected = dense.decode_step(&token, 7).unwrap();
            assert_tensor_close(
                &expected,
                &bounded.decode_step(&token, 7).unwrap(),
                "decode",
            );
            bounded.rewind(checkpoint).unwrap();
            assert_tensor_close(
                &expected,
                &bounded.decode_step(&token, 7).unwrap(),
                "rewound decode",
            );
            assert_checkpoint_close(&dense.checkpoint(), &bounded.checkpoint());
        }
    }

    #[test]
    fn shared_weight_sessions_interleave_without_state_leakage() {
        let bytes = tiny_gguf();
        let mut first = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut first);
        let mut second = first.new_session();
        let mut reference = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut reference);
        let a = Tensor::new(&[[1u32, 2]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[3u32, 1, 2]], &Device::Cpu).unwrap();
        first.prefill(&a).unwrap();
        second.prefill(&b).unwrap();
        let token = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
        reference.prefill(&a).unwrap();
        assert_tensor_close(
            &reference.decode_step(&token, 2).unwrap(),
            &first.decode_step(&token, 2).unwrap(),
            "first interleaved session",
        );
        reference.prefill(&b).unwrap();
        assert_tensor_close(
            &reference.decode_step(&token, 3).unwrap(),
            &second.decode_step(&token, 3).unwrap(),
            "second interleaved session",
        );
        assert!(first.rewind(second.checkpoint()).is_err());
        drop(first);
        assert_tensor_close(
            &reference.decode_step(&token, 4).unwrap(),
            &second.decode_step(&token, 4).unwrap(),
            "surviving session",
        );
    }

    #[test]
    fn text_continuation_appends_only_the_forwarded_prefix() {
        let bytes = tiny_gguf();
        let mut session = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut session);
        let mut reference = session.new_session();
        let first = [1u32, 2, 1];
        let extended = [1u32, 2, 1, 3, 2];
        let (_, reused) = session.resume_prompt(&first, &mut || Ok(())).unwrap();
        assert_eq!(reused, 0);
        let (actual, reused) = session.resume_prompt(&extended, &mut || Ok(())).unwrap();
        assert_eq!(reused, 3);
        let expected = reference
            .prefill(&Tensor::new(&[extended], &Device::Cpu).unwrap())
            .unwrap();
        assert_tensor_close(&expected, &actual, "extended prompt");
        let (same, reused) = session.resume_prompt(&extended, &mut || Ok(())).unwrap();
        assert_eq!(reused, 5);
        assert_tensor_close(&actual, &same, "identical prompt");
    }

    #[test]
    fn text_decode_and_checkpoint_track_only_forwarded_tokens() {
        let bytes = tiny_gguf();
        let mut session = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut session);
        let mut reference = session.new_session();
        session.resume_prompt(&[1, 2, 3], &mut || Ok(())).unwrap();
        let saved = session.checkpoint();
        session.forward_token(1, &mut || Ok(())).unwrap();
        let (actual, reused) = session
            .resume_prompt(&[1, 2, 3, 1, 2], &mut || Ok(()))
            .unwrap();
        assert_eq!(reused, 4);
        let expected = reference
            .prefill(&Tensor::new(&[[1u32, 2, 3, 1, 2]], &Device::Cpu).unwrap())
            .unwrap();
        assert_tensor_close(&actual, &expected, "decode then continuation");
        session.rewind(saved).unwrap();
        let (_, reused) = session.resume_prompt(&[1, 2, 3], &mut || Ok(())).unwrap();
        assert_eq!(reused, 3);
        let (actual, reused) = session
            .resume_prompt(&[1, 2, 3, 2], &mut || Ok(()))
            .unwrap();
        assert_eq!(reused, 3);
        let expected = reference
            .prefill(&Tensor::new(&[[1u32, 2, 3, 2]], &Device::Cpu).unwrap())
            .unwrap();
        assert_tensor_close(&actual, &expected, "rewound continuation");
    }

    #[test]
    fn cancelled_text_extension_and_decode_recover_with_fresh_state() {
        let bytes = tiny_gguf();
        let mut session = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut session);
        let mut reference = session.new_session();
        let prompt = [1u32, 2, 3];
        let extended = [1u32, 2, 3, 1, 2];
        session.resume_prompt(&prompt, &mut || Ok(())).unwrap();
        let mut checks = 0;
        assert!(
            session
                .resume_prompt(&extended, &mut || {
                    checks += 1;
                    if checks == 3 {
                        candle_core::bail!("cancel after suffix chunk");
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(checks, 3);
        assert!(session.forward_token(1, &mut || Ok(())).is_err());
        let (actual, reused) = session.resume_prompt(&extended, &mut || Ok(())).unwrap();
        assert_eq!(reused, 0);
        let expected = reference
            .prefill(&Tensor::new(&[extended], &Device::Cpu).unwrap())
            .unwrap();
        assert_tensor_close(&actual, &expected, "cancelled suffix recovery");
        let mut checks = 0;
        assert!(
            session
                .forward_token(2, &mut || {
                    checks += 1;
                    if checks == 2 {
                        candle_core::bail!("cancel after decode");
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(checks, 2);
        let (actual, reused) = session.resume_prompt(&extended, &mut || Ok(())).unwrap();
        assert_eq!(reused, 0);
        assert_tensor_close(&actual, &expected, "cancelled decode recovery");
    }

    #[test]
    fn divergent_or_untracked_text_reprefills_and_limits_do_not_truncate() {
        let bytes = tiny_gguf();
        let mut session = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut session);
        let mut reference = session.new_session();
        session.resume_prompt(&[1, 2, 3], &mut || Ok(())).unwrap();
        for prompt in [&[1u32, 3, 2][..], &[1u32, 3][..]] {
            let (actual, reused) = session.resume_prompt(prompt, &mut || Ok(())).unwrap();
            assert_eq!(reused, 0);
            let expected = reference
                .prefill(&Tensor::from_slice(prompt, (1, prompt.len()), &Device::Cpu).unwrap())
                .unwrap();
            assert_tensor_close(&actual, &expected, "divergent/shorter prompt");
        }
        assert!(session.resume_prompt(&[1; 17], &mut || Ok(())).is_err());
        assert!(session.resume_prompt(&[], &mut || Ok(())).is_err());
        assert_eq!(session.resume_prompt(&[1, 3], &mut || Ok(())).unwrap().1, 2);
        session
            .decode_step(&Tensor::new(&[[2u32]], &Device::Cpu).unwrap(), 2)
            .unwrap();
        assert_eq!(
            session
                .resume_prompt(&[1, 3, 2, 1], &mut || Ok(()))
                .unwrap()
                .1,
            0
        );
    }

    #[test]
    fn bounded_prefill_validates_and_cancellation_clears_partial_state() {
        let bytes = tiny_gguf();
        let mut model = load_tiny_model(&bytes);
        let tokens = Tensor::new(&[[1u32, 2, 1, 3, 1]], &Device::Cpu).unwrap();
        model.prefill(&tokens).unwrap();
        let saved = model.checkpoint();
        for (length, chunk) in [(0, 2), (17, 2), (5, 0)] {
            let invalid = Tensor::zeros((1, length), DType::U32, &Device::Cpu).unwrap();
            assert!(
                model
                    .prefill_controlled(&invalid, chunk, &mut || Ok(()))
                    .is_err()
            );
            assert_checkpoint_close(&saved, &model.checkpoint());
        }
        for fail_at in [1, 2, 4] {
            let mut calls = 0;
            let result = model.prefill_controlled(&tokens, 2, &mut || {
                calls += 1;
                if calls == fail_at {
                    candle_core::bail!("cancelled")
                }
                Ok(())
            });
            assert!(result.is_err());
            assert_eq!(calls, fail_at);
            assert!(model.mask_cache.is_none());
            assert!(model.checkpoint().layers.iter().all(|layer| matches!(
                layer,
                MixerState::Attention(None) | MixerState::GatedDelta(None)
            )));
            // Historical same-model snapshots survive reset/cancellation.
            model.rewind(saved.clone()).unwrap();
            assert_checkpoint_close(&saved, &model.checkpoint());
        }
        let mut foreign = load_tiny_model(&bytes);
        foreign.prefill(&tokens).unwrap();
        assert!(model.rewind(foreign.checkpoint()).is_err());
        assert_checkpoint_close(&saved, &model.checkpoint());
        let stale = model.checkpoint();
        drop(model);
        let mut reloaded = load_tiny_model(&bytes);
        assert!(reloaded.rewind(stale).is_err());
    }

    #[test]
    fn chunked_prefill_matches_steps_and_checkpoint_rewinds_state() {
        let bytes = tiny_gguf();
        let mut chunked = load_tiny_model(&bytes);
        let mut stepped = load_tiny_model(&bytes);
        make_recurrent_state_observable(&mut chunked);
        make_recurrent_state_observable(&mut stepped);
        let tokens = Tensor::new(&[[1u32, 1, 1]], &Device::Cpu).unwrap();

        chunked.prefill(&tokens).unwrap();
        let chunked_state = chunked.checkpoint();
        stepped.prefill(&tokens.narrow(1, 0, 1).unwrap()).unwrap();
        let before_second = stepped.checkpoint();
        stepped
            .decode_step(&tokens.narrow(1, 1, 1).unwrap(), 1)
            .unwrap();
        let after_second = stepped.checkpoint();
        stepped
            .decode_step(&tokens.narrow(1, 2, 1).unwrap(), 2)
            .unwrap();
        let stepped_state = stepped.checkpoint();

        let recurrent = match &chunked_state.layers[0] {
            MixerState::GatedDelta(Some(state)) => state
                .recurrent
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            _ => panic!("first target layer must own recurrent state"),
        };
        assert!(
            recurrent.iter().any(|value| value.abs() > 1e-4),
            "fixture must exercise non-zero recurrent state"
        );
        assert_checkpoint_close(&chunked_state, &stepped_state);

        stepped.rewind(before_second).unwrap();
        stepped
            .decode_step(&tokens.narrow(1, 1, 1).unwrap(), 1)
            .unwrap();
        assert_checkpoint_close(&after_second, &stepped.checkpoint());
    }

    #[test]
    fn installed_artifact_schema_when_path_is_supplied() {
        let Some(path) = std::env::var_os("AKH_QWEN35_TEST_MODEL") else {
            return;
        };
        let mut reader = std::fs::File::open(path).unwrap();
        let config = ModelWeights::validate_gguf(&mut reader).unwrap();

        assert_eq!(config.total_block_count, 65);
        assert_eq!(config.block_count, 64);
        assert_eq!(config.nextn_block_count, 1);
        assert_eq!(config.embedding_length, 5_120);
        assert_eq!(config.context_length, 262_144);
    }

    #[test]
    fn append_without_owned_state_fails_closed() {
        let bytes = tiny_gguf();
        let mut model = load_tiny_model(&bytes);
        let token = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();

        let error = model.forward(&token, 1).unwrap_err().to_string();
        assert!(error.contains("has no state to append"), "{error}");
    }

    #[test]
    fn installed_artifact_streaming_cpu_matches_llama_cpp() {
        let Some(path) = std::env::var_os("AKH_QWEN35_TEST_MODEL") else {
            return;
        };
        // The installed tokenizer maps `Hello world` -> [9419, 1814]. These
        // prefix target IDs and log-probabilities were captured from the
        // production llama.cpp target forward with sampling disabled. Position
        // one exercises causal masking, non-identity RoPE, and prior-token KV.
        let logits = streaming_cpu_logits(std::path::Path::new(&path), &[9_419, 1_814]);
        for (prompt, logits, oracle_token, oracle_log_probability) in [
            ("Hello", &logits[0], 11, -0.568_946_3f32),
            ("Hello world", &logits[1], 0, -1.457_515_6f32),
        ] {
            let actual = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .unwrap();
            assert_eq!(
                actual, oracle_token,
                "streaming CPU Candle must select llama.cpp's greedy next token for {prompt}"
            );
            let actual_log_probability = log_probability(logits, oracle_token);
            assert!(
                (actual_log_probability - oracle_log_probability).abs() < 0.05,
                "{prompt}: Candle log-probability {actual_log_probability}, llama.cpp {oracle_log_probability}"
            );
        }
    }

    /// Exact-artifact stepped-decode canary. `Hello world` tokenizes to
    /// `[9419, 1814]`; the production llama.cpp oracle selects token 0 (`!`).
    /// Kept opt-in because it needs the full model and GPU.
    #[cfg(feature = "hip")]
    #[test]
    fn installed_artifact_stepped_decode_matches_llama_cpp() {
        let _route_lock = Q8_ROUTE_ENV
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _route = EnvOverride::new("AKH_HIP_Q8_GEMV", "0");
        let Some(path) = std::env::var_os("AKH_QWEN35_TEST_MODEL") else {
            return;
        };
        assert_ptah_model_fingerprint(Path::new(&path));
        let device = Device::new_cuda(0).unwrap();
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut reader, &device).unwrap();
        let hello = Tensor::new(&[[9_419u32]], &device).unwrap();
        model.prefill(&hello).unwrap();
        let world = Tensor::new(&[[1_814u32]], &device).unwrap();
        let (logits, hidden) = model.decode_step_with_hidden(&world, 1).unwrap();
        let logits = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut cpu_reader =
            std::fs::File::open(std::env::var_os("AKH_QWEN35_TEST_MODEL").unwrap()).unwrap();
        let cpu_content = gguf_file::Content::read(&mut cpu_reader).unwrap();
        let cpu_output = QMatMul::from_qtensor(
            cpu_content
                .tensor(&mut cpu_reader, "output.weight", &Device::Cpu)
                .unwrap(),
        )
        .unwrap();
        let cpu_logits = cpu_output
            .forward(&hidden.to_device(&Device::Cpu).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let non_finite = logits.iter().filter(|value| !value.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "Candle produced non-finite exact-model logits"
        );
        let actual = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap();
        let cpu_actual = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap();

        assert_eq!(
            actual, cpu_actual,
            "HIP and CPU Q6_K output projections disagree"
        );

        assert_eq!(
            cpu_actual, 0,
            "Candle must select llama.cpp's greedy next token (actual_logit={}, oracle_logit={})",
            cpu_logits[cpu_actual], cpu_logits[0]
        );
        let actual_log_probability = log_probability(&logits, 0);
        assert!(
            (actual_log_probability - -1.457_515_6).abs() < 0.05,
            "HIP log-probability {actual_log_probability}, llama.cpp -1.4575156"
        );
    }

    /// Exact-model gate for the RDNA3 Q8-activation/packed-Q4_K
    /// decode tracer. It compares two forwards from the same owned checkpoint,
    /// so recurrent and KV state cannot disguise a kernel difference.
    #[cfg(feature = "hip")]
    #[test]
    #[ignore = "exclusive GPU: exact Ptah prefill threshold/forced-decode diagnostic"]
    fn installed_artifact_prefill_threshold_probe() {
        let _route_lock = Q8_ROUTE_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL")
            .expect("set AKH_QWEN35_TEST_MODEL to the exact Ptah GGUF");
        assert_ptah_model_fingerprint(Path::new(&path));
        let repeats: usize = std::env::var("AKH_QWEN35_PROBE_REPEATS")
            .map(|value| value.parse().expect("probe repeats must be an integer"))
            .unwrap_or(3);
        assert!((1..=12).contains(&repeats));
        let _q8 = EnvOverride::new("AKH_HIP_Q8_GEMV", "1");
        let device = Device::new_cuda(0).unwrap();
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut reader, &device).unwrap();
        device.synchronize().unwrap();

        let vram = || {
            let mut cards = std::collections::BTreeMap::new();
            for entry in std::fs::read_dir("/sys/class/drm").unwrap().flatten() {
                let path = entry.path().join("device/mem_info_vram_used");
                if let Ok(value) = std::fs::read_to_string(path) {
                    if let Ok(bytes) = value.trim().parse::<u64>() {
                        cards.insert(entry.file_name().to_string_lossy().into_owned(), bytes);
                    }
                }
            }
            cards
        };
        let vector = |tensor: &Tensor| tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cosine = |left: &[f32], right: &[f32]| {
            let (mut dot, mut ln, mut rn) = (0f64, 0f64, 0f64);
            for (&l, &r) in left.iter().zip(right) {
                assert!(l.is_finite() && r.is_finite());
                dot += l as f64 * r as f64;
                ln += (l as f64).powi(2);
                rn += (r as f64).powi(2);
            }
            dot / (ln * rn).sqrt()
        };
        let mut run = |length, batched: bool, iteration: Option<usize>| {
            let _batch = EnvOverride::new("AKH_GEMV_BATCH", if batched { "1" } else { "0" });
            device.synchronize().unwrap();
            candle::cuda::reset_pcie_counts();
            candle::cuda::reset_malloc_counts();
            let start = std::time::Instant::now();
            let input = Tensor::new(vec![9_419u32; length], &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let (logits, hidden) = model.prefill_with_hidden(&input).unwrap();
            let first_token = logits
                .i(0)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_scalar::<u32>()
                .unwrap();
            device.synchronize().unwrap();
            let prefill_secs = start.elapsed().as_secs_f64();
            let prefill_pcie = candle::cuda::pcie_counts();
            let prefill_alloc = candle::cuda::malloc_profile();
            let prefill_vram = vram();
            // Full vectors are diagnostics only: exclude their readback from stage timing/counters.
            let first = (vector(&logits), vector(&hidden));
            drop((logits, hidden));
            candle::cuda::reset_pcie_counts();
            candle::cuda::reset_malloc_counts();
            let start = std::time::Instant::now();
            let mut last = None;
            let mut greedy = vec![first_token];
            for step in 0..8 {
                let input = Tensor::new(&[[9_419u32]], &device).unwrap();
                let (logits, hidden) = model
                    .decode_step_with_hidden(&input, length + step)
                    .unwrap();
                greedy.push(
                    logits
                        .i(0)
                        .unwrap()
                        .argmax(0)
                        .unwrap()
                        .to_scalar::<u32>()
                        .unwrap(),
                );
                device.synchronize().unwrap();
                last = Some((logits, hidden));
            }
            let decode_secs = start.elapsed().as_secs_f64();
            let decode_pcie = candle::cuda::pcie_counts();
            let decode_alloc = candle::cuda::malloc_profile();
            let decode_vram = vram();
            let (logits, hidden) = last.unwrap();
            let last = (vector(&logits), vector(&hidden));
            if let Some(iteration) = iteration {
                eprintln!(
                    "QWEN35_PREFILL_PROBE {}",
                    serde_json::json!({
                        "prompt_token_id": 9419, "prompt_length": length,
                        "forced_decode_token_id": 9419, "decode_steps": 8,
                        "batched_gemv": batched, "q8_gemv": true, "iteration": iteration,
                        "prefill_secs": prefill_secs, "decode_secs": decode_secs,
                        "prefill_pcie_h2d_d2h": prefill_pcie, "decode_pcie_h2d_d2h": decode_pcie,
                        "prefill_malloc_calls_misses_nanos": prefill_alloc,
                        "decode_malloc_calls_misses_nanos": decode_alloc,
                        "prefill_vram_used_bytes_by_card": prefill_vram,
                        "decode_vram_used_bytes_by_card": decode_vram,
                        "greedy_tokens": greedy,
                    })
                );
            }
            (first, last, greedy)
        };
        for length in [15, 16, 17, 19, 32] {
            // Warm both paths at each shape; counterbalance subsequent measured order.
            let _ = run(length, false, None);
            let _ = run(length, true, None);
            for iteration in 0..repeats {
                let (reference, candidate) = if iteration % 2 == 0 {
                    let candidate = run(length, true, Some(iteration));
                    (run(length, false, Some(iteration)), candidate)
                } else {
                    let reference = run(length, false, Some(iteration));
                    (reference, run(length, true, Some(iteration)))
                };
                for (stage, left, right) in [
                    ("prefill", &reference.0, &candidate.0),
                    ("decode8", &reference.1, &candidate.1),
                ] {
                    let logits = cosine(&left.0, &right.0);
                    let hidden = cosine(&left.1, &right.1);
                    eprintln!(
                        "QWEN35_PREFILL_PARITY length={length} iteration={iteration} stage={stage} logits={logits:.9} hidden={hidden:.9}"
                    );
                    assert!(logits >= 0.999 && hidden >= 0.999);
                }
                assert_eq!(
                    reference.2, candidate.2,
                    "batch route changed greedy IDs at length {length}"
                );
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_block_gemv_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_GEMV_BLOCKS");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_submission_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_BATCH_SUBMIT");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_rms_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_QWEN35_FUSED_RMS");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_encoder_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_BATCH_ENCODER");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_convolution_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_QWEN35_FUSED_CONV");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "exact model parity requires an exclusive Vulkan GPU window"]
    fn installed_artifact_vulkan_compute_pass_hidden_and_token_parity() {
        assert_vulkan_artifact_routes_match("AKH_VULKAN_BATCH_PASS");
    }

    #[cfg(all(feature = "hip", feature = "vulkan"))]
    #[test]
    #[ignore = "exact cross-backend parity requires an exclusive HIP/Vulkan GPU window"]
    fn installed_artifact_vulkan_matches_hip_f32_hidden_and_tokens() {
        let _route_lock = Q8_ROUTE_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL")
            .expect("set AKH_QWEN35_TEST_MODEL to the exact Ptah GGUF");
        assert_ptah_model_fingerprint(Path::new(&path));
        // Compare the independent f32 HIP arithmetic route before composing
        // lossy Q8 activation quantization or experimental recurrence fusion.
        let _q8 = EnvOverride::new("AKH_HIP_Q8_GEMV", "0");
        let _fusion = EnvOverride::new("AKH_HIP_QWEN35_FUSED_DELTA", "0");
        // Vulkan releases its device-owned allocations on drop. HIP's current
        // process-global allocator can retain freed blocks, so run HIP last.
        let candidate = artifact_hidden_trace(Path::new(&path), Device::new_vulkan(0).unwrap());
        let reference = artifact_hidden_trace(Path::new(&path), Device::new_cuda(0).unwrap());
        assert_artifact_traces_match("VULKAN_VS_HIP_F32", false, &reference, &candidate);
    }

    #[cfg(all(feature = "hip", feature = "vulkan"))]
    #[test]
    #[ignore = "exact-model session isolation requires an exclusive GPU window"]
    fn installed_artifact_owned_sessions_match_serial_execution() {
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL").expect("exact Ptah GGUF required");
        assert_ptah_model_fingerprint(Path::new(&path));
        let backend =
            std::env::var("AKH_QWEN35_SESSION_DEVICE").expect("explicit hip or vulkan required");
        let device = match backend.as_str() {
            "hip" => Device::new_cuda(0).unwrap(),
            "vulkan" => Device::new_vulkan(0).unwrap(),
            _ => panic!("unsupported session test device"),
        };
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let model = InferenceSession::from_gguf(content, &mut reader, &device).unwrap();
        let prompts: [&[u32]; 2] = [&[9_419, 1_814, 1], &[1_814, 9_419, 2, 1, 3]];
        let token_at =
            |step: usize| Tensor::new(&[[[9_419u32, 1_814, 1, 2][step % 4]]], &device).unwrap();
        let bits = |pair: (Tensor, Tensor)| {
            [pair.0, pair.1].map(|tensor| {
                tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(|value| {
                        assert!(value.is_finite());
                        value.to_bits()
                    })
                    .collect::<Vec<_>>()
            })
        };
        let reference = prompts.map(|prompt| {
            let mut session = model.new_session();
            let input = Tensor::from_slice(prompt, (1, prompt.len()), &device).unwrap();
            let mut trace = vec![bits(session.prefill_with_hidden(&input).unwrap())];
            for step in 0..32 {
                trace.push(bits(
                    session
                        .decode_step_with_hidden(&token_at(step), prompt.len() + step)
                        .unwrap(),
                ));
            }
            trace
        });
        let mut sessions = [model.new_session(), model.new_session()];
        for (index, session) in sessions.iter_mut().enumerate() {
            let input =
                Tensor::from_slice(prompts[index], (1, prompts[index].len()), &device).unwrap();
            assert_eq!(
                bits(session.prefill_with_hidden(&input).unwrap()),
                reference[index][0],
                "prefill session {index}"
            );
        }
        assert!(sessions[0].rewind(sessions[1].checkpoint()).is_err());
        for step in 0..32 {
            for (index, session) in sessions.iter_mut().enumerate() {
                assert_eq!(
                    bits(
                        session
                            .decode_step_with_hidden(&token_at(step), prompts[index].len() + step)
                            .unwrap()
                    ),
                    reference[index][step + 1],
                    "session {index}, decode {step}"
                );
            }
        }
        eprintln!(
            "OWNED_SESSION_PARITY device={backend} sessions=2 positions=33 hidden_and_logits=bit_exact"
        );
    }

    #[cfg(feature = "vulkan")]
    fn artifact_hidden_trace(path: &Path, device: Device) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut reader, &device).unwrap();
        let prompt = Tensor::new(&[[9_419u32, 1_814, 1]], &device).unwrap();
        let mut pair = model.prefill_with_hidden(&prompt).unwrap();
        let mut steps = Vec::new();
        for step in 0..=32 {
            steps.push((
                pair.0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                pair.1.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            ));
            if step < 32 {
                let token = Tensor::new(&[[[9_419u32, 1_814, 1, 2][step % 4]]], &device).unwrap();
                pair = model.decode_step_with_hidden(&token, 3 + step).unwrap();
            }
        }
        steps
    }

    #[cfg(all(feature = "hip", feature = "vulkan"))]
    #[test]
    #[ignore = "exact-model continuation requires an exclusive GPU window"]
    fn installed_artifact_text_continuation_matches_fresh_prefill() {
        if std::env::var("AKH_QWEN35_SESSION_DEVICE").as_deref() == Ok("hip")
            && std::env::var("AKH_QWEN35_RAW_Q8_CONTINUATION").as_deref() != Ok("1")
        {
            eprintln!("TEXT_CONTINUATION activation_policy=scoped_f32");
            candle_core::cuda::with_full_precision_activations(text_continuation_gate);
        } else {
            text_continuation_gate();
        }
    }

    #[cfg(all(feature = "hip", feature = "vulkan"))]
    fn text_continuation_gate() {
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL").expect("exact Ptah GGUF required");
        assert_ptah_model_fingerprint(Path::new(&path));
        let backend =
            std::env::var("AKH_QWEN35_SESSION_DEVICE").expect("explicit hip or vulkan required");
        let device = match backend.as_str() {
            "hip" => Device::new_cuda(0).unwrap(),
            "vulkan" => Device::new_vulkan(0).unwrap(),
            _ => panic!("unsupported session test device"),
        };
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut continued = InferenceSession::from_gguf(content, &mut reader, &device).unwrap();
        let mut fresh = continued.new_session();
        let mut tokens = vec![9_419u32, 1_814, 1];
        continued.resume_prompt(&tokens, &mut || Ok(())).unwrap();
        for token in [9_419, 1_814, 1, 2, 1_814, 9_419, 2, 1] {
            continued.forward_token(token, &mut || Ok(())).unwrap();
            tokens.push(token);
        }
        let prefix_len = tokens.len();
        tokens.extend([2, 9_419, 1_814, 1]);
        let (actual, reused) = continued.resume_prompt(&tokens, &mut || Ok(())).unwrap();
        assert_eq!(reused, prefix_len);
        let expected = fresh
            .prefill(&Tensor::from_slice(&tokens, (1, tokens.len()), &device).unwrap())
            .unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&a, &b) in expected.iter().zip(&actual) {
            assert!(a.is_finite() && b.is_finite());
            dot += a as f64 * b as f64;
            aa += (a as f64).powi(2);
            bb += (b as f64).powi(2);
        }
        let cosine = dot / (aa * bb).sqrt();
        assert!(cosine >= 0.9999, "continuation prompt logits: {cosine}");
        let greedy = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(greedy(&expected), greedy(&actual));
        eprintln!(
            "TEXT_CONTINUATION device={backend} reused={reused} prompt_logits_cosine={cosine:.9}"
        );
        let mut reference_trace = Vec::new();
        let mut continued_trace = Vec::new();
        for step in 0..33 {
            let token = Tensor::new(&[[[9_419u32, 1_814, 1, 2][step % 4]]], &device).unwrap();
            for (session, trace) in [
                (&mut fresh, &mut reference_trace),
                (&mut continued, &mut continued_trace),
            ] {
                let (logits, hidden) = session
                    .decode_step_with_hidden(&token, tokens.len() + step)
                    .unwrap();
                trace.push((
                    logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                ));
            }
        }
        assert_artifact_traces_match(
            "TEXT_CONTINUATION",
            false,
            &reference_trace,
            &continued_trace,
        );
    }

    #[test]
    #[ignore = "requires exact Ptah model and exclusive GPU access"]
    #[cfg(all(feature = "hip", feature = "vulkan"))]
    fn installed_artifact_same_pass_taps_preserve_target_state() {
        let run = || {
            let path = std::env::var_os("AKH_QWEN35_TEST_MODEL").expect("exact Ptah GGUF required");
            assert_ptah_model_fingerprint(Path::new(&path));
            let backend = std::env::var("AKH_QWEN35_SESSION_DEVICE").unwrap();
            let device = match backend.as_str() {
                "hip" => Device::new_cuda(0).unwrap(),
                "vulkan" => Device::new_vulkan(0).unwrap(),
                _ => panic!("explicit hip or vulkan required"),
            };
            let mut reader = std::fs::File::open(path).unwrap();
            let content = gguf_file::Content::read(&mut reader).unwrap();
            let mut tapped = InferenceSession::from_gguf(content, &mut reader, &device).unwrap();
            let mut plain = tapped.new_session();
            let tokens = [9_419u32, 1_814, 1, 2, 9_419];
            let capture = tapped
                .resume_prompt_with_taps(&tokens, 1..5, &[5, 19, 33, 47, 61], &mut || Ok(()))
                .unwrap();
            assert_eq!(capture.reused_tokens, 0);
            let taps = capture.taps.unwrap();
            assert_eq!(taps.len(), 5);
            for (tap, layer) in taps.iter().zip([5, 19, 33, 47, 61]) {
                assert_eq!(tap.layer, layer);
                assert_eq!(tap.mean.dims(), &[5120]);
                let values = tap.mean.to_vec1::<f32>().unwrap();
                assert!(values.iter().all(|value| value.is_finite()));
                assert!(values.iter().any(|value| *value != 0.0));
            }
            let plain_logits = plain.resume_prompt(&tokens, &mut || Ok(())).unwrap().0;
            assert_eq!(
                capture
                    .logits
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                plain_logits
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            let mut reference = Vec::new();
            let mut candidate = Vec::new();
            for step in 0..33 {
                let token = Tensor::new(&[[[9_419u32, 1_814, 1, 2][step % 4]]], &device).unwrap();
                for (session, trace) in
                    [(&mut plain, &mut reference), (&mut tapped, &mut candidate)]
                {
                    let (logits, hidden) = session
                        .decode_step_with_hidden(&token, tokens.len() + step)
                        .unwrap();
                    trace.push((
                        logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    ));
                }
            }
            assert_artifact_traces_match("SAME_PASS_TAPS", true, &reference, &candidate);
            eprintln!(
                "SAME_PASS_TAPS device={backend} layers=5 pooled_width=5120 state_parity=bit_exact"
            );
        };
        if std::env::var("AKH_QWEN35_SESSION_DEVICE").as_deref() == Ok("hip") {
            candle_core::cuda::with_full_precision_activations(run);
        } else {
            run();
        }
    }

    #[cfg(feature = "vulkan")]
    fn assert_vulkan_artifact_routes_match(selector: &'static str) {
        let _route_lock = Q8_ROUTE_ENV.lock().unwrap_or_else(|p| p.into_inner());
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL")
            .expect("set AKH_QWEN35_TEST_MODEL to the exact Ptah GGUF");
        assert_ptah_model_fingerprint(Path::new(&path));
        let run = |enabled: bool| {
            let _route = EnvOverride::new(selector, if enabled { "1" } else { "0" });
            // A device captures submission policy at construction. Load routes
            // sequentially so two full models never need to fit in VRAM.
            let device = Device::new_vulkan(0).unwrap();
            artifact_hidden_trace(Path::new(&path), device)
        };
        let reference = run(false);
        let candidate = run(true);
        let requires_exact = matches!(
            selector,
            "AKH_VULKAN_BATCH_SUBMIT" | "AKH_VULKAN_BATCH_ENCODER" | "AKH_VULKAN_BATCH_PASS"
        );
        assert_artifact_traces_match(selector, requires_exact, &reference, &candidate);
    }

    #[cfg(feature = "vulkan")]
    fn assert_artifact_traces_match(
        selector: &str,
        requires_exact: bool,
        reference: &[(Vec<f32>, Vec<f32>)],
        candidate: &[(Vec<f32>, Vec<f32>)],
    ) {
        assert_eq!(reference.len(), 33);
        assert_eq!(candidate.len(), reference.len());
        let cosine = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in a.iter().zip(b) {
                assert!(a.is_finite() && b.is_finite());
                dot += a as f64 * b as f64;
                aa += (a as f64).powi(2);
                bb += (b as f64).powi(2);
            }
            dot / (aa * bb).sqrt()
        };
        let greedy = |values: &[f32]| {
            values.iter().enumerate().fold(
                0,
                |best, (i, value)| {
                    if *value > values[best] { i } else { best }
                },
            )
        };
        let mut failures = Vec::new();
        for (step, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
            let logits = cosine(&reference.0, &candidate.0);
            let hidden = cosine(&reference.1, &candidate.1);
            let token_matches = greedy(&reference.0) == greedy(&candidate.0);
            let bit_exact = reference
                .0
                .iter()
                .chain(&reference.1)
                .zip(candidate.0.iter().chain(&candidate.1))
                .all(|(a, b)| a.to_bits() == b.to_bits());
            eprintln!(
                "QWEN35_VULKAN_PARITY selector={selector} step={step} logits={logits:.9} hidden={hidden:.9} token_matches={token_matches} bit_exact={bit_exact}"
            );
            if !(logits >= 0.9999 && hidden >= 0.9999 && token_matches)
                || (requires_exact && !bit_exact)
            {
                failures.push(step);
            }
        }
        assert!(
            failures.is_empty(),
            "{selector} parity failed at positions {failures:?}"
        );
    }

    #[cfg(feature = "hip")]
    #[test]
    #[ignore = "exact model parity requires an exclusive HIP GPU window"]
    fn installed_artifact_fused_delta_hidden_and_token_parity() {
        let _route_lock = Q8_ROUTE_ENV
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let path = std::env::var_os("AKH_QWEN35_TEST_MODEL")
            .expect("set AKH_QWEN35_TEST_MODEL to the exact Ptah GGUF");
        assert_ptah_model_fingerprint(Path::new(&path));
        let _batch = EnvOverride::new("AKH_GEMV_BATCH", "1");
        let device = Device::new_cuda(0).unwrap();
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut reader, &device).unwrap();
        let vector = |tensor: &Tensor| tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut run = |fused: bool| {
            let _route =
                EnvOverride::new("AKH_HIP_QWEN35_FUSED_DELTA", if fused { "1" } else { "0" });
            let prompt = Tensor::new(&[[9_419u32, 1_814, 1]], &device).unwrap();
            let mut pair = model.prefill_with_hidden(&prompt).unwrap();
            let mut steps = Vec::new();
            for step in 0..=32 {
                steps.push((vector(&pair.0), vector(&pair.1)));
                if step < 32 {
                    let token =
                        Tensor::new(&[[[9_419u32, 1_814, 1, 2][step % 4]]], &device).unwrap();
                    pair = model.decode_step_with_hidden(&token, 3 + step).unwrap();
                }
            }
            steps
        };
        let cosine = |a: &[f32], b: &[f32]| {
            let (mut dot, mut aa, mut bb) = (0., 0., 0.);
            for (&a, &b) in a.iter().zip(b) {
                dot += a as f64 * b as f64;
                aa += (a as f64).powi(2);
                bb += (b as f64).powi(2);
            }
            dot / (aa * bb).sqrt()
        };
        let greedy = |logits: &[f32]| {
            logits.iter().enumerate().fold(
                0,
                |best, (index, value)| {
                    if *value > logits[best] { index } else { best }
                },
            )
        };
        let mut failures = Vec::new();
        // Change only activation quantization between cases. Repeating the
        // unfused run also catches nondeterminism or incomplete state reset.
        // Keep the original accuracy threshold, but collect every position so
        // the first failure does not hide later drift or greedy-token changes.
        for q8 in [false, true] {
            let _q8 = EnvOverride::new("AKH_HIP_Q8_GEMV", if q8 { "1" } else { "0" });
            let reference = run(false);
            let repeated = run(false);
            let candidate = run(true);
            let mut minimum_logits = 1.0f64;
            let mut minimum_hidden = 1.0f64;
            let mut worst_hidden_step = 0;
            let mut greedy_mismatches = Vec::new();
            let mut repeat_mismatches = Vec::new();
            for (step, ((reference, repeated), candidate)) in
                reference.iter().zip(&repeated).zip(&candidate).enumerate()
            {
                if reference != repeated {
                    repeat_mismatches.push(step);
                }
                let logits = cosine(&reference.0, &candidate.0);
                let hidden = cosine(&reference.1, &candidate.1);
                minimum_logits = minimum_logits.min(logits);
                if hidden < minimum_hidden {
                    minimum_hidden = hidden;
                    worst_hidden_step = step;
                }
                if greedy(&reference.0) != greedy(&candidate.0) {
                    greedy_mismatches.push(step);
                }
                if !(logits >= 0.9999 && hidden >= 0.9999) {
                    failures.push(format!(
                        "q8={q8} step={step} logits={logits:.9} hidden={hidden:.9}"
                    ));
                }
            }
            eprintln!(
                "QWEN35_FUSED_DELTA_PARITY q8={q8} minimum_logits={minimum_logits:.9} minimum_hidden={minimum_hidden:.9} worst_hidden_step={worst_hidden_step} greedy_mismatches={greedy_mismatches:?} repeat_mismatches={repeat_mismatches:?}"
            );
            if !repeat_mismatches.is_empty() || !greedy_mismatches.is_empty() {
                failures.push(format!(
                    "q8={q8} greedy={greedy_mismatches:?} repeat={repeat_mismatches:?}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "fused recurrence parity failures: {}",
            failures.join("; ")
        );
        // This diagnostic deliberately transfers full vectors each step. Its wall
        // time is not a throughput measurement; use matched native generation.
    }

    #[cfg(feature = "hip")]
    #[test]
    fn installed_artifact_q8_q4k_hidden_and_token_parity() {
        let _route_lock = Q8_ROUTE_ENV
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(path) = std::env::var_os("AKH_QWEN35_TEST_MODEL") else {
            return;
        };
        assert_ptah_model_fingerprint(Path::new(&path));
        let device = Device::new_cuda(0).unwrap();
        let mut reader = std::fs::File::open(path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let mut model = ModelWeights::from_gguf(content, &mut reader, &device).unwrap();
        let hello = Tensor::new(&[[9_419u32]], &device).unwrap();
        model.prefill(&hello).unwrap();
        let checkpoint = model.checkpoint();
        // `world` followed by two forced continuation tokens exercises recurrent/KV
        // accumulation instead of proving only an isolated one-step forward.
        let continuation = [1_814u32, 1, 1];

        let mut run = |q8: bool| {
            model.rewind(checkpoint.clone()).unwrap();
            let _route = EnvOverride::new("AKH_HIP_Q8_GEMV", if q8 { "1" } else { "0" });
            let started = std::time::Instant::now();
            let mut steps = Vec::with_capacity(continuation.len());
            for (offset, token) in continuation.into_iter().enumerate() {
                let token = Tensor::new(&[[token]], &device).unwrap();
                let (logits, hidden) = model.decode_step_with_hidden(&token, 1 + offset).unwrap();
                let logits = logits
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let hidden = hidden
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                steps.push((logits, hidden));
            }
            let elapsed = started.elapsed();
            (steps, elapsed)
        };

        // Warm both routes before measuring, then alternate order so allocation-pool and
        // GPU-clock ramp do not masquerade as a kernel speedup.
        let _ = run(false);
        let _ = run(true);
        let mut reference_times = Vec::new();
        let mut q8_times = Vec::new();
        let mut parity_pair = None;
        for iteration in 0..12 {
            let (reference, q8) = if iteration % 2 == 0 {
                let q8 = run(true);
                let reference = run(false);
                (reference, q8)
            } else {
                let reference = run(false);
                let q8 = run(true);
                (reference, q8)
            };
            reference_times.push(reference.1);
            q8_times.push(q8.1);
            parity_pair = Some((reference, q8));
        }
        reference_times.sort_unstable();
        q8_times.sort_unstable();
        let middle = reference_times.len() / 2;
        let reference_elapsed = (reference_times[middle - 1] + reference_times[middle]) / 2;
        let q8_elapsed = (q8_times[middle - 1] + q8_times[middle]) / 2;
        let ((reference_steps, _), (q8_steps, _)) = parity_pair.unwrap();
        let greedy = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .unwrap()
        };
        let cosine = |left_values: &[f32], right_values: &[f32]| {
            let (mut dot, mut left_norm, mut right_norm) = (0.0f64, 0.0f64, 0.0f64);
            for (&left, &right) in left_values.iter().zip(right_values) {
                let (left, right) = (left as f64, right as f64);
                dot += left * right;
                left_norm += left * left;
                right_norm += right * right;
            }
            dot / (left_norm.sqrt() * right_norm.sqrt()).max(f64::MIN_POSITIVE)
        };
        let mut hidden_cosine = 1.0f64;
        let mut logit_cosine = 1.0f64;
        let mut reference_tokens = Vec::with_capacity(reference_steps.len());
        let mut q8_tokens = Vec::with_capacity(q8_steps.len());
        for ((reference_logits, reference_hidden), (q8_logits, q8_hidden)) in
            reference_steps.iter().zip(&q8_steps)
        {
            hidden_cosine = hidden_cosine.min(cosine(reference_hidden, q8_hidden));
            logit_cosine = logit_cosine.min(cosine(reference_logits, q8_logits));
            reference_tokens.push(greedy(reference_logits));
            q8_tokens.push(greedy(q8_logits));
        }
        eprintln!(
            "exact Qwen3.8 Q8/Q4_K: hidden cosine={hidden_cosine:.9}, \
             logit cosine={logit_cosine:.9}, greedy={q8_tokens:?}, \
             f32={:.3}s q8={:.3}s speedup={:.3}x",
            reference_elapsed.as_secs_f64(),
            q8_elapsed.as_secs_f64(),
            reference_elapsed.as_secs_f64() / q8_elapsed.as_secs_f64()
        );
        assert!(
            hidden_cosine >= 0.999,
            "Q8/Q4_K exact-model hidden cosine {hidden_cosine} is below the acceptance gate"
        );
        assert!(
            logit_cosine >= 0.999,
            "Q8/Q4_K exact-model logit cosine {logit_cosine} is below the acceptance gate"
        );
        assert_eq!(
            q8_tokens, reference_tokens,
            "Q8/Q4_K changed an exact-model greedy token after recurrent-state accumulation"
        );
        assert_eq!(
            q8_tokens[0], 0,
            "Q8/Q4_K must preserve llama.cpp's `Hello world` oracle token"
        );
    }
}
