//! Reference Qwen3.5/Qwen3.8 hybrid decoder for quantized GGUF artifacts.
//!
//! This module deliberately starts at the artifact seam: metadata is converted
//! into a strongly typed configuration and the complete target tensor schema is
//! validated before any multi-gigabyte weight is read.

use std::collections::HashMap;

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

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
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
        if device.is_cuda() {
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
        let rotated = if x.device().is_cuda() {
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
    key: Tensor,
    value: Tensor,
}

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
    state: Option<AttentionState>,
}

impl FullAttention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
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

        let (key, value) = match &self.state {
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
        self.state = Some(AttentionState {
            key: key.clone(),
            value: value.clone(),
        });

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
    state: Option<GatedDeltaState>,
}

fn causal_depthwise_conv(
    input: &Tensor,
    kernel: &Tensor,
    history: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
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
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        if index_pos > 0 && self.state.is_none() {
            candle_core::bail!(
                "Qwen35 recurrent mixer has no state to append at position {index_pos}"
            )
        }
        if let Some(state) = &self.state {
            if index_pos > 0 && state.position != index_pos {
                candle_core::bail!(
                    "Qwen35 recurrent state is at position {}, cannot append at {index_pos}",
                    state.position
                )
            }
        }
        let prior = if index_pos == 0 {
            None
        } else {
            self.state.as_ref()
        };
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
        self.state = Some(GatedDeltaState {
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
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        match self {
            Self::Attention(attention) => attention.forward(x, mask, index_pos),
            Self::GatedDelta(gated_delta) => gated_delta.forward(x, index_pos),
        }
    }

    fn state(&self) -> MixerState {
        match self {
            Self::Attention(attention) => MixerState::Attention(attention.state.clone()),
            Self::GatedDelta(gated_delta) => MixerState::GatedDelta(gated_delta.state.clone()),
        }
    }

    fn restore(&mut self, state: MixerState) -> Result<()> {
        match (self, state) {
            (Self::Attention(attention), MixerState::Attention(state)) => attention.state = state,
            (Self::GatedDelta(gated_delta), MixerState::GatedDelta(state)) => {
                gated_delta.state = state
            }
            _ => candle_core::bail!("Qwen35 checkpoint layer kind does not match model"),
        }
        Ok(())
    }

    fn reset(&mut self) {
        match self {
            Self::Attention(attention) => attention.state = None,
            Self::GatedDelta(gated_delta) => gated_delta.state = None,
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
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let residual = x;
        let normalized = self.input_norm.forward(x)?;
        let mixed = self.mixer.forward(&normalized, mask, index_pos)?;
        let x = (residual + mixed)?;
        let mlp = self.mlp.forward(&self.post_attention_norm.forward(&x)?)?;
        x + mlp
    }

    fn state(&self) -> MixerState {
        self.mixer.state()
    }

    fn restore(&mut self, state: MixerState) -> Result<()> {
        self.mixer.restore(state)
    }

    fn reset(&mut self) {
        self.mixer.reset();
    }
}

/// Snapshot of all mutable target-model state. Immutable weights are shared by
/// the owning model and deliberately absent.
#[derive(Debug, Clone)]
pub(crate) struct Checkpoint {
    layers: Vec<MixerState>,
}

pub(crate) struct ModelWeights {
    embedding: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
}

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
            state: None,
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
            state: None,
        })
    };
    Ok(Layer {
        input_norm,
        post_attention_norm,
        mixer,
        mlp,
    })
}

impl ModelWeights {
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
        Ok(Self {
            embedding,
            layers,
            norm,
            output,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            return Ok(mask.clone());
        }
        let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
        self.masks.insert((seq_len, kv_len), mask.clone());
        Ok(mask)
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            layers: self.layers.iter().map(Layer::state).collect(),
        }
    }

    pub(crate) fn rewind(&mut self, checkpoint: Checkpoint) -> Result<()> {
        if checkpoint.layers.len() != self.layers.len() {
            candle_core::bail!(
                "Qwen35 checkpoint has {} layers, model has {}",
                checkpoint.layers.len(),
                self.layers.len()
            )
        }
        for (layer, state) in self.layers.iter_mut().zip(checkpoint.layers) {
            layer.restore(state)?;
        }
        Ok(())
    }

    fn run_hidden_states(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len) = tokens.dims2()?;
        if index_pos == 0 {
            self.reset();
        }
        let mask = if seq_len > 1 {
            Some(self.mask(seq_len, index_pos, tokens.device())?)
        } else {
            None
        };
        let mut hidden = self.embedding.forward(tokens)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, mask.as_ref(), index_pos)?;
        }
        self.norm.forward(&hidden)?.i((.., seq_len - 1, ..))
    }

    fn run_with_hidden(&mut self, tokens: &Tensor, index_pos: usize) -> Result<(Tensor, Tensor)> {
        let hidden = self.run_hidden_states(tokens, index_pos)?;
        let logits = self.output.forward(&hidden)?;
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
        let (_, seq_len) = tokens.dims2()?;
        if seq_len == 0 {
            candle_core::bail!("Qwen35 prefill requires at least one token")
        }
        self.run_with_hidden(tokens, 0)
    }

    fn decode_step_with_hidden(
        &mut self,
        token: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len) = token.dims2()?;
        if seq_len != 1 || index_pos == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Shape, quantized::GgmlDType};
    use sha2::{Digest, Sha256};
    use std::ffi::OsString;
    use std::io::{BufReader, Cursor, Read};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

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

    fn tiny_metadata() -> Vec<(String, gguf_file::Value)> {
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

    fn quantized(values: Vec<f32>, shape: &[usize]) -> QTensor {
        let tensor = Tensor::from_vec(values, Shape::from_dims(shape), &Device::Cpu).unwrap();
        QTensor::quantize(&tensor, GgmlDType::F32).unwrap()
    }

    fn push_tensor(
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
    fn tiny_gguf() -> Vec<u8> {
        let mut tensors = Vec::new();
        let mut embeddings = vec![0.0f32; 16 * 8];
        embeddings[8..16].fill(1.0);
        tensors.push(("token_embd.weight".into(), quantized(embeddings, &[16, 8])));
        push_tensor(&mut tensors, "output_norm.weight", &[8], 1.0);
        let mut output = vec![0.0f32; 16 * 8];
        for index in 0..8 {
            output[index * 8 + index] = 1.0;
        }
        tensors.push(("output.weight".into(), quantized(output, &[16, 8])));

        for layer in 0..4 {
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
            if layer < 3 {
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

        let prefix = "blk.4";
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

        let metadata = tiny_metadata();
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

    fn load_tiny_model(bytes: &[u8]) -> ModelWeights {
        let mut reader = Cursor::new(bytes);
        let content = gguf_file::Content::read(&mut reader).unwrap();
        ModelWeights::from_gguf(content, &mut reader, &Device::Cpu).unwrap()
    }

    fn make_recurrent_state_observable(model: &mut ModelWeights) {
        for (layer_index, layer) in model.layers.iter_mut().enumerate() {
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
            let mut layer = load_layer(
                &content,
                &mut reader,
                &config,
                &rope,
                &neg_inf,
                layer_index,
                &device,
            )
            .unwrap();
            hidden = layer.forward(&hidden, Some(&mask), 0).unwrap();
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
