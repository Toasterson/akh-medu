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

/// Position index for every rotary pair under a multimodal RoPE split.
///
/// Qwen-VL gives each token three indices — temporal, height and width — and
/// assigns a contiguous run of rotary pairs to each. `sections` lists those runs
/// in pairs. Any pair beyond the declared sections keeps the temporal index,
/// which is what a text-only model would have used throughout.
///
/// With `t == h == w`, every pair gets the same index and the result is
/// indistinguishable from ordinary 1-D RoPE. That identity is why the current
/// text-only forward pass is correct while ignoring the sections entirely.
pub(crate) fn mrope_pair_positions(
    sections: &[usize],
    pairs: usize,
    t: usize,
    h: usize,
    w: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(pairs);
    let indices = [t, h, w];
    for (section, width) in sections.iter().enumerate() {
        let index = indices.get(section).copied().unwrap_or(t);
        for _ in 0..*width {
            if out.len() == pairs {
                return out;
            }
            out.push(index);
        }
    }
    while out.len() < pairs {
        out.push(t);
    }
    out
}

/// Which rope positions a forward pass uses.
///
/// Text advances one index per token and every rotary pair shares it. Image
/// tokens do not: the temporal, height and width sections advance independently
/// over the patch grid, so a token's pairs take different positions depending on
/// which section they fall in.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RopePositions<'a> {
    /// `seq_len` consecutive indices from this one, all pairs alike.
    Contiguous(usize),
    /// A `(temporal, height, width)` triple per token.
    PerToken(&'a [[usize; 3]]),
}

#[derive(Debug, Clone)]
struct PartialRope {
    cos: Tensor,
    sin: Tensor,
    width: usize,
    /// Rotary pairs per section, empty when the artifact does not declare a
    /// usable split. Empty means image positions cannot be threaded, which is a
    /// refusal rather than a silent fallback to text positions.
    sections: Vec<usize>,
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
            sections: config.rope_sections.clone(),
        })
    }

    /// Gather one table row per token, each pair taking its section's index.
    ///
    /// `index_select` over the flattened table, which both device backends
    /// implement, so this needs no new kernel. Text never reaches it.
    fn gathered(&self, positions: &[[usize; 3]], pairs: usize) -> Result<(Tensor, Tensor)> {
        // A triple with t == h == w places a token wherever the split falls, so
        // it needs no split. Refuse only a genuine image layout on an artifact
        // that cannot describe one, which is the case that would silently place
        // image tokens as text.
        if self.sections.is_empty()
            && positions
                .iter()
                .any(|[t, h, w]| t != h || h != w)
        {
            candle_core::bail!(
                "this artifact declares no usable rope dimension sections, so image \
                 positions cannot be threaded; text positions would silently mis-place them"
            )
        }
        let rows = self.cos.dim(0)?;
        let mut ids = Vec::with_capacity(positions.len() * pairs);
        for [t, h, w] in positions {
            for (pair, index) in mrope_pair_positions(&self.sections, pairs, *t, *h, *w)
                .into_iter()
                .enumerate()
            {
                if index >= rows {
                    candle_core::bail!("rope position {index} beyond the {rows}-row table")
                }
                ids.push((index * pairs + pair) as u32);
            }
        }
        let device = self.cos.device();
        let ids = Tensor::from_vec(ids, (positions.len() * pairs,), device)?;
        let take = |table: &Tensor| -> Result<Tensor> {
            table
                .flatten_all()?
                .index_select(&ids, 0)?
                .reshape((positions.len(), pairs))
        };
        Ok((take(&self.cos)?, take(&self.sin)?))
    }

    fn apply(&self, x: &Tensor, positions: RopePositions<'_>) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let head_dim = x.dim(3)?;
        let rotated = x.narrow(3, 0, self.width)?.contiguous()?;
        let (cos, sin) = match positions {
            RopePositions::Contiguous(index_pos) => (
                self.cos.narrow(0, index_pos, seq_len)?,
                self.sin.narrow(0, index_pos, seq_len)?,
            ),
            RopePositions::PerToken(triples) => {
                if triples.len() != seq_len {
                    candle_core::bail!(
                        "{} rope position triples for {seq_len} tokens",
                        triples.len()
                    )
                }
                self.gathered(triples, self.width / 2)?
            }
        };
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

/// How codes are laid out in device memory.
///
/// The Vulkan allocator hands out four bytes per element for every dtype it
/// supports, so a `u8` code tensor there costs exactly what the f32 cache it
/// replaces costs and the compression is nominal only. Packing three codes into
/// each `u32` word restores a real 3x. Three and not four because the unpack
/// runs through f32, which represents integers exactly only up to `2^24`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeLayout {
    /// One code per element, `u8`. What HIP and the CPU backend already give.
    Byte,
    /// Three codes per `u32` word, packed along the head dimension.
    PackedTriple { words_per_row: usize },
}

impl CodeLayout {
    fn packed(dim: usize) -> Self {
        Self::PackedTriple {
            words_per_row: dim.div_ceil(3),
        }
    }

    /// The layout a device wants, unless `AKH_QWEN35_KV_PACK` overrides it.
    ///
    /// The override exists so the packing can be measured against the layout it
    /// replaces on the same backend; without it the two differ by device and the
    /// comparison confounds the packing with everything else that differs.
    fn for_device(device: &Device, dim: usize) -> Self {
        match std::env::var("AKH_QWEN35_KV_PACK").ok().as_deref() {
            Some("packed") => return Self::packed(dim),
            Some("byte") => return Self::Byte,
            _ => {}
        }
        #[cfg(feature = "vulkan")]
        if device.is_vulkan() {
            return Self::packed(dim);
        }
        let _ = (device, dim);
        Self::Byte
    }

    /// Bytes one code element occupies on this device.
    ///
    /// Vulkan gives every dtype it supports a full 32-bit word, so a `u8` code
    /// there costs four bytes and not one. Charging one unconditionally is what
    /// made the byte layout look like a 3.9x saving on a backend where it is
    /// actually a net loss.
    fn code_element_bytes(device: &Device) -> usize {
        #[cfg(feature = "vulkan")]
        if device.is_vulkan() {
            return 4;
        }
        let _ = device;
        std::mem::size_of::<u8>()
    }

    fn label(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::PackedTriple { .. } => "packed-triple",
        }
    }
}

/// How faithfully the live paged path reconstructs what it replaced.
///
/// Accumulated at the moment positions are absorbed, which is the only point
/// where the exact rows and their device decode both exist. Measuring here
/// means the figure covers the real codes, the real layout and the real device
/// arithmetic, rather than a host simulation of them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvPageFidelity {
    pub vectors: usize,
    pub min_cosine: f64,
    pub sum_cosine: f64,
    pub max_relative_l2: f64,
}

impl Default for KvPageFidelity {
    fn default() -> Self {
        Self {
            vectors: 0,
            min_cosine: f64::INFINITY,
            sum_cosine: 0.0,
            max_relative_l2: 0.0,
        }
    }
}

impl KvPageFidelity {
    fn merge(&mut self, other: &Self) {
        self.vectors += other.vectors;
        self.min_cosine = self.min_cosine.min(other.min_cosine);
        self.sum_cosine += other.sum_cosine;
        self.max_relative_l2 = self.max_relative_l2.max(other.max_relative_l2);
    }

    pub fn mean_cosine(&self) -> f64 {
        if self.vectors == 0 {
            f64::NAN
        } else {
            self.sum_cosine / self.vectors as f64
        }
    }
}

/// Measuring costs a device-to-host copy per absorbed chunk, so it is opt-in.
fn kv_page_fidelity_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AKH_QWEN35_KV_PAGE_FIDELITY")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

/// What compressed KV pages occupy right now, summed over attention layers.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KvPageResidency {
    pub layout: &'static str,
    pub stored_bytes: usize,
    pub uncompressed_bytes: usize,
    /// Paged positions summed over layers, so layers x positions per layer.
    pub positions: usize,
    pub layers: usize,
    /// Present only when `AKH_QWEN35_KV_PAGE_FIDELITY=1`.
    pub fidelity: Option<KvPageFidelity>,
}

/// Pack rows of codes three to a `u32` word, lowest lane in the low byte.
///
/// Rows are padded with zero codes up to `words_per_row * 3`; the padding is
/// dropped again on the way out, so it costs at most the tail of one word.
fn pack_triples(codes: &[u8], dim: usize, words_per_row: usize) -> Vec<u32> {
    let mut packed = Vec::with_capacity((codes.len() / dim) * words_per_row);
    for row in codes.chunks_exact(dim) {
        for word in 0..words_per_row {
            let lane = |offset: usize| u32::from(row.get(word * 3 + offset).copied().unwrap_or(0));
            packed.push(lane(0) | (lane(1) << 8) | (lane(2) << 16));
        }
    }
    packed
}

/// Compressed older positions for one attention layer, resident on the device.
///
/// Codes are `u8`, one byte each, and the decode widens them for the gather
/// rather than storing them wide. Nothing returns to the host after the initial
/// quantization, which is what makes this usable in the decode loop: the
/// host-side version starved the GPU badly enough that a six-turn benchmark did
/// not finish in thirty minutes.
#[derive(Debug, Clone)]
struct KvPages {
    /// `(batch, heads, positions, dim)` of codebook indices.
    key_codes: Option<Tensor>,
    value_codes: Option<Tensor>,
    /// `(batch, heads, positions, 1)` full-precision norms.
    key_norms: Option<Tensor>,
    value_norms: Option<Tensor>,
    /// Shared across both tensors and every slot.
    codebook: Tensor,
    rotation: Tensor,
    quantizer: std::sync::Arc<crate::nlu::turbo_quant::TurboQuant>,
    layout: CodeLayout,
    /// What one code element costs on the device holding it.
    code_element_bytes: usize,
    /// Present only when fidelity measurement is switched on.
    fidelity: Option<KvPageFidelity>,
    positions: usize,
}

impl KvPages {
    fn new(policy: KvQuantization, dim: usize, device: &Device) -> Result<Self> {
        let quantizer = std::sync::Arc::new(crate::nlu::turbo_quant::TurboQuant::new(
            dim,
            policy.bits,
            KV_PAGE_SEED,
        ));
        let codebook = Tensor::from_vec(
            quantizer.codebook().to_vec(),
            (quantizer.codebook().len(),),
            device,
        )?;
        let rotation = Tensor::from_vec(quantizer.rotation().to_vec(), (dim, dim), device)?;
        Ok(Self {
            key_codes: None,
            value_codes: None,
            key_norms: None,
            value_norms: None,
            codebook,
            rotation,
            quantizer,
            layout: CodeLayout::for_device(device, dim),
            code_element_bytes: CodeLayout::code_element_bytes(device),
            fidelity: kv_page_fidelity_enabled().then(KvPageFidelity::default),
            positions: 0,
        })
    }

    fn positions(&self) -> usize {
        self.positions
    }

    /// Resident code bytes against the f32 they replace.
    fn stored_and_uncompressed_bytes(&self, heads: usize, dim: usize, batch: usize) -> (usize, usize) {
        let vectors = batch * heads * self.positions;
        let code_bytes = match self.layout {
            CodeLayout::Byte => vectors * dim * self.code_element_bytes,
            // A word is four bytes on every backend, three codes to a word.
            CodeLayout::PackedTriple { words_per_row } => {
                vectors * words_per_row * std::mem::size_of::<u32>()
            }
        };
        // Codes plus an f32 norm per vector, for key and value alike.
        let stored = 2 * (code_bytes + vectors * std::mem::size_of::<f32>());
        (stored, 2 * vectors * dim * std::mem::size_of::<f32>())
    }

    /// Quantize the oldest `count` positions of `tensor` and append them.
    ///
    /// Quantization stays on the host because it happens once per position as it
    /// ages out, not once per decode step. Only the decode is hot.
    fn absorb(&self, tensor: &Tensor, count: usize) -> Result<(Tensor, Tensor)> {
        let (batch, heads, positions, dim) = tensor.dims4()?;
        let values: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut codes = Vec::with_capacity(batch * heads * count * dim);
        let mut norms = Vec::with_capacity(batch * heads * count);
        for b in 0..batch {
            for head in 0..heads {
                let base = (b * heads + head) * positions * dim;
                for position in 0..count {
                    let row = &values[base + position * dim..base + (position + 1) * dim];
                    let (row_codes, norm) = self.quantizer.quantize_row(row);
                    codes.extend_from_slice(&row_codes);
                    norms.push(norm);
                }
            }
        }
        let device = tensor.device();
        let code_tensor = match self.layout {
            CodeLayout::Byte => Tensor::from_vec(codes, (batch, heads, count, dim), device)?,
            CodeLayout::PackedTriple { words_per_row } => Tensor::from_vec(
                pack_triples(&codes, dim, words_per_row),
                (batch, heads, count, words_per_row),
                device,
            )?,
        };
        Ok((
            code_tensor,
            Tensor::from_vec(norms, (batch, heads, count, 1), device)?,
        ))
    }

    /// Append freshly quantized pages to whatever is already held.
    fn extend(
        existing: &mut Option<Tensor>,
        addition: Tensor,
    ) -> Result<()> {
        *existing = Some(match existing.take() {
            None => addition,
            Some(held) => Tensor::cat(&[&held, &addition], 2)?.contiguous()?,
        });
        Ok(())
    }

    /// Recover one code per element from words that hold three each.
    ///
    /// Only `affine`, `floor` and the binary ops appear here, all of which every
    /// backend already implements, so the packed layout needs no new kernel. The
    /// arithmetic runs in f32 and is exact: the largest packed word is `2^24 - 1`
    /// and both divisions are by powers of two.
    fn unpack_triples(words: &Tensor, dim: usize) -> Result<Tensor> {
        let (batch, heads, positions, words_per_row) = words.dims4()?;
        let word = words.to_dtype(DType::F32)?;
        let above_low = word.affine(1.0 / 256.0, 0.0)?.floor()?;
        let high = word.affine(1.0 / 65_536.0, 0.0)?.floor()?;
        let low = (&word - above_low.affine(256.0, 0.0)?)?;
        let middle = (&above_low - high.affine(256.0, 0.0)?)?;
        // Interleaving is a stack on a new trailing axis: element `(word, lane)`
        // lands at `word * 3 + lane`, which is where it was packed from.
        let interleaved = Tensor::stack(&[&low, &middle, &high], 4)?.reshape((
            batch,
            heads,
            positions,
            words_per_row * 3,
        ))?;
        if words_per_row * 3 == dim {
            Ok(interleaved)
        } else {
            interleaved.narrow(3, 0, dim)?.contiguous()
        }
    }

    /// Decode compressed pages without leaving the device.
    ///
    /// Three primitives: gather the centroids the codes point at, scale each
    /// position by its stored norm, then apply the inverse rotation. The last is
    /// `rows * Pi` because reconstruction is `Pi^T y` and these rows are
    /// transposed relative to that.
    ///
    /// Codes are held one per element under [`CodeLayout::Byte`] and three per
    /// `u32` word under [`CodeLayout::PackedTriple`]; either way they are
    /// widened to the `u32` that `index_select` requires rather than stored
    /// wide, which is what makes the compression real rather than nominal.
    fn decode_on_device(
        codes: &Tensor,
        norms: &Tensor,
        codebook: &Tensor,
        rotation: &Tensor,
        layout: CodeLayout,
        batch: usize,
        heads: usize,
        positions: usize,
        dim: usize,
    ) -> Result<Tensor> {
        let ids = match layout {
            CodeLayout::Byte => codes.flatten_all()?.to_dtype(DType::U32)?,
            CodeLayout::PackedTriple { .. } => Self::unpack_triples(codes, dim)?
                .flatten_all()?
                .to_dtype(DType::U32)?,
        };
        codebook
            .index_select(&ids, 0)?
            .reshape((batch * heads * positions, dim))?
            .broadcast_mul(&norms.reshape((batch * heads * positions, 1))?)?
            .matmul(rotation)?
            .reshape((batch, heads, positions, dim))
    }

    /// Compare a freshly absorbed chunk against the exact rows it replaces.
    ///
    /// Runs the same `decode_on_device` the live path runs, so what is scored is
    /// the reconstruction attention will actually read, not an approximation of
    /// it. No-op unless measurement is switched on.
    fn record_fidelity(
        &mut self,
        exact: &Tensor,
        codes: &Tensor,
        norms: &Tensor,
        count: usize,
    ) -> Result<()> {
        if self.fidelity.is_none() || count == 0 {
            return Ok(());
        }
        let (batch, heads, _, dim) = exact.dims4()?;
        let decoded = Self::decode_on_device(
            codes,
            norms,
            &self.codebook,
            &self.rotation,
            self.layout,
            batch,
            heads,
            count,
            dim,
        )?;
        let want: Vec<f32> = exact
            .narrow(2, 0, count)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let got: Vec<f32> = decoded.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let stats = self
            .fidelity
            .as_mut()
            .expect("fidelity presence checked above");
        for (original, restored) in want.chunks_exact(dim).zip(got.chunks_exact(dim)) {
            let (mut dot, mut left, mut right, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (a, b) in original.iter().zip(restored) {
                let (a, b) = (f64::from(*a), f64::from(*b));
                dot += a * b;
                left += a * a;
                right += b * b;
                diff += (a - b) * (a - b);
            }
            // A zero row has no direction to agree about; skip rather than
            // score it as perfect or as total disagreement.
            if left == 0.0 || right == 0.0 {
                continue;
            }
            stats.vectors += 1;
            let cosine = dot / (left.sqrt() * right.sqrt());
            stats.min_cosine = stats.min_cosine.min(cosine);
            stats.sum_cosine += cosine;
            stats.max_relative_l2 = stats.max_relative_l2.max(diff.sqrt() / left.sqrt());
        }
        Ok(())
    }

    /// The stored keys and values, reconstructed.
    fn decode_pair(&self, batch: usize, heads: usize, dim: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(kc), Some(kn), Some(vc), Some(vn)) = (
            self.key_codes.as_ref(),
            self.key_norms.as_ref(),
            self.value_codes.as_ref(),
            self.value_norms.as_ref(),
        ) else {
            return Ok(None);
        };
        Ok(Some((
            Self::decode_on_device(
                kc,
                kn,
                &self.codebook,
                &self.rotation,
                self.layout,
                batch,
                heads,
                self.positions,
                dim,
            )?,
            Self::decode_on_device(
                vc,
                vn,
                &self.codebook,
                &self.rotation,
                self.layout,
                batch,
                heads,
                self.positions,
                dim,
            )?,
        )))
    }
}

/// Attend without materializing the GQA head expansion.
///
/// `repeat_kv` builds a full copy of K and V with every kv head repeated to
/// query-head count, on every forward, in every attention layer. Folding the
/// repeat into the row dimension instead - `(batch, kv_heads, ratio * seq, dim)`
/// against `(batch, kv_heads, dim, positions)` - is the same arithmetic as a
/// plain 4-D batched matmul, with nothing expanded.
///
/// The head ordering matches: `repeat_kv` emits head `kv * ratio + rep`, and
/// reshaping the grouped rows back to `(batch, heads, seq, ...)` puts row
/// `rep * seq + s` of group `kv` at exactly that head.
///
/// **Measured slower, and off for that reason.** It is numerically sound -
/// exactly equal to `repeat_kv` on CPU, element for element, and cosine
/// 1.0000000000 against the materializing path on the full model - but three
/// interleaved pairs at ~5,200 tokens put it **10.1% slower**, 184.8 s against
/// 167.8 s, with every materializing run beating every grouped run. Dropping the
/// batch dimension from query heads to kv-groups appears to cost more GEMM
/// parallelism than the avoided copy saves.
///
/// Kept because the result is counter-intuitive and will be proposed again: a
/// flag and a measurement are cheaper than reimplementing it to rediscover this.
fn grouped_attention_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AKH_QWEN35_GROUPED_ATTENTION")
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

/// Tokens forwarded per prefill step.
///
/// Each chunk re-expands the whole K and V cache through `repeat_kv` in every
/// attention layer, so the expansion traffic scales with
/// `(prompt / chunk) * positions`: halving the chunk count halves it. 128 was
/// the standing value; `AKH_QWEN35_PREFILL_CHUNK` makes it measurable.
fn prefill_chunk_size() -> usize {
    use std::sync::OnceLock;
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("AKH_QWEN35_PREFILL_CHUNK")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|value| *value > 0)
            .unwrap_or(128)
    })
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
        rope: Option<&[[usize; 3]]>,
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
        // Image tokens carry their own (t, h, w); text advances one index per
        // token. Absent triples mean text, never a guess at image placement.
        let positions = match rope {
            Some(triples) => RopePositions::PerToken(triples),
            None => RopePositions::Contiguous(index_pos),
        };
        let query = self.rope.apply(&query, positions)?.contiguous()?;
        let key = self.rope.apply(&key, positions)?.contiguous()?;

        let (key, value) = match state.as_ref() {
            Some(state) if index_pos > 0 => {
                // The resident tensors hold only the exact window once older
                // positions have been paged out, so the invariant is over the
                // total the state represents, not the length of what is held.
                let resident = state.key.dim(2)?;
                let held = resident + state.pages.as_ref().map_or(0, |p| p.positions());
                if held != index_pos {
                    candle_core::bail!(
                        "Qwen35 attention state is at position {held}, cannot append at {index_pos}"
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
                let mut pages = match state.take().and_then(|previous| previous.pages) {
                    Some(pages) => pages,
                    None => KvPages::new(policy, dim, key.device())?,
                };
                let overflow = positions.saturating_sub(policy.recent);
                if overflow > 0 {
                    let (key_codes, key_norms) = pages.absorb(&key, overflow)?;
                    let (value_codes, value_norms) = pages.absorb(&value, overflow)?;
                    pages.record_fidelity(&key, &key_codes, &key_norms, overflow)?;
                    pages.record_fidelity(&value, &value_codes, &value_norms, overflow)?;
                    KvPages::extend(&mut pages.key_codes, key_codes)?;
                    KvPages::extend(&mut pages.key_norms, key_norms)?;
                    KvPages::extend(&mut pages.value_codes, value_codes)?;
                    KvPages::extend(&mut pages.value_norms, value_norms)?;
                    pages.positions += overflow;
                }
                let recent_key = key.narrow(2, overflow, positions - overflow)?.contiguous()?;
                let recent_value =
                    value.narrow(2, overflow, positions - overflow)?.contiguous()?;
                let full = match pages.decode_pair(batch, heads, dim)? {
                    Some((old_key, old_value)) => (
                        Tensor::cat(&[&old_key.to_dtype(key.dtype())?, &recent_key], 2)?,
                        Tensor::cat(&[&old_value.to_dtype(value.dtype())?, &recent_value], 2)?,
                    ),
                    None => (recent_key.clone(), recent_value.clone()),
                };
                *state = Some(AttentionState {
                    key: recent_key,
                    value: recent_value,
                    pages: Some(pages),
                });
                full
            }
        };

        let ratio = self.head_count / self.kv_head_count;
        let grouped = grouped_attention_enabled() && ratio > 1;
        // Grouped: fold the repeat into the row dimension so the expanded K and
        // V are never built. Otherwise materialize them as before.
        let (key, value, query) = if grouped {
            (key, value, query.contiguous()?)
        } else {
            (repeat_kv(key, ratio)?, repeat_kv(value, ratio)?, query)
        };
        let key_positions = key.dim(2)?;
        let scores = if grouped {
            query
                .reshape((batch, self.kv_head_count, ratio * seq_len, self.head_dim))?
                .matmul(&key.t()?)?
                .reshape((batch, self.head_count, seq_len, key_positions))?
        } else {
            query.matmul(&key.t()?)?
        };
        let scores = (scores / (self.head_dim as f64).sqrt())?;
        let scores = match mask {
            Some(mask) => {
                let mask = mask.broadcast_as(scores.shape())?;
                masked_fill(&scores, &mask, &self.neg_inf)?
            }
            None => scores,
        };
        let probabilities = softmax_last_dim(&scores)?;
        let attended = if grouped {
            probabilities
                .reshape((batch, self.kv_head_count, ratio * seq_len, key_positions))?
                .matmul(&value.contiguous()?)?
                .reshape((batch, self.head_count, seq_len, self.head_dim))?
        } else {
            probabilities.matmul(&value.contiguous()?)?
        };
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
        rope: Option<&[[usize; 3]]>,
    ) -> Result<Tensor> {
        match (self, state) {
            (Self::Attention(attention), MixerState::Attention(state)) => {
                attention.forward(state, x, mask, index_pos, rope)
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
        rope: Option<&[[usize; 3]]>,
    ) -> Result<Tensor> {
        let residual = x;
        let normalized = self.input_norm.forward(x)?;
        let mixed = self.mixer.forward(state, &normalized, mask, index_pos, rope)?;
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
    /// Multimodal RoPE section widths, in rotary *pairs*, as the GGUF declares
    /// them. Qwen-VL splits the rotary dimensions between temporal, height and
    /// width position indices. Empty when the model declares none.
    ///
    /// Text-only inference does not need them: a text token's three indices are
    /// all its sequence position, so every section rotates by the same angle and
    /// the split collapses to ordinary 1-D RoPE. Image tokens are what make the
    /// sections observable, which is why the current forward pass is correct
    /// without them and cannot stay so once images arrive.
    pub(crate) rope_sections: Vec<usize>,
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
        // Sections are written as signed integers by some writers and unsigned
        // by others, so accept either. They are metadata for the image path and
        // are unused by text inference: a shape we cannot read must not stop the
        // model loading, it must stop us claiming to know the split.
        let rope_sections: Vec<usize> = metadata
            .get("qwen35.rope.dimension_sections")
            .and_then(|value| value.to_vec().ok())
            .map(|values| {
                values
                    .iter()
                    .map(|v| {
                        v.to_u32()
                            .map(|v| v as usize)
                            .or_else(|_| v.to_i32().map(|v| v.max(0) as usize))
                            .ok()
                    })
                    .collect::<Option<Vec<usize>>>()
            })
            .unwrap_or_default()
            .unwrap_or_else(|| {
                tracing::warn!(
                    "qwen35.rope.dimension_sections is present but unreadable; \
                     multimodal position splitting will be unavailable"
                );
                Vec::new()
            });
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
        // Sections are counted in rotary pairs, so they must account for every
        // pair of the rotary width. A mismatch means the split we would apply to
        // image positions does not describe this model.
        let section_pairs: usize = rope_sections.iter().sum();
        let rope_sections = if !rope_sections.is_empty() && section_pairs * 2 != rope_dim {
            // Text inference does not consult the split, so a description we
            // cannot trust must not stop the model loading. Discard it instead,
            // so nothing later mistakes it for a usable multimodal layout.
            tracing::warn!(
                sections = ?rope_sections,
                rope_dim,
                "rope sections do not cover the rotary width; ignoring them"
            );
            Vec::new()
        } else {
            rope_sections
        };
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
            rope_sections,
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

    /// What the compressed pages of every attention layer currently occupy,
    /// against the f32 they stand in for.
    ///
    /// Returns `None` when KV compression is off, so an absent figure reads as
    /// "not compressing" rather than "compressing nothing". The stored side is
    /// what the device allocator actually charges for the chosen
    /// [`CodeLayout`], not a nominal one-byte-per-code count.
    pub(crate) fn kv_page_residency(&self) -> Option<KvPageResidency> {
        let mut residency = KvPageResidency::default();
        for state in &self.states {
            let MixerState::Attention(Some(state)) = state else {
                continue;
            };
            let Some(pages) = state.pages.as_ref() else {
                continue;
            };
            let (batch, heads, _, dim) = match state.key.dims4() {
                Ok(dims) => dims,
                Err(_) => continue,
            };
            let (stored, uncompressed) = pages.stored_and_uncompressed_bytes(heads, dim, batch);
            residency.layout = pages.layout.label();
            residency.stored_bytes += stored;
            residency.uncompressed_bytes += uncompressed;
            residency.positions += pages.positions();
            residency.layers += 1;
            if let Some(measured) = pages.fidelity.as_ref() {
                residency
                    .fidelity
                    .get_or_insert_with(KvPageFidelity::default)
                    .merge(measured);
            }
        }
        (residency.layers > 0).then_some(residency)
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
        let (logits, _) =
            self.prefill_at_position(&input, reused, prefill_chunk_size(), check, collector)?;
        self.text = Some(TextState {
            tokens: tokens.to_vec(),
            logits: logits.clone(),
        });
        Ok((logits, reused))
    }

    /// Embed token ids through the model's own table.
    ///
    /// Exposed so a caller can build a mixed sequence - text rows from here,
    /// image rows from the projector - and hand the whole thing back.
    pub(crate) fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.weights.embedding.forward(tokens)
    }

    /// Residual width, so a caller can check projector output before splicing.
    pub(crate) fn embedding_width(&self) -> usize {
        self.weights.hidden
    }

    /// Forward a sequence whose embeddings and rope positions are both supplied.
    ///
    /// This is the image path: `embedded` carries projector rows spliced among
    /// text rows, and `positions` carries the `(t, h, w)` triple for every token,
    /// so image tokens reach the rotary sections instead of being placed as text.
    /// `tokens` is still required, for length and for the state bookkeeping that
    /// the text path shares.
    pub(crate) fn prefill_embedded(
        &mut self,
        tokens: &Tensor,
        embedded: &Tensor,
        positions: &[[usize; 3]],
    ) -> Result<Tensor> {
        let (_, seq_len) = tokens.dims2()?;
        if positions.len() != seq_len {
            candle_core::bail!(
                "{} rope position triples for {seq_len} tokens",
                positions.len()
            )
        }
        let hidden =
            self.run_hidden_states_positioned(tokens, 0, None, Some(positions), Some(embedded))?;
        self.weights.output.forward(&hidden)
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
        self.run_hidden_states_positioned(tokens, index_pos, collector, None, None)
    }

    /// The same forward, with explicit rope positions for image tokens.
    ///
    /// `rope` is `None` for text, which is every existing caller. Supplying it
    /// is the only way image tokens reach the rotary sections; there is no
    /// inference of image layout from the token stream.
    fn run_hidden_states_positioned(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
        mut collector: Option<&mut TapCollector>,
        rope: Option<&[[usize; 3]]>,
        embedded: Option<&Tensor>,
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
        // Supplied embeddings replace the table lookup wholesale: an image's
        // rows come from the projector, not from any token id.
        let mut hidden = match embedded {
            Some(rows) => {
                let (_, rows_len, _) = rows.dims3()?;
                if rows_len != seq_len {
                    candle_core::bail!(
                        "{rows_len} supplied embedding rows for {seq_len} tokens"
                    )
                }
                rows.clone()
            }
            None => self.weights.embedding.forward(tokens)?,
        };
        for (index, (layer, state)) in self.weights.layers.iter().zip(&mut self.states).enumerate()
        {
            hidden = layer.forward(state, &hidden, mask.as_ref(), index_pos, rope)?;
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
            // Two rotary pairs split one apiece between the temporal and height
            // sections, so this fixture can exercise image positions at all.
            // Text never consults the split, so text behaviour is unchanged.
            (
                "qwen35.rope.dimension_sections".into(),
                gguf_file::Value::Array(vec![
                    gguf_file::Value::I32(1),
                    gguf_file::Value::I32(1),
                    gguf_file::Value::I32(0),
                    gguf_file::Value::I32(0),
                ]),
            ),
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
    fn device_decode_reproduces_the_host_decode_exactly() {
        // Same codes, same codebook, same rotation: the device path must be a
        // relocation of the arithmetic, not a different approximation. Runs on
        // CPU here; the device backends use the identical tensor ops.
        let device = Device::Cpu;
        let (heads, positions, dim, bits) = (2usize, 5usize, 64usize, 4u8);
        let quantizer =
            std::sync::Arc::new(crate::nlu::turbo_quant::TurboQuant::new(dim, bits, 31));

        let mut codes: Vec<u8> = Vec::new();
        let mut norms: Vec<f32> = Vec::new();
        let mut host = Vec::new();
        for slot in 0..heads {
            let mut store =
                crate::nlu::turbo_quant::KvPageStore::with_quantizer(quantizer.clone());
            for position in 0..positions {
                let row: Vec<f32> = (0..dim)
                    .map(|d| ((slot * 31 + position * 7 + d) % 53) as f32 / 26.0 - 1.0)
                    .collect();
                let (row_codes, norm) = quantizer.quantize_row(&row);
                codes.extend_from_slice(&row_codes);
                norms.push(norm);
                store.push_rows(&row).unwrap();
            }
            host.extend(store.decode());
        }

        let code_tensor = Tensor::from_vec(codes, (1, heads, positions, dim), &device).unwrap();
        let norm_tensor = Tensor::from_vec(norms, (1, heads, positions, 1), &device).unwrap();
        let codebook = Tensor::from_vec(
            quantizer.codebook().to_vec(),
            (quantizer.codebook().len(),),
            &device,
        )
        .unwrap();
        let rotation =
            Tensor::from_vec(quantizer.rotation().to_vec(), (dim, dim), &device).unwrap();

        let on_device = KvPages::decode_on_device(
            &code_tensor,
            &norm_tensor,
            &codebook,
            &rotation,
            CodeLayout::Byte,
            1,
            heads,
            positions,
            dim,
        )
        .unwrap();
        assert_eq!(on_device.dims(), &[1, heads, positions, dim]);
        let got: Vec<f32> = on_device.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(got.len(), host.len());
        let worst = host
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // f32 accumulation order differs between the loop and the matmul, so
        // this is agreement rather than bit-identity.
        assert!(worst < 1e-4, "worst absolute difference {worst}");
    }

    #[test]
    fn kv_pages_absorb_the_oldest_positions_and_decode_them_back_in_order() {
        let device = Device::Cpu;
        let (heads, positions, dim) = (2usize, 8usize, 64usize);
        let values: Vec<f32> = (0..heads * positions * dim)
            .map(|i| ((i % 97) as f32 / 48.0) - 1.0)
            .collect();
        let tensor = Tensor::from_vec(values.clone(), (1, heads, positions, dim), &device).unwrap();

        let policy = KvQuantization { bits: 4, recent: 3 };
        let mut pages = KvPages::new(policy, dim, &device).unwrap();
        let overflow = positions - policy.recent;
        let (kc, kn) = pages.absorb(&tensor, overflow).unwrap();
        KvPages::extend(&mut pages.key_codes, kc).unwrap();
        KvPages::extend(&mut pages.key_norms, kn).unwrap();
        let (vc, vn) = pages.absorb(&tensor, overflow).unwrap();
        KvPages::extend(&mut pages.value_codes, vc).unwrap();
        KvPages::extend(&mut pages.value_norms, vn).unwrap();
        pages.positions += overflow;
        assert_eq!(pages.positions(), overflow);

        let (decoded, _) = pages.decode_pair(1, heads, dim).unwrap().unwrap();
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

        // Compression is real: one byte per code, not four.
        let (stored, plain) = pages.stored_and_uncompressed_bytes(heads, dim, 1);
        assert!(
            plain as f64 / stored as f64 > 3.5,
            "ratio {}",
            plain as f64 / stored as f64
        );
    }

    #[test]
    fn paged_fidelity_tracks_the_bit_width_it_is_measuring() {
        let device = Device::Cpu;
        let (heads, positions, dim) = (2usize, 8usize, 64usize);
        // Smoothly varying rows, so a coarse codebook has something to lose.
        let values: Vec<f32> = (0..heads * positions * dim)
            .map(|i| ((i % 97) as f32 / 48.0) - 1.0)
            .collect();
        let tensor = Tensor::from_vec(values, (1, heads, positions, dim), &device).unwrap();

        let measure = |bits: u8| {
            let mut pages =
                KvPages::new(KvQuantization { bits, recent: 2 }, dim, &device).unwrap();
            // Switch measurement on directly; the env flag is read once per
            // process and a test cannot rely on winning that race.
            pages.fidelity = Some(KvPageFidelity::default());
            let (codes, norms) = pages.absorb(&tensor, 6).unwrap();
            pages.record_fidelity(&tensor, &codes, &norms, 6).unwrap();
            pages.fidelity.unwrap()
        };

        let coarse = measure(2);
        let fine = measure(8);
        assert_eq!(coarse.vectors, heads * 6);
        assert_eq!(fine.vectors, heads * 6);
        // More bits must reconstruct at least as well on every summary.
        assert!(
            fine.min_cosine > coarse.min_cosine,
            "8-bit {} against 2-bit {}",
            fine.min_cosine,
            coarse.min_cosine
        );
        assert!(fine.mean_cosine() > coarse.mean_cosine());
        assert!(fine.max_relative_l2 < coarse.max_relative_l2);
        // A cosine is a cosine, whatever the width.
        for stats in [coarse, fine] {
            assert!(stats.min_cosine > -1.000_001 && stats.min_cosine < 1.000_001);
        }
    }

    #[test]
    fn per_token_rope_positions_collapse_onto_the_contiguous_path() {
        // The real artifact's split: 11 temporal, 11 height, 10 width pairs.
        let sections = vec![11usize, 11, 10];
        let device = Device::Cpu;
        let (width, context) = (64usize, 32usize);
        let pairs = width / 2;
        let theta: Vec<f32> = (0..width)
            .step_by(2)
            .map(|i| 1.0 / 10_000f32.powf(i as f32 / width as f32))
            .collect();
        let theta = Tensor::from_vec(theta, (1, pairs), &device).unwrap();
        let angles = Tensor::arange(0u32, context as u32, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((context, 1))
            .unwrap()
            .matmul(&theta)
            .unwrap();
        let rope = PartialRope {
            cos: angles.cos().unwrap(),
            sin: angles.sin().unwrap(),
            width,
            sections: sections.clone(),
        };

        let (batch, heads, seq) = (1usize, 2usize, 5usize);
        let values: Vec<f32> = (0..batch * heads * seq * width)
            .map(|i| ((i % 71) as f32 / 35.0) - 1.0)
            .collect();
        let x = Tensor::from_vec(values, (batch, heads, seq, width), &device).unwrap();

        // A triple per token with t == h == w is the text case written the long
        // way, so it must reproduce the contiguous path exactly.
        let start = 3usize;
        let degenerate: Vec<[usize; 3]> = (start..start + seq).map(|p| [p, p, p]).collect();
        let contiguous = rope.apply(&x, RopePositions::Contiguous(start)).unwrap();
        let per_token = rope
            .apply(&x, RopePositions::PerToken(&degenerate))
            .unwrap();
        assert_eq!(
            contiguous.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            per_token.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "a degenerate triple must be the contiguous path exactly"
        );

        // A genuine image triple must not be the text path, or the sections are
        // doing nothing and image tokens would be placed as if they were text.
        let image: Vec<[usize; 3]> = (0..seq).map(|i| [1, 2 + i, 9 - i]).collect();
        let threaded = rope.apply(&x, RopePositions::PerToken(&image)).unwrap();
        assert_ne!(
            contiguous.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            threaded.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "distinct t/h/w must change the rotation"
        );

        // Wrong count is a refusal, not a silent truncation.
        assert!(rope.apply(&x, RopePositions::PerToken(&image[..2])).is_err());

        // And an artifact with no declared split refuses image positions rather
        // than quietly placing them as text.
        let unsplit = PartialRope { sections: Vec::new(), ..rope.clone() };
        assert!(unsplit.apply(&x, RopePositions::PerToken(&image)).is_err());
        assert!(unsplit.apply(&x, RopePositions::Contiguous(start)).is_ok());
    }

    #[test]
    fn grouped_attention_is_the_same_arithmetic_as_materializing_the_repeat() {
        use candle_transformers::utils::repeat_kv;
        let device = Device::Cpu;
        let (batch, kv_heads, ratio, seq, positions, dim) = (1usize, 3usize, 4usize, 5usize, 7usize, 8usize);
        let heads = kv_heads * ratio;

        // Distinct values everywhere, so a head-ordering error cannot cancel.
        let make = |n: usize, salt: usize| {
            let values: Vec<f32> = (0..n)
                .map(|i| ((i * 7 + salt * 13) % 61) as f32 / 30.0 - 1.0)
                .collect();
            values
        };
        let query = Tensor::from_vec(
            make(batch * heads * seq * dim, 1),
            (batch, heads, seq, dim),
            &device,
        )
        .unwrap();
        let key = Tensor::from_vec(
            make(batch * kv_heads * positions * dim, 2),
            (batch, kv_heads, positions, dim),
            &device,
        )
        .unwrap();
        let value = Tensor::from_vec(
            make(batch * kv_heads * positions * dim, 3),
            (batch, kv_heads, positions, dim),
            &device,
        )
        .unwrap();

        // Reference: build the expanded K and V, then attend.
        let wide_key = repeat_kv(key.clone(), ratio).unwrap();
        let wide_value = repeat_kv(value.clone(), ratio).unwrap();
        let reference_scores = query.matmul(&wide_key.t().unwrap()).unwrap();

        // Grouped: fold the repeat into the row dimension, expand nothing.
        let grouped_scores = query
            .reshape((batch, kv_heads, ratio * seq, dim))
            .unwrap()
            .matmul(&key.t().unwrap())
            .unwrap()
            .reshape((batch, heads, seq, positions))
            .unwrap();

        assert_eq!(reference_scores.dims(), grouped_scores.dims());
        let a: Vec<f32> = reference_scores.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = grouped_scores.flatten_all().unwrap().to_vec1().unwrap();
        // Identical operand order per output element, so this is exact, not close.
        assert_eq!(a, b, "grouped scores must match head for head");

        // And the value matmul, which reuses the same grouping.
        let probabilities = softmax_last_dim(&reference_scores).unwrap();
        let reference_attended = probabilities.matmul(&wide_value.contiguous().unwrap()).unwrap();
        let grouped_attended = probabilities
            .reshape((batch, kv_heads, ratio * seq, positions))
            .unwrap()
            .matmul(&value.contiguous().unwrap())
            .unwrap()
            .reshape((batch, heads, seq, dim))
            .unwrap();
        assert_eq!(
            reference_attended.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            grouped_attended.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "grouped attention output must match head for head"
        );

        // The test would pass trivially if the expansion were a no-op; it is not.
        assert_eq!(wide_key.dims(), &[batch, heads, positions, dim]);
        assert_ne!(key.dims(), wide_key.dims());
    }

    #[test]
    fn packed_triples_survive_the_device_unpack_for_every_code_and_row_width() {
        // 128 is the head width the model actually uses and leaves one lane of
        // the last word padded; 129 divides evenly and takes the other branch.
        for dim in [128usize, 129] {
            let words_per_row = dim.div_ceil(3);
            let rows = 5usize;
            let codes: Vec<u8> = (0..rows * dim).map(|i| ((i * 37) % 256) as u8).collect();
            let packed = pack_triples(&codes, dim, words_per_row);
            assert_eq!(packed.len(), rows * words_per_row);
            assert!(
                packed.iter().all(|word| *word < 1 << 24),
                "a packed word has to stay inside the exactly representable f32 integers"
            );

            let words =
                Tensor::from_vec(packed, (1, 1, rows, words_per_row), &Device::Cpu).unwrap();
            let unpacked = KvPages::unpack_triples(&words, dim).unwrap();
            assert_eq!(unpacked.dims(), &[1, 1, rows, dim]);
            let got: Vec<f32> = unpacked.flatten_all().unwrap().to_vec1().unwrap();
            let want: Vec<f32> = codes.iter().map(|code| f32::from(*code)).collect();
            assert_eq!(got, want, "dim {dim}");
        }
    }

    #[test]
    fn the_packed_layout_decodes_to_exactly_what_the_byte_layout_decodes() {
        let device = Device::Cpu;
        let (heads, positions, dim) = (2usize, 6usize, 128usize);
        let values: Vec<f32> = (0..heads * positions * dim)
            .map(|i| ((i % 97) as f32 / 48.0) - 1.0)
            .collect();
        let tensor = Tensor::from_vec(values, (1, heads, positions, dim), &device).unwrap();
        // Eight bits so every code in the byte range gets exercised.
        let policy = KvQuantization { bits: 8, recent: 2 };
        let overflow = positions - policy.recent;

        let decode = |layout: CodeLayout, code_element_bytes: usize| {
            let mut pages = KvPages::new(policy, dim, &device).unwrap();
            pages.layout = layout;
            pages.code_element_bytes = code_element_bytes;
            let (kc, kn) = pages.absorb(&tensor, overflow).unwrap();
            KvPages::extend(&mut pages.key_codes, kc).unwrap();
            KvPages::extend(&mut pages.key_norms, kn).unwrap();
            let (vc, vn) = pages.absorb(&tensor, overflow).unwrap();
            KvPages::extend(&mut pages.value_codes, vc).unwrap();
            KvPages::extend(&mut pages.value_norms, vn).unwrap();
            pages.positions += overflow;
            let (key, _) = pages.decode_pair(1, heads, dim).unwrap().unwrap();
            (
                key.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                pages.stored_and_uncompressed_bytes(heads, dim, 1),
            )
        };

        let packed_layout = CodeLayout::PackedTriple {
            words_per_row: dim.div_ceil(3),
        };
        let (plain, (plain_stored, uncompressed)) = decode(CodeLayout::Byte, 1);
        let (packed, (packed_stored, packed_uncompressed)) = decode(packed_layout, 1);
        // Packing changes where the codes live, never which codes they are.
        assert_eq!(plain, packed);
        assert_eq!(uncompressed, packed_uncompressed);

        // Where an element can be one byte, the plain layout is the cheaper of
        // the two; the packed layout exists for backends where it cannot.
        assert!(packed_stored > plain_stored);
        assert!(
            uncompressed as f64 / plain_stored as f64 > 3.5,
            "ratio {}",
            uncompressed as f64 / plain_stored as f64
        );

        // On a backend whose narrowest element is a whole word, the byte layout
        // is not a saving at all: it stores more than the f32 it replaces. That
        // is the case the packed layout exists for.
        let (_, (wide_stored, wide_uncompressed)) = decode(CodeLayout::Byte, 4);
        assert!(
            wide_stored > wide_uncompressed,
            "byte layout on a four-byte element stores {wide_stored} against {wide_uncompressed}"
        );
        let (_, (wide_packed, _)) = decode(packed_layout, 4);
        assert!(
            wide_uncompressed as f64 / wide_packed as f64 > 2.9,
            "ratio {}",
            wide_uncompressed as f64 / wide_packed as f64
        );
    }

    #[test]
    fn mrope_collapses_to_one_dimensional_rope_when_the_indices_agree() {
        // The exact Ptah model's declared split: 11 temporal, 11 height, 10
        // width pairs, covering all 32 pairs of a 64-wide rotary dimension.
        let sections = [11usize, 11, 10, 0];
        assert_eq!(sections.iter().sum::<usize>() * 2, 64);

        // A text token carries the same index three times, so every pair
        // rotates by the same angle and the split is unobservable.
        let text = mrope_pair_positions(&sections, 32, 7, 7, 7);
        assert_eq!(text, vec![7usize; 32]);

        // An image token's indices differ, and the split becomes visible at
        // exactly the declared boundaries.
        let image = mrope_pair_positions(&sections, 32, 3, 5, 9);
        assert_eq!(&image[..11], &[3usize; 11]);
        assert_eq!(&image[11..22], &[5usize; 11]);
        assert_eq!(&image[22..32], &[9usize; 10]);
    }

    #[test]
    fn mrope_falls_back_to_the_temporal_index_beyond_the_declared_sections() {
        // Short sections leave a tail; a model with none declared is all text.
        assert_eq!(mrope_pair_positions(&[2, 1], 5, 1, 2, 3), vec![1, 1, 2, 1, 1]);
        assert_eq!(mrope_pair_positions(&[], 3, 4, 5, 6), vec![4, 4, 4]);
        // Over-long sections are truncated to the rotary width, never past it.
        assert_eq!(mrope_pair_positions(&[9], 2, 1, 2, 3), vec![1, 1]);
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
            // Written as signed integers, as the real artifact does; the
            // loader must accept that rather than refusing to load the model.
            (
                "qwen35.rope.dimension_sections".into(),
                gguf_file::Value::Array(vec![
                    gguf_file::Value::I32(11),
                    gguf_file::Value::I32(11),
                    gguf_file::Value::I32(10),
                    gguf_file::Value::I32(0),
                ]),
            ),
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
                .forward(&mut layer.mixer.empty_state(), &hidden, Some(&mask), 0, None)
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
    fn a_text_and_image_prompt_runs_end_to_end_with_grid_derived_positions() {
        use crate::nlu::vision_positions::{position_triples, ImageGrid, PromptPiece};
        let bytes = tiny_gguf();
        let device = Device::Cpu;

        // Two text tokens, a 2x2 image, one more text token.
        let grid = ImageGrid { rows: 2, cols: 2 };
        let pieces = [
            PromptPiece::Text(2),
            PromptPiece::Image(grid),
            PromptPiece::Text(1),
        ];
        let positions = position_triples(&pieces);
        let total = 2 + grid.tokens() + 1;
        assert_eq!(positions.len(), total);

        // Placeholders stand in for the image's tokens; their ids never reach
        // the stack because their rows are replaced by the projector's.
        let ids: Vec<u32> = (0..total as u32).map(|i| i % 4).collect();
        let tokens = Tensor::from_vec(ids, (1, total), &device).unwrap();

        let mut model = load_tiny_model(&bytes);
        let width = model.embedding_width();
        let text_rows = model.embed_tokens(&tokens).unwrap();
        assert_eq!(text_rows.dims(), &[1, total, width]);

        // Stand-in projector output, distinct from any token embedding.
        let projected = Tensor::from_vec(
            (0..grid.tokens() * width)
                .map(|i| ((i % 13) as f32 / 6.0) - 1.0)
                .collect::<Vec<f32>>(),
            (1, grid.tokens(), width),
            &device,
        )
        .unwrap();
        let spliced = Tensor::cat(
            &[
                &text_rows.narrow(1, 0, 2).unwrap(),
                &projected,
                &text_rows.narrow(1, 2 + grid.tokens(), 1).unwrap(),
            ],
            1,
        )
        .unwrap()
        .contiguous()
        .unwrap();
        assert_eq!(spliced.dims(), text_rows.dims());

        let logits = model
            .prefill_embedded(&tokens, &spliced, &positions)
            .unwrap();
        assert_eq!(logits.dims()[0], 1);
        let values: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "logits must be finite");

        // This fixture's mixer weights are zero, so every path through it
        // produces zero logits. Assert that rather than work around it: the
        // moment the fixture stops being degenerate this fails, and whoever
        // changed it should add the discrimination this cannot carry.
        assert!(
            values.iter().all(|v| *v == 0.0),
            "fixture is no longer degenerate; assert that image rows and grid \
             positions change the logits here, not just that they are accepted"
        );

        // What this fixture *can* establish is that the composition type-checks
        // end to end and every stage agrees on length: grid -> triples ->
        // spliced rows -> forward. That the triples actually change the rotation
        // is covered where it is observable, in
        // `per_token_rope_positions_collapse_onto_the_contiguous_path`.
        let mut mismatched = load_tiny_model(&bytes);
        assert!(
            mismatched
                .prefill_embedded(&tokens, &spliced, &positions[..total - 1])
                .is_err(),
            "a triple per token is required, not merely supplied"
        );
    }

    #[test]
    fn the_embedded_path_reproduces_the_token_path_when_nothing_is_spliced() {
        let bytes = tiny_gguf();
        let device = Device::Cpu;
        let tokens = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();

        // Token path.
        let mut plain = load_tiny_model(&bytes);
        let reference: Vec<f32> = plain
            .forward(&tokens, 0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Embedded path, handing back exactly what the table would have given
        // and degenerate position triples, which is text written the long way.
        let mut embedded_model = load_tiny_model(&bytes);
        let rows = embedded_model.embed_tokens(&tokens).unwrap();
        let positions: Vec<[usize; 3]> = (0..3).map(|p| [p, p, p]).collect();
        let got: Vec<f32> = embedded_model
            .prefill_embedded(&tokens, &rows, &positions)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(reference, got, "the image path must contain the text path");

        // A row count that disagrees with the tokens is a refusal.
        let short = rows.narrow(1, 0, 2).unwrap().contiguous().unwrap();
        let mut third = load_tiny_model(&bytes);
        assert!(third.prefill_embedded(&tokens, &short, &positions).is_err());
        // As is a triple count that disagrees.
        let mut fourth = load_tiny_model(&bytes);
        assert!(fourth.prefill_embedded(&tokens, &rows, &positions[..2]).is_err());

        // Splicing different rows must change the answer, or the supplied
        // embeddings are being ignored and images would silently do nothing.
        let mut fifth = load_tiny_model(&bytes);
        let altered = rows.affine(1.0, 0.25).unwrap();
        let spliced: Vec<f32> = fifth
            .prefill_embedded(&tokens, &altered, &positions)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_ne!(reference, spliced, "supplied embeddings must reach the stack");
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
