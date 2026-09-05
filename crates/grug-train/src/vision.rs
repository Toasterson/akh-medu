//! Native pixel-prefix input for the BabyLLM decoder.
//!
//! The stem follows the encoder-free Gemma 4 principle: RGB patches pass
//! through one projection, factorized row/column embeddings and LayerNorm,
//! then become a causal prefix for the existing [`TinyTransformer`]. Patch
//! size remains configurable because 48x48 reference patches can erase small
//! document text at BabyLLM scale.

use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::activation::gelu;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::tokenizer::Vocab;
use crate::train::Batch;
use crate::transformer::{TinyTransformer, TinyTransformerConfig};

pub const OCR_PROMPT: &str = "transcribe image exactly";

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("patch size must be positive")]
    ZeroPatchSize,
    #[error("image dimensions must be positive, got {width}x{height}")]
    EmptyImage { width: usize, height: usize },
    #[error("RGB length {actual} does not match {width}x{height}x3 ({expected})")]
    InvalidRgbLength {
        width: usize,
        height: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "visual prefix ({visual_tokens}) plus text ({text_tokens}) exceeds context {max_seq_len}"
    )]
    ContextOverflow {
        visual_tokens: usize,
        text_tokens: usize,
        max_seq_len: usize,
    },
    #[error("vision/text batch must contain at least one case")]
    EmptyBatch,
    #[error(
        "patch grid {index} has shape {rows}x{columns}x{patch_dim}, expected {expected_rows}x{expected_columns}x{expected_patch_dim}"
    )]
    PatchGridMismatch {
        index: usize,
        rows: usize,
        columns: usize,
        patch_dim: usize,
        expected_rows: usize,
        expected_columns: usize,
        expected_patch_dim: usize,
    },
    #[error("RGB image {index} has length {actual}, expected {expected} for {width}x{height}")]
    RgbBatchMismatch {
        index: usize,
        width: usize,
        height: usize,
        expected: usize,
        actual: usize,
    },
    #[error("tokenizer failed while building an OCR sequence: {0}")]
    Tokenizer(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchGrid {
    /// Row-major `[patch, local_y, local_x, rgb_channel]`, normalized to 0..1.
    pub values: Vec<f32>,
    pub rows: usize,
    pub columns: usize,
    pub patch_size: usize,
    pub patch_dim: usize,
    pub padded_width: usize,
    pub padded_height: usize,
}

impl PatchGrid {
    pub fn token_count(&self) -> usize {
        self.rows * self.columns
    }

    /// Create a `[1, patches, patch_dim]` tensor for one image.
    pub fn into_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, 3> {
        let token_count = self.token_count();
        let patch_dim = self.patch_dim;
        Tensor::from_data(
            TensorData::new(self.values, [1, token_count, patch_dim]),
            device,
        )
    }
}

/// Patchify RGB bytes. Partial edge patches are padded white rather than
/// discarded, with padding dimensions retained in [`PatchGrid`].
pub fn patchify_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    patch_size: usize,
) -> Result<PatchGrid, VisionError> {
    if patch_size == 0 {
        return Err(VisionError::ZeroPatchSize);
    }
    if width == 0 || height == 0 {
        return Err(VisionError::EmptyImage { width, height });
    }
    let expected = width * height * 3;
    if rgb.len() != expected {
        return Err(VisionError::InvalidRgbLength {
            width,
            height,
            expected,
            actual: rgb.len(),
        });
    }
    let columns = width.div_ceil(patch_size);
    let rows = height.div_ceil(patch_size);
    let patch_dim = patch_size * patch_size * 3;
    let mut values = Vec::with_capacity(rows * columns * patch_dim);
    for patch_row in 0..rows {
        for patch_column in 0..columns {
            for local_y in 0..patch_size {
                for local_x in 0..patch_size {
                    let x = patch_column * patch_size + local_x;
                    let y = patch_row * patch_size + local_y;
                    if x < width && y < height {
                        let start = (y * width + x) * 3;
                        values.extend(
                            rgb[start..start + 3]
                                .iter()
                                .map(|value| f32::from(*value) / 255.0),
                        );
                    } else {
                        values.extend([1.0, 1.0, 1.0]);
                    }
                }
            }
        }
    }
    Ok(PatchGrid {
        values,
        rows,
        columns,
        patch_size,
        patch_dim,
        padded_width: columns * patch_size,
        padded_height: rows * patch_size,
    })
}

/// Stack equal-shaped patch grids into `[batch, patches, patch_dim]`.
pub fn stack_patch_grids<B: Backend>(
    grids: &[PatchGrid],
    device: &B::Device,
) -> Result<Tensor<B, 3>, VisionError> {
    let first = grids.first().ok_or(VisionError::EmptyBatch)?;
    let rows = first.rows;
    let columns = first.columns;
    let patch_dim = first.patch_dim;
    let mut values = Vec::with_capacity(grids.len() * first.values.len());
    for (index, grid) in grids.iter().enumerate() {
        if grid.rows != rows || grid.columns != columns || grid.patch_dim != patch_dim {
            return Err(VisionError::PatchGridMismatch {
                index,
                rows: grid.rows,
                columns: grid.columns,
                patch_dim: grid.patch_dim,
                expected_rows: rows,
                expected_columns: columns,
                expected_patch_dim: patch_dim,
            });
        }
        values.extend_from_slice(&grid.values);
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [grids.len(), rows * columns, patch_dim]),
        device,
    ))
}

/// Convert equal-sized RGB byte images into the NCHW tensor consumed by the
/// compact convolutional fallback stem.
pub fn rgb_batch_to_nchw<B: Backend>(
    images: &[Vec<u8>],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>, VisionError> {
    if images.is_empty() {
        return Err(VisionError::EmptyBatch);
    }
    if width == 0 || height == 0 {
        return Err(VisionError::EmptyImage { width, height });
    }
    let expected = width * height * 3;
    let mut values = vec![0.0f32; images.len() * expected];
    for (batch, image) in images.iter().enumerate() {
        if image.len() != expected {
            return Err(VisionError::RgbBatchMismatch {
                index: batch,
                width,
                height,
                expected,
                actual: image.len(),
            });
        }
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    let source = (y * width + x) * 3 + channel;
                    let destination = ((batch * 3 + channel) * height + y) * width + x;
                    values[destination] = f32::from(image[source]) / 255.0;
                }
            }
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [images.len(), 3, height, width]),
        device,
    ))
}

/// Build teacher-forced OCR targets using
/// `BOS <instruction> SEP <transcript> EOS`. The image is supplied separately
/// as a prefix; only transcript/EOS targets are enabled in the loss mask.
pub fn build_vision_ocr_batch<B: Backend, V: Vocab>(
    tokenizer: &V,
    transcripts: &[String],
    max_text_seq_len: usize,
    device: &B::Device,
) -> Result<Batch<B>, VisionError> {
    if transcripts.is_empty() {
        return Err(VisionError::EmptyBatch);
    }
    let prompt = tokenizer
        .encode(OCR_PROMPT)
        .map_err(|error| VisionError::Tokenizer(error.to_string()))?;
    let mut encoded = Vec::with_capacity(transcripts.len());
    for transcript in transcripts {
        let mut ids = Vec::new();
        ids.push(tokenizer.bos_id());
        ids.extend_from_slice(&prompt);
        ids.push(tokenizer.sep_id());
        let target_start = ids.len();
        ids.extend(
            tokenizer
                .encode(transcript)
                .map_err(|error| VisionError::Tokenizer(error.to_string()))?,
        );
        ids.push(tokenizer.eos_id());
        encoded.push((ids, target_start));
    }
    let longest = encoded
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or(2)
        .min(max_text_seq_len.max(2));
    let seq_len = longest.saturating_sub(1).max(1);
    let batch_size = encoded.len();
    let mut input_flat = Vec::with_capacity(batch_size * seq_len);
    let mut target_flat = Vec::with_capacity(batch_size * seq_len);
    let mut loss_mask = Vec::with_capacity(batch_size * seq_len);
    for (mut ids, target_start) in encoded {
        ids.truncate(longest);
        while ids.len() < longest {
            ids.push(tokenizer.pad_id());
        }
        for position in 0..seq_len {
            input_flat.push(i64::from(ids[position]));
            target_flat.push(i64::from(ids[position + 1]));
            loss_mask.push(position + 1 >= target_start && ids[position + 1] != tokenizer.pad_id());
        }
    }
    Ok(Batch {
        input_ids: Tensor::from_data(TensorData::new(input_flat, [batch_size, seq_len]), device),
        targets: Tensor::from_data(TensorData::new(target_flat, [batch_size, seq_len]), device),
        copy_mask: vec![false; batch_size * seq_len],
        loss_mask,
        batch_size,
        seq_len,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualTokenBudget {
    pub width: usize,
    pub height: usize,
    pub patch_size: usize,
    pub visual_tokens: usize,
    pub text_tokens: usize,
    pub total_tokens: usize,
    pub max_seq_len: usize,
    pub visual_fraction: f64,
}

/// Compute the image-prefix context cost before training or allocation.
pub fn visual_token_budget(
    width: usize,
    height: usize,
    patch_size: usize,
    text_tokens: usize,
    max_seq_len: usize,
) -> Result<VisualTokenBudget, VisionError> {
    if patch_size == 0 {
        return Err(VisionError::ZeroPatchSize);
    }
    if width == 0 || height == 0 {
        return Err(VisionError::EmptyImage { width, height });
    }
    let visual_tokens = width.div_ceil(patch_size) * height.div_ceil(patch_size);
    let total_tokens = visual_tokens + text_tokens;
    if total_tokens > max_seq_len {
        return Err(VisionError::ContextOverflow {
            visual_tokens,
            text_tokens,
            max_seq_len,
        });
    }
    Ok(VisualTokenBudget {
        width,
        height,
        patch_size,
        visual_tokens,
        text_tokens,
        total_tokens,
        max_seq_len,
        visual_fraction: visual_tokens as f64 / max_seq_len.max(1) as f64,
    })
}

#[derive(Config, Debug)]
pub struct VisionPatchStemConfig {
    pub channels: usize,
    pub patch_size: usize,
    pub d_model: usize,
    pub max_grid_rows: usize,
    pub max_grid_columns: usize,
}

impl VisionPatchStemConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionPatchStem<B> {
        assert!(self.channels > 0, "vision channels must be positive");
        assert!(self.patch_size > 0, "vision patch size must be positive");
        assert!(self.max_grid_rows > 0 && self.max_grid_columns > 0);
        let patch_dim = self.channels * self.patch_size * self.patch_size;
        VisionPatchStem {
            patch_projection: LinearConfig::new(patch_dim, self.d_model).init(device),
            row_embedding: EmbeddingConfig::new(self.max_grid_rows, self.d_model).init(device),
            column_embedding: EmbeddingConfig::new(self.max_grid_columns, self.d_model)
                .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            patch_dim,
            d_model: self.d_model,
            max_grid_rows: self.max_grid_rows,
            max_grid_columns: self.max_grid_columns,
        }
    }
}

#[derive(Module, Debug)]
pub struct VisionPatchStem<B: Backend> {
    patch_projection: Linear<B>,
    row_embedding: Embedding<B>,
    column_embedding: Embedding<B>,
    norm: LayerNorm<B>,
    patch_dim: usize,
    d_model: usize,
    max_grid_rows: usize,
    max_grid_columns: usize,
}

impl<B: Backend> VisionPatchStem<B> {
    /// Project `[batch, rows*columns, patch_dim]` to visual prefix embeddings.
    pub fn forward(&self, patches: Tensor<B, 3>, rows: usize, columns: usize) -> Tensor<B, 3> {
        let [batch, patch_count, patch_dim] = patches.dims();
        assert_eq!(
            patch_count,
            rows * columns,
            "patch count does not match grid"
        );
        assert_eq!(patch_dim, self.patch_dim, "patch vector width mismatch");
        assert!(
            rows <= self.max_grid_rows,
            "row grid exceeds configured maximum"
        );
        assert!(
            columns <= self.max_grid_columns,
            "column grid exceeds configured maximum"
        );
        let device = patches.device();
        let row_ids: Vec<i64> = (0..rows)
            .flat_map(|row| std::iter::repeat_n(row as i64, columns))
            .collect();
        let column_ids: Vec<i64> = (0..rows)
            .flat_map(|_| (0..columns).map(|column| column as i64))
            .collect();
        let row_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::new(row_ids, [patch_count]), &device)
                .unsqueeze::<2>()
                .repeat_dim(0, batch);
        let column_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::new(column_ids, [patch_count]), &device)
                .unsqueeze::<2>()
                .repeat_dim(0, batch);
        let projected = self.patch_projection.forward(patches);
        let positioned = projected
            + self.row_embedding.forward(row_ids)
            + self.column_embedding.forward(column_ids);
        let output = self.norm.forward(positioned);
        debug_assert_eq!(output.dims(), [batch, patch_count, self.d_model]);
        output
    }
}

/// A full-resolution residual pre-block followed by four local-convolution
/// plus 2x2 max-pooling stages forms the measured fallback when the raw patch
/// projection cannot retain small translated glyphs. The pre-block lets the
/// stem encode sub-stride strokes before spatial reduction; max pooling adds
/// bounded translation tolerance. The output stride remains 16, matching the
/// token count of a 16x16 patch stem, before the shared decoder.
#[derive(Config, Debug)]
pub struct CompactConvVisionStemConfig {
    pub channels: usize,
    pub hidden_channels: usize,
    pub d_model: usize,
    pub max_grid_rows: usize,
    pub max_grid_columns: usize,
}

impl CompactConvVisionStemConfig {
    pub const OUTPUT_STRIDE: usize = 16;

    pub fn init<B: Backend>(&self, device: &B::Device) -> CompactConvVisionStem<B> {
        assert!(self.channels > 0 && self.hidden_channels > 0 && self.d_model > 0);
        assert!(self.max_grid_rows > 0 && self.max_grid_columns > 0);
        let hidden_twice = self.hidden_channels.saturating_mul(2).max(1);
        let same_resolution = |channels| {
            Conv2dConfig::new(channels, [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let local = |channels| {
            Conv2dConfig::new(channels, [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        CompactConvVisionStem {
            pre_conv1: same_resolution([self.channels, self.hidden_channels]),
            pre_conv2: same_resolution([self.hidden_channels, self.hidden_channels]),
            conv1: local([self.hidden_channels, self.hidden_channels]),
            conv2: local([self.hidden_channels, hidden_twice]),
            conv3: local([hidden_twice, self.d_model]),
            conv4: local([self.d_model, self.d_model]),
            downsample: MaxPool2dConfig::new([2, 2]).with_ceil_mode(true).init(),
            row_embedding: EmbeddingConfig::new(self.max_grid_rows, self.d_model).init(device),
            column_embedding: EmbeddingConfig::new(self.max_grid_columns, self.d_model)
                .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            channels: self.channels,
            d_model: self.d_model,
            max_grid_rows: self.max_grid_rows,
            max_grid_columns: self.max_grid_columns,
        }
    }
}

#[derive(Module, Debug)]
pub struct CompactConvVisionStem<B: Backend> {
    pre_conv1: Conv2d<B>,
    pre_conv2: Conv2d<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    downsample: MaxPool2d,
    row_embedding: Embedding<B>,
    column_embedding: Embedding<B>,
    norm: LayerNorm<B>,
    channels: usize,
    d_model: usize,
    max_grid_rows: usize,
    max_grid_columns: usize,
}

impl<B: Backend> CompactConvVisionStem<B> {
    /// Convert `[batch, channels, height, width]` pixels into row-major visual
    /// prefix tokens at an effective stride of 16 pixels.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, channels, _, _] = images.dims();
        assert_eq!(channels, self.channels, "vision channel count mismatch");
        let features = gelu(self.pre_conv1.forward(images));
        let features = gelu(self.pre_conv2.forward(features.clone())) + features;
        let features = self.downsample.forward(gelu(self.conv1.forward(features)));
        let features = self.downsample.forward(gelu(self.conv2.forward(features)));
        let features = self.downsample.forward(gelu(self.conv3.forward(features)));
        let features = self.downsample.forward(self.conv4.forward(features));
        let [_, width, rows, columns] = features.dims();
        assert_eq!(width, self.d_model, "compact stem width mismatch");
        assert!(
            rows <= self.max_grid_rows && columns <= self.max_grid_columns,
            "compact stem grid {rows}x{columns} exceeds configured maximum {}x{}",
            self.max_grid_rows,
            self.max_grid_columns
        );
        let patch_count = rows * columns;
        let device = features.device();
        let tokens =
            features
                .swap_dims(1, 2)
                .swap_dims(2, 3)
                .reshape([batch, patch_count, self.d_model]);
        let row_ids: Vec<i64> = (0..rows)
            .flat_map(|row| std::iter::repeat_n(row as i64, columns))
            .collect();
        let column_ids: Vec<i64> = (0..rows)
            .flat_map(|_| (0..columns).map(|column| column as i64))
            .collect();
        let row_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::new(row_ids, [patch_count]), &device)
                .unsqueeze::<2>()
                .repeat_dim(0, batch);
        let column_ids =
            Tensor::<B, 1, Int>::from_data(TensorData::new(column_ids, [patch_count]), &device)
                .unsqueeze::<2>()
                .repeat_dim(0, batch);
        // Normalize visual content independently, then add position. Mixing a
        // random absolute-position vector into LayerNorm made identical small
        // glyphs acquire different content directions at every output cell.
        // The decoder still receives ordered positions, while the convolutional
        // representation retains the translation tolerance established above.
        self.norm.forward(tokens)
            + self.row_embedding.forward(row_ids)
            + self.column_embedding.forward(column_ids)
    }
}

#[derive(Config, Debug)]
pub struct NativeVisionTransformerConfig {
    pub transformer: TinyTransformerConfig,
    pub vision: VisionPatchStemConfig,
}

impl NativeVisionTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NativeVisionTransformer<B> {
        assert_eq!(
            self.transformer.d_model, self.vision.d_model,
            "vision and transformer widths must match"
        );
        NativeVisionTransformer {
            transformer: self.transformer.init(device),
            vision: self.vision.init(device),
        }
    }
}

/// Shared-decoder multimodal model. The visual stem is deliberately exposed
/// as a first-class module so checkpoints include and train its parameters.
#[derive(Module, Debug)]
pub struct NativeVisionTransformer<B: Backend> {
    transformer: TinyTransformer<B>,
    vision: VisionPatchStem<B>,
}

impl<B: Backend> NativeVisionTransformer<B> {
    pub fn forward(
        &self,
        patches: Tensor<B, 3>,
        rows: usize,
        columns: usize,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let prefix = self.vision.forward(patches, rows, columns);
        self.transformer
            .forward_with_prefix_embeddings(prefix, input_ids)
    }

    /// Text-only control path for retention comparisons on the same trunk.
    pub fn forward_text_only(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.transformer.forward(input_ids)
    }

    /// Logits aligned with textual input positions only. Visual prefix logits
    /// are removed so an ordinary text [`Batch`] target/mask can train the
    /// multimodal model without assigning fake token IDs to image patches.
    pub fn forward_text_logits(
        &self,
        patches: Tensor<B, 3>,
        rows: usize,
        columns: usize,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, text_len] = input_ids.dims();
        let prefix_len = rows * columns;
        let logits = self.forward(patches, rows, columns, input_ids);
        let vocab = logits.dims()[2];
        logits.slice([0..batch, prefix_len..prefix_len + text_len, 0..vocab])
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

/// Shared-decoder model using the compact local-convolution fallback. It
/// preserves the same continuous-prefix and text-token ABI as
/// [`NativeVisionTransformer`], so the two stems are directly comparable.
#[derive(Config, Debug)]
pub struct CompactVisionTransformerConfig {
    pub transformer: TinyTransformerConfig,
    pub vision: CompactConvVisionStemConfig,
}

impl CompactVisionTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CompactVisionTransformer<B> {
        assert_eq!(
            self.transformer.d_model, self.vision.d_model,
            "vision and transformer widths must match"
        );
        CompactVisionTransformer {
            transformer: self.transformer.init(device),
            vision: self.vision.init(device),
        }
    }

    /// Attach a fresh compact visual stem to an already trained textual trunk.
    /// The caller reconstructs and verifies the trunk from its artifact before
    /// calling this method; no textual parameter is reinitialized here.
    pub fn init_with_transformer<B: Backend>(
        &self,
        transformer: TinyTransformer<B>,
        device: &B::Device,
    ) -> CompactVisionTransformer<B> {
        assert_eq!(
            self.transformer.d_model, self.vision.d_model,
            "vision and transformer widths must match"
        );
        CompactVisionTransformer {
            transformer,
            vision: self.vision.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct CompactVisionTransformer<B: Backend> {
    transformer: TinyTransformer<B>,
    vision: CompactConvVisionStem<B>,
}

impl<B: Backend> CompactVisionTransformer<B> {
    pub fn forward(&self, images: Tensor<B, 4>, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let prefix = self.vision.forward(images);
        self.transformer
            .forward_with_prefix_embeddings(prefix, input_ids)
    }

    pub fn forward_text_only(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.transformer.forward(input_ids)
    }

    pub fn forward_text_logits(
        &self,
        images: Tensor<B, 4>,
        input_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, text_len] = input_ids.dims();
        let prefix = self.vision.forward(images);
        let prefix_len = prefix.dims()[1];
        let logits = self
            .transformer
            .forward_with_prefix_embeddings(prefix, input_ids);
        let vocab = logits.dims()[2];
        logits.slice([0..batch, prefix_len..prefix_len + text_len, 0..vocab])
    }

    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Borrow the shared textual trunk for retention evaluation and ordinary
    /// provider-free language/tool inference.
    pub fn transformer(&self) -> &TinyTransformer<B> {
        &self.transformer
    }
}

/// Initialize the native vision model and the same clipped Adam family used by
/// the text trunk. This is the runnable V0/V1 training boundary: data code can
/// build ordinary text targets and supply one patch tensor.
pub fn init_vision_training<B: AutodiffBackend>(
    config: &NativeVisionTransformerConfig,
    device: &B::Device,
) -> (
    NativeVisionTransformer<B>,
    OptimizerAdaptor<Adam, NativeVisionTransformer<B>, B>,
) {
    let model = config.init::<B>(device);
    let optimizer = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init::<B, NativeVisionTransformer<B>>();
    (model, optimizer)
}

pub fn init_compact_vision_training<B: AutodiffBackend>(
    config: &CompactVisionTransformerConfig,
    device: &B::Device,
) -> (
    CompactVisionTransformer<B>,
    OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
) {
    let model = config.init::<B>(device);
    let optimizer = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init::<B, CompactVisionTransformer<B>>();
    (model, optimizer)
}

/// One teacher-forced multimodal optimization step. Loss is evaluated only at
/// textual positions selected by `batch.loss_mask`; gradients still flow
/// through the preceding visual prefix and patch projection.
pub fn train_vision_step<B: AutodiffBackend>(
    model: NativeVisionTransformer<B>,
    optimizer: &mut OptimizerAdaptor<Adam, NativeVisionTransformer<B>, B>,
    patches: Tensor<B, 3>,
    rows: usize,
    columns: usize,
    batch: &Batch<B>,
    learning_rate: f64,
) -> (NativeVisionTransformer<B>, f32) {
    let device = batch.input_ids.device();
    let mask: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|enabled| i64::from(*enabled))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask, [batch.batch_size * batch.seq_len]),
        &device,
    );
    let logits = model.forward_text_logits(patches, rows, columns, batch.input_ids.clone());
    let loss = crate::train::masked_cross_entropy(logits, batch.targets.clone(), &mask);
    let loss_value: f32 = loss.clone().into_scalar().elem();
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value)
}

pub fn train_compact_vision_step<B: AutodiffBackend>(
    model: CompactVisionTransformer<B>,
    optimizer: &mut OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    images: Tensor<B, 4>,
    batch: &Batch<B>,
    learning_rate: f64,
) -> (CompactVisionTransformer<B>, f32) {
    let (model, loss, _grad_norm) = train_compact_vision_step_guarded(
        model,
        optimizer,
        images,
        batch,
        learning_rate,
        f32::INFINITY,
        f32::INFINITY,
    );
    (model, loss)
}

/// Forward-only OCR loss on the shared decoder, suitable for an inner backend
/// evaluation pass that does not retain gradients.
pub fn compact_vision_batch_loss<B: Backend>(
    model: &CompactVisionTransformer<B>,
    images: Tensor<B, 4>,
    batch: &Batch<B>,
) -> f32 {
    let device = batch.input_ids.device();
    let mask: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|enabled| i64::from(*enabled))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask, [batch.batch_size * batch.seq_len]),
        &device,
    );
    crate::train::masked_cross_entropy(
        model.forward_text_logits(images, batch.input_ids.clone()),
        batch.targets.clone(),
        &mask,
    )
    .into_scalar()
    .elem()
}

/// Guarded visual update. A non-finite or capped loss/whole-model gradient
/// returns the model untouched, preventing one hard OCR batch from poisoning
/// the shared text trunk or Adam state.
#[allow(clippy::too_many_arguments)]
pub fn train_compact_vision_step_guarded<B: AutodiffBackend>(
    model: CompactVisionTransformer<B>,
    optimizer: &mut OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    images: Tensor<B, 4>,
    batch: &Batch<B>,
    learning_rate: f64,
    loss_cap: f32,
    grad_norm_cap: f32,
) -> (CompactVisionTransformer<B>, f32, f32) {
    let device = batch.input_ids.device();
    let mask: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|enabled| i64::from(*enabled))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask, [batch.batch_size * batch.seq_len]),
        &device,
    );
    let logits = model.forward_text_logits(images, batch.input_ids.clone());
    let loss = crate::train::masked_cross_entropy(logits, batch.targets.clone(), &mask);
    let loss_value: f32 = loss.clone().into_scalar().elem();
    if !loss_value.is_finite() || loss_value > loss_cap {
        return (model, loss_value, f32::NAN);
    }
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let grad_norm = crate::train::global_grad_norm(&gradients, &model);
    if !grad_norm.is_finite() || grad_norm > grad_norm_cap {
        return (model, loss_value, grad_norm);
    }
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value, grad_norm)
}

/// Guarded text-only retention update through the exact same shared trunk and
/// optimizer as visual training. This enables an explicit interleaved mixture
/// instead of hoping OCR-only updates preserve tool/language behavior.
pub fn train_compact_text_step_guarded<B: AutodiffBackend>(
    model: CompactVisionTransformer<B>,
    optimizer: &mut OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    batch: &Batch<B>,
    learning_rate: f64,
    loss_cap: f32,
    grad_norm_cap: f32,
) -> (CompactVisionTransformer<B>, f32, f32) {
    let device = batch.input_ids.device();
    let mask: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|enabled| i64::from(*enabled))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask, [batch.batch_size * batch.seq_len]),
        &device,
    );
    let logits = model.forward_text_only(batch.input_ids.clone());
    let loss = crate::train::masked_cross_entropy(logits, batch.targets.clone(), &mask);
    let loss_value: f32 = loss.clone().into_scalar().elem();
    if !loss_value.is_finite() || loss_value > loss_cap {
        return (model, loss_value, f32::NAN);
    }
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let grad_norm = crate::train::global_grad_norm(&gradients, &model);
    if !grad_norm.is_finite() || grad_norm > grad_norm_cap {
        return (model, loss_value, grad_norm);
    }
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value, grad_norm)
}

/// Project a training model into its non-autodiff inference form.
pub fn to_eval_vision_model<B: AutodiffBackend>(
    model: &NativeVisionTransformer<B>,
) -> NativeVisionTransformer<B::InnerBackend> {
    model.valid()
}

pub fn to_eval_compact_vision_model<B: AutodiffBackend>(
    model: &CompactVisionTransformer<B>,
) -> CompactVisionTransformer<B::InnerBackend> {
    model.valid()
}

/// Greedy OCR generation using the same instruction/framing as
/// [`build_vision_ocr_batch`]. The visual prefix stays fixed while generated
/// text is appended autoregressively.
#[allow(clippy::too_many_arguments)] // grid geometry belongs to the patch tensor contract
pub fn generate_vision_ocr<B: Backend, V: Vocab>(
    model: &NativeVisionTransformer<B>,
    patches: Tensor<B, 3>,
    rows: usize,
    columns: usize,
    tokenizer: &V,
    max_new_tokens: usize,
    max_seq_len: usize,
    device: &B::Device,
) -> String {
    let mut ids = vec![tokenizer.bos_id()];
    ids.extend(tokenizer.encode(OCR_PROMPT).unwrap_or_default());
    ids.push(tokenizer.sep_id());
    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        if rows * columns + ids.len() >= max_seq_len {
            break;
        }
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                ids.iter().map(|id| i64::from(*id)).collect(),
                [1, ids.len()],
            ),
            device,
        );
        let logits = model.forward(patches.clone(), rows, columns, input);
        let [_, sequence, vocab] = logits.dims();
        let last: Vec<f32> = logits
            .slice([0..1, sequence - 1..sequence, 0..vocab])
            .reshape([vocab])
            .into_data()
            .to_vec()
            .unwrap_or_default();
        let Some((next, _)) = last
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
        else {
            break;
        };
        let next = next as u32;
        if next == tokenizer.eos_id() {
            break;
        }
        ids.push(next);
        generated.push(next);
    }
    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|token| *token < 256)
            .map(|token| token as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

pub fn generate_compact_vision_ocr<B: Backend, V: Vocab>(
    model: &CompactVisionTransformer<B>,
    images: Tensor<B, 4>,
    tokenizer: &V,
    max_new_tokens: usize,
    max_seq_len: usize,
    device: &B::Device,
) -> String {
    let prefix = model.vision.forward(images);
    let visual_tokens = prefix.dims()[1];
    let mut ids = vec![tokenizer.bos_id()];
    ids.extend(tokenizer.encode(OCR_PROMPT).unwrap_or_default());
    ids.push(tokenizer.sep_id());
    if visual_tokens + ids.len() >= max_seq_len {
        return String::new();
    }
    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            ids.iter().map(|id| i64::from(*id)).collect(),
            [1, ids.len()],
        ),
        device,
    );
    let (mut logits, mut cache) = model
        .transformer
        .prefill_cached_with_prefix_embeddings(prefix, input);
    let mut generated = Vec::new();
    for generated_index in 0..max_new_tokens {
        let [_, sequence, vocab] = logits.dims();
        let last: Vec<f32> = logits
            .slice([0..1, sequence - 1..sequence, 0..vocab])
            .reshape([vocab])
            .into_data()
            .to_vec()
            .unwrap_or_default();
        let Some((next, _)) = last
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
        else {
            break;
        };
        let next = next as u32;
        if next == tokenizer.eos_id() {
            break;
        }
        generated.push(next);
        if generated_index + 1 == max_new_tokens
            || visual_tokens + ids.len() + generated.len() >= max_seq_len
        {
            break;
        }
        let next_input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![i64::from(next)], [1, 1]), device);
        (logits, cache) = model.transformer.decode_cached(next_input, cache);
    }
    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|token| *token < 256)
            .map(|token| token as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ByteTokenizer, CpuAutodiffBackend, CpuBackend};
    use burn::nn::loss::{MseLoss, Reduction};

    #[test]
    fn patchify_pads_edges_white_in_row_major_order() {
        let rgb = vec![0u8; 3 * 3 * 3];
        let grid = patchify_rgb(&rgb, 3, 3, 2).unwrap();
        assert_eq!(grid.rows, 2);
        assert_eq!(grid.columns, 2);
        assert_eq!(grid.patch_dim, 12);
        assert_eq!(grid.padded_width, 4);
        assert_eq!(grid.padded_height, 4);
        assert_eq!(grid.values.len(), 4 * 12);
        assert!(grid.values.contains(&1.0));
    }

    #[test]
    fn token_budget_rejects_unworkable_full_page_inputs() {
        let error = visual_token_budget(1024, 1024, 16, 128, 1024).unwrap_err();
        assert!(matches!(error, VisionError::ContextOverflow { .. }));
        let budget = visual_token_budget(384, 64, 16, 128, 1024).unwrap();
        assert_eq!(budget.visual_tokens, 96);
        assert!(budget.visual_fraction < 0.10);
    }

    fn config() -> NativeVisionTransformerConfig {
        NativeVisionTransformerConfig::new(
            TinyTransformerConfig::new(37, 16, 2, 2, 32, 64),
            VisionPatchStemConfig::new(3, 2, 16, 4, 4),
        )
    }

    fn byte_config() -> NativeVisionTransformerConfig {
        NativeVisionTransformerConfig::new(
            TinyTransformerConfig::new(260, 16, 2, 2, 32, 64),
            VisionPatchStemConfig::new(3, 2, 16, 4, 4),
        )
    }

    fn compact_config() -> CompactVisionTransformerConfig {
        CompactVisionTransformerConfig::new(
            TinyTransformerConfig::new(37, 16, 2, 2, 32, 64),
            CompactConvVisionStemConfig::new(3, 4, 16, 4, 4),
        )
    }

    fn compact_byte_config() -> CompactVisionTransformerConfig {
        CompactVisionTransformerConfig::new(
            TinyTransformerConfig::new(260, 16, 2, 2, 32, 64),
            CompactConvVisionStemConfig::new(3, 4, 16, 4, 4),
        )
    }

    #[test]
    fn rgb_batch_conversion_preserves_nchw_channel_order() {
        let device = Default::default();
        let tensor =
            rgb_batch_to_nchw::<CpuBackend>(&[vec![255, 0, 0, 0, 128, 255]], 2, 1, &device)
                .unwrap();
        let values: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(values, vec![1.0, 0.0, 0.0, 128.0 / 255.0, 0.0, 1.0]);
    }

    #[test]
    fn compact_stem_has_stride_sixteen_and_shared_sequence() {
        let device = Default::default();
        let model = compact_config().init::<CpuBackend>(&device);
        let images = Tensor::<CpuBackend, 4>::zeros([1, 3, 17, 33], &device);
        let ids = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3], [1, 3]),
            &device,
        );
        // ceil(17/16) * ceil(33/16) = 2 * 3 visual tokens.
        assert_eq!(model.forward(images, ids).dims(), [1, 9, 37]);
    }

    #[test]
    fn compact_stem_gradient_reaches_first_convolution() {
        let device = Default::default();
        let model = compact_config().init::<CpuAutodiffBackend>(&device);
        let images = Tensor::<CpuAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..3 * 16 * 16)
                    .map(|index| index as f32 / (3 * 16 * 16) as f32)
                    .collect(),
                [1, 3, 16, 16],
            ),
            &device,
        );
        let ids = Tensor::<CpuAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3], [1, 3]),
            &device,
        );
        let output = model.forward(images, ids);
        let target = Tensor::<CpuAutodiffBackend, 3>::zeros(output.dims(), &device);
        let loss = MseLoss::new().forward(output, target, Reduction::Mean);
        let grads = loss.backward();
        let gradient = model
            .vision
            .pre_conv1
            .weight
            .grad(&grads)
            .expect("first convolution must receive a gradient");
        let magnitude: f32 = gradient.abs().sum().into_scalar();
        assert!(magnitude > 0.0, "first convolution gradient is zero");
    }

    #[test]
    fn compact_training_and_generation_boundaries_run() {
        let device = Default::default();
        let (model, mut optimizer) =
            init_compact_vision_training::<CpuAutodiffBackend>(&compact_byte_config(), &device);
        let images = Tensor::<CpuAutodiffBackend, 4>::ones([1, 3, 16, 16], &device);
        let tokenizer = ByteTokenizer::new();
        let batch = build_vision_ocr_batch::<CpuAutodiffBackend, _>(
            &tokenizer,
            &["hello".into()],
            48,
            &device,
        )
        .unwrap();
        let (model, loss) = train_compact_vision_step(model, &mut optimizer, images, &batch, 1e-3);
        assert!(loss.is_finite() && loss > 0.0);
        let model = to_eval_compact_vision_model(&model);
        let output = generate_compact_vision_ocr(
            &model,
            Tensor::<CpuBackend, 4>::ones([1, 3, 16, 16], &device),
            &tokenizer,
            4,
            64,
            &device,
        );
        assert!(output.len() <= 16);
    }

    #[test]
    fn compact_cached_prefill_matches_full_prefix_forward() {
        let device = Default::default();
        let model = compact_config().init::<CpuBackend>(&device);
        let images = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                (0..3 * 16 * 16)
                    .map(|index| index as f32 / (3 * 16 * 16) as f32)
                    .collect(),
                [1, 3, 16, 16],
            ),
            &device,
        );
        let ids = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let full: Vec<f32> = model
            .forward(images.clone(), ids.clone())
            .into_data()
            .to_vec()
            .unwrap();
        let prefix = model.vision.forward(images);
        let (cached, _) = model
            .transformer
            .prefill_cached_with_prefix_embeddings(prefix, ids);
        let cached: Vec<f32> = cached.into_data().to_vec().unwrap();
        assert_eq!(full.len(), cached.len());
        let maximum = full
            .iter()
            .zip(cached)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum <= 1e-5, "cached prefix drifted by {maximum}");
    }

    #[test]
    fn compact_cached_ocr_matches_legacy_full_forward_decoding() {
        let device = Default::default();
        let model = compact_byte_config().init::<CpuBackend>(&device);
        let image = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                (0..3 * 16 * 16)
                    .map(|index| (index % 251) as f32 / 250.0)
                    .collect(),
                [1, 3, 16, 16],
            ),
            &device,
        );
        let tokenizer = ByteTokenizer::new();
        let cached =
            generate_compact_vision_ocr(&model, image.clone(), &tokenizer, 12, 64, &device);

        let mut ids = vec![tokenizer.bos_id()];
        ids.extend(tokenizer.encode(OCR_PROMPT));
        ids.push(tokenizer.sep_id());
        let visual_tokens = model.vision.forward(image.clone()).dims()[1];
        let mut generated = Vec::new();
        for _ in 0..12 {
            if visual_tokens + ids.len() >= 64 {
                break;
            }
            let input = Tensor::<CpuBackend, 2, Int>::from_data(
                TensorData::new(
                    ids.iter().map(|id| i64::from(*id)).collect(),
                    [1, ids.len()],
                ),
                &device,
            );
            let logits = model.forward(image.clone(), input);
            let [_, sequence, vocab] = logits.dims();
            let last: Vec<f32> = logits
                .slice([0..1, sequence - 1..sequence, 0..vocab])
                .reshape([vocab])
                .into_data()
                .to_vec()
                .unwrap();
            let next = last
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .unwrap()
                .0 as u32;
            if next == tokenizer.eos_id() {
                break;
            }
            ids.push(next);
            generated.push(next);
        }
        let legacy = tokenizer.decode(&generated).unwrap_or_else(|_| {
            let bytes: Vec<u8> = generated
                .iter()
                .copied()
                .filter(|token| *token < 256)
                .map(|token| token as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        });
        assert_eq!(cached, legacy);
    }

    #[test]
    fn frozen_text_trunk_is_unchanged_by_visual_warmup_update() {
        let device = Default::default();
        let config = compact_byte_config();
        let text = config.transformer.init::<CpuAutodiffBackend>(&device);
        let control_ids = Tensor::<CpuAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 11, 12, 13], [1, 4]),
            &device,
        );
        let before: Vec<f32> = text
            .forward(control_ids.clone())
            .into_data()
            .to_vec()
            .unwrap();
        let model = config.init_with_transformer(text.no_grad(), &device);
        let mut optimizer = AdamConfig::new()
            .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
            .init::<CpuAutodiffBackend, CompactVisionTransformer<CpuAutodiffBackend>>();
        let tokenizer = ByteTokenizer::new();
        let batch = build_vision_ocr_batch::<CpuAutodiffBackend, _>(
            &tokenizer,
            &["hello".into()],
            48,
            &device,
        )
        .unwrap();
        let images = Tensor::<CpuAutodiffBackend, 4>::ones([1, 3, 16, 16], &device);
        let (model, loss, grad_norm) = train_compact_vision_step_guarded(
            model,
            &mut optimizer,
            images,
            &batch,
            1e-3,
            20.0,
            100.0,
        );
        assert!(loss.is_finite() && grad_norm.is_finite());
        let after: Vec<f32> = model
            .forward_text_only(control_ids)
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(
            before, after,
            "visual warmup changed a frozen text parameter"
        );
    }

    #[test]
    fn joint_text_retention_update_changes_the_shared_text_trunk() {
        let device = Default::default();
        let (model, mut optimizer) =
            init_compact_vision_training::<CpuAutodiffBackend>(&compact_byte_config(), &device);
        let control_ids = Tensor::<CpuAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 11, 12, 13], [1, 4]),
            &device,
        );
        let before: Vec<f32> = model
            .forward_text_only(control_ids.clone())
            .into_data()
            .to_vec()
            .unwrap();
        let batch = build_vision_ocr_batch::<CpuAutodiffBackend, _>(
            &ByteTokenizer::new(),
            &["retention update".into()],
            48,
            &device,
        )
        .unwrap();
        let (model, loss, grad_norm) =
            train_compact_text_step_guarded(model, &mut optimizer, &batch, 1e-2, 20.0, 100.0);
        assert!(loss.is_finite() && grad_norm.is_finite() && grad_norm > 0.0);
        let after: Vec<f32> = model
            .forward_text_only(control_ids)
            .into_data()
            .to_vec()
            .unwrap();
        assert_ne!(before, after, "text retention update left trunk unchanged");
    }

    #[test]
    fn native_vision_prefix_and_text_share_one_sequence() {
        let device = Default::default();
        let model = config().init::<CpuBackend>(&device);
        let patches = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.25f32; 4 * 12], [1, 4, 12]),
            &device,
        );
        let ids = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 4, 5], [1, 5]),
            &device,
        );
        let logits = model.forward(patches, 2, 2, ids);
        assert_eq!(logits.dims(), [1, 9, 37]);
    }

    #[test]
    fn different_images_produce_different_prefix_conditioned_logits() {
        let device = Default::default();
        let model = config().init::<CpuBackend>(&device);
        let dark = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; 4 * 12], [1, 4, 12]),
            &device,
        );
        let light = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![1.0f32; 4 * 12], [1, 4, 12]),
            &device,
        );
        let ids = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3], [1, 3]),
            &device,
        );
        let dark_logits: Vec<f32> = model
            .forward(dark, 2, 2, ids.clone())
            .into_data()
            .to_vec()
            .unwrap();
        let light_logits: Vec<f32> = model
            .forward(light, 2, 2, ids)
            .into_data()
            .to_vec()
            .unwrap();
        assert_ne!(dark_logits, light_logits);
    }

    #[test]
    fn loss_gradient_reaches_patch_projection() {
        let device = Default::default();
        let model = config().init::<CpuAutodiffBackend>(&device);
        let patches = Tensor::<CpuAutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..4 * 12).map(|index| index as f32 / 48.0).collect(),
                [1, 4, 12],
            ),
            &device,
        );
        let ids = Tensor::<CpuAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3], [1, 3]),
            &device,
        );
        let output = model.forward(patches, 2, 2, ids);
        let target = Tensor::<CpuAutodiffBackend, 3>::zeros(output.dims(), &device);
        let loss = MseLoss::new().forward(output, target, Reduction::Mean);
        let grads = loss.backward();
        let gradient = model
            .vision
            .patch_projection
            .weight
            .grad(&grads)
            .expect("patch projection must receive a gradient");
        let magnitude: f32 = gradient.abs().sum().into_scalar();
        assert!(magnitude > 0.0, "patch projection gradient is zero");
    }

    #[test]
    fn exported_vision_training_step_runs() {
        let device = Default::default();
        let (model, mut optimizer) =
            init_vision_training::<CpuAutodiffBackend>(&byte_config(), &device);
        let patches = Tensor::<CpuAutodiffBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 4 * 12], [1, 4, 12]),
            &device,
        );
        let tokenizer = ByteTokenizer::new();
        let batch = build_vision_ocr_batch::<CpuAutodiffBackend, _>(
            &tokenizer,
            &["hello".into()],
            48,
            &device,
        )
        .unwrap();
        let (_model, loss) = train_vision_step(model, &mut optimizer, patches, 2, 2, &batch, 1e-3);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn untrained_ocr_generation_is_bounded_and_total() {
        let device = Default::default();
        let model = byte_config().init::<CpuBackend>(&device);
        let patches = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 4 * 12], [1, 4, 12]),
            &device,
        );
        let output =
            generate_vision_ocr(&model, patches, 2, 2, &ByteTokenizer::new(), 4, 64, &device);
        assert!(output.len() <= 16);
    }
}
