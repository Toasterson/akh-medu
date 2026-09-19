//! The exact Ptah vision projector: geometry, configuration and weights.
//!
//! The pinned model ships a `qwen3vl_merger` CLIP tower as a separate manifest
//! layer (see [`crate::nlu::projector_artifact`]). This module reads that GGUF's
//! declared geometry and locates every tensor the forward pass needs, failing
//! loudly when the artifact does not match what the metadata claims.
//!
//! Loading is separated from the forward pass deliberately: the artifact is 931
//! MB of BF16 and every shape can be checked without a device, so a mismatch is
//! caught by a CPU test rather than by a GPU kernel producing wrong numbers.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use super::error::{NluError, NluResult};

/// Geometry declared by the projector GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// Transformer blocks in the vision tower.
    pub block_count: usize,
    /// Residual width.
    pub embedding_length: usize,
    /// MLP inner width.
    pub feed_forward_length: usize,
    /// Attention heads. `embedding_length` must divide evenly by this.
    pub head_count: usize,
    /// Square patch side, in pixels.
    pub patch_size: usize,
    /// Reference image side; not a resize target, see `vision_preprocess`.
    pub image_size: usize,
    /// Patches merged per side before projection.
    pub spatial_merge_size: usize,
    /// Width the merger projects into: the language model's residual stream.
    pub projection_dim: usize,
    /// LayerNorm epsilon. These are LayerNorms with bias, not RMSNorms.
    pub layer_norm_epsilon: f64,
    /// Per-channel normalization statistics.
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Whether any block feeds the language model directly. This artifact
    /// declares all false, so there is no deepstack routing to implement.
    pub deepstack_layers: Vec<bool>,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.embedding_length / self.head_count
    }

    /// Width entering the merger: one merge block of patches, concatenated.
    pub fn merged_width(&self) -> usize {
        self.embedding_length * self.spatial_merge_size * self.spatial_merge_size
    }

    /// True when no block requires deepstack routing.
    pub fn deepstack_is_disabled(&self) -> bool {
        self.deepstack_layers.iter().all(|enabled| !enabled)
    }

    /// Read the geometry from an open projector GGUF.
    pub fn from_gguf(content: &gguf_file::Content) -> NluResult<Self> {
        let fail = |reason: String| NluError::ModelLoadFailed { reason };
        let get = |key: &str| {
            content
                .metadata
                .get(key)
                .ok_or_else(|| fail(format!("projector GGUF is missing {key}")))
        };
        let u32_value = |key: &str| -> NluResult<usize> {
            get(key)?
                .to_u32()
                .map(|v| v as usize)
                .map_err(|e| fail(format!("{key}: {e}")))
        };
        let f32_value = |key: &str| -> NluResult<f32> {
            get(key)?.to_f32().map_err(|e| fail(format!("{key}: {e}")))
        };
        let triple = |key: &str| -> NluResult<[f32; 3]> {
            let values = get(key)?
                .to_vec()
                .map_err(|e| fail(format!("{key}: {e}")))?
                .iter()
                .map(|v| v.to_f32().map_err(|e| fail(format!("{key}: {e}"))))
                .collect::<NluResult<Vec<f32>>>()?;
            values
                .try_into()
                .map_err(|_| fail(format!("{key} is not three channels")))
        };

        let architecture = get("general.architecture")?
            .to_string()
            .map_err(|e| fail(format!("general.architecture: {e}")))?
            .clone();
        if architecture != "clip" {
            return Err(fail(format!(
                "expected a clip projector, got architecture {architecture}"
            )));
        }
        let projector = get("clip.projector_type")?
            .to_string()
            .map_err(|e| fail(format!("clip.projector_type: {e}")))?
            .clone();
        if projector != "qwen3vl_merger" {
            return Err(fail(format!(
                "this loader implements qwen3vl_merger, got {projector}"
            )));
        }

        let deepstack_layers = match content.metadata.get("clip.vision.is_deepstack_layers") {
            Some(value) => value
                .to_vec()
                .map_err(|e| fail(format!("is_deepstack_layers: {e}")))?
                .iter()
                .map(|v| v.to_bool().map_err(|e| fail(format!("deepstack: {e}"))))
                .collect::<NluResult<Vec<bool>>>()?,
            None => Vec::new(),
        };

        let config = Self {
            block_count: u32_value("clip.vision.block_count")?,
            embedding_length: u32_value("clip.vision.embedding_length")?,
            feed_forward_length: u32_value("clip.vision.feed_forward_length")?,
            head_count: u32_value("clip.vision.attention.head_count")?,
            patch_size: u32_value("clip.vision.patch_size")?,
            image_size: u32_value("clip.vision.image_size")?,
            spatial_merge_size: u32_value("clip.vision.spatial_merge_size")?,
            projection_dim: u32_value("clip.vision.projection_dim")?,
            layer_norm_epsilon: f32_value("clip.vision.attention.layer_norm_epsilon")? as f64,
            image_mean: triple("clip.vision.image_mean")?,
            image_std: triple("clip.vision.image_std")?,
            deepstack_layers,
        };
        if config.head_count == 0 || config.embedding_length % config.head_count != 0 {
            return Err(fail(format!(
                "embedding width {} is not divisible by {} heads",
                config.embedding_length, config.head_count
            )));
        }
        if config.spatial_merge_size == 0 || config.patch_size == 0 {
            return Err(fail("patch and merge sizes must be positive".into()));
        }
        if !config.deepstack_layers.is_empty()
            && config.deepstack_layers.len() != config.block_count
        {
            return Err(fail(format!(
                "{} deepstack flags for {} blocks",
                config.deepstack_layers.len(),
                config.block_count
            )));
        }
        Ok(config)
    }

    /// The preprocessing policy implied by this geometry.
    pub fn resize_policy(&self) -> super::vision_preprocess::ResizePolicy {
        let patch_area = (self.patch_size * self.spatial_merge_size).pow(2) as u32;
        super::vision_preprocess::ResizePolicy {
            align: (self.patch_size * self.spatial_merge_size) as u32,
            min_pixels: 256 * patch_area,
            max_pixels: 16_384 * patch_area,
        }
    }
}

/// Every tensor name the forward pass requires, in load order.
pub fn required_tensors(config: &VisionConfig) -> Vec<String> {
    let mut names = vec![
        "v.patch_embd.weight".to_string(),
        "v.patch_embd.bias".to_string(),
        "v.position_embd.weight".to_string(),
        "v.post_ln.weight".to_string(),
        "v.post_ln.bias".to_string(),
        "mm.0.weight".to_string(),
        "mm.0.bias".to_string(),
        "mm.2.weight".to_string(),
        "mm.2.bias".to_string(),
    ];
    for block in 0..config.block_count {
        for suffix in [
            "ln1.weight",
            "ln1.bias",
            "ln2.weight",
            "ln2.bias",
            "attn_qkv.weight",
            "attn_qkv.bias",
            "attn_out.weight",
            "attn_out.bias",
            "ffn_up.weight",
            "ffn_up.bias",
            "ffn_down.weight",
            "ffn_down.bias",
        ] {
            names.push(format!("v.blk.{block}.{suffix}"));
        }
    }
    names
}

/// Check that the artifact holds every required tensor, with the shapes the
/// declared geometry implies.
///
/// Returns the names of the optional temporal patch-embedding weights that are
/// present, since a video-capable projector carries a second frame.
pub fn validate_tensors(
    content: &gguf_file::Content,
    config: &VisionConfig,
) -> NluResult<Vec<String>> {
    let fail = |reason: String| NluError::ModelLoadFailed { reason };
    let width = config.embedding_length;
    let expected: Vec<(String, Vec<usize>)> = {
        let mut v = vec![
            // candle reports GGUF dimensions reversed, so weights read as
            // (out, in, ...) — the conventional layout, not the file's order.
            (
                "v.patch_embd.weight".to_string(),
                vec![width, 3, config.patch_size, config.patch_size],
            ),
            ("v.patch_embd.bias".to_string(), vec![width]),
            ("v.post_ln.weight".to_string(), vec![width]),
            ("v.post_ln.bias".to_string(), vec![width]),
            (
                "mm.0.weight".to_string(),
                vec![config.merged_width(), config.merged_width()],
            ),
            ("mm.0.bias".to_string(), vec![config.merged_width()]),
            (
                "mm.2.weight".to_string(),
                vec![config.projection_dim, config.merged_width()],
            ),
            ("mm.2.bias".to_string(), vec![config.projection_dim]),
        ];
        for block in 0..config.block_count {
            v.push((format!("v.blk.{block}.ln1.weight"), vec![width]));
            v.push((format!("v.blk.{block}.ln1.bias"), vec![width]));
            v.push((format!("v.blk.{block}.ln2.weight"), vec![width]));
            v.push((format!("v.blk.{block}.ln2.bias"), vec![width]));
            v.push((
                format!("v.blk.{block}.attn_qkv.weight"),
                vec![width * 3, width],
            ));
            v.push((format!("v.blk.{block}.attn_qkv.bias"), vec![width * 3]));
            v.push((
                format!("v.blk.{block}.attn_out.weight"),
                vec![width, width],
            ));
            v.push((format!("v.blk.{block}.attn_out.bias"), vec![width]));
            v.push((
                format!("v.blk.{block}.ffn_up.weight"),
                vec![config.feed_forward_length, width],
            ));
            v.push((
                format!("v.blk.{block}.ffn_up.bias"),
                vec![config.feed_forward_length],
            ));
            v.push((
                format!("v.blk.{block}.ffn_down.weight"),
                vec![width, config.feed_forward_length],
            ));
            v.push((format!("v.blk.{block}.ffn_down.bias"), vec![width]));
        }
        v
    };

    for (name, shape) in &expected {
        let info = content
            .tensor_infos
            .get(name.as_str())
            .ok_or_else(|| fail(format!("projector is missing tensor {name}")))?;
        let actual: Vec<usize> = info.shape.dims().to_vec();
        if &actual != shape {
            return Err(fail(format!(
                "tensor {name} has shape {actual:?}, geometry implies {shape:?}"
            )));
        }
    }

    // The position grid is learned at one resolution and interpolated for
    // others; record what it actually covers rather than assuming a side.
    let position = content
        .tensor_infos
        .get("v.position_embd.weight")
        .ok_or_else(|| fail("projector is missing v.position_embd.weight".into()))?;
    let dims = position.shape.dims();
    if dims.last() != Some(&width) {
        return Err(fail(format!(
            "position embedding width {dims:?} does not match {width}"
        )));
    }

    Ok(content
        .tensor_infos
        .keys()
        .filter(|name| name.starts_with("v.patch_embd.weight."))
        .cloned()
        .collect())
}

/// Positions the learned position grid covers, and its square side.
pub fn position_grid(content: &gguf_file::Content) -> NluResult<(usize, usize)> {
    let info = content.tensor_infos.get("v.position_embd.weight").ok_or(
        NluError::ModelLoadFailed {
            reason: "projector is missing v.position_embd.weight".into(),
        },
    )?;
    let positions = info.shape.dims()[0];
    let side = (positions as f64).sqrt().round() as usize;
    if side * side != positions {
        return Err(NluError::ModelLoadFailed {
            reason: format!("position grid of {positions} is not square"),
        });
    }
    Ok((positions, side))
}

/// One transformer block of the tower.
#[derive(Debug)]
struct VisionBlock {
    ln1_weight: Tensor,
    ln1_bias: Tensor,
    ln2_weight: Tensor,
    ln2_bias: Tensor,
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
    up_weight: Tensor,
    up_bias: Tensor,
    down_weight: Tensor,
    down_bias: Tensor,
}

/// The loaded vision tower: weights plus the geometry that shapes them.
#[derive(Debug)]
pub struct VisionTower {
    config: VisionConfig,
    /// Conv weights for each temporal frame. A still image reuses one frame for
    /// every entry, which is why they are summed rather than concatenated.
    patch_weights: Vec<Tensor>,
    patch_bias: Tensor,
    /// Learned position grid, `(positions, width)`.
    position: Tensor,
    position_side: usize,
    blocks: Vec<VisionBlock>,
    post_ln_weight: Tensor,
    post_ln_bias: Tensor,
    mm0_weight: Tensor,
    mm0_bias: Tensor,
    mm2_weight: Tensor,
    mm2_bias: Tensor,
    device: Device,
}

impl VisionTower {
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Load every tensor onto `device` as f32.
    ///
    /// The artifact is BF16; dequantizing once at load keeps the forward pass
    /// in one dtype rather than converting per block.
    pub fn load<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> NluResult<Self> {
        let config = VisionConfig::from_gguf(content)?;
        let temporal = validate_tensors(content, &config)?;
        let (_, position_side) = position_grid(content)?;
        let fail = |reason: String| NluError::ModelLoadFailed { reason };
        let mut get = |name: &str| -> NluResult<Tensor> {
            content
                .tensor(reader, name, device)
                .map_err(|e| fail(format!("projector tensor {name}: {e}")))?
                .dequantize(device)
                .map_err(|e| fail(format!("projector tensor {name} dequantize: {e}")))?
                .to_dtype(DType::F32)
                .map_err(|e| fail(format!("projector tensor {name} dtype: {e}")))
        };

        let mut patch_weights = vec![get("v.patch_embd.weight")?];
        for name in temporal {
            patch_weights.push(get(&name)?);
        }
        let patch_bias = get("v.patch_embd.bias")?;
        let position = get("v.position_embd.weight")?;
        let post_ln_weight = get("v.post_ln.weight")?;
        let post_ln_bias = get("v.post_ln.bias")?;
        let mm0_weight = get("mm.0.weight")?;
        let mm0_bias = get("mm.0.bias")?;
        let mm2_weight = get("mm.2.weight")?;
        let mm2_bias = get("mm.2.bias")?;

        let mut blocks = Vec::with_capacity(config.block_count);
        for block in 0..config.block_count {
            blocks.push(VisionBlock {
                ln1_weight: get(&format!("v.blk.{block}.ln1.weight"))?,
                ln1_bias: get(&format!("v.blk.{block}.ln1.bias"))?,
                ln2_weight: get(&format!("v.blk.{block}.ln2.weight"))?,
                ln2_bias: get(&format!("v.blk.{block}.ln2.bias"))?,
                qkv_weight: get(&format!("v.blk.{block}.attn_qkv.weight"))?,
                qkv_bias: get(&format!("v.blk.{block}.attn_qkv.bias"))?,
                out_weight: get(&format!("v.blk.{block}.attn_out.weight"))?,
                out_bias: get(&format!("v.blk.{block}.attn_out.bias"))?,
                up_weight: get(&format!("v.blk.{block}.ffn_up.weight"))?,
                up_bias: get(&format!("v.blk.{block}.ffn_up.bias"))?,
                down_weight: get(&format!("v.blk.{block}.ffn_down.weight"))?,
                down_bias: get(&format!("v.blk.{block}.ffn_down.bias"))?,
            });
        }

        Ok(Self {
            config,
            patch_weights,
            patch_bias,
            position,
            position_side,
            blocks,
            post_ln_weight,
            post_ln_bias,
            mm0_weight,
            mm0_bias,
            mm2_weight,
            mm2_bias,
            device: device.clone(),
        })
    }

    /// Project one preprocessed image into language-model embeddings.
    ///
    /// `pixels` is `(3, height, width)` f32, already resized to the policy in
    /// [`VisionConfig::resize_policy`] and normalized with the declared mean and
    /// standard deviation. Returns `(merged_patches, projection_dim)`.
    pub fn forward(&self, pixels: &Tensor) -> NluResult<Tensor> {
        let fail = |reason: String| NluError::ModelLoadFailed { reason };
        let map = |e: candle_core::Error, what: &str| fail(format!("vision {what}: {e}"));

        let (channels, height, width) = pixels
            .dims3()
            .map_err(|e| map(e, "input must be (3, height, width)"))?;
        if channels != 3 {
            return Err(fail(format!("expected 3 channels, got {channels}")));
        }
        let patch = self.config.patch_size;
        let merge = self.config.spatial_merge_size;
        if height % (patch * merge) != 0 || width % (patch * merge) != 0 {
            return Err(fail(format!(
                "{width}x{height} is not a multiple of patch {patch} times merge {merge}; \
                 resize with VisionConfig::resize_policy first"
            )));
        }
        let (grid_h, grid_w) = (height / patch, width / patch);

        // Patch embedding. Every temporal weight sees the same still frame, so
        // their contributions sum; a video path would feed distinct frames.
        let input = pixels
            .unsqueeze(0)
            .map_err(|e| map(e, "batch"))?
            .contiguous()
            .map_err(|e| map(e, "contiguous"))?;
        let mut embedded: Option<Tensor> = None;
        for weight in &self.patch_weights {
            let conv = input
                .conv2d(weight, 0, patch, 1, 1)
                .map_err(|e| map(e, "patch conv"))?;
            embedded = Some(match embedded {
                None => conv,
                Some(sum) => (sum + conv).map_err(|e| map(e, "temporal sum"))?,
            });
        }
        let embedded = embedded.ok_or_else(|| fail("projector has no patch weights".into()))?;

        // (1, width, gh, gw) -> (patches, width)
        let hidden = embedded
            .squeeze(0)
            .map_err(|e| map(e, "squeeze"))?
            .flatten_from(1)
            .map_err(|e| map(e, "flatten grid"))?
            .t()
            .map_err(|e| map(e, "transpose"))?
            .contiguous()
            .map_err(|e| map(e, "contiguous patches"))?;
        let hidden = hidden
            .broadcast_add(&self.patch_bias)
            .map_err(|e| map(e, "patch bias"))?;
        let hidden = hidden
            .add(&self.interpolated_positions(grid_h, grid_w)?)
            .map_err(|e| map(e, "position embedding"))?;

        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = self.block_forward(block, &hidden)?;
        }
        let hidden = layer_norm(
            &hidden,
            &self.post_ln_weight,
            &self.post_ln_bias,
            self.config.layer_norm_epsilon,
        )
        .map_err(|e| map(e, "post layer norm"))?;

        let merged = self.spatial_merge(&hidden, grid_h, grid_w)?;
        let projected = linear(&merged, &self.mm0_weight, &self.mm0_bias)
            .map_err(|e| map(e, "merger first layer"))?;
        let projected = projected.gelu().map_err(|e| map(e, "merger gelu"))?;
        linear(&projected, &self.mm2_weight, &self.mm2_bias)
            .map_err(|e| map(e, "merger second layer"))
    }

    fn block_forward(&self, block: &VisionBlock, hidden: &Tensor) -> NluResult<Tensor> {
        let map = |e: candle_core::Error, what: &str| NluError::ModelLoadFailed {
            reason: format!("vision {what}: {e}"),
        };
        let eps = self.config.layer_norm_epsilon;
        let heads = self.config.head_count;
        let head_dim = self.config.head_dim();
        let width = self.config.embedding_length;
        let patches = hidden.dim(0).map_err(|e| map(e, "patch count"))?;

        let normed = layer_norm(hidden, &block.ln1_weight, &block.ln1_bias, eps)
            .map_err(|e| map(e, "ln1"))?;
        let qkv = linear(&normed, &block.qkv_weight, &block.qkv_bias)
            .map_err(|e| map(e, "qkv"))?;
        let split = |offset: usize| -> NluResult<Tensor> {
            qkv.narrow(1, offset * width, width)
                .map_err(|e| map(e, "qkv split"))?
                .reshape((patches, heads, head_dim))
                .map_err(|e| map(e, "qkv reshape"))?
                .transpose(0, 1)
                .map_err(|e| map(e, "qkv transpose"))?
                .contiguous()
                .map_err(|e| map(e, "qkv contiguous"))
        };
        let (query, key, value) = (split(0)?, split(1)?, split(2)?);

        // No causal mask: the tower attends over all patches bidirectionally.
        let scores = query
            .matmul(&key.t().map_err(|e| map(e, "key transpose"))?)
            .map_err(|e| map(e, "scores"))?;
        let scores = (scores / (head_dim as f64).sqrt()).map_err(|e| map(e, "scale"))?;
        let weights = candle_nn::ops::softmax_last_dim(&scores).map_err(|e| map(e, "softmax"))?;
        let attended = weights
            .matmul(&value)
            .map_err(|e| map(e, "attend"))?
            .transpose(0, 1)
            .map_err(|e| map(e, "attend transpose"))?
            .reshape((patches, width))
            .map_err(|e| map(e, "attend reshape"))?;
        let attended = linear(&attended, &block.out_weight, &block.out_bias)
            .map_err(|e| map(e, "attn out"))?;
        let hidden = (hidden + attended).map_err(|e| map(e, "attn residual"))?;

        let normed = layer_norm(&hidden, &block.ln2_weight, &block.ln2_bias, eps)
            .map_err(|e| map(e, "ln2"))?;
        let up = linear(&normed, &block.up_weight, &block.up_bias)
            .map_err(|e| map(e, "ffn up"))?;
        let activated = up.gelu().map_err(|e| map(e, "ffn gelu"))?;
        let down = linear(&activated, &block.down_weight, &block.down_bias)
            .map_err(|e| map(e, "ffn down"))?;
        (hidden + down).map_err(|e| map(e, "ffn residual"))
    }

    /// Bilinearly resample the learned square position grid onto `(grid_h, grid_w)`.
    fn interpolated_positions(&self, grid_h: usize, grid_w: usize) -> NluResult<Tensor> {
        let map = |e: candle_core::Error, what: &str| NluError::ModelLoadFailed {
            reason: format!("vision {what}: {e}"),
        };
        let width = self.config.embedding_length;
        let side = self.position_side;
        if grid_h == side && grid_w == side {
            return self
                .position
                .to_dtype(DType::F32)
                .map_err(|e| map(e, "position dtype"));
        }
        let table: Vec<f32> = self
            .position
            .flatten_all()
            .map_err(|e| map(e, "position flatten"))?
            .to_vec1()
            .map_err(|e| map(e, "position read"))?;
        let mut out = vec![0.0f32; grid_h * grid_w * width];
        for row in 0..grid_h {
            // Align corners: the first and last learned positions pin the edges.
            let y = if grid_h == 1 {
                0.0
            } else {
                row as f32 * (side - 1) as f32 / (grid_h - 1) as f32
            };
            let (y0, fy) = (y.floor() as usize, y - y.floor());
            let y1 = (y0 + 1).min(side - 1);
            for col in 0..grid_w {
                let x = if grid_w == 1 {
                    0.0
                } else {
                    col as f32 * (side - 1) as f32 / (grid_w - 1) as f32
                };
                let (x0, fx) = (x.floor() as usize, x - x.floor());
                let x1 = (x0 + 1).min(side - 1);
                let base = (row * grid_w + col) * width;
                for channel in 0..width {
                    let at = |r: usize, c: usize| table[(r * side + c) * width + channel];
                    let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                    let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                    out[base + channel] = top * (1.0 - fy) + bottom * fy;
                }
            }
        }
        Tensor::from_vec(out, (grid_h * grid_w, width), &self.device)
            .map_err(|e| map(e, "position build"))
    }

    /// Group each `merge x merge` patch block into one row.
    fn spatial_merge(&self, hidden: &Tensor, grid_h: usize, grid_w: usize) -> NluResult<Tensor> {
        let map = |e: candle_core::Error, what: &str| NluError::ModelLoadFailed {
            reason: format!("vision {what}: {e}"),
        };
        let merge = self.config.spatial_merge_size;
        let width = self.config.embedding_length;
        hidden
            .reshape((grid_h / merge, merge, grid_w / merge, merge, width))
            .map_err(|e| map(e, "merge reshape"))?
            // (blocks_h, blocks_w, merge, merge, width) so one block's patches
            // are contiguous before flattening.
            .permute((0, 2, 1, 3, 4))
            .map_err(|e| map(e, "merge permute"))?
            .contiguous()
            .map_err(|e| map(e, "merge contiguous"))?
            .reshape(((grid_h / merge) * (grid_w / merge), merge * merge * width))
            .map_err(|e| map(e, "merge flatten"))
    }
}

/// `x * weight^T + bias` for a `(rows, in)` input and `(out, in)` weight.
fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
    x.matmul(&weight.t()?.contiguous()?)?.broadcast_add(bias)
}

/// LayerNorm over the last dimension, with weight and bias.
fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    centered
        .broadcast_div(&(variance + eps)?.sqrt()?)?
        .broadcast_mul(weight)?
        .broadcast_add(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_projector() -> Option<std::path::PathBuf> {
        let model = Path::new(
            "/usr/share/ollama/.ollama/models/blobs/\
             sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec",
        );
        super::super::projector_artifact::locate(model)
    }

    /// Artifact test: the real projector matches its own declared geometry.
    #[test]
    #[ignore]
    fn the_pinned_projector_matches_its_declared_geometry() {
        let path = pinned_projector().expect("pinned projector");
        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let config = VisionConfig::from_gguf(&content).unwrap();

        assert_eq!(config.block_count, 27);
        assert_eq!(config.embedding_length, 1152);
        assert_eq!(config.feed_forward_length, 4304);
        assert_eq!(config.head_count, 16);
        assert_eq!(config.head_dim(), 72);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.spatial_merge_size, 2);
        assert_eq!(config.projection_dim, 5120);
        assert_eq!(config.merged_width(), 4608);
        assert_eq!(config.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(config.image_std, [0.5, 0.5, 0.5]);

        // No deepstack routing in this artifact, so the language model needs no
        // change to accept its output.
        assert_eq!(config.deepstack_layers.len(), 27);
        assert!(config.deepstack_is_disabled());

        // Every tensor the forward pass needs, at the implied shape.
        let temporal = validate_tensors(&content, &config).unwrap();
        assert_eq!(temporal, vec!["v.patch_embd.weight.1".to_string()]);

        // The learned position grid is square and covers the reference image.
        let (positions, side) = position_grid(&content).unwrap();
        assert_eq!(positions, 2304);
        assert_eq!(side, 48);
        assert_eq!(side * config.patch_size, config.image_size);

        // The policy this geometry implies is the one preprocessing uses.
        assert_eq!(
            config.resize_policy(),
            super::super::vision_preprocess::ResizePolicy::qwen3vl()
        );
    }

    #[test]
    fn layer_norm_standardizes_then_applies_weight_and_bias() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let weight = Tensor::from_vec(vec![1.0f32; 4], (4,), &device).unwrap();
        let bias = Tensor::from_vec(vec![0.0f32; 4], (4,), &device).unwrap();
        let normed: Vec<f32> = layer_norm(&x, &weight, &bias, 1e-6)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mean: f32 = normed.iter().sum::<f32>() / 4.0;
        let variance: f32 = normed.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
        assert!(normed[0] < normed[3], "order must be preserved");

        // Weight scales and bias shifts after standardizing.
        let weight = Tensor::from_vec(vec![2.0f32; 4], (4,), &device).unwrap();
        let bias = Tensor::from_vec(vec![5.0f32; 4], (4,), &device).unwrap();
        let scaled: Vec<f32> = layer_norm(&x, &weight, &bias, 1e-6)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (plain, scaled) in normed.iter().zip(&scaled) {
            assert!((plain * 2.0 + 5.0 - scaled).abs() < 1e-4);
        }
    }

    #[test]
    fn linear_applies_an_out_by_in_weight() {
        let device = Device::Cpu;
        // Two rows of width 3, projected to width 2.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();
        let weight =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3), &device).unwrap();
        let bias = Tensor::from_vec(vec![10.0f32, 20.0], (2,), &device).unwrap();
        let out: Vec<f32> = linear(&x, &weight, &bias)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(out, vec![11.0, 22.0, 14.0, 25.0]);
    }

    /// Artifact test: run the real tower on CPU and check its output shape.
    #[test]
    #[ignore]
    fn the_pinned_projector_runs_a_forward_pass() {
        let path = pinned_projector().expect("pinned projector");
        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let device = Device::Cpu;
        let tower = VisionTower::load(&content, &mut file, &device).unwrap();
        let config = tower.config().clone();

        // 64x64 is a multiple of patch x merge, giving a 4x4 patch grid that
        // merges to 2x2. Small enough to run on CPU in a test.
        let (side, channels) = (64usize, 3usize);
        let pixels: Vec<f32> = (0..channels * side * side)
            .map(|i| ((i % 255) as f32 / 127.5) - 1.0)
            .collect();
        let input = Tensor::from_vec(pixels, (channels, side, side), &device).unwrap();

        let out = tower.forward(&input).unwrap();
        let grid = side / config.patch_size;
        let merged = (grid / config.spatial_merge_size).pow(2);
        assert_eq!(out.dims(), &[merged, config.projection_dim]);

        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "projector produced non-finite embeddings"
        );
        // A real projection is not degenerate: it must vary across channels.
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        assert!(variance > 1e-6, "embeddings are constant: variance {variance}");
    }

    /// The forward pass must refuse geometry it cannot patch evenly.
    #[test]
    #[ignore]
    fn the_pinned_projector_rejects_unaligned_input() {
        let path = pinned_projector().expect("pinned projector");
        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let device = Device::Cpu;
        let tower = VisionTower::load(&content, &mut file, &device).unwrap();
        // 48 is a multiple of the patch size but not of patch x merge.
        let input = Tensor::zeros((3, 48, 64), DType::F32, &device).unwrap();
        let error = tower.forward(&input).unwrap_err().to_string();
        assert!(error.contains("multiple of patch"), "{error}");
    }

    #[test]
    fn merged_width_and_head_dim_follow_the_geometry() {
        let config = VisionConfig {
            block_count: 2,
            embedding_length: 1152,
            feed_forward_length: 4304,
            head_count: 16,
            patch_size: 16,
            image_size: 768,
            spatial_merge_size: 2,
            projection_dim: 5120,
            layer_norm_epsilon: 1e-6,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            deepstack_layers: vec![false, false],
        };
        assert_eq!(config.head_dim(), 72);
        assert_eq!(config.merged_width(), 4608);
        assert!(config.deepstack_is_disabled());
        assert_eq!(config.resize_policy().align, 32);
    }

    #[test]
    fn required_tensor_list_covers_every_block() {
        let config = VisionConfig {
            block_count: 3,
            embedding_length: 8,
            feed_forward_length: 16,
            head_count: 2,
            patch_size: 2,
            image_size: 8,
            spatial_merge_size: 2,
            projection_dim: 4,
            layer_norm_epsilon: 1e-6,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            deepstack_layers: vec![],
        };
        let names = required_tensors(&config);
        assert_eq!(names.len(), 9 + 3 * 12);
        assert!(names.contains(&"v.blk.2.attn_qkv.weight".to_string()));
        assert!(names.contains(&"mm.2.bias".to_string()));
    }
}
