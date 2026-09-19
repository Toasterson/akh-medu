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
