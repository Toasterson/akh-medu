//! Image geometry and normalization for the exact Ptah vision projector.
//!
//! The projector is `qwen3vl_merger`, which llama.cpp drives with a
//! dynamic-resolution preprocessor rather than a fixed square: `image_size` in
//! the GGUF is not a resize target. Sides are aligned to `patch_size *
//! spatial_merge_size` and the area is held between a minimum and maximum pixel
//! count derived from token limits. Getting this wrong misaligns every patch
//! silently, so the policy is reproduced here rather than approximated.
//!
//! Reference: `ggml-org/llama.cpp`, `tools/mtmd/mtmd-image.cpp`,
//! `calc_size_preserved_ratio` — the `smart_resize` of the transformers
//! implementation.

/// Aligned, area-bounded target geometry for one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizePolicy {
    /// Both sides are multiples of this. `patch_size * spatial_merge_size`.
    pub align: u32,
    /// Lower bound on the resized area, in pixels. 0 disables.
    pub min_pixels: u32,
    /// Upper bound on the resized area, in pixels. 0 disables.
    pub max_pixels: u32,
}

impl ResizePolicy {
    /// The policy for this model's projector.
    ///
    /// `align` is 16 x 2. The bounds are the Qwen3-VL token limits, 256 and
    /// 16,384, times a patch area of `16^2 * 2^2` = 1024, because the artifact
    /// carries no `clip.vision.image_{min,max}_pixels` keys of its own.
    pub fn qwen3vl() -> Self {
        Self {
            align: 32,
            min_pixels: 256 * 1024,
            max_pixels: 16_384 * 1024,
        }
    }

    /// Resized `(width, height)` preserving aspect ratio.
    ///
    /// Sides round to `align`, then the area is pulled under `max_pixels` or up
    /// to `min_pixels`. A side never falls below `align`, so a sliver image
    /// still yields at least one patch row.
    pub fn target_size(&self, width: u32, height: u32) -> (u32, u32) {
        let align = self.align.max(1);
        let round_by = |x: f64| ((x / align as f64).round() as u32).max(0) * align;
        let ceil_by = |x: f64| ((x / align as f64).ceil() as u32) * align;
        let floor_by = |x: f64| ((x / align as f64).floor() as u32) * align;

        let (w, h) = (width as f64, height as f64);
        let mut w_bar = round_by(w).max(align);
        let mut h_bar = round_by(h).max(align);

        let area = |a: u32, b: u32| a as u64 * b as u64;
        if self.max_pixels > 0 && area(w_bar, h_bar) > self.max_pixels as u64 {
            let beta = (w * h / self.max_pixels as f64).sqrt();
            w_bar = floor_by(w / beta).max(align);
            h_bar = floor_by(h / beta).max(align);
        } else if self.min_pixels > 0 && area(w_bar, h_bar) < self.min_pixels as u64 {
            let beta = (self.min_pixels as f64 / (w * h)).sqrt();
            w_bar = ceil_by(w * beta).max(align);
            h_bar = ceil_by(h * beta).max(align);
        }
        (w_bar, h_bar)
    }

    /// Patch grid the resized image produces, before spatial merging.
    pub fn patch_grid(&self, width: u32, height: u32, patch_size: u32) -> (u32, u32) {
        let (w, h) = self.target_size(width, height);
        (w / patch_size.max(1), h / patch_size.max(1))
    }
}

/// Scale 8-bit channel values to the projector's normalized range.
///
/// `clip.vision.image_mean` and `image_std` are 0.5 for every channel, so this
/// is `value / 127.5 - 1`. Kept explicit rather than folded into a constant so
/// a projector with different statistics cannot silently reuse it.
pub fn normalize_channel(value: u8, mean: f32, std: f32) -> f32 {
    ((value as f32 / 255.0) - mean) / std
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sides_are_aligned_and_aspect_is_preserved_within_a_step() {
        let policy = ResizePolicy::qwen3vl();
        for (w, h) in [(1920, 1080), (640, 480), (1000, 1000), (333, 777)] {
            let (tw, th) = policy.target_size(w, h);
            assert_eq!(tw % policy.align, 0, "width {tw} not aligned");
            assert_eq!(th % policy.align, 0, "height {th} not aligned");
            let original = w as f64 / h as f64;
            let resized = tw as f64 / th as f64;
            // Alignment quantizes the ratio; one alignment step of slack.
            let slack = policy.align as f64 / th.min(tw) as f64;
            assert!(
                (original - resized).abs() <= original.max(resized) * slack,
                "{w}x{h} -> {tw}x{th}"
            );
        }
    }

    #[test]
    fn a_huge_image_is_pulled_under_the_maximum_area() {
        let policy = ResizePolicy::qwen3vl();
        let (w, h) = policy.target_size(12_000, 9_000);
        assert!(
            w as u64 * h as u64 <= policy.max_pixels as u64,
            "{w}x{h} exceeds {}",
            policy.max_pixels
        );
        assert_eq!(w % policy.align, 0);
        assert_eq!(h % policy.align, 0);
    }

    #[test]
    fn a_tiny_image_is_pushed_up_to_the_minimum_area() {
        let policy = ResizePolicy::qwen3vl();
        let (w, h) = policy.target_size(48, 32);
        assert!(
            w as u64 * h as u64 >= policy.min_pixels as u64,
            "{w}x{h} below {}",
            policy.min_pixels
        );
    }

    #[test]
    fn a_sliver_never_loses_its_last_patch_row() {
        let policy = ResizePolicy::qwen3vl();
        let (w, h) = policy.target_size(4000, 1);
        assert!(w >= policy.align && h >= policy.align, "{w}x{h}");
        let (gw, gh) = policy.patch_grid(4000, 1, 16);
        assert!(gw >= 2 && gh >= 2, "grid {gw}x{gh} must span a merge block");
    }

    #[test]
    fn bounds_can_be_disabled_independently() {
        let unbounded = ResizePolicy {
            align: 32,
            min_pixels: 0,
            max_pixels: 0,
        };
        assert_eq!(unbounded.target_size(100, 100), (96, 96));
        let only_min = ResizePolicy {
            align: 32,
            min_pixels: 256 * 1024,
            max_pixels: 0,
        };
        let (w, h) = only_min.target_size(100, 100);
        assert!(w as u64 * h as u64 >= 256 * 1024);
    }

    #[test]
    fn normalization_maps_the_byte_range_onto_minus_one_to_one() {
        assert_eq!(normalize_channel(0, 0.5, 0.5), -1.0);
        assert_eq!(normalize_channel(255, 0.5, 0.5), 1.0);
        assert!((normalize_channel(128, 0.5, 0.5)).abs() < 0.01);
    }
}

/// An image decoded, resized to the policy, and normalized for the tower.
#[cfg(feature = "candle-backend")]
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// `(3, height, width)` in channel-major order, normalized.
    pub pixels: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Merged patch grid: the tokens this image occupies.
    pub grid: super::vision_positions::ImageGrid,
}

/// Decode an encoded image and prepare it for the tower.
///
/// Resampling uses a Catmull-Rom cubic filter, the closest the `image` crate
/// offers to the bicubic resize the reference preprocessor uses. That is an
/// approximation and is flagged as such; no numerical equivalence to the
/// reference is claimed for the resize itself.
#[cfg(feature = "candle-backend")]
pub fn prepare_image(
    bytes: &[u8],
    policy: &ResizePolicy,
    patch_size: usize,
    merge_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> super::error::NluResult<PreparedImage> {
    use super::error::NluError;
    let decoded = image::load_from_memory(bytes).map_err(|e| NluError::LlmGenerationFailed {
        reason: format!("image decode: {e}"),
    })?;
    let (width, height) = policy.target_size(decoded.width(), decoded.height());
    let resized = decoded
        .resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        .to_rgb8();

    let (width, height) = (width as usize, height as usize);
    let mut pixels = vec![0.0f32; 3 * width * height];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            pixels[channel * width * height + index] =
                normalize_channel(pixel.0[channel], mean[channel], std[channel]);
        }
    }
    let per_side = patch_size.max(1) * merge_size.max(1);
    Ok(PreparedImage {
        pixels,
        width,
        height,
        grid: super::vision_positions::ImageGrid {
            rows: height / per_side,
            cols: width / per_side,
        },
    })
}

#[cfg(all(test, feature = "candle-backend"))]
mod prepare_tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, _| {
            if x < width / 2 { image::Rgb([255, 0, 0]) } else { image::Rgb([255, 255, 255]) }
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        bytes
    }

    #[test]
    fn a_prepared_image_is_aligned_normalized_and_grid_consistent() {
        let policy = ResizePolicy::qwen3vl();
        let prepared = prepare_image(&png(300, 200), &policy, 16, 2, [0.5; 3], [0.5; 3]).unwrap();
        assert_eq!(prepared.width % 32, 0);
        assert_eq!(prepared.height % 32, 0);
        assert_eq!(prepared.pixels.len(), 3 * prepared.width * prepared.height);
        assert!(prepared.pixels.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert_eq!(prepared.grid.rows, prepared.height / 32);
        assert_eq!(prepared.grid.cols, prepared.width / 32);
        assert!(prepared.grid.tokens() >= 256, "minimum pixel bound implies >= 256 tokens");

        // Channel-major: the left half is red, so at a left-half pixel the red
        // channel is +1 and green is -1.
        let (w, h) = (prepared.width, prepared.height);
        let left = h / 2 * w + w / 8;
        assert!((prepared.pixels[left] - 1.0).abs() < 1e-6, "red channel");
        assert!((prepared.pixels[w * h + left] + 1.0).abs() < 1e-6, "green channel");
    }

    #[test]
    fn undecodable_bytes_are_refused() {
        let policy = ResizePolicy::qwen3vl();
        assert!(prepare_image(b"not an image", &policy, 16, 2, [0.5; 3], [0.5; 3]).is_err());
    }
}
