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
