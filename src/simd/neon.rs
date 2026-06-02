//! ARM NEON (128-bit) SIMD kernel for aarch64 VSA operations.
//!
//! Targets Apple M-series (M1/M2/M3) and other aarch64 CPUs with NEON.
//! NEON is mandatory on aarch64, so no runtime feature detection is needed.
//!
//! Falls through to the generic kernel for tail elements that don't fill
//! a full 128-bit (16-byte) register.

use super::{IsaLevel, VsaKernel};

/// NEON-accelerated VSA kernel for aarch64.
///
/// Uses 128-bit SIMD (16 bytes/register). NEON is baseline on aarch64 —
/// no runtime feature detection required.
#[derive(Debug, Clone, Copy)]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl VsaKernel for NeonKernel {
    fn isa_level(&self) -> IsaLevel {
        IsaLevel::Neon
    }

    fn xor_bind(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        // Safety: NEON is always available on aarch64.
        unsafe { self.xor_bind_neon(a, b, out) }
    }

    fn bundle_add_i8(&self, acc: &mut [i8], src: &[i8]) {
        debug_assert_eq!(acc.len(), src.len());
        unsafe { self.bundle_add_i8_neon(acc, src) }
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len());
        // NEON has vcntq_u8 (byte-level popcount) — much faster than scalar.
        unsafe { self.hamming_distance_neon(a, b) }
    }

    fn cosine_similarity_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        unsafe { self.cosine_similarity_i8_neon(a, b) }
    }

    fn permute(&self, data: &[u8], shift: usize, out: &mut [u8]) {
        // Bit-level permutation doesn't benefit from SIMD for arbitrary shifts.
        super::generic::GenericKernel.permute(data, shift, out);
    }
}

#[cfg(target_arch = "aarch64")]
impl NeonKernel {
    /// XOR bind using `veorq_u8` (128-bit XOR, 16 bytes per op).
    ///
    /// # Safety
    /// Caller must ensure NEON is available (always true on aarch64).
    unsafe fn xor_bind_neon(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let offset = i * 16;
            unsafe {
                let va = vld1q_u8(a.as_ptr().add(offset));
                let vb = vld1q_u8(b.as_ptr().add(offset));
                let result = veorq_u8(va, vb);
                vst1q_u8(out.as_mut_ptr().add(offset), result);
            }
        }

        // Scalar tail
        let tail_start = chunks * 16;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] ^ b[tail_start + i];
        }
    }

    /// Saturating i8 add using `vqaddq_s8` (16 elements per op).
    ///
    /// # Safety
    /// Caller must ensure NEON is available.
    unsafe fn bundle_add_i8_neon(&self, acc: &mut [i8], src: &[i8]) {
        use std::arch::aarch64::*;

        let len = acc.len();
        let chunks = len / 16;
        let remainder = len % 16;

        for i in 0..chunks {
            let offset = i * 16;
            unsafe {
                let va = vld1q_s8(acc.as_ptr().add(offset));
                let vb = vld1q_s8(src.as_ptr().add(offset));
                let result = vqaddq_s8(va, vb); // saturating add
                vst1q_s8(acc.as_mut_ptr().add(offset), result);
            }
        }

        // Scalar tail
        let tail_start = chunks * 16;
        for i in 0..remainder {
            acc[tail_start + i] = acc[tail_start + i].saturating_add(src[tail_start + i]);
        }
    }

    /// Hamming distance using `vcntq_u8` (byte popcount) + horizontal add.
    ///
    /// NEON's `vcntq_u8` counts set bits per byte — this is the key advantage
    /// over AVX2 which lacks native popcount for packed bytes.
    ///
    /// # Safety
    /// Caller must ensure NEON is available.
    unsafe fn hamming_distance_neon(&self, a: &[u8], b: &[u8]) -> u32 {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        // Accumulate popcount in u16 lanes to avoid overflow (max 16 * 8 = 128 per chunk).
        let mut total: u64 = 0;

        for i in 0..chunks {
            let offset = i * 16;
            unsafe {
                let va = vld1q_u8(a.as_ptr().add(offset));
                let vb = vld1q_u8(b.as_ptr().add(offset));
                let xored = veorq_u8(va, vb);
                // Count bits per byte
                let popcnt = vcntq_u8(xored);
                // Pairwise add bytes→u16, then u16→u32, then u32→u64
                let sum16 = vpaddlq_u8(popcnt);
                let sum32 = vpaddlq_u16(sum16);
                let sum64 = vpaddlq_u32(sum32);
                // Extract the two u64 lanes and add
                total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
            }
        }

        // Scalar tail
        let tail_start = chunks * 16;
        for i in 0..remainder {
            total += (a[tail_start + i] ^ b[tail_start + i]).count_ones() as u64;
        }

        total as u32
    }

    /// Cosine similarity using i16 multiplication + pairwise accumulation.
    ///
    /// Strategy: unpack i8→i16 (8 elements), multiply, accumulate as i32.
    ///
    /// # Safety
    /// Caller must ensure NEON is available.
    unsafe fn cosine_similarity_i8_neon(&self, a: &[i8], b: &[i8]) -> f32 {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 16;

        let mut dot_acc: i64 = 0;
        let mut norm_a_acc: i64 = 0;
        let mut norm_b_acc: i64 = 0;

        for i in 0..chunks {
            let offset = i * 16;
            unsafe {
                let va = vld1q_s8(a.as_ptr().add(offset));
                let vb = vld1q_s8(b.as_ptr().add(offset));

                // Unpack low/high halves from i8 to i16
                let va_lo = vmovl_s8(vget_low_s8(va)); // 8 × i16
                let va_hi = vmovl_s8(vget_high_s8(va)); // 8 × i16
                let vb_lo = vmovl_s8(vget_low_s8(vb));
                let vb_hi = vmovl_s8(vget_high_s8(vb));

                // Multiply i16×i16→i16 (no overflow risk: i8 range fits in i16 product)
                // then widen-accumulate to i32
                let dot_lo = vmull_s16(vget_low_s16(va_lo), vget_low_s16(vb_lo));
                let dot_lo = vmlal_s16(dot_lo, vget_high_s16(va_lo), vget_high_s16(vb_lo));
                let dot_hi = vmull_s16(vget_low_s16(va_hi), vget_low_s16(vb_hi));
                let dot_hi = vmlal_s16(dot_hi, vget_high_s16(va_hi), vget_high_s16(vb_hi));
                let dot_sum = vaddq_s32(dot_lo, dot_hi);
                dot_acc += vaddlvq_s32(dot_sum);

                let na_lo = vmull_s16(vget_low_s16(va_lo), vget_low_s16(va_lo));
                let na_lo = vmlal_s16(na_lo, vget_high_s16(va_lo), vget_high_s16(va_lo));
                let na_hi = vmull_s16(vget_low_s16(va_hi), vget_low_s16(va_hi));
                let na_hi = vmlal_s16(na_hi, vget_high_s16(va_hi), vget_high_s16(va_hi));
                let na_sum = vaddq_s32(na_lo, na_hi);
                norm_a_acc += vaddlvq_s32(na_sum);

                let nb_lo = vmull_s16(vget_low_s16(vb_lo), vget_low_s16(vb_lo));
                let nb_lo = vmlal_s16(nb_lo, vget_high_s16(vb_lo), vget_high_s16(vb_lo));
                let nb_hi = vmull_s16(vget_low_s16(vb_hi), vget_low_s16(vb_hi));
                let nb_hi = vmlal_s16(nb_hi, vget_high_s16(vb_hi), vget_high_s16(vb_hi));
                let nb_sum = vaddq_s32(nb_lo, nb_hi);
                norm_b_acc += vaddlvq_s32(nb_sum);
            }
        }

        // Scalar tail
        let tail_start = chunks * 16;
        for i in tail_start..len {
            let av = a[i] as i64;
            let bv = b[i] as i64;
            dot_acc += av * bv;
            norm_a_acc += av * av;
            norm_b_acc += bv * bv;
        }

        let denom = ((norm_a_acc as f64).sqrt() * (norm_b_acc as f64).sqrt()) as f32;
        if denom == 0.0 {
            return 0.0;
        }
        (dot_acc as f32) / denom
    }
}

// Provide a stub for non-aarch64 targets so the module compiles.
#[cfg(not(target_arch = "aarch64"))]
impl VsaKernel for NeonKernel {
    fn isa_level(&self) -> IsaLevel {
        IsaLevel::Generic
    }
    fn xor_bind(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        super::generic::GenericKernel.xor_bind(a, b, out);
    }
    fn bundle_add_i8(&self, acc: &mut [i8], src: &[i8]) {
        super::generic::GenericKernel.bundle_add_i8(acc, src);
    }
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        super::generic::GenericKernel.hamming_distance(a, b)
    }
    fn cosine_similarity_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        super::generic::GenericKernel.cosine_similarity_i8(a, b)
    }
    fn permute(&self, data: &[u8], shift: usize, out: &mut [u8]) {
        super::generic::GenericKernel.permute(data, shift, out);
    }
}
