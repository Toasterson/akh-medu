//! AVX2 (256-bit) SIMD kernel for x86_64 VSA operations.
//!
//! Falls through to the generic kernel for tail elements that don't fill
//! a full 256-bit (32-byte) register.

use super::{IsaLevel, VsaKernel};

/// AVX2-accelerated VSA kernel.
///
/// Uses 256-bit SIMD for bulk operations and scalar fallback for tails.
#[derive(Debug, Clone, Copy)]
pub struct Avx2Kernel;

#[cfg(target_arch = "x86_64")]
impl VsaKernel for Avx2Kernel {
    fn isa_level(&self) -> IsaLevel {
        IsaLevel::Avx2
    }

    fn xor_bind(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());

        if is_x86_feature_detected!("avx2") {
            // Safety: we've confirmed AVX2 support at runtime.
            unsafe { self.xor_bind_avx2(a, b, out) }
        } else {
            super::generic::GenericKernel.xor_bind(a, b, out);
        }
    }

    fn bundle_add_i8(&self, acc: &mut [i8], src: &[i8]) {
        debug_assert_eq!(acc.len(), src.len());

        if is_x86_feature_detected!("avx2") {
            unsafe { self.bundle_add_i8_avx2(acc, src) }
        } else {
            super::generic::GenericKernel.bundle_add_i8(acc, src);
        }
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len());
        // Use generic for now — popcount is tricky with AVX2 (no native vpopcount).
        super::generic::GenericKernel.hamming_distance(a, b)
    }

    fn cosine_similarity_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        if is_x86_feature_detected!("avx2") {
            unsafe { self.cosine_similarity_i8_avx2(a, b) }
        } else {
            super::generic::GenericKernel.cosine_similarity_i8(a, b)
        }
    }

    fn permute(&self, data: &[u8], shift: usize, out: &mut [u8]) {
        // Bit-level permutation doesn't benefit as much from SIMD for arbitrary shifts.
        super::generic::GenericKernel.permute(data, shift, out);
    }
}

#[cfg(target_arch = "x86_64")]
impl Avx2Kernel {
    #[target_feature(enable = "avx2")]
    unsafe fn xor_bind_avx2(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 32;
        let remainder = len % 32;

        for i in 0..chunks {
            let offset = i * 32;
            unsafe {
                let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
                let result = _mm256_xor_si256(va, vb);
                _mm256_storeu_si256(out.as_mut_ptr().add(offset) as *mut __m256i, result);
            }
        }

        // Scalar tail
        let tail_start = chunks * 32;
        for i in 0..remainder {
            out[tail_start + i] = a[tail_start + i] ^ b[tail_start + i];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn bundle_add_i8_avx2(&self, acc: &mut [i8], src: &[i8]) {
        use std::arch::x86_64::*;

        let len = acc.len();
        let chunks = len / 32;
        let remainder = len % 32;

        for i in 0..chunks {
            let offset = i * 32;
            unsafe {
                let va = _mm256_loadu_si256(acc.as_ptr().add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(src.as_ptr().add(offset) as *const __m256i);
                let result = _mm256_adds_epi8(va, vb); // saturating add
                _mm256_storeu_si256(acc.as_mut_ptr().add(offset) as *mut __m256i, result);
            }
        }

        // Scalar tail
        let tail_start = chunks * 32;
        for i in 0..remainder {
            acc[tail_start + i] = acc[tail_start + i].saturating_add(src[tail_start + i]);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn cosine_similarity_i8_avx2(&self, a: &[i8], b: &[i8]) -> f32 {
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 32;

        let mut dot_acc;
        let mut norm_a_acc;
        let mut norm_b_acc;

        unsafe {
            dot_acc = _mm256_setzero_si256();
            norm_a_acc = _mm256_setzero_si256();
            norm_b_acc = _mm256_setzero_si256();

            for i in 0..chunks {
                let offset = i * 32;
                let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

                // Unpack to 16-bit for proper signed multiplication
                let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
                let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
                let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
                let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

                // 16-bit multiply and horizontal pair-add to 32-bit
                let dot_lo = _mm256_madd_epi16(va_lo, vb_lo);
                let dot_hi = _mm256_madd_epi16(va_hi, vb_hi);
                dot_acc = _mm256_add_epi32(dot_acc, _mm256_add_epi32(dot_lo, dot_hi));

                let na_lo = _mm256_madd_epi16(va_lo, va_lo);
                let na_hi = _mm256_madd_epi16(va_hi, va_hi);
                norm_a_acc = _mm256_add_epi32(norm_a_acc, _mm256_add_epi32(na_lo, na_hi));

                let nb_lo = _mm256_madd_epi16(vb_lo, vb_lo);
                let nb_hi = _mm256_madd_epi16(vb_hi, vb_hi);
                norm_b_acc = _mm256_add_epi32(norm_b_acc, _mm256_add_epi32(nb_lo, nb_hi));
            }
        }

        // Horizontal sum of 256-bit i32 vectors
        let dot = unsafe { hsum_epi32(dot_acc) };
        let norm_a = unsafe { hsum_epi32(norm_a_acc) };
        let norm_b = unsafe { hsum_epi32(norm_b_acc) };

        // Scalar tail
        let tail_start = chunks * 32;
        let mut dot_tail: i64 = dot as i64;
        let mut norm_a_tail: i64 = norm_a as i64;
        let mut norm_b_tail: i64 = norm_b as i64;
        for i in tail_start..len {
            let av = a[i] as i64;
            let bv = b[i] as i64;
            dot_tail += av * bv;
            norm_a_tail += av * av;
            norm_b_tail += bv * bv;
        }

        let denom = ((norm_a_tail as f64).sqrt() * (norm_b_tail as f64).sqrt()) as f32;
        if denom == 0.0 {
            return 0.0;
        }
        (dot_tail as f32) / denom
    }
}

/// Horizontal sum of 8 packed i32 values in a __m256i register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    // All intrinsics below are safe when AVX2 is enabled (guaranteed by #[target_feature]).
    // This function is `unsafe fn` so the body is an unsafe context in edition 2024
    // only when called from an unsafe block, which the callers already handle.
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b_00_01_00_01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

// Provide a stub for non-x86_64 targets so the module compiles.
#[cfg(not(target_arch = "x86_64"))]
impl VsaKernel for Avx2Kernel {
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
