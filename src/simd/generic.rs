//! Pure-Rust scalar fallback for VSA kernel operations.
//!
//! This implementation works on all platforms (including illumos) and serves
//! as the reference implementation for correctness testing.

use super::{IsaLevel, VsaKernel};

/// Pure-Rust scalar VSA kernel — no SIMD intrinsics.
#[derive(Debug, Clone, Copy)]
pub struct GenericKernel;

impl VsaKernel for GenericKernel {
    fn isa_level(&self) -> IsaLevel {
        IsaLevel::Generic
    }

    fn xor_bind(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        for ((o, &av), &bv) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = av ^ bv;
        }
    }

    fn bundle_add_i8(&self, acc: &mut [i8], src: &[i8]) {
        debug_assert_eq!(acc.len(), src.len());
        for (a, &s) in acc.iter_mut().zip(src.iter()) {
            *a = a.saturating_add(s);
        }
    }

    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&av, &bv)| (av ^ bv).count_ones())
            .sum()
    }

    fn cosine_similarity_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot: i64 = 0;
        let mut norm_a: i64 = 0;
        let mut norm_b: i64 = 0;
        for (&av, &bv) in a.iter().zip(b.iter()) {
            let av = av as i64;
            let bv = bv as i64;
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }
        let denom = ((norm_a as f64).sqrt() * (norm_b as f64).sqrt()) as f32;
        if denom == 0.0 {
            return 0.0;
        }
        (dot as f32) / denom
    }

    fn permute(&self, data: &[u8], shift: usize, out: &mut [u8]) {
        debug_assert_eq!(data.len(), out.len());
        let total_bits = data.len() * 8;
        if total_bits == 0 {
            return;
        }
        let shift = shift % total_bits;
        if shift == 0 {
            out.copy_from_slice(data);
            return;
        }

        // Circular left shift in bit-space:
        // For each output bit position `i`, the source bit is at `(i - shift) mod total_bits`.
        // Equivalently, output bit `i` comes from input bit `(i + total_bits - shift) mod total_bits`.
        for o in out.iter_mut() {
            *o = 0;
        }
        for bit_pos in 0..total_bits {
            let src_bit_pos = (bit_pos + total_bits - shift) % total_bits;
            let src_byte = src_bit_pos / 8;
            let src_bit = 7 - (src_bit_pos % 8); // MSB-first
            let dst_byte = bit_pos / 8;
            let dst_bit = 7 - (bit_pos % 8);
            let bit_val = (data[src_byte] >> src_bit) & 1;
            out[dst_byte] |= bit_val << dst_bit;
        }
    }
}
