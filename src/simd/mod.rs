//! SIMD-accelerated kernels for VSA operations.
//!
//! This module provides a `VsaKernel` trait with CPU-specific implementations.
//! At runtime, [`detect_isa`] determines the best available instruction set and
//! [`best_kernel`] returns the fastest implementation for the current CPU.
//!
//! # Supported ISA levels
//!
//! - **Generic**: Pure-Rust scalar fallback — works everywhere (illumos, ARM, etc.)
//! - **AVX2**: 256-bit SIMD for x86_64 Linux/illumos systems with AVX2 support

pub mod avx2;
pub mod generic;

/// Instruction set architecture level detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IsaLevel {
    /// Pure-Rust scalar operations, no SIMD.
    Generic,
    /// x86_64 AVX2 (256-bit vectors).
    Avx2,
}

impl std::fmt::Display for IsaLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsaLevel::Generic => write!(f, "Generic (scalar)"),
            IsaLevel::Avx2 => write!(f, "AVX2 (256-bit)"),
        }
    }
}

/// Detect the best ISA level available on the current CPU.
pub fn detect_isa() -> IsaLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return IsaLevel::Avx2;
        }
    }
    IsaLevel::Generic
}

/// Trait for SIMD-accelerated VSA kernels.
///
/// Each method operates on raw byte/i8 slices representing hypervectors.
/// Implementations must handle alignment and length requirements internally.
pub trait VsaKernel: Send + Sync {
    /// The ISA level this kernel targets.
    fn isa_level(&self) -> IsaLevel;

    /// XOR-bind two binary hypervectors (byte-wise XOR).
    ///
    /// `a` and `b` must have the same length.
    /// Writes result into `out`.
    fn xor_bind(&self, a: &[u8], b: &[u8], out: &mut [u8]);

    /// Bundle (element-wise addition) of `i8` hypervectors.
    ///
    /// Adds `src` element-wise into `acc` (accumulator), saturating at i8 bounds.
    fn bundle_add_i8(&self, acc: &mut [i8], src: &[i8]);

    /// Compute Hamming distance between two binary hypervectors.
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32;

    /// Cosine similarity between two `i8` hypervectors.
    ///
    /// Returns a value in `[-1.0, 1.0]`.
    fn cosine_similarity_i8(&self, a: &[i8], b: &[i8]) -> f32;

    /// Permute a binary hypervector by `shift` positions (circular left shift).
    fn permute(&self, data: &[u8], shift: usize, out: &mut [u8]);
}

/// Return the best available kernel for the current CPU.
pub fn best_kernel() -> Box<dyn VsaKernel> {
    match detect_isa() {
        #[cfg(target_arch = "x86_64")]
        IsaLevel::Avx2 => Box::new(avx2::Avx2Kernel),
        _ => Box::new(generic::GenericKernel),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_isa_returns_valid_level() {
        let level = detect_isa();
        // Should always be at least Generic.
        assert!(level >= IsaLevel::Generic);
    }

    #[test]
    fn best_kernel_returns_working_kernel() {
        let kernel = best_kernel();
        let level = kernel.isa_level();
        assert!(level >= IsaLevel::Generic);
    }

    /// Run the full kernel test suite against any implementation.
    pub fn kernel_conformance_tests(kernel: &dyn VsaKernel) {
        // XOR bind
        let a = vec![0xFF_u8; 128];
        let b = vec![0xAA_u8; 128];
        let mut out = vec![0u8; 128];
        kernel.xor_bind(&a, &b, &mut out);
        for &byte in &out {
            assert_eq!(byte, 0xFF ^ 0xAA);
        }

        // XOR self = zero
        kernel.xor_bind(&a, &a, &mut out);
        for &byte in &out {
            assert_eq!(byte, 0);
        }

        // Bundle add i8
        let mut acc = vec![0i8; 128];
        let src = vec![1i8; 128];
        kernel.bundle_add_i8(&mut acc, &src);
        for &val in &acc {
            assert_eq!(val, 1);
        }
        kernel.bundle_add_i8(&mut acc, &src);
        for &val in &acc {
            assert_eq!(val, 2);
        }

        // Bundle saturation
        let mut acc = vec![126i8; 128];
        let src = vec![10i8; 128];
        kernel.bundle_add_i8(&mut acc, &src);
        for &val in &acc {
            assert_eq!(val, 127); // saturated
        }

        // Hamming distance
        let a = vec![0xFF_u8; 128];
        let b = vec![0x00_u8; 128];
        let dist = kernel.hamming_distance(&a, &b);
        assert_eq!(dist, 128 * 8); // all bits differ

        let dist_self = kernel.hamming_distance(&a, &a);
        assert_eq!(dist_self, 0);

        // Cosine similarity
        let a = vec![1i8; 128];
        let b = vec![1i8; 128];
        let sim = kernel.cosine_similarity_i8(&a, &b);
        assert!((sim - 1.0).abs() < 0.001, "identical vectors should have cosine ~1.0");

        let c = vec![-1i8; 128];
        let sim_neg = kernel.cosine_similarity_i8(&a, &c);
        assert!(
            (sim_neg - (-1.0)).abs() < 0.001,
            "opposite vectors should have cosine ~-1.0"
        );

        // Permute
        let mut data = vec![0u8; 16];
        data[0] = 0b1000_0000; // MSB of byte 0 = bit position 0 in our scheme
        let mut out = vec![0u8; 16];
        kernel.permute(&data, 1, &mut out);
        // After circular left shift by 1 bit: bit position 0 moves to position 1,
        // which is byte 0, bit 6 (7 - 1) → 0b0100_0000
        assert_eq!(out[0], 0b0100_0000);
    }

    #[test]
    fn generic_kernel_conformance() {
        kernel_conformance_tests(&generic::GenericKernel);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_conformance() {
        if detect_isa() >= IsaLevel::Avx2 {
            kernel_conformance_tests(&avx2::Avx2Kernel);
        }
    }
}
