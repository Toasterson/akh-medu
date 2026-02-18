//! Vector Symbolic Architecture (VSA) core.
//!
//! This module implements hyperdimensional computing (HDC) with configurable
//! dimensions (default 10,000). It provides:
//!
//! - [`HyperVec`] — the fundamental hypervector type
//! - [`VsaOps`] — bind, bundle, permute, similarity operations
//! - [`ItemMemory`] — symbol-to-vector mapping with ANN search
//! - Encoding from symbols to vectors

pub mod code_encode;
pub mod encode;
pub mod grounding;
pub mod item_memory;
pub mod ops;

use serde::{Deserialize, Serialize};

/// Configurable hypervector dimensionality.
///
/// Typical values: 10,000 for good capacity, 1,000 for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension(pub usize);

impl Dimension {
    /// Standard high-capacity dimension.
    pub const DEFAULT: Self = Self(10_000);

    /// Smaller dimension for fast testing.
    pub const TEST: Self = Self(1_000);

    /// Number of bytes needed to store a binary vector at this dimension.
    /// Each bit is one component, packed into bytes.
    pub fn binary_byte_len(self) -> usize {
        (self.0 + 7) / 8
    }

    /// Number of bytes for an i8 vector (one byte per component).
    pub fn i8_byte_len(self) -> usize {
        self.0
    }
}

impl Default for Dimension {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Encoding scheme for hypervectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Encoding {
    /// Binary bipolar: each component is ±1, stored as bits (1 = +1, 0 = -1).
    /// Bind = XOR, Bundle = majority vote, Similarity = Hamming distance.
    Bipolar,
    // Future: FHRR, SSP
}

impl Default for Encoding {
    fn default() -> Self {
        Self::Bipolar
    }
}

impl std::fmt::Display for Encoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Encoding::Bipolar => write!(f, "Bipolar"),
        }
    }
}

/// A hypervector — the fundamental unit of VSA computation.
///
/// In bipolar encoding, this stores `dim` bits packed into a `Vec<u8>`.
/// Component `i` is at bit `i` of the packed representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperVec {
    /// The raw data (bit-packed for Bipolar, one byte per component for integer encodings).
    data: Vec<u8>,
    /// Number of components (not bytes).
    dim: Dimension,
    /// Encoding scheme.
    encoding: Encoding,
}

impl HyperVec {
    /// Create a new hypervector from raw data.
    ///
    /// The caller must ensure `data.len()` matches the expected size
    /// for the given dimension and encoding.
    pub fn from_raw(data: Vec<u8>, dim: Dimension, encoding: Encoding) -> Self {
        debug_assert_eq!(
            data.len(),
            match encoding {
                Encoding::Bipolar => dim.binary_byte_len(),
            }
        );
        Self {
            data,
            dim,
            encoding,
        }
    }

    /// Create a zero hypervector (all bits 0).
    pub fn zero(dim: Dimension, encoding: Encoding) -> Self {
        let len = match encoding {
            Encoding::Bipolar => dim.binary_byte_len(),
        };
        Self {
            data: vec![0u8; len],
            dim,
            encoding,
        }
    }

    /// Raw byte data of this hypervector.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable raw byte data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// The dimension of this hypervector.
    pub fn dim(&self) -> Dimension {
        self.dim
    }

    /// The encoding scheme.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Number of raw bytes.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Get a single bit (for Bipolar encoding). Returns `true` if the bit is 1 (+1).
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index < self.dim.0);
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        (self.data[byte_idx] >> bit_idx) & 1 == 1
    }

    /// Set a single bit (for Bipolar encoding).
    pub fn set_bit(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.dim.0);
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        if value {
            self.data[byte_idx] |= 1 << bit_idx;
        } else {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_byte_lengths() {
        assert_eq!(Dimension(8).binary_byte_len(), 1);
        assert_eq!(Dimension(10).binary_byte_len(), 2);
        assert_eq!(Dimension(10_000).binary_byte_len(), 1250);
        assert_eq!(Dimension(10_000).i8_byte_len(), 10_000);
    }

    #[test]
    fn hypervec_zero() {
        let hv = HyperVec::zero(Dimension::TEST, Encoding::Bipolar);
        assert_eq!(hv.dim(), Dimension::TEST);
        assert_eq!(hv.encoding(), Encoding::Bipolar);
        assert!(hv.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn hypervec_bit_operations() {
        let mut hv = HyperVec::zero(Dimension(16), Encoding::Bipolar);
        assert!(!hv.get_bit(0));
        hv.set_bit(0, true);
        assert!(hv.get_bit(0));
        hv.set_bit(7, true);
        assert!(hv.get_bit(7));
        hv.set_bit(8, true);
        assert!(hv.get_bit(8));
        hv.set_bit(0, false);
        assert!(!hv.get_bit(0));
    }
}
