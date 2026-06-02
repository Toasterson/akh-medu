//! PolarQuant KV cache compression for Candle LLM backend.
//!
//! Implements stage 1 of TurboQuant (ICLR 2026) — compressing KV cache vectors
//! from fp16/f32 to 3-bit representation via:
//!
//! 1. **Fast Walsh-Hadamard Transform (FWHT)** — deterministic random rotation
//!    that decorrelates dimensions, O(d log d)
//! 2. **Polar transform** — converts RoPE dimension pairs (x,y) → (r,θ),
//!    smoothing outliers that appear in single dimensions
//! 3. **Lloyd-Max scalar quantization** — data-oblivious 3-bit quantization
//!    with pre-computed optimal reconstruction levels for Beta distribution
//!
//! Target: 5.3× compression on Qwen2.5-7B KV cache (~4 GB → ~750 MB),
//! enabling 7B models in 16 GB RAM alongside the engine.

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform
// ---------------------------------------------------------------------------

/// Apply the Fast Walsh-Hadamard Transform in-place.
///
/// Operates on f32 slices. The input length must be a power of 2.
/// If not, the caller should pad to the next power of 2.
///
/// Complexity: O(n log n) additions/subtractions, no multiplications.
pub fn fwht_in_place(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");

    let mut step = 1;
    while step < n {
        for i in (0..n).step_by(step * 2) {
            for j in i..i + step {
                let a = data[j];
                let b = data[j + step];
                data[j] = a + b;
                data[j + step] = a - b;
            }
        }
        step *= 2;
    }
}

/// Apply FWHT with normalization (preserves Euclidean norm).
pub fn fwht_normalized(data: &mut [f32]) {
    fwht_in_place(data);
    let n = data.len() as f32;
    let scale = 1.0 / n.sqrt();
    for x in data.iter_mut() {
        *x *= scale;
    }
}

/// Apply a seeded sign-flip before FWHT (randomized Hadamard).
///
/// This creates a pseudo-random rotation matrix deterministically:
/// multiply each element by +1 or -1 based on the seed, then apply FWHT.
/// The seed ensures reproducibility across sessions.
pub fn randomized_hadamard(data: &mut [f32], seed: u64) {
    // Simple deterministic sign-flip based on seed.
    let mut hash = seed;
    for x in data.iter_mut() {
        // Simple hash: xorshift64
        hash ^= hash << 13;
        hash ^= hash >> 7;
        hash ^= hash << 17;
        if hash & 1 == 1 {
            *x = -*x;
        }
    }
    fwht_normalized(data);
}

/// Inverse randomized Hadamard transform.
///
/// FWHT is its own inverse (up to scaling), so we apply FWHT then undo sign-flips.
pub fn inverse_randomized_hadamard(data: &mut [f32], seed: u64) {
    fwht_normalized(data);
    // Undo the sign-flips (same operation — flipping twice is identity).
    let mut hash = seed;
    for x in data.iter_mut() {
        hash ^= hash << 13;
        hash ^= hash >> 7;
        hash ^= hash << 17;
        if hash & 1 == 1 {
            *x = -*x;
        }
    }
}

// ---------------------------------------------------------------------------
// Polar transform (for RoPE dimension pairs)
// ---------------------------------------------------------------------------

/// Convert consecutive dimension pairs from Cartesian (x,y) to polar (r,θ).
///
/// RoPE applies rotations to consecutive pairs of dimensions, so the polar
/// representation is natural and smooths outlier values.
///
/// Input: `[x0, y0, x1, y1, ...]` → Output: `[r0, θ0, r1, θ1, ...]`
pub fn cartesian_to_polar(data: &mut [f32]) {
    debug_assert!(data.len() % 2 == 0, "polar transform requires even length");
    let pairs = data.len() / 2;
    for i in 0..pairs {
        let x = data[i * 2];
        let y = data[i * 2 + 1];
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        data[i * 2] = r;
        data[i * 2 + 1] = theta;
    }
}

/// Convert consecutive dimension pairs from polar (r,θ) back to Cartesian (x,y).
pub fn polar_to_cartesian(data: &mut [f32]) {
    debug_assert!(data.len() % 2 == 0, "polar transform requires even length");
    let pairs = data.len() / 2;
    for i in 0..pairs {
        let r = data[i * 2];
        let theta = data[i * 2 + 1];
        let x = r * theta.cos();
        let y = r * theta.sin();
        data[i * 2] = x;
        data[i * 2 + 1] = y;
    }
}

// ---------------------------------------------------------------------------
// Lloyd-Max 3-bit scalar quantization
// ---------------------------------------------------------------------------

/// Pre-computed optimal decision boundaries for 3-bit Lloyd-Max quantization.
///
/// These are derived for a standard normal distribution (the rotated KV values
/// approximate a normal distribution after FWHT). 8 reconstruction levels.
const LLOYD_MAX_BOUNDARIES: [f32; 7] = [
    -1.748, -1.050, -0.500, 0.000, 0.500, 1.050, 1.748,
];

/// Pre-computed optimal reconstruction levels for 3-bit Lloyd-Max.
const LLOYD_MAX_LEVELS: [f32; 8] = [
    -2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152,
];

/// Quantize a single f32 value to a 3-bit index (0..7).
pub fn quantize_scalar(value: f32) -> u8 {
    for (i, &boundary) in LLOYD_MAX_BOUNDARIES.iter().enumerate() {
        if value < boundary {
            return i as u8;
        }
    }
    7
}

/// Dequantize a 3-bit index back to f32.
pub fn dequantize_scalar(index: u8) -> f32 {
    LLOYD_MAX_LEVELS[index.min(7) as usize]
}

/// Quantize an f32 vector to packed 3-bit representation.
///
/// Output format: each group of 8 values packs into 3 bytes (24 bits = 8×3).
/// For non-multiple-of-8 lengths, trailing bits are zero-padded.
pub fn quantize_vector(values: &[f32]) -> QuantizedVector {
    let indices: Vec<u8> = values.iter().map(|&v| quantize_scalar(v)).collect();
    let packed = pack_3bit(&indices);
    QuantizedVector {
        packed,
        len: values.len(),
    }
}

/// Dequantize a packed 3-bit vector back to f32.
pub fn dequantize_vector(qvec: &QuantizedVector) -> Vec<f32> {
    let indices = unpack_3bit(&qvec.packed, qvec.len);
    indices.iter().map(|&i| dequantize_scalar(i)).collect()
}

/// Packed 3-bit quantized vector.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Packed 3-bit indices.
    pub packed: Vec<u8>,
    /// Original vector length.
    pub len: usize,
}

impl QuantizedVector {
    /// Compression ratio compared to f32 (32-bit → 3-bit).
    pub fn compression_ratio(&self) -> f32 {
        32.0 / 3.0
    }

    /// Compressed size in bytes.
    pub fn byte_size(&self) -> usize {
        self.packed.len()
    }

    /// Original uncompressed size in bytes (f32).
    pub fn original_byte_size(&self) -> usize {
        self.len * 4
    }
}

// ---------------------------------------------------------------------------
// Full PolarQuant pipeline
// ---------------------------------------------------------------------------

/// Compress a KV cache vector using the full PolarQuant pipeline.
///
/// Steps:
/// 1. Pad to next power of 2
/// 2. Apply randomized Hadamard transform (decorrelation)
/// 3. Apply polar transform to RoPE dimension pairs
/// 4. Quantize to 3-bit Lloyd-Max
pub fn compress(values: &[f32], seed: u64) -> QuantizedVector {
    let original_len = values.len();
    let padded_len = original_len.next_power_of_two();

    let mut buf = vec![0.0f32; padded_len];
    buf[..original_len].copy_from_slice(values);

    // Step 1: Randomized Hadamard transform.
    randomized_hadamard(&mut buf, seed);

    // Step 2: Polar transform on dimension pairs.
    // Only apply if length is even (which it is since padded to power of 2).
    cartesian_to_polar(&mut buf);

    // Step 3: Quantize.
    // Normalize by empirical std dev for better quantization fit.
    let std_dev = empirical_std(&buf);
    if std_dev > 1e-8 {
        let inv_std = 1.0 / std_dev;
        for x in buf.iter_mut() {
            *x *= inv_std;
        }
    }

    let mut qvec = quantize_vector(&buf);
    qvec.len = original_len; // Store original length for decompression.
    qvec
}

/// Decompress a PolarQuant-compressed vector.
pub fn decompress(qvec: &QuantizedVector, seed: u64, std_dev_hint: f32) -> Vec<f32> {
    let padded_len = qvec.len.next_power_of_two();

    // Dequantize.
    let mut padded_qvec = qvec.clone();
    padded_qvec.len = padded_len;
    let mut buf = dequantize_vector(&padded_qvec);
    buf.resize(padded_len, 0.0);

    // Undo normalization.
    if std_dev_hint > 1e-8 {
        for x in buf.iter_mut() {
            *x *= std_dev_hint;
        }
    }

    // Inverse polar transform.
    polar_to_cartesian(&mut buf);

    // Inverse Hadamard.
    inverse_randomized_hadamard(&mut buf, seed);

    buf.truncate(qvec.len);
    buf
}

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------

/// Pack a slice of 3-bit indices (0..7) into bytes.
///
/// Every 8 indices → 3 bytes (24 bits).
fn pack_3bit(indices: &[u8]) -> Vec<u8> {
    let n_bits = indices.len() * 3;
    let n_bytes = n_bits.div_ceil(8);
    let mut packed = vec![0u8; n_bytes];

    for (i, &idx) in indices.iter().enumerate() {
        let bit_pos = i * 3;
        let byte_pos = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        packed[byte_pos] |= (idx & 0x07) << bit_offset;
        // Handle spanning into next byte.
        if bit_offset > 5 && byte_pos + 1 < n_bytes {
            packed[byte_pos + 1] |= (idx & 0x07) >> (8 - bit_offset);
        }
    }

    packed
}

/// Unpack 3-bit indices from packed bytes.
fn unpack_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(count);

    for i in 0..count {
        let bit_pos = i * 3;
        let byte_pos = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let mut val = (packed[byte_pos] >> bit_offset) & 0x07;
        // Handle spanning into next byte.
        if bit_offset > 5 && byte_pos + 1 < packed.len() {
            val |= (packed[byte_pos + 1] << (8 - bit_offset)) & 0x07;
        }

        indices.push(val);
    }

    indices
}

/// Compute empirical standard deviation of an f32 slice.
fn empirical_std(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_identity() {
        // FWHT of [1,0,0,0] should give [1,1,1,1] (unnormalized).
        let mut data = vec![1.0, 0.0, 0.0, 0.0];
        fwht_in_place(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn fwht_self_inverse() {
        // FWHT applied twice (with proper scaling) is identity.
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        fwht_normalized(&mut data);
        fwht_normalized(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "FWHT should be self-inverse: {a} != {b}");
        }
    }

    #[test]
    fn randomized_hadamard_roundtrip() {
        let original = vec![1.0, -0.5, 0.3, 2.1, -1.0, 0.7, 0.0, -0.2];
        let seed = 42;
        let mut data = original.clone();
        randomized_hadamard(&mut data, seed);
        // Data should be different after transform.
        assert_ne!(data, original);
        // Inverse should recover original.
        inverse_randomized_hadamard(&mut data, seed);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "roundtrip failed: {a} != {b}"
            );
        }
    }

    #[test]
    fn polar_roundtrip() {
        let original = vec![3.0, 4.0, -1.0, 2.0, 0.0, 5.0, -3.0, -4.0];
        let mut data = original.clone();
        cartesian_to_polar(&mut data);
        // r values should be positive.
        assert!(data[0] > 0.0);
        assert!(data[2] > 0.0);
        polar_to_cartesian(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "polar roundtrip: {a} != {b}"
            );
        }
    }

    #[test]
    fn quantize_dequantize_range() {
        // Quantize and dequantize values spanning the range.
        let values = vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 0.1];
        let qvec = quantize_vector(&values);
        let recovered = dequantize_vector(&qvec);
        // 3-bit quantization has limited precision, but direction should be preserved.
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(
                orig.signum() == rec.signum() || orig.abs() < 0.3,
                "sign mismatch: {orig} → {rec}"
            );
        }
    }

    #[test]
    fn pack_unpack_3bit_roundtrip() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 5, 7];
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn compression_ratio() {
        let values: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let qvec = quantize_vector(&values);
        // 3-bit per element: 1024 * 3 / 8 = 384 bytes.
        assert_eq!(qvec.byte_size(), 384);
        // Original: 1024 * 4 = 4096 bytes.
        assert_eq!(qvec.original_byte_size(), 4096);
        assert!((qvec.compression_ratio() - 10.67).abs() < 0.1);
    }

    #[test]
    fn full_pipeline_roundtrip_preserves_direction() {
        // Full compress/decompress should roughly preserve the vector direction.
        let original: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.7).sin() * 2.0)
            .collect();
        let seed = 123;
        let std_dev = empirical_std(&original);

        let qvec = compress(&original, seed);
        let recovered = decompress(&qvec, seed, std_dev);

        // Compute cosine similarity between original and recovered.
        let dot: f32 = original.iter().zip(recovered.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = original.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_b: f32 = recovered.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cosine = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };
        // 3-bit quantization with FWHT should preserve direction reasonably well.
        assert!(
            cosine > 0.5,
            "full pipeline cosine similarity too low: {cosine}"
        );
    }
}
