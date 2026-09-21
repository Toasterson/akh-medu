//! TurboQuant: MSE-optimal online vector quantization.
//!
//! Zandieh, Daliri, Hadian and Mirrokni, *TurboQuant: Online Vector
//! Quantization with Near-optimal Distortion Rate*, arXiv:2504.19874 (ICLR
//! 2026), Algorithm 1.
//!
//! A random rotation makes every coordinate of a unit vector follow the same
//! Beta density, so coordinates can be quantized independently with a scalar
//! codebook that is optimal for *that* density. This is the difference from the
//! project's earlier PolarQuant path, whose Lloyd-Max levels assume a shape the
//! rotated coordinates do not have; measured on this model's real KV cache that
//! cost 43% relative error on keys at three bits.
//!
//! Data-oblivious: the rotation and codebook depend on dimension, bit-width and
//! seed alone, never on the vectors, so it is valid for an online KV cache with
//! no calibration pass.
//!
//! This implements the MSE-optimal quantizer. The paper's inner-product variant
//! composes it with a 1-bit QJL transform on the residual to remove estimation
//! bias; that is deliberately not implemented here, so callers must not treat
//! reconstructed vectors as unbiased for inner-product estimation.

/// SplitMix64. Inline and fixed so a rotation is reproducible from its seed
/// regardless of which RNG crate version is linked; dequantization depends on
/// regenerating exactly the rotation that quantized the data.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the half-open unit interval.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A quantized vector: one codebook index per coordinate, plus the norm.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    /// Codebook index per coordinate. Each fits in `bits`.
    pub indices: Vec<u8>,
    /// Original L2 norm, kept in full precision so the quantizer only has to
    /// represent direction. The paper's unit-norm assumption is restored by
    /// rescaling on reconstruction.
    pub norm: f32,
}

impl QuantizedVector {
    /// Bits actually needed for the payload, excluding the stored norm.
    pub fn payload_bits(&self, bits: u8) -> usize {
        self.indices.len() * bits as usize
    }
}

/// A rotation and codebook fixed for one (dimension, bit-width, seed).
#[derive(Debug, Clone)]
pub struct TurboQuant {
    dim: usize,
    bits: u8,
    /// Row-major `dim x dim` rotation.
    rotation: Vec<f32>,
    /// Ascending centroids, `2^bits` of them.
    codebook: Vec<f32>,
}

impl TurboQuant {
    /// Build the rotation and codebook. Cost is `O(dim^3)`; do this once.
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(dim > 1, "TurboQuant needs at least two dimensions");
        assert!(
            (1..=8).contains(&bits),
            "TurboQuant supports 1 to 8 bits per coordinate"
        );
        Self {
            dim,
            bits,
            rotation: random_rotation(dim, seed),
            codebook: lloyd_max_codebook(dim, bits),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Centroids, ascending. Exposed for evidence and tests.
    pub fn codebook(&self) -> &[f32] {
        &self.codebook
    }

    /// The rotation, row-major `dim x dim`.
    ///
    /// A device decoder needs it as a tensor: reconstruction is `Pi^T y`, and
    /// for row-major batches that is `rows * Pi`, so this is uploaded once and
    /// reused for every page.
    pub fn rotation(&self) -> &[f32] {
        &self.rotation
    }

    /// Quantize a row and return its codes and norm, without storing them.
    ///
    /// The page store keeps codes on the device; this gives it the bytes to
    /// upload without going through the host-side page representation.
    pub fn quantize_row(&self, values: &[f32]) -> (Vec<u8>, f32) {
        let quantized = self.quantize(values);
        (quantized.indices, quantized.norm)
    }

    /// Rotate, then store the nearest centroid index per coordinate.
    pub fn quantize(&self, values: &[f32]) -> QuantizedVector {
        assert_eq!(values.len(), self.dim, "vector does not match this codebook");
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm == 0.0 {
            return QuantizedVector {
                indices: vec![self.nearest(0.0); self.dim],
                norm: 0.0,
            };
        }
        let rotated = self.rotate(values, false);
        let indices = rotated
            .iter()
            .map(|value| self.nearest(value / norm))
            .collect();
        QuantizedVector { indices, norm }
    }

    /// Look up centroids, rotate back, restore the norm.
    pub fn dequantize(&self, quantized: &QuantizedVector) -> Vec<f32> {
        assert_eq!(
            quantized.indices.len(),
            self.dim,
            "quantized vector does not match this codebook"
        );
        if quantized.norm == 0.0 {
            return vec![0.0; self.dim];
        }
        let centroids: Vec<f32> = quantized
            .indices
            .iter()
            .map(|index| self.codebook[*index as usize] * quantized.norm)
            .collect();
        self.rotate(&centroids, true)
    }

    fn nearest(&self, value: f32) -> u8 {
        // Codebook is ascending, so binary search beats a linear scan at 4+ bits.
        let position = self
            .codebook
            .partition_point(|centroid| *centroid < value)
            .min(self.codebook.len() - 1);
        let lower = position.saturating_sub(1);
        if (value - self.codebook[lower]).abs() <= (value - self.codebook[position]).abs() {
            lower as u8
        } else {
            position as u8
        }
    }

    /// `rotation * v`, or `rotation^T * v` when `transpose`.
    fn rotate(&self, values: &[f32], transpose: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        for row in 0..self.dim {
            let mut sum = 0.0f32;
            for col in 0..self.dim {
                let entry = if transpose {
                    self.rotation[col * self.dim + row]
                } else {
                    self.rotation[row * self.dim + col]
                };
                sum += entry * values[col];
            }
            out[row] = sum;
        }
        out
    }
}

/// Orthonormal matrix from Gram-Schmidt on seeded Gaussian columns.
///
/// The paper generates it by QR decomposition of a Gaussian matrix; modified
/// Gram-Schmidt is the same factor and keeps the dependency surface empty.
fn random_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Prng::new(seed);
    let mut rows: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| gaussian(&mut rng)).collect())
        .collect();
    for i in 0..dim {
        for j in 0..i {
            let projection: f64 = (0..dim).map(|k| rows[i][k] * rows[j][k]).sum();
            for k in 0..dim {
                rows[i][k] -= projection * rows[j][k];
            }
        }
        let norm: f64 = rows[i].iter().map(|v| v * v).sum::<f64>().sqrt();
        // A degenerate draw is astronomically unlikely; fall back to a basis
        // vector rather than emitting NaNs into every later quantization.
        if norm < 1e-9 {
            rows[i] = (0..dim).map(|k| if k == i { 1.0 } else { 0.0 }).collect();
        } else {
            for value in rows[i].iter_mut() {
                *value /= norm;
            }
        }
    }
    rows.into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect()
}

fn gaussian(rng: &mut Prng) -> f64 {
    // Box-Muller; the uniform is nudged off zero so ln() stays finite.
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Lloyd-Max centroids for the coordinate density of a random unit vector.
///
/// The density is `f(x) = C * (1 - x^2)^((d-3)/2)` on `[-1, 1]` (paper Lemma 1),
/// which tends to `N(0, 1/d)`. Solved numerically on a fixed grid; the constant
/// `C` cancels because every step is a ratio of integrals against the same
/// weight.
fn lloyd_max_codebook(dim: usize, bits: u8) -> Vec<f32> {
    const GRID: usize = 200_001;
    let levels = 1usize << bits;
    let exponent = (dim as f64 - 3.0) / 2.0;
    let mut grid = Vec::with_capacity(GRID);
    for i in 0..GRID {
        let x = -1.0 + 2.0 * i as f64 / (GRID - 1) as f64;
        let inside = (1.0 - x * x).max(0.0);
        // exp/ln keeps the tail representable for large d, where (1-x^2)^~127
        // underflows for all but a narrow band around zero.
        let weight = if inside <= 0.0 {
            0.0
        } else {
            (exponent * inside.ln()).exp()
        };
        grid.push((x, weight));
    }
    let total: f64 = grid.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return vec![0.0; levels];
    }

    // Seed centroids at equal-probability quantiles so Lloyd-Max starts inside
    // the mass rather than out on the empty tails.
    let mut centroids: Vec<f64> = Vec::with_capacity(levels);
    let mut cumulative = 0.0;
    let mut index = 0;
    for level in 0..levels {
        let target = total * (level as f64 + 0.5) / levels as f64;
        while index + 1 < grid.len() && cumulative + grid[index].1 < target {
            cumulative += grid[index].1;
            index += 1;
        }
        centroids.push(grid[index].0);
    }

    for _ in 0..200 {
        let mut sums = vec![0.0f64; levels];
        let mut weights = vec![0.0f64; levels];
        let mut cursor = 0usize;
        for (x, w) in &grid {
            if *w == 0.0 {
                continue;
            }
            while cursor + 1 < levels
                && (x - centroids[cursor]).abs() > (x - centroids[cursor + 1]).abs()
            {
                cursor += 1;
            }
            sums[cursor] += x * w;
            weights[cursor] += w;
        }
        let mut moved = 0.0f64;
        for level in 0..levels {
            if weights[level] > 0.0 {
                let updated = sums[level] / weights[level];
                moved = moved.max((updated - centroids[level]).abs());
                centroids[level] = updated;
            }
        }
        if moved < 1e-12 {
            break;
        }
    }
    centroids.sort_by(|a, b| a.partial_cmp(b).expect("finite centroids"));
    centroids.into_iter().map(|c| c as f32).collect()
}

/// A packed array of small unsigned integers.
///
/// Indices are written at exactly `bits` each with no alignment padding, so a
/// 3-bit code costs three bits rather than a byte. Kept minimal and local: it
/// only has to serve [`KvPageStore`].
#[derive(Debug, Default, Clone)]
struct BitVec {
    data: Vec<u8>,
    len: usize,
}

impl BitVec {
    fn new() -> Self {
        Self::default()
    }

    fn bytes(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, value: u8, bits: u8) {
        for bit in 0..bits {
            let set = (value >> bit) & 1 == 1;
            let byte = self.len / 8;
            if byte >= self.data.len() {
                self.data.push(0);
            }
            if set {
                self.data[byte] |= 1 << (self.len % 8);
            }
            self.len += 1;
        }
    }

    fn get(&self, index: usize, bits: u8) -> u8 {
        let mut value = 0u8;
        for bit in 0..bits {
            let at = index * bits as usize + bit as usize;
            if self.data[at / 8] >> (at % 8) & 1 == 1 {
                value |= 1 << bit;
            }
        }
        value
    }

    /// Drop the first `count` values, rewriting the remainder.
    fn drop_front(&mut self, count: usize, bits: u8) {
        let total = self.len / bits as usize;
        let mut rebuilt = BitVec::new();
        for index in count..total {
            rebuilt.push(self.get(index, bits), bits);
        }
        *self = rebuilt;
    }
}

/// Device-resident compressed KV pages.
///
/// Older cache positions are held as one codebook index per coordinate plus a
/// full-precision norm, and decoded back to f32 when attention needs them. The
/// codebook and rotation are shared across every page, so the per-position cost
/// is `bits` per coordinate plus one norm rather than 32 bits per coordinate.
///
/// This is the storage half of KV compression. The quality half is settled
/// separately by the round-trip gate; a store that decodes to the same values
/// the in-place path produced inherits that result.
#[derive(Debug, Clone)]
pub struct KvPageStore {
    /// Shared: the rotation and codebook are identical for every store at a
    /// given dimension and bit-width, and a dense rotation is too large to
    /// duplicate per attention head per layer.
    quantizer: std::sync::Arc<TurboQuant>,
    /// Codebook indices packed at `bits` each, with no padding between rows.
    /// Packing is what makes the saving real: one byte per index would cap a
    /// 4-bit code at 4x against f32 rather than the ~7.8x it should reach.
    indices: BitVec,
    /// Norm per stored position.
    norms: Vec<f32>,
    positions: usize,
}

impl KvPageStore {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Self {
        Self::with_quantizer(std::sync::Arc::new(TurboQuant::new(dim, bits, seed)))
    }

    /// Share one prepared quantizer across many stores.
    pub fn with_quantizer(quantizer: std::sync::Arc<TurboQuant>) -> Self {
        Self {
            quantizer,
            indices: BitVec::new(),
            norms: Vec::new(),
            positions: 0,
        }
    }

    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    pub fn is_empty(&self) -> bool {
        self.positions == 0
    }

    /// Bytes held, against what the same positions would cost as f32.
    pub fn stored_bytes(&self) -> usize {
        self.indices.bytes() + self.norms.len() * std::mem::size_of::<f32>()
    }

    pub fn uncompressed_bytes(&self) -> usize {
        self.positions * self.quantizer.dim() * std::mem::size_of::<f32>()
    }

    /// Append `positions` rows, each `dim` values, compressing as they arrive.
    pub fn push_rows(&mut self, rows: &[f32]) -> Result<(), String> {
        let dim = self.quantizer.dim();
        if rows.len() % dim != 0 {
            return Err(format!("{} values is not a multiple of {dim}", rows.len()));
        }
        let bits = self.quantizer.bits();
        for row in rows.chunks_exact(dim) {
            let quantized = self.quantizer.quantize(row);
            for index in &quantized.indices {
                self.indices.push(*index, bits);
            }
            self.norms.push(quantized.norm);
            self.positions += 1;
        }
        Ok(())
    }

    /// Decode every stored position back to f32, in insertion order.
    pub fn decode(&self) -> Vec<f32> {
        let dim = self.quantizer.dim();
        let bits = self.quantizer.bits();
        let mut out = Vec::with_capacity(self.positions * dim);
        for position in 0..self.positions {
            let start = position * dim;
            let indices = (0..dim)
                .map(|i| self.indices.get(start + i, bits))
                .collect();
            out.extend_from_slice(&self.quantizer.dequantize(&QuantizedVector {
                indices,
                norm: self.norms[position],
            }));
        }
        out
    }

    /// Drop the oldest `count` positions, as a paging policy would on eviction.
    pub fn drop_oldest(&mut self, count: usize) {
        let count = count.min(self.positions);
        let dim = self.quantizer.dim();
        self.indices.drop_front(count * dim, self.quantizer.bits());
        self.norms.drain(..count);
        self.positions -= count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Prng::new(seed);
        let raw: Vec<f32> = (0..dim).map(|_| gaussian(&mut rng) as f32).collect();
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        raw.into_iter().map(|v| v / norm).collect()
    }

    fn mean_distortion(dim: usize, bits: u8, samples: usize) -> f64 {
        let quantizer = TurboQuant::new(dim, bits, 7);
        let mut total = 0.0;
        for sample in 0..samples {
            let x = unit_vector(dim, 1000 + sample as u64);
            let restored = quantizer.dequantize(&quantizer.quantize(&x));
            total += x
                .iter()
                .zip(&restored)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>();
        }
        total / samples as f64
    }

    #[test]
    fn rotation_is_orthonormal() {
        let dim = 64;
        let rotation = random_rotation(dim, 11);
        for i in 0..dim {
            for j in 0..dim {
                let dot: f32 = (0..dim)
                    .map(|k| rotation[i * dim + k] * rotation[j * dim + k])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-3, "row {i}.{j} = {dot}");
            }
        }
    }

    #[test]
    fn codebook_is_symmetric_and_ordered() {
        let codebook = lloyd_max_codebook(256, 3);
        assert_eq!(codebook.len(), 8);
        for pair in codebook.windows(2) {
            assert!(pair[0] < pair[1], "codebook not ascending: {codebook:?}");
        }
        for (low, high) in codebook.iter().zip(codebook.iter().rev()) {
            assert!((low + high).abs() < 1e-3, "not symmetric: {codebook:?}");
        }
    }

    #[test]
    fn one_bit_centroids_match_the_published_closed_form() {
        // Paper section 3.1: for moderately high d the 1-bit centroids are
        // +/- sqrt(2/pi)/sqrt(d).
        let dim = 256;
        let codebook = lloyd_max_codebook(dim, 1);
        let expected = (2.0f64 / std::f64::consts::PI).sqrt() / (dim as f64).sqrt();
        assert!(
            (codebook[1] as f64 - expected).abs() < expected * 0.02,
            "got {:?}, expected +/-{expected}",
            codebook
        );
    }

    #[test]
    fn two_bit_centroids_match_the_published_closed_form() {
        // Paper section 3.1: +/-0.453/sqrt(d) and +/-1.51/sqrt(d).
        let dim = 256;
        let codebook = lloyd_max_codebook(dim, 2);
        let scale = (dim as f64).sqrt();
        let inner = codebook[2] as f64 * scale;
        let outer = codebook[3] as f64 * scale;
        assert!((inner - 0.453).abs() < 0.02, "inner {inner}");
        assert!((outer - 1.51).abs() < 0.05, "outer {outer}");
    }

    #[test]
    fn distortion_matches_the_published_rates() {
        // Paper Theorem 1: D_mse ~ 0.36, 0.117, 0.03, 0.009 for b = 1..4 on
        // unit vectors. Tolerance is loose because this averages few samples.
        for (bits, expected) in [(1u8, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)] {
            let measured = mean_distortion(128, bits, 12);
            assert!(
                measured < expected * 1.35 && measured > expected * 0.65,
                "{bits} bits: measured {measured}, paper {expected}"
            );
        }
    }

    #[test]
    fn distortion_falls_by_about_four_times_per_added_bit() {
        // The bound is proportional to 4^-b; consecutive ratios should sit near
        // 4 rather than the 2 a naive scalar quantizer would give.
        let three = mean_distortion(128, 3, 8);
        let four = mean_distortion(128, 4, 8);
        let ratio = three / four;
        assert!((2.5..5.5).contains(&ratio), "ratio {ratio}");
    }

    #[test]
    fn norm_is_preserved_and_a_zero_vector_round_trips() {
        let quantizer = TurboQuant::new(64, 4, 3);
        let mut x = unit_vector(64, 5);
        for value in x.iter_mut() {
            *value *= 17.5;
        }
        let quantized = quantizer.quantize(&x);
        assert!((quantized.norm - 17.5).abs() < 1e-2, "{}", quantized.norm);
        let restored = quantizer.dequantize(&quantized);
        let restored_norm = restored.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((restored_norm - 17.5).abs() < 17.5 * 0.2, "{restored_norm}");

        let zero = vec![0.0f32; 64];
        assert_eq!(quantizer.dequantize(&quantizer.quantize(&zero)), zero);
    }

    #[test]
    fn page_store_round_trips_rows_in_order_and_reports_its_saving() {
        let (dim, rows) = (128usize, 6usize);
        let mut store = KvPageStore::new(dim, 4, 21);
        let original: Vec<f32> = (0..rows)
            .flat_map(|r| unit_vector(dim, 400 + r as u64))
            .collect();
        store.push_rows(&original).unwrap();
        assert_eq!(store.positions(), rows);

        let decoded = store.decode();
        assert_eq!(decoded.len(), original.len());
        for r in 0..rows {
            let a = &original[r * dim..(r + 1) * dim];
            let b = &decoded[r * dim..(r + 1) * dim];
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            // Rows must not be transposed or reordered: each decodes to itself.
            assert!(dot / (na * nb) > 0.98, "row {r} cosine {}", dot / (na * nb));
        }

        // Indices are packed at 4 bits, so two share a byte, plus one norm.
        assert_eq!(store.stored_bytes(), rows * dim / 2 + rows * 4);
        assert_eq!(store.uncompressed_bytes(), rows * dim * 4);
        let ratio = store.uncompressed_bytes() as f64 / store.stored_bytes() as f64;
        assert!(ratio > 7.0, "compression ratio {ratio}");
    }

    #[test]
    fn page_store_evicts_from_the_front() {
        let dim = 64usize;
        let mut store = KvPageStore::new(dim, 3, 5);
        let rows: Vec<f32> = (0..4).flat_map(|r| unit_vector(dim, 700 + r as u64)).collect();
        store.push_rows(&rows).unwrap();
        let before = store.decode();

        store.drop_oldest(2);
        assert_eq!(store.positions(), 2);
        let after = store.decode();
        // What remains is exactly the tail of what was there.
        assert_eq!(after, before[2 * dim..].to_vec());

        store.drop_oldest(99);
        assert!(store.is_empty());
        assert!(store.decode().is_empty());
    }

    #[test]
    fn page_store_rejects_a_ragged_push() {
        let mut store = KvPageStore::new(64, 3, 1);
        assert!(store.push_rows(&vec![0.0; 63]).is_err());
        assert_eq!(store.positions(), 0);
    }

    #[test]
    fn payload_is_exactly_the_advertised_bit_width() {
        let quantizer = TurboQuant::new(256, 3, 1);
        let quantized = quantizer.quantize(&unit_vector(256, 9));
        assert_eq!(quantized.payload_bits(3), 256 * 3);
        assert!(quantized.indices.iter().all(|i| *i < 8));
    }
}
