//! Item Memory: symbol-to-hypervector mapping with ANN search.
//!
//! The item memory is the central registry of all symbol hypervectors.
//! It provides:
//! - Lazy allocation: `get_or_create(symbol)` generates a deterministic vector on first access
//! - Fast ANN search: find the most similar symbols to a query vector
//! - Concurrent access via DashMap for the symbol registry

use std::collections::HashMap;
use std::sync::RwLock;

use anndists::dist::Distance;
use dashmap::DashMap;
use hnsw_rs::hnsw::Hnsw;

use crate::error::VsaError;
use crate::symbol::SymbolId;

use super::encode::encode_symbol;
use super::ops::{VsaOps, VsaResult};
use super::{Dimension, Encoding, HyperVec};

/// Bit-level Hamming distance for u32-packed binary vectors.
///
/// Counts the number of BITS that differ via XOR + popcount, producing a
/// distance in [0, dim_bits]. This is correct for binary hypervectors packed
/// as u32 slices from `bytes_to_u32_vec`.
///
/// `anndists::DistHamming<u32>` counts u32 ELEMENTS that differ (not bits),
/// which gives a max distance of ~313 for 10 000-bit vectors and causes
/// similarity values to cluster near 1.0. This custom struct fixes that.
#[derive(Clone, Copy)]
struct BitHamming;

impl Distance<u32> for BitHamming {
    fn eval(&self, va: &[u32], vb: &[u32]) -> f32 {
        va.iter()
            .zip(vb.iter())
            .map(|(a, b)| (a ^ b).count_ones() as f32)
            .sum()
    }
}

/// Search result from item memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// The matching symbol ID.
    pub symbol_id: SymbolId,
    /// Similarity score (0.0 = unrelated, 1.0 = identical).
    pub similarity: f32,
}

/// Item Memory: the registry of all symbol hypervectors.
///
/// Combines a concurrent hashmap for exact lookups with an HNSW index
/// for approximate nearest-neighbor search.
pub struct ItemMemory {
    /// Exact symbol → hypervector mapping.
    vectors: DashMap<SymbolId, HyperVec>,
    /// HNSW ANN index for similarity search.
    /// Uses u32 data with bit-level Hamming distance for bipolar vectors.
    hnsw: RwLock<Hnsw<'static, u32, BitHamming>>,
    /// Mapping from HNSW internal IDs to SymbolIds.
    id_to_symbol: DashMap<usize, SymbolId>,
    /// Canonical (latest) HNSW internal ID for each symbol.
    /// Used to filter stale index entries after re-grounding.
    symbol_to_hnsw_id: DashMap<SymbolId, usize>,
    /// Next HNSW internal ID to assign.
    next_hnsw_id: std::sync::atomic::AtomicUsize,
    /// Dimension and encoding config.
    dim: Dimension,
    encoding: Encoding,
}

// Safety: Hnsw uses internal synchronization via atomics/locks.
// The RwLock wrapper provides the outer synchronization needed.
unsafe impl Send for ItemMemory {}
unsafe impl Sync for ItemMemory {}

impl ItemMemory {
    /// Create a new empty item memory.
    ///
    /// `max_elements` is a capacity hint for the HNSW index.
    pub fn new(dim: Dimension, encoding: Encoding, max_elements: usize) -> Self {
        // HNSW parameters:
        // max_nb_connection: 16 (standard for moderate dimensions)
        // ef_construction: 200 (higher = better recall during build)
        // max_layer: computed from expected elements
        let max_layer = (max_elements as f64).log2().ceil() as usize;
        let max_layer = max_layer.clamp(4, 16);

        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, BitHamming {});

        Self {
            vectors: DashMap::new(),
            hnsw: RwLock::new(hnsw),
            id_to_symbol: DashMap::new(),
            symbol_to_hnsw_id: DashMap::new(),
            next_hnsw_id: std::sync::atomic::AtomicUsize::new(0),
            dim,
            encoding,
        }
    }

    /// Get the hypervector for a symbol, creating it if it doesn't exist.
    ///
    /// The vector is deterministically derived from the symbol ID,
    /// so this is idempotent.
    pub fn get_or_create(&self, ops: &VsaOps, symbol: SymbolId) -> HyperVec {
        if let Some(entry) = self.vectors.get(&symbol) {
            return entry.value().clone();
        }

        let vec = encode_symbol(ops, symbol);
        self.insert(symbol, vec.clone());
        vec
    }

    /// Insert or update the hypervector for a symbol.
    ///
    /// If the symbol already has an HNSW entry, the old entry becomes stale.
    /// The `symbol_to_hnsw_id` map is updated so `search` always sees the
    /// latest vector and filters out stale entries.
    pub fn insert(&self, symbol: SymbolId, vec: HyperVec) {
        // Convert to u32 slices for HNSW (bit-packed representation)
        let data_u32 = bytes_to_u32_vec(vec.data());
        let hnsw_id = self
            .next_hnsw_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Insert into HNSW — insert takes &self, not &mut self
        if let Ok(hnsw) = self.hnsw.read() {
            hnsw.insert((&data_u32, hnsw_id));
        }

        self.id_to_symbol.insert(hnsw_id, symbol);
        // Track the canonical (latest) HNSW ID for this symbol so that
        // search can skip the old stale entry when the symbol is re-grounded.
        self.symbol_to_hnsw_id.insert(symbol, hnsw_id);
        self.vectors.insert(symbol, vec);
    }

    /// Get the hypervector for a symbol, if it exists.
    pub fn get(&self, symbol: SymbolId) -> Option<HyperVec> {
        self.vectors.get(&symbol).map(|v| v.value().clone())
    }

    /// Check if a symbol has a vector in item memory.
    pub fn contains(&self, symbol: SymbolId) -> bool {
        self.vectors.contains_key(&symbol)
    }

    /// Number of symbols stored.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Return all symbol IDs currently stored in item memory.
    pub fn all_symbols(&self) -> Vec<SymbolId> {
        self.vectors.iter().map(|entry| *entry.key()).collect()
    }

    /// Search for the `k` most similar symbols to a query vector.
    ///
    /// Returns up to `k` results sorted by descending similarity.
    /// Stale HNSW entries (from symbols that have been re-grounded) are
    /// filtered out, and results are deduplicated by symbol_id so each
    /// symbol appears at most once.
    pub fn search(&self, query: &HyperVec, k: usize) -> VsaResult<Vec<SearchResult>> {
        if query.dim() != self.dim {
            return Err(VsaError::DimensionMismatch {
                expected: self.dim.0,
                actual: query.dim().0,
            });
        }

        let query_u32 = bytes_to_u32_vec(query.data());
        // Request more candidates than k to compensate for stale-entry filtering
        // and per-symbol deduplication.
        let fetch_k = (k * 3).max(k + 16);
        let ef_search = (fetch_k * 2).max(64);

        let hnsw = self.hnsw.read().map_err(|_| VsaError::HnswError {
            message: "HNSW lock poisoned".into(),
        })?;

        let neighbours = hnsw.search(&query_u32, fetch_k, ef_search);

        let total_bits = self.dim.0 as f32;
        // Collect best similarity per symbol, skipping stale HNSW entries.
        let mut best: HashMap<SymbolId, f32> = HashMap::new();
        for n in &neighbours {
            let symbol_id = match self.id_to_symbol.get(&n.d_id) {
                Some(s) => s.value().to_owned(),
                None => continue,
            };
            // Skip stale entries: only the canonical HNSW ID for this symbol
            // reflects the current (possibly re-grounded) vector.
            let is_canonical = self
                .symbol_to_hnsw_id
                .get(&symbol_id)
                .map(|id| *id.value() == n.d_id)
                .unwrap_or(false);
            if !is_canonical {
                continue;
            }
            let similarity = 1.0 - (n.distance / total_bits);
            let entry = best.entry(symbol_id).or_insert(f32::NEG_INFINITY);
            if similarity > *entry {
                *entry = similarity;
            }
        }

        let mut results: Vec<SearchResult> = best
            .into_iter()
            .map(|(symbol_id, similarity)| SearchResult {
                symbol_id,
                similarity,
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        Ok(results)
    }

    /// Rebuild the HNSW index from the current contents of `vectors`.
    ///
    /// After a bulk re-encoding operation (e.g. `ground_all_symbols`), the
    /// HNSW index accumulates stale entries from previous vector versions.
    /// Rebuilding replaces the index with only the current vectors, restoring
    /// search recall and removing stale entries entirely.
    ///
    /// This holds the HNSW write lock for the duration of the swap to prevent
    /// concurrent searches from observing inconsistent state.
    pub fn rebuild_hnsw(&self) {
        let n = self.vectors.len();
        if n == 0 {
            return;
        }
        let max_layer = ((n as f64).log2().ceil() as usize).clamp(4, 16);
        let new_hnsw = Hnsw::new(max_layer, n + 64, 16, 200, BitHamming {});

        // Build (hnsw_id, symbol, data) triples outside the write lock.
        let mut pairs: Vec<(usize, SymbolId)> = Vec::with_capacity(n);
        let mut next_id: usize = 0;
        for entry in &self.vectors {
            let symbol = *entry.key();
            let data_u32 = bytes_to_u32_vec(entry.value().data());
            new_hnsw.insert((&data_u32, next_id));
            pairs.push((next_id, symbol));
            next_id += 1;
        }

        // Swap under the write lock so concurrent reads don't see torn state.
        if let Ok(mut hnsw_guard) = self.hnsw.write() {
            self.id_to_symbol.clear();
            self.symbol_to_hnsw_id.clear();
            for &(hnsw_id, symbol) in &pairs {
                self.id_to_symbol.insert(hnsw_id, symbol);
                self.symbol_to_hnsw_id.insert(symbol, hnsw_id);
            }
            self.next_hnsw_id
                .store(next_id, std::sync::atomic::Ordering::SeqCst);
            *hnsw_guard = new_hnsw;
        }
    }

    /// Batch insert using rayon for parallel encoding.
    pub fn insert_batch(&self, ops: &VsaOps, symbols: &[SymbolId]) {
        use rayon::prelude::*;

        let vecs: Vec<(SymbolId, HyperVec)> = symbols
            .par_iter()
            .map(|&sym| {
                let vec = encode_symbol(ops, sym);
                (sym, vec)
            })
            .collect();

        for (sym, vec) in vecs {
            self.insert(sym, vec);
        }
    }
}

impl std::fmt::Debug for ItemMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ItemMemory")
            .field("dim", &self.dim)
            .field("encoding", &self.encoding)
            .field("len", &self.vectors.len())
            .finish()
    }
}

/// Convert a byte slice to a `Vec<u32>` for HNSW Hamming distance.
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    let mut result = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        result.push(u32::from_le_bytes(word));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use rand::SeedableRng;

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    #[test]
    fn bit_hamming_counts_bits_not_elements() {
        // 0b00000011 XOR 0b00000000 = 0b00000011 → 2 bits set
        let va = vec![0b00000011u32];
        let vb = vec![0b00000000u32];
        let d = BitHamming {}.eval(&va, &vb);
        assert_eq!(d, 2.0, "BitHamming should count differing BITS, got {d}");

        // Single-bit difference
        let vc = vec![0b00000001u32];
        let vd = vec![0b00000000u32];
        assert_eq!(BitHamming {}.eval(&vc, &vd), 1.0);

        // Identical: zero distance
        let ve = vec![0xDEADBEEFu32, 0xCAFEBABEu32];
        assert_eq!(BitHamming {}.eval(&ve, &ve), 0.0);
    }

    #[test]
    fn get_or_create_is_idempotent() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 1000);
        let sym = SymbolId::new(1).unwrap();

        let v1 = mem.get_or_create(&ops, sym);
        let v2 = mem.get_or_create(&ops, sym);
        assert_eq!(v1, v2);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn search_finds_self() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        // Insert a few symbols
        for i in 1..=10u64 {
            let sym = SymbolId::new(i).unwrap();
            mem.get_or_create(&ops, sym);
        }

        // Search for symbol 5 — should find itself
        let query_sym = SymbolId::new(5).unwrap();
        let query_vec = mem.get(query_sym).unwrap();
        let results = mem.search(&query_vec, 3).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].symbol_id, query_sym);
        assert!((results[0].similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn search_similarity_range_for_random_symbols() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 200);

        for i in 1..=30u64 {
            let sym = SymbolId::new(i).unwrap();
            mem.get_or_create(&ops, sym);
        }

        let sym1 = SymbolId::new(1).unwrap();
        let vec1 = mem.get(sym1).unwrap();
        let results = mem.search(&vec1, 10).unwrap();

        // sym1 itself must be at the top with similarity 1.0
        assert!(!results.is_empty());
        assert_eq!(results[0].symbol_id, sym1);
        assert!((results[0].similarity - 1.0).abs() < 0.001);

        // Other random symbols should NOT all have similarity 1.0 —
        // that would indicate the normalization is wrong.
        let other_max = results[1..]
            .iter()
            .filter(|r| r.symbol_id != sym1)
            .map(|r| r.similarity)
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            other_max < 0.85,
            "Random unrelated symbols should have similarity < 0.85 (uniform-1.0 bug); \
             got {other_max:.4}. This indicates DistHamming is not counting bits correctly."
        );
    }

    #[test]
    fn search_deduplicates_after_regrounding() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 100);

        // Insert several symbols so HNSW has something to index
        for i in 1..=5u64 {
            mem.get_or_create(&ops, SymbolId::new(i).unwrap());
        }

        let sym = SymbolId::new(1).unwrap();
        // Simulate grounding: re-insert with a different vector
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let new_vec = ops.random(&mut rng);
        mem.insert(sym, new_vec.clone());

        // Search should return sym at most once even though HNSW has
        // two entries for it (old random + new grounded).
        let results = mem.search(&new_vec, 5).unwrap();
        let count = results.iter().filter(|r| r.symbol_id == sym).count();
        assert_eq!(count, 1, "Symbol should appear exactly once after re-grounding");

        // The one result for sym should be the LATEST vector (sim = 1.0)
        let sym_sim = results
            .iter()
            .find(|r| r.symbol_id == sym)
            .map(|r| r.similarity)
            .unwrap();
        assert!(
            (sym_sim - 1.0).abs() < 0.001,
            "Re-grounded symbol should have similarity 1.0, got {sym_sim}"
        );
    }

    #[test]
    fn rebuild_hnsw_removes_stale_entries() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 200);

        // Insert initial symbols
        for i in 1..=10u64 {
            mem.get_or_create(&ops, SymbolId::new(i).unwrap());
        }

        // Re-ground all symbols (simulates ground_all_symbols)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for i in 1..=10u64 {
            let sym = SymbolId::new(i).unwrap();
            mem.insert(sym, ops.random(&mut rng));
        }

        // Rebuild removes stale entries
        mem.rebuild_hnsw();

        // Every symbol should appear exactly once in a broad search
        let sym = SymbolId::new(1).unwrap();
        let vec = mem.get(sym).unwrap();
        let results = mem.search(&vec, 10).unwrap();

        let mut seen = std::collections::HashSet::new();
        for r in &results {
            assert!(
                seen.insert(r.symbol_id),
                "Symbol {:?} appeared more than once after rebuild",
                r.symbol_id
            );
        }
    }

    #[test]
    fn batch_insert() {
        let ops = test_ops();
        let mem = ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 1000);
        let syms: Vec<SymbolId> = (1..=100u64).map(|i| SymbolId::new(i).unwrap()).collect();

        mem.insert_batch(&ops, &syms);
        assert_eq!(mem.len(), 100);
    }

    #[test]
    fn bytes_to_u32_roundtrip() {
        let bytes = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let u32s = bytes_to_u32_vec(&bytes);
        assert_eq!(u32s.len(), 2);
        assert_eq!(u32s[0], u32::from_le_bytes([0x01, 0x02, 0x03, 0x04]));
        assert_eq!(u32s[1], u32::from_le_bytes([0x05, 0x00, 0x00, 0x00]));
    }
}
