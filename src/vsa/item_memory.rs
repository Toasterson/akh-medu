//! Item Memory: symbol-to-hypervector mapping with ANN search.
//!
//! The item memory is the central registry of all symbol hypervectors.
//! It provides:
//! - Lazy allocation: `get_or_create(symbol)` generates a deterministic vector on first access
//! - Fast ANN search: find the most similar symbols to a query vector
//! - Concurrent access via DashMap for the symbol registry

use std::sync::RwLock;

use anndists::dist::DistHamming;
use dashmap::DashMap;
use hnsw_rs::hnsw::Hnsw;

use crate::error::VsaError;
use crate::symbol::SymbolId;

use super::encode::encode_symbol;
use super::ops::{VsaOps, VsaResult};
use super::{Dimension, Encoding, HyperVec};

/// Search result from item memory.
#[derive(Debug, Clone)]
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
    /// Uses u32 data with Hamming distance for bipolar vectors.
    hnsw: RwLock<Hnsw<'static, u32, DistHamming>>,
    /// Mapping from HNSW internal IDs to SymbolIds.
    id_to_symbol: DashMap<usize, SymbolId>,
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
        let max_layer = max_layer.max(4).min(16);

        let hnsw = Hnsw::new(max_layer, max_elements, 16, 200, DistHamming {});

        Self {
            vectors: DashMap::new(),
            hnsw: RwLock::new(hnsw),
            id_to_symbol: DashMap::new(),
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

    /// Insert a specific hypervector for a symbol.
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

    /// Search for the `k` most similar symbols to a query vector.
    ///
    /// Returns results sorted by descending similarity.
    pub fn search(&self, query: &HyperVec, k: usize) -> VsaResult<Vec<SearchResult>> {
        if query.dim() != self.dim {
            return Err(VsaError::DimensionMismatch {
                expected: self.dim.0,
                actual: query.dim().0,
            });
        }

        let query_u32 = bytes_to_u32_vec(query.data());
        let ef_search = (k * 2).max(32); // ef_search should be >= k

        let hnsw = self.hnsw.read().map_err(|_| VsaError::HnswError {
            message: "HNSW lock poisoned".into(),
        })?;

        let neighbours = hnsw.search(&query_u32, k, ef_search);

        let total_bits = self.dim.0 as f32;
        let mut results: Vec<SearchResult> = neighbours
            .into_iter()
            .filter_map(|n| {
                let symbol_id = self.id_to_symbol.get(&n.d_id)?.value().to_owned();
                // HNSW returns Hamming distance; convert to similarity
                let hamming_dist = n.distance;
                let similarity = 1.0 - (hamming_dist / total_bits);
                Some(SearchResult {
                    symbol_id,
                    similarity,
                })
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        Ok(results)
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
    let mut result = Vec::with_capacity((bytes.len() + 3) / 4);
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

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
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
