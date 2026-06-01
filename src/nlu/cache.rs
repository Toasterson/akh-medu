//! Semantic result cache for the n akh router.
//!
//! Caches NLU parse results so that repeated or near-identical queries skip
//! backend inference entirely.  "Near-identical" is defined as VSA hypervector
//! similarity ≥ 0.95.  Entries expire after a configurable TTL.
//!
//! The cache is an LRU store backed by a plain `Vec`.  With a typical capacity
//! of 256 entries the linear scan is negligible compared to backend latency.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::vsa::HyperVec;
use crate::vsa::ops::VsaOps;

use super::NluParseResult;

/// Minimum VSA similarity required for a cache hit (near-identical match).
pub const SIMILARITY_THRESHOLD: f32 = 0.95;

struct CacheEntry {
    key_hash: u64,
    key_vec: HyperVec,
    result: NluParseResult,
    inserted_at: Instant,
    last_accessed: Instant,
}

/// LRU semantic cache for NLU parse results.
///
/// Lookup order:
/// 1. Exact hash match (fast path, O(n) scan but cache is tiny).
/// 2. VSA similarity search (slow path, O(n) but still small n).
///
/// Eviction: expired entries are pruned on `insert`; when the cache is
/// still full after pruning, the entry with the oldest `last_accessed`
/// timestamp is removed.
pub struct SemanticCache {
    entries: Vec<CacheEntry>,
    max_size: usize,
    ttl: Duration,
}

impl SemanticCache {
    /// Create a cache with the given capacity and TTL (seconds).
    pub fn new(max_size: usize, ttl_secs: u64) -> Self {
        Self {
            entries: Vec::with_capacity(max_size.min(256)),
            max_size,
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Look up a result for `input`.
    ///
    /// First checks for an exact hash match, then falls back to VSA similarity
    /// search across unexpired entries.  Returns `Some(result)` on hit,
    /// updating the entry's `last_accessed` timestamp.
    pub fn find(
        &mut self,
        input: &str,
        query_vec: &HyperVec,
        ops: &VsaOps,
    ) -> Option<NluParseResult> {
        let hash = hash_input(input);
        let now = Instant::now();

        // Fast path: exact hash match
        for entry in &mut self.entries {
            if entry.key_hash == hash
                && now.duration_since(entry.inserted_at) < self.ttl
            {
                entry.last_accessed = now;
                tracing::trace!(cache = "hit:exact", "semantic cache exact match");
                return Some(entry.result.clone());
            }
        }

        // Slow path: VSA similarity search
        let mut best_sim = 0.0f32;
        let mut best_idx = usize::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            if now.duration_since(entry.inserted_at) >= self.ttl {
                continue;
            }
            if let Ok(sim) = ops.similarity(query_vec, &entry.key_vec) {
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
        }

        if best_sim >= SIMILARITY_THRESHOLD && best_idx < self.entries.len() {
            tracing::trace!(sim = best_sim, "semantic cache similarity hit");
            let result = self.entries[best_idx].result.clone();
            self.entries[best_idx].last_accessed = now;
            return Some(result);
        }

        None
    }

    /// Insert a result for `input` with its pre-computed hypervector key.
    pub fn insert(&mut self, input: &str, key_vec: HyperVec, result: NluParseResult) {
        let hash = hash_input(input);
        let now = Instant::now();

        // Prune expired entries
        self.entries
            .retain(|e| now.duration_since(e.inserted_at) < self.ttl);

        // LRU eviction if still over capacity
        if self.entries.len() >= self.max_size {
            if let Some(oldest_idx) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(i, _)| i)
            {
                self.entries.swap_remove(oldest_idx);
            }
        }

        self.entries.push(CacheEntry {
            key_hash: hash,
            key_vec,
            result,
            inserted_at: now,
            last_accessed: now,
        });
    }

    /// Invalidate all entries.
    ///
    /// Call this after knowledge-graph mutations to prevent stale results.
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
        tracing::debug!("semantic cache invalidated");
    }

    /// Count of live (unexpired) entries.
    pub fn live_count(&self) -> usize {
        let now = Instant::now();
        self.entries
            .iter()
            .filter(|e| now.duration_since(e.inserted_at) < self.ttl)
            .count()
    }
}

fn hash_input(input: &str) -> u64 {
    let mut h = DefaultHasher::new();
    input.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::abs::AbsTree;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};
    use crate::vsa::encode::encode_label;
    use crate::vsa::ops::VsaOps;

    fn make_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    fn make_result() -> NluParseResult {
        NluParseResult {
            tree: AbsTree::entity("test"),
            source_tier: 1,
            confidence: 0.85,
            exemplar_similarity: None,
            neural_concepts: vec![],
        }
    }

    #[test]
    fn exact_hit_after_insert() {
        let ops = make_ops();
        let mut cache = SemanticCache::new(16, 300);
        let input = "dogs are mammals";
        let vec = encode_label(&ops, input).unwrap();
        let result = make_result();

        cache.insert(input, vec.clone(), result);

        // Same input should produce an exact cache hit
        let found = cache.find(input, &vec, &ops);
        assert!(found.is_some(), "expected cache hit");
    }

    #[test]
    fn miss_on_empty_cache() {
        let ops = make_ops();
        let mut cache = SemanticCache::new(16, 300);
        let vec = encode_label(&ops, "hello world").unwrap();
        assert!(cache.find("hello world", &vec, &ops).is_none());
    }

    #[test]
    fn lru_eviction() {
        let ops = make_ops();
        let mut cache = SemanticCache::new(2, 300);
        let result = make_result();

        let v1 = encode_label(&ops, "input one").unwrap();
        let v2 = encode_label(&ops, "input two").unwrap();
        let v3 = encode_label(&ops, "input three").unwrap();

        cache.insert("input one", v1.clone(), result.clone());
        cache.insert("input two", v2.clone(), result.clone());

        // Access "input two" to make it more recent
        let _ = cache.find("input two", &v2, &ops);

        // Insert third entry — should evict "input one" (least recently accessed)
        cache.insert("input three", v3.clone(), result.clone());

        assert_eq!(cache.live_count(), 2);
        // "input one" should be evicted
        assert!(cache.find("input one", &v1, &ops).is_none());
    }

    #[test]
    fn invalidate_clears_all() {
        let ops = make_ops();
        let mut cache = SemanticCache::new(16, 300);
        let v = encode_label(&ops, "hello").unwrap();
        cache.insert("hello", v.clone(), make_result());
        assert_eq!(cache.live_count(), 1);

        cache.invalidate_all();
        assert_eq!(cache.live_count(), 0);
        assert!(cache.find("hello", &v, &ops).is_none());
    }
}
