//! Device-local shader specialization cache. Independent of GPU allocation.
use std::collections::HashMap;

/// Observable cache behaviour. Counters are cumulative for the device's
/// lifetime; `entries` and `source_bytes` are the live residency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    /// Specializations served from residency.
    pub hits: u64,
    /// Specializations that required compilation, including failures.
    pub misses: u64,
    /// Least-recently-used entries dropped to stay inside the budget.
    pub evictions: u64,
    /// Shaders compiled and executed without becoming resident.
    pub bypassed: u64,
    /// Compilations that returned an error; these never insert or evict.
    pub compile_failures: u64,
    /// Whole-cache clears forced by recency-counter wrap.
    pub recency_resets: u64,
    /// Live resident specializations.
    pub entries: usize,
    /// Live resident source bytes.
    pub source_bytes: usize,
}

#[derive(Debug)]
pub(super) struct PipelineCache<T> {
    entries: HashMap<String, (T, u64)>,
    source_bytes: usize,
    clock: u64,
    max_entries: usize,
    max_source_bytes: usize,
    stats: PipelineCacheStats,
}

impl<T: Clone> PipelineCache<T> {
    pub(super) fn new(max_entries: usize, max_source_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            source_bytes: 0,
            clock: 0,
            max_entries,
            max_source_bytes,
            stats: PipelineCacheStats::default(),
        }
    }

    /// Snapshot of cumulative counters plus live residency.
    pub(super) fn stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            entries: self.entries.len(),
            source_bytes: self.source_bytes,
            ..self.stats
        }
    }

    pub(super) fn get_or_try_insert_with<E>(
        &mut self,
        source: String,
        compile: impl FnOnce(&str) -> Result<T, E>,
    ) -> Result<T, E> {
        // Reset recency before counter wrap; cached objects are only an optimization.
        if self.clock == u64::MAX {
            self.entries.clear();
            self.source_bytes = 0;
            self.clock = 0;
            self.stats.recency_resets += 1;
        }
        self.clock += 1;
        if let Some((value, stamp)) = self.entries.get_mut(&source) {
            *stamp = self.clock;
            self.stats.hits += 1;
            return Ok(value.clone());
        }
        self.stats.misses += 1;
        let value = match compile(&source) {
            Ok(value) => value,
            Err(error) => {
                self.stats.compile_failures += 1;
                return Err(error);
            }
        };
        // A single large shader may execute without becoming resident.
        if self.max_entries == 0 || source.len() > self.max_source_bytes {
            self.stats.bypassed += 1;
            return Ok(value);
        }
        while self.entries.len() >= self.max_entries
            || self.source_bytes > self.max_source_bytes - source.len()
        {
            // Misses already compile a shader. Scanning only on eviction keeps
            // hits O(1) without duplicating potentially large source keys.
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, stamp))| *stamp)
                .map(|(key, _)| key.clone())
                .expect("nonempty cache over budget");
            self.entries.remove(&oldest);
            self.source_bytes -= oldest.len();
            self.stats.evictions += 1;
        }
        self.source_bytes += source.len();
        self.entries.insert(source, (value.clone(), self.clock));
        Ok(value)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(cache: &mut PipelineCache<usize>, key: &str, value: usize) -> usize {
        cache
            .get_or_try_insert_with(key.into(), |_| Ok::<_, ()>(value))
            .unwrap()
    }

    #[test]
    fn varied_specializations_are_bounded_and_hot_entries_survive() {
        let mut cache = PipelineCache::new(2, 100);
        insert(&mut cache, "a", 1);
        insert(&mut cache, "b", 2);
        assert_eq!(insert(&mut cache, "a", 99), 1);
        insert(&mut cache, "c", 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.entries.contains_key("a"));
        assert!(!cache.entries.contains_key("b"));
        for i in 0..1000 {
            insert(&mut cache, &format!("shape_{i}"), i);
            assert!(cache.entries.len() <= 2);
            assert!(cache.source_bytes <= 100);
        }
    }

    #[test]
    fn source_byte_limit_and_oversized_bypass() {
        let mut cache = PipelineCache::new(10, 4);
        insert(&mut cache, "aa", 1);
        insert(&mut cache, "bbb", 2);
        assert_eq!(cache.source_bytes, 3);
        assert_eq!(insert(&mut cache, "oversized", 3), 3);
        assert_eq!(cache.source_bytes, 3);
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn failed_compilation_does_not_evict_or_cache() {
        let mut cache = PipelineCache::new(1, 10);
        insert(&mut cache, "a", 1);
        assert_eq!(
            cache.get_or_try_insert_with("b".into(), |_| Err::<usize, _>("bad")),
            Err("bad")
        );
        assert_eq!(insert(&mut cache, "a", 99), 1);
        assert_eq!(cache.source_bytes, 1);
    }

    #[test]
    fn zero_capacity_bypasses_cache() {
        let mut cache = PipelineCache::new(0, 100);
        assert_eq!(insert(&mut cache, "a", 1), 1);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn eviction_releases_cache_ownership_not_caller_ownership() {
        use std::sync::Arc;
        let mut cache = PipelineCache::new(1, 10);
        let held = cache
            .get_or_try_insert_with("a".into(), |_| Ok::<_, ()>(Arc::new(1)))
            .unwrap();
        let weak = Arc::downgrade(&held);
        cache
            .get_or_try_insert_with("b".into(), |_| Ok::<_, ()>(Arc::new(2)))
            .unwrap();
        assert_eq!(*held, 1);
        assert_eq!(Arc::strong_count(&held), 1);
        drop(held);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn byte_budget_can_evict_multiple_entries() {
        let mut cache = PipelineCache::new(10, 6);
        for key in ["aa", "bb", "cc"] {
            insert(&mut cache, key, 1);
        }
        insert(&mut cache, "dddddd", 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.source_bytes, 6);
    }

    #[test]
    fn counters_separate_hits_misses_evictions_and_bypasses() {
        let mut cache = PipelineCache::new(1, 4);
        insert(&mut cache, "a", 1);
        insert(&mut cache, "a", 99);
        insert(&mut cache, "b", 2);
        assert_eq!(insert(&mut cache, "oversized", 3), 3);
        assert!(cache
            .get_or_try_insert_with("c".into(), |_| Err::<usize, _>("bad"))
            .is_err());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.bypassed, 1);
        assert_eq!(stats.compile_failures, 1);
        assert_eq!(stats.recency_resets, 0);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.source_bytes, 1);
    }

    #[test]
    fn counters_record_recency_reset_without_losing_residency_accounting() {
        let mut cache = PipelineCache::new(2, 10);
        insert(&mut cache, "a", 1);
        cache.clock = u64::MAX;
        insert(&mut cache, "b", 2);
        let stats = cache.stats();
        assert_eq!(stats.recency_resets, 1);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.source_bytes, 1);
    }

    #[test]
    fn recency_counter_wrap_resets_safely() {
        let mut cache = PipelineCache::new(1, 10);
        insert(&mut cache, "a", 1);
        cache.clock = u64::MAX;
        assert_eq!(insert(&mut cache, "b", 2), 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.source_bytes, 1);
        assert_eq!(cache.clock, 1);
    }
}
