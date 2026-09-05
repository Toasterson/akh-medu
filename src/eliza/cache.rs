//! In-memory TTL cache for ELIZA KDL scripts.
//!
//! Entries expire after 60 seconds. Both fresh reads and stale fallback
//! (when Seshat is unreachable) are supported via [`ScriptCache::get`]
//! and [`ScriptCache::get_stale`].

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache entry lifetime before a fresh Seshat fetch is preferred.
const TTL: Duration = Duration::from_secs(60);

/// One cached script entry.
#[derive(Debug, Clone)]
pub struct CachedScript {
    /// The verbatim KDL script content.
    pub content: String,
    /// The Seshat document UUID that produced this content.
    pub revision: String,
    /// When this entry was inserted or last refreshed.
    pub fetched_at: Instant,
}

impl CachedScript {
    /// Returns `true` if the entry is still within the TTL window.
    pub fn is_fresh(&self) -> bool {
        self.fetched_at.elapsed() < TTL
    }
}

/// Thread-unsafe in-memory TTL cache for ELIZA scripts.
///
/// Wrap in `Arc<RwLock<ScriptCache>>` for concurrent access.
#[derive(Debug, Default)]
pub struct ScriptCache {
    entries: HashMap<String, CachedScript>,
}

impl ScriptCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a fresh (within TTL) entry for `name`, or `None` if absent/expired.
    pub fn get(&self, name: &str) -> Option<&CachedScript> {
        self.entries.get(name).filter(|e| e.is_fresh())
    }

    /// Return an entry regardless of staleness — used as a degraded fallback
    /// when Seshat is temporarily unreachable.
    pub fn get_stale(&self, name: &str) -> Option<&CachedScript> {
        self.entries.get(name)
    }

    /// Insert or refresh a cache entry.
    pub fn insert(&mut self, name: impl Into<String>, content: String, revision: String) {
        self.entries.insert(
            name.into(),
            CachedScript {
                content,
                revision,
                fetched_at: Instant::now(),
            },
        );
    }

    /// Remove a single entry (e.g. after a save invalidation).
    pub fn invalidate(&mut self, name: &str) {
        self.entries.remove(name);
    }

    /// Remove all entries older than the TTL.
    pub fn evict_stale(&mut self) {
        self.entries.retain(|_, e| e.is_fresh());
    }

    /// Retrieve just the revision ID for `name` if it is currently cached (fresh or stale).
    pub fn revision(&self, name: &str) -> Option<&str> {
        self.entries.get(name).map(|e| e.revision.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_entry_retrieved() {
        let mut cache = ScriptCache::new();
        cache.insert("sage", "kdl content".into(), "rev-1".into());
        let entry = cache.get("sage").expect("should be fresh");
        assert_eq!(entry.content, "kdl content");
        assert_eq!(entry.revision, "rev-1");
    }

    #[test]
    fn missing_entry_returns_none() {
        let cache = ScriptCache::new();
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn invalidated_entry_gone() {
        let mut cache = ScriptCache::new();
        cache.insert("sage", "body".into(), "r1".into());
        cache.invalidate("sage");
        assert!(cache.get("sage").is_none());
        assert!(cache.get_stale("sage").is_none());
    }

    #[test]
    fn stale_fallback_available() {
        let mut cache = ScriptCache::new();
        cache.entries.insert(
            "sage".into(),
            CachedScript {
                content: "stale body".into(),
                revision: "r0".into(),
                fetched_at: Instant::now()
                    .checked_sub(Duration::from_secs(120))
                    .unwrap_or_else(Instant::now),
            },
        );
        assert!(cache.get("sage").is_none(), "should be expired");
        assert!(
            cache.get_stale("sage").is_some(),
            "stale fallback available"
        );
    }
}
