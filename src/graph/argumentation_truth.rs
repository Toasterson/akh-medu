//! Argumentation-based truth values: dynamic truth computation via evidence weighing.
//!
//! Instead of relying on a single stored confidence value, this module computes
//! truth by collecting pro/con arguments (via Phase 9e) and caching the verdict.
//! When new evidence arrives or TMS retracts a support, the cached truth is
//! invalidated and recomputed on next query.
//!
//! This is opt-in — callers use `query_with_argumentation()` instead of plain queries
//! to activate dynamic truth computation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::argumentation::{ArgumentSet, MetaRule};
use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Cache key: a (subject, predicate) pair for which argumentation was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArgumentationKey {
    pub subject: SymbolId,
    pub predicate: SymbolId,
}

/// Cached argumentation result for a (subject, predicate) query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedVerdict {
    /// The full argument set with verdict.
    pub argument_set: ArgumentSet,
    /// The effective confidence from the verdict.
    pub effective_confidence: f64,
    /// Monotonically increasing version counter — incremented when cache is invalidated.
    pub version: u64,
}

/// Cache of argumentation verdicts.
///
/// Stores computed verdicts for (subject, predicate) pairs so they don't need
/// to be recomputed on every query. Verdicts are invalidated when relevant
/// triples change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArgumentationCache {
    entries: HashMap<ArgumentationKey, CachedVerdict>,
    /// Global version counter, incremented on each invalidation.
    global_version: u64,
}

impl ArgumentationCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cached verdict, if available.
    pub fn get(&self, subject: SymbolId, predicate: SymbolId) -> Option<&CachedVerdict> {
        let key = ArgumentationKey { subject, predicate };
        self.entries.get(&key)
    }

    /// Store a verdict in the cache.
    pub fn put(&mut self, argument_set: ArgumentSet) {
        let key = ArgumentationKey {
            subject: argument_set.subject,
            predicate: argument_set.predicate,
        };
        let confidence = argument_set.verdict.confidence;
        self.global_version += 1;
        self.entries.insert(
            key,
            CachedVerdict {
                argument_set,
                effective_confidence: confidence,
                version: self.global_version,
            },
        );
    }

    /// Invalidate all cache entries that involve the given symbol
    /// (as subject or predicate).
    ///
    /// Called when a triple involving this symbol is added, removed, or modified.
    pub fn invalidate_for_symbol(&mut self, symbol: SymbolId) {
        self.entries.retain(|key, _| {
            key.subject != symbol && key.predicate != symbol
        });
    }

    /// Invalidate a specific (subject, predicate) entry.
    pub fn invalidate(&mut self, subject: SymbolId, predicate: SymbolId) {
        let key = ArgumentationKey { subject, predicate };
        self.entries.remove(&key);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.global_version += 1;
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Argumentation-based query
// ---------------------------------------------------------------------------

/// Query triples for `(subject, predicate, ?)` using argumentation-based truth values.
///
/// Instead of returning raw triples with their stored confidence, this function:
/// 1. Finds all candidate triples
/// 2. Runs argumentation (meta-rule ranking from Phase 9e)
/// 3. Returns triples sorted by argumentation-derived confidence
///
/// If a cached verdict exists, uses it. Otherwise computes fresh.
pub fn query_with_argumentation(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    cache: &mut ArgumentationCache,
) -> AkhResult<Vec<(Triple, f64)>> {
    // Check cache first
    if let Some(cached) = cache.get(subject, predicate) {
        return Ok(triples_from_argument_set(
            engine,
            &cached.argument_set,
        ));
    }

    // Compute fresh argumentation
    let arg_set = crate::argumentation::argue(engine, subject, predicate)?;

    // Extract triples with argumentation-derived confidence
    let results = triples_from_argument_set(engine, &arg_set);

    // Cache the result
    cache.put(arg_set);

    Ok(results)
}

/// Query with a custom set of meta-rules.
pub fn query_with_argumentation_rules(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    meta_rules: &[MetaRule],
    cache: &mut ArgumentationCache,
) -> AkhResult<Vec<(Triple, f64)>> {
    // Always compute fresh for custom rules (don't cache)
    let arg_set =
        crate::argumentation::argue_with_rules(engine, subject, predicate, meta_rules)?;
    let results = triples_from_argument_set(engine, &arg_set);
    cache.put(arg_set);
    Ok(results)
}

/// Extract triples with their argumentation-derived confidence from an ArgumentSet.
fn triples_from_argument_set(
    engine: &Engine,
    arg_set: &ArgumentSet,
) -> Vec<(Triple, f64)> {
    let kg = engine.knowledge_graph();
    let mut results = Vec::new();

    // The verdict's answer is the winning candidate
    if let Some(winner) = arg_set.verdict.answer {
        // Find the actual triple
        let triples = kg.triples_from(arg_set.subject);
        for t in triples {
            if t.predicate == arg_set.predicate && t.object == winner {
                results.push((t, arg_set.verdict.confidence));
                break;
            }
        }
    }

    // Also include other candidates with their computed scores
    for (&candidate_id, args) in &arg_set.candidates {
        if Some(candidate_id) == arg_set.verdict.answer {
            continue; // Already added as winner
        }
        let triples = kg.triples_from(arg_set.subject);
        for t in triples {
            if t.predicate == arg_set.predicate && t.object == candidate_id {
                // Compute a confidence from this candidate's arguments
                // Use default meta-rule order for scoring
                let default_order = crate::argumentation::MetaRule::default_order();
                let total_strength: f64 = args
                    .iter()
                    .map(|a| a.scores.weighted_total(&default_order))
                    .sum::<f64>()
                    / args.len().max(1) as f64;
                results.push((t, total_strength.clamp(0.0, 1.0)));
                break;
            }
        }
    }

    // Sort by confidence descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;

    #[test]
    fn cache_put_and_get() {
        let mut cache = ArgumentationCache::new();
        assert!(cache.is_empty());

        let subject = SymbolId::new(1).unwrap();
        let predicate = SymbolId::new(2).unwrap();
        let answer = SymbolId::new(3).unwrap();

        let arg_set = ArgumentSet {
            subject,
            predicate,
            candidates: HashMap::new(),
            verdict: crate::argumentation::Verdict {
                answer: Some(answer),
                confidence: 0.85,
                pro: vec![],
                con: vec![],
                reasoning: "test".to_string(),
                decisive_rule: None,
            },
        };

        cache.put(arg_set);
        assert_eq!(cache.len(), 1);

        let cached = cache.get(subject, predicate).unwrap();
        assert!((cached.effective_confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_invalidate_for_symbol() {
        let mut cache = ArgumentationCache::new();
        let s1 = SymbolId::new(1).unwrap();
        let s2 = SymbolId::new(2).unwrap();
        let p1 = SymbolId::new(10).unwrap();
        let p2 = SymbolId::new(11).unwrap();

        let make_set = |s, p| ArgumentSet {
            subject: s,
            predicate: p,
            candidates: HashMap::new(),
            verdict: crate::argumentation::Verdict {
                answer: None,
                confidence: 0.5,
                pro: vec![],
                con: vec![],
                reasoning: String::new(),
                decisive_rule: None,
            },
        };

        cache.put(make_set(s1, p1));
        cache.put(make_set(s2, p2));
        assert_eq!(cache.len(), 2);

        cache.invalidate_for_symbol(s1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(s1, p1).is_none());
        assert!(cache.get(s2, p2).is_some());
    }

    #[test]
    fn cache_invalidate_specific() {
        let mut cache = ArgumentationCache::new();
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();

        let arg_set = ArgumentSet {
            subject: s,
            predicate: p,
            candidates: HashMap::new(),
            verdict: crate::argumentation::Verdict {
                answer: None,
                confidence: 0.5,
                pro: vec![],
                con: vec![],
                reasoning: String::new(),
                decisive_rule: None,
            },
        };

        cache.put(arg_set);
        assert_eq!(cache.len(), 1);

        cache.invalidate(s, p);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_clear() {
        let mut cache = ArgumentationCache::new();
        let s = SymbolId::new(1).unwrap();
        let p = SymbolId::new(2).unwrap();

        cache.put(ArgumentSet {
            subject: s,
            predicate: p,
            candidates: HashMap::new(),
            verdict: crate::argumentation::Verdict {
                answer: None,
                confidence: 0.5,
                pro: vec![],
                con: vec![],
                reasoning: String::new(),
                decisive_rule: None,
            },
        });

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn query_with_argumentation_basic() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
        let is_a = engine
            .create_symbol(SymbolKind::Relation, "is-a")
            .unwrap();
        let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
        engine
            .add_triple(&Triple::new(sun.id, is_a.id, star.id))
            .unwrap();

        let mut cache = ArgumentationCache::new();
        let results =
            query_with_argumentation(&engine, sun.id, is_a.id, &mut cache).unwrap();

        // Should have at least the winning triple
        assert!(!results.is_empty());
        assert_eq!(results[0].0.object, star.id);
    }
}
