//! KG context injection for LLM prompts.
//!
//! Before each LLM inference, this module:
//! 1. Extracts topic keywords from user input
//! 2. Resolves keywords to symbols via the engine's label registry
//! 3. Runs VSA similarity search to find related symbols (top_k=5)
//! 4. Retrieves entity triples from the knowledge graph for matched symbols
//! 5. Formats triples as a structured block for prompt injection
//! 6. Caches results per session to avoid redundant KG queries

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::symbol::SymbolId;

// ── Constants ──────────────────────────────────────────────────────────────

/// Maximum characters in the context block (conservative ~2000 token estimate).
const MAX_CONTEXT_CHARS: usize = 8_000;

/// Maximum triples to include per context block.
const MAX_TRIPLES: usize = 15;

/// Maximum triples to retrieve per matched symbol (outgoing or incoming).
const TRIPLES_PER_SYMBOL: usize = 3;

/// Maximum similar symbols to fetch per resolved keyword.
const TOP_K_SIMILAR: usize = 5;

/// Maximum cache entries per session.
const MAX_CACHE_ENTRIES: usize = 64;

// ── Cache ──────────────────────────────────────────────────────────────────

/// Session-scoped cache for KG context lookups.
///
/// Bounded at `MAX_CACHE_ENTRIES`. When full, the oldest entry is evicted
/// (FIFO). The cache key is a normalized keyword fingerprint derived from
/// the user input.
#[derive(Debug, Default)]
pub struct KgContextCache {
    entries: HashMap<String, String>,
    insertion_order: Vec<String>,
}

impl KgContextCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    fn insert(&mut self, key: String, context: String) {
        if self.entries.len() >= MAX_CACHE_ENTRIES && !self.entries.contains_key(&key) {
            if let Some(oldest) = self.insertion_order.first().cloned() {
                self.entries.remove(&oldest);
                self.insertion_order.remove(0);
            }
        }
        if !self.entries.contains_key(&key) {
            self.insertion_order.push(key.clone());
        }
        self.entries.insert(key, context);
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

// ── Public API ─────────────────────────────────────────────────────────────

/// Build a KG context block for the given user input.
///
/// Queries the engine for triples related to topic keywords extracted from
/// `input`. Results are cached using `cache` to avoid redundant lookups on
/// repeated topics within a session.
///
/// Returns `None` when:
/// - No content keywords are found in `input`
/// - No keywords resolve to known symbols
/// - The KG contains no triples for matched symbols
pub fn build_kg_context(
    engine: &Engine,
    input: &str,
    cache: &mut KgContextCache,
) -> Option<String> {
    let keywords = extract_keywords(input);
    if keywords.is_empty() {
        return None;
    }

    // Cache key: sorted, lowercased keywords joined by "|"
    let mut sorted_lower: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
    sorted_lower.sort_unstable();
    let cache_key = sorted_lower.join("|");

    if let Some(cached) = cache.get(&cache_key) {
        tracing::debug!(keywords = ?keywords, "KG context cache hit");
        return if cached.is_empty() {
            None
        } else {
            Some(cached.to_string())
        };
    }

    let t0 = std::time::Instant::now();
    let triples = collect_relevant_triples(engine, &keywords);
    let context = if triples.is_empty() {
        None
    } else {
        let texts: Vec<String> = triples.into_iter().map(|(_, t)| t).collect();
        Some(format_context_block(&texts))
    };
    tracing::debug!(
        keywords = ?keywords,
        elapsed_ms = t0.elapsed().as_millis(),
        found = context.is_some(),
        "KG context query complete"
    );

    // Cache the result (empty string for negative results, avoids re-querying).
    cache.insert(cache_key, context.clone().unwrap_or_default());
    context
}

/// Build a KG context block of only the facts **not yet shown this session** (44d).
///
/// Identical relevance retrieval to [`build_kg_context`], but each triple already
/// recorded in `seen` is omitted and every newly-emitted triple is inserted into
/// `seen`. On the first turn this is the full block; on later turns it is the
/// *delta* — the activation frontier that moved — so unchanged facts the model
/// already holds in its (large-window) history aren't re-sent. Returns `None`
/// when no relevant facts, or when every relevant fact was already shown.
///
/// Caveat (documented, follow-up): if recent history is later evicted under tight
/// budget, an already-shown fact won't be re-injected. Safe while windows are
/// large (the measure-first finding); revisit alongside history summarisation.
pub fn build_kg_context_delta(
    engine: &Engine,
    input: &str,
    seen: &mut std::collections::HashSet<(u64, u64, u64)>,
) -> Option<String> {
    let keywords = extract_keywords(input);
    if keywords.is_empty() {
        return None;
    }
    let fresh: Vec<String> = collect_relevant_triples(engine, &keywords)
        .into_iter()
        .filter(|(key, _)| seen.insert(*key))
        .map(|(_, text)| text)
        .collect();
    if fresh.is_empty() {
        None
    } else {
        Some(format_context_block(&fresh))
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Extract topic keywords from user input.
///
/// Splits on non-alphanumeric characters (allowing hyphens and underscores),
/// filters common English stop words, and discards very short tokens.
/// Returns up to 8 unique keywords preserving original casing.
pub fn extract_keywords(input: &str) -> Vec<String> {
    let raw: Vec<String> = input
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter_map(|w| {
            let w = w.trim();
            if w.len() >= 3 && !is_stop_word(w) {
                Some(w.to_string())
            } else {
                None
            }
        })
        .collect();

    // Deduplicate (case-insensitive) while preserving original order and casing.
    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for w in raw {
        if seen.insert(w.to_lowercase()) {
            result.push(w);
        }
    }
    result.truncate(8);
    result
}

/// Returns `true` if `word` is a common English stop word.
fn is_stop_word(word: &str) -> bool {
    static STOP_WORDS: &[&str] = &[
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
        "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "now", "old", "see",
        "two", "way", "who", "did", "let", "put", "say", "she", "too", "use", "what", "when",
        "with", "this", "that", "from", "your", "they", "been", "have", "more", "will", "also",
        "about", "which", "there", "their", "would", "other", "after", "first", "these", "could",
        "into", "then", "than", "some", "just", "over", "such", "tell", "ask", "give", "show",
        "list", "find", "know", "does", "make", "take", "look", "come", "think", "help", "need",
        "want", "like",
    ];
    let lower = word.to_lowercase();
    STOP_WORDS.contains(&lower.as_str())
}

/// Collect KG context by resolving keywords → symbols → triples.
///
/// Returns up to [`MAX_TRIPLES`] `((subject, predicate, object), "s p o")` pairs,
/// deduplicated. The key lets callers do session-level delta filtering (44d);
/// [`build_kg_context`] just formats the texts.
fn collect_relevant_triples(
    engine: &Engine,
    keywords: &[String],
) -> Vec<((u64, u64, u64), String)> {
    let mut triples: Vec<((u64, u64, u64), String)> = Vec::new();
    let mut seen: HashSet<(u64, u64, u64)> = HashSet::new();

    for keyword in keywords {
        if triples.len() >= MAX_TRIPLES {
            break;
        }

        // Try exact label lookup, then lowercase fallback.
        let seed = engine
            .resolve_symbol(keyword)
            .or_else(|_| engine.resolve_symbol(&keyword.to_lowercase()));

        let seed_sym = match seed {
            Ok(s) => s,
            Err(_) => continue,
        };

        collect_triples(engine, seed_sym, &mut triples, &mut seen);

        if triples.len() >= MAX_TRIPLES {
            break;
        }

        // VSA similarity search: expand to related symbols.
        if let Ok(similar) = engine.search_similar_to(seed_sym, TOP_K_SIMILAR) {
            for sr in similar {
                if triples.len() >= MAX_TRIPLES {
                    break;
                }
                // Skip very low-similarity matches (noise threshold).
                if sr.similarity < 0.05 {
                    continue;
                }
                collect_triples(engine, sr.symbol_id, &mut triples, &mut seen);
            }
        }
    }

    triples
}

/// Append keyed triples for `symbol` to `out`, respecting the dedup set.
///
/// Collects up to `TRIPLES_PER_SYMBOL` outgoing triples and up to
/// `TRIPLES_PER_SYMBOL` incoming triples.
fn collect_triples(
    engine: &Engine,
    symbol: SymbolId,
    out: &mut Vec<((u64, u64, u64), String)>,
    seen: &mut HashSet<(u64, u64, u64)>,
) {
    let mut push = |t: &crate::graph::Triple, out: &mut Vec<((u64, u64, u64), String)>| {
        let key = (t.subject.get(), t.predicate.get(), t.object.get());
        if seen.insert(key) {
            out.push((
                key,
                format!(
                    "{} {} {}",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                ),
            ));
        }
    };
    for t in engine.triples_from(symbol).iter().take(TRIPLES_PER_SYMBOL) {
        push(t, out);
    }
    for t in engine.triples_to(symbol).iter().take(TRIPLES_PER_SYMBOL) {
        push(t, out);
    }
}

/// Format a list of "subject predicate object" strings into a context block.
pub fn format_context_block(triples: &[String]) -> String {
    let mut block = String::from("## Knowledge Graph Context\n");
    let mut total = block.len();

    for line in triples {
        let entry = format!("- {line}\n");
        if total + entry.len() > MAX_CONTEXT_CHARS {
            break;
        }
        block.push_str(&entry);
        total += entry.len();
    }

    block
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keywords_filters_stopwords() {
        let kws = extract_keywords("what are the properties of Ennead");
        assert!(!kws.contains(&"what".to_string()));
        assert!(!kws.contains(&"are".to_string()));
        assert!(!kws.contains(&"the".to_string()));
        assert!(kws.contains(&"properties".to_string()));
        assert!(kws.contains(&"Ennead".to_string()));
    }

    #[test]
    fn extract_keywords_deduplicates_case_insensitive() {
        let kws = extract_keywords("Ennead ennead members");
        let ennead_count = kws.iter().filter(|k| k.to_lowercase() == "ennead").count();
        assert_eq!(ennead_count, 1);
    }

    #[test]
    fn extract_keywords_filters_short_words() {
        let kws = extract_keywords("go to top");
        assert!(!kws.contains(&"go".to_string()));
        assert!(!kws.iter().any(|k| k.len() < 3));
    }

    #[test]
    fn extract_keywords_caps_at_eight() {
        let input = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda";
        let kws = extract_keywords(input);
        assert!(kws.len() <= 8);
    }

    #[test]
    fn extract_keywords_preserves_original_casing() {
        let kws = extract_keywords("Ennead members");
        assert!(kws.contains(&"Ennead".to_string()));
    }

    #[test]
    fn extract_keywords_handles_punctuation() {
        let kws = extract_keywords("hello, world! foo-bar baz_qux");
        assert!(kws.contains(&"hello".to_string()));
        assert!(kws.contains(&"world".to_string()));
    }

    #[test]
    fn format_context_block_has_header() {
        let block = format_context_block(&["Sun is-a Star".to_string()]);
        assert!(block.starts_with("## Knowledge Graph Context\n"));
        assert!(block.contains("- Sun is-a Star\n"));
    }

    #[test]
    fn format_context_block_empty_input() {
        let block = format_context_block(&[]);
        assert_eq!(block, "## Knowledge Graph Context\n");
    }

    #[test]
    fn format_context_block_respects_char_limit() {
        let long_triple = "a".repeat(MAX_CONTEXT_CHARS / 2);
        let triples = vec![
            long_triple.clone(),
            long_triple.clone(),
            long_triple.clone(),
        ];
        let block = format_context_block(&triples);
        assert!(block.len() <= MAX_CONTEXT_CHARS + 10);
    }

    #[test]
    fn cache_hit_returns_stored_value() {
        let mut cache = KgContextCache::new();
        cache.insert("key".to_string(), "ctx".to_string());
        assert_eq!(cache.get("key"), Some("ctx"));
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = KgContextCache::new();
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn cache_evicts_oldest_when_full() {
        let mut cache = KgContextCache::new();
        for i in 0..MAX_CACHE_ENTRIES {
            cache.insert(format!("key_{i}"), format!("val_{i}"));
        }
        cache.insert("overflow".to_string(), "new".to_string());
        assert!(
            cache.get("key_0").is_none(),
            "oldest entry should be evicted"
        );
        assert_eq!(cache.get("overflow"), Some("new"));
        assert_eq!(cache.len(), MAX_CACHE_ENTRIES);
    }

    #[test]
    fn cache_update_does_not_grow_insertion_order() {
        let mut cache = KgContextCache::new();
        cache.insert("k".to_string(), "v1".to_string());
        cache.insert("k".to_string(), "v2".to_string());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.insertion_order.len(), 1);
    }

    #[test]
    fn build_kg_context_returns_none_for_stopword_only_input() {
        // Can test without an engine: keyword extraction returns empty.
        let kws = extract_keywords("what are the and for");
        assert!(kws.is_empty());
    }

    #[test]
    fn delta_emits_each_fact_once_per_session() {
        use crate::engine::{Engine, EngineConfig};
        use crate::graph::Triple;

        let engine = Engine::new(EngineConfig::default()).unwrap();
        let rust = engine.resolve_or_create_entity("rustlang").unwrap();
        let prop = engine.resolve_or_create_entity("has-property").unwrap();
        let safe = engine.resolve_or_create_entity("memory-safe").unwrap();
        engine.add_triple(&Triple::new(rust, prop, safe)).unwrap();

        let mut seen = std::collections::HashSet::new();
        // First turn: the fact is delivered.
        let first = build_kg_context_delta(&engine, "tell me about rustlang", &mut seen);
        assert!(
            first.as_deref().unwrap_or("").contains("memory-safe"),
            "first turn delivers the fact"
        );
        // Second turn, same topic: already shown ⇒ nothing new.
        let second = build_kg_context_delta(&engine, "tell me about rustlang", &mut seen);
        assert!(
            second.is_none(),
            "already-shown fact is not re-sent: {second:?}"
        );
    }
}
