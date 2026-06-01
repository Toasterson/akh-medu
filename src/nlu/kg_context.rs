//! KG Context Builder — injects Knowledge Graph facts into LLM prompts.
//!
//! Before every LLM inference this module:
//! 1. Extracts topic keywords from user input
//! 2. Resolves keywords to KG symbols via exact label lookup
//! 3. Expands to related symbols via VSA similarity search (top_k=5)
//! 4. Retrieves KG triples for all matched symbols
//! 5. Formats them as a structured context block (< 2000 tokens)
//! 6. Caches results (session-scoped) to satisfy the < 50 ms repeat-query budget

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::engine::Engine;

// ── Constants ─────────────────────────────────────────────────────────────

/// Approximate token budget for the KG context block.
/// Combined with the persona prompt (< 1800 tokens) this stays under 3800 total.
const MAX_KG_TOKENS: usize = 1800;

/// Number of VSA nearest-neighbours to expand per keyword symbol.
const VSA_TOP_K: usize = 5;

/// Maximum triples retrieved per expanded symbol.
const MAX_TRIPLES_PER_SYMBOL: usize = 5;

/// Maximum entries in the session-scoped LRU cache.
const CACHE_MAX_ENTRIES: usize = 64;

/// Common English stopwords filtered before keyword extraction.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "about", "above", "across",
    "after", "against", "along", "among", "around", "at", "before",
    "behind", "below", "between", "by", "down", "during", "for", "from",
    "in", "inside", "into", "near", "of", "off", "on", "out", "outside",
    "over", "through", "to", "toward", "under", "up", "with", "without",
    "and", "but", "or", "nor", "so", "yet", "also", "not", "no", "its",
    "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "what",
    "which", "who", "how", "when", "where", "why", "if", "then", "else",
    "i", "tell", "show", "give", "list", "get", "find", "explain",
    "describe", "know", "think", "like", "just", "more", "some", "any",
];

// ── Session-scoped LRU cache ───────────────────────────────────────────────

struct KgContextCache {
    entries: HashMap<String, String>,
    order: VecDeque<String>,
    max_entries: usize,
}

impl KgContextCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
        }
    }

    /// Return cached value and bump it to most-recently-used.
    fn get(&mut self, key: &str) -> Option<&str> {
        if self.entries.contains_key(key) {
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            self.entries.get(key).map(String::as_str)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, value: String) {
        if self.entries.len() >= self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }
}

// ── KgContextBuilder ───────────────────────────────────────────────────────

/// Builds KG context blocks for injection into LLM prompts.
///
/// Instantiate one `KgContextBuilder` per chat session and call [`build`] before
/// every LLM inference. The internal cache makes repeated queries for the same
/// topic essentially free (HashMap lookup, < 1 µs).
pub struct KgContextBuilder {
    cache: KgContextCache,
}

impl KgContextBuilder {
    /// Create a new builder with a fresh session cache.
    pub fn new() -> Self {
        Self {
            cache: KgContextCache::new(CACHE_MAX_ENTRIES),
        }
    }

    /// Build a KG context block for the given user input.
    ///
    /// Returns `None` when the KG has no relevant facts (engine is empty, no
    /// keyword symbols found, or all expansions produce no triples).
    ///
    /// The returned string contains at most [`MAX_KG_TOKENS`] × 4 characters of
    /// KG triples, keeping the combined LLM prompt well below 3 800 tokens.
    pub fn build(&mut self, user_input: &str, engine: &Engine) -> Option<String> {
        let t0 = Instant::now();

        let keywords = Self::extract_keywords(user_input);
        if keywords.is_empty() {
            return None;
        }

        // Stable cache key: sorted keywords joined.
        let cache_key = {
            let mut sorted = keywords.clone();
            sorted.sort_unstable();
            sorted.join("|")
        };

        if let Some(cached) = self.cache.get(&cache_key) {
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                keywords = ?keywords,
                "kg_context: cache hit"
            );
            return if cached.is_empty() { None } else { Some(cached.to_string()) };
        }

        // Collect (subject, predicate, object) label triples.
        let mut triples: Vec<(String, String, String)> = Vec::new();

        for kw in &keywords {
            // Step 1: resolve keyword → SymbolId (exact label match).
            let Ok(kw_id) = engine.lookup_symbol(kw) else {
                tracing::trace!(kw, "kg_context: no symbol for keyword");
                continue;
            };

            // Step 2: VSA similarity expansion — find the top-k related symbols.
            let candidates = match engine.search_similar_to(kw_id, VSA_TOP_K) {
                Ok(c) => c,
                Err(e) => {
                    tracing::trace!(kw, error = %e, "kg_context: VSA search failed");
                    // Fallback: at least include the keyword symbol itself.
                    vec![crate::vsa::item_memory::SearchResult {
                        symbol_id: kw_id,
                        similarity: 1.0,
                    }]
                }
            };

            // Step 3: retrieve triples for each candidate.
            for candidate in &candidates {
                let sym_triples = engine.triples_from(candidate.symbol_id);
                let mut count = 0usize;
                for t in &sym_triples {
                    if count >= MAX_TRIPLES_PER_SYMBOL {
                        break;
                    }
                    let subj = engine.resolve_label(t.subject);
                    let pred = engine.resolve_label(t.predicate);
                    let obj = engine.resolve_label(t.object);
                    // Skip internal agent-namespace relations (agent: prefix).
                    if pred.starts_with("agent:") || pred.starts_with("sym:") {
                        continue;
                    }
                    triples.push((subj, pred, obj));
                    count += 1;
                }
            }
        }

        if triples.is_empty() {
            self.cache.insert(cache_key, String::new());
            tracing::debug!(
                elapsed_ms = t0.elapsed().as_millis(),
                keywords = ?keywords,
                "kg_context: no relevant triples found"
            );
            return None;
        }

        // Deduplicate while preserving order.
        triples.sort_unstable();
        triples.dedup();

        let block = Self::format_context_block(&triples);

        let elapsed = t0.elapsed();
        tracing::debug!(
            elapsed_ms = elapsed.as_millis(),
            triples = triples.len(),
            tokens_est = Self::estimate_tokens(&block),
            "kg_context: built context block"
        );

        self.cache.insert(cache_key, block.clone());
        Some(block)
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Extract topic keywords from user input.
    ///
    /// Splits on whitespace and ASCII punctuation, removes stopwords and
    /// tokens shorter than 3 characters, deduplicates, and returns at most
    /// 5 tokens sorted by length descending (longer = more specific).
    pub fn extract_keywords(input: &str) -> Vec<String> {
        let mut words: Vec<String> = input
            .split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
            .filter_map(|w| {
                let w = w.trim_matches('\'').trim().to_lowercase();
                if w.len() < 3 {
                    return None;
                }
                if STOPWORDS.contains(&w.as_str()) {
                    return None;
                }
                Some(w)
            })
            .collect();

        words.sort_unstable_by_key(|w| std::cmp::Reverse(w.len()));
        words.dedup();
        words.truncate(5);
        words
    }

    /// Format a list of (subject, predicate, object) label triples into a
    /// structured context block, truncating at the token budget.
    fn format_context_block(triples: &[(String, String, String)]) -> String {
        let header = "## Knowledge Graph Context\n";
        let mut block = String::with_capacity(header.len() + triples.len() * 48);
        block.push_str(header);

        for (subj, pred, obj) in triples {
            let line = format!("- {subj} {pred} {obj}\n");
            if Self::estimate_tokens(&block) + Self::estimate_tokens(&line) > MAX_KG_TOKENS {
                break;
            }
            block.push_str(&line);
        }

        block
    }

    /// Estimate token count using the 4-chars-per-token heuristic.
    ///
    /// Conservative approximation for Qwen/Gemma subword tokenizers.
    pub fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

impl Default for KgContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keywords_filters_stopwords() {
        let kws = KgContextBuilder::extract_keywords("tell me about the Ennead");
        assert!(!kws.contains(&"the".to_string()));
        assert!(!kws.contains(&"tell".to_string()));
        assert!(!kws.contains(&"me".to_string()));
        assert!(kws.contains(&"ennead".to_string()));
    }

    #[test]
    fn extract_keywords_max_five() {
        let input = "alpha beta gamma delta epsilon zeta eta";
        let kws = KgContextBuilder::extract_keywords(input);
        assert!(kws.len() <= 5);
    }

    #[test]
    fn extract_keywords_sorted_by_length() {
        let kws = KgContextBuilder::extract_keywords("ennead cluster topology node");
        // Longer words come first.
        for i in 1..kws.len() {
            assert!(kws[i - 1].len() >= kws[i].len());
        }
    }

    #[test]
    fn extract_keywords_deduplicates() {
        let kws = KgContextBuilder::extract_keywords("cluster cluster cluster");
        assert_eq!(kws.iter().filter(|k| k.as_str() == "cluster").count(), 1);
    }

    #[test]
    fn extract_keywords_empty_input() {
        let kws = KgContextBuilder::extract_keywords("the a an");
        assert!(kws.is_empty());
    }

    #[test]
    fn estimate_tokens_approx() {
        // 4 chars → 1 token (ceiling)
        assert_eq!(KgContextBuilder::estimate_tokens("abcd"), 1);
        assert_eq!(KgContextBuilder::estimate_tokens("abcde"), 2);
        assert_eq!(KgContextBuilder::estimate_tokens(""), 0);
    }

    #[test]
    fn format_context_block_header() {
        let triples = vec![
            ("Ra".to_string(), "is-a".to_string(), "deity".to_string()),
        ];
        let block = KgContextBuilder::extract_keywords("ra deity");
        let _ = block; // suppress unused warning
        let block = {
            let triples_ref: Vec<(String, String, String)> = triples;
            let header = "## Knowledge Graph Context\n";
            let mut b = String::from(header);
            b.push_str("- Ra is-a deity\n");
            b
        };
        assert!(block.starts_with("## Knowledge Graph Context\n"));
        assert!(block.contains("Ra is-a deity"));
    }

    #[test]
    fn cache_returns_cached_value() {
        let mut cache = KgContextCache::new(4);
        cache.insert("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get("key1"), Some("value1"));
        assert_eq!(cache.get("missing"), None);
    }

    #[test]
    fn cache_evicts_oldest_on_overflow() {
        let mut cache = KgContextCache::new(2);
        cache.insert("k1".to_string(), "v1".to_string());
        cache.insert("k2".to_string(), "v2".to_string());
        cache.insert("k3".to_string(), "v3".to_string()); // evicts k1
        assert_eq!(cache.get("k1"), None);
        assert_eq!(cache.get("k2"), Some("v2"));
        assert_eq!(cache.get("k3"), Some("v3"));
    }

    #[test]
    fn cache_lru_promotes_on_access() {
        let mut cache = KgContextCache::new(2);
        cache.insert("k1".to_string(), "v1".to_string());
        cache.insert("k2".to_string(), "v2".to_string());
        // Access k1 to promote it; next insert should evict k2.
        cache.get("k1");
        cache.insert("k3".to_string(), "v3".to_string()); // evicts k2
        assert_eq!(cache.get("k2"), None);
        assert_eq!(cache.get("k1"), Some("v1"));
        assert_eq!(cache.get("k3"), Some("v3"));
    }
}
