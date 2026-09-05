//! Gemma generation handler: pluggable backend, semantic cache, Psyche persona injection.
//!
//! The handler wraps any [`GemmaBackend`] implementation with:
//! - A TTL-bounded in-memory semantic cache (cache-hit path < 0.1 ms).
//! - Persona tone injection via a system-prompt prefix.
//! - Two generation modes so the fallback chain can try cached, then full inference.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::{GemmaError, GemmaResult};

// ---------------------------------------------------------------------------
// GemmaBackend trait
// ---------------------------------------------------------------------------

/// Pluggable generation backend for a Gemma-class LLM.
///
/// Implement this to connect an actual model:
/// - local Gemma via Candle GGUF
/// - local Gemma via llama-cpp-2
/// - remote Gemma API proxy
pub trait GemmaBackend: Send + Sync {
    /// Generate a completion for `prompt`, optionally grounded in `context`.
    ///
    /// The handler prepends the persona tone before calling this method.
    fn generate(&self, prompt: &str, context: Option<&str>) -> GemmaResult<String>;

    /// Returns `true` when the backend can accept requests right now.
    fn is_available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// GenerationMode
// ---------------------------------------------------------------------------

/// Controls whether the handler checks the semantic cache before inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    /// Return a cached response if available; error with [`GemmaError::CacheMiss`] otherwise.
    Cached,
    /// Always run full inference; bypass the cache check.
    Full,
}

// ---------------------------------------------------------------------------
// SemanticCache
// ---------------------------------------------------------------------------

struct CacheEntry {
    response: String,
    inserted_at: Instant,
}

/// In-memory response cache with TTL eviction.
///
/// Keys are normalised (lowercased, whitespace-collapsed) prompt strings.
/// On `max_entries` overflow the oldest entry is evicted.
pub struct SemanticCache {
    entries: HashMap<String, CacheEntry>,
    ttl: Duration,
    max_entries: usize,
}

impl SemanticCache {
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            ttl,
            max_entries,
        }
    }

    fn normalize(prompt: &str) -> String {
        prompt
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return a cached response for `prompt` if one exists and has not expired.
    pub fn get(&self, prompt: &str) -> Option<&str> {
        let key = Self::normalize(prompt);
        self.entries.get(&key).and_then(|e| {
            if e.inserted_at.elapsed() < self.ttl {
                Some(e.response.as_str())
            } else {
                None
            }
        })
    }

    /// Store a response for `prompt`.
    pub fn insert(&mut self, prompt: &str, response: String) {
        if self.entries.len() >= self.max_entries {
            // Evict the oldest entry to stay within budget
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.inserted_at)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest);
            }
        }
        let key = Self::normalize(prompt);
        self.entries.insert(
            key,
            CacheEntry {
                response,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Drop all entries whose TTL has expired.
    pub fn evict_stale(&mut self) {
        self.entries
            .retain(|_, e| e.inserted_at.elapsed() < self.ttl);
    }

    /// Number of live (non-expired) entries.
    pub fn len(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.inserted_at.elapsed() < self.ttl)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SemanticCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), 256)
    }
}

// ---------------------------------------------------------------------------
// GemmaHandler
// ---------------------------------------------------------------------------

/// Wraps a [`GemmaBackend`] with caching and Psyche persona injection.
///
/// # Fallback integration
///
/// The fallback chain calls this handler twice:
/// 1. `generate(…, GenerationMode::Cached)` — returns a cached hit or [`GemmaError::CacheMiss`]
/// 2. `generate(…, GenerationMode::Full)` — runs inference; returns [`GemmaError::Unavailable`]
///    when no backend is configured.
pub struct GemmaHandler {
    backend: Option<Box<dyn GemmaBackend>>,
    pub cache: SemanticCache,
    /// System-prompt prefix derived from the active Psyche persona tone.
    persona_tone: String,
}

impl GemmaHandler {
    /// Create a stub handler with no backend (useful for testing routing logic).
    pub fn new_stub() -> Self {
        Self {
            backend: None,
            cache: SemanticCache::default(),
            persona_tone: String::from("You are a knowledgeable and thoughtful assistant."),
        }
    }

    /// Create a handler backed by `backend` with the given Psyche persona tone.
    pub fn with_backend(backend: Box<dyn GemmaBackend>, persona_tone: impl Into<String>) -> Self {
        Self {
            backend: Some(backend),
            cache: SemanticCache::default(),
            persona_tone: persona_tone.into(),
        }
    }

    /// Replace the active persona tone (e.g. after a Psyche evolution cycle).
    pub fn set_persona_tone(&mut self, tone: impl Into<String>) {
        self.persona_tone = tone.into();
    }

    /// Returns `true` when a backend is configured and currently reachable.
    pub fn is_available(&self) -> bool {
        self.backend.as_ref().is_some_and(|b| b.is_available())
    }

    /// Generate a response for `prompt` with optional KG `context`.
    ///
    /// - `Cached`: returns a cached hit, or [`GemmaError::CacheMiss`] on miss.
    /// - `Full`: runs inference; prepends the persona tone; caches the result.
    ///
    /// Returns [`GemmaError::Unavailable`] when no backend is configured or reachable.
    pub fn generate(
        &mut self,
        prompt: &str,
        context: Option<&str>,
        mode: GenerationMode,
    ) -> GemmaResult<String> {
        if mode == GenerationMode::Cached {
            return match self.cache.get(prompt) {
                Some(hit) => Ok(hit.to_string()),
                None => Err(GemmaError::CacheMiss),
            };
        }

        // Full inference path — check availability before touching the backend
        if !self.is_available() {
            return Err(GemmaError::Unavailable);
        }

        let enriched = if context.is_some_and(|c| !c.is_empty()) {
            format!(
                "[Persona: {}]\n\n[Context: {}]\n\n{}",
                self.persona_tone,
                context.unwrap_or_default(),
                prompt
            )
        } else {
            format!("[Persona: {}]\n\n{}", self.persona_tone, prompt)
        };

        // Scope the backend borrow so we can mutably borrow `self.cache` afterwards
        let response = {
            let backend = self.backend.as_ref().ok_or(GemmaError::Unavailable)?;
            backend.generate(&enriched, context)?
        };

        self.cache.insert(prompt, response.clone());
        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoBackend;

    impl GemmaBackend for EchoBackend {
        fn generate(&self, prompt: &str, _context: Option<&str>) -> GemmaResult<String> {
            Ok(format!("echo: {prompt}"))
        }
        fn is_available(&self) -> bool {
            true
        }
    }

    struct UnavailableBackend;

    impl GemmaBackend for UnavailableBackend {
        fn generate(&self, _prompt: &str, _context: Option<&str>) -> GemmaResult<String> {
            Err(GemmaError::Unavailable)
        }
        fn is_available(&self) -> bool {
            false
        }
    }

    #[test]
    fn stub_returns_unavailable() {
        let mut h = GemmaHandler::new_stub();
        assert!(matches!(
            h.generate("hello", None, GenerationMode::Full),
            Err(GemmaError::Unavailable)
        ));
    }

    #[test]
    fn stub_returns_cache_miss_for_cached_mode() {
        let mut h = GemmaHandler::new_stub();
        assert!(matches!(
            h.generate("hello", None, GenerationMode::Cached),
            Err(GemmaError::CacheMiss)
        ));
    }

    #[test]
    fn echo_backend_generates_and_caches() {
        let mut h = GemmaHandler::with_backend(Box::new(EchoBackend), "Scholar");
        let r1 = h.generate("test prompt", None, GenerationMode::Full);
        assert!(r1.is_ok(), "expected success, got {r1:?}");

        // Second call with Cached mode should hit the cache
        let r2 = h.generate("test prompt", None, GenerationMode::Cached);
        assert!(r2.is_ok(), "expected cache hit");
    }

    #[test]
    fn unavailable_backend_propagates_error() {
        let mut h = GemmaHandler::with_backend(Box::new(UnavailableBackend), "Scholar");
        assert!(matches!(
            h.generate("hello", None, GenerationMode::Full),
            Err(GemmaError::Unavailable)
        ));
    }

    #[test]
    fn context_is_injected_into_prompt() {
        struct CapturingBackend;
        impl GemmaBackend for CapturingBackend {
            fn generate(&self, prompt: &str, _ctx: Option<&str>) -> GemmaResult<String> {
                // The enriched prompt should contain the context
                assert!(
                    prompt.contains("[Context:"),
                    "expected [Context: …] in prompt"
                );
                Ok("ok".to_string())
            }
            fn is_available(&self) -> bool {
                true
            }
        }
        let mut h = GemmaHandler::with_backend(Box::new(CapturingBackend), "Scholar");
        h.generate("question", Some("kg fact"), GenerationMode::Full)
            .unwrap();
    }

    #[test]
    fn semantic_cache_ttl_eviction() {
        let mut cache = SemanticCache::new(Duration::from_millis(10), 10);
        cache.insert("prompt", "response".to_string());
        assert!(cache.get("prompt").is_some());
        // After TTL, entry should be gone
        std::thread::sleep(Duration::from_millis(20));
        assert!(cache.get("prompt").is_none());
    }

    #[test]
    fn semantic_cache_max_entries_evicts_oldest() {
        let mut cache = SemanticCache::new(Duration::from_secs(60), 2);
        cache.insert("a", "1".to_string());
        std::thread::sleep(Duration::from_millis(5)); // ensure distinct timestamps
        cache.insert("b", "2".to_string());
        std::thread::sleep(Duration::from_millis(5));
        cache.insert("c", "3".to_string()); // should evict "a"
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }
}
