//! Autonomous triple extraction from text via LLM backends.
//!
//! Extracts structured `(subject, predicate, object, confidence)` triples from
//! natural-language text using a three-tier fallback:
//! 1. Local LLM (Qwen2.5-1.5B via llama-cpp-2, shared with NLU pipeline)
//! 2. External OpenAI-compatible API (OpenRouter, DeepSeek, etc.)
//! 3. Regex fallback (basic is-a/part-of patterns)
//!
//! Designed for autonomous operation in daemon mode — the akh populates its
//! own knowledge graph without human intervention.

pub mod config;
mod error;
mod prompt;
mod backends;

pub use config::{ExternalApiConfig, ExtractionConfig};
pub use error::{ExtractionError, ExtractionResult};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};

// ── Core types ──────────────────────────────────────────────────────────

/// A single extracted triple with confidence.
#[derive(Debug, Clone)]
pub struct RawTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

/// Which backend produced the extraction.
#[derive(Debug, Clone)]
pub enum ExtractionBackend {
    LocalLlm,
    ExternalApi { model: String },
    RegexFallback,
}

impl std::fmt::Display for ExtractionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalLlm => write!(f, "local_llm"),
            Self::ExternalApi { model } => write!(f, "external_api:{model}"),
            Self::RegexFallback => write!(f, "regex_fallback"),
        }
    }
}

/// Result of extraction from a text chunk.
#[derive(Debug)]
pub struct ExtractionOutput {
    pub triples: Vec<RawTriple>,
    pub backend_used: ExtractionBackend,
}

// ── Orchestrator ────────────────────────────────────────────────────────

/// Three-tier extraction orchestrator.
///
/// Tries local LLM first (if available), falls back to external API (if
/// configured), and finally falls back to regex patterns. Each tier
/// produces structured triples from text.
pub struct TripleExtractor {
    config: ExtractionConfig,
    #[cfg(feature = "nlu-llm")]
    local_backend: Option<backends::local::LocalLlmBackend>,
    api_backend: Option<backends::api::ExternalApiBackend>,
}

impl TripleExtractor {
    /// Create a new extractor with the given config.
    ///
    /// The local LLM backend is initialized lazily (shares model with NLU).
    /// The API backend is created if `external_api` config is present.
    pub fn new(config: ExtractionConfig) -> Self {
        let api_backend = config
            .external_api
            .as_ref()
            .map(|api_config| backends::api::ExternalApiBackend::new(api_config.clone()));

        Self {
            config,
            #[cfg(feature = "nlu-llm")]
            local_backend: None,
            api_backend,
        }
    }

    /// Set the local LLM backend (shared model from NLU pipeline).
    #[cfg(feature = "nlu-llm")]
    pub fn with_local_llm(
        mut self,
        translator: std::sync::Arc<crate::nlu::llm_translator::LlmTranslator>,
    ) -> Self {
        self.local_backend = Some(backends::local::LocalLlmBackend::new(translator));
        self
    }

    /// Extract triples from text using the three-tier fallback.
    pub fn extract(&self, text: &str) -> ExtractionResult<ExtractionOutput> {
        if !self.config.enabled {
            return Ok(ExtractionOutput {
                triples: vec![],
                backend_used: ExtractionBackend::RegexFallback,
            });
        }

        // Chunk text if too long.
        let chunks = chunk_text(text, self.config.max_chunk_chars);
        let mut all_triples = Vec::new();
        let mut backend_used = ExtractionBackend::RegexFallback;

        for chunk in &chunks {
            // Tier 1: Local LLM
            #[cfg(feature = "nlu-llm")]
            if self.config.prefer_local {
                if let Some(ref local) = self.local_backend {
                    match local.extract(chunk, self.config.max_triples_per_chunk) {
                        Ok(triples) if !triples.is_empty() => {
                            all_triples.extend(triples);
                            backend_used = ExtractionBackend::LocalLlm;
                            continue;
                        }
                        Ok(_) => {} // empty, try next tier
                        Err(e) => {
                            tracing::debug!("local LLM extraction failed: {e}");
                        }
                    }
                }
            }

            // Tier 2: External API
            if let Some(ref api) = self.api_backend {
                match api.extract(chunk, self.config.max_triples_per_chunk) {
                    Ok(triples) if !triples.is_empty() => {
                        all_triples.extend(triples);
                        backend_used = ExtractionBackend::ExternalApi {
                            model: self
                                .config
                                .external_api
                                .as_ref()
                                .map(|c| c.model.clone())
                                .unwrap_or_default(),
                        };
                        continue;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!("external API extraction failed: {e}");
                    }
                }
            }

            // Tier 3: Regex fallback
            let regex_triples = backends::regex::extract_regex(chunk);
            if !regex_triples.is_empty() {
                all_triples.extend(regex_triples);
                // backend_used stays RegexFallback
            }
        }

        // Filter by min confidence and deduplicate.
        all_triples.retain(|t| t.confidence >= self.config.min_confidence);
        dedup_triples(&mut all_triples);

        Ok(ExtractionOutput {
            triples: all_triples,
            backend_used,
        })
    }

    /// Extract triples and insert them into the engine with provenance.
    ///
    /// Returns the number of triples successfully inserted.
    pub fn extract_and_store(
        &self,
        text: &str,
        engine: &Engine,
    ) -> ExtractionResult<usize> {
        let output = self.extract(text)?;
        let mut inserted = 0;

        let source_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };

        for raw in &output.triples {
            let s = match engine.resolve_or_create_entity(&raw.subject) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let p = match engine.resolve_or_create_relation(&raw.predicate) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let o = match engine.resolve_or_create_entity(&raw.object) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let triple = Triple::new(s, p, o).with_confidence(raw.confidence);
            if engine.add_triple(&triple).is_ok() {
                // Store provenance.
                let mut record = ProvenanceRecord::new(
                    s,
                    DerivationKind::LlmTripleExtraction {
                        backend: output.backend_used.to_string(),
                        source_text_hash: source_hash,
                        model: match &output.backend_used {
                            ExtractionBackend::LocalLlm => "qwen2.5-1.5b".to_string(),
                            ExtractionBackend::ExternalApi { model } => model.clone(),
                            ExtractionBackend::RegexFallback => "regex".to_string(),
                        },
                    },
                )
                .with_confidence(raw.confidence)
                .with_sources(vec![s, p, o]);

                let _ = engine.store_provenance(&mut record);
                inserted += 1;
            }
        }

        Ok(inserted)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Split text into chunks at paragraph boundaries.
fn chunk_text(text: &str, max_chars: usize) -> Vec<&str> {
    if text.len() <= max_chars {
        return vec![text];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        // Try to break at a paragraph boundary.
        let break_at = if end < text.len() {
            text[start..end]
                .rfind("\n\n")
                .or_else(|| text[start..end].rfind('\n'))
                .or_else(|| text[start..end].rfind(". "))
                .map(|pos| start + pos + 1)
                .unwrap_or(end)
        } else {
            end
        };
        chunks.push(&text[start..break_at]);
        start = break_at;
    }

    chunks
}

/// Deduplicate triples by (subject, predicate, object) keeping highest confidence.
fn dedup_triples(triples: &mut Vec<RawTriple>) {
    let mut seen = std::collections::HashMap::new();
    for triple in triples.iter() {
        let key = (
            triple.subject.to_lowercase(),
            triple.predicate.to_lowercase(),
            triple.object.to_lowercase(),
        );
        let entry = seen.entry(key).or_insert(0.0f32);
        if triple.confidence > *entry {
            *entry = triple.confidence;
        }
    }

    let mut unique = Vec::new();
    let mut emitted = std::collections::HashSet::new();
    for triple in triples.drain(..) {
        let key = (
            triple.subject.to_lowercase(),
            triple.predicate.to_lowercase(),
            triple.object.to_lowercase(),
        );
        if emitted.insert(key) {
            unique.push(triple);
        }
    }
    *triples = unique;
}
