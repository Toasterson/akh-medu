//! LLM knowledge elicitation engine (Phase 32a + 32b).
//!
//! Prompts a local generative model (T5-Knowledge from Phase 35, currently
//! stubbed to the available Candle backend) to produce structured triples about
//! concepts through multi-pass, self-consistency-voted queries.  A multi-layer
//! validation pipeline (syntactic → dedup → arity → contradiction → neural
//! bridge → self-consistency scoring) filters hallucinations before KG
//! ingestion.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use akh_medu::bootstrap::knowledge_elicit::{
//!     ElicitConfig, ElicitPass, KnowledgeElicitor, ElicitValidator,
//! };
//! use akh_medu::engine::{Engine, EngineConfig};
//! use akh_medu::vsa::Dimension;
//!
//! let engine = Engine::new(EngineConfig { dimension: Dimension(64), ..EngineConfig::default() }).unwrap();
//! let config = ElicitConfig::default();
//! let mut elicitor = KnowledgeElicitor::new(config, None);
//! let result = elicitor.elicit("astronomy", &["stellar evolution".into()], &[ElicitPass::Taxonomy]);
//! let validator = ElicitValidator::new(None);
//! let validated = validator.validate(&result.triples, result.total_rounds, &engine);
//! ```
//!
//! Feature gate: the generative backend requires `candle-backend`.  Without it
//! all elicitation calls return empty results gracefully.

use miette::Diagnostic;
#[cfg(feature = "candle-backend")]
use std::collections::HashMap;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::graph::arity::ConstraintViolation;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the knowledge elicitation pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum ElicitError {
    #[error("LLM backend unavailable: {reason}")]
    #[diagnostic(
        code(akh::elicit::backend_unavailable),
        help(
            "Enable the `candle-backend` feature and ensure the model file exists at \
             `models/llm/t5-knowledge-q6k.gguf`.  Elicitation degrades gracefully \
             without the model — bootstrap continues via web APIs alone."
        )
    )]
    BackendUnavailable { reason: String },

    #[error("generation failed: {reason}")]
    #[diagnostic(
        code(akh::elicit::generation_failed),
        help(
            "The LLM produced no output or a fatal tokenization error occurred. \
             Try lowering `max_tokens` or switching to a smaller model."
        )
    )]
    GenerationFailed { reason: String },

    #[error("JSON parse failed: {reason}")]
    #[diagnostic(
        code(akh::elicit::json_parse),
        help(
            "The LLM output could not be parsed as a JSON triple array. \
             The lenient parser already handles trailing commas and partial arrays — \
             if this still fails the output was severely malformed.  \
             Try increasing `max_tokens` or adjusting the temperature."
        )
    )]
    JsonParseFailed { reason: String },

    #[error("engine error: {0}")]
    #[diagnostic(
        code(akh::elicit::engine),
        help("An engine-level error occurred while resolving or storing elicited triples.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ElicitError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type ElicitResult<T> = std::result::Result<T, ElicitError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which elicitation passes to run.
///
/// Each pass uses a different focused prompt template targeting a specific
/// relation family.  Running all five gives maximum coverage; running only
/// Taxonomy + Properties is fastest for a quick first fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElicitPass {
    /// Hierarchical structure: `is-a`, `subclass-of`, `part-of`.
    Taxonomy,
    /// Attribute facts: `has-property`, `characterized-by`, `measured-in`.
    Properties,
    /// Inter-concept relations: `causes`, `enables`, `prevents`, `depends-on`.
    Relationships,
    /// Action schemas for skill authoring (Phase 30): steps, preconditions, effects.
    Procedures,
    /// Temporal ordering: event sequences, `before`, `after`, `co-occurs-with`.
    Temporal,
}

impl ElicitPass {
    /// Short snake-case name, used in provenance records.
    pub fn name(self) -> &'static str {
        match self {
            Self::Taxonomy => "taxonomy",
            Self::Properties => "properties",
            Self::Relationships => "relationships",
            Self::Procedures => "procedures",
            Self::Temporal => "temporal",
        }
    }
}

/// Configuration for the knowledge elicitor.
#[derive(Debug, Clone)]
pub struct ElicitConfig {
    /// Maximum triples to generate per pass (caps JSON array length).
    pub max_triples_per_pass: usize,
    /// Maximum tokens to generate per inference call.
    pub max_tokens: u32,
    /// Number of self-consistency re-prompt rounds (≥1; 1 = single shot).
    pub self_consistency_rounds: u8,
    /// Minimum fraction of rounds in which a triple must appear to be kept.
    pub min_agreement: f32,
    /// Inference temperature (0.0 = greedy determinism for round 1,
    /// then 0.3 for diversity rounds).
    pub temperature: f32,
    /// Model identifier written into provenance records.
    pub model_id: String,
    /// Maximum number of concepts to include per prompt (avoids token overflow).
    pub max_concepts_per_prompt: usize,
}

impl Default for ElicitConfig {
    fn default() -> Self {
        Self {
            max_triples_per_pass: 30,
            max_tokens: 512,
            self_consistency_rounds: 2,
            min_agreement: 0.5,
            temperature: 0.0,
            model_id: "t5-knowledge-q6k".to_string(),
            max_concepts_per_prompt: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A single triple produced by elicitation before validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ElicitedTriple {
    /// Normalized (lowercase-trimmed) subject label.
    pub subject: String,
    /// Normalized predicate label.
    pub predicate: String,
    /// Normalized object label.
    pub object: String,
    /// How many self-consistency rounds independently produced this triple.
    pub agreement_count: u8,
}

/// Output of a single elicitation batch (one domain, one or more passes).
#[derive(Debug, Clone)]
pub struct ElicitBatch {
    /// Deduplicated, agreement-filtered triples across all requested passes.
    pub triples: Vec<ElicitedTriple>,
    /// Total self-consistency rounds run (for confidence calculation).
    pub total_rounds: u8,
    /// FNV-64 fingerprint of the last prompt issued (for provenance).
    pub prompt_hash: u64,
    /// Model identifier from config.
    pub model_id: String,
}

// ---------------------------------------------------------------------------
// Prompt templates (candle-backend only)
// ---------------------------------------------------------------------------

#[cfg(feature = "candle-backend")]
fn taxonomy_prompt(domain: &str, concepts: &[String]) -> String {
    let truncated: Vec<_> = concepts.iter().take(15).collect();
    format!(
        "List the main categories and subcategories of {domain}.\n\
         Format each fact as a JSON object: {{\"s\": \"subject\", \"p\": \"predicate\", \"o\": \"object\"}}\n\
         Use predicates: is-a, subclass-of, part-of, instance-of.\n\
         Output a JSON array of at most 30 objects. Known concepts: {}\n",
        truncated
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(feature = "candle-backend")]
fn properties_prompt(concepts: &[String], domain: &str) -> String {
    let truncated: Vec<_> = concepts
        .iter()
        .take(5)
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    format!(
        "List all known properties and attributes of {} in the domain of {domain}.\n\
         Format: JSON array of {{\"s\": \"concept\", \"p\": \"predicate\", \"o\": \"value\"}}.\n\
         Use predicates: has-property, characterized-by, measured-in, typical-value, range.\n\
         Output at most 30 objects.\n",
        truncated.join(", ")
    )
}

#[cfg(feature = "candle-backend")]
fn relationships_prompt(concepts: &[String], domain: &str) -> String {
    let truncated: Vec<_> = concepts
        .iter()
        .take(10)
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    format!(
        "How do these concepts relate to each other in {domain}?\n\
         Concepts: {}\n\
         Format: JSON array of {{\"s\": \"concept1\", \"p\": \"relation\", \"o\": \"concept2\"}}.\n\
         Use predicates: causes, enables, prevents, depends-on, precedes, co-occurs-with.\n\
         Output at most 30 objects.\n",
        truncated.join(", ")
    )
}

#[cfg(feature = "candle-backend")]
fn procedures_prompt(concepts: &[String], domain: &str) -> String {
    let truncated: Vec<_> = concepts
        .iter()
        .take(5)
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    format!(
        "Describe the steps, preconditions and effects of performing: {} in {domain}.\n\
         Format: JSON array of {{\"s\": \"action\", \"p\": \"predicate\", \"o\": \"value\"}}.\n\
         Use predicates: has-step, has-precondition, has-effect, followed-by, requires.\n\
         Output at most 30 objects.\n",
        truncated.join(", ")
    )
}

#[cfg(feature = "candle-backend")]
fn temporal_prompt(concepts: &[String], domain: &str) -> String {
    let truncated: Vec<_> = concepts
        .iter()
        .take(5)
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    format!(
        "Describe the temporal sequence of events for {} in {domain}.\n\
         Format: JSON array of {{\"s\": \"event\", \"p\": \"predicate\", \"o\": \"event_or_state\"}}.\n\
         Use predicates: precedes, follows, co-occurs-with, triggers, terminates.\n\
         Output at most 30 objects.\n",
        truncated.join(", ")
    )
}

/// Build a prompt for the given pass, optionally shuffling concept order for
/// diversity rounds (round ≥ 2).
#[cfg(feature = "candle-backend")]
fn build_prompt(pass: ElicitPass, domain: &str, concepts: &[String], round: u8) -> String {
    // Rotate concepts for diversity: each round shifts the starting index.
    let offset = (round as usize).min(concepts.len().saturating_sub(1));
    let rotated: Vec<String> = concepts
        .iter()
        .skip(offset)
        .chain(concepts.iter().take(offset))
        .cloned()
        .collect();

    match pass {
        ElicitPass::Taxonomy => taxonomy_prompt(domain, &rotated),
        ElicitPass::Properties => properties_prompt(&rotated, domain),
        ElicitPass::Relationships => relationships_prompt(&rotated, domain),
        ElicitPass::Procedures => procedures_prompt(&rotated, domain),
        ElicitPass::Temporal => temporal_prompt(&rotated, domain),
    }
}

// ---------------------------------------------------------------------------
// JSON parsing — lenient, recovers partial arrays
// ---------------------------------------------------------------------------

/// A raw JSON triple object from the LLM.
///
/// `pub(crate)` (rather than the original `#[cfg(any(test, feature =
/// "candle-backend"))]`-only visibility): `agent::librarian` (ADR 081 slice 4)
/// reuses this lenient parser for its own LLM extraction seam rather than
/// duplicating it — see that module's `parse_candidate_triples`. The parser
/// has no candle dependency (pure string scanning), so widening its
/// visibility does not pull in anything new.
#[derive(Debug)]
pub(crate) struct RawJsonTriple {
    pub(crate) s: String,
    pub(crate) p: String,
    pub(crate) o: String,
}

/// Parse the LLM output into raw triple objects.
///
/// Lenient: handles trailing commas, partial arrays (truncated generation),
/// and extra whitespace.  Returns whatever valid objects were found.
pub(crate) fn parse_json_triples(text: &str) -> Vec<RawJsonTriple> {
    // Find the outermost `[` to locate the array start.
    let start = match text.find('[') {
        Some(i) => i,
        None => return vec![],
    };
    // Collect everything from `[` onward, stopping at the matching `]` if found.
    let tail = &text[start..];
    let array_text = if let Some(end) = find_matching_bracket(tail) {
        &tail[..=end]
    } else {
        // Truncated output — close the array and try anyway.
        tail
    };

    parse_objects_from_array(array_text)
}

/// Locate the `]` that closes the opening `[` at position 0, respecting nesting.
fn find_matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '[' | '{' => depth += 1,
            ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract individual `{...}` objects from an array string, parsing `s`/`p`/`o` keys.
fn parse_objects_from_array(array: &str) -> Vec<RawJsonTriple> {
    let mut triples = Vec::new();
    let mut i = 0;
    let bytes = array.as_bytes();

    while i < bytes.len() {
        // Find next `{`.
        let Some(start_rel) = array[i..].find('{') else {
            break;
        };
        let start = i + start_rel;
        let obj_tail = &array[start..];
        let Some(end_rel) = find_matching_bracket(obj_tail) else {
            break;
        };
        let obj_str = &obj_tail[..=end_rel];

        if let Some(t) = extract_spo(obj_str) {
            triples.push(t);
        }

        i = start + end_rel + 1;
    }

    triples
}

/// Extract `s`, `p`, `o` string values from a JSON object literal `{...}`.
fn extract_spo(obj: &str) -> Option<RawJsonTriple> {
    let s = extract_string_field(obj, "\"s\"")?;
    let p = extract_string_field(obj, "\"p\"")?;
    let o = extract_string_field(obj, "\"o\"")?;
    if s.is_empty() || p.is_empty() || o.is_empty() {
        return None;
    }
    Some(RawJsonTriple { s, p, o })
}

/// Find the value for a JSON key like `"s"` in a flat object literal.
fn extract_string_field(obj: &str, key: &str) -> Option<String> {
    let key_pos = obj.find(key)?;
    let after_key = &obj[key_pos + key.len()..];
    // Skip whitespace and colon.
    let after_colon = after_key.trim_start_matches([' ', '\t', '\n', '\r', ':']);
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    // Find closing quote (simple: no escape handling here for value).
    let end = inner.find('"')?;
    Some(inner[..end].trim().to_lowercase())
}

// ---------------------------------------------------------------------------
// FNV-64 hash (no external dependency)
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "candle-backend"))]
fn fnv64(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// ---------------------------------------------------------------------------
// 32a — KnowledgeElicitor
// ---------------------------------------------------------------------------

/// Drives multi-pass LLM knowledge elicitation with self-consistency voting.
///
/// The elicitor holds an optional Candle backend reference.  When the backend
/// is absent (feature not compiled, or model file missing) all elicitation
/// calls return empty batches — the bootstrap pipeline degrades gracefully to
/// web-API-only mode.
pub struct KnowledgeElicitor {
    config: ElicitConfig,
    /// Data directory for locating the T5 Knowledge model file.
    data_dir: Option<std::path::PathBuf>,
}

impl KnowledgeElicitor {
    /// Create a new elicitor.
    ///
    /// `data_dir` should point to the workspace data directory so the elicitor
    /// can locate `models/llm/t5-knowledge-q6k.gguf`.  Pass `None` for a
    /// stub that always returns empty results (useful in tests).
    pub fn new(config: ElicitConfig, data_dir: Option<std::path::PathBuf>) -> Self {
        Self { config, data_dir }
    }

    /// Create with default config from the given data directory.
    pub fn from_data_dir(data_dir: std::path::PathBuf) -> Self {
        Self::new(ElicitConfig::default(), Some(data_dir))
    }

    /// Run elicitation for a domain and set of seed concepts.
    ///
    /// Runs the requested `passes` with `self_consistency_rounds` rounds each.
    /// Returns all triples that appear in ≥ `min_agreement` fraction of rounds.
    pub fn elicit(
        &mut self,
        domain: &str,
        seed_concepts: &[String],
        passes: &[ElicitPass],
    ) -> ElicitBatch {
        if seed_concepts.is_empty() || passes.is_empty() {
            return ElicitBatch {
                triples: vec![],
                total_rounds: 0,
                prompt_hash: 0,
                model_id: self.config.model_id.clone(),
            };
        }

        // Without the Candle backend there is no generative model — return empty.
        #[cfg(not(feature = "candle-backend"))]
        {
            let _ = (domain, seed_concepts, passes);
            ElicitBatch {
                triples: vec![],
                total_rounds: 0,
                prompt_hash: 0,
                model_id: self.config.model_id.clone(),
            }
        }

        #[cfg(feature = "candle-backend")]
        {
            // If the model file is absent, degrade gracefully (same as no-backend).
            if !self.is_available() {
                return ElicitBatch {
                    triples: vec![],
                    total_rounds: 0,
                    prompt_hash: 0,
                    model_id: self.config.model_id.clone(),
                };
            }

            let rounds = self.config.self_consistency_rounds.max(1);
            // Track (subject, predicate, object) → agreement_count.
            let mut agreement: HashMap<(String, String, String), u8> = HashMap::new();
            let mut last_prompt_hash: u64 = 0;

            for pass in passes {
                for round in 0..rounds {
                    let prompt = build_prompt(*pass, domain, seed_concepts, round);
                    last_prompt_hash = fnv64(&prompt);

                    let output = self.generate_raw(&prompt);
                    let raw_triples = parse_json_triples(&output);

                    for rt in raw_triples {
                        let key = (rt.s, rt.p, rt.o);
                        *agreement.entry(key).or_insert(0) += 1;
                    }
                }
            }

            let total_rounds = rounds * passes.len() as u8;
            let min_count = ((self.config.min_agreement * rounds as f32).ceil() as u8).max(1);

            let triples: Vec<ElicitedTriple> = agreement
                .into_iter()
                .filter(|(_, count)| *count >= min_count)
                .map(
                    |((subject, predicate, object), agreement_count)| ElicitedTriple {
                        subject,
                        predicate,
                        object,
                        agreement_count,
                    },
                )
                .take(self.config.max_triples_per_pass * passes.len())
                .collect();

            ElicitBatch {
                triples,
                total_rounds,
                prompt_hash: last_prompt_hash,
                model_id: self.config.model_id.clone(),
            }
        } // end #[cfg(feature = "candle-backend")]
    }

    /// Attempt to generate text from the Candle backend.
    ///
    /// Returns an empty string if the backend is unavailable or fails.
    #[cfg(feature = "candle-backend")]
    fn generate_raw(&mut self, prompt: &str) -> String {
        let result = self.generate_candle(prompt);
        match result {
            Ok(text) => {
                tracing::debug!(
                    prompt_len = prompt.len(),
                    output_len = text.len(),
                    "elicitor: Candle generation succeeded"
                );
                text
            }
            Err(e) => {
                tracing::debug!(error = %e, "elicitor: Candle generation failed, returning empty");
                String::new()
            }
        }
    }

    #[cfg(feature = "candle-backend")]
    fn generate_candle(&mut self, prompt: &str) -> ElicitResult<String> {
        use crate::nlu::candle_backend::{CandleBackend, CandleConfig};

        let data_dir = self
            .data_dir
            .as_deref()
            .ok_or_else(|| ElicitError::BackendUnavailable {
                reason: "no data_dir configured".to_string(),
            })?;

        // T5 Knowledge model path (Phase 35 will produce this file).
        // Falls back to the standard Qwen model if T5 is not yet available.
        let t5_path = data_dir.join("models/llm").join("t5-knowledge-q6k.gguf");
        let qwen_path = data_dir
            .join("models/llm")
            .join("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        let tokenizer_path = data_dir.join("models/llm").join("tokenizer.json");

        let model_path = if t5_path.exists() {
            t5_path
        } else if qwen_path.exists() {
            qwen_path
        } else {
            return Err(ElicitError::BackendUnavailable {
                reason: "neither t5-knowledge-q6k.gguf nor qwen2.5-1.5b-instruct-q4_k_m.gguf found"
                    .to_string(),
            });
        };

        let cfg = CandleConfig {
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            ..Default::default()
        };

        let mut backend = CandleBackend::load_with_config(&model_path, &tokenizer_path, cfg)
            .map_err(|e| ElicitError::BackendUnavailable {
                reason: format!("model load: {e}"),
            })?;

        backend
            .generate(prompt, self.config.max_tokens)
            .map_err(|e| ElicitError::GenerationFailed {
                reason: format!("{e}"),
            })
    }

    /// Whether the backend is expected to be available (model file exists).
    pub fn is_available(&self) -> bool {
        let Some(ref dir) = self.data_dir else {
            return false;
        };

        #[cfg(feature = "candle-backend")]
        {
            dir.join("models/llm/t5-knowledge-q6k.gguf").exists()
                || dir
                    .join("models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf")
                    .exists()
        }

        #[cfg(not(feature = "candle-backend"))]
        {
            let _ = dir;
            false
        }
    }
}

// ---------------------------------------------------------------------------
// 32b — Validation pipeline
// ---------------------------------------------------------------------------

/// Why an elicited triple was rejected outright.
#[derive(Debug, Clone)]
pub enum RejectionReason {
    /// Subject, predicate, or object is empty after normalization.
    MalformedTriple,
    /// Subject and object are the same token (self-referential).
    SelfReferential,
    /// Identical (or near-identical) triple already exists in the KG.
    Duplicate {
        existing_subject: SymbolId,
        existing_predicate: SymbolId,
        existing_object: SymbolId,
    },
    /// The triple violates an arity/type constraint.
    ArityViolation { description: String },
    /// The triple contradicts existing knowledge.
    Contradiction { description: String },
}

/// Why an elicited triple was accepted but flagged for human review.
#[derive(Debug, Clone)]
pub enum FlagReason {
    /// Neural bridge coherence below threshold (possible hallucination).
    NeuralBridgeIncoherent { similarity: f32 },
    /// Self-consistency agreement below 0.75 — accepted but uncertain.
    LowSelfConsistency { agreement: f32 },
    /// No web source could later corroborate this triple.
    NoWebCorroboration,
    /// The predicate label was not seen during training — novel.
    NovelPredicate { predicate: String },
}

/// A validated, confidence-scored triple ready for KG ingestion.
#[derive(Debug, Clone)]
pub struct ValidatedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Computed confidence in \[0.3, 0.95\].
    pub final_confidence: f32,
    /// Self-consistency agreement fraction.
    pub agreement: f32,
    /// Optional neural bridge coherence bonus used in confidence.
    pub neural_coherence: Option<f32>,
    /// Whether a web cross-reference later matched this triple.
    pub web_corroborated: bool,
    pub flags: Vec<FlagReason>,
}

/// Outcome of the full validation pipeline.
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub accepted: Vec<ValidatedTriple>,
    pub rejected: Vec<(ElicitedTriple, RejectionReason)>,
    pub flagged_count: usize,
}

/// Context used to compute final confidence.
struct ValidationContext<'a> {
    total_rounds: u8,
    engine: &'a Engine,
    well_known_predicates: &'a [&'a str],
}

/// Multi-layer validation pipeline for elicited triples.
///
/// Stateless — the engine is passed per-call so the validator can be used from
/// contexts that only have `&Engine` (e.g., inside `CurriculumIngestor`).
pub struct ElicitValidator;

impl ElicitValidator {
    /// Create a validator.  Neural bridge integration (Phase 26d) is optional
    /// and not yet wired — the `_bridge` argument is reserved for future use.
    pub fn new(_bridge: Option<()>) -> Self {
        Self
    }

    /// Run the full validation pipeline on a set of elicited triples.
    pub fn validate(
        &self,
        triples: &[ElicitedTriple],
        total_rounds: u8,
        engine: &Engine,
    ) -> ValidationResult {
        let well_known: &[&str] = &[
            "is-a",
            "subclass-of",
            "part-of",
            "instance-of",
            "has-property",
            "characterized-by",
            "measured-in",
            "typical-value",
            "range",
            "causes",
            "enables",
            "prevents",
            "depends-on",
            "precedes",
            "follows",
            "co-occurs-with",
            "triggers",
            "terminates",
            "has-step",
            "has-precondition",
            "has-effect",
            "requires",
            "expand:instance_of",
            "expand:subclass_of",
            "expand:part_of",
            "expand:related_to",
        ];

        let ctx = ValidationContext {
            total_rounds,
            engine,
            well_known_predicates: well_known,
        };

        let mut result = ValidationResult::default();

        for triple in triples {
            match self.validate_one(triple, &ctx) {
                Ok((validated, flags)) => {
                    let flag_count = flags.len();
                    result.flagged_count += flag_count;
                    result.accepted.push(ValidatedTriple {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                        final_confidence: validated,
                        agreement: triple.agreement_count as f32 / (total_rounds.max(1)) as f32,
                        neural_coherence: None,
                        web_corroborated: false,
                        flags,
                    });
                }
                Err(reason) => {
                    result.rejected.push((triple.clone(), reason));
                }
            }
        }

        result
    }

    /// Validate a single triple through all layers.
    ///
    /// Returns `(final_confidence, flags)` on acceptance, `RejectionReason` on rejection.
    fn validate_one(
        &self,
        triple: &ElicitedTriple,
        ctx: &ValidationContext<'_>,
    ) -> Result<(f32, Vec<FlagReason>), RejectionReason> {
        // Layer 1: Syntactic
        self.check_syntactic(triple)?;

        // Layer 2: Self-referential
        if triple.subject == triple.object {
            return Err(RejectionReason::SelfReferential);
        }

        // Layer 3: Duplicate (exact label match against existing KG)
        if let Some((s, p, o)) = self.check_duplicate(triple, ctx.engine) {
            return Err(RejectionReason::Duplicate {
                existing_subject: s,
                existing_predicate: p,
                existing_object: o,
            });
        }

        // Layer 4: Arity constraints (best-effort — skipped if symbols not resolvable)
        if let Some(desc) = self.check_arity(triple, ctx.engine) {
            return Err(RejectionReason::ArityViolation { description: desc });
        }

        // Layer 5: Contradiction detection (best-effort)
        if let Some(desc) = self.check_contradiction(triple, ctx.engine) {
            return Err(RejectionReason::Contradiction { description: desc });
        }

        // Layer 6: Confidence + flags
        let mut flags = Vec::new();
        let agreement = triple.agreement_count as f32 / (ctx.total_rounds.max(1)) as f32;

        if agreement < 0.75 {
            flags.push(FlagReason::LowSelfConsistency { agreement });
        }

        if !ctx
            .well_known_predicates
            .contains(&triple.predicate.as_str())
        {
            flags.push(FlagReason::NovelPredicate {
                predicate: triple.predicate.clone(),
            });
        }

        let confidence = compute_confidence(triple, ctx.total_rounds, None);

        Ok((confidence, flags))
    }

    fn check_syntactic(&self, triple: &ElicitedTriple) -> Result<(), RejectionReason> {
        if triple.subject.trim().is_empty()
            || triple.predicate.trim().is_empty()
            || triple.object.trim().is_empty()
        {
            return Err(RejectionReason::MalformedTriple);
        }
        Ok(())
    }

    /// Check if an identical (s-label, p-label, o-label) triple already exists.
    fn check_duplicate(
        &self,
        triple: &ElicitedTriple,
        engine: &Engine,
    ) -> Option<(SymbolId, SymbolId, SymbolId)> {
        let s_id = engine.resolve_symbol(&triple.subject).ok()?;
        let p_id = engine.resolve_symbol(&triple.predicate).ok()?;
        let o_id = engine.resolve_symbol(&triple.object).ok()?;

        // Walk outgoing triples from s and look for (p, o) match.
        let outgoing = engine.triples_from(s_id);
        for t in outgoing {
            if t.predicate == p_id && t.object == o_id {
                return Some((s_id, p_id, o_id));
            }
        }
        None
    }

    /// Check arity constraints using the engine's stored constraint registry.
    fn check_arity(&self, triple: &ElicitedTriple, engine: &Engine) -> Option<String> {
        let s_id = engine.resolve_symbol(&triple.subject).ok()?;
        let p_id = engine.resolve_symbol(&triple.predicate).ok()?;
        let o_id = engine.resolve_symbol(&triple.object).ok()?;
        let candidate = Triple::new(s_id, p_id, o_id);

        let registry = engine.constraint_reg();
        let violations = registry.check_triple(&candidate, engine);
        if violations.is_empty() {
            return None;
        }

        let desc = violations
            .iter()
            .map(|v| match v {
                ConstraintViolation::Arity {
                    expected, actual, ..
                } => {
                    format!("arity expects {expected} got {actual}")
                }
                ConstraintViolation::Type { arg_position, .. } => {
                    format!("type mismatch at arg {arg_position}")
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(desc)
    }

    /// Check for contradictions against existing KG knowledge using the engine's registries.
    fn check_contradiction(&self, triple: &ElicitedTriple, engine: &Engine) -> Option<String> {
        let s_id = engine.resolve_symbol(&triple.subject).ok()?;
        let p_id = engine.resolve_symbol(&triple.predicate).ok()?;
        let o_id = engine.resolve_symbol(&triple.object).ok()?;
        let candidate = Triple::new(s_id, p_id, o_id);

        let contradictions = engine.check_contradictions(&candidate);
        if contradictions.is_empty() {
            return None;
        }

        let desc = contradictions
            .iter()
            .map(|c| format!("{:?}", c.kind))
            .collect::<Vec<_>>()
            .join("; ");
        Some(desc)
    }
}

// ---------------------------------------------------------------------------
// Confidence computation
// ---------------------------------------------------------------------------

/// Compute final confidence for a validated elicited triple.
///
/// `base = 0.55` for LLM-only facts.  Self-consistency and neural bridge
/// bonuses shift up; low agreement shifts down.  Clamped to \[0.3, 0.95\].
pub fn compute_confidence(
    triple: &ElicitedTriple,
    total_rounds: u8,
    neural_coherence: Option<f32>,
) -> f32 {
    let base = 0.55_f32;
    let mut conf = base;

    let rounds = total_rounds.max(1) as f32;
    let agreement = triple.agreement_count as f32 / rounds;
    // +0.10 for perfect agreement, –0.10 for single-round (agreement = 0.5).
    conf += (agreement - 0.5) * 0.2;

    if let Some(coherence) = neural_coherence {
        conf += (coherence - 0.5) * 0.15;
    }

    conf.clamp(0.3, 0.95)
}

// ---------------------------------------------------------------------------
// KG ingestion helpers
// ---------------------------------------------------------------------------

/// Ingest a validated batch of triples into the engine with `KnowledgeElicitation` provenance.
///
/// Accepts a plain `&Engine` so it can be called from any context.
/// Returns the number of triples successfully inserted.
pub fn ingest_validated(
    validated: &ValidationResult,
    engine: &Engine,
    batch: &ElicitBatch,
    pass: ElicitPass,
) -> usize {
    let mut inserted = 0usize;

    for vt in &validated.accepted {
        let s_id = match engine.resolve_or_create_entity(&vt.subject) {
            Ok(id) => id,
            Err(e) => {
                tracing::debug!(label = %vt.subject, error = %e, "elicitor: failed to resolve subject");
                continue;
            }
        };
        let p_id = match engine.resolve_or_create_relation(&vt.predicate) {
            Ok(id) => id,
            Err(e) => {
                tracing::debug!(label = %vt.predicate, error = %e, "elicitor: failed to resolve predicate");
                continue;
            }
        };
        let o_id = match engine.resolve_or_create_entity(&vt.object) {
            Ok(id) => id,
            Err(e) => {
                tracing::debug!(label = %vt.object, error = %e, "elicitor: failed to resolve object");
                continue;
            }
        };

        let triple = Triple::new(s_id, p_id, o_id).with_confidence(vt.final_confidence);
        match engine.add_triple(&triple) {
            Ok(_) => {
                let mut record = ProvenanceRecord::new(
                    s_id,
                    DerivationKind::KnowledgeElicitation {
                        model_id: batch.model_id.clone(),
                        pass: pass.name().to_string(),
                        prompt_hash: batch.prompt_hash,
                        self_consistency: vt.agreement,
                        neural_coherence: vt.neural_coherence,
                        web_corroborated: vt.web_corroborated,
                    },
                )
                .with_confidence(vt.final_confidence)
                .with_sources(vec![s_id, p_id, o_id]);

                if let Err(e) = engine.store_provenance(&mut record) {
                    tracing::debug!(error = %e, "elicitor: failed to store provenance");
                }
                inserted += 1;
            }
            Err(e) => {
                tracing::debug!(error = %e, "elicitor: add_triple failed");
            }
        }
    }

    inserted
}

/// Apply web-corroboration boost: for any validated triple whose (s, p, o)
/// labels appear in the provided web-extracted label set, increment confidence
/// and mark `web_corroborated = true`.
///
/// `web_labels` is a set of `(subject_label, predicate_label, object_label)` tuples
/// already found by Wikidata / Wikipedia / ConceptNet.
pub fn apply_web_corroboration(
    validated: &mut ValidationResult,
    web_labels: &std::collections::HashSet<(String, String, String)>,
) {
    for vt in &mut validated.accepted {
        let key = (vt.subject.clone(), vt.predicate.clone(), vt.object.clone());
        if web_labels.contains(&key) {
            vt.final_confidence = (vt.final_confidence + 0.2).min(0.95);
            vt.web_corroborated = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Domain-level entry points for orchestrator integration (32c)
// ---------------------------------------------------------------------------

/// Run taxonomy + relationships elicitation for a domain expansion context.
///
/// Called from `DomainExpander` or `BootstrapOrchestrator` after web-API expansion.
/// Returns `None` when the backend is unavailable so callers can degrade gracefully.
pub fn elicit_for_expansion(
    domain: &str,
    seed_concepts: &[String],
    engine: &Engine,
    data_dir: Option<std::path::PathBuf>,
) -> Option<usize> {
    let config = ElicitConfig::default();
    let mut elicitor = KnowledgeElicitor::new(config, data_dir);
    if !elicitor.is_available() {
        return None;
    }

    let batch = elicitor.elicit(
        domain,
        seed_concepts,
        &[ElicitPass::Taxonomy, ElicitPass::Relationships],
    );
    if batch.triples.is_empty() {
        return Some(0);
    }

    let validator = ElicitValidator::new(None);
    let validated = validator.validate(&batch.triples, batch.total_rounds, engine);

    let count = ingest_validated(&validated, engine, &batch, ElicitPass::Taxonomy);
    tracing::info!(
        domain,
        accepted = validated.accepted.len(),
        rejected = validated.rejected.len(),
        inserted = count,
        "elicitor: expansion elicitation complete"
    );
    Some(count)
}

/// Run properties + relationships elicitation for curriculum ingestion (32c).
///
/// Called per-concept during `CurriculumIngestor` runs.
/// Returns `None` when the backend is unavailable so callers can degrade gracefully.
pub fn elicit_for_concept(
    concept: &str,
    domain: &str,
    engine: &Engine,
    data_dir: Option<std::path::PathBuf>,
) -> Option<usize> {
    let config = ElicitConfig::default();
    let mut elicitor = KnowledgeElicitor::new(config, data_dir);
    if !elicitor.is_available() {
        return None;
    }

    let concepts = vec![concept.to_string()];
    let batch = elicitor.elicit(
        domain,
        &concepts,
        &[ElicitPass::Properties, ElicitPass::Relationships],
    );
    if batch.triples.is_empty() {
        return Some(0);
    }

    let validator = ElicitValidator::new(None);
    let validated = validator.validate(&batch.triples, batch.total_rounds, engine);

    let count = ingest_validated(&validated, engine, &batch, ElicitPass::Properties);
    tracing::debug!(
        concept,
        accepted = validated.accepted.len(),
        rejected = validated.rejected.len(),
        inserted = count,
        "elicitor: concept elicitation complete"
    );
    Some(count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension(64),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn parse_well_formed_json() {
        let json = r#"[
            {"s": "dog", "p": "is-a", "o": "animal"},
            {"s": "cat", "p": "is-a", "o": "animal"}
        ]"#;
        let triples = parse_json_triples(json);
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].s, "dog");
        assert_eq!(triples[0].p, "is-a");
        assert_eq!(triples[0].o, "animal");
    }

    #[test]
    fn parse_trailing_comma() {
        let json = r#"[{"s": "a", "p": "b", "o": "c"},]"#;
        let triples = parse_json_triples(json);
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn parse_truncated_array() {
        // Simulates LLM cutting off in the middle.
        let json = r#"[{"s": "stellar evolution", "p": "is-a", "o": "process"}, {"s": "ma"#;
        let triples = parse_json_triples(json);
        // At least the complete object should be recovered.
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn parse_no_array() {
        let json = "I cannot generate a JSON array.";
        let triples = parse_json_triples(json);
        assert!(triples.is_empty());
    }

    #[test]
    fn elicitor_no_backend_returns_empty() {
        let mut elicitor = KnowledgeElicitor::new(ElicitConfig::default(), None);
        let batch = elicitor.elicit("astronomy", &["star".into()], &[ElicitPass::Taxonomy]);
        assert!(batch.triples.is_empty());
        assert_eq!(batch.total_rounds, 0);
    }

    #[test]
    fn confidence_perfect_agreement() {
        let triple = ElicitedTriple {
            subject: "a".into(),
            predicate: "is-a".into(),
            object: "b".into(),
            agreement_count: 3,
        };
        let conf = compute_confidence(&triple, 3, None);
        // base 0.55 + (1.0 - 0.5) * 0.2 = 0.55 + 0.10 = 0.65
        assert!((conf - 0.65).abs() < 0.01, "conf={conf}");
    }

    #[test]
    fn confidence_single_round() {
        let triple = ElicitedTriple {
            subject: "a".into(),
            predicate: "is-a".into(),
            object: "b".into(),
            agreement_count: 1,
        };
        let conf = compute_confidence(&triple, 1, None);
        // base 0.55 + (1.0 - 0.5) * 0.2 = 0.65
        assert!((conf - 0.65).abs() < 0.01, "conf={conf}");
    }

    #[test]
    fn validation_rejects_malformed() {
        let engine = test_engine();
        let validator = ElicitValidator::new(None);
        let triples = vec![ElicitedTriple {
            subject: "".into(),
            predicate: "is-a".into(),
            object: "animal".into(),
            agreement_count: 1,
        }];
        let result = validator.validate(&triples, 1, &engine);
        assert_eq!(result.accepted.len(), 0);
        assert_eq!(result.rejected.len(), 1);
        assert!(matches!(
            result.rejected[0].1,
            RejectionReason::MalformedTriple
        ));
    }

    #[test]
    fn validation_rejects_self_referential() {
        let engine = test_engine();
        let validator = ElicitValidator::new(None);
        let triples = vec![ElicitedTriple {
            subject: "star".into(),
            predicate: "is-a".into(),
            object: "star".into(),
            agreement_count: 1,
        }];
        let result = validator.validate(&triples, 1, &engine);
        assert_eq!(result.accepted.len(), 0);
        assert!(matches!(
            result.rejected[0].1,
            RejectionReason::SelfReferential
        ));
    }

    #[test]
    fn validation_accepts_valid_triple() {
        let engine = test_engine();
        let validator = ElicitValidator::new(None);
        let triples = vec![ElicitedTriple {
            subject: "stellar evolution".into(),
            predicate: "is-a".into(),
            object: "astrophysical process".into(),
            agreement_count: 2,
        }];
        let result = validator.validate(&triples, 2, &engine);
        assert_eq!(result.accepted.len(), 1);
        assert!(result.rejected.is_empty());
    }

    #[test]
    fn fnv64_is_deterministic() {
        let h1 = fnv64("hello world");
        let h2 = fnv64("hello world");
        assert_eq!(h1, h2);

        let h3 = fnv64("different");
        assert_ne!(h1, h3);
    }
}
