//! Daily Librarian projection — ADR 081 slice 4
//! (`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`, work-item 4).
//!
//! The sleep-cycle harvest phase (`agent::harvest`, ADR 091) turns raw
//! `kind="turn"` WAL rows into lightweight `episode:turn:{seq}` indices using
//! crude keyword→symbol linking — deliberately dumb, so the transcript is
//! *findable* without waiting for an LLM. This module is the pass ADR 091
//! named as its designed successor: it re-reads the same WAL tail (plus, if
//! supplied, note bodies) through an **LLM extraction seam** and asserts
//! **typed** triples against the akh's in-force schema (`schema::compile`'s
//! native-triple vocabulary — `is-a` / `onto:property`), reusing the
//! Phase-32 `KnowledgeElicitor`'s validated pattern (lenient JSON-triple
//! parsing, provenance-tracked ingestion) rather than reinventing it.
//!
//! ## Two watermarks, not one
//!
//! The Librarian keeps its **own** watermark (`akhomed_akh_state.librarian_watermark`),
//! independent of harvest's `harvest_watermark`. The two passes serve
//! different purposes and progress on different cadences: harvest is
//! always-on, cheap, and runs every sleep tick; the Librarian is an
//! off-by-default (`AKH_LIBRARIAN=1`), real LLM-cost pass that runs roughly
//! daily. Neither may gate or clobber the other — see the migration doc
//! comment (`crates/akhomed-store/src/migration/m20260711_000001_librarian_watermark.rs`).
//!
//! ## Module shape
//!
//! - [`LibrarianModel`] — the LLM seam (`extract`). [`FakeLibrarianModel`] is
//!   a deterministic double for tests; [`OllamaLibrarianModel`] rides the
//!   existing [`super::model_manager::AgentModelManager`] / Ollama plumbing
//!   (config-selected via `AKH_LIBRARIAN_MODEL`, default `gpt-oss:120b` per
//!   ADR 081 decision 10).
//! - [`LibrarianWatermarkStore`] — where the watermark lives. [`InMemoryWatermark`]
//!   is the engine-local fallback (and what unit tests inject);
//!   [`PgLibrarianWatermark`] (postgres feature) is the real column, mirroring
//!   `harvest_watermark`'s precedent exactly.
//! - [`run_librarian_pass`] — the pure orchestration: engine + model +
//!   watermark store + already-gathered `[LibrarianSource]`. No Postgres
//!   dependency, so it is directly unit-testable.
//! - [`run_librarian_pass_over_wal`] (postgres feature) — the thin DB-gathering
//!   wrapper daemon.rs calls: reads the un-harvested `turn` tail + optional
//!   notes, then calls [`run_librarian_pass`].
//!
//! ## Validation + frontier (ADR 081 decision 7: writes are never blocked)
//!
//! A candidate triple's predicate is checked against the installed schema
//! digest ([`schema_digest`]): `is-a` to a known type, or a property name
//! declared by *some* installed type, lands as a **typed** triple; anything
//! else is still asserted — via `schema::project::FRONTIER_PREFIX` (the same
//! `note:frontier/{predicate}` vocabulary Slice 1's mechanical projector
//! uses) — never dropped. [`frontier_terms`] tallies frontier predicates
//! across the whole KG for the promotion pass (slice 5) to consume later.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::schema::compile::{PRED_IS_A, PRED_PROPERTY, TYPE_MARKER};
use crate::schema::project::FRONTIER_PREFIX;
use crate::symbol::SymbolId;

use super::error::{AgentError, AgentResult};

// ---------------------------------------------------------------------------
// LLM seam
// ---------------------------------------------------------------------------

/// One candidate fact the [`LibrarianModel`] proposes extracting from a
/// source's text. Labels only (no `SymbolId`s) — resolution against the KG
/// happens after validation, in [`assert_candidate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A digest of one installed schema type, for prompting the model with what
/// vocabulary is already in force (mirrors what `schema::project::project_note`
/// checks candidates against, but schema-wide rather than per-note).
#[derive(Debug, Clone)]
pub struct SchemaTypeDigest {
    pub name: String,
    pub properties: Vec<String>,
}

/// Input to one [`LibrarianModel::extract`] call.
#[derive(Debug, Clone)]
pub struct ExtractionRequest {
    /// Stable id for provenance + tally lookups (`"turn:{seq}"` / `"note:{id}"`).
    pub source_id: String,
    /// The raw material to extract from.
    pub text: String,
    /// The in-force schema, so the model can prefer known types/properties.
    pub schema_types: Vec<SchemaTypeDigest>,
}

/// Errors from a [`LibrarianModel`] call. Miette-diagnosable per CLAUDE.md.
#[derive(Debug, Error, Diagnostic)]
pub enum LibrarianModelError {
    #[error("librarian model backend unavailable: {reason}")]
    #[diagnostic(
        code(akh::librarian::backend_unavailable),
        help("Start Ollama (`ollama serve`) and ensure AKH_LIBRARIAN_MODEL is pulled.")
    )]
    Unavailable { reason: String },

    #[error("librarian model reply could not be parsed as JSON triples: {reason}")]
    #[diagnostic(
        code(akh::librarian::malformed_reply),
        help(
            "The model did not reply with a `[{{\"s\":..,\"p\":..,\"o\":..}}, ...]` JSON array. \
             This source is retried on the next pass; the watermark is not advanced past it."
        )
    )]
    MalformedReply { reason: String },

    #[error("librarian model generation failed: {reason}")]
    #[diagnostic(
        code(akh::librarian::generation_failed),
        help("The backend was reached but the generation call itself failed — check its logs.")
    )]
    GenerationFailed { reason: String },
}

/// Convenience alias.
pub type LibrarianModelResult<T> = std::result::Result<T, LibrarianModelError>;

/// Input to one [`LibrarianModel::propose_promotion`] call (ADR 081 Slice 5,
/// `agent::promotion`): a frontier predicate that has already crossed
/// [`super::promotion::promotion_candidates`]'s mechanical recurrence +
/// value-shape thresholds, with a sample of its observed usage.
#[derive(Debug, Clone)]
pub struct PromotionRequest {
    /// The frontier predicate (unprefixed — no `note:frontier/`).
    pub predicate: String,
    /// How many distinct subjects use this predicate.
    pub distinct_subjects: usize,
    /// A capped sample of subject labels observed with this predicate.
    pub example_subjects: Vec<String>,
    /// A capped sample of object labels observed with this predicate.
    pub example_values: Vec<String>,
    /// The in-force schema, so the model can name a real type/alias target
    /// instead of guessing blind.
    pub schema_types: Vec<SchemaTypeDigest>,
}

/// The model's advice on how a recurring frontier predicate should join the
/// typed core — never applied automatically (ADR 081 decision 8: admission
/// always requires an explicit human confirm). See
/// `agent::promotion::run_promotion_pass` / `promote_proposal`.
#[derive(Debug, Clone)]
pub struct PromotionAdvice {
    /// An already-installed type this property should extend, or `None` for
    /// a new standalone type.
    pub proposed_type: Option<String>,
    /// The property name to mint (usually the frontier predicate itself,
    /// unless a clearer schema.org-style name fits better).
    pub proposed_property_name: String,
    /// When set, this predicate is a synonym of an existing core
    /// type/property name — promotion should alias onto it instead of
    /// minting anything new (R5: keep the core small).
    pub alias_of: Option<String>,
    /// One-sentence justification, kept with the stored proposal for the
    /// human confirming it.
    pub rationale: String,
}

/// The Librarian's LLM extraction seam: given a source's text and the
/// in-force schema, propose candidate typed triples. Implementations must
/// NOT decide what is typed vs. frontier — that validation happens uniformly
/// in [`assert_candidate`] regardless of which model produced the candidate.
pub trait LibrarianModel: Send + Sync {
    /// Extract candidates from `request`. An empty `Vec` is a legitimate
    /// answer (the source had nothing worth extracting); an `Err` means the
    /// model/backend itself failed — the poison-safe path (the caller does
    /// not advance the watermark past a source that errors here).
    fn extract(&self, request: &ExtractionRequest) -> LibrarianModelResult<Vec<CandidateTriple>>;

    /// Advise on promoting a recurring frontier predicate (ADR 081 Slice 5).
    /// Default: propose the predicate verbatim as a new standalone
    /// property, no alias — a safe, LLM-free fallback for implementations
    /// that don't override it (nothing is applied without a human confirm
    /// either way, so a trivial default cannot corrupt the core).
    fn propose_promotion(
        &self,
        request: &PromotionRequest,
    ) -> LibrarianModelResult<PromotionAdvice> {
        Ok(PromotionAdvice {
            proposed_type: None,
            proposed_property_name: request.predicate.clone(),
            alias_of: None,
            rationale: "default advisor: no LLM configured — proposing the frontier predicate \
                        verbatim as a new standalone property"
                .to_string(),
        })
    }

    /// Short label recorded in provenance / logs (e.g. `"fake"`, an Ollama tag).
    fn label(&self) -> &str;
}

/// Deterministic test double: returns pre-registered candidates for a given
/// `source_id`, or can be told to fail on one — used to exercise the
/// poison-safe (watermark-not-advanced) path without a live model. Shipped
/// as a real type (not `cfg(test)`-only) so it also demonstrates the seam to
/// callers outside this crate.
#[derive(Debug, Default)]
pub struct FakeLibrarianModel {
    canned: std::collections::HashMap<String, Vec<CandidateTriple>>,
    fail_sources: std::collections::HashSet<String>,
    /// Canned [`PromotionAdvice`] replies, keyed by [`PromotionRequest::predicate`]
    /// (ADR 081 Slice 5 — `agent::promotion` tests inject these the same way
    /// [`Self::with_extraction`] injects triple extractions).
    canned_proposals: std::collections::HashMap<String, PromotionAdvice>,
}

impl FakeLibrarianModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the candidates to return for `source_id` (builder style).
    pub fn with_extraction(
        mut self,
        source_id: impl Into<String>,
        triples: Vec<CandidateTriple>,
    ) -> Self {
        self.canned.insert(source_id.into(), triples);
        self
    }

    /// Make `extract` return [`LibrarianModelError::MalformedReply`] for this
    /// `source_id` (builder style) — the poison-safe test path. Also honored
    /// by [`LibrarianModel::propose_promotion`] when `source_id` names a
    /// frontier predicate (the two seams share one failure-injection set —
    /// there is only ever one reason a fake model fails: "configured to").
    pub fn failing_on(mut self, source_id: impl Into<String>) -> Self {
        self.fail_sources.insert(source_id.into());
        self
    }

    /// Register the [`PromotionAdvice`] to return for a given frontier
    /// `predicate` (builder style) — ADR 081 Slice 5's canned-proposals
    /// extension. A predicate with no registered advice falls back to the
    /// trait's default (verbatim standalone property, no alias).
    pub fn with_proposal(mut self, predicate: impl Into<String>, advice: PromotionAdvice) -> Self {
        self.canned_proposals.insert(predicate.into(), advice);
        self
    }
}

impl LibrarianModel for FakeLibrarianModel {
    fn extract(&self, request: &ExtractionRequest) -> LibrarianModelResult<Vec<CandidateTriple>> {
        if self.fail_sources.contains(&request.source_id) {
            return Err(LibrarianModelError::MalformedReply {
                reason: format!("fake model configured to fail on \"{}\"", request.source_id),
            });
        }
        Ok(self
            .canned
            .get(&request.source_id)
            .cloned()
            .unwrap_or_default())
    }

    fn propose_promotion(
        &self,
        request: &PromotionRequest,
    ) -> LibrarianModelResult<PromotionAdvice> {
        if self.fail_sources.contains(&request.predicate) {
            return Err(LibrarianModelError::MalformedReply {
                reason: format!(
                    "fake model configured to fail on \"{}\"",
                    request.predicate
                ),
            });
        }
        match self.canned_proposals.get(&request.predicate) {
            Some(advice) => Ok(advice.clone()),
            None => Ok(PromotionAdvice {
                proposed_type: None,
                proposed_property_name: request.predicate.clone(),
                alias_of: None,
                rationale: "fake model: no canned proposal registered for this predicate".into(),
            }),
        }
    }

    fn label(&self) -> &str {
        "fake"
    }
}

/// Agent id the Ollama-backed model registers itself under in
/// [`super::model_manager::AgentModelManager`].
const LIBRARIAN_AGENT_ID: &str = "librarian";

/// Default Ollama model tag when `AKH_LIBRARIAN_MODEL` is unset (ADR 081
/// decision 10: gpt-oss 120B is the daily Librarian).
pub const DEFAULT_LIBRARIAN_MODEL_TAG: &str = "gpt-oss:120b";

/// The real LLM backend: rides the existing per-akh Ollama plumbing (ADR
/// 070's selectable-backing precedent — this is a second, purpose-built
/// `AgentModelManager` instance rather than sharing the akh's conversational
/// one, so a slow/failing Librarian call can never contend with live chat
/// turns). Config-selected via `AKH_LIBRARIAN_MODEL` (see [`Self::from_env`]).
///
/// Correctness against a real model is **not** covered by the automated
/// suite (no live model in CI) — see the `#[ignore]`d integration test at the
/// bottom of this module.
pub struct OllamaLibrarianModel {
    mgr: super::model_manager::AgentModelManager,
    model_tag: String,
}

impl OllamaLibrarianModel {
    /// Build a Librarian model backed by Ollama, loading `model_tag`.
    pub fn new(model_tag: impl Into<String>) -> Self {
        let model_tag = model_tag.into();
        let ollama = super::ollama::OllamaConfig {
            model: model_tag.clone(),
            ..Default::default()
        };
        let mgr = super::model_manager::AgentModelManager::new(
            super::model_manager::ModelManagerConfig {
                ollama,
                ..Default::default()
            },
        );
        // Librarian has no tools - it's a specialized extraction model
        mgr.load_agent(LIBRARIAN_AGENT_ID, LIBRARIAN_AGENT_ID, None, &[]);
        Self { mgr, model_tag }
    }

    /// Read the model tag from `AKH_LIBRARIAN_MODEL`, defaulting to
    /// [`DEFAULT_LIBRARIAN_MODEL_TAG`].
    pub fn from_env() -> Self {
        let tag = std::env::var("AKH_LIBRARIAN_MODEL")
            .unwrap_or_else(|_| DEFAULT_LIBRARIAN_MODEL_TAG.to_string());
        Self::new(tag)
    }
}

impl LibrarianModel for OllamaLibrarianModel {
    fn extract(&self, request: &ExtractionRequest) -> LibrarianModelResult<Vec<CandidateTriple>> {
        let prompt = build_extraction_prompt(request);
        let reply = self
            .mgr
            .generate(LIBRARIAN_AGENT_ID, &prompt)
            .map_err(|e| LibrarianModelError::GenerationFailed {
                reason: e.to_string(),
            })?;
        parse_candidate_triples(&reply).ok_or_else(|| LibrarianModelError::MalformedReply {
            reason: format!(
                "no `[{{\"s\":..}}]` JSON array found in a {}-byte reply",
                reply.len()
            ),
        })
    }

    fn propose_promotion(
        &self,
        request: &PromotionRequest,
    ) -> LibrarianModelResult<PromotionAdvice> {
        let prompt = build_promotion_prompt(request);
        let reply = self
            .mgr
            .generate(LIBRARIAN_AGENT_ID, &prompt)
            .map_err(|e| LibrarianModelError::GenerationFailed {
                reason: e.to_string(),
            })?;
        parse_promotion_advice(&reply).ok_or_else(|| LibrarianModelError::MalformedReply {
            reason: format!(
                "no `{{\"proposed_property_name\":..}}` JSON object found in a {}-byte reply",
                reply.len()
            ),
        })
    }

    fn label(&self) -> &str {
        &self.model_tag
    }
}

/// Build the extraction prompt: the in-force schema digest + the source text,
/// asking for a JSON triple array (same wire format as
/// `bootstrap::knowledge_elicit`'s passes, reused via [`parse_candidate_triples`]).
fn build_extraction_prompt(request: &ExtractionRequest) -> String {
    let mut schema_block = String::new();
    for t in &request.schema_types {
        schema_block.push_str("- ");
        schema_block.push_str(&t.name);
        if !t.properties.is_empty() {
            schema_block.push_str(" (properties: ");
            schema_block.push_str(&t.properties.join(", "));
            schema_block.push(')');
        }
        schema_block.push('\n');
    }
    if schema_block.is_empty() {
        schema_block.push_str("(no types installed yet)\n");
    }

    format!(
        "Extract factual triples from the text below. Prefer the installed schema types and \
         properties listed when a fact fits one — use `is-a` to assert a type, and a listed \
         property name directly as the predicate for an attribute. If a fact does not fit any \
         installed type/property, still emit it with whatever predicate best names the \
         relation — it will be filed as frontier knowledge, not dropped.\n\n\
         Installed schema:\n{schema_block}\n\
         Format each fact as a JSON object: {{\"s\": \"subject\", \"p\": \"predicate\", \"o\": \"object\"}}\n\
         Output only a JSON array of such objects (at most 50), nothing else.\n\n\
         Source ({source}):\n{text}\n",
        source = request.source_id,
        text = request.text,
    )
}

/// Parse a model reply into candidates, reusing
/// `bootstrap::knowledge_elicit::parse_json_triples`'s lenient JSON-array
/// scanner (trailing commas, truncated generation) rather than duplicating
/// it. `None` when the reply contains no `[` at all — treated as a malformed
/// reply, not a legitimate empty extraction (a well-behaved model always
/// emits at least `[]`).
fn parse_candidate_triples(text: &str) -> Option<Vec<CandidateTriple>> {
    if !text.contains('[') {
        return None;
    }
    Some(
        crate::bootstrap::knowledge_elicit::parse_json_triples(text)
            .into_iter()
            .map(|t| CandidateTriple {
                subject: t.s,
                predicate: t.p,
                object: t.o,
            })
            .collect(),
    )
}

/// Build the promotion-advice prompt (ADR 081 Slice 5): the in-force schema
/// digest + a sample of one recurring frontier predicate's usage, asking for
/// a single JSON object naming/typing/aliasing it.
fn build_promotion_prompt(request: &PromotionRequest) -> String {
    let mut schema_block = String::new();
    for t in &request.schema_types {
        schema_block.push_str("- ");
        schema_block.push_str(&t.name);
        if !t.properties.is_empty() {
            schema_block.push_str(" (properties: ");
            schema_block.push_str(&t.properties.join(", "));
            schema_block.push(')');
        }
        schema_block.push('\n');
    }
    if schema_block.is_empty() {
        schema_block.push_str("(no types installed yet)\n");
    }

    format!(
        "A frontier (not-yet-typed) predicate has recurred across {n} distinct subjects in this \
         akh's knowledge graph. Decide how it should join the typed core.\n\n\
         Installed schema:\n{schema_block}\n\
         Frontier predicate: \"{predicate}\"\n\
         Example subjects: {subjects}\n\
         Example values: {values}\n\n\
         Reply with exactly one JSON object, nothing else:\n\
         {{\"proposed_type\": \"<an installed type name this property should extend, or null for \
         a new standalone type>\", \"proposed_property_name\": \"<the property name to mint — \
         usually the frontier predicate itself, unless a clearer schema.org-style name fits \
         better>\", \"alias_of\": \"<an existing installed property or type name this predicate \
         is a synonym of, or null>\", \"rationale\": \"<one sentence>\"}}\n",
        n = request.distinct_subjects,
        predicate = request.predicate,
        subjects = request.example_subjects.join(", "),
        values = request.example_values.join(", "),
    )
}

/// Parse a model reply into [`PromotionAdvice`] — lenient like
/// [`parse_candidate_triples`]: scans for the first JSON object in the reply
/// (via [`serde_json::Deserializer`]'s streaming parser) rather than
/// requiring the reply to be *only* JSON, so a chatty model prefacing its
/// answer with prose still parses. `None` when no JSON object with a
/// `proposed_property_name` string field is found.
fn parse_promotion_advice(text: &str) -> Option<PromotionAdvice> {
    let idx = text.find('{')?;
    let value = serde_json::Deserializer::from_str(&text[idx..])
        .into_iter::<serde_json::Value>()
        .next()?
        .ok()?;
    let obj = value.as_object()?;
    let proposed_property_name = obj.get("proposed_property_name")?.as_str()?.to_string();
    let proposed_type = obj
        .get("proposed_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let alias_of = obj
        .get("alias_of")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let rationale = obj
        .get("rationale")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(PromotionAdvice {
        proposed_type,
        proposed_property_name,
        alias_of,
        rationale,
    })
}

// ---------------------------------------------------------------------------
// Material to extract from
// ---------------------------------------------------------------------------

/// One piece of captured material to run extraction over.
#[derive(Debug, Clone)]
pub struct LibrarianSource {
    /// Stable id for provenance + tally lookups (`"turn:{seq}"` / `"note:{id}"`).
    pub source_id: String,
    /// The raw text.
    pub text: String,
    /// The WAL seq this material came from, when it is turn-sourced — the
    /// watermark unit. `None` for note-sourced material: notes do not
    /// participate in the seq-based watermark, so the caller is responsible
    /// for only passing not-yet-processed notes.
    pub seq: Option<i64>,
}

/// Convert a loaded `turn` WAL row into a [`LibrarianSource`]. Always
/// returns `Some` (unlike `agent::harvest::mint_turn_episode`, which skips
/// malformed payloads outright) — an empty-text source is still carried
/// through so its `seq` counts toward the watermark advance in
/// [`run_librarian_pass`]; [`extract_and_assert_source`] skips the model
/// call for empty text without spending an LLM round-trip.
#[cfg(feature = "postgres")]
fn turn_row_to_source(row: &akhomed_store::TurnRow) -> LibrarianSource {
    let user_prompt = row
        .payload
        .get("user_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let agent_answer = row
        .payload
        .get("agent_answer")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    LibrarianSource {
        source_id: format!("turn:{}", row.seq),
        text: format!("{user_prompt}\n{agent_answer}").trim().to_string(),
        seq: Some(row.seq),
    }
}

/// Convert a [`crate::library::note::Note`] into a [`LibrarianSource`].
/// `None` for a note with an empty body (nothing to extract).
pub fn note_to_source(note: &crate::library::note::Note) -> Option<LibrarianSource> {
    if note.body.trim().is_empty() {
        return None;
    }
    Some(LibrarianSource {
        source_id: format!("note:{}", note.id.as_str()),
        text: note.body.clone(),
        seq: None,
    })
}

// ---------------------------------------------------------------------------
// Schema digest + validation
// ---------------------------------------------------------------------------

/// A queryable snapshot of the akh's in-force schema, built once per
/// [`run_librarian_pass`] call and reused across all its sources.
struct SchemaDigest {
    /// Type name → its directly-declared properties (prompt-building).
    types: BTreeMap<String, Vec<String>>,
    /// Property name → the (first) type name that declares it (validation).
    known_properties: BTreeMap<String, String>,
}

impl SchemaDigest {
    fn as_type_digests(&self) -> Vec<SchemaTypeDigest> {
        self.types
            .iter()
            .map(|(name, properties)| SchemaTypeDigest {
                name: name.clone(),
                properties: properties.clone(),
            })
            .collect()
    }
}

/// Build the [`SchemaDigest`] from whatever schema packs are currently
/// installed (`schema::install::install_pack`'s `(T, is-a, onto:Type)` /
/// `(T, onto:property, p)` triples — the same vocabulary
/// `schema::project::project_note` reads). Empty digest (not an error) when
/// no pack is installed yet — the Librarian then files everything as
/// frontier, which is the correct degraded behaviour.
fn schema_digest(engine: &Engine) -> SchemaDigest {
    let mut digest = SchemaDigest {
        types: BTreeMap::new(),
        known_properties: BTreeMap::new(),
    };

    let (Ok(is_a), Ok(type_marker)) = (
        engine.resolve_symbol(PRED_IS_A),
        engine.resolve_symbol(TYPE_MARKER),
    ) else {
        return digest;
    };
    let property_pred = engine.resolve_symbol(PRED_PROPERTY).ok();

    let type_symbols: Vec<SymbolId> = engine
        .knowledge_graph()
        .triples_for_predicate(is_a)
        .into_iter()
        .filter(|(_, object)| *object == type_marker)
        .map(|(subject, _)| subject)
        .collect();

    for type_symbol in type_symbols {
        let type_name = engine.resolve_label(type_symbol);
        // Effective (inherited-included) property set, so the extraction
        // prompt shows e.g. Person with `name` from Thing (Slice 3's
        // is-a chain walk). Falls back to directly-declared properties if
        // the chain is degenerate (e.g. a cycle in a hand-built engine).
        let properties: Vec<String> = match crate::schema::shape::effective_properties(
            engine,
            type_symbol,
        ) {
            Ok(set) => set.properties.into_keys().collect(),
            Err(_) => property_pred
                .map(|p| engine.knowledge_graph().objects_of(type_symbol, p))
                .unwrap_or_default()
                .into_iter()
                .map(|s| engine.resolve_label(s))
                .collect(),
        };
        for property in &properties {
            digest
                .known_properties
                .entry(property.clone())
                .or_insert_with(|| type_name.clone());
        }
        digest.types.insert(type_name, properties);
    }

    digest
}

/// Query the KG for how many triples exist per `note:frontier/{predicate}`
/// marker (`schema::project::FRONTIER_PREFIX`) — the promotion pass (slice
/// 5) consumes this to find recurring frontier terms worth proposing for the
/// core. Not scoped to the Librarian alone: slice 1's mechanical projector
/// uses the same marker convention, so this reports both sources uniformly.
pub fn frontier_terms(engine: &Engine) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for symbol in engine.item_memory().all_symbols() {
        let label = engine.resolve_label(symbol);
        let Some(key) = label.strip_prefix(FRONTIER_PREFIX) else {
            continue;
        };
        let count = engine.knowledge_graph().triples_for_predicate(symbol).len();
        if count > 0 {
            counts.insert(key.to_string(), count);
        }
    }
    let mut tallied: Vec<(String, usize)> = counts.into_iter().collect();
    tallied.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    tallied
}

// ---------------------------------------------------------------------------
// Extraction + assertion
// ---------------------------------------------------------------------------

/// Tuning for one [`run_librarian_pass`] call.
#[derive(Debug, Clone)]
pub struct LibrarianConfig {
    /// Recorded in provenance as the extraction model id when the
    /// [`LibrarianModel`] itself doesn't carry one worth using verbatim.
    /// Currently unused (each model's own [`LibrarianModel::label`] is
    /// authoritative) — reserved for a future override.
    pub model_id_override: Option<String>,
    /// Candidates beyond this count (per source) are dropped, not asserted —
    /// guards against one runaway reply flooding the KG.
    pub max_triples_per_source: usize,
}

impl Default for LibrarianConfig {
    fn default() -> Self {
        Self {
            model_id_override: None,
            max_triples_per_source: 50,
        }
    }
}

/// Outcome of one [`run_librarian_pass`] call.
#[derive(Debug, Clone, Default)]
pub struct LibrarianReport {
    /// Sources that reached the model (excludes empty-text skips).
    pub sources_processed: usize,
    /// Sources skipped because their text was empty (no model call spent).
    pub sources_skipped_empty: usize,
    pub typed_triples_asserted: usize,
    pub frontier_triples_asserted: usize,
    /// Frontier predicate label → count asserted *this call* (not the
    /// whole-KG tally — see [`frontier_terms`] for that).
    pub frontier_tally: BTreeMap<String, usize>,
    /// The watermark after this call (unchanged from the input watermark if
    /// nothing advanced, e.g. an empty `sources` slice — "no new material"
    /// is a no-op).
    pub last_seq: i64,
}

#[derive(Debug, Clone, Default)]
struct SourceOutcome {
    typed: usize,
    frontier: usize,
    frontier_tally: BTreeMap<String, usize>,
}

/// Where the Librarian persists its own watermark. Abstracted so
/// [`run_librarian_pass`] never depends on Postgres to be exercised: unit
/// tests inject [`InMemoryWatermark`]; production wires [`PgLibrarianWatermark`]
/// (postgres feature) via [`run_librarian_pass_over_wal`].
pub trait LibrarianWatermarkStore {
    fn get(&self) -> AgentResult<i64>;
    fn advance(&self, seq: i64) -> AgentResult<()>;
}

/// Non-durable engine-local fallback — also the type unit tests inject.
/// Production without a configured Postgres store simply does not run the
/// Librarian pass across restarts (matching every other store-backed
/// background task's local-only degradation — see `daemon.rs`'s `db: Option<..>`
/// gating); this type exists so the pure orchestration seam
/// ([`run_librarian_pass`]) never requires Postgres to be exercised.
#[derive(Debug, Default)]
pub struct InMemoryWatermark(AtomicI64);

impl InMemoryWatermark {
    pub fn new(initial: i64) -> Self {
        Self(AtomicI64::new(initial))
    }
}

impl LibrarianWatermarkStore for InMemoryWatermark {
    fn get(&self) -> AgentResult<i64> {
        Ok(self.0.load(Ordering::SeqCst))
    }

    fn advance(&self, seq: i64) -> AgentResult<()> {
        self.0.store(seq, Ordering::SeqCst);
        Ok(())
    }
}

/// Postgres-backed watermark: `akhomed_akh_state.librarian_watermark`,
/// written only via the dedicated setter (mirrors `harvest_watermark`'s
/// clobber-guard precedent — see the migration's doc comment).
///
/// **Sync/async boundary**, same discipline as `agent::harvest::harvest_daily_log`
/// (ADR 027: never hold the agent mutex across an `.await`): callers must
/// invoke [`get`](LibrarianWatermarkStore::get) / [`advance`](LibrarianWatermarkStore::advance)
/// from within a tokio runtime context (true for every daemon/TUI/MCP surface
/// that runs the Librarian pass), typically wrapped in
/// `tokio::task::block_in_place`.
#[cfg(feature = "postgres")]
pub struct PgLibrarianWatermark<'a> {
    pub db: &'a akhomed_store::DatabaseConnection,
    pub akh_id: akhomed_store::uuid::Uuid,
}

#[cfg(feature = "postgres")]
impl LibrarianWatermarkStore for PgLibrarianWatermark<'_> {
    fn get(&self) -> AgentResult<i64> {
        let seq = tokio::runtime::Handle::current()
            .block_on(akhomed_store::get_librarian_watermark(self.db, self.akh_id))?;
        Ok(seq)
    }

    fn advance(&self, seq: i64) -> AgentResult<()> {
        tokio::runtime::Handle::current().block_on(akhomed_store::set_librarian_watermark(
            self.db,
            self.akh_id,
            seq,
        ))?;
        Ok(())
    }
}

/// Extract + assert against one source. Pure engine + model call — no
/// watermark logic (that lives in [`run_librarian_pass`]). Skips the model
/// call entirely for empty-text sources (a legitimate, zero-cost no-op, not
/// an error).
fn extract_and_assert_source(
    engine: &Engine,
    model: &dyn LibrarianModel,
    source: &LibrarianSource,
    digest: &SchemaDigest,
    config: &LibrarianConfig,
) -> AgentResult<SourceOutcome> {
    let mut outcome = SourceOutcome::default();
    if source.text.trim().is_empty() {
        return Ok(outcome);
    }

    let request = ExtractionRequest {
        source_id: source.source_id.clone(),
        text: source.text.clone(),
        schema_types: digest.as_type_digests(),
    };

    let candidates = model.extract(&request).map_err(|e| AgentError::Librarian {
        source_id: source.source_id.clone(),
        reason: e.to_string(),
    })?;

    let model_id = config
        .model_id_override
        .clone()
        .unwrap_or_else(|| model.label().to_string());

    for candidate in candidates.iter().take(config.max_triples_per_source) {
        if candidate.subject.trim().is_empty()
            || candidate.predicate.trim().is_empty()
            || candidate.object.trim().is_empty()
        {
            continue; // malformed candidate — silently dropped, mirrors ElicitValidator's syntactic layer
        }
        assert_candidate(
            engine,
            candidate,
            digest,
            &source.source_id,
            &model_id,
            &mut outcome,
        )?;
    }

    Ok(outcome)
}

/// Validate one candidate against `digest` and assert it — typed if the
/// predicate resolves against the installed schema, frontier otherwise
/// (ADR 081 decision 7: never blocked, never silently dropped).
fn assert_candidate(
    engine: &Engine,
    candidate: &CandidateTriple,
    digest: &SchemaDigest,
    source_id: &str,
    model_id: &str,
    outcome: &mut SourceOutcome,
) -> AgentResult<()> {
    let subject = engine.resolve_or_create_entity(&candidate.subject)?;

    let (predicate_label, object_label, schema_type, is_frontier) =
        if candidate.predicate == PRED_IS_A {
            if digest.types.contains_key(&candidate.object) {
                (
                    PRED_IS_A.to_string(),
                    candidate.object.clone(),
                    Some(candidate.object.clone()),
                    false,
                )
            } else {
                (
                    format!("{FRONTIER_PREFIX}{PRED_IS_A}"),
                    candidate.object.clone(),
                    None,
                    true,
                )
            }
        } else if let Some(owning_type) = digest.known_properties.get(&candidate.predicate) {
            (
                candidate.predicate.clone(),
                candidate.object.clone(),
                Some(owning_type.clone()),
                false,
            )
        } else {
            (
                format!("{FRONTIER_PREFIX}{}", candidate.predicate),
                candidate.object.clone(),
                None,
                true,
            )
        };

    let predicate = engine.resolve_or_create_relation(&predicate_label)?;
    let object = engine.resolve_or_create_entity(&object_label)?;
    engine.add_triple(&Triple::new(subject, predicate, object))?;

    let mut prov = ProvenanceRecord::new(
        subject,
        DerivationKind::LibrarianExtraction {
            model_id: model_id.to_string(),
            source_id: source_id.to_string(),
            schema_type,
        },
    )
    .with_sources(vec![subject, predicate, object])
    .with_confidence(0.6);
    let _ = engine.store_provenance(&mut prov);

    if is_frontier {
        outcome.frontier += 1;
        let tally_key = if candidate.predicate == PRED_IS_A {
            PRED_IS_A.to_string()
        } else {
            candidate.predicate.clone()
        };
        *outcome.frontier_tally.entry(tally_key).or_insert(0) += 1;
    } else {
        outcome.typed += 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Run one Librarian pass over already-gathered `sources`. Pure — no
/// Postgres dependency, so this is what unit tests exercise directly.
///
/// Batch-atomic on failure: sources are processed in order, and the whole
/// call returns `Err` (propagated from [`extract_and_assert_source`])
/// immediately on the first source whose model call fails, **before**
/// touching the watermark — so a poison source blocks the watermark's
/// advance rather than being silently skipped (contrast
/// `agent::harvest::harvest_daily_log`'s per-turn skip-and-advance, which is
/// correct there because a malformed *payload* is data corruption, not a
/// model failure to retry). Triples already asserted for earlier sources in
/// the failed batch remain in the KG — harmless, since re-processing them on
/// the retried pass is triple-level idempotent
/// (`schema::project`'s dedup note applies equally here).
///
/// Sources whose `seq` is at or below the current watermark are skipped
/// (already processed in an earlier pass) — an empty `sources` slice, or a
/// slice containing only such already-processed sources, is a no-op that
/// still returns `Ok` without advancing the watermark.
pub fn run_librarian_pass(
    engine: &Engine,
    model: &dyn LibrarianModel,
    watermark: &dyn LibrarianWatermarkStore,
    sources: &[LibrarianSource],
    config: &LibrarianConfig,
) -> AgentResult<LibrarianReport> {
    let current_watermark = watermark.get()?;

    if sources.is_empty() {
        return Ok(LibrarianReport {
            last_seq: current_watermark,
            ..Default::default()
        });
    }

    let digest = schema_digest(engine);
    let mut report = LibrarianReport {
        last_seq: current_watermark,
        ..Default::default()
    };
    let mut max_seq = current_watermark;

    for source in sources {
        if let Some(seq) = source.seq
            && seq <= current_watermark
        {
            continue; // already processed in an earlier pass
        }

        let outcome = extract_and_assert_source(engine, model, source, &digest, config)?;

        if source.text.trim().is_empty() {
            report.sources_skipped_empty += 1;
        } else {
            report.sources_processed += 1;
            report.typed_triples_asserted += outcome.typed;
            report.frontier_triples_asserted += outcome.frontier;
            for (key, count) in outcome.frontier_tally {
                *report.frontier_tally.entry(key).or_insert(0) += count;
            }
        }

        if let Some(seq) = source.seq {
            max_seq = max_seq.max(seq);
        }
    }

    if max_seq > current_watermark {
        watermark.advance(max_seq)?;
        report.last_seq = max_seq;
    }

    Ok(report)
}

/// Batch size cap for one [`run_librarian_pass_over_wal`] call — a very
/// chatty akh degrades gracefully (later ticks pick up where this one left
/// off), mirroring `agent::harvest`'s `HARVEST_BATCH_LIMIT`.
#[cfg(feature = "postgres")]
const LIBRARIAN_BATCH_LIMIT: u64 = 256;

/// Gather un-harvested `kind="turn"` WAL rows (+ any supplied notes) since
/// the Librarian's own watermark, then run [`run_librarian_pass`] against
/// them. The thin DB-facing wrapper `daemon.rs` calls.
#[cfg(feature = "postgres")]
pub fn run_librarian_pass_over_wal(
    db: &akhomed_store::DatabaseConnection,
    engine: &Engine,
    model: &dyn LibrarianModel,
    akh_id: akhomed_store::uuid::Uuid,
    notes: &[crate::library::note::Note],
    config: &LibrarianConfig,
) -> AgentResult<LibrarianReport> {
    let handle = tokio::runtime::Handle::current();
    let watermark_seq = handle.block_on(akhomed_store::get_librarian_watermark(db, akh_id))?;
    let turns = handle.block_on(akhomed_store::load_turns_since(
        db,
        akh_id,
        watermark_seq,
        LIBRARIAN_BATCH_LIMIT,
    ))?;

    let mut sources: Vec<LibrarianSource> = turns.iter().map(turn_row_to_source).collect();
    sources.extend(notes.iter().filter_map(note_to_source));

    let watermark = PgLibrarianWatermark { db, akh_id };
    run_librarian_pass(engine, model, &watermark, &sources, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    /// A persistent (temp-dir-backed) engine — `store_provenance` errors with
    /// `NoPersistence` on a pure in-memory engine (mirrors
    /// `agent::harvest`/`schema::project`'s test setup).
    fn test_engine() -> (tempfile::TempDir, Engine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        (dir, engine)
    }

    const SEED_PACK: &str = r#"{
        "pack": {"name": "librarian-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "email", "value_type": "text"}]}
        ]
    }"#;

    fn install_seed(engine: &Engine) {
        let pack = crate::schema::compile::compile(SEED_PACK).unwrap();
        crate::schema::install::install_pack(engine, &pack).unwrap();
    }

    fn source(source_id: &str, seq: i64) -> LibrarianSource {
        LibrarianSource {
            source_id: source_id.to_string(),
            text: "Jane works here.".to_string(),
            seq: Some(seq),
        }
    }

    #[test]
    fn typed_extraction_against_installed_schema_is_asserted_with_provenance() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let model = FakeLibrarianModel::new().with_extraction(
            "turn:1",
            vec![
                CandidateTriple {
                    subject: "Jane".into(),
                    predicate: PRED_IS_A.into(),
                    object: "Person".into(),
                },
                CandidateTriple {
                    subject: "Jane".into(),
                    predicate: "email".into(),
                    object: "jane@example.org".into(),
                },
            ],
        );
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![source("turn:1", 1)];

        let report = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();

        assert_eq!(report.sources_processed, 1);
        assert_eq!(report.typed_triples_asserted, 2, "is-a Person + email");
        assert_eq!(report.frontier_triples_asserted, 0);

        let jane = engine.resolve_symbol("Jane").unwrap();
        let triples = engine.triples_from(jane);
        assert!(
            triples
                .iter()
                .any(|t| engine.resolve_label(t.predicate) == PRED_IS_A
                    && engine.resolve_label(t.object) == "Person")
        );
        assert!(
            triples
                .iter()
                .any(|t| engine.resolve_label(t.predicate) == "email")
        );

        let prov = engine.provenance_of(jane).unwrap();
        assert!(prov.iter().any(|p| matches!(
            &p.kind,
            DerivationKind::LibrarianExtraction { source_id, schema_type, .. }
                if source_id == "turn:1" && schema_type.as_deref() == Some("Person")
        )));
    }

    #[test]
    fn unknown_predicate_lands_in_frontier_and_is_tallied() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let model = FakeLibrarianModel::new().with_extraction(
            "turn:1",
            vec![CandidateTriple {
                subject: "Jane".into(),
                predicate: "favoriteColor".into(),
                object: "teal".into(),
            }],
        );
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![source("turn:1", 1)];

        let report = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();

        assert_eq!(report.typed_triples_asserted, 0);
        assert_eq!(report.frontier_triples_asserted, 1);
        assert_eq!(report.frontier_tally.get("favoriteColor"), Some(&1));

        let jane = engine.resolve_symbol("Jane").unwrap();
        let triples = engine.triples_from(jane);
        assert!(triples.iter().any(|t| {
            engine.resolve_label(t.predicate) == format!("{FRONTIER_PREFIX}favoriteColor")
        }));

        let prov = engine.provenance_of(jane).unwrap();
        assert!(prov.iter().any(|p| matches!(
            &p.kind,
            DerivationKind::LibrarianExtraction {
                schema_type: None,
                ..
            }
        )));

        // The KG-wide tally query helper (slice 5 will consume this).
        let tallied = frontier_terms(&engine);
        assert!(tallied.contains(&("favoriteColor".to_string(), 1)));
    }

    #[test]
    fn watermark_advances_past_processed_seq() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let model = FakeLibrarianModel::new().with_extraction("turn:1", vec![]);
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![source("turn:1", 7)];

        run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();

        assert_eq!(watermark.get().unwrap(), 7);
    }

    #[test]
    fn second_pass_with_no_new_material_is_a_noop() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let model = FakeLibrarianModel::new().with_extraction(
            "turn:1",
            vec![CandidateTriple {
                subject: "Jane".into(),
                predicate: PRED_IS_A.into(),
                object: "Person".into(),
            }],
        );
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![source("turn:1", 1)];

        run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();
        let triple_count_after_first = engine
            .triples_from(engine.resolve_symbol("Jane").unwrap())
            .len();

        // Re-run with the SAME already-processed source (seq 1 <= watermark 1):
        // internally skipped, nothing new asserted, watermark unchanged.
        let report = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();

        assert_eq!(report.sources_processed, 0);
        assert_eq!(watermark.get().unwrap(), 1);
        let triple_count_after_second = engine
            .triples_from(engine.resolve_symbol("Jane").unwrap())
            .len();
        assert_eq!(triple_count_after_first, triple_count_after_second);

        // An outright empty `sources` slice is equally a no-op.
        let empty_report = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &[],
            &LibrarianConfig::default(),
        )
        .unwrap();
        assert_eq!(empty_report.last_seq, 1);
    }

    #[test]
    fn malformed_llm_reply_returns_diagnostic_error_and_does_not_advance_watermark() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let model = FakeLibrarianModel::new().failing_on("turn:1");
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![source("turn:1", 1)];

        let result = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        );

        assert!(matches!(result, Err(AgentError::Librarian { .. })));
        assert_eq!(
            watermark.get().unwrap(),
            0,
            "watermark must not advance past a failed source"
        );
    }

    #[test]
    fn empty_source_text_skips_the_model_call_but_still_advances_watermark() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        // Configured to fail if called — proves the empty-text source never
        // reaches the model.
        let model = FakeLibrarianModel::new().failing_on("turn:1");
        let watermark = InMemoryWatermark::new(0);
        let sources = vec![LibrarianSource {
            source_id: "turn:1".to_string(),
            text: "   ".to_string(),
            seq: Some(1),
        }];

        let report = run_librarian_pass(
            &engine,
            &model,
            &watermark,
            &sources,
            &LibrarianConfig::default(),
        )
        .unwrap();

        assert_eq!(report.sources_skipped_empty, 1);
        assert_eq!(report.sources_processed, 0);
        assert_eq!(watermark.get().unwrap(), 1);
    }

    #[test]
    fn frontier_terms_reports_nothing_with_no_frontier_triples() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert!(frontier_terms(&engine).is_empty());
    }

    #[test]
    fn note_to_source_skips_empty_body() {
        use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
        use crate::library::note::Note;

        let doc = DocumentRecord {
            id: "doc-1".into(),
            title: "Test".into(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        };
        let empty = Note::new(doc.clone(), "   ");
        assert!(note_to_source(&empty).is_none());

        let with_body = Note::new(doc, "Jane's email is jane@example.org.");
        let src = note_to_source(&with_body).unwrap();
        assert!(src.source_id.starts_with("note:"));
        assert_eq!(src.seq, None);
    }

    #[test]
    fn fake_model_default_promotion_advice_proposes_verbatim_standalone_property() {
        let model = FakeLibrarianModel::new();
        let request = PromotionRequest {
            predicate: "favoriteColor".into(),
            distinct_subjects: 3,
            example_subjects: vec!["Jane".into()],
            example_values: vec!["teal".into()],
            schema_types: vec![],
        };
        let advice = model.propose_promotion(&request).unwrap();
        assert_eq!(advice.proposed_property_name, "favoriteColor");
        assert!(advice.proposed_type.is_none());
        assert!(advice.alias_of.is_none());
    }

    #[test]
    fn fake_model_with_proposal_returns_the_canned_advice() {
        let model = FakeLibrarianModel::new().with_proposal(
            "favoriteColor",
            PromotionAdvice {
                proposed_type: Some("Person".into()),
                proposed_property_name: "favoriteColor".into(),
                alias_of: None,
                rationale: "recurs on Person subjects".into(),
            },
        );
        let request = PromotionRequest {
            predicate: "favoriteColor".into(),
            distinct_subjects: 3,
            example_subjects: vec![],
            example_values: vec![],
            schema_types: vec![],
        };
        let advice = model.propose_promotion(&request).unwrap();
        assert_eq!(advice.proposed_type.as_deref(), Some("Person"));
        assert_eq!(advice.rationale, "recurs on Person subjects");
    }

    #[test]
    fn fake_model_failing_on_also_poisons_promotion_advice() {
        let model = FakeLibrarianModel::new().failing_on("favoriteColor");
        let request = PromotionRequest {
            predicate: "favoriteColor".into(),
            distinct_subjects: 3,
            example_subjects: vec![],
            example_values: vec![],
            schema_types: vec![],
        };
        assert!(matches!(
            model.propose_promotion(&request),
            Err(LibrarianModelError::MalformedReply { .. })
        ));
    }

    #[test]
    fn parse_promotion_advice_is_lenient_about_prose_prefix_and_null_fields() {
        let reply = "Sure! Here you go:\n\
            {\"proposed_type\": null, \"proposed_property_name\": \"favoriteColor\", \
             \"alias_of\": \"color\", \"rationale\": \"synonym of an existing term\"}\ntrailing junk";
        let advice = parse_promotion_advice(reply).unwrap();
        assert_eq!(advice.proposed_property_name, "favoriteColor");
        assert!(advice.proposed_type.is_none());
        assert_eq!(advice.alias_of.as_deref(), Some("color"));
    }

    #[test]
    fn parse_promotion_advice_rejects_reply_with_no_json_object() {
        assert!(parse_promotion_advice("no json here at all").is_none());
    }

    #[test]
    #[ignore = "requires a live Ollama endpoint serving AKH_LIBRARIAN_MODEL (or the default \
                gpt-oss:120b) — not exercised by the automated suite, per ADR 081 slice 4"]
    fn ollama_librarian_model_extracts_against_a_live_backend() {
        let model = OllamaLibrarianModel::from_env();
        let request = ExtractionRequest {
            source_id: "manual".into(),
            text: "Jane's email is jane@example.org.".into(),
            schema_types: vec![SchemaTypeDigest {
                name: "Person".into(),
                properties: vec!["email".into()],
            }],
        };
        let result = model.extract(&request);
        assert!(result.is_ok(), "extract() failed: {result:?}");
    }
}
