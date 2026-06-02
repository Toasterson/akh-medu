//! KG-grounded response templates with `{fact_N}` placeholders (PR #174).
//!
//! This module adds a **KG-grounded templating capability** on top of the
//! existing VSA pattern-matching [`ElizaEngine`](super::engine::ElizaEngine).
//! It was reconciled from PR #174 (`[ELIZA] KG-grounded response templates with
//! fact insertion`, Anima WI `019e404a`), whose original design reimplemented a
//! different `ElizaEngine` from scratch. Rather than overwrite the canonical
//! VSA engine, the reusable templating logic is lifted here so any caller — the
//! engine, the chat path, or a script author — can render facts pulled live from
//! the knowledge graph into response text.
//!
//! ## Template syntax
//!
//! - `{N}` — replace with the Nth regex capture group (1-indexed).
//! - `{fact_N:type:entity}` — query the knowledge graph and insert the result.
//!
//! ### Fact query types
//! | Type | Syntax | Action |
//! |------|--------|--------|
//! | triple | `{fact_0:triple:Ennead}` | Fetch triples for the named entity |
//! | triple (dynamic) | `{fact_0:triple:{1}}` | Fetch triples for the entity in capture 1 |
//! | vsa | `{fact_0:vsa:{1}:3}` | VSA top-3 similar symbols |
//! | competence | `{fact_0:competence:{1}}` | Known-fact count + confidence |
//!
//! During parsing, `{fact_N:type:entity}` tokens are normalised to `{fact_N}` in
//! the stored render text; the full spec is kept in [`FactQuery`] structs. On
//! render, each `{fact_N}` is replaced by the formatted result — or the
//! [`FALLBACK`] string when the query timed out or returned nothing.
//!
//! ## KG queries
//!
//! Each query runs in a dedicated thread and is subject to a hard 100ms
//! wall-clock timeout enforced via [`std::sync::mpsc::Receiver::recv_timeout`].
//! On timeout the query is abandoned and the caller receives `None`, triggering
//! the [`FALLBACK`] string in the rendered template.
//!
//! ## Per-session cache
//!
//! [`ElizaSession`] caches `(entity, query-type)` results so the same pair is
//! never re-queried within a single conversation session.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crate::agent::synthesize::is_metadata_label;
use crate::engine::Engine;

use super::error::{ElizaError, ElizaResult};

// ── FactQueryKind ────────────────────────────────────────────────────────────

/// The type of KG query backing a `{fact_N}` placeholder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactQueryKind {
    /// Fetch outgoing triples and render them as natural-language sentences.
    Triple,
    /// VSA approximate-nearest-neighbour search; returns up to `k` similar symbols.
    Vsa { k: usize },
    /// Competence assessment: fact count + average confidence for the entity.
    Competence,
}

impl FactQueryKind {
    /// Stable string key used as the second half of the session cache key.
    pub fn cache_key(&self) -> String {
        match self {
            Self::Triple => "triple".to_string(),
            Self::Vsa { k } => format!("vsa:{k}"),
            Self::Competence => "competence".to_string(),
        }
    }
}

// ── EntityRef ────────────────────────────────────────────────────────────────

/// How to resolve the entity name for a fact query at render time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityRef {
    /// A hard-coded entity name.
    Literal(String),
    /// Use the text of the Nth regex capture group (1-indexed).
    Capture(usize),
}

impl EntityRef {
    /// Resolve the entity name from the regex capture map.
    pub fn resolve<'a>(&'a self, captures: &'a HashMap<usize, String>) -> &'a str {
        match self {
            Self::Literal(s) => s.as_str(),
            Self::Capture(n) => captures.get(n).map(String::as_str).unwrap_or(""),
        }
    }
}

// ── FactQuery ────────────────────────────────────────────────────────────────

/// A KG query bound to a specific `{fact_N}` placeholder.
#[derive(Debug, Clone)]
pub struct FactQuery {
    /// Which `{fact_N}` placeholder this query fills (N is the index here).
    pub fact_index: usize,
    /// The type of query to execute.
    pub kind: FactQueryKind,
    /// How to resolve the entity name at render time.
    pub entity: EntityRef,
}

// ── Template ─────────────────────────────────────────────────────────────────

/// A reassembly template that may contain capture references and KG fact queries.
#[derive(Debug, Clone)]
pub struct Template {
    /// Render-time text; `{fact_N:type:entity}` tokens have been normalised to `{fact_N}`.
    pub stored_text: String,
    /// All KG queries extracted from the original template string.
    pub fact_queries: Vec<FactQuery>,
}

/// Fallback text used when a KG query times out or returns no data.
pub const FALLBACK: &str = "I don't have data on that yet";

impl Template {
    /// Parse a template string into a [`Template`].
    ///
    /// # Errors
    /// Returns [`ElizaError::TemplateParse`] if a `{fact_N:...}` spec is malformed.
    pub fn parse(text: &str) -> ElizaResult<Self> {
        let mut stored_text = String::with_capacity(text.len());
        let mut fact_queries: Vec<FactQuery> = Vec::new();
        let mut last_end = 0;

        for (range, token) in brace_tokens(text) {
            stored_text.push_str(&text[last_end..range.start]);

            let inner = &token[1..token.len() - 1]; // strip surrounding { }

            if inner.starts_with("fact_") {
                let query = parse_fact_spec(inner)?;
                stored_text.push_str(&format!("{{fact_{}}}", query.fact_index));
                fact_queries.push(query);
            } else {
                // Capture reference or unknown token — keep as-is.
                stored_text.push_str(token);
            }

            last_end = range.end;
        }

        stored_text.push_str(&text[last_end..]);
        Ok(Self { stored_text, fact_queries })
    }

    /// Render the template: substitute captures and pre-resolved facts.
    pub fn render(
        &self,
        captures: &HashMap<usize, String>,
        facts: &HashMap<usize, String>,
    ) -> String {
        let mut out = String::with_capacity(self.stored_text.len() * 2);
        let mut last = 0;

        for (range, token) in brace_tokens(&self.stored_text) {
            out.push_str(&self.stored_text[last..range.start]);

            let inner = &token[1..token.len() - 1];

            if let Some(rest) = inner.strip_prefix("fact_") {
                let replacement = rest
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| facts.get(&n))
                    .map(String::as_str)
                    .unwrap_or(FALLBACK);
                out.push_str(replacement);
            } else if let Ok(n) = inner.parse::<usize>() {
                let replacement = captures.get(&n).map(String::as_str).unwrap_or("");
                out.push_str(replacement);
            } else {
                out.push_str(token);
            }

            last = range.end;
        }

        out.push_str(&self.stored_text[last..]);
        out
    }

    /// Resolve all fact queries against `engine` (with `session` cache) and render
    /// the final response text in one call.
    ///
    /// Each KG query is subject to the 100ms timeout enforced by
    /// [`query_with_timeout`]. Results are cached in `session` so repeated
    /// `(entity, query-type)` pairs within a conversation are not re-queried.
    pub fn render_grounded(
        &self,
        captures: &HashMap<usize, String>,
        engine: &Arc<Engine>,
        session: &mut ElizaSession,
    ) -> String {
        let mut facts: HashMap<usize, String> = HashMap::new();

        for fq in &self.fact_queries {
            let entity = fq.entity.resolve(captures);
            if entity.is_empty() {
                continue;
            }

            let query_key = fq.kind.cache_key();
            let fragments: Vec<String> = if let Some(cached) = session.get(entity, &query_key) {
                cached.clone()
            } else {
                let resolved = query_with_timeout(entity, &fq.kind, engine).unwrap_or_default();
                session.insert(entity, query_key, resolved.clone());
                resolved
            };

            let formatted = format_facts(&fragments, &fq.kind);
            if !formatted.is_empty() {
                facts.insert(fq.fact_index, formatted);
            }
        }

        self.render(captures, &facts)
    }
}

// ── ElizaSession ──────────────────────────────────────────────────────────────

/// Per-session state: caches KG fact results to avoid redundant queries.
///
/// The cache key is `(normalised_entity_name, query_cache_key)` where
/// `query_cache_key` is produced by [`FactQueryKind::cache_key`].
#[derive(Debug, Default)]
pub struct ElizaSession {
    /// `(entity, query_key)` → formatted fact fragments.
    cache: HashMap<(String, String), Vec<String>>,
    /// Number of times a cached result was returned.
    cache_hits: u32,
    /// Number of times the cache was missed (a live query was fired).
    cache_misses: u32,
}

impl ElizaSession {
    /// Create a new empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up cached fact results for `(entity, query_key)`.
    ///
    /// Entity names are normalised to lowercase before lookup.
    pub fn get(&mut self, entity: &str, query_key: &str) -> Option<&Vec<String>> {
        let key = (entity.to_lowercase(), query_key.to_string());
        let hit = self.cache.contains_key(&key);
        if hit {
            self.cache_hits += 1;
        }
        self.cache.get(&key)
    }

    /// Store fact results for `(entity, query_key)`.
    ///
    /// Entity names are normalised to lowercase before storage.
    pub fn insert(
        &mut self,
        entity: impl Into<String>,
        query_key: impl Into<String>,
        facts: Vec<String>,
    ) {
        let key = (entity.into().to_lowercase(), query_key.into());
        if !self.cache.contains_key(&key) {
            self.cache_misses += 1;
        }
        self.cache.insert(key, facts);
    }

    /// Number of distinct `(entity, query_key)` pairs currently cached.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Cache hit count for this session.
    pub fn cache_hits(&self) -> u32 {
        self.cache_hits
    }

    /// Cache miss count for this session.
    pub fn cache_misses(&self) -> u32 {
        self.cache_misses
    }

    /// Evict all cached facts (e.g., on topic change).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

// ── KG query execution ─────────────────────────────────────────────────────────

/// Hard timeout for all KG queries (100 ms, non-negotiable per spec).
const QUERY_TIMEOUT: Duration = Duration::from_millis(100);

/// Maximum triples to include in a Triple query result.
const MAX_TRIPLE_FACTS: usize = 5;

/// Execute a KG query with a 100ms timeout.
///
/// Returns `None` when the query times out, the entity cannot be resolved,
/// or the KG returns no relevant data.
pub fn query_with_timeout(
    entity: &str,
    kind: &FactQueryKind,
    engine: &Arc<Engine>,
) -> Option<Vec<String>> {
    let (tx, rx) = std::sync::mpsc::channel::<Option<Vec<String>>>();
    let engine = Arc::clone(engine);
    let entity = entity.to_string();
    let kind = kind.clone();

    std::thread::spawn(move || {
        let result = match kind {
            FactQueryKind::Triple => query_triple(&entity, &engine),
            FactQueryKind::Vsa { k } => query_vsa(&entity, k, &engine),
            FactQueryKind::Competence => query_competence(&entity, &engine),
        };
        // Ignore the send error: it means the receiver already timed out.
        let _ = tx.send(result);
    });

    rx.recv_timeout(QUERY_TIMEOUT).ok().flatten()
}

/// Format a slice of fact strings for the given query kind into a single string.
///
/// - **Triple**: sentences joined with `"; "`.
/// - **VSA**: Oxford-ish list.
/// - **Competence**: single summary sentence (always one item).
pub fn format_facts(facts: &[String], kind: &FactQueryKind) -> String {
    if facts.is_empty() {
        return String::new();
    }
    match kind {
        FactQueryKind::Triple => facts.join("; "),
        FactQueryKind::Vsa { .. } => {
            if facts.len() == 1 {
                facts[0].clone()
            } else {
                let (last, rest) = facts.split_last().expect("checked non-empty");
                format!("{} and {last}", rest.join(", "))
            }
        }
        FactQueryKind::Competence => facts.join(""),
    }
}

fn query_triple(entity: &str, engine: &Engine) -> Option<Vec<String>> {
    let sym = engine.resolve_symbol(entity).ok()?;
    let triples = engine.triples_from(sym);

    let facts: Vec<String> = triples
        .iter()
        .filter(|t| {
            let pred = engine.resolve_label(t.predicate);
            let obj = engine.resolve_label(t.object);
            !is_metadata_label(&pred) && !is_metadata_label(&obj)
        })
        .take(MAX_TRIPLE_FACTS)
        .map(|t| {
            triple_to_nl(
                &engine.resolve_label(t.subject),
                &engine.resolve_label(t.predicate),
                &engine.resolve_label(t.object),
            )
        })
        .collect();

    if facts.is_empty() { None } else { Some(facts) }
}

fn query_vsa(entity: &str, k: usize, engine: &Engine) -> Option<Vec<String>> {
    let sym = engine.resolve_symbol(entity).ok()?;
    let results = engine.search_similar_to(sym, k + 1).ok()?; // +1 to exclude self

    let labels: Vec<String> = results
        .into_iter()
        // Exclude the query symbol itself (similarity ≥ 0.99 heuristic).
        .filter(|r| r.symbol_id != sym && r.similarity < 0.99)
        .take(k)
        .map(|r| engine.resolve_label(r.symbol_id))
        .filter(|label| !is_metadata_label(label))
        .collect();

    if labels.is_empty() { None } else { Some(labels) }
}

fn query_competence(entity: &str, engine: &Engine) -> Option<Vec<String>> {
    let sym = engine.resolve_symbol(entity).ok()?;
    let triples = engine.triples_from(sym);
    let count = triples.len();

    let avg_conf = if count > 0 {
        let sum: f32 = triples.iter().map(|t| t.confidence).sum();
        sum / count as f32
    } else {
        return None;
    };

    let sentence = format!(
        "{count} fact{} recorded about {entity} (average confidence: {:.0}%)",
        if count == 1 { "" } else { "s" },
        avg_conf * 100.0,
    );
    Some(vec![sentence])
}

// ── NL rendering ─────────────────────────────────────────────────────────────

/// Render a triple as a natural-language sentence.
///
/// Well-known predicates map to idiomatic English; unknown predicates are used verbatim.
pub fn triple_to_nl(subject: &str, predicate: &str, object: &str) -> String {
    match predicate {
        "is-a" => format!("{subject} is a {object}"),
        "subclass-of" => format!("{subject} is a kind of {object}"),
        "part-of" => format!("{subject} is part of {object}"),
        "has-a" | "has-part" => format!("{subject} has {object}"),
        "has-property" => format!("{subject} is {object}"),
        "located-in" => format!("{subject} is located in {object}"),
        "causes" => format!("{subject} causes {object}"),
        "depends-on" => format!("{subject} depends on {object}"),
        "similar-to" => format!("{subject} is similar to {object}"),
        "contains" => format!("{subject} contains {object}"),
        "created-by" => format!("{subject} was created by {object}"),
        "member-of" => format!("{subject} is a member of {object}"),
        "composed-of" => format!("{subject} is composed of {object}"),
        "implements" => format!("{subject} implements {object}"),
        "instance-of" => format!("{subject} is an instance of {object}"),
        "written-in" => format!("{subject} is written in {object}"),
        "defines" => format!("{subject} defines {object}"),
        _ => format!("{subject} {predicate} {object}"),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Iterator over brace-delimited tokens in `text`, respecting nested braces.
///
/// Yields `(byte_range, token_str)` for each `{...}` token found.
pub fn brace_tokens(text: &str) -> impl Iterator<Item = (Range<usize>, &str)> {
    BraceTokenIter { text, pos: 0 }
}

struct BraceTokenIter<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Iterator for BraceTokenIter<'a> {
    type Item = (Range<usize>, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        // Scan for the next opening brace.
        while self.pos < bytes.len() && bytes[self.pos] != b'{' {
            self.pos += 1;
        }
        if self.pos >= bytes.len() {
            return None;
        }

        let start = self.pos;
        let mut depth = 1usize;
        self.pos += 1;

        while self.pos < bytes.len() && depth > 0 {
            match bytes[self.pos] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            self.pos += 1;
        }

        if depth == 0 {
            let range = start..self.pos;
            Some((range.clone(), &self.text[range]))
        } else {
            None // unmatched brace — stop
        }
    }
}

/// Parse the inner content of a `{fact_N:type:entity}` token (braces already stripped).
fn parse_fact_spec(inner: &str) -> ElizaResult<FactQuery> {
    // Split on ':' but only for the outer level (entity may contain '{N}').
    // Since entity names and capture refs don't contain ':', a simple split is fine.
    let parts: Vec<&str> = inner.splitn(4, ':').collect();

    let [fact_name, query_type, entity_spec, rest @ ..] = parts.as_slice() else {
        return Err(ElizaError::TemplateParse {
            message: format!("expected 'fact_N:type:entity' but got '{inner}'"),
        });
    };

    let fact_index = fact_name
        .strip_prefix("fact_")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| ElizaError::TemplateParse {
            message: format!("expected 'fact_N' prefix but got '{fact_name}'"),
        })?;

    let entity = parse_entity_ref(entity_spec);

    let kind = match *query_type {
        "triple" => FactQueryKind::Triple,
        "vsa" => {
            let k = rest.first().and_then(|s| s.parse().ok()).unwrap_or(3);
            FactQueryKind::Vsa { k }
        }
        "competence" => FactQueryKind::Competence,
        other => {
            return Err(ElizaError::TemplateParse {
                message: format!(
                    "unknown fact query type '{other}'; expected 'triple', 'vsa', or 'competence'"
                ),
            });
        }
    };

    Ok(FactQuery { fact_index, kind, entity })
}

/// Parse `{N}` as a capture reference or anything else as a literal.
fn parse_entity_ref(s: &str) -> EntityRef {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('{').and_then(|t| t.strip_suffix('}')) {
        if let Ok(n) = inner.parse::<usize>() {
            return EntityRef::Capture(n);
        }
    }
    EntityRef::Literal(s.to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_literal_template() {
        let t = Template::parse("Hello world").unwrap();
        assert_eq!(t.stored_text, "Hello world");
        assert!(t.fact_queries.is_empty());
    }

    #[test]
    fn parse_capture_ref() {
        let t = Template::parse("You said {1}.").unwrap();
        assert_eq!(t.stored_text, "You said {1}.");
        assert!(t.fact_queries.is_empty());
    }

    #[test]
    fn parse_triple_fact_literal_entity() {
        let t = Template::parse("I know: {fact_0:triple:Ennead}.").unwrap();
        assert_eq!(t.stored_text, "I know: {fact_0}.");
        assert_eq!(t.fact_queries.len(), 1);
        let fq = &t.fact_queries[0];
        assert_eq!(fq.fact_index, 0);
        assert_eq!(fq.kind, FactQueryKind::Triple);
        assert_eq!(fq.entity, EntityRef::Literal("Ennead".into()));
    }

    #[test]
    fn parse_triple_fact_capture_entity() {
        let t = Template::parse("About {1}: {fact_0:triple:{1}}.").unwrap();
        assert_eq!(t.stored_text, "About {1}: {fact_0}.");
        let fq = &t.fact_queries[0];
        assert_eq!(fq.entity, EntityRef::Capture(1));
    }

    #[test]
    fn parse_vsa_with_k() {
        let t = Template::parse("Similar: {fact_0:vsa:{1}:5}.").unwrap();
        let fq = &t.fact_queries[0];
        assert_eq!(fq.kind, FactQueryKind::Vsa { k: 5 });
    }

    #[test]
    fn parse_vsa_default_k() {
        let t = Template::parse("Similar: {fact_0:vsa:{1}}.").unwrap();
        let fq = &t.fact_queries[0];
        assert_eq!(fq.kind, FactQueryKind::Vsa { k: 3 });
    }

    #[test]
    fn parse_competence_fact() {
        let t = Template::parse("Competence: {fact_0:competence:{1}}.").unwrap();
        let fq = &t.fact_queries[0];
        assert_eq!(fq.kind, FactQueryKind::Competence);
    }

    #[test]
    fn parse_unknown_type_errors() {
        let err = Template::parse("{fact_0:unknown:entity}").unwrap_err();
        assert!(matches!(err, ElizaError::TemplateParse { .. }));
    }

    #[test]
    fn render_substitutes_captures_and_facts() {
        let t = Template::parse("About {1}: {fact_0:triple:{1}}.").unwrap();

        let mut caps = HashMap::new();
        caps.insert(1, "Dogs".to_string());

        let mut facts = HashMap::new();
        facts.insert(0, "Dogs are mammals".to_string());

        let out = t.render(&caps, &facts);
        assert_eq!(out, "About Dogs: Dogs are mammals.");
    }

    #[test]
    fn render_uses_fallback_when_fact_missing() {
        let t = Template::parse("About {1}: {fact_0:triple:{1}}.").unwrap();

        let mut caps = HashMap::new();
        caps.insert(1, "Unicorns".to_string());

        let out = t.render(&caps, &HashMap::new());
        assert!(out.contains(FALLBACK));
    }

    #[test]
    fn brace_tokens_nested() {
        let text = "x {fact_0:triple:{1}} y";
        let tokens: Vec<_> = brace_tokens(text).collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].1, "{fact_0:triple:{1}}");
    }

    #[test]
    fn entity_ref_capture_resolve() {
        let mut caps = HashMap::new();
        caps.insert(2, "Ra".to_string());
        let er = EntityRef::Capture(2);
        assert_eq!(er.resolve(&caps), "Ra");
    }

    #[test]
    fn entity_ref_literal_resolve() {
        let er = EntityRef::Literal("Osiris".into());
        assert_eq!(er.resolve(&HashMap::new()), "Osiris");
    }

    #[test]
    fn fact_query_kind_cache_keys() {
        assert_eq!(FactQueryKind::Triple.cache_key(), "triple");
        assert_eq!(FactQueryKind::Vsa { k: 5 }.cache_key(), "vsa:5");
        assert_eq!(FactQueryKind::Competence.cache_key(), "competence");
    }

    #[test]
    fn triple_to_nl_known_predicates() {
        assert_eq!(triple_to_nl("Rust", "is-a", "language"), "Rust is a language");
        assert_eq!(triple_to_nl("X", "part-of", "Y"), "X is part of Y");
        assert_eq!(triple_to_nl("A", "member-of", "B"), "A is a member of B");
    }

    #[test]
    fn triple_to_nl_unknown_predicate() {
        assert_eq!(triple_to_nl("Ennead", "worships", "Ra"), "Ennead worships Ra");
    }

    #[test]
    fn format_facts_triple_joins_semicolons() {
        let facts = vec!["A is B".into(), "C is D".into()];
        assert_eq!(format_facts(&facts, &FactQueryKind::Triple), "A is B; C is D");
    }

    #[test]
    fn format_facts_vsa_single() {
        let facts = vec!["Ra".into()];
        assert_eq!(format_facts(&facts, &FactQueryKind::Vsa { k: 3 }), "Ra");
    }

    #[test]
    fn format_facts_vsa_multiple() {
        let facts = vec!["Ra".into(), "Osiris".into(), "Horus".into()];
        let out = format_facts(&facts, &FactQueryKind::Vsa { k: 3 });
        assert_eq!(out, "Ra, Osiris and Horus");
    }

    #[test]
    fn format_facts_empty_returns_empty() {
        assert_eq!(format_facts(&[], &FactQueryKind::Triple), "");
    }

    #[test]
    fn session_insert_then_hit() {
        let mut s = ElizaSession::new();
        s.insert("Ennead", "triple", vec!["Ennead is a deity group".into()]);
        let result = s.get("Ennead", "triple");
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], "Ennead is a deity group");
        assert_eq!(s.cache_hits(), 1);
    }

    #[test]
    fn session_entity_name_normalised_lowercase() {
        let mut s = ElizaSession::new();
        s.insert("ENNEAD", "triple", vec!["fact".into()]);
        assert!(s.get("ennead", "triple").is_some());
        assert!(s.get("Ennead", "triple").is_some());
    }

    #[test]
    fn session_clear_cache_empties() {
        let mut s = ElizaSession::new();
        s.insert("Ra", "triple", vec!["Ra is a sun god".into()]);
        assert_eq!(s.cache_size(), 1);
        s.clear_cache();
        assert_eq!(s.cache_size(), 0);
        assert!(s.get("Ra", "triple").is_none());
    }
}
