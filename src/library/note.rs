//! `Note` = library [`DocumentRecord`] + optional typed frontmatter + wikilinks
//! + a stable identity that survives edits (ADR 081 work-item P1).
//!
//! This is the *carrier* prefactor only — no projection into the typed KG
//! happens here (that is slice 1 in
//! `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`). The critical
//! invariant this module must uphold: **a frontmatter-less [`Note`] behaves
//! byte-identically to today's [`DocumentRecord`]** — [`Note`] wraps
//! [`DocumentRecord`] without mutating it, so existing ingest/catalog code
//! and tests are untouched by this module's existence.
//!
//! Frontmatter values are typed per the Obsidian value set already declared
//! in [`crate::schema::source::ValueType`] (text / number / checkbox / date /
//! datetime / list / link) — see ADR 081 decision 4 (layered vocabulary
//! borrow). Wikilinks (`[[target]]` / `[[target|alias]]`) are extracted from
//! the prose body. Embedded task references are a typed placeholder only —
//! the task tool itself is a separate slice (work-item 6); no task logic
//! lives here.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::LazyLock;

use rand::RngCore;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::library::error::LibraryError;
use crate::library::model::DocumentRecord;

// ---------------------------------------------------------------------------
// Stable note identity
// ---------------------------------------------------------------------------

/// A stable identity for a [`Note`], minted once at creation and preserved
/// across body/frontmatter edits.
///
/// Deliberately independent of content: renaming a note's title or editing
/// its body/frontmatter must never change the id (P1 acceptance: "a stable
/// id that survives edits"). Built from `rand` + `blake3` (both already
/// unconditional workspace dependencies) rather than pulling in a `uuid`
/// direct dependency.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NoteId(String);

impl NoteId {
    /// Mint a fresh, random note id.
    pub fn new() -> Self {
        let mut entropy = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut entropy);
        let hash = blake3::hash(&entropy);
        Self(format!("note-{}", &hash.to_hex()[..32]))
    }

    /// Borrow the id as a plain string slice (e.g. for use as a map key or
    /// KG symbol label suffix).
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for NoteId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NoteId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Wikilinks
// ---------------------------------------------------------------------------

/// A `[[target]]` or `[[target|alias]]` reference extracted from a note body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wikilink {
    /// The linked note/concept label (the text inside `[[...]]` before `|`).
    pub target: String,
    /// Optional display alias (the text after `|`).
    pub alias: Option<String>,
}

impl fmt::Display for Wikilink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.alias {
            Some(alias) => write!(f, "[[{}|{}]]", self.target, alias),
            None => write!(f, "[[{}]]", self.target),
        }
    }
}

static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]").expect("valid regex"));

/// Extract all `[[target]]` / `[[target|alias]]` references from `text`, in
/// order of appearance.
pub fn extract_wikilinks(text: &str) -> Vec<Wikilink> {
    WIKILINK_RE
        .captures_iter(text)
        .map(|caps| Wikilink {
            target: caps.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default(),
            alias: caps.get(2).map(|m| m.as_str().trim().to_string()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Embedded task references (placeholder only — see work-item 6)
// ---------------------------------------------------------------------------

/// A typed reference to a task tracked by the (separate) structured task
/// tool. This is a placeholder field only: no task creation/completion logic
/// lives in the library — the task store is authoritative elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRef(pub String);

impl fmt::Display for TaskRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Typed frontmatter
// ---------------------------------------------------------------------------

/// A single frontmatter value, typed per Obsidian's value set
/// ([`crate::schema::source::ValueType`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontmatterValue {
    Text(String),
    Number(f64),
    Checkbox(bool),
    /// `YYYY-MM-DD`, kept as the validated source string (no calendar-math
    /// dependency belongs in a note carrier).
    Date(String),
    /// `YYYY-MM-DDThh:mm[:ss]`, kept as the validated source string.
    Datetime(String),
    List(Vec<String>),
    Link(Wikilink),
}

impl FrontmatterValue {
    /// The [`crate::schema::source::ValueType`] this value carries.
    pub fn value_type(&self) -> crate::schema::source::ValueType {
        use crate::schema::source::ValueType;
        match self {
            FrontmatterValue::Text(_) => ValueType::Text,
            FrontmatterValue::Number(_) => ValueType::Number,
            FrontmatterValue::Checkbox(_) => ValueType::Checkbox,
            FrontmatterValue::Date(_) => ValueType::Date,
            FrontmatterValue::Datetime(_) => ValueType::Datetime,
            FrontmatterValue::List(_) => ValueType::List,
            FrontmatterValue::Link(_) => ValueType::Link,
        }
    }

    /// Render as a single-line frontmatter value (the inverse of
    /// [`parse_value`]).
    fn render(&self) -> String {
        match self {
            FrontmatterValue::Text(s) => {
                if needs_quoting(s) {
                    format!("\"{}\"", s.replace('"', "\\\""))
                } else {
                    s.clone()
                }
            }
            FrontmatterValue::Number(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    n.to_string()
                }
            }
            FrontmatterValue::Checkbox(b) => b.to_string(),
            FrontmatterValue::Date(d) => d.clone(),
            FrontmatterValue::Datetime(dt) => dt.clone(),
            FrontmatterValue::List(items) => format!("[{}]", items.join(", ")),
            FrontmatterValue::Link(link) => link.to_string(),
        }
    }
}

fn needs_quoting(s: &str) -> bool {
    s.is_empty()
        || s.starts_with(['[', '"'])
        || s.contains(':')
        || s.trim() != s
        || matches!(s.to_ascii_lowercase().as_str(), "true" | "false")
}

static DATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}$").expect("valid regex"));
static DATETIME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?").expect("valid regex"));

/// Infer and parse a single frontmatter value from its textual form.
///
/// `pub(crate)` (not just module-private) because [`crate::agent::promotion`]
/// (ADR 081 Slice 5) reuses this exact inference to check whether a frontier
/// predicate's observed object *labels* (plain KG entity strings, not
/// frontmatter text) share a consistent [`crate::schema::source::ValueType`]
/// shape — the same textual heuristics that already drive frontmatter typing,
/// applied to the same kind of raw string.
pub(crate) fn parse_value(raw: &str) -> FrontmatterValue {
    let trimmed = raw.trim();

    if let Some(inner) = trimmed.strip_prefix("[[").and_then(|s| s.strip_suffix("]]")) {
        let mut parts = inner.splitn(2, '|');
        let target = parts.next().unwrap_or_default().trim().to_string();
        let alias = parts.next().map(|s| s.trim().to_string());
        return FrontmatterValue::Link(Wikilink { target, alias });
    }

    if let Some(inner) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let items = if inner.trim().is_empty() {
            Vec::new()
        } else {
            inner
                .split(',')
                .map(|s| unquote(s.trim()).to_string())
                .collect()
        };
        return FrontmatterValue::List(items);
    }

    if trimmed.eq_ignore_ascii_case("true") {
        return FrontmatterValue::Checkbox(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return FrontmatterValue::Checkbox(false);
    }

    if DATETIME_RE.is_match(trimmed) {
        return FrontmatterValue::Datetime(trimmed.to_string());
    }
    if DATE_RE.is_match(trimmed) {
        return FrontmatterValue::Date(trimmed.to_string());
    }

    // `f64::parse` rejects the empty string on its own, so no extra guard
    // is needed here.
    if let Ok(n) = trimmed.parse::<f64>() {
        return FrontmatterValue::Number(n);
    }

    FrontmatterValue::Text(unquote(trimmed).to_string())
}

fn unquote(s: &str) -> &str {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Optional typed frontmatter: schema.org-flavoured `key: value` pairs.
///
/// An empty [`Frontmatter`] is the frontmatter-less case — a [`Note`] with
/// no frontmatter fields behaves exactly like today's [`DocumentRecord`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Frontmatter(BTreeMap<String, FrontmatterValue>);

impl Frontmatter {
    /// An empty frontmatter block.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this frontmatter carries no fields (the frontmatter-less
    /// case).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Set (or overwrite) a field.
    pub fn insert(&mut self, key: impl Into<String>, value: FrontmatterValue) {
        self.0.insert(key.into(), value);
    }

    /// Look up a field by key.
    pub fn get(&self, key: &str) -> Option<&FrontmatterValue> {
        self.0.get(key)
    }

    /// Iterate fields in key order (stable, for deterministic rendering).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &FrontmatterValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Parse a `key: value` frontmatter block (one field per line; blank
    /// lines ignored; optional surrounding `---` fences are stripped).
    ///
    /// Value types are inferred from syntax:
    /// - `true` / `false` → checkbox
    /// - `YYYY-MM-DD` → date, `YYYY-MM-DDThh:mm[:ss]...` → datetime
    /// - `[[target]]` / `[[target|alias]]` → link
    /// - `[a, b, c]` → list
    /// - a bare number → number
    /// - everything else → text (surrounding `"..."` quotes are stripped)
    pub fn parse(text: &str) -> Result<Self, LibraryError> {
        let mut fields = BTreeMap::new();

        for (idx, raw_line) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.trim_end();
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed == "---" {
                continue;
            }

            let Some((key, value)) = trimmed.split_once(':') else {
                return Err(LibraryError::FrontmatterParse {
                    line: line_no,
                    message: format!("no `:` separator in {trimmed:?}"),
                });
            };

            let key = key.trim();
            if key.is_empty() {
                return Err(LibraryError::FrontmatterParse {
                    line: line_no,
                    message: "empty key before `:`".to_string(),
                });
            }

            fields.insert(key.to_string(), parse_value(value));
        }

        Ok(Self(fields))
    }

    /// Render back to a `key: value` block (without `---` fences), in key
    /// order. Not guaranteed to be the exact original text (e.g. quoting/
    /// list-spacing may normalize), but re-[`parse`](Self::parse)-ing the
    /// output always yields an equal [`Frontmatter`].
    pub fn to_block(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.0 {
            out.push_str(key);
            out.push_str(": ");
            out.push_str(&value.render());
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Note
// ---------------------------------------------------------------------------

/// A `Note` = the existing library [`DocumentRecord`] + optional typed
/// frontmatter + wikilinks + embedded task references, carrying a stable
/// identity that survives edits.
///
/// **Invariant:** a [`Note`] with empty [`Frontmatter`] renders exactly its
/// body text — no frontmatter block is ever injected — so a frontmatter-less
/// `Note` behaves byte-identically to today's [`DocumentRecord`] + prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    /// Stable identity, minted once, unaffected by body/frontmatter edits.
    pub id: NoteId,
    /// The wrapped catalog record — untouched, so existing catalog/ingest
    /// code keeps working on it unchanged.
    pub document: DocumentRecord,
    /// Prose body (what would be chunked/embedded for vector recall).
    pub body: String,
    /// Optional typed frontmatter (empty = frontmatter-less).
    pub frontmatter: Frontmatter,
    /// Wikilinks extracted from the body.
    pub wikilinks: Vec<Wikilink>,
    /// Embedded task references (placeholder — see work-item 6).
    pub tasks: Vec<TaskRef>,
}

impl Note {
    /// Create a new note wrapping `document`, with a prose `body` and no
    /// frontmatter. Wikilinks are extracted from the body immediately. Mints
    /// a fresh, stable [`NoteId`].
    pub fn new(document: DocumentRecord, body: impl Into<String>) -> Self {
        let body = body.into();
        let wikilinks = extract_wikilinks(&body);
        Self {
            id: NoteId::new(),
            document,
            body,
            frontmatter: Frontmatter::new(),
            wikilinks,
            tasks: Vec::new(),
        }
    }

    /// Wrap an existing [`DocumentRecord`] as a frontmatter-less `Note` with
    /// no body text (the record's prose lives in the KG/vector store, not in
    /// this carrier — see the module docs). Existing `Document` machinery is
    /// untouched by this call; it only mints a fresh `Note` identity around
    /// the record.
    pub fn from_document(document: DocumentRecord) -> Self {
        Self::new(document, String::new())
    }

    /// Attach frontmatter (builder style).
    pub fn with_frontmatter(mut self, frontmatter: Frontmatter) -> Self {
        self.frontmatter = frontmatter;
        self
    }

    /// Attach task references (builder style).
    pub fn with_tasks(mut self, tasks: Vec<TaskRef>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Edit the body in place. Re-extracts wikilinks. The note's `id` is
    /// never touched by edits (P1 acceptance: identity survives edits).
    pub fn set_body(&mut self, body: impl Into<String>) {
        self.body = body.into();
        self.wikilinks = extract_wikilinks(&self.body);
    }

    /// Replace the frontmatter in place. The note's `id` is never touched by
    /// edits.
    pub fn set_frontmatter(&mut self, frontmatter: Frontmatter) {
        self.frontmatter = frontmatter;
    }

    /// Whether this note has no frontmatter fields (the frontmatter-less
    /// case that must behave byte-identically to a plain `Document`).
    pub fn is_frontmatter_less(&self) -> bool {
        self.frontmatter.is_empty()
    }

    /// Render the note as it would appear on disk: an optional `---`-fenced
    /// frontmatter block followed by the body. A frontmatter-less note
    /// renders as exactly its body — nothing is ever prepended.
    pub fn render(&self) -> String {
        if self.frontmatter.is_empty() {
            self.body.clone()
        } else {
            format!("---\n{}---\n{}", self.frontmatter.to_block(), self.body)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::model::{ContentFormat, DocumentSource};

    fn sample_document(id: &str) -> DocumentRecord {
        DocumentRecord {
            id: id.into(),
            title: "Test Document".into(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec!["test".into()],
            chunk_count: 1,
            triple_count: 2,
            ingested_at: 1_700_000_000,
        }
    }

    // -- frontmatter parse round-trip --------------------------------------

    #[test]
    fn frontmatter_parse_round_trip() {
        let text = "\
title: Hello World
count: 42
ratio: 3.5
done: true
pending: false
due: 2026-07-15
updated: 2026-07-15T10:30:00
tags: [alpha, beta, gamma]
see_also: [[Other Note|alias text]]
plain_link: [[Bare Target]]
";
        let fm = Frontmatter::parse(text).unwrap();
        assert_eq!(fm.len(), 10);
        assert_eq!(
            fm.get("title"),
            Some(&FrontmatterValue::Text("Hello World".into()))
        );
        assert_eq!(fm.get("count"), Some(&FrontmatterValue::Number(42.0)));
        assert_eq!(fm.get("ratio"), Some(&FrontmatterValue::Number(3.5)));
        assert_eq!(fm.get("done"), Some(&FrontmatterValue::Checkbox(true)));
        assert_eq!(fm.get("pending"), Some(&FrontmatterValue::Checkbox(false)));
        assert_eq!(
            fm.get("due"),
            Some(&FrontmatterValue::Date("2026-07-15".into()))
        );
        assert_eq!(
            fm.get("updated"),
            Some(&FrontmatterValue::Datetime("2026-07-15T10:30:00".into()))
        );
        assert_eq!(
            fm.get("tags"),
            Some(&FrontmatterValue::List(vec![
                "alpha".into(),
                "beta".into(),
                "gamma".into()
            ]))
        );
        assert_eq!(
            fm.get("see_also"),
            Some(&FrontmatterValue::Link(Wikilink {
                target: "Other Note".into(),
                alias: Some("alias text".into())
            }))
        );
        assert_eq!(
            fm.get("plain_link"),
            Some(&FrontmatterValue::Link(Wikilink {
                target: "Bare Target".into(),
                alias: None
            }))
        );

        // Re-parsing the rendered block must yield an equal Frontmatter.
        let rendered = fm.to_block();
        let reparsed = Frontmatter::parse(&rendered).unwrap();
        assert_eq!(fm, reparsed);
    }

    #[test]
    fn frontmatter_parse_strips_fences_and_blanks() {
        let text = "---\ntitle: A\n\nkind: B\n---\n";
        let fm = Frontmatter::parse(text).unwrap();
        assert_eq!(fm.len(), 2);
        assert_eq!(fm.get("title"), Some(&FrontmatterValue::Text("A".into())));
    }

    #[test]
    fn frontmatter_parse_rejects_missing_separator() {
        let err = Frontmatter::parse("not-a-kv-line").unwrap_err();
        assert!(matches!(err, LibraryError::FrontmatterParse { line: 1, .. }));
    }

    #[test]
    fn frontmatter_parse_rejects_empty_key() {
        let err = Frontmatter::parse(": value").unwrap_err();
        assert!(matches!(err, LibraryError::FrontmatterParse { .. }));
    }

    #[test]
    fn value_type_matches_schema_value_types() {
        use crate::schema::source::ValueType;
        assert_eq!(FrontmatterValue::Text("x".into()).value_type(), ValueType::Text);
        assert_eq!(FrontmatterValue::Number(1.0).value_type(), ValueType::Number);
        assert_eq!(FrontmatterValue::Checkbox(true).value_type(), ValueType::Checkbox);
        assert_eq!(
            FrontmatterValue::Date("2026-01-01".into()).value_type(),
            ValueType::Date
        );
        assert_eq!(
            FrontmatterValue::Datetime("2026-01-01T00:00:00".into()).value_type(),
            ValueType::Datetime
        );
        assert_eq!(
            FrontmatterValue::List(vec![]).value_type(),
            ValueType::List
        );
        assert_eq!(
            FrontmatterValue::Link(Wikilink {
                target: "x".into(),
                alias: None
            })
            .value_type(),
            ValueType::Link
        );
    }

    // -- wikilink extraction -------------------------------------------------

    #[test]
    fn wikilink_extraction_finds_bare_and_aliased_links() {
        let text = "See [[Rust Programming]] and also [[Cargo|the build tool]] for details.";
        let links = extract_wikilinks(text);
        assert_eq!(
            links,
            vec![
                Wikilink {
                    target: "Rust Programming".into(),
                    alias: None
                },
                Wikilink {
                    target: "Cargo".into(),
                    alias: Some("the build tool".into())
                },
            ]
        );
    }

    #[test]
    fn wikilink_extraction_empty_when_none_present() {
        assert!(extract_wikilinks("plain prose, no links here").is_empty());
    }

    #[test]
    fn note_extracts_wikilinks_from_body_on_construction() {
        let note = Note::new(sample_document("doc-1"), "Links to [[Target A]].");
        assert_eq!(note.wikilinks.len(), 1);
        assert_eq!(note.wikilinks[0].target, "Target A");
    }

    // -- stable id across edits ----------------------------------------------

    #[test]
    fn stable_id_across_body_and_frontmatter_edits() {
        let mut note = Note::new(sample_document("doc-1"), "original body");
        let id_before = note.id.clone();

        note.set_body("a completely different body with [[New Link]]");
        assert_eq!(note.id, id_before, "id must survive body edits");

        let mut fm = Frontmatter::new();
        fm.insert("title", FrontmatterValue::Text("Renamed".into()));
        note.set_frontmatter(fm);
        assert_eq!(note.id, id_before, "id must survive frontmatter edits");
    }

    #[test]
    fn note_ids_are_unique_per_mint() {
        let a = NoteId::new();
        let b = NoteId::new();
        assert_ne!(a, b);
    }

    // -- frontmatter-less byte-identity ---------------------------------------

    #[test]
    fn frontmatter_less_note_renders_body_byte_identical() {
        let body = "Exactly this text, no more, no less.";
        let note = Note::new(sample_document("doc-1"), body);
        assert!(note.is_frontmatter_less());
        assert_eq!(note.render(), body, "no frontmatter block may be injected");
    }

    #[test]
    fn from_document_wraps_record_unchanged() {
        let record = sample_document("doc-1");
        let note = Note::from_document(record.clone());
        assert_eq!(note.document, record, "DocumentRecord must be untouched");
        assert!(note.is_frontmatter_less());
        assert_eq!(note.render(), "");
    }

    #[test]
    fn note_with_frontmatter_renders_fenced_block_then_body() {
        let mut fm = Frontmatter::new();
        fm.insert("title", FrontmatterValue::Text("Hi".into()));
        let note = Note::new(sample_document("doc-1"), "body text").with_frontmatter(fm);
        let rendered = note.render();
        assert!(rendered.starts_with("---\ntitle: Hi\n---\n"));
        assert!(rendered.ends_with("body text"));
    }

    // -- persistence round-trip -----------------------------------------------

    #[test]
    fn note_serde_round_trip_preserves_frontmatter_and_wikilinks() {
        let mut fm = Frontmatter::new();
        fm.insert("title", FrontmatterValue::Text("Hi".into()));
        fm.insert("count", FrontmatterValue::Number(7.0));
        fm.insert(
            "related",
            FrontmatterValue::Link(Wikilink {
                target: "Other".into(),
                alias: None,
            }),
        );
        let note = Note::new(sample_document("doc-1"), "See [[Other]] for more.")
            .with_frontmatter(fm)
            .with_tasks(vec![TaskRef("task-123".into())]);

        let json = serde_json::to_string(&note).unwrap();
        let restored: Note = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, note);
        assert_eq!(restored.frontmatter.len(), 3);
        assert_eq!(restored.wikilinks, note.wikilinks);
        assert_eq!(restored.tasks, vec![TaskRef("task-123".into())]);
        assert_eq!(restored.id, note.id);
    }

    #[test]
    fn frontmatter_less_note_serde_round_trip_matches_document_shape() {
        let record = sample_document("doc-1");
        let note = Note::from_document(record.clone());

        let json = serde_json::to_string(&note).unwrap();
        let restored: Note = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, note);
        assert!(restored.is_frontmatter_less());

        // The wrapped DocumentRecord alone still round-trips exactly as it
        // does today, independent of the Note wrapper.
        let doc_json = serde_json::to_string(&record).unwrap();
        let doc_restored: DocumentRecord = serde_json::from_str(&doc_json).unwrap();
        assert_eq!(doc_restored.id, record.id);
        assert_eq!(doc_restored.chunk_count, record.chunk_count);
    }
}
