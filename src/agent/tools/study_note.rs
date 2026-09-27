//! Provenance-backed, append-only study-note authoring.
//!
//! This is deliberately narrower than a generic `take_note` tool. Ordinary
//! conversation capture remains structural (ADR 081); this tool is for an
//! explicit learning workflow after source passages have been retrieved. It
//! requires at least one source locator and supporting quote, stores every edit
//! as an immutable revision, and then refreshes the typed KG projection.
//! Persistence is the truth path, so a projection failure is reported but
//! never discards a note.
//!
//! These tool executors validate citation shape and storage semantics. Every
//! model-driven invocation must still pass through `Agent::execute_tool_call`,
//! which binds those citations to successful earlier retrieval results. Direct
//! executor calls are reserved for trusted host code and focused unit tests.

use std::collections::HashSet;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::library::catalog::slugify;
use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
use crate::library::note::{Frontmatter, FrontmatterValue, Note, NoteId, Wikilink};
use crate::library::note_store::{NoteCatalog, PersistentNoteCatalog};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CitationInput {
    source_id: String,
    locator: String,
    quote: String,
}

/// Append a new source-backed study note.
pub struct StudyNoteAppendTool {
    store: Arc<PersistentNoteCatalog>,
    allow_revision: bool,
}

/// Append a conflict-safe revision of an existing study note.
///
/// Creation and revision are separate model-facing contracts so their required
/// inputs remain expressible in ordinary JSON Schema. This is especially
/// important for generic all-tool curricula: `note_id` and
/// `expected_revision` cannot be sampled independently.
pub struct StudyNoteReviseTool {
    store: Arc<PersistentNoteCatalog>,
}

impl StudyNoteReviseTool {
    pub fn new(store: Arc<PersistentNoteCatalog>) -> Self {
        Self { store }
    }
}

impl StudyNoteAppendTool {
    pub fn new(store: Arc<PersistentNoteCatalog>) -> Self {
        Self {
            store,
            allow_revision: false,
        }
    }

    fn for_revision(store: Arc<PersistentNoteCatalog>) -> Self {
        Self {
            store,
            allow_revision: true,
        }
    }
}

/// Shared compatibility executor for bounded study-note listing and reading.
///
/// The model-facing registry exposes [`StudyNoteListTool`] and
/// [`StudyNoteReadTool`] separately so every required field is unconditional.
/// This combined type remains available to older in-process callers only.
pub struct StudyNoteCatalogTool {
    catalog: Arc<dyn NoteCatalog>,
}

impl StudyNoteCatalogTool {
    pub fn new(catalog: Arc<dyn NoteCatalog>) -> Self {
        Self { catalog }
    }
}

/// Body-free, paginated discovery over an akh's current note heads.
pub struct StudyNoteListTool {
    catalog: Arc<dyn NoteCatalog>,
}

impl StudyNoteListTool {
    pub fn new(catalog: Arc<dyn NoteCatalog>) -> Self {
        Self { catalog }
    }
}

/// Full identity-based read of one current note head.
pub struct StudyNoteReadTool {
    catalog: Arc<dyn NoteCatalog>,
}

impl StudyNoteReadTool {
    pub fn new(catalog: Arc<dyn NoteCatalog>) -> Self {
        Self { catalog }
    }
}

const DEFAULT_NOTE_PAGE: u64 = 20;
const MAX_NOTE_PAGE: u64 = 50;
const MAX_CATALOG_QUERY_CHARS: usize = 256;
const MIN_SUPPORTING_QUOTE_CHARS: usize = 8;
const REVISION_HEX_CHARS: usize = 64;

fn catalog_operation_schema() -> Value {
    serde_json::json!({
        "type": "string",
        "enum": ["list", "read"]
    })
}

fn catalog_limit_schema() -> Value {
    serde_json::json!({
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_NOTE_PAGE,
        "default": DEFAULT_NOTE_PAGE
    })
}

fn catalog_offset_schema() -> Value {
    serde_json::json!({
        "type": "integer",
        "minimum": 0,
        "default": 0
    })
}

fn catalog_query_schema() -> Value {
    serde_json::json!({
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_CATALOG_QUERY_CHARS
    })
}

fn note_id_schema() -> Value {
    serde_json::json!({
        "type": "string",
        "pattern": "^note-[0-9a-f]{32}$"
    })
}

impl Tool for StudyNoteCatalogTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "study_note_catalog".into(),
            description: "Filter/list body-free study-note metadata or read one full note by stable id. Use list to discover challenge notes; use read only after selecting a note_id.".into(),
            parameters: vec![
                ToolParam {
                    name: "operation".into(),
                    description: "Either list or read.".into(),
                    required: true,
                },
                ToolParam {
                    name: "note_id".into(),
                    description: "Stable note id; required only for read.".into(),
                    required: false,
                },
                ToolParam {
                    name: "query".into(),
                    description: "Optional metadata-only filter for list.".into(),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    description: "List page size, 1-50 (default 20).".into(),
                    required: false,
                },
                ToolParam {
                    name: "offset".into(),
                    description: "Zero-based list offset (default 0).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        match input.require("operation", "study_note_catalog")? {
            "list" => {
                if input.get_value("note_id").is_some() {
                    return Err(catalog_error("note_id is not valid for list"));
                }
                let query = input.get("query").map(str::trim);
                if query.is_some_and(str::is_empty) {
                    return Err(catalog_error("query must not be empty"));
                }
                if query.is_some_and(|query| query.chars().count() > MAX_CATALOG_QUERY_CHARS) {
                    return Err(catalog_error(format!(
                        "query must contain at most {MAX_CATALOG_QUERY_CHARS} characters"
                    )));
                }
                let limit =
                    bounded_catalog_number(&input, "limit", DEFAULT_NOTE_PAGE, 1, MAX_NOTE_PAGE)?;
                let offset = bounded_catalog_number(&input, "offset", 0, 0, u64::MAX)?;
                let offset = usize::try_from(offset)
                    .map_err(|_| catalog_error("offset is too large for this host"))?;
                let limit = usize::try_from(limit)
                    .map_err(|_| catalog_error("limit is too large for this host"))?;

                let entries = self.catalog.catalog_entries();
                let catalog_total = entries.len();
                let matching: Vec<_> = entries
                    .into_iter()
                    .map(|entry| {
                        let frontmatter = entry.note.frontmatter.without("supporting_quotes");
                        (entry, frontmatter)
                    })
                    .filter(|(entry, frontmatter)| {
                        query.is_none_or(|query| {
                            catalog_metadata_matches(&entry.note, frontmatter, query)
                        })
                    })
                    .collect();
                let matched_total = matching.len();
                let page: Vec<Value> = matching
                    .into_iter()
                    .skip(offset)
                    .take(limit)
                    .map(|(entry, frontmatter)| {
                        let note = entry.note;
                        serde_json::json!({
                            "note_id": note.id,
                            "document_id": note.document.id,
                            "title": note.document.title,
                            "tags": note.document.tags,
                            "frontmatter": frontmatter,
                            "revision": entry.revision,
                            "stored_at": entry.stored_at,
                        })
                    })
                    .collect();
                let returned = page.len();
                let page_end = offset.saturating_add(returned).min(matched_total);
                let has_more = page_end < matched_total;
                let next_offset = has_more.then_some(page_end);
                Ok(ToolOutput::ok(
                    serde_json::json!({
                        "operation": "list",
                        "query": query,
                        "catalog_total": catalog_total,
                        "total": matched_total,
                        "offset": offset,
                        "limit": limit,
                        "returned": returned,
                        "has_more": has_more,
                        "next_offset": next_offset,
                        "notes": page,
                    })
                    .to_string(),
                ))
            }
            "read" => {
                if input.get_value("limit").is_some()
                    || input.get_value("offset").is_some()
                    || input.get_value("query").is_some()
                {
                    return Err(catalog_error(
                        "query, limit, and offset are not valid for read",
                    ));
                }
                let raw_id = input.require("note_id", "study_note_catalog")?;
                let note_id = NoteId::parse(raw_id.to_string())
                    .map_err(|error| catalog_error(error.to_string()))?;
                let entry = self
                    .catalog
                    .get_entry(&note_id)
                    .ok_or_else(|| catalog_error(format!("note {note_id} does not exist")))?;
                let note = entry.note;
                Ok(ToolOutput::ok(
                    serde_json::json!({
                        "operation": "read",
                        "note_id": note.id,
                        "revision": entry.revision,
                        "stored_at": entry.stored_at,
                        "document": note.document,
                        "frontmatter": note.frontmatter,
                        "body": note.body,
                        "wikilinks": note.wikilinks,
                        "tasks": note.tasks,
                    })
                    .to_string(),
                ))
            }
            other => Err(catalog_error(format!(
                "unknown operation {other:?}; expected list or read"
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "study_note_catalog".into(),
            description: "Browse or filter an akh's durable study-note catalog without loading bodies, or explicitly read one selected note by stable id.".into(),
            parameters: vec![
                ToolParamSchema::required_schema(
                    "operation",
                    "Catalog operation: list metadata or read one note.",
                    catalog_operation_schema(),
                ),
                ToolParamSchema::optional("note_id", "Stable note id required for read."),
                ToolParamSchema::optional_schema(
                    "query",
                    "Optional list filter over note id, title, tags, and non-archival frontmatter.",
                    catalog_query_schema(),
                ),
                ToolParamSchema::optional_schema(
                    "limit",
                    "List page size, from 1 through 50.",
                    catalog_limit_schema(),
                ),
                ToolParamSchema::optional_schema(
                    "offset",
                    "Zero-based list offset.",
                    catalog_offset_schema(),
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::ReadOnly,
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Read-only, bounded access to the akh's own study-note catalog."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

impl Tool for StudyNoteListTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "study_note_list".into(),
            description: "List body-free study-note metadata, optionally filtered and paginated. Select a stable note_id from this result before calling study_note_read.".into(),
            parameters: vec![
                ToolParam {
                    name: "query".into(),
                    description: "Optional metadata-only filter.".into(),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    description: "Page size, 1-50 (default 20).".into(),
                    required: false,
                },
                ToolParam {
                    name: "offset".into(),
                    description: "Zero-based list offset (default 0).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, mut input: ToolInput) -> AgentResult<ToolOutput> {
        reject_unexpected_params(&input, &["query", "limit", "offset"], list_error)?;
        input
            .params
            .insert("operation".into(), Value::String("list".into()));
        StudyNoteCatalogTool::new(Arc::clone(&self.catalog))
            .execute(engine, input)
            .map_err(|error| remap_tool_error(error, list_error))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "study_note_list".into(),
            description: "Browse or filter an akh's durable study-note catalog without loading note bodies or supporting quotes.".into(),
            parameters: vec![
                ToolParamSchema::optional_schema(
                    "query",
                    "Optional filter over note id, title, tags, and non-archival frontmatter.",
                    catalog_query_schema(),
                ),
                ToolParamSchema::optional_schema(
                    "limit",
                    "Page size, from 1 through 50.",
                    catalog_limit_schema(),
                ),
                ToolParamSchema::optional_schema(
                    "offset",
                    "Zero-based list offset.",
                    catalog_offset_schema(),
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::ReadOnly,
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Read-only, bounded metadata access to the akh's own study-note catalog."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

impl Tool for StudyNoteReadTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "study_note_read".into(),
            description: "Read one full current study-note head by a stable note_id selected from study_note_list or archival_search.".into(),
            parameters: vec![ToolParam {
                name: "note_id".into(),
                description: "Stable note id in note- plus 32 lowercase hexadecimal characters format.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, mut input: ToolInput) -> AgentResult<ToolOutput> {
        reject_unexpected_params(&input, &["note_id"], read_error)?;
        input
            .params
            .insert("operation".into(), Value::String("read".into()));
        StudyNoteCatalogTool::new(Arc::clone(&self.catalog))
            .execute(engine, input)
            .map_err(|error| remap_tool_error(error, read_error))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "study_note_read".into(),
            description: "Read the full current head of one explicitly selected study note, including its revision token for conflict-safe updates.".into(),
            parameters: vec![ToolParamSchema::required_schema(
                "note_id",
                "Stable note id selected from study_note_list or archival_search.",
                note_id_schema(),
            )],
            danger: DangerInfo {
                effect: ToolEffect::ReadOnly,
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Read-only identity-based access to one study note.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

fn reject_unexpected_params(
    input: &ToolInput,
    allowed: &[&str],
    error: fn(String) -> AgentError,
) -> AgentResult<()> {
    if let Some(name) = input
        .params
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
    {
        return Err(error(format!("unexpected parameter: {name}")));
    }
    Ok(())
}

fn remap_tool_error(error: AgentError, remap: fn(String) -> AgentError) -> AgentError {
    match error {
        AgentError::ToolExecution { message, .. } => remap(message),
        other => other,
    }
}

fn catalog_metadata_matches(note: &Note, frontmatter: &Frontmatter, query: &str) -> bool {
    let searchable = format!(
        "{} {} {} {} {}",
        note.id,
        note.document.id,
        note.document.title,
        note.document.tags.join(" "),
        frontmatter.to_block()
    )
    .to_lowercase();
    query
        .split_whitespace()
        .map(str::to_lowercase)
        .all(|term| searchable.contains(&term))
}

fn bounded_catalog_number(
    input: &ToolInput,
    name: &str,
    default: u64,
    minimum: u64,
    maximum: u64,
) -> AgentResult<u64> {
    let Some(raw) = input.get_value(name) else {
        return Ok(default);
    };
    let value = input
        .get_u64(name)
        .ok_or_else(|| catalog_error(format!("{name} must be an unsigned integer")))?;
    if value < minimum || value > maximum {
        return Err(catalog_error(format!(
            "{name} must be between {minimum} and {maximum}, got {raw}"
        )));
    }
    Ok(value)
}

fn citations_schema() -> Value {
    serde_json::json!({
        "type": "array",
        "minItems": 1,
        "description": "Passages supporting this note.",
        "items": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1},
                "locator": {"type": "string", "minLength": 1, "description": "Chapter, section, page, heading, or chunk id."},
                "quote": {"type": "string", "minLength": MIN_SUPPORTING_QUOTE_CHARS, "description": "Short supporting excerpt observed in an earlier retrieval result."}
            },
            "required": ["source_id", "locator", "quote"],
            "additionalProperties": false
        }
    })
}

fn string_array_schema(description: &str) -> Value {
    serde_json::json!({
        "type": "array",
        "description": description,
        "items": {"type": "string"}
    })
}

fn frontmatter_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "description": "Additional typed frontmatter. Values may be strings, numbers, booleans, or string arrays.",
        "additionalProperties": {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "array", "items": {"type": "string"}}
            ]
        }
    })
}

impl Tool for StudyNoteAppendTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "study_note_append".into(),
            description: "Create a source-backed study note after reading verified textbook/archive passages. Requires a title and citations; use study_note_revise for an existing note.".into(),
            parameters: vec![
                ToolParam {
                    name: "body".into(),
                    description: "The concise study note in your own words.".into(),
                    required: true,
                },
                ToolParam {
                    name: "citations".into(),
                    description: "Non-empty array of {source_id, locator, quote}.".into(),
                    required: true,
                },
                ToolParam {
                    name: "title".into(),
                    description: "Title for the new note.".into(),
                    required: true,
                },
                ToolParam {
                    name: "challenge_id".into(),
                    description: "Goal/task/challenge this note was created to solve.".into(),
                    required: false,
                },
                ToolParam {
                    name: "language".into(),
                    description: "BCP-47-ish source/note language label, such as en, de, or fr.".into(),
                    required: false,
                },
                ToolParam {
                    name: "tags".into(),
                    description: "Short topical tags.".into(),
                    required: false,
                },
                ToolParam {
                    name: "frontmatter".into(),
                    description: "Optional additional typed frontmatter object.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        if !self.allow_revision
            && (input.get_value("note_id").is_some()
                || input.get_value("expected_revision").is_some())
        {
            return Err(tool_error(
                "study_note_append creates new notes only; use study_note_revise for note_id updates",
            ));
        }
        let body = input
            .require("body", "study_note_append")?
            .trim()
            .to_string();
        if body.is_empty() {
            return Err(tool_error("body must not be empty"));
        }
        let citations = parse_citations(&input)?;
        let tags = input.get_str_list("tags");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let raw_id = optional_string(&input, "note_id")?;
        let expected_revision = optional_string(&input, "expected_revision")?;
        if raw_id.is_none() && expected_revision.is_some() {
            return Err(tool_error(
                "expected_revision is valid only when note_id identifies an existing note",
            ));
        }
        if raw_id.is_some() && expected_revision.is_none() {
            return Err(tool_error(
                "expected_revision is required when revising an existing note; read the note with study_note_read first",
            ));
        }
        if let Some(revision) = expected_revision.as_deref()
            && !is_revision_token(revision)
        {
            return Err(tool_error(format!(
                "expected_revision must be exactly {REVISION_HEX_CHARS} lowercase hexadecimal characters"
            )));
        }

        let mut note = if let Some(raw_id) = raw_id.as_deref() {
            let id = NoteId::parse(raw_id.to_string()).map_err(|e| tool_error(e.to_string()))?;
            self.store
                .get(&id)
                .ok_or_else(|| tool_error(format!("note {id} does not exist")))?
        } else {
            let title = input
                .get("title")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| tool_error("title is required when note_id is omitted"))?;
            Note::new(
                DocumentRecord {
                    id: slugify(title),
                    title: title.to_string(),
                    source: DocumentSource::Inline,
                    format: ContentFormat::PlainText,
                    tags: tags.clone(),
                    chunk_count: 1,
                    triple_count: 0,
                    ingested_at: now,
                },
                String::new(),
            )
        };

        if let Some(title) = input.get("title").map(str::trim).filter(|s| !s.is_empty()) {
            note.document.title = title.to_string();
            note.document.id = slugify(title);
        }
        if !tags.is_empty() {
            note.document.tags = tags.clone();
        }
        note.set_body(body);

        let mut frontmatter = note.frontmatter.clone();
        if input.get_value("frontmatter").is_some() {
            for (key, value) in parse_frontmatter(&input)?.iter() {
                frontmatter.insert(key, value.clone());
            }
        }
        if frontmatter.get("type").is_none() {
            frontmatter.insert("type", FrontmatterValue::Text("StudyNote".into()));
        }
        frontmatter.insert(
            "sources",
            FrontmatterValue::List(unique_source_ids(&citations)),
        );
        frontmatter.insert(
            "citations",
            FrontmatterValue::List(
                citations
                    .iter()
                    .map(|c| format!("{}#{}", c.source_id, c.locator))
                    .collect(),
            ),
        );
        if let Some(challenge_id) = input.get("challenge_id") {
            frontmatter.insert(
                "challenge",
                FrontmatterValue::Link(Wikilink {
                    target: challenge_id.to_string(),
                    alias: None,
                }),
            );
        }
        if let Some(language) = input.get("language") {
            frontmatter.insert("language", FrontmatterValue::Text(language.to_string()));
        }
        if !tags.is_empty() {
            frontmatter.insert("tags", FrontmatterValue::List(tags));
        }
        frontmatter.insert(
            "supporting_quotes",
            FrontmatterValue::List(citations.iter().map(|c| c.quote.clone()).collect()),
        );
        note.set_frontmatter(frontmatter);

        let receipt = self
            .store
            .append_revision_checked(note.clone(), expected_revision.as_deref())
            .map_err(|e| tool_error(e.to_string()))?;

        let projection = match crate::schema::reproject_note(engine, &note) {
            Ok(report) => {
                let diff = report.diff.unwrap_or_default();
                serde_json::json!({
                    "status": "projected",
                    "typed_properties": report.typed_property_count,
                    "frontier_properties": report.frontier_property_count,
                    "added": diff.added,
                    "retained": diff.retained,
                    "retracted": diff.retracted,
                    "shape_violations": report.shape_violations.len()
                })
            }
            Err(error) => serde_json::json!({
                "status": "deferred",
                "error": error.to_string()
            }),
        };

        let result = serde_json::json!({
            "note_id": receipt.note_id,
            "revision": receipt.revision,
            "created": receipt.created,
            "stored_at": receipt.stored_at,
            "citation_count": citations.len(),
            "projection": projection
        });
        Ok(ToolOutput::ok(result.to_string()))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "study_note_append".into(),
            description: "Create a provenance-backed study note after retrieving and checking source passages. Use study_note_revise for conflict-safe updates; conversation capture is automatic elsewhere.".into(),
            parameters: vec![
                ToolParamSchema::required("body", "Concise source-grounded note body."),
                ToolParamSchema::required_schema(
                    "citations",
                    "At least one source locator and observed supporting quote.",
                    citations_schema(),
                ),
                ToolParamSchema::required("title", "Title for the new study note."),
                ToolParamSchema::optional("challenge_id", "Goal/task/challenge this supports."),
                ToolParamSchema::optional("language", "Language label, such as en, de, or fr."),
                ToolParamSchema::optional_schema("tags", "Topical tags.", string_array_schema("Topical tags.")),
                ToolParamSchema::optional_schema(
                    "frontmatter",
                    "Additional typed frontmatter.",
                    frontmatter_schema(),
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([
                    Capability::WriteFilesystem,
                    Capability::WriteKg,
                    Capability::MemoryAccess,
                    Capability::ProvenanceAccess,
                ]),
                description: "Creates an immutable first note revision and refreshes its derived KG projection.".into(),
                shadow_triggers: vec!["study note".into(), "learn from source".into()],
            },
            source: ToolSource::Native,
        }
    }
}

impl Tool for StudyNoteReviseTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "study_note_revise".into(),
            description: "Append an immutable, source-backed revision of an existing study note. Read the note first and pass its exact expected_revision token so a stale turn cannot overwrite newer knowledge.".into(),
            parameters: vec![
                ToolParam {
                    name: "note_id".into(),
                    description: "Stable id of the existing note.".into(),
                    required: true,
                },
                ToolParam {
                    name: "expected_revision".into(),
                    description: "Current 64-character revision token returned by study_note_read.".into(),
                    required: true,
                },
                ToolParam {
                    name: "body".into(),
                    description: "The revised concise study note in your own words.".into(),
                    required: true,
                },
                ToolParam {
                    name: "citations".into(),
                    description: "Non-empty array of {source_id, locator, quote}.".into(),
                    required: true,
                },
                ToolParam {
                    name: "title".into(),
                    description: "Optional replacement title.".into(),
                    required: false,
                },
                ToolParam {
                    name: "challenge_id".into(),
                    description: "Goal/task/challenge this note supports.".into(),
                    required: false,
                },
                ToolParam {
                    name: "language".into(),
                    description: "BCP-47-ish source/note language label, such as en, de, or fr.".into(),
                    required: false,
                },
                ToolParam {
                    name: "tags".into(),
                    description: "Short topical tags.".into(),
                    required: false,
                },
                ToolParam {
                    name: "frontmatter".into(),
                    description: "Optional additional typed frontmatter object.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        if input.get_value("note_id").is_none() || input.get_value("expected_revision").is_none() {
            return Err(revise_error(
                "note_id and expected_revision are required; read the current note with study_note_read first",
            ));
        }
        let writer = StudyNoteAppendTool::for_revision(Arc::clone(&self.store));
        writer.execute(engine, input).map_err(|error| match error {
            AgentError::ToolExecution { message, .. } => revise_error(message),
            other => other,
        })
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "study_note_revise".into(),
            description: "Append a provenance-backed revision against an observed note head. A stale expected_revision fails without advancing history.".into(),
            parameters: vec![
                ToolParamSchema::required_schema(
                    "note_id",
                    "Stable id of the existing note.",
                    note_id_schema(),
                ),
                ToolParamSchema::required_schema(
                    "expected_revision",
                    "Current revision token returned by study_note_read.",
                    serde_json::json!({
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$"
                    }),
                ),
                ToolParamSchema::required("body", "Revised concise source-grounded note body."),
                ToolParamSchema::required_schema(
                    "citations",
                    "At least one source locator and observed supporting quote.",
                    citations_schema(),
                ),
                ToolParamSchema::optional("title", "Optional replacement title."),
                ToolParamSchema::optional("challenge_id", "Goal/task/challenge this supports."),
                ToolParamSchema::optional("language", "Language label, such as en, de, or fr."),
                ToolParamSchema::optional_schema(
                    "tags",
                    "Topical tags.",
                    string_array_schema("Topical tags."),
                ),
                ToolParamSchema::optional_schema(
                    "frontmatter",
                    "Additional typed frontmatter merged into the current note.",
                    frontmatter_schema(),
                ),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([
                    Capability::WriteFilesystem,
                    Capability::WriteKg,
                    Capability::MemoryAccess,
                    Capability::ProvenanceAccess,
                ]),
                description: "Appends a checked immutable note revision and refreshes its derived KG projection.".into(),
                shadow_triggers: vec!["revise study note".into(), "update learned fact".into()],
            },
            source: ToolSource::Native,
        }
    }
}

fn parse_citations(input: &ToolInput) -> AgentResult<Vec<CitationInput>> {
    let value = input
        .get_value("citations")
        .ok_or_else(|| tool_error("citations is required"))?;
    let citations: Vec<CitationInput> = serde_json::from_value(value.clone())
        .map_err(|e| tool_error(format!("invalid citations: {e}")))?;
    if citations.is_empty() {
        return Err(tool_error(
            "citations must contain at least one source locator",
        ));
    }
    if citations
        .iter()
        .any(|c| c.source_id.trim().is_empty() || c.locator.trim().is_empty())
    {
        return Err(tool_error(
            "citation source_id, locator, and quote must not be empty",
        ));
    }
    if citations
        .iter()
        .any(|c| c.quote.trim().chars().count() < MIN_SUPPORTING_QUOTE_CHARS)
    {
        return Err(tool_error(format!(
            "citation quote must contain at least {MIN_SUPPORTING_QUOTE_CHARS} characters"
        )));
    }
    Ok(citations)
}

fn optional_string(input: &ToolInput, name: &str) -> AgentResult<Option<String>> {
    match input.get_value(name) {
        None => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.trim().into())),
        Some(Value::String(_)) => Err(tool_error(format!("{name} must not be empty"))),
        Some(_) => Err(tool_error(format!("{name} must be a string"))),
    }
}

fn is_revision_token(value: &str) -> bool {
    value.len() == REVISION_HEX_CHARS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn unique_source_ids(citations: &[CitationInput]) -> Vec<String> {
    let mut seen = HashSet::new();
    citations
        .iter()
        .filter_map(|citation| {
            seen.insert(citation.source_id.clone())
                .then(|| citation.source_id.clone())
        })
        .collect()
}

fn parse_frontmatter(input: &ToolInput) -> AgentResult<Frontmatter> {
    let Some(value) = input.get_value("frontmatter") else {
        return Ok(Frontmatter::new());
    };
    let object = value
        .as_object()
        .ok_or_else(|| tool_error("frontmatter must be a JSON object"))?;
    let mut result = Frontmatter::new();
    for (key, value) in object {
        let converted = match value {
            Value::String(s) => FrontmatterValue::Text(s.clone()),
            Value::Number(n) => n
                .as_f64()
                .map(FrontmatterValue::Number)
                .ok_or_else(|| tool_error(format!("frontmatter {key} is not a finite number")))?,
            Value::Bool(b) => FrontmatterValue::Checkbox(*b),
            Value::Array(items) => {
                let values: Option<Vec<String>> = items
                    .iter()
                    .map(|item| item.as_str().map(str::to_string))
                    .collect();
                FrontmatterValue::List(values.ok_or_else(|| {
                    tool_error(format!("frontmatter {key} array must contain only strings"))
                })?)
            }
            _ => {
                return Err(tool_error(format!(
                    "frontmatter {key} must be string, number, boolean, or string array"
                )));
            }
        };
        result.insert(key, converted);
    }
    Ok(result)
}

fn tool_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "study_note_append".into(),
        message: message.into(),
    }
}

fn catalog_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "study_note_catalog".into(),
        message: message.into(),
    }
}

fn list_error(message: String) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "study_note_list".into(),
        message,
    }
}

fn read_error(message: String) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "study_note_read".into(),
        message,
    }
}

fn revise_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "study_note_revise".into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::{Engine, EngineConfig};

    fn fixture() -> (Engine, tempfile::TempDir, Arc<PersistentNoteCatalog>) {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        let store = Arc::new(PersistentNoteCatalog::open(tmp.path().join("notes")).unwrap());
        (engine, tmp, store)
    }

    #[test]
    fn requires_grounding_citations() {
        let (engine, _tmp, store) = fixture();
        let tool = StudyNoteAppendTool::new(store);
        let input = ToolInput::new()
            .with_param("title", "Mutexes")
            .with_param("body", "A mutex serializes access.")
            .with_value("citations", serde_json::json!([]));
        assert!(tool.execute(&engine, input).is_err());
    }

    #[test]
    fn create_and_revision_contracts_have_no_conditional_required_fields() {
        let (_engine, _tmp, store) = fixture();
        let create = StudyNoteAppendTool::new(Arc::clone(&store)).manifest();
        let revise = StudyNoteReviseTool::new(store).manifest();
        let create_required: HashSet<_> = create
            .parameters
            .iter()
            .filter(|parameter| parameter.required)
            .map(|parameter| parameter.name.as_str())
            .collect();
        let revise_required: HashSet<_> = revise
            .parameters
            .iter()
            .filter(|parameter| parameter.required)
            .map(|parameter| parameter.name.as_str())
            .collect();
        assert_eq!(
            create_required,
            HashSet::from(["body", "citations", "title"])
        );
        assert!(!create.parameters.iter().any(|p| p.name == "note_id"));
        assert!(
            !create
                .parameters
                .iter()
                .any(|p| p.name == "expected_revision")
        );
        assert_eq!(
            revise_required,
            HashSet::from(["body", "citations", "expected_revision", "note_id"])
        );
    }

    #[test]
    fn requires_a_supporting_quote_for_every_citation() {
        let (engine, _tmp, store) = fixture();
        let tool = StudyNoteAppendTool::new(store);
        let input = ToolInput::new()
            .with_param("title", "Mutexes")
            .with_param("body", "A mutex serializes access.")
            .with_value(
                "citations",
                serde_json::json!([{
                    "source_id": "book:rust",
                    "locator": "ch. 16 / p. 301"
                }]),
            );

        let error = tool.execute(&engine, input).unwrap_err().to_string();
        assert!(error.contains("quote"), "unexpected error: {error}");

        let too_short = ToolInput::new()
            .with_param("title", "Mutexes")
            .with_param("body", "A mutex serializes access.")
            .with_value(
                "citations",
                serde_json::json!([{
                    "source_id": "book:rust",
                    "locator": "p. 301",
                    "quote": "mutex"
                }]),
            );
        let error = tool.execute(&engine, too_short).unwrap_err().to_string();
        assert!(error.contains("at least 8"), "unexpected error: {error}");
    }

    #[test]
    fn create_executor_rejects_revision_inputs_and_unknown_citation_fields() {
        let (engine, _tmp, store) = fixture();
        let writer = StudyNoteAppendTool::new(store);

        let revision_input = writer
            .execute(
                &engine,
                ToolInput::new().with_param("note_id", "note-00000000000000000000000000000000"),
            )
            .unwrap_err()
            .to_string();
        assert!(revision_input.contains("creates new notes only"));

        let unknown_citation_field = writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Strict citation")
                    .with_param("body", "The source-backed value is QZ-731.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:alpha",
                            "locator": "chapter 3",
                            "quote": "The source-backed value is QZ-731.",
                            "invented_confidence": 1.0
                        }]),
                    ),
            )
            .unwrap_err()
            .to_string();
        assert!(unknown_citation_field.contains("unknown field"));
    }

    #[test]
    fn creates_and_revises_one_stable_note() {
        let (engine, _tmp, store) = fixture();
        let writer = StudyNoteAppendTool::new(Arc::clone(&store));
        let reviser = StudyNoteReviseTool::new(Arc::clone(&store));
        let first = writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Locking")
                    .with_param("body", "Use a mutex for exclusive access.")
                    .with_param("language", "en")
                    .with_param("challenge_id", "challenge:locking")
                    .with_value("tags", serde_json::json!(["rust", "concurrency"]))
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:rust",
                            "locator": "ch. 16 / p. 301",
                            "quote": "A mutex allows only one thread to access data."
                        }]),
                    ),
            )
            .unwrap();
        let first: Value = serde_json::from_str(&first.result).unwrap();
        let id = first["note_id"].as_str().unwrap().to_string();
        let revision = first["revision"].as_str().unwrap().to_string();

        let second = reviser
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", &id)
                    .with_param("expected_revision", revision)
                    .with_param(
                        "body",
                        "Use RwLock when reads dominate and writes are rare.",
                    )
                    .with_value("frontmatter", serde_json::json!({"status": "revised"}))
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:rust",
                            "locator": "ch. 16 / p. 305",
                            "quote": "An RwLock permits many readers or one writer."
                        }]),
                    ),
            )
            .unwrap();
        let second: Value = serde_json::from_str(&second.result).unwrap();
        assert_eq!(second["note_id"], id);

        let parsed_id = NoteId::parse(id).unwrap();
        assert_eq!(store.revision_count(&parsed_id).unwrap(), 2);
        let revised = store.get(&parsed_id).unwrap();
        assert!(revised.body.contains("RwLock"));
        assert_eq!(
            revised.frontmatter.get("status"),
            Some(&FrontmatterValue::Text("revised".into()))
        );
        assert_eq!(
            revised.frontmatter.get("challenge"),
            Some(&FrontmatterValue::Link(Wikilink {
                target: "challenge:locking".into(),
                alias: None,
            }))
        );
    }

    #[test]
    fn identical_retry_is_idempotent() {
        let (engine, _tmp, store) = fixture();
        let writer = StudyNoteAppendTool::new(Arc::clone(&store));
        let reviser = StudyNoteReviseTool::new(Arc::clone(&store));
        let make = || {
            ToolInput::new()
                .with_param("title", "Stable note")
                .with_param("body", "The source says X.")
                .with_value(
                    "citations",
                    serde_json::json!([{
                        "source_id": "doc:x",
                        "locator": "section 1",
                        "quote": "The source says X."
                    }]),
                )
        };
        let first = writer.execute(&engine, make()).unwrap();
        let first_json: Value = serde_json::from_str(&first.result).unwrap();
        let id = first_json["note_id"].as_str().unwrap();
        let revision = first_json["revision"].as_str().unwrap();

        let retry = reviser
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", id)
                    .with_param("expected_revision", revision)
                    .with_param("body", "The source says X.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "doc:x",
                            "locator": "section 1",
                            "quote": "The source says X."
                        }]),
                    ),
            )
            .unwrap();
        let retry_json: Value = serde_json::from_str(&retry.result).unwrap();
        assert_eq!(retry_json["created"], false);
        let parsed = NoteId::parse(id.to_string()).unwrap();
        assert_eq!(store.revision_count(&parsed).unwrap(), 1);
    }

    #[test]
    fn revision_requires_the_observed_head_and_rejects_stale_writes() {
        let (engine, _tmp, store) = fixture();
        let writer = StudyNoteAppendTool::new(Arc::clone(&store));
        let reviser = StudyNoteReviseTool::new(Arc::clone(&store));
        let created = writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Conflict-safe note")
                    .with_param("body", "Version one comes from the source.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:one",
                            "locator": "section 1",
                            "quote": "Version one comes from the source."
                        }]),
                    ),
            )
            .unwrap();
        let created: Value = serde_json::from_str(&created.result).unwrap();
        let note_id = created["note_id"].as_str().unwrap();
        let first_revision = created["revision"].as_str().unwrap();

        let missing_token = reviser
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", note_id)
                    .with_param("body", "Version two comes from a newer source.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:two",
                            "locator": "section 2",
                            "quote": "Version two comes from a newer source."
                        }]),
                    ),
            )
            .unwrap_err()
            .to_string();
        assert!(missing_token.contains("expected_revision"));

        let updated = reviser
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", note_id)
                    .with_param("expected_revision", first_revision)
                    .with_param("body", "Version two comes from a newer source.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:two",
                            "locator": "section 2",
                            "quote": "Version two comes from a newer source."
                        }]),
                    ),
            )
            .unwrap();
        let updated: Value = serde_json::from_str(&updated.result).unwrap();
        assert_ne!(updated["revision"], first_revision);

        let stale = reviser
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", note_id)
                    .with_param("expected_revision", first_revision)
                    .with_param("body", "A stale model turn should not win.")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:stale",
                            "locator": "section 9",
                            "quote": "A stale model turn should not win."
                        }]),
                    ),
            )
            .unwrap_err()
            .to_string();
        assert!(
            stale.contains("revision conflict"),
            "unexpected error: {stale}"
        );
        assert_eq!(
            store
                .get(&NoteId::parse(note_id.to_string()).unwrap())
                .unwrap()
                .body,
            "Version two comes from a newer source."
        );
    }

    #[test]
    fn narrow_list_and_read_tools_preserve_catalog_privacy_and_identity() {
        let (engine, _tmp, store) = fixture();
        let writer = StudyNoteAppendTool::new(Arc::clone(&store));
        let written = writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Challenge Alpha")
                    .with_param("body", "The private answer is keph QZ-731.")
                    .with_param("challenge_id", "challenge:alpha")
                    .with_value("tags", serde_json::json!(["textbook", "alpha"]))
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:alpha",
                            "locator": "chapter 3",
                            "quote": "The private answer is keph QZ-731."
                        }]),
                    ),
            )
            .unwrap();
        let note_id = serde_json::from_str::<Value>(&written.result).unwrap()["note_id"]
            .as_str()
            .unwrap()
            .to_string();
        writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Challenge Beta")
                    .with_param("body", "The second private answer is keph LM-204.")
                    .with_param("challenge_id", "challenge:beta")
                    .with_value(
                        "citations",
                        serde_json::json!([{
                            "source_id": "book:beta",
                            "locator": "chapter 9",
                            "quote": "The second private answer is keph LM-204."
                        }]),
                    ),
            )
            .unwrap();
        let list = StudyNoteListTool::new(store.clone());
        let read = StudyNoteReadTool::new(store);

        let listed = list
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("query", "challenge:alpha")
                    .with_value("limit", serde_json::json!(10)),
            )
            .unwrap();
        let listed: Value = serde_json::from_str(&listed.result).unwrap();
        assert_eq!(listed["total"], 1);
        assert_eq!(listed["catalog_total"], 2);
        assert_eq!(listed["returned"], 1);
        assert_eq!(listed["has_more"], false);
        assert!(listed["next_offset"].is_null());
        assert_eq!(listed["notes"][0]["note_id"], note_id);
        assert_eq!(
            listed["notes"][0]["revision"],
            serde_json::from_str::<Value>(&written.result).unwrap()["revision"]
        );
        assert!(listed["notes"][0]["stored_at"].is_u64());
        assert_eq!(
            listed["notes"][0]["frontmatter"]["challenge"]["link"]["target"],
            "challenge:alpha"
        );
        assert!(
            !listed.to_string().contains("private answer"),
            "catalog listing must not leak a note body"
        );

        let body_only_query = list
            .execute(
                &engine,
                ToolInput::new().with_param("query", "private answer"),
            )
            .unwrap();
        let body_only_query: Value = serde_json::from_str(&body_only_query.result).unwrap();
        assert_eq!(body_only_query["catalog_total"], 2);
        assert_eq!(body_only_query["total"], 0);

        let first_page = list
            .execute(
                &engine,
                ToolInput::new().with_value("limit", serde_json::json!(1)),
            )
            .unwrap();
        let first_page: Value = serde_json::from_str(&first_page.result).unwrap();
        assert_eq!(first_page["total"], 2);
        assert_eq!(first_page["returned"], 1);
        assert_eq!(first_page["has_more"], true);
        assert_eq!(first_page["next_offset"], 1);

        let second_page = list
            .execute(
                &engine,
                ToolInput::new()
                    .with_value("limit", serde_json::json!(1))
                    .with_value("offset", serde_json::json!(1)),
            )
            .unwrap();
        let second_page: Value = serde_json::from_str(&second_page.result).unwrap();
        assert_eq!(second_page["returned"], 1);
        assert_eq!(second_page["has_more"], false);
        assert!(second_page["next_offset"].is_null());

        let read = read
            .execute(&engine, ToolInput::new().with_param("note_id", note_id))
            .unwrap();
        let read: Value = serde_json::from_str(&read.result).unwrap();
        assert_eq!(read["body"], "The private answer is keph QZ-731.");
        assert!(read["revision"].is_string());
        assert!(read["stored_at"].is_u64());
    }

    #[test]
    fn catalog_rejects_cross_operation_and_unbounded_inputs() {
        let (engine, _tmp, store) = fixture();
        let catalog = StudyNoteCatalogTool::new(store);
        assert!(
            catalog
                .execute(
                    &engine,
                    ToolInput::new()
                        .with_param("operation", "list")
                        .with_value("limit", serde_json::json!(51)),
                )
                .is_err()
        );
        assert!(
            catalog
                .execute(
                    &engine,
                    ToolInput::new()
                        .with_param("operation", "read")
                        .with_param("query", "alpha")
                        .with_value("offset", serde_json::json!(0)),
                )
                .is_err()
        );
        assert!(
            catalog
                .execute(&engine, ToolInput::new().with_param("operation", "delete"),)
                .is_err()
        );
    }

    #[test]
    fn narrow_catalog_contracts_have_no_conditional_or_cross_operation_inputs() {
        let (engine, _tmp, store) = fixture();
        let list = StudyNoteListTool::new(store.clone());
        let read = StudyNoteReadTool::new(store);
        let list_manifest = list.manifest();
        let read_manifest = read.manifest();

        assert_eq!(list_manifest.name, "study_note_list");
        assert!(
            list_manifest
                .parameters
                .iter()
                .all(|parameter| !parameter.required)
        );
        assert!(!list_manifest.parameters.iter().any(|p| p.name == "note_id"));
        assert_eq!(read_manifest.name, "study_note_read");
        assert_eq!(read_manifest.parameters.len(), 1);
        assert!(read_manifest.parameters[0].required);
        assert_eq!(
            read_manifest.parameters[0]
                .schema
                .as_ref()
                .and_then(|schema| schema.get("pattern"))
                .and_then(Value::as_str),
            Some("^note-[0-9a-f]{32}$")
        );

        let list_error = list
            .execute(
                &engine,
                ToolInput::new().with_param("note_id", "note-00000000000000000000000000000000"),
            )
            .unwrap_err();
        assert!(matches!(
            list_error,
            AgentError::ToolExecution { ref tool_name, .. } if tool_name == "study_note_list"
        ));
        let read_error = read
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("note_id", "not-a-note-id")
                    .with_param("query", "alpha"),
            )
            .unwrap_err();
        assert!(matches!(
            read_error,
            AgentError::ToolExecution { ref tool_name, .. } if tool_name == "study_note_read"
        ));
    }
}
