//! Document ingestion pipeline.
//!
//! Orchestrates: parse → chunk → symbols → structural triples → NLP extraction
//! → VSA embeddings → provenance → catalog.

use std::path::Path;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::library::catalog::{LibraryCatalog, slugify};
use crate::library::chunker::{ChunkConfig, normalize_chunks};
use crate::library::error::{LibraryError, LibraryResult};
use crate::library::model::*;
use crate::library::parser;
use crate::library::predicates::LibraryPredicates;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_label;

/// Configuration for the ingestion pipeline.
pub struct IngestConfig {
    /// Override the detected title.
    pub title: Option<String>,
    /// Tags to assign to the document.
    pub tags: Vec<String>,
    /// Override the detected format.
    pub format: Option<ContentFormat>,
    /// Chunk normalization settings.
    pub chunk_config: ChunkConfig,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            title: None,
            tags: Vec::new(),
            format: None,
            chunk_config: ChunkConfig::default(),
        }
    }
}

/// Result of a successful document ingestion.
pub struct IngestResult {
    /// The document's catalog record.
    pub record: DocumentRecord,
    /// The document's root symbol ID in the KG.
    pub document_symbol: SymbolId,
    /// Total triples created.
    pub triple_count: usize,
    /// Total chunks after normalization.
    pub chunk_count: usize,
}

/// Ingest a document from raw bytes with a known source.
///
/// This is the main entry point for the ingestion pipeline. It:
/// 1. Detects format and parses the document
/// 2. Normalizes chunks to target size
/// 3. Creates KG symbols for the document and its structural elements
/// 4. Creates structural triples linking elements
/// 5. Runs NLP extraction per chunk
/// 6. Creates VSA embeddings per chunk
/// 7. Stores provenance records
/// 8. Registers the document in the catalog
pub fn ingest_document(
    engine: &Engine,
    catalog: &mut LibraryCatalog,
    data: &[u8],
    source: DocumentSource,
    config: IngestConfig,
) -> LibraryResult<IngestResult> {
    // 1. Determine format.
    let format = config.format.unwrap_or_else(|| {
        let source_str = source.to_string();
        parser::detect_format(&source_str).unwrap_or(ContentFormat::PlainText)
    });

    // 2. Parse.
    let parser = parser::parser_for(format)?;
    let parsed = parser.parse(data)?;

    if parsed.raw_chunks.is_empty() {
        return Err(LibraryError::EmptyDocument {
            origin: source.to_string(),
        });
    }

    // 3. Determine title and slug.
    let title = config
        .title
        .or(parsed.metadata.title.clone())
        .unwrap_or_else(|| {
            // Fallback: use first 60 chars of first chunk.
            let first = &parsed.raw_chunks[0].text;
            if first.len() > 60 {
                format!("{}...", &first[..60])
            } else {
                first.clone()
            }
        });
    let slug = slugify(&title);

    // Check for duplicates before doing expensive work.
    if catalog.get(&slug).is_some() {
        return Err(LibraryError::Duplicate { id: slug });
    }

    // 4. Normalize chunks.
    let chunks = normalize_chunks(&parsed.raw_chunks, &config.chunk_config);

    // 5. Initialize library predicates.
    let preds = LibraryPredicates::init(engine).map_err(|e| LibraryError::IngestFailed {
        document: slug.clone(),
        message: format!("predicate init: {e}"),
    })?;

    // 6. Create document symbol + metadata triples.
    let doc_sym = create_entity(engine, &title, &slug)?;
    let mut triple_count = 0usize;

    // Metadata triples.
    triple_count += add_metadata_triples(
        engine,
        doc_sym,
        &parsed.metadata,
        &preds,
        &source,
        format,
        &config.tags,
        &slug,
    )?;

    // 7. Create structural elements + triples.
    let mut chapter_symbols: Vec<(usize, SymbolId)> = Vec::new();
    let mut section_symbols: Vec<(usize, usize, SymbolId)> = Vec::new();

    for el in &parsed.elements {
        match &el.kind {
            ElementKind::Document => {} // Already handled.
            ElementKind::Chapter { ordinal } => {
                let label = format!("ch:{ordinal}:{}", el.heading);
                let sym = create_entity(engine, &label, &slug)?;
                add_triple(engine, doc_sym, preds.has_chapter, sym, &slug)?;
                triple_count += 1;
                chapter_symbols.push((*ordinal, sym));
            }
            ElementKind::Section { chapter, ordinal } => {
                let label = format!("sec:{chapter}.{ordinal}:{}", el.heading);
                let sym = create_entity(engine, &label, &slug)?;
                // Link section to its chapter if we have one.
                if let Some((_, ch_sym)) = chapter_symbols.iter().find(|(o, _)| o == chapter) {
                    add_triple(engine, *ch_sym, preds.has_section, sym, &slug)?;
                    triple_count += 1;
                }
                section_symbols.push((*chapter, *ordinal, sym));
            }
            ElementKind::Paragraph { .. } => {} // Handled in chunk loop.
        }
    }

    // 8. Per-chunk: create paragraph symbols, structural triples, NLP extraction, VSA.
    let mut prev_chunk_sym: Option<SymbolId> = None;

    for chunk in &chunks {
        let para_label = format!("para:{slug}:{}", chunk.index);
        let para_sym = create_entity(engine, &para_label, &slug)?;

        // Link to document.
        add_triple(engine, doc_sym, preds.has_paragraph, para_sym, &slug)?;
        triple_count += 1;

        // Link to chapter if applicable.
        if let Some((_, ch_sym)) = chapter_symbols.iter().find(|(o, _)| *o == chunk.chapter) {
            add_triple(engine, *ch_sym, preds.has_paragraph, para_sym, &slug)?;
            triple_count += 1;
        }

        // Chain: previous chunk -> next_chunk -> this chunk.
        if let Some(prev) = prev_chunk_sym {
            add_triple(engine, prev, preds.next_chunk, para_sym, &slug)?;
            triple_count += 1;
        }
        prev_chunk_sym = Some(para_sym);

        // Store chunk index as entity.
        let idx_label = format!("{}", chunk.index);
        let idx_sym = create_entity(engine, &idx_label, &slug)?;
        add_triple(engine, para_sym, preds.chunk_index, idx_sym, &slug)?;
        triple_count += 1;

        // NLP extraction: run regex-based triple extraction on chunk text.
        triple_count += run_nlp_extraction(engine, &chunk.text, &slug)?;

        // VSA embedding: encode the chunk text and insert into item memory.
        if let Ok(vec) = encode_label(engine.ops(), &chunk.text) {
            engine.item_memory().insert(para_sym, vec);
        }

        // Provenance: record document ingestion origin.
        store_provenance(engine, para_sym, &slug, format, chunk.index as u32);
    }

    // 9. Register in catalog.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let record = DocumentRecord {
        id: slug.clone(),
        title: title.clone(),
        source,
        format,
        tags: config.tags,
        chunk_count: chunks.len(),
        triple_count,
        ingested_at: now,
    };
    catalog.add(record.clone())?;

    Ok(IngestResult {
        record,
        document_symbol: doc_sym,
        triple_count,
        chunk_count: chunks.len(),
    })
}

/// Ingest a document from a filesystem path.
pub fn ingest_file(
    engine: &Engine,
    catalog: &mut LibraryCatalog,
    path: &Path,
    config: IngestConfig,
) -> LibraryResult<IngestResult> {
    let data = std::fs::read(path).map_err(|e| LibraryError::Io { source: e })?;
    let source = DocumentSource::File(path.display().to_string());

    // Override format from extension if not explicitly set.
    let config = if config.format.is_none() {
        IngestConfig {
            format: parser::detect_format(&path.display().to_string()),
            ..config
        }
    } else {
        config
    };

    ingest_document(engine, catalog, &data, source, config)
}

/// Ingest a document from a URL via HTTP GET.
pub fn ingest_url(
    engine: &Engine,
    catalog: &mut LibraryCatalog,
    url: &str,
    config: IngestConfig,
) -> LibraryResult<IngestResult> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| LibraryError::FetchError {
            url: url.into(),
            message: e.to_string(),
        })?;

    // Detect format from Content-Type.
    let content_type = response.content_type().to_string();
    let format_from_ct = parser::detect_format_from_content_type(&content_type);

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| LibraryError::FetchError {
            url: url.into(),
            message: format!("read body: {e}"),
        })?;

    let source = DocumentSource::Url(url.into());

    let config = if config.format.is_none() {
        IngestConfig {
            format: format_from_ct.or_else(|| parser::detect_format(url)),
            ..config
        }
    } else {
        config
    };

    ingest_document(engine, catalog, &data, source, config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an entity symbol, tolerating duplicate labels by looking up existing.
fn create_entity(engine: &Engine, label: &str, _slug: &str) -> LibraryResult<SymbolId> {
    engine
        .resolve_or_create_entity(label)
        .map_err(|e| LibraryError::IngestFailed {
            document: _slug.into(),
            message: format!("create entity '{label}': {e}"),
        })
}

/// Add a single triple.
fn add_triple(
    engine: &Engine,
    s: SymbolId,
    p: SymbolId,
    o: SymbolId,
    _slug: &str,
) -> LibraryResult<()> {
    let triple = Triple::new(s, p, o);
    engine
        .add_triple(&triple)
        .map_err(|e| LibraryError::IngestFailed {
            document: _slug.into(),
            message: format!("add triple: {e}"),
        })?;
    Ok(())
}

/// Add metadata triples for the document.
fn add_metadata_triples(
    engine: &Engine,
    doc_sym: SymbolId,
    metadata: &DocumentMetadata,
    preds: &LibraryPredicates,
    source: &DocumentSource,
    format: ContentFormat,
    tags: &[String],
    slug: &str,
) -> LibraryResult<usize> {
    let mut count = 0;

    if let Some(ref title) = metadata.title {
        let val = create_entity(engine, title, slug)?;
        add_triple(engine, doc_sym, preds.has_title, val, slug)?;
        count += 1;
    }
    if let Some(ref author) = metadata.author {
        let val = create_entity(engine, author, slug)?;
        add_triple(engine, doc_sym, preds.has_author, val, slug)?;
        count += 1;
    }
    if let Some(ref desc) = metadata.description {
        let val = create_entity(engine, desc, slug)?;
        add_triple(engine, doc_sym, preds.has_description, val, slug)?;
        count += 1;
    }
    if let Some(ref lang) = metadata.language {
        let val = create_entity(engine, lang, slug)?;
        add_triple(engine, doc_sym, preds.has_language, val, slug)?;
        count += 1;
    }
    for kw in &metadata.keywords {
        let val = create_entity(engine, kw, slug)?;
        add_triple(engine, doc_sym, preds.has_keyword, val, slug)?;
        count += 1;
    }

    // Format triple.
    let fmt_val = create_entity(engine, format.as_str(), slug)?;
    add_triple(engine, doc_sym, preds.has_format, fmt_val, slug)?;
    count += 1;

    // Source triple.
    let src_label = source.to_string();
    let src_val = create_entity(engine, &src_label, slug)?;
    add_triple(engine, doc_sym, preds.has_source, src_val, slug)?;
    count += 1;

    // Tag triples.
    for tag in tags {
        let val = create_entity(engine, tag, slug)?;
        add_triple(engine, doc_sym, preds.has_tag, val, slug)?;
        count += 1;
    }

    Ok(count)
}

/// Run regex-based NLP extraction on chunk text.
///
/// Reuses the same pattern-matching approach as `text_ingest` tool.
fn run_nlp_extraction(engine: &Engine, text: &str, slug: &str) -> LibraryResult<usize> {
    let mut count = 0;
    let sentences = split_sentences(text);

    for sentence in &sentences {
        for (subject, predicate, object, confidence) in extract_triples(sentence) {
            let s = create_entity(engine, &subject, slug)?;
            let p = engine.resolve_or_create_relation(&predicate).map_err(|e| {
                LibraryError::IngestFailed {
                    document: slug.into(),
                    message: format!("create relation '{predicate}': {e}"),
                }
            })?;
            let o = create_entity(engine, &object, slug)?;
            let triple = Triple::new(s, p, o).with_confidence(confidence);
            let _ = engine.add_triple(&triple); // Tolerate duplicates.
            count += 1;
        }
    }
    Ok(count)
}

/// Store a provenance record for a derived chunk symbol.
fn store_provenance(
    engine: &Engine,
    symbol: SymbolId,
    slug: &str,
    format: ContentFormat,
    chunk_index: u32,
) {
    let kind = DerivationKind::DocumentIngested {
        document_id: slug.to_string(),
        format: format.as_str().to_string(),
        chunk_index,
    };
    let mut record = ProvenanceRecord::new(symbol, kind).with_confidence(1.0);
    // Best-effort: provenance requires a durable store.
    let _ = engine.store_provenance(&mut record);
}

// ---------------------------------------------------------------------------
// Inline NLP (same patterns as text_ingest tool, kept private)
// ---------------------------------------------------------------------------

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() > 1 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.len() > 1 {
        sentences.push(trimmed);
    }
    sentences
}

type ExtractedTriple = (String, String, String, f32);

fn extract_triples(sentence: &str) -> Vec<ExtractedTriple> {
    let mut results = Vec::new();
    let s = sentence
        .trim()
        .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 3 {
        return results;
    }
    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();

    try_pattern(&words, &lower, &["is", "a"], "is-a", 0.9, &mut results);
    try_pattern(&words, &lower, &["is", "an"], "is-a", 0.9, &mut results);
    try_pattern(&words, &lower, &["has", "a"], "has-a", 0.85, &mut results);
    try_pattern(&words, &lower, &["has", "an"], "has-a", 0.85, &mut results);
    try_pattern(
        &words,
        &lower,
        &["contains"],
        "contains",
        0.85,
        &mut results,
    );
    try_pattern(
        &words,
        &lower,
        &["is", "part", "of"],
        "part-of",
        0.9,
        &mut results,
    );
    try_pattern(&words, &lower, &["causes"], "causes", 0.85, &mut results);
    try_pattern(
        &words,
        &lower,
        &["is", "located", "in"],
        "located-in",
        0.9,
        &mut results,
    );
    try_pattern(
        &words,
        &lower,
        &["is", "similar", "to"],
        "similar-to",
        0.8,
        &mut results,
    );
    try_pattern(
        &words,
        &lower,
        &["is", "made", "of"],
        "composed-of",
        0.85,
        &mut results,
    );
    try_pattern(&words, &lower, &["are"], "is-a", 0.85, &mut results);
    try_pattern(&words, &lower, &["has"], "has-a", 0.85, &mut results);

    results
}

fn try_pattern(
    words: &[&str],
    lower: &[String],
    pattern: &[&str],
    predicate: &str,
    confidence: f32,
    results: &mut Vec<ExtractedTriple>,
) {
    let plen = pattern.len();
    if lower.len() < plen + 2 {
        return;
    }
    for i in 0..lower.len().saturating_sub(plen + 1) {
        let matches = pattern
            .iter()
            .enumerate()
            .all(|(j, p)| i + 1 + j < lower.len() && lower[i + 1 + j] == *p);
        if matches {
            let subject = capitalize(
                words[..=i]
                    .join(" ")
                    .trim_matches(|c: char| !c.is_alphanumeric()),
            );
            let obj_start = i + 1 + plen;
            if obj_start >= words.len() {
                continue;
            }
            let object = capitalize(
                words[obj_start..]
                    .join(" ")
                    .trim_matches(|c: char| !c.is_alphanumeric()),
            );
            if !subject.is_empty() && !object.is_empty() {
                results.push((subject, predicate.into(), object, confidence));
                return;
            }
        }
    }
}

fn capitalize(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap().to_uppercase().to_string();
    first + chars.as_str()
}
