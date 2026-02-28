//! Document ingestion pipeline.
//!
//! Orchestrates: parse → chunk → symbols → structural triples → concept extraction
//! → VSA embeddings → provenance → catalog.

use std::collections::HashSet;
use std::path::Path;

use crate::engine::Engine;
use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::ParseContext;
use crate::graph::Triple;
use crate::library::catalog::{LibraryCatalog, slugify};
use crate::library::chunker::{ChunkConfig, normalize_chunks};
use crate::library::concepts::{
    extract_concepts_from_chunk, extract_head_noun_phrase, extract_richer_triples, ConceptSource,
};
use crate::library::error::{LibraryError, LibraryResult};
use crate::library::model::*;
use crate::library::parser;
use crate::library::predicates::LibraryPredicates;
use crate::nlu::NluPipeline;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_label;

/// Configuration for the ingestion pipeline.
#[derive(Default)]
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


/// Result of a successful document ingestion.
pub struct IngestResult {
    /// The document's catalog record.
    pub record: DocumentRecord,
    /// The document's root symbol ID in the KG.
    pub document_symbol: SymbolId,
    /// Total triples created.
    pub triple_count: usize,
    /// Total atomic concepts extracted.
    pub concept_count: usize,
    /// Total chunks after normalization.
    pub chunk_count: usize,
    /// Triples added by the NLU pipeline (Phase C). Subset of `triple_count`.
    pub nlu_triple_count: usize,
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
    nlu: Option<&mut NluPipeline>,
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

    // 8. Per-chunk: create paragraph symbols, structural triples, concept extraction, VSA.
    let mut prev_chunk_sym: Option<SymbolId> = None;
    let mut total_concepts = 0usize;
    let mut total_nlu_triples = 0usize;
    let mut nlu = nlu;

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

        // Concept extraction: three-phase concept extraction from chunk text.
        //   Phase A: relational patterns, Phase B: standalone concepts,
        //   Phase C: NLU pipeline (if available).
        let extraction = run_concept_extraction(
            engine,
            para_sym,
            &chunk.text,
            &preds,
            &slug,
            format,
            chunk.index as u32,
            nlu.as_mut().map(|p| &mut **p),
        )?;
        triple_count += extraction.triple_count;
        total_concepts += extraction.concept_count;
        total_nlu_triples += extraction.nlu_triple_count;

        // VSA embedding: encode the chunk text and insert into item memory.
        if let Ok(vec) = encode_label(engine.ops(), &chunk.text) {
            engine.item_memory().insert(para_sym, vec);
        }

        // Provenance: record document ingestion origin.
        store_provenance(engine, para_sym, &slug, format, chunk.index as u32);
    }

    // Audit: record the content ingestion event (best-effort).
    // Capture source string before it's moved into the catalog record.
    let source_str = source.to_string();
    if let Some(ledger) = engine.audit_ledger() {
        let mut audit_entry = crate::audit::AuditEntry::new(
            crate::audit::AuditKind::ContentIngestion {
                document_title: title.clone(),
                source: source_str.clone(),
                format: format.as_str().to_string(),
                chunk_count: chunks.len(),
                triple_count,
            },
            "default",
        );
        let _ = ledger.append(&mut audit_entry);
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
        concept_count: total_concepts,
        chunk_count: chunks.len(),
        nlu_triple_count: total_nlu_triples,
    })
}

/// Ingest a document from a filesystem path.
pub fn ingest_file(
    engine: &Engine,
    catalog: &mut LibraryCatalog,
    path: &Path,
    config: IngestConfig,
    nlu: Option<&mut NluPipeline>,
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

    ingest_document(engine, catalog, &data, source, config, nlu)
}

/// Ingest a document from a URL via HTTP GET.
pub fn ingest_url(
    engine: &Engine,
    catalog: &mut LibraryCatalog,
    url: &str,
    config: IngestConfig,
    nlu: Option<&mut NluPipeline>,
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

    ingest_document(engine, catalog, &data, source, config, nlu)
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
#[allow(clippy::too_many_arguments)]
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

/// Result of concept extraction for a single chunk.
struct ExtractionResult {
    triple_count: usize,
    concept_count: usize,
    nlu_triple_count: usize,
}

/// Three-phase concept extraction from chunk text.
///
/// Phase A: Relational patterns (existing `extract_triples` + `extract_richer_triples`)
///   with head noun phrase extraction to trim spans to atomic concepts.
/// Phase B: Standalone concepts (capitalized terms, technical compounds, repeated terms).
/// Phase C: NLU pipeline parsing (if available) — produces richer AbsTree structures
///   that capture negation, conditionals, temporals, modals, comparisons, etc.
#[allow(clippy::too_many_arguments)]
fn run_concept_extraction(
    engine: &Engine,
    para_sym: SymbolId,
    text: &str,
    preds: &LibraryPredicates,
    slug: &str,
    _format: ContentFormat,
    chunk_index: u32,
    nlu: Option<&mut NluPipeline>,
) -> LibraryResult<ExtractionResult> {
    let mut triple_count = 0usize;
    let mut concept_count = 0usize;
    let mut nlu_triple_count = 0usize;
    let mut relational_labels: HashSet<String> = HashSet::new();

    let sentences = split_sentences(text);

    // Collect all extracted triples (existing patterns + richer patterns), dedup.
    let mut all_triples: Vec<(String, String, String, f32)> = Vec::new();
    let mut seen_triple_keys: HashSet<String> = HashSet::new();

    for sentence in &sentences {
        // Phase A: existing relational patterns.
        for t in extract_triples(sentence) {
            let key = format!("{}|{}|{}", t.0.to_lowercase(), t.1, t.2.to_lowercase());
            if seen_triple_keys.insert(key) {
                all_triples.push(t);
            }
        }
        // Phase A extended: richer patterns (conjunctions, bare verbs, adverb modulation).
        for t in extract_richer_triples(sentence) {
            let key = format!("{}|{}|{}", t.0.to_lowercase(), t.1, t.2.to_lowercase());
            if seen_triple_keys.insert(key) {
                all_triples.push(t);
            }
        }
    }

    // Process each extracted triple: apply head noun phrase extraction.
    for (subject_span, predicate, object_span, confidence) in &all_triples {
        let subj_words: Vec<&str> = subject_span.split_whitespace().collect();
        let obj_words: Vec<&str> = object_span.split_whitespace().collect();

        let atomic_subj = extract_head_noun_phrase(&subj_words)
            .or_else(|| {
                // Fallback: if original span ≤3 words after determiner stripping, use directly.
                if subj_words.len() <= 3 {
                    Some(capitalize(&subj_words.join(" ")))
                } else {
                    None
                }
            });
        let atomic_obj = extract_head_noun_phrase(&obj_words)
            .or_else(|| {
                if obj_words.len() <= 3 {
                    Some(capitalize(&obj_words.join(" ")))
                } else {
                    None
                }
            });

        if let (Some(subj), Some(obj)) = (atomic_subj, atomic_obj) {
            if subj.is_empty() || obj.is_empty() {
                continue;
            }

            let s = create_entity(engine, &subj, slug)?;
            let p = engine.resolve_or_create_relation(predicate).map_err(|e| {
                LibraryError::IngestFailed {
                    document: slug.into(),
                    message: format!("create relation '{predicate}': {e}"),
                }
            })?;
            let o = create_entity(engine, &obj, slug)?;
            let triple = Triple::new(s, p, o).with_confidence(*confidence);
            let _ = engine.add_triple(&triple);
            triple_count += 1;

            // Add doc:mentions triples linking paragraph to each concept.
            add_triple(engine, para_sym, preds.mentions, s, slug)?;
            add_triple(engine, para_sym, preds.mentions, o, slug)?;
            triple_count += 2;

            // Track for dedup with standalone concepts.
            relational_labels.insert(subj.to_lowercase());
            relational_labels.insert(obj.to_lowercase());

            // Provenance for each concept.
            store_concept_provenance(
                engine, s, slug, chunk_index, ConceptSource::RelationalHead,
            );
            store_concept_provenance(
                engine, o, slug, chunk_index, ConceptSource::RelationalHead,
            );
            concept_count += 2;
        }
    }

    // Phase B: Standalone concepts not captured by relational extraction.
    let standalone = extract_concepts_from_chunk(text, &relational_labels);
    for concept in &standalone {
        let sym = create_entity(engine, &concept.label, slug)?;
        add_triple(engine, para_sym, preds.mentions, sym, slug)?;
        triple_count += 1;

        store_concept_provenance(engine, sym, slug, chunk_index, concept.source);
        concept_count += 1;
    }

    // Phase C: NLU pipeline sentence-level parsing (when available).
    //
    // The NLU cascade produces rich AbsTree structures with negation, conditionals,
    // temporal scoping, modals, comparatives, and relative clauses — constructs that
    // the regex-based Phase A cannot capture. Parse failures are non-fatal since
    // Phase A/B already extracted what they could.
    if let Some(nlu) = nlu {
        let ctx = ParseContext::with_engine(engine.registry(), engine.ops(), engine.item_memory());

        for sentence in &sentences {
            let parse_result = match nlu.parse(sentence, &ctx) {
                Ok(r) => r,
                Err(_) => continue, // Non-fatal: Phase A/B already handled this sentence.
            };

            let grounded = parse_result.tree.ground(engine.registry());

            // Extract triple keys from the grounded tree to dedup against Phase A.
            let tree_keys = collect_triple_keys(&grounded);
            let has_new = tree_keys.iter().any(|k| !seen_triple_keys.contains(k));
            if !has_new && !tree_keys.is_empty() {
                // All triples from this parse were already captured by Phase A.
                continue;
            }
            // Record the keys so later sentences don't re-add them either.
            for key in tree_keys {
                seen_triple_keys.insert(key);
            }

            // Commit the grounded tree to the KG.
            let committed = engine.commit_abs_tree(&grounded).map_err(|e| {
                LibraryError::IngestFailed {
                    document: slug.into(),
                    message: format!("NLU commit: {e}"),
                }
            })?;
            nlu_triple_count += committed;
            triple_count += committed;

            // Add doc:mentions triples for all entity labels discovered by NLU.
            for label in grounded.collect_labels() {
                if relational_labels.contains(&label.to_lowercase()) {
                    continue; // Already linked by Phase A.
                }
                if let Ok(sym) = engine.resolve_or_create_entity(label) {
                    let _ = add_triple(engine, para_sym, preds.mentions, sym, slug);
                    triple_count += 1;
                    relational_labels.insert(label.to_lowercase());
                }
            }

            // NLU-specific provenance.
            let nlu_source = ConceptSource::NluParsed {
                tier: parse_result.source_tier,
            };
            store_concept_provenance(engine, para_sym, slug, chunk_index, nlu_source);

            // Also store the richer DerivationKind::NluParsed provenance.
            let nlu_kind = DerivationKind::NluParsed {
                source_tier: parse_result.source_tier,
                confidence: parse_result.confidence,
                exemplar_similarity: parse_result.exemplar_similarity,
            };
            let mut nlu_record =
                ProvenanceRecord::new(para_sym, nlu_kind).with_confidence(parse_result.confidence);
            let _ = engine.store_provenance(&mut nlu_record);
        }
    }

    Ok(ExtractionResult {
        triple_count,
        concept_count,
        nlu_triple_count,
    })
}

/// Walk an [`AbsTree`] and collect dedup keys for any `Triple` nodes.
///
/// Keys are `"subject|predicate|object"` (lowercased), matching the format
/// used by Phase A's `seen_triple_keys` set.
fn collect_triple_keys(tree: &AbsTree) -> Vec<String> {
    let mut keys = Vec::new();
    collect_triple_keys_inner(tree, &mut keys);
    keys
}

fn collect_triple_keys_inner(tree: &AbsTree, keys: &mut Vec<String>) {
    match tree {
        AbsTree::Triple {
            subject,
            predicate,
            object,
        } => {
            let s = subject.label().unwrap_or("?").to_lowercase();
            let p = predicate.label().unwrap_or("?").to_lowercase();
            let o = object.label().unwrap_or("?").to_lowercase();
            keys.push(format!("{s}|{p}|{o}"));
        }
        AbsTree::WithConfidence { inner, .. }
        | AbsTree::WithProvenance { inner, .. }
        | AbsTree::Negation { inner }
        | AbsTree::Temporal { inner, .. }
        | AbsTree::Modal { inner, .. } => {
            collect_triple_keys_inner(inner, keys);
        }
        AbsTree::Quantified { scope, .. } => {
            collect_triple_keys_inner(scope, keys);
        }
        AbsTree::Conjunction { items, .. } | AbsTree::Section { body: items, .. } => {
            for item in items {
                collect_triple_keys_inner(item, keys);
            }
        }
        AbsTree::Conditional {
            condition,
            consequent,
        } => {
            collect_triple_keys_inner(condition, keys);
            collect_triple_keys_inner(consequent, keys);
        }
        AbsTree::RelativeClause { head, clause } => {
            collect_triple_keys_inner(head, keys);
            collect_triple_keys_inner(clause, keys);
        }
        _ => {}
    }
}

/// Store a provenance record for an extracted concept.
fn store_concept_provenance(
    engine: &Engine,
    symbol: SymbolId,
    slug: &str,
    chunk_index: u32,
    source: ConceptSource,
) {
    let kind = DerivationKind::ConceptExtracted {
        document_id: slug.to_string(),
        chunk_index,
        extraction_method: source.method_str().to_string(),
    };
    let mut record = ProvenanceRecord::new(symbol, kind).with_confidence(1.0);
    let _ = engine.store_provenance(&mut record);
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
        .trim_end_matches(['.', '!', '?']);
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
