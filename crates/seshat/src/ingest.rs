//! Document ingestion: paragraph-aware chunking + DB writes.
//!
//! Splits a document body into chunks and persists both the
//! `pa_documents` row and the `pa_chunks` rows inside a single
//! transaction. Embedding is not done here — the background
//! `embed_worker` polls for rows with `embedded_at IS NULL` and
//! embeds them asynchronously. Callers can also use the
//! `seshat_embed_now` MCP tool to force an immediate batch pass.
//!
//! Chunking strategy is **paragraph-aware greedy packing**:
//!
//! 1. The body is split on blank lines into paragraphs.
//! 2. For markdown-formatted input, a leading `# … ######`
//!    heading is pulled off the start of each paragraph and
//!    propagated onto the resulting chunk (and every chunk
//!    that follows, until the next heading is seen).
//! 3. Paragraphs are packed into chunks up to `target_chars`
//!    characters. A paragraph that won't fit gets flushed into
//!    a new chunk; one that is *itself* too large falls back to
//!    sentence-boundary splitting, and finally to hard-character
//!    splits for anything sentence-splitting can't rescue
//!    (e.g. giant URLs or minified JSON).
//!
//! The default budget of 1500 chars keeps each chunk under
//! `all-MiniLM-L6-v2`'s 512-WordPiece-token cap with healthy
//! headroom for English prose. Callers can override it via the
//! `target_chunk_chars` field on [`IngestRequest`].

use sea_orm::{ActiveModelTrait, DatabaseConnection, EntityTrait, Set, TransactionTrait};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity::pa_chunk;
use crate::entity::pa_document;
use crate::error::{SeshatError, SeshatResult};

/// Default target chunk size in characters.
///
/// Chosen to stay under `all-MiniLM-L6-v2`'s 512-WordPiece-token cap.
/// English averages ~0.25–0.3 tokens per character, so 1500 chars is
/// roughly 400 tokens — enough headroom that the tokenizer's
/// truncation doesn't silently chop content off our chunks.
pub const DEFAULT_TARGET_CHUNK_CHARS: usize = 1500;

/// Minimum chunk-size budget. Guards against pathologically small
/// inputs (`target_chars = 0`) that would otherwise loop.
const MIN_TARGET_CHARS: usize = 200;

/// A chunk-to-be, before it's assigned an ID and persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkDraft {
    pub text: String,
    pub word_count: i32,
    pub heading: Option<String>,
}

/// A request to ingest a single document.
///
/// Build one with the fields you have; the MCP layer maps its tool
/// params onto this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub title: String,
    /// URL-safe unique slug. Auto-derived from title if `None`.
    pub slug: Option<String>,
    /// Document format: `markdown`, `text`, `html`, `pdf`, `epub`, …
    pub format: String,
    /// Full document body.
    pub body: String,
    /// Original file path or URL, if any.
    pub source_path: Option<String>,
    /// Freeform tags for categorization; stored as a JSONB array.
    pub tags: Vec<String>,
    /// Freeform metadata (author, language, …); stored as JSONB.
    pub metadata: serde_json::Value,
    /// Identifier of the pipeline that ingested this document.
    pub ingested_by: Option<String>,
    /// Target chunk size in characters.
    pub target_chunk_chars: usize,
}

/// Summary returned from a successful ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSummary {
    pub document_id: Uuid,
    pub slug: String,
    pub chunk_count: usize,
    pub word_count: usize,
}

/// Ingest a document: chunk, persist, and leave embedding to the worker.
pub async fn ingest_document(
    db: &DatabaseConnection,
    req: IngestRequest,
) -> SeshatResult<IngestSummary> {
    if req.title.trim().is_empty() {
        return Err(SeshatError::Config(
            "ingest: title must not be empty".to_string(),
        ));
    }
    if req.body.trim().is_empty() {
        return Err(SeshatError::Config(
            "ingest: body must not be empty".to_string(),
        ));
    }

    let chunks = chunk_paragraph_aware(&req.body, req.target_chunk_chars, &req.format);
    if chunks.is_empty() {
        return Err(SeshatError::Config(
            "ingest: chunker produced zero chunks — is body non-empty whitespace only?"
                .to_string(),
        ));
    }

    let doc_id = Uuid::new_v4();
    let slug = match req.slug.clone() {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => derive_slug(&req.title, &doc_id),
    };

    let total_words: usize = chunks.iter().map(|c| c.word_count as usize).sum();
    let chunk_count = chunks.len();
    let now = chrono::Utc::now();

    // Transaction so a chunk-insert failure rolls back the document row.
    let txn = db.begin().await?;

    let tags_json = serde_json::to_value(&req.tags)?;
    let doc_am = pa_document::ActiveModel {
        id: Set(doc_id),
        title: Set(req.title.clone()),
        slug: Set(slug.clone()),
        source_path: Set(req.source_path.clone()),
        format: Set(req.format.clone()),
        tags: Set(tags_json),
        metadata: Set(req.metadata.clone()),
        chunk_count: Set(chunk_count as i32),
        ingested_at: Set(now.into()),
        ingested_by: Set(req.ingested_by.clone()),
    };
    doc_am.insert(&txn).await?;

    let chunk_models: Vec<pa_chunk::ActiveModel> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| pa_chunk::ActiveModel {
            id: Set(Uuid::new_v4()),
            document_id: Set(doc_id),
            chunk_index: Set(i as i32),
            text: Set(c.text),
            word_count: Set(c.word_count),
            chapter: Set(None),
            section: Set(None),
            heading: Set(c.heading),
            embedded_at: Set(None),
        })
        .collect();

    pa_chunk::Entity::insert_many(chunk_models)
        .exec(&txn)
        .await?;

    txn.commit().await?;

    Ok(IngestSummary {
        document_id: doc_id,
        slug,
        chunk_count,
        word_count: total_words,
    })
}

/// Derive a URL-safe slug from a title. Appends an 8-char UUID suffix
/// to keep slugs unique when the same title is ingested more than once.
fn derive_slug(title: &str, id: &Uuid) -> String {
    let mut slug: String = title
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-').to_string();
    let suffix: String = id.simple().to_string().chars().take(8).collect();
    if slug.is_empty() {
        format!("doc-{suffix}")
    } else {
        format!("{slug}-{suffix}")
    }
}

/// Paragraph-aware greedy chunker.
///
/// See the module docstring for the full strategy. `format` is used
/// only to decide whether to pull markdown-style headings off the front
/// of paragraphs; passing `"markdown"` or `"md"` enables heading
/// extraction, anything else treats the input as opaque prose.
pub fn chunk_paragraph_aware(
    body: &str,
    target_chars: usize,
    format: &str,
) -> Vec<ChunkDraft> {
    let target = target_chars.max(MIN_TARGET_CHARS);
    let is_markdown = matches!(format, "markdown" | "md");

    let mut chunks: Vec<ChunkDraft> = Vec::new();
    let mut current = String::new();
    let mut current_heading: Option<String> = None;

    for para in split_paragraphs(body) {
        // If the first line of this paragraph is a markdown heading,
        // split it off and use it as the propagated heading.
        let (heading_update, content) = if is_markdown {
            match split_leading_heading(&para) {
                Some((h, rest)) => (Some(h), rest),
                None => (None, Some(para)),
            }
        } else {
            (None, Some(para))
        };

        if let Some(h) = heading_update {
            flush(&mut chunks, &mut current, current_heading.clone());
            current_heading = Some(h);
        }

        let content = match content {
            Some(c) if !c.trim().is_empty() => c,
            _ => continue,
        };

        if content.chars().count() > target {
            flush(&mut chunks, &mut current, current_heading.clone());
            for sub in split_long_paragraph(&content, target) {
                let wc = word_count(&sub);
                chunks.push(ChunkDraft {
                    text: sub,
                    word_count: wc,
                    heading: current_heading.clone(),
                });
            }
            continue;
        }

        // Will this paragraph push us over the budget?
        let sep = if current.is_empty() { 0 } else { 2 };
        let would_overflow =
            !current.is_empty() && current.chars().count() + sep + content.chars().count() > target;
        if would_overflow {
            flush(&mut chunks, &mut current, current_heading.clone());
            current = content;
        } else if current.is_empty() {
            current = content;
        } else {
            current.push_str("\n\n");
            current.push_str(&content);
        }
    }

    flush(&mut chunks, &mut current, current_heading);
    chunks
}

fn flush(chunks: &mut Vec<ChunkDraft>, current: &mut String, heading: Option<String>) {
    if current.is_empty() {
        return;
    }
    let text = std::mem::take(current);
    let wc = word_count(&text);
    chunks.push(ChunkDraft {
        text,
        word_count: wc,
        heading,
    });
}

/// Split `body` on blank lines, trim each paragraph, drop empties.
/// `\r\n` is normalized to `\n`.
fn split_paragraphs(body: &str) -> Vec<String> {
    let normalized = body.replace("\r\n", "\n");
    let mut out = Vec::new();
    let mut current = String::new();
    for line in normalized.split('\n') {
        if line.trim().is_empty() {
            if !current.trim().is_empty() {
                out.push(current.trim().to_string());
            }
            current.clear();
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

/// If the first line of `para` is a markdown ATX heading (`#` through
/// `######` followed by whitespace), return `(heading_text, remaining_body)`.
/// `remaining_body` is `None` if the heading was the only line.
fn split_leading_heading(para: &str) -> Option<(String, Option<String>)> {
    let mut lines = para.lines();
    let first = lines.next()?;
    let trimmed = first.trim_start();
    let mut chars = trimmed.chars();
    let mut hashes = 0;
    while let Some('#') = chars.clone().next() {
        chars.next();
        hashes += 1;
        if hashes > 6 {
            return None;
        }
    }
    if hashes == 0 {
        return None;
    }
    match chars.next() {
        Some(c) if c.is_whitespace() => {
            let heading = chars.as_str().trim().trim_end_matches('#').trim().to_string();
            let rest: String = lines.collect::<Vec<_>>().join("\n");
            let rest = if rest.trim().is_empty() {
                None
            } else {
                Some(rest.trim().to_string())
            };
            Some((heading, rest))
        }
        _ => None,
    }
}

fn word_count(s: &str) -> i32 {
    s.split_whitespace().count() as i32
}

/// Split an over-budget paragraph on sentence boundaries and greedily
/// repack. Sentences that are themselves too long fall through to
/// `hard_split` as a last resort.
fn split_long_paragraph(para: &str, target: usize) -> Vec<String> {
    let sentences = split_sentences(para);
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for sent in sentences {
        if sent.chars().count() > target {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.extend(hard_split(&sent, target));
            continue;
        }
        let sep = if current.is_empty() { 0 } else { 1 };
        if !current.is_empty() && current.chars().count() + sep + sent.chars().count() > target {
            out.push(std::mem::take(&mut current));
            current = sent;
        } else if current.is_empty() {
            current = sent;
        } else {
            current.push(' ');
            current.push_str(&sent);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Simple prose sentence splitter: split on `". "`, `"? "`, `"! "`
/// while keeping the terminator with the preceding sentence. Not an
/// NLP-grade tokenizer — intentionally minimal because the fallback
/// below (`hard_split`) catches degenerate cases.
fn split_sentences(para: &str) -> Vec<String> {
    let chars: Vec<char> = para.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let is_term = matches!(c, '.' | '?' | '!');
        let next_ws = i + 1 < chars.len() && chars[i + 1].is_whitespace();
        if is_term && next_ws {
            let piece: String = chars[start..=i].iter().collect();
            let piece = piece.trim().to_string();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = i + 2;
            i = start;
            continue;
        }
        i += 1;
    }
    if start < chars.len() {
        let tail: String = chars[start..].iter().collect();
        let tail = tail.trim().to_string();
        if !tail.is_empty() {
            out.push(tail);
        }
    }
    if out.is_empty() {
        out.push(para.trim().to_string());
    }
    out
}

/// Last-resort character-boundary split for sentences that don't fit.
/// Used for things like minified JSON, long URLs, or base64 blobs
/// masquerading as a single "sentence".
fn hard_split(s: &str, target: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let end = (i + target).min(chars.len());
        out.push(chars[i..end].iter().collect::<String>());
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_small_paragraphs_into_one_chunk() {
        let body = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_paragraph_aware(body, 1500, "text");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Para one."));
        assert!(chunks[0].text.contains("Para three."));
        assert!(chunks[0].heading.is_none());
    }

    #[test]
    fn splits_when_budget_exceeded() {
        let para = "Lorem ipsum dolor sit amet consectetur adipiscing elit.";
        let body = std::iter::repeat(para).take(9).collect::<Vec<_>>().join("\n\n");
        let chunks = chunk_paragraph_aware(&body, 200, "text");
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.chars().count() <= 200, "chunk {} chars", c.text.chars().count());
        }
    }

    #[test]
    fn extracts_markdown_heading_from_standalone_paragraph() {
        let body = "# Introduction\n\nThis is the intro paragraph.\n\n## Next Section\n\nMore content.";
        let chunks = chunk_paragraph_aware(body, 1500, "markdown");
        assert_eq!(chunks.len(), 2, "got {:#?}", chunks);
        assert_eq!(chunks[0].heading.as_deref(), Some("Introduction"));
        assert!(!chunks[0].text.trim_start().starts_with('#'));
        assert_eq!(chunks[1].heading.as_deref(), Some("Next Section"));
    }

    #[test]
    fn extracts_markdown_heading_with_inline_content() {
        // Heading and content in the same paragraph (no blank line between).
        let body = "## Title\nContent line one.\nContent line two.\n\nNext para.";
        let chunks = chunk_paragraph_aware(body, 1500, "markdown");
        assert_eq!(chunks[0].heading.as_deref(), Some("Title"));
        assert!(chunks[0].text.contains("Content line one."));
        assert!(chunks[0].text.contains("Next para."));
    }

    #[test]
    fn splits_long_paragraph_on_sentence_boundaries() {
        let long = "One short sentence. ".repeat(50); // ~1000 chars
        let chunks = chunk_paragraph_aware(&long, 300, "text");
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.chars().count() <= 300);
        }
    }

    #[test]
    fn hard_splits_ridiculously_long_non_sentence() {
        let body = "a".repeat(5000);
        let chunks = chunk_paragraph_aware(&body, 1000, "text");
        assert!(chunks.len() >= 5);
        for c in &chunks {
            assert!(c.text.chars().count() <= 1000);
        }
    }

    #[test]
    fn ignores_tiny_budget_by_enforcing_floor() {
        let body = "para one.\n\npara two.";
        // target_chars = 10 would be impossible, floor bumps to MIN_TARGET_CHARS.
        let chunks = chunk_paragraph_aware(body, 10, "text");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn derive_slug_handles_special_chars() {
        let id = Uuid::new_v4();
        let s = derive_slug("Phase 37: Seshat's Archive (plan)", &id);
        assert!(s.starts_with("phase-37-seshat-s-archive-plan-"), "got {s}");
        assert_eq!(s.split('-').last().unwrap().len(), 8);
    }

    #[test]
    fn derive_slug_fallback_for_all_specials() {
        let id = Uuid::new_v4();
        let s = derive_slug("!!!???", &id);
        assert!(s.starts_with("doc-"));
    }

    #[test]
    fn rejects_hashtag_without_whitespace_as_heading() {
        // "#tag" is not a heading — must have whitespace after the hashes.
        assert!(split_leading_heading("#tag not a heading").is_none());
        assert_eq!(
            split_leading_heading("# Real Heading"),
            Some(("Real Heading".to_string(), None))
        );
        // More than 6 hashes is not an ATX heading.
        assert!(split_leading_heading("####### too many").is_none());
    }

    #[test]
    fn propagates_heading_across_packed_chunks() {
        let para = "Lorem ipsum dolor sit amet consectetur adipiscing elit.";
        // Fill enough paragraphs under one heading to force 2+ chunks.
        let body = format!(
            "# Topic\n\n{paras}",
            paras = std::iter::repeat(para).take(8).collect::<Vec<_>>().join("\n\n")
        );
        let chunks = chunk_paragraph_aware(&body, 200, "markdown");
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert_eq!(c.heading.as_deref(), Some("Topic"));
        }
    }
}
