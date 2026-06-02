//! Journal ingestion: parse OpenClaw daily memory journals (markdown) into episodic memory.
//!
//! Journals are markdown files in a configurable directory (default:
//! `$XDG_DATA_HOME/akh-medu/journals/`; overridable by `AKH_JOURNAL_DIR`).
//! Each file is split into heading-delimited chunks; each chunk becomes one
//! episodic KG entry tagged `tag:journal` so `agent_recall` can find it.
//!
//! Re-running is idempotent: the presence of a `journal:ingested:{stem}` entity
//! in the KG signals that a file has been processed; it is skipped on re-runs.

use std::path::{Path, PathBuf};

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::agent::AgentPredicates;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the journal ingestion pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum JournalIngestError {
    #[error("cannot read journal directory \"{path}\": {source}")]
    #[diagnostic(
        code(akh::agent::journal::read_dir),
        help(
            "Ensure the directory exists. Create it manually or run `akh memory init-journals`."
        )
    )]
    ReadDir {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot read journal file \"{path}\": {source}")]
    #[diagnostic(
        code(akh::agent::journal::read_file),
        help("Check that the file exists and you have read permission.")
    )]
    ReadFile {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("KG write failed during journal ingest for \"{file}\": {message}")]
    #[diagnostic(
        code(akh::agent::journal::kg_write),
        help("Ensure akhomed is running and the workspace engine is accessible.")
    )]
    KgWrite { file: String, message: String },
}

/// Convenience alias.
pub type JournalIngestResult<T> = Result<T, JournalIngestError>;

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// A single content chunk from a parsed markdown journal file.
#[derive(Debug, Clone)]
pub struct JournalChunk {
    /// The heading text that opens this section (empty for front-matter prose).
    pub heading: String,
    /// Body text (trimmed).
    pub content: String,
    /// Zero-based chunk index within the file.
    pub index: usize,
}

/// Split a markdown document into heading-delimited chunks.
///
/// Any line whose first non-space character is `#` starts a new chunk.
/// Content before the first heading becomes chunk 0 with an empty heading.
/// Empty chunks (no body text) are omitted.
/// Chunks with bodies exceeding `max_chars` are truncated with an ellipsis.
pub fn chunk_markdown(text: &str, max_chars: usize) -> Vec<JournalChunk> {
    let mut chunks: Vec<JournalChunk> = Vec::new();
    let mut current_heading = String::new();
    let mut current_body = String::new();
    let mut index = 0usize;

    let flush = |heading: &str, body: &str, idx: &mut usize, out: &mut Vec<JournalChunk>| {
        let body = body.trim();
        if !body.is_empty() {
            let content = if body.len() > max_chars {
                format!("{}…", &body[..max_chars])
            } else {
                body.to_string()
            };
            out.push(JournalChunk {
                heading: heading.to_string(),
                content,
                index: *idx,
            });
            *idx += 1;
        }
    };

    for line in text.lines() {
        if line.trim_start().starts_with('#') {
            flush(
                &current_heading,
                &current_body,
                &mut index,
                &mut chunks,
            );
            current_heading = line.trim_start_matches('#').trim().to_string();
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    flush(&current_heading, &current_body, &mut index, &mut chunks);
    chunks
}

// ---------------------------------------------------------------------------
// Keyword extraction
// ---------------------------------------------------------------------------

/// Extract candidate keyword tokens from a chunk for KG symbol linking.
///
/// Heuristics used:
/// - Words ≥ 4 characters that begin with an uppercase letter at mid-sentence
///   (position > 0) are likely proper nouns or technical terms.
/// - The heading words themselves are always included.
/// - Results are lower-cased, de-duplicated, and limited to 15.
fn extract_keywords(heading: &str, body: &str) -> Vec<String> {
    let mut kws: Vec<String> = Vec::new();

    // Always include heading words.
    for w in heading.split_whitespace() {
        let clean = w
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if clean.len() >= 3 {
            kws.push(clean);
        }
    }

    // Scan body for mid-sentence capitalized tokens.
    let words: Vec<&str> = body.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() < 4 {
            continue;
        }
        if i > 0 && clean.chars().next().is_some_and(|c| c.is_uppercase()) {
            kws.push(clean.to_lowercase());
        }
    }

    kws.sort();
    kws.dedup();
    kws.truncate(15);
    kws
}

// ---------------------------------------------------------------------------
// Ingestion statistics
// ---------------------------------------------------------------------------

/// Aggregate results from a journal ingest run.
#[derive(Debug, Default)]
pub struct JournalIngestStats {
    /// Total `*.md` files found in the scanned directory.
    pub files_scanned: usize,
    /// Files actually ingested this run.
    pub files_ingested: usize,
    /// Files skipped (already ingested or unreadable).
    pub files_skipped: usize,
    /// Episodic KG entries created across all ingested files.
    pub episodes_created: usize,
}

// ---------------------------------------------------------------------------
// Idempotency helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the journal with the given stem has already been ingested.
///
/// Checks by looking for the marker entity `journal:ingested:{stem}` in the KG.
fn is_ingested(engine: &Engine, stem: &str) -> bool {
    engine
        .resolve_symbol(&format!("journal:ingested:{stem}"))
        .is_ok()
}

/// Write a `journal:ingested:{stem}` marker entity so future runs skip the file.
fn mark_ingested(engine: &Engine, stem: &str) -> Result<(), crate::error::AkhError> {
    engine.resolve_or_create_entity(&format!("journal:ingested:{stem}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-file ingestion
// ---------------------------------------------------------------------------

/// Ingest a single markdown journal file into the episodic KG store.
///
/// Each heading-delimited chunk becomes one episodic memory entry tagged with
/// `tag:journal`, `tag:daily_ritual`, and a per-file source tag. Keywords
/// extracted from headings and body prose are linked via `agent:learned`.
///
/// Returns the number of new episodes created (0 if the file was already
/// ingested). Callers should call `engine.persist()` when convenient.
pub fn ingest_journal_file(
    engine: &Engine,
    path: &Path,
    predicates: &AgentPredicates,
) -> JournalIngestResult<usize> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("journal.md")
        .to_string();
    let stem = filename
        .strip_suffix(".md")
        .or_else(|| filename.strip_suffix(".MD"))
        .unwrap_or(&filename)
        .to_string();

    if is_ingested(engine, &stem) {
        return Ok(0);
    }

    let text = std::fs::read_to_string(path).map_err(|e| JournalIngestError::ReadFile {
        path: path.display().to_string(),
        source: e,
    })?;

    if text.trim().is_empty() {
        mark_ingested(engine, &stem).map_err(|e| JournalIngestError::KgWrite {
            file: filename.clone(),
            message: format!("{e}"),
        })?;
        return Ok(0);
    }

    let chunks = chunk_markdown(&text, 800);

    // Well-known tags resolved once per file.
    let tag_journal = engine
        .resolve_or_create_entity("tag:journal")
        .map_err(|e| JournalIngestError::KgWrite {
            file: filename.clone(),
            message: e.to_string(),
        })?;
    let tag_daily = engine
        .resolve_or_create_entity("tag:daily_ritual")
        .map_err(|e| JournalIngestError::KgWrite {
            file: filename.clone(),
            message: format!("{e}"),
        })?;
    let tag_source = engine
        .resolve_or_create_entity(&format!("journal:source:{stem}"))
        .map_err(|e| JournalIngestError::KgWrite {
            file: filename.clone(),
            message: format!("{e}"),
        })?;

    let file_tags = FileTags {
        journal: tag_journal,
        daily: tag_daily,
        source: tag_source,
    };
    let mut episodes_created = 0usize;

    for chunk in &chunks {
        match create_episode(engine, chunk, &filename, &stem, predicates, &file_tags) {
            Ok(_) => episodes_created += 1,
            Err(e) => {
                tracing::warn!(
                    file = %filename,
                    chunk = chunk.index,
                    error = %e,
                    "failed to persist journal chunk, continuing"
                );
            }
        }
    }

    mark_ingested(engine, &stem).map_err(|e| JournalIngestError::KgWrite {
        file: filename.clone(),
        message: format!("{e}"),
    })?;

    Ok(episodes_created)
}

/// Pre-resolved per-file tag symbols passed to `create_episode`.
struct FileTags {
    journal: SymbolId,
    daily: SymbolId,
    source: SymbolId,
}

/// Create one episodic KG entry for a single markdown chunk.
///
/// Episode label: `journal:{stem}:chunk{index}` (idempotent via resolve_or_create).
fn create_episode(
    engine: &Engine,
    chunk: &JournalChunk,
    filename: &str,
    stem: &str,
    predicates: &AgentPredicates,
    tags: &FileTags,
) -> Result<SymbolId, crate::error::AkhError> {
    // Episode entity (idempotent label).
    let episode_label = format!("journal:{stem}:chunk{}", chunk.index);
    let episode = engine.resolve_or_create_entity(&episode_label)?;

    // Summary entity: "{filename} § {heading}: {first 80 chars}"
    let preview = if chunk.content.len() > 80 {
        format!("{}…", &chunk.content[..80])
    } else {
        chunk.content.clone()
    };
    let summary_text = if chunk.heading.is_empty() {
        format!("{filename}: {preview}")
    } else {
        format!("{filename} § {}: {preview}", chunk.heading)
    };
    let summary_label = if summary_text.len() > 200 {
        format!("summary:{}", &summary_text[..200])
    } else {
        format!("summary:{summary_text}")
    };
    let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
    let _ = engine.add_triple(&Triple::new(episode, predicates.has_summary, summary_sym));

    // Tags.
    let _ = engine.add_triple(&Triple::new(episode, predicates.has_tag, tags.journal));
    let _ = engine.add_triple(&Triple::new(episode, predicates.has_tag, tags.daily));
    let _ = engine.add_triple(&Triple::new(episode, predicates.has_tag, tags.source));

    if !chunk.heading.is_empty() {
        let slug: String = chunk
            .heading
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let heading_tag =
            engine.resolve_or_create_entity(&format!("journal:section:{slug}"))?;
        let _ = engine.add_triple(&Triple::new(episode, predicates.has_tag, heading_tag));
    }

    // memory_type marker — makes `agent_recall` treat this as an episode.
    let mem_type = engine.resolve_or_create_entity("episodic_memory")?;
    let _ = engine.add_triple(&Triple::new(episode, predicates.memory_type, mem_type));

    // Learned symbols from keyword extraction.
    let keywords = extract_keywords(&chunk.heading, &chunk.content);
    for kw in &keywords {
        let sym = engine.resolve_or_create_entity(kw)?;
        let _ = engine.add_triple(&Triple::new(episode, predicates.learned, sym));
    }

    // Provenance record.
    let reason = if chunk.heading.is_empty() {
        format!("Journal ingest: {filename} (front-matter)")
    } else {
        format!("Journal ingest: {filename} § {}", chunk.heading)
    };
    let mut prov = ProvenanceRecord::new(
        episode,
        DerivationKind::AgentConsolidation {
            reason,
            relevance_score: 0.75,
        },
    )
    .with_confidence(0.75);
    let _ = engine.store_provenance(&mut prov);

    Ok(episode)
}

// ---------------------------------------------------------------------------
// Directory scanner
// ---------------------------------------------------------------------------

/// Scan `dir` for `*.md` files and ingest any not yet processed.
///
/// Files are processed in lexicographic filename order so that
/// `YYYY-MM-DD.md`-named journals are ingested chronologically.
/// Already-ingested files are skipped without error.
/// Per-file errors are logged as warnings and counted as skipped.
pub fn ingest_journals_from_dir(
    engine: &Engine,
    dir: &Path,
    predicates: &AgentPredicates,
) -> JournalIngestResult<JournalIngestStats> {
    let mut stats = JournalIngestStats::default();

    let entries = std::fs::read_dir(dir).map_err(|e| JournalIngestError::ReadDir {
        path: dir.display().to_string(),
        source: e,
    })?;

    let mut md_files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("md"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    // Chronological order via filename sort (YYYY-MM-DD.md naming).
    md_files.sort();
    stats.files_scanned = md_files.len();

    for path in &md_files {
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        if is_ingested(engine, &file_stem) {
            stats.files_skipped += 1;
            continue;
        }

        match ingest_journal_file(engine, path, predicates) {
            Ok(n) => {
                stats.files_ingested += 1;
                stats.episodes_created += n;
                tracing::info!(
                    file = %path.display(),
                    episodes = n,
                    "journal ingested"
                );
            }
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "journal ingest failed, skipping"
                );
                stats.files_skipped += 1;
            }
        }
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Standalone public API (for binary crates that cannot access pub(crate) items)
// ---------------------------------------------------------------------------

/// Ingest a single journal file, resolving `AgentPredicates` from the engine.
///
/// Convenience wrapper for callers (CLI, scripts) that don't hold an
/// `AgentPredicates` instance. Idempotent.
pub fn ingest_journal_file_auto(
    engine: &Engine,
    path: &Path,
) -> JournalIngestResult<usize> {
    let predicates =
        super::agent::AgentPredicates::init(engine).map_err(|e| JournalIngestError::KgWrite {
            file: path.display().to_string(),
            message: format!("{e}"),
        })?;
    ingest_journal_file(engine, path, &predicates)
}

/// Scan `dir` for `*.md` journals and ingest new ones, resolving
/// `AgentPredicates` from the engine.
///
/// Convenience wrapper for callers (CLI, scripts) that don't hold an
/// `AgentPredicates` instance.
pub fn ingest_journals_auto(
    engine: &Engine,
    dir: &Path,
) -> JournalIngestResult<JournalIngestStats> {
    let predicates =
        super::agent::AgentPredicates::init(engine).map_err(|e| JournalIngestError::ReadDir {
            path: dir.display().to_string(),
            source: std::io::Error::other(format!("{e}")),
        })?;
    ingest_journals_from_dir(engine, dir, &predicates)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_markdown_with_headings() {
        let md = "\
# Daily Summary

Everything went well.

## Cluster Topology

Nodes: 3 workers.

## Decisions

Chose Rust.
";
        let chunks = chunk_markdown(md, 800);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, "Daily Summary");
        assert!(chunks[0].content.contains("Everything went well"));
        assert_eq!(chunks[1].heading, "Cluster Topology");
        assert_eq!(chunks[2].heading, "Decisions");
    }

    #[test]
    fn chunk_markdown_no_headings() {
        let md = "Some prose without headings.\nMore prose here.\n";
        let chunks = chunk_markdown(md, 800);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading.is_empty());
        assert!(chunks[0].content.contains("prose"));
    }

    #[test]
    fn chunk_markdown_empty_body() {
        let md = "## Heading Only\n\n## Another\n\nText here.";
        let chunks = chunk_markdown(md, 800);
        // First heading has no body → no chunk; second has body → 1 chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "Another");
    }

    #[test]
    fn chunk_markdown_max_chars_truncation() {
        let body = "A".repeat(1000);
        let md = format!("## Section\n\n{body}");
        let chunks = chunk_markdown(&md, 100);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.ends_with('…'));
        // Content should be 100 bytes of 'A' + '…'
        assert!(chunks[0].content.len() <= 104); // 100 + 3-byte '…'
    }

    #[test]
    fn extract_keywords_from_heading_and_body() {
        let kws = extract_keywords("Cluster Topology", "The Kubernetes cluster has 3 Workers.");
        assert!(kws.contains(&"cluster".to_string()));
        assert!(kws.contains(&"topology".to_string()));
        // "Kubernetes" is mid-sentence capitalized → included
        assert!(kws.contains(&"kubernetes".to_string()));
    }

    #[test]
    fn ingest_journal_file_idempotent() {
        use crate::agent::agent::AgentPredicates;
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let dir = tempfile::TempDir::new().unwrap();
        let file_path = dir.path().join("2026-05-25.md");
        std::fs::write(
            &file_path,
            "# Daily Summary\n\nAll good today.\n\n## Decisions\n\nChose Rust for the project.\n",
        )
        .unwrap();

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let predicates = AgentPredicates::init(&engine).unwrap();

        // First ingest creates episodes.
        let n1 = ingest_journal_file(&engine, &file_path, &predicates).unwrap();
        assert!(n1 > 0, "first ingest should create episodes");

        // Second ingest is a no-op.
        let n2 = ingest_journal_file(&engine, &file_path, &predicates).unwrap();
        assert_eq!(n2, 0, "second ingest must be idempotent");
    }

    #[test]
    fn ingest_journals_from_dir_basic() {
        use crate::agent::agent::AgentPredicates;
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("2026-05-24.md"), "## Topology\n\nAll nodes up.\n").unwrap();
        std::fs::write(dir.path().join("2026-05-25.md"), "## Goals\n\nImplemented recall.\n")
            .unwrap();
        std::fs::write(dir.path().join("notes.txt"), "ignored file").unwrap();

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let predicates = AgentPredicates::init(&engine).unwrap();

        let stats = ingest_journals_from_dir(&engine, dir.path(), &predicates).unwrap();
        assert_eq!(stats.files_scanned, 2);
        assert_eq!(stats.files_ingested, 2);
        assert_eq!(stats.files_skipped, 0);
        assert!(stats.episodes_created >= 2);

        // Re-run: all files skipped.
        let stats2 = ingest_journals_from_dir(&engine, dir.path(), &predicates).unwrap();
        assert_eq!(stats2.files_ingested, 0);
        assert_eq!(stats2.files_skipped, 2);
    }
}
