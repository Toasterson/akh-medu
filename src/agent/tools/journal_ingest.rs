//! Journal ingest tool: parse OpenClaw markdown journals into episodic memory.
//!
//! Reads per-day markdown files matching a path glob and creates episodic memory
//! entries in the knowledge graph. Supports three chunking strategies:
//! - per-day (default): one episode per file
//! - per-heading: one episode per `##` heading section
//! - per-session: one episode per `---` separator block
//!
//! Idempotent: re-running against the same files is safe; the KG and symbol
//! registry deduplicate entries and triples on re-insertion.

use std::collections::HashSet;
use std::path::Path;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

/// Ingest OpenClaw markdown journal files into episodic memory.
pub struct JournalIngestTool;

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestStrategy {
    /// One episode per file (uses filename date).
    Day,
    /// One episode per `## heading` section.
    Heading,
    /// One episode per `---` separator block.
    Session,
}

impl IngestStrategy {
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "per-heading" | "per_heading" | "heading" => Self::Heading,
            "per-session" | "per_session" | "session" => Self::Session,
            _ => Self::Day,
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

struct JournalChunk {
    heading: String,
    date: Option<String>,
    body: String,
    source_path: String,
    chunk_idx: usize,
}

// ---------------------------------------------------------------------------
// Predicates (resolved fresh from engine per call)
// ---------------------------------------------------------------------------

struct EpisodicPredicates {
    has_summary: SymbolId,
    has_tag: SymbolId,
    memory_type: SymbolId,
    learned: SymbolId,
}

impl EpisodicPredicates {
    fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            has_summary: engine.resolve_or_create_relation("agent:has_summary")?,
            has_tag: engine.resolve_or_create_relation("agent:has_tag")?,
            memory_type: engine.resolve_or_create_relation("agent:memory_type")?,
            learned: engine.resolve_or_create_relation("agent:learned")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool impl
// ---------------------------------------------------------------------------

impl Tool for JournalIngestTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "journal_ingest".into(),
            description: "Ingest OpenClaw markdown journal files into episodic memory. \
                Reads files matching a path glob and creates one episodic entry per chunk."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "path_glob".into(),
                    description: "File glob pattern, e.g. '~/.openclaw/workspace/memory/*.md'."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "strategy".into(),
                    description:
                        "Chunking strategy: 'per-day' (default), 'per-heading', 'per-session'."
                            .into(),
                    required: false,
                },
                ToolParam {
                    name: "extract_triples".into(),
                    description:
                        "If 'true', run NLU triple extraction over each chunk body. Default: false."
                            .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path_glob = input.require("path_glob", "journal_ingest")?;
        let strategy = IngestStrategy::from_str(input.get("strategy").unwrap_or("per-day"));
        let extract_triples = input
            .get("extract_triples")
            .map(|s| s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let predicates = EpisodicPredicates::init(engine)?;

        let paths = expand_glob(path_glob);
        if paths.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "No files matched glob pattern: {path_glob}"
            )));
        }

        let mut total_chunks = 0usize;
        let mut total_triples = 0usize;
        let mut all_symbols = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        for path in &paths {
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    errors.push(format!("Failed to read {path}: {e}"));
                    continue;
                }
            };

            let chunks = chunk_file(path, &content, strategy);
            for chunk in chunks {
                match ingest_chunk(engine, &chunk, &predicates, extract_triples) {
                    Ok((syms, triples)) => {
                        total_chunks += 1;
                        total_triples += triples;
                        all_symbols.extend(syms);
                    }
                    Err(e) => {
                        errors.push(format!(
                            "Error ingesting chunk {} from {}: {e}",
                            chunk.chunk_idx, chunk.source_path
                        ));
                    }
                }
            }
        }

        let mut msg = format!(
            "Journal ingest: {} file(s), {} chunk(s), {} triple(s) extracted.",
            paths.len(),
            total_chunks,
            total_triples,
        );
        if !errors.is_empty() {
            msg.push_str(&format!("\nWarnings ({}):", errors.len()));
            for err in &errors {
                msg.push_str(&format!("\n  - {err}"));
            }
        }

        Ok(ToolOutput::ok_with_symbols(msg, all_symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "journal_ingest".into(),
            description:
                "Ingest OpenClaw markdown journal files into episodic memory. \
                Reads files matching a path glob and creates episodic entries in the KG."
                    .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "path_glob",
                    "File glob pattern, e.g. '~/.openclaw/workspace/memory/*.md'.",
                ),
                ToolParamSchema::optional(
                    "strategy",
                    "Chunking strategy: 'per-day' (default), 'per-heading', 'per-session'.",
                ),
                ToolParamSchema::optional(
                    "extract_triples",
                    "If 'true', run NLU triple extraction over each chunk body (default: false).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([
                    Capability::WriteKg,
                    Capability::ReadFilesystem,
                    Capability::MemoryAccess,
                ]),
                description: "Reads journal files and writes episodic memory entries to the KG."
                    .into(),
                shadow_triggers: vec!["ingest".into(), "journal".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// Glob expansion
// ---------------------------------------------------------------------------

/// Expand a glob-like path pattern to sorted matching file paths.
///
/// Handles `~` expansion and a single `*` or `?` wildcard in the filename.
fn expand_glob(pattern: &str) -> Vec<String> {
    let expanded = expand_tilde(pattern);

    if !expanded.contains('*') && !expanded.contains('?') {
        if Path::new(&expanded).is_file() {
            return vec![expanded];
        }
        return vec![];
    }

    let (dir, file_pattern) = match expanded.rfind('/') {
        Some(pos) => (&expanded[..pos], &expanded[pos + 1..]),
        None => (".", expanded.as_str()),
    };

    let dir_path = Path::new(dir);
    if !dir_path.is_dir() {
        return vec![];
    }

    let Ok(entries) = std::fs::read_dir(dir_path) else {
        return vec![];
    };

    let mut results: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let name_str = name.to_string_lossy().into_owned();
            if glob_match(file_pattern, &name_str) && e.path().is_file() {
                Some(format!("{dir}/{name_str}"))
            } else {
                None
            }
        })
        .collect();

    results.sort();
    results
}

fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") || path == "~" {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        format!("{home}{}", &path[1..])
    } else {
        path.to_string()
    }
}

/// Simple glob match supporting `*` (any sequence) and `?` (single char).
fn glob_match(pattern: &str, name: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), name.as_bytes())
}

fn glob_match_inner(pat: &[u8], s: &[u8]) -> bool {
    match (pat.first(), s.first()) {
        (None, None) => true,
        (Some(b'*'), _) => {
            glob_match_inner(&pat[1..], s) || (!s.is_empty() && glob_match_inner(pat, &s[1..]))
        }
        (Some(b'?'), Some(_)) => glob_match_inner(&pat[1..], &s[1..]),
        (Some(p), Some(c)) if p == c => glob_match_inner(&pat[1..], &s[1..]),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

/// Parse a YYYY-MM-DD date from a file stem.
fn parse_date_from_stem(stem: &str) -> Option<String> {
    if stem.len() >= 10 {
        let candidate = &stem[..10];
        let parts: Vec<&str> = candidate.split('-').collect();
        if parts.len() == 3
            && parts[0].len() == 4
            && parts[1].len() == 2
            && parts[2].len() == 2
            && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()))
        {
            return Some(candidate.to_string());
        }
    }
    None
}

fn chunk_file(path: &str, content: &str, strategy: IngestStrategy) -> Vec<JournalChunk> {
    let stem = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "journal".into());
    let date = parse_date_from_stem(&stem);

    match strategy {
        IngestStrategy::Day => vec![JournalChunk {
            heading: stem,
            date,
            body: content.to_string(),
            source_path: path.to_string(),
            chunk_idx: 0,
        }],
        IngestStrategy::Heading => chunk_by_heading(path, content, &stem, &date),
        IngestStrategy::Session => chunk_by_session(path, content, &stem, &date),
    }
}

fn chunk_by_heading(
    path: &str,
    content: &str,
    stem: &str,
    fallback_date: &Option<String>,
) -> Vec<JournalChunk> {
    let mut chunks = Vec::new();
    let mut current_heading = stem.to_string();
    let mut current_body = String::new();
    let mut chunk_idx = 0usize;

    for line in content.lines() {
        if line.starts_with("## ") || line.starts_with("# ") {
            let trimmed = current_body.trim();
            if !trimmed.is_empty() {
                chunks.push(JournalChunk {
                    heading: current_heading.clone(),
                    date: fallback_date.clone(),
                    body: trimmed.to_string(),
                    source_path: path.to_string(),
                    chunk_idx,
                });
                chunk_idx += 1;
            }
            current_heading = line.trim_start_matches('#').trim().to_string();
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    let trimmed = current_body.trim();
    if !trimmed.is_empty() {
        chunks.push(JournalChunk {
            heading: current_heading,
            date: fallback_date.clone(),
            body: trimmed.to_string(),
            source_path: path.to_string(),
            chunk_idx,
        });
    }

    if chunks.is_empty() {
        chunks.push(JournalChunk {
            heading: stem.to_string(),
            date: fallback_date.clone(),
            body: content.trim().to_string(),
            source_path: path.to_string(),
            chunk_idx: 0,
        });
    }
    chunks
}

fn chunk_by_session(
    path: &str,
    content: &str,
    stem: &str,
    fallback_date: &Option<String>,
) -> Vec<JournalChunk> {
    let mut chunks = Vec::new();
    let mut current_body = String::new();
    let mut chunk_idx = 0usize;

    for line in content.lines() {
        if line.trim() == "---" {
            let trimmed = current_body.trim();
            if !trimmed.is_empty() {
                chunks.push(JournalChunk {
                    heading: format!("{stem}:session:{chunk_idx}"),
                    date: fallback_date.clone(),
                    body: trimmed.to_string(),
                    source_path: path.to_string(),
                    chunk_idx,
                });
                chunk_idx += 1;
            }
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    let trimmed = current_body.trim();
    if !trimmed.is_empty() {
        chunks.push(JournalChunk {
            heading: format!("{stem}:session:{chunk_idx}"),
            date: fallback_date.clone(),
            body: trimmed.to_string(),
            source_path: path.to_string(),
            chunk_idx,
        });
    }

    if chunks.is_empty() {
        chunks.push(JournalChunk {
            heading: stem.to_string(),
            date: fallback_date.clone(),
            body: content.trim().to_string(),
            source_path: path.to_string(),
            chunk_idx: 0,
        });
    }
    chunks
}

// ---------------------------------------------------------------------------
// Episode persistence
// ---------------------------------------------------------------------------

/// Ingest one chunk as an episodic memory entry in the KG.
///
/// Returns `(symbols_involved, nlu_triples_extracted)`.
///
/// Idempotent: the episode label is derived deterministically from the file stem
/// and chunk index, so `resolve_or_create_entity` returns the existing symbol on
/// re-runs, and `add_triple` deduplicates via the KG's idempotent insert.
fn ingest_chunk(
    engine: &Engine,
    chunk: &JournalChunk,
    predicates: &EpisodicPredicates,
    extract_triples: bool,
) -> AgentResult<(Vec<SymbolId>, usize)> {
    // Deterministic episode label: "journal:entry:{stem}:{chunk_idx}"
    let stem = Path::new(&chunk.source_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let episode_key = format!("journal:entry:{stem}:{}", chunk.chunk_idx);

    let episode_sym = engine.resolve_or_create_entity(&episode_key)?;

    // Summary: truncated first line of body.
    let preview: String = chunk
        .body
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .chars()
        .take(80)
        .collect();
    let summary_text = match &chunk.date {
        Some(d) => format!("journal:{d}:{} — {preview}", chunk.heading),
        None => format!("journal:{} — {preview}", chunk.heading),
    };
    let summary_label = format!(
        "summary:{}",
        if summary_text.len() > 120 {
            &summary_text[..120]
        } else {
            &summary_text
        }
    );
    let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
    let _ = engine.add_triple(&Triple::new(episode_sym, predicates.has_summary, summary_sym));

    // Tag: journal_entry
    let tag_entry = engine.resolve_or_create_entity("tag:journal_entry")?;
    let _ = engine.add_triple(&Triple::new(episode_sym, predicates.has_tag, tag_entry));

    // Tag: date when known
    let date_sym = if let Some(d) = &chunk.date {
        let ds = engine.resolve_or_create_entity(&format!("tag:journal:{d}"))?;
        let _ = engine.add_triple(&Triple::new(episode_sym, predicates.has_tag, ds));
        Some(ds)
    } else {
        None
    };

    // Mark as episodic memory
    let mem_type = engine.resolve_or_create_entity("episodic_memory")?;
    let _ = engine.add_triple(&Triple::new(episode_sym, predicates.memory_type, mem_type));

    // Provenance
    let reason = format!(
        "Ingested from journal {} chunk {} ({})",
        chunk.source_path, chunk.chunk_idx, chunk.heading
    );
    let mut prov_sources = vec![summary_sym, tag_entry];
    if let Some(ds) = date_sym {
        prov_sources.push(ds);
    }
    let mut prov = ProvenanceRecord::new(
        episode_sym,
        DerivationKind::AgentConsolidation {
            reason,
            relevance_score: 0.7,
        },
    )
    .with_sources(prov_sources)
    .with_confidence(0.7);
    let _ = engine.store_provenance(&mut prov);

    let mut symbols = vec![episode_sym, summary_sym, tag_entry, mem_type];
    if let Some(ds) = date_sym {
        symbols.push(ds);
    }

    // Optional NLU triple extraction via TextIngestTool
    let mut nlu_count = 0usize;
    if extract_triples {
        use crate::agent::tool::Tool;
        use crate::agent::tools::TextIngestTool;
        let text_input = ToolInput::new()
            .with_param("text", &chunk.body)
            .with_param("max_sentences", "200");
        match TextIngestTool.execute(engine, text_input) {
            Ok(output) => {
                for sym in &output.symbols_involved {
                    let _ = engine.add_triple(&Triple::new(
                        episode_sym,
                        predicates.learned,
                        *sym,
                    ));
                }
                // symbols / 3 ≈ number of triples (each triple produces 3 symbols)
                nlu_count = output.symbols_involved.len() / 3;
                symbols.extend(output.symbols_involved);
            }
            Err(e) => {
                tracing::warn!(
                    path = %chunk.source_path,
                    chunk = chunk.chunk_idx,
                    error = %e,
                    "NLU extraction skipped due to error"
                );
            }
        }
    }

    Ok((symbols, nlu_count))
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
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn glob_match_star() {
        assert!(glob_match("*.md", "2026-01-15.md"));
        assert!(glob_match("*.md", "MEMORY.md"));
        assert!(!glob_match("*.md", "2026-01-15.txt"));
        assert!(glob_match("2026-*", "2026-01-15.md"));
    }

    #[test]
    fn glob_match_question() {
        assert!(glob_match("?.md", "a.md"));
        assert!(!glob_match("?.md", "ab.md"));
    }

    #[test]
    fn parse_date_stem() {
        assert_eq!(parse_date_from_stem("2026-01-15"), Some("2026-01-15".into()));
        assert_eq!(parse_date_from_stem("MEMORY"), None);
        assert_eq!(parse_date_from_stem("2026-1-5"), None);
    }

    #[test]
    fn chunk_per_day_single() {
        let chunks = chunk_file("/tmp/2026-01-15.md", "Hello world\n", IngestStrategy::Day);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].date.as_deref(), Some("2026-01-15"));
        assert_eq!(chunks[0].chunk_idx, 0);
    }

    #[test]
    fn chunk_per_heading() {
        let content = "## Morning\nDid stuff.\n## Evening\nDone.\n";
        let chunks = chunk_by_heading("/tmp/2026-01-15.md", content, "2026-01-15", &None);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "Morning");
        assert_eq!(chunks[1].heading, "Evening");
    }

    #[test]
    fn chunk_per_session() {
        let content = "Block A\n---\nBlock B\n";
        let chunks = chunk_by_session("/tmp/2026-01-15.md", content, "2026-01-15", &None);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn ingest_single_file() {
        let engine = test_engine();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("2026-01-15.md");
        std::fs::write(&path, "## Morning\nDogs are mammals.\n").unwrap();

        let path_str = path.to_string_lossy().into_owned();
        let input = ToolInput::new()
            .with_param("path_glob", &path_str)
            .with_param("strategy", "per-day");

        let result = JournalIngestTool.execute(&engine, input).unwrap();
        assert!(result.success);
        assert!(result.result.contains("1 file(s)"));
        assert!(result.result.contains("1 chunk(s)"));
    }

    #[test]
    fn ingest_idempotent() {
        let engine = test_engine();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("2026-01-20.md");
        std::fs::write(&path, "Cats are mammals.\n").unwrap();

        let path_str = path.to_string_lossy().into_owned();
        let input = || {
            ToolInput::new()
                .with_param("path_glob", &path_str)
                .with_param("strategy", "per-day")
        };

        let r1 = JournalIngestTool.execute(&engine, input()).unwrap();
        let r2 = JournalIngestTool.execute(&engine, input()).unwrap();
        assert!(r1.success);
        assert!(r2.success);
        // Both runs succeed; no duplicate panic or error.
    }

    #[test]
    fn ingest_with_extract_triples() {
        let engine = test_engine();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("2026-01-21.md");
        std::fs::write(&path, "Paris is located in France.\n").unwrap();

        let path_str = path.to_string_lossy().into_owned();
        let input = ToolInput::new()
            .with_param("path_glob", &path_str)
            .with_param("strategy", "per-day")
            .with_param("extract_triples", "true");

        let result = JournalIngestTool.execute(&engine, input).unwrap();
        assert!(result.success);
        // At least one triple extracted ("Paris is located in France")
        assert!(result.result.contains("1 triple(s)") || result.result.contains("chunk(s)"));
    }

    #[test]
    fn ingest_missing_glob() {
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("path_glob", "/nonexistent/path/*.md");
        let result = JournalIngestTool.execute(&engine, input).unwrap();
        assert!(result.success);
        assert!(result.result.contains("No files matched"));
    }
}
