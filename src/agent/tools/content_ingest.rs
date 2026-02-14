//! Content ingest tool: add documents (files/URLs) to the shared library.
//!
//! Accepts a `source` parameter (file path or URL), optional `title` and `tags`.
//! Delegates to the library ingestion pipeline for parsing, chunking, triple
//! extraction, and VSA embedding.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::library::catalog::LibraryCatalog;
use crate::library::ingest::{IngestConfig, ingest_file, ingest_url};
use crate::paths::AkhPaths;

/// Tool for ingesting documents (files, URLs) into the shared content library.
pub struct ContentIngestTool;

impl Tool for ContentIngestTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "content_ingest".into(),
            description: "Ingest a document (file or URL) into the shared content library. \
                          Parses HTML, PDF, EPUB, or plain text. Extracts triples and \
                          creates VSA embeddings for semantic search."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "source".into(),
                    description: "File path or URL to ingest.".into(),
                    required: true,
                },
                ToolParam {
                    name: "title".into(),
                    description: "Override document title (optional).".into(),
                    required: false,
                },
                ToolParam {
                    name: "tags".into(),
                    description: "Comma-separated tags for categorization (optional).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let source = input.require("source", "content_ingest")?;
        let title = input.get("title").map(|s| s.to_string());
        let tags: Vec<String> = input
            .get("tags")
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();

        // Resolve library directory from XDG paths.
        let library_dir = match AkhPaths::resolve() {
            Ok(paths) => paths.library_dir(),
            Err(_) => {
                return Ok(ToolOutput::err(
                    "Cannot resolve library directory. Set HOME environment variable.",
                ));
            }
        };

        // Open catalog.
        let mut catalog = match LibraryCatalog::open(&library_dir) {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::err(format!("Cannot open catalog: {e}"))),
        };

        let config = IngestConfig {
            title,
            tags,
            ..Default::default()
        };

        let result = if source.starts_with("http://") || source.starts_with("https://") {
            ingest_url(engine, &mut catalog, source, config)
        } else {
            let path = PathBuf::from(source);
            ingest_file(engine, &mut catalog, &path, config)
        };

        match result {
            Ok(res) => {
                let msg = format!(
                    "Ingested \"{}\" (id={}, {} chunks, {} triples, format={}).",
                    res.record.title,
                    res.record.id,
                    res.chunk_count,
                    res.triple_count,
                    res.record.format,
                );
                Ok(ToolOutput::ok_with_symbols(msg, vec![res.document_symbol]))
            }
            Err(e) => Ok(ToolOutput::err(format!("Ingestion failed: {e}"))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "content_ingest".into(),
            description: "Ingest documents (files, URLs) into the shared content library."
                .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "source",
                    "File path or URL to ingest into the library.",
                ),
                ToolParamSchema::optional(
                    "title",
                    "Override document title.",
                ),
                ToolParamSchema::optional(
                    "tags",
                    "Comma-separated tags for categorization.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([
                    Capability::WriteKg,
                    Capability::ReadFilesystem,
                    Capability::Network,
                    Capability::VsaAccess,
                ]),
                description: "Fetches/reads documents, parses them, and writes triples + VSA \
                              embeddings to the knowledge graph."
                    .into(),
                shadow_triggers: vec!["ingest".into(), "import".into()],
            },
            source: ToolSource::Native,
        }
    }
}
