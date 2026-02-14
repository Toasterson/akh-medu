//! Library search tool: find relevant content in the shared library via VSA.
//!
//! Encodes a natural language query as a hypervector, searches the engine's
//! item memory, and filters results to library paragraph symbols only.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::vsa::grounding::encode_text_as_vector;

/// Search the shared content library for paragraphs matching a natural language query.
pub struct LibrarySearchTool;

impl Tool for LibrarySearchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "library_search".into(),
            description: "Search the shared content library for paragraphs matching a \
                          natural language query via VSA semantic similarity."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "query".into(),
                    description: "Natural language search text.".into(),
                    required: true,
                },
                ToolParam {
                    name: "top_k".into(),
                    description: "Number of results to return (default: 5).".into(),
                    required: false,
                },
                ToolParam {
                    name: "document".into(),
                    description: "Filter to a specific document slug (optional).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query = input.require("query", "library_search")?;
        let top_k: usize = input.get("top_k").and_then(|k| k.parse().ok()).unwrap_or(5);
        let document_filter = input.get("document").map(|s| s.to_string());

        let ops = engine.ops();
        let im = engine.item_memory();

        // Encode the query text as a hypervector.
        let query_vec = match encode_text_as_vector(query, engine, ops, im) {
            Ok(v) => v,
            Err(_) => {
                return Ok(ToolOutput::err(
                    "Failed to encode query as hypervector. The VSA item memory may be empty.",
                ));
            }
        };

        // Over-fetch to have enough candidates after filtering.
        let fetch_count = (top_k * 5).min(100);
        let results = match engine.search_similar(&query_vec, fetch_count) {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutput::err(format!("Similarity search failed: {e}"))),
        };

        // Filter to paragraph symbols only (label prefix "para:").
        let mut para_results: Vec<_> = results
            .into_iter()
            .filter(|sr| {
                let label = engine.resolve_label(sr.symbol_id);
                if !label.starts_with("para:") {
                    return false;
                }
                // If a document filter is specified, require "para:{slug}:" prefix.
                if let Some(ref slug) = document_filter {
                    let expected_prefix = format!("para:{slug}:");
                    if !label.starts_with(&expected_prefix) {
                        return false;
                    }
                }
                true
            })
            .collect();

        para_results.truncate(top_k);

        if para_results.is_empty() {
            let filter_msg = document_filter
                .as_ref()
                .map(|s| format!(" in document \"{s}\""))
                .unwrap_or_default();
            return Ok(ToolOutput::ok(format!(
                "No library content found for \"{query}\"{filter_msg}."
            )));
        }

        let mut lines = Vec::new();
        let mut symbols = Vec::new();
        for (i, sr) in para_results.iter().enumerate() {
            let label = engine.resolve_label(sr.symbol_id);
            symbols.push(sr.symbol_id);

            // Try to extract document slug and chunk index from "para:{slug}:{index}".
            let context = label
                .strip_prefix("para:")
                .map(|rest| {
                    if let Some(colon_pos) = rest.rfind(':') {
                        let slug = &rest[..colon_pos];
                        let idx = &rest[colon_pos + 1..];
                        format!("doc=\"{slug}\" chunk={idx}")
                    } else {
                        format!("id=\"{rest}\"")
                    }
                })
                .unwrap_or_default();

            lines.push(format!(
                "  {}. [{context}] (similarity: {:.4})",
                i + 1,
                sr.similarity,
            ));
        }

        let filter_msg = document_filter
            .as_ref()
            .map(|s| format!(" in \"{s}\""))
            .unwrap_or_default();
        let result = format!(
            "Library search for \"{query}\"{filter_msg} ({} results):\n{}",
            para_results.len(),
            lines.join("\n"),
        );
        Ok(ToolOutput::ok_with_symbols(result, symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "library_search".into(),
            description: "Search the shared content library for paragraphs via VSA similarity."
                .into(),
            parameters: vec![
                ToolParamSchema::required("query", "Natural language search text."),
                ToolParamSchema::optional("top_k", "Number of results to return (default: 5)."),
                ToolParamSchema::optional("document", "Filter to a specific document slug."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::VsaAccess, Capability::ReadKg]),
                description:
                    "VSA-based semantic search of library content — read-only, no side effects."
                        .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
