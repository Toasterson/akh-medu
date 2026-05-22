//! Agent write tools for Akhipedia — Phase 22b.
//!
//! Provides three tools registered in the OODA tool registry:
//! - `WriteArticleTool`: create or update a wiki article
//! - `CiteArticleTool`: record a citation link between articles
//! - `RespondToArticleTool`: create a response article with `responds_to` relation
//!
//! Each tool restores the `ArticleStore` from durable storage, performs its
//! mutation, persists back, and returns a `ToolOutput` with the affected
//! article's `SymbolId`.

use std::collections::HashSet;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::wiki::store::{ArticleStore, BlockRef, BlockType, Stance};

// ═══════════════════════════════════════════════════════════════════════
// WriteArticleTool
// ═══════════════════════════════════════════════════════════════════════

/// Create or update a wiki article.
///
/// If an article with the given `title` already exists, its body and blocks
/// are replaced (update). Otherwise a new article is created.
pub struct WriteArticleTool;

impl Tool for WriteArticleTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "write_article".into(),
            description: "Create or update a wiki article with a Markdown body.".into(),
            parameters: vec![
                ToolParam {
                    name: "title".into(),
                    description: "Article title (unique identifier).".into(),
                    required: true,
                },
                ToolParam {
                    name: "body".into(),
                    description: "Markdown body content of the article.".into(),
                    required: true,
                },
                ToolParam {
                    name: "author".into(),
                    description: "Author entity label (defaults to the calling akh).".into(),
                    required: false,
                },
                ToolParam {
                    name: "subjects".into(),
                    description: "Comma-separated KG entity labels for article subjects.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let title = input.require("title", "write_article")?;
        let body = input.require("body", "write_article")?;

        let author = if let Some(author_label) = input.get("author") {
            engine
                .resolve_or_create_entity(author_label)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("author entity: {e}"),
                })?
        } else {
            engine
                .resolve_or_create_entity("akh:self")
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("akh:self entity: {e}"),
                })?
        };

        let subjects: Vec<_> = input
            .get("subjects")
            .map(|s| {
                s.split(',')
                    .map(|t| t.trim())
                    .filter(|t| !t.is_empty())
                    .filter_map(|t| engine.resolve_or_create_entity(t).ok())
                    .collect()
            })
            .unwrap_or_default();

        let blocks = auto_detect_blocks(body);

        let mut store = ArticleStore::restore(engine);

        if store.get_by_title(title).is_some() {
            // Update path.
            let block_count = blocks.len();
            store
                .update_article(title, body, blocks)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("update_article: {e}"),
                })?;
            store
                .persist(engine)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("persist: {e}"),
                })?;
            let article = store.get_by_title(title).unwrap();
            Ok(ToolOutput::ok_with_symbols(
                format!("Updated article \"{title}\" ({block_count} blocks)."),
                vec![article.header.id, author],
            ))
        } else {
            // Create path.
            let article = store
                .create_article(title, body, author, &subjects, blocks, engine)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("create_article: {e}"),
                })?;
            let block_count = article.header.blocks.len();
            let article_id = article.header.id;
            store
                .persist(engine)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write_article".into(),
                    message: format!("persist: {e}"),
                })?;
            Ok(ToolOutput::ok_with_symbols(
                format!("Created article \"{title}\" ({block_count} blocks, id={article_id})."),
                vec![article_id, author],
            ))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "write_article".into(),
            description: "Create or update a wiki article with a Markdown body.".into(),
            parameters: vec![
                ToolParamSchema::required("title", "Article title (unique identifier)."),
                ToolParamSchema::required("body", "Markdown body content of the article."),
                ToolParamSchema::optional("author", "Author entity label."),
                ToolParamSchema::optional(
                    "subjects",
                    "Comma-separated KG entity labels for article subjects.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Creates or updates wiki articles in durable storage.".into(),
                shadow_triggers: vec!["write article".into(), "create article".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CiteArticleTool
// ═══════════════════════════════════════════════════════════════════════

/// Record a citation from one article to another.
pub struct CiteArticleTool;

impl Tool for CiteArticleTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "cite_article".into(),
            description: "Record a citation link from one wiki article to another.".into(),
            parameters: vec![
                ToolParam {
                    name: "from".into(),
                    description: "Title of the citing article.".into(),
                    required: true,
                },
                ToolParam {
                    name: "to".into(),
                    description: "Title of the article being cited.".into(),
                    required: true,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let from = input.require("from", "cite_article")?;
        let to = input.require("to", "cite_article")?;

        let mut store = ArticleStore::restore(engine);
        store
            .add_citation(from, to)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "cite_article".into(),
                message: format!("add_citation: {e}"),
            })?;
        store
            .persist(engine)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "cite_article".into(),
                message: format!("persist: {e}"),
            })?;

        let symbols: Vec<_> = [from, to]
            .iter()
            .filter_map(|t| store.get_by_title(t).map(|a| a.header.id))
            .collect();

        Ok(ToolOutput::ok_with_symbols(
            format!("Citation recorded: \"{from}\" → \"{to}\"."),
            symbols,
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "cite_article".into(),
            description: "Record a citation link between two wiki articles.".into(),
            parameters: vec![
                ToolParamSchema::required("from", "Title of the citing article."),
                ToolParamSchema::required("to", "Title of the article being cited."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Adds a citation relation between wiki articles.".into(),
                shadow_triggers: vec!["cite".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RespondToArticleTool
// ═══════════════════════════════════════════════════════════════════════

/// Create a new article that explicitly responds to an existing one.
pub struct RespondToArticleTool;

impl Tool for RespondToArticleTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "respond_to_article".into(),
            description: "Create a new wiki article that responds to an existing one.".into(),
            parameters: vec![
                ToolParam {
                    name: "title".into(),
                    description: "Title of the new response article.".into(),
                    required: true,
                },
                ToolParam {
                    name: "body".into(),
                    description: "Markdown body of the response.".into(),
                    required: true,
                },
                ToolParam {
                    name: "responds_to".into(),
                    description: "Title of the article being responded to.".into(),
                    required: true,
                },
                ToolParam {
                    name: "author".into(),
                    description: "Author entity label (defaults to akh:self).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let title = input.require("title", "respond_to_article")?;
        let body = input.require("body", "respond_to_article")?;
        let responds_to_title = input.require("responds_to", "respond_to_article")?;

        let author = if let Some(label) = input.get("author") {
            engine
                .resolve_or_create_entity(label)
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "respond_to_article".into(),
                    message: format!("author: {e}"),
                })?
        } else {
            engine
                .resolve_or_create_entity("akh:self")
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "respond_to_article".into(),
                    message: format!("akh:self: {e}"),
                })?
        };

        let mut store = ArticleStore::restore(engine);

        // Verify the article being responded to exists.
        let parent_id = store
            .get_by_title(responds_to_title)
            .map(|a| a.header.id)
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "respond_to_article".into(),
                message: format!("article not found: \"{responds_to_title}\""),
            })?;

        let blocks = auto_detect_blocks(body);
        let article = store
            .create_article(title, body, author, &[], blocks, engine)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "respond_to_article".into(),
                message: format!("create_article: {e}"),
            })?;

        // Mark the `responds_to` link by adding a citation and setting the field.
        let article_id = article.header.id;
        store
            .set_responds_to(title, parent_id)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "respond_to_article".into(),
                message: format!("set_responds_to: {e}"),
            })?;
        store
            .persist(engine)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "respond_to_article".into(),
                message: format!("persist: {e}"),
            })?;

        Ok(ToolOutput::ok_with_symbols(
            format!(
                "Response article \"{title}\" created (responds to \"{responds_to_title}\")."
            ),
            vec![article_id, parent_id, author],
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "respond_to_article".into(),
            description: "Create a wiki article that responds to an existing one.".into(),
            parameters: vec![
                ToolParamSchema::required("title", "Title of the new response article."),
                ToolParamSchema::required("body", "Markdown body of the response."),
                ToolParamSchema::required("responds_to", "Title of the article being responded to."),
                ToolParamSchema::optional("author", "Author entity label."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Creates a response article with a responds_to relation.".into(),
                shadow_triggers: vec!["respond".into(), "reply".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SearchWikiTool
// ═══════════════════════════════════════════════════════════════════════

/// Simple in-memory title/body keyword search (observe-phase tool).
///
/// For full Tantivy-based ranked search see `src/wiki/search.rs` (requires
/// the `wiki` feature flag).
pub struct SearchWikiTool;

impl Tool for SearchWikiTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "search_wiki".into(),
            description: "Search wiki articles by keyword.".into(),
            parameters: vec![ToolParam {
                name: "query".into(),
                description: "Keyword(s) to search for in article titles and bodies.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query = input.require("query", "search_wiki")?;
        let lower = query.to_lowercase();

        let store = ArticleStore::restore(engine);

        let mut hits: Vec<_> = store
            .articles
            .values()
            .filter(|a| {
                a.header.title.to_lowercase().contains(&lower)
                    || a.body.to_lowercase().contains(&lower)
            })
            .collect();

        if hits.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "No wiki articles found matching \"{query}\"."
            )));
        }

        hits.sort_by(|a, b| a.header.title.cmp(&b.header.title));
        let symbols: Vec<_> = hits.iter().map(|a| a.header.id).collect();
        let listing = hits
            .iter()
            .map(|a| format!("  • {}", a.header.title))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolOutput::ok_with_symbols(
            format!("Found {} article(s) matching \"{query}\":\n{listing}", hits.len()),
            symbols,
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "search_wiki".into(),
            description: "Search wiki articles by keyword (title + body).".into(),
            parameters: vec![ToolParamSchema::required(
                "query",
                "Keyword(s) to search for.",
            )],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::new(),
                description: "Read-only keyword search over wiki articles.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Scan Markdown body and emit a rough block manifest by looking for
/// section headings and annotated fences (```claim, ```opinion, etc.).
fn auto_detect_blocks(body: &str) -> Vec<BlockRef> {
    let mut blocks = Vec::new();
    let mut offset = 0usize;

    for line in body.lines() {
        let len = line.len() + 1; // +1 for newline
        let trimmed = line.trim();

        let block_type = if trimmed.starts_with("```claim") {
            Some(BlockType::Claim { confidence: 0.8 })
        } else if trimmed.starts_with("```opinion") {
            Some(BlockType::Opinion {
                stance: Stance {
                    position: "nuanced".into(),
                    confidence: 0.7,
                    reasoning: String::new(),
                },
            })
        } else if trimmed.starts_with("```example") || trimmed.starts_with("```rust")
            || trimmed.starts_with("```python")
        {
            Some(BlockType::Example)
        } else if trimmed.starts_with("## ") || trimmed.starts_with("# ") {
            Some(BlockType::Prose)
        } else if trimmed.starts_with("> **Definition") {
            Some(BlockType::Definition)
        } else if trimmed.starts_with("> **Relationship") {
            Some(BlockType::Relationship)
        } else {
            None
        };

        if let Some(bt) = block_type {
            let id = format!("block-{}", blocks.len() + 1);
            blocks.push(BlockRef {
                id,
                block_type: bt,
                offset,
                length: len,
            });
        }

        offset += len;
    }

    blocks
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::{Engine, EngineConfig};
    use tempfile::TempDir;

    /// In-memory engine (no durable store) — sufficient for single-tool tests.
    fn engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    /// Engine backed by a temp directory so that `persist`/`restore` round-trips
    /// work correctly across multiple tool invocations within one test.
    fn persistent_engine() -> (Engine, TempDir) {
        let dir = TempDir::new().unwrap();
        let config = EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        (Engine::new(config).unwrap(), dir)
    }

    fn input(pairs: &[(&str, &str)]) -> ToolInput {
        let mut inp = ToolInput::new();
        for (k, v) in pairs {
            inp = inp.with_param(*k, *v);
        }
        inp
    }

    #[test]
    fn write_article_create() {
        let eng = engine();
        let tool = WriteArticleTool;
        let out = tool
            .execute(
                &eng,
                input(&[("title", "Rust Basics"), ("body", "# Rust\n\nA systems language.")]),
            )
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Created"));
        assert!(out.result.contains("Rust Basics"));
    }

    #[test]
    fn write_article_update() {
        let (eng, _dir) = persistent_engine();
        let tool = WriteArticleTool;
        // Create first.
        tool.execute(
            &eng,
            input(&[("title", "Update Me"), ("body", "v1")]),
        )
        .unwrap();
        // Update — article now exists in persisted store.
        let out = tool
            .execute(&eng, input(&[("title", "Update Me"), ("body", "v2")]))
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Updated"));
    }

    #[test]
    fn cite_article() {
        let (eng, _dir) = persistent_engine();
        let write = WriteArticleTool;
        write
            .execute(&eng, input(&[("title", "Source"), ("body", "src body")]))
            .unwrap();
        write
            .execute(
                &eng,
                input(&[("title", "Citing"), ("body", "cites src")]),
            )
            .unwrap();
        let cite = CiteArticleTool;
        let out = cite
            .execute(&eng, input(&[("from", "Citing"), ("to", "Source")]))
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Citation recorded"));
    }

    #[test]
    fn respond_to_article() {
        let (eng, _dir) = persistent_engine();
        let write = WriteArticleTool;
        write
            .execute(&eng, input(&[("title", "Original"), ("body", "original body")]))
            .unwrap();
        let respond = RespondToArticleTool;
        let out = respond
            .execute(
                &eng,
                input(&[
                    ("title", "Response"),
                    ("body", "response body"),
                    ("responds_to", "Original"),
                ]),
            )
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Response"));
        assert!(out.result.contains("Original"));
    }

    #[test]
    fn respond_to_missing_article() {
        let eng = engine();
        let tool = RespondToArticleTool;
        let out = tool.execute(
            &eng,
            input(&[
                ("title", "Response"),
                ("body", "body"),
                ("responds_to", "NonExistent"),
            ]),
        );
        assert!(out.is_err());
    }

    #[test]
    fn search_wiki_basic() {
        let (eng, _dir) = persistent_engine();
        let write = WriteArticleTool;
        write
            .execute(&eng, input(&[("title", "Rust"), ("body", "systems language")]))
            .unwrap();
        write
            .execute(
                &eng,
                input(&[("title", "Python"), ("body", "scripting language")]),
            )
            .unwrap();
        let search = SearchWikiTool;
        let out = search
            .execute(&eng, input(&[("query", "systems")]))
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Rust"));
        assert!(!out.result.contains("Python"));
    }

    #[test]
    fn search_wiki_no_results() {
        let eng = engine();
        let search = SearchWikiTool;
        let out = search
            .execute(&eng, input(&[("query", "nonexistent")]))
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("No wiki articles found"));
    }

    #[test]
    fn auto_detect_blocks_kinds() {
        let body = "# Introduction\n```claim\nassert x\n```\n```opinion\nthink y\n```\n";
        let blocks = auto_detect_blocks(body);
        // At least claim and opinion blocks should be detected.
        let kinds: Vec<_> = blocks.iter().map(|b| b.block_type.as_label()).collect();
        assert!(kinds.contains(&"prose"));
        assert!(kinds.contains(&"claim"));
        assert!(kinds.contains(&"opinion"));
    }
}
