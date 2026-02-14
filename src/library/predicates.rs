//! Well-known `doc:*` relation predicates for the content library.
//!
//! Follows the same pattern as `AgentPredicates`: a set of well-known
//! SymbolIds resolved (or created) at library initialization time.

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::SymbolId;

/// Well-known relation predicates for document structure and metadata.
///
/// All predicate labels are prefixed with `doc:` to avoid collisions with
/// other subsystems.
pub struct LibraryPredicates {
    // Structural relations
    pub has_chapter: SymbolId,
    pub has_section: SymbolId,
    pub has_paragraph: SymbolId,
    pub next_chunk: SymbolId,

    // Metadata relations
    pub has_title: SymbolId,
    pub has_author: SymbolId,
    pub has_format: SymbolId,
    pub has_source: SymbolId,
    pub has_language: SymbolId,
    pub has_description: SymbolId,
    pub has_keyword: SymbolId,
    pub has_tag: SymbolId,

    // Chunk-level relations
    pub chunk_text: SymbolId,
    pub chunk_index: SymbolId,
}

impl LibraryPredicates {
    /// Resolve or create all 14 well-known doc predicates in the engine.
    pub fn init(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            has_chapter: engine.resolve_or_create_relation("doc:has_chapter")?,
            has_section: engine.resolve_or_create_relation("doc:has_section")?,
            has_paragraph: engine.resolve_or_create_relation("doc:has_paragraph")?,
            next_chunk: engine.resolve_or_create_relation("doc:next_chunk")?,
            has_title: engine.resolve_or_create_relation("doc:has_title")?,
            has_author: engine.resolve_or_create_relation("doc:has_author")?,
            has_format: engine.resolve_or_create_relation("doc:has_format")?,
            has_source: engine.resolve_or_create_relation("doc:has_source")?,
            has_language: engine.resolve_or_create_relation("doc:has_language")?,
            has_description: engine.resolve_or_create_relation("doc:has_description")?,
            has_keyword: engine.resolve_or_create_relation("doc:has_keyword")?,
            has_tag: engine.resolve_or_create_relation("doc:has_tag")?,
            chunk_text: engine.resolve_or_create_relation("doc:chunk_text")?,
            chunk_index: engine.resolve_or_create_relation("doc:chunk_index")?,
        })
    }
}
