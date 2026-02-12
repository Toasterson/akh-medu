//! The `ConcreteGrammar` trait — the interface every archetype implements.
//!
//! A concrete grammar defines both directions of the bidirectional mapping:
//! - `linearize()`: abstract syntax → prose (symbol→prose)
//! - `parse()`: prose → abstract syntax (prose→symbol)
//!
//! Each archetype (formal, terse, narrative, custom) implements this trait
//! with its own style of rendering and pattern matching.

use crate::registry::SymbolRegistry;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

use super::abs::AbsTree;
use super::cat::Cat;
use super::error::GrammarResult;
use super::lexer::{Language, Lexicon};

/// Context available during linearization (AbsTree → prose).
///
/// Provides access to the symbol registry for label resolution and
/// optional engine for enrichment lookups.
pub struct LinContext<'a> {
    /// Symbol registry for resolving IDs to labels.
    pub registry: Option<&'a SymbolRegistry>,
    /// Override lexicon for language-specific surface forms.
    pub lexicon: Option<Lexicon>,
    /// Target language for linearization.
    pub language: Language,
}

impl<'a> LinContext<'a> {
    /// Create a context with a symbol registry.
    pub fn with_registry(registry: &'a SymbolRegistry) -> Self {
        Self {
            registry: Some(registry),
            lexicon: None,
            language: Language::Auto,
        }
    }

    /// Resolve a label: if a symbol registry is available, look up the
    /// canonical label; otherwise return the original.
    pub fn resolve_label(&self, label: &str, symbol_id: Option<crate::symbol::SymbolId>) -> String {
        if let (Some(registry), Some(id)) = (self.registry, symbol_id) {
            if let Some(meta) = registry.get(id) {
                return meta.label;
            }
        }
        label.to_string()
    }
}

impl Default for LinContext<'_> {
    fn default() -> Self {
        Self {
            registry: None,
            lexicon: None,
            language: Language::Auto,
        }
    }
}

/// Context available during parsing (prose → AbsTree).
///
/// Provides access to the symbol registry for exact token resolution
/// and VSA item memory for fuzzy/similarity-based disambiguation.
pub struct ParseContext<'a> {
    /// Symbol registry for exact label lookups.
    pub registry: Option<&'a SymbolRegistry>,
    /// VSA operations for encoding tokens to hypervectors.
    pub ops: Option<&'a VsaOps>,
    /// Item memory for similarity-based token resolution.
    pub item_memory: Option<&'a ItemMemory>,
    /// Override lexicon (if `None`, selected by `language`).
    pub lexicon: Option<Lexicon>,
    /// Language to parse in.
    pub language: Language,
}

impl<'a> ParseContext<'a> {
    /// Create a context with full engine access.
    pub fn with_engine(
        registry: &'a SymbolRegistry,
        ops: &'a VsaOps,
        item_memory: &'a ItemMemory,
    ) -> Self {
        Self {
            registry: Some(registry),
            ops: Some(ops),
            item_memory: Some(item_memory),
            lexicon: None,
            language: Language::Auto,
        }
    }

    /// Create a context with full engine access and explicit language.
    pub fn with_engine_and_language(
        registry: &'a SymbolRegistry,
        ops: &'a VsaOps,
        item_memory: &'a ItemMemory,
        language: Language,
    ) -> Self {
        Self {
            registry: Some(registry),
            ops: Some(ops),
            item_memory: Some(item_memory),
            lexicon: None,
            language,
        }
    }
}

impl Default for ParseContext<'_> {
    fn default() -> Self {
        Self {
            registry: None,
            ops: None,
            item_memory: None,
            lexicon: None,
            language: Language::Auto,
        }
    }
}

/// A concrete grammar — one voice for the abstract syntax.
///
/// Implementations provide both directions of the bidirectional mapping:
/// linearization (symbol→prose) and parsing (prose→symbol).
pub trait ConcreteGrammar: Send + Sync {
    /// The name of this grammar (e.g., "formal", "terse", "narrative").
    fn name(&self) -> &str;

    /// A human-readable description of this grammar's style.
    fn description(&self) -> &str;

    /// Linearize an abstract syntax tree into prose.
    ///
    /// The `ctx` provides optional access to the symbol registry for
    /// resolving IDs to canonical labels.
    fn linearize(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String>;

    /// Parse prose input into an abstract syntax tree.
    ///
    /// `expected_cat` optionally constrains what category the parser
    /// should produce (e.g., `Some(Cat::Statement)` for a triple).
    /// The `ctx` provides optional access to the registry and VSA
    /// for token resolution.
    fn parse(
        &self,
        input: &str,
        expected_cat: Option<Cat>,
        ctx: &ParseContext,
    ) -> GrammarResult<AbsTree>;

    /// The categories this grammar can linearize.
    fn supported_categories(&self) -> &[Cat];
}
