//! Hieroglyphic notation system for knowledge representation.
//!
//! This module provides a visual notation for displaying knowledge graph
//! structures using a combination of fixed hieroglyphic glyphs (for common
//! predicates, types, and structural markers) and dynamic sigils generated
//! from VSA vectors (giving each concept a unique visual identity).
//!
//! ## Components
//!
//! - [`catalog`] — 35 fixed glyphs mapped to PUA codepoints U+E000–U+E022
//! - [`sigil`] — 32 hieroglyphic radicals (U+E100–U+E11F) composed from VSA vectors
//! - [`notation`] — triple/subgraph → hieroglyphic string formatting
//! - [`render`] — ANSI terminal renderer with color coding

pub mod catalog;
pub mod notation;
pub mod render;
pub mod sigil;

pub use catalog::{Glyph, GlyphCategory};
pub use notation::NotationConfig;
pub use render::RenderConfig;
pub use sigil::{Radical, RadicalCategory};

use miette::Diagnostic;
use thiserror::Error;

/// Errors that can occur during glyph operations.
#[derive(Debug, Error, Diagnostic)]
pub enum GlyphError {
    #[error("unknown glyph label: '{label}'")]
    #[diagnostic(
        code(akh::glyph::unknown),
        help("Use catalog::all_glyphs() to see available glyphs.")
    )]
    UnknownGlyph { label: String },

    #[error("symbol {symbol_id} has no VSA vector for sigil generation")]
    #[diagnostic(
        code(akh::glyph::no_vector),
        help("Ingest the symbol first so it gets a VSA vector.")
    )]
    NoVector { symbol_id: String },
}

/// Result type for glyph operations.
pub type GlyphResult<T> = Result<T, GlyphError>;
