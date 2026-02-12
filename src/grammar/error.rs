//! Rich diagnostic error types for the grammar subsystem.
//!
//! Follows the akh-medu miette pattern: every error variant carries
//! `#[diagnostic(code(...), help(...))]` so the user knows exactly
//! what went wrong and how to fix it.

use miette::Diagnostic;
use thiserror::Error;

use super::cat::Cat;

/// Errors produced by the bidirectional grammar system.
#[derive(Debug, Error, Diagnostic)]
pub enum GrammarError {
    #[error("parse failed for input: \"{input}\"")]
    #[diagnostic(
        code(akh::grammar::parse_failed),
        help(
            "The input could not be parsed into a known pattern. \
             Try simple subject-predicate-object sentences like \
             \"Dogs are mammals\" or questions like \"What is a dog?\". \
             You can also use explicit triple syntax: \"<subject> <predicate> <object>\"."
        )
    )]
    ParseFailed { input: String },

    #[error("linearization failed for category {cat:?} in grammar \"{grammar}\": {message}")]
    #[diagnostic(
        code(akh::grammar::lin_failed),
        help(
            "The concrete grammar could not produce prose for this abstract syntax node. \
             Check that the grammar supports the category, or try a different archetype."
        )
    )]
    LinearizationFailed {
        cat: Cat,
        grammar: String,
        message: String,
    },

    #[error("type mismatch: expected {expected:?}, got {actual:?}")]
    #[diagnostic(
        code(akh::grammar::type_mismatch),
        help(
            "An abstract syntax node has an unexpected category. \
             This usually means a child node has the wrong type \
             (e.g., a Triple's predicate slot contains an Entity instead of a Relation)."
        )
    )]
    TypeMismatch { expected: Cat, actual: Cat },

    #[error("unresolved entity: \"{label}\"")]
    #[diagnostic(
        code(akh::grammar::unresolved),
        help(
            "No symbol with this label was found in the registry or via VSA similarity. \
             To create a new symbol, use it in an assertion: \"{label} is a <category>\"."
        )
    )]
    UnresolvedEntity { label: String },

    #[error("unknown grammar: \"{name}\"")]
    #[diagnostic(
        code(akh::grammar::unknown_grammar),
        help(
            "No grammar with this name is registered. \
             Built-in grammars: \"formal\", \"terse\", \"narrative\". \
             Use GrammarRegistry::list() to see all available grammars."
        )
    )]
    UnknownGrammar { name: String },

    #[error("invalid custom grammar: {message}")]
    #[diagnostic(
        code(akh::grammar::invalid_custom),
        help(
            "The custom grammar definition is malformed. \
             Check the TOML syntax and ensure all required sections are present: \
             [grammar], [linearization], [parse.patterns]."
        )
    )]
    InvalidCustomGrammar { message: String },

    #[error("VSA grounding error: {message}")]
    #[diagnostic(
        code(akh::grammar::vsa_error),
        help(
            "Failed to ground abstract syntax against the VSA item memory. \
             Ensure the engine is initialized and symbols have been registered."
        )
    )]
    VsaError { message: String },

    #[error("ambiguous parse: {candidate_count} possible interpretations for \"{fragment}\"")]
    #[diagnostic(
        code(akh::grammar::ambiguous),
        help(
            "The input matches multiple parse patterns. \
             The highest-confidence interpretation was selected. \
             Rephrase for clarity, or use explicit triple syntax: \
             \"<subject> <predicate> <object>\"."
        )
    )]
    Ambiguous {
        fragment: String,
        candidate_count: usize,
    },

    #[error("incomplete sentence: expected {expected} after \"{fragment}\"")]
    #[diagnostic(
        code(akh::grammar::incomplete),
        help("The sentence appears truncated. Add the missing {expected} to complete it.")
    )]
    Incomplete { fragment: String, expected: String },

    #[error("grounding incomplete: {unresolved_count} label(s) unresolved, first: \"{first_unresolved}\"")]
    #[diagnostic(
        code(akh::grammar::grounding_incomplete),
        help(
            "Some labels in the abstract syntax tree could not be resolved to \
             known symbols. Ingest the missing entities first, or use fuzzy \
             resolution via VSA to match approximate labels. \
             First unresolved: \"{first_unresolved}\"."
        )
    )]
    GroundingIncomplete {
        unresolved_count: usize,
        first_unresolved: String,
    },

    #[error("unsupported language: \"{language}\"")]
    #[diagnostic(
        code(akh::grammar::unsupported_language),
        help(
            "The language \"{language}\" is not supported. \
             Supported languages: en (English), ru (Russian), ar (Arabic), \
             fr (French), es (Spanish), auto (auto-detect)."
        )
    )]
    UnsupportedLanguage { language: String },
}

/// Result type for grammar operations.
pub type GrammarResult<T> = std::result::Result<T, GrammarError>;
