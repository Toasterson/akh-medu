//! GF-inspired bidirectional grammar system.
//!
//! This module implements a Grammatical Framework-inspired system for
//! translating between prose and symbolic representations. The key insight
//! is the **abstract/concrete syntax split**: one shared semantic
//! representation ([`AbsTree`]) can be linearized through multiple
//! concrete grammars (archetypes) to produce different styles of output.
//!
//! ## Architecture
//!
//! ```text
//! Prose Input ──→ Lexer ──→ Parser ──→ AbsTree ──→ ConcreteGrammar ──→ Styled Prose
//!                   │          │          ↕              ↕
//!             SymbolRegistry  VSA    bridge.rs     GrammarRegistry
//!             (exact match)  (fuzzy)                (formal/terse/
//!                                                    narrative/custom)
//! ```
//!
//! ## Built-in archetypes
//!
//! - **Formal**: precise, structured, academic-style output
//! - **Terse**: minimal prose, dense symbol-heavy notation
//! - **Narrative**: flowing, story-like prose for interactive sessions
//!
//! ## Usage
//!
//! ```rust,no_run
//! use akh_medu::grammar::{GrammarRegistry, AbsTree};
//!
//! let registry = GrammarRegistry::new();
//! let tree = AbsTree::triple(
//!     AbsTree::entity("Dog"),
//!     AbsTree::relation("is-a"),
//!     AbsTree::entity("Mammal"),
//! );
//! let prose = registry.linearize("formal", &tree).unwrap();
//! ```

pub mod abs;
pub mod bridge;
pub mod cat;
pub mod concrete;
pub mod custom;
pub mod detect;
pub mod entity_resolution;
pub mod equivalences;
pub mod error;
pub mod formal;
pub mod lexer;
pub mod morpho;
pub mod narrative;
pub mod parser;
pub mod preprocess;
pub mod terse;

use std::collections::HashMap;

pub use abs::{AbsTree, ProvenanceTag, VsaRoleSymbols};
pub use cat::Cat;
pub use concrete::{ConcreteGrammar, LinContext, ParseContext};
pub use error::{GrammarError, GrammarResult};
pub use lexer::Language;
pub use parser::ParseResult;

/// Runtime registry of available concrete grammars.
///
/// Ships with three built-in archetypes (formal, terse, narrative).
/// Users can register custom grammars at runtime.
pub struct GrammarRegistry {
    grammars: HashMap<String, Box<dyn ConcreteGrammar>>,
    default: String,
}

impl GrammarRegistry {
    /// Create a new registry with the three built-in archetypes.
    pub fn new() -> Self {
        let mut grammars: HashMap<String, Box<dyn ConcreteGrammar>> = HashMap::new();
        grammars.insert("formal".into(), Box::new(formal::FormalGrammar));
        grammars.insert("terse".into(), Box::new(terse::TerseGrammar));
        grammars.insert("narrative".into(), Box::new(narrative::NarrativeGrammar::new()));

        Self {
            grammars,
            default: "formal".into(),
        }
    }

    /// Register a custom grammar. Overwrites any existing grammar with the same name.
    pub fn register(&mut self, grammar: Box<dyn ConcreteGrammar>) {
        let name = grammar.name().to_string();
        self.grammars.insert(name, grammar);
    }

    /// Set the default grammar name.
    pub fn set_default(&mut self, name: impl Into<String>) -> GrammarResult<()> {
        let name = name.into();
        if !self.grammars.contains_key(&name) {
            return Err(GrammarError::UnknownGrammar { name });
        }
        self.default = name;
        Ok(())
    }

    /// Get a grammar by name.
    pub fn get(&self, name: &str) -> GrammarResult<&dyn ConcreteGrammar> {
        self.grammars
            .get(name)
            .map(|g| g.as_ref())
            .ok_or_else(|| GrammarError::UnknownGrammar {
                name: name.to_string(),
            })
    }

    /// Get the default grammar.
    pub fn default_grammar(&self) -> &dyn ConcreteGrammar {
        self.grammars
            .get(&self.default)
            .expect("default grammar must exist")
            .as_ref()
    }

    /// The name of the default grammar.
    pub fn default_name(&self) -> &str {
        &self.default
    }

    /// List all registered grammar names.
    pub fn list(&self) -> Vec<&str> {
        self.grammars.keys().map(|s| s.as_str()).collect()
    }

    /// Linearize an abstract syntax tree using the named grammar.
    pub fn linearize(&self, grammar_name: &str, tree: &AbsTree) -> GrammarResult<String> {
        let grammar = self.get(grammar_name)?;
        let ctx = LinContext::default();
        grammar.linearize(tree, &ctx)
    }

    /// Linearize using the default grammar.
    pub fn linearize_default(&self, tree: &AbsTree) -> GrammarResult<String> {
        self.linearize(&self.default, tree)
    }

    /// Parse prose input using the named grammar.
    pub fn parse(
        &self,
        grammar_name: &str,
        input: &str,
        expected_cat: Option<Cat>,
    ) -> GrammarResult<AbsTree> {
        let grammar = self.get(grammar_name)?;
        let ctx = ParseContext::default();
        grammar.parse(input, expected_cat, &ctx)
    }
}

impl Default for GrammarRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_three_builtins() {
        let reg = GrammarRegistry::new();
        let mut names = reg.list();
        names.sort();
        assert!(names.contains(&"formal"));
        assert!(names.contains(&"terse"));
        assert!(names.contains(&"narrative"));
    }

    #[test]
    fn default_is_formal() {
        let reg = GrammarRegistry::new();
        assert_eq!(reg.default_name(), "formal");
    }

    #[test]
    fn unknown_grammar_errors() {
        let reg = GrammarRegistry::new();
        assert!(reg.get("nonexistent").is_err());
    }

    #[test]
    fn set_default_validates() {
        let mut reg = GrammarRegistry::new();
        assert!(reg.set_default("terse").is_ok());
        assert!(reg.set_default("nonexistent").is_err());
    }

    #[test]
    fn linearize_triple_through_all_archetypes() {
        let reg = GrammarRegistry::new();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        for name in &["formal", "terse", "narrative"] {
            let result = reg.linearize(name, &tree);
            assert!(result.is_ok(), "grammar {name} failed: {result:?}");
            let prose = result.unwrap();
            assert!(!prose.is_empty(), "grammar {name} produced empty output");
        }
    }
}
