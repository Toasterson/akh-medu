//! Grammar categories mapping to existing akh-medu types.
//!
//! Each [`Cat`] variant corresponds to a semantic role in the abstract syntax.
//! Categories guide both parsing (what the parser expects next) and
//! linearization (which concrete rules to apply).

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolKind;

/// Grammatical category — the semantic type of an abstract syntax node.
///
/// Maps directly to existing akh-medu types:
/// - `Entity` ↔ `SymbolKind::Entity`
/// - `Relation` ↔ `SymbolKind::Relation`
/// - `Statement` ↔ `Triple` / `FactKind::Triple`
/// - etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cat {
    /// A concrete entity (person, place, concept).
    Entity,
    /// A relation between entities.
    Relation,
    /// A complete statement (subject-predicate-object triple).
    Statement,
    /// A similarity assertion between two entities.
    Similarity,
    /// A knowledge gap — something unknown or missing.
    Gap,
    /// An inference or reasoning result.
    Inference,
    /// A code structure fact (function, struct, module, etc.).
    CodeFact,
    /// A quantifier or qualifier modifying another node.
    Modifier,
    /// A conjunction or disjunction of multiple items.
    Conjunction,
    /// A complete document with sections.
    Document,
    /// A section within a document.
    Section,
    /// A confidence annotation.
    Confidence,
    /// A provenance annotation.
    Provenance,
    /// Free-form text that didn't parse into a structured category.
    Freeform,
}

impl Cat {
    /// Whether this category represents a leaf node (no children).
    pub fn is_leaf(self) -> bool {
        matches!(self, Cat::Entity | Cat::Relation | Cat::Freeform)
    }

    /// Whether this category can be a child of a `Statement`.
    pub fn valid_in_statement(self) -> bool {
        matches!(self, Cat::Entity | Cat::Relation | Cat::Freeform)
    }
}

impl From<SymbolKind> for Cat {
    fn from(kind: SymbolKind) -> Self {
        match kind {
            SymbolKind::Entity => Cat::Entity,
            SymbolKind::Relation => Cat::Relation,
            SymbolKind::Composite => Cat::Entity,
            SymbolKind::Glyph { .. } => Cat::Entity,
        }
    }
}

impl std::fmt::Display for Cat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cat::Entity => write!(f, "Entity"),
            Cat::Relation => write!(f, "Relation"),
            Cat::Statement => write!(f, "Statement"),
            Cat::Similarity => write!(f, "Similarity"),
            Cat::Gap => write!(f, "Gap"),
            Cat::Inference => write!(f, "Inference"),
            Cat::CodeFact => write!(f, "CodeFact"),
            Cat::Modifier => write!(f, "Modifier"),
            Cat::Conjunction => write!(f, "Conjunction"),
            Cat::Document => write!(f, "Document"),
            Cat::Section => write!(f, "Section"),
            Cat::Confidence => write!(f, "Confidence"),
            Cat::Provenance => write!(f, "Provenance"),
            Cat::Freeform => write!(f, "Freeform"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_is_leaf() {
        assert!(Cat::Entity.is_leaf());
        assert!(Cat::Relation.is_leaf());
        assert!(!Cat::Statement.is_leaf());
        assert!(!Cat::Document.is_leaf());
    }

    #[test]
    fn symbol_kind_converts_to_cat() {
        assert_eq!(Cat::from(SymbolKind::Entity), Cat::Entity);
        assert_eq!(Cat::from(SymbolKind::Relation), Cat::Relation);
        assert_eq!(Cat::from(SymbolKind::Composite), Cat::Entity);
    }

    #[test]
    fn valid_statement_children() {
        assert!(Cat::Entity.valid_in_statement());
        assert!(Cat::Relation.valid_in_statement());
        assert!(!Cat::Statement.valid_in_statement());
    }
}
