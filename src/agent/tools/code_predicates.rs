//! Well-known code relation predicates for code ingestion.
//!
//! Follows the `AgentPredicates` pattern: a set of SymbolIds for code-specific
//! relations, resolved or created on initialization.

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::super::error::AgentResult;

/// Well-known relation SymbolIds for code structure.
///
/// All labels are prefixed `code:` to avoid collision with agent or ontological
/// predicates.
#[derive(Debug, Clone)]
pub struct CodePredicates {
    // Module structure
    /// "code:contains-mod" — a module contains a submodule
    pub contains_mod: SymbolId,
    /// "code:defined-in" — an item is defined in a file/module
    pub defined_in: SymbolId,

    // Definitions
    /// "code:defines-fn" — a module/impl defines a function
    pub defines_fn: SymbolId,
    /// "code:defines-struct" — a module defines a struct
    pub defines_struct: SymbolId,
    /// "code:defines-enum" — a module defines an enum
    pub defines_enum: SymbolId,
    /// "code:defines-trait" — a module defines a trait
    pub defines_trait: SymbolId,

    // Relationships
    /// "code:implements-trait" — a type implements a trait
    pub implements_trait: SymbolId,
    /// "code:has-method" — a type has a method
    pub has_method: SymbolId,
    /// "code:has-field" — a struct has a field
    pub has_field: SymbolId,
    /// "code:has-variant" — an enum has a variant
    pub has_variant: SymbolId,
    /// "code:has-param" — a function has a parameter
    pub has_param: SymbolId,
    /// "code:returns-type" — a function returns a type
    pub returns_type: SymbolId,
    /// "code:depends-on" — a module/item depends on another
    pub depends_on: SymbolId,
    /// "code:derives-trait" — a type derives a trait
    pub derives_trait: SymbolId,

    // Metadata
    /// "code:has-visibility" — an item has a visibility (pub, pub(crate), etc.)
    pub has_visibility: SymbolId,
    /// "code:has-doc" — an item has a doc comment
    pub has_doc: SymbolId,
    /// "code:circular-dep" — circular dependency (derived by rules)
    pub circular_dep: SymbolId,
}

impl CodePredicates {
    /// Resolve or create all well-known code predicates in the engine.
    pub fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            contains_mod: engine.resolve_or_create_relation("code:contains-mod")?,
            defined_in: engine.resolve_or_create_relation("code:defined-in")?,
            defines_fn: engine.resolve_or_create_relation("code:defines-fn")?,
            defines_struct: engine.resolve_or_create_relation("code:defines-struct")?,
            defines_enum: engine.resolve_or_create_relation("code:defines-enum")?,
            defines_trait: engine.resolve_or_create_relation("code:defines-trait")?,
            implements_trait: engine.resolve_or_create_relation("code:implements-trait")?,
            has_method: engine.resolve_or_create_relation("code:has-method")?,
            has_field: engine.resolve_or_create_relation("code:has-field")?,
            has_variant: engine.resolve_or_create_relation("code:has-variant")?,
            has_param: engine.resolve_or_create_relation("code:has-param")?,
            returns_type: engine.resolve_or_create_relation("code:returns-type")?,
            depends_on: engine.resolve_or_create_relation("code:depends-on")?,
            derives_trait: engine.resolve_or_create_relation("code:derives-trait")?,
            has_visibility: engine.resolve_or_create_relation("code:has-visibility")?,
            has_doc: engine.resolve_or_create_relation("code:has-doc")?,
            circular_dep: engine.resolve_or_create_relation("code:circular-dep")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    #[test]
    fn code_predicates_init() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let preds = CodePredicates::init(&engine).unwrap();

        // All predicates should be distinct.
        let ids = [
            preds.contains_mod,
            preds.defined_in,
            preds.defines_fn,
            preds.defines_struct,
            preds.defines_enum,
            preds.defines_trait,
            preds.implements_trait,
            preds.has_method,
            preds.has_field,
            preds.has_variant,
            preds.has_param,
            preds.returns_type,
            preds.depends_on,
            preds.derives_trait,
            preds.has_visibility,
            preds.has_doc,
            preds.circular_dep,
        ];

        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "all code predicates must be unique"
        );
    }

    #[test]
    fn code_predicates_idempotent() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let first = CodePredicates::init(&engine).unwrap();
        let second = CodePredicates::init(&engine).unwrap();

        assert_eq!(first.defines_fn, second.defines_fn);
        assert_eq!(first.implements_trait, second.implements_trait);
    }
}
