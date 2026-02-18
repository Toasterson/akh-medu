//! Non-Atomic Reified Terms (NARTs): functional term construction.
//!
//! NARTs are complex terms built from functions applied to arguments:
//! `(GovernmentFn France)`, `(FruitFn AppleTree)`. They are reified as
//! first-class entities, can appear as arguments to predicates, and are
//! unified structurally.
//!
//! A NART is identified by its function + arguments structure. Two NARTs
//! with the same function and arguments are the same entity (deduplication).
//!
//! ## Design
//!
//! NARTs use the existing `SymbolKind::Composite` variant. The NART metadata
//! (function, args) is stored in a `NartRegistry` rather than modifying SymbolKind,
//! keeping the symbol system lightweight.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::{SymbolId, SymbolKind};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to NART operations.
#[derive(Debug, Error, Diagnostic)]
pub enum NartError {
    #[error("NART not found: symbol {id}")]
    #[diagnostic(
        code(akh::nart::not_found),
        help("No NART with this symbol ID is registered. Check the NART registry.")
    )]
    NotFound { id: u64 },

    #[error("function symbol {function_id} not found")]
    #[diagnostic(
        code(akh::nart::function_not_found),
        help("The function symbol used to construct the NART must exist. Create it first.")
    )]
    FunctionNotFound { function_id: u64 },

    #[error("argument symbol {arg_id} at position {position} not found")]
    #[diagnostic(
        code(akh::nart::arg_not_found),
        help("All argument symbols must exist before constructing a NART.")
    )]
    ArgNotFound { arg_id: u64, position: usize },

    #[error("NART requires at least one argument")]
    #[diagnostic(
        code(akh::nart::empty_args),
        help("A NART must have at least one argument. Use a plain Entity for zero-argument terms.")
    )]
    EmptyArgs,
}

/// Result type for NART operations.
pub type NartResult<T> = std::result::Result<T, NartError>;

// ---------------------------------------------------------------------------
// NART definition
// ---------------------------------------------------------------------------

/// A Non-Atomic Reified Term: a function applied to arguments.
///
/// The NART is backed by a `SymbolKind::Composite` symbol in the knowledge graph.
/// Its identity is determined by `(function, args)` — structural equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NartDef {
    /// The SymbolId of this NART in the KG (a Composite symbol).
    pub id: SymbolId,
    /// The function symbol (e.g., `GovernmentFn`).
    pub function: SymbolId,
    /// The arguments (e.g., `[France]`).
    pub args: Vec<SymbolId>,
    /// Human-readable label (e.g., "(GovernmentFn France)").
    pub label: String,
}

/// A structural key for NART deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct NartKey {
    function: SymbolId,
    args: Vec<SymbolId>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of Non-Atomic Reified Terms.
///
/// Manages NART creation with deduplication and provides structural lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NartRegistry {
    /// SymbolId → NartDef for all registered NARTs.
    narts: HashMap<SymbolId, NartDef>,
    /// Structural index: (function, args) → SymbolId for deduplication.
    structural_index: HashMap<NartKey, SymbolId>,
}

impl NartRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a NART, or return the existing one if structurally identical.
    ///
    /// The NART is backed by a `Composite` symbol in the engine.
    pub fn create_nart(
        &mut self,
        engine: &Engine,
        function: SymbolId,
        args: Vec<SymbolId>,
    ) -> AkhResult<NartDef> {
        if args.is_empty() {
            return Err(NartError::EmptyArgs.into());
        }

        let key = NartKey {
            function,
            args: args.clone(),
        };

        // Deduplication check
        if let Some(&existing_id) = self.structural_index.get(&key) {
            if let Some(existing) = self.narts.get(&existing_id) {
                return Ok(existing.clone());
            }
        }

        // Build the label: (FunctionName Arg1 Arg2 ...)
        let fn_label = engine.resolve_label(function);
        let arg_labels: Vec<String> = args.iter().map(|a| engine.resolve_label(*a)).collect();
        let label = format!("({} {})", fn_label, arg_labels.join(" "));

        // Create a Composite symbol
        let meta = engine.create_symbol(SymbolKind::Composite, &label)?;

        let nart_def = NartDef {
            id: meta.id,
            function,
            args: args.clone(),
            label,
        };

        self.narts.insert(meta.id, nart_def.clone());
        self.structural_index.insert(key, meta.id);

        Ok(nart_def)
    }

    /// Look up a NART by its SymbolId.
    pub fn get(&self, id: SymbolId) -> Option<&NartDef> {
        self.narts.get(&id)
    }

    /// Find a NART by its structural identity (function + args).
    pub fn find_structural(
        &self,
        function: SymbolId,
        args: &[SymbolId],
    ) -> Option<&NartDef> {
        let key = NartKey {
            function,
            args: args.to_vec(),
        };
        self.structural_index
            .get(&key)
            .and_then(|id| self.narts.get(id))
    }

    /// Check if a SymbolId is a NART.
    pub fn is_nart(&self, id: SymbolId) -> bool {
        self.narts.contains_key(&id)
    }

    /// List all NARTs.
    pub fn all_narts(&self) -> Vec<&NartDef> {
        self.narts.values().collect()
    }

    /// Find all NARTs that use a given function.
    pub fn narts_for_function(&self, function: SymbolId) -> Vec<&NartDef> {
        self.narts
            .values()
            .filter(|n| n.function == function)
            .collect()
    }

    /// Find all NARTs that have a given symbol as an argument.
    pub fn narts_with_arg(&self, arg: SymbolId) -> Vec<&NartDef> {
        self.narts
            .values()
            .filter(|n| n.args.contains(&arg))
            .collect()
    }

    /// Structural unification: match a pattern (function, arg_patterns) against
    /// registered NARTs. `None` in a pattern position matches any argument.
    pub fn unify(
        &self,
        function: SymbolId,
        arg_patterns: &[Option<SymbolId>],
    ) -> Vec<&NartDef> {
        self.narts
            .values()
            .filter(|n| {
                if n.function != function {
                    return false;
                }
                if n.args.len() != arg_patterns.len() {
                    return false;
                }
                n.args
                    .iter()
                    .zip(arg_patterns.iter())
                    .all(|(actual, pattern)| match pattern {
                        Some(expected) => actual == expected,
                        None => true,
                    })
            })
            .collect()
    }

    /// Number of registered NARTs.
    pub fn len(&self) -> usize {
        self.narts.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.narts.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;

    #[test]
    fn create_nart_basic() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();

        let mut registry = NartRegistry::new();
        let nart = registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();

        assert_eq!(nart.function, gov_fn.id);
        assert_eq!(nart.args, vec![france.id]);
        assert!(nart.label.contains("GovernmentFn"));
        assert!(nart.label.contains("France"));
    }

    #[test]
    fn create_nart_deduplication() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();

        let mut registry = NartRegistry::new();
        let n1 = registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();
        let n2 = registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();

        assert_eq!(n1.id, n2.id);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn create_nart_different_args() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();
        let germany = engine
            .create_symbol(SymbolKind::Entity, "Germany")
            .unwrap();

        let mut registry = NartRegistry::new();
        let n1 = registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();
        let n2 = registry
            .create_nart(&engine, gov_fn.id, vec![germany.id])
            .unwrap();

        assert_ne!(n1.id, n2.id);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn create_nart_empty_args_fails() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();

        let mut registry = NartRegistry::new();
        let result = registry.create_nart(&engine, gov_fn.id, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn find_structural() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();

        let mut registry = NartRegistry::new();
        registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();

        let found = registry.find_structural(gov_fn.id, &[france.id]);
        assert!(found.is_some());

        let not_found = registry.find_structural(france.id, &[gov_fn.id]);
        assert!(not_found.is_none());
    }

    #[test]
    fn narts_for_function() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let cap_fn = engine
            .create_symbol(SymbolKind::Relation, "CapitalFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();
        let germany = engine
            .create_symbol(SymbolKind::Entity, "Germany")
            .unwrap();

        let mut registry = NartRegistry::new();
        registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();
        registry
            .create_nart(&engine, gov_fn.id, vec![germany.id])
            .unwrap();
        registry
            .create_nart(&engine, cap_fn.id, vec![france.id])
            .unwrap();

        let gov_narts = registry.narts_for_function(gov_fn.id);
        assert_eq!(gov_narts.len(), 2);

        let cap_narts = registry.narts_for_function(cap_fn.id);
        assert_eq!(cap_narts.len(), 1);
    }

    #[test]
    fn narts_with_arg() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let cap_fn = engine
            .create_symbol(SymbolKind::Relation, "CapitalFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();

        let mut registry = NartRegistry::new();
        registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();
        registry
            .create_nart(&engine, cap_fn.id, vec![france.id])
            .unwrap();

        let france_narts = registry.narts_with_arg(france.id);
        assert_eq!(france_narts.len(), 2);
    }

    #[test]
    fn structural_unification() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();
        let germany = engine
            .create_symbol(SymbolKind::Entity, "Germany")
            .unwrap();

        let mut registry = NartRegistry::new();
        registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();
        registry
            .create_nart(&engine, gov_fn.id, vec![germany.id])
            .unwrap();

        // Match with wildcard arg
        let matches = registry.unify(gov_fn.id, &[None]);
        assert_eq!(matches.len(), 2);

        // Match with specific arg
        let matches = registry.unify(gov_fn.id, &[Some(france.id)]);
        assert_eq!(matches.len(), 1);

        // No match — wrong function
        let matches = registry.unify(france.id, &[None]);
        assert!(matches.is_empty());
    }

    #[test]
    fn is_nart_check() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let gov_fn = engine
            .create_symbol(SymbolKind::Relation, "GovernmentFn")
            .unwrap();
        let france = engine
            .create_symbol(SymbolKind::Entity, "France")
            .unwrap();

        let mut registry = NartRegistry::new();
        let nart = registry
            .create_nart(&engine, gov_fn.id, vec![france.id])
            .unwrap();

        assert!(registry.is_nart(nart.id));
        assert!(!registry.is_nart(france.id));
    }

    #[test]
    fn multi_arg_nart() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let between_fn = engine
            .create_symbol(SymbolKind::Relation, "BetweenFn")
            .unwrap();
        let a = engine.create_symbol(SymbolKind::Entity, "A").unwrap();
        let b = engine.create_symbol(SymbolKind::Entity, "B").unwrap();

        let mut registry = NartRegistry::new();
        let nart = registry
            .create_nart(&engine, between_fn.id, vec![a.id, b.id])
            .unwrap();

        assert_eq!(nart.args.len(), 2);
        assert!(nart.label.contains("BetweenFn"));
        assert!(nart.label.contains("A"));
        assert!(nart.label.contains("B"));
    }
}
