//! Symbolic reasoning via e-graphs (egg).
//!
//! Defines AkhLang for the e-graph language and provides rewrite rules
//! for symbolic reasoning over the knowledge graph. Full implementation in Phase 2.

pub mod anti_unify;
pub mod second_order;

use egg::define_language;

define_language! {
    /// The language for e-graph-based symbolic reasoning.
    ///
    /// Expressions in AkhLang represent knowledge graph patterns:
    /// - `Entity(id)` — a concrete entity reference
    /// - `Triple(s, p, o)` — a knowledge triple
    /// - `Bind(a, b)` — VSA bind operation
    /// - `Bundle(a, b)` — VSA bundle operation
    /// - `Similar(a, b)` — similarity assertion
    /// - `And/Or/Not` — logical connectives
    pub enum AkhLang {
        // Numeric literal (symbol ID as integer)
        Num(i64),

        // Knowledge operations
        "triple" = Triple([egg::Id; 3]),
        "bind" = Bind([egg::Id; 2]),
        "bundle" = Bundle([egg::Id; 2]),
        "similar" = Similar([egg::Id; 2]),
        "permute" = Permute([egg::Id; 2]),

        // Logical connectives
        "and" = And([egg::Id; 2]),
        "or" = Or([egg::Id; 2]),
        "not" = Not([egg::Id; 1]),

        // Named symbol references
        Symbol(egg::Symbol),
    }
}

/// Create the built-in rewrite rules for AkhLang.
pub fn builtin_rules() -> Vec<egg::Rewrite<AkhLang, ()>> {
    vec![
        // Bind is commutative: bind(a, b) = bind(b, a) for XOR-based binding
        egg::rewrite!("bind-commute"; "(bind ?a ?b)" => "(bind ?b ?a)"),
        // Bind is self-inverse: bind(a, bind(a, b)) = b
        egg::rewrite!("bind-self-inverse"; "(bind ?a (bind ?a ?b))" => "?b"),
        // Bundle is commutative
        egg::rewrite!("bundle-commute"; "(bundle ?a ?b)" => "(bundle ?b ?a)"),
        // Similarity is commutative
        egg::rewrite!("similar-commute"; "(similar ?a ?b)" => "(similar ?b ?a)"),
        // And is commutative
        egg::rewrite!("and-commute"; "(and ?a ?b)" => "(and ?b ?a)"),
        // Or is commutative
        egg::rewrite!("or-commute"; "(or ?a ?b)" => "(or ?b ?a)"),
        // Double negation elimination
        egg::rewrite!("double-neg"; "(not (not ?a))" => "?a"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{AstSize, Extractor, Runner};

    #[test]
    fn double_negation_simplifies() {
        let expr: egg::RecExpr<AkhLang> = "(not (not x))".parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(&builtin_rules());
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (cost, best) = extractor.find_best(runner.roots[0]);
        assert_eq!(best.to_string(), "x");
        assert_eq!(cost, 1);
    }

    #[test]
    fn bind_self_inverse_simplifies() {
        let expr: egg::RecExpr<AkhLang> = "(bind a (bind a b))".parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(&builtin_rules());
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (cost, best) = extractor.find_best(runner.roots[0]);
        assert_eq!(best.to_string(), "b");
        assert_eq!(cost, 1);
    }

    #[test]
    fn builtin_rules_load() {
        let rules = builtin_rules();
        assert!(!rules.is_empty());
    }
}
