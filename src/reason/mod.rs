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

        // Causal & Event Calculus (Phase 15a-b)
        "causes" = Causes([egg::Id; 2]),
        "enables" = Enables([egg::Id; 2]),
        "prevents" = Prevents([egg::Id; 2]),
        "initiates" = Initiates([egg::Id; 2]),
        "terminates" = Terminates([egg::Id; 2]),
        "happens" = Happens([egg::Id; 2]),
        "holds-at" = HoldsAt([egg::Id; 2]),
        "terminated" = Terminated([egg::Id; 2]),

        // Epistemic Logic (Phase 19)
        "knows" = Knows([egg::Id; 2]),
        "believes" = Believes([egg::Id; 2]),
        "ignorant-about" = IgnorantAbout([egg::Id; 2]),
        "implies" = Implies([egg::Id; 2]),

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

/// PIM-specific rewrite rules (Phase 13e).
///
/// - `pim-unblock`: when a blocker is done, the blocked task becomes next.
/// - `pim-deadline-chain`: earliest-start constraint from blocker's deadline.
pub fn pim_rules() -> Vec<egg::Rewrite<AkhLang, ()>> {
    vec![
        // If triple(blocked, blocked-by, blocker) and triple(blocker, gtd-state, done)
        // then triple(blocked, gtd-state, next).
        egg::rewrite!("pim-unblock";
            "(and (triple ?blocked ?blocked_by ?blocker) (triple ?blocker ?gtd_state ?done))"
            => "(triple ?blocked ?gtd_state ?done)"
        ),
        // Deadline chain: triple(A, deadline, D) and triple(B, blocked-by, A) implies
        // triple(B, deadline, D) as earliest-start.
        egg::rewrite!("pim-deadline-chain";
            "(and (triple ?a ?deadline ?d) (triple ?b ?blocked_by ?a))"
            => "(and (triple ?a ?deadline ?d) (and (triple ?b ?blocked_by ?a) (triple ?b ?deadline ?d)))"
        ),
    ]
}

/// Calendar-specific rewrite rules (Phase 13f).
///
/// - `before-trans`: the "before" relation is transitive.
/// - `cal-conflict`: overlapping events that require the same resource conflict.
pub fn calendar_rules() -> Vec<egg::Rewrite<AkhLang, ()>> {
    vec![
        // Transitivity of "before": if A before B and B before C then A before C.
        egg::rewrite!("before-trans";
            "(and (triple ?a ?before ?b) (triple ?b ?before ?c))"
            => "(and (triple ?a ?before ?b) (and (triple ?b ?before ?c) (triple ?a ?before ?c)))"
        ),
        // Conflict: if two events overlap and require the same resource, they conflict.
        egg::rewrite!("cal-conflict";
            "(and (triple ?e1 ?overlaps ?e2) (and (triple ?e1 ?requires ?r) (triple ?e2 ?requires ?r)))"
            => "(and (triple ?e1 ?overlaps ?e2) (and (triple ?e1 ?requires ?r) (and (triple ?e2 ?requires ?r) (triple ?e1 ?conflicts ?e2))))"
        ),
    ]
}

/// Causal and Event Calculus rewrite rules (Phase 15a-b).
///
/// - `cause-trans`: causal transitivity — if A causes B and B causes C, then A causes C.
/// - `enable-cause`: enabling chains — if A enables B and B causes C, then A enables C.
/// - `ec-persist`: EC persistence — if event initiates fluent, fluent holds at that time.
/// - `ec-terminate`: EC termination — if event terminates fluent, fluent is terminated.
pub fn causal_rules() -> Vec<egg::Rewrite<AkhLang, ()>> {
    vec![
        // Causal transitivity: causes(a, b) ∧ causes(b, c) → causes(a, c)
        egg::rewrite!("cause-trans";
            "(and (causes ?a ?b) (causes ?b ?c))"
            => "(and (causes ?a ?b) (and (causes ?b ?c) (causes ?a ?c)))"
        ),
        // Enable + cause = enable: enables(a, b) ∧ causes(b, c) → enables(a, c)
        egg::rewrite!("enable-cause";
            "(and (enables ?a ?b) (causes ?b ?c))"
            => "(and (enables ?a ?b) (and (causes ?b ?c) (enables ?a ?c)))"
        ),
        // EC persistence: initiates(e, f) ∧ happens(e, t) → holds-at(f, t)
        egg::rewrite!("ec-persist";
            "(and (initiates ?e ?f) (happens ?e ?t))"
            => "(and (initiates ?e ?f) (and (happens ?e ?t) (holds-at ?f ?t)))"
        ),
        // EC termination: terminates(e, f) ∧ happens(e, t) → terminated(f, t)
        egg::rewrite!("ec-terminate";
            "(and (terminates ?e ?f) (happens ?e ?t))"
            => "(and (terminates ?e ?f) (and (happens ?e ?t) (terminated ?f ?t)))"
        ),
    ]
}

/// Epistemic logic rewrite rules (Phase 19).
///
/// - `knows-implies-believes`: knowing implies believing
/// - `positive-introspection`: if you know P, you know that you know P
/// - `k-axiom`: knowledge distributes over implication
pub fn epistemic_rules() -> Vec<egg::Rewrite<AkhLang, ()>> {
    vec![
        // Knowing implies believing.
        egg::rewrite!("knows-implies-believes";
            "(knows ?a ?p)"
            => "(and (knows ?a ?p) (believes ?a ?p))"
        ),
        // Positive introspection: K(a, p) → K(a, K(a, p)).
        egg::rewrite!("positive-introspection";
            "(knows ?a ?p)"
            => "(and (knows ?a ?p) (knows ?a (knows ?a ?p)))"
        ),
        // K-axiom: K(a, p→q) ∧ K(a, p) → K(a, q).
        egg::rewrite!("k-axiom";
            "(and (knows ?a (implies ?p ?q)) (knows ?a ?p))"
            => "(and (knows ?a (implies ?p ?q)) (and (knows ?a ?p) (knows ?a ?q)))"
        ),
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

    #[test]
    fn causal_rules_load() {
        let rules = causal_rules();
        assert_eq!(rules.len(), 4);
    }

    #[test]
    fn ec_persist_rule_fires() {
        // If initiates(e, f) and happens(e, t) then holds-at(f, t) should be derivable.
        let expr: egg::RecExpr<AkhLang> =
            "(and (initiates ev fluent) (happens ev time))".parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(&causal_rules());

        // Check that holds-at(fluent, time) is in the e-graph.
        let target: egg::RecExpr<AkhLang> = "(holds-at fluent time)".parse().unwrap();
        let target_id = runner.egraph.lookup_expr(&target);
        assert!(
            target_id.is_some(),
            "ec-persist rule should derive (holds-at fluent time)"
        );
    }

    #[test]
    fn cause_transitivity_rule_fires() {
        let expr: egg::RecExpr<AkhLang> =
            "(and (causes a b) (causes b c))".parse().unwrap();
        let runner = Runner::default().with_expr(&expr).run(&causal_rules());

        let target: egg::RecExpr<AkhLang> = "(causes a c)".parse().unwrap();
        let target_id = runner.egraph.lookup_expr(&target);
        assert!(
            target_id.is_some(),
            "cause-trans rule should derive (causes a c)"
        );
    }
}
