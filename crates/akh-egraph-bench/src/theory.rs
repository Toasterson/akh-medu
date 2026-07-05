//! Deterministic synthetic theory generator.
//!
//! Produces an `AkhLang` expression encoding `n_facts` facts as a balanced
//! `and`-conjunction, with a fact mix chosen so every production rule pack
//! has real fodder:
//!
//! - **40 % causal edges** laid out as a `branching`-ary tree emitted in DFS
//!   preorder, so `(and (causes a b) (causes b c))` subterms actually occur
//!   adjacently and `cause-trans` fires. `branching = 1` is a pure chain
//!   (maximum transitive-closure depth); higher branching is shallower.
//! - **15 % event-calculus pairs** — `initiates(e,f)` next to `happens(e,t)`
//!   so `ec-persist` fires.
//! - **15 % epistemic facts** — `knows(agent, prop)`. Under the `large`
//!   pack, `positive-introspection` grows these without bound; this is the
//!   deliberate worst-case driver.
//! - **15 % generic `before` triples** in chain-adjacent order for
//!   `before-trans`.
//! - **15 % VSA algebra terms** — `(bind a (bind a b))`, `(not (not x))`,
//!   `(bundle x y)` for the builtin pack.
//!
//! Everything is derived from the seed; two runs with the same
//! `(n_facts, branching, seed)` produce byte-identical expressions.
//!
//! Note an architectural fact this generator makes measurable: egg's
//! conjunction rules only match *structural* `(and x y)` subterms — the
//! e-graph does no KG-wide join. The and-tree shape therefore bounds how
//! many conjunctive rules can fire; the adjacency layout here is the
//! *favourable* case.

use akh_medu::reason::AkhLang;
use egg::{Id, RecExpr};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Entity-id namespaces so fact families never collide.
const CAUSAL_BASE: i64 = 1_000_000;
const EC_BASE: i64 = 2_000_000;
const EPISTEMIC_BASE: i64 = 3_000_000;
const TRIPLE_BASE: i64 = 4_000_000;
const VSA_BASE: i64 = 5_000_000;

/// Build the synthetic theory expression. Returns the expression and the
/// number of fact terms actually emitted (== `n_facts` for n >= 8).
pub fn build_theory(n_facts: usize, branching: usize, seed: u64) -> RecExpr<AkhLang> {
    assert!(n_facts >= 1, "need at least one fact");
    let branching = branching.max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut expr = RecExpr::default();
    let mut facts: Vec<Id> = Vec::with_capacity(n_facts);

    let n_causal = (n_facts * 40) / 100;
    let n_ec = (n_facts * 15) / 100;
    let n_epistemic = (n_facts * 15) / 100;
    let n_triple = (n_facts * 15) / 100;
    // Remainder goes to the VSA family so the total is exactly n_facts.
    let n_vsa = n_facts - n_causal - n_ec - n_epistemic - n_triple;

    causal_facts(&mut expr, &mut facts, n_causal, branching);
    ec_facts(&mut expr, &mut facts, n_ec);
    epistemic_facts(&mut expr, &mut facts, n_epistemic);
    before_triples(&mut expr, &mut facts, n_triple);
    vsa_facts(&mut expr, &mut facts, n_vsa, &mut rng);

    debug_assert_eq!(facts.len(), n_facts);
    and_tree(&mut expr, &facts);
    expr
}

/// `causes` edges of a `b`-ary tree (heap numbering) in DFS preorder.
fn causal_facts(expr: &mut RecExpr<AkhLang>, facts: &mut Vec<Id>, n: usize, b: usize) {
    // Child ids come from a sequential counter, NOT heap numbering
    // (`parent*b + k + 1`): heap ids grow as b^depth, and a DFS descends
    // O(n/b) levels, so heap numbering overflows u64 (debug panic; silent
    // wrap + entity collisions in release). Sequential ids keep every
    // entity < n while preserving the tree shape and the DFS adjacency
    // that lets cause-trans fire on (and (causes a b) (causes b c)).
    let mut stack: Vec<u64> = vec![0];
    let mut next_id: u64 = 1;
    let mut emitted = 0usize;
    while emitted < n {
        let parent = match stack.pop() {
            Some(p) => p,
            None => {
                // Tree exhausted — start a new root (kept for totality).
                let root = next_id;
                next_id += 1;
                root
            }
        };
        for _ in 0..b {
            if emitted >= n {
                break;
            }
            let child = next_id;
            next_id += 1;
            let p = expr.add(AkhLang::Num(CAUSAL_BASE + parent as i64));
            let c = expr.add(AkhLang::Num(CAUSAL_BASE + child as i64));
            facts.push(expr.add(AkhLang::Causes([p, c])));
            stack.push(child);
            emitted += 1;
        }
    }
}

/// Adjacent `initiates(e,f)` / `happens(e,t)` pairs for `ec-persist`.
fn ec_facts(expr: &mut RecExpr<AkhLang>, facts: &mut Vec<Id>, n: usize) {
    let mut added = 0usize;
    let mut i = 0i64;
    while added < n {
        let e = expr.add(AkhLang::Num(EC_BASE + 3 * i));
        let f = expr.add(AkhLang::Num(EC_BASE + 3 * i + 1));
        let t = expr.add(AkhLang::Num(EC_BASE + 3 * i + 2));
        facts.push(expr.add(AkhLang::Initiates([e, f])));
        added += 1;
        if added < n {
            facts.push(expr.add(AkhLang::Happens([e, t])));
            added += 1;
        }
        i += 1;
    }
}

/// `knows(agent, prop)` facts — 16 agents, distinct propositions.
fn epistemic_facts(expr: &mut RecExpr<AkhLang>, facts: &mut Vec<Id>, n: usize) {
    for i in 0..n {
        let a = expr.add(AkhLang::Num(EPISTEMIC_BASE + (i % 16) as i64));
        let p = expr.add(AkhLang::Num(EPISTEMIC_BASE + 100_000 + i as i64));
        facts.push(expr.add(AkhLang::Knows([a, p])));
    }
}

/// Chain-ordered `triple(s, before, o)` facts for `before-trans`.
fn before_triples(expr: &mut RecExpr<AkhLang>, facts: &mut Vec<Id>, n: usize) {
    let before = expr.add(AkhLang::Symbol("before".into()));
    for i in 0..n {
        let s = expr.add(AkhLang::Num(TRIPLE_BASE + i as i64));
        let o = expr.add(AkhLang::Num(TRIPLE_BASE + i as i64 + 1));
        facts.push(expr.add(AkhLang::Triple([s, before, o])));
    }
}

/// VSA algebra terms exercising the builtin pack.
fn vsa_facts(expr: &mut RecExpr<AkhLang>, facts: &mut Vec<Id>, n: usize, rng: &mut StdRng) {
    for i in 0..n {
        let a = expr.add(AkhLang::Num(VSA_BASE + rng.gen_range(0..1_000)));
        let b = expr.add(AkhLang::Num(VSA_BASE + rng.gen_range(0..1_000)));
        let fact = match i % 3 {
            0 => {
                // (bind a (bind a b)) — simplifies to b via bind-self-inverse.
                let inner = expr.add(AkhLang::Bind([a, b]));
                expr.add(AkhLang::Bind([a, inner]))
            }
            1 => {
                // (not (not a)) — simplifies via double-neg.
                let inner = expr.add(AkhLang::Not([a]));
                expr.add(AkhLang::Not([inner]))
            }
            _ => expr.add(AkhLang::Bundle([a, b])),
        };
        facts.push(fact);
    }
}

/// Balanced binary `and`-tree over the fact ids (preserves order, so the
/// adjacency the generator arranged survives into `(and x y)` subterms).
fn and_tree(expr: &mut RecExpr<AkhLang>, ids: &[Id]) -> Id {
    match ids.len() {
        0 => unreachable!("and_tree requires at least one fact"),
        1 => ids[0],
        n => {
            let mid = n / 2;
            let (left_ids, right_ids) = ids.split_at(mid);
            let left_ids = left_ids.to_vec();
            let right_ids = right_ids.to_vec();
            let l = and_tree(expr, &left_ids);
            let r = and_tree(expr, &right_ids);
            expr.add(AkhLang::And([l, r]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let a = build_theory(200, 3, 42);
        let b = build_theory(200, 3, 42);
        assert_eq!(a.to_string(), b.to_string());
    }

    #[test]
    fn fact_count_is_exact() {
        for n in [8usize, 100, 1_000] {
            let expr = build_theory(n, 3, 7);
            // n facts + (n-1) and-nodes is a lower bound on node count.
            assert!(expr.len() >= 2 * n - 1, "n={n} len={}", expr.len());
        }
    }
}
