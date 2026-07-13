//! The deductive world: the ground truth for the Phase-51 P4 rule-based arm.
//!
//! A **stratified single-rule derivation chain** — the canonical
//! penguin→bird→flies nonmonotonic taxonomy, made synthetic and deterministic.
//! Predicates are *unary* and *sort-typed* (Q2); each derived predicate is
//! defined by exactly **one** rule over strictly lower-level predicates, so
//! proving a level-`d` predicate is a `d`-deep modus-ponens chain — the
//! rule-arm analogue of the graph-walking corpus's depth, which lets the
//! capacity ladder and depth axes carry straight over.
//!
//! # Why single-rule, stratified, single-variable
//!
//! - **Single rule per predicate** keeps a refutation (`no`) to a *single*
//!   witness (one failing premise), so "no" traces stay one line long instead
//!   of needing a universal "every rule fails" argument.
//! - **Stratified** (rule head strictly above its body, negated guards
//!   strictly below the head) guarantees a unique, deterministic well-founded
//!   model computable in level order — no fixpoint ambiguity.
//! - **Single variable** (`x`) makes the `bind` component a single binding, so
//!   the `select/bind/fire/chain` decomposition (Q5) reads cleanly off the
//!   trace. Binary structural relations (`is_a`/`part_of`/…) and multi-variable
//!   Horn rules are the reserved-dark set, lit by a later binary-rule corpus.
//!
//! Gold answers are `yes` (chain completes) or `no` (a premise is
//! closed-world false); the open-world `gap` value stays exercised by the
//! existing graph-walking corpus, whose functional-relation totality gives the
//! redaction-vs-false signal this world's non-total base predicates do not.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::reason::register::Sort;

/// A typed unary predicate, identified by its `(sort, level, index)`
/// coordinates. Level 0 predicates are **base** (assertable as ground facts);
/// higher levels are **derived** (each defined by one [`Rule`]). Predicates are
/// open-lexicon (they go through NL BPE, not the reserved-token region), so the
/// surface form is chosen only to be deterministic, compact, and collision-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Predicate {
    pub sort: Sort,
    pub level: u32,
    pub index: u32,
}

impl Predicate {
    /// The 2-letter stable code for a sort, used to build collision-free
    /// predicate/rule surfaces.
    fn sort_code(sort: Sort) -> &'static str {
        match sort {
            Sort::Thing => "th",
            Sort::Animal => "an",
            Sort::Place => "pl",
            Sort::Tool => "to",
            Sort::Agent => "ag",
        }
    }

    fn sort_from_code(code: &str) -> Option<Sort> {
        Sort::ALL
            .iter()
            .copied()
            .find(|&s| Self::sort_code(s) == code)
    }

    /// The predicate's surface word (`pred_an_2_1`). Fixed-width fields make it
    /// uniquely decodable back via [`Predicate::parse`].
    pub fn surface(&self) -> String {
        format!(
            "pred_{}_{}_{}",
            Self::sort_code(self.sort),
            self.level,
            self.index
        )
    }

    /// Parse a predicate surface (exact inverse of [`Predicate::surface`]).
    /// Total: `None` for anything not matching `pred_<code>_<level>_<index>`.
    pub fn parse(s: &str) -> Option<Self> {
        let rest = s.strip_prefix("pred_")?;
        let mut it = rest.splitn(3, '_');
        let code = it.next()?;
        let level: u32 = it.next()?.parse().ok()?;
        let index: u32 = it.next()?.parse().ok()?;
        let sort = Self::sort_from_code(code)?;
        Some(Predicate { sort, level, index })
    }

    pub fn is_base(&self) -> bool {
        self.level == 0
    }
}

/// A universally-quantified rule `spine(x) [and not guard(x)] implies head(x)`
/// — one per derived predicate. `spine` is a predicate one level below `head`
/// (the derivation spine); `guard`, when present, is a level-0 base predicate
/// whose *presence* blocks the rule (the nonmonotonic `not` premise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    pub head: Predicate,
    pub spine: Predicate,
    pub guard: Option<Predicate>,
}

impl Rule {
    /// The rule's stable id surface (`r_an_2_1`), mirroring its head
    /// predicate's coordinates (the 1:1 head↔rule correspondence). This is what
    /// a `fire <id>` line names, and it is validated by the replayer against
    /// the rule actually applicable at that point (the `select` component).
    pub fn id(&self) -> String {
        format!(
            "r_{}_{}_{}",
            Predicate::sort_code(self.head.sort),
            self.head.level,
            self.head.index
        )
    }
}

/// Config for [`RuleWorld::generate`].
#[derive(Debug, Clone, PartialEq)]
pub struct RuleWorldConfig {
    pub seed: u64,
    /// Predicates per `(sort, level)` cell. Bigger = more distractor room and a
    /// wider rule-selection space (harder `select`).
    pub preds_per_cell: u32,
    /// Highest predicate level, i.e. the deepest provable chain. Depth-`d`
    /// goals require `d <= max_level`.
    pub max_level: u32,
    /// Probability a derived predicate's rule carries a nonmonotonic guard.
    pub guard_prob: f64,
}

impl Default for RuleWorldConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            preds_per_cell: 6,
            max_level: 8,
            guard_prob: 0.5,
        }
    }
}

fn cell_seed(seed: u64, sort: Sort, level: u32, index: u32) -> u64 {
    seed ^ (sort.reserved_id() as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((level as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((index as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
        .wrapping_add(0xD1B5_4A32_D192_ED03)
}

/// The generated world: a fixed rule per derived predicate, plus the fixed
/// per-cell predicate counts. Deterministic in `config.seed`.
pub struct RuleWorld {
    config: RuleWorldConfig,
    /// head coordinates -> the one rule defining that predicate.
    rules: HashMap<(Sort, u32, u32), Rule>,
}

impl RuleWorld {
    /// Build the world: for every derived predicate `(sort, level>=1, index)`,
    /// draw its spine (a predicate at `level-1`, same sort) and, with
    /// `guard_prob`, a level-0 guard — all from an independent per-cell seeded
    /// stream, so re-generating at the same seed reproduces byte-identical
    /// rules.
    pub fn generate(config: RuleWorldConfig) -> Self {
        let mut rules = HashMap::new();
        let ppc = config.preds_per_cell.max(1);
        for &sort in Sort::ALL.iter() {
            for level in 1..=config.max_level {
                for index in 0..ppc {
                    let mut rng = StdRng::seed_from_u64(cell_seed(config.seed, sort, level, index));
                    let spine = Predicate {
                        sort,
                        level: level - 1,
                        index: rng.gen_range(0..ppc),
                    };
                    let guard = if rng.gen_bool(config.guard_prob.clamp(0.0, 1.0)) {
                        Some(Predicate {
                            sort,
                            level: 0,
                            index: rng.gen_range(0..ppc),
                        })
                    } else {
                        None
                    };
                    let head = Predicate { sort, level, index };
                    rules.insert((sort, level, index), Rule { head, spine, guard });
                }
            }
        }
        Self { config, rules }
    }

    pub fn config(&self) -> &RuleWorldConfig {
        &self.config
    }

    pub fn preds_per_cell(&self) -> u32 {
        self.config.preds_per_cell.max(1)
    }

    /// The rule defining `pred`, or `None` if `pred` is a base (level-0)
    /// predicate (which has no rule — it can only be a ground fact).
    pub fn rule_for(&self, pred: Predicate) -> Option<Rule> {
        if pred.is_base() {
            return None;
        }
        self.rules
            .get(&(pred.sort, pred.level, pred.index))
            .copied()
    }

    /// The spine chain from `goal` down to its level-0 base, **head-first**:
    /// `[goal, spine(goal), spine(spine(goal)), …, base]`. Length is
    /// `goal.level + 1`. Every non-final element has a rule ([`rule_for`]);
    /// the final element is base.
    pub fn spine_chain(&self, goal: Predicate) -> Vec<Predicate> {
        let mut chain = Vec::with_capacity(goal.level as usize + 1);
        let mut cur = goal;
        chain.push(cur);
        while let Some(rule) = self.rule_for(cur) {
            cur = rule.spine;
            chain.push(cur);
        }
        chain
    }

    /// A predicate at `(sort, level, index)`, if `index` is in range and
    /// `level <= max_level`.
    pub fn predicate(&self, sort: Sort, level: u32, index: u32) -> Option<Predicate> {
        if level > self.config.max_level || index >= self.preds_per_cell() {
            return None;
        }
        Some(Predicate { sort, level, index })
    }

    /// Pick a uniformly-random predicate at a given `(sort, level)` cell.
    pub fn random_predicate(&self, rng: &mut StdRng, sort: Sort, level: u32) -> Predicate {
        Predicate {
            sort,
            level,
            index: rng.gen_range(0..self.preds_per_cell()),
        }
    }

    /// All level-0 base predicates of a sort, in index order (used to draw
    /// propositional-arm atoms and rule-arm distractor facts).
    pub fn base_predicates(&self, sort: Sort) -> Vec<Predicate> {
        (0..self.preds_per_cell())
            .map(|index| Predicate {
                sort,
                level: 0,
                index,
            })
            .collect()
    }
}

/// A uniformly-random sort (shared helper for the corpus builders).
pub fn random_sort(rng: &mut StdRng) -> Sort {
    *Sort::ALL.choose(rng).expect("Sort::ALL is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> RuleWorld {
        RuleWorld::generate(RuleWorldConfig {
            seed: 7,
            preds_per_cell: 5,
            max_level: 6,
            guard_prob: 0.5,
        })
    }

    #[test]
    fn predicate_surface_round_trips_and_is_distinct() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for sort in Sort::ALL {
            for level in 0..4 {
                for index in 0..5 {
                    let p = Predicate { sort, level, index };
                    assert!(seen.insert(p.surface()), "duplicate surface for {p:?}");
                    assert_eq!(Predicate::parse(&p.surface()), Some(p));
                }
            }
        }
        assert_eq!(Predicate::parse("pred_zz_1_0"), None);
        assert_eq!(Predicate::parse("not_a_pred"), None);
    }

    #[test]
    fn generation_is_deterministic_for_a_fixed_seed() {
        let a = world();
        let b = world();
        for sort in Sort::ALL {
            for level in 1..=6 {
                for index in 0..5 {
                    let p = Predicate { sort, level, index };
                    assert_eq!(a.rule_for(p), b.rule_for(p));
                }
            }
        }
    }

    #[test]
    fn every_derived_predicate_has_exactly_one_rule_pointing_strictly_down() {
        let w = world();
        for sort in Sort::ALL {
            for level in 1..=6 {
                for index in 0..5 {
                    let p = Predicate { sort, level, index };
                    let rule = w.rule_for(p).expect("derived predicate must have a rule");
                    assert_eq!(rule.head, p);
                    // spine strictly one level below, same sort (stratification).
                    assert_eq!(rule.spine.level, level - 1);
                    assert_eq!(rule.spine.sort, sort);
                    // guard, if any, is a strictly-lower level-0 base predicate.
                    if let Some(g) = rule.guard {
                        assert_eq!(g.level, 0);
                        assert!(g.level < level);
                        assert_eq!(g.sort, sort);
                    }
                }
            }
        }
    }

    #[test]
    fn base_predicates_have_no_rule() {
        let w = world();
        for sort in Sort::ALL {
            for p in w.base_predicates(sort) {
                assert!(p.is_base());
                assert_eq!(w.rule_for(p), None);
            }
        }
    }

    #[test]
    fn spine_chain_reaches_base_at_the_right_length() {
        let w = world();
        let goal = Predicate {
            sort: Sort::Animal,
            level: 5,
            index: 2,
        };
        let chain = w.spine_chain(goal);
        assert_eq!(chain.len(), 6); // levels 5,4,3,2,1,0
        assert_eq!(chain.first().copied(), Some(goal));
        assert!(chain.last().unwrap().is_base());
        // Strictly descending levels, each linked by a real rule.
        for pair in chain.windows(2) {
            let rule = w.rule_for(pair[0]).unwrap();
            assert_eq!(rule.spine, pair[1]);
            assert_eq!(pair[1].level + 1, pair[0].level);
        }
    }
}
