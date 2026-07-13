//! The two deductive corpus arms and their High-Grug-v2 trace rendering.
//!
//! - **Propositional arm** ([`RuleDataset::propositional_examples`]): a boolean
//!   formula over ground unary atoms, evaluated closed-world (`and`/`or`/`not`,
//!   answer `yes`/`no`). Exercises the logical connectives + the `derive`
//!   reduction op.
//! - **Rule-based arm** ([`RuleDataset::rule_examples`]): a depth-`d`
//!   modus-ponens chain over the [`super::world`] rules (`fire`/`bind`, answer
//!   `yes`/`no`). Exercises `select`/`bind`/`fire`/`chain` (Q5) + `implies`.
//!
//! Every gold trace is **correct by construction** — the renderer computes the
//! truth values it prints — so a gold target replays at executability 1.0 with
//! the correct answer (the `gold_*_replays_perfectly` tests). The slot
//! determinative (Q2) is threaded by the `typed_slots` knob: `x1·ANIMAL` typed,
//! `x1` bare, the same example otherwise byte-for-byte.

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::world::{Predicate, Rule, RuleWorld, RuleWorldConfig, random_sort};
use crate::error::GrugResult;
use crate::reason::TraceArm;
use crate::reason::register::{Op, Slot, Sort, Truth};

/// Which deductive skill an example exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicArm {
    Propositional,
    RuleBased,
}

/// A tag on how a rule-arm example reaches its answer — for diagnostics and
/// answer-class-balanced eval slices (not needed for scoring, which uses
/// `gold_answer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExampleKind {
    /// Propositional formula evaluating true/false.
    PropFormula,
    /// Rule chain completes → `yes`.
    RuleProved,
    /// Base premise absent (closed-world false) → `no`.
    RuleNoBaseAbsent,
    /// A nonmonotonic guard holds, blocking the chain → `no`.
    RuleNoGuardBlocked,
}

/// The structured goal an example poses — an atom to prove (rule arm) or a
/// formula to evaluate (propositional arm). The replayer reads this rather than
/// re-parsing the prompt's `goal` line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GoalSpec {
    /// Prove `pred(slot)` (rule arm).
    Atom { pred: Predicate, slot: Slot },
    /// Evaluate a single-connective formula of literals over `slot` (prop arm).
    Formula {
        conn: PropConn,
        literals: Vec<PropLiteral>,
        slot: Slot,
    },
}

/// A propositional connective (the closed subset the prop arm uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropConn {
    And,
    Or,
}

impl PropConn {
    fn op(self) -> Op {
        match self {
            PropConn::And => Op::And,
            PropConn::Or => Op::Or,
        }
    }

    /// Fold `acc` with `next` under this connective (two-valued; the prop arm
    /// never introduces `Gap`). Shared with the replayer's `derive` check.
    pub(crate) fn apply(self, acc: Truth, next: Truth) -> Truth {
        let a = acc == Truth::Yes;
        let b = next == Truth::Yes;
        let r = match self {
            PropConn::And => a && b,
            PropConn::Or => a || b,
        };
        if r { Truth::Yes } else { Truth::No }
    }
}

/// One literal in a propositional formula: a base predicate, possibly negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropLiteral {
    pub pred: Predicate,
    pub negated: bool,
}

/// One deductive example: prompt + High-Grug-v2 target, the gold answer, and
/// the structured ground truth [`super::replay::replay_rule`] checks output
/// against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleExample {
    pub prompt: String,
    pub target: String,
    pub gold_answer: Truth,
    /// Chain depth (rule arm) or literal count (prop arm).
    pub depth: usize,
    pub logic: LogicArm,
    pub kind: ExampleKind,
    pub arm: TraceArm,
    pub typed_slots: bool,
    /// The single individual's sort and slot (all atoms are about this slot).
    pub sort: Sort,
    pub slot: Slot,
    pub goal: GoalSpec,
    /// Every unary ground fact present in the prompt (`pred(slot)` holds).
    pub prompt_atoms: Vec<(Predicate, Slot)>,
    /// Every rule present in the prompt (rule arm; empty for the prop arm).
    pub prompt_rules: Vec<Rule>,
}

/// Config for [`RuleDataset::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct RuleCorpusConfig {
    pub world: RuleWorldConfig,
    /// Extra (present) noise base facts about the individual, per example.
    pub distractor_facts: usize,
    /// Extra distractor rules (any sort) mixed into a rule-arm prompt.
    pub distractor_rules: usize,
    /// Fraction of rule-arm examples that should be provable (`yes`); the rest
    /// are refutations (`no`).
    pub yes_frac: f64,
    /// Inclusive literal-count range for propositional formulas.
    pub prop_literals: (usize, usize),
}

impl Default for RuleCorpusConfig {
    fn default() -> Self {
        Self {
            world: RuleWorldConfig::default(),
            distractor_facts: 4,
            distractor_rules: 6,
            yes_frac: 0.5,
            prop_literals: (2, 4),
        }
    }
}

const PROP_SEED_SALT: u64 = 0x51F0_9C2A_77B4_1E63;
const RULE_SEED_SALT: u64 = 0x1D8E_44C7_A0B9_2F35;
// Eval streams use distinct salts so a held-out draw is a *fresh individual*
// (a novel base-fact pattern over the same shared rules) — genuine in-context
// deductive generalization, since the rules themselves live in the prompt, not
// in the weights. Not overlapping the train stream is what makes the eval a
// generalization test rather than a memorization readout.
const PROP_EVAL_SEED_SALT: u64 = 0xB7A2_4E19_6C03_51DF;
const RULE_EVAL_SEED_SALT: u64 = 0x3C61_D9F5_2A87_44BE;

/// A deductive corpus over a fixed [`RuleWorld`], generating both arms
/// deterministically from a seed.
pub struct RuleDataset {
    config: RuleCorpusConfig,
    world: RuleWorld,
}

impl RuleDataset {
    pub fn new(config: RuleCorpusConfig) -> GrugResult<Self> {
        let world = RuleWorld::generate(config.world.clone());
        Ok(Self { config, world })
    }

    pub fn config(&self) -> &RuleCorpusConfig {
        &self.config
    }

    pub fn world(&self) -> &RuleWorld {
        &self.world
    }

    /// `n` propositional TRAIN examples, targets rendered under `arm`, slots
    /// typed or bare per `typed_slots`.
    pub fn propositional_examples(
        &self,
        n: usize,
        arm: TraceArm,
        typed_slots: bool,
    ) -> Vec<RuleExample> {
        self.prop_stream(PROP_SEED_SALT, n, arm, typed_slots)
    }

    /// The held-out propositional EVAL stream (fresh individuals; see the
    /// eval-salt note above).
    pub fn propositional_examples_eval(
        &self,
        n: usize,
        arm: TraceArm,
        typed_slots: bool,
    ) -> Vec<RuleExample> {
        self.prop_stream(PROP_EVAL_SEED_SALT, n, arm, typed_slots)
    }

    fn prop_stream(&self, salt: u64, n: usize, arm: TraceArm, typed: bool) -> Vec<RuleExample> {
        let mut rng = StdRng::seed_from_u64(self.config.world.seed ^ salt);
        (0..n)
            .map(|_| {
                let sort = random_sort(&mut rng);
                build_prop(&mut rng, &self.world, &self.config, sort, arm, typed)
            })
            .collect()
    }

    /// `n` rule-based TRAIN examples over `depths` (each `<= world.max_level`),
    /// balanced `yes`/`no` by `config.yes_frac`.
    pub fn rule_examples(
        &self,
        n: usize,
        depths: &[usize],
        arm: TraceArm,
        typed_slots: bool,
    ) -> Vec<RuleExample> {
        self.rule_stream(RULE_SEED_SALT, n, depths, arm, typed_slots)
    }

    /// The held-out rule-based EVAL stream (fresh individuals + fresh chains
    /// over the same rules; `depths` may be train depths for the
    /// in-distribution axis or unseen depths for length generalization).
    pub fn rule_examples_eval(
        &self,
        n: usize,
        depths: &[usize],
        arm: TraceArm,
        typed_slots: bool,
    ) -> Vec<RuleExample> {
        self.rule_stream(RULE_EVAL_SEED_SALT, n, depths, arm, typed_slots)
    }

    fn rule_stream(
        &self,
        salt: u64,
        n: usize,
        depths: &[usize],
        arm: TraceArm,
        typed: bool,
    ) -> Vec<RuleExample> {
        assert!(!depths.is_empty(), "rule_examples needs at least one depth");
        let mut rng = StdRng::seed_from_u64(self.config.world.seed ^ salt);
        (0..n)
            .map(|_| {
                let depth = *depths.choose(&mut rng).expect("depths non-empty");
                let sort = random_sort(&mut rng);
                let want_yes = rng.gen_bool(self.config.yes_frac.clamp(0.0, 1.0));
                build_rule(
                    &mut rng, &self.world, &self.config, sort, depth, want_yes, arm, typed,
                )
            })
            .collect()
    }
}

// --- rendering helpers (shared by the gold-trace builders) -------------------

fn slot_render(slot: Slot, sort: Sort, typed: bool) -> String {
    slot.render(if typed { Some(sort) } else { None })
}

fn fact_line(pred: Predicate, slot: Slot, sort: Sort, typed: bool) -> String {
    format!(
        "{} {} {}",
        Op::Fact,
        pred.surface(),
        slot_render(slot, sort, typed)
    )
}

/// `rule <id> | <spine> x [and not <guard> x] implies <head> x`. The rule
/// variable is the literal `x` (not a slot), unambiguous because a bare `x`
/// never parses as a slot (slots require digits).
fn rule_line(rule: &Rule) -> String {
    let body = match rule.guard {
        Some(g) => format!(
            "{} x {} {} {} x",
            rule.spine.surface(),
            Op::And,
            Op::Not,
            g.surface()
        ),
        None => format!("{} x", rule.spine.surface()),
    };
    format!(
        "{} {} | {} {} {} x",
        Op::Rule,
        rule.id(),
        body,
        Op::Implies,
        rule.head.surface()
    )
}

fn goal_atom_line(pred: Predicate, slot: Slot, sort: Sort, typed: bool) -> String {
    format!(
        "{} {} {}",
        Op::Goal,
        pred.surface(),
        slot_render(slot, sort, typed)
    )
}

fn goal_formula_line(
    conn: PropConn,
    literals: &[PropLiteral],
    slot: Slot,
    sort: Sort,
    typed: bool,
) -> String {
    let sr = slot_render(slot, sort, typed);
    let parts: Vec<String> = literals
        .iter()
        .map(|lit| {
            if lit.negated {
                format!("{} {} {}", Op::Not, lit.pred.surface(), sr)
            } else {
                format!("{} {}", lit.pred.surface(), sr)
            }
        })
        .collect();
    format!("{} {}", Op::Goal, parts.join(&format!(" {} ", conn.op())))
}

/// Assemble a prompt: shuffle the fact + rule lines, then the goal line last
/// (mirroring the P3-R corpus's "goal is the final line" convention).
fn assemble_prompt(mut body_lines: Vec<String>, rng: &mut StdRng, goal_line: String) -> String {
    body_lines.shuffle(rng);
    body_lines.push(goal_line);
    body_lines.join("\n")
}

// --- propositional arm -------------------------------------------------------

fn build_prop(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    arm: TraceArm,
    typed: bool,
) -> RuleExample {
    let slot = Slot(1);
    let (lo, hi) = config.prop_literals;
    let n_lit = rng.gen_range(lo.max(1)..=hi.max(lo.max(1)));
    let base = world.base_predicates(sort);
    let n_lit = n_lit.min(base.len()).max(1);

    // Distinct predicates for the literals.
    let mut pool = base.clone();
    pool.shuffle(rng);
    let lit_preds: Vec<Predicate> = pool.into_iter().take(n_lit).collect();

    // Per literal: negation + whether its atom is present (true).
    let literals: Vec<PropLiteral> = lit_preds
        .iter()
        .map(|&pred| PropLiteral {
            pred,
            negated: rng.gen_bool(0.5),
        })
        .collect();
    let present: Vec<bool> = (0..n_lit).map(|_| rng.gen_bool(0.5)).collect();

    let conn = if rng.gen_bool(0.5) {
        PropConn::And
    } else {
        PropConn::Or
    };

    // Prompt facts: the present literal atoms + distractor noise atoms. Every
    // literal predicate is excluded from the distractor draw — otherwise a
    // distractor could re-add a predicate a literal needs *absent*, flipping
    // its closed-world truth and desyncing the gold trace.
    let literal_preds: HashSet<Predicate> = literals.iter().map(|l| l.pred).collect();
    let mut atom_present: HashSet<Predicate> = HashSet::new();
    for (lit, &pres) in literals.iter().zip(present.iter()) {
        if pres {
            atom_present.insert(lit.pred);
        }
    }
    add_distractor_facts_excluding(
        rng,
        world,
        sort,
        config.distractor_facts,
        &literal_preds,
        &mut atom_present,
    );

    let prompt_atoms = sorted_atoms(&atom_present, slot);
    let mut fact_lines: Vec<String> = prompt_atoms
        .iter()
        .map(|&(p, s)| fact_line(p, s, sort, typed))
        .collect();
    // Stable order before shuffle so the seed fully determines the layout.
    fact_lines.sort();
    let goal_line = goal_formula_line(conn, &literals, slot, sort, typed);
    let prompt = assemble_prompt(fact_lines, rng, goal_line);

    // Evaluate + render the trace.
    let lit_truths: Vec<Truth> = literals
        .iter()
        .zip(present.iter())
        .map(|(lit, &pres)| {
            let base_truth = if pres { Truth::Yes } else { Truth::No };
            if lit.negated {
                base_truth.negate()
            } else {
                base_truth
            }
        })
        .collect();
    let mut answer = lit_truths[0];
    for &t in &lit_truths[1..] {
        answer = conn.apply(answer, t);
    }

    let target = match arm {
        TraceArm::Trace => render_prop_trace(&literals, &present, conn, slot, sort, typed),
        TraceArm::AnswerOnly => render_answer_only(answer),
    };

    RuleExample {
        prompt,
        target,
        gold_answer: answer,
        depth: n_lit,
        logic: LogicArm::Propositional,
        kind: ExampleKind::PropFormula,
        arm,
        typed_slots: typed,
        sort,
        slot,
        goal: GoalSpec::Formula {
            conn,
            literals,
            slot,
        },
        prompt_atoms,
        prompt_rules: Vec::new(),
    }
}

fn render_prop_trace(
    literals: &[PropLiteral],
    present: &[bool],
    conn: PropConn,
    slot: Slot,
    sort: Sort,
    typed: bool,
) -> String {
    let sr = slot_render(slot, sort, typed);
    let mut lines: Vec<String> = Vec::new();
    let mut step = 1usize;
    // Per-literal result step + truth.
    let mut running_step = 0usize;
    let mut running_truth = Truth::Yes;
    for (i, (lit, &pres)) in literals.iter().zip(present.iter()).enumerate() {
        let atom_truth = if pres { Truth::Yes } else { Truth::No };
        lines.push(format!(
            "{} {} | {} {} {} | {}",
            Op::Step,
            step,
            Op::Check,
            lit.pred.surface(),
            sr,
            atom_truth
        ));
        let check_step = step;
        step += 1;
        let (lit_step, lit_truth) = if lit.negated {
            let flipped = atom_truth.negate();
            lines.push(format!(
                "{} {} | {} {} | {} {} | {} {}",
                Op::Step,
                step,
                Op::Derive,
                Op::Not,
                Op::By,
                check_step,
                Op::Got,
                flipped
            ));
            let s = step;
            step += 1;
            (s, flipped)
        } else {
            (check_step, atom_truth)
        };

        if i == 0 {
            running_step = lit_step;
            running_truth = lit_truth;
        } else {
            let combined = conn.apply(running_truth, lit_truth);
            lines.push(format!(
                "{} {} | {} {} | {} {} {} | {} {}",
                Op::Step,
                step,
                Op::Derive,
                conn.op(),
                Op::By,
                running_step,
                lit_step,
                Op::Got,
                combined
            ));
            running_step = step;
            running_truth = combined;
            step += 1;
        }
    }
    lines.push(render_answer_only(running_truth));
    lines.join("\n")
}

// --- rule-based arm ----------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_rule(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    depth: usize,
    want_yes: bool,
    arm: TraceArm,
    typed: bool,
) -> RuleExample {
    let slot = Slot(1);
    let depth = depth.min(world.config().max_level as usize).max(1);
    let goal = world.random_predicate(rng, sort, depth as u32);
    // head-first: [P_depth, ..., P_0]; ascending base-first: reverse.
    let mut chain_desc = world.spine_chain(goal);
    chain_desc.reverse(); // now [P_0, P_1, ..., P_depth]
    let asc = chain_desc;
    let base_pred = asc[0];

    let chain_rules: Vec<Rule> = asc[1..]
        .iter()
        .map(|p| {
            world
                .rule_for(*p)
                .expect("chain predicate above base has a rule")
        })
        .collect();
    let chain_guards: Vec<Predicate> = chain_rules.iter().filter_map(|r| r.guard).collect();

    // Controlled predicates whose presence we set explicitly (never distractor).
    let mut controlled: HashSet<Predicate> = HashSet::new();
    controlled.insert(base_pred);
    for &g in &chain_guards {
        controlled.insert(g);
    }

    if want_yes {
        build_rule_yes(
            rng,
            world,
            config,
            sort,
            slot,
            depth,
            goal,
            &asc,
            &chain_rules,
            &controlled,
            arm,
            typed,
        )
    } else {
        // Prefer a guard-break refutation when the chain has a *distinct*,
        // collision-free guard; else fall back to the always-available
        // base-absent refutation.
        build_rule_no_guard(
            rng,
            world,
            config,
            sort,
            slot,
            depth,
            goal,
            &asc,
            &chain_rules,
            arm,
            typed,
        )
        .unwrap_or_else(|| {
            build_rule_no_base_absent(
                rng,
                world,
                config,
                sort,
                slot,
                depth,
                goal,
                &controlled,
                &chain_rules,
                arm,
                typed,
            )
        })
    }
}

/// Common prompt assembly for a rule-arm example given the present base atoms
/// and the rules to show.
#[allow(clippy::too_many_arguments)]
fn assemble_rule_prompt(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    slot: Slot,
    present_atoms: &HashSet<Predicate>,
    chain_rules: &[Rule],
    goal: Predicate,
    typed: bool,
) -> (String, Vec<(Predicate, Slot)>, Vec<Rule>) {
    // Facts.
    let prompt_atoms = sorted_atoms(present_atoms, slot);
    let mut lines: Vec<String> = prompt_atoms
        .iter()
        .map(|&(p, s)| fact_line(p, s, sort, typed))
        .collect();

    // Rules: chain rules + distractor rules (deduplicated by id).
    let mut rules: Vec<Rule> = chain_rules.to_vec();
    let chain_ids: HashSet<String> = chain_rules.iter().map(Rule::id).collect();
    let mut attempts = 0;
    while rules.len() < chain_rules.len() + config.distractor_rules && attempts < 200 {
        attempts += 1;
        let dsort = random_sort(rng);
        let level = rng.gen_range(1..=world.config().max_level);
        let pred = world.random_predicate(rng, dsort, level);
        if let Some(r) = world.rule_for(pred)
            && !chain_ids.contains(&r.id())
            && !rules.iter().any(|x| x.id() == r.id())
        {
            rules.push(r);
        }
    }

    let mut rule_lines: Vec<String> = rules.iter().map(rule_line).collect();
    lines.append(&mut rule_lines);
    lines.sort();

    let goal_line = goal_atom_line(goal, slot, sort, typed);
    let prompt = assemble_prompt(lines, rng, goal_line);
    (prompt, prompt_atoms, rules)
}

#[allow(clippy::too_many_arguments)]
fn build_rule_yes(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    slot: Slot,
    depth: usize,
    goal: Predicate,
    asc: &[Predicate],
    chain_rules: &[Rule],
    controlled: &HashSet<Predicate>,
    arm: TraceArm,
    typed: bool,
) -> RuleExample {
    // yes: base present, all guards absent.
    let mut present: HashSet<Predicate> = HashSet::new();
    present.insert(asc[0]);
    add_distractor_facts_excluding(
        rng,
        world,
        sort,
        config.distractor_facts,
        controlled,
        &mut present,
    );

    let (prompt, prompt_atoms, prompt_rules) = assemble_rule_prompt(
        rng,
        world,
        config,
        sort,
        slot,
        &present,
        chain_rules,
        goal,
        typed,
    );

    let target = match arm {
        TraceArm::Trace => render_rule_yes_trace(asc, chain_rules, slot, sort, typed),
        TraceArm::AnswerOnly => render_answer_only(Truth::Yes),
    };

    RuleExample {
        prompt,
        target,
        gold_answer: Truth::Yes,
        depth,
        logic: LogicArm::RuleBased,
        kind: ExampleKind::RuleProved,
        arm,
        typed_slots: typed,
        sort,
        slot,
        goal: GoalSpec::Atom { pred: goal, slot },
        prompt_atoms,
        prompt_rules,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rule_no_base_absent(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    slot: Slot,
    depth: usize,
    goal: Predicate,
    controlled: &HashSet<Predicate>,
    chain_rules: &[Rule],
    arm: TraceArm,
    typed: bool,
) -> RuleExample {
    // no: base predicate absent (closed-world false) -> chain cannot start.
    let base_pred = {
        let mut c = world.spine_chain(goal);
        c.reverse();
        c[0]
    };
    let mut present: HashSet<Predicate> = HashSet::new();
    // Distractors, but the base predicate stays absent.
    let mut exclude = controlled.clone();
    exclude.insert(base_pred);
    add_distractor_facts_excluding(
        rng,
        world,
        sort,
        config.distractor_facts,
        &exclude,
        &mut present,
    );
    present.remove(&base_pred);

    let (prompt, prompt_atoms, prompt_rules) = assemble_rule_prompt(
        rng,
        world,
        config,
        sort,
        slot,
        &present,
        chain_rules,
        goal,
        typed,
    );

    let target = match arm {
        TraceArm::Trace => render_rule_no_base_trace(base_pred, slot, sort, typed),
        TraceArm::AnswerOnly => render_answer_only(Truth::No),
    };

    RuleExample {
        prompt,
        target,
        gold_answer: Truth::No,
        depth,
        logic: LogicArm::RuleBased,
        kind: ExampleKind::RuleNoBaseAbsent,
        arm,
        typed_slots: typed,
        sort,
        slot,
        goal: GoalSpec::Atom { pred: goal, slot },
        prompt_atoms,
        prompt_rules,
    }
}

/// Guard-break refutation: fire the chain up to level `b-1`, then show the
/// guard `g_b` of rule `R_b` holds, blocking `R_b`. `None` (caller falls back to
/// base-absent) when the chain has no guard, or the guard set collides (a guard
/// appearing at two levels, or equal to the base) — collisions make the
/// "fire-below-then-block" story unsound.
#[allow(clippy::too_many_arguments)]
fn build_rule_no_guard(
    rng: &mut StdRng,
    world: &RuleWorld,
    config: &RuleCorpusConfig,
    sort: Sort,
    slot: Slot,
    depth: usize,
    goal: Predicate,
    asc: &[Predicate],
    chain_rules: &[Rule],
    arm: TraceArm,
    typed: bool,
) -> Option<RuleExample> {
    // Find the lowest level b (1-indexed into chain_rules => rule for asc[b])
    // whose rule has a guard.
    let b = chain_rules.iter().position(|r| r.guard.is_some())? + 1; // asc index of the blocked head
    let blocked_rule = chain_rules[b - 1];
    let g_b = blocked_rule.guard.expect("selected a guarded rule");

    // The rules strictly below `b` must all fire, so their guards must be
    // absent. The block requires `g_b` present. The only unsound case is a
    // below-guard equal to `g_b` (both present and absent). Guards above `b`
    // never fire, so their presence is irrelevant. (The disjoint level-0 pools
    // already guarantee `g_b != base`.)
    let base_pred = asc[0];
    let below_guards: Vec<Predicate> = chain_rules[..b - 1]
        .iter()
        .filter_map(|r| r.guard)
        .collect();
    if below_guards.contains(&g_b) {
        return None;
    }

    // Present set: base present; below-guards absent; g_b present.
    let mut present: HashSet<Predicate> = HashSet::new();
    present.insert(base_pred);
    present.insert(g_b);
    // Controlled = base + every chain guard (kept out of the distractor draw so
    // no below-guard is accidentally flipped present).
    let guards: Vec<Predicate> = chain_rules.iter().filter_map(|r| r.guard).collect();
    let mut controlled: HashSet<Predicate> = HashSet::new();
    controlled.insert(base_pred);
    for &g in &guards {
        controlled.insert(g);
    }
    add_distractor_facts_excluding(
        rng,
        world,
        sort,
        config.distractor_facts,
        &controlled,
        &mut present,
    );

    let (prompt, prompt_atoms, prompt_rules) = assemble_rule_prompt(
        rng,
        world,
        config,
        sort,
        slot,
        &present,
        chain_rules,
        goal,
        typed,
    );

    let target = match arm {
        TraceArm::Trace => render_rule_no_guard_trace(asc, chain_rules, b, g_b, slot, sort, typed),
        TraceArm::AnswerOnly => render_answer_only(Truth::No),
    };

    Some(RuleExample {
        prompt,
        target,
        gold_answer: Truth::No,
        depth,
        logic: LogicArm::RuleBased,
        kind: ExampleKind::RuleNoGuardBlocked,
        arm,
        typed_slots: typed,
        sort,
        slot,
        goal: GoalSpec::Atom { pred: goal, slot },
        prompt_atoms,
        prompt_rules,
    })
}

/// Render a full bottom-up modus-ponens proof of the goal. Returns the trace
/// ending in `call answer | value yes`.
fn render_rule_yes_trace(
    asc: &[Predicate],
    chain_rules: &[Rule],
    slot: Slot,
    sort: Sort,
    typed: bool,
) -> String {
    let sr = slot_render(slot, sort, typed);
    let mut lines: Vec<String> = Vec::new();
    let mut step = 1usize;
    let mut step_of: HashMap<Predicate, usize> = HashMap::new();

    // Recall the base fact.
    lines.push(format!(
        "{} {} | {} {} {} | {} {}",
        Op::Step,
        step,
        Op::Recall,
        asc[0].surface(),
        sr,
        Op::Got,
        Truth::Yes
    ));
    step_of.insert(asc[0], step);
    step += 1;

    // Fire up the chain.
    for (k, rule) in chain_rules.iter().enumerate() {
        let head = asc[k + 1];
        let mut by: Vec<usize> = vec![*step_of.get(&rule.spine).expect("spine established")];
        if let Some(g) = rule.guard {
            lines.push(format!(
                "{} {} | {} {} {} | {}",
                Op::Step,
                step,
                Op::Check,
                g.surface(),
                sr,
                Truth::No
            ));
            by.push(step);
            step += 1;
        }
        let by_s: Vec<String> = by.iter().map(|n| n.to_string()).collect();
        lines.push(format!(
            "{} {} | {} {} | {} x {} | {} {} {} | {} {}",
            Op::Step,
            step,
            Op::Fire,
            rule.id(),
            Op::Bind,
            sr,
            Op::Got,
            head.surface(),
            sr,
            Op::By,
            by_s.join(" ")
        ));
        step_of.insert(head, step);
        step += 1;
    }

    lines.push(render_answer_only(Truth::Yes));
    lines.join("\n")
}

fn render_rule_no_base_trace(base_pred: Predicate, slot: Slot, sort: Sort, typed: bool) -> String {
    let sr = slot_render(slot, sort, typed);
    let mut lines = Vec::new();
    lines.push(format!(
        "{} 1 | {} {} {} | {}",
        Op::Step,
        Op::Check,
        base_pred.surface(),
        sr,
        Truth::No
    ));
    lines.push(render_answer_only(Truth::No));
    lines.join("\n")
}

/// Fire levels `1..b-1`, then recall the holding guard `g_b` as the block
/// witness, then answer `no`.
fn render_rule_no_guard_trace(
    asc: &[Predicate],
    chain_rules: &[Rule],
    b: usize,
    g_b: Predicate,
    slot: Slot,
    sort: Sort,
    typed: bool,
) -> String {
    let sr = slot_render(slot, sort, typed);
    let mut lines: Vec<String> = Vec::new();
    let mut step = 1usize;
    let mut step_of: HashMap<Predicate, usize> = HashMap::new();

    lines.push(format!(
        "{} {} | {} {} {} | {} {}",
        Op::Step,
        step,
        Op::Recall,
        asc[0].surface(),
        sr,
        Op::Got,
        Truth::Yes
    ));
    step_of.insert(asc[0], step);
    step += 1;

    // Fire rules for levels below b (chain_rules indices 0..b-1).
    for (k, rule) in chain_rules.iter().enumerate().take(b - 1) {
        let head = asc[k + 1];
        let mut by: Vec<usize> = vec![*step_of.get(&rule.spine).expect("spine established")];
        if let Some(g) = rule.guard {
            lines.push(format!(
                "{} {} | {} {} {} | {}",
                Op::Step,
                step,
                Op::Check,
                g.surface(),
                sr,
                Truth::No
            ));
            by.push(step);
            step += 1;
        }
        let by_s: Vec<String> = by.iter().map(|n| n.to_string()).collect();
        lines.push(format!(
            "{} {} | {} {} | {} x {} | {} {} {} | {} {}",
            Op::Step,
            step,
            Op::Fire,
            rule.id(),
            Op::Bind,
            sr,
            Op::Got,
            head.surface(),
            sr,
            Op::By,
            by_s.join(" ")
        ));
        step_of.insert(head, step);
        step += 1;
    }

    // The block: the guard g_b holds.
    lines.push(format!(
        "{} {} | {} {} {} | {} {}",
        Op::Step,
        step,
        Op::Recall,
        g_b.surface(),
        sr,
        Op::Got,
        Truth::Yes
    ));
    lines.push(render_answer_only(Truth::No));
    lines.join("\n")
}

fn render_answer_only(value: Truth) -> String {
    format!("{} {} | {} {}", Op::Call, Op::Answer, Op::Value, value)
}

// --- distractor helpers ------------------------------------------------------

/// Collect a present-atom set into a `(Predicate, Slot)` Vec in a **stable**
/// order (by predicate surface, then slot) — the atoms come out of a `HashSet`,
/// whose iteration order is randomized per process, so sorting is what makes
/// the example byte-identical across `ReasonDataset` instances at the same seed.
fn sorted_atoms(present: &HashSet<Predicate>, slot: Slot) -> Vec<(Predicate, Slot)> {
    let mut v: Vec<(Predicate, Slot)> = present.iter().map(|&p| (p, slot)).collect();
    v.sort_by(|a, b| a.0.surface().cmp(&b.0.surface()).then(a.1.0.cmp(&b.1.0)));
    v
}

/// Add up to `count` present noise base facts of `sort`, never touching a
/// predicate in `exclude` (the controlled set whose presence the caller pins).
fn add_distractor_facts_excluding(
    rng: &mut StdRng,
    world: &RuleWorld,
    sort: Sort,
    count: usize,
    exclude: &HashSet<Predicate>,
    present: &mut HashSet<Predicate>,
) {
    let base = world.base_predicates(sort);
    let mut candidates: Vec<Predicate> = base
        .into_iter()
        .filter(|p| !exclude.contains(p) && !present.contains(p))
        .collect();
    candidates.shuffle(rng);
    for p in candidates.into_iter().take(count) {
        present.insert(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::rule::replay::replay_rule;

    fn dataset() -> RuleDataset {
        RuleDataset::new(RuleCorpusConfig {
            world: RuleWorldConfig {
                seed: 11,
                preds_per_cell: 6,
                max_level: 6,
                guard_prob: 0.6,
            },
            distractor_facts: 4,
            distractor_rules: 6,
            yes_frac: 0.5,
            prop_literals: (2, 4),
        })
        .unwrap()
    }

    #[test]
    fn same_seed_reproduces_byte_identical_examples() {
        let a = dataset();
        let b = dataset();
        for typed in [false, true] {
            assert_eq!(
                a.propositional_examples(40, TraceArm::Trace, typed),
                b.propositional_examples(40, TraceArm::Trace, typed)
            );
            assert_eq!(
                a.rule_examples(40, &[2, 3, 4], TraceArm::Trace, typed),
                b.rule_examples(40, &[2, 3, 4], TraceArm::Trace, typed)
            );
        }
    }

    #[test]
    fn train_and_eval_streams_differ() {
        // The held-out eval stream must be fresh individuals, not a replay of
        // the train stream (else "generalization" would be memorization).
        let ds = dataset();
        let train = ds.rule_examples(100, &[2, 3, 4], TraceArm::Trace, true);
        let eval = ds.rule_examples_eval(100, &[2, 3, 4], TraceArm::Trace, true);
        let train_prompts: HashSet<&str> = train.iter().map(|e| e.prompt.as_str()).collect();
        let overlap = eval
            .iter()
            .filter(|e| train_prompts.contains(e.prompt.as_str()))
            .count();
        assert!(overlap < 5, "train/eval streams overlap too much: {overlap}/100");

        let ptrain = ds.propositional_examples(100, TraceArm::Trace, false);
        let peval = ds.propositional_examples_eval(100, TraceArm::Trace, false);
        let ptp: HashSet<&str> = ptrain.iter().map(|e| e.prompt.as_str()).collect();
        let poverlap = peval
            .iter()
            .filter(|e| ptp.contains(e.prompt.as_str()))
            .count();
        assert!(poverlap < 15, "prop train/eval overlap too high: {poverlap}/100");
    }

    #[test]
    fn prop_gold_traces_replay_perfectly() {
        let ds = dataset();
        for typed in [false, true] {
            for ex in ds.propositional_examples(80, TraceArm::Trace, typed) {
                let report = replay_rule(&ex, &ex.target);
                assert_eq!(
                    report.executability, 1.0,
                    "prop gold should replay at 1.0: {ex:?}\n{report:?}"
                );
                assert!(report.answer_correct, "prop gold answer wrong: {ex:?}");
                assert!(report.malformed_lines.is_empty(), "{report:?}");
            }
        }
    }

    #[test]
    fn rule_gold_traces_replay_perfectly_all_classes() {
        let ds = dataset();
        for typed in [false, true] {
            let examples = ds.rule_examples(200, &[1, 2, 3, 4, 5, 6], TraceArm::Trace, typed);
            // All three rule classes should appear across 200 examples.
            let mut kinds: HashSet<ExampleKind> = HashSet::new();
            for ex in &examples {
                kinds.insert(ex.kind);
                let report = replay_rule(ex, &ex.target);
                assert_eq!(
                    report.executability, 1.0,
                    "rule gold should replay at 1.0: {ex:?}\n{report:?}"
                );
                assert!(
                    report.answer_correct,
                    "rule gold answer wrong ({:?}): {ex:?}",
                    ex.kind
                );
                assert!(report.malformed_lines.is_empty(), "{report:?}");
            }
            assert!(
                kinds.contains(&ExampleKind::RuleProved),
                "no proved examples generated"
            );
            assert!(
                kinds.contains(&ExampleKind::RuleNoBaseAbsent)
                    || kinds.contains(&ExampleKind::RuleNoGuardBlocked),
                "no refutation examples generated"
            );
        }
    }

    #[test]
    fn no_gold_trace_or_prompt_contains_a_dark_op() {
        // Governance: only live tokens may be emitted (Q3/Q6). Every
        // reserved-dark op's surface must be absent from generated text.
        let ds = dataset();
        let dark: Vec<&str> = Op::ALL
            .iter()
            .filter(|o| !o.is_live())
            .map(|o| o.surface())
            .collect();
        let mut all_text = String::new();
        for typed in [false, true] {
            for ex in ds.propositional_examples(30, TraceArm::Trace, typed) {
                all_text.push_str(&ex.prompt);
                all_text.push('\n');
                all_text.push_str(&ex.target);
                all_text.push('\n');
            }
            for ex in ds.rule_examples(60, &[2, 3, 4, 5], TraceArm::Trace, typed) {
                all_text.push_str(&ex.prompt);
                all_text.push('\n');
                all_text.push_str(&ex.target);
                all_text.push('\n');
            }
        }
        for tok in &dark {
            // Match as a whitespace-delimited token to avoid substring false
            // positives (e.g. "is_a" inside a hypothetical longer word).
            let is_present = all_text
                .split(|c: char| c.is_whitespace() || c == '|')
                .any(|w| &w == tok);
            assert!(!is_present, "dark op `{tok}` leaked into generated text");
        }
    }

    #[test]
    fn typed_slots_carry_the_sort_determinative() {
        let ds = dataset();
        for ex in ds.rule_examples(20, &[3], TraceArm::Trace, true) {
            let want = format!("x1·{}", ex.sort.surface());
            assert!(
                ex.prompt.contains(&want),
                "typed prompt should carry the determinative {want}: {}",
                ex.prompt
            );
        }
        for ex in ds.rule_examples(20, &[3], TraceArm::Trace, false) {
            assert!(
                !ex.prompt.contains('·'),
                "bare prompt must have no determinative: {}",
                ex.prompt
            );
        }
    }

    #[test]
    fn answer_only_targets_are_a_single_call_line() {
        let ds = dataset();
        for ex in ds.rule_examples(20, &[3, 4], TraceArm::AnswerOnly, true) {
            assert_eq!(ex.target.lines().count(), 1, "{}", ex.target);
            assert!(ex.target.starts_with("call answer | value "));
        }
    }
}
