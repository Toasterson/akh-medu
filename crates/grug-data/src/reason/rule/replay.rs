//! The rule/prop replayer with the `select/bind/fire/chain` component
//! decomposition (ADR 092-v2 Q5).
//!
//! [`replay_rule`] parses a model's output against the *example's own* prompt
//! facts and rules and scores it on two independent axes: **executability**
//! (every statement checks out, exactly as the graph-walking replayer does for
//! its grammar) and — for the rule arm — the **component decomposition**, so an
//! 11M failure can distinguish *broken* from *capacity-gated* (Q5's
//! false-abandon guard):
//!
//! - **select** — a `fire` names a prompt rule whose head is the concluded
//!   predicate (the right rule for the goal).
//! - **bind** — the rule's variable is bound to the individual slot, and (typed
//!   ablation) the slot's determinative matches the rule's sort.
//! - **fire** — modus ponens is licensed: the cited `by` steps establish the
//!   rule's body (spine `yes`, guard `no`), and the `got` is the rule's head.
//! - **chain** — the fires compose all the way to the goal predicate (the
//!   terminal conclusion is the goal, established `yes`).
//!
//! The parser is total: an unrecognised line lands in `malformed_lines`, never
//! a panic.

use std::collections::{HashMap, HashSet};

use super::corpus::{GoalSpec, LogicArm, PropConn, RuleExample};
use super::world::{Predicate, Rule};
use crate::reason::register::{Op, Slot, Sort, Truth};

/// The result of replaying one rule/prop output.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleReplayReport {
    pub statements_total: usize,
    pub statements_valid: usize,
    pub executability: f64,
    pub answer: Option<Truth>,
    pub answer_correct: bool,
    pub malformed_lines: Vec<String>,

    // --- component decomposition (rule arm; all zero for the prop arm) ---
    pub select_ok: usize,
    pub select_total: usize,
    pub bind_ok: usize,
    pub bind_total: usize,
    pub fire_ok: usize,
    pub fire_total: usize,
    /// Whether a full derivation to the goal was expected (a provable rule-arm
    /// example).
    pub chain_expected: bool,
    /// Whether the fires actually composed to the goal predicate.
    pub chain_ok: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum RuleLine {
    Recall {
        step: usize,
        pred: Predicate,
        slot: u32,
    },
    Check {
        step: usize,
        pred: Predicate,
        slot: u32,
        truth: Truth,
    },
    Fire {
        step: usize,
        rule_id: String,
        bind_slot: (u32, Option<Sort>),
        got_pred: Predicate,
        got_slot: (u32, Option<Sort>),
        by: Vec<usize>,
    },
    DeriveNot {
        step: usize,
        by: usize,
        got: Truth,
    },
    DeriveConn {
        step: usize,
        conn: PropConn,
        by: Vec<usize>,
        got: Truth,
    },
    Call {
        value: Truth,
    },
    Malformed,
}

fn parse_slot(s: &str) -> Option<(u32, Option<Sort>)> {
    Slot::parse(s).map(|(slot, sort)| (slot.0, sort))
}

fn parse_by(seg: &str) -> Option<Vec<usize>> {
    let rest = seg.strip_prefix(&format!("{} ", Op::By))?;
    let by: Option<Vec<usize>> = rest
        .split_whitespace()
        .map(|t| t.parse::<usize>().ok())
        .collect();
    by.filter(|v| !v.is_empty())
}

fn parse_line(line: &str) -> RuleLine {
    let segs: Vec<&str> = line.split(" | ").map(str::trim).collect();
    if segs.len() < 2 {
        return RuleLine::Malformed;
    }

    // call answer | value X
    if segs[0] == format!("{} {}", Op::Call, Op::Answer) {
        let Some(v) = segs[1].strip_prefix(&format!("{} ", Op::Value)) else {
            return RuleLine::Malformed;
        };
        return match Truth::parse(v.trim()) {
            Some(value) => RuleLine::Call { value },
            None => RuleLine::Malformed,
        };
    }

    let Some(step_s) = segs[0].strip_prefix(&format!("{} ", Op::Step)) else {
        return RuleLine::Malformed;
    };
    let Ok(step) = step_s.trim().parse::<usize>() else {
        return RuleLine::Malformed;
    };

    let mut op_tokens = segs[1].split_whitespace();
    match op_tokens.next().and_then(Op::parse) {
        Some(Op::Recall) => {
            // step N | recall <pred> <slot> | got <truth>
            if segs.len() != 3 {
                return RuleLine::Malformed;
            }
            let (Some(pred_s), Some(slot_s)) = (op_tokens.next(), op_tokens.next()) else {
                return RuleLine::Malformed;
            };
            let (Some(pred), Some((slot, _))) = (Predicate::parse(pred_s), parse_slot(slot_s))
            else {
                return RuleLine::Malformed;
            };
            let Some(got_s) = segs[2].strip_prefix(&format!("{} ", Op::Got)) else {
                return RuleLine::Malformed;
            };
            // A recall always asserts presence; the `got yes` is the phonetic
            // complement. Anything but `yes` is malformed.
            if Truth::parse(got_s.trim()) != Some(Truth::Yes) {
                return RuleLine::Malformed;
            }
            RuleLine::Recall { step, pred, slot }
        }
        Some(Op::Check) => {
            // step N | check <pred> <slot> | <truth>
            if segs.len() != 3 {
                return RuleLine::Malformed;
            }
            let (Some(pred_s), Some(slot_s)) = (op_tokens.next(), op_tokens.next()) else {
                return RuleLine::Malformed;
            };
            let (Some(pred), Some((slot, _)), Some(truth)) = (
                Predicate::parse(pred_s),
                parse_slot(slot_s),
                Truth::parse(segs[2]),
            ) else {
                return RuleLine::Malformed;
            };
            RuleLine::Check {
                step,
                pred,
                slot,
                truth,
            }
        }
        Some(Op::Fire) => {
            // step N | fire <id> | bind x <slot> | got <pred> <slot> | by <nums>
            if segs.len() != 5 {
                return RuleLine::Malformed;
            }
            let Some(rule_id) = op_tokens.next().map(str::to_string) else {
                return RuleLine::Malformed;
            };
            if op_tokens.next().is_some() {
                return RuleLine::Malformed; // extra tokens after the rule id
            }
            // bind x <slot>
            let mut bind_tokens = segs[2].split_whitespace();
            if bind_tokens.next().and_then(Op::parse) != Some(Op::Bind) {
                return RuleLine::Malformed;
            }
            if bind_tokens.next() != Some("x") {
                return RuleLine::Malformed;
            }
            let Some(bind_slot) = bind_tokens.next().and_then(parse_slot) else {
                return RuleLine::Malformed;
            };
            // got <pred> <slot>
            let mut got_tokens = segs[3].split_whitespace();
            if got_tokens.next().and_then(Op::parse) != Some(Op::Got) {
                return RuleLine::Malformed;
            }
            let (Some(got_pred), Some(got_slot)) = (
                got_tokens.next().and_then(Predicate::parse),
                got_tokens.next().and_then(parse_slot),
            ) else {
                return RuleLine::Malformed;
            };
            let Some(by) = parse_by(segs[4]) else {
                return RuleLine::Malformed;
            };
            RuleLine::Fire {
                step,
                rule_id,
                bind_slot,
                got_pred,
                got_slot,
                by,
            }
        }
        Some(Op::Derive) => {
            // step N | derive <conn> | by <nums> | got <truth>
            if segs.len() != 4 {
                return RuleLine::Malformed;
            }
            let Some(conn_op) = op_tokens.next().and_then(Op::parse) else {
                return RuleLine::Malformed;
            };
            if op_tokens.next().is_some() {
                return RuleLine::Malformed;
            }
            let Some(by) = parse_by(segs[2]) else {
                return RuleLine::Malformed;
            };
            let Some(got_s) = segs[3].strip_prefix(&format!("{} ", Op::Got)) else {
                return RuleLine::Malformed;
            };
            let Some(got) = Truth::parse(got_s.trim()) else {
                return RuleLine::Malformed;
            };
            match conn_op {
                Op::Not => {
                    if by.len() != 1 {
                        return RuleLine::Malformed;
                    }
                    RuleLine::DeriveNot {
                        step,
                        by: by[0],
                        got,
                    }
                }
                Op::And => RuleLine::DeriveConn {
                    step,
                    conn: PropConn::And,
                    by,
                    got,
                },
                Op::Or => RuleLine::DeriveConn {
                    step,
                    conn: PropConn::Or,
                    by,
                    got,
                },
                _ => RuleLine::Malformed,
            }
        }
        _ => RuleLine::Malformed,
    }
}

/// Replay `output` against `example` (ADR 092-v2 Q5). Total, never panics.
pub fn replay_rule(example: &RuleExample, output: &str) -> RuleReplayReport {
    let present: HashSet<(Predicate, u32)> = example
        .prompt_atoms
        .iter()
        .map(|&(p, s)| (p, s.0))
        .collect();
    let rules_by_id: HashMap<String, Rule> =
        example.prompt_rules.iter().map(|r| (r.id(), *r)).collect();
    let guard_set: HashSet<Predicate> = example
        .prompt_rules
        .iter()
        .filter_map(|r| r.guard)
        .collect();
    let individual_slot = example.slot.0;

    // Per-step recorded atom + its established truth (for `by`-ref premise
    // checks) and per-step concluded truth (for `derive` folds).
    let mut step_atom: HashMap<usize, (Predicate, u32, Truth)> = HashMap::new();
    let mut result_truth: HashMap<usize, Truth> = HashMap::new();
    let mut atom_status: HashMap<(Predicate, u32), Truth> = HashMap::new();
    let mut refutation_shown = false;
    let mut last_valid_result: Option<Truth> = None;

    let mut statements_total = 0usize;
    let mut statements_valid = 0usize;
    let mut malformed_lines = Vec::new();
    let mut answer: Option<Truth> = None;

    let mut select_ok = 0usize;
    let mut select_total = 0usize;
    let mut bind_ok = 0usize;
    let mut bind_total = 0usize;
    let mut fire_ok = 0usize;
    let mut fire_total = 0usize;

    let goal_atom = match &example.goal {
        GoalSpec::Atom { pred, slot } => Some((*pred, slot.0)),
        GoalSpec::Formula { .. } => None,
    };

    for raw in output.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        statements_total += 1;

        match parse_line(line) {
            RuleLine::Malformed => malformed_lines.push(line.to_string()),

            RuleLine::Recall { step, pred, slot } => {
                if present.contains(&(pred, slot)) {
                    statements_valid += 1;
                    step_atom.insert(step, (pred, slot, Truth::Yes));
                    result_truth.insert(step, Truth::Yes);
                    atom_status.insert((pred, slot), Truth::Yes);
                    last_valid_result = Some(Truth::Yes);
                    // Recalling a rule's guard (which therefore holds) is a
                    // block witness for a `no` answer.
                    if guard_set.contains(&pred) {
                        refutation_shown = true;
                    }
                }
            }

            RuleLine::Check {
                step,
                pred,
                slot,
                truth,
            } => {
                let actual = if present.contains(&(pred, slot)) {
                    Truth::Yes
                } else {
                    Truth::No
                };
                // Closed-world: only yes/no are valid check outcomes here.
                if truth == actual && truth != Truth::Gap {
                    statements_valid += 1;
                    step_atom.insert(step, (pred, slot, truth));
                    result_truth.insert(step, truth);
                    atom_status.insert((pred, slot), truth);
                    last_valid_result = Some(truth);
                    if truth == Truth::No {
                        refutation_shown = true;
                    }
                }
            }

            RuleLine::Fire {
                step,
                rule_id,
                bind_slot,
                got_pred,
                got_slot,
                by,
            } => {
                select_total += 1;
                bind_total += 1;
                fire_total += 1;

                let rule = rules_by_id.get(&rule_id).copied();
                let sel = rule.is_some_and(|r| r.head == got_pred);
                if sel {
                    select_ok += 1;
                }

                // bind: single consistent binding to the individual, sort tag
                // (if present) matching the rule head's sort.
                let (bind_num, bind_sort) = bind_slot;
                let (got_num, got_sort) = got_slot;
                let mut bnd = bind_num == got_num && got_num == individual_slot;
                if let Some(r) = rule {
                    if let Some(gs) = got_sort {
                        bnd &= gs == r.head.sort;
                    }
                    if let Some(bs) = bind_sort {
                        bnd &= bs == r.head.sort;
                    }
                }
                if bnd {
                    bind_ok += 1;
                }

                // fire: modus ponens licensed by cited premises.
                let mut premises_ok = false;
                if let Some(r) = rule {
                    let spine_ok = by
                        .iter()
                        .any(|s| step_atom.get(s) == Some(&(r.spine, got_num, Truth::Yes)));
                    let guard_ok = match r.guard {
                        Some(g) => by
                            .iter()
                            .any(|s| step_atom.get(s) == Some(&(g, got_num, Truth::No))),
                        None => true,
                    };
                    premises_ok = spine_ok && guard_ok && r.head == got_pred;
                }
                let fok = sel && bnd && premises_ok;
                if fok {
                    fire_ok += 1;
                    statements_valid += 1;
                    step_atom.insert(step, (got_pred, got_num, Truth::Yes));
                    result_truth.insert(step, Truth::Yes);
                    atom_status.insert((got_pred, got_num), Truth::Yes);
                    last_valid_result = Some(Truth::Yes);
                }
            }

            RuleLine::DeriveNot { step, by, got } => {
                if let Some(&t) = result_truth.get(&by)
                    && got == t.negate()
                {
                    statements_valid += 1;
                    result_truth.insert(step, got);
                    last_valid_result = Some(got);
                }
            }

            RuleLine::DeriveConn {
                step,
                conn,
                by,
                got,
            } => {
                if by.len() >= 2 {
                    let truths: Option<Vec<Truth>> =
                        by.iter().map(|s| result_truth.get(s).copied()).collect();
                    if let Some(ts) = truths {
                        let mut acc = ts[0];
                        for &t in &ts[1..] {
                            acc = conn.apply(acc, t);
                        }
                        if got == acc {
                            statements_valid += 1;
                            result_truth.insert(step, got);
                            last_valid_result = Some(got);
                        }
                    }
                }
            }

            RuleLine::Call { value } => {
                if answer.is_none() {
                    answer = Some(value);
                }
                let valid = match example.logic {
                    LogicArm::Propositional => last_valid_result == Some(value),
                    LogicArm::RuleBased => {
                        let goal_yes = goal_atom.and_then(|g| atom_status.get(&g).copied())
                            == Some(Truth::Yes);
                        match value {
                            Truth::Yes => goal_yes,
                            Truth::No => !goal_yes && refutation_shown,
                            Truth::Gap => false,
                        }
                    }
                };
                if valid {
                    statements_valid += 1;
                }
            }
        }
    }

    let answer_correct = answer == Some(example.gold_answer);
    let executability = if statements_total == 0 {
        0.0
    } else {
        statements_valid as f64 / statements_total as f64
    };
    let chain_expected = example.logic == LogicArm::RuleBased && example.gold_answer == Truth::Yes;
    let chain_ok = match goal_atom {
        Some(g) => atom_status.get(&g).copied() == Some(Truth::Yes),
        None => false,
    };

    RuleReplayReport {
        statements_total,
        statements_valid,
        executability,
        answer,
        answer_correct,
        malformed_lines,
        select_ok,
        select_total,
        bind_ok,
        bind_total,
        fire_ok,
        fire_total,
        chain_expected,
        chain_ok,
    }
}

// --- aggregate scoring -------------------------------------------------------

/// Per-depth slice of a [`RuleEvalSummary`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuleDepthStats {
    pub n: usize,
    pub answer_accuracy: f64,
    pub executability_rate: f64,
}

/// The `select/bind/fire/chain` component pass-rates over a batch (Q5). Rates
/// are `NaN`-free: a zero denominator reports `0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentSummary {
    pub select_rate: f64,
    pub bind_rate: f64,
    pub fire_rate: f64,
    /// Over examples where a full chain was expected (provable rule-arm
    /// examples), the fraction whose fires composed to the goal.
    pub chain_rate: f64,
    pub select_total: usize,
    pub bind_total: usize,
    pub fire_total: usize,
    pub chain_expected_total: usize,
}

/// The full aggregate report over a scored deductive batch.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleEvalSummary {
    pub n: usize,
    pub answer_accuracy: f64,
    pub executability_rate: f64,
    pub trace_valid_rate: f64,
    pub per_depth: HashMap<usize, RuleDepthStats>,
    pub components: ComponentSummary,
}

/// Score a batch of raw `outputs` against their source `examples` (paired by
/// index; the shorter slice wins).
pub fn score_rule_outputs(examples: &[RuleExample], outputs: &[String]) -> RuleEvalSummary {
    let n = examples.len().min(outputs.len());
    if n == 0 {
        return RuleEvalSummary {
            n: 0,
            answer_accuracy: 0.0,
            executability_rate: 0.0,
            trace_valid_rate: 0.0,
            per_depth: HashMap::new(),
            components: ComponentSummary {
                select_rate: 0.0,
                bind_rate: 0.0,
                fire_rate: 0.0,
                chain_rate: 0.0,
                select_total: 0,
                bind_total: 0,
                fire_total: 0,
                chain_expected_total: 0,
            },
        };
    }

    let reports: Vec<(&RuleExample, RuleReplayReport)> = (0..n)
        .map(|i| (&examples[i], replay_rule(&examples[i], &outputs[i])))
        .collect();

    let answer_accuracy =
        reports.iter().filter(|(_, r)| r.answer_correct).count() as f64 / n as f64;
    let executability_rate = reports.iter().map(|(_, r)| r.executability).sum::<f64>() / n as f64;
    let trace_valid_rate = reports
        .iter()
        .filter(|(_, r)| {
            r.statements_total > 0
                && r.statements_valid == r.statements_total
                && r.malformed_lines.is_empty()
        })
        .count() as f64
        / n as f64;

    let mut raw_depth: HashMap<usize, (usize, usize, f64)> = HashMap::new();
    for (ex, r) in &reports {
        let e = raw_depth.entry(ex.depth).or_insert((0, 0, 0.0));
        e.0 += 1;
        if r.answer_correct {
            e.1 += 1;
        }
        e.2 += r.executability;
    }
    let per_depth = raw_depth
        .into_iter()
        .map(|(d, (cnt, correct, exec))| {
            (
                d,
                RuleDepthStats {
                    n: cnt,
                    answer_accuracy: correct as f64 / cnt.max(1) as f64,
                    executability_rate: exec / cnt.max(1) as f64,
                },
            )
        })
        .collect();

    let (mut so, mut st, mut bo, mut bt, mut fo, mut ft) = (0, 0, 0, 0, 0, 0);
    let (mut chain_ok, mut chain_exp) = (0usize, 0usize);
    for (_, r) in &reports {
        so += r.select_ok;
        st += r.select_total;
        bo += r.bind_ok;
        bt += r.bind_total;
        fo += r.fire_ok;
        ft += r.fire_total;
        if r.chain_expected {
            chain_exp += 1;
            if r.chain_ok {
                chain_ok += 1;
            }
        }
    }
    let rate = |num: usize, den: usize| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    let components = ComponentSummary {
        select_rate: rate(so, st),
        bind_rate: rate(bo, bt),
        fire_rate: rate(fo, ft),
        chain_rate: rate(chain_ok, chain_exp),
        select_total: st,
        bind_total: bt,
        fire_total: ft,
        chain_expected_total: chain_exp,
    };

    RuleEvalSummary {
        n,
        answer_accuracy,
        executability_rate,
        trace_valid_rate,
        per_depth,
        components,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::TraceArm;
    use crate::reason::rule::corpus::{RuleCorpusConfig, RuleDataset};
    use crate::reason::rule::world::RuleWorldConfig;

    fn dataset() -> RuleDataset {
        RuleDataset::new(RuleCorpusConfig {
            world: RuleWorldConfig {
                seed: 23,
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
    fn perfect_rule_batch_scores_perfectly() {
        let ds = dataset();
        let examples = ds.rule_examples(120, &[2, 3, 4, 5], TraceArm::Trace, true);
        let outputs: Vec<String> = examples.iter().map(|e| e.target.clone()).collect();
        let s = score_rule_outputs(&examples, &outputs);
        assert_eq!(s.answer_accuracy, 1.0);
        assert_eq!(s.executability_rate, 1.0);
        assert_eq!(s.trace_valid_rate, 1.0);
        // Every fire component is perfect on gold, and every provable example
        // chained to the goal.
        assert_eq!(s.components.select_rate, 1.0);
        assert_eq!(s.components.bind_rate, 1.0);
        assert_eq!(s.components.fire_rate, 1.0);
        assert_eq!(s.components.chain_rate, 1.0);
        assert!(s.components.fire_total > 0, "fires should have happened");
    }

    #[test]
    fn perfect_prop_batch_scores_perfectly_with_no_fire_components() {
        let ds = dataset();
        let examples = ds.propositional_examples(80, TraceArm::Trace, false);
        let outputs: Vec<String> = examples.iter().map(|e| e.target.clone()).collect();
        let s = score_rule_outputs(&examples, &outputs);
        assert_eq!(s.answer_accuracy, 1.0);
        assert_eq!(s.executability_rate, 1.0);
        assert_eq!(s.components.fire_total, 0, "prop arm fires nothing");
        assert_eq!(s.components.select_rate, 0.0);
    }

    #[test]
    fn garbage_scores_near_zero() {
        let ds = dataset();
        let examples = ds.rule_examples(20, &[3, 4], TraceArm::Trace, true);
        let outputs: Vec<String> = examples
            .iter()
            .map(|_| "total nonsense".to_string())
            .collect();
        let s = score_rule_outputs(&examples, &outputs);
        assert_eq!(s.answer_accuracy, 0.0);
        assert_eq!(s.executability_rate, 0.0);
    }

    #[test]
    fn wrong_rule_id_fails_select_but_may_keep_bind() {
        let ds = dataset();
        // A provable, guard-free-ish deep example so it has >= 1 fire.
        let ex = ds
            .rule_examples(200, &[4], TraceArm::Trace, true)
            .into_iter()
            .find(|e| e.gold_answer == Truth::Yes && e.target.contains("| fire "))
            .expect("a provable multi-fire example exists");
        // Corrupt the first fired rule id to a non-existent one (a single
        // token, so the line still parses as a well-formed fire).
        let real_id = ex
            .target
            .lines()
            .find_map(|l| l.split(" | ").nth(1)?.strip_prefix("fire "))
            .expect("a fire line with a rule id")
            .to_string();
        let corrupted = ex
            .target
            .replacen(&format!("fire {real_id}"), "fire r_zz_9_9", 1);
        assert_ne!(corrupted, ex.target);
        let report = replay_rule(&ex, &corrupted);
        assert!(
            report.malformed_lines.is_empty(),
            "corruption should stay well-formed: {report:?}"
        );
        assert!(
            report.select_ok < report.select_total,
            "a bogus rule id must fail select: {report:?}"
        );
    }

    #[test]
    fn fire_without_established_premise_fails_fire_component() {
        let ds = dataset();
        let ex = ds
            .rule_examples(200, &[4], TraceArm::Trace, true)
            .into_iter()
            .find(|e| e.gold_answer == Truth::Yes && e.target.matches("| fire ").count() >= 2)
            .expect("a provable example with >=2 fires exists");
        // Drop the base `recall` line: now the first fire's spine premise is
        // never established, so that fire (and the chain) must fail.
        let without_recall: String = ex
            .target
            .lines()
            .filter(|l| !l.contains("| recall "))
            .collect::<Vec<_>>()
            .join("\n");
        let report = replay_rule(&ex, &without_recall);
        assert!(
            report.fire_ok < report.fire_total,
            "a fire citing an unestablished premise must fail: {report:?}"
        );
        assert!(!report.chain_ok, "chain must not complete: {report:?}");
    }

    #[test]
    fn empty_output_is_zero_not_nan() {
        let ds = dataset();
        let ex = ds.rule_examples(1, &[3], TraceArm::Trace, true).remove(0);
        let report = replay_rule(&ex, "");
        assert_eq!(report.statements_total, 0);
        assert_eq!(report.executability, 0.0);
        assert!(!report.answer_correct);
    }
}
