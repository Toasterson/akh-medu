//! High-Grug **v2** reasoning register: the closed, governed operator core, its
//! reserved-token id layout, the sort/determinative classifiers, and the
//! positional entity slots.
//!
//! This is the *shared vocabulary* ADR 092-v2 (Q3/Q6) commits to: the organ,
//! System-2, and the trace verifier all read and write the same closed set of
//! operators, so the ADR-093 handoff needs no translation layer. The set is
//! **closed by governance** — the governance rule is "a corpus must measurably
//! exercise a token," not "it would be nice" — so this module distinguishes
//! **live** operators (emitted by the Phase-51 P4 slice-1 corpora, see
//! [`Op::is_live`]) from **reserved-but-dark** ones (id-carved for
//! forward-compatibility, lit only when a corpus that exercises them lands).
//!
//! # The governed region (Q6)
//!
//! The definitive tokenizer reserves one contiguous region for this register.
//! Absolute offsets are assigned by the tokenizer builder in a later slice;
//! here we fix only the *relative* layout and block sizes, so the renderer and
//! that future tokenizer agree on ordering:
//!
//! ```text
//! [ OP block    ] Op::ALL.len()   ids   — Op::reserved_id()   in 0 .. OP_BLOCK_LEN
//! [ TRUTH block ] Truth::ALL.len() ids  — value literals yes/no/gap
//! [ SORT block  ] Sort::ALL.len() ids   — determinative classifiers (Q2)
//! [ SLOT block  ] MAX_SLOTS ids         — positional entity slots x1..xMAX_SLOTS (Q1)
//! [ DARK block  ] reserved, unfilled    — logogram SymbolId promotion (Q1, dark)
//! ```
//!
//! The renderer emits surface strings ([`Op::surface`], [`Slot::render`], …);
//! the reserved ids exist so the tokenizer↔renderer lockstep Q6 makes
//! load-bearing is checkable *now* (contiguous, distinct, round-tripping — the
//! `reserved_ids_are_contiguous_and_distinct` test), long before the definitive
//! tokenizer exists.
//!
//! # Egyptian mapping (ADR 092-v2, made literal)
//!
//! - **phonograms** → [`Op`] (the small closed, composed operator set)
//! - **determinatives** → [`Sort`] (unpronounced category glyph on a slot)
//! - **phonetic complements** → `got` / `by` (self-verifying redundancy: every
//!   statement carries what it needs to be re-checked)
//! - **logograms** → the dark SymbolId block ([`Op::Logogram`], reserved)

use std::fmt;

use serde::{Deserialize, Serialize};

/// The closed High-Grug-v2 operator core (Q3). Ordering is stable and
/// load-bearing: [`Op::reserved_id`] is the position in [`Op::ALL`], so
/// reordering variants renumbers the governed region — don't.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    // --- prompt / boundary markers ---
    /// A ground fact line in the prompt (`fact <pred> <slot>` unary, or the v1
    /// `fact <subj> <rel> <obj>` binary form).
    Fact,
    /// A universally-quantified rule line in the prompt (`rule <id> | <body>
    /// implies <head>`).
    Rule,
    /// The query line (`goal <atom>` — the thing to prove / evaluate).
    Goal,

    // --- trace mechanics (one workspace state per line) ---
    /// Statement/line marker with an explicit step number (ADR 092 D2).
    Step,
    /// Bring a *present* ground fact into a numbered step (`recall <pred>
    /// <slot> | got yes`). Must match a prompt fact or the statement is
    /// invalid.
    Recall,
    /// The phonetic-complement result segment of a `recall`/`fire` line —
    /// self-verifying redundancy.
    Got,
    /// Probe an atom's epistemic status (`check <pred> <slot> | yes|no|gap`) —
    /// the truth-valued generalization of v1's absence probe. `gap` = unknown
    /// (open-world), `no` = false (closed-world), `yes` = present.
    Check,
    /// Combine prior steps under a connective (`derive <connective> | by i j |
    /// got <truth>`) — the propositional-arm reduction op.
    Derive,
    /// Back-reference segment citing the prior steps a `derive`/`fire` rests on
    /// (`by i j`). The phonetic complement of a derivation.
    By,
    /// Apply modus ponens with a bound rule (`fire <rule-id> | bind x <slot> |
    /// got <head-pred> <slot> | by <steps>`) — the rule-arm core op.
    Fire,
    /// The variable-binding segment of a `fire` line (`bind x <slot>`).
    Bind,

    // --- boundary call (ADR 088 carryover) ---
    Call,
    Answer,
    Value,

    // --- logical connectives ---
    And,
    Or,
    Not,
    Implies,

    // --- reserved-but-dark structural relations (NOT emitted in slice 1) ---
    // Binary structural relations; lit when the binary-rule corpus lands (a
    // later P4 slice). Id-carved now so the governed region is forward-stable.
    IsA,
    PartOf,
    Causes,
    Before,
    After,

    // --- reserved-but-dark marker ---
    /// The logogram-promotion marker for the dark SymbolId block (Q1). Reserved
    /// so promotion is forward-compatible; never emitted while the block is
    /// dark.
    Logogram,
}

impl Op {
    /// Every operator, live then dark, in the fixed governed-region order.
    pub const ALL: [Op; 24] = [
        Op::Fact,
        Op::Rule,
        Op::Goal,
        Op::Step,
        Op::Recall,
        Op::Got,
        Op::Check,
        Op::Derive,
        Op::By,
        Op::Fire,
        Op::Bind,
        Op::Call,
        Op::Answer,
        Op::Value,
        Op::And,
        Op::Or,
        Op::Not,
        Op::Implies,
        Op::IsA,
        Op::PartOf,
        Op::Causes,
        Op::Before,
        Op::After,
        Op::Logogram,
    ];

    /// The canonical surface form (what the renderer emits and [`Op::parse`]
    /// accepts). Underscored where the token is multi-word conceptually
    /// (`is_a`) so it stays a single whitespace-delimited token.
    pub const fn surface(self) -> &'static str {
        match self {
            Op::Fact => "fact",
            Op::Rule => "rule",
            Op::Goal => "goal",
            Op::Step => "step",
            Op::Recall => "recall",
            Op::Got => "got",
            Op::Check => "check",
            Op::Derive => "derive",
            Op::By => "by",
            Op::Fire => "fire",
            Op::Bind => "bind",
            Op::Call => "call",
            Op::Answer => "answer",
            Op::Value => "value",
            Op::And => "and",
            Op::Or => "or",
            Op::Not => "not",
            Op::Implies => "implies",
            Op::IsA => "is_a",
            Op::PartOf => "part_of",
            Op::Causes => "causes",
            Op::Before => "before",
            Op::After => "after",
            Op::Logogram => "logogram",
        }
    }

    /// Whether this operator is **emitted by a slice-1 corpus** (governance:
    /// only live tokens may appear in generated text). The dark set is
    /// id-reserved but never rendered — a test enforces that no gold trace
    /// contains a dark surface form.
    pub const fn is_live(self) -> bool {
        !matches!(
            self,
            Op::IsA | Op::PartOf | Op::Causes | Op::Before | Op::After | Op::Logogram
        )
    }

    /// This operator's id within the OP block of the governed region (its
    /// index in [`Op::ALL`]). Stable across builds; the tokenizer adds the
    /// block's absolute base offset.
    pub fn reserved_id(self) -> u32 {
        Op::ALL
            .iter()
            .position(|&o| o == self)
            .expect("every Op is in Op::ALL") as u32
    }

    /// Parse a surface token back to its [`Op`] (exact inverse of
    /// [`Op::surface`]). Total: `None` for anything not in the closed set.
    pub fn parse(s: &str) -> Option<Self> {
        Op::ALL.iter().copied().find(|o| o.surface() == s)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.surface())
    }
}

/// The value-literal set: the closed vocabulary a `call answer | value _` line
/// or a `check`/`derive` result segment can carry. `Gap` is the ADR 092 D1
/// abstention value (identical surface to [`super::GAP_VALUE`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Truth {
    /// The atom holds / the formula evaluates true / the goal is derivable.
    Yes,
    /// The atom is false under closed-world (propositional arm, full
    /// information).
    No,
    /// Unknown / not derivable — the reused abstention value (open-world).
    Gap,
}

impl Truth {
    pub const ALL: [Truth; 3] = [Truth::Yes, Truth::No, Truth::Gap];

    pub const fn surface(self) -> &'static str {
        match self {
            Truth::Yes => "yes",
            Truth::No => "no",
            Truth::Gap => super::GAP_VALUE,
        }
    }

    /// Id within the TRUTH block (index in [`Truth::ALL`]).
    pub fn reserved_id(self) -> u32 {
        Truth::ALL
            .iter()
            .position(|&t| t == self)
            .expect("every Truth is in Truth::ALL") as u32
    }

    pub fn parse(s: &str) -> Option<Self> {
        Truth::ALL.iter().copied().find(|t| t.surface() == s)
    }

    /// Boolean negation over the two-valued literals; `Gap` negates to `Gap`
    /// (negating the unknown is still unknown — the propositional arm only ever
    /// negates `Yes`/`No`, but keeping this total avoids a panic path).
    pub const fn negate(self) -> Truth {
        match self {
            Truth::Yes => Truth::No,
            Truth::No => Truth::Yes,
            Truth::Gap => Truth::Gap,
        }
    }
}

impl fmt::Display for Truth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.surface())
    }
}

/// A sort/determinative classifier (Q2): the always-present coarse category
/// glyph attached to a slot (`x7·PLACE`). Orthogonal to the fine unary
/// predicates a rule reasons over; its job is to **type-gate** rule
/// applicability so the model can prune candidate rules by sort — the sharp
/// hypothesis that typed slots cut off-path derailment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sort {
    Thing,
    Animal,
    Place,
    Tool,
    Agent,
}

impl Sort {
    pub const ALL: [Sort; 5] = [
        Sort::Thing,
        Sort::Animal,
        Sort::Place,
        Sort::Tool,
        Sort::Agent,
    ];

    /// The uppercase determinative surface (`ANIMAL`), rendered after the
    /// [`DETERMINATIVE_SEP`] on a typed slot.
    pub const fn surface(self) -> &'static str {
        match self {
            Sort::Thing => "THING",
            Sort::Animal => "ANIMAL",
            Sort::Place => "PLACE",
            Sort::Tool => "TOOL",
            Sort::Agent => "AGENT",
        }
    }

    /// Id within the SORT block (index in [`Sort::ALL`]).
    pub fn reserved_id(self) -> u32 {
        Sort::ALL
            .iter()
            .position(|&s| s == self)
            .expect("every Sort is in Sort::ALL") as u32
    }

    pub fn parse(s: &str) -> Option<Self> {
        Sort::ALL.iter().copied().find(|so| so.surface() == s)
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.surface())
    }
}

/// The determinative separator between a slot and its sort classifier
/// (`x7·PLACE`) — the U+00B7 MIDDLE DOT, matching ADR 092-v2's notation. A
/// distinctive reserved joining glyph the definitive tokenizer treats
/// atomically; byte-fallback BPE handles the two UTF-8 bytes transparently.
pub const DETERMINATIVE_SEP: char = '·';

/// The maximum number of distinct positional slots a single example may bind
/// (Q1). Bounds the SLOT block of the governed region. A depth-8 chain touches
/// at most ~9 entities, so 32 is comfortable headroom for slice 1's corpora.
pub const MAX_SLOTS: u32 = 32;

/// A positional entity slot (Q1): `x1..xN`, **entity-agnostic**, bound to a
/// concrete SymbolId only at the boundary. The corpus renders slots, never
/// global entity ids — this is what decouples the tokenizer from any KG
/// snapshot (the sovereignty/relocatability reason Q1 supersedes v1's
/// entity-words).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Slot(pub u32);

impl Slot {
    /// Render the slot, optionally with its sort determinative (`x7·PLACE`
    /// typed, `x7` bare). The `typed` flag is the Q2 ablation knob threaded
    /// from the corpus config.
    pub fn render(self, sort: Option<Sort>) -> String {
        match sort {
            Some(s) => format!("x{}{}{}", self.0, DETERMINATIVE_SEP, s.surface()),
            None => format!("x{}", self.0),
        }
    }

    /// Parse a rendered slot token back into `(Slot, Option<Sort>)` (exact
    /// inverse of [`Slot::render`]). Total: `None` for anything that isn't
    /// `x<digits>` optionally followed by `·<SORT>`. The replayer relies on
    /// this to read a model's emitted slot text back into structure.
    pub fn parse(s: &str) -> Option<(Slot, Option<Sort>)> {
        let s = s.trim();
        let (slot_part, sort) = match s.split_once(DETERMINATIVE_SEP) {
            Some((slot_part, sort_part)) => (slot_part, Some(Sort::parse(sort_part)?)),
            None => (s, None),
        };
        let digits = slot_part.strip_prefix('x')?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let n: u32 = digits.parse().ok()?;
        Some((Slot(n), sort))
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn op_surface_round_trips_and_is_distinct() {
        let mut seen = HashSet::new();
        for op in Op::ALL {
            assert!(seen.insert(op.surface()), "duplicate surface: {op:?}");
            assert_eq!(Op::parse(op.surface()), Some(op));
        }
        assert_eq!(Op::parse("not_an_op"), None);
    }

    #[test]
    fn reserved_ids_are_contiguous_and_distinct() {
        // The tokenizer↔renderer lockstep invariant, checkable now (Q6).
        let ids: Vec<u32> = Op::ALL.iter().map(|o| o.reserved_id()).collect();
        assert_eq!(ids, (0..Op::ALL.len() as u32).collect::<Vec<_>>());
        let truth_ids: Vec<u32> = Truth::ALL.iter().map(|t| t.reserved_id()).collect();
        assert_eq!(truth_ids, (0..Truth::ALL.len() as u32).collect::<Vec<_>>());
        let sort_ids: Vec<u32> = Sort::ALL.iter().map(|s| s.reserved_id()).collect();
        assert_eq!(sort_ids, (0..Sort::ALL.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn live_and_dark_partition_the_operator_core() {
        let dark: Vec<Op> = Op::ALL.iter().copied().filter(|o| !o.is_live()).collect();
        assert_eq!(
            dark,
            vec![
                Op::IsA,
                Op::PartOf,
                Op::Causes,
                Op::Before,
                Op::After,
                Op::Logogram
            ],
            "the reserved-but-dark set must be exactly the structural relations + logogram"
        );
        // Every live op is exercised by a slice-1 renderer path; that is
        // enforced end-to-end by the `no_gold_trace_contains_a_dark_op` corpus
        // test, not here (this only fixes the partition).
    }

    #[test]
    fn truth_negation_is_involutive_on_two_valued() {
        assert_eq!(Truth::Yes.negate(), Truth::No);
        assert_eq!(Truth::No.negate(), Truth::Yes);
        assert_eq!(Truth::Yes.negate().negate(), Truth::Yes);
        assert_eq!(Truth::Gap.negate(), Truth::Gap);
    }

    #[test]
    fn slot_render_parse_round_trips_typed_and_bare() {
        for n in [0u32, 1, 7, 31] {
            let slot = Slot(n);
            assert_eq!(Slot::parse(&slot.render(None)), Some((slot, None)));
            for sort in Sort::ALL {
                let rendered = slot.render(Some(sort));
                assert_eq!(Slot::parse(&rendered), Some((slot, Some(sort))));
                // The typed form is exactly slot + '·' + sort.
                assert_eq!(rendered, format!("x{n}·{}", sort.surface()));
            }
        }
    }

    #[test]
    fn slot_parse_rejects_garbage() {
        assert_eq!(Slot::parse(""), None);
        assert_eq!(Slot::parse("x"), None);
        assert_eq!(Slot::parse("y7"), None);
        assert_eq!(Slot::parse("x7a"), None);
        assert_eq!(Slot::parse("x7·NOTASORT"), None);
        assert_eq!(Slot::parse("7"), None);
    }

    #[test]
    fn gap_literal_agrees_with_module_constant() {
        assert_eq!(Truth::Gap.surface(), super::super::GAP_VALUE);
    }
}
