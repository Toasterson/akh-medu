//! Phase 51 P4 slice 1 — the propositional + rule-based reasoning arms, in the
//! High-Grug **v2** register (ADR 092-v2).
//!
//! Where the sibling P3-R corpus ([`super::corpus`]) walks a functional-relation
//! graph, these arms exercise *deductive* reasoning: boolean evaluation of a
//! ground formula (propositional arm) and universally-quantified rule
//! application by modus ponens (rule arm). Both render the shared reserved-token
//! register ([`super::register`]) and both are scored by a
//! `select/bind/fire/chain`-decomposed replayer ([`replay`]) so partial
//! capability is visible at 11M (Q5's false-abandon guard).

pub mod world;
// corpus + replay land in the same slice, added next.
