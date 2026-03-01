# Phase 13c — Email Triage & Priority

**Status**: Complete
**Date**: 2026-02-21

## Goal

Add sender reputation tracking, four-feature importance scoring, VSA priority
prototypes, HEY-style screening queue, and routing logic on top of Phase 13b's
spam classifier.

## Design

### Triage Pipeline
1. Look up or create SenderStats, call `record_message()`
2. Check screening: `needs_screening()` -> return ScreeningQueue
3. Compute four scores: social + content + thread + label
4. Combine with weights: social*0.35 + content*0.25 + thread*0.20 + label*0.20
5. Compute VSA similarity to important/low-priority prototypes (if trained)
6. Route decision: >=0.65 -> Important, >=0.35 -> Feed, else PaperTrail

### Scoring Formulas

**Social (0.0-1.0):** 40% reply_rate (EMA) + 30% frequency (log2-based) + 20%
recency (7-day window) + 10% relationship weight (Colleague=0.9, Friend=0.8,
Service=0.3, Newsletter=0.1, Unknown=0.5)

**Content (0.0-1.0):** VSA similarity to important/low-priority prototypes,
normalized. 0.5 neutral when untrained.

**Thread (0.0-1.0):** 40% in_reply_to presence + 40% references depth (min(len,10)/10)
+ 20% has_references binary.

**Label (0.0-1.0):** Important=1.0, Feed=0.6, PaperTrail=0.3, Spam=0.0,
ScreeningQueue/None=0.5.

### Key Decisions
- Hybrid persistence: SenderStats HashMap via put_meta (authoritative), key
  metrics synced to KG via sync_sender_to_kg() for SPARQL queryability
- HEY screening returns EmailRoute::ScreeningQueue; operator approval via
  set_sender_routing() is handled by caller
- Standalone module: called explicitly, not auto-wired on inbound yet

## New Files
- `src/email/triage.rs` — TriageEngine, types, scoring, persistence, tests

## Modified Files
- `src/email/mod.rs` — add `pub mod triage` + re-exports
- `src/provenance.rs` — add `EmailTriaged` variant (tag 51)
- `src/agent/explain.rs` — add `derivation_kind_prose` arm
- `src/main.rs` — add `format_derivation_kind` arm
- `docs/ai/architecture.md` — update provenance count, add Phase 13c checklist
- `CLAUDE.md` — add Phase 13c completion checklist

## Verification
- `cargo build` passes
- `cargo build --features email` passes
- `cargo test --lib` — 1122 tests pass
- `cargo test --lib --features email` — 26 new triage tests + 86 existing email tests pass
- `cargo test --lib --features oxifed` — 1138 tests pass, no regressions
