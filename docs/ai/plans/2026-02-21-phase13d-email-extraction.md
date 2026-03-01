# Phase 13d — Structured Email Extraction

**Status**: Complete
**Date**: 2026-02-21
**Depends on**: Phase 13c (email triage & priority)

## Goal

Extract structured information from triaged emails: dates, tracking numbers, URLs,
phone numbers, email addresses, and action items. Hybrid regex + grammar approach
for multi-language support. Compartment-scoped KG persistence.

## Design

### Regex + Grammar Hybrid

- **Regex** handles structured patterns where format is fixed (ISO dates, tracking
  numbers, URLs, phone numbers, email addresses).
- **Grammar framework** handles natural language extraction (relative dates, action
  items) where multi-language support matters.

### Compartment-Scoped Persistence

Extraction triples stored in microtheories following the interlocutor pattern:
- Account MT: `mt:email:account:{email}` (parent)
- Correspondent MT: `mt:email:correspondent:{sender}` (specializes account)

### Goal Specs, Not Goal Creation

`action_items_to_goals()` produces `ActionItemGoalSpec` structs. The agent
decides whether to actually create goals. Separation of concerns.

## Implementation

### New: `src/email/extract.rs`

- Types: `ExtractedItemKind`, `SourceField`, `ExtractedItem`, `ExtractionResult`
- Predicates: `ExtractionPredicates` (8 relations in `extract:` namespace)
- Scoping: `ExtractionScope`, `ensure_extraction_scope()`
- Goals: `ActionItemGoalSpec`, `action_items_to_goals()`
- 10 regex patterns via `LazyLock<Regex>`
- Grammar extractors: `extract_temporal_via_grammar()`, `extract_actions_via_grammar()`
- Pipeline: `extract_all()` — regex + grammar, deduplication
- Storage: `store_extractions()` — compartment-scoped KG triples
- Provenance: `record_extraction_provenance()` — `DerivationKind::EmailExtracted` (tag 52)
- Quick predicates: `has_action_items()`, `has_calendar_event()`, `has_shipment_info()`
- ~26 unit tests

### Modified

- `Cargo.toml`: added `regex` optional dep, included in `email` feature
- `src/email/mod.rs`: added `pub mod extract` + re-exports
- `src/provenance.rs`: added `EmailExtracted` variant (tag 52)
- `src/agent/explain.rs`: added `derivation_kind_prose` arm
- `src/main.rs`: added `format_derivation_kind` arm

## Verification

- `cargo build` — passes
- `cargo build --features email` — passes
- `cargo test --lib` — 1122 tests pass
- `cargo test --lib --features email` — 1270 tests pass (148 email tests)
- `cargo test --lib --features oxifed` — 1138 tests pass (no regressions)
