# Phase 14a+14b — Purpose-Driven Identity & Ritual of Awakening

**Status**: Complete
**Date**: 2026-02-22

## Summary

Implemented the purpose/identity parser (Phase 14a) and identity resolution with the Ritual of Awakening (Phase 14b). This gives the agent the ability to parse operator purpose declarations, resolve cultural/historical/fictional references, construct a personalized Psyche, and generate a self-name via culture-specific morpheme composition.

## New Modules

### `src/bootstrap/purpose.rs` (~340 lines)
- `BootstrapError` (4 variants) + `BootstrapResult<T>`
- `DreyfusLevel`, `EntityType`, `IdentityRef`, `PurposeModel`, `BootstrapIntent`
- 5 LazyLock regex patterns for identity extraction
- `parse_purpose()`, `classify_entity_type()`, `extract_competence()`
- 12 unit tests

### `src/bootstrap/identity.rs` (~720 lines)
- `IdentityError` (5 variants) + `IdentityResult<T>`
- `CultureOrigin`, `CharacterKnowledge`, `OceanProfile`, `ArchetypeProfile`
- `MorphemeTable`, `NameCandidate`, `RitualResult`
- Static tables: DOMAIN_TRAITS, TRAIT_ARCHETYPE, ARCHETYPE_OCEAN, ARCHETYPE_SHADOWS
- 4 culture morpheme tables (Egyptian, Greek, Norse, Latin)
- Multi-source resolution: static tables -> Wikidata -> Wikipedia
- Psyche construction with domain-augmented traits
- Ritual of Awakening with VSA-scored name generation
- 13 unit tests

## Cross-Cutting Changes

- `src/lib.rs`: Added `pub mod bootstrap;`
- `src/provenance.rs`: 2 new DerivationKind variants (tags 59-60)
- `src/agent/error.rs`: Bootstrap + Identity transparent variants
- `src/agent/nlp.rs`: `UserIntent::AwakenCommand` variant
- `src/agent/explain.rs`: 2 new `derivation_kind_prose()` arms
- `src/main.rs`: `Commands::Awaken` (Parse/Resolve/Status), headless handler, format_derivation_kind arms
- `src/tui/mod.rs`: AwakenCommand handler
- `Cargo.toml`: regex promoted from optional to always-on dependency
- `docs/ai/architecture.md`: Updated module count, provenance tag count, last-updated

## Verification

- `cargo build`: clean
- `cargo test`: 1,363 tests pass (27 new bootstrap tests)
- `cargo clippy`: no new warnings in bootstrap module
