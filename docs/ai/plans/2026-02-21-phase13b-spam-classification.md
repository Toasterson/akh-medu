# Phase 13b — OnlineHD Spam & Relevance Classification

**Status:** Complete
**Date:** 2026-02-21
**Phase:** 13b (Personal Assistant — Email)

## Goal

Add VSA-native spam/ham classification to the email subsystem using OnlineHD
prototype vectors, Bayesian token probability supplement (Robinson chi-square),
and deterministic rule overrides.

## Approach

1. **OnlineHD prototypes** — maintain a spam prototype and ham prototype HyperVec,
   updated via majority-vote bundling (`ops.bundle([existing, new])`). Natural
   diminishing-contribution property as more examples accumulate.

2. **Robinson chi-square** — Bayesian supplement using per-token spam/ham counts.
   Top-15 most informative tokens combined via Fisher's method. No external crate.

3. **Deterministic rules** — whitelist/blacklist domains and mailing-list detection
   (List-Id header) override all statistical signals.

4. **Combined scoring** — `0.7 * VSA_normalized + 0.3 * Bayesian` with thresholds
   at 0.55 (spam) and 0.45 (ham); uncertain zone between.

5. **Persistence** — bincode serialization to engine's durable store via
   `put_meta`/`get_meta`.

6. **Provenance** — `DerivationKind::SpamClassification` (tag 50) records
   message ID, decision, VSA confidence, and Bayesian score.

## New File

- `src/email/classify.rs` — ~600 lines implementation + ~200 lines tests (24 tests)

## Modified Files

- `src/email/mod.rs` — add `pub mod classify` and re-exports
- `src/provenance.rs` — add `SpamClassification` variant (tag 50)
- `src/agent/explain.rs` — add `derivation_kind_prose` arm
- `src/main.rs` — add `format_derivation_kind` arm
- `docs/ai/architecture.md` — update provenance count, add Phase 13b checklist
- `CLAUDE.md` — add Phase 13b completion checklist

## Key Patterns Reused

- VSA role-filler binding from `src/vsa/encode.rs`
- `encode_token`/`encode_label` for feature encoding
- `ops.similarity()` for prototype comparison
- `put_meta`/`get_meta` + bincode for persistence
- `engine.store_provenance()` for classification provenance
- EmailError pattern from `src/email/error.rs`

## Verification

- `cargo build` — passes
- `cargo build --features email` — passes
- `cargo test --lib` — 1122 tests pass
- `cargo test --lib --features email` — 1208 tests pass (24 new)
- `cargo test --lib --features oxifed` — 1138 tests pass (no regressions)
