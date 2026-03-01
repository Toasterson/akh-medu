# ADR 018 — Structured Email Extraction Design

**Date**: 2026-02-21
**Status**: Accepted
**Phase**: 13d

## Context

Phase 13c delivers triaged, non-spam emails. Downstream features (calendar 13f,
tasks 13e, preference learning 13g) need structured data extracted from email
content: dates, tracking numbers, URLs, phone numbers, and action items.

## Decision

### 1. Regex for structured patterns, grammar for NLP

Regex handles fixed-format patterns (ISO dates `2026-02-21`, tracking numbers
`1Z...`, URLs, phone numbers, email addresses) with high confidence. The grammar
framework (`grammar::detect::detect_language()`) handles relative dates ("tomorrow",
"завтра", "demain") and action items ("please send...", "veuillez...") where
multi-language support and context matter.

**Rationale**: Playing to each tool's strength. Regex is fast and precise for
known formats. Grammar infrastructure already supports 5 languages and provides
canonical forms.

### 2. Canonical temporal forms

Relative dates normalize to language-independent canonical forms:
`temporal:tomorrow`, `temporal:next-monday`, `temporal:in-3-days`. This enables
downstream temporal resolution without knowing the source language.

**Rationale**: Calendar integration (Phase 13f) should not care whether the user
received "demain" or "завтра" — both mean "tomorrow".

### 3. Compartment-scoped persistence

Extraction triples stored in correspondent-scoped microtheories that specialize
account-scoped microtheories. Pattern follows Phase 12d interlocutor design.

**Rationale**: Natural query scoping ("what did Alice send me tracking numbers
for?"), good redb locality, and inheritance lets account-level queries aggregate
across correspondents.

### 4. Goal specs, not goal creation

`action_items_to_goals()` returns `ActionItemGoalSpec` structs; it does NOT call
`create_goal`. The agent decides whether to act.

**Rationale**: Separation of concerns. Extraction should not have side effects on
agent state. The agent's OODA loop or operator can review and approve.

### 5. FedEx context gating

12-22 digit FedEx tracking numbers are too broad as a regex. Context keywords
("tracking", "shipment", "delivery", "FedEx", "package") within 100 chars raise
confidence from 0.5 to 0.9.

**Rationale**: Without context gating, invoice numbers, account numbers, and phone
numbers would false-positive as FedEx tracking numbers.

### 6. Deduplication by normalized form

If the same URL appears in subject and body, keep only the highest-confidence
instance. Dedup key is `(kind, normalized)`.

**Rationale**: Prevents duplicate KG triples and inflated counts.

## Consequences

- Email extraction is purely synchronous and feature-gated (`--features email`)
- Multi-language support covers EN, RU, FR, ES, AR for temporal and action patterns
- `regex` crate added as optional dependency (standard, well-maintained)
- Provenance tag 52 (`EmailExtracted`) added to the ledger
- Downstream phases (13e-13i) can query extraction results via KG predicates
