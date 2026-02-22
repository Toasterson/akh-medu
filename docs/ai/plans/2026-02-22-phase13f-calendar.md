# Phase 13f — Calendar & Temporal Reasoning

- **Status**: Complete
- **Date**: 2026-02-22
- **Phase**: 13f

## Summary

Calendar event management with Allen interval algebra for temporal reasoning,
iCalendar import, CalDAV sync, scheduling conflict detection, and VSA temporal
pattern encoding.

## Implementation

### New File
- `src/agent/calendar.rs` — CalendarError, AllenRelation (13 variants), CalendarEvent,
  CalendarPredicates (19 predicates), CalendarRoleVectors (4 roles), CalendarManager
  (add/remove/get, detect_conflicts sweep-line, events_in_range, today/week_events,
  Allen relation computation, VSA temporal encoding, KG sync, provenance, persist/restore),
  import_ical (feature-gated), sync_caldav (feature-gated), 32 tests

### Modified Files
- `Cargo.toml` — icalendar + chrono optional deps, calendar feature
- `src/provenance.rs` — DerivationKind::CalendarEventManaged (tag 54)
- `src/agent/error.rs` — Calendar transparent variant
- `src/agent/mod.rs` — pub mod calendar + re-exports
- `src/agent/agent.rs` — calendar_manager field, init/resume/persist lifecycle, accessors
- `src/agent/nlp.rs` — UserIntent::CalCommand
- `src/agent/explain.rs` — derivation_kind_prose for tag 54
- `src/reason/mod.rs` — calendar_rules() (before-trans, cal-conflict)
- `src/main.rs` — CalAction enum, Commands::Cal, CLI handlers, format_derivation_kind arm
- `src/tui/mod.rs` — CalCommand handler

## Key Decisions

1. Allen algebra is pure — takes four u64 timestamps, no engine dependency
2. chrono stays at the boundary — internal storage is u64 UNIX seconds
3. Core module always compiles — only import_ical/sync_caldav are feature-gated
4. Reuses pim::Recurrence for recurring events
5. Sweep-line conflict detection — O(n log n) average case
