# Phase 25 — People & Contacts Implementation Plan

**Date:** 2026-03-01
**Status:** Complete
**ADR:** 028

## Summary

Build a cohesive people-awareness layer on top of the existing agent infrastructure in 5 sub-phases. Dependency: 25a first, then 25b-25e independently.

## Dependency Graph

```
25a (Contact Entity) <- 25b (Relationships)
25a (Contact Entity) <- 25c (Person Memory)
25a (Contact Entity) <- 25d (Style Profiles)
25a (Contact Entity) <- 25e (Calendar Integration)
```

## Implementation

### 25a — Contact Entity & Identity Resolution
- **File**: `src/agent/contact.rs` (~550 LOC)
- ContactError (4 variants), ContactPredicates (13 relations), Contact struct, ContactManager
- Alias-based identity resolution with O(1) reverse index
- Merge support: consolidates aliases, interlocutor links, sender links
- Provenance: ContactResolved (tag 72), ContactMerged (tag 73)

### 25b — Relationship Graph
- **File**: `src/agent/contact_rel.rs` (~500 LOC)
- RelationshipKind (10 variants), Relationship, RelationshipGraph
- Strength reinforcement and exponential decay
- VSA-based social circle detection via greedy clustering
- Provenance: RelationshipRecorded (tag 74)

### 25c — Per-Person Conversation Memory
- **File**: `src/agent/contact_memory.rs` (~250 LOC)
- PersonMemoryIndex: episodes_by_contact + topic_index
- Recall by person, by person+topic, shared context between contacts
- Rebuild from KG on restore

### 25d — Communication Style Profiles
- **File**: `src/agent/contact_style.rs` (~400 LOC)
- Rule-based formality heuristic (no LLM dependency)
- EMA tracking (alpha=0.3) for formality, verbosity, message length, response time
- VSA prototype vectors for style similarity search
- Provenance: StyleObserved (tag 75)

### 25e — Calendar-Contact Integration
- **File**: `src/agent/contact_calendar.rs` (~250 LOC)
- Stateless utility (no Agent field needed)
- Links events to contacts, co-attendance queries, meeting frequency, availability heatmap
- Provenance: CalendarAttendeeLinked (tag 76)

## Integration Changes

| File | Change |
|------|--------|
| `src/agent/mod.rs` | 5 new modules + re-exports |
| `src/agent/agent.rs` | 4 new fields (contact_manager, relationship_graph, person_memory, style_manager) with init/persist/restore |
| `src/agent/error.rs` | `Contact(#[from] ContactError)` variant |
| `src/provenance.rs` | 5 new DerivationKind variants (tags 72-76) |
| `src/agent/explain.rs` | 5 new match arms in derivation_kind_prose |

## Verification

- 33 new unit tests, all passing
- Clean build (no warnings in new code)
- Persist/restore round-trip verified for contact_manager, relationship_graph, style_manager
