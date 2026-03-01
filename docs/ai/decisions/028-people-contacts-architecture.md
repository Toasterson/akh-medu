# ADR-028: People & Contacts Architecture

**Date:** 2026-03-01
**Status:** Accepted

## Context

The agent had foundational social modeling (Phase 12d InterlocutorRegistry) but lacked unified contact management. People were generic `Entity` symbols with string-prefix conventions (`interlocutor:{id}`). Email senders (`SenderStats`) and calendar attendees were disconnected from the interlocutor model. There was no identity resolution (alice@work.com and alice@personal.com were separate interlocutors), no relationship graph, no per-person conversation memory, and no learned communication style profiles.

## Decision

Build a cohesive people-awareness layer in 5 sub-phases (25a-25e) on top of existing infrastructure, turning the agent from "knows about channels" into "knows about people."

### Sub-phase Structure

- **25a — Contact Entity & Identity Resolution**: Unified `Contact` struct, `ContactManager` with alias-based identity resolution and merge support.
- **25b — Relationship Graph**: `RelationshipGraph` with 10 relationship kinds, strength decay, VSA-based social circle detection.
- **25c — Per-Person Conversation Memory**: `PersonMemoryIndex` mapping contacts to episodic memories and discussed topics.
- **25d — Communication Style Profiles**: `StyleManager` with rule-based formality/verbosity heuristics, EMA tracking, VSA prototypes.
- **25e — Calendar-Contact Integration**: `ContactCalendar` stateless utility bridging events and contacts.

### Key Design Choices

1. **Alias-based identity resolution** — All known email addresses, handles, and names are aliases. `find_by_alias()` resolves any alias to a contact in O(1) via a reverse index. `merge()` consolidates two contacts and rewires all aliases.

2. **KG-backed with in-memory indexes** — Contacts, relationships, and style profiles are stored as KG triples with well-known predicates. In-memory HashMaps provide fast lookups. Persisted via bincode + `put_meta`.

3. **Stateless utility for calendar integration** — `ContactCalendar` has no state of its own; it queries CalendarManager and Engine directly. This avoids adding another persisted manager.

4. **Rule-based style heuristics (no LLM)** — Formality detection uses keyword markers (formal: "dear", "sincerely", etc.; casual: "hey", "lol", etc.) and sentence length. VSA prototypes learn what rules miss.

5. **EMA for style convergence** — Exponential moving averages (alpha=0.3) balance responsiveness and stability for formality, verbosity, and response time tracking.

### DerivationKind Tags

| Tag | Variant | Sub-phase |
|-----|---------|-----------|
| 72 | `ContactResolved` | 25a |
| 73 | `ContactMerged` | 25a |
| 74 | `RelationshipRecorded` | 25b |
| 75 | `StyleObserved` | 25d |
| 76 | `CalendarAttendeeLinked` | 25e |

### Agent Integration

Four new fields on Agent: `contact_manager`, `relationship_graph`, `person_memory`, `style_manager`. All follow the standard init/persist/restore lifecycle.

## Consequences

- Identity resolution enables the agent to recognize the same person across channels (email, chat, calendar).
- Relationship graph provides social context for priority reasoning and audience adaptation.
- Per-person memory enables "what did I discuss with Alice about Rust?" queries.
- Style profiles enable the agent to match communication tone to each contact.
- Calendar-contact integration enables "when is my next meeting with Bob?" queries.
