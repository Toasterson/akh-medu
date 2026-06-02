# Phase 25 — People & Contacts

Status: **Complete**

Cohesive people-awareness layer: unified contact management with identity resolution,
relationship graphs with decay and social circles, per-person conversation memory,
communication style profiles with rule-based heuristics, and calendar-contact integration.

- **Implementation plan**: `docs/ai/plans/2026-03-01-phase25-people-contacts.md`
- **ADR**: `docs/ai/decisions/028-people-contacts-architecture.md`

## Phase 25a — Contact Entity & Identity Resolution

- [x] `ContactError` miette diagnostic enum (4 variants: NotFound, DuplicateAlias, SelfMerge, Engine)
- [x] `ContactPredicates` — 13 well-known relations in `contact:` namespace (is-person, has-alias, display-name, has-organization, has-phone, has-note, merged-from, linked-interlocutor, linked-sender, created-at, participated-in, discussed-topic, shared-context)
- [x] `Contact` struct with contact_id, symbol_id, display_name, aliases, interlocutor_ids, sender_addresses, organization, phones, notes, timestamps
- [x] `ContactManager` — HashMap + alias_index reverse index, create/find/resolve_or_create/merge/search/link/persist/restore
- [x] `AgentError::Contact` transparent variant
- [x] `DerivationKind::ContactResolved` (tag 72), `ContactMerged` (tag 73)
- [x] Agent field `contact_manager` with init/persist/restore lifecycle
- [x] 8 unit tests (create, alias lookup, case insensitive, duplicate error, merge, self-merge, resolve idempotency, persist/restore, search, link)

## Phase 25b — Relationship Graph

- [x] `RelationshipKind` enum (10 variants: Family, Friend, Colleague, Acquaintance, Mentor, Mentee, Manager, Report, Neighbor, Service)
- [x] `Relationship` struct with from/to, kind, strength, timestamps, symbol_ids
- [x] `RelationshipPredicates` — per-kind predicates + strength + social-circle
- [x] `RelRoleVectors` — 4 deterministic VSA role vectors (partner, kind, strength, recency)
- [x] `RelationshipGraph` — edges HashMap, adjacency index, VSA rel_vectors, add/remove/query/reinforce/decay/detect_circles/rewire_after_merge/persist/restore
- [x] `SocialCircle` struct with member_ids and cluster label
- [x] `DerivationKind::RelationshipRecorded` (tag 74)
- [x] Agent field `relationship_graph` with init/persist/restore lifecycle
- [x] 6 unit tests (add/query, kind filter, reinforce+decay, rewire after merge, persist/restore, empty circles)

## Phase 25c — Per-Person Conversation Memory

- [x] `PersonMemoryIndex` — episodes_by_contact, topic_index, max_per_contact
- [x] tag_episode, recall_with_person, recall_with_person_about, topics_discussed_with, shared_context_between, rebuild_from_kg
- [x] Agent field `person_memory` with persist/restore lifecycle
- [x] 6 unit tests (tag+recall, person+topic, shared_context, prune, rebuild, topics ranked)

## Phase 25d — Communication Style Profiles

- [x] `StylePredicates` — 6 relations (formality, verbosity, expertise-area, preferred-channel, response-pattern, style-observation)
- [x] `StyleRoleVectors` — 5 deterministic VSA role vectors
- [x] `CommunicationStyle` struct with EMA fields (formality, verbosity, avg_message_length, avg_response_time), expertise_areas, VSA prototype
- [x] `StyleObservation` input struct
- [x] `score_formality()` — rule-based keyword/contraction/emoticon/length heuristic
- [x] `score_verbosity()` — linear mapping 20-500 chars to 0.0-1.0
- [x] `StyleManager` — HashMap, EMA updates (alpha=0.3), observe/style_for/suggest/similar_style/decay/persist/restore
- [x] `DerivationKind::StyleObserved` (tag 75)
- [x] Agent field `style_manager` with init/persist/restore lifecycle
- [x] 6 unit tests (observe+retrieve, EMA convergence, formality heuristic, verbosity scoring, decay, persist/restore)

## Phase 25e — Calendar-Contact Integration

- [x] `CalendarContactPredicates` — has-attendee, organizer
- [x] `ContactCalendar` — stateless utility with add_attendee, set_organizer, events_with, shared_events, meeting_frequency, next_event_with, import_ical_attendees, availability_heatmap
- [x] `DerivationKind::CalendarAttendeeLinked` (tag 76)
- [x] 6 unit tests (add+query, shared events, frequency, import ical, heatmap, next_event)
