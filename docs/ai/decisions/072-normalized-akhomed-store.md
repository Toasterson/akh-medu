# ADR 072 — Normalized akhomed-store schema (edge/child tables, not JSON lists)

> Date: 2026-06-20
> Status: Accepted (implemented)
> Related: ADR 047 (Seshat SeaORM precedent), ADR 049 / Phase 38 plan
>          (relocatable Akh — the schema this normalizes), CLAUDE.md
>          ("use strongly typed idiomatic rust… express the function in datatypes",
>          "never use raw SQL, use SeaORM entities")

## Context

Phase 38a–c shipped the `akhomed.*` / `workspaces.*` schema fast, and several
columns that are *relational* (lists of ids, ordered sub-records) were stored as
JSON arrays for expedience — even where the entity's own doc comment said the
intended type was an array or relation. The schema is days old and greenfield
(no released data), so this is the right moment to normalize before any Akh data
exists to migrate.

## Decision

Replace JSON list-columns with proper edge/child tables; keep JSON only for
genuinely polymorphic/opaque documents; keep BYTEA for VSA hypervectors.

**Normalized → relational tables**

| Was (JSON)              | Now (table)                                                        |
|-------------------------|-------------------------------------------------------------------|
| `goal.children`         | `akhomed_goal_hierarchy(akh_id, parent_id, child_id, position)`    |
| `goal.blocked_by`       | `akhomed_goal_dependencies(akh_id, goal_id, blocking_goal_id)`     |
| `learned_method.steps`  | `akhomed_learned_method_step(akh_id, method_id, step_index, tool, arg_pattern, expected_outcome)` |
| `provenance.inputs`     | `akhomed_provenance_input(provenance_id, position, input_symbol_id)` |
| `workspace.seed_packs`  | `workspaces_workspace_seed_pack(workspace_id, position, pack_name)`  |

Ordering is preserved by an explicit `position` / `step_index` column (so a
reload re-sorts to the original order regardless of row insertion order). Child
tables (not Postgres `BIGINT[]`/`TEXT[]`) are used so the schema stays portable
to SQLite/libSQL, matching the rest of the store.

**Kept as JSONB** (polymorphic or opaque, no relational query value):
`akh_state.psyche`, `akh_state.extra`, `goal.detail`, `provenance.derivation`,
`episode.payload`, `mutation_log.payload`, `effort_case.tool_usage`.

**Kept as BYTEA**: all `*_vector` columns (VSA hypervectors — opaque to SQL,
compared only by the in-process HNSW).

The snapshot writer/loader (`crates/akhomed-store/src/snapshots.rs`,
`src/snapshot.rs`) build and reconstruct these tables instead of the JSON
columns; new `replace_*`/`load_*` helpers follow the existing per-Akh
full-replace pattern (and two hand-written helpers for the
non-`akh_id`-keyed `provenance_input` / `workspace_seed_pack`).

## Consequences

- Goal hierarchies, dependency graphs, method steps, derivation inputs, and seed
  packs are now queryable relationally (e.g. "all children of goal X") and have
  typed columns rather than opaque blobs — matching the project's "express the
  function in datatypes" rule.
- Migrations were edited in place (greenfield). The change is invisible at the
  agent layer: `Goal.children`, `LearnedMethod.steps`, etc. are unchanged Rust
  types; only their persistence shape changed.
- Entities keep app-side referential integrity (no hard DB FK constraints), matching
  the existing store style and SQLite portability; new child/edge entities use the
  same minimal `Relation`/`ActiveModelBehavior` shape.

## Verification

- Round-trip unit tests: the five new entities, goals with children + blocked_by,
  and a learned method with steps all survive `replace → load` with order
  preserved (out-of-order input rows re-sorted on load).
- `cargo build` (default) + `cargo test -p akhomed-store` green; snapshot
  round-trip tests (`--features postgres`) 13 pass; `cargo test --lib` 2587 pass.
