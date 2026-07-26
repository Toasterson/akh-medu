# Plan: Generic JSON-Schema → microtheory dataset import + real-graph benchmark

> **Date:** 2026-07-08 · **Status:** Designed (not yet built) · **ADR:** [090](../decisions/090-jsonschema-dataset-import.md)
> **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
> **Builds on:** [ADR 081](../decisions/081-akh-memory-schema-typing.md) — schema packs (`src/schema/`, built)

## Thesis

**A JSON Schema gives you the shape; a small mapping manifest gives you the meaning. Together
they turn any published structured dataset into a typed microtheory (ADR 081) plus instance
truth — with zero bespoke translation code per dataset.**

The worked example is the Marble Skill Taxonomy (`../../../os-taxonomy/`): a verified 1,590-node
/ 3,221-edge prerequisite DAG with JSON Schemas and per-edge provenance. Importing it yields
both a reusable capability *and* a real multi-hop reasoning benchmark with ground truth.

## Why now

- ADR 081 shipped the schema-pack pipeline (`compile`/`install`/immutable-guard) but only a
  bespoke front-end; §R6 asked for reusable, non-bespoke schemas.
- A real, clean, license-clear dataset with JSON Schemas just landed next to the repo.
- The [harness-value experiment](../../../.claude/projects/-home-toasty-ws-agi-akh-medu/memory/project_harness_value_experiment.md)
  and Phase-41f scaling law both need a **real** multi-hop KG with ground truth; their current
  graphs are synthetic/toy. This import produces one for free.

## Requirements

- **R1 — Reuse ADR 081 unchanged.** No new blob format, registry semantics, or install path.
  The importer emits a `SchemaPackSource`; everything downstream is free.
- **R2 — Dataset-agnostic manifest.** The semantics layer must work for future datasets, not
  just Marble. Marble is the first worked example, not the only shape.
- **R3 — Two layers.** Schema→pack (types, shared/immutable) and data→triples (truth, own
  scope), mirroring ADR 081's permissive-truth + typed-projection split.
- **R4 — Faithful graph.** Prerequisite edges must land as KG *edges* (not flat blobs), with
  `strength`→confidence and `reason`→provenance. Losing the graph defeats the purpose.
- **R5 — Prose out of the typed core.** Descriptions/evidence/prompts (CC-BY-SA) attach as
  attributes/note bodies, never as typed-core triples (ADR 081 discipline).
- **R6 — License survives.** ODbL/CC-BY-SA attribution + per-record provenance are recorded.
- **R7 — Dry-run first.** A wrong manifest silently mis-types; the importer must validate
  pointers and emit an inspectable report before writing.

## The mapping manifest (v1 format)

A dataset-agnostic JSON artifact, published beside the schema. Sketch:

```jsonc
{
  "dataset":  { "name": "marble-skill-taxonomy", "version": "v1",
                "source_url": "https://github.com/withmarbleapp/os-taxonomy",
                "license": "ODbL-1.0 AND CC-BY-SA-4.0",
                "attribution": "Marble Skill Taxonomy (v1) · © Generative Spark, Inc. …" },

  "pack":     { "name": "marble-curriculum", "version": "1.0.0", "scope": "own",
                "assumes": ["CreativeWork", "Thing"] },   // anchors from schema-org-core

  "entities": [
    { "type": "Topic", "is_a": "CreativeWork",
      "schema": "schema/topics.schema.json",
      "pointer": "#/properties/topics/items",             // which subschema is the node
      "data":   "data/topics.json", "collection": "#/topics",
      "id_field": "id",
      "subtype_from": "type",                             // enum → is_a subtypes
      "properties": { "include": ["subject","domain","ageRangeStart","ageRangeEnd","centrality"],
                      "prose":   ["description","evidence","assessmentPrompt"] },  // → note body
      "value_type_overrides": { "centrality": "number" } },

    { "type": "Standard", "is_a": "Thing",
      "schema": "schema/curriculum-standards.schema.json", "pointer": "…",
      "data": "data/curriculum-standards.json", "id_field": "code" }
  ],

  "relations": [
    { "predicate": "dependsOn", "arg1_type": "Topic", "arg2_type": "Topic",
      "data": "data/dependencies.json", "collection": "#/dependencies",
      "subject_field": "topicId", "object_field": "prerequisiteId",
      "confidence_from": { "field": "strength", "map": { "hard": 0.95, "soft": 0.6 } },
      "provenance_field": "reason" },

    { "predicate": "alignsWith", "arg1_type": "Topic", "arg2_type": "Standard",
      "from_entity": "Topic", "list_field": "standards" }   // inline FK list on the node
  ]
}
```

Manifest is validated against the schema (every pointer resolves; every `subject_field`/
`object_field`/`id_field` exists; enum values for `subtype_from` are enumerable).

## Slices (implementation, when approved)

Design-only for now; this is the intended breakdown.

- **S1 — Manifest model + validator** (`src/schema/import/manifest.rs`): serde types for the
  format above; validation resolving JSON Pointers against the referenced schemas with rich
  miette diagnostics (`akh::schema::import::*`).
- **S2 — `jsonschema_to_source`** (`src/schema/import/jsonschema.rs`): lower `(schema, manifest)`
  → `SchemaPackSource` per the ADR 090 structural table. Unit-test that
  `compile(source_of(marble))` yields the expected `Topic`/subtype/`dependsOn` triples.
- **S3 — Instance ingest** (`src/schema/import/instances.rs`): stream `data/*.json`, emit
  entity + edge triples under a dataset compartment (`mt:dataset/<name>@<ver>`), attach prose
  as attributes/library docs, write provenance (needs `DerivationKind::DatasetImported` — small
  additive variant, or `Extracted` with a source tag; **decide at build**). `strength`→confidence
  via Phase-17/18 path.
- **S4 — Dry-run report + CLI**: `akh schema import --schema … --manifest … [--out pack.json]`
  (S1+S2, offline, no engine) and `akh ingest dataset --manifest … [--dry-run]` (S3, needs an
  engine/workspace). Dry-run prints types, relations, counts, and sample triples; writes nothing.
- **S5 — Marble manifests** committed as the worked example (`data/dataset-manifests/marble-*.json`)
  + a golden test that imports the real files and asserts node/edge counts (1,590 / 3,221) and
  DAG acyclicity post-import.
- **S6 — Real-graph benchmark** (`src/thinking/bench.rs` extension or `examples/`): reusing
  `BenchTask`/`BenchRun`, generate tasks from the imported DAG — pick topics at hop-depth *d* from
  a target prerequisite, `expected` = that ancestor — and sweep `STEP_BUDGETS`. Produces a
  real-graph counterpart to the Phase-41f synthetic curve and a clean input for the harness-value
  experiment. CSV to `target/bench-results/`.

## Marble worked example (concrete mapping)

- `topics.json` → `Topic is_a CreativeWork`; `type` enum → 5 subtypes
  (`CONCEPTUAL`/`PROCEDURAL`/`REPRESENTATIONAL`/`LANGUAGE`/`META`) `is_a Topic`; scalar props
  `subject`/`domain`/`ageRangeStart`/`ageRangeEnd`/`centrality`; prose
  `description`/`evidence`/`assessmentPrompt` → note body/attributes.
- `dependencies.json` → 3,221 `(topic, dependsOn, prerequisite)` edges; `strength`→confidence
  (hard 0.95 / soft 0.6); `reason`→provenance note.
- `standards[]` on each topic → `(topic, alignsWith, Standard)`; `curriculum-standards.json` →
  `Standard` entities.
- `clusters.json` → optional; parent-friendly `summary` prose → note bodies keyed by
  (subject, domain, ageBand), not typed-core.

## Testing

- Manifest validator: pointer-resolution failures, missing fields, non-enumerable
  `subtype_from` → correct diagnostics.
- `jsonschema_to_source`: golden triples for the Marble schema (mirrors
  `compile::tests::compiles_expected_triples`).
- Instance ingest: counts (1,590 topics, 3,221 edges, 0 dangling, 0 cycles) on the real files;
  `strength`/`reason` land as confidence/provenance; prose absent from typed-core compartment.
- Benchmark: deterministic task generation (seeded), solvable-depth vs budget curve emitted.

## Out of scope

- JSON-LD `@context` as an alternate manifest source (future; ADR 090 alternatives).
- Registry admission of the Marble pack to the *shared* core (stays `own`/frontier unless a
  human promotes it — ADR 081 §7/§8).
- Turtle/YAML schema front-ends (orthogonal ADR 081 extension points).
- Re-import / diff-based supersede of an updated dataset version (ADR 081 §12 TMS path — later).

## References

- [ADR 090](../decisions/090-jsonschema-dataset-import.md) (this design).
- [ADR 081](../decisions/081-akh-memory-schema-typing.md) + [plan](2026-06-30-akh-memory-schema.md) + `src/schema/`.
- Phase-41f: [scaling-law report](../2026-07-05-phase41f-scaling-law.md) + `src/thinking/bench.rs`.
- Marble Skill Taxonomy v1: `github.com/withmarbleapp/os-taxonomy`.
