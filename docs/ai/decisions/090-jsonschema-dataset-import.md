# ADR 090: Generic JSON-Schema → microtheory dataset import (structure + mapping manifest)

- **Status:** Accepted (design) — not yet implemented
- **Date:** 2026-07-08
- **Plan:** [docs/ai/plans/2026-07-08-jsonschema-dataset-import.md](../plans/2026-07-08-jsonschema-dataset-import.md)
- **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
- **Builds on:** [ADR 081](081-akh-memory-schema-typing.md) (schema packs = installable typed microtheories)

## Context

[ADR 081](081-akh-memory-schema-typing.md) established schema packs: a typed microtheory
shipped as an installable artifact, compiled once from a portable, GitHub-reviewable source
and installed per-akh as a byte-fetch + bulk-insert. The prefactor slice is built (`src/schema/`:
`source.rs` parser, `compile.rs` lowering to native-triple blobs, `install.rs` with the
immutable-global-type-name guard, `akh schema compile|show`). ADR 081 §R6 called for
**"reusable schemas, not bespoke-per-domain."**

The current front-end is a *bespoke* JSON dialect: a `SchemaPackSource` object of
`{pack, types, relations, assumes}` hand-authored to match akh-medu's own vocabulary. The
module's own doc-comment (`src/schema/source.rs`) anticipates the gap:

> The v1 front-end is a JSON document […]; Turtle/YAML front-ends can be added later behind
> [`parse_source`] without touching the compiled artifact format.

Meanwhile, third parties are now publishing high-quality structured knowledge graphs *with
their own JSON Schemas*. The motivating instance is the **Marble Skill Taxonomy v1**
(`withmarbleapp/os-taxonomy`, ODbL 1.0 + CC BY-SA 4.0): 1,590 curriculum micro-topics (graph
nodes) + 3,221 prerequisite edges (a verified DAG — 0 cycles, 0 dangling references, a `reason`
on every edge), each file carrying a JSON Schema (`schema/*.schema.json`). This is exactly the
shape ADR 081 wants to consume: typed nodes + typed edges + per-edge provenance + external
standard alignment.

We want to ingest such a dataset into an Akh **(a)** as a typed microtheory (the *types* — a
schema pack) and **(b)** as instance truth (the *records* — KG triples with provenance), while
**reusing the ADR 081 compile/install/immutable-guard machinery unchanged** and writing **no
bespoke translation code per dataset**.

## The load-bearing finding

**A JSON Schema is purely *structural*; it does not carry the *semantics* import needs.**
JSON Schema describes the *shape* of a document (this field is a `string`, that one is
`required`, this array holds objects) but it cannot express:

- **Which subschema is the entity.** In `topics.schema.json` the real graph node is
  `#/properties/topics/items`, not the wrapping `{version, topicCount, topics}` object.
- **Which field is identity.** `id` (`^mt_`) is the node's stable key; nothing in the schema
  marks it as such.
- **Which fields are relations vs. properties.** `dependencies.json`'s
  `{topicId, prerequisiteId, strength, reason}` is really an **edge type**
  (`topicId ──dependsOn──▶ prerequisiteId`, with `strength`/`reason` as edge qualifiers). JSON
  Schema sees four flat string fields and cannot know two of them are foreign keys into the
  node set.
- **How to anchor foreign types to the seed core.** Nothing says Marble's `Topic` should be an
  `is_a` of schema.org `CreativeWork` (ADR 081 §4 vocabulary anchoring).
- **How qualifiers map to engine semantics.** `strength: hard|soft` → edge confidence;
  `reason` → a provenance record. This is a modelling choice, not a shape.

Attempting a "pure JSON Schema, zero-config" importer would force us to *guess* these
semantics heuristically, and the dependencies-are-edges case is precisely where a heuristic
would get it wrong (it would import 3,221 four-attribute `Dependency` blobs instead of 3,221
graph edges — losing the entire prerequisite graph, which is the point of the dataset).

**Therefore a faithful generic importer is `JSON Schema (structure) + a mapping manifest
(semantics)`.** The manifest is small, declarative, and reusable, and it *is* the
community-sharing artifact: publish `dataset.mapping.json` beside the schema and anyone can turn
the dataset into typed Akh memory.

## Decision

Add a JSON-Schema import front-end and a companion **mapping manifest**, both feeding the
existing ADR 081 pipeline. Concretely:

1. **New authoring front-end, same artifact format.** Add
   `jsonschema_to_source(schema, manifest) -> SchemaPackSource`. Its output is an ordinary
   `SchemaPackSource`, so `compile` → `CompiledSchemaPack` → `install_pack` →
   `ctx:specializes` activation and the immutable-global-type-name guard are all reused
   **verbatim**. No new blob format, no new registry semantics.

2. **The mapping manifest is a first-class, dataset-agnostic, GitHub-reviewable artifact.**
   It carries exactly the semantics JSON Schema omits (entity pointer, identity field, relation
   collections, qualifier mappings, enum-as-subtype, schema.org anchor, license/attribution).
   It is *not* folded into `SchemaPackSource`: the pack source is the **type** artifact (shared,
   immutable, registry-admitted), whereas the manifest is **dataset-ingest config** (own scope,
   per-import). Different lifecycles ⇒ separate artifacts.

3. **Two-layer import, matching ADR 081's permissive-truth + typed-projection split.**
   - **Schema layer** (`schema + manifest → schema pack`): the *types*. Shared/immutable, goes
     through the human-confirm registry path unchanged.
   - **Instance layer** (`data + manifest → KG triples`): the *truth* — the actual records —
     landing in a **dataset compartment** (own scope / frontier), never the immutable core.

4. **Structural → vocabulary mapping is fixed and deterministic** (see table below). Value-type
   inference (`string`→text, `integer`/`number`→number, `boolean`→checkbox, `array`→list,
   `format:date`→date, `format:date-time`→datetime, `format:uri`/`$ref`→link) is overridable
   per field in the manifest.

5. **`enum` on a manifest-designated `subtype_from` field lowers to `is_a` subtypes** — the only
   ADR 081 extension mechanism. Marble's `type ∈ {CONCEPTUAL, PROCEDURAL, REPRESENTATIONAL,
   LANGUAGE, META}` becomes five `is_a Topic` subtype declarations, and each instance gets a
   `(mt_id, is-a, CONCEPTUAL)` edge. Enums *not* named as `subtype_from` stay text properties.

6. **Relations come only from manifest-declared edge collections.** For each, the manifest names
   the subject field, object field, predicate name, and how qualifier fields map: `strength` →
   edge confidence, `reason` → the provenance record's note. Foreign-key-into-node-set is a
   semantic assertion the operator makes, not an inference.

7. **Prose stays out of the typed core.** Consistent with ADR 081 (§source.rs drops
   `description` into artifact metadata, not triples), the importer does **not** lower
   `description` / `evidence` / `assessmentPrompt` / cluster `summary` into typed-core triples.
   These are content (CC BY-SA); they belong in note bodies / attributes / the vector library,
   attached to the entity but outside the immutable type projection.

8. **The human confirm is untouched.** The importer is an *authoring aid on the operator
   pack-install inflow* (ADR 081 §8) — it turns a third-party JSON Schema into a reviewable pack
   source so an operator need not hand-write the bespoke dialect. Registry admission still
   requires the ADR 081 human confirm; the importer never bypasses it.

9. **Provenance + license are recorded, not discarded.** Instance triples carry a provenance
   record naming the dataset, version, source URL, and license; the manifest's
   `dataset.attribution` string is stored so the ODbL/CC-BY-SA attribution obligation travels
   with the data. This needs a `DerivationKind::DatasetImported { dataset, version, record_id }`
   variant (or reuse of `Extracted` with a source tag — decided in the plan).

### Structural mapping (JSON Schema construct → ADR 081 pack vocabulary)

| JSON Schema | Lowered to | Native triples (via existing `compile`) |
|---|---|---|
| `object` at manifest `entity.pointer` | `TypeDef` (+ `is_a` anchor) | `(T, is-a, onto:Type)`, `(T, is-a, Anchor)` |
| property, JSON `string` | `PropertyDef{value_type: text}` | `(T, onto:property, p)`, `(p, onto:value-type, onto:value/text)` |
| property, `integer`/`number` | `…{number}` | `… onto:value/number` |
| property, `boolean` | `…{checkbox}` | `… onto:value/checkbox` |
| property, `array` | `…{list}` | `… onto:value/list` |
| property, `format: date` / `date-time` | `…{date}` / `…{datetime}` | `… onto:value/date` … |
| property, `format: uri` or `$ref` | `…{link}` | `… onto:value/link` |
| `required: [...]` | `PropertyDef.required = true` | `(T, onto:required, p)` |
| `enum` on `subtype_from` field | one `TypeDef{is_a: T}` per value | `(V, is-a, onto:Type)`, `(V, is-a, T)` |
| manifest `relation` collection | `RelationDef{arg1_type, arg2_type}` | `(r, is-a, relation)`, `(r, onto:arity, onto:arity/2)`, `(r, onto:arg1type, T)`, `(r, onto:arg2type, T)` |
| `description` | (metadata only — **not** lowered) | — |

### Instance mapping (record → triples, under a dataset compartment)

| Record | Triples |
|---|---|
| entity record | `(id, is-a, Type)` + `(id, is-a, <enum-subtype>)` + one property triple per scalar (link/list per shape) |
| edge record | `(subject_id, predicate, object_id)`; `strength` → confidence; `reason` → provenance note |
| standards link | `(id, alignsWith, <Standard>)` cross-reference (Standard is its own manifest entity or an assumed type) |
| prose fields | attached as attributes / note body (NOT typed-core triples) |

## Alternatives considered

- **Pure JSON Schema, zero manifest (heuristic semantics).** Rejected — see "load-bearing
  finding": identity and relation are semantic, not structural. It would silently mis-model the
  dependency graph as flat blobs.
- **A bespoke per-dataset importer** (the original Option 1 shape). Rejected by design intent
  (ADR 081 §R6) and by the user: it doesn't scale, produces no reusable community artifact, and
  re-implements the type-lowering that `compile` already does.
- **JSON-LD `@context` instead of a manifest.** JSON-LD is the "textbook" semantic layer, and we
  may accept it later as an *alternate* manifest source. Rejected as the v1 mechanism because
  (a) Marble — and most datasets — ship plain JSON + JSON Schema, not JSON-LD; (b) JSON-LD does
  not cleanly express edge reification with qualifiers over our binary-triple model; (c) a
  purpose-built manifest is smaller and matches ADR 081's portable-source ethos.
- **Fold the manifest into `SchemaPackSource`.** Rejected: conflates the shared/immutable *type*
  artifact with per-dataset *ingest* config. The pack source is registry-admitted and versioned
  as a type system; the manifest changes whenever you re-map a dataset. Separate lifecycles.
- **Import instances as one opaque JSON blob per record (no relation extraction).** Rejected:
  this is what loses the graph. The dataset's value *is* the edges.

## Consequences

**Positive**

- Reuses the entire ADR 081 machinery (compile, blob, install, immutable guard,
  `ctx:specializes`, `TripleIdentity`) — the new code is a front-end + an instance loader.
- Delivers ADR 081 §R6 ("reusable schemas, not bespoke-per-domain") concretely: schema +
  manifest = "any structured dataset → typed Akh memory."
- Turns the Marble taxonomy into (a) a typed microtheory and (b) a real 1,590-node / 3,221-edge
  prerequisite DAG in the KG — a genuine multi-hop reasoning substrate with ground truth, which
  is exactly what the harness-value experiment and the Phase-41f scaling-law work lacked (their
  graphs were synthetic or toy).
- License/attribution and per-edge provenance survive the import, keeping FLOSS-sovereignty
  obligations (ODbL share-alike, CC-BY-SA attribution) attached to the data.

**Negative / risks**

- The manifest is a new artifact operators must learn; a *wrong* manifest silently mis-types
  (typing errors surface at read time, per ADR 081's projection risk). Mitigation: the importer
  validates every manifest JSON Pointer against the schema and emits a **dry-run report**
  (types, relations, sample instance triples, dropped fields) before anything is written.
- Foreign type names (`Topic`, `CONCEPTUAL`) are not schema.org; they enter as a **scoped
  domain pack**, anchored to the seed core via `is_a`. This is sanctioned (ADR 081 import stack)
  but does grow the shared vocabulary — so a Marble-style pack should default to `own`/frontier
  scope unless deliberately promoted.
- Prose handling: descriptions/evidence are the CC-BY-SA layer and must be routed to note
  bodies/attributes, not the typed core — an easy thing to get wrong and pull prose into triples.
- Adds a `DerivationKind` variant (small, additive) and a new dataset compartment convention.

## References

- Builds on: [ADR 081](081-akh-memory-schema-typing.md) (schema packs), and its
  [plan](../plans/2026-06-30-akh-memory-schema.md) + `src/schema/`.
- Related: [ADR 057](057-code-understanding-kg.md) (relational truth + projected read-model),
  [ADR 072](072-normalized-akhomed-store.md).
- Reuses: Phase-41f benchmark harness (`src/thinking/bench.rs`) for the real-graph scaling
  benchmark; Phase-17/18 (Dempster-Shafer + source reliability) for `strength`→confidence.
- External: Marble Skill Taxonomy v1 — `github.com/withmarbleapp/os-taxonomy`
  (ODbL 1.0 database + CC BY-SA 4.0 content). Attribution: *Marble Skill Taxonomy (v1) ·
  © Generative Spark, Inc. (Marble) · https://withmarble.com*.
