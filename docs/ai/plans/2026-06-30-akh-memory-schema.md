# Plan: Akh Memory & Schema-Typing Subsystem

> **Date:** 2026-06-30 · **Status:** Designed (not yet built) · **ADR:** [081](../decisions/081-akh-memory-schema-typing.md)
> **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
> **Work breakdown:** [2026-06-30-akh-memory-schema-issues.md](2026-06-30-akh-memory-schema-issues.md)

## Thesis

**Obsidian on the write side. Orbit on the read side. An LLM Librarian as the projector between them. A human-gated schema-pack registry as the type system.**

The KG today is a simple triple store tuned for symbolic reasoning. LLMs — especially the small local ones — produce unstructured prose. We need a memory that lets cloud and local models build a **shared** and **own** memory they can work against *consistently*, where a small model with weak world-knowledge but decent behaviour can lean on the memory as its knowledge substrate, and where long-horizon work survives without context compaction.

## Grounding

This design was developed by grounding against two reference systems that sit at opposite ends of two axes:

| Axis | **GitLab Orbit** (the *new* "context layer for AI agents") | **Obsidian / Karpathy LLM-Wiki** |
|---|---|---|
| Schema rigidity | Closed, enumerated, immutable; 24 node types / 6 domains; introspectable `get_graph_schema` | Open, emergent; typed links are an overlay synced to frontmatter |
| Who authors | Mechanical indexer; the LLM *queries* a projected read-model | The LLM *authors* notes and maintains coherence |

GitLab Orbit is **not** the old `orbit.love` community CRM — it is a property graph **projected** from GitLab's real data, which the LLM reads but does not write.

Akh takes **Obsidian's permissive substrate** (notes, an open frontier, never block a write) and **Orbit's typed-core discipline** (globally-unique immutable types, an introspectable schema, a small sacred core), joined by a **promotion gate**. The **Librarian** is the bridge — the LLM-as-indexer that turns Obsidian-style notes into an Orbit-style typed projection.

## Requirements

- **R1 — Shared + own.** Cloud and local LLMs maintain a *shared* consistent memory *and* private per-model memory.
- **R2 — Small-LLM leverage.** Small models have behaviour but weak knowledge; the memory must *be* the knowledge substrate they lean on.
- **R3 — Bidirectional schema.** Schema assists note-writing *and* is captured back out of notes/transcripts.
- **R4 — Context-manager sink.** Feeds Passive + MemGPT (`VirtualContextManager`) to hold working context <100K and enable long-horizon work without compaction.
- **R5 — Two modes.** Typed store + permissive store; making *both* permissive weakens each for its purpose.
- **R6 — Reusable schemas**, not bespoke-per-domain.

## The twelve locked decisions

### Architecture
1. **One design.** Schema-typing and note-taking are inseparable — the note-taking pipeline's *output is the schema*, and how rigidly the schema is typed determines how much the LLM must conform vs. free-write. Orbit and Obsidian each chose a coherent *pair*; the failure mode is mismatched halves.
2. **Permissive truth + typed projection, crystallizing at runtime.** Everything lands in a permissive truth store (notes, transcripts, observations — low bar, small models can always write). The typed store is a **read-model projected** from it (the GitLab-Orbit pattern, and our own [ADR 057 `codex_kg`](../decisions/057-code-understanding-kg.md) / [ADR 072](../decisions/072-normalized-akhomed-store.md)). **Build-time** starts as projection; a **typed core crystallizes at runtime** as library Akhs (Seshat, Thoth, …) do the projection work. Provenance on both sides. The **Akh** is the unit of ownership — a workspace = its model + KG + notes + capabilities + behaviour.

### Units
3. **Notes + episodes (dual).** A **Note** = the library `Document` (vector-indexed prose, which was always intended to be a frontmatter-less note — the split exists only for the vector store) **+ frontmatter + embedded tasks** — an Obsidian-style concept-page. The **`EpisodicEntry`** is the temporal spine (past events *and* scheduled/recurring future events → projectable to calendar / task management). **Daily logs are data-maintenance staging, not memory**; their processed residue becomes episodic entries + note edits (log-lines like "Built X from git XY", "Deployed X on user request").

### Type system
4. **Layered borrow for the vocabulary.** Entity/note types ← curated **schema.org** `Thing` subset + Akh-native (Akh, Goal, Skill, Decision); frontmatter value types ← Obsidian's set (text / list / number / date / datetime / checkbox / link); relation/wikilink types ← the *existing* predicate namespaces (`is-a`, `part-of`, `agent:`, `ctx:`, `doc:`, the `ElicitPass` relations). Seeded build-time, extended run-time. **schema.org is chosen because small LLMs already emit it** — anchoring on a known vocabulary turns "invent a schema" (which small models fail at) into "recall a vocabulary" (which they do well). It lives at the human/LLM boundary and projects into VSA/symbolic internally, consistent with the "human language at the boundary only" principle, and is FLOSS-clean (terms, not a dependency).
5. **Type semantics: global, immutable, additive.** Type names are **globally unique and may never be redefined** (like a programming language / RDFS `subClassOf`). This makes conflict *impossible by construction* and **forces the core to stay small** — solving R5 structurally rather than by policy. Extension is **`is-a` subtyping only** (`CompanyEmployee is-a Employee is-a Person`, each adding its own property shape + validation). Properties are **additive** (everything is triples), so adding a subtype or attribute never mutates an existing projected view — growth is monotonic.
6. **Composition = layered typed microtheories.** A schema pack *is* a typed microtheory (a compartment declaring its types + `onto:` arity/shape constraints). An Akh's effective schema = the **import stack** resolved via `ctx:specializes` (seeded-core ← installed-pack ← the Akh's own private *frontier* MT). **Shared = registry microtheories; own = the Akh's frontier MT** — R1 expressed as microtheory scope, one consistency model. This reuses existing machinery: "install a schema at runtime, no new code" is what microtheories already do.

### Governance / registry
7. **Open frontier, curated core.** The LLM may use *any* term in note frontmatter/wikilinks (writes never blocked — R2); frontier terms live only in the permissive store. The **core** is curated and enforced. Frontier→core promotion is **gated by cross-akh consensus**; the Librarian **aliases synonyms** onto core terms, which is what actually keeps R5 satisfied. LLMs design the *frontier* freely; the shared *core* grows only by consensus + ratification.
8. **Schema is an installable artifact.** New memory *capability* arrives as schema, not as code/deploys/MCP-tools — schema packs are to memory what skillpacks are to behaviour. Two inflows: **operator pack-install** (non-LLM path; e.g. a community-curated "Orbit schema" from GitHub) and **Librarian bottom-up proposal**. A **human always confirms** admission to the shared registry — so operators (even non-developers of Akh-Medu) can curate types.
9. **Registry = a dedicated table holding native-triple byte blobs.** Source is authored/reviewed on GitHub in a portable form (Turtle / JSON-LD / YAML) and **compiled once, at the human-confirm step**, into a native-triple blob; per-akh install is then a dumb byte-fetch + bulk-insert-as-microtheory (no bespoke translation code). Row = `{id, version, name, scope, source_url, checksum, confirmed_by, blob}`. **Available vs. active**: the registry holds *available* packs; each Akh *activates* by importing (workspace membership scopes defaults). **Lazy convergence** (refresh on wake + at the start of each daily Librarian pass + an operator `reload-schema` command); `pg_notify` deferred — because immutable+additive types make distribution **monotonic**, lazy convergence is always safe. **Push-to-workspace-default, pull-elsewhere.**

### Engine
10. **Tiered authoring.** The small local model (gemma4) writes in real time, **append-only**, emitting the type-guessing `WorkingMemoryEntry` salience markers it already produces (Observation/Decision/Inference — *appends*, not judgments; **no `take_note` tool exists**, capture is structural). **gpt-oss 120B is the daily Librarian (Seshat)** — projector / extractor / promoter, cheap via the Ollama subscription, run once per day as a batch pass; it **complements** (does not replace) the Phase-11i VSA sleep cycle.

### Read path
11. **core / recall / archival, tier-aware.** Core (always-resident, bounded, typed) = identity + prefs + active goals (`SafetyEnforced` pins) + active-note **frontmatter only** + a `get_graph_schema` digest. Recall = recent episodic + recently-touched frontmatter. Archival = note **bodies** (vector / `archival_search`) + the typed KG (an Orbit-style `query_graph`). **Bodies are never resident → the footprint is horizon-independent → there is nothing to compact.** Small models are append-only (no core self-edit tools); capable cloud models get the MemGPT self-edit tools. Passive + Virtual share one substrate.

### Evolution
12. **Diff-based supersede + weighted multi-akh conflict.** Notes are versioned (append-only edits); each projection tags triples with the note-version provenance. On edit, the Librarian **diffs** old vs new projection and retracts only the *changed* triples via TMS (`was_retracted`, Phase 44g), preserving history in provenance + note versions ("strong information evolution"). This needs a **stable triple-identity hash**. Multi-akh conflict in the shared core is represented as a **Dempster-Shafer belief interval weighted by source reliability** (Phase 17 + 18) — never last-writer-wins.

**Tasks (the one exception to append-only):** a structured **task tool** (OpenCode/`TodoWrite`-style: create/update/complete/list) the LLM drives directly, **grammar-constrained + schema-validated** (reliability comes from the tool schema, not the model's judgment — R2-safe; OpenCode shows agents do this well). The **structured task store is the source of truth** (markdown checkboxes are a rendered view); **no parser**. Lifecycle in the handler: `complete` → past `EpisodicEntry` + spawn recurrence; `create` → episodic future-spine / calendar / KG commitment.

**Migration (no flag-day):** legacy triples are grandfathered as *frontier* (the system is already permissive); the existing relation namespaces are front-loaded into the core vocabulary at install; entity typing stays lazy (the Librarian pass is the migration engine). An opt-in, non-LLM **schema-coverage report** shows typed-vs-frontier proportions.

## Open work / de-risk order

1. **★ R2 is an empirical bet — de-risk first.** Does gemma4-12b actually emit usable schema.org frontmatter *and* drive the task tool reliably? Cheap probe with `examples/agentic_flow.rs` + `AKH_AGENT_DEBUG=1` before building anything. If it can't, the tiered append-only premise needs rework.
2. **Stable triple-identity hash** (gates diff-supersede).
3. **Property-shape validation language** (SHACL-ish, extending `onto:` arity).
4. **Seed-core curation** — which schema.org subset + Akh-native types.
5. **Pack compiler** (portable source → native blob).
6. **Librarian pipeline** — adapt the Phase-32 `KnowledgeElicitor` to the gpt-oss 120B backend.
7. **Context-manager wiring** to the real Phase 44h/44i `PassiveManager` / `VirtualContextManager`.

## Phasing

See the [work-item breakdown](2026-06-30-akh-memory-schema-issues.md) — a parent work item + `D0` (this plan + ADR) + two prefactors + the `S0` feasibility spike + nine vertical-slice (tracer-bullet) issues, dependency-ordered, mapped to Anima `current_phase` triage tags.

## References

- ADR [081](../decisions/081-akh-memory-schema-typing.md) — this decision.
- Builds on: [ADR 057](../decisions/057-code-understanding-kg.md) (relational truth + projected read-model), [ADR 072](../decisions/072-normalized-akhomed-store.md), [ADR 065](../decisions/065-context-assembly-runtime.md) (context runtime), [ADR 070](../decisions/070-selectable-chat-backing.md) (selectable Candle/Ollama backing).
- Reuses: Phase 11i (VSA sleep), Phase 17 (Dempster-Shafer), Phase 18 (source reliability), Phase 32 (knowledge elicitation), Phase 44g/h/i (recall hygiene, Passive + Virtual context managers).
