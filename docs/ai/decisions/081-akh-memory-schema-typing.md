# ADR 081: Akh Memory & Schema-Typing — permissive notes + projected immutable typed core

- **Status:** Implemented (2026-07-11) — all work items landed (P1, P2, S0-structural, slices 1–9): Note carrier, mechanical + Librarian projection, PG registry + admit/install/converge, is-a shapes + digest, task tool, context tiering, promotion + human-confirm + aliasing, TripleIdentity diff-supersede + TMS + DS conflict, grandfather classification + coverage report. Full lib suite + live-PG integration green. **Still open:** the S0 *empirical* bet (live small-model schema.org emission — structural seams shipped with deterministic fakes instead), cross-akh promotion consensus signal (no cross-akh telemetry channel yet), Heka vector search for archival (substring fallback documented), re-assert-under-alias + alias-aware coverage classification.
- **Date:** 2026-06-30
- **Plan:** [docs/ai/plans/2026-06-30-akh-memory-schema.md](../plans/2026-06-30-akh-memory-schema.md)
- **Work breakdown:** [docs/ai/plans/2026-06-30-akh-memory-schema-issues.md](../plans/2026-06-30-akh-memory-schema-issues.md)
- **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e`

## Context

The KG is a simple triple store tuned for symbolic reasoning. LLMs — particularly the small in-process models (gemma4-12b) — emit unstructured prose. We need cloud and local models to build a **shared** and **own** memory they can work against *consistently*: a small model with weak world-knowledge but adequate behaviour must be able to lean on the memory as its knowledge substrate, and long-horizon work must survive without context compaction (feeding the Passive and MemGPT context managers under ~100K tokens).

The design was grounded against two reference systems:

- **GitLab Orbit** — the *new* "context layer for AI agents" (not the old `orbit.love` CRM): a property graph with a **closed, immutable, introspectable schema** (24 node types / 6 domains, `get_graph_schema`) that is **projected** from real data and read — never written — by the LLM.
- **Obsidian / Karpathy LLM-Wiki** — an **emergent, loosely-typed** markdown + wikilink store the **LLM authors**, where typed links are an overlay synced to frontmatter.

A survey of the current code showed Akh is already ~90% Obsidian-shaped: only four nominal `SymbolKind` variants, an **open/emergent** predicate vocabulary (`resolve_or_create_relation` mints predicates on demand), advisory-only `onto:` arity constraints, and an LLM that already authors the graph (elicitation, extraction, sleep consolidation). What it lacks is Orbit's typed-core discipline.

## Decision

Adopt a **permissive truth store + a typed projection (read-model) that crystallizes an immutable typed core at runtime**, with schema as an installable, human-gated artifact. Concretely:

1. **One design.** Schema-typing and note-taking are a single design: capture's output *is* the schema.
2. **Permissive truth, projected typed read-model.** All capture is permissive (notes, transcripts, episodes). The typed KG is a projection — the GitLab-Orbit pattern, and our own [ADR 057](057-code-understanding-kg.md) `codex_kg` model. Build-time is projection; the typed core crystallizes at runtime. The **Akh** (workspace = model + KG + notes + capabilities + behaviour) is the unit of ownership.
3. **Units.** A **Note** = library `Document` + frontmatter + embedded tasks (Obsidian concept-page). `EpisodicEntry` is the temporal spine (past + scheduled/recurring future). Daily logs are staging, not memory.
4. **Vocabulary = layered borrow.** schema.org entity types + Obsidian value types + the existing relation namespaces. schema.org is chosen primarily because **small models already emit it** — recall, not invention.
5. **Type semantics = global, immutable, additive.** Type names are globally unique and never redefined; extension is `is-a` subtyping only; properties are additive. Conflict is impossible by construction and the core is forced small — **R5 (typed store must not rot into a permissive one) is solved structurally.**
6. **Composition = typed microtheories.** A schema pack is a typed microtheory; an Akh's effective schema is the `ctx:specializes` import stack. Shared = registry MTs; own = the Akh's private frontier MT.
7. **Open frontier / curated core, consensus-gated promotion.** Writes are never blocked (any frontier term allowed); the core grows only by cross-akh consensus + Librarian aliasing.
8. **Schema is an installable artifact**, with two inflows (operator pack-install on a non-LLM path; Librarian bottom-up proposal) and a **mandatory human confirm** for registry admission.
9. **Registry = a dedicated table of native-triple byte blobs**, compiled once from a portable GitHub-curated source at confirm time; per-akh install is a byte-fetch. Available-vs-active via workspace-scoped imports; lazy convergence (`pg_notify` deferred — monotonic types make lazy safe).
10. **Tiered authoring.** Small local model writes append-only with type-guessing salience markers (no `take_note` tool — capture is structural); **gpt-oss 120B is the daily Librarian** projector/extractor/promoter, complementing the VSA sleep cycle.
11. **core / recall / archival, tier-aware.** Bodies live only in archival, so the resident footprint is horizon-independent (no compaction). Small models are append-only; capable models get MemGPT self-edit tools.
12. **Diff-based supersede + Dempster-Shafer multi-akh conflict.** Edits diff against the prior projection and retract only changed triples via TMS, preserving history; shared-core conflict is a belief interval weighted by source reliability, never last-writer-wins.

**Tasks** are the deliberate exception to append-only: an LLM-driven structured task tool (grammar-constrained, schema-validated) whose store is authoritative — no markdown parser. **Migration** is no-flag-day: legacy triples are grandfathered as frontier; the Librarian pass is the migration engine.

## Alternatives considered

- **Pure Orbit (closed schema, mechanical indexer).** Rejected: kills the "LLMs design their own memory" goal, requires structured sources we don't have, and a small model can't query a rich 24-type schema reliably.
- **Pure Obsidian (fully emergent, no fixed core).** Rejected: violates R5 — with no immutable spine the typed store is just another permissive store, and shared multi-akh consistency (R1) has nothing to anchor on.
- **Two physically separate stores (typed + permissive).** Rejected in favour of one truth + a projection: two physical stores create a sync/divergence problem, worse across shared/own copies. The projection enforces typing at the boundary with a single source of truth.
- **A hand-pinned typed core authored directly, no projection.** Folded in only as the runtime-crystallized core; the bulk is derived, so there is no large hand-maintained ontology.
- **Model-decided capture (`take_note` tool) / markdown task parser.** Rejected: deciding "is this noteworthy?" and scraping structure from free text are exactly what small models / parsers do unreliably. Capture is structural; tasks get structure from a typed tool schema instead.
- **`pg_notify` live convergence now.** Deferred: immutable+additive types make schema distribution monotonic, so lazy convergence (wake / daily / manual reload) is always safe.

## Consequences

**Positive**

- R5 is structural (immutable global type names), not a policy that can erode.
- R2 leverages a vocabulary small models already know (schema.org) and a structured task tool, instead of betting on small-model judgment.
- R4 is achieved by construction: bodies never reside in context, so there is nothing to compact.
- New memory capability ships as a human-confirmed schema pack — no code, deploy, or new MCP tool.
- Heavily reuses existing machinery: microtheories (registry/composition), `onto:` constraints (shapes), TMS (supersede), Dempster-Shafer + source reliability (conflict), Phase-32 elicitation (Librarian), Phase-44 context managers.

**Negative / risks**

- The whole tiered-authoring premise rests on the **R2 empirical bet** — must be probed first.
- The typed projection can be **stale**; typing errors surface at read time, not write time.
- Diff-based supersede requires a **stable triple-identity hash** not present today.
- The daily Librarian adds an external `gpt-oss 120B` dependency (mitigated: cheap, batched once/day, degrades gracefully — projection can fall back to the trivial frontmatter map + VSA sleep).

## References

- Plan: [2026-06-30-akh-memory-schema.md](../plans/2026-06-30-akh-memory-schema.md)
- Related ADRs: [057](057-code-understanding-kg.md) (relational truth + projected read-model), [072](072-normalized-akhomed-store.md), [065](065-context-assembly-runtime.md), [070](070-selectable-chat-backing.md).
- Reused phases: 11i (VSA sleep), 17 (Dempster-Shafer), 18 (source reliability), 32 (knowledge elicitation), 44g/h/i (recall hygiene, Passive + Virtual context managers).
- External grounding: GitLab Orbit (`about.gitlab.com/gitlab-orbit/`, `docs.gitlab.com/orbit/`); Obsidian / Karpathy LLM-Wiki pattern.
