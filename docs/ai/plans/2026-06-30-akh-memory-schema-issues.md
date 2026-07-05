# Akh Memory & Schema-Typing — Work-Item Breakdown

**Status:** Ready for Anima import · **Design locked:** 2026-06-30 (grilling session)
**Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")
**Design source of truth:** this repo + the locked-design note (`project_akh_memory_schema_design.md` in agent memory). Author the canonical plan + ADR via **D0** below.

**Thesis:** *Obsidian on the write side, Orbit on the read side, an LLM Librarian as the projector between them, a human-gated schema-pack registry as the type system.*

> Grounding: **GitLab Orbit** = the new "context layer for AI agents" — closed property graph (24 node types / 6 domains), introspectable `get_graph_schema`, mechanically projected from truth. **Obsidian / Karpathy LLM-Wiki** = emergent schema, markdown + wikilinks, LLM authors notes. Akh takes Obsidian's permissive substrate + Orbit's typed-core discipline, joined by a promotion gate.

### Requirements traced by these items
- **R1** shared + own memory (cloud + local LLMs consistent) → 2, 5, 8
- **R2** small-LLM leverage (behavior, weak knowledge) → S0, 1, 6
- **R3** bidirectional schema (assists writing / captured from notes) → 1, 3, 4
- **R4** context-manager sink (<100K, no compaction) → 7
- **R5** typed + permissive split, neither weakened → 2, 3, 5
- **R6** reusable schemas (not bespoke per domain) → 2, 3

### How to import on the Anima-MCP device
1. Create the **Parent** work item first under project `019d9844-…`.
2. Create children in the order below (blockers before dependents).
3. As you create each, **replace the slice-label `Blocked by` refs** (e.g. "Slice 2") with the **real Anima WI IDs**.
4. Set each item's `current_phase` to the **Triage** value shown. Items marked `triaged` want a human decision before an AFK agent runs; the rest are `agent-ready`.

### Dependency DAG
`D0, P1, P2, S0` (no blockers) → `1` → `2, 4, 7` → `3, 5, 9` and `8` ; `6` rides on `S0`, otherwise standalone.

---

# PARENT — Akh Memory & Schema-Typing Subsystem

**Triage:** `triaged` (umbrella)

## What to build
The end-to-end memory subsystem that lets cloud and local LLMs build a **shared** and **own** memory they can work against consistently: a **permissive truth store** (notes + transcripts + episodic spine) plus a **typed read-model projection** that crystallizes a small, sacred typed core at runtime. Schema is an **installable, community-curatable, human-confirmed** artifact (not code). A small local model writes notes append-only; a daily **Librarian** (gpt-oss 120B) projects prose into typed knowledge; the result feeds the Passive + MemGPT context managers so long-horizon work stays <100K with no compaction.

Children = the vertical slices D0 / P1 / P2 / S0 / 1–9.

## Acceptance criteria
- [ ] All child slices shipped and individually demoable
- [ ] R1–R6 each demonstrably covered by at least one shipped slice
- [ ] Canonical plan + ADR authored and kept in sync (D0)
- [ ] The locked-design memory note references the plan + ADR paths

## Blocked by
None — umbrella.

---

# D0 — Author plan + ADR for the subsystem

**Triage:** `agent-ready` · do first (docs)

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
The canonical design docs, per the project's documentation flow: a timestamped **plan** capturing the 12 locked decisions + open work, and a sequentially-numbered **ADR** recording the architecture choice (permissive-truth + typed-projection that crystallizes at runtime; globally-immutable typed core; schema-pack registry; tiered small-model + daily-Librarian authoring) and the **GitLab-Orbit vs Obsidian** grounding with rejected alternatives. Update the architecture doc and push both to the corpus.

## Acceptance criteria
- [ ] Plan committed under `docs/ai/plans/` (the 12 decisions + open-work list)
- [ ] ADR committed under `docs/ai/decisions/`, numbered after the latest existing ADR, capturing the decision + grounding + alternatives considered
- [ ] `docs/ai/architecture.md` updated and its "last updated" date bumped
- [ ] Both docs ingested into Seshat (`seshat_ingest`)
- [ ] Parent Anima WI description references the plan + ADR file paths

## Blocked by
None — can start immediately.

---

# P1 — Prefactor: `Note` = `Document` + frontmatter + stable id

**Triage:** `agent-ready` · prefactor · covers R3

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Extend the existing library `Document` (vector-indexed prose, no frontmatter — it was always meant to be a frontmatter-less note) into a **Note**: optional typed frontmatter (key/value, schema.org-flavoured) + wikilinks + a **stable note identity** that survives edits, while staying vector-indexed. Frontmatter is optional and degrades gracefully — a frontmatter-less Note behaves byte-identically to today's Document. This is only the carrier; no projection yet.

## Acceptance criteria
- [ ] A Note can be created with a prose body and zero-or-more frontmatter fields
- [ ] A Note has a stable id that survives body/frontmatter edits
- [ ] An existing Document loads as a frontmatter-less Note with no behavior change (byte-identical recall)
- [ ] Frontmatter and wikilinks round-trip through persistence

## Blocked by
None — can start immediately.

---

# P2 — Prefactor: stable triple-identity hash

**Triage:** `agent-ready` · prefactor · enables R1 (evolution)

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Give every projected triple a **stable content identity** — a hash over (subject, predicate, object, compartment), excluding volatile fields (timestamp, confidence, provenance id) — so a later projection can tell "same fact, still true" from "new fact." Identity only; no supersession logic here (that's slice 8).

## Acceptance criteria
- [ ] Each triple exposes a stable identity hash
- [ ] Re-asserting the same (s, p, o, compartment) yields the same hash
- [ ] Changing any of s / p / o / compartment yields a different hash
- [ ] The hash is persisted and queryable for diffing

## Blocked by
None — can start immediately.

---

# S0 — Spike: R2 feasibility (small-model schema.org + task tool)

**Triage:** `agent-ready` · **de-risk before building** · covers R2

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
An empirical probe answering the bet the whole design rests on: can **gemma4-12b** (1) emit usable schema.org frontmatter for a concept note, and (2) reliably drive a minimal structured task tool (`create` / `complete`) under grammar-constrained decoding? Use the existing agentic-flow example harness with `AKH_AGENT_DEBUG=1`. Output a short **go / no-go** report with measured success rates. If it fails, the tiered append-only-small-model premise needs rework (heavier grammar constraint, a repair pass, or capable-model-only writes).

## Acceptance criteria
- [ ] ≥20 note-write trials measured; report % producing parseable frontmatter with a valid `@type`
- [ ] ≥20 task-tool trials measured; report % producing schema-valid task-tool calls
- [ ] A go/no-go recommendation written into the design memory / plan
- [ ] On no-go: the chosen fallback documented

## Blocked by
None — de-risk first.

---

# 1 — Thin note → projection → typed-KG round-trip

**Triage:** `agent-ready` · covers R3

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
The smallest complete vertical slice through write → store → project → read. With a **hardcoded single-type seed** (e.g. `Person`), a Note carrying `@type: Person` projects — via a trivial frontmatter→triple map, **no LLM** — into one typed triple (`note is-a Person`) in the KG, readable via a typed query. Proves the spine end-to-end before any registry or Librarian exists.

## Acceptance criteria
- [ ] Writing a Note with `@type: Person` yields a `note is-a Person` triple
- [ ] The triple carries provenance back to the source note + version
- [ ] A typed query returns the note as a `Person`
- [ ] A Note with no `@type` produces no type triple (no crash)

## Blocked by
- Slice P1 (Note carrier)
- Slice S0 (small-model write path validated)

---

# 2 — Schema-pack registry, one seed pack end-to-end

**Triage:** `triaged` (pack format wants human review) · covers R5, R6, R1

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
A **schema pack** is a typed microtheory stored as a **native-triple byte blob in a dedicated registry table**. Build the full path for one tiny seed pack (schema.org core: `Person`, `Organization`, `is-a`): a **confirm-time compiler** (portable source — Turtle / JSON-LD / YAML — → native blob), an **operator install** command on a **non-LLM path** that requires a **human confirm** before admission, and **akh activation** (import the pack as a microtheory via `ctx:specializes` into the effective schema). After activation, notes can be typed with the pack's types — with no code deploy or restart.

## Acceptance criteria
- [ ] A pack authored in portable text compiles to a native-triple blob; one registry row holds `{id, version, name, scope, checksum, confirmed_by, blob}`
- [ ] Operator install is non-LLM and requires an explicit human confirm before the row is admitted
- [ ] An akh activates the pack by import; a `get_graph_schema`-style introspection lists the now-in-force types
- [ ] A note typed with a pack type projects correctly (extends slice 1 beyond the hardcoded seed)
- [ ] Installing/activating needs no code deploy or restart

## Blocked by
- Slice 1 (projection spine)

---

# 3 — `is-a` subtyping + property-shape validation

**Triage:** `agent-ready` · covers R5, R6

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Extend the type system with **subtyping and shapes**. A second pack defines `Employee is-a Person` carrying its own required property shape (e.g. `employer`). Validation checks a note against its declared type's shape — **advisory at write**, surfacing rich **miette diagnostics**; enforcement is the operator/Librarian's call. Proves the three locked type semantics: **global immutable names** (no redefinition), **`is-a`-only extension**, and **additive properties** (adding `Employee` never mutates the `Person` view).

## Acceptance criteria
- [ ] A pack can declare a subtype via `is-a` with its own required/optional properties
- [ ] A note typed `Employee` missing a required property surfaces a miette diagnostic (does not hard-block the write)
- [ ] Completing the property validates cleanly
- [ ] Attempting to redefine an existing global type name is rejected
- [ ] The projected `Person` view is unchanged by adding `Employee` (additive)

## Blocked by
- Slice 2 (registry + packs)

---

# 4 — Daily Librarian projection (real LLM extraction)

**Triage:** `triaged` (Librarian prompt/pipeline wants human review) · covers R3

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Replace the trivial projector with the **daily Librarian** pass backed by **gpt-oss 120B** (Ollama). It harvests prose note bodies + transcripts (the "daily log") and **extracts typed triples** against the in-force schema, recording provenance and **proposing frontier→core candidates**. Adapt the existing Phase-32 knowledge-elicitation pipeline to the 120B backend. Runs as a scheduled/batch pass; **complements, does not replace,** the VSA sleep cycle.

## Acceptance criteria
- [ ] A freeform prose note, after a Librarian run, yields typed triples consistent with the in-force schema
- [ ] Every extracted triple carries provenance (source note + version, model, pass)
- [ ] Frontier terms (not in core) are recorded as candidates, never silently promoted
- [ ] The pass is batch/scheduled and cost-bounded (per cycle, not per turn)
- [ ] The VSA sleep cycle still runs independently

## Blocked by
- Slice 1 (typed-KG projection target)

---

# 5 — Frontier → core promotion + human-confirm + aliasing

**Triage:** `agent-ready` · covers R1, R5

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Close the governance loop. When a frontier term recurs across **≥N notes / ≥M akhs** with consistent usage, the Librarian **proposes** it for the core; a **human-confirm** step (reusing slice 2's confirm path) admits it to the **shared registry**; the Librarian **aliases synonyms** onto existing core terms so the core stays small and clean (R5). LLMs design the *frontier* freely; the shared *core* grows only by consensus + ratification.

## Acceptance criteria
- [ ] A recurring frontier term generates a promotion proposal carrying its evidence (counts, example notes)
- [ ] No core type/predicate is added without an explicit human confirm
- [ ] On confirm, the term enters the registry and later notes type against it
- [ ] A detected synonym is aliased onto the existing core term instead of being added
- [ ] Rejected proposals do not re-surface every run

## Blocked by
- Slice 2 (registry + confirm path)
- Slice 4 (Librarian proposals)

---

# 6 — Structured task tool end-to-end

**Triage:** `agent-ready` · covers R2 · can start right after S0

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
A **structured task tool** (`create` / `update` / `complete` / `list`) the LLM drives directly — **grammar-constrained + schema-validated** (miette diagnostics on bad args), so reliability comes from the tool schema, not the model's judgment (this is the one deliberate exception to small-models-are-append-only; OpenCode shows agents do this well). The **structured task store is the source of truth**; any markdown checkboxes are a *rendered view*, not the source — **no markdown parser**. Lifecycle lives in the handler: `complete` → emit a past `EpisodicEntry` ("Deployed X…") + spawn the next recurrence; `create` → project to the episodic future-spine / calendar / KG commitment. Adopt Obsidian Tasks emoji syntax (`📅` due, `⏳` scheduled, `🔁` recur, `✅` done) as the boundary form.

## Acceptance criteria
- [ ] The model creates a task with due/recurrence via a schema-validated call; bad args yield a miette diagnostic, not a corrupt task
- [ ] Completing a task emits a past `EpisodicEntry` and, if recurring, the next occurrence
- [ ] Tasks project to a KG commitment + the episodic future-spine
- [ ] The structured store is authoritative; no markdown scraping is involved

## Blocked by
- Slice S0 (model can drive the tool reliably)

---

# 7 — Context-manager core / recall / archival integration

**Triage:** `agent-ready` · covers R4

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Wire notes + `EpisodicEntry` + typed-KG into the **PassiveManager** and **VirtualContextManager**. **Core** = identity / prefs / goals (`SafetyEnforced` pins) + **active-note frontmatter only** + a `get_graph_schema` digest. **Recall** = recent episodic + recently-touched frontmatter. **Archival** = note **bodies** (vector / `archival_search`) + KG (`query_graph`). Bodies are **never resident**, so the resident footprint is **horizon-independent** — that's how long-horizon work avoids compaction. **Tier-aware:** small models are append-only (no core self-edit tools); capable models get the MemGPT self-edit tools. Both managers consume the same substrate.

## Acceptance criteria
- [ ] A long-horizon session keeps resident context ≤100K, with bodies paged to archival
- [ ] Recall pulls a previously-evicted note body back on demand via search
- [ ] Resident footprint does not grow with conversation length (no compaction triggered)
- [ ] Small-model path exposes no core self-edit tools; capable-model path does
- [ ] Passive and Virtual managers consume the same memory substrate

## Blocked by
- Slice 1 (notes/projection exist)
- Slice 4 (typed content to populate core)

---

# 8 — Evolution: diff-supersede + TMS + multi-akh DS conflict

**Triage:** `agent-ready` · covers R1

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Keep the typed core trustworthy over time. When a note is edited, **re-project and diff** against the prior projection; **retract only the changed triples** via the TMS (`was_retracted`) so recall filters them, while **preserving full history** in provenance + note versions (that's the "strong information evolution"). When two akhs assert contradictory facts in the **shared core**, represent the contested fact as a **Dempster-Shafer belief interval weighted by source reliability** — not last-writer-wins.

## Acceptance criteria
- [ ] Editing a fact retracts only the changed triple(s); unchanged triples keep their original provenance/confidence
- [ ] Retracted triples are filtered from recall but remain in provenance history
- [ ] The prior note version + the change are auditable
- [ ] Two akhs disagreeing yields a DS belief interval (no silent overwrite), weighted by source reliability

## Blocked by
- Slice 4 (projection emits the triples to diff)
- Slice P2 (stable triple-identity hash)

---

# 9 — Migration: grandfather + schema-coverage report

**Triage:** `agent-ready` · sustains R5 over time

## Parent
PARENT — Akh Memory & Schema-Typing Subsystem

## What to build
Fold today's corpus in with **no flag-day**. Existing triples are **grandfathered as frontier** (they already work permissively). **Front-load the safe part:** register the existing relation namespaces (`is-a`, `part-of`, `doc:`, `agent:`, `ctx:`, the `ElicitPass` relations) into the seed **core** vocabulary at install; entity typing stays **lazy** (the normal Librarian pass does it). Ship an **opt-in, non-LLM schema-coverage report** (typed vs frontier %) the operator can run anytime — visibility without forcing change, doubling as the "is the core crystallizing?" dashboard.

## Acceptance criteria
- [ ] Existing triples keep working unchanged after the core is introduced (no downtime)
- [ ] Existing relation namespaces are registered into the core vocabulary
- [ ] A non-LLM coverage report prints typed-vs-frontier proportions over the live KG
- [ ] No bulk retyping job is required; lazy typing happens via the Librarian

## Blocked by
- Slice 2 (registry + core vocabulary)
