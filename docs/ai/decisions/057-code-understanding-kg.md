# ADR 057 — Code-Understanding Knowledge Graph: out-of-process tree-sitter ingester, AGE read-model, live edits live in the LLM

> Date: 2026-06-15
> Status: Proposed
> Phase: 43 (Code-Understanding Knowledge Graph)
> Depends on: ADR 049 (Postgres truth + AGE read-projection), ADR 047 —
>             Seshat's Archive (SeaORM/CNPG precedent), ADR 029 (MCP server)
> Enhances: Phase 10 (generative functions / code generation)
> Supersedes: the in-process `CodeIngestTool` (`src/agent/tools/code_ingest.rs:33`)
>             and Rust-only `syn` code ingestion (`src/agent/tools/code_predicates.rs`)
> Reference ontology: GitLab Orbit schema (https://docs.gitlab.com/orbit/remote/schema/)
> Anima ticket: TBD

## Context

Today, source code enters the Akh's knowledge graph through an **in-process
agent tool**: `CodeIngestTool` (`src/agent/tools/code_ingest.rs:33`) parses
Rust with `syn`, walks the AST, and emits ~20 `code:` predicates
(`src/agent/tools/code_predicates.rs`) — `code:defines-fn`,
`code:implements-trait`, `code:has-field`, `code:depends-on`, etc. — into the
agent's belief KG, scoped to a per-repo compartment (`repo-X::code`).
Re-ingestion clears the compartment and re-asserts. Each symbol carries a
`SourceRef` (document_id, byte_start, byte_end).

Three structural problems, plus a conceptual one:

1. **Rust-only.** `syn` parses only Rust. The agent must understand polyglot
   repositories (Python, TypeScript, Go, Java, C#, PHP, Ruby — the languages
   GitLab Orbit covers). A `syn`-based ingester cannot grow to that.

2. **Wrong store, wrong layer.** Code structure is *machine-derived,
   high-volume, and not a belief.* Yet — after ADR 049 makes `akhomed.facts`
   the authoritative fact store — it lands in belief space as if it were the
   Akh's knowledge. This:
   - pollutes belief space with millions of structural triples (a monorepo
     dwarfs the Akh's actual belief count);
   - inflates the WAL and snapshots that ADR 049 is careful to keep bounded;
   - risks the temporal-decay (`src/temporal.rs`), defeasible
     (`src/graph/defeasible.rs`), and argumentation machinery being applied
     to facts like "function `foo` exists" — which is category-incorrect.
     Confidence does not decay on a function signature.

3. **In-process cost.** Parsing whole repositories blocks the OODA loop and
   competes with cognition for the single-process budget ADR 049 guards.

4. **The KG should *be* the understanding.** Today the agent still falls back
   to reading raw source files to understand code. We want the opposite: a
   structural read-model rich enough that the agent **understands the code
   without reading it** — it queries "definition of X", "callers of Y",
   "what imports Z", "blast radius of changing W" instead of opening files
   and spending LLM context on raw text.

A related principle falls out of (4): when the agent *edits* code (Phase 10
generative functions), those in-flight edits must **not** mutate this KG.
They belong to the LLM/working-state side of akh-medu until they are written
to disk and re-indexed. This is exactly GitLab Orbit's stance — "designed for
point-in-time SDLC insight, not real-time or transactional use cases" — adopted
here deliberately.

## Decision

Replace in-process `syn` ingestion with an **out-of-process, polyglot
tree-sitter ingester daemon** that maintains a **separate** code-structure
store, projected into Apache AGE as a read-only Cypher graph. The agent
*queries* this graph; it never asserts code structure into its own belief KG.
Live edits stay in the LLM.

In bullet form:

- **Separate ingester daemon — Shen (`shend`).** A standalone process,
  sibling to `akhomed` (`src/bin/akhomed.rs`) and `seshd` (Seshat, ADR 047).
  *Shen* (šn) is the Ancient Egyptian word "to encircle" and the name of the
  shen-ring — the loop of rope that encircles everything the sun orbits, the
  hieroglyph of eternity and the all-encompassing. It is the Egyptian reading
  of "Orbit": the daemon encircles the whole codebase and holds it as a
  queryable graph, just as GitLab Orbit "indexes your entire SDLC". It owns
  code parsing, the code store, and the AGE projection refresh. The in-process
  `CodeIngestTool` is retired.

- **tree-sitter, polyglot.** Parsing uses the `tree-sitter` crate plus grammar
  crates (`tree-sitter-rust`, `-python`, `-typescript`/`-tsx`, `-go`,
  `-java`, `-c-sharp`, `-php`, `-ruby`; extensible). Per-language tree-sitter
  query files (`.scm`) extract a **language-agnostic model**: `File`,
  `Directory`, `Definition` (function / method / class / struct / enum /
  trait / module / …), `ImportedSymbol`, and `Reference`. Node fields and edge
  names align with Orbit's schema (`Definition` with `fqn`/`definition_type`/
  `start_line`/`end_line`, `ImportedSymbol` with `import_type`/`import_path`/
  `identifier_name`) so the graph is cross-tool-comparable and import-friendly.

- **Its own store — not belief space.** Code facts live in a dedicated
  PostgreSQL schema **`codex.*`** (relational truth: `codex.repo`,
  `codex.file`, `codex.definition`, `codex.import`, `codex.reference`), owned
  by the ingester, **distinct from `akhomed.facts`**. They are projected into a
  dedicated AGE graph **`codex_kg`** (its own Postgres namespace, per AGE's
  graph-is-a-schema model) for Cypher traversal. This is exactly ADR 049's
  "AGE = read-only projection over relational truth" pattern, applied to a
  second, separately-owned store. The Akh's belief KG never carries code
  structure.

- **The KG is a read model; queries replace file reads.** akhomed consumes
  `codex_kg` through a **read interface** — primarily an **MCP query tool**
  exposed by `shend` (parallel to seshd/Seshat per ADR 029, and matching
  Orbit's MCP exposure), with an optional direct SQL/Cypher read pool as a
  latency escape hatch. Code-understanding agent flows query the graph
  ("definition of", "callers of", "imports of", "impact of changing") instead
  of reading source.

- **Live edits live in the LLM, not the KG.** `codex_kg` reflects only
  *indexed, on-disk* code at a known git ref. In-flight edits the agent makes
  (Phase 10 code generation, working memory, LLM context) are **not**
  represented in `codex_kg`; they remain in the LLM/agent state until written
  to disk and re-indexed by `shend`. Point-in-time, not transactional — by
  design.

- **Incremental, event-driven refresh (CDC-style).** `shend` re-indexes on
  git commit / filesystem change / Forgejo webhook, updating only the touched
  files' subgraph rather than full rescans. Each index is tagged with
  `(repo, commit_sha)` so queries can be point-in-time and so a relocation or
  branch switch is just a different watermark.

- **Predicate vocabulary retained and generalized.** The `code:` namespace
  survives as the schema vocabulary — now language-agnostic and
  producer-independent — so existing consumers keep stable labels. SDLC
  objects (Forgejo MRs/issues, Solstice CI runs, security findings) are an
  **optional second ingestion source under the same daemon** using an
  `sdlc:` namespace, realizing the broader Orbit-style SDLC graph. Code
  understanding is the primary remit; SDLC ingestion is phase 43g, opt-in.

## Alternatives considered

### Keep `syn` in-process, add languages incrementally

**Why considered.** Smallest change; reuses the working `code:` pipeline.

**Why rejected.** `syn` is Rust-only by construction; "add languages" means
adding a parser per language anyway. It keeps ingestion on the OODA-blocking
hot path and keeps code structure polluting belief space (ADR 049 WAL/snapshot
cost). The three structural problems above are all properties of *where and how*
ingestion runs, not of the parser alone.

### tree-sitter in-process (library, not a daemon)

**Why considered.** Polyglot parsing without standing up a new process.

**Why rejected.** Still competes with cognition for the single-process budget
ADR 049 protects. File-watching, incremental re-indexing, and webhook intake
want their own lifecycle and restart semantics. Decoupling lets the ingester
crash, scale, and be deployed independently — and matches GitLab Orbit Local,
which is itself a separate process building a code graph.

### Code facts into `akhomed.facts`, projected to `akhomed_kg`

**Why considered.** One store, one AGE graph; reuses ADR 049 wholesale.

**Why rejected.** Conflates machine-derived structure with belief. It inflates
the WAL and snapshots ADR 049 keeps bounded, risks temporal-decay /
argumentation running over structural facts, and lets repository scale swamp
the Akh's actual knowledge. A separate `codex.*` store with its own `codex_kg`
projection keeps both clean while reusing the identical ADR-049 pattern.

### Represent live edits transactionally in the KG

**Why considered.** A always-current code graph; no staleness window.

**Why rejected.** Turns the KG into an editor/IDE model, fights ADR 049's
snapshot cadence, and duplicates state the LLM already holds during generation.
Orbit explicitly scopes itself to point-in-time for the same reason. The
staleness window (edit on disk → re-index) is acceptable precisely because
live edits are supposed to live in the LLM.

### LSP / stack-graphs instead of tree-sitter

**Why considered.** Precise cross-file symbol resolution (which import binds to
which definition).

**Why rejected (for now).** Requires a language server runtime per language and
heavier operational surface. tree-sitter gives breadth (one engine, many
grammars), no per-language server, and aligns with the tooling already in the
stack (lean-ctx's tree-sitter AST layer). A precise resolution pass can be
layered on top later (Phase 43+); start with tree-sitter's syntactic structure
plus heuristic import resolution.

## Consequences

### Positive

- **Polyglot code understanding** (8+ languages from day one) — the capability
  gap versus Orbit closes.
- **OODA loop unblocked.** Ingestion is out-of-process; cognition no longer
  competes with repo parsing.
- **Belief KG stays clean and small.** WAL/snapshot cost (ADR 049) is
  unaffected by code volume; temporal/argumentation machinery never sees
  structural facts.
- **"No need to read the code."** Structural queries replace file reads, cutting
  LLM context spend and giving the agent a stable, queryable model of the code.
- **External Cypher / MCP access** to the code graph for visualisers,
  debuggers, and tooling — the same read-view rationale ADR 049 gives for
  `akhomed_kg`.
- **Cross-tool schema parity** with Orbit eases future import/interop.

### Negative

- **A new daemon to operate.** `shend` adds a Helm/Talos workload and another
  MCP endpoint, and connects to the CNPG cluster (the AGE extension is already
  committed there per ADR 049 resolution P38Q6).
- **Syntactic, not semantic, by default.** tree-sitter yields structure;
  binding each import/reference to its target definition needs a resolution
  pass that is approximate without a compiler. Accept approximate edges first;
  refine later.
- **Staleness window.** Between an edit landing on disk and re-index, `codex_kg`
  lags. Acceptable by design — live edits live in the LLM and queries are
  point-in-time — but agent flows must not assume the graph reflects
  uncommitted work.
- **Mediation tension with ADR 047.** ADR 047 holds that akh-medu never connects
  to Postgres directly. ADR 049 already relaxed this for the Akh fact store
  (akhomed gets a SeaORM pool). Here the default is mediation (read via
  `shend`'s MCP interface); a direct read pool is an opt-in perf escape hatch,
  consciously chosen, not assumed.
- **Eventual-consistency complexity** in the incremental indexer
  (rename/move/delete handling, watermark bookkeeping) — needs a test harness.

### Migration

- **Retire `CodeIngestTool`** (`src/agent/tools/code_ingest.rs:33`) and the
  `syn` dependency path for code ingestion.
- **Drop or relocate** existing `code:` triples from belief compartments;
  one-time migration into `codex.*`.
- **Update stale references:** the help string at
  `src/agent/tools/code_gen.rs:144` ("Use code_ingest or kg_mutate…") and any
  agent flows / prompts that invoked `code_ingest` switch to `codex_kg`
  queries.

## Implementation (sub-phases)

Staged so each step is independently runnable; the agent keeps working
throughout (falling back to today's behaviour until the relevant phase lands).

1. **43a** — `shend` skeleton + tree-sitter **Rust** parity with the current
   `syn` output, validated against the existing `code:` predicate set.
2. **43b** — `codex.*` relational schema + `codex_kg` AGE projection
   (mirrors ADR 049's migration pattern; the `create_graph('codex_kg')` step
   is a no-op when the AGE extension is unavailable).
3. **43c** — additional grammars (python, typescript/tsx, go, java, c#, php,
   ruby) via `.scm` query files against the language-agnostic model.
4. **43d** — MCP read interface on `shend`; code-understanding agent flows
   switch from file reads / `code_ingest` to `codex_kg` queries.
5. **43e** — incremental, event-driven refresh (filesystem watch + Forgejo
   webhook); `(repo, commit_sha)` watermarking for point-in-time queries.
6. **43f** — retire `CodeIngestTool`; one-time migration of legacy `code:`
   triples; fix `code_gen.rs` help text and agent prompts.
7. **43g** (optional) — `sdlc:` ingestion (Forgejo MRs/issues, Solstice CI,
   security findings) under the same daemon, completing the Orbit-style SDLC
   graph.

## References

- ADR 049 — Relocatable Session-Based Akh (Postgres truth + AGE read-projection)
- ADR 047 — Seshat's Archive (SeaORM-backed CNPG precedent; sibling daemon `seshd`)
- ADR 029 — MCP server integration (read-interface precedent)
- Phase 10 — Generative Functions (the LLM-side home of live edits)
- GitLab Orbit schema (reference ontology): https://docs.gitlab.com/orbit/remote/schema/
- GitLab Orbit Local (separate-process code-graph precedent): https://docs.gitlab.com/orbit/local/
- Apache AGE (graph-is-a-schema, `cypher()` read views): https://age.apache.org/
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Retired: `src/agent/tools/code_ingest.rs`, `src/agent/tools/code_predicates.rs`

## Naming note

The daemon is **Shen** (`shend`), after *šn* — the Ancient Egyptian verb "to
encircle" and the name of the **shen-ring** (Gardiner V9), the loop of rope
that encircles everything the sun orbits and the hieroglyph for eternity and
the all-encompassing. It is the Egyptian reading of "Orbit": the process that
encircles the whole codebase and holds it as a queryable graph. It joins
`akhomed` and `seshd` (Seshat the archivist, ADR 047) — Shen the encircler is
their sibling. The shen-ring glyph can back a `SymbolKind::Glyph` for the
daemon's identity in the registry, the way the wedjat (`𓂀`) is used elsewhere.
