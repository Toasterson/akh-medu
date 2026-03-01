# Post-Trip Autonomy Fixes

**Created:** 2026-03-01
**Status:** Planned
**Context:** Issues discovered during Mac Mini M2 autonomous deployment testing

---

## P0 — Goal Quality (garbage entity filtering)

The goal generator creates "Learn more about X" goals for every under-connected
entity, including sentence fragments from PDF ingestion. Examples of garbage goals:

- "Learn more about 'But this equality'"
- "Learn more about 'Painstaking analyses of probate records'"
- "Learn more about 'In the middle* There'"
- "Learn more about 'Just hit ENTER to accept'"

**Root cause:** The PDF/epub parsers produce entities from text fragments. The
grammar framework doesn't distinguish real concepts from noise. The goal
generator treats all entities equally.

**Proposed fixes:**
1. **Entity quality scoring** — score entities by connectivity, label quality
   (length, capitalization, punctuation), and provenance. Skip goal generation
   for entities below a quality threshold.
2. **PDF parser improvements** — filter sentence fragments during ingestion,
   not after. Only create entities for proper nouns, defined terms, and
   structured headings.
3. **Goal deduplication** — several goals target the same concept at different
   fill levels (e.g., two separate "Learn more about 'CA'" goals). Merge or
   skip duplicates.
4. **Goal pruning idle task** — periodic background task that removes goals
   targeting garbage entities (e.g., labels starting with lowercase, containing
   punctuation fragments, or shorter than 3 characters).

## P0 — Failed Tool Dispatch (OODA parameter mapping)

Three goals failed with `csv_ingest — missing required parameter: path` and one
with `kg_mutate — missing required parameter: subject`. The OODA loop selects
tools but can't map goal context to tool parameters.

**Root cause:** The tool selection picks a tool by name/description similarity,
but the parameter extraction from goal descriptions is too naive — it doesn't
know how to derive a `path` or `subject` from a goal like "investigate: Whats
your name?".

**Proposed fixes:**
1. **Guard against parameter-less invocations** — before executing a tool, verify
   all required parameters are present. If not, mark the goal as blocked (not
   failed) with a diagnostic explaining which parameters are missing.
2. **Improve parameter extraction** — the OODA decide phase should map goal
   context to tool parameters more intelligently, using the tool's parameter
   schema as a template.
3. **Tool applicability filter** — not all tools are applicable to all goals.
   `csv_ingest` should never be selected for an "investigate" goal.

## P1 — Conversation Compartments

Currently only the psyche compartment exists. Each conversation partner should
have their own microtheory compartment storing:
- Conversation history (as triples, not raw text)
- Learned preferences and interaction patterns
- Social KG relationships (Phase 12e)

This requires extending the compartment system with a `Person` kind and
auto-creating compartments when new interlocutors are detected.

## P1 — Seed Auto-Application

Seeds (core-ontology, etc.) are not automatically applied to new workspaces.
The user must manually run `akh seed apply core-ontology`. The daemon or
workspace creation should check if essential seeds are applied and apply them
automatically.

**Proposed fix:** Add a startup check in `run_daemon_task` that applies
`core-ontology` if no seeds are marked as applied in the workspace.

## P2 — TemporalConflict Investigation

The enriched contradiction logs (added in v0.5.4) will reveal what triples are
conflicting. After collecting data from a few days of autonomous operation:
1. Determine if the conflicts are from re-insertion of existing knowledge
   (benign — should be deduplicated upstream) or genuine contradictions.
2. If benign: add a "skip if identical triple exists" check before
   `add_triple`, or deduplicate during synthesis.
3. If genuine: consider promoting contradiction policy from `Warn` to
   `Replace` with confidence-based resolution.

## P2 — Preference Bootstrapping

Preferences are all zeros because preference signals only come from user
interaction. For autonomous operation, the daemon should:
1. Infer initial preferences from the psyche's domain/traits
2. Generate preference signals from autonomous learning choices
3. Use curiosity-driven exploration as implicit positive feedback

## P3 — Goal Accumulation Over Time

Completed/failed goals stay in the KG forever. After weeks/months this becomes
a performance issue. Add a periodic cleanup task that:
- Archives completed goals older than N days
- Removes failed goals after retry limit
- Compacts goal metadata

## P3 — Daemon Agent / Handler Agent Coherence

The daemon runs a long-lived Agent instance while HTTP handlers create throwaway
Agent instances. Changes made by handlers (e.g., `causal bootstrap`) can be
overwritten by the daemon's next persist. The v0.5.4 fix (skip empty causal
persist) is a band-aid. The proper fix is one of:
1. **Shared agent** — handlers route through the daemon's agent via a channel
2. **Merge-on-persist** — read-modify-write instead of blind overwrite
3. **Separate stores** — each subsystem (causal, preferences, etc.) persists
   independently to its own key, with atomic compare-and-swap
