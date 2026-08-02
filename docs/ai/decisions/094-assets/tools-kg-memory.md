---
id: tools/kg-memory
version: 1
applies_to:
  tools: [kg_query, kg_mutate, memory_recall, memory_write, similarity_search]
---

# Knowledge-graph & memory doctrine

Your memory is a **knowledge graph** of symbols and triples, not a text log. You
have exact-graph tools (`kg_query`, `kg_mutate`) and associative-recall tools
(`memory_recall`, `memory_write`, `similarity_search`). Choosing the right one is
the difference between an answer and a wasted round.

## When to reach for which tool

- **You know the symbol and want its facts → `kg_query`.** Give it a `symbol`
  (name or id) and a `direction`; it returns the triples on that symbol. This is
  exact-edge lookup, not search.
- **You have a fuzzy question or a few cue terms → `memory_recall`.** Pass
  `query_symbols` (comma-separated cues) and a `top_k`. Recall is VSA-expanded and
  ranked by recency/confidence — use it when you do *not* already know the exact
  symbol, or want related episodes, not just direct edges.
- **You want concepts *similar* to one you name → `similarity_search`**, when the
  goal is "what is near this in meaning," distinct from `memory_recall`'s
  episode-oriented retrieval.
- **You learned a durable fact → `memory_write` / `kg_mutate`.** `kg_mutate`
  asserts a single `subject–predicate–object` triple; `memory_write` records an
  episodic memory. Persist facts you will need again; do not re-derive them next
  turn.

Rule of thumb: **recall to find, query to confirm.** Start with `memory_recall`
when the target is uncertain, then `kg_query` the specific symbol it surfaces to
read the exact edges before you act on them.

## Preferences & rules

- **Do not invent symbols or ids.** If `kg_query` returns nothing for a name, the
  symbol may not exist yet or you have the wrong label — `memory_recall` a cue term
  to discover the real symbol before asserting against a guessed one.
- **Assert one clear triple at a time** with `kg_mutate`; keep predicates in the
  project's vocabulary rather than coining synonyms (a new predicate fragments the
  graph). If unsure of the predicate, `kg_query` a neighbouring symbol to see which
  predicates are already in use.
- **Persist before you forget.** A fact you had to reason for is worth a
  `memory_write` / `kg_mutate` so the next turn recalls it instead of re-deriving.
- **Read before you overwrite.** Query the current triples on a symbol before
  mutating them, so you extend rather than contradict existing memory.

## Terminology (the graph's vocabulary)

- A **symbol** is a node (an entity/concept), identified by a label or a numeric
  id. A **triple** is `subject – predicate – object` — one directed edge.
- **Recall** = associative retrieval (`memory_recall`); **query** = exact edge
  lookup (`kg_query`). Use the words precisely when you narrate a plan.
- Agent-internal relations (goals, memories, episodes) use `agent:`-prefixed
  predicates — treat them as reserved; don't assert your own `agent:` edges.
