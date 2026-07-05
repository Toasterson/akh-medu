# Phase 44j — Context-Strategy A/B (manager-level)

**Date:** 2026-06-21
**Harness:** `src/bench_phase44/` (`cargo test --lib bench_phase44`; regenerate the
tables with `cargo test --lib bench_phase44::tests::dump_report -- --ignored --nocapture`)
**Scope:** deterministic, no LLM, no network. Compares
`SafetyEnforced<{CompactingManager, PassiveManager, VirtualContextManager}>` on a
fixed scenario set, measuring the properties the *assembly layer* controls:
assembled token cost, salient-fact retention, safety-pin violations, message count.

This is **not** an NL-quality benchmark. Answer quality and "context rot" need an
LLM judge + the live 3-akh harness and are the recorded follow-up (below). The
honest claim here is about the *mechanics of what enters the window*.

## Why these scenarios

The strategies only diverge when something must be dropped — i.e. long
conversations and/or tight windows. With a large window every strategy keeps
everything (and the live akhs run 128K–1M windows, per the measure-first report).
So the scenarios use deliberately tight budgets to surface the trade-offs:

1. **`grounding_in_kg`** — the salient fact is a KG triple; every strategy should
   keep it (grounding parity).
2. **`early_fact_tight_budget`** — the fact is stated in an early turn; retention
   depends on the history policy.
3. **`core_memory_retention`** — the same early fact, but persisted to core memory
   (the mechanism only `virtual` has).

## Per-scenario cells

| scenario | strategy | tokens | msgs | salient kept | safety viol |
|---|---|---:|---:|:---:|---:|
| grounding_in_kg | compacting | 39 | 4 | ✓ | 0 |
| grounding_in_kg | passive | 39 | 4 | ✓ | 0 |
| grounding_in_kg | virtual | 39 | 4 | ✓ | 0 |
| early_fact_tight_budget | compacting | 47 | 8 | ✓ | 0 |
| early_fact_tight_budget | passive | 16 | 3 | ✗ | 0 |
| early_fact_tight_budget | virtual | 19 | 4 | ✗ | 0 |
| core_memory_retention | compacting | 45 | 8 | ✓ | 0 |
| core_memory_retention | passive | 16 | 3 | ✗ | 0 |
| core_memory_retention | virtual | 42 | 5 | ✓ | 0 |

## Per-strategy aggregates

| strategy | scenarios | avg tokens | lost salient | safety viol |
|---|---:|---:|---:|---:|
| compacting | 3 | 43.7 | 0 | 0 |
| passive | 3 | 23.7 | 2 | 0 |
| virtual | 3 | 33.3 | 1 | 0 |

## Reading the result

- **Grounding is preserved by all three** — KG facts ride the atomic, high-priority
  `KnowledgeGraph` section, so no strategy drops them.
- **`compacting` retains the most, at the highest token cost** (43.7 avg, 0 lost):
  it carries full conversation history within budget. The right default.
- **`passive` is the cheapest and loses the most** (23.7 avg, 2 lost): it keeps
  only the immediate tail and trusts background consolidation + recall for the
  rest — so a fact never re-surfaced by recall falls out. Best when token cost
  dominates and consolidation is healthy.
- **`virtual` sits in between** (33.3 avg, 1 lost): it keeps the early fact when the
  model persisted it to core memory (`core_memory_retention`), but — like passive —
  drops a fact left only in history (`early_fact_tight_budget`). Its retention is
  exactly as good as the model's use of the self-edit tools.
- **Safety-pin violations are 0 everywhere**, as required (ADR 065 D6). The curating
  managers anchor every dropped span as `None` (no decision/veto/danger content
  reaches the eviction path), so the `SafetyEnforced` guard is satisfied by
  construction; the actual rejection path is exercised by the unit tests in
  `src/agent/context/safety.rs`.

## Recommendation

Keep **`compacting` as the daemon default** (`AKH_CONTEXT_STRATEGY=compacting`).
Reach for `passive` when token cost dominates and consolidation is healthy, and for
`virtual` when the akh should curate durable facts itself (long-running
assistant/persona). All three are live and selectable per akh.

## Deferred (needs the live harness, not this one)

- **NL answer quality** A/B across strategies on the 3-akh setup (real models).
- **Context-rot incidence** (an LLM-judge metric).
- **Recall precision/recall (Q3)** on a labelled query set, against the 44g VSA blend.
- A tighter `compacting` budget point to show its own drop-off (here 47 tok fit the
  64-token window, so it never had to trim).
