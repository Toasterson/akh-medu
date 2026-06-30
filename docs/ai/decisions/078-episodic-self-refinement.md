# ADR 078 — Episodic memory self-refinement (the akh cleans its own memory)

> Date: 2026-06-23
> Status: Prototype (opt-in via `AKH_ENRICH_MEMORY`)
> Related: ADR 076 (OpenClaw self-import — how facts get stored), the agent
>          goal system (`src/agent/goal_generation.rs`, `GapAnalysisTool`),
>          `src/agent/tools/{memory_write,memory_recall}.rs`,
>          `_akh/triples/remove` verb (ADR-less, this session)

## Context

Import is lossy. `memory_write` stores the LLM's *distilled* facts as `(S,P,O)`
triples and links the episode to each fact's **subject** (`episode —learned→ S`).
Three granularity losses follow (measured on the OpenClaw import):

1. **Extraction drops detail.** The model keeps salient concepts (`redb —issue→…`)
   but drops specific values — the redb DB *path* was never emitted as a fact, so
   it isn't in the KG at all. This is the dominant gap.
2. **Subject-only linkage.** Recall walked `learned → subject`, surfacing concept
   labels, not the predicate/object — fixed this session by rendering full S–P–O
   in `memory_recall` and the real summary (not the discriminator hash).
3. (resolved) summary showed a hash; `reconstruct_episode` now prefers `has_summary`.

#1 can't be fixed by recall alone — the detail must be *re-extracted from source*.

## Decision

Don't re-import externally; let the akh **refine its own memory as a goal**. All
the machinery already exists — `GapAnalysisTool` + the `generate_goals` loop to
notice gaps, `memory_write`/`add-triple` to add facts, and the new
`remove-triple` verb to prune. The only missing piece is *noticing* that an
episode is thin.

`proposals_from_episodic_gaps` (in `goal_generation.rs`, opt-in via
`AKH_ENRICH_MEMORY`) scans recent episodes and proposes enrichment for the *thin*
ones — fewer than `THIN_FACT_THRESHOLD` (5) **grounded** facts (learned subjects
that actually have outgoing triples) **and** a known `tag:source:` to re-read.
Each becomes a `GapDetection` goal ("Enrich memory \"…\" by re-reading <source>"),
priority scaled by thinness. The akh's OODA loop then reads the source and writes
the missing facts — closing the loop the import opened.

The two recall fixes are the **enabler**: the akh can only judge a memory as thin
once it can *see* its full facts + real summary.

## Why this shape

- **Reuse `GapDetection`, don't add a `GoalSource` variant** — 18 external match
  sites; a thin memory genuinely *is* a knowledge gap (`gap_kind:
  "episodic_enrichment"`). Cheap and non-invasive.
- **Opt-in env gate** while prototyping — it adds a proposal source to a live
  autonomous loop; gating avoids surprising existing deployments. Deliberation /
  feasibility / dedup still filter these like any other proposal.
- **Grounded-fact count, not raw learning count** — an episode with 10 concept
  tags but no statements is still thin; counting subjects-with-triples measures
  actual recall value.

## Status / next

- Detection + proposal: implemented, compiles, env-gated.
- **Not yet:** the execution half (a focused "enrich this episode" sub-routine /
  prompt that reads the source and calls `memory_write` with detail-extraction
  emphasis), and validation over real daemon cycles (needs a stable LLM backend).
- Threshold (5) and the 50-episode scan window are untuned heuristics.
