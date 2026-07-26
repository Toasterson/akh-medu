# Tecton capability exercise — episodic memory, retrieval+action, context managers

> Date: 2026-06-23
> Status: Active
> Driver: live over ACP via the Python adapter (`scripts/acp_adapter.py`), not the CLI.
> Subject: akh `tecton` (GLM via ollama.com cloud), 38 OpenClaw journals + 7 skills imported & durable in PG.

## Goal

Prove the storage we built is *usable*: that Tecton's episodic memory **grows**, that he can
**retrieve** the right memory and **act** on it, and measure how the three context-manager
strategies affect this. Plus a feasibility probe of two agentic-coding benchmarks.

## Harnessing

- **Driver:** `scripts/acp_adapter.py` (`AkhClient`) — line-JSON ACP over tcp:8201, streams
  `agent_thought_chunk` (reasoning + `[tool:status]` steps) and `agent_message_chunk` (answer).
- **Strategy switch:** `launchctl setenv AKH_CONTEXT_STRATEGY <compacting|passive|virtual>` then
  `kickstart -k`; confirm via daemon log `context manager attached strategy=…`. Restore to
  `compacting` at the end.
- **Truth probe:** PG over mTLS (`akhomed_mutation_log` WAL seq = live deltas; `akhomed_symbols`
  `episode:*` = materialized episodes). Episode growth measured here, not via the lagging recall view.

## Tracks

### T1 — Episodic memory growth
Measure `(WAL max seq, episode-symbol count)` before/after a batch of fresh conversational turns
that contain memorable facts. Expect: new `episode:*` symbols appear and WAL advances. Confirms
the consolidation path writes episodes for live dialogue (not only the bulk import).

### T2 — Retrieval + action (grounded in the 38 journals)  ← the core test
Auto-generate probes from a sample of the imported journals: each probe is a question whose answer
lives in one specific journal (e.g. cluster topology, the May-9 Switch-2 outage, the
`agent/memory.rs:456` UTF-8 panic, the observability alert routing). Drive each as a **fresh
session** (so the answer must come from KG recall, not conversation carry-over). Score per probe:
- **retrieved**: did Tecton invoke `memory_recall` (visible in `tool_steps`)?
- **correct**: does the answer contain the expected fact/keyword(s)? (keyword + LLM-judge)
- latency, tools used.
Report accuracy = correct / total; retrieval-rate = retrieved / total.

### T3 — Context-manager comparison
Run T1 + T2 under each of `compacting`, `passive`, `virtual`. Compare retrieval-rate, answer
accuracy, episode growth, latency, and tool mix (esp. `virtual`'s `archival_search` /
`core_memory_*`). Output a small table.

### T4 — External memory benchmarks
- **LoCoMo** — multi-session dialogue QA; load the public JSON, replay each conversation as
  sequential turns into one ACP session, then ask the recall questions; score EM/F1 per strategy.
- **LongMemEval** — 500 long-term-memory questions (extraction, multi-session, temporal,
  knowledge-update, abstention); larger haystacks stress the managers harder.
Both adapted to drive Tecton over the adapter; scored per strategy. (Download + license verified
before fetch.)

### T5 — Agentic coding (feasibility, separate harness)
**SWE-Bench Pro** + **FrontierCode (Diamond)** — frontier agents score low (≈5–30% on Diamond).
Build a *minimal* ACP coding harness: repo read + scratch-write + run-tests exposed to Tecton via
tools; run a small slice (10–20 instances) as a feasibility probe, not a leaderboard run.
Availability/licensing of both datasets to be web-verified first; fall back to a SWE-bench Verified
slice if Diamond/Pro aren't publicly fetchable.

## Order of execution
1. T2 harness on `compacting` (prove retrieval+action end-to-end live). 2. T1 growth check.
3. T3 sweep across all three strategies. 4. T4 external benchmarks. 5. T5 coding feasibility.
Each track reports before moving on; restore `compacting` + clean any scratch state at the end.
