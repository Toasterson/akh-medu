# Exploration policy in the loop and Dream-RSI implementability (2026-09-18 thread)

Research harvest of T3/Codex thread `6adfb261-5682-4d80-89fc-4875480d9876`, requested by the user on 2026-09-19 after the thread had
been missed by the ADR 112 sweep (it was never archived; only the T3 provider log existed). Start with the
[conversation](transcripts/conversation.md).

## Contents

- `transcripts/conversation.{json,md}`: exact user-facing thread snapshot; source and cutoff in `context/transcript-provenance.json`.
- `context/git-head.txt`: repository HEAD at harvest time.
- `seshat/`: ingest receipt for the conversation.
- `SHA256SUMS`: integrity manifest for every other archived file.

## What the thread established (verified against source on 2026-09-19)

- Exploration today is three unrelated mechanisms with no shared policy: the agentic LLM loop (`src/agent/agentic_chat.rs`,
  `RoundBudget`: iteration backstop + wall-clock cap + repeat-loop detection; one conversation, no branch comparison), the OODA
  ranking (`src/agent/ooda.rs`: relevance, novelty bonus, recency penalty, prior success; UCB1 under `VsaNative`), and the NLU
  backend bandit (`src/nlu/mod.rs:430-467`) whose `CloudApi` and `BurnLocal` arms are unimplemented and fall back to the cascade.
- Dream-RSI (arXiv 2609.14858) fits the missing layer for any backing: improve exploration-controller *code* by replaying recorded
  discovery trees, weights frozen; evaluate cloud and local backends separately; faithful replay of stateful tool execution is the
  hard part; start with isolated, restorable code-repair tasks.

## Where it went

Folded into [ADR 112](../../decisions/112-kaine-theory-first-policy-training-inference-environments.md) (starting facts, D5) and
work item P7.18 (`01a0bb18-32fe-7830-9358-aee23664aee8`) in the
[P7 extension plan](../../plans/2026-09-19-phase51-p7-kaine-theory-first-policy.md). No training, deployment, dispatch or
architecture change is part of this harvest.

## Related archives

- [Dream-RSI for ADAM and Kaine](../2026-09-16-dream-rsi-adam-kaine/README.md) (paper, sources, analysis).
- [Kaine foresight and compression-progress curiosity](../2026-09-16-kaine-foresight-curiosity/README.md).

## Seshat retrieval

Search for `exploration policy loop cloud local Dream-RSI`. Record slug: `exploration-policy-dream-rsi-2026-09-18-conversation`.
