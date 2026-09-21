**Yes—highly relevant to Kaine’s exploration loop, and useful for ADAM’s surrounding harness now.** It gives us a concrete way to improve how the system spends attempts before changing model architecture.

The paper is [*Dream-RSI: Recursive Self-Improvement through Evolving Worlds*](https://arxiv.org/abs/2609.14858), from Google, DeepMind and university researchers.

**What actually improves**

An agent records its discovery attempts as a tree: starting state, attempted solution, evaluation and descendants. A separate agent revises executable exploration-policy code—when to continue a branch, open another, batch attempts or stop. Candidate policies replay recorded outcomes cheaply; the winner guides new online exploration. Model weights remain fixed. [Paper, §3](https://arxiv.org/html/2609.14858v1#S3)

The important limitation: **replay only covers recorded possibilities.** It cannot establish what an untried action would accomplish. The guarantee of improvement applies to the historical replay score, not future performance. [Paper, §3](https://arxiv.org/html/2609.14858v1#S3)

The headline needs context:

- **162× fewer discovery calls** compares against SimpleTES using a different underlying model.
- The controlled Lasso comparison is **550 → 317 calls**, about **1.74× fewer**, with better average downstream solver runtime.
- Discovery-call counts do not establish equivalent savings in total dollars, tokens or policy-development work. [Paper, §4.1](https://arxiv.org/html/2609.14858v1#S4.SS1)

**Where I would apply it**

I checked the current [architecture ladder](/Users/toasty/ws/akh-medu/docs/ai/decisions/111-kaine-architecture-ladder.md) and [character/environment design](/Users/toasty/ws/akh-medu/docs/ai/decisions/110-kaine-character-guided-reward-live-environment.md).

| Area | Relevance | Proposed application |
|---|---|---|
| **ADAM inference** | High for tasks needing multiple attempts | Improve retry, branch and stopping choices around a frozen checkpoint. Measure whether the same model completes more tasks per budget. |
| **ADAM training** | Indirect | Use successful exploration traces as potential corrected-sibling material. Selecting training exercises this way would be a separate extension requiring validation. |
| **Kaine exploration** | Very high | Add an engine-owned controller that learns how to allocate effort across hypotheses and recovery paths. |
| **Backbone architecture** | Low | This provides no reason to change the transformer/hybrid ladder. |

For Kaine, this connects neatly with today’s foresight and curiosity discussions. My proposed division is:

- **Foresight:** propose a hypothesis and an experiment that could distinguish it.
- **Curiosity:** estimate which unfamiliar problem is worth learning about.
- **Replay-based exploration:** test which allocation of attempts has worked across prior searches.

That combination remains our research proposal; this paper does not validate the combined system.

It also fits the engine-as-captain boundary. Keep policy versions, observations, permissions and verification in the engine. Preserve ADR 110’s separate effect, validity, provenance, disposition and budget signals. A controller-selection objective need not become a new scalar reward for training Kaine.

**The first experiment I’d prioritize**

Use a frozen ADAM checkpoint on a small sandboxed repair-task set. Record branching attempts with parent snapshots, diagnostics, verified outcomes and costs. Compare:

1. A fixed retry/branch policy.
2. A simple hand-written adaptive policy.
3. A policy improved through historical replay.

Split by **whole task**, hide unrevealed outcomes, and test the selected policy on **fresh online episodes**. Count policy-development cost too. Success means equal or better verified completion at lower total cost, without regressing the existing gates.

The main engineering trap is replaying stateful workflows: changing action order can change later outcomes. Start with isolated branches restored from snapshots; ordinary chronological tool logs are insufficient.

**I would prioritize a bounded replay-controller experiment alongside P7’s environment work.** The mechanism is concrete, but the authors’ [repository](https://github.com/zhengkid/Dream-RSI) still lists the full implementation and reproduction scripts as pending, so this is currently an adaptation opportunity rather than a ready-to-run dependency.

<oai-mem-citation>
<citation_entries>
MEMORY.md:49-51|note=[Prior ADAM experiments and Kaine reward boundaries checked against current ADRs]
</citation_entries>
<rollout_ids>
01a0a98e-10f7-7af3-9aae-70685457a1a2
</rollout_ids>
</oai-mem-citation>
