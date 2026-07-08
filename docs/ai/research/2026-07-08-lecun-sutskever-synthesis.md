# What emerges from LeCun 2022 × Sutskever 2025 × akh-medu — a synthesis

- **Date**: 2026-07-08
- **Status**: research note (synthesis, not a plan)
- **Sources**:
  - LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — [full text](2022-lecun-path-autonomous-machine-intelligence-fulltext.md), Seshat `lecun-2022-path-autonomous-machine-intelligence`
  - Sutskever/Dwarkesh, *We're moving from the age of scaling to the age of research* (2025-11-25) — [edited transcript](2025-11-25-sutskever-dwarkesh-age-of-research-transcript.md) ([verbatim captions](2025-11-25-sutskever-dwarkesh-age-of-research-captions-verbatim.md)), Seshat `sutskever-2025-age-of-research-dwarkesh`
  - akh-medu state as of Phase 50 planning (Phases 15/16/17, 41e/41f, 44, 48a, ADR 082/085/086)
- **Context**: written while setting the akh world-model direction — a world model for akhs
  means predicting the consequences of an akh's own actions before executing them
  (LeCun Mode-2 / model-predictive control, not RL).

## 1. The rivals now agree on the diagnosis — akh-medu is a third answer to their shared problem

LeCun 2022 and Sutskever 2025 were on opposite sides of the scaling debate, and the
interesting thing is what happened between the documents: Sutskever *won* the scaling
era, then conceded LeCun's diagnosis. LeCun argued a priori that generative token models
can't represent uncertainty, can't plan, and have shallow ungrounded common sense.
Sutskever, three years later and from the inside, reports the empirical version:
pre-training data is finite, RL-tuned models are jagged, and "these models somehow just
generalize dramatically worse than people — it's super obvious." Same missing piece,
argued from theory and from exhaustion of the alternative.

Where they split is the remedy. LeCun says it's an *architecture* problem (JEPA, world
models, cost modules). Sutskever says it's an unknown *recipe* problem — his
value-function line is telling: value functions make things *efficient*, but "anything
you can do with a value function, you can do without, just more slowly"; the fundamental
thing is generalization itself. akh-medu is a third position on the same diagnosis: put
the world knowledge in an explicit, inspectable substrate (KG + VSA + Dempster-Shafer +
e-graphs) and buy deliberation with cheap discrete search instead of gradients.
Sutskever's "slower but equivalent" concession is exactly the license the symbolic route
needs — it trades gradient efficiency for auditability, and he says that trade is
legitimate.

## 2. Three independent derivations of the same functional decomposition

Line up LeCun's modules, Sutskever's remarks, and what akh-medu has already built, and
the same anatomy appears three times:

| LeCun 2022 | Sutskever 2025 | akh-medu |
|---|---|---|
| world model | (the thing that must generalize) | Phase 15 causal model + event calculus |
| actor as search | — | Phase 16 MCTS |
| trainable critic | value functions | TD values + 41e VOC controller |
| intrinsic cost ("emotions") | "emotions as value functions" | affective system |
| configurator | — | Phase 44 context assembly + schedulers |
| short-term associative memory | — | working memory + episodic store |

These were arrived at from deep learning (LeCun), from scaling-lab experience
(Sutskever), and from the symbolic cognitive-architecture lineage (akh-medu). When three
very different search processes converge on one decomposition, that is evidence the
decomposition is real rather than fashionable. What's unusual about akh-medu is that the
whole loop is *implemented*, just in the discrete regime — the architecture isn't
missing; the learned transition function inside it is. That is precisely the world-model
direction.

## 3. Jaggedness has a mechanistic reading: capability in the wrong representation

Sutskever describes the puzzle (a model aces competition programming, fails an adjacent
trivial task) but doesn't explain it. The 41f scaling-law data suggests an explanation
in miniature: multi-hop performance scaled smoothly with thinking budget, while analogy
was *flat* — because VSA solves analogy in one shot; the operation is native to the
representation. The curve shape diagnosed the substrate (ADR 085).

Generalized: **jaggedness is what you get when one representation (weight space) is
forced to carry every operation; robustness comes from routing each cognitive operation
to a substrate where it's native.** LLMs can't route — everything is the same forward
pass. An akh can: analogy → VSA, equational → e-graph, evidence fusion → DS, consequence
prediction → world model + MCTS. This reframes akh-medu's substrate heterogeneity from
an engineering inconvenience into the direct counter-mechanism to the failure mode
Sutskever calls the most fundamental thing.

## 4. Phase 50 quietly unifies benchmark, world-model calibrator, and learning-from-deployment

Sutskever says the decisive jump will come from models that keep learning on the job,
but not how. LeCun supplies the mechanism: Mode-2 deliberation distilled into Mode-1
policies (amortized inference). akh-medu has the third ingredient neither of them had:
**ground truth**. The Phase 50 harness with deterministic verifiers produces exact
(state, action, predicted-outcome, actual-outcome) tuples. One corpus feeds three
consumers:

1. **score** the akh (benchmark leaderboard);
2. **calibrate** the world model and critic (prediction-vs-actual diff);
3. **consolidate** — deliberated solutions compiled into skills/KG facts so the akh gets
   cheaper at repeated tasks (Mode-2 → Mode-1; existing skills + AgentConsolidation
   machinery).

Design principle to elevate: Phase 50 is not a measurement phase that precedes the
learning phase — it *is* the deployment-learning flywheel.

## 5. The irony in LeCun §4.9: the KG smuggled into the differentiable church

LeCun's own Section 4.9 argues that world state should live in a key-value memory of
*entities*, where an event updates only the affected entries (the bottle moves; the
kitchen and dining-room entries change, nothing else does). That is a knowledge graph
with tensor upholstery. Even the arch-anti-symbolist, when forced to make world-state
tracking concrete, reinvents entity-level structured state with sparse updates. Combined
with his admission that the configurator — the executive — is "the most mysterious
module" (mystery being what the differentiable program can't specify), the paper reads
as inadvertent testimony that the symbolic layer re-enters exactly where LLMs and
gradients are weakest: persistent state and executive control. Those are the two places
akh-medu put its foundations.

## 6. For agentic systems, the world model *is* the safety mechanism

LeCun's immutable-intrinsic-cost-as-guardrail, Sutskever's entire SSI premise, and the
predict-before-execute framing collapse into one statement: in an acting system,
capability and safety are the same module. A world model that predicts action
consequences is simultaneously what makes the akh competent (choose the action whose
predicted outcome scores best) and what makes it safe (refuse the action whose predicted
branch crosses an immutable cost). With DS-massed branches you get the honest version —
"I can't rule out the bad branch" is a first-class, *auditable* reason not to act, which
no end-to-end policy can give you. The alignment of all three sources on this point is
the strongest single argument for prioritizing the world model after Phase 50 lands.

## 7. The meta-point: the age of research is an argument for this project's existence

Sutskever's thesis — the frontier moved from compute to ideas, small clarifying
experiments beat giant runs, taste matters — describes a regime where a one-person
symbolic-cognition lab with a measure-first culture (Phase 40's "persist nothing"
decision, 41f's scaling law, 44j's A/B harness) is playing the actual game, not a toy
version of it. And the thread that produced this note was itself the akh workflow
performed by hand: primary sources pulled into a durable corpus, cross-linked, then
reasoned over for emergent structure — KG ingestion, provenance, deliberation. The
architecture is being dogfooded at the process level before it exists at the code level.

## One-sentence synthesis

Both camps now agree the missing piece is a generalizing world model and that scale
won't provide it; akh-medu's wager is that the world model can be explicit rather than
learned end-to-end, its heterogeneous substrates are a mechanistic answer to jaggedness,
and Phase 50 converts that wager into a measurable flywheel — prediction vs. verified
outcome — which is the experiment neither LeCun nor Sutskever is in a position to run.
