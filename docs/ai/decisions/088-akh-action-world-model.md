# ADR 088: The Akh Action World Model — predict-before-execute as capability *and* safety

- Status: **Proposed** (draft 2026-07-08; gated on Phase 50 harness landing)
- Phase: successor to Phase 50; builds on Phases 15/16/17, 41e, 48a
- Relates to: [synthesis note](../research/2026-07-08-lecun-sutskever-synthesis.md),
  [ADR 082](082-privilege-separated-executor-backend.md) (executor Backend seam),
  [ADR 086](086-feature-test-vm-harness-benchmarks.md) (Phase 50 harness),
  [ADR 085](085-thinking-engine-findings.md) (41f scaling law)
- Sources of record: LeCun 2022 (`lecun-2022-path-autonomous-machine-intelligence`),
  Sutskever 2025 (`sutskever-2025-age-of-research-dwarkesh`) in Seshat

## Context

An akh is an *acting* system: it issues tool calls (shell, file I/O, edits, remote
exec via the ADR 082 backends) that change a real environment. Today it acts, then
observes what happened. It has no faculty for asking *what will happen if I do this?*
before committing the action.

Two external sources, archived in the corpus, converge on the same gap:

- **LeCun 2022** argues the core of an autonomous agent is a learned, configurable
  **world model** used in a Mode-2 loop: propose an action sequence → roll it forward →
  score predicted outcomes with a cost module → execute only the lowest-cost sequence.
  This is model-predictive control, not reinforcement learning — the agent avoids
  dangerous actions by *predicting* their consequences, not by trying them.
- **Sutskever 2025** concedes, from inside the scaling era, that models "generalize
  dramatically worse than people" and that scale alone will not close the gap; the
  frontier is now new ideas, not bigger runs.

The [synthesis note](../research/2026-07-08-lecun-sutskever-synthesis.md) draws out two
load-bearing claims that this ADR promotes from observation to decision:

1. **The world model is simultaneously the competence mechanism and the safety
   mechanism.** Choosing the best-scoring predicted outcome is capability; refusing an
   action whose predicted outcome crosses an immutable cost is safety. They are the same
   module.
2. **Phase 50 is not merely a benchmark; it is the ground-truth flywheel.** Its
   deterministic verifiers can supply exact (state, action, predicted-outcome,
   actual-outcome) tuples — the one thing neither LeCun nor Sutskever had — turning
   predict-before-execute into a *measurable, self-improving* loop.

akh-medu already implements the LeCun decomposition in the discrete regime (Phase 15
causal model + event calculus = transition function skeleton; Phase 16 MCTS = actor
search; Phase 17 Dempster-Shafer = branch uncertainty; 41e VOC = critic-style halting;
affective system = intrinsic cost). What is missing is the transition function wired
into the act loop. This ADR specifies that faculty.

## Decision

### 1. Add an Action World Model that predicts KG-level state deltas of a candidate action

Before executing a tool call, the akh consults an **Action World Model (AWM)**:
`predict(state, action) → {outcome_branch}`. The prediction target is a **KG-level state
delta** — the set of asserted/retracted facts the action is expected to produce (e.g.
`service:nginx status→running`, `file:/etc/x mtime↑`, `pkg:foo installed`) — **not** raw
stdout, exit codes, or byte-level effects.

Rationale (LeCun §4.4/§4.5, "predict in representation space"): raw output is
high-entropy and largely unpredictable (timestamps, log noise, formatting); the
*meaningful* consequence is low-entropy and lives at the entity level. Predicting the
delta lets the model be confident about what matters and silent about what doesn't.

### 2. Represent the unpredictable remainder as Dempster-Shafer-massed outcome branches

A single (state, action) does not determine one outcome — the package may not exist, the
network may be down, a file may already be present. The AWM emits a small set of
**plausible branches**, each carrying a DS mass (Phase 17), rather than a point estimate
or a normalized probability. This is LeCun's latent-variable multimodality expressed in
akh-medu's native evidence calculus.

The honest failure mode is first-class: **"I cannot rule out the destructive branch"** is
an auditable, mass-backed reason to *not act* — something no end-to-end policy can
produce.

### 3. Cost is split: immutable intrinsic cost (safety) + trainable critic (competence)

Following LeCun §3.2 and echoed by Sutskever's "emotions as value functions":

- **Intrinsic cost (immutable):** hard safety constraints wired to the destructive-action
  guard. If *any* branch above a mass threshold crosses an intrinsic-cost boundary
  (data loss, irreversible mutation, scope escape), the action is refused or escalated —
  regardless of expected benefit. Immutable so behavior cannot drift or be trained around.
- **Trainable critic:** learned "this kind of action tends to end badly here" estimates,
  sourced from episodic memory and Phase 50 outcomes. Speeds up good action selection;
  never overrides the intrinsic guard.

### 4. Capability and safety are the same evaluation

The actor (Phase 16 MCTS over the discrete tool-call space — LeCun explicitly endorses
tree search where the action space is discrete) scores candidate actions by expected cost
over predicted branches, optionally penalizing branch *variance* to minimize risk. The
selected action is the one whose predicted outcomes score best **and** clear the intrinsic
guard. There is no separate "safety check" bolted on afterward; the guard is a term in the
same objective.

### 5. Phase 50 is the ground-truth calibrator and the deployment-learning flywheel

The Phase 50 harness (ADR 086) runs sysadmin tasks in `vmctl` VMs with **deterministic
verifiers**. Every executed action yields the actual post-state. This makes the AWM
*measurable*: log `predict(state, action)`, execute, diff predicted delta vs. verified
actual delta. One corpus of (state, action, predicted, actual) tuples feeds three
consumers:

1. **Score** — AWM prediction accuracy becomes a leaderboard axis alongside task success.
2. **Calibrate** — the diff trains/tunes the AWM and the trainable critic; DS masses are
   calibrated against observed branch frequencies.
3. **Consolidate** — a deliberated Mode-2 plan that succeeds is compiled into a skill / KG
   facts (existing skills + `DerivationKind::AgentConsolidation` machinery), so the akh
   gets cheaper at repeated tasks. This is LeCun's **Mode-2 → Mode-1 distillation**
   (amortized inference) and Sutskever's **learning-from-deployment**, realized with the
   ground truth neither of them had.

Design principle to elevate: **Phase 50 does not precede the learning phase — it *is* the
learning loop.**

### 6. Scope: opt-in, KG-level, act-loop only

- The AWM is consulted in the agentic act loop before backend-mutating tool calls
  (shell_exec, file_io, edit). Read-only calls (glob, grep, read) bypass it.
- Gated behind a flag (proposed `AKH_ACTION_WORLD_MODEL`, default off until Phase 50
  calibration exists); the intrinsic-cost guard may ship enabled independently.
- Prediction lives at KG abstraction. No attempt to predict raw command output.

## Consequences

**Enables**
- Predict-before-execute: dangerous actions are caught by prediction, not by damage.
- Auditable refusals grounded in DS mass, not opaque policy.
- A measurable world model — a research instrument, per the age-of-research thesis.
- A concrete mechanism for the akh to improve on repeated tasks without weight training.

**Costs / risks**
- Latency: every mutating action now pays a predict + score step (bounded by Mode-2→Mode-1
  distillation — hot paths compile to reactive skills).
- Prediction quality bootstrapping: before Phase 50 supplies calibration data, the AWM is
  weak; the intrinsic guard must carry safety alone in that window.
- A miscalibrated AWM that is *over*-confident about a benign branch is more dangerous than
  no AWM. Mitigation: the intrinsic guard triggers on *any* above-threshold destructive
  branch, and calibration is measured continuously against verifier ground truth.

## Open questions

- Where does the transition function live — extend the Phase 15 causal model, or a
  separate predictor keyed by tool + argument shape? (Lean toward extending Phase 15;
  measure first, per Phase 40 discipline.)
- Branch enumeration policy: how many branches, and how are they proposed (KG neighborhood
  of the action's effect predicates, prior episodes, or a small LLM proposer)?
- Intrinsic-cost boundary specification: declarative predicate set vs. learned-from-labels;
  ADR 082's "OS as sole security boundary" is the backstop, but the AWM guard is
  defense-in-depth *above* it.
- Does the AWM subsume or complement the existing destructive-action confirmation prompt?

## Alternatives considered

- **RL from environment reward** (learn "don't do the bad thing" by doing it) — rejected:
  Sutskever's sample-efficiency argument and the fact that akh actions are real and often
  irreversible. Predict-before-execute is the whole point.
- **Learned end-to-end policy with implicit safety** — rejected: no auditable refusal, and
  it re-creates the jaggedness the synthesis note attributes to single-representation
  systems.
- **Raw-output prediction** — rejected: high-entropy, low-value; LeCun's representation-
  space argument. Predict the KG delta.
- **Ship nothing until the AWM is strong** — rejected: the intrinsic-cost guard is valuable
  on day one and can ship ahead of the predictive machinery.
