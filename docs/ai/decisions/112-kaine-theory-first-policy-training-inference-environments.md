# ADR 112 — Kaine theory-first policy: theories live in the engine, the model learns to conjecture, predict, intervene and revise; one policy stack serves training and inference

> Date: 2026-09-19
> Status: Proposed (architect-session; every work item `draft`, HITL, after the P6 window queue and gate — same gate as ADR 110/111)
> Phase: 51 P7 extension (plan: `docs/ai/plans/2026-09-19-phase51-p7-kaine-theory-first-policy.md`)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `01a0a24e-062f-7881-9fd6-1d2017d52b34` (P7); WIs P7.14–P7.18 filed 2026-09-19 (IDs in the plan)
> Related: ADR 005 §2 (compression-progress curiosity), ADR 033 (counterfactuals), ADR 035 (Dempster–Shafer), ADR 052 (thinking loop), ADR 117 (action world model, predict-before-execute), ADR 104 (copy-faithful decoding), ADR 105 (run governance), ADR 110 (character, guided reward, live environment), ADR 111 (architecture ladder)
> Evidence: T3/Codex thread `6adfb261-5682-4d80-89fc-4875480d9876` (2026-09-18, exploration policy + Dream-RSI, not archived; findings folded into the starting facts and D5); papers sourced through `paper-search` on 2026-09-19 (§"What the literature establishes"); five Codex-Astra research harvests of 2026-09-16 under `docs/ai/research/2026-09-16-*` and the measured `experiments/typed-decisions/` artifact; code map in §"Starting facts"

## Question

Felin and Holweg (2024) argue that data-driven prediction is backward-looking and imitative, while human cognition is
forward-looking: it holds beliefs that outrun the data, builds a causal theory, and runs directed experiments that *generate*
the data that did not exist. Can Kaine get that faculty, and what training environment, inference environment and policy
machinery make it trainable and measurable — given the 2026-09-16 Astra findings on typed decisions, replay-improved
exploration and curiosity?

## What the literature establishes

Sources were fetched in full text where open access allowed; the Felin and Holweg text is archived in Seshat and under
`docs/ai/research/2026-09-16-kaine-foresight-curiosity/papers/` (CC BY-ND 4.0).

1. **Felin & Holweg 2024**, *Theory Is All You Need*, Strategy Science 9(4):346–371, DOI 10.1287/stsc.2024.0189.
   *Data–belief asymmetry*: "forward-looking beliefs … are more in search of data rather than based on existing data"; "it is
   hard to ex ante distinguish between beliefs that indeed are delusional versus those that simply are ahead of their time" —
   the discriminator is the experiment, not confidence. A theory is a problem decomposition plus a causal logic plus directed
   experimentation: the Wrights reduced flight to lift, propulsion and steering, distrusted Lilienthal's lift coefficients,
   built a wind tunnel and "measured everything … data that they generated through ongoing experimental manipulation".
   Polanyi's line carries the design: "things are not labeled evidence in nature" — the theory decides what counts as data.
   The paper is a conceptual argument; it contains no impossibility theorem (the tweet's claim is the tweeter's).
2. **Felin, Gambardella, Novelli & Zenger 2024**, *A Scientific Method for Startups*, J. Management, DOI
   10.1177/01492063231226136. Generative rationality vs bounded rationality: "there is no 'lookup table' for the right answer";
   theories are "what-if forms of causal logic"; prescribes a *discriminating alignment* between the kind of theory and the kind
   of experiment.
3. **Ehrig & Schmidt 2022**, *Theory-based learning and experimentation*, SMJ, DOI 10.1002/smj.3381. Formulate a theory as the
   necessary conditions for an envisioned future; test the cheap necessary conditions to draw inferences about elements that
   cannot be tested directly ("learning from arguments"). This is the experiment-selection rule below.
4. **Ehrig & Zenger 2024**, *Competing with Theories*, Strategy Science, DOI 10.1287/stsc.2024.0177 (full text read).
   Awareness ≠ information ≠ confidence: "One cannot improve knowledge of the odds of A being true without first being aware of
   A." Theories define the state space; beliefs define probabilities over it. Expanding the hypothesis space is a distinct act
   from updating belief inside it.
5. **Valentine, Novelli & Agarwal 2024**, Strategy Science, DOI 10.1287/stsc.2024.0183 (RCT, 261 entrepreneurs, 1,600
   interviews). Theorization and experimentation are *complements*: each alone is negatively associated with focused radical
   pivots, the interaction is strongly positive. Consequence for reward design: never credit a conjecture without its
   experiment, nor an experiment without its conjecture.
6. **Felin & Koenderink 2022**, *A Generative View of Rationality and Growing Awareness*, Front. Psychol., DOI
   10.3389/fpsyg.2022.807261. Organisms are "probing", not "intuitive statisticians"; cues become clues only through
   organism-directed probing ("cue-to-clue transformation").
7. **Murphy 2026**, *Model Discovery Agent*, arXiv 2608.09696 (full text read). The ML operationalization of the same loop:
   a posterior over mechanistic models (SMC over models, parameters, latents); the LLM proposes a *new* mechanism only when a
   predictive check fails (M-open expansion); experiments chosen by expected information gain / value of information; evaluation
   is **out-of-sample interventional prediction accuracy** (Richens & Everitt 2024: doing well on it implies a causal world
   model). In-context prediction from the same data fails where the explicit model succeeds; "prediction error tracks form
   recovery".
8. **Schmidhuber 2009** (compression progress; already mapped in ADR 005 §2) and **Poli et al. 2024**, *Curiosity and the
   dynamics of optimal exploration*, TICS, DOI 10.1016/j.tics.2024.02.001: curiosity as a common currency over uncertainty,
   information gain and learning progress, balanced against other drives.
9. **Zheng et al. 2026**, *Dream-RSI*, arXiv 2609.14858 (harvest `2026-09-16-dream-rsi-adam-kaine`): exploration-policy *code*
   improved offline by replaying recorded discovery trees while weights stay frozen; replay "cannot establish what an untried
   action would accomplish".
10. **Jev ecosystem + `experiments/typed-decisions/`** (harvests `2026-09-16-jev*`, `2026-09-16-typesafe-jev-adam-kaine`):
    measured on synthetic panels — a 240k-parameter head over a frozen MiniLM hits 719/720 in-distribution and 51 % on
    reworded urgency; joint scoring of complete outcomes puts 13 % mass on unsupported outcomes vs 50 % from multiplied
    marginals; every end-to-end replay is won by the deterministic host verifier (receipt lookup 48/48), never by a learned
    head; calibration is per schema and "not a transferable safety certificate". No head is ready for a live path.

## Starting facts (code, 2026-09-19)

| seam | where | status |
|---|---|---|
| Causal transition model | `src/agent/causal.rs` `CausalManager::{predict_effects, verify_prediction, applicable_actions}` | live; E3 reward source |
| Counterfactual abduct→intervene→predict→compare, `PredictionTracker` (predict-before-execute accuracy, EMA) | `src/agent/counterfactual.rs`, tracker held in `Agent` (`src/agent/agent.rs:236`) | live, tracker unused by the LLM path |
| DS-massed hypotheses with belief interval | `src/thinking/state.rs` `Hypothesis { mass: MassFunction }`, `belief()`/`interval()` | live |
| Hypothesis forking/merging | `src/infer/superposition.rs`; Phase 29f explorer surface | unexposed |
| Epistemic modality (Knows / Believes / ConsidersPossible) | `src/agent/epistemic.rs` | live |
| Directed curiosity (information-gap, ZPD) | `src/agent/curiosity.rs`, `DerivationKind::DirectedCuriosity` | live (Phase 11j) |
| Action world model, predicted KG delta, DS branches, intrinsic-cost guard | ADR 117 | proposed, gated on Phase 50 |
| Decide step | `src/agent/ooda.rs` `Decision { chosen_tool, tool_input, predicted_outcome_vec }`, `DecideMode::{VsaNative, LlmDriven}` (`vsa_decide.rs`: VSA ranking + UCB1/MCTS fallback) | live |
| Planner / values / active inference | `mcts.rs` (`MctsPlanner` over `CausalManager`), `state_value.rs` (`ValueFunction`), `active_inference.rs` (`Policy { actions, efe }`, EFE softmax) | live |
| Halting controller | 41e VOC two-regime rule; WI-399/400 (B3) | live / draft |
| Exploration today (three unrelated mechanisms, no shared policy) | agentic LLM loop `src/agent/agentic_chat.rs` (`RoundBudget`: iteration backstop + wall-clock cap + repeat-loop detection; one evolving conversation, no branch comparison); OODA `src/agent/ooda.rs` (relevance + novelty bonus + recency penalty + prior success; `VsaNative` adds UCB1); NLU backend router `src/nlu/mod.rs:430-467` (contextual bandit, but the `CloudApi` and `BurnLocal` arms are unimplemented and fall back to the cascade — not a working cloud-vs-local exploration controller) | live; nothing learns whether branching, revisiting or abandoning would have helped |
| Permission and executor posture | `permission.rs` `PermissionPolicy`, `src/backend/` `Confinement`/`Posture` (ADR 082) | live, rule-based |
| Training loop | `crates/grug-train` `train_knowledge --remediate-from-bundle --record-weights use`; `KnowledgeSftRecord.weight` | live (P6.4/P6.5) |
| Episode producer | `src/bin/akh_episode_generator.rs` (frozen `ToolContractSnapshot`, sentinels, E1–E4) | live |
| Training environment `AkhEnvironment` / dojo | P7.10 (WI-456) | design only, no code |
| Gates | `certify_knowledge_model` (knowledge + tools), `eval_tool_contract_exam`/`eval_constrained` (copy), P7.13 environment gate | live / draft |
| Inference backing | `ChatBacking::Grug` → `GrugChatBackend` (native Burn, certified `active-model.json` only, constrained copy-faithful decoding, live `ToolContractSnapshot` + `ContractDriftJournal`) | live; drift is journaled, not closed |

## Decision

### D1 — A theory is an engine object; Kaine learns the procedure, not the theories

Add `Theory` to the agent module (`src/agent/theory.rs`), built from existing seams and stored as regular Entity symbols with
`agent:` predicates (no new `SymbolKind`):

- `problem`: the goal symbol and its decomposition into sub-problems (Wright: lift, propulsion, steering).
- `claims`: `CausalRelation` triples asserted in a microtheory context with `EpistemicModality::Believes`, never `Knows`.
- `necessary_conditions`: ordered by test cost (Ehrig & Schmidt); each one a fluent the event calculus can project.
- `support`: the Dempster–Shafer mass over the claims; **the data–belief asymmetry is the interval** `Pl(H) − Bel(H)`. A
  Wright-style theory is low `Bel`, high `Pl`: the data argue against it, nothing excludes it. A theory whose `Pl` has collapsed
  is dead, whatever the conviction.
- `experiment`: a candidate action (from `applicable_actions` + ADR 117 branches) whose predicted divergent fluents
  (`counterfactual_query`) separate the theory from the null. Selection rule: the cheapest necessary condition whose outcome
  most reduces the interval (expected information gain over the live hypotheses; Murphy's EIG in DS terms).
- `revision`: the TMS retraction / assertion that the observed outcome forces (reuses the E4 exact-owned revision curriculum).

Provenance records the loop: `DerivationKind::TheoryConjecture` and `DerivationKind::DirectedExperiment` (new variants,
explainable through `akh acp explain`). Weights never hold theories; the engine remains the store of record (ADR 111 §6).

### D2 — One policy stack, engine authority, "enumerate → choose → spell → verify"

The Astra typed-decision evidence fixes the division of labour:

1. **Enumerate (engine).** The legal candidate set is *complete actions*: tool + arguments from the `ToolContractSnapshot`,
   ADR 117 outcome branches, and the theory's candidate experiments. Never independent fields scored separately.
2. **Choose (model, engine-ranked).** Constrained generation over the enumerated set stays the primary chooser. Engine rankers
   (`VsaNative` + MCTS, EFE, VOC) supply ordering and pruning. A typed-decision head is a **shadow path only** (P7.15), promoted
   to a live path solely if it beats constrained generation on a sealed frozen-ADAM panel with no generation regression.
3. **Spell (model).** Copy-faithful constrained decoding under the runtime grammar (ADR 104), unchanged.
4. **Verify (engine).** Completion is gated by the host verifier: engine replay, `verify_prediction`, receipts, `PermissionPolicy`
   and backend `Posture` — never by a learned confidence. Calibration numbers are per schema and do not travel.

**Foresight sits between 2 and 3.** Every mutating action carries a `predicted_delta` (KG-level, ADR 117) before execution;
`PredictionTracker` records the match after execution at training *and* inference time.

### D3 — Reward vector: add `foresight`; keep learning progress out of the reward

ADR 110's vector `{effect, validity, provenance, disposition, budget}` gains a sixth component:

- `foresight ∈ [0, 1]`: the pre-committed `predicted_delta` matched the verified delta. Weighted by prior surprise so a verified
  low-prior prediction earns more than a verified obvious one; the exact function is fixed in P7.16 and versioned in
  `weight_policy`. A miss is **0 with a corrected sibling** whose prediction is the verified delta — never a penalty.
- A conjecture earns `foresight` only through its experiment, and an experiment earns it only under a stated conjecture
  (Valentine et al.: complements). Contrarianism alone scores nothing.
- **Learning progress** (compression progress, ADR 005 §2; Phase 11j) is a *curriculum and experiment-selection* signal, not a
  reward component. This keeps ADR 110's rule: weighted SFT + sibling contrast, no scalar-judge RL. Adopting a full
  compression-progress RL controller would be a separate decision.

### D4 — Third episode family: theory-first

P7.12 adds `theory_first` next to the emergent-property-bug and act-to-perceive families, generated in `AkhEnvironment`:
the KG is seeded with majority evidence for X; the engine-owned mechanism makes ¬X true on a hidden fluent. The correct
trajectory is `conjecture → predict → intervene → observe → revise → next experiment`; the passive control answers from
retrieval. A retrieval-only attempt trains at weight 0 with the theory-first trajectory as its corrected sibling. Disposition
labels: Curiosity (Explorer, slot 3), Courage (Creator, slot 1; supporting Candor in slot 4 Rebel), Rigor (Sage, slot 2).
Sentinel decontamination applies as for every family.

> 2026-09-27: the Courage label was corrected from "Rebel" to match ADR 110 Decision 1, which is canonical for the value→slot
> mapping (Courage primary in slot 1 with Optimism/Urgency, supporting slot 4 Candor).

### D5 — Exploration-policy code improves by replay, weights stay frozen

The engine-side controller (VOC halting, branch/continue/stop, experiment selection) is code. Recorded discovery trees from
the dojo form a replay corpus; candidate controllers are scored offline (Dream-RSI) before an online dojo run validates them,
because replay covers only recorded branches. This is the cheap, auditable way to "learn to explore" without training the
model on exploration. The controller is engine code, so it serves every backing — cloud akhs and frozen local checkpoints alike.
Discovery records carry parent attempt, environment snapshot, model/prompt version, proposal, verified outcome and cost; the
replay evaluator exposes only outcomes the candidate policy has actually reached and marks unrecorded branches unknown;
backends are evaluated separately with the model identity recorded, because a better model can masquerade as a better
exploration policy. Faithful replay of stateful tool execution is the hard part, so the first tasks are isolated code-repair
and algorithm tasks whose branches can be restored and objectively verified.

### D6 — Training environment (assembled)

`AkhEnvironment` (P7.10) = real `Engine` + full `ToolRegistry` + Agent Board + Anima + executor `Backend`, sandboxed dojo
tenancy, per-step JSONL recorder. Producers inside it: `akh_episode_generator` (frozen contracts, sentinels, E1–E4), the
ADR 117 world model for `predicted_delta`, `Theory` objects (D1), the replay corpus (D5). Trainer: `train_knowledge
--record-weights use` with sidecar v1 (P7.11) extended to v1.1 by `foresight`. Gates: knowledge, tools, copy, the P7.13
environment gate, plus a new **interventional-prediction axis** (held-out `predicted_delta` accuracy on unseen interventions,
Murphy's metric) recorded as a P7.13 row.

### D7 — Inference environment (assembled)

The same code path serves: `backend = "grug"` loads only a certified `active-model.json` into `GrugChatBackend`; constrained
decoding; the live `ToolContractSnapshot` is compared to the trained one every turn and drift is journaled. Predict-before-
execute runs under `AKH_ACTION_WORLD_MODEL`; `PredictionTracker` foresight accuracy is emitted as telemetry so deployment
tuples feed the next window (the ADR 117 flywheel). Theory-first behaviour at inference needs the Phase 29f explorer exposed
as tools (`hypothesis_fork`, `hypothesis_merge`, `hypothesis_confidence`, `theory_propose`, `experiment_select`) and those
tools present in the frozen contract so train and inference see one surface (P7.17). Known gap, kept: contract drift is
journaled, not closed, until a remediation window re-certifies against the live contract.

### D8 — Guards and what this ADR does not decide

- A DS interval is not a calibrated probability; no threshold learned on one schema is reused on another.
- No scalar-judge policy gradient; no learned head in a live path without the P7.15 verdict; host verification stays
  authoritative.
- Kaine's size and backbone are untouched (ADR 111); this ADR adds objects, signals, families and tools, not layers.
- Whether the compression-progress controller becomes an RL controller (ADR 005 §2 open item) is not decided here.
- All items are `draft`, `source=architect-session`, after P6, promoted by the operator (memory rule: P7 gated on P6, HITL).

## Consequences

- Kaine can be measured on the one number the literature says matters for a causal world model: out-of-sample interventional
  prediction accuracy — not on how often its answers match the data it was shown.
- Novelty comes from the intervention loop producing data that did not exist, which is exactly the paper's mechanism; the 33M
  ADAM rung will not invent flight, but it can learn to predict before acting, to hold a mechanism-backed belief against the
  evidence, and to spend a tool call on an experiment instead of a retrieval.
- Our evidence calculus is closer to the authors' position than a Bayesian prior: explicit ignorance mass is the room a theory
  lives in (cf. Ehrig & Foss 2022 on going beyond Bayesianism).
- Every new signal is auditable to an engine verification, in keeping with ADR 110's "a weight of 0 means this taught nothing".

## Alternatives considered

- **Train a "theory head" or a curiosity reward into the weights** — rejected: ADR 111 keeps memory in the engine; ADR 110
  forbids scalar-judge RL; the typed-decisions results show small heads memorize wording.
- **Reward contrarian beliefs** — rejected: the RCT evidence says theorization alone is harmful; only conjecture + experiment
  is credited.
- **Make the typed-decision head the chooser now** — rejected by its own numbers (51 % on reworded panels, 4/8 end-to-end);
  kept as a measured shadow path.
- **Learn exploration by RL on the model** — rejected: replay-improved controller code is cheaper, auditable and reversible.
