# Phase 51 P7 extension — Kaine theory-first policy: training environment, inference environment, policy machinery

- **Status:** Planned (2026-09-19), **gated on P6** like every P7 item: nothing starts before the P6.4/P6.5 window queue and its
  release gate have finished; all items HITL (`source=architect-session`, `draft` until the operator promotes them, never
  `ready`/`working`).
- **ADR:** [ADR 112](../decisions/112-kaine-theory-first-policy-training-inference-environments.md); builds on
  [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md) (reward vector, live environment),
  [ADR 111](../decisions/111-kaine-architecture-ladder.md) (backbone untouched), [ADR 088](../decisions/088-akh-action-world-model.md)
  (predict-before-execute), [ADR 005 §2](../decisions/005-bootstrapping-research.md) (compression-progress curiosity),
  [ADR 105](../decisions/105-autonomous-run-governance.md) (runs).
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `01a0a24e-062f-7881-9fd6-1d2017d52b34` (P7).
- **Parent plan:** [P7 ladder + engine](2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md) (P7.1–P7.13).
- **Research inputs (Codex, research-only):** T3 thread `6adfb261-5682-4d80-89fc-4875480d9876` (2026-09-18: exploration policy today + Dream-RSI implementation sketch; not archived, only the provider log); 2026-09-16 Astra harvests `docs/ai/research/2026-09-16-kaine-foresight-curiosity/`,
  `2026-09-16-dream-rsi-adam-kaine/`, `2026-09-16-jevlike-adam-kaine/`, `2026-09-16-jev-ecosystem/`,
  `2026-09-16-typesafe-jev-adam-kaine/`, and the measured `experiments/typed-decisions/` artifact.

## Goal

Give Kaine the forward-looking loop the theory-based-view literature describes — conjecture, predict, intervene, observe,
revise — as an engine-owned procedure the model is trained on and measured against, without changing the backbone, the
weighted-SFT training rule, or engine authority over verification.

## What is assembled

| layer | today | added by this plan |
|---|---|---|
| Training environment | `akh_episode_generator` (frozen contracts, sentinels, E1–E4); P7.10 `AkhEnvironment` design; `train_knowledge --record-weights use`; sidecar v1 (P7.11) | `Theory` objects as a producer; `theory_first` family; sidecar v1.1 with `foresight`; replay corpus of discovery trees; interventional-prediction axis on the P7.13 gate |
| Inference environment | `ChatBacking::Grug` → `GrugChatBackend` (certified bundle only, constrained copy-faithful decoding, live `ToolContractSnapshot` + drift journal); `PredictionTracker` held but unused by the LLM path | 29f explorer + theory tools in the contract; foresight telemetry per turn; deployment tuples exported in sidecar shape |
| Policy machinery | `DecideMode::{VsaNative, LlmDriven}`, MCTS + `ValueFunction`, active-inference EFE, VOC halting, `PermissionPolicy`, backend `Posture` | the explicit stack *enumerate → choose → spell → verify*; typed-decision head as a measured shadow path only; replay-improved controller code |

## Work items

| WI | scope | exit condition |
|---|---|---|
| `01a0bb17-f092-75e0-9e9b-44fe0d419589` P7.14 `Theory` engine object | `src/agent/theory.rs`: problem decomposition, `Believes` causal claims in a microtheory, necessary conditions by cost, DS interval = data–belief asymmetry, experiment = cheapest necessary condition with the largest expected interval reduction, TMS revision; `DerivationKind::{TheoryConjecture, DirectedExperiment}` | unit tests; engine demo where seeded contrary evidence + one selected experiment flips a belief via TMS with the trace in provenance; no lane touched |
| `01a0bb18-1207-7ff1-9acc-dd00e73807d0` P7.15 spike: decision seam | frozen ADAM corpus (observation, complete candidate set, verifier outcome); sealed panel; four arms: constrained generation / candidate scoring / joint complete-action scoring / + per-schema calibration; 3 seeds; generation regression on the 1,272-case gate | findings document + go/no-go; "no" keeps constrained generation the sole chooser |
| `01a0bb19-3665-7d01-89a1-216489d25a51` P7.16 `foresight` + `theory_first` (depends P7.10, P7.11, P7.14) | sidecar v1.1 sixth component, surprise-weighted, miss = 0 + corrected sibling; `theory_first` family with passive control and Curiosity/Courage/Rigor labels; learning progress for curriculum/experiment selection only; interventional-prediction axis on P7.13 | one A/B window from 401,577 on the unchanged sentinels (vector+siblings vs +foresight) with all gates and the new axis in one RESULT table; generator hash + decontamination receipts |
| `01a0bb19-62d0-7b33-831a-b801fc547c86` P7.17 inference hooks (depends P7.14) | `hypothesis_fork/merge/confidence`, `theory_propose`, `experiment_select` as `impl Tool` in the registry and the frozen contract; `PredictionTracker` foresight telemetry on `GrugChatBackend` turns; tuples exportable in sidecar v1.1 | contract parity test; one dojo dispatch showing propose → select → mutating call with prediction, verified delta and score in the trace |
| `01a0bb18-32fe-7830-9358-aee23664aee8` P7.18 spike: replay-improved controller (depends P7.10, WI-399) | discovery-tree recorder (parent attempt, environment snapshot, model/prompt version, proposal, verified outcome, cost); replay harness with hidden futures, isolated-branch restore and unrecorded branches marked unknown; three arms (fixed / hand-written adaptive / replay-improved) on frozen-ADAM repair tasks; per-backend evaluation with model identity recorded (controller serves cloud and local backings); online dojo validation under ADR 105 | findings document with both tables and a flag-gated decision on the VOC defaults |

Amendment recorded on the Anima item: **P7.13** gains the interventional-prediction axis (held-out `predicted_delta` accuracy
on unseen interventions); "axis pending" until P7.16 lands.

## Sequence

1. P7.14 is engine-side Rust with no GPU or recipe contention; like P7.10 the operator *may* promote it before the ladder items.
2. P7.15 needs only the existing generator and the ADAM candidate; it can run any time after P6, on CPU.
3. P7.16 waits for P7.10 + P7.11 (the dojo and the vector) and P7.14; its window is an ADR 105 run WI.
4. P7.17 after P7.14; contract regeneration means one remediation window before the ADAM candidate sees the new tools.
5. P7.18 after P7.10 has recorded enough discovery trees and WI-399 has landed the B3 halting it will compete against.

## Guards

- No scalar-judge RL, no learned head in a live path without the P7.15 verdict, host verification authoritative (ADR 110, ADR 112 D8).
- Contrarianism alone earns nothing; conjecture and experiment are credited together (Valentine et al. 2024).
- DS intervals are not calibrated probabilities; no threshold travels across schemas.
- Every run is an Anima WI with an exit condition and board-routed events; no chat-wakeup bridges (ADR 105).

## Reconciliation notes

- **ADR 005 §2** already maps compression progress to `curiosity_bonus` in OODA utility; ADR 112 D3 keeps that as a
  selection signal and explicitly does not promote it to a training reward. KG-coverage delta is a proxy, not the compressor's
  learning progress; the held-out generalization check from the foresight harvest guards against memorization.
- **ADR 088** is the home of `predicted_delta`; until the world model lands, `CausalManager::predict_effects` supplies the
  prediction and `verify_prediction` the check, exactly as E3 does today.
- **Phase 29f** hypothesis explorer: the tool surface is closed by P7.17; the stateful ACP surface still needs its own design.
