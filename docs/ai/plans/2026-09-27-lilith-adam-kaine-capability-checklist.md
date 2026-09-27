# Lilith baseline: the full ADAM + Kaine capability checklist

- **Date:** 2026-09-27
- **Status:** For operator check (slice C1b, Anima WI `01a0e2a0-6fa3`)
- **Purpose:** the operator decided that Lilith's basics are the full ADAM and Kaine capability set.
  This lists every capability the docs define, where it is written, and how well: **S** = specified
  (definition + exam or threshold), **P** = partially specified (named, but no threshold or an open
  question), **R** = research only.
- **Method:** a source inventory of `origin/main` on 2026-09-27, plus ADR 112 and P7.14–P7.18 from
  branch `docs/adr-112-kaine-theory-first` (pushed to origin 2026-09-27, not on main). The family
  grouping is editorial; the capabilities are the documents' own.
- **Related:** [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md), [Lilith CoS plan](2026-09-27-lilith-chief-of-staff-rl-runpod.md), [Project Lilith capability ledger](2026-09-21-project-lilith.md#capability-transfer-ledger)

## 1. Capabilities

| # | Capability | Lineage | Source | Status | Evidence | Gap in the docs |
|---|---|---|---|---|---|---|
| 1 | Tool-native action: choose among offered tools, schema-valid arguments against the current `ToolContractSnapshot` (53 tools × 24 cases, ≥85% pass, ≤1% catastrophic per family) | ADAM | ADR 102 §2, §4, §5 | S | `training/results/tool-native-8k-lane-20260911-RESULT.md` (1257/1272); `p6-2-decoder-rescore-20260913-RESULT.md` (1259/1272); P6.4 window 1 (1255, failed). Not certified | Lilith needs the gate on Gemma's own contract |
| 2 | Contract-drift journal + delta curriculum (react to removed or changed tools) | ADAM | ADR 102 §2 | S | Implemented; generator smoke only | Gemma-side drift journal |
| 3 | Copy-faithful constrained decoding, identifier-aware BPE, RoPE + chunked attention, ladder L0–L5, length-generalization sentinel | ADAM | ADR 104 D1–D6; P5 plan | S | P5.1–P5.4 landed; length generalization 800/800 in v3 | Vision decoder still unconstrained |
| 4 | Candidate vs active certification, lineage (`adapted_from`, `steered_from`), adaptive exam controller | ADAM | ADR 102 §5; ADR 100 | S | Implemented; no active model | **Inherited-model manifest + verifier for Gemma not written** |
| 5 | Seven stateful knowledge workflows (acquire, recall after restart, revise under conflict, provider fallback, abstain, newest edition, note navigation) + provenance write gate | ADAM | ADR 102 §1, §3 | S | 800/800 knowledge at 493,261 and after window 1 | — |
| 6 | Graph acquisition, identity, revision, invalidation of dependent beliefs (≥99% precision, ≥95% recall) | Both | R1 plan `2026-09-11-akh-350m-graph-grounded-release-plan.md` | S (targets only) | R1 harness not built | R1 not marked current or superseded; a cited RESULT is missing from every ref |
| 7 | Cold-restart recall with frozen weights (≥200 episodes, ≥95%) | Both | R1; homelab exam stage 7 | S (targets) | Only ADR 102 `recall_after_restart` measured | R1 episode runner missing |
| 8 | High Grug v2 internal register (≈19 operators, typed slots); R1 ≤2 pp non-inferiority, ≥25% fewer tokens | ADAM | ADR 092; ADR 088 (HG boundary format) | S design / P test | `p350m-RESULT.md` 0.928/0.935 | TMS operators (assume, contradict, retract) deferred, but R1 revision needs them |
| 9 | Dual-process handoff (model proposes, engine completes; neural fraction) | Both | ADR 093 | S | Lab and engine slices measured | Reverse direction out of scope; no threshold on the iterative arm |
| 10 | Step rewards from engine replay, VSA judge, recurrent depth | ADAM | ADR 096 | R | none | **Conflicts with ADR 110** (see §2) |
| 11 | Engine-in-the-loop training E1–E4 (symbolic completion, event-calculus validity, causal step reward, exact-owned revision); E5 breadth | ADAM | P6 plan; ADR 106 | P (controls, no success margin) | E1 treatment failed (window 1). No RESULT for E1-control, E2–E4, window 2 | Record P6.5 outcomes; E5 undesigned |
| 12 | Learn commands at runtime, Unix composition, versioned skill records, changed interfaces (8 homelab families) | Both | Homelab release exam | S behaviours / P thresholds | Not implemented | **GNU/BSD only; no illumos variant** |
| 13 | Languages EN/DE/FR certified; Portuguese basics (PT-BR, then PT-PT) | ADAM | ADR 102 §3, §5; R1; `2026-09-10-portuguese-language-stage.md` | EN/DE/FR S; PT P | EN/DE/FR in every sentinel; PT none | Certifier and sidecar reject anything outside en/de/fr; no PT threshold |
| 14 | Perception → graph → action (native OCR) | ADAM | ADRs 100, 101, 103; R1 | S (gates) | `vision-v1-a40-RESULT.md`: overfit, promotion held | **No audio capability in ADAM docs** (Gemma has audio input) |
| 15 | Crew of eight × Oxide's 15 values; three principles as captain invariants | Both | Dojo §2; ADR 110 D1 | P | none (P7.12 draft) | Observables without thresholds; slot names unset |
| 16 | Eight starter kata | Both | Dojo §3; R1 dojo section | P | White-belt v0 (Qwen3-0.6B LoRA); 476 dojo records in the ADAM base corpus | No dojo gate in ADR 102 certification |
| 17 | Sparring: 6 adversaries, rotating judges | Both | Dojo §4, §7 | P | none | Judges depend on Phase 50 (planned) |
| 18 | Belts White → Black, rank held in the self-model | Both | Dojo §5; R1 belt table; P7.12 | P | none | **Two conflicting belt definitions**; no frozen gates |
| 19 | Captaining: voice selection at Brown, vs engine as captain | Both | Dojo §2, §8 Q5; ADR 099; ADR 110 | P | none | **Resolved for Lilith by ADR 116 D2: internalized** |
| 20 | Lineage record `{situation, move_chosen, why_this_move, observed_effect}`; Black-belt transmission | Both | Dojo §5–6; R1 stretch row | P | none | Student-speedup gate has no margin |
| 21 | Kata in High Grug (hidden trace, spoken move in the user's language) | Both | Dojo §9 | P | none | HG has no kata operators |
| 22 | Process consultation: 6 forms, whole-process sparring, promotion contract | Both | Dojo §10.1–10.5 | S forms / P numbers | none | No converter or evaluator (§10.6) |
| 23 | Emotional intelligence and repair (paired needs; a correction changes the next action) | Both | R1; homelab "operator correction"; ADR 099 | S (R1: 85% per family) | none | No blinded rubric artifact |
| 24 | Coordination through Board and Anima dispatch, delegation, Teamwork (slot 8) | Kaine | ADR 110 D3; P7.10, P7.13 | P | none | P7.13 gate not frozen; **no R1 coordination family** |
| 25 | Guided reward: five-component vector, corrected siblings, no scalar-judge policy gradient; `AkhEnvironment` dojo tenancy; emergent-bug and act-to-perceive families | Kaine (baselined on ADAM) | ADR 110 D2–D4; P7.10–P7.13 | S exits / P family scoring | E3 weighted-loss mechanics only | "Bounded hypotheses" and "board ask" thresholds unset |
| 26 | Theory-first: `Theory` object, sixth reward component `foresight`, `theory_first` family, interventional prediction, 29f explorer tools | Kaine | ADR 112, P7.14–P7.18 (**branch only**); foresight archive | R / Proposed | none | **ADR 112 not on main, but ADR 114 cites it** |
| 27 | Action world model (predict before execute) | Both | ADR 117 (action world model; was 088) | P (Proposed) | none | Gated on Phase 50 |
| 28 | Curiosity as compression / learning progress | Kaine | Foresight handoff; ADR 005 §2; ADR 112 D3 | R | none | — |
| 29 | Replay-improved exploration (Dream-RSI) | ADAM harness + Kaine | Dream-RSI analysis; ADR 112 D5 | R | none | — |
| 30 | Typed decisions, joint scoring of complete actions (Jev, Jevlike) | Both | TypeSafe/Jev README; Jevlike handoff; ADR 112 P7.15 | R | `experiments/typed-decisions/` (**untracked**) | ADR 114 keeps Jev-class models out of the live path |
| 31 | Kaine architecture ladder (SWA + global, MobileLLM, Gated DeltaNet, 1.1B fit, product-key memory layer) | Kaine | ADR 111; P7.5–P7.9 | S (exits) | none run | Kaine size open; **not weight-compatible with Gemma** |
| 32 | Training-engine memory (recompute attention, fusion/f16, recompute FFN, Cut CE, Muon) | Both | ADRs 107, 109; P7.1–P7.4 | S | P6.1b: 8192×2 in 7.0 GB | — |
| 33 | Local lane and run governance | Both | ADRs 105, 106 | S | In effect | — |

## 2. Missing, under-specified, or conflicting (please check)

1. **Design docs not on main.** ADR 112 and P7.14–P7.18 are on a branch only (pushed 2026-09-27),
   and ADR 114 on main cites them. Duplicate ADR numbers exist: 088, 097 and 108 are each used twice.
2. **Which ADAM target is current.** R1 (350M–1.1B, 12 families) is neither marked current nor
   superseded. P6 narrowed the target to the `small_8k` release gate.
3. **No dojo retention gate.** Dojo §10.5 requires one, but ADR 102 certifies only knowledge and
   tools. P7.12 uses "≤3-case regression on 1,272" as a proxy.
4. **No numeric thresholds** for kata, belts, consultation, the environment gate, disposition credit,
   or the emergent-bug and act-to-perceive families.
5. **Two belt definitions:** dojo §5 by form and voice; R1 by memory-to-action exercises.
6. **Reward conflicts:**
   - ADR 096 uses negative redundancy and an RL advantage, while ADR 110 allows weights in [0,1] only.
   - Dojo §4/§7 have judge-graded "sparring RL", while ADR 110 rule 7 forbids a scalar-judge
     gradient.
   - The Lilith KLPO arm uses terminal reward, while ADR 110 rule 1 says every step is scored. ADR
     116 keeps KLPO as an ablation only.
7. **Courage placement** is inconsistent: ADR 110 puts it in slot 1 (+ slot 4), while ADR 112 D4
   puts it in slot 4 (Rebel).
8. **"Maat"** appears in dojo §10.2 next to Caregiver but in no crew table.
9. **Coordination is thin in ADAM/Kaine.** There is no R1 family and no exam, only ADR 110 and
   P7.13. Lilith's primary role therefore needs its own exams (ADR 116 families).
10. **illumos is absent from every ADAM/Kaine source.** The homelab exam targets GNU/BSD. Lilith's
    illumos guidance role has no inherited exam; ADR 116 adds `illumos-guidance` and the Project
    Lilith illumos pack.
11. **Portuguese** is a measured target with no threshold, and the certifier rejects it.
12. **Kaine naming** is ambiguous in the hardware plan; the Lilith plan resolves it.
13. **P6.5 outcomes unrecorded:** E1-control, E2–E4, window 2.
14. **Operator-owned open questions:**
    - voice names (dojo §8 Q1);
    - belt count and names (Q2);
    - scenario authorship and judges (Q3);
    - where High Grug meets the kata (Q4);
    - the Portuguese threshold.

## 3. Lineage-bound capabilities (Gemma needs a behavioural-equivalent exam)

- **Tokenizer as ABI:** identifier-aware BPE, HG reserved tokens, the dark SymbolId block. Gemma
  needs its own grammars and copy-fidelity exams.
- **Sovereign foundation:** ≥500M en/de/fr tokens, a teacher-free base. This is not applicable to an
  inherited model; write the rationale down.
- **Artifacts:** `candidate-model.json`/`active-model.json`, Burn checkpoint v2, RoPE and window
  settings. Replace them with the Gemma inherited-model manifest (row 4 gap).
- **HG fluency pretraining and gloss embeddings:** only the runtime HG register transfers.
- **Measured results** (E1–E4, windows, white-belt LoRA, persona vectors): the curricula transfer,
  the results do not.
- **Kaine architecture and training engine:** not weight-compatible. The behavioural aims transfer
  through the engine.
- **Belt rank in the self-model and per-organ character adapters:** Gemma-specific adapters, and the
  belt exams re-run.
- **Transfer as behaviour (re-run on Gemma's contract and backend):** the all-tool matrix, the seven
  knowledge families + length generalization, the R1 episode families and homelab exam (illumos
  variants to write), the consultation and kata rubrics, and the P7.13 environment gate.

## 4. What the operator is asked to do

For each row, mark **keep / adapt / not applicable**. Answer the §2 questions you own. C1b then maps
every kept row to a Lilith exam and fixes the docs gaps you confirm.
