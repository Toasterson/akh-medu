# Lilith baseline: the full ADAM + Kaine capability checklist

- **Date:** 2026-09-27
- **Status:** For operator check (slice C1b, Anima WI `01a0e2a0-6fa3`)
- **Purpose:** the operator decided that Lilith's basics are the full ADAM and Kaine capability set.
  This lists every capability the docs define, where it is written, and how well: **S** = specified
  (definition + exam or threshold), **P** = partially specified (named, but no threshold or an open
  question), **R** = research only.
- **Method:** a source inventory of `origin/main` on 2026-09-27, plus ADR 112 and P7.14–P7.18 (on main since PR #419;
  the checklist first cited branch `docs/adr-112-kaine-theory-first`). The family
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
| 11 | Engine-in-the-loop training E1–E4 (symbolic completion, event-calculus validity, causal step reward, exact-owned revision); E5 breadth | ADAM | P6 plan; ADR 106 | P (controls, no success margin) | **Recorded 2026-09-28** (`training/results/p6-5-windows-RESULT.md`): all 9 completed windows FAILED the release gate. Best: window 2 (sentinel-shaped siblings) 1265/1272, 1 blocker family. E1 control ≈ identical to treatment (no effect); E2 projection/raw-events byte-identical (no measured effect); E3 treatment trades a `reactor_manage` fix for a worse `memory_write` catastrophic rate; E4 treatment is the one pairwise win (1 fewer blocker family) | E5 still undesigned |
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
| 26 | Theory-first: `Theory` object, sixth reward component `foresight`, `theory_first` family, interventional prediction, 29f explorer tools | Kaine | ADR 112, P7.14–P7.18 (on main, PR #419); foresight archive | R / Proposed | none | Fixed 2026-09-27: ADR 112 is on main |
| 27 | Action world model (predict before execute) | Both | ADR 117 (action world model; was 088) | P (Proposed) | none | Gated on Phase 50 |
| 28 | Curiosity as compression / learning progress | Kaine | Foresight handoff; ADR 005 §2; ADR 112 D3 | R | none | — |
| 29 | Replay-improved exploration (Dream-RSI) | ADAM harness + Kaine | Dream-RSI analysis; ADR 112 D5 | R | none | — |
| 30 | Typed decisions, joint scoring of complete actions (Jev, Jevlike) | Both | TypeSafe/Jev README; Jevlike handoff; ADR 112 P7.15 | R | `experiments/typed-decisions/` (**untracked**) | ADR 114 keeps Jev-class models out of the live path |
| 31 | Kaine architecture ladder (SWA + global, MobileLLM, Gated DeltaNet, 1.1B fit, product-key memory layer) | Kaine | ADR 111; P7.5–P7.9 | S (exits) | none run | Kaine size open; **not weight-compatible with Gemma** |
| 32 | Training-engine memory (recompute attention, fusion/f16, recompute FFN, Cut CE, Muon) | Both | ADRs 107, 109; P7.1–P7.4 | S | P6.1b: 8192×2 in 7.0 GB | — |
| 33 | Local lane and run governance | Both | ADRs 105, 106 | S | In effect | — |

## 2. Missing, under-specified, or conflicting (please check)

Doc-gap pass 2026-09-27 (branch `docs/fix-adam-kaine-doc-gaps`): items marked **Fixed** changed only
documents; items marked **→ Lilith exam design (supervisor)** need numeric exam design and stay open.

1. **Design docs not on main.** ~~ADR 112 and P7.14–P7.18 are on a branch only~~ — **Fixed 2026-09-27
   (commit `034fff4c`)**: they reached main through PR #419 (`fd711f92`); the merge of
   `origin/docs/adr-112-kaine-theory-first` brought no tree change and only records the branch as integrated.
   No number collision for 112. Duplicate ADR numbers 088, 097, 108 — **Fixed 2026-09-27 (commit
   `7ddf0f04`)**: the later-dated ADR of each pair is renumbered: action world model 088 → **117**,
   shared-store migration compatibility 097 → **118**, transcript harvest 108 → **119**; references updated.
2. **Which ADAM target is current.** **Fixed 2026-09-27 (commit `b0959628`)**: the R1 plan carries a status
   line (later release target; the current gate is the P5/P6 `small_8k` release gate) and marks the missing
   `release-v3-failure-analysis.md` as missing.
3. **No dojo retention gate.** Dojo §10.5 requires one, but ADR 102 certifies only knowledge and
   tools. P7.12 uses "≤3-case regression on 1,272" as a proxy. **Open → Lilith exam design (supervisor).**
4. **No numeric thresholds** for kata, belts, consultation, the environment gate, disposition credit,
   or the emergent-bug and act-to-perceive families. **Open → Lilith exam design (supervisor).**
5. **Two belt definitions.** **Fixed 2026-09-27 (commit `b0959628`)**: dojo §5 is canonical for belt identity
   and order (note in the manual, commit `2474e08f`); the R1 table is now the memory-to-action exam content
   per dojo belt, one row per §5 belt. Belt gate numbers stay open with item 4.
6. **Reward conflicts.** **Fixed 2026-09-27 (commit `2474e08f`)**:
   - ADR 096 carries "superseded in part by ADR 110"; Decision 3 (scalar advantage) and the negative
     redundancy clause of Decision 4 are marked in place.
   - Dojo §4/§5/§7 sparring now follows ADR 110 rule 7 (weighted SFT on corrected siblings +
     sibling-contrast preference; judges score social effect only, blinded).
   - The KLPO research note states that ADR 116 keeps KLPO as a verifier-only ablation arm.
7. **Courage placement.** **Fixed 2026-09-27 (commit `2474e08f`)**: ADR 112 D4 now labels Courage as slot 1
   Creator (supporting slot 4 Candor), matching ADR 110 Decision 1.
8. **"Maat".** **Fixed 2026-09-27 (commit `2474e08f`)**: dojo §10.2 defines Maat at first use as the captain's
   field (the three Oxide principles as captain invariants, ADR 110 D1), not a voice.
9. **Coordination is thin in ADAM/Kaine.** There is no R1 family and no exam, only ADR 110 and
   P7.13. Lilith's primary role therefore needs its own exams (ADR 116 families). **Open → Lilith exam
   design (supervisor)** (coordination exam).
10. **illumos is absent from every ADAM/Kaine source.** **Fixed 2026-09-27 (commit `b0959628`)** as a
    definition: the homelab exam has an "illumos variant" section (SMF, IPS, zones, ZFS, RBAC/`pfexec`,
    illumos userland and man pages; same thresholds as GNU/BSD, frozen before evaluation). The
    `illumos-guidance` family itself stays with ADR 116.
11. **Portuguese.** **Fixed 2026-09-27 (commit `b0959628`)** as a proposal: PT-BR basics per family ≥ 0.85 ×
    the same model's EN score, PT-PT reported without a gate ("Proposed 2026-09-27, freeze before
    evaluation"), in the Portuguese plan and R1. The certifier still rejects `pt`; the ADR 102 language list
    must add it before PT can be certified (doc note only; no code changed).
12. **Kaine naming.** **Fixed 2026-09-27 (commit `b0959628`)**: the hardware plan notes that "Lilith" names the
    inherited Gemma track and "Kaine" keeps its ADR 111 meaning.
13. **P6.5 outcomes unrecorded.** **Fixed 2026-09-28**: the nine completed windows' evidence was recovered
    from the read-only NFS backup (`/Volumes/akh-medu-runs/mac-toasty/2026-09-24/akh-medu/t3code-bf7ae675/runs/p6-4/`,
    never copied back to `origin/main` before the pods/queue were torn down) and written up in
    `training/results/p6-5-windows-RESULT.md`, with the P6 plan's outcome table (§Experiments) updated to match.
    **All nine windows FAILED the release gate**; window 2 (sentinel-shaped siblings, not an E1–E4 arm) is the
    best result at 1265/1272 with a single blocker family. E1 and E2 show no measurable treatment effect; E3
    trades one blocker family for a worse catastrophic rate elsewhere; E4 is the only clean pairwise win, and
    still fails the gate.
14. **Operator-owned open questions:**
    - voice names (dojo §8 Q1);
    - belt count and names (Q2);
    - scenario authorship and judges (Q3);
    - where High Grug meets the kata (Q4);
    - the Portuguese threshold (now a proposal to confirm or change; item 11).

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
