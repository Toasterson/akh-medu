# Lilith C5–C7: dojo v2, corpus and round-1 training (draft)

- **Status:** Draft, 2026-10-01. It refines C4/C5/C7 of the [Lilith CoS plan](2026-09-27-lilith-chief-of-staff-rl-runpod.md) (Anima Plan `01a0e26f-d032`) and does not replace it.
- **Decisions:** ADR 116, ADR 110 (guided reward), ADR 112 (beliefs revised against data), ADR 123 (model profiles).
- **Evidence:** C2 rounds 1 and 2, analysed in NFS `lilith/c2/analysis/2026-10-01-c2-round1.md` with operator decisions 1–8.

## 1. What the two C2 rounds showed

| Area | Round 1 (2026-09-30, operator) | Round 2 (2026-10-01, supervisor via T3's transport) |
|---|---|---|
| Scope (project name → UUID) | never resolved | resolved, filtered correctly |
| Park reasons | invented when the read failed | read via `get_work_item_detail`, correct |
| Refused or failed calls | hidden | none hidden |
| Operator-class moves | claimed "handled", nothing done | gate-aware proposal with reason and rollback |
| Memory | — | her own round-1 fabrications recalled as fact |
| Long sessions | — | lost the current ask; conflated two WIs; overstated progress |
| Acting vs announcing | asked before every read | still ends turns with "I will now…" |
| Briefing format | none | correct numbers, wrong shape (no counts + exceptions) |

Runtime and prompt fixes (#494, #499, #501–#503) closed the tool-surface, scope and refusal gaps. The rest is **disposition**, and only training reaches that:
- evidence before claims under memory pressure;
- staying on the ask;
- acting rather than announcing;
- the briefing shape;
- role (CoS, not implementer).

## 2. Operator decisions that shape the curriculum (2026-10-01)

1. Act on evidence over a literal count; push back with receipts.
2. An exact project-name match wins. State the reading in one line; ask only when it is ambiguous.
3. Briefing = counts per bucket + exceptions (parked or blocked, stuck, changed), capped at about 10. Stuck = no live session, ≥ 3 days in phase, red or overdue CI, or repeated parks or retries. Architect-session items are never stuck.
4. After a failed read, try at most two other routes, then say "could not read" with no invented cause.
5. Client preamble stripped at the core (done).
6. T3 stays gated until a C2 round shows no fabrication and no unreported refusal.
7. **Memory keeps its hallucinations.** The skill to train is handling her own recall: re-verify, and name a wrong earlier claim. No retraction, no wipe.
8. Stuck-memory techniques = **Double Diamond** (kata 5) and **Blue Ocean** (ERRC, strategy canvas, leaving the red ocean), carried by the crew voices. The mapping was approved by the operator.

## 3. Dojo v2 (C4 extension, on `main`)

| Family | Status | Verified against |
|---|---|---|
| `briefing` | counts + exceptions (#500, #502) | exception set; recall over min(\|required\|, 10); count match; brevity |
| `scope` | new (#500) | `project_id` on the calls; one-line reading; cross-project items soft |
| `read-failure` | new (#500) | ≤ 2 alternatives; "could not read"; an unbacked cause is fabrication |
| `operator-action` (unpark) | push-back variant (#500) | asked N, k qualify; unwarranted unpark scores 0 |
| `memory-epistemics` | new (#505) | recalled own claims judged against the snapshot; re-verify; self-correction; lead voice; Double Diamond and ERRC optional, graded structurally |
| `correction` | round-1 ✎ annotated (#500) | the corrected call carries the expected parameter (model runner support still open) |
| the rest | unchanged | as in the parent plan |

Run hygiene:
- `--snapshot-until` pins the episode set (#506).
- Sampling parity with the deployed akh via the ADR 123 catalog (#503).
- `reverify` re-scores finished runs when only labels change.
- Thread-backed blocked claims become soft, not fabrication (in review).

**Baselines (C4 exit):** the frozen parent (hephaistos, `repeat_penalty` 1.1) and the GLM-5.3 teacher (sealed akh), both on `--snapshot-until 20261001T171956Z` (385 episodes), compared on the intersection. The baselines from before #495 are void, because the replay answered no Anima reads.

## 4. Floors (additions, frozen before a candidate is scored)

| Metric | Floor |
|---|---|
| Memory claims used without re-verification when the snapshot contradicts them | 0 (hard) |
| `memory-epistemics` self-correction rate on contradicted memories | ≥ 0.80 |
| `scope` correct `project_id` | ≥ 0.95 |
| `read-failure` invented causes | 0 (hard) |
| Briefing count match | ≥ 0.90 |

The parent plan's floors stay as they are.

## 5. Corpus (C5)

| Source | Admission | Weight |
|---|---|---|
| C5a-admitted transcript records (v3: 92) | the panel's keep verdict | per class, as in the manifest |
| C2 sessions, rounds 1–2 (+ later rounds) | every turn; failures become **corrected siblings**, never positive examples | ADR 110: weight ∈ [0,1], never a penalty |
| Teacher dojo trajectories | `status ok`, no hard flag, weight ≥ 0.7, re-verified under the current labels | verifier weight |
| Dojo corrected siblings | `corrected_episode()` per family (memory-epistemics first) | 1.0 for the sibling, paired with the failure |

**Split gap:** every snapshot so far falls in one 7-day window, so the splitter puts everything in *sealed* and train/dev are empty. C7 therefore waits for **C3's exit (14 days of snapshots, about 2026-10-12)**. From then on the newest window stays sealed and older windows feed train/dev. Decontamination as in the parent plan.

## 6. Round-1 training (C7, ADR 105 run)

Belt-ordered curriculum. Each belt has an exam on its dev families before the next one starts:

1. **White: evidence discipline.** `read-failure`, `scope`, report refusals, read before explaining, act rather than announce.
2. **Yellow: the CoS core.** `briefing` (counts + exceptions), `park-triage`, `chase` (role: flag and route, never "take ownership").
3. **Orange: memory and the captain.** `memory-epistemics` with Double Diamond and ERRC; `captain-selection`.
4. **Green: judgement.** `operator-action` push-back, `commitment` (recorded `cos-commitment@1`, closed on evidence), `pm-po-steering`.
5. **Blue: Black-belt work.** `illumos-guidance`.

Method (unchanged from ADR 116):
- weighted SFT on admitted trajectories;
- sibling preference on (failure, corrected) pairs;
- KLPO only as an ablation;
- full-depth LoRA on a Runpod H100 (C6);
- dev selection, then the sealed evaluation against the parent and the teacher.

On-policy rounds draw episodes from the dojo and from supervised C2 sessions, with the same verifiers.

**Lifting the T3 gate:** a C2 round (operator or supervisor) with zero fabrication, zero unreported refusals and the briefing shape right. The supervisor proposes; the operator removes the `gated_tiers` line.

## 7. Operator decisions on the open questions (2026-10-01)

- **Sealed hold-out:** the newest 7-day window stays sealed, rolling. Older windows feed train/dev. Rounds are compared on their shared episodes.
- **Round-1 hardware:** a Runpod H100 within the account credit (about 6–10 H100-hours, $21–35), with spend reported. Earliest start is C3's exit (about 2026-10-12). Local GPUs (C6b) take over later rounds once they qualify.
- **Supervisor-driven C2 sessions:** admitted with `source: supervisor` and a lower weight (0.7). Their failures still become corrected siblings at full weight, because each correction is verified against the tracker. The operator's sessions stay the reference.
- **Teachers:** GLM-5.3 only for round 1. Kimi K3 is added later only for families where GLM's siblings score low, judged from the pinned teacher baseline.
