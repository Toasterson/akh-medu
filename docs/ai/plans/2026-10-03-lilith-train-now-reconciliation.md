# Lilith: transcript acceleration and the next training run

- Date: 2026-10-03. This is the current execution plan for the next development run.
- Parent: [Lilith CoS plan](2026-09-27-lilith-chief-of-staff-rl-runpod.md), Anima Plan `01a0e26f-d032-7541-80cb-34e3f29c8842`.
- Supersedes the **training-start wait** in [C5–C7](2026-10-01-lilith-c5-c7-dojo-and-training.md) §§5/7. Fresh evaluation, active-adapter approval and the T3 gate remain separate decisions.
- Operator direction: use harvested transcripts to bring training forward; include the voices curriculum; keep testing bounded. This audit starts no GPU workload.

## What drifted

The October 1 plan assumed that the only training worlds came from the new C3 recorder. Its time-window splitter consequently had no train/dev data and deferred C7 until roughly October 12. The October 2 historical work created an independent training corpus and completed a pilot, but its document was only a supplement. Neither the parent Anima description nor the C5–C7 training-start condition was reconciled. The follow-up preparation then repeated the old temporal gate.

Claude's September 27–October 1 operator transcript confirms the approved eight voices, internalized captain, Double Diamond and Blue Ocean exercises, and the instruction to move into training. The later Codex conversation authorized historical reconstruction and the first historical training run. Compaction may have helped preserve the stale condition, but the demonstrable fault is the inconsistent persisted plan.

**Training can proceed from historical development data. C3's elapsed-day requirement is not a prerequisite for that run.** Historical receipts provide partial dated worlds, not fabricated full-project snapshots. Continuing the recorder supplies future evaluation worlds independently of GPU training or Hephaistos uptime.

## Audited state

| Part | Evidence as of this audit | Remaining work |
| --- | --- | --- |
| Historical facts | 545 transcript files scanned; 338 complete receipts, including 76 Claude receipts; 636 train lessons and 40 analyst-visible development lessons | Retain provenance and exclude these held-out sessions/WIs from the mixed training corpus |
| First training | Main `89b1e40b`, 200 optimizer steps completed; parent 0/20 and candidate 20/20 exact receipt answers; reload identical; saved candidate inactive | Receipt skill is trained; this does not establish the complete CoS role |
| C5 role material | 92 panel-admitted trajectories; 88 staged after one overlapping session was excluded (78 train / 10 validation) | Export supported assistant decisions with observation masking; recompute splits across all sources |
| Voices and memory | Voice labels, replayed real C2 memories, four memory variants, corrected-sibling builder and structural verifiers are on main | Export verified corrections into the same trainer format; cover all eight voices and the approved reasoning forms |
| Teacher material | Existing teacher runs and CPU re-verification artifacts; 154 candidates in the 208-episode re-verified run meet the stored status/hard-flag/weight filter; one additional candidate in the pinned 24-episode run | Recheck current labels, deduplicate, split and exclude held-out exposure; these counts are candidates, not final admitted totals |
| Recorder | 54 snapshots from September 28 to October 3, about five days | Keep collecting for later evaluation; do not wait for its fourteenth day to train |
| GPU preparation | Existing EU-FR-1 volume, parent, CUDA environment and guarded pilot path | Mixed-curriculum entry point/payload still needs preparation; capacity is checked only when launching |

Private audit root: `/Users/toasty/lilith-private/plan-reconciliation-20261003/`. Completed pilot: `/Users/toasty/lilith-private/history-20261001/launch/pilot-summary.json`. The old receipt-only trainer validates its own restricted schema and unit weights; simply concatenating C5/dojo records into it is not a working mixed-curriculum launch.

## Data boundary for the accelerated run

1. Keep the original 40 historical development cases and four prepared behavior spots out of training. Exclude exposure through full tool responses as well as subject IDs, and keep a session/shared-WI connected component in one split across sources.
2. The C2 corrections and pre-October-2 dojo worlds already examined during development may feed this run. Explicitly record them as retrospective development, even where old filenames say `sealed`. Preserve the original results and manifests. They cannot subsequently be called an untouched release exam.
3. Proposed cutoff: `2026-10-02T00:00:00Z` exclusive for recorder-derived training worlds **and their hindsight labels**. Later recorder worlds remain outside this training corpus. Do not move the cutoff forward to fill a missing family. Final selection checks entity and text overlap with any later evaluation set.
4. Admit assistant transcript prose as panel-approved imitation at its source weight, with its evidence limitations. Facts/corrections require complete receipts or replay verification. Failed original answers have zero positive training weight; corrected siblings earn their verifier weight. No invented historical probabilities or preference labels.
5. Mask prompts, memories and tool observations. Train supported assistant tool decisions and answers. Use the pinned Gemma tokenizer/template; step windows must preserve required observations, and records that cannot fit are reported rather than silently truncated.

This explicitly replaces the temporal wait for development training. It does not alter the default dojo time-window splitter or count historical partial worlds as fresh C3 coverage.

## What the voices must learn

The [dojo manual §2a](../manuals/2026-09-01-babyllm-dojo-curriculum.md) is canonical for names; its §5 is canonical for belt identity. The October 1 color sequence is a CoS lesson ordering, not a replacement character ladder.

| Voice | Training action |
| --- | --- |
| Maker | Execute the supported next move after checking it; finish with the result |
| Truthkeeper | Read the receipt and distinguish it from remembered retelling |
| Seeker | Open a useful alternative source or state the remaining unknown |
| Challenger | Name and correct an earlier claim disproved by the evidence |
| Guardian | Check consequences, scope and standing rules before an action |
| Sacred Clown | Break a conflated identity or stuck explanation by checking an alternative |
| Steward | Keep verification load-bearing and bounded; stay on the current ask |
| Weaver | Write a memory with a verifiable source and coordinate the right recipient |

Start with evidence/scope/read-failure, then briefing/triage/chase, then memory/voice selection. Include the eight voice kata in this first mixed run rather than postponing character until after CoS training. Double Diamond examples must show distinct live options/reads and a criterion for convergence. ERRC examples must eliminate contradicted claims, raise verified ones, reduce unsupported certainty and create the missing check. Naming a voice or a technique without the corresponding behavior is insufficient. Existing memory corrected siblings do not automatically contain every optional Diamond/ERRC form; the export must make coverage explicit.

## Execution order

The current Codex supervisor owns preparation and reporting. Private data stays local/NFS. Code changes use the local-pickup workflow and a focused PR; GPU payloads use merged main.

1. **Finish one CPU corpus/export task (C5).** Recover/cache the three annotated C2 session files locally first: the audit's attempt to read the NFS copy stalled and was interrupted before it produced an export. This is a source-retrieval issue, not a recorder-duration requirement. Combine the existing historical lessons, admitted C5 decisions, eligible teacher decisions and verified C2/dojo corrected siblings. Produce source hashes, exclusions, connected-component splits, per-family/eight-voice coverage and token masks. Replay the corrections using existing verifiers. No new panel campaign or general benchmark suite is required. Missing Blue/illumos outcomes are recorded and do not stop the available core curriculum.
2. **Prepare one mixed SFT launch (C7 development stage).** Reuse the proven full-depth LoRA core and supervisor/reload controls. Warm-start the saved receipt candidate after checking parent/rank/tensor compatibility, retaining historical rehearsal; do not restart the project from seed expansion. Freeze the mixed data manifest and run ID. Replace the pilot's receipt-only schema/grade wrapper with the mixed trajectory exporter and bounded family evaluation.
3. **Run when EU-FR-1 H100 capacity is available.** Reuse volume `c6kdciozex`; no region move/new volume. One curriculum pass, at most 1,500 optimizer updates, and a three-GPU-hour host stop limit including setup, evaluation and save. Report incomplete coverage if the limit is reached. At the previously observed $3.49/h this is at most $10.47 GPU cost, plus storage; refresh rate and remaining prepaid credit before launch and reduce the cap if needed. This is a proposed next-run budget, not a new paid launch during this audit.
4. **Check once, then interact.** Fold the already prepared 20 remaining receipt comparisons into that pod session; reuse the original 20 parent answers. Score the new candidate on all 40 receipt cases and the four excluded behavior spots, report results on at most 12 selected CoS/voice development episodes from the new split, and cold-reload once. Do not require a separate evaluation-only pod or completion of the 385-world baseline before training. Then run one supervised C2 conversation focused on briefing and memory correction and feed actual failures into the next correction round.
5. **Iterate from observed failures.** Sibling preference and on-policy training remain subsequent stages of the approved guided-RL program. KLPO stays an optional ablation. They are not additional mandatory arms before this mixed SFT run. Fresh recorder evaluation, native illumos evidence, active-adapter selection and a canary follow when their actual data/serving prerequisites are ready.

Completion of preparation means a tokenized mixed archive and launch command with verified hashes, not a recipe containing source paths. As of this audit that artifact is still missing. The next work is the exporter and launch preparation, not another twelve-day wait or another planning cycle.
