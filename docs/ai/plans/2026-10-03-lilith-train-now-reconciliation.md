# Lilith: transcript acceleration through the full guided curriculum

- Date: 2026-10-03. This is the current execution plan for the complete C5/C7 guided training program.
- Parent: [Lilith CoS plan](2026-09-27-lilith-chief-of-staff-rl-runpod.md), Anima Plan `01a0e26f-d032-7541-80cb-34e3f29c8842`.
- Supersedes the **training-start wait** in [C5–C7](2026-10-01-lilith-c5-c7-dojo-and-training.md) §§5/7. Fresh evaluation, active-adapter approval and the T3 gate remain separate decisions.
- Operator direction: use harvested transcripts to bring training forward; include the voices curriculum; complete the existing guided curriculum, then test the resulting candidate. This plan revision starts no GPU workload.

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

## Full training scope: SFT is the opening stage

Operator clarification, 2026-10-03: finish the planned training sequence, then test. A short SFT pass is not the deliverable and cannot establish all of Lilith. The guided program remains weighted SFT **plus** same-situation corrected-sibling preference and on-policy experience under `cos-v1` (ADR 110/116). No new pilot, mandatory KLPO arm or qualification campaign is inserted before it.

| Stage, in canonical order | What is trained | Training signal |
| --- | --- | --- |
| White — Form | All eight voice behaviors and eight starter kata; evidence/read discipline, scope, concise action and honest correction | Admitted demonstrations and verified corrected siblings, weighted SFT |
| Yellow — Echo | The same forms under paraphrase, EN/DE/FR variation, multi-turn memory and budget pressure; counts-plus-exceptions briefings | Varied trajectories and corrected siblings, weighted SFT |
| Green — Trial | Existing sparring adversaries; refusal/redirection, candor, uncertainty, conflicting memories, Double Diamond and Blue Ocean/ERRC under pressure | On-policy episodes, verifier-guided corrections, weighted sibling SFT and sibling-contrast preference |
| Brown — Choice | Internalized captain: select the appropriate voice and explain the move; CoS triage/chase/commitments, PM/PO steering and warranted announced operator actions | Situation/disposition labels, real tool trajectories and corrected-sibling preference |
| Black — Transmission | illumos guidance of implementers/PM/PO using native evidence and source citations; explain the technique so another agent can use it; lineage commentary | Verified guidance trajectories, on-policy corrections and the existing transmission format |

CoS families span the stages; they are not an alternative belt ladder. Keep the inherited [ADAM/Kaine behavioral checklist](2026-09-27-lilith-adam-kaine-capability-checklist.md) in the corpus coverage matrix: tool-native calls, stateful knowledge/provenance/revision, retrieval and long-context behavior, coordination, coding and language rehearsal. Existing runtime capabilities are reused. A LoRA pass does not implement proposed architecture features or make their unmeasured targets achieved.

Historical transcripts accelerate demonstrations and corrections. They do not replace live practice, adversarial variation, captain selection or native illumos experience. Generate missing curriculum examples through the existing teacher/akh and replay/live paths as part of training preparation. Missing Black-stage evidence is an unfinished part of the program, not grounds to rename a White-stage adapter “all Lilith”.

## Execution order

The current Codex supervisor owns preparation and reporting. Private data stays local/NFS. Code changes use the local-pickup workflow and a focused PR; GPU payloads use merged main.

1. **Prepare one corpus and stage manifest (C5).** Recover/cache the three annotated C2 session files locally first; the audit NFS read stalled before export. Combine historical lessons, admitted C5 decisions, eligible teacher decisions, verified C2/dojo corrections, eight-voice/kata examples and inherited-capability rehearsal. Fill missing variation/sparring/guidance examples using existing teacher/akh and replay/live generation. Produce source hashes, exclusions, mixed-source connected-component splits, observation masks and a coverage matrix for every stage/family. Freeze per-stage example selection, passes and on-policy round quotas before launch. No separate panel or broad benchmark campaign is required to start the program.
2. **Prepare the complete training job (C7).** Reuse the proven full-depth LoRA core, compatible saved receipt candidate and historical rehearsal. Implement the common trajectory format and weighted loss, same-situation sibling preference, on-policy rollout/correction ingestion and stage transitions. Save adapter, optimizer/scheduler/RNG state, corpus/stage cursor and remaining budget so resumption continues the same program. Freeze manifests and main-only source hashes. Preparation is complete only when the full job can execute all five stages; a seed-only launch is insufficient.
3. **Execute White → Yellow → Green → Brown → Black.** Finish each stage's declared training work and continue to the next without an intervening acceptance exam or evaluation-only pod. Existing per-episode verifiers supply the guided learning weights and corrections. Their role is training supervision; the model acceptance suite runs at the end. Live situations use the existing single-executor rule, own scoped identity and standing operating rules. Keep training and inference on the full akh tool/runtime surface.
4. **Bound spending, not the curriculum.** Reuse EU-FR-1 volume `c6kdciozex`; the operator chose to wait for its H100 capacity rather than move regions. Refresh price and remaining prepaid credit at launch. Use checkpointed resource windows with host-enforced stop times; the former three-hour/1,500-update proposal is withdrawn as the whole-program exit. A resource window ending saves and resumes the unfinished stage, without launching another diagnostic campaign or declaring the curriculum completed. Total spend remains within the supplied account credit; no funds are added. If credit, capacity, sources or the native executor prevent completion, report the exact unfinished stage rather than silently drop it.
5. **Test the finished candidate once using the existing plan.** After all declared training stages finish, freeze the candidate and run the already planned final evaluation: CoS/eight-voice/captain behavior, inherited-capability retention, native illumos/guidance and the supervised operator C2 interaction. Reuse existing parent/teacher results when comparable; the prepared receipt cases and behavior spots belong here, not between stages. Reload/serving checks are done once for the final artifact. Report every untested or failed capability explicitly. Active-adapter acceptance, the current T3 gate-lift condition and the canary then follow the existing governance.

The deliverable is the fully trained candidate from the declared guided curriculum plus its final report, not proof that another short SFT pass can run. This does not promise that every capability will pass: training completion is the intended endpoint before acceptance testing. Actual source preparation, the mixed archive, preference/on-policy integration and the complete launch command are still unfinished; the next work is to produce those artifacts and execute the program.
