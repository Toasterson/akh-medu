# Akh-Medu parked-item audit — 2026-09-14

Scope: the user-requested `/babysit-lane` audit followed by explicitly authorized code repair and execution recovery. All 44 initially parked Akh-Medu items were opened individually through Anima MCP; each received a recovery disposition in its thread. This is an operational audit and bounded recovery, not implementation completion.

## Implemented recovery after the audit

- Anima integration PR719: https://code.aopc.cloud/toasterson/Anima/pulls/719,
  head `f1b07ff1` (supersedes `2a8e336d`). Integrates the preserved
  WI855/WI857 fixes and concurrent PR718 through `4d1d2c4a`, with exact-head
  PR lookup, terminal follow-up publication, bounded admission reconciliation,
  safe retirement diagnostics, and migration144. Existing phase budgets and
  owner/retirement/unknown-operation fences remain intact.
- WI858 accounting: exact `operator_resolved` becomes an external-cancellation
  witness; `unknown_outcome` becomes a non-charging runtime-refusal witness.
  Both retain failed execution truth and use a durable scheduling backoff of
  at least60seconds. Replays charge the scheduling counter only once. True
  code failures still exhaust their normal budget. Unresolved unknown turns
  still retain ownership. Added native nightly registration for concurrent
  outbox and cancelled-primary/completed-follow-up publication tests.
- Local Anima evidence:4client tests,20Forgejo tests,1PostgreSQL concurrent
  outbox fixture, and completion/retirement/follow-up publication fixtures pass.
  The combined completion fixture covers both unknown-outcome codes, pending
  cleanup, exhausted implementation budget, real failures and replay. Native
  push run4451 failed the project-binding source enumeration because its parser
  rejected a rustfmt-wrapped cancel_session dispatch arm. Fixed the parser without
  removing any binding checks; three local enumeration tests now pass. The
  current f1b07ff1 head has push4456 waiting in native CI and a full nightly
  dispatch accepted with HTTP204. Superseded own nightly4452 was cancelled; do not treat local tests as native CI, merge, deployment or resumed work.
- WI403: existing PR366 advanced to `0abffd0e` including repair `e5c69eca`.
  Valid MuSiQue paragraph0 is preserved via Option; absent index is distinct.
  Embedded pinned IDs prevent collision protection silently disappearing when
  cwd changes. Fixed a test-only QueryTrait import and merged current main.
  All22offline multihop tests pass. Canonical resync verified the new head;
  unparked with atomic gate-processing context. Review/native CI remain pending.
- WI243: existing PR351 advanced to `803d91b9`, including repair `51547837`.
  Uses WorkspacePaths.root for training.redb, supplies the collector on automatic
  daemon startup, merges current main and removes duplicate OTLP imports.
  Server cargo check and11collector persistence/concurrency tests pass.
  After network recovery, canonical PR resync resolved PR351 at803d91b9 and
  invalidated stale AI approval. Individually unparked with atomic validation
  context; readback ready with no park and real PR identity. Current-head native
  CI/review and runtime collection remain unproved. WI50 identity also recovered
  canonically as PR324 atddd1b302; its conflicts and hard edges remain.
- Additionally restored dependency-driven eligibility for WI317,316,311,310,
  309,269,247,246,235. Each received an atomic individual brief; live readbacks
  retain all unsatisfied hard edges and show ready with no park. These are
  waiting on prerequisites, not newly witnessed execution. WI318 was already
  restored during the initial audit.
- Native logs now prove that WI390,296,223,225,224 share shell_exec subprocess
  spawn ENOMEM failures. Their feature-specific tests were not the failed
  assertions. All three isolated tests pass locally with both server and native
  postgres features. The shared CI repair is committed and
  published as PR370 at5992e248 (https://code.aopc.cloud/toasterson/akh-medu/pulls/370):
  move three real-shell cwd tests into an explicitly registered integration
  executable, run serially, and use quiet libtest output to reduce native log
  transport volume. Do not infer that this repair has landed until its PR,
  native CI and merge are recorded. PR370 native run3497 is waiting; none of
  its six current-head statuses is green yet.
- The latest bounded scheduler pass still assigned nothing: Tecton no_capacity,
  Horus dispatch_failed;40hard-dependency candidates excluded in the post-network
  pass. Released claims
  `scheduler:1:01a09f98-74d0-7652-858f-4c4ead9a7cd8` and following claims are
  not execution evidence. Eleven fleet sessions were provisioning at latest
  occupancy, including new WI403 maat/sekhmet review reservations and concurrent
  Anima WI846/WI851 reservations; no first
  witnesses were reported. No owner was cancelled or artificially released.

### Shared service interruption during recovery

Forgejo Git push/PR creation returned HTTP500 and health returned424 during
PostgreSQL HA failover. After the database regained a primary, the stor01
Cilium desired service state listed the primary but the actual BPF service map
had zero backend entries. A direct primary TCP probe succeeded while the
postgres-ha-rw ClusterIP failed. Other checked nodes had correct forwarding maps.

The user explicitly approved restarting only the stor01 Cilium agent. Deleted
only cilium-6tlrc; replacement cilium-chgcj became Ready. Verified the BPF
postgres-ha-rw backend returned, Forgejo TCP access through the ClusterIP
succeeded, health returned200, PR370 Git push and API creation succeeded, and
Anima/Solstice MCP reads recovered. No production SQL was executed. A later
managed PostgreSQL switchover changed the backend again; the new Cilium map
tracked the then-current EndpointSlice. This restores observed connectivity,
not a claim that all future Cilium/PostgreSQL failovers are permanently fixed.

## Initial audit outcome

- Parked count: 44 → 43.
- WI-318 individually unparked with atomic cleanup context and both hard edges retained. The reconciler later projected `working`; final readback has no clock, lease or session. It is DAG-held queue state, not dispatch.
- WI-35 canonical PR resync updated the recorded head from d6a5f6f to 7e03510 and cleared stale CI green. Its operator hold and three dependencies remain.
- WI-305/PR339 and WI-298/PR325 resync confirmed current identities without changes.
- WI-243 and WI-50 canonical identity repair timed out. Readbacks still show unresolved PR identity; no successful repair inferred.
- WI-180/WI-366 dependency witness reconciliation did not qualify done predecessor labels as completion evidence; edges were retained.
- Added individual recovery notes to all 44 threads and fresh regression evidence to existing Anima WI-858, WI-365, WI-855, WI-857, WI-450 and Akh-Medu WI-354. No duplicate prevention WIs filed.
- No merge, code fix, deployment, owner release, forced approval, retry-budget bypass or new witnessed session was claimed.

## Why the lane still cannot progress

The bounded canonical scheduler tick had background scheduling enabled, excluded 31 dependency-held candidates, and assigned nothing. Tecton returned `no_capacity`; Horus returned `dispatch_failed`. Its released allocation claim `scheduler:1:01a09f37-6f4b-77e0-9750-2d8f7e4328ee` is not an assignment.

Seven fleet reservations remain `provisioning` since September 13 with no first witness. This legacy-looking projection is insufficient to infer runtime inactivity: WI-426 already has a published Git head and a retained pending PR intent.

| Project / WI | Executor | Session | Current evidence |
|---|---|---|---|
| Anima WI-305 | maat | 01a09b7a-6bba-7aa3-b7c2-71342b0c997e | provisioning; first witness absent |
| Anima WI-305 | sekhmet | 01a09b7a-6a2f-7d43-81b8-279a9c3caa4c | provisioning; first witness absent |
| Anima WI-857 | astra-impl-4 | 01a09b5c-0242-7790-8bf0-bc708407c186 | provisioning; first witness absent |
| Anima WI-855 | astra-impl-1 | 01a09b53-2d08-79f2-b7f1-beebc6811a25 | provisioning; first witness absent |
| Akh-Medu WI-417 | tecton | 01a09b37-051b-7a52-9b0f-58583439c353 | provisioning; first witness absent |
| Akh-Medu WI-405 | astra-impl-3 | 01a09b36-74ee-7842-b5fb-c567a6564788 | provisioning; first witness absent |
| Akh-Medu WI-426 | astra-impl-2 | 01a09b35-994f-7ca2-82af-ded4886aaca5 | provisioning; first witness absent |

Server logs around 09:21 UTC still show WI-417's immutable dispatch/central identity mapping conflict and WI-405/WI-426/WI-857's unconfirmed scoped Forgejo observations. Original owners and uncertain actions remain authoritative. WI-426 publication readback: Git head `8acbe1495c5f3c9db5dfc772668536e0888d3a98` published; PR intent `86c27316adff2e02e1aa9a66c12bf741974f4b3503ab164257dba69abd376765` pending, `creation_reserved=false`.

Existing Anima WI-855 and WI-857 contain concrete preserved implementations and local validation evidence; that does not prove integration, native CI, deployment or live reconciliation. The active work must retain its existing owner. Age-only cancellation would risk losing or duplicating publication.

## Recorded stop reasons at the initial snapshot

| Machine label | Items |
|---|---:|
| no-progress | 9 |
| lane-circuit-breaker | 12 |
| retry-budget | 5 |
| rebase-conflict | 2 |
| turn-failure | 2 |
| operator-hold | 5 |
| too-big | 4 |
| ci-retry-generations-exhausted | 2 |
| no-new-commits-loop | 3 |

These are machine stop labels. The item table separates them from current code, dependency and delivery requirements. In particular, empty turns, provider faults, elapsed waiting and genuine code failures require different recovery.

## Item-by-item disposition

| Item | Initial park | Live hard blockers | Current branch evidence | Diagnosis and required recovery |
|---|---|---|---|---|
| WI-403 | no-progress | None reported | c80435b6; 4 files changed, 1517 insertions(+), 1 deletion(-); merge-tree clean | Three recent failed attempts report operator_resolved unknown-turn reconciliation, not a new implementation. PR366 still drops a real MuSiQue paragraph index 0. Preserve the 4-file implementation; distinguish absent index with Option<usize>, retain index 0, and add the regression. Merge-tree is clean. Retry needs available admission plus that exact brief. Prevention: Anima WI-858/WI-855. |
| WI-401 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | None reported | 658f105a; 2 files changed, 96 insertions(+), 1 deletion(-); conflicts: CONFLICT (content): Merge conflict in src/agent/context/compacting.rs | Imhotep's executor-wide circuit breaker parked this item. The two-file PR359 only adds directive types; compacting rendering/budget/atomic-section wiring is absent, and compacting.rs now conflicts. Recover the existing types and finish all three context managers after a scoped conflict resolution. Prevention: Anima WI-365/WI-845; Akh WI-354. |
| WI-390 | retry-budget | None reported | 9d20e37c; 3 files changed, 757 insertions(+), 8 deletions(-); merge-tree clean | Native CI failed exit101 on the exact approved head 9d20e37c; its 3-file reward implementation merges cleanly. Full CI log retrieval was unavailable (Solstice bearer absent), so the concrete failing diagnostic is not newly established. Preserve approval/code; obtain request bd7fe1a7-01ee-44b9-8839-76a14d9f8875 before any retry or code edit. Prevention: Anima WI-450 for proven infrastructure faults only. |
| WI-367 | no-progress | None reported | d7ce6aa7; 1 file changed, 604 insertions(+); merge-tree clean | Repeated unchanged published head after a reported ENOSPC compile-gate failure and exhausted editing turns. PR360 contains only synthesis.rs; registration, FixtureExtractor::backend_name, and MCP wiring need recovery from preserved Changes and completion. Merge-tree is clean. Do not count a preserved-but-unpublished diff as no work. Prevention: Anima WI-452/WI-471; Akh WI-354. |
| WI-366 | rebase-conflict | WI-363 (done; witness unqualified) | 7cbaefc2; ancestor of main; no unique diff | The historical rebase/581-file diagnosis is stale: old head 7cbaefc2 is now an ancestor of main with zero unique diff. Current load_checkpoint/load_model_only still load without the requested positional-encoding comparison. WI-363 remains without a qualifying dependency witness after reconciliation. Recover its existing evidence, then implement the narrow typed mismatch guard on current main; do not close merely because the old head landed. |
| WI-358 | no-progress | None reported | No usable resolved head in bounded detail; see recovery note | Twenty recorded attempts culminated in empty turns; latest handoff says tool-call limit before first edit. No published implementation. Reuse the existing per-round KV analysis and implement a bounded suffix-prefill seam with greedy parity and measured per-round token counts. No timer reset justified from the bounded evidence. Prevention: Akh WI-354 and Anima WI-471. |
| WI-354 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | None reported | 4a35c4ba; 39 files changed, 150 insertions(+), 261 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/agent/tools/agent_management.rs; CONFLICT (content): Merge conflict in src/agent/tools/code_gen.rs; CONFLICT (content): Merge conflict in src/agent/tools/delegate.rs; CONFLICT (content): Merge conflict in src/agent/tools/entity_schema.rs; CONFLICT (content): Merge conflict in src/agent/tools/kg_delete.rs; CONFLICT (content): Merge conflict in src/agent/tools/memory_recall.rs; CONFLICT (content): Merge conflict in src/agent/tools/pattern_mine.rs; CONFLICT (content): Merge conflict in src/agent/tools/skill_author.rs | The executor circuit breaker interrupted the tool-description audit itself; latest reports also show repeated reconnaissance exhausting the tool limit. PR357 has 39 changed files and 8 current conflicts. Recover edits without losing current descriptions, complete missing descriptions, and publish the before/after retrieval sweep. Add first-edit and durable-output measurements for real implementation turns; description prose alone is not prevention proof. |
| WI-343 | turn-failure: error (Internal error: "ACP session \"akh-1788930517991362347\" already has a prompt in flight; wait for that turn to finish") | None reported | 8a919297; 2 files changed, 776 insertions(+), 10 deletions(-); merge-tree clean | Imhotep repeatedly reused ACP session akh-1788930517991362347 while a prompt remained in flight, interspersed with quota-stuck cancellations. PR356 is a cleanly mergeable 2-file config change. Fix/reconcile the retained session mapping and provider admission before retrying the config brief. Prevention: Anima WI-855/WI-858/WI-365. |
| WI-342 | operator-hold | None reported | No usable resolved head in bounded detail; see recovery note | Historical attempts never reached the runner and hit ACP connection refused; now operator-held. No PR exists. The current slice combines opt-in Oxigraph and a measured mmap decision. Retain the hold while its reason and healthy admission are resolved; do not equate this with code failure or claim mmap qualification. Prevention: Anima WI-365/WI-471. |
| WI-324 | too-big | None reported | No usable resolved head in bounded detail; see recovery note | An in-flight deadline followed repeated empty analysis turns, including a final report of no first edit. This does not prove the metrics slice is too large. Reuse the mapped OODA/tool/subsystem seams, deliver one real measured path first, and account actual execution time before any credit. Prevention: Akh WI-354 and Anima WI-506/WI-471. |
| WI-318 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-231, WI-345 | No usable resolved head in bounded detail; see recovery note | Executor-wide Imhotep circuit breaker affected an item with no session or PR. Individually unparked with atomic exact-file cleanup context; WI-231 and WI-345 edges preserved. Readback: the reconciler moved it to working, but no scheduling clock/session exists; still only DAG-held queue state. Stopped further routing to this executor. Prevention: Anima WI-365/WI-471. |
| WI-317 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-231, WI-345 | No usable resolved head in bounded detail; see recovery note | Executor-wide circuit breaker, not an observed implementation failure; no PR. Wait for WI-231 and WI-345, then implement authoritative errored advertisement plus wake-as-retry, preserving fail-closed diagnostics. Keep hard edges; further routing stopped after WI-318 had no scheduling decision. |
| WI-316 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-231, WI-345 | No usable resolved head in bounded detail; see recovery note | Executor-wide circuit breaker, no published implementation. WI-231 and WI-345 are real prerequisites. Required next work is the actual seeds status/apply daemon handlers with lifecycle-correct CLI and idempotency tests. Further routing stopped after WI-318 had no scheduling decision. |
| WI-311 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-308 | No usable resolved head in bounded detail; see recovery note | Executor-wide circuit breaker while waiting for WI-308. No measurement artifact exists. First finish the shared harness, then run only vocabulary recall on a pinned real model and publish per-entity outputs. Keep DAG edge and do not fabricate benchmark results. Further Imhotep routing stopped. |
| WI-310 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-308 | No usable resolved head in bounded detail; see recovery note | Executor-wide circuit breaker while WI-308 remains unfinished. Wait for its matched harness, then run constraints-on/off with identical prompts, budget and model, including selection scoring. No experiment result is implied by queue admission. Further Imhotep routing stopped. |
| WI-309 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-308 | No usable resolved head in bounded detail; see recovery note | Executor-wide circuit breaker with WI-308 pending. Wait for repaired structural scoring and matched harness, then measure content-plus-frontmatter correctness under real load. Retain hard edge; further Imhotep routing stopped. |
| WI-305 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | None reported | 42e2e9e5; 1 file changed, 103 insertions(+), 6 deletions(-); merge-tree clean | Executor-wide park despite a later review. Current PR339 head42e2e9e contains the canonicalized macOS temp-path assertion; the recorded green belongs to older head18eac635. Current review/CI gates are not cleared. Canonical PR resync confirmed identity; merge-tree is clean. Needs exact-current-head gates on healthy capacity, not another implementation rewrite. |
| WI-299 | too-big | WI-298 | No usable resolved head in bounded detail; see recovery note | In-flight deadline, no implementation PR or detailed turn diagnostic in the bounded trace. WI-298's per-source manifest is still a real prerequisite. Full re-projection must wait for that contract. Do not infer actual oversized scope or grant unmeasured time credit. |
| WI-298 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | None reported | 0cde8c70; 3 files changed, 469 insertions(+), 56 deletions(-); merge-tree clean | Executor circuit breaker after earlier deadline. PR325 head0cde8c7 is a clean 3-file merge with exact-head AI approval; historical red CI targets d5d1af0, not this head. Canonical resync confirmed head. Resolve the recorded shared-triple-retention regression against the current code and obtain fresh native CI without replaying broad implementation. |
| WI-296 | retry-budget | None reported | a33e412b; 2 files changed, 49 insertions(+), 14 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/bin/akhomed.rs | Exact-head CI red plus later worktree clone collision (destination already exists) and quota cancellation. Current 2-file PR362 also conflicts in akhomed.rs. Preserve approved ritual fixes; reconcile owned worktree rather than delete it, resolve the conflict, then retrieve/fix CI request3a5a8983-67ca-4049-a7fd-c02af2226763. Prevention: Anima WI-844/WI-845/WI-450; secret-redaction issue already WI-796. |
| WI-285 | too-big | WI-345 | No usable resolved head in bounded detail; see recovery note | Deadline after an earlier documented dispatch-only retry park; no PR. Deployment docs depend on WI-345. Finish deployment wiring before writing truthful chart examples. No evidence justifies splitting this docs slice or resetting its whole historical timer. |
| WI-269 | no-progress | WI-403 | No usable resolved head in bounded detail; see recovery note | Connection-refused attempts produced no-progress park. Its real multi-hop evaluation prerequisite WI-403 is still unfinished. Keep the dependency, then implement note_search/note_read traversal against that contract. Further Tecton routing is capacity-blocked. |
| WI-263 | turn-failure: error (Pre-push compile failure (diff hash: 36de86a4)) | None reported | c88dd9b5; 1 file changed, 667 insertions(+); merge-tree clean | Recorded pre-push compile failure 36de86a4 plus a handoff claiming fixes were unstaged while the gate compiled a stashed HEAD. PR361 contains only harness.rs and now merges cleanly. Recover the exact persisted edits, verify ReferencePair ordering derives and removed private re-export, then validate the candidate diff actually submitted. Prevention: Anima WI-452/WI-530. |
| WI-247 | no-progress | WI-245, WI-246 | No usable resolved head in bounded detail; see recovery note | Connection-refused attempts misrecorded as no-progress. Training scheduler still depends on WI-245 and WI-246. Preserve those edges and inference starvation acceptance; no reason to run against guessed trainer/verifier APIs. |
| WI-246 | no-progress | WI-245 | No usable resolved head in bounded detail; see recovery note | Connection-refused attempts, with WI-245 still required. Wait for the actual trainer, then implement verifier and automatic rollback. No implementation failure or justified time credit is established. |
| WI-243 | no-progress | None reported | No usable resolved head in bounded detail; see recovery note | Not infrastructure-only: preserved CI diagnostics identify nonexistent WorkspacePaths.data_dir and a missing TrainingDataCollector argument. Repeated witnesses also cite invalid PR0 although real PR is351. Canonical identity repair timed out; readback still has no resolved PR; repair is unconfirmed. Fix those precise compiler sites on the preserved branch after identity/admission recovery. Prevention: Anima canonical PR repair and WI-844/WI-452. |
| WI-242 | retry-budget | WI-239 | f2f1cea8; 2 files changed, 569 insertions(+), 7 deletions(-); merge-tree clean | Retry budget plus historical false claims of being on main. Current PR319 headf2f1cea8 is not an ancestor of main, but now merges cleanly. The 2-file neural bridge implementation still depends on WI-239. Preserve and validate it only after that prerequisite; do not confuse branch history with main. |
| WI-239 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | None reported | 939eddd2; 15 files changed, 680 insertions(+), 729 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/nlu/llm_translator.rs; CONFLICT (content): Merge conflict in src/nlu/mod.rs | Executor circuit breaker after prior deadline; PR349 has exact-head review but no current CI and real llm_translator.rs/nlu/mod.rs conflicts. Preserve backend/lockfile retirement work and resolve against current APIs, then run exact-head CI. Prevention: Anima WI-845/WI-365. |
| WI-237 | too-big | WI-236 | 06d400f7; ancestor of main; no unique diff | Deadline after unproductive baseline-fix turns. PR251's head is an unrelated main ancestor with zero unique diff; render feature is not delivered. WI-236 remains required. Wait for birth extras, then start a bounded real render implementation; no completion from this empty PR. |
| WI-236 | retry-budget | WI-235, WI-429 | 7b3a1554; 2 files changed, 57 insertions(+), 45 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/bin/generate_qa_dataset.rs | Retry budget with recorded runner restart. PR250 contains a small partial QA change and now conflicts in generate_qa_dataset.rs. WI-235 and WI-429 remain required. Recover only birth-extra scope after those children, preserve prior code, and use a fresh reviewed head. |
| WI-235 | no-progress | WI-234, WI-315 | 06d400f7; ancestor of main; no unique diff | Connection-refused no-progress park; PR249 head is unrelated main history with no unique implementation. WI-234 and WI-315 remain prerequisites. Keep DAG waiting, then implement the wizard against the completed lifecycle/Genesis contracts. |
| WI-234 | operator-hold | WI-428, WI-429, WI-430, WI-431, WI-432, WI-433 | 168950c3; 6 files changed, 32 insertions(+), 23 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/acp/mod.rs | Deliberate protective hold: previous automated review graded a decomposed parent before its children completed. Six child edges WI-428..433 remain pending; old PR330 is only 6 files and conflicts in acp/mod.rs. Retain containment until children satisfy or Anima WI-846's review fix is integrated and live-verified. Do not weaken legitimate current-diff review policy. |
| WI-231 | no-progress | WI-345 | afb26ca9; 17 files changed, 611 insertions(+), 88 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/acp/mod.rs; CONFLICT (content): Merge conflict in src/main.rs; CONFLICT (content): Merge conflict in src/snapshot.rs | Unchanged-head completed/cancelled turns caused no-progress; WI-345 is pending. Current ownership/autostart branch conflicts in ACP, main and snapshot. Preserve its 17-file work, wait for deployment contracts, and recover only the remaining lifecycle scope. |
| WI-225 | ci-retry-generations-exhausted | None reported | a9f76a56; 1 file changed, 95 insertions(+); merge-tree clean | CI retry generations exhausted for a review-only item whose 95-line report merges cleanly. Current kind is review; historical toolchain/ownership faults and old-head approval are not current CI proof. Verify the linked findings artifact and current-head gates; do not re-run repository implementation to produce a second report. Prevention: Anima WI-450 and deliverable-aware lifecycle. |
| WI-224 | ci-retry-generations-exhausted | None reported | f36d1dad; 1 file changed, 214 insertions(+); merge-tree clean | CI retry generations exhausted although the review produced a linked findings Document. Current 214-line report merges cleanly; approval is stale for current head. Verify the findings delivery and correct document-aware terminal gate, preserving report and CI history. Full old CI log is not newly available. Prevention: Anima WI-450 and deliverable-aware lifecycle. |
| WI-223 | retry-budget | None reported | d7d6f0b8; 9 files changed, 878 insertions(+), 9 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/acp/mod.rs | Native CI exit101 on exact approved head d7d6f0b8 exhausted retries; current PR242 also conflicts in acp/mod.rs. Preserve session registry/mailbox implementation and corrected test helper. Obtain CI request608790b8-f4de-4ee5-952e-a6fb3807adeb and resolve current conflicts before a single justified retry. |
| WI-180 | no-new-commits-loop: 3 consecutive sessions with no commits | WI-293 (done; witness unqualified) | 03585ad0; 12 files changed, 1569 insertions(+), 43 deletions(-); conflicts: CONFLICT (content): Merge conflict in crates/akhomed-store/src/lib.rs; CONFLICT (content): Merge conflict in crates/akhomed-store/src/migration/mod.rs | No-new-commits review loop with substantive snapshot gaps; current PR329 conflicts in store exports/migration registration. Canonical reconciliation still does not qualify WI-293's done label as a witness. Preserve snapshot work; recover prerequisite evidence, enforce WAL flush failures, FK/concurrency and six-table restore proof. Correct historical guessed Anima paths: this is Akh-Medu/akhomed-store. |
| WI-95 | no-new-commits-loop: 3 consecutive sessions with no commits | None reported | 741c6506; 4 files changed, 526 insertions(+), 45 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/agent/mod.rs | Review-only unchanged-head loop; current PR236 still has recorded fabricated historical event timestamps and now conflicts in agent/mod.rs. Keep AI gate uncleared. Replace synthetic history with observed lifecycle events/unknown states and restart tests in Akh-Medu, not the guessed Anima migration paths. Do not merge a passing test that merely asserts invented events. |
| WI-50 | rebase-conflict | WI-405, WI-406 | ddd1b302; 16 files changed, 1719 insertions(+), 245 deletions(-); conflicts: CONFLICT (content): Merge conflict in src/agent/delegation/mod.rs; CONFLICT (content): Merge conflict in src/agent/delegation/peer.rs; CONFLICT (content): Merge conflict in src/agent/delegation/types.rs; CONFLICT (content): Merge conflict in src/provenance.rs | Historical rebase park is still real: provenance and delegation conflicts now exist. WI-405 and WI-406 remain hard prerequisites. Preserve old envelope code and address recorded bincode variant compatibility, delegation livelock and owned-access test issues only within final integration. Canonical PR identity repair timed out; readback retains unresolved number/owner/repo fields; no successful repair claimed. |
| WI-44 | operator-hold | WI-378, WI-379, WI-380 | 8edd59dc; 1 file changed, 23 insertions(+), 9 deletions(-); merge-tree clean | Operator containment after a genuinely decomposed broad bridge task. PR123 is merged upstream but its unique delta is a WIP marker, not the feature. WI-378..380 remain pending. Retain child ownership and complete final external-client/browser evidence only after real implementations. |
| WI-35 | operator-hold | WI-376, WI-377, WI-328 | 7e03510d; ancestor of main; no unique diff | Operator containment for logs/trace propagation with WI-376/WI-377/WI-328 pending. Live head7e03510 is already main history; old stored greend6a5f6f was stale. Canonical resync updated head and cleared stale CI green. No feature closure inferred; preserve logs/propagation child work and final live correlation requirements. |
| WI-28 | operator-hold | WI-370, WI-371, WI-372, WI-373, WI-374, WI-375 | 27c10ea4; 1 file changed, 101 insertions(+), 128 deletions(-); merge-tree clean | Operator containment on a decomposed inference fabric; PR270 is a one-file WIP rewrite, even though CI is green. WI-370..375 own the real code and device evidence. Retain hold/edges until real child witnesses or verified safe review scheduling; do not treat WIP-marker CI as implementation. |
| WI-26 | lane-circuit-breaker: 3 lane dispatch failures (executor faults) | WI-367, WI-368, WI-369 | 96cc16d1; 1 file changed, 46 insertions(+), 40 deletions(-); merge-tree clean | Executor circuit breaker on a parent whose PR269 is only a WIP rewrite. Children WI-367..369 remain pending. Wait for synthesis, validation and scheduler wiring, then implement remaining health/end-to-end acceptance. Preserve exact original scope; no generic parent replay. |
| WI-21 | no-new-commits-loop: 3 consecutive sessions with no commits | WI-404 | 60a3787b; 3 files changed, 424 insertions(+); conflicts: CONFLICT (content): Merge conflict in src/workspace.rs | No-new-commits loop after provider faults; PR272 contains config/client only and now conflicts in workspace.rs. WI-404's findings are in testing and explicitly report the historical queue/fetch/ack API does not exist. Resolve that API capability gap before implementing polling/ingest/ack; never invent endpoints from the old PA5 sketch. |

## Prevention ownership and completion criteria

| Existing work | Required prevention | Proof needed before calling it fixed |
|---|---|---|
| Anima WI-858 + WI-470/WI-471 | Central `operator_resolved`/unknown reconciliation, quota and non-execution outcomes must not spend implementation progress budgets. Keep true repeated code/no-output failures bounded. | WI-403 regression plus exactly-once accounting; durable infrastructure backoff; ownership fences preserved; fresh recovered witness. |
| Anima WI-365 | Stop provider/executor dispatch before exhausted capacity burns attempts; avoid parking never-dispatched DAG-held items when a lane trips. | WI-318/316/317/309/310/311 regression with unchanged edges, no failed-item counter charge, visible provider reset/wake and automatic eligible recovery. |
| Anima WI-855 / WI-857 | Reconcile immutable central mappings and retained exact Git/PR publication intents. | Current owners finish safely through canonical APIs; no duplicate PR creation; exact head delivered, reviewed and native-CI validated; live reconciliation succeeds after separately authorized rollout. |
| Anima WI-845 / WI-844 | Explicit preserved-branch conflict recovery reaches the implementer; failed preparation does not create invisible replay loops or overwrite occupied worktrees. | Head-bound recovery intent, preserved source, canonical witnessed implementation and normal Change/CI gates. |
| Anima WI-846 | Prevent obsolete decomposed-parent artifacts from being repeatedly graded/prep-parked while preserving valid current-diff review semantics. | Integrate the current conflicting PR; preserve dimensional review and atomic admission; verify appropriate live blocked-parent behavior before removing containment holds. |
| Anima WI-450 / WI-452 | Distinguish CI infrastructure faults and candidate/subturn gate iterations from genuine code failure; validate the actual submitted candidate. | Exact CI diagnostic attribution, one intended accounting event, bounded native retry, changed-head invalidation; no blanket reset. |
| Akh-Medu WI-354 | Reduce repeated reconnaissance that consumes the entire turn without editing. | Before/after recall and actual implementation measurements: rounds to first edit, durable Change/head, completed acceptance; wording or witness count alone is insufficient. |

## Other lane inventory

Initial snapshot: 38 working, 4 testing, 0 ready, 42 draft. Thirty-five working rows were reported sessionless by the stranded view; many are correctly dependency-held.

Testing WI-418 and WI-419 are analysis items; WI-404 and WI-319 are spikes. Their loop views have no merge-ready PR. WI-404's linked findings explicitly say the old transcript queue/fetch/mark-graphed sketch is not a live API contract; WI-21 must not implement against invented endpoints. No testing item was merged.

Draft items remained unpromoted: capacity/admission was not proved, and their predecessor/operator conditions were not waived.

## Initial-audit evidence limits (later updates above supersede these)

- Main was freshly fetched at `e7e70bfe` (Postgres store transaction gate documentation). Local ancestry/diff checks covered 29 available recorded/live heads; merge-tree checked every non-ancestor head in that set. These checks are not builds or tests.
- Current head comparisons take precedence over a stale `pinned_to_current` field attached to a historical CI row.
- Forensics reads were bounded to 8 recent events and thread messages to 1–5 recent messages per item. Older handoffs are identified as reported evidence. No historical elapsed-time credits were calculated or granted.
- Solstice MCP was reachable but unauthorized without its configured bearer token in this shell. Exact full CI logs were therefore unavailable; native exit101 summaries were not relabelled as infrastructure failures.
- No direct Anima database access, production restarts, deployments or destructive worktree cleanup.
- The babysit skill requires stopping further routing to an executor when the canonical recovery produces no scheduling clock/session. That applied after WI-318. Its remaining prerequisites and all protective containment holds were preserved.

Final full live MCP readback:32 of the original44 remain parked;12 were individually unparked with recovery context. Full execution recovery remains incomplete. New review reservations for WI403 still have no first witness. The original table records diagnosis; the implementation section above records subsequent repairs and changed dispositions.

