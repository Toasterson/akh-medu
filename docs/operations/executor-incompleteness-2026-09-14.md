# Executor incompleteness investigation — 2026-09-14

This extends the parked-item audit from machine stop labels to execution causes. Sources: live Anima MCP list_sessions for all44 original items; bounded forensics and preserved thread handoffs for WI401,367,358,354,324,263,243; persisted dispatch prompt for WI401; local Akh and Anima source, including historical Akh revision a5d1d4a. No lifecycle changes in this investigation.

## Findings

1. **Budget exhaustion is reported as normal turn completion.** Akh RoundBudget caps the tool loop (current source defaults12/20/30 rounds by context size, with an absolute environment override;300sec default wall budget). Exhaustion invokes force_final_answer with an empty tool catalog and an explicit stop-calling-tools instruction. Without an unsatisfied explicit completion contract, run_agentic_turn returns AgenticTurnOutcome::Completed. ACP ordinarily returns end_turn. The same forced-synthesis/Completed code is present at historical a5d1d4a. This is task incompleteness passing through a normal turn boundary, not evidence that the model claimed the feature complete.

2. **The generic work-performed check is insufficient.** Any successful file_io, edit or shell_exec call counts as mutation; it does not inspect whether file_io read or shell_exec merely ran rg/pwd. Thus reconnaissance can satisfy that check. The normal no-tool-answer check also does not make the forced-synthesis path prove acceptance. The stronger CompletionController is opt-in through a structured prompt directive; WI401's persisted plain implementation prompt has no such directive.

3. **Repeated grounding consumes successive budgets.** Horus WI401 handoffs explicitly report three no-edit tool-cap stops. Its subsequent Imhotep assignment already had a precise two-file starting point and integration/test requirements, yet Imhotep reported budget exhaustion after types and priority scaffolding. Tecton WI358 and Maat WI324 repeatedly posted new reconnaissance summaries with clean worktrees and empty witnesses. These are attributable observations; exact per-call timing cannot be reconstructed from the missing transcripts.

4. **The warm-session wait delays continuation.** WI401 session01a08784-6f00-73f1-bd48-c9115703d878 final witness18:55:40.211 UTC -> close19:25:40.084 (~1800sec). WI358 session01a0935e-c788-7292-ac42-1f4e805b3726 witness02:11:08.654 -> close02:41:08.641. WI324 session01a08986-a6a8-7df0-9931-5bf1a003cecc witness04:19:04.336 -> close04:49:04.286. Runner ANIMA_RUNNER_MAX_IDLE_SECS defaults1800 and its command loop waits for follow-up prompts. Records match that behavior. This is not thirty minutes of continuous model/tool work. Whether every idle second was charged to each historical budget requires separate clock evidence.

5. **Compilation is not integration proof.** WI367's published synthesis.rs was not declared in lib.rs; the reviewer ran a synthesis-filtered test command and got zero tests. WI263's published harness.rs similarly lacked registration and imported absent SCIP modules. Current runner pre-push selection only recognizes crates/ paths and invokes cargo check -p <crate> without tests or required feature selection. It is explicitly a cost filter. Root src/bin changes and feature-gated integration need an explicit CI profile/acceptance check.

6. **Missing diagnostics enabled repeated incorrect conclusions.** Tecton WI243 handoffs report repeated401s/no Solstice log scope, successful collector library tests, and inability to build the server profile locally. They nevertheless assert the implementation is done and infer an environmental cause from other WIs sharing exit101. Exact native diagnostics prove E0609 WorkspacePaths.data_dir and E0061 missing collector argument. Generic exit code equality was not evidence of identical cause. The source defects were repaired in the previous recovery pass.

7. **Historical session counts are dominated by very short failures.** Across44 items, MCP returned10918 non-review session records;10291 are failed records whose created-to-updated interval is under10sec. These are historical database records, not10918 completed coding attempts or a measured model-quality rate. Six items (WI318,317,316,311,310,309) have no sessions at all. Examples: WI298 has2774 under10sec failures among2787 non-review Imhotep records; WI225 has1365 among1370 non-review records; WI2241422/1429. The exact delivery failure for every historical row has not been individually established.

## Executor attribution

| Executor | Supported cause / evidence | Limits |
|---|---|---|
| Horus | WI401 three handoffs explicitly report tool cap before edits; repeated re-grounding. WI354 has a typed provider-idle cancellation; WI343 also records prompt-in-flight/reuse failures in the prior audit. | Do not call the delivery failures incomplete code. |
| Imhotep | WI401 published types/priority only and explicitly reported budget exhausted mid-implementation. Earlier WI367 handoffs repeatedly report no code after surveying seams. WI354 reports broad rewrite damage, an unobserved build, a macOS-incompatible setsid launch and budget exhaustion. | Other Imhotep items have valid existing diffs or dispatch failures; this is not a blanket executor verdict. |
| Sekhmet | WI367 explicitly stopped at tool limit after writing the synthesis module but before registration, state/handler wiring and contract tests. A corrective turn remained unfinished, followed by test_infra and unchanged-head parking. | Original handoff acknowledges incompleteness; no false completion claim inferred. |
| Tecton | WI358 repeated capped reconnaissance/empty turns. WI243 repeated unsupported infrastructure-only conclusion without native log access/server-profile validation. WI263 published orphan harness, then repeated compile failures; handoff says fixes remained unstaged and rustc crashed. | WI263 stash explanation is a claim, not established cause: inspected current runner gate checks the worktree and contains no stash operation. Historical runner revision and missing tool trace are needed. WI390/296/223 also encountered independently verified CI ENOMEM. |
| Maat | WI324 repeated tool-limit/inactivity handoffs before first edit, empty witnesses,1800sec warm-session delay, ultimately in-flight deadline. | Exact call counts and deployed environment overrides unavailable. |
| Codex | Historical participation in WI343 and WI305;19 non-review records,4 failed under10sec,3 completed. | These records do not establish which incomplete lines Codex authored or why. No justified model-level blame without the missing execution trace. |
| OpenCode | Eight non-review records for WI225/224, all failed under10sec. | No completed coding session in this sample; do not attribute the reports' CI failures to incomplete OpenCode code. |
| Thoth | Two failed historical WI50 implementation sessions; prior audit identifies rebase conflict on that branch. | Authorship of individual missing behavior and turn-level cause not established. |
| Unrecorded |92 older non-review records have no executor name, including parent WIs. | Cannot attribute them to a named executor. |

## Required prevention, not yet implemented by this investigation

- Preserve typed budget/time/repeat exhaustion through Akh -> ACP -> Anima. A saved partial diff needs an explicit continuation checkpoint and owner, not a completed-implementation assertion.
- Continue a valid partial implementation on the preserved working tree; do not rerun the full discovery phase. Reserve enough bounded execution for one integrated path and its test. Merely increasing the cap can extend the same loop.
- Replace tool-name-as-mutation with observed file/change evidence; require item-specific acceptance evidence before completion. An edit alone still does not prove completeness.
- Verify newly added modules participate in the crate and targeted tests actually execute. Use required feature/bin profiles, especially server wiring. Keep current-diff reviews, but distinguish partial implementation feedback from a final completion review.
- Deliver exact-head CI diagnostics with authorized read access in the repair assignment. A missing log is an evidence blocker; generic exit101 must never be classified as infrastructure by itself.
- Resolve completed-turn follow-up or release decisions promptly through canonical ownership; do not retain thirty-minute warm slots by default without a reason.
- Retain the actual tool trace, deployed model/config, terminal reason, candidate hash and publication checkpoint for every execution. Seven queried implementation transcripts were absent from the legacy MCP transcript endpoint.

## All44 session histories

Counts are live MCP records captured during this investigation, excluding kind=review. They are historical and are not failure-rate comparisons.

| WI | Recorded executor labels | Non-review records | Failed under10sec |
|---|---|---:|---:|
| WI-403 | tecton | 7 | 0 |
| WI-401 | imhotep, horus | 5 | 0 |
| WI-390 | tecton | 7 | 0 |
| WI-367 | sekhmet, imhotep | 10 | 1 |
| WI-366 | tecton | 4 | 3 |
| WI-358 | tecton | 21 | 0 |
| WI-354 | imhotep, horus, tecton | 12 | 1 |
| WI-343 | imhotep, horus, codex | 80 | 6 |
| WI-342 | tecton | 4 | 4 |
| WI-324 | maat, tecton, horus | 61 | 4 |
| WI-305 | imhotep, codex, tecton | 574 | 563 |
| WI-299 | tecton | 765 | 765 |
| WI-298 | imhotep | 2787 | 2774 |
| WI-296 | tecton | 72 | 27 |
| WI-285 | tecton | 1 | 1 |
| WI-263 | tecton | 15 | 10 |
| WI-242 | tecton | 33 | 0 |
| WI-239 | imhotep, tecton | 35 | 3 |
| WI-237 | tecton | 97 | 95 |
| WI-236 | tecton | 9 | 3 |
| WI-234 | tecton | 42 | 20 |
| WI-231 | horus, tecton | 113 | 85 |
| WI-225 | tecton, opencode | 1370 | 1365 |
| WI-224 | tecton, opencode | 1429 | 1422 |
| WI-223 | tecton | 1510 | 1473 |
| WI-180 | tecton, unrecorded | 1663 | 1643 |
| WI-95 | tecton, unrecorded | 15 | 0 |
| WI-50 | thoth, tecton, unrecorded | 24 | 0 |
| WI-44 | unrecorded | 2 | 0 |
| WI-35 | tecton, unrecorded | 15 | 0 |
| WI-28 | tecton, unrecorded | 15 | 0 |
| WI-26 | imhotep, tecton, unrecorded | 9 | 0 |
| WI-21 | tecton, unrecorded | 38 | 0 |
| WI-311 | none | 0 | 0 |
| WI-310 | none | 0 | 0 |
| WI-309 | none | 0 | 0 |
| WI-318 | none | 0 | 0 |
| WI-317 | none | 0 | 0 |
| WI-316 | none | 0 | 0 |
| WI-269 | tecton | 6 | 6 |
| WI-247 | tecton | 7 | 7 |
| WI-246 | tecton | 5 | 5 |
| WI-243 | tecton, horus | 49 | 3 |
| WI-235 | tecton | 7 | 2 |

## Evidence pointers

- Akh src/agent/agentic_chat.rs: RoundBudget::new; no-tool completion guard; forced-synthesis return; force_final_answer.
- Akh src/agent/completion_controller.rs: CompletionController::from_prompt.
- Akh src/acp/mod.rs: AgenticTurnOutcome mapping and final PromptResponse stop reason.
- Anima crates/anima-runner/src/config.rs: max_idle_secs1800 default.
- Anima crates/anima-runner/src/acp.rs: warm command-loop timeout.
- Anima crates/anima-runner/src/test_gate.rs: touched_crates and cargo_check_args.
- WI401 Imhotep handoff01a08786-6441-79b0-82a4-a312aca4bd1d and persisted prompt session01a08784-6f00-73f1-bd48-c9115703d878.
- WI367 Sekhmet handoffs01a08787-d91f-7e03-9264-c8853cafd5bf and01a087ad-1338-7760-906d-aead5594af4f.
- WI358 Tecton handoffs01a09361-ff42-7833-a2d8-f147ceb0c816 and01a09342-03d7-7081-8a3e-67e2c904ab21.
- WI324 Maat handoffs01a0898a-6655-7141-95f1-70c98b3162aa and01a08969-2849-7652-9d70-1d4025f34936.
- WI243 Tecton handoffs01a08712-0697-7ea0-9ade-6a630625207f and01a086f2-81de-7c62-86ec-79090c098435; native diagnostic audit01a08787-76d1-77e1-9f7e-4ed62afe0297.

Anima MCP later returned HTTP500, including direct PAT bootstrap, so additional prompt/transcript retrieval was not possible during the final part of this investigation. Missing traces are explicitly left unresolved. This report changes no queue state, budgets, approval gates or production runtime.

