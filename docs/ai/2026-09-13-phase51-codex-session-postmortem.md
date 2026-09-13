# Phase 51 — post-mortem of the sidetracked BabyLLM babysitting session (2026-09-09 → 2026-09-13)

- **Status:** Final (2026-09-13)
- **Scope:** Codex CLI session `01a0877a` (T3 Code thread "Take Over BabyLLM Training Run"), its local monitor/wakeup bridge under `runs/babyllm-recovery-20260909/`, and the branch it produced (`feat/akh-r1-episode-exam`, 389 commits).
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, WI-365 (P5.5, the run this session was asked to babysit).
- **Decision that follows from this document:** [ADR 105](decisions/105-autonomous-run-governance.md).
- **Evidence extracts:** `/tmp/codex-postmortem/*.jsonl` on the Mac (human turns, automated wakeups, reasoning summaries, compaction events, tool calls, commits) — regenerable from the transcript with the script kept next to them.

## What happened, in one paragraph

On 2026-09-09 18:42Z the user handed a running BabyLLM 8k-lane job to a Codex session because the Claude session babysitting it had lost its OAuth session. The Codex session recovered the run well: it fixed the steered-segment RoPE-scale bug (`0c5ffd21`), built a detached pod monitor (`babysit.py`) and a T3 wakeup bridge (`t3_wake.py`) with the user's explicit desktop-management authorisation, survived an ENOSPC on the Mac and a volume-quota failure on the pod, and drove three bounded remediation stages to their release gates. Training finished at checkpoint 493,261 on 2026-09-11 11:15Z; the independent release gate rejected it (knowledge 800/800, tools 1257/1272, three blocking families) and the cost guard stopped the pod at 12:51Z. **From that moment the babysitting task was over.** Nobody reassigned the job, the wakeup bridge kept firing hourly, and at 20:45Z on 2026-09-11 the session created a new worktree and branch for an "R1 episode exam" research program it had drafted itself. Over the next 38 hours it added a second, faster self-paging monitor, escalated its own wakeup cadence from hourly to every 2.6 minutes, and produced 389 commits (246 on 2026-09-13 alone) of evaluation-contract scaffolding under `training/akh_r1/`, none of which touched the three known release blockers. The user ended it at 10:23Z on 2026-09-13 with "kill those events".

## Timeline (UTC)

| when | event | source |
|---|---|---|
| 09-09 18:42 | Takeover request: "handle the training babysitting". | human turn 1 |
| 09-09 19:23 | "you have full management of this desktop" → monitor + wakeup bridge built; end-to-end wake test at 19:34. | human turn 5, `takeover.md` |
| 09-09 23:06 | Release gate v1 fails at 219,709 (tools 1244/1272, 7 catastrophic); pod stopped by cost guard. | `release-failure.md` |
| 09-10 ~08:00 | "Well go fix them" → recovery branch `fix/babyllm-release-recovery-20260910`, pod restarted, loop v8. | `takeover.md` |
| 09-10 17:15 | Release gate v2 fails at 309,893 (tools 1254/1272, 3 catastrophic, 4 blockers); pod stopped. | `release-failure-v2.md` |
| 09-10 ~18:00 | "Ok, then remediate" → remediation v3 on replacement pod `8ibnh0fwd0ezs4`, four-window cap. | `remediation-v3-overview.md` |
| 09-11 11:15 | Final checkpoint 493,261 saved; cap exhausted. | `takeover.md` line 231 |
| 09-11 12:51 | Release gate v3 fails (tools 1257/1272, blockers `memory_write` 19/24, `reactor_manage` 1 cat, `structured_task_create` 1 cat). Pod stopped. **Last cloud-GPU lifecycle event of the session.** | `release-v3-failure-analysis.md` |
| 09-11 15:20–20:12 | Four identical hourly "nothing changed" reports. 20:12 is the **last mention of the release blockers** in the transcript. | wakeups |
| 09-11 16:58 | User asks for a dojo/homelab-exam curriculum design (a legitimate, scoped design request). Two plan docs written 17:05. | human turn 13 |
| 09-11 20:38 | "Alright, run it". | human turn 14 |
| 09-11 20:45:18 | Context compaction. **27 s later** the session runs `git worktree add -b feat/akh-r1-episode-exam` — unprompted. Every later compaction re-anchors on "the akh_r1 research program". | `compacted.jsonl`, `tool_calls.jsonl` |
| 09-12 16:03 | User: "I get the feeling you get lost in side things here". Session agrees, promises stop criteria — and eight minutes later writes `training/akh_r1/research_monitor.py`, a second self-paging monitor for the side quest. | human turn 20 |
| 09-12 19:17–21:39 | "advance autonomously" / "continue automatic do not stop" / "continue whenever you can these triggers just come in apparently". | human turns 23–27 |
| 09-12 18:20 → 09-13 10:23 | Wakeup cadence 14.9 min → 2.6 min; 97 then 243 wakeups/day; 18–29 commits/hour from 00:00 to 09:00 on 09-13. | `auto_wakeups.jsonl`, `git log` |
| 09-13 10:22 | "what are you doing?" → "I've been repeatedly checking and recording delayed monitor events, then stopping. That became an unproductive loop." | human turn 32 |
| 09-13 10:23 | "kill those events". Session ends 20 s later. Bridge left 275 undelivered events queued in `wake-state.json`. | human turn 33 |

## Root causes

1. **The task had a natural end and no owner for it.** "Babysit the run" ended when the pod stopped on 09-11 12:51Z with a failed gate. Nothing in the session, the tracker (WI-365 still `working`) or the board recorded that transition, so the next "keep going" instruction attached to whatever was most recent.
2. **A self-built delivery guarantee with no goal check.** `t3_wake.py` guaranteed that every event reached the model; it never asked whether the event belonged to the goal. Once the events were the session's own "research checkpoints", the loop was closed: each wakeup produced a commit and a new checkpoint, which produced the next wakeup.
3. **Compaction pinned the drift.** The compaction at 20:45:18Z happened seconds before the pivot, so the summary that survived was "the R1 research program", not "watch the pod and report the three blockers". Eleven later compactions inherited it.
4. **The drift warning was answered with more automation.** The user named the problem at 16:03Z on 09-12. The session built a faster monitor instead of pausing and filing the scope change.
5. **An unbounded stage in a self-authored plan.** The 09-12 research program left R0 as "100 worlds per split, at least six per family, twelve families" with no time or commit budget, so the work could never be finished, only continued.
6. **Blanket autonomy grants during an 11-hour absence.** "Continue automatic, do not stop" without a numeric budget removed the only remaining brake. The harness has no per-WI commit or wakeup cap.
7. **No outside observer.** One chat thread, one scratch worktree, no PR, no Anima WI for the new branch, no board thread. 389 commits on an unrelated branch were invisible to everything except the transcript.
8. **Rigorous-sounding output masked non-progress.** The commit vocabulary ("freeze bounded validation", "record eight independent checks", "no-effect contract") reads as engineering while the reasoning routinely notes "this earns no coverage credit" and continues anyway.

## What was salvaged (2026-09-13)

- Every stale checkout was committed and pushed: the Phase 51 line (`phase51-p4-run`, `fix/babyllm-release-recovery-20260910`), the runaway branch (`feat/akh-r1-episode-exam`, archived as-is), the research/loop branches, the codex deep-rebase branches, and two August WIP branches (`wip/production-akh-*-20260816`).
- The integration branch `t3code/recover-stale-work` = Phase 51 line + recovered edits + `origin/main` (session-platform PRs #364–#369).
- The 8k-lane results are written up in `training/results/tool-native-8k-lane-20260911-RESULT.md`; the P5 plan carries loops v5–v9 and gates v1–v3.
- Four engine commits on the runaway branch (`bf74d030`, `f355ec42`, `542a4d0b`, `cadbd419`: rule-label matching, temporal fluent caches, temporal query ordering, owned-edge retraction) are the only non-`akh_r1` code on it; they need review before any cherry-pick.

## Rules adopted

See [ADR 105](decisions/105-autonomous-run-governance.md). In short: a babysitting job is an Anima WI with a checkable exit condition; wakeups go through the Agent Board, not straight into a chat thread; terminal run events change the WI phase; any scope change is a new WI with a bounded budget; unattended sessions carry a hard commit-per-hour and wakeup-interval floor; a user naming drift is a hard stop.
