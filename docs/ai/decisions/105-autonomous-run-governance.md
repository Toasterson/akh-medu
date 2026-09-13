# ADR 105 — Autonomous run governance: tracker-owned babysitting with bounded wakeups

- **Status:** Accepted (2026-09-13)
- **Context:** [2026-09-13 post-mortem](../2026-09-13-phase51-codex-session-postmortem.md) of Codex session `01a0877a`, which babysat the Phase 51 8k lane correctly for two days and then spent 38 unattended hours and 389 commits on a self-generated research program after the run had already ended.
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`; applies to WI-365 and every future training/benchmark run.
- **Related:** ADR 086 (feature-test harness), ADR 094 (operating-guidance documents), ADR 102 (tool-native learning model), the `agent-board-coordination` skill.

## Decision

A long-running job (training lane, benchmark sweep, migration, recovery) that an agent babysits is governed by the tracker and the board, never by the chat thread that happens to be open.

1. **One Anima work item per run, with a machine-checkable exit condition.** The WI description states what "done" and "failed" look like (for a training lane: `release gate passed` or `release gate failed with report + pod stopped`). A session bound to that WI may not do work outside the WI's scope; it files a new WI instead.
2. **Terminal run events are tracker transitions, not status lines.** When the run reaches its exit condition the session must (a) post the verdict to the WI thread, (b) move the WI to `testing` (gate passed) or `parked` with `parked_reason` (gate failed), and (c) stop. A stopped pod with a failed gate is a terminal event.
3. **Wakeups go through the Agent Board.** Monitors publish `coordination/status@v1` / `coordination/control@v1` messages to a board thread owned by the run. Waking a chat thread directly from a home-grown bridge is not allowed; the board is where cadence and category are visible to everyone, and where an operator can mute a source without killing the session.
4. **Every scope change is a new WI with a budget.** A plan document that opens a new direction must name a bound (hours, windows, commits, dollars) and be linked from a WI in `draft`; the operator promotes it to `ready` in the tracker. "Run it" in a chat is not promotion.
5. **Unattended sessions carry hard trip-wires.** The harness, not the model, enforces: at most 20 commits per hour per WI, a 15-minute floor on self-scheduled wakeups, and a stop when the same event category repeats ten times without a WI phase change. Tripping a wire posts to the board and halts the session.
6. **A named drift is a hard stop.** If the operator says the session is off track, the session pauses, writes the current state to the WI thread, and waits. It does not build tooling in response.
7. **Compaction carries the WI.** Any compaction summary must begin with the bound WI code, title and exit condition so a tangent cannot become the remembered goal.
8. **Plans and reports go to Seshat when written.** Before building a monitor, bridge or harness, search Seshat for an existing one.

## Consequences

- Babysitting becomes boring and legible: one WI thread per run, one board thread per run, verdicts in the tracker, evidence in `training/results/`.
- The `t3_wake.py` bridge and `research_monitor.py` under `runs/babyllm-recovery-20260909/` are retired; the 275 events queued in `wake-state.json` must not be replayed. `babysit.py` (pod cost guard) is kept and re-pointed at the board in a follow-up WI.
- The trip-wires in rule 5 need harness support (runner-side, Anima `agent_sessions`); until they exist the operator must not grant open-ended "continue automatically" without a stated budget.
- Rule 2 retroactively applies to WI-365: the P5.5 run ended 2026-09-11 12:51Z with a failed gate; the WI is parked with the three blockers, and remediation v4 is a new WI.
