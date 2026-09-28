# Phase 51 P6.5 — world-model experiment windows E1–E4 + window 2 — ALL GATES FAILED (recorded 2026-09-28)

Anima WI P6.5 `01a09ac7-f938-7e43-9044-94cdbebbc14e`; plan `docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md`;
ADR 105/106/107. Nine local windows on the M2 (f32 Metal, recompute-backward attention), all supervised by
`scripts/p6/local_window.sh` via `scripts/p6/window_queue.sh runs/p6-4/queue-20260914.txt`, launched 2026-09-14,
completed 2026-09-21T12:57Z per `runs/p6-4/queue-20260914.txt.log`. Window 1 (E1 treatment) was already recorded in
`training/results/p6-4-window-1-RESULT.md` (gate FAILED 1255/1272); this doc records the remaining eight windows,
which were run but never written up before the evidence moved off the Mac.

**Verdict: no window in this set cleared the release gate.** Every one of the nine completed windows (window 1,
window 2, E1-control, E2-projection, E2-raw-events, E3-treatment, E3-control, E4-treatment, E4-control) has at least
one blocker family under the unchanged rule (pass < 85% or catastrophic > 1% per family, computed the same way as
`local_window.sh`'s summary). The best result in the whole P6.4/P6.5 run is **window 2** (sentinel-shaped siblings,
not an E1–E4 experiment arm): 1265/1272, 1 catastrophic, only 1 blocker family (`reactor_manage`). One window,
`window-1-attempt-1-failed`, aborted during training before any gate ran and has no gate data — noted, not invented.

## Provenance

The evidence lives only on the read-only NFS backup — it was never copied back to `origin/main` before the pods/queue
were torn down. Root path (abbreviated `NFS …/runs/p6-4/` in the tables below):

```
/Volumes/akh-medu-runs/mac-toasty/2026-09-24/akh-medu/t3code-bf7ae675/runs/p6-4/
```

Every number below is read directly from `<window>/status.json` (phase history + final `detail` summary, produced by
`local_window.sh`'s own tool-gate computation) and `<window>/gate/knowledge-*.evidence.json` (summed pass/total/
catastrophic per case across the 8 knowledge families, computed the same way `eval_knowledge_exam` reports a case:
`report.cases[].passed` / `.catastrophic`). `macOS ._*` AppleDouble files were ignored per instruction. Nothing here
is inferred from a missing file.

## Config (same across all nine windows unless noted)

| item | value | source |
|---|---|---|
| parent | candidate 401,577 (`runs/p6-4/parent-401577`), retained Adam | P6 plan §Decisions, `p6-4-window-1-RESULT.md` |
| start mode | `train_knowledge --remediate-from-bundle`, LR 4e-6, 1 epoch, batch 1, seed 62000, f32 Metal + recompute-backward attention | `scripts/p6/local_window.sh` defaults; `runs/p6-4/queue-20260914.txt` (no LR/epoch/batch overrides per line) |
| eval/ckpt cadence | `--eval-every 2000 --ckpt-every 5000` on every line; E3-treatment additionally `--record-weights use` | `runs/p6-4/queue-20260914.txt` |
| mixture | base 43,842 (knowledge 40,822 + tools 2,544 + dojo 476) + ~2,000–2,300 arm-specific siblings | P6 plan §Experiments |
| gate | unchanged quarantined sentinels: tools 1,272 (53 × 24), knowledge 8 × 100; constrained decoding | `scripts/p6/local_window.sh` (`SENTINELS` default), every `status.json` `detail` string |
| release-gate rule | per family: pass ≥ 0.85 **and** catastrophic ≤ 1% — any family failing either is a blocker | `scripts/p6/local_window.sh` SUMMARY computation (`rate<0.85 or catrate>0.01`) |

## Per-window gate table

| window | arm | final checkpoint | wall clock (build→done) | tools pass/1272 | catastrophic | blocker families | knowledge pass/800 |
|---|---|---|---|---|---|---|---|
| window-1-attempt-1-failed | E1 treatment, attempt 1 | none (aborted before first checkpoint save) | 2026-09-13T19:09Z → 20:28Z (1.3 h) | **no gate ran** — failed at line 76 (lineage bug, fixed `3ba92a12`) | — | — | — |
| window-1 | E1 treatment | `step_00447719` | 2026-09-14T09:21Z → 2026-09-15T04:19Z (19.0 h) | 1255/1272 | 2 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 19/24, `reactor_manage` 23/24 cat1, `study_note_revise` 18/24 | 800/800 |
| window-e1-control | E1 control | `step_00447719` | 2026-09-15T04:20Z → 23:08Z (18.8 h) | 1255/1272 | 3 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 19/24 cat1, `reactor_manage` 23/24 cat1, `study_note_revise` 20/24 | 799/800 (`revise_from_new_edition` 99/100) |
| window-2 | sentinel-shaped siblings (follow-up to window-1) | `step_00447692` | 2026-09-15T23:08Z → 2026-09-16T17:50Z (18.7 h) | **1265/1272 (best of set)** | **1 (best of set)** | `reactor_manage` 23/24 cat1 (only family) | 800/800 |
| window-e2-projection | E2 treatment (`holds_at` projection prompt) | `step_00447719` | 2026-09-16T17:50Z → 2026-09-17T12:39Z (18.8 h) | 1254/1272 | 3 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 18/24 cat1, `reactor_manage` 23/24 cat1, `study_note_revise` 18/24 | 800/800 |
| window-e2-raw-events | E2 control (raw event list prompt) | `step_00447719` | 2026-09-17T12:39Z → 2026-09-18T08:07Z (19.5 h) | 1254/1272 | 3 | identical list to window-e2-projection | 800/800 |
| window-e4-treatment | E4 treatment (owned-edge retraction revision) | `step_00447419` | 2026-09-18T08:07Z → 2026-09-19T03:23Z (19.3 h) | 1255/1272 | 3 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 18/24 cat1, `reactor_manage` 23/24 cat1, `study_note_revise` 20/24 | 800/800 |
| window-e4-control | E4 control (append-only) | `step_00447419` | 2026-09-19T03:23Z → 22:28Z (19.1 h) | 1254/1272 | 3 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 19/24 cat1, `reactor_manage` 23/24 cat1, `study_note_append` 20/24, `study_note_revise` 20/24 | 800/800 |
| window-e3-treatment | E3 treatment (`--record-weights use`, causal per-step reward) | `step_00447519` | 2026-09-19T22:28Z → 2026-09-20T17:36Z (19.1 h) | 1255/1272 | 4 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 19/24 cat2, `study_note_revise` 18/24 cat1 (`reactor_manage` clears) | 800/800 |
| window-e3-control | E3 control (sequence-level loss) | `step_00447519` | 2026-09-20T17:36Z → 2026-09-21T12:57Z (19.3 h) | 1254/1272 | 3 | `mcp__seshat__seshat_wish` 23/24 cat1, `memory_write` 19/24, `reactor_manage` 23/24 cat1, `study_note_revise` 18/24 cat1 | 800/800 |

All knowledge families not shown above (`acquire_textbook`, `grounded_abstention`, `length_generalization`,
`note_navigation`, `provider_fallback`, `recall_after_restart`, `textbook_inventory`) score 100/100 with 0
catastrophic in every window; the knowledge gate is saturated across this whole run and does not discriminate any
arm.

## Per-experiment treatment vs control

**E1 — symbolic-completion targets (window 1) vs raw-valid targets (window-e1-control).**
Tool pass counts are identical (1255/1272) and the blocker-family list is identical (same 4 families). The control
has 1 more catastrophic case overall and drops 1 knowledge case (`revise_from_new_edition` 99/100 vs 100/100 in
treatment). **No measurable difference in favor of the treatment** — the E1 hypothesis ("the completed target is
always schema-valid and value-exact, so copy-faithful families gain") is not supported by this control; both arms
fail on the same shapes documented in `p6-4-window-1-RESULT.md` (multi-fact `memory_write`, nested-schedule
`reactor_manage`, `study_note_revise` precursor cases, one French `seshat_wish` misroute).

**E2 — event-calculus `holds_at` projection prompt (window-e2-projection) vs raw event list (window-e2-raw-events).**
The two arms produced **byte-identical tool-gate summaries**: 1254/1272, 3 catastrophic, the same 4 blocker
families with the same per-family counts. Knowledge gates also match at 800/800. This sentinel set shows **zero
measured separation** between the two prompt designs — either the treatment has no effect at this scale/LR, or the
current sentinel mix does not exercise the stale-intent cases the hypothesis targets (the plan notes a held-out
revoked-case probe the generator emits, which was not part of this gate run).

**E3 — causal per-step reward weighting (window-e3-treatment, `--record-weights use`) vs sequence-level loss
(window-e3-control).**
Treatment moves 1 case from fail to pass (1255 vs 1254) and, notably, `reactor_manage` clears the blocker list
entirely in treatment (23/24 cat1 in control, not listed as a blocker in treatment — meaning it crossed back above
threshold). But treatment's `memory_write` catastrophic count rises to 2 (from 0 in control) and its total
catastrophic count is higher (4 vs 3). **This is a trade, not a net improvement**: the per-step causal reward
appears to help `reactor_manage` while making `memory_write` catastrophic failures worse in the same window.

**E4 — owned-edge retraction revision curriculum (window-e4-treatment) vs append-only control
(window-e4-control).**
Treatment has 1 more pass (1255 vs 1254) and 1 fewer blocker family — `study_note_append` clears in treatment
(22/24-equivalent threshold met) but remains a blocker in control (20/24). Both arms still fail on
`mcp__seshat__seshat_wish`, `memory_write` and `reactor_manage`. **This is the one pairwise win for a treatment in
the whole set**, though it does not clear the gate.

**Window 2 (not an E1–E4 arm) vs window 1.** Window 2's sentinel-shaped siblings (`--copy-fidelity` curriculum at
sentinel difficulty for `memory_write`/`reactor_manage`/`study_note_revise`/`seshat_wish`, plus 700 multi-fact
`memory_write` engine episodes) took the same base recipe from 4 blocker families down to 1, and catastrophic count
from 2 to 1. This is by a wide margin the largest single change measured anywhere in P6.4/P6.5, larger than any of
the E1–E4 treatment/control deltas.

## Retained artifacts (NFS, read-only)

- `NFS …/runs/p6-4/window-1/{status.json,gate/}` — E1 treatment, already reported in `p6-4-window-1-RESULT.md`.
- `NFS …/runs/p6-4/window-1-attempt-1-failed/status.json` — failed attempt, no gate.
- `NFS …/runs/p6-4/window-2/{status.json,gate/}` — best result of the run.
- `NFS …/runs/p6-4/window-e1-control/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e2-projection/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e2-raw-events/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e3-treatment/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e3-control/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e4-treatment/{status.json,gate/}`
- `NFS …/runs/p6-4/window-e4-control/{status.json,gate/}`
- `NFS …/runs/p6-4/{base/,parent-401577/,queue-20260914.txt,queue-20260914.txt.log,queue-20260914.nohup}` — base
  mixture, parent checkpoint reference, and the queue driver's own log (source for the per-window `detail` strings
  quoted above).
- Full checkpoints, predictions and per-case evidence for every window (`artifacts/checkpoints/step_*`,
  `gate/all-tools.{evidence,predictions,decode-reports}.json`, `gate/knowledge-*.{evidence,predictions}.json`) are
  present under each window directory on the same NFS path; not copied into the repo (large, and the repo only
  tracks RESULT docs per `training/results/` convention).

## Open hypotheses

- **E1** is not refuted, only untested by this control — the failing shapes (multi-fact copies, nested schedules)
  were absent from both arms' siblings alike. Window 2's siblings (which do target those shapes) are the actual
  test of the underlying idea and they work; a cleaner E1 rerun would need siblings shaped like window 2's but with
  raw-valid (not completed) targets, which does not yet exist.
- **E2** needs a gate that actually exercises revoked/stale-intent cases (the plan's "held-out revoked-case probe")
  before any conclusion about `holds_at` projection can be drawn; the current sentinel set is insensitive to it.
- **E3**'s reward weighting trade (reactor_manage up, memory_write catastrophic down) suggests the per-step reward
  is redistributing gradient signal across families rather than improving the model overall at LR 4e-6/1 epoch;
  worth retesting at a lower weight-scale or with `memory_write` specifically excluded from the weighted set.
- **E4** is the one measured pairwise win; a second window combining E4's revision curriculum with window 2's
  sentinel-shaped siblings is the natural next step, since neither change alone clears the gate but they hit
  different blocker families (E4 → `study_note_append`; window 2 → three of four families).
- No window here should be promoted to `testing`; P6.6 (pod decision) remains unopened because no window measured a
  local-lane throughput miss — the misses are model-quality misses (blocker families), not budget misses.
