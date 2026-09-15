# Phase 51 P6.4 — remediation v4, window 1 (E1 treatment) — GATE FAILED (2026-09-15)

Anima WI P6.4 `01a09a61-f4d2-7db3-bc2c-0f3ff513191e`; plan `docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md`;
ADR 105/106/107. Local run on the M2 (f32 Metal, recompute-backward attention), supervised by `scripts/p6/local_window.sh`.

## Run

| item | value |
|---|---|
| parent | candidate 401,577 (`runs/p6-4/parent-401577`), retained Adam |
| start mode | `train_knowledge --remediate-from-bundle`, LR 4e-6, warmup 250, 1 epoch, batch 1, seed 62000 |
| mixture | base 43,842 (knowledge 40,822 + tools 2,544 + dojo 476) + 2,300 engine siblings (memory_write 1,000; reactor_manage 750 EN/DE/FR; structured_task_create 300; structured_task_update 250) = 46,142; share 49,846 ppm |
| siblings | `training/episodes/p6-3b-20260913-indist` (frozen contract, 4 offered tools, grug-data request rendering; decontaminated, 0 dropped); mixture manifest `runs/p6-4/window-1/siblings/manifest.json` |
| steps | 401,578 → 447,719 (46,142 updates), 0.67–0.73 updates/s, ≈18 h (attempt 1 died at the first checkpoint on a lineage bug, fixed in `3ba92a12`) |
| held-out evals | parse/action/wire-exact 1.000 (n=32) at every 2,000 steps |
| final candidate | `runs/p6-4/window-1/artifacts` (checkpoint `step_00447719`, pruned to the referenced checkpoint), lineage `remediated_from` recorded |
| gate | unchanged quarantined sentinels: tools 1,272 (53 × 24), knowledge 8 × 100; constrained decoding with the fixed decoder |

## Gate result

**Tools 1255/1272, 15 precursor, 2 catastrophic — FAILED** (baseline 401,577 with the fixed decoder: 1259/1272, 12 precursor, 1 catastrophic). Knowledge: all 8 families pass 1.000.

| family | 401,577 (fixed decoder) | window 1 | verdict |
|---|---|---|---|
| memory_write | 19/24 | 19/24 (5 precursor) | **unchanged — blocker** |
| reactor_manage | 23/24, 1 cat | 23/24, 1 cat | **unchanged — blocker** |
| structured_task_create | 24/24 | 24/24 | fixed by the decoder (P6.2) |
| study_note_revise | 21/24 | **18/24** (6 precursor) | regressed — blocker (<85 %) |
| mcp__seshat__seshat_wish | 24/24 | **23/24, 1 cat** | regressed — blocker (>1 % cat) |
| study_note_append | 23/24 | 22/24 | regressed, still ≥85 % |
| compile_feedback | 23/24 | 24/24 | improved |

## Case analysis (from `runs/p6-4/window-1/gate/all-tools.{evidence,predictions}.json`)

- **memory_write (5 fails)**: the sentinels are synthetic copy-fidelity cases — `facts` arrays of 1–4 objects with a
  randomly present `confidence`, optional top-level `origin`/`source`/`tags`. Failures: one of three facts dropped;
  `source` omitted; two facts merged into one; an unrequested `timestamp` plus invented facts when `facts` was absent.
  **Our siblings never exercised this**: every engine memory_write episode had exactly one fact with fixed fields.
- **reactor_manage (1 catastrophic, French)**: output starts `L'id-revision-381…` — the model emitted a French token
  before `<tool_call>`, so the grammar never engaged; the case is a copy-fidelity call with a nested `schedule`
  (`lease_seconds`, `misfire_grace_seconds`, `rule: {}`) and placeholder values. Our reactor siblings were realistic
  `add`/`list` operations, not this structure.
- **study_note_revise (6 precursor)** and **seshat_wish (1 cat, French: chose `seshat_search`)**: families we did not
  target; a 3-case / 1-case movement at LR 4e-6 over one epoch is within the noise the pod loop also saw, but it is a
  regression under the gate's rules and must be watched in the E1-control window (same base, different targets).

## Verdict and exit condition

Gate failed → WI P6.4 stays `draft` (parked per the plan's exit condition), candidate not promoted. The E1 hypothesis
("completed targets are schema-valid and value-exact") is not refuted — the siblings simply did not cover the failing
shapes. **Window 2** (queued after the running E1-control window): siblings that mirror the sentinel structure —
copy-fidelity records for memory_write / reactor_manage / study_note_revise / seshat_wish generated with grug-data's own
curriculum generator at sentinel difficulty (fresh seed, decontaminated), plus engine memory_write episodes upgraded to
1–4 facts with optional fields. Same LR/epoch/share so window 1 and window 2 differ only in sibling design.

Disk: window pruned to 383 MB; `runs/akh-r1-*` (39 GB of venvs, build objects and scratch DBs from the September 12
drift session) archived to `runs/akh-r1-evidence-20260915.tar.zst` (8,836 text files, 184 MB) and deleted.
