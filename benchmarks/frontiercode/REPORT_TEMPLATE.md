# FrontierCode-Diamond-style run — Tecton

> Date: YYYY-MM-DD
> Subject: akh `tecton` (GLM via ollama.com cloud) over ACP `tcp:127.0.0.1:8201`
> Harness: `benchmarks/frontiercode/` (SWE-bench Verified hard slice, 6-dim grading)
> Context strategy: `AKH_CONTEXT_STRATEGY=<...>`  ·  `AKH_AGENT_MAX_ROUNDS=<...>`

## Headline

**Diamond-style score: N / M  (X.X%)**  <- blocker-gated (all blockers pass else 0).

(Expected ~0 — frontier models score 6–13% on real Diamond. This run measures
feasibility + failure modes, not a leaderboard position.)

## Per-task results

| instance_id | repo | diff | score | failure stage / dominant blocker | notes |
|-------------|------|------|-------|----------------------------------|-------|
| sympy__sympy-17630 | sympy | ... | 0/1 | ... | ... |

Failure-stage vocabulary: `no-patch` (Tecton emitted no diff) · `apply-failed`
(diff didn't apply) · `behavioral` (F2P failed) · `regression` (P2P broke) ·
`reverse-classical` (test patch didn't bite) · `scope` (too large) · `env-infra`
(could not build test env -> not-evaluable).

## Dimension roll-up

| dimension | blocker | passed | failed | not-evaluable |
|-----------|---------|--------|--------|---------------|
| behavioral_correctness | yes | | | |
| regression_safety | yes | | | |
| test_correctness_reverse_classical | yes | | | |
| scope | yes | | | |
| mechanical_cleanliness | no | | | |
| code_quality | no | | | (stubbed) |

## What happened (narrative)

- Patch production: did Tecton emit a parseable unified diff? how often?
  (`stages.model.patch_extracted` in each result JSON.)
- Patch application: apply success rate; common apply failures.
- Behavioral: of patches that applied, how many made the injected tests pass?
- Infra: how many tasks blocked by env build failures (and which repos)?

## Faithful vs approximated (recap)

- Fully implemented: dims 1, 2, 4, 5 (+ dim 3 when a linter is present).
- Stubbed: dim 6 (code-quality LLM rubric) — `grader.judge_stub`.

## Reproduce

```bash
cd benchmarks/frontiercode
python3 harness/extract_tasks.py
python3 harness/runner.py
```

## Raw artefacts

- `results/<instance_id>.json` — per-task stages, patch stats, full grade.
- `results/summary.json` — roll-up.
- Candidate patches: re-run with `--keep` to retain `candidate.patch`.
