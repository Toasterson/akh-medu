# FrontierCode-Diamond-style feasibility slice — Tecton

> Date: 2026-06-23
> Harness: benchmarks/frontiercode/ (SWE-bench Verified hard slice, 6-dim grading)
> Subject: akh `tecton` (GLM via ollama.com cloud) over ACP tcp:127.0.0.1:8201

## Headline

- **Harness validated end-to-end** via the gold-oracle path: `--dry-run-gold` on
  `sympy__sympy-17630` scores **1/1**, all four blocker dimensions evaluable and
  passing (including the reverse-classical gate).
- **Tecton diamond-style score on the live slice: 0/1** — but for an
  **infrastructure** reason, not a model-quality verdict: the ollama.com cloud
  backend was unreachable during the run.

## What happened, per attempt (task: sympy__sympy-17630)

| attempt | prompt | timeout | result | failure_kind | why |
|---------|--------|---------|--------|--------------|-----|
| gold oracle | n/a (gold patch) | n/a | **SCORE 1/1** | — | validates grader+env |
| Tecton #1 | v1 (allowed tools) | 480s | no patch | timeout | Tecton entered agentic tool-use (glob/grep/http_fetch/shell_exec) on the 28 KB context prompt and never emitted a final message before the 480s cap |
| Tecton #2 | v2 ("do NOT call tools, answer with one diff") | 900s | no patch | backend_error | tighter prompt stopped the tool-spiral (tools=[]), but the GLM call failed: `Ollama server unreachable at https://ollama.com/api/chat … timed out reading response` after 240s |
| Tecton #3 | v2 | 700s | no patch | backend_error | reproduced identically — 240s network timeout to ollama.com |

## Findings

1. **The grader, test-env build, and reverse-classical gate are mechanically
   correct.** The gold sympy patch builds a venv + editable install in ~8s, passes
   19/19 sampled PASS_TO_PASS, the injected test passes post-patch, and the same
   injected test *fails* on the unpatched base (rc=1, ValueError) — so the
   reverse-classical blocker fires correctly. Diamond-style gating (all blockers →
   1 else 0) works.

2. **Prompt shape matters for Tecton.** With tools allowed, Tecton treats a
   repo-edit task as an *agentic* task and burns the turn on tool calls without
   producing the requested diff. The directive "do NOT call tools, reply with one
   ```diff block" (PROMPT_TEMPLATE v2) eliminated the tool-spiral.

3. **Live runs are currently blocked by the cloud LLM backend.** Tecton's GLM is
   served from ollama.com; that endpoint timed out (~240s) on every live attempt
   ("intermittent LAN access" per the task brief). This is captured as
   `failure_kind: backend_error` in the result JSON — the harness does not mistake
   an infra outage for a model failure.

## Retry

Re-run the live slice once ollama.com is reachable from the host:

```bash
cd benchmarks/frontiercode
python3 harness/runner.py --ids sympy__sympy-17630 --timeout 700
# or a broader slice:
python3 harness/runner.py --ids sympy__sympy-17630 sympy__sympy-13852 pytest-dev__pytest-6197
```

Confirm backend health first, e.g. an unrelated short ACP prompt:
`python3 ../../scripts/acp_adapter.py -w tecton -p "Reply with one word: ok"`.

## Artefacts

- `results/GOLD_VALIDATION.json` — the 1/1 gold-oracle grade (harness proof).
- `results/sympy__sympy-17630.json` — last live attempt (backend_error).
- `results/summary.json` — roll-up.

## Addendum (additional live attempts)

- An unrelated tiny ACP probe (`-p "Reply with one word: ok"`) **succeeded**
  (end_turn, 132.8s, tools=none) in one window — confirming the ACP path and akh
  are healthy; the failures are purely the upstream cloud LLM.
- Re-running the task with a **5000-char** context (vs 28 KB) still hit the same
  240s `backend_error`. So the timeout is **not** prompt-size driven — it is a hard
  intermittent-connectivity ceiling on the ollama.com path for these turns. Added
  `--max-context <chars>` to the runner anyway, so a future operator can shrink the
  prompt if a size limit does surface on a healthier link.

## Net feasibility verdict

The harness is **done and proven** (gold 1/1, all blockers + reverse-classical
working). A genuine Tecton-vs-Diamond score awaits a stable ollama.com window;
re-run `python3 harness/runner.py` then (no code changes needed). Observed Tecton
failure modes so far: (a) tool-spiral on permissive prompts — fixed by the v2
prompt; (b) cloud-backend timeouts — infra, surfaced as `failure_kind:backend_error`.
