# fleet-orchestration (W6) — first real run interpretation (2026-07-07)

First live run of the Phase-50 **W6 fleet-orchestration benchmark**: does splitting
a decomposable illumos task across an **orchestrator akh + N worker akhs** beat a
**single akh** doing the whole thing? Two arms, same task, same model, over the live
`exec1` guest. See `docs/ai/2026-07-07-phase50-w6-orchestration-design.md`.

Task: `relay-group-3svc` — three independent, differently-broken SMF services
(`relay-a` maintenance / `relay-b` disabled / `relay-c` misconfigured), distinct
FMRIs+paths so parallel workers never collide. Both arms graded by the same
composite `verify.sh` (all three `online`).

## Result

| Model | Control wall | Orchestrated wall | **Speedup** | Reqs ctrl→orch | Decompose tax | Redundant cmds |
|---|---|---|---|---|---|---|
| deepseek-v4-pro | 55.3 s | 34.0 s | **1.63×** | 8 → 24 | 5 677 tok / 1 req | 0/36 |
| qwen3.5:397b | 111.8 s | 75.9 s | **1.47×** | 15 → 32 | 7 469 tok / 1 req | 0/28 |
| glm-5.2 | 100.4 s | 115.8 s | **0.87× (slower)** | 8 → 36 | 5 512 tok / 1 req | 2/55 |

All six arms **passed** (2/2 per model) — every model, in both single and fleet mode,
correctly fixed all three services. The interesting axis is cost/latency, not success.

## The headline: fleet orchestration is *model-dependent*, not a free win

- **deepseek-v4-pro parallelizes best (1.63×)** — and does it at the lowest request
  amplification (3×: 8→24) and zero redundant commands. Its workers were balanced and
  its per-turn work efficient, so overlapping their cloud-model latency actually paid.
- **qwen (1.47×)** — solid benefit, but the priciest decomposition (7 469 tok) and most
  requests (32).
- **glm-5.2 REGRESSED to 0.87×** — orchestration made it *slower*. Three compounding
  costs beat the parallelism: the ~5.5k-token decompose turn, a **4.5× request blow-up**
  (8→36), and the slowest-worker bottleneck (the orchestrated wall can't beat
  `decompose + max(worker)`; if one subtask dominates, parallelism buys little). glm also
  had the only non-zero redundant commands (2/55 — two workers re-ran a shared discovery).

**Takeaway:** star-topology fleet orchestration is worth it only when (a) subtasks are
balanced, (b) the model is request-efficient per turn, and (c) the coordination tax
(decompose turn + N× request amplification) is small relative to the work. It is NOT an
automatic speedup — for glm on this task it was a net loss. This is the measured answer
to "does orchestrating a fleet of akhs help?": *it depends on the model, and you can
measure which.*

## Caveats (read before citing the numbers)

- **Single trial, high cloud-latency variance.** glm's control arm measured 58.1 s in a
  scratch validation run and 100.4 s in this one — cloud inference latency swings widely
  run-to-run. These speedups are indicative, not precise; a rigorous version needs
  several trials per (model, arm) and a median. The *direction* (deepseek≈qwen benefit,
  glm regress) is the robust signal; the exact ratios are noisy.
- **One task, N=3 workers.** A single composite task with three subtasks. More tasks and
  larger fan-out (N) would sharpen the balance/amplification story.
- **Coordination tax = decompose turn only.** "Messages" is proxied by the fan-out count
  (one delegated subtask = one message); there is no inter-akh message audit event yet
  (would need an `AuditKind::Delegation`/`Message` variant — deliberately deferred, ADR
  082 §7 correlation-index is the seam). Redundant-command ratio IS real, read from the
  shared audit ledger grouped by `ExecEnvelope.correlation_id`.
- Substrate/grading identical to illumos-e2e: live OpenIndiana `exec1`, deterministic
  `verify.sh`, `ollama.com` cloud models, autonomous mode.

## Provenance

Harness `fleet-orchestration`; two task rows per run (`…-control`, `…-orchestrated`);
`HostTier::Vm`; regenerate the board/site with `bench-tool leaderboard` / `bench-tool site`.
