# illumos-e2e — first real run interpretation (2026-07-07)

First live run of the Phase-50 W3 runner: three rented frontier models driven
through the **48c remote executor** against a live OpenIndiana Hipster guest
(`exec1`), one real illumos sysadmin task at a time, deterministically graded
by each task's `verify.sh`. See `docs/ai/2026-07-07-phase50-w3-runner-design.md`.

## Headline

| Model | Pass | Rate | Mean score | Requests | Wall |
|---|---|---|---|---|---|
| glm-5.2 | 8/15 | 53.3% | 0.533 | 244 | ~49 min |
| deepseek-v4-pro | 8/15 | 53.3% | 0.533 | 193 | ~28 min |
| qwen3.5:397b | 8/15 | 53.3% | 0.533 | 280 | ~42 min |

All three converge on the **same 8 passes and 7 fails**. That is not a
coincidence — it is a property of the substrate, not the models (read on).

## The 53.3% ceiling is a substrate cap, not a model wall

The guest is the minimal `openindiana-hipster-cloud-fixed.qcow2` image — it
has **no C compiler and no Rust toolchain**. Six of the fifteen tasks require
compiling against illumos system libraries:

- `c-01/02/03` (libkstat / libscf against `cc`)
- `rust-01/02/03` (illumos crates against `cargo`/`rustc`)

Every model fails all six for the same reason: `cc`/`cargo` are absent, so no
answer can be produced regardless of the model's competence. The seventh fail,
`rbac-03-restrict-privileges`, is a **genuine model miss** (all three got the
privilege-restriction end-state wrong).

**On the 9 toolchain-free tasks, all three models score 8/9** (only rbac-03
lost). The interesting comparative signal is therefore NOT the pass rate — it
is the manpage grounding (below).

A **v2 sweep on a toolchain-provisioned guest** (the deferred `builder` VM, or
`pkg install gcc rust` on `exec1`) is needed to grade the compile families
fairly. Until then, read illumos-e2e as a **9-task core** with 6 tasks
substrate-blocked and uniformly zero across models.

## The real inter-model signal: manpage grounding

Each task declares `expected_manpages` — the illumos man pages a competent
engineer would consult. The runner grades, from the executor audit trail,
whether the model actually ran `man`/`apropos` on those pages (non-gating; the
deterministic `verify.sh` decides pass/fail). This is the "does the akh ground
in local documentation, or does it rely on parametric memory that may be
Linux/Solaris-11-wrong?" signal the corpus was built to expose (many tasks are
**version-traps**).

Observed behavior (per-task notes in each `run.jsonl`):

- **deepseek-v4-pro** grounds the *least* — repeatedly `0/N` on SMF, discovery
  and RBAC tasks (solved them from parametric knowledge alone).
- **glm-5.2** grounds moderately (`2/4`, `3/3` on several tasks).
- **qwen3.5:397b** grounds the *most* on RBAC (`3/4` on rbac-01).

All three pass the same tasks, so on this corpus parametric recall was
*sufficient* for the toolchain-free work — but a model that grounds is the one
you'd trust on a version-trap where its memory is wrong. That distinction is
exactly what a deterministic pass/fail alone would miss, and what this metric
surfaces.

## Provenance

- Task-set hash: `0c05049432a8…` (all three share it — same corpus).
- Host tier: `Vm` (`code.aopc.cloud/CloudNebulaProject`), recipe hash
  `c93aaef5…` (the `bake-hipster-cloud-image.sh` guest).
- Grading: `autonomous` (deterministic `verify.sh`, no LLM judge).
- Models: `ollama.com` cloud, bare published tags.

Regenerate the board/site: `bench-tool leaderboard` / `bench-tool site`.
