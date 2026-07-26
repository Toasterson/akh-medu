# Phase 50 — Feature-test VM harness + task-performance benchmark program

- Status: Planned (grilling-locked 2026-07-05)
- ADR: [086](../decisions/086-feature-test-vm-harness-benchmarks.md)
- Depends on: Phase 48a (Backend seam, built); pulls in 48b/48c (built here)

## Goal

A self-hosting feature-test bed that exercises every akh-medu feature in the
cheapest tier that genuinely runs it, using `vmctl`-spun **OpenIndiana** VMs —
and, on the same substrate, a longitudinal, publishable benchmark program
measuring task performance of the frontier models we rent (latest deepseek,
GLM 5.2, latest qwen) and eventually our self-trained tiny LLMs.

See ADR 086 for the locked decisions. This plan is the work breakdown.

## Prerequisites / facts (verified 2026-07-05)

- `vmctl` present (`~/.cargo/bin/vmctl`): create/up/down/reload/provision/
  ssh/console/suspend/snapshot; declarative `VMFile.kdl`. KVM available
  (`/dev/kvm`), qemu + virsh present.
- OpenIndiana + OmniOS images built by refraction-forger, published at
  `code.aopc.cloud/CloudNebulaProject/-/packages`; image sources under
  `solstice-ci/images`. **Pull prebuilt — do not rebuild.**
- Release feature sets (release.yml): `akh --no-default-features --features
  client`; `akhomed --features server`; `seshd --no-default-features
  --features synthesis`. illumos is ORT-free (no onnxruntime build).
- Existing harnesses to reuse: `benchmarks/frontiercode/` (Diamond
  replication), `benchmarks/` LoCoMo/LongMemEval (`run_memory_bench.py`),
  `scripts/acp_adapter.py` (live ACP driver).

## Work items

### W1 — VM substrate  *(feature-test deliverable, part 1)*
- `tests/vm/VMFile.kdl`: `exec1` (sshd + `akh-exec` unprivileged user),
  `pg` (postgres + pgvector), `builder` (rust toolchain). 2 vCPU/4GB baseline.
- Image pull + `vmctl image` wiring from the CloudNebulaProject registry;
  document OpenIndiana-under-KVM boot quirks discovered (UEFI/virtio).
- Provisioners: users/keys, postgres init, rust toolchain.
- `vmctl up` smoke test in CI-able form.
- **Portability shakeout**: run full `cargo test --features server` natively
  in `builder`; triage/file the first illumos failures (paths, libc, redb
  mmap, socket semantics). Record a baseline pass list.

### W2 — Executors against the substrate  *(feature-test deliverable, part 2)*
- **48c russh remote** first (E2E tasks need it): host akhomed → `exec1`
  over SSH via the `Backend` trait; plain provisioned keys (48d deferred,
  per ADR 086). Wire into ACP.
- **48b LocalDrop**: akhomed in `exec1` dropping to `akh-exec` via illumos
  RBAC/privileges; confinement posture reported (ADR 082 §confinement).
- Executor audit trail (ADR 082 correlation-index) emits the command log the
  W3 grader consumes.

### W3 — illumos E2E corpus + runner  *(feature-test deliverable, part 3)*
- Fleet authors ~15 tasks (human-reviewed), 5 families (SMF, manpage
  discovery, C/libkstat+libscf, Rust/illumos, RBAC), **version-traps
  included**. Shape `(setup.sh, prompt, verify.sh)`; akh never sees verify.
- Runner: clean snapshot per task (vmctl restore/reload), autonomous mode,
  deterministic grade = `verify.sh` exit code.
- Manpage grading: deterministic (audit-trail `man`/`apropos` for the right
  page) + rubric.

### W4 — Advisory mode
- Engineer-simulator (cheap scripted qwen tag): states goal, executes akh's
  suggested commands verbatim in the VM, logs transcript; `verify.sh` still
  grades end state where applicable.
- Rotation LLM grader (no self-grading; version-pinned) on advice quality
  (command correctness, real-vs-hallucinated flags, destructive-safety,
  turns-to-solution). Report judge-vs-deterministic calibration.

### W5 — Public benchmark legs (per model: deepseek / GLM 5.2 / qwen)
- frontiercode run harness per-model.
- LoCoMo + LongMemEval across all three (was Tecton-only).
- Terminal-Bench core-subset adapter (reuse the `(setup,prompt,verify)`
  runner from W3), Linux host tier.

### W6 — Fleet-orchestration benchmark
- Orchestrator akh + worker akhs (agent_spawn/message/delegation) on
  decompose-and-conquer tasks over the VM fleet.
- Metrics: deterministic success; **speedup vs single-akh control**;
  coordination overhead (messages, redundant commands from audit logs);
  delegation-quality rubric (rotation judge).

### W7 — Results, leaderboard, site
- Self-describing JSONL + summary schema (harness SHA, task-set hash, model
  tag/backing, judge+seed, VM image id, host tier) under
  `benchmarks/results/<date>-<harness>-<model>/`. Commit scores; gitignore
  transcripts/artifacts.
- `benchmarks/LEADERBOARD.md` generator (latest committed per model×harness);
  task-hash freeze on publish (edits → v2 series).
- In-repo Rust site generator → self-contained single-page HTML (inline CSS
  + inline-SVG charts, no CDN/JS).
- `just bench <harness> <model>` entry point; per-run budget caps; manual
  cadence → weekly after two stable runs.

## Out of scope (v1)
48d OIDC/cert auth (production, not lab); peer-to-peer swarm orchestration;
SWE-bench-full / GAIA / AgentBench; GPU passthrough to guests; Phase-49
console (advisory mode here is benchmark-only, not the co-piloted console).
