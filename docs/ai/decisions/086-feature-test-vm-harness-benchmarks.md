# ADR 086: OpenIndiana VM feature-test harness + task-performance benchmark program

- Status: Accepted (grilling-locked 2026-07-05)
- Phase: 50
- Plan: [docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md](../plans/2026-07-05-phase50-feature-test-benchmarks.md)

## Context

Phase 48 designed the privilege-separated executor (ADR 082); 48a (Backend
seam) is built, but 48b (LocalDrop) and 48c (russh remote) are unbuildable
without a second user / a real sshd / a separate host — infrastructure the dev
box alone doesn't offer. Separately we want task-performance benchmarks that
are comparable over time, across the frontier cloud models we rent (latest
deepseek, GLM 5.2, latest qwen) and, eventually, our own self-trained tiny
LLMs — so training improvements become a one-line diff on a leaderboard.

## Decision

### 1. VM harness (vmctl + OpenIndiana) is the executor test bed — not a mirror of `cargo test`

`tests/vm/VMFile.kdl` defines three guests, all **OpenIndiana**, images pulled
prebuilt from the CloudNebulaProject package registry (built by
[refraction-forger](https://code.aopc.cloud/CloudNebulaProject/refraction-forger);
image sources in
[solstice-ci/images](https://code.aopc.cloud/CloudNebulaProject/solstice-ci/src/branch/main/images)):

| VM | Role |
|---|---|
| `exec1` | executor target: sshd + unprivileged `akh-exec` user. 48b LocalDrop tested *inside* it (illumos RBAC/privileges = the ADR 082 "OS as sole security boundary", natively); 48c russh tested host→VM over real SSH |
| `pg` | plain postgres+pgvector; Phase-38 relocation/split-brain/WAL tests get a real network boundary + host↔VM relocation |
| `builder` | native rust toolchain; builds the illumos artifacts (release.yml feature sets: `akh --features client`, `akhomed --features server`, `seshd --features synthesis`) and runs the **full `cargo test --features server` natively** — the suite has never executed on illumos, a shipping release target |

Sizing: 2 vCPU/4GB each on the 8-core/31GB host (builder may take more while
compiling). Native build in-VM, not cross-compilation (matches release.yml;
also continuously validates the still-unvalidated illumos release recipe).

**Two-tier feature matrix**: host tier (Linux) keeps full default features +
GPU/candle + nlu-ml/ORT; VM tier covers the release feature sets plus
everything only VMs can test (executors, RBAC confinement, network-boundary
ACP, relocation). "All features" = the union; each feature is tested in the
cheapest tier that genuinely exercises it.

**Auth deviation (explicit)**: the lab harness uses plain provisioned SSH
keys. 48d OIDC→SSH-cert (ADR 082) remains the production auth story and is
NOT displaced by this; the harness must not silently become the de-facto
auth model.

### 2. E2E tests are real illumos system-engineering work, agentically performed

Not unit tests in a guest: an akh, through the Phase-48 executor, does the
work an illumos system engineer does. Task corpus v1 = ~15 fleet-authored,
human-reviewed tasks in the shape `(setup.sh, prompt, verify.sh)` across 5
families: SMF operations, manpage-driven command discovery, C against native
libs (libkstat/libscf), Rust against illumos concepts, RBAC/users. Each task
runs from a clean snapshot (vmctl restore/reload between tasks).

- **Autonomous mode**: deterministic grading only — `verify.sh` inspects VM
  state after the run; the akh never sees it. No LLM judge on outcomes.
- **Advisory mode**: a cheap scripted engineer-simulator LLM executes the
  akh's suggested commands verbatim (so `verify.sh` still grades end state)
  plus a rubric LLM grader for advice quality. **Rotation rule: no model
  grades its own transcript**; graders pinned by version; judge scores are
  calibrated against the deterministic subset and the divergence is reported.
- **Manpage grounding is graded**: deterministically via the executor audit
  trail (did the akh run `man`/`apropos` for the relevant pages?) and as a
  rubric item; **version-trap tasks** (where Linux/Solaris-11 trained
  knowledge is wrong on OpenIndiana and only the local manpage yields the
  working flag) make manpage use load-bearing on outcomes, not a checkbox.

### 3. Benchmark program: four legs per model + fleet orchestration

Models under test: latest deepseek, GLM 5.2, latest qwen (ollama-cloud tags,
per-akh `ChatBackend`); later our self-trained tiny LLMs via candle GGUF —
same harness, same task hashes.

1. **Coding**: existing `benchmarks/frontiercode/` (FrontierCode-Diamond
   replication over public SWE-bench tasks) run per-model.
2. **Memory**: existing LoCoMo + LongMemEval harness run per-model.
3. **Terminal agency**: Terminal-Bench core subset via one adapter (its
   task format ≈ our `(setup, prompt, verify)` shape), Linux host tier.
4. **illumos sysadmin E2E**: the corpus above (our novel challenge set).
5. **Fleet orchestration**: orchestrator-shaped — one orchestrator akh
   delegates to worker akhs (agent_spawn/agent_message/delegation) on
   decompose-and-conquer tasks over the VM fleet. Metrics: deterministic
   task success, **speedup vs a single-akh control run of the same task**,
   coordination overhead (messages, redundant commands via per-akh audit
   logs), delegation quality (rotation judge). Peer-to-peer swarms are out
   of scope for v1.

Skipped for v1: SWE-bench-full (per-task Docker infra, low marginal signal
over frontiercode), GAIA (open-web + messy grading), AgentBench (aged).

### 4. Results discipline (publishability + tiny-LLM comparability)

Every run emits self-describing JSONL + summary markdown under
`benchmarks/results/<date>-<harness>-<model>/`, stamped with: harness git
SHA, **task-set content hash**, model tag + backing (cloud tag / GGUF hash),
judge model + rotation seed, VM image identity (registry/recipe hash), host
tier. Scores/summaries are committed (regression tracking = git log); raw
transcripts and VM artifacts are gitignored. `benchmarks/LEADERBOARD.md` is
generated from the latest committed run per (model, harness). **Once
published, a task-set hash freezes** — task improvements fork a v2 series.

A small in-repo Rust **site generator** renders the committed results into a
self-contained single-page HTML leaderboard (inline CSS + inline-SVG trend
charts, no CDN/JS toolchain) for publishing.

### 5. Execution logistics

Runs are host-configurable: this box for cloud-backed runs + the VM tier;
hephaistos only when local GPU inference is the row under test. Cadence:
manual (`just bench <harness> <model>`) until two consecutive runs are
stable, then weekly. Hard per-run budget caps (max tokens / cloud requests
per task; abort counts as failure) so a looping akh can't drain the sub.

## Consequences

- 48b/48c get their missing test bed and are in scope to build (fleet).
- The illumos release path finally gets exercised continuously (build AND
  test), de-risking ADR 079's untested first tag.
- Benchmarks become longitudinal instruments: publishable, and the wall the
  Grug-Brain tiny LLMs will be measured against.
- Cost: VM tier adds wall-time (native illumos builds); accepted. Single-akh
  control runs double fleet-bench cost; accepted by user.

## Work items (dependency-sorted, fleet-executable)

W1 VM substrate (VMFile.kdl, image pull, provisioners, `vmctl up` smoke,
   native suite in builder) → W2 executors (48c russh host→exec1, then 48b
   RBAC LocalDrop in exec1) → W3 illumos E2E corpus + runner + audit-log
   manpage grading (**W1–W3 = the feature-test deliverable**) → W4 advisory
   mode (simulator + rotation judge + calibration) → W5 public legs
   (frontiercode / memory / Terminal-Bench across 3 models) → W6 fleet
   orchestration + control → W7 results schema + leaderboard + site-gen.
