# Lilith: illumos benchmark and native-runtime inventory

Date: 2026-09-21. Scope: read-only audit of repository-owned code, task
fixtures, ADRs, and committed historical results. No VM was started, changed,
or contacted. Historical run reports establish prior activity; they do not
establish that this machine or its guests are ready today.

The repository has a useful development baseline: 15 illumos tasks, one
three-worker coordination task, real historical VM results, and a public
repository-patching harness. It does **not** yet supply a clean Lilith release
evaluation or the requested Qwen 3.8 27B comparison. The historical Qwen runs
identify **qwen3.5:397b**. Preserve that identity; do not relabel its scores.
Sources: [E2E result](../../../benchmarks/results/2026-07-07-illumos-e2e-qwen/run.jsonl),
[coding result](../../../benchmarks/results/2026-07-06-frontiercode-qwen/run.jsonl).

## Reusable artifacts and coverage

| Artifact | Actual coverage | Reuse for Lilith |
|---|---|---|
| [illumos-e2e/tasks](../../../benchmarks/illumos-e2e/tasks) | 15 authored tasks: three each in SMF, manpage discovery, C native libraries, Rust illumos FFI, RBAC | Public development/regression suite; native outcome grading |
| [illumos-orchestration/tasks](../../../benchmarks/illumos-orchestration/tasks) | One task, `relay-group-3svc`, with three independent SMF subtasks | Paired single-agent/fleet control; too small for a coordination release gate |
| [phase50_w3_runner.rs](../../../examples/phase50_w3_runner.rs) | Setup, remote executor, agent turn, deterministic verification, heuristic audit-based manpage lookup, result emission | Existing execution path; isolate guests and strengthen provenance before a baseline |
| [phase50_w6_orchestration.rs](../../../examples/phase50_w6_orchestration.rs) | Control arm plus decomposition/fan-out arm, composite verification, usage and redundant-command metrics | Reuse paired design; distinguish model decomposition from supplied metadata |
| [frontiercode harness](../../../benchmarks/frontiercode/harness/runner.py) | Public SWE-bench-derived patch generation with source context selected from gold-touched files | A constrained patching diagnostic, not an autonomous repository-discovery score |
| [harness-core](../../../benchmarks/harness-core) | Task hashing, results schema/store, static corpus tests | Preserve task hashes and per-task records; extend evidence rather than replacing this plumbing |
| [VMFile.kdl](../../../tests/vm/VMFile.kdl) and [builder shakeout](../../../tests/vm/scripts/builder-shakeout.sh) | OpenIndiana guest definitions and native build/test commands | Starting point for a reproducible, toolchain-baked execution image |
| [native API KG](../../../benchmarks/illumos-e2e/kg/illumos-native-api-kg.toml) | Authored API-relation fixture covering the existing tasks' concepts | Development knowledge fixture; explicitly tag exposure and exclude it as evidence of novel learning |

The 15 E2E task metadata files describe five version traps spanning **four**
families: SMF (1), discovery (2), Rust (1), RBAC (1). The README's sentence
saying three families disagrees with its own table. The corpus tests enforce
minimum coverage, but static validation cannot validate native ABI behavior.
Sources: [corpus README](../../../benchmarks/illumos-e2e/README.md),
[metadata tests](../../../benchmarks/harness-core/tests/illumos_e2e_task_set.rs).

| Family | Existing tasks | Missing quality evidence |
|---|---|---|
| SMF | Clear maintenance, configure/enable service, import manifest | Dependency ordering, rollback, recovery after reboot, durable service lifecycle |
| Discovery / ZFS | CPU facts, NIC state/speed, pool/dataset properties | Fault diagnosis across changing configurations, storage recovery, network repair |
| C native libraries | libkstat CPU count, libscf property, libscf state | Larger patches, ownership/error paths, ABI and resource-lifetime checks |
| Rust illumos | processor_info FFI, Doors round trip, extended attributes | Binding correctness across architectures, malformed inputs, concurrency and lifecycle failures |
| RBAC | Role, rights profile, restricted default privileges | Persistent execution identity, negative permission tests, attributed audit trail |
| Coordination | Three independent services | Dependencies, conflicting edits, worker failure/reassignment, integration review, honest human status reports |

Missing entries above are proposed extensions, inferred from the narrowly
scoped [E2E](../../../benchmarks/illumos-e2e/README.md) and
[orchestration](../../../benchmarks/illumos-orchestration/README.md) fixtures.

## What the historical evidence really establishes

- July 7 E2E v1: all three models passed 8/15. Six compile tasks lacked a
  compiler/toolchain; those failures cannot diagnose model competence. The
  report attributes the remaining RBAC failure to the models, but that still
  merits an independently validated oracle before tuning.
  [v1 interpretation](../../../benchmarks/results/2026-07-07-illumos-e2e-NOTES.md)
- July 7 v2: six compile tasks rerun after updating the guest and installing
  compilers; GLM passed 2/6, DeepSeek 1/6, Qwen 0/6. Reports identify shelling
  out instead of native APIs and incorrect property retrieval as recurring
  failures. These are native task executions, but combining them with v1 is
  a mixed-image composite, not a fresh 15-task matched run.
  [v2 interpretation](../../../benchmarks/results/2026-07-07-illumos-e2e-v2-NOTES.md)
- July 8 GLM development runs report 8/15 with the initial harness/knowledge
  setup and 11/15 with the atomic KG fixture. Different harness SHAs and an
  exposed task set mean this is a useful intervention observation, not proof
  of generalization, VSA learning, or a causal controlled comparison.
  [initial result](../../../benchmarks/results/2026-07-08-illumos-e2e-glm-5.2-harness/summary.md),
  [KG result](../../../benchmarks/results/2026-07-08-illumos-e2e-glm-5.2-harness-kg/summary.md)
- Fleet runs contain two arms per model on **one** task; all six arms passed.
  Reported speedups are 1.63× DeepSeek, 1.47× Qwen, 0.87× GLM. These are single
  trials with substantial cloud-latency variation; message counts are a proxy,
  while redundant-command counts derive from audit records.
  [fleet interpretation](../../../benchmarks/results/2026-07-07-fleet-orchestration-NOTES.md)
- July 6 coding runs cover exactly two public SymPy tasks per model, all 0/2.
  Notes are sparse; failure causes must not be invented from the aggregate.
  Earlier feasibility notes describe one successful gold-oracle run and
  backend timeouts. This is neither private FrontierCode data nor an illumos
  coding benchmark.
  [Qwen](../../../benchmarks/results/2026-07-06-frontiercode-qwen/summary.md),
  [DeepSeek](../../../benchmarks/results/2026-07-06-frontiercode-deepseek/summary.md),
  [GLM](../../../benchmarks/results/2026-07-06-frontiercode-glm-5.2/summary.md),
  [feasibility](../../../benchmarks/frontiercode/results/FEASIBILITY_FINDINGS.md)

The fixtures are authored tasks, not harvested production incidents. Their
historical VM outcomes are real execution evidence according to the committed
records. Gold-oracle and runner dry-run paths are harness validation only.
The coding grader has an optional quality judge, a size-based scope heuristic,
sampled regression tests, and a reverse check on canonical held-out tests;
that does not assess Lilith's own test-writing competence. Its current prompt
explicitly forbids added tests.
Sources: [coding runner](../../../benchmarks/frontiercode/harness/runner.py),
[grader](../../../benchmarks/frontiercode/harness/grader.py).

## Evaluation integrity gaps to fix before tuning

1. **Clean state and hidden verification.** ADR 086 requires snapshot restore
   per task. The current W3 runner connects once and runs setup per task;
   W6 resets by rerunning setup. Neither is an implemented snapshot boundary.
   W3 uploads verification after the turn to a predictable persistent path;
   without restore/removal, a repeat run could expose an earlier verifier.
   Keep evaluator artifacts outside the agent's accessible filesystem and
   restore a known image per task/arm.
   [ADR 086](../decisions/086-feature-test-vm-harness-benchmarks.md),
   [W3 code](../../../examples/phase50_w3_runner.rs),
   [W6 code](../../../examples/phase50_w6_orchestration.rs)
2. **Audit the grader before calling failures ignorance.** `c-03` compares
   libscf output to `svcs -o state`; committed failures include `online`
   versus `online*`. This is a concrete oracle/transition-format suspicion,
   not proven model failure. Native-API checks partly use source-text grep;
   complement those with fresh inputs, runtime behavior, and review. Validate
   good and bad reference implementations on the exact image first.
   [c-03 verifier](../../../benchmarks/illumos-e2e/tasks/c-03-libscf-smf-state/verify.sh),
   [recorded discrepancy](../../../benchmarks/results/2026-07-08-illumos-e2e-glm-5.2-harness-kg/summary.md)
3. **Separate public development from held-out transfer.** SWE-bench issues,
   gold patches and tests are public. This checkout exposes illumos prompts,
   verifiers, outcomes and the task-targeted KG. Existing exposure is known;
   inclusion in any model's pretraining is unknown. Keep these tasks as
   development/regression data. Author fresh hidden variants and whole held-out
   problem families; exclude their solutions/verifiers from VSA lessons and
   tuning. Score knowledge-off, approved-VSA, and specialist-assisted arms
   separately. [public coding methodology](../../../benchmarks/frontiercode/README.md),
   [KG provenance comments](../../../benchmarks/illumos-e2e/kg/illumos-native-api-kg.toml)
4. **Pin actual identities and environment.** Record model artifact hash,
   quantization, inference backend, prompt/tool policy, image digest plus guest
   package/compiler versions, task hash, memory snapshot and trial seed. Current
   historical bare cloud tags and the reused recipe identity across manually
   upgraded guests are insufficient for the new parity claim.
   [results contract](../decisions/086-feature-test-vm-harness-benchmarks.md),
   [guest mutation history](../../../benchmarks/results/2026-07-07-illumos-e2e-v2-NOTES.md)

## Native Lilith runtime and next steps

Running a cloud model against an illumos guest does not establish that Lilith's
Gemma inference, daemon, VSA persistence, executor and packaged service work
natively. ADR 079 specifies ORT-free native packaging and records unproven
Candle/TLS edges; ADR 082 specifies per-principal execution and correlated
auditing. Lab root SSH and permissive host-key acceptance are not proof of that
deployment contract. The builder script supplies release builds and server
tests, but ends by printing completion after best-effort steps; consumers must
parse individual results or fix aggregate failure reporting before using it
as a gate. Sources: [ADR 079](../decisions/079-release-packaging-workflow.md),
[ADR 082](../decisions/082-privilege-separated-executor-backend.md),
[builder script](../../../tests/vm/scripts/builder-shakeout.sh),
[remote runner](../../../examples/phase50_w3_runner.rs).

Recommended first executable slice:

1. Resolve exact Gemma and comparator artifacts separately; retain the user's
   Qwen 3.8 27B target without silently substituting the older 397B model.
2. Validate the native toolchain image and one known-good/known-bad task in
   each family; classify setup, harness and grader defects before any tuning.
3. Freeze an isolated development baseline with matched budgets, image and
   repeated trials. Emit per-family results and diagnosis records, not only
   aggregate accuracy. Distinguish infrastructure, oracle, API/ABI knowledge,
   reasoning, implementation, verification and coordination failures.
4. Introduce verified, provenance-bearing VSA lessons from development failures;
   test gains on fresh hidden variants and measure regressions with memory off.
5. Add native daemon/inference/service lifecycle checks and a real illumos
   repository patch task including agent-authored tests. Evaluate coordination
   and the human-facing account of results independently of specialist coding.

These are proposed next steps, not claims that new runs have already passed.
