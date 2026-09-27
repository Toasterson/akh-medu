# Lilith cancellation slice — 2026-09-21

The controller now accepts durable operator cancellation while its writer is
active or services are stopped, reconciles previously persisted submissions,
cancels only owned turns, and waits for confirmed terminal states. Artifact
changes and author retirement also stop child work. The mailbox and completion
transaction prevent a committed cancellation from being overwritten by success.

See [the operating guide](../../../examples/lilith/COORDINATION.md#cancel-an-assignment)
for the CLI, recovery rules and cooperative-cancellation limits.

## Investigation evidence

- `run1/result.json`: the initial probe reused an old session; the service's idle
  retirement sweep had retired it. The controller correctly fenced delegation.
  Subsequent probes use a fresh Lilith session.
- `run2/result.json`: the original runtime stopped Ptah during native preparation
  but represented cancellation as `unknown`. The controller stayed `cancelling`
  until its deadline instead of inventing a confirmed outcome. All probe-owned
  processes stopped. This is the observed failure that motivated the Rust fix.
- `run2/audited-recovery.json`: a one-off, evidence-backed reconciliation of that
  exact job after process exit. It had no tool grants, had not loaded Ptah, and
  recorded a cancellation error. This is not an automatic policy for unknown
  runtime outcomes. The API checked the exact request fingerprint and inactivity.

- [run3/result.json](run3/result.json): passed with the rebuilt runtime. After
  observing the Ptah turn as running, the probe persisted cancellation, reopened
  the controller ledger, and received confirmed cancellation in **2.07 seconds**.
  No return/report turn was submitted; the parent turn list stayed unchanged.
  All four owned services stopped. This exercised cancellation during native
  preparation, not interruption during token generation or an outstanding tool.
- `run2/after-audited-recovery.json`: the earlier failed assignment also reached
  cancelled after its explicit recovery; original failure evidence remains intact.

## Runtime fix

Native compute-lane admission, artifact hashing, and the post-load check now
produce the existing typed `model_cancelled`/`model_incomplete` boundary errors.
At these points no broker callback is outstanding and prior tool results are
already durable. The runtime can therefore retire the continuation and record
cancelled/failed correctly. Later generation/callback uncertainty is unchanged;
`unknown` remains non-confirmation for the cancellation controller.

## Validation

- `unit-tests.log`: 40 scoped Python tests pass, including concurrent mailbox
  writes, queued/review/report cancellation, lost submission and cancel responses,
  restart, stale artifacts, immutable outcomes, unknown jobs, and completion races.
- `native-tests.log`: 78 Rust library tests pass, 3 artifact-dependent tests ignored.
  Includes native admission/hash cancellation, occupied compute-lane deadlines,
  runtime boundary handling, and existing session/executor contracts.
- `runtime-build.log`: build log for the local HIP/Vulkan runtime.

The live fixture creates a fresh Lilith readiness turn, host-seeds the already
verified clamp artifact, and seeds a scoped assignment directly in its separate
ledger. It tests the real controller/session/runtime cancellation boundary; it
is not another model-requested delegation test or a coding benchmark. The first
slice retains the model-requested delegation evidence.

Native illumos validation, a second specialist and VSA lesson admission remain
future work. Cancellation does not roll back earlier writes, and callbacks with
uncertain effects still require audited recovery.
