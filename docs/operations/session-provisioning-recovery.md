# Session provisioning and lost-executor recovery

A healthy pod is not execution evidence. Resolve the Anima session to its exact
central session with `get_session_retirement`, then inspect the central turn,
workspace and tool receipts. The legacy lane status and witness counters can
remain `provisioning`/zero for a native turn that is already executing tools.
Do not change business phases to repair these display fields.

## Provisioning starvation

A central `dispatch_deferred` event with `busy` may mean normal sandbox startup,
repository bootstrap, or runtime capacity. Inspect its message and subsequent
progress. An `expired` workspace lease cannot be implicitly reacquired.

The session controller limits archived cleanup to four workers, prioritizes open
sessions, skips missed scan ticks, and reuses durable credential-cleanup receipts.
These bounds prevent historical cleanup from occupying every dispatch worker.
Retire an expired attempt through Anima's canonical cancellation path; preserve
its ownership until cleanup is confirmed. Verify a fresh exact checkout and tool
receipt from its replacement before calling the incident recovered.

The cognitive core can need more than the default 30-second request deadline to
restore a cold Akh. Tecton's measured restore took 100 seconds. Set the session
controller's core endpoint `request_timeout_secs` to 180; unrelated endpoints
retain the 30-second default. Allowed values are 1 through 300 seconds. A
supported `GET /v1/akhs/{akh}/tools` with an adequate deadline can warm the core
without starting model work. Verify a subsequent catalog/context response and
actual session execution; do not reset or discard saved memory.

## Executor already gone

A sandbox allocation can remain `ready` or `attaching` after its executor stops.
Its normal release request cannot contact that executor. Recovery is explicit:

1. Verify the exact central retirement, terminal runtime job, stopped executor,
   original Sandbox/Pod UIDs, and absence of pending commands or publications.
   Inspect unresolved tool receipts; preserve unknown outcomes without asserting
   successful execution. A missing HTTP response alone is not quiescence proof.
2. Preserve remaining workspace files, complete central events and tool records,
   allocation state, and Kubernetes metadata. Record hashes. Check for prior
   kubelet removal of ephemeral volumes; do not claim files were backed up when
   they were already absent.
3. Remove only the verified terminated Sandbox with a UID precondition and
   foreground propagation. Observe resource absence; DELETE acceptance alone
   does not establish cleanup.
4. As the sandbox administrator, POST `/v1/allocations/{session}/reconcile` with
   `observed_absent: true`, no observed binding, and a `resolution` containing a
   unique request ID, the canonical allocation-request SHA-256, terminal
   `completed` resolution, `quiescent: true`, and bounded evidence references.
   Leave `output` and `result` null: this resolves allocation cleanup, not work.
   Canonical request hashing sorts object keys recursively and hashes compact
   UTF-8 JSON. The service also performs its own Kubernetes absence check and
   rejects a present resource, including a replacement with the same name.
5. Verify allocation `released`, the original retained binding/audit, central
   lifecycle `workspace_released: true`, and Anima `retirement_complete` with
   ownership released. When necessary, requeue the existing exact retirement
   intent using `reconcile_session_retirement` and a fresh UUID request ID.

Never edit Anima's database, erase journals, replay uncertain operations, or
manufacture a successful work result to release a lane.
