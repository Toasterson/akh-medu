# ADR 097: Shared-store migrations require fleet compatibility

- Status: Accepted
- Date: 2026-08-24
- Work item: WI-338 (`01a03078-bdc3-7142-82cf-9a948df60ebc`)

## Context

The akhomed Postgres schema is shared by several independently deployed daemon
instances. During the incident which led to WI-338,
`m20260823_000001_residency_epoch` was applied before every daemon image had
been rebuilt from a compatible revision. Older binaries then refused the
forward schema. Restarting another old pod would have turned a local outage
into a fleet outage.

Schema version is therefore a deployment compatibility boundary, not an
implementation detail of one daemon. A transactionally safe migration can
still be operationally unsafe when readers and writers at other revisions use
the same store.

## Decision

Until the store publishes and enforces an explicit forward-compatible schema
range, shared-store migrations deploy in lockstep:

1. Build and distribute a migration-compatible image to every host first.
2. Prevent restart of old images once the migration is applied.
3. Apply the migration once under the existing advisory migration lock.
4. Roll or restart the prepared images, verifying hydration on one canary Akh
   before sweeping the fleet.
5. Record the minimum compatible binary revision in the rollout/runbook.

An operator must not apply a shared-store migration while any restartable fleet
member still points at an image that rejects or misinterprets that schema.

For online mixed-version deployments, a future change may replace lockstep with
an expand/contract protocol plus a manifest describing the binary's supported
schema interval. That work must prove that old readers tolerate every expanded
column/table and that old writers cannot erase new state. Merely ignoring an
unknown migration name is not forward compatibility.

## Consequences

- Migration rollout is slower and requires an image inventory, but failures are
  bounded to the canary rather than discovered by arbitrary pod restarts.
- Emergency data repair remains an explicit engine/store operation (as in
  WI-338), never hand SQL against one Akh.
- The migration advisory lock continues to serialize DDL; it does not replace
  fleet compatibility checks.
- Cluster and workstation runbooks must treat image build, migration, and
  restart as one ordered change.
