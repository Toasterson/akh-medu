# Runbook — PVC-less akhomed on arm64 (Postgres mode)

The in-cluster akhomed is meant to run on the cluster's 7 arm64 nodes (Raspberry
Pis), which boot from SD cards. A persistent data dir there is redundant — in
Postgres mode Postgres is the source of truth (ADR 049: PG = truth, redb = cache)
— and continuous small writes to flash wear it out. This runbook deploys
akhomed with **no PVC at all**, putting the data dir on tmpfs.

> Prerequisite: `[postgres]` is configured in akhomed's `akhomed.toml` (or
> `DATABASE_URL` is set). PVC-less mode without Postgres is **not supported** —
> there is no source of truth to wake from.

## What you give up

An **ungraceful** exit (power cut, OOM kill, SIGKILL) drops everything since the
last WAL flush. The exposure is bounded by `postgres.wal_flush_interval_secs`
(default 2 s) plus anything buffered in the in-memory WAL up to
`postgres.wal_max_buffer_mb` (default 16 MiB, beyond which the overflow drop
policy applies). A **graceful** pod delete is safe: `shutdown_wals` stops the
periodic snapshot ticker, takes one final snapshot per loaded workspace, and
drains the WAL flusher before exit. The `terminationGracePeriodSeconds` chart
value (default 120 s) gives the kubelet room for that final snapshot to land.

## Chart values

```yaml
akhomed:
  persistence:
    enabled: false
    medium: "Memory"        # tmpfs — never touches the SD card
    sizeLimit: "2Gi"         # bound the RAM cost; tune to the workspace
  terminationGracePeriodSeconds: 120
  models:
    # Separate decision. See "Models volume" below.
    enabled: true            # or false, with the tradeoff noted
```

The values.yaml file carries the full commentary on each field.

## Verify the data dir is tmpfs

After the pod is up:

```sh
kubectl -n <ns> exec -it <akhomed-pod> -- sh -c 'mount | grep /data'
# expect:  none on /data type tmpfs (rw,relatime,size=...)
```

If you see the node's filesystem (overlay, ext4, …) the chart values did not
take effect — check that `persistence.enabled: false` and `medium: "Memory"`.

## Cold start: wake from Postgres

A PVC-less pod boots with an empty data dir. `akhomed`'s auto-start path opens
a fresh redb, so the in-memory KG + HNSW start **empty** — there is no
auto-wake. Rebuild from Postgres with the control-plane command:

```sh
kubectl -n <ns> exec -it <akhomed-pod> -- akh -w <workspace> akh wake
```

`snapshot::wake` (`src/snapshot.rs:599`) reconstructs the symbol registry, KG
triples, item-memory HNSW, agent meta (drives, agenda, goals, …), and replays
the WAL tail newer than the snapshot watermark. It also persists the rebuilt
state to local redb so a plain restart (without Postgres) still recovers.

The live test `wake_reconstructs_identical_akh` in
`tests/snapshot_live.rs:131` is the gating experiment for this path: snapshot
on a source host, append a WAL tail, then `wake` into a fresh tempdir and
assert the labels match and the tail replays. Run it against a throwaway DB:

```sh
AKH_TEST_PG_URL=postgres://user@127.0.0.1:5432/akhmedu?sslmode=disable \
  cargo test --features server --test snapshot_live -- --ignored
```

If that test does not pass, **stop** — PVC-less mode is not safe.

## Graceful pod delete + restart

```sh
kubectl -n <ns> delete pod <akhomed-pod>
# pod terminates: SIGTERM → shutdown_wals → final snapshot + WAL drain → exit
# kubelet reschedules; new pod starts with an empty tmpfs
kubectl -n <ns> exec -it <new-akhomed-pod> -- akh -w <workspace> akh wake
```

No committed state is lost: the final snapshot covers everything up to
SIGTERM, and the WAL tail between the snapshot watermark and the SIGTERM is
drained by `shutdown_wals`.

## Measure the final-snapshot duration

`snapshot_once` logs `snapshot persisted` with `capture_ms`, `persist_ms`,
and `total_ms` fields. On a graceful shutdown look for the
`final snapshot written on shutdown` log line and read the elapsed from the
preceding `snapshot persisted` for the same workspace.

```sh
kubectl -n <ns> logs <akhomed-pod> | grep -E 'final snapshot|snapshot persisted'
```

Pick `terminationGracePeriodSeconds` to comfortably exceed the largest
`total_ms` you observe across all loaded workspaces (the shutdown path takes
one final snapshot per workspace serially, so sum them). The default 120 s is
conservative for a single workspace on a Pi.

## Models volume

The 5 GiB models PVC (ONNX NER + LLM GGUF + Candle models) is read-mostly and
a tmpfs copy would cost real RAM on an 8 GiB Pi (~1.1 GiB for the GGUF alone).
Three options:

1. **Keep the PVC** on a `storageClass` that works on arm64 (nfs, a manual PV
   to a USB SSD, …). Best for >1 replica or frequent restarts.
2. **Disable the PVC** and let models re-download on each start
   (`akh setup models`). Fine for a single replica with cheap egress; painful
   on a slow Pi uplink.
3. **Pin to amd64** (override `supportedArchitectures` + `nodeSelector`) and
   keep the zfs PVC. Trades the arm64 win for a working models cache.

The chart does not pick — decide per deployment.

## RAM cost

The data dir tmpfs holds the redb cache + KG petgraph + HNSW dump for each
loaded workspace. For a small workspace this is well under 1 GiB; the WAL
buffer adds up to `wal_max_buffer_mb` (default 16 MiB) on top. `sizeLimit`
bounds the tmpfs — if it's exceeded, the kernel evicts to the node's
ephemeral storage (the SD card), so keep `sizeLimit` above the working set.
On a wktiny node (1.8 GiB RAM) leave headroom for the container; on a full
Pi (8 GiB) `2Gi` is comfortable.
