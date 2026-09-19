# Deploy artifacts — transcript harvest store

Prepared 2026-09-19. **Neither has been built, pushed, nor applied** — both are
outward-facing actions against live infrastructure and need a human to run them.

## Why a new cluster

[Extension verification](../../tickets/extension-verification.md) checked every
CloudNativePG cluster in the fleet:

| cluster | image | `vector` | `timescaledb` |
|---|---|---|---|
| `akh-medu/akh-pg` | `cloudnative-pg/postgresql:18.3-system-trixie` | available | **absent** |
| `djehuty-board/djehuty-board-postgres` | same | available | **absent** |
| `postgres/postgres-ha` | `code.aopc.cloud/toasterson/cnpg-age:18-v1.7.0-rc0` | available | **absent** |

No cluster can host a hypertable. Adding the extension to `akh-pg` means an image
swap and rolling restart of the database the akhs depend on, so the transcript
store takes the change on its own cluster instead.

## Files

- `Dockerfile` — CNPG 18 base + TimescaleDB 2.25.1 (trixie/pg18 packages exist
  for amd64 and arm64). Cache mounts are keyed on `TARGETPLATFORM` so parallel
  multi-arch builds do not share an apt cache. Ends with an assertion that
  `timescaledb.so`, `timescaledb.control` and `vector.control` are present —
  a build that silently lacks the extension is the exact failure that left
  `akhomed_mutation_log` a plain table for 769,001 rows.
- `harvest-pg-cluster.yaml` — the `harvest-pg` Cluster, modelled on
  `akh-pg-cluster.yaml`: mTLS-only, ZFS storage, 2 instances.

## To deploy

```sh
# 1. build + push (needs registry credentials)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 --push .

# 2. place the manifest in the Flux repo and let it reconcile
cp harvest-pg-cluster.yaml \
  ~/ws/homelab/talos/clusters/talos.wegmueller.it/cloudnativepg/
# add to that directory's kustomization.yaml, commit, push
```

Flux owns these manifests — applying with `kubectl` directly will be reverted.

## Open question this reopens

With a TimescaleDB-capable cluster, session 5's choice to put `observed_at` in
the raw primary key **regains its justification**: compression becomes available,
so the partly-synthetic key column buys something again. The flag on
[store schema](../../tickets/store-schema.md) can be cleared if this cluster is
built. If it is not, dropping `observed_at` from the key remains the
recommendation.
