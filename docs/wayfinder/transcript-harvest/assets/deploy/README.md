# Deploy artifacts — transcript harvest store

Prepared 2026-09-19.

**Status:** the image is **built and pushed** —
`code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1` (amd64; verified with
`docker manifest inspect`). The cluster manifest is **staged in the Flux repo**
(`~/ws/homelab/talos`, `clusters/talos.wegmueller.it/cloudnativepg/`) but **not
committed**: that repo requires signed commits and its SSH signing key is
passphrase-protected with no agent running. Finish with:

```sh
cd ~/ws/homelab/talos
ssh-add ~/.ssh/id_ed25519      # or let the commit prompt
git commit                      # message is staged-ready; see git log for style
git push origin main            # Flux reconciles from here
```

amd64 only: `harvest-pg` uses the `zfs-postgres` storage class, which
`akh-pg-cluster.yaml` pins to the amd64 storage tier. An arm64 build can be added
later with the existing `akh-builder` buildx builder, which supports it.

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

Both steps below are **done** except the final signed commit:

```sh
docker buildx build --platform linux/amd64 --provenance=false \
  -t code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 --push .   # DONE
```

Flux owns these manifests — applying with `kubectl` directly will be reverted.

**Note:** `--push` with the default buildx driver exported to the local image
store rather than the registry; the `akh-builder` (docker-container) builder
pushes correctly. Do not run a second `docker push` for the same tag while a
buildx push is in flight — the registry returns HTTP 500.

### Not yet addressed: no LAN endpoint

The Cilium pool `homelab-pool` (192.168.1.11-19) is **fully allocated**, so
`harvest-pg` has no LoadBalancer. In-cluster consumers (derivation worker, read
MCP server) are fine; **harvesters on external hosts are not**. Either widen the
pool in `infrastructure/cilium-manifests.yaml` or free an address, then add a
`harvest-pg-lan` Service modelled on `akh-pg-lb.yaml`. Widening a LAN range risks
conflicts with addresses outside the cluster's knowledge, so it was left alone.

## Open question this reopens

With a TimescaleDB-capable cluster, session 5's choice to put `observed_at` in
the raw primary key **regains its justification**: compression becomes available,
so the partly-synthetic key column buys something again. The flag on
[store schema](../../tickets/store-schema.md) can be cleared if this cluster is
built. If it is not, dropping `observed_at` from the key remains the
recommendation.
