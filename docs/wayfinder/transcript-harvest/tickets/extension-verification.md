---
type: task
status: closed
resolved: 2026-09-19
blocked-by: []
graduated-from: store-schema.md, 2026-09-19
---

## Question

Confirm that the target Postgres actually has pgvector and TimescaleDB, and
provision them if not.

[The store schema](store-schema.md) is now gated on this.
`scripts/pg-standalone.sh` records that the base CNPG image
`ghcr.io/cloudnative-pg/postgresql:16` ships **neither**: pgvector comes from a
wrapper image (referenced as Anima WI `019dc614`) and TimescaleDB is optional.
Every existing `akhomed-store` migration guards both and no-ops when absent —
which was correct while they were optional, and is not correct now.

Work:

1. Query the live akh-pg cluster (LAN 192.168.1.16:5432) for
   `pg_available_extensions` and report what is actually installed.
2. If pgvector is missing, determine which image the cluster runs and what
   changing it costs — a CNPG cluster image swap is a rolling restart of a
   database akhs depend on, so this is not a free action.
3. If TimescaleDB is missing, no action is required to proceed: the layout
   degrades to a plain table with an identical primary key. Record the decision
   either way.
4. Record the verified extension set so [the store schema](store-schema.md) and
   [the embedding pipeline](embedding-pipeline.md) can stop hedging.

AFK where the credentials allow; otherwise a precise checklist for the human.
Note the standing obstacle that several lab services currently answer 401 from
this machine — see [tracker restoration](tracker-restoration.md).

## Resolution (2026-09-19) — verified against the live cluster

Queried `192.168.1.16:5432` directly (`psql`, client-cert auth from
`~/.config/akh-medu/config.toml`):

```
select name, installed_version from pg_available_extensions
 where name in ('vector','timescaledb');
  vector -> AVAILABLE, not installed
```

- **`vector` (pgvector): AVAILABLE**, not yet installed. `CREATE EXTENSION vector`
  will succeed. [The embedding pipeline](embedding-pipeline.md) and
  [the read surface](read-mcp-surface.md)'s semantic half are **viable**, and the
  store schema's fail-loudly-on-missing-pgvector path will not fire.
- **`timescaledb`: NOT AVAILABLE.** It is absent from `pg_available_extensions`
  entirely — not merely uninstalled. **No hypertable can be created in this
  cluster as it stands.**

### Incidental findings

- **Auth is mTLS, not passwords.** The connection string carries
  `sslcert`/`sslkey`/`sslrootcert`; the client certificate's subject is
  **`CN = akhomed`**, valid to 2028-09-24. This refines
  [ingest transport](ingest-transport.md): per-machine identity would be issued as
  **per-machine client certificates** mapped to per-machine PG roles, not as
  passwords in a URL. That is strictly better than what that ticket assumed, and
  the issuing path already exists.
- **The configured `sslrootcert` does not verify the server**: connecting with
  `sslrootcert=…/ca.crt` fails `certificate verify failed`, while omitting it
  succeeds. The CA bundle on this machine appears stale relative to the server's
  current certificate (local certs are dated 2026-08-23). Not this map's problem,
  but it will bite whoever provisions the next machine.

### Consequence: [store schema](store-schema.md) is flagged

Session 5 chose raw-as-hypertable with a forward-filled `observed_at` **inside the
primary key**, explicitly accepting "a partly-synthetic column inside the primary
key" in exchange for columnar compression. **That exchange is currently
unavailable.** As it stands the design pays the cost and receives nothing:

- `observed_at` sits in the PK, so idempotency depends on carry-forward
  determinism ([ingest transport](ingest-transport.md)'s self-healing re-scan
  rests on it).
- The compression it was meant to enable cannot be turned on.

Three options, **not decided here** — this reverses a user decision and is theirs
to make:

1. **Keep as designed.** Plain table, identical PK, forward-compatible if
   TimescaleDB is ever added. Costs nothing but the complexity already accepted.
2. **Drop `observed_at` from the PK**, reverting to the natural key
   `(source, session_id, stream_id, seq)`. Simpler, and idempotency stops
   depending on carry-forward determinism entirely. This is what the original
   recommendation in session 5 would have produced.
3. **Add TimescaleDB to the cluster.** A CNPG image change is a rolling restart of
   a database the akhs depend on — not free, and for ~3.4 GB/year the compression
   argument is weak on its own.

Given the measured volume, option 2 has the strongest case and option 1 the
lowest cost of inaction.

## Addendum 2026-09-19 — checked the whole fleet, then built our own

The user asked whether the other clusters are affected. **All of them are.**

| cluster | image | `vector` | `timescaledb` |
|---|---|---|---|
| `akh-medu/akh-pg` | `cloudnative-pg/postgresql:18.3-system-trixie` | available | **absent** |
| `djehuty-board/djehuty-board-postgres` | same | available | **absent** |
| `postgres/postgres-ha` | `code.aopc.cloud/toasterson/cnpg-age:18-v1.7.0-rc0` | available | **absent** |

All three run PostgreSQL 18 and are healthy. No cluster in the fleet can host a
hypertable. Note the third already proves a **custom CNPG image pipeline exists**
(`cnpg-age`, built into the user's own registry) — so building one is a known,
travelled path rather than new infrastructure.

**TimescaleDB does support PG 18** — added in 2.23, and
`timescaledb-2-postgresql-18` / `-loader-` packages exist for Debian trixie
(the CNPG base) on both amd64 and arm64.

**Resolution: our own cluster.** `assets/deploy/` holds a `Dockerfile`
(CNPG 18 base + TimescaleDB 2.25.1) and `harvest-pg-cluster.yaml`, modelled on
`akh-pg-cluster.yaml`. The manifest **validates against the live CNPG CRD**
(`kubectl apply --dry-run=server` → `cluster.postgresql.cnpg.io/harvest-pg created`).
Neither has been built, pushed nor applied — both are outward-facing.

### Separate finding: ADR 091's hypertable does not exist

`akhomed_mutation_log` holds **769,001 rows** and
`select count(*) from pg_extension where extname='timescaledb'` returns **0**.
It is a **plain table**. [ADR 091](../../../ai/decisions/091-transcript-episodic-capture.md)
decision 3 describes it as "a TimescaleDB hypertable" and its compression as the
reason raw turns can be kept indefinitely. The guarded migration
(`m20260620_000002_mutation_log_hypertable.rs`) logged its warning and no-opped,
and nothing checked afterwards.

Outside this map's scope, but it is the precise failure the new image's build-time
assertion exists to prevent.
