# Runbook — transcript harvest (`harvest-pg`, `akh-harvest`, `akh-derive`, `akh-transcripts-mcp`)

Design: [ADR 108](../ai/decisions/108-agent-session-transcript-harvest.md) ·
[plan](../ai/plans/2026-09-19-transcript-harvest.md) ·
[map](../wayfinder/transcript-harvest/map.md). Everything below was verified live
on 2026-09-19 unless marked otherwise.

## Topology

| piece | where | identity |
|---|---|---|
| `harvest-pg` (CNPG, PG 18.3, timescaledb 2.25.1, vector 0.8.2) | `akh-medu` ns, 2 instances | LAN `192.168.1.60:5432` (`harvest-pg-lan`, pool `homelab-pool-2`), in-cluster `harvest-pg-rw.akh-medu.svc` |
| `akh-harvest` | every machine that runs a coding agent; systemd **user** timer, 15 min | PG role `harvest_<machine>`, **INSERT only** |
| `akh-derive` | in-cluster singleton (advisory lock) | role `harvest_derive` (full DML); migrations run as owner `harvest` |
| `akh-transcripts-mcp` | in-cluster, `http://transcripts.lab.home/mcp` | role `harvest_read` (SELECT), bearer = fleet shared token |

Database `transcripts`, owner `harvest`. Tables `th_raw_records` (hypertable on
`observed_at`, compression after 30 d segmented by `(source, session_id)`),
`th_events`, `th_sessions`, `th_chunks` (HNSW cosine), `th_drift_counters`.

## Auth is mTLS only — and the certificates must be X.509 v3

`pg_hba` is `hostssl transcripts all all cert clientcert=verify-full`; plaintext
is rejected. The client certificate's **CN must equal the role name**.

- CA: the CNPG secret `harvest-pg-ca` in `akh-medu`. **It expires 2026-12-18**
  (CNPG rotates its CA); every client cert below must be re-issued on that
  cadence. Put it in the calendar.
- **Mint client certs as v3.** `openssl x509 -req` without `-extfile` produces
  X.509 **v1** certificates. libpq accepts them; **rustls (sqlx, i.e. every Rust
  binary here) rejects them** with `UnsupportedCertVersion`. Use:
  ```
  cat > client-v3.ext <<'X'
  basicConstraints=CA:FALSE
  keyUsage=critical,digitalSignature,keyEncipherment
  extendedKeyUsage=clientAuth
  subjectKeyIdentifier=hash
  authorityKeyIdentifier=keyid,issuer
  X
  kubectl get secret -n akh-medu harvest-pg-ca -o jsonpath='{.data.ca\.key}' | base64 -d > ca.key
  openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -keyout ROLE.key -out ROLE.csr -subj "/CN=ROLE"
  openssl x509 -req -in ROLE.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 730 -sha256 -extfile client-v3.ext -out ROLE.crt
  shred -u ca.key
  ```
- **rustls also verifies the hostname even under `sslmode=verify-ca`.** The
  server SANs are `harvest-pg-rw|-r|-ro.akh-medu(.svc(.cluster.local))` and
  `harvest-pg.lab.home` — not the LAN IP. External harvesters therefore connect
  to `harvest-pg.lab.home` with `sslmode=verify-full`; on hosts where lab DNS is
  not configured, one `/etc/hosts` line does it:
  `192.168.1.60 harvest-pg.lab.home`. Do not fall back to `sslmode=require`
  (it authenticates the client but not the server).
- In-cluster secrets holding certs: `harvest-pg-client-harvest` (owner, migrate
  initContainer only), `harvest-pg-client-harvest-derive`,
  `harvest-pg-client-harvest-read`. Keys `ca.crt`, `tls.crt`, `tls.key`.
- Note: the older `akh-pg` cluster's `sslrootcert` on hephaistos is **stale**
  (dated 2026-08-23) and fails verification; that is unrelated to this store but
  will confuse anyone reusing it.

## Roles and why tables must be created by `harvest`

Default privileges are declared `FOR ROLE harvest`:
```
ALTER DEFAULT PRIVILEGES FOR ROLE harvest IN SCHEMA public GRANT INSERT ON TABLES TO harvest_hephaistos;
ALTER DEFAULT PRIVILEGES FOR ROLE harvest IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO harvest_derive;
ALTER DEFAULT PRIVILEGES FOR ROLE harvest IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO harvest_derive;
ALTER DEFAULT PRIVILEGES FOR ROLE harvest IN SCHEMA public GRANT SELECT ON TABLES TO harvest_read;
```
They apply only to objects **created by `harvest`**. Run migrations as `harvest`
(owner cert). If a member role creates a table, the grants silently do not
apply. Verified result after migration: `harvest_hephaistos=INSERT`,
`harvest_read=SELECT`, `harvest_derive=DELETE,INSERT,SELECT,UPDATE`.

Adding a machine:
```
CREATE ROLE harvest_<machine> LOGIN;
GRANT CONNECT ON DATABASE transcripts TO harvest_<machine>;
GRANT USAGE ON SCHEMA public TO harvest_<machine>;
GRANT INSERT ON th_raw_records TO harvest_<machine>;   -- existing table
ALTER DEFAULT PRIVILEGES FOR ROLE harvest IN SCHEMA public GRANT INSERT ON TABLES TO harvest_<machine>;
```
then mint its v3 cert (CN = role), install the binary, env file and timer.
Removing one: `DROP ROLE harvest_<machine>` and delete its cert.

## Harvester on a machine (hephaistos is the reference)

- Env: `~/.config/akh-transcripts/harvest.env` (mode 600) — `AKH_TRANSCRIPTS_DATABASE_URL`,
  `AKH_TRANSCRIPTS_MACHINE`, `AKH_TRANSCRIPTS_STATE_DIR`, `RUST_LOG`.
- Units: `~/.config/systemd/user/akh-harvest.{service,timer}`; binary at
  `~/.local/bin/akh-harvest`; `systemctl --user enable --now akh-harvest.timer`.
- The source files are the spool: a failed push retries next tick. Watermarks
  are `(inode, byte offset)` per stream under the state dir; deleting them is
  harmless — a full re-scan collides into `ON CONFLICT DO NOTHING`.
- **Quote the URL in the env file.** systemd's `EnvironmentFile=` reads it
  fine unquoted, but anyone who `source`s the file in a shell gets every `&` in
  the query string interpreted as a job separator and the variable silently
  unset. Write `AKH_TRANSCRIPTS_DATABASE_URL='postgres://…'` (single quotes).
  Also: never edit it with `sed s/…/URL/` — an unescaped `&` in a sed
  replacement re-inserts the match (this produced a quadruplicated URL once).
- **Transcripts can contain NUL bytes.** Postgres `text`/`jsonb` reject `\0`
  outright (`unsupported Unicode escape sequence`); the harvester strips exactly
  that byte from `raw`/`payload`. It is the one deviation from byte-exact
  storage, and it is logged.
- **Health is the watermark, not the process.** A harvester that runs and
  harvests nothing looks identical to a broken one. Check watermark age against
  the source file's mtime.

## Status — hephaistos (2026-09-20)

`akh-harvest.timer` **enabled and active**, 15-minute cadence; binary at
`~/.local/bin/akh-harvest` (release build of `feat/transcript-harvest`). First
full harvest: 93 streams, ~75k records in 74 s; re-scan after deleting the
watermarks deduped 75,120/75,160 rows. Steady state: a few hundred new records
per tick (this machine's own live Claude Code sessions). Store at that point:
75,530 raw rows, 13 sessions, 92 streams. 8 records quarantined per run — the
workflow `journal.jsonl` with no timestamps anywhere, the documented trap.

## Known limits / open items

- Only `hephaistos` is provisioned. Other machines need the "Adding a machine"
  steps; `armini`, `archibald`, `exec1` were not reachable at rollout time.
- `harvest-pg.lab.home` is not in lab DNS (only `/etc/hosts` on hephaistos).
- Compression has not yet triggered (needs rows older than 30 d).
- Rollout status of `akh-derive` / `akh-transcripts-mcp`: see the plan; update
  this section when they are live.
