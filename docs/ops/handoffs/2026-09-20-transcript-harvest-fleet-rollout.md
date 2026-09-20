# Handoff — roll the transcript harvester out to every machine, and monitor it

**For:** the agent that can reach the whole fleet (tailnet + LAN + illumos hosts).
**From:** the session that built and deployed ADR 108 on 2026-09-19/20.
**Read first:** `docs/ops/transcript-harvest.md` (runbook — auth, roles, certs,
quirks), `docs/ai/plans/2026-09-19-transcript-harvest.md`, ADR 108.

## What exists and is verified (as of 2026-09-20 19:06 UTC)

- `harvest-pg` (CNPG, TimescaleDB 2.25.1 + pgvector) at LAN `192.168.1.60:5432`
  = `harvest-pg.lab.home`, in-cluster `harvest-pg-rw.akh-medu.svc`. mTLS only;
  cert CN = role. Database `transcripts`.
- **In-cluster**: `akh-medu-transcripts-derive` (singleton worker) and
  `akh-medu-transcripts-mcp` (read surface at `http://transcripts.lab.home/mcp`,
  bearer = `akhomed-shared-secret`). Running image `d72262cacdc2`.
- **hephaistos**: harvester installed, systemd user timer every 15 min. It has
  produced everything in the store so far: 77,526 raw rows / 94 streams, all
  derived, 5.3k chunks embedded. Nothing from any other machine yet.
- **Pre-provisioned for you** (so each machine is one command): roles
  `harvest_armini`, `harvest_archibald`, `harvest_exec1` (INSERT-only, proven),
  v3 client certs, and sealed bundles on hephaistos at
  `~/.config/akh-transcripts/bundles/harvest_<machine>-certs.tgz`.
- **Installer**: `scripts/install-akh-harvest.sh` (on `main`) — Linux systemd
  user timer, macOS launchd agent, illumos SMF manifest. **Status**:
  `scripts/harvest-status.sh` (read-only, per-machine rows/age, pending, lock
  holder, drift).

## Your job

### 1. Find every producer
A producer is any user home with `~/.claude/projects/**/*.jsonl`,
`~/.codex/sessions/**/*.jsonl`, or `~/.local/share/opencode/opencode.db`.
Known candidates: **`armini`** (Mac mini, macOS arm64, tailnet `100.122.96.27`,
the most likely second producer), **`archibald`**, **`exec1`** (OpenIndiana).
On 2026-09-20 from hephaistos: Tailscale was **logged out**, `archibald`/`exec1`
had no DNS, `node03` was reachable but has no transcripts. Discover rather than
assume; report the inventory before installing.

### 2. Reachability rule — do not work around it
`harvest-pg` is **LAN-only** (`harvest-pg-lan`, Cilium pool `homelab-pool-2`).
A machine must reach `192.168.1.60:5432`. A tailnet-only or Internet-only host
cannot push; that needs either the host on the LAN or an Internet endpoint for
`harvest-pg` (akh-pg has one, harvest-pg deliberately does not). If a producer is
not LAN-reachable, **stop and report** — that is the operator's call. Never
downgrade `sslmode` to `require`; rustls verifies the server name even under
`verify-ca`, so the URL must use `harvest-pg.lab.home` + `verify-full`. If lab
DNS is absent on a host, add `192.168.1.60 harvest-pg.lab.home` to `/etc/hosts`.

### 3. Install per machine
```sh
# from hephaistos
scp ~/.config/akh-transcripts/bundles/harvest_<m>-certs.tgz <host>:
# on the machine (needs cargo; the binary is built natively — no cross-compile)
git clone https://code.aopc.cloud/toasterson/akh-medu.git && cd akh-medu   # or pull main
sh scripts/install-akh-harvest.sh --machine <m> --bundle ~/harvest_<m>-certs.tgz --build
```
`<m>` must be the role suffix (`armini`, `archibald`, `exec1`): the installer
checks the cert CN equals `harvest_<m>` because `pg_hba clientcert=verify-full`
maps CN → role. On illumos the installer writes the SMF manifest and prints the
`svccfg import` line (needs `pfexec`). For a machine not yet provisioned, follow
"Adding a machine" in the runbook (roles are created **as the `postgres`
superuser via `kubectl exec -n akh-medu harvest-pg-1`** — the `harvest` owner
lacks CREATEROLE; certs are minted from the `harvest-pg-ca` secret with the v3
extfile in the runbook — v1 certs are rejected by rustls).

### 4. Acceptance per machine
- First run prints `harvest complete: streams=<n> records_read=<n> inserted=<n> …
  failed_streams=0`, with `discovered streams` per source logged above it.
  Claude Code discovery must be **recursive** (`<sessionId>/subagents/...`); a
  count that looks like "number of top-level files" is wrong.
- Second run: `inserted=0` for static sources (the machine's own live sessions
  keep adding a few rows — that is fine).
- `scripts/harvest-status.sh` shows the machine with rows and `age` under the
  timer interval; `pending` returns to 0 within the worker interval (30 s).
- `quarantined=<n>` is expected only for records with no timestamp anywhere in
  their stream (Claude workflow `journal.jsonl`); anything else, investigate.

### 5. Monitor (what "healthy" means)
- **Health is the watermark, not the process.** A harvester that runs and
  harvests nothing is indistinguishable from a broken one by exit status. Use
  `harvest-status.sh`: per-machine `age` must stay below ~2× the interval while
  the machine is producing. State/watermarks live in
  `~/.local/state/akh-transcripts/`; deleting them is harmless (re-scan dedupes).
- Worker: `worker singleton held by: <pod ip>` and `pending = 0`. The worker pod
  currently logs **nothing** (image `d72262cacdc2` logs under `akh_derive`, the
  chart filter is `akh_transcripts=info`) — this is fixed in image
  `0b5fe0f963f8`; the HelmRelease bump to it is **staged, uncommitted** in
  `~/ws/homelab/talos` (`clusters/talos.wegmueller.it/akh-medu/helmrelease.yaml`)
  and needs a signed commit (`ssh-add ~/.ssh/id_ed25519` first). Until then,
  silence from that pod is not a failure.
- Read surface: `curl -H 'Host: transcripts.lab.home' http://192.168.1.12/readyz`
  → 200; `/mcp` without token → 401.
- Drift counters: `codex/world_state` undetermined and `claude_code/attachment`
  degraded are **known**; a *new* record type appearing is the drift alarm.
- Cluster CA expires **2026-12-18**: every client cert (all `harvest_*`) must be
  re-issued from the rotated CA before then — put it in a calendar.

### 6. Do not
- Print record content from the store or logs (it contains credentials); use
  counts and keys.
- Query OpenCode's `account`/`credential` tables (live OAuth tokens).
- Run `akh-derive` outside the cluster while the in-cluster one runs (it refuses
  via the advisory lock; don't fight it).
- Edit `harvest.env` with `sed` and an unescaped `&`; quote the URL in single
  quotes (see runbook).

## Open items you may hit
- **Solstice CI was stalled** on 2026-09-20 (no `rust` job started; zombie runs
  from 09-14). Images/chart were built and pushed locally
  (`docker buildx --builder akh-builder`, helm via `alpine/helm` container —
  `/usr/local/bin/helm` is owned by `twe`, mode `--x--x---`).
- `docker/akh-transcripts.Dockerfile` uses BuildKit cache mounts; unverified on
  the Solstice runner.
- Claude `attachment` records derive as degraded `context` (normalizer does not
  find their text) — a normalizer follow-up, tracked by the drift counters.
