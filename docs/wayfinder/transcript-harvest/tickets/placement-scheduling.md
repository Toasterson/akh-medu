---
type: task
status: closed
resolved: 2026-09-19 (partial — remote inventory handed on as a checklist)
blocked-by: []
unblocked: 2026-09-19
---

## Question

Which machines actually run the harvester, and how is it scheduled on each?

Manual work that unblocks nothing conceptual but must happen before the
destination is reached — the destination is a *working* harvester, and a harvester
that only runs on one box does not cover the sessions.

- **Provision a per-machine PG role.** [Ingest transport](ingest-transport.md)
  chose direct Postgres with an insert-only role per machine, its URL delivered by
  the platform's own secret mechanism rather than a config file. Each machine
  added is a role created; each machine retired is a `DROP ROLE`.
- Inventory: which machines have produced sessions in any of the three tools.
  This box, the Mac mini, archibald and exec1 are candidates; confirm rather than
  assume.
- Per platform: systemd user timer (Linux), launchd agent (macOS), SMF manifest
  (illumos). Interval, and what happens on a missed window (laptop asleep).
- Deployment: how the binary reaches each machine and how it is updated.
- Verification: a check that proves a machine is actually harvesting and its
  watermark is advancing, so silent stoppage is visible.

## Resolution (2026-09-19)

### Verified on this workstation

- **Linux x86_64**, `systemctl --user` available, and a **working precedent
  already running**: `khepri-sweep.timer` is an active user timer. The systemd
  user-timer mechanism from [the process model](process-model.md) needs no proving
  here.
- **All three CLIs are installed**: `claude` (`~/.local/bin`), `codex` and
  `opencode` (both under nvm). This machine is a first-class harvest target for
  every source.
- **It holds the entire measured corpus**: 46 Claude Code streams and 33 Codex
  rollouts, 182 MB logical.
- **akh-pg is reachable** from here — `192.168.1.16:5432` accepts connections
  (ICMP is blocked, which is why ping misleads) and authenticates by **client
  certificate**, not password.

### The scheduling decision

**Timer interval is chosen for freshness, not for safety.**
[Source file lifecycle](source-file-lifecycle.md) measured a retention floor of at
least 76 days, so any interval from minutes to days is safe against data loss.
**15 minutes** is a reasonable default: a session finished during a work block is
harvested before the next one starts, at a cost of 96 no-op ticks a day, each of
which is a `stat` per stream.

Per platform: systemd user timer (Linux, proven here), launchd agent (macOS), SMF
manifest (illumos). Deployment is a single static binary plus a
`~/.config/…/harvest.toml` holding the connection URL and cert paths.

**The health check is the watermark, not the process.** A harvester that runs and
harvests nothing looks identical to one that is broken. Report per
`(machine, source, stream)` watermark age; a stream whose watermark has not
advanced while its file has grown is the alarm.

### Remote inventory — handed on as a checklist

Could not be completed AFK, and deliberately not forced:

- **`hephaistos` SSH host key has changed.** `ssh` refused with
  `REMOTE HOST IDENTIFICATION HAS CHANGED`. Not overridden — most likely a
  rebuild, but that is the operator's determination to make, not an agent's.
- **`armini`** (in `~/.ssh/config`, presumably the Mac mini) returned nothing
  within the timeout.
- **`archibald` and `exec1`** do not resolve or respond from this machine and are
  not in `~/.ssh/config`.

Remaining work, precisely:

1. Resolve the `hephaistos` host-key change, then confirm which of hephaistos,
   armini, archibald and exec1 actually run any of the three CLIs. **Only machines
   that produce transcripts need a harvester** — hephaistos runs the akhs, which
   is a different thing.
2. For each confirmed machine, issue a **client certificate** and a matching
   insert-only PG role (see below), install the binary and the platform timer.
3. Verify by watching the watermark advance after a real session.

### Refinement to [ingest transport](ingest-transport.md)

That ticket specified "one role per machine, its URL delivered by the platform's
own secret mechanism". The cluster is better than assumed: it uses **mTLS**, and
the existing client certificate's subject is `CN = akhomed`.

So per-machine identity should be **a per-machine client certificate** (`CN =
harvest-<machine>`) mapped to a per-machine insert-only role, rather than a
password in a URL. Revocation becomes certificate revocation *and* `DROP ROLE`.
The issuing path already exists — the current certs live in
`~/.config/akh-medu/certs/`.

**Caution for whoever provisions the next machine:** the configured `sslrootcert`
on this workstation **no longer verifies the server** (`certificate verify failed`
with it, success without it). The local CA bundle is dated 2026-08-23 and appears
stale relative to the server's current certificate. Fix that before minting new
client certs against it.
