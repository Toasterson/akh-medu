# Runbook — Relocating an Akh to a new host (Phase 38g)

When a host dies (cellar M2 power loss, hardware refresh, cluster pod eviction),
an Akh whose truth lives in Postgres can be **woken on another host** with the
same identity, resuming from the last committed mutation. This is a *recovery*
mechanism, not a routine load-balancer — Akhs are statically homed by default.

> Prerequisite: the Akh's daemon must have been running with a `[postgres]`
> backing store. With local-only persistence there is nothing to relocate.

## Concepts

- **Truth in Postgres, cache on the host.** `akhomed.*` holds the snapshot + WAL;
  the daemon holds an in-process oxigraph/petgraph/HNSW cache. A relocation
  rebuilds the cache from Postgres on the new host.
- **Bounded dataloss.** Recovery replays from the latest snapshot watermark
  forward through the WAL tail. A clean `relocate`/`sleep`/SIGTERM loses zero; a
  hard kill loses at most one WAL flush interval (default ≤5 s).
- **Split-brain guard.** Two guards prevent two hosts owning one Akh:
  1. the `akhomed.akh.last_seen` heartbeat (updated ~every 30 s), and
  2. a session-scoped Postgres **advisory lock** keyed on the `akh_id`
     (`pg_try_advisory_lock(hashtext('akh:' || akh_id))`).
  `akh akh wake` / `akh akh relocate` refuse fast if either guard rejects.

## Graceful relocation (planned move)

On the **source** host (still alive):

```sh
# Snapshot + flush so another host can take over; updates last_seen.
akh -w <workspace> akh relocate
```

Then on the **recovery** host (pointed at the same Postgres via `[postgres]` in
`akhomed.toml`):

```sh
akh -w <workspace> akh wake
akh -w <workspace> acp status      # verify symbol/triple counts match
```

## Dead-host recovery (unplanned)

1. **Confirm the source host is actually down** (don't wake while it may still be
   writing — the advisory lock will refuse, but confirm anyway). If it is merely
   slow, prefer fixing it; a forced takeover requires the source to be gone.
2. **Pick a recovery host** (the cluster primary, or another M2). Ensure its
   `akhomed.toml` `[postgres]` points at the same cluster Postgres.
3. **Wake the Akh:**
   ```sh
   akh -w <workspace> akh wake
   ```
   - If it prints a split-brain error ("advisory lock held" or "last_seen updated
     within 30 s by another host"), the source is still alive/heartbeating. Wait
     for `last_seen` to age out (>30 s) or stop the source, then retry.
4. **Verify:**
   ```sh
   akh -w <workspace> acp status
   akh -w <workspace> daemon status
   ```
   The triple/symbol counts should match the source's last known state (cold wake
   replays snapshot + WAL tail and rebuilds the HNSW from `akhomed.facts` or the
   local-disk cache — Phase 38e).

## Notes

- **HNSW wake cost.** Same-host warm wake = 0; same-host cold wake from the
  local-disk cache = sub-second; relocated cold wake rebuilds the index from
  `akhomed.facts` (~hundreds of ms for 10 K vectors). The e-graph is rebuilt from
  facts on wake (Phase 40 will persist it if rebuild becomes the bottleneck).
- **Migrations.** If the recovery host is a fresh deployment, run
  `akh migrate` (or rely on `migrate_on_start`) before the first wake.
- **Multi-Akh workspaces.** Membership lives in `workspaces.membership`; use
  `akh workspace list-members <id>` to see who shares a workspace.
