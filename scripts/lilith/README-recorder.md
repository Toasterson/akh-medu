# Lilith CoS read-only tracker recorder

`cos_recorder.py` captures a content-addressed snapshot of what Lilith would
read as Chief of Staff: Anima (plans, work items, live-item detail/loop-state,
changed-item threads, sessions, changes, CI status, scheduling state), the
akh-medu git refs, the Agent Board (when configured) and fleet status (stub).

This is the read-only evidence base for the CoS dojo hindsight labels (Lilith
plan slice C4) and for gap auditing. It never calls a write tool — see
"Read-only guarantee" below.

- Anima WI: `01a0e270-137c-7923-9d03-aadeb2740f3e` (WI-514)
- Plan: `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md` (slice C3)
- ADR: `docs/ai/decisions/116-lilith-chief-of-staff-guided-rl.md` (Decision 3)

## Requirements

- Python 3.11+ (stdlib only; `urllib`/`json`/`hashlib`/`argparse`/`subprocess`).
- `git` on `PATH` (for the ref snapshot).
- Either `ANIMA_PAT` in the environment, or `~/.local/bin/anima-pat` available
  (the recorder runs `anima-pat get`). Use a **read-only-scoped** PAT so the
  allowlist guarantee is backed by the server side too.
- `pytest` for the test suite (not required to run the recorder itself).

## Run once

```sh
LILITH_SNAPSHOT_ROOT=$(mktemp -d) python3 scripts/lilith/cos_recorder.py run
```

Prints a one-line JSON summary (schema, timestamp, per-collector status /
response count / error count) and writes:

```
<root>/<UTC-timestamp>/manifest.json   # schema "lilith/cos-snapshot@1"
<root>/objects/<sha256>.json           # deduplicated response bodies, shared across runs
```

Omit `LILITH_SNAPSHOT_ROOT` to use the default,
`/Volumes/akh-medu-runs/lilith/cos-snapshots` (the NFS share mount point on
the Mac).

## Gap report

```sh
python3 scripts/lilith/cos_recorder.py gap-report --day 2026-09-28
```

Lists the day's expected 2-hour capture slots (UTC), which were actually
captured, which are missing, and which captured snapshots had a collector in
`error`/`partial` status (with the recorded, redacted error strings).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LILITH_SNAPSHOT_ROOT` | `/Volumes/akh-medu-runs/lilith/cos-snapshots` | Snapshot root; can be overridden per-invocation with `--root`. |
| `ANIMA_PAT` | — | Anima PAT. If unset, the recorder runs `~/.local/bin/anima-pat get`. Use a read-only-scoped token. |
| `BOARD_MCP_URL` | — | Agent Board MCP endpoint. If unset (with `BOARD_TOKEN`), the board collector records `skipped` with a reason — nothing is contacted. |
| `BOARD_TOKEN` | — | Agent Board bearer token. |
| `BOARD_READ_TOOLS` | — (empty) | Comma-separated allowlist of board tool names the collector is permitted to call. The board's tool contract isn't known ahead of time, so with this unset the collector still records `tools/list`'s contract (useful evidence on its own) but calls nothing. |

Fleet/run status has no read-only MCP surface yet; `FleetCollector` always
records `not_configured` with a reason. It is the documented extension point
for wiring one in later — implement `Collector.collect(ctx)` following the
`BoardCollector` pattern and add an instance to `DEFAULT_COLLECTORS`.

## Read-only guarantee

Every Anima MCP tool call goes through `call_read_tool(client, allowlist,
name, args)`, which raises `ReadOnlyToolRefused` and never builds a request
if `name` isn't in `ANIMA_READ_TOOLS`:

```
list_plans, list_work_items, get_work_item_detail, get_wi_loop_state,
get_thread, list_sessions, list_changes, get_project_ci_status,
get_scheduling_state
```

The Board collector applies the same guard against its own
`BOARD_READ_TOOLS` allowlist. `test_cos_recorder.py` asserts the refusal
happens *before* any transport call (`transport.calls == []`), and that no
write-tool name (`update_work_item`, `create_work_item`, `post_message`,
`hold_work_item`, `cancel_session`, ...) is present in the allowlist. The exit
condition for WI-514 also calls for a PAT-scope audit on the Anima side —
that is an operational check on the token's server-side scope, not something
this script can prove by itself.

## Fail-closed collectors, atomic writes, no secrets

- Each of the four collectors (anima, git, board, fleet) runs independently;
  an exception in one is caught by the orchestrator and recorded as
  `{"status": "error", "errors": [...]}` for that collector only — it never
  aborts the others or the manifest write.
- Every file write (`manifest.json`, `objects/<sha256>.json`) is
  tmp-file-then-`os.replace` (atomic).
- Response bodies are deduplicated by content: the same response captured in
  two different runs (or across two live items with identical results) is
  stored once under its sha256.
- Error strings are passed through `redact()` before being stored or printed,
  stripping `Bearer ...` tokens and `Authorization: ...` header values.

## Tests

```sh
uv run --with pytest python -m pytest scripts/lilith/test_cos_recorder.py -v
```

(or any Python 3.11+ with `pytest` installed). The suite uses a fake MCP
transport (`ScriptedTransport`) — no network access, no real Anima PAT.
Covers: allowlist refusal (and that no request is sent), pagination across
`list_work_items` pages, live-item / changed-item filtering, per-collector
fail-closed behaviour, content-addressed dedup, atomic writes, manifest
schema, board skip/contract/call paths, fleet stub, and the gap report.

## Installing the launchd schedule (not done by this change)

The plist template lives at
`deploy/lilith/it.akh.lilith-recorder.plist.template` — a macOS **LaunchAgent**
(not a LaunchDaemon: it must run in the logged-in user's session so
`anima-pat get` can read the login Keychain). To install it by hand:

```sh
sed \
  -e "s#__HOME__#$HOME#g" \
  -e "s#__REPO_ROOT__#$HOME/ws/akh-medu#g" \
  -e "s#__PYTHON_BIN__#$(command -v python3)#g" \
  -e "s#__SNAPSHOT_ROOT__#/Volumes/akh-medu-runs/lilith/cos-snapshots#g" \
  -e "s#__LOG_DIR__#$HOME/Library/Logs/akh-medu#g" \
  deploy/lilith/it.akh.lilith-recorder.plist.template \
  > ~/Library/LaunchAgents/it.akh.lilith-recorder.plist

mkdir -p "$HOME/Library/Logs/akh-medu"
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/it.akh.lilith-recorder.plist
launchctl kickstart "gui/$(id -u)/it.akh.lilith-recorder"   # optional: run once immediately
```

To remove: `launchctl bootout "gui/$(id -u)/it.akh.lilith-recorder"`.

## Known limitations

- The board and fleet collectors are stubs/pluggable seams, not full
  implementations: the board collector only calls tools named in
  `BOARD_READ_TOOLS` (empty by default — contract-only capture), and the
  fleet collector always reports `not_configured`. Neither the Agent Board
  MCP contract nor a fleet-status source were available while implementing
  this recorder.
- Pagination and "live item" filtering use small, documented heuristics
  (`_extract_page` takes the first list-valued response field as the items
  page and the first populated `next_page_token`/`next_cursor`/`cursor` field
  as the next-page token; "live" excludes `current_phase == "done"` and
  `status == "cancelled"`, case-insensitively) because the exact Anima
  response field names were not pinned down in the work item brief. Adjust
  `_extract_page`/`_is_live` if a live capture shows a mismatch.
- "Changed since the last snapshot" is computed only from the most recent
  prior `manifest.json` under the same root; if that file or its referenced
  objects are missing (first run, or a pruned root), every item is treated as
  changed, so a fresh root always captures every thread once.
