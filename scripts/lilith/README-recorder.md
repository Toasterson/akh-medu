# Lilith CoS read-only tracker recorder

`cos_recorder.py` captures a content-addressed snapshot of what Lilith would
read as Chief of Staff: Anima (plans, work items, live-item detail/loop-state,
changed-item threads, sessions, changes, CI status, scheduling state), the
akh-medu git refs, the Agent Board (REST, GET-only, when a token file is
configured) and fleet status (GET-only endpoints from a config file). Anima can
be captured for several projects in one run.

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

**Deployed layout: local root + mirror.** The recorder writes to a *local*
root (`~/Library/Application Support/akh-medu/lilith/cos-snapshots`) because
writing the NFS share directly from a launchd job hit EINTR. A separate job,
`cos_mirror.py`, copies completed snapshots to the share (see "Mirroring to
the NFS share"). The built-in default root is still the NFS mount point, for
interactive runs.

## Collectors

| Collector | Status values | What it does |
|---|---|---|
| `anima` | `ok`, `partial`, `error` | Akh-Medu project: plans, work items (paginated), per live item detail / loop state / sessions / changes / scheduling state, per-item thread, project CI status. |
| `anima:<project_id>` | same | One extra collector per additional Anima project, identical shape (see "Several Anima projects"). |
| `git` | `ok`, `error` | `git ls-remote` of the akh-medu remote. |
| `board` | `ok`, `partial`, `error`, `not_configured` | Agent Board REST, GET only: `/v1/agents`, `/v1/message-types`, `/v1/threads`, and the messages of every **open** thread (forward `next-cursor` pagination, up to 50 pages of 200 per thread; hitting the cap marks the run `partial`). The inbox is never read. `not_configured` until `--board-secrets-file` / `LILITH_BOARD_SECRETS` names a token file. |
| `fleet` | `ok`, `partial`, `error`, `not_configured` | GET of each endpoint in a fleet config (`--fleet-config` / `LILITH_FLEET_CONFIG`); see below. `not_configured` without one. |

### Threads: absence is a result, not an error

A work item without a thread used to make `get_thread` fail (`{'code': -32603,
'message': 'Thread not found'}`), which marked every snapshot `partial`
(118 of 202 live items). Now:

1. For a live item the recorder already holds `get_work_item_detail`. If its
   `thread_messages` is an empty list and `thread_messages_elided` is false,
   `get_thread` is **not called**; the snapshot records an explicit absent
   result under the same call and args: object `{"thread": null,
   "thread_state": "none"}`, manifest record field `"thread_state": "none"`.
   (Measured 2026-09-29: this also covers 3 items whose thread exists but has
   no messages, whose `get_thread` answer was only `Thread ID: ...`.)
2. Otherwise `get_thread` is called. Only the documented contract
   (`THREAD_ABSENT_CONTRACT` in the code: tool `get_thread`, code `-32603`,
   message exactly `Thread not found`) is read as absence and recorded the same
   way. Any other error (other code, other message, other tool) stays an error
   and marks the collector `partial`.

A clean run therefore reports `status: "ok"`, and `extra.thread_states` counts
`none` / `present`.

### Several Anima projects

`--project-id ID` (repeatable or comma-separated) or `LILITH_ANIMA_PROJECT_IDS`
selects the projects; the default is the Akh-Medu project alone. The Akh-Medu
project id keeps the collector name `anima`, every other id becomes
`anima:<project_id>`. Every response record carries an additive `project_id`
field and each collector's `extra.project_id` names its project. The schema
stays `lilith/cos-snapshot@1` (unknown record fields are ignored by readers,
older snapshots simply have no `project_id`). `scripts/lilith/dojo/snapshot.py`
loads both: `Snapshot.by_call`/`work_items`/`plans()` keep returning the
Akh-Medu view, `Snapshot.project_collectors` maps project id to collector, and
`work_items_of(collector)` / `plans(collector)` read another project.

Deployed set (operator decision 2026-09-29): Akh-Medu
`019d9844-ce74-7993-a9f0-c953b8f04a4e` plus the illumos family so the
illumos-guidance dojo family has evidence. Configure them in the LaunchAgent
(`LILITH_ANIMA_PROJECT_IDS`), not as a code default:

```
019d9844-ce74-7993-a9f0-c953b8f04a4e  Akh-Medu
019dda27-b1c5-79b0-929b-f9debcbd0b0b  illumos
019de483-cdd9-7f52-9f49-7db6177102b3  OpenIndiana
019f998a-ac52-7a82-80cd-814a6aac2d76  ips
019f998a-ac75-7270-8798-c398fb98d5bd  zmgr
019f998a-b1e5-7cb2-b534-cb4cd22a2c10  reddwarf
019f998a-c42a-7e30-97eb-ba18b1c1c1f5  sysconfig
019f998a-c9ac-7771-9e23-4488eda20ce0  machined
019f998a-d107-7ce1-bab6-2b09770d87d1  installer-ui
019f998a-d4c1-7503-8fe8-bec7b6a9ffec  installadm
019f998a-da51-7550-a556-5d1ec9557055  instcomd
019f998a-6aca-7681-a7e2-55e768d68455  refraction-forger
```

### Read volume

Measured on 2026-09-29 with the full 12-project set (Akh-Medu 202 live of 525
work items, illumos 307 live of 454, OpenIndiana 706 live of 711, nine small
projects with 0 to 4 items). Anima MCP calls per run, counting the three
handshake calls per project and excluding the skipped `get_thread` calls:

| Profile | Anima calls per run | Per day (12 runs) |
|---|---|---|
| Today, Akh-Medu only (before this change, incl. 118 failing `get_thread`) | 1,107 | ~13,300 |
| Akh-Medu `full` + the 11 others `detail` (recommended, steady state) | ~2,210 (1,107 + 322 + 726 + 57) | ~26,500 |
| All 12 projects `full` (steady state estimate) | ~7,100 | ~85,000 |
| First run into an empty root (every item counts as changed) | ~7,500 | once |

The `detail` profile (`--detail-project-id`) reads only `get_work_item_detail`
per live item (which already includes the thread messages) plus `get_thread`
for the rare elided thread; it skips loop state, sessions, changes and
scheduling state. Use `full` for projects whose gate/session state the dojo
needs; the illumos family needs the item text and threads, so `detail`. The
other local reads (Board about 60 GETs, fleet 3, git 1) do not touch Anima.
Snapshots are large mostly because of one Board thread (observer traffic,
about 66 MB for 2,000 messages, stored once per page and deduplicated across
runs); expect the local root to grow with that thread.


### Board identity

The Board REST API has no read-only credential class, so the read-only
guarantee is enforced client-side by `ReadOnlyHttpClient` (GET only, refuses
any other method and any non-server-relative path before the transport is
touched). Which identity to point it at:

- Recommended: a **dedicated recorder identity** (for example handle
  `akh-lilith-recorder`), so the C3 "zero tracker writes" audit can show, from
  the server side, that this principal never wrote. It has to be minted on the
  Board by the operator; nothing in this repo can do that.
- Interim (works today): Lilith's own token,
  `~/Library/Application Support/roman-akhs/secrets/board-lilith.json`
  (handle `akh-lilith`). Reads only. Caveat: her relay posts under the same
  identity, so a server-side audit cannot separate recorder traffic from hers;
  rely on the client-side guarantee and the unit tests.

### Fleet

Fleet needs GET-able status endpoints and a token per endpoint. On the Mac
mini the Roman akhs already expose them on loopback: the sessions service
(`:18431/v1/sessions?compact=true`), the event service deliveries
(`:18435/v1/deliveries?pending=true`) and the `roman-alerts` monitor
(`:18435/v1/monitors/roman-alerts`). `deploy/lilith/cos-recorder-fleet.example.json`
lists them; the token files (`operator.key`, `observer.key`) stay under
`roman-akhs/secrets/`. Caveats: `operator.key` is an operator-class credential
(the client is GET-only, but the key itself is not read-only); a dedicated
read-scoped session token would be better. The sessions payload is ~200 KB and
changes as sessions progress. Without a config the collector stays
`not_configured`, and the Talos-cluster akhs are not covered.

## Gap report

```sh
python3 scripts/lilith/cos_recorder.py gap-report --day 2026-09-28
python3 scripts/lilith/cos_recorder.py gap-report --since 2026-09-15 --until 2026-09-28 \
    --format markdown --markdown-out /tmp/cos-gap.md
```

For the day or range (UTC), JSON on stdout (or `--format markdown`) with:

- `expected_slots` / `captured_slots` / `missing_slots` (2-hour slots);
- `time_gaps`: consecutive snapshots more than 2.5 h apart, anchored on the
  snapshot just before the range and the next one after it (or now / range end
  when the recorder has stopped);
- `collector_status_counts`: per collector, how many snapshots had each status;
- `error_kinds`: recorded errors bucketed as `<call>/<code|http status|other>`
  from the structured code, not the message text;
- `failures`: the recorded, redacted error strings per snapshot.

`--markdown-out FILE` also writes the short markdown summary. This is the
"daily gap report" of plan slice C3; run it from any scheduler or by hand
against the local root.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LILITH_SNAPSHOT_ROOT` | `/Volumes/akh-medu-runs/lilith/cos-snapshots` | Snapshot root; `--root` overrides. Deployed: the local Application Support path. |
| `LILITH_ANIMA_PROJECT_IDS` | Akh-Medu only | Comma-separated Anima project ids; `--project-id` overrides. |
| `LILITH_ANIMA_DETAIL_PROJECT_IDS` | - | Subset captured with the cheap `detail` profile (see "Read volume"); `--detail-project-id` overrides. |
| `ANIMA_PAT` | - | Anima PAT. If unset, the recorder runs `~/.local/bin/anima-pat get`. Use a read-only-scoped token. |
| `LILITH_BOARD_SECRETS` | - | Board token file (JSON with `bearer_token`, or a plain one-line file); `--board-secrets-file` overrides. Unset = board `not_configured`. |
| `LILITH_BOARD_URL` | `http://127.0.0.1:18089` | Board base URL; `--board-url` overrides. |
| `LILITH_FLEET_CONFIG` | - | Fleet endpoints JSON; `--fleet-config` overrides. Unset = fleet `not_configured`. |
| `LILITH_MIRROR_DEST` | `/Volumes/akh-medu-runs/lilith/cos-snapshots` | Mirror destination (mirror job only). |
| `LILITH_MIRROR_MOUNT` | `/Volumes/akh-medu-runs` | Mount point that must exist before the mirror writes (mirror job only). |

## Read-only guarantee

Every Anima MCP tool call goes through `call_read_tool(client, allowlist,
name, args)`, which raises `ReadOnlyToolRefused` and never builds a request
if `name` isn't in `ANIMA_READ_TOOLS`:

```
list_plans, list_work_items, get_work_item_detail, get_wi_loop_state,
get_thread, list_sessions, list_changes, get_project_ci_status,
get_scheduling_state
```

The Board and fleet collectors go through `ReadOnlyHttpClient`, whose only
send path raises `ReadOnlyHttpRefused` for every method other than GET before
the HTTP transport is touched (`test_cos_recorder_collectors.py` covers POST,
PUT, PATCH, DELETE, HEAD and foreign/relative paths). `test_cos_recorder.py` asserts the refusal
happens *before* any transport call (`transport.calls == []`), and that no
write-tool name (`update_work_item`, `create_work_item`, `post_message`,
`hold_work_item`, `cancel_session`, ...) is present in the allowlist. The exit
condition for WI-514 also calls for a PAT-scope audit on the Anima side —
that is an operational check on the token's server-side scope, not something
this script can prove by itself.

## Fail-closed collectors, atomic writes, no secrets

- Every collector (one `anima*` per project, git, board, fleet) runs independently;
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

## Mirroring to the NFS share

`cos_mirror.py` is a one-shot job (LaunchAgent template
`deploy/lilith/it.akh.lilith-recorder-mirror.plist.template`, every 2 h):

- copies **completed** snapshots only (a `manifest.json` exists; the recorder
  writes it last, atomically) from the local root to the share;
- copies referenced objects first, verifies each (sha256 of the copy equals its
  file name), and copies a snapshot's `manifest.json` last and only when all
  its objects are verified, so the share never holds a manifest with a missing
  object; a copied manifest is checked against the local sha256;
- uses `/usr/bin/rsync` (openrsync, rsync 2.6.9 compatible) with only
  `-t --ignore-existing --files-from` (no `-a`; ownership and mode bits do not
  survive NFS as non-root), retries rsync failures such as EINTR with backoff
  (4 attempts, 2 s doubling) and retries transient read errors while
  verifying;
- is idempotent: a second run copies nothing. An existing object with a
  different size is re-copied with `--checksum`; an existing manifest that
  differs from the local one is reported as a conflict and never overwritten;
- never deletes on either side (rsync's temp files from a killed run may stay
  on the share; they are dot-files and harmless);
- refuses to run when `/Volumes/akh-medu-runs` is not a mount point (exit 2,
  message in the log) instead of writing under the boot volume.

Exit codes: 0 done and verified, 1 usage, 2 share not mounted, 3 copy /
verification failure or conflict (details in the JSON summary line).

```sh
python3 scripts/lilith/cos_mirror.py            # uses the env / defaults above
python3 scripts/lilith/cos_mirror.py --src /tmp/local --dest /tmp/share --no-require-mount
```

## Tests

```sh
uv run --python /opt/homebrew/bin/python3 --with pytest python -m pytest \
  scripts/lilith/test_cos_recorder.py scripts/lilith/test_cos_recorder_collectors.py \
  scripts/lilith/test_cos_mirror.py scripts/lilith/dojo/tests -v
```

(or any Python 3.11+ with `pytest`). No network and no real credentials: the
MCP and HTTP transports are fakes; `test_cos_mirror.py` runs the real
`/usr/bin/rsync` against temp dirs. Covers: allowlist refusal (before any
request), pagination, live/changed filtering, per-collector fail-closed,
thread absence handling, multi-project naming and record tagging, the GET-only
HTTP client, board and fleet collectors, the mirror (verification, retry,
idempotence, no-delete, conflict, unmounted share), the gap report and the
dojo loader on old and new snapshots.

## Deploying the schedules (done by the supervisor, not by this change)

The templates are macOS **LaunchAgents** (not LaunchDaemons: the recorder reads
the Anima PAT through `anima-pat get`, which needs the login Keychain). Run
from a clean checkout of `origin/main` (`.claude/rules/deploy-from-main.md`);
the deployed recorder runs from `~/.local/share/akh-medu/recorder-main`.

```sh
REPO=$HOME/.local/share/akh-medu/recorder-main   # git fetch origin && checkout origin/main first
LOCAL_ROOT="$HOME/Library/Application Support/akh-medu/lilith/cos-snapshots"
SECRETS="$HOME/Library/Application Support/roman-akhs/secrets"
PROJECTS=019d9844-ce74-7993-a9f0-c953b8f04a4e,019dda27-b1c5-79b0-929b-f9debcbd0b0b,019de483-cdd9-7f52-9f49-7db6177102b3,019f998a-ac52-7a82-80cd-814a6aac2d76,019f998a-ac75-7270-8798-c398fb98d5bd,019f998a-b1e5-7cb2-b534-cb4cd22a2c10,019f998a-c42a-7e30-97eb-ba18b1c1c1f5,019f998a-c9ac-7771-9e23-4488eda20ce0,019f998a-d107-7ce1-bab6-2b09770d87d1,019f998a-d4c1-7503-8fe8-bec7b6a9ffec,019f998a-da51-7550-a556-5d1ec9557055,019f998a-6aca-7681-a7e2-55e768d68455
mkdir -p "$HOME/Library/Logs/akh-medu"
cp "$REPO/deploy/lilith/cos-recorder-fleet.example.json" "$HOME/.config/akh-medu/cos-recorder-fleet.json"   # edit paths if needed

# 1. recorder (multi-project + board + fleet)
sed -e "s#__HOME__#$HOME#g" -e "s#__REPO_ROOT__#$REPO#g" -e "s#__PYTHON_BIN__#/opt/homebrew/bin/python3#g" \
    -e "s#__SNAPSHOT_ROOT__#$LOCAL_ROOT#g" -e "s#__LOG_DIR__#$HOME/Library/Logs/akh-medu#g" \
    -e "s#__ANIMA_PROJECT_IDS__#$PROJECTS#g" -e "s#__ANIMA_DETAIL_PROJECT_IDS__#${PROJECTS#*,}#g" \
    -e "s#__BOARD_SECRETS__#$SECRETS/board-lilith.json#g" \
    -e "s#__FLEET_CONFIG__#$HOME/.config/akh-medu/cos-recorder-fleet.json#g" \
    "$REPO/deploy/lilith/it.akh.lilith-recorder.plist.template" > ~/Library/LaunchAgents/it.akh.lilith-recorder.plist
launchctl bootout "gui/$(id -u)/it.akh.lilith-recorder" 2>/dev/null
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/it.akh.lilith-recorder.plist

# 2. mirror (install about an hour after the recorder's last start; order is not load-bearing)
sed -e "s#__HOME__#$HOME#g" -e "s#__REPO_ROOT__#$REPO#g" -e "s#__PYTHON_BIN__#/opt/homebrew/bin/python3#g" \
    -e "s#__SNAPSHOT_ROOT__#$LOCAL_ROOT#g" -e "s#__MIRROR_DEST__#/Volumes/akh-medu-runs/lilith/cos-snapshots#g" \
    -e "s#__MIRROR_MOUNT__#/Volumes/akh-medu-runs#g" -e "s#__LOG_DIR__#$HOME/Library/Logs/akh-medu#g" \
    "$REPO/deploy/lilith/it.akh.lilith-recorder-mirror.plist.template" > ~/Library/LaunchAgents/it.akh.lilith-recorder-mirror.plist
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/it.akh.lilith-recorder-mirror.plist
launchctl kickstart "gui/$(id -u)/it.akh.lilith-recorder-mirror"   # optional first run
tail -n 3 ~/Library/Logs/akh-medu/lilith-recorder-mirror.log
```

To remove: `launchctl bootout "gui/$(id -u)/it.akh.lilith-recorder-mirror"` (and
`.../it.akh.lilith-recorder`). Verify the mirror after its first run: the share
should hold `objects/` plus the snapshot directories, and a second run prints
`snapshots_mirrored: 0`. If launchd cannot reach the share (macOS "Network
Volumes" privacy prompt for the interpreter), the log shows the failure; run it
once by hand from a terminal that has the permission, then retry.

## Known limitations

- Thread presence for live items is inferred from `thread_messages` in the item
  detail; an existing-but-empty thread is recorded as `thread_state: "none"`.
  Non-live items whose list entry changed have no detail, so they call
  `get_thread` and rely on the `Thread not found` contract above.
- Board: the token is a general agent credential (no server-side read-only
  class); reads are limited to open threads, and the inbox is not captured.
  Fleet: covers the Mac-mini Roman akhs only, and needs an operator-class key
  until a read-scoped token exists.
- Pagination and "live item" filtering use small, documented heuristics
  (`_extract_page` takes the first list-valued response field as the items
  page and the first populated `next_page_token`/`next_cursor`/`cursor` field
  as the next-page token; "live" excludes `current_phase == "done"` and
  `status == "cancelled"`, case-insensitively).
- "Changed since the last snapshot" is computed only from the most recent
  prior `manifest.json` under the same root (all `anima*` collectors merged);
  if that file or its referenced objects are missing (first run, or a pruned
  root), every item is treated as changed, so a fresh root captures every
  thread once. A project added later is captured in full on its first run.
- The dojo replay server still serves the primary (Akh-Medu) project; serving
  the extra projects is future work for the illumos-guidance family.
