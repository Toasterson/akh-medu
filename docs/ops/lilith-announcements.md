# Runbook — Lilith Chief-of-Staff announcement relay + ancestry fetch

> ADR: [116 — Lilith as Chief of Staff](../ai/decisions/116-lilith-chief-of-staff-guided-rl.md)
> (Decision 1 revision 4: T3 operator-class actions are taken autonomously
> and announced afterwards on the operator's Chief-of-Staff thread, with
> receipts, her reason and the rollback path)
> Code: `src/agent/chief_of_staff/{guard.rs,ledger.rs,contracts.rs}`,
> `skills/chief-of-staff/schemas/cos-announcement.schema.json`
> Scripts: `scripts/lilith/cos_announce_relay.py`, `scripts/lilith/ancestry_fetch.sh`
> Plist templates (not installed): `deploy/lilith/it.akh.lilith-cos-announce-relay.plist.template`,
> `deploy/lilith/it.akh.lilith-ancestry-fetch.plist.template`

## What this closes

Two gaps left after the guard/ledger landed:

1. **The announcement relay.** `CosGuard::settle` (`guard.rs`) writes every
   T3 action Lilith takes as a `lilith/cos-announcement@1` line to her
   Chief-of-Staff JSONL log (`ledger.rs`). `ledger.rs`'s own module doc says
   "The relay to the operator's CoS thread posts these lines and skips the
   others" — but nothing did that relaying. `cos_announce_relay.py` is it.
2. **The ancestry fetch.** `GitAncestry` (`policy.rs`) runs
   `git -C <repository> merge-base --is-ancestor <sha> <main-ref>` against
   whatever refs already exist in the guard's dedicated clone — it never
   fetches. Left alone, `refs/remotes/origin/main` in that clone goes stale
   and a legitimately main-landed deploy/restart SHA starts looking like a
   non-ancestor, so the standing rule (`CosError::NonMainArtifact`) refuses a
   real, correct action. `ancestry_fetch.sh` is the scheduled fetch that
   keeps it current.

Neither script is wired into the live Roman deployment by this change —
`deploy/roman-mini/` has no `[chief_of_staff]` wiring yet (that lands in a
separate PR). Both scripts are complete and independently tested; the
sections below are the install steps for whenever that wiring exists.

## `cos_announce_relay.py`

Tails the Chief-of-Staff log by byte offset (state file
`<log>.relay-state.json`, tmp-then-`os.replace`d beside the log — crash-safe
across restarts) and posts each `lilith/cos-announcement@1` line, once, to
the operator's thread on the Agent Board as a `board/note@v1` message: the
action, her reason, an outcome, the receipts (with a `Links:` subsection
pulled from receipt kinds `pull-request`/`work-item`/`commit`/`board-message`/
`document`), the rollback path, and the rule checks that passed.
`lilith/cos-spend-reserved@1` / `lilith/cos-spend-settled@1` lines in the
same log are read past and never posted.

It runs once per invocation and exits — no built-in polling loop — following
the same pattern `cos_recorder.py` already uses with a launchd
`StartInterval`.

### Ordering and failure handling

- **Poison line** (bad JSON, an unrecognized `contract`, or an announcement
  missing a required field): appended to `<log>.relay-quarantine.jsonl` with
  a reason and the offset skips past it. It never blocks later lines.
- **Board post failure** (network error, or the board rejects the write):
  retried with bounded exponential backoff inside the run. If every attempt
  fails, the byte offset is left exactly before that line — the *next*
  scheduled run retries it, in order, before anything behind it. A systemic
  failure (bad token, board down, wrong thread) therefore blocks the stream
  until it is fixed rather than silently dropping an announcement or
  reordering the feed.
- **Exactly-once**: each line's sha256 digest is both the board write's
  `idempotency_key` (a retry after a mid-post crash can't double-post on the
  board's side) and is recorded in the state file's bounded digest ring (a
  defense-in-depth check if the byte offset is ever reverted, e.g. by an
  operator restoring an old state file).

### CLI

```sh
scripts/lilith/cos_announce_relay.py run \
  --log /path/to/cos-log.jsonl \
  --thread-id <operator-cos-thread-uuid> \
  --root "$HOME/Library/Application Support/roman-akhs" \
  --identity lilith
```

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--log` | `LILITH_COS_LOG` | — (required) | The `[chief_of_staff].announcements` JSONL path — the same file the guard writes. |
| `--thread-id` | `LILITH_COS_THREAD_ID` | — (required) | The operator's CoS thread UUID on the Agent Board. |
| `--root` | `LILITH_ROOT` | `~/Library/Application Support/roman-akhs` | Deployment root; only used to default the secrets file path. |
| `--identity` | `LILITH_BOARD_IDENTITY` | `lilith` | Board identity name — secrets file defaults to `<root>/secrets/board-<identity>.json`. The Lilith board identity itself lands in a parallel PR; this flag is why the relay does not hardcode it. |
| `--secrets-file` | `LILITH_BOARD_SECRETS` | derived from `--root`/`--identity` | Overrides the secrets file path directly. |
| `--board-url` | `LILITH_BOARD_URL` | `http://127.0.0.1:18089` | Agent Board base URL — the Roman deployment's local `board-tunnel` port (`configure_integrations.py`). |
| `--state` | — | `<log>.relay-state.json` | Offset + digest-ring state file. |
| `--quarantine` | — | `<log>.relay-quarantine.jsonl` | Poison-line quarantine file. |
| `--max-attempts` | — | `6` | Board post retry attempts before giving up for this run. |

The secrets file is the same shape `daemon_credentials.py` and
`observer.py` already use for every other Roman board identity:
`{"bearer_token": "..."}`, mode `0600`. The token is never written to
stdout, stderr, the state file or the quarantine file — every string that
might carry it is passed through the script's `redact()` first (same
regexes as `cos_recorder.py`'s).

### Install (LaunchDaemon, not automated)

```sh
sed \
  -e "s#__RUN_AS_USER__#toasty#g" \
  -e "s#__HOME__#$HOME#g" \
  -e "s#__REPO_ROOT__#$HOME/ws/akh-medu#g" \
  -e "s#__PYTHON_BIN__#$(command -v python3)#g" \
  -e "s#__COS_LOG__#/path/to/cos-log.jsonl#g" \
  -e "s#__THREAD_ID__#<operator-cos-thread-uuid>#g" \
  -e "s#__DEPLOY_ROOT__#$HOME/Library/Application Support/roman-akhs#g" \
  -e "s#__BOARD_IDENTITY__#lilith#g" \
  -e "s#__LOG_DIR__#$HOME/Library/Logs/akh-medu#g" \
  deploy/lilith/it.akh.lilith-cos-announce-relay.plist.template \
  | sudo tee /Library/LaunchDaemons/it.akh.lilith-cos-announce-relay.plist >/dev/null

mkdir -p "$HOME/Library/Logs/akh-medu"
sudo launchctl bootstrap system /Library/LaunchDaemons/it.akh.lilith-cos-announce-relay.plist
sudo launchctl kickstart -k "system/it.akh.lilith-cos-announce-relay"   # optional: run once immediately
```

It is a **LaunchDaemon**, not a LaunchAgent: like `dev.roman-akhs.{sessions,
events,core}`, it authenticates from a plain `secrets/board-<identity>.json`
file, not the login Keychain, so it does not need the logged-in user's
session — it runs as `UserName` (the deployment owner, **not** root) so it
can read files the operator owns.

To remove: `sudo launchctl bootout system/it.akh.lilith-cos-announce-relay`.

### Tests

```sh
uv run --with pytest python -m pytest scripts/lilith/test_cos_announce_relay.py -v
```

Covers: offset persistence across a simulated restart, a partial trailing
line left unconsumed, exactly-once via the digest ring plus the deterministic
idempotency key, spend-reservation/settlement lines skipped without posting,
poison-line quarantine (invalid JSON, missing fields, unknown contract) never
blocking later valid lines, the bearer token never appearing in the state
file, the quarantine file, stdout or stderr (including an adversarial board
response that echoes the `Authorization` header back), transient-failure
retry-then-success, and a permanent failure leaving the offset exactly before
the failed line so the next run resumes there.

## `ancestry_fetch.sh`

```sh
scripts/lilith/ancestry_fetch.sh <clone-dir> [remote-url] [main-ref]
```

Runs, in order: acquire a portable mkdir-based lock at
`<clone-dir>.fetch.lock` (macOS ships no `flock(1)`; `mkdir` is atomic on
every filesystem this runs on) → bootstrap a **bare** mirror clone at
`<clone-dir>` if it does not exist yet → ensure a remote named `origin`
points at `remote-url` (repairs a missing or wrong one) →
`git -C <clone-dir> fetch --prune origin +refs/heads/<main-ref>:refs/remotes/origin/<main-ref>`.
Defaults: `remote-url` = `https://code.aopc.cloud/toasterson/akh-medu.git`
(public, read-only, no credential — the same URL `cos_recorder.py`'s git
collector already fetches unauthenticated), `main-ref` = `main`.

`<clone-dir>` must be the exact path configured as `[chief_of_staff].repository`
in the akh's config — this is a **dedicated** clone the guard reads through,
never the operator's own working checkout.

### Locking

- A live lock (owning pid still running, checked with `kill -0`) makes the
  script log and exit `0` without touching anything — a skip, not an error.
- A stale lock (owning pid no longer alive) is cleared automatically and the
  fetch proceeds.
- The lock is only released by the process that acquired it — a run that was
  skipped because another fetch already held the lock never removes that
  still-live lock.

### Install (LaunchDaemon, not automated)

```sh
sed \
  -e "s#__RUN_AS_USER__#toasty#g" \
  -e "s#__HOME__#$HOME#g" \
  -e "s#__ANCESTRY_CLONE__#$HOME/Library/Application Support/roman-akhs/state/lilith/ancestry-mirror#g" \
  -e "s#__SCRIPT_PATH__#$HOME/ws/akh-medu/scripts/lilith/ancestry_fetch.sh#g" \
  -e "s#__LOG_DIR__#$HOME/Library/Logs/akh-medu#g" \
  deploy/lilith/it.akh.lilith-ancestry-fetch.plist.template \
  | sudo tee /Library/LaunchDaemons/it.akh.lilith-ancestry-fetch.plist >/dev/null

sudo launchctl bootstrap system /Library/LaunchDaemons/it.akh.lilith-ancestry-fetch.plist
sudo launchctl kickstart -k "system/it.akh.lilith-ancestry-fetch"   # optional: run once immediately
```

`__ANCESTRY_CLONE__` **must** match `[chief_of_staff].repository` in the
akh's config exactly, or the guard checks one clone while this script keeps
another one fresh. It is a LaunchDaemon for the same reason as the relay: no
Keychain access is needed (the fetch is unauthenticated HTTPS), so it runs as
`UserName` rather than root.

To remove: `sudo launchctl bootout system/it.akh.lilith-ancestry-fetch`.

### Tests

```sh
uv run --with pytest python -m pytest scripts/lilith/test_ancestry_fetch.py -v
```

Real `git` against local bare repos (no network). Covers: bare-clone
bootstrap and initial fetch, an idempotent re-fetch with nothing new, a new
commit picked up on the next fetch, a missing/mismatched `origin` remote
repaired, a stale lock cleared and the fetch proceeding, a concurrent fetch
(a real live subprocess holding the lock) skipped with the lock left intact,
and refusal on a directory that exists but is not a git repository.

## Known limitations

- Neither script is wired into `deploy/roman-mini/` yet: there is no
  `[chief_of_staff]` table in the Roman config, no `board-lilith` secrets
  file, and no operator CoS thread ID recorded here. That wiring, and the
  actual board identity, are a separate PR (the Lilith board identity is
  being added in parallel — both scripts take it as a configurable
  `--identity`/`LILITH_BOARD_IDENTITY`, never a hardcoded name).
- The relay's message body is a plain-text `board/note@v1` payload
  (`{"text": ...}`), not a dedicated registered protocol — registering a
  `lilith/cos-announcement@1`-shaped board message type is out of scope for
  this repository (the board is a separate service,
  `code.aopc.cloud/toasterson/djehuty-board`).
- `--max-attempts` bounds the relay's retry loop to the current run; it does
  not itself loop or daemonize. Its retry budget should stay well inside the
  LaunchDaemon's `StartInterval` (120 s) so one invocation cannot still be
  backing off when the next one starts.
