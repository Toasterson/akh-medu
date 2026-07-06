# Phase 50 / W2b — `RemoteBackend` validated against the LIVE illumos `exec1` guest

*2026-07-06. ADR 082 (privilege-separated executor backend, Phase 48c) + ADR
086 (feature-test VM harness). Closes the loop opened by 48c: until now
`RemoteBackend` (`src/backend/remote.rs`) was only exercised against an
in-process loopback `russh::server` fixture. This is the first run against a
real sshd on a real illumos guest.*

## TL;DR

`RemoteBackend` drives the live OpenIndiana `exec1` guest correctly: **15/15
checks passed** — `exec` (svcs/uname/kstat/man), the exec error path, and all
six `Filesystem` verbs (write/read round-trip, exists, is_symlink, file_size,
read_dir), plus the `handshake()` posture. One real bug surfaced that the
loopback fixture could never have caught: **`RemoteBackend::write` shelled
out to `base64 -d`, and OpenIndiana's default image ships no `base64` binary
at all.** Fixed by switching the remote-side decoder to `openssl base64 -d`
(present on both illumos and every Linux tested) — see "Bug found" below.

## What was validated

Target: `akh-exec@127.0.0.1:10042` (port-forwarded to the running `exec1`
VM), staged ed25519 key, `HostKeyPolicy::InsecureAcceptAny` (lab/loopback
posture per ADR 086 §1 — the backend logs a loud warning on every connect
under this policy by design).

Harness: `examples/phase50_remote_exec1.rs`, gated behind the
`remote-executor` feature and an `AKH_EXEC1_SSH` env guard (skips cleanly,
exit 0, when unset — never runs as part of `cargo test`).

| Check | Result |
|---|---|
| `connect()` — SSH handshake + publickey auth | PASS |
| `exec: svcs -H svc:/network/ssh` | PASS — `online 13:20:12 svc:/network/ssh:default` |
| `exec: uname -s` | PASS — `SunOS` |
| `exec: kstat cpu_info:0:cpu_info0:brand` | PASS — `Intel(r) Core(tm) i7-6700K CPU @ 4.00GHz` |
| `exec: man -k svcs` | PASS — `svcs(1) - report service status` |
| `exec: man svcs` (page body) | PASS — `SVCS(1) User Commands SVCS(1)` |
| exec error path: `exit 42` | PASS — `ExecOutcome{exit_code:42, success:false}`, no panic |
| exec error path: `ls` of a missing path | PASS — exit_code=2, stderr captured |
| `write()` → `/tmp/akh-probe-<pid>` | PASS (after the base64 fix — see below) |
| `read_to_string()` round-trip | PASS — 75 bytes back, byte-identical |
| `exists()` | PASS |
| `is_symlink()` (false positive check) | PASS |
| `file_size()` | PASS — matches payload length exactly |
| `read_dir("/etc")` | PASS — 197 entries |
| `handshake()` | PASS — `Confinement::RemoteUnverified` |

Full transcript is reproducible — see "How to run it" below.

## Bug found + fixed: `base64 -d` doesn't exist on illumos

`RemoteBackend::write` (pre-fix) built the remote command as:

```text
printf '%s' <base64-of-content> | base64 -d > <path>
```

Manual probing of the live guest found:

```
$ which base64 ; find / -xdev -name 'base64*'
no base64 in /usr/bin /usr/sbin /sbin /usr/gnu/bin
(only Python's base64.py module, no CLI)
```

OpenIndiana's default coreutils set has no `base64(1)` anywhere on `$PATH`.
The old command would fail closed with "command not found" on **every**
`RemoteBackend::write` call against a real illumos target — a bug the
loopback fixture (which runs the "remote" shell on the Linux test host) can
never surface, because the test host always has GNU coreutils' `base64`.

Fix (`src/backend/remote.rs`, `Filesystem::write`): decode with
**`openssl base64 -d`** instead. `openssl` is present on both illumos and
every Linux distro tested — far more universal than the GNU-only `base64`
CLI. One wrinkle: `openssl base64 -d` needs its input newline-terminated to
flush the last line (verified manually: `printf '%s'` with no trailing
newline decodes as empty output, `RC=0`, no error — a silent-truncation
failure mode, not a loud one). Fixed by using `printf '%s\n'` instead of
`printf '%s'`. Verified byte-exact round-trip for short and 300-byte
payloads directly over SSH before touching the Rust, then again through the
fixed `RemoteBackend` (15/15 above).

Also confirmed (so they're *not* misattributed as bugs): `ls --`, `cat --`,
`test -d`, `test -e`, `test -L`, `wc -c` all behave identically on illumos
and Linux — the existing shell-quoting/file-verb commands needed no other
changes. The SFTP subsystem (`Subsystem sftp internal-sftp`) *is* enabled on
exec1, but is moot here since `RemoteBackend` deliberately uses shell
commands, not SFTP (ADR 082 §2 note in the module doc).

## Audit trail: honest gap, not a fabricated pass

Checked whether `RemoteBackend` (or anything downstream) records the
ADR-082 §7 `{ akh_id, session_id, agent_id, backend, host, os_user,
exit_code, correlation_id }` correlation-index envelope. It does not exist
yet:

- `RemoteBackend` itself has no audit/logging hook at all beyond two
  `tracing::warn!` calls (insecure host-key policy, unsafe env var name) —
  no per-command log line, no ledger call.
- `AuditKind::ToolInvocation` (`src/audit.rs`) only carries
  `tool_name`/`params_summary`/`success`/`output_summary`/`duration_ms` — no
  `backend`/`host`/`os_user`/`correlation_id` fields.
- That generic ledger is written one layer up, in `ToolRegistry::execute`
  (`src/agent/tool.rs`), which is backend-agnostic by construction — it
  audits *tool* calls, not which `Backend` served them.

Net: driving `RemoteBackend` directly (as this validation does, bypassing
`ToolRegistry`) produces **no** audit entry at all today. This is real,
unbuilt 48f-adjacent scope, not something this task should paper over — the
Phase 50 W3 runner cannot yet consume a per-command remote correlation index
from this path. Flagged for whoever picks up 48f.

## How to run it

```sh
export CARGO_TARGET_DIR=/home/toasty/ws/agi/akh-medu/target
AKH_EXEC1_SSH=akh-exec@127.0.0.1:10042 \
AKH_EXEC1_KEY=/path/to/staged/akh-exec_ed25519 \
cargo run --quiet --features remote-executor --example phase50_remote_exec1
```

Without `AKH_EXEC1_SSH` set, the example prints a `SKIP:` message and exits
0 — safe in any script, never runs under plain `cargo test`.

## Regression check

- `cargo test --quiet --lib --tests` (default features): 2812 passed, 1
  failed once (`thinking::tests::analogy_goal_recovers_structured_binding`),
  then passed clean on immediate re-run and in isolation ×3 — pre-existing
  VSA-analogy flake (see "Binary VSA analogy is degenerate" in project
  memory), unrelated to this change. Full re-run afterward: all green.
- `cargo test --quiet --lib --features remote-executor 'backend::remote::'`:
  10/10 passed, including the loopback `write_then_read_round_trips_...`
  test — confirms the `openssl base64 -d` fix does not regress the
  Linux-loopback path either.

## Files changed

- `src/backend/remote.rs` — `Filesystem::write`: `base64 -d` → `openssl
  base64 -d`, `printf '%s'` → `printf '%s\n'`; doc comments updated.
- `Cargo.toml` — updated the `base64` dependency's doc comment to match.
- `examples/phase50_remote_exec1.rs` (new) — the live-guest validation
  harness described above.

Branch: `worktree-agent-ac7a798e11cb0f8c4` (worktree off `develop`), based on
`develop@f9a2556`.
