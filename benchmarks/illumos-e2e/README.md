# illumos E2E task corpus (Phase 50 / W3)

Real illumos system-engineering tasks an akh is graded on, run through the
Phase-48 executor against an OpenIndiana Hipster guest (`exec1` in
`tests/vm/VMFile.kdl`, W1/W2 — not yet built at authoring time). This
directory is the **authoring half** of W3: the task content, deterministic
verifiers, and the format a future runner consumes. It does not itself run
anything.

See:
- Plan: [`docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md`](../../docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md) — W3 section
- ADR: [`docs/ai/decisions/086-feature-test-vm-harness-benchmarks.md`](../../docs/ai/decisions/086-feature-test-vm-harness-benchmarks.md) — §2

## Task shape: `(meta.toml, setup.sh, prompt.md, verify.sh)`

Each task is a directory `tasks/<family>-<nn>-<slug>/` containing exactly
four files:

| File | Runs as | Seen by the akh? | Purpose |
|---|---|---|---|
| `meta.toml` | — | No | Machine-readable task metadata (see below) |
| `setup.sh` | root, on `exec1`, before the task | No | Establishes starting VM state. Idempotent — safe to re-run against a pool of pre-restored snapshots. |
| `prompt.md` | — | **Yes** | The natural-language task, exactly as a sysadmin would receive it. Never leaks what `verify.sh` checks. |
| `verify.sh` | root, on `exec1`, after the task | **No, never** | Deterministic end-state check. Exit `0` = pass, non-zero = fail. Echoes what it checked, so a failed run is diagnosable from its stdout. |

### `meta.toml` fields

```toml
id = "smf-01-clear-maintenance"      # matches the directory's <family>-<nn>-<slug>
family = "smf"                       # one of: smf, manpage-discovery, c-native-libs, rust-illumos, rbac
description = "One-line human summary"
timeout_secs = 600                   # wall-clock budget for the akh's turn (not setup/verify time)
expected_manpages = ["svcs", "svcadm"]  # illumos.org/man pages the akh should have consulted
mode = ["autonomous", "advisory"]    # grading modes this task supports (both, currently, for all 15)

[version_trap]
present = true                       # whether this task is a version-trap (see below)
note = "..."                         # required iff present = true: what wrong answer it traps, and why
```

`version_trap.note` is required (non-empty) when `present = true`, and must
be absent/empty when `present = false` — enforced by the harness-core test
(see "Static validation" below).

## The 5 families

Per ADR 086 §2, ~15 tasks across 5 families, 3 each:

1. **`smf`** — SMF operations (`svcs`/`svcadm`/`svccfg`): diagnosing and
   clearing a service stuck in maintenance, enabling + configuring a
   disabled service, authoring a manifest from scratch for an unmanaged
   daemon. Man pages: [svcadm(1M)](https://illumos.org/man/8/svcadm),
   [svcs(1)](https://illumos.org/man/1/svcs),
   [svccfg(1M)](https://illumos.org/man/8/svccfg),
   [service_bundle(5)](https://illumos.org/man/5/service_bundle),
   [smf_method(5)](https://illumos.org/man/7/smf_method).
2. **`manpage-discovery`** — the right flag from `prstat`/`kstat`/`dladm`/
   `zfs`/`zpool` man pages, no shell-out to Linux-only tools: CPU count +
   clock speed, NIC link state + speed, a `zpool`/`zfs` create with exact
   properties. Man pages: [psrinfo(1M)](https://illumos.org/man/8/psrinfo),
   [kstat(1M)](https://illumos.org/man/8/kstat),
   [dladm(1M)](https://illumos.org/man/8/dladm),
   [zpool(1M)](https://illumos.org/man/8/zpool),
   [zfs(1M)](https://illumos.org/man/8/zfs).
3. **`c-native-libs`** — a small C program compiled against libkstat or
   libscf: count `cpu_info` kstats, read a custom SMF property, read an
   SMF instance's restarter state. Man pages:
   [kstat(3KSTAT)](https://illumos.org/man/3KSTAT/kstat),
   [libscf(3lib)](https://illumos.org/man/3LIB/libscf).
4. **`rust-illumos`** — a small Rust program using illumos-specific FFI:
   the `processor_info(2)` syscall, the Doors API
   (`door_create`/`door_call`/`door_return`), and extended attributes
   (`attropen`/`openat`+`O_XATTR`). Man pages:
   [processor_info(2)](https://illumos.org/man/2/processor_info),
   [door_create(3C)](https://illumos.org/man/3C/door_create),
   [attropen(3C)](https://illumos.org/man/3C/attropen),
   [fsattr(5)](https://illumos.org/man/7/fsattr).
5. **`rbac`** — creating a role bound to a rights profile, authoring a
   custom rights profile granting one elevated command, and narrowing a
   service account's default privilege set — dogfoods the "OS as sole
   security boundary" posture from ADR 082 (Phase 48b). Man pages:
   [roleadd(1M)](https://illumos.org/man/8/roleadd),
   [usermod(1M)](https://illumos.org/man/8/usermod),
   [profiles(1)](https://illumos.org/man/1/profiles),
   [ppriv(1)](https://illumos.org/man/1/ppriv).

## Grading contract

- **Autonomous mode**: `verify.sh` exit code is the *only* signal. The akh
  never sees `verify.sh`, before or after the run. No LLM judge on outcome.
- **Advisory mode** (W4, not built here): an engineer-simulator executes
  the akh's suggested commands verbatim; `verify.sh` still grades the
  resulting end state, plus a rotation-judge rubric on advice quality.
- **Manpage grounding**: graded from the Phase-48 executor's audit trail
  (ADR 082 correlation-index), by checking whether `man`/`apropos` was
  invoked for (at least one of) each task's `expected_manpages` during the
  run. This repo does not implement that check — it is a runner
  responsibility (see "What the runner still has to do" below) — but the
  metadata each task needs for it is in `meta.toml`.
- **Version-trap tasks** make manpage grounding load-bearing on the
  *outcome*, not just a checkbox: the "obvious" Linux/Solaris-11 answer is
  wrong on OpenIndiana Hipster, and only reading the local manpage yields
  the flag/command that actually works. `version_trap.note` documents the
  specific wrong turn each trap catches; `verify.sh` grades the end state
  exactly like any other task (it does not re-detect the wrong turn — that
  is what the audit-trail manpage check is for).

## Static validation (what could be checked without a live guest)

- Every `setup.sh`/`verify.sh` passes `sh -n` **and** `dash -n` (strict
  POSIX; illumos `/bin/sh` is ksh93-based, so anything dash accepts is a
  safe subset — no bashisms).
- `benchmarks/harness-core/tests/illumos_e2e_task_set.rs` parses every
  `meta.toml`, asserts required fields and cross-field invariants
  (`version_trap.note` present iff `version_trap.present`, `family` is one
  of the 5 known families, `expected_manpages` non-empty, `timeout_secs >
  0`), confirms all four required files exist per task, counts >= 15 tasks
  and >= 3 version-traps spanning more than one family, and calls
  `harness_core::hash::hash_task_set_dir` on `tasks/` twice to confirm the
  hash is stable (the W7 "task-set content hash" contract).
- What is **not** validated here (needs a live OpenIndiana guest): whether
  each `setup.sh` actually converges to the state its comment claims
  (e.g. that repeated start-method failures really land an instance in
  `maintenance` within the retry budget), whether every man page/command
  name is exactly right on Hipster's current package set, whether the
  Rust/C FFI declarations are byte-correct against the actual libc/libscf/
  libkstat headers on that image, and whether `pkg install` package names
  guessed in `c-01`'s `setup.sh` match what's actually published.

## Confidence: what needs live-guest scrutiny before trusting a red/green result

All 15 tasks are authored against illumos semantics researched via
illumos.org/man and illumos-gate source, not executed on a live guest.
Relative confidence, most to least:

- **High confidence** (standard, well-trodden CLI/SMF surface): `smf-01`,
  `smf-02`, `smf-03`, `discover-01`, `discover-02`, `discover-03`,
  `c-01`, `c-02`, `c-03`, `rbac-01`, `rbac-02`.
- **Needs live-guest scrutiny**:
  - `rust-01-processor-info` — `processor_info_t` struct layout
    (`pi_state`/`pi_processor_type[16]`/`pi_fputypes[32]`/`pi_clock`) is
    taken from the illumos-gate header, but untested via an actual FFI
    call from Rust.
  - `rust-02-door-roundtrip` — the riskiest task in the corpus. Doors FFI
    without a wrapper crate is delicate (thread creation semantics for the
    door server procedure, `door_arg_t` layout, whether a single-process
    `door_call` from the main thread against a door created in the same
    process behaves as expected without an explicit `fattach`). The
    `verify.sh` watchdog (kills the binary after 10s) prevents a hang from
    blocking the whole grading run, but a wrong-but-plausible
    implementation could still misbehave in ways only a live run surfaces.
  - `rust-03-xattr` — `attropen`/`O_XATTR` semantics researched from
    fsattr(5); the runat(1) cross-check assumes that tool is present on
    Hipster (base OS, should be, but unconfirmed on this image).
  - `rbac-03-restrict-privileges` — the `dd if=/proc/1/psinfo` functional
    check assumes illumos denies opening another user's `/proc/<pid>`
    files once `proc_info` is removed from the caller's effective set;
    this is documented `privileges(5)` behavior but the exact denial mode
    (permission vs. the entry looking empty/absent) is unverified live.
  - `c-01-kstat-cpu-count`'s `setup.sh` guesses at OpenIndiana Hipster
    package FMRIs for a C compiler (`developer/gcc`, `developer/gcc-13`,
    …) — the actual published name is unconfirmed.

## The 15 tasks

| id | family | description | version-trap | expected_manpages |
|---|---|---|---|---|
| `smf-01-clear-maintenance` | smf | Diagnose a service stuck in maintenance and bring it online | **yes** — `svcadm restart` doesn't clear maintenance | svcs, svcadm |
| `smf-02-enable-configure` | smf | Enable a disabled service and set a property via svccfg | no | svccfg, svcadm, svcprop |
| `smf-03-import-manifest` | smf | Author a manifest for an unmanaged daemon and import it | no | svccfg, service_bundle, smf_method |
| `discover-01-cpu-info` | manpage-discovery | Report CPU count + clock speed | **yes** — no lscpu/proc/cpuinfo | psrinfo, kstat |
| `discover-02-nic-speed` | manpage-discovery | Report NIC link state + speed | **yes** — no ethtool/ip | dladm |
| `discover-03-zpool-create` | manpage-discovery | Create zpool+dataset with exact properties | no | zpool, zfs |
| `c-01-kstat-cpu-count` | c-native-libs | Count cpu_info kstats via libkstat | no | kstat_open, kstat_lookup, libkstat |
| `c-02-libscf-read-prop` | c-native-libs | Read a custom SMF property via libscf | no | scf_simple_prop_get, libscf |
| `c-03-libscf-smf-state` | c-native-libs | Read an SMF instance's state via libscf | no | scf_simple_prop_get, libscf, smf |
| `rust-01-processor-info` | rust-illumos | FFI to processor_info(2) for CPU clock | no | processor_info, psrinfo |
| `rust-02-door-roundtrip` | rust-illumos | Doors API round trip | no | door_create, door_call, door_return |
| `rust-03-xattr` | rust-illumos | Extended attributes via attropen/O_XATTR | **yes** — no Linux setxattr/xattr crate | attropen, fsattr |
| `rbac-01-create-role` | rbac | Create a role with a rights profile | **yes** — group-based sudo habit grants nothing | roleadd, usermod, profiles, roles |
| `rbac-02-custom-profile` | rbac | Author a custom rights profile, assign directly to a user | no | prof_attr, exec_attr, pfexec, usermod |
| `rbac-03-restrict-privileges` | rbac | Narrow a service account's default privilege set | no | usermod, ppriv, privileges |

5 version-trap tasks, spanning 3 of the 5 families (smf, manpage-discovery
x2, rust-illumos, rbac) — exceeds the ADR 086 "at least 3" bar.

## Task-set hashing (W7 compatibility)

`harness_core::hash::hash_task_set_dir("benchmarks/illumos-e2e/tasks")`
walks every file under `tasks/`, sorts by relative path, and blake3-hashes
`(path, content)` pairs — independent of directory-walk order or where the
checkout lives (see `benchmarks/harness-core/src/hash.rs`). Any edit to any
task file changes the hash; per ADR 086 §4, a published run's task-set hash
freezes, and further edits fork a v2 series (e.g.
`benchmarks/illumos-e2e-v2/`). The harness-core test in this PR calls this
function twice against `tasks/` and asserts the result is identical, which
is the property W7's leaderboard/results-schema code relies on.

## What the runner (separate work, not built here) still has to do

This directory has no VM, no executor wiring, and no grading loop — it is
purely the task content plus static checks. A future runner must:

1. **Snapshot-restore** `exec1` to a clean baseline before each task
   (`vmctl restore`/`reload` per ADR 086 §2 — "clean snapshot per task").
2. Copy/run `setup.sh` as root inside `exec1` for the task about to run.
3. Hand `prompt.md`'s content to the akh over the Phase-48c executor
   (russh `RemoteBackend`), in whichever grading mode (autonomous /
   advisory) the run is exercising, respecting `meta.toml`'s
   `timeout_secs`.
4. After the akh's turn ends (or times out), run `verify.sh` as root
   inside `exec1` and record its exit code as the deterministic
   pass/fail signal (`harness_core::schema::TaskOutcome`).
5. Scrape the executor's audit trail (ADR 082 correlation-index) for
   `man`/`apropos` invocations, cross-reference against each task's
   `expected_manpages`, and fold that into the manpage-grounding rubric
   score (deterministic component) — this repo does not implement that
   scraper.
6. Emit a `harness_core::schema::RunRecord` per run (task-set hash via
   `hash_task_set_dir`, harness git SHA, model identity, grading mode,
   VM image identity) and write it via `harness_core::store::write_run`,
   per the W7 results schema this corpus is designed to feed.
