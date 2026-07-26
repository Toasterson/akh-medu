# OpenIndiana VM harness (Phase 50 W1) — runbook + boot quirks

- Plan: [docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md](../ai/plans/2026-07-05-phase50-feature-test-benchmarks.md) (W1)
- ADR: [086](../ai/decisions/086-feature-test-vm-harness-benchmarks.md)
- VMFile: [tests/vm/VMFile.kdl](../../tests/vm/VMFile.kdl)

This is the operational record for standing up the `exec1` / `pg` / `builder`
OpenIndiana guests described in ADR 086, including the real gaps found in
`vmctl` (installed from source at `~/.cargo/bin/vmctl`, developed in the
sibling `vm-manager` repo) while bringing the substrate up for the first
time, and the exact image/package facts verified against the live registry
and OpenIndiana Hipster catalog on 2026-07-05.

## Quick start

```bash
cd tests/vm
vmctl up --name exec1          # bring up one guest; omit --name for all three
vmctl log --name exec1 --provision --tail 200
vmctl ssh --name exec1
vmctl down --name exec1 --destroy
```

`vmctl provision --name <vm>` re-runs provisioners on an already-running
guest (fast iteration on the scripts under `tests/vm/scripts/` without a full
reboot). `vmctl reload --name <vm>` destroys and recreates from a clean
image overlay — this is the "clean snapshot per task" primitive W3's task
runner needs.

For the `builder` VM specifically, the portability shakeout (native
`cargo test --features server` run) is **not** an automatic provisioner —
run it manually per the instructions at the top of
`tests/vm/scripts/run-native-tests.sh`.

## Image: only one OpenIndiana qcow2 is published today

ADR 086 calls for three OpenIndiana guests. As of 2026-07-05 the
CloudNebulaProject registry (`code.aopc.cloud/cloudnebulaproject/-/packages`)
only publishes **one** OpenIndiana disk image:

```
oci://code.aopc.cloud/cloudnebulaproject/openindiana-rust-ci:latest
```

built from `solstice-ci/images/openindiana-rust-ci/*.kdl` by
refraction-forger. It's OpenIndiana Hipster + rustc + sccache +
build-essential + the virtio drivers KVM boot needs, provisioned at first
boot by illumos `sysconfig` (the NoCloud-only cloud-init replacement baked
into the image). There is no lighter/bare OpenIndiana base image published
alongside it.

Per the plan's prerequisite ("pull prebuilt — do not rebuild"), `VMFile.kdl`
reuses this **one image for all three roles** (`exec1`, `pg`, `builder`) and
differentiates them entirely through provisioners. This is a deliberate
deviation from an implied "three distinct images" reading of the ADR; the
rust toolchain/sccache baked into `exec1` and `pg` is unused weight but
harmless. If a lighter base image is published later, only the `image-url`
lines in `VMFile.kdl` need to change — the provisioners are written against
package/service operations, not against "what's already installed."

Image size: the qcow2 blob is **14,765,129,728 bytes (~13.75 GiB)** — much
larger than the `8G` ZFS pool the forger target spec declares, because the
"entire" incorporation + full rust toolchain fill it. Budget bandwidth/disk
accordingly; `vmctl image pull "oci://code.aopc.cloud/cloudnebulaproject/openindiana-rust-ci:latest"`
took ~8 minutes on this host's link.

## Boot quirk #1 (found + fixed): `VMFile.kdl` couldn't express UEFI

The `openindiana-rust-ci` qcow2 target was built with `bootloader "uefi"`
(GPT + ESP, illumos `loader64.efi` — no legacy MBR boot code), so it **will
not boot under legacy BIOS** (SeaBIOS has nothing to chain-load). `vmctl`'s
QEMU backend has always supported UEFI boot (`VmSpec::uefi` + OVMF pflash
wiring in `backends/qemu.rs`, also exposed via `vmctl create --uefi`), but
the **declarative `VMFile.kdl` parser hardcoded `uefi: false`** when
resolving a `VmDef` — there was no `uefi` node to set it from a VMFile at
all. This is exactly the kind of "boot quirk discovered" the plan asked to
be recorded.

Fixed in the sibling `vm-manager` repo (not part of this git history —
`/home/toasty/ws/nebula/vm-manager`, uncommitted, needs its own review/push):

- `crates/vm-manager/src/vmfile.rs`: added a `uefi` bool node
  (`doc.get_arg("uefi").and_then(|v| v.as_bool()).unwrap_or(false)`), a
  `VmDef.uefi` field, wired through `resolve()` into `VmSpec.uefi`. Two new
  parser tests (`parse_uefi_flag`, `uefi_defaults_false`); all 11
  `vmfile::tests` pass.
- `docs/src/vmfile/resources.md`: documented the new `uefi` node.
- Rebuilt `cargo build --release -p vmctl` and reinstalled to
  `~/.cargo/bin/vmctl` so this harness's `VMFile.kdl` (`uefi #true` on all
  three guests) actually works.

`tests/vm/VMFile.kdl` sets `uefi #true` on all three guests as a result.

## Boot quirk #2 (found + fixed): illumos vioblk panics on q35's modern virtio-pci

Even with `uefi #true`, the guest panicked on every boot attempt:

```
NOTICE: Performing full ZFS device scan!
WARNING: vioblk0: Failed to map CAP 2 @ BAR4 0x3000+1000
WARNING: vioblk0: failed to start Virtio init
WARNING: vioif0: Failed to map CAP 2 @ BAR4 0x3000+1000
NOTICE: Cannot read the pool label from '/pseudo/lofi@2:b'
NOTICE: spa_import_rootpool: error 5
Cannot mount root on /pseudo/lofi@2:b fstype zfs
panic[cpu0]/thread=...: vfs_mountroot: cannot mount root
```

The illumos loader (firmware-stage EFI Block I/O) reads the disk fine —
this only happens once the illumos kernel's *native* `vioblk`/`vioif`
drivers try to reattach the same device and parse the "modern" (1.0)
virtio-pci capability list, whose BAR QEMU places in a way these drivers
fail to map, specifically under the `q35` (PCIe) machine type vmctl's QEMU
backend hardcodes.

Fixed with a new opt-in VMFile.kdl node, `virtio-transitional #true`, which
appends `disable-modern=on,disable-legacy=off` to the `virtio-blk-pci` and
`virtio-net-pci` `-device` arguments — forcing QEMU to present only the
legacy (0.9.5) virtio-pci capability layout these drivers have long, mature
support for, sidestepping the modern-capability BAR parsing bug entirely.
Verified empirically: a standalone one-VM smoke test
(`uefi #true` + `virtio-transitional #true`) got past the exact point that
previously panicked (`Performing full ZFS device scan!` → pool imported via
an alternate device path, no `Failed to map CAP` / no panic) on the first
try. Added to `crates/vm-manager/src/{types.rs,vmfile.rs,backends/qemu.rs}`
+ `crates/vmctl/src/commands/create.rs` (`--virtio-transitional` CLI flag
for parity) + a parser test pair, mirroring the `uefi` node exactly.
`tests/vm/VMFile.kdl` sets `virtio-transitional #true` on all three guests.

This is a generically useful, backward-compatible fix (opt-in, default
unchanged) for any illumos/BSD-family guest hitting the same class of
q35 + modern-virtio bug — not specific to this harness.

## Boot quirk #3 (found + fixed): VNC display collision across concurrent VMs

vmctl's QEMU backend hardcoded `-vnc 127.0.0.1:0`, which binds VNC display
`0` (TCP port 5900) literally — it does not auto-search for a free port.
Bringing up a second VM (any second VM, not specific to this harness) while
one was already running failed outright:
`qemu-system-x86_64: -vnc 127.0.0.1:0: Failed to find an available port:
Address already in use`. This matters for `tests/vm/VMFile.kdl` specifically
because it defines three concurrent guests. Fixed by widening the VNC
option to a range, `127.0.0.1:0,to=99` (`crates/vm-manager/src/backends/qemu.rs`),
so QEMU auto-picks the first free display in 5900–5999; single-VM behavior
is unchanged (still binds 5900 when free).

## Boot quirk #4 (found, NOT yet root-caused): post-panic-fix stall before login

With both fixes above, a standalone smoke-test boot got past ZFS root
import cleanly but then produced no further console output for several
minutes after `WARNING: consconfig_prepare_dev: can't push streams module
"vuid3ps2", error 22` (a console-input/PS2-mouse-streams warning that looks
benign on its own). `qemu-system-x86_64`'s CPU usage during the stall was
low (~3–13%, not pegged), and the SLIRP-forwarded SSH port accepted a raw
TCP connection but the SSH banner exchange itself timed out — i.e. the
guest's network stack may be up, but sshd (or the multi-user milestone
generally) had not answered yet as of this writing. Not distinguished yet
between: (a) normal non-verbose-boot silence during SMF first-boot
manifest-import (which can genuinely take minutes, worse under host
contention — this box was shared with at least one other concurrent
`vmctl` invocation during this session, see below), (b) a DNS/network-
dependent SMF service timing out slowly, or (c) a second, distinct hang.
**Next step**: reboot with verbose boot (`boot -v` at the loader prompt, or
`boot-args="-v"` via a VMFile `cloud-init`/one-off `vmctl console`) to see
which SMF service the guest is stuck on, on a quiet host.

## Shared-host hazard (observed, not a harness bug): concurrent `vmctl` state races

This session's host had **another, independent `vmctl up --file
VMFile.kdl --name exec1` process running concurrently** (different PID,
different QEMU machine type — plain `pc`/i440fx, no UEFI — meaning a
different vmctl binary build than the one this session installed), which
is strong evidence multiple agent sessions share this machine and its
`~/.local/share/vmctl/` state. Concretely observed: `~/.local/share/vmctl/vms.json`
(the global VM registry) went from tracking this session's VM to `{}`
(empty) with no action by this session — a later write from the other
process raced and clobbered it, orphaning a still-running QEMU process
from vmctl's own tracking (it kept running fine; only `vmctl list`/`vmctl
destroy` lost track of it). **Do not assume `~/.cargo/bin/vmctl` or
`~/.local/share/vmctl/` are exclusively yours on this host** — verify with
`ps -ef | grep -E 'qemu|vmctl'` before destroying/recreating VMs by name,
and prefer unique VM names for throwaway experiments (this document's own
verification used `w1smoketest`, not `exec1`, for exactly this reason).

## Boot quirk #5 (found + fixed): `vmctl image pull oci://...` was broken

The imperative CLI (`vmctl image pull <url>`) always called
`ImageManager::pull` (the plain-HTTP path) regardless of URL scheme —
`pull_oci` was only ever reached through the declarative `VMFile.kdl`
`resolve()` path. `vmctl image pull "oci://code.aopc.cloud/..."` failed
immediately with "builder error for url". Fixed in
`crates/vmctl/src/commands/image.rs`: the pull subcommand now dispatches to
`pull_oci` when the URL has an `oci://` prefix, matching what `resolve()`
already did. `vmctl up`/`vmctl reload` (the declarative path) were never
affected by this bug — only the standalone `vmctl image pull` convenience
command was.

## Networking

All three guests use `network "user"` (QEMU SLIRP, no root/bridge setup
required). This is a real TCP/IP hop (not in-process), which is enough of a
"real network boundary" for the Phase-38 relocation/split-brain tests `pg`
exists for, and enough for `48c` russh host→`exec1` over real SSH — SLIRP's
`hostfwd` gives each guest a host-side port. `network "tap"` remains
available (see `docs/src/vmfile/network.md` in vm-manager) if a later test
needs a shared L2 between guests rather than each guest only reaching the
host.

## SSH / login user

The image's `sysconfig` first-boot provisioning reads the same
`#cloud-config` `users[0].name` that solstice-ci's own orchestrator uses:
`sol`. `vmctl`'s standard cloud-init block (auto-generated Ed25519 keypair,
no explicit `ssh-key`/`user-data`) produces a `#cloud-config` in exactly that
shape (see `build_cloud_config` in vm-manager's `cloudinit.rs`), so
`VMFile.kdl` uses `ssh { user "sol" }` on all three guests rather than
inventing a new convention this image has never been exercised against.
`48d` OIDC→SSH-cert auth is explicitly out of scope here (ADR 086 §1); this
is the "lab uses plain provisioned SSH keys" auth deviation, by design.

**Open verification item**: whether illumos `sysconfig`'s NoCloud parsing
actually honors the `groups: [sudo]` / `sudo: ALL=(ALL) NOPASSWD:ALL` /
`shell: /bin/bash` fields `build_cloud_config` emits (`sysconfig`'s own
source wasn't found published anywhere in `solstice-ci`, only the compiled
artifact reference) is **not yet confirmed** — the provisioner scripts
degrade gracefully (`pfexec` first, `sudo -n` second, loud failure third)
specifically so the first real `vmctl up` run surfaces which path actually
works, and that becomes the next fact to record here.

## `exec1` — executor target (48b/48c)

`tests/vm/scripts/provision-exec1.sh` creates the unprivileged `akh-exec`
user (`useradd -m -s /bin/bash`, password locked), copies the
`sol`-authorized SSH key so the same host-side private key drives both
`48c` (host → `exec1` as `sol` or `akh-exec`) and `48b` (in-guest RBAC/
privilege testing as `akh-exec`), and verifies/enables the `ssh` SMF
service. Confinement posture (`profiles`/`auths` on `akh-exec`) is dumped to
`provision.log` for the ADR 082 "OS as sole security boundary" audit trail —
no custom RBAC profile is assigned yet (plain default-user confinement);
tightening that is 48b's own work, not W1's.

## `pg` — postgres + pgvector

Verified against the live OpenIndiana Hipster catalog
(`pkg.openindiana.org/hipster/catalog/1/catalog.base.C`, 2026-07-05):

- `database/postgres-16`, `database/postgres-16/developer` (headers +
  `pg_config`, needed to build extensions), `service/database/postgres-16`
  (SMF manifest + method script) are real, installable packages.
- **No `pgvector` package exists in Hipster** (`vector`/`pgvector` don't
  appear anywhere in the catalog). `tests/vm/scripts/provision-postgres.sh`
  builds it from upstream source (`git clone
  https://github.com/pgvector/pgvector`, `make PG_CONFIG=... install`)
  against the postgres-16 dev headers — itself a small illumos
  system-engineering exercise in the spirit of the W3 task corpus.
- Binaries land at `/usr/postgres/16/bin/`; data dir is
  `/var/postgres/16/data` (owned `postgres:postgres`). The SMF FMRI for the
  postgres-16 service **was not confirmed from the package manifest alone**
  (the manifest XML file's contents weren't independently fetched), so the
  provisioner discovers it at runtime (`svcs -aH -o fmri | grep -i
  postgres`) rather than hardcoding a guess — record the real FMRI here
  once a live run confirms it.
- `pg_hba.conf`/`postgresql.conf` are set to accept `md5` auth from
  `0.0.0.0/0` and listen on all interfaces — **lab-only**, matching the
  "plain provisioned keys, not the 48d OIDC/cert story" posture ADR 086
  already accepts for this harness.

## `builder` — native illumos toolchain + portability shakeout

The image already ships the toolchain (`rustc`, `cargo`, `sccache`,
`build-essential`, `git`) so `provision-builder.sh` only confirms versions
and network egress to `code.aopc.cloud` (the akh-medu git remote). The full
"run `cargo test --features server` natively, triage the first illumos
failures, record a baseline pass list" work item is **not completed by this
change** — it's a multi-hour-to-multi-day effort against a 113K LOC
workspace that has never built on illumos, and is explicitly sized in the
plan as ongoing discovery work, not a single-session deliverable.
`tests/vm/scripts/run-native-tests.sh` is the concrete mechanism to drive it
(checkout + release.yml feature-set builds + full test suite, `--no-fail-fast`,
writing `~/akh-medu-native-baseline.txt`) — run it manually per the
instructions in its header and append the first findings to this document.

Sizing: `builder` gets 4 vCPU / 8 GB / 40 GB disk (above the 2 vCPU/4 GB
baseline `exec1`/`pg` use) since native compiles of this workspace
(candle, hnsw_rs, onnxruntime-adjacent deps, ...) need the headroom, and the
host has an 8-core/31GB budget to give it.

## Verified facts checklist (for the next person / next run)

- [x] `vmctl` binary present, patched (uefi node, virtio-transitional node,
      VNC port-range fix, oci pull dispatch), rebuilt, reinstalled to
      `~/.cargo/bin/vmctl`.
- [x] KVM (`/dev/kvm`), `qemu-system-x86_64`, `qemu-img`, OVMF firmware
      (`/usr/share/OVMF/OVMF_{CODE,VARS}*.fd`) all present on this host.
- [x] `openindiana-rust-ci` OCI image pulled and cached
      (~13.75 GiB, sha256 `664af2f44d5bc0765f675ae585624b8fad3a790411554ffd6157c6739f1c4334`).
- [x] `database/postgres-16` + `service/database/postgres-16` package names
      confirmed against the live Hipster catalog; `pgvector` confirmed
      **not** packaged (source build required).
- [x] The fatal `vfs_mountroot` panic is fixed (`virtio-transitional #true`)
      — confirmed via a standalone `w1smoketest` VM getting cleanly past
      ZFS root import on the first attempt.
- [ ] First full boot to an SSH-reachable login confirmed end-to-end. As of
      this writing the `w1smoketest` guest's SLIRP-forwarded SSH port
      accepts a raw TCP connection but the SSH banner exchange itself times
      out, several minutes after the last console output
      (`consconfig_prepare_dev: ... vuid3ps2, error 22`) — see boot quirk
      #4. Retry on a quiet host with verbose boot and append the result.
- [ ] `sysconfig`'s actual support for the `users[].sudo`/`groups` cloud-config
      fields confirmed one way or the other.
- [ ] postgres-16 SMF FMRI confirmed from a live `svcs -a` listing.
- [ ] First native `cargo test --features server` pass/fail baseline on
      illumos recorded (W1's largest remaining item; tracked in the plan,
      not blocking the rest of W1's deliverable).

---

## UPDATE (2026-07-05, consolidated sole W1 agent) — SSH blocker ROOT-CAUSED + fix reconciliation

This runbook above was authored by a concurrent W1 agent (now retired). The
two sessions independently found the same boot quirks. This section records
what the surviving agent added, and supersedes the two open SSH items.

### Boot quirk #4 is now ROOT-CAUSED: it's a provisioning-format mismatch, not a stall

The earlier "post-panic-fix stall before login (sshd never answers)" is NOT
a timing/contention stall. With a fully instrumented manual boot (i440fx via
the q35→pc shim, `-vga std` + QMP `screendump` for console visibility, a
`virtio-net-pci` + SLIRP `hostfwd`, and a NoCloud `cidata` seed carrying a
standard cloud-init `#cloud-config`), the guest boots to multi-user and
**sshd DOES answer** — `SSH-2.0-OpenSSH_9.9` on the forwarded port. What
fails is the *login*: no key is accepted for any user (`sol`, `vmctl`,
`root`, …). The serial console (the image's kernel console is `ttya`, so the
full agent log lands there — the VGA framebuffer only shows the loader)
explains exactly why:

```
INFO searching for cidata device with metadata...
DEBG found cidata device /dev/dsk/c1t0d0p0 of type hsfs
INFO mounting cidata device /dev/dsk/c1t0d0p0 to /var/metadata/files
INFO WRITE NODENAME "oiman"
INFO ----- WRITE FILE: /etc/nodename ------
INFO ----- WRITE FILE: /etc/inet/hosts ------
INFO found these ethernet interfaces: ["vioif0"]
INFO ENSURE IPv4 DHCP INTERFACE: vioif0
INFO ok, interface vioif0 address 10.0.2.15/24 (dhcp) complete
INFO ok, run complete
Hostname: oiman
oiman console login:  ... svc:/system/illumos/useragent:default
```

**OpenIndiana Hipster's cloud image provisions via
`svc:/system/illumos/useragent`, a minimal illumos-native metadata agent —
NOT cloud-init.** It reads the NoCloud `cidata` seed's **meta-data** and does
exactly three things: sets `/etc/nodename` + `/etc/inet/hosts` from
`local-hostname`, and brings up DHCP on the first NIC. Then "ok, run
complete". It **never reads the cloud-init `#cloud-config` user-data** at all
— there is no user-creation, no `ssh_authorized_keys` injection, no `sudo`
step in its run. So `vmctl`'s `build_cloud_config` output (a Debian-shaped
`#cloud-config` with `users:` / `ssh_authorized_keys:` / `groups:[sudo]`) is
simply ignored on this image, and no login is ever created.

The `openindiana-rust-ci` image is worse: it uses `svc:/system/sysconfig`
instead, whose `sysconfig provision --source auto` method is **SIGKILLed**
(SMF method timeout) and transitions to maintenance — again no user, plus
`network/ipmp` left unconfigured. (Screendumped from its VGA console:
`system/sysconfig:default failed: transitioned to maintenance`.)

**Net:** neither published OpenIndiana image creates a login user from
vmctl's cloud-config user-data. sshd comes up; you just can't authenticate.

### What this means for W1

- The three-VM `vmctl up` + SSH `smoke.sh` **cannot go green as-is** — no
  SSH login exists on either image via vmctl's seeding.
- The `builder` native-build **portability shakeout is blocked too** (it
  needs to rsync + build over SSH into the guest). It could still be driven
  over the **serial console** if a console login exists (default root
  password unknown — not attempted), but that's not automatable this session.

### Fix options (for whoever continues W1 — none yet implemented/verified)

1. **Match `useragent`'s expected key channel.** OI's metadata agent likely
   injects SSH keys from a specific meta-data field (SmartOS/Joyent lineage
   → e.g. `root_authorized_keys` / `public-keys`), into a specific default
   user — NOT from cloud-config. The agent's own source wasn't found
   published; determine the field it reads (from the illumos `userland`
   metadata-agent source or by experiment) and have vmctl (or a custom
   `user-data`/`meta-data` VMFile override) emit that instead of
   cloud-config. This is the cleanest path and keeps OpenIndiana per ADR 086.
2. **Bake the login into the image.** Rebuild `openindiana-rust-ci` (or a new
   lighter OI base) with a pre-created `sol`/`akh-exec` user + authorized_key,
   or a working sysconfig/useragent profile — an image-build change, out of
   W1's "pull prebuilt" scope but the durable fix.
3. **Fall back to `omnios-bloody`** (already cached; catalog lists
   `ssh_user: root`, `nocloud: true`). OmniOS's metadata handling may inject
   `root`'s key from meta-data. Deviates from ADR 086's OpenIndiana choice
   (note it), but would unblock the executor/pg/relocation substrate fastest.

### Virtio panic-fix reconciliation: shim vs source

Both agents fixed the illumos `vioblk` q35 panic; two routes exist:

- **This branch ships a self-contained PATH shim**
  (`tests/vm/scripts/qemu-illumos-shim/qemu-system-x86_64`) that rewrites
  vmctl's `-machine q35,accel=kvm` → `-machine pc,accel=kvm` (i440fx,
  conventional PCI). Proven: same image boots past the panic, ZFS root
  imports. No dependency on any uncommitted external change; `smoke.sh`
  prepends it to `PATH`.
- **The sibling `vm-manager` repo has an uncommitted source fix** (the
  retired agent's): a `virtio-transitional #true` VMFile node appending
  `disable-modern=on,disable-legacy=off` to the virtio `-device`s (legacy
  virtio-pci on q35), plus a `uefi #true` node, a VNC port-range fix for
  concurrent VMs, and an `oci://` dispatch fix for `vmctl image pull`. The
  installed `~/.cargo/bin/vmctl` was rebuilt from these (mtime 22:58), so it
  already understands those nodes. **These live in the USER's separate repo
  (`/home/toasty/ws/nebula/vm-manager`) and are left UNCOMMITTED for the
  user's own review — do not push.** If/when they land, `tests/vm/VMFile.kdl`
  can drop the shim and set `virtio-transitional #true` (+ `uefi #true` if a
  given image needs it) instead. Both routes solve the same panic; the shim
  is what this branch depends on so it stands alone.

  **UPDATE:** the user has since pulled newer upstream `vm-manager` +
  `refraction-forger`. Upstream now already carries UEFI + the VNC-port fix +
  an OS-ephemeral SSH port + OCI auth, so the retired agent's hand-rolled
  UEFI/VNC/OCI edits are redundant — but the pull collided with those
  uncommitted edits and left `/home/toasty/ws/nebula/vm-manager` in an
  **unresolved merge conflict** (`UU crates/vm-manager/src/backends/qemu.rs`).
  That repo won't build until the user resolves it, and it is **entirely the
  user's to handle — do not touch/build/commit/reset it.** The
  already-installed `~/.cargo/bin/vmctl` binary still runs (conflict is only
  in source), so the harness keeps using it + the self-contained shim. The
  `virtio_transitional` field is **not merged upstream yet** — it remains a
  needed upstream fix, but it is the user's to land. `tests/vm` deliberately
  does NOT depend on it; the shim is the standalone path.

### Verified-facts delta (this agent)

- [x] 14.7 GiB `openindiana-rust-ci` OCI image pulled + **sha256-verified**
      (`664af2…c4334`) via a streaming curl against the registry's anonymous
      OCI bearer-token endpoint — the `vmctl` OCI pull path buffers the whole
      layer in RAM (`oci_client::pull()` → `Vec<u8>`) and OOMs on a 14.7 GiB
      layer, so it must NOT be used for this image (a concurrent `vmctl image
      pull` ballooned to 5.6 GiB RSS and had to be killed to save the box).
- [x] `openindiana-hipster` general cloud image (`OI-hipster-cloudimage.img.zstd`,
      ~747 MiB compressed → 1.95 GiB raw) pulled + decompressed; the HTTP+zstd
      pull path **streams** (no OOM). This is the NoCloud-capable OI image the
      solstice catalog distinguishes from the CI-runner image.
- [x] i440fx (`-machine pc`) gets **both** images past the `vfs_mountroot`
      vioblk panic (independent confirmation of quirk #2's fix, via a
      different route than virtio-transitional).
- [x] **sshd comes up on OI-hipster** (banner answers) — quirk #4's "sshd
      never answers" was image/agent-specific; the real wall is the
      user-data provisioning-format mismatch above.
- [ ] SSH login into any guest — **BLOCKED** (root-caused above). Blocks
      `smoke.sh` green + the native shakeout this session.

---

## UPDATE (2026-07-06) — committed VMFile.kdl regression fixed; console-visibility artifact resolved; sysconfig failure re-characterized with hard evidence

Picked this up fresh from `develop` (worktree reset to `2a2949b`, the W1
merge commit). This section records what changed and what is now know with
much higher confidence than the sections above.

### Regression found: the committed `VMFile.kdl` never actually set `uefi`/`virtio-transitional`

The "UPDATE (2026-07-05)" section above says the installed `~/.cargo/bin/vmctl`
was rebuilt with `uefi` and `virtio-transitional` VMFile nodes (confirmed
again this session via `strings ~/.cargo/bin/vmctl | grep -i uefi` — both
flags are compiled in, with a real `create --uefi`/`--virtio-transitional`
CLI surface and a `VMFile.kdl` parser that accepts `uefi #true` /
`virtio-transitional #true` bool nodes on a `vm` block). **But the
`tests/vm/VMFile.kdl` actually committed to `develop` never set either
node** — only the PATH shim (`-machine q35` → `pc` rewrite) was wired in.
Reproduced from a clean worktree: `vmctl up --file VMFile.kdl --name exec1`
with the shim on `PATH` produces a QEMU invocation with **no OVMF pflash
drives at all** (`-machine pc,accel=kvm ... -drive ...overlay.qcow2 -device
virtio-blk-pci,drive=drive0` — no `disable-modern`, no `if=pflash`). Since
`openindiana-rust-ci` is GPT/UEFI-only (see quirk #1's original finding, real
despite the earlier "not a blocker" note in the older shakeout doc — that
note was about a *different*, older image build), this boots to a bare
SeaBIOS spinner with **zero further console output ever** (confirmed: 8
bytes of ANSI cursor-reposition codes in `console.log`, no progress after
minutes). And even with `uefi #true` alone (no `virtio-transitional`), the
guest reaches the OpenIndiana loader fine but then hits the **exact quirk-#2
vioblk panic again**:

```
WARNING: vioblk0: Failed to map CAP 2 @ BAR4 0x3000+1000
...
panic[cpu0]/thread=...: vfs_mountroot: cannot mount root
```

**Fix applied:** added `uefi #true` and `virtio-transitional #true` to all
three `vm` blocks in `tests/vm/VMFile.kdl` (matching the parser test shapes
in `vm-manager/src/vmfile.rs`). Also bumped `builder` to `vcpus 4` / `memory
8192` per this doc's own sizing note (§"`builder` — native illumos
toolchain..."), which the committed file also hadn't done. With both flags
set, `vmctl up` produces `-drive if=pflash,...OVMF_CODE.fd -drive
if=pflash,...efivars.fd` plus `disable-modern=on,disable-legacy=off` on both
virtio-blk-pci and virtio-net-pci, and the guest gets past the panic on the
first try (`ZFS is trying an alternate path
(/pci@0,0/pci1af4,2@3/blkdev@0,0:b)` — pool imports).

### The "several minutes of silence" from quirk #4 is a console-routing artifact, NOT a hang

This was the single most time-consuming thing to pin down this session, and
it's worth recording precisely so nobody re-diagnoses it as a timing/host-
contention bug again. `vmctl`'s QEMU backend passes `-nodefaults` and never
adds a `-vga`/display device (confirmed: `screendump` over QMP against a
running vmctl-launched guest fails with `"There is no console to take a
screendump from"`). OpenIndiana's kernel, once it takes over from the
loader, picks its console from whichever devices are actually present — with
no VGA device, it should fall back to `ttya` (serial), and early on it does
(the loader banner, `Welcome to illumos`, the `NOTICE: Performing full ZFS
device scan!` lines, and the `vuid3ps2` STREAMS warning all land in
`console.log` fine). But **after that point, on this image, `console.log`
goes completely silent for 10+ minutes** with the qemu process sitting at
low, non-pegged CPU (~5-17%) — which looks exactly like a hung guest, and is
what the 2026-07-05 "boot quirk #4" section left as an open question.

It is not a hang. Booting the *same* overlay+seed manually with `-vga std
-vnc 127.0.0.1:0,to=99` added (bypassing vmctl, reusing its overlay.qcow2/
seed.iso/efivars.fd files directly) and reading the framebuffer via QMP
`screendump` shows the **entire rest of boot happening normally, in well
under a minute**, all on the `text` (VGA) console — `Configuring devices.`,
SMF service imports, the `system/sysconfig:default failed` / `network/ipmp:
default failed fatally` messages, and finally an `unknown console login:`
prompt. None of this ever reaches serial. **The guest was never stuck; the
console was just pointed somewhere vmctl doesn't capture.** (Note: adding
`-vga std` also changes OVMF's own boot behavior — with a GOP-capable
display present, OVMF drops into its interactive `UiApp`/Boot Manager
front page or a `Shell>` prompt on cold NVRAM instead of silently
autobooting the ESP's default loader the way it does headless. Worked
around by connecting to the serial `chardev` socket directly — the *loader*
stage, unlike the running kernel, still listens on both ttya and the
keyboard — and typing `FS0:` then `EFI\BOOT\bootx64.efi` to chainload
manually. Once the illumos kernel takes over, further input has to go
through QMP `send-key`, since the login prompt reads the PS/2 keyboard, not
the serial UART.) **Practical upshot: don't trust "no vmctl `console.log`
output" as a liveness signal for these images once boot passes the loader —
use SSH reachability, or a `-vga std` + `screendump` side-channel, instead.**

### `svc:/system/sysconfig:default` failure re-characterized: fast repeated failure, not a 120s SIGKILL

With the console now actually visible, the failure is:

```
Jul  6 06:25:33 svc.startd[9]: network/ipmp:default: Method
  "/lib/svc/method/net-ipmp" failed with exit status 95.
Jul  6 06:25:33 svc.startd[9]: network/ipmp:default failed fatally:
  transitioned to maintenance
...
Jul  6 06:25:35 svc.startd[9]: system/sysconfig:default failed:
  transitioned to maintenance
unknown console login:
SUNW-MSG-ID: SMF-8000-YX ... network/ipmp:default ... "a method is failing
  in a retryable manner but too often" ...
SUNW-MSG-ID: SMF-8000-YX ... system/sysconfig:default ... same ...
```

This reached the login prompt in **under 15 real seconds** from the loader
handoff — not the ~120s SMF `exec_method timeout_seconds` the 2026-07-05
section assumed (that number is real, from `service-configs/smf/
sysconfig.xml` in `~/ws/nebula/sysconfig`, but this run never hit it: the
method fails and exits on its own well before the timeout, repeatedly, and
SMF's own maintenance-transition policy (a handful of quick retries then
give up) is what actually ends it). `network/ipmp` failing *first*, fatally,
is suspicious as a possible trigger/co-factor for `sysconfig`'s own failure
(sysconfig's SMF manifest only has an *optional* dependency on
`network-physical`, not `ipmp`, so this isn't an explicit dependency chain,
but a network-configuration step inside `sysconfig apply` itself could well
be tripping over the same missing IPMP group setup) — flagged as the next
concrete thing to check from inside a shell, not yet confirmed.

**Read `~/ws/nebula/sysconfig`'s own source this session** (read-only,
per instructions) to understand exactly what's being attempted:
`crates/sysconfig-providers/src/cloud_init.rs` scans `/dev/dsk/c*{s2,p0}`
via `fstyp`, mounts whatever has an `hsfs` filesystem and a `CIDATA`/`cidata`
volume label, and parses a real subset of `#cloud-config` (`hostname`,
`ssh_authorized_keys` → root, and a `users:` list with `name`/`shell`/
`sudo`/`ssh_authorized_keys`/`groups` — confirmed by the crate's own unit
tests, e.g. `parse_cloud_config_per_user_ssh_keys`). That format is exactly
what vmctl's `cloud-init` block produces, so the seed *format* was probably
never the problem. `crates/sysconfig-providers/src/lib.rs`'s `auto_detect()`
tries `smartos` (checks for `/usr/sbin/mdata-get`), `ec2` (SMBIOS-gated, its
HTTP fetch is a stub that always returns `NotAvailable` in this build — not
a hang risk), `digitalocean`, then `cloud-init`, then `generic_cpio` — so if
`mdata-get` happens to exist on this image and its `fetch()` blocks on a
metadata channel that doesn't exist under plain QEMU/SLIRP, it would run
*before* cloud-init ever gets a chance and could produce exactly this
"failing repeatedly" signature. **Not confirmed** — would need a shell
inside the guest (`/var/svc/log/system-sysconfig:default.log` has the real
answer) to settle definitively; single-user boot / root-password login
attempts this session did not succeed (root has no working/blank
password on this image either).

### omnios-sysconfig fallback: tried, not a quick win either

Per the task's documented fallback option, pulled
`oci://code.aopc.cloud/cloudnebulaproject/omnios-sysconfig:latest` (small —
OCI in-RAM pull peaked around ~1.3 GiB RSS, fine memory-wise) with a custom
`user-data` (`akh-exec` user + root SSH key, matching the confirmed sysconfig
parser shape) via a standalone throwaway VMFile (not committed — this was a
quick side-experiment, not part of the harness). Result: boots cleanly,
reaches an `console login:` prompt within ~35s (no visible SMF failures this
time, unlike rust-ci), but **no SSH banner ever answered in ~2 minutes of
polling** the forwarded port. Inconclusive whether sysconfig actually
succeeded there (hostname still showed `unknown` at the login banner, which
may just be a pre-provisioning artifact rather than a failure signal) —
this needs its own dedicated investigation and is **not** a proven
drop-in fix; do not assume it "just works" without further verification.

### Net status for the next session

- `tests/vm/VMFile.kdl`'s `uefi`/`virtio-transitional` gap is fixed —
  `vmctl up` now reliably reaches the same failure point the 2026-07-05
  manual investigation found, automatically, for all three guests.
- `smoke.sh` still cannot go green: no guest is SSH-reachable. This remains
  the accurate, current, root-caused status.
- The `builder` native `cargo build`/`cargo test` shakeout is still
  blocked on SSH and was **not run** this session (would be fabricating
  results to claim otherwise).
- Most valuable next step: get a **shell** inside a booted guest (single-user
  boot, or a console with a known root password baked at image-build time)
  and read `/var/svc/log/system-sysconfig:default.log` directly — that
  settles the smartos/mdata-get hypothesis (or rules it out) with certainty
  instead of inference from FMA messages.
