# Phase 50 / W1 — illumos portability shakeout (baseline)

> Date: 2026-07-05
> Status: In progress — VM substrate up, native build/test run pending/partial
> Plan: [2026-07-05-phase50-feature-test-benchmarks.md](plans/2026-07-05-phase50-feature-test-benchmarks.md) (W1)
> ADR: [086 — Feature-test VM harness + benchmark program](decisions/086-feature-test-vm-harness-benchmarks.md)
> Harness: `tests/vm/VMFile.kdl` (`builder` guest), `tests/vm/smoke.sh`

## Purpose

akh-medu's `illumos-pkg` release job (`.forgejo/workflows/release.yml`) has
never actually run — there is no illumos runner registered. This is the
first real attempt at building and testing the release feature sets
natively on illumos (OpenIndiana), inside the Phase-50 `builder` VM. The
goal is an honest baseline of what breaks, categorized, not a fix-everything
pass.

Commands under test (mirrors `release.yml`'s `illumos-pkg` job, plus the
native test suite ADR 086 asks for that release.yml does not run):

```bash
export CARGO_BUILD_JOBS=<reduced-for-4GB-RAM>   # see below
cargo build --release --no-default-features --features client --bin akh
cargo build --release --features server --bin akhomed
cargo build --release -p seshat --bin seshd --no-default-features --features synthesis
cargo test --features server
```

## vmctl quirks discovered (host-side, before any guest code ran)

These aren't akh-medu bugs, but they blocked the substrate and are worth
recording for whoever touches `tests/vm/VMFile.kdl` next (vmctl lives at
`~/.cargo/bin/vmctl`, source at `~/ws/nebula/vm-manager` on this box).

1. **`vmctl image pull oci://...` and `vmctl create --image-url oci://...`
   never dispatch to the OCI client.** `crates/vmctl/src/commands/image.rs`
   (`ImageAction::Pull`) and `crates/vmctl/src/commands/create.rs` both call
   `vm_manager::image::ImageManager::pull(url, name)` unconditionally, which
   does a plain `reqwest` GET — it doesn't check for the `oci://` scheme at
   all. Result: `Error: failed to download image ...: builder error for url
   (oci://...)` (reqwest chokes building a request for a non-http(s)
   scheme). Only `VMFile.kdl`'s `image-url "oci://..."` field, resolved by
   `vmfile::resolve()` → `ImageManager::pull_oci()`, takes the correct path
   (`crates/vm-manager/src/vmfile.rs:481-483`). **Workaround: none needed
   for the imperative commands — just don't use them for `oci://` refs; use
   a VMFile.**

2. **`pull_oci` buffers the entire OCI layer in process memory before
   writing to disk.** `crates/vm-manager/src/oci.rs::pull_qcow2()` calls
   `oci_client::Client::pull()`, which returns a fully materialized
   `ImageData { layers: Vec<ImageLayer> }` — each layer's bytes held in a
   `Vec<u8>` — and only *then* does `image.rs::pull_oci()` `tokio::fs::write`
   it out. For the 14.7GB `openindiana-rust-ci` layer this means ~14.7GB of
   process RSS before a single byte hits disk. Confirmed empirically: RSS
   grew ~2GB in the first 90s of a live `vmctl up` run (`45.6% CPU, 1.97GB
   RSS` at t+90s per `ps`); this host had only ~8.8GB "available" at the
   time (shared with other concurrent agent sessions — not `vmctl`'s
   fault, but it makes the bug load-bearing here). Killed the run before it
   could OOM the box. **Workaround applied: pre-pulled the blob by hand**
   with a streaming `curl` against the registry's anonymous OCI
   bearer-token endpoint (`code.aopc.cloud`'s `/v2/token` issues anonymous
   pull tokens — no `GITHUB_TOKEN`-equivalent credential needed for this
   registry), writing directly to
   `~/.local/share/vmctl/images/openindiana-rust-ci.qcow2`, then referenced
   it in `VMFile.kdl` via the local `image` field instead of `image-url
   oci://`. The exact commands are in a comment at the top of
   `tests/vm/VMFile.kdl`. **This is a real vmctl bug worth fixing upstream**
   (stream the layer to a temp file during pull instead of materializing it)
   — flagging for whoever owns `vm-manager` next; not fixed here since it's
   a different project/repo than akh-medu.

3. **OCI references must be lowercase.** `oci://code.aopc.cloud/CloudNebulaProject/...`
   fails `Reference` parsing (`invalid reference format` — Docker reference
   spec requires a lowercase repository path); `cloudnebulaproject` (all
   lowercase) parses fine. The registry's own package-listing API and `/v2/`
   endpoints are also case-normalized to lowercase, so this matches what the
   registry itself expects.

4. **VMFile.kdl has no `uefi` field — but that is NOT a blocker here.**
   `vmfile.rs` hardcodes `uefi: false` when converting a parsed `VmDef` to a
   `VmSpec` (only the imperative `vmctl create --uefi` flag can request
   UEFI). The `refraction-forger` *build* target
   (`images/omnios-bloody-disk.kdl`) sets `bootloader "uefi"`, which raised
   the question of whether the published cloud images need UEFI to boot.
   Resolved by reading the solstice orchestrator's own **image-catalog**
   (pulled from the `image-catalog` OCI artifact in the same registry): it
   boots *every* illumos image — `omnios-bloody` and `openindiana-hipster` —
   with **legacy BIOS** (no `uefi: true` field; the catalog documents a
   `uefi:` per-image field and simply doesn't set it for the illumos rows).
   So vmctl's BIOS default is the intended boot path for these OpenIndiana
   cloud images, and the missing VMFile `uefi` field does not block W1.
   *(Actual boot result recorded in the "boot quirks" section below once the
   fleet comes up.)*

   Side note: that same catalog **independently confirms the in-memory OCI
   buffering bug** from quirk #2 above, from the *push* side: "Forger reads
   the entire qcow2 into memory for OCI push, and the multipart-form body
   buffers a second copy — for our 4.4 GiB image that's ~9–10 GB peak
   resident. 8 GB OOM-killed forger silently on run 6." The pull side
   (`oci_client::pull()` → `Vec<u8>`) has the same shape. Both are known,
   both want streaming; flag upstream.

## OpenIndiana-under-KVM boot + provisioning quirks

**Boot: confirmed working under legacy BIOS.** `vmctl up --name exec1`
(QEMU q35 + `-cpu host` + SeaBIOS, no OVMF) booted the
`openindiana-rust-ci` qcow2 cleanly — `QEMU: started status=running` — then
proceeded to provisioning. This settles quirk #4: the OpenIndiana cloud
image does **not** need UEFI, so vmctl's hardcoded `uefi: false` is fine.
The image reports `virtual size: 24 GiB` (qcow2, zlib-compressed, 7.27 GiB
on disk) via `qemu-img info`, so no `disk` resize is needed in VMFile.kdl
for a base checkout+build (the release build tree fits in 24 GiB; bump
`disk` later if the in-VM cargo target dir + image outputs need more).

### ★ BLOCKER (root-caused + fixed): vmctl's `-machine q35` panics illumos

`vmctl up` boots exec1 fine to the OpenIndiana loader, but the kernel then
**panics** and vmctl's provisioner fails with `Failed getting banner`
(sshd never comes up because the guest never finishes booting). The serial
console (`~/.local/share/vmctl/vms/exec1/console.log`) shows the real cause:

```
WARNING: vioblk0: Failed to map CAP 2 @ BAR4 0x3000+1000
WARNING: vioblk0: failed to start Virtio init
WARNING: vioif0: Failed to map CAP 2 @ BAR4 0x3000+1000
NOTICE: Cannot read the pool label from '/pseudo/lofi@2:b'
NOTICE: spa_import_rootpool: error 5
Cannot mount root on /pseudo/lofi@2:b fstype zfs
panic[cpu0]/thread=...: vfs_mountroot: cannot mount root
```

**Root cause:** vmctl's QEMU backend hardcodes `-machine q35,accel=kvm`
(`crates/vm-manager/src/backends/qemu.rs`). q35 is a PCIe machine; illumos's
`vioblk`/`vioif` drivers fail to map the *modern* virtio PCI capability
(CAP 2) on the PCIe root complex, so the virtio-blk root disk never
initializes and the root ZFS pool can't mount.

**Fix (proven):** use the **i440fx** machine (`-machine pc`) — conventional
PCI, where illumos maps the virtio caps correctly. This is exactly what the
proven-working `refraction-forger` libvirt backend uses
(`<type ... machine="pc">`, `crates/vm-manager/src/backends/libvirt.rs:111`).
Booting the same image manually with `qemu-system-x86_64 -machine pc ...`
(otherwise identical) gets past the panic — the serial log then shows:

```
OpenIndiana Hipster 2026.04 Version illumos-44e9e35ce0 64-bit
NOTICE: Performing full ZFS device scan!
NOTICE: Original /devices path (/pseudo/lofi@2:b) not available; ZFS is
  trying an alternate path (/pci@0,0/pci1af4,2@4/blkdev@0,0:b)
```

i.e. vioblk maps cleanly (`blkdev@0,0`) and ZFS imports the root pool.

**What vmctl needs (one-line-ish upstream change):** a machine-type knob.
Either (a) a `machine "pc"` field on the `vm` block in VMFile.kdl plumbed
into `VmSpec`, or (b) default to `pc` (q35 buys nothing for these
workloads), or (c) auto-select `pc` for illumos/BSD images. Until then, W1
ships a **non-invasive PATH shim**
(`tests/vm/scripts/qemu-illumos-shim/qemu-system-x86_64`) that rewrites
`-machine q35,accel=kvm` → `-machine pc,accel=kvm` and execs the real qemu;
prepend it to `PATH` when running vmctl against these guests (smoke.sh does
this automatically). It does not touch the shared `~/.cargo/bin/vmctl` or
system qemu.

**Console visibility note:** the image directs its console to VGA, not the
serial ttya, after early boot — so vmctl's serial `console.log` captures
only the loader + any kernel panic (panics write to every console), and
normal multi-user boot output is invisible on serial. That's why the first
diagnosis was hard: the serial log looked empty on a healthy boot. To watch
a normal boot, screendump the VGA via QMP (`screendump`) on a VM started
with a VGA device, or rely on SSH reachability as the liveness signal.

### ★ SECOND BLOCKER: `openindiana-rust-ci` image's first-boot provisioning fails

With the `-machine pc` fix the guest boots to multi-user, but SSH is still
unreachable and vmctl provisioning still fails. QMP `screendump` of the VGA
console (the only way to see past the serial cutover) shows the real
reason:

```
svc:/system/sysconfig:default: Method "/usr/lib/sysconfig/sysconfig
  provision --source auto" failed due to signal KILL
svc.startd: system/sysconfig:default failed: transitioned to maintenance
...
unknown console login:
IMPACT: svc:/network/ipmp:default is unavailable.
IMPACT: svc:/system/sysconfig:default is unavailable.
PLATFORM: Standard-PC-(i440FX-+-PIIX,-1996)   <- confirms the pc fix is live
```

**Root cause:** `openindiana-rust-ci` is a **CI-runner image** (built by the
solstice pipeline to host the solstice runner agent), not a general-purpose
cloud image. Its first-boot provisioning service `svc:/system/sysconfig`
runs OpenIndiana's native `sysconfig provision --source auto`, which is
**killed by an SMF method timeout** — it doesn't consume vmctl's cloud-init
NoCloud `cidata` seed (illumos `sysconfig` reads an SC-profile, and/or the
runner image expects the solstice runner's own metadata channel, not
cloud-config YAML). Because that service dies: no login user is created,
`network/ipmp` never configures, hostname stays `unknown`, and sshd is
unreachable. State is terminal (identical console after 45s+).

This is independent of vmctl and of the q35 fix — it's an
**image/provisioning-format mismatch**. Consequences:
- vmctl's SSH-based provisioners (and `tests/vm/smoke.sh`) can't run against
  this image: there's no SSH user and no network.
- The image DOES boot and run illumos (so the `builder` native-build
  shakeout could still be done via console or a pre-baked user), but
  unattended cloud-init provisioning is out.

**Fix / path forward (recommended):** use a **general-purpose OpenIndiana
cloud image** that supports NoCloud cloud-init, not the CI-runner image. The
solstice orchestrator's own image-catalog lists exactly this distinction —
its OpenIndiana entry is `openindiana-hipster`:
`https://dlc.openindiana.org/isos/hipster/<date>/OI-hipster-cloudimage.img.zstd`
with `nocloud: true`. That image (≈747 MB compressed, HTTP+zstd — and the
HTTP pull path *streams* to disk, so it sidesteps the OCI in-RAM-buffering
bug too) is the right base for exec1/pg/builder. W1 attempts this image
next; if cloud-init provisioning succeeds on it, point `VMFile.kdl`'s three
`vm` blocks at it (local `image "…/openindiana-hipster-cloud.img"`) and the
smoke test should run end-to-end. If it also fails, the fallback is the
already-cached, catalog-blessed `omnios-bloody` image (`ssh_user: root`,
`nocloud: true`) — OmniOS rather than OpenIndiana, a deviation from ADR 086
to note but a working illumos substrate.

*(Result of the OI-hipster-cloudimage attempt appended below.)*

**Anticipated cloud-init/RBAC risk (watch for it):** vmctl's auto-generated
cloud-config (`crates/vm-manager/src/cloudinit.rs::build_cloud_config`)
creates the SSH user with `groups: [sudo]` + `sudo: ALL=(ALL) NOPASSWD:ALL`
and `shell: /bin/bash`. On illumos:
- there is no `sudo` group by default (it's a Linux/Debian convention);
  cloud-init typically logs a warning and proceeds, but if the OpenIndiana
  cloud-init port treats a missing group as fatal, user creation could fail.
- `/bin/bash` exists on OpenIndiana (unlike some minimal illumos images),
  so the shell should be fine.
- the `sudo:` directive writes `/etc/sudoers.d/90-cloud-init-users`, which
  works regardless of group membership **as long as the `sudo` binary is
  installed** — `openindiana-rust-ci` is a CI image so it very likely has
  it, but every provisioner script here (`exec1-`, `pg-`, `builder-`) leans
  on `sudo`, so if sudo is absent or cloud-init didn't grant it, ALL three
  provisioners fail at the first `sudo` call. If that happens the fix is to
  add a `user-data` block to each VM in VMFile.kdl with an
  illumos-appropriate admin grant (RBAC `Primary Administrator` profile via
  `profiles`, or `root` login) instead of relying on the Debian-shaped
  auto-config. Recorded here so the next runner recognizes the signature
  fast rather than re-diagnosing it.

## Native build results

*(Pending — `builder` VM not yet provisioned/built against at time of
writing.)*

| Step | Result | Notes |
|---|---|---|
| `cargo build --release --no-default-features --features client --bin akh` | pending | |
| `cargo build --release --features server --bin akhomed` | pending | |
| `cargo build --release -p seshat --bin seshd --no-default-features --features synthesis` | pending | |
| `cargo test --features server` | pending | |

## Failure categories (framework for triage)

Per the task brief, categorize first failures — do not attempt to fix all
of them:

- **Paths**: hardcoded `/proc`, `/sys`, Linux-specific FHS assumptions
- **libc**: GNU libc extensions not in illumos libc, `#[cfg(target_os =
  "linux")]` gates that should also cover illumos or vice versa
- **redb mmap**: `redb`'s mmap-backed storage relying on Linux-specific
  mmap flags/behavior (`MAP_POPULATE`, `madvise` flags, etc.)
- **socket semantics**: `SO_REUSEPORT`, abstract Unix sockets (Linux-only,
  no illumos equivalent), epoll-vs-event-ports assumptions leaking through
  a dependency
- **missing syscalls**: anything gated on `libc::` calls illumos doesn't
  implement (e.g. `memfd_create`, `pidfd_*`, Linux-only `clone` flags)

## Next steps

1. Finish bringing up `exec1`/`pg`/`builder` via `tests/vm/smoke.sh` and
   record the smoke-test pass/fail per VM.
2. Checkout/rsync the akh-medu worktree into `builder`, run the build
   sequence above with `CARGO_BUILD_JOBS` reduced to fit 4GB RAM, capture
   the first compile failure (if any) verbatim.
3. If it compiles, run `cargo test --features server`, capture the first
   test failures verbatim, categorize per the framework above.
4. Do **not** attempt to fix illumos portability issues as part of W1 — the
   deliverable is the substrate + this baseline, not a green illumos build.
