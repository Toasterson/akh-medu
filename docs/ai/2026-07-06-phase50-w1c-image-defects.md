# Phase 50 / W1c — OpenIndiana `exec1` SSH blocker: ROOT-CAUSED (image-build defects)

*2026-07-06. ADR 086 (feature-test VM harness). Supersedes the "seed / login
provisioning" framing in the earlier W1 shakeout notes and the VMFile status
comment.*

## TL;DR

The `openindiana-rust-ci` OI Hipster 2026.04 image published in the
CloudNebulaProject registry is a **broken build**. It boots to multi-user
under vmctl (after the W1b `uefi`/`virtio-transitional` fix + the q35→pc
shim) but **never becomes SSH-reachable**, because of **three independent,
fatal image-build defects**. This is NOT a cloud-init seed / user-data /
provisioner-format problem — every earlier session had that framing wrong.

Method that settled it: instead of fighting the VGA-silent console, I mounted
the image **offline** on the Linux host (`qemu-nbd -c /dev/nbd0` + `zpool
import -R <altroot>`) and read the guest's own SMF logs (`/var/svc/log/*.log`),
device tree (`/etc/path_to_inst`, `/dev/dsk`), and binaries (`file(1)`)
directly. No boot-timing guesswork.

## The three defects

### 1. `sysconfig` is a Linux ELF (cannot exec on illumos)

```
$ file .../usr/lib/sysconfig/sysconfig
ELF 64-bit LSB pie executable, x86-64 ... interpreter /lib64/ld-linux-x86-64.so.2,
    for GNU/Linux 3.2.0 ...
```

`/var/svc/log/system-sysconfig:default.log`:

```
Executing start method ("/usr/lib/sysconfig/sysconfig provision --source auto").
Method "start" failed due to signal KILL.      (×3, all in the same second)
```

It is a `x86_64-unknown-linux-gnu` cross-build (needs GLIBC_2.x + ld-linux),
shipped into an illumos image. illumos cannot exec it; the kernel SIGKILLs it
instantly, svc.startd retries a few times and drops
`svc:/system/sysconfig:default` into maintenance. That is why no login user is
ever created — **not** a 120s SMF timeout, and **not** the seed format.
(`mdata-get` next to it *is* a correct 32-bit Solaris ELF, confirming only
sysconfig is mis-targeted.)

### 2. No IDE/ATAPI cdrom driver → the CIDATA seed is unreadable

`/etc/path_to_inst` on a booted guest lists only:

```
/pci@0,0/isa@1 ... "isa"
/pci@0,0/pci1af4,4@2 "viorand"
/pci@0,0/pci1af4,2@3 "vioblk"
/pci@0,0/pci1af4,1@4 "vioif"
```

No `pci-ide`/`ata`/`sd` node; `/dev/dsk` has only the virtio root disk
`c1t0d0`. vmctl attaches the NoCloud CIDATA seed as an **IDE cdrom**
(`crates/vm-manager/src/backends/qemu.rs`: `file=…,if=ide,media=cdrom`), so
the seed — and the **per-run SSH key** vmctl puts inside it — is invisible to
the guest. Even a *working* sysconfig could not have read it. (The native
provisioner's own log confirms: `no CIDATA seed found (scanned /dev/dsk/c*)`.)

### 3. `/sbin/dladm`, `/sbin/ipadm`, `/sbin/flowadm` are MISSING → no networking

`/var/svc/log/network-physical:default.log`:

```
/lib/svc/method/net-physical: /sbin/dladm: not found
ifconfig: cannot plumb net0: Could not open DLPI link
ifconfig: cannot plumb vioif0: Interface does not exist
/sbin/ipadm: not found
Method "start" exited with status 96.
```

Only the SUNWcs **compat symlinks** are installed —
`usr/sbin/dladm -> ../../sbin/dladm` (dangling) — and the SUNWcs manifest
declares only those symlinks, never `file path=sbin/dladm`. The package that
ships the actual command binaries (illumos `system/network` core-os) was
**never installed** by the recipe. Without `dladm`/`ipadm` the datalink+IP
layer cannot be configured, so the guest gets no IP and nothing answers the
QEMU SSH host-forward. `libdladm.so.1` and `dlmgmtd` *are* present — only the
command binaries are gone.

This is the hard blocker. It cannot be repaired offline: the image carries no
IPS file-content cache (`/var/pkg/.../file/` is empty), and the only other OI
image on the host (`openindiana-hipster-cloud.img`, OI **2025.04**) ships a
different, non-ABI-compatible build of `libdladm`/`libipadm`/`dlmgmtd`
(checksums differ), so grafting its `dladm`/`ipadm` onto the 2026.04 kernel is
not safe.

## What was fixed reproducibly (defects #1 and the ipmp fatal)

`tests/vm/scripts/bake-illumos-fixed-image.sh` (offline transform) +
`tests/vm/scripts/illumos-image-fix/`:

- **native `/sbin/sh` `sysconfig`** replacement — reads the seed *if*
  readable, creates users + SSH keys, grants `Primary Administrator`, sets
  hostname, DHCPs the primary NIC, always exits 0. Verified on a real boot:
  the SMF service now runs and **exits 0** (no more SIGKILL / maintenance).
- **`ipmp-disable.xml`** site profile — verified: after baking,
  `network/ipmp:default` boots **Disabled** (log: `[ Disabled. ]`) instead of
  failing fatally. (First attempt failed silently because the XML comment
  contained a `--`; `svccfg` rejects that. Fixed.)
- **`sudo`→`pfexec` shim** — the image ships no `sudo(8)` but every
  provisioner + `smoke.sh` calls `sudo`.
- declarative `/etc/hostname.{net0,vioif0}` + `/etc/dhcp.{net0,vioif0}`.

Produced image: `~/.local/share/vmctl/images/openindiana-rust-ci-fixed.qcow2`
(VMFile.kdl points all three guests at it). It is strictly better than the
original but **still not SSH-reachable** because defect #3 remains.

## Where it stands

- `smoke.sh` is **red** and honestly cannot be green on this image — no guest
  can get networking (defect #3).
- The `builder` native `cargo build/test --features server` shakeout was
  **not run** (it needs SSH into `builder`, which is blocked). Claiming
  illumos build results would be fabrication.
- Seed-side? No — the seed format is correct and, worse, undeliverable
  (defect #2). Provisioner-side? Partly — #1 is fixed by a native
  provisioner. **Image-side is mandatory** for #3 (and cleanly for #1/#2).

## The fix: rebuild the image (Phase B)

A correct `openindiana-rust-ci` rebuild (refraction-forger / solstice-ci OI
recipe) must:

1. **Install the network-commands package** providing `/sbin/dladm`,
   `/sbin/ipadm`, `/sbin/flowadm` (`system/network` core-os). — unblocks #3.
2. **Ship a native-illumos `sysconfig`** (not an `x86_64-linux-gnu`
   cross-build), or drop it and bake the native provisioner here. — fixes #1.
3. **Deliver the seed on a readable bus** — package the `pci-ide`/`ata`
   driver so the IDE cdrom enumerates, or have vmctl attach the seed as a
   second virtio-blk disk. — fixes #2.

The local `refraction-forger` checkout has only OmniOS + Ubuntu recipes (no OI
`rust-ci` recipe); the OI recipe lives in `solstice-ci` under `images/`. A
full illumos image build boots a builder VM and runs IPS installs against the
OI repo (multi-hour, network-dependent) — not completed this session.

### Cheaper alternative to evaluate first

`openindiana-hipster-cloud.img` (OI 2025.04) on the host is a **complete,
non-broken** OI image with working `dladm`/`ipadm`. For the `exec1`/`pg`
roles (which don't need the baked Rust toolchain) it may be a drop-in
substitute — worth a single boot test before committing to a multi-hour
rebuild. It still needs a readable seed (defect #2 may or may not apply) and
a working provisioner, but it clears the hard #3 blocker for free.
