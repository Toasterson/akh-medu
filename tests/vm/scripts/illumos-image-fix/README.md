# illumos-image-fix — offline repairs for `openindiana-rust-ci`

Phase 50 / W1c. These files repair the broken `openindiana-rust-ci`
OI Hipster 2026.04 image so that it can (partly) self-provision under vmctl.
They are applied by [`../bake-illumos-fixed-image.sh`](../bake-illumos-fixed-image.sh),
which mounts the image offline (`qemu-nbd` + `zpool import`, no boot needed)
and drops them in.

## The three image-build defects (root cause, 2026-07-06)

The blocker was NOT the cloud-init seed / user-data format (which matches
sysconfig's own parser). It is three independent, fatal defects baked into
the published image:

| # | Defect | Evidence | Fixable offline? |
|---|--------|----------|------------------|
| 1 | `/usr/lib/sysconfig/sysconfig` is a **GNU/Linux x86-64 ELF** (interp `/lib64/ld-linux-x86-64.so.2`) in an illumos image | `file(1)`; SMF log shows `Method "start" failed due to signal KILL` ×3 instantly | **Yes** — replace with native shell (`sysconfig-native`) |
| 2 | **No IDE/ATAPI cdrom driver**; running device tree is only `isa@1` + `vioblk`/`vioif`/`viorand`. vmctl attaches the CIDATA seed as an IDE cdrom, so it is unreadable | `path_to_inst`; `/dev/dsk` has only the virtio root disk | Partly — the per-run key in the seed is undeliverable; needs a static key or a virtio-bus seed |
| 3 | **`/sbin/dladm`, `/sbin/ipadm`, `/sbin/flowadm` missing** — only the SUNWcs compat symlinks are installed; the package shipping the binaries was never installed | dangling `usr/sbin/dladm -> ../../sbin/dladm`; `network/physical` log: `/sbin/dladm: not found`, `cannot plumb net0: Could not open DLPI link` | **No** — needs an image rebuild |

Defect #3 is the hard blocker: without `dladm`/`ipadm` the datalink+IP layer
can't come up, so the guest never gets an IP and never answers SSH. It cannot
be repaired offline (the image carries no IPS file-content cache, and the
2025.04 OI image on the host ships a different, non-ABI-compatible build of
`libdladm`/`libipadm`/`dlmgmtd`).

## Files

- **`sysconfig-native`** — native `/sbin/sh` reimplementation of the
  provisioner. Reads the NoCloud CIDATA seed *if readable*, creates the
  seed's login user(s) + SSH keys, grants them the `Primary Administrator`
  RBAC profile (so the `sudo` shim gives them root), sets hostname, and
  brings up DHCP on the primary NIC. Always exits 0 so the transient SMF
  instance reaches `online` instead of `maintenance`. Installed over
  `/usr/lib/sysconfig/sysconfig` (the Linux ELF is kept as
  `sysconfig.linux-elf.bak`).
- **`sudo`** — `sudo`→`pfexec` shim. The image ships no `sudo(8)`, but the
  vmctl provisioner scripts and `smoke.sh` call `sudo`. Maps `sudo cmd` →
  `pfexec cmd` and `sudo -u user cmd` → `pfexec su - user -c "cmd"`.
- **`ipmp-disable.xml`** — SMF site profile that disables
  `svc:/network/ipmp:default`, which fails fatally on a single-NIC guest
  (`in.mpathd: setup_listener failed for both IPv4 and IPv6`). Applied
  automatically by `manifest-import`. (Note: XML comments must not contain
  a double hyphen — an earlier version was silently rejected by `svccfg`
  for exactly that.)

## What a proper image REBUILD must fix

The refraction-forger / solstice-ci OI recipe must:

1. **Install the network-commands package** that provides `/sbin/dladm`,
   `/sbin/ipadm`, `/sbin/flowadm` (illumos `system/network` / core-os).
   This is the one defect that blocks networking.
2. **Ship a native-illumos `sysconfig`** binary, not a Linux cross-build
   (or drop sysconfig and rely on a native provisioner like the one here).
3. **Deliver the seed on a bus illumos can read** — either package the
   `pci-ide`/`ata` driver in the boot archive so the IDE cdrom enumerates,
   or (simpler) have vmctl attach the seed as a second virtio-blk disk.
