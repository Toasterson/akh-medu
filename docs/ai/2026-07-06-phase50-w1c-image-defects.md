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

---

# Phase 50 / W1d — hipster-cloud evaluated: ONE remaining root cause (pool-name → no DHCP)

*2026-07-06, follow-up. The "cheaper alternative" above was taken to its
conclusion: offline-inspected, seed/key-channel reverse-engineered, booted,
and forensically re-inspected offline.*

## Verdict: hipster-cloud is FAR better than rust-ci, blocked by ONE cheap defect

`openindiana-hipster-cloud.img` clears **all three** rust-ci defects for
free (verified by offline `qemu-nbd` + `zpool import`, no boot):

- **networking binaries PRESENT** — `/sbin/dladm`, `/sbin/ipadm`,
  `/sbin/flowadm` are real 32-bit Solaris ELF (rust-ci defect #3 gone).
- **IDE/ATAPI cdrom driver PRESENT** — `kernel/drv/{ata,ahci,cmdk,sd}`
  installed, `driver_aliases` has `ata "ide"` + `ahci "pciclass,010601"`
  (rust-ci defect #2's driver gap gone; the CIDATA seed *is* enumerable).
- **native provisioner** — it does NOT use the Linux-ELF `sysconfig`;
  instead it ships illumos's own `svc:/system/illumos/{metadata,useragent}`
  agents (`/usr/lib/{metadata,useragent}`, genuine Solaris x86-64 ELF, Rust
  binaries — the upstream `illumos/metadata-agent`) (rust-ci defect #1 gone).

But a real boot still never answers SSH ("Connection timed out during
banner exchange"). The precise, forensically-confirmed root cause is a
**single, different, cheaply-fixable defect**:

### Root cause: root pool is `oirpool`, the metadata agent hardcodes `rpool`, so networking is never configured

Console + the guest's own SMF log
(`/var/svc/log/system-illumos-metadata:default.log`, read offline from the
booted overlay):

```
[ Enabled. ] [ Executing start method ("/usr/lib/metadata | tee /dev/msglog"). ]
INFO SMF instance: svc:/system/illumos/metadata:default
INFO mkdir /var/metadata
ERRO fatal error: zpool list failure: exit code 1: cannot open 'rpool': no such pool
[ Method "start" exited with status 0. ]
```

The image's root pool is named **`oirpool`** (loader: `Setting currdev to
zfs:oirpool/ROOT/0242d35e`), but the illumos metadata agent's very first
real action after `mkdir /var/metadata` is a `zpool list rpool` (for its
"expand rpool" step). That fails, and the agent bails **before it reaches
its network-config stage** — where it, and *only* it, would run
`ipadm create-addr -T dhcp vioif0/v4`. (It exits status 0, so it does NOT go
to maintenance; it just silently does nothing further.)

Downstream, confirmed by mounting the post-boot overlay offline:

- `/etc/ipadm/ipadm.conf` is **empty** (no interface objects); there is no
  `aobjmap.conf` and no `/var/db/dhcpagent/` lease → **`vioif0` has a
  datalink but never got an IP.** NWAM is inactive in this cloud image, and
  there were no `/etc/hostname.*`/`/etc/dhcp.*` files, so nothing else
  configured the NIC.
- `svc:/network/ssh:default` **DID start** (`Method "start" exited with
  status 0`, generated host keys) — sshd is up and healthy. It just has no
  network address to answer on, so QEMU's `hostfwd tcp::10042-:22` completes
  the TCP handshake (QEMU accepts) but the guest sends no banner → the exact
  "Failed getting banner" / "timed out during banner exchange" symptom.

So this is **not** a seed-format problem and **not** an SSH/PermitRootLogin
problem — the login channel is fine, the guest simply has no IP.

### The key/login channel (reverse-engineered, works, and is now moot for the blocker)

For the record, the metadata/useragent key channel *was* precisely
identified (via `strings` + DWARF debug-info on `/usr/lib/{metadata,useragent}`):

- **NoCloud/cidata IS supported**: `metadata` scans `/dev/dsk` + `/dev/rdsk`
  for a `cidata`-labelled volume, mounts it (`meta-data` + `user-data` +
  `network-config`), and reads `public-keys/0/openssh-key` → installs to
  **`/root/.ssh/authorized_keys`** (root is the key target, confirmed by the
  `checking SSH public keys for user .../root/.ssh` code path).
- **`useragent` parses cloud-config `#cloud-config`** with a genuine
  cloud-init-shaped deserializer — DWARF shows `struct CloudConfig { groups,
  users, write_files, ca_certs, bootcmd, runcms, final_message, packages,
  package_update, phone_home, growpart, ssh_authorized_keys, ssh_keys,
  no_ssh_fingerprints, ssh }` and `struct UserConfig { name, expire_date,
  gecos, homedir, primary_group, groups, lock_passwd, inactive, passwd,
  no_create_home, no_user_group, ssh_authorized_keys, system, shell }`. So
  vmctl's default `#cloud-config` (users + ssh_authorized_keys) is
  format-compatible — **but note `UserConfig` has no sudo/RBAC-profile
  field**, so a cloud-init-created user gets key auth and zero admin rights.

Because both the seed read AND the useragent run happen *after* the network
stage that the pool-name abort skips — and because relying on them still
wouldn't grant admin — the login channel was **baked statically offline**
instead (independent of any agent): static `/root/.ssh/authorized_keys`,
`sshd_config PermitRootLogin no → prohibit-password`, and the `sudo→pfexec`
shim. All three were confirmed present + correct in the booted guest.

### The fix (cheap, offline) — VERIFIED GREEN: DHCP + DNS, root SSH login live

The missing pieces were networking + DNS, neither of which needs the
metadata agent:

- **DHCP** (`bake` step 4): two empty declarative files —
  `/etc/hostname.vioif0` + `/etc/dhcp.vioif0` (illumos's classic "DHCP this
  NIC" mechanism, read by `svc:/network/physical:default`, which is enabled
  and runs fine) — make the guest DHCP `vioif0` on its own, bypassing the
  aborting metadata agent.
- **DNS** (`bake` step 5): OpenIndiana does **not** wire DNS by default
  (again, that's the metadata agent's job, and it aborted). Two files, both
  required (the illumos gotcha): `/etc/resolv.conf` →
  `nameserver 10.0.2.3` (QEMU usernet's DNS forwarder) + `1.1.1.1` fallback,
  **and** `nsswitch.conf` `hosts:`/`ipnodes:` must list `dns` (resolv.conf
  alone is insufficient on illumos; this image already ships
  `files dns mdns`, so the bake's `sed` is a no-op here but idempotent for
  other bases). Without this, the NIC is up + SSH works but `pkg`/`git`/
  `curl` fail with "Could not resolve host: pkg.openindiana.org".

**VERIFIED on a real boot (2026-07-06):**
```
# ssh -i .keys/root_ed25519 -p 10042 root@127.0.0.1
LOGIN_OK   uid=0(root) gid=0(root)
ipadm show-addr:  vioif0/?  dhcp  ok  10.0.2.15/24
grep '^hosts:' /etc/nsswitch.conf:  hosts:  files dns mdns
cat /etc/resolv.conf:  nameserver 10.0.2.3 / nameserver 1.1.1.1
getent hosts pkg.openindiana.org:  65.21.23.2  pkg.openindiana.org
id akh-exec:  uid=100(akh-exec) gid=1(other)   # exec1 provisioner ran
svcs network/ssh:  online
```
Both `exec1` (:10042) and `pg` (:10043) answer `SSH-2.0-OpenSSH_9.9` and
accept the baked root key. **The SSH/login unblock is real and
reproducible.**

### Reproducible recipe (fresh operator, from clean)

```
cd tests/vm
bash scripts/gen-keys.sh                       # mints .keys/{root,akh-exec}_ed25519
scripts/bake-hipster-cloud-image.sh \          # offline bake: key+sshd+sudo+DHCP+DNS
    ~/.local/share/vmctl/images/openindiana-hipster-cloud.img \
    ~/.local/share/vmctl/images/openindiana-hipster-cloud-fixed.qcow2 \
    .keys/root_ed25519.pub
./smoke.sh                                       # PATH-shims q35->pc, vmctl up, checks
```
`vmctl up` creates any VM not already in `~/.local/share/vmctl/vms.json`
(fresh CoW overlay over the shared `-fixed` base) and reuses ones already
running; `vmctl down --destroy` removes them. exec1 comes up green first and
independently of pg/builder.

### Provisioning-content status (secondary; SSH unblock is the win)

- **exec1 = GREEN**: `akh-exec` non-root user created, `network/ssh` online,
  DHCP IP, DNS resolves. This is the load-bearing guest (48c executor / W3
  runner / 48b target).
- **pg**: DNS resolves and the IPS catalog fetches fine, but the first run
  FAILed on a provisioner-CONTENT bug — `pg-provision.sh` looked for
  `database/postgresql-<N>`, whereas OI ships `database/postgres-<NN>` (and
  the `<NN>` legacy trains 82..96 out-sort the modern 10..12 numerically).
  Fixed in `pg-provision.sh` (selects `database/postgres-12`, installs the
  `/developer` + `/library` subpackages for pgvector); **not yet run through
  a full postgres+pgvector build** (slow over guest egress — deferred).
- **builder**: never reached this run — `smoke.sh` bails when `vmctl up`
  returns non-zero (pg's provisioner failure), before the SSH assertion
  checks. hipster-cloud ships no Rust toolchain, so `builder-provision.sh`'s
  rustup fallback (now that DNS works) is the path; deferred with pg.

Note: `smoke.sh` today treats *any* guest's provisioner failure as fatal to
the whole run (it exits on `vmctl up` non-zero before the per-guest SSH
assertions). exec1 still boots green regardless, but a future tidy-up could
decouple the exec1 assertions from pg/builder provisioning-content.

## Recommendation for the image rebuild (single clearest path)

**Base the rebuild on `openindiana-hipster-cloud` (OI 2025.04), NOT
`openindiana-rust-ci`.** It already solves rust-ci's three fatal defects
(net binaries, cdrom driver, native provisioner). Then the *only* fixes a
clean rebuild needs are:

1. **Name the root pool `rpool`** (the illumos metadata agent hardcodes it),
   OR ship declarative `/etc/hostname.vioif0` + `/etc/dhcp.vioif0` so
   networking is agent-independent. Either one unblocks SSH. (The offline
   bake here does the latter, non-invasively.)
2. **For the `builder` role only**: install the Rust toolchain + `gcc`/
   `clang`/`gmake`/`git` (hipster-cloud is minimal and ships none). `exec1`
   and `pg` don't need it.
3. **Optional niceties already handled by the offline bake**: a `sudo(8)`
   (or the pfexec shim), and `PermitRootLogin prohibit-password` + a root
   authorized_keys — needed only if you want key-based root SSH without
   depending on the metadata/useragent seed path.

Everything else (UEFI + virtio-transitional, the q35→pc shim) is already in
`tests/vm/VMFile.kdl` + `tests/vm/scripts/qemu-illumos-shim/` and works.

## Consolidated "what a bootable, SSH-reachable illumos guest needs" (all images tested)

| Requirement | rust-ci 2026.04 | hipster-cloud 2025.04 | why it matters |
|---|---|---|---|
| `/sbin/dladm`,`ipadm`,`flowadm` binaries | ❌ missing | ✅ present | no datalink/IP layer without them |
| IDE/ATAPI cdrom driver (`ata`/`ahci`) | ❌ absent | ✅ present | else the CIDATA seed is unreadable |
| native-illumos provisioner (not Linux ELF) | ❌ Linux ELF `sysconfig` | ✅ illumos `metadata`/`useragent` | Linux ELF is SIGKILLed on exec |
| provisioner actually configures the NIC | n/a (never ran) | ❌ aborts on `rpool` name | **the hipster-cloud blocker** |
| root pool named `rpool` (or declarative DHCP) | unknown | ❌ `oirpool` → fix needed | metadata agent hardcodes `rpool` |
| `svc:/network/ssh` enabled by default | — | ✅ (generic profile) | sshd must be up |
| root SSH key channel | — | ✅ metadata→`/root/.ssh` (or bake) | login target |
| `sudo(8)` present | ❌ | ❌ (shim baked) | provisioner scripts call `sudo` |
| Rust toolchain (builder role) | claimed, unusable | ❌ absent | `builder` native build |
| UEFI + virtio-transitional + q35→pc shim | ✅ (W1b/W1) | ✅ (same) | boots to multi-user |

Net: **rust-ci is unfixable offline (missing net binaries); hipster-cloud is
one non-invasive offline fix (declarative DHCP, already baked) away from
green, and is the correct rebuild base.**
