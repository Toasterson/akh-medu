# Lilith illumos native validation VM — 2026-09-22

## Result

The reusable guest `lilith-illumos-20260922` is operational on this host:

- OpenIndiana Hipster 2026.04, `illumos-c4b084c090`
- 2 vCPUs, 4096 MiB RAM, no GPU, SLIRP networking
- loopback-only SSH at `127.0.0.1:10115` as unprivileged `akh-exec`
- Python 3.9.25, GCC 14.3.0, Rust/Cargo 1.96.0, GNU make/tar, ZFS
- project-only QCOW2 overlay under
  `~/.local/share/akh-medu/lilith-illumos-20260922/`
- credentials and pinned known-host data outside the repository under
  `~/.local/share/akh-medu/lilith-native-20260922/`

The base images and the pre-provisioned source overlay were not modified. The
Lilith disk is copy-on-write. No Refraction Forger rebuild was needed after a
working local Solstice/OpenIndiana provision layer was found.

## Image selection and repair

The July `openindiana-rust-ci-fixed.qcow2` bake reaches its 2026.04 kernel but
does not import the relocated ZFS root. This reproduced with 4 and 6 GiB under
q35 and with 4 GiB under i440fx. Serial evidence is retained in the three
`attempt-*-console.log` files. The older cloud image boots, but its illumos
metadata agent does not consume the newer sysconfig/NoCloud user contract.

The repair began from a COW child of the already provisioned local Solstice
`exec1` layer. Only Lilith's child disk was imported offline to add its public
SSH key. It was then flattened into a private, standalone golden image so a
later `exec1` start, deletion, or edit cannot affect Lilith. The golden SHA-256
is `437edebceb20931ffee43d0838b0caf284e174368a1def421128a4375ac4f8ee`;
the active runtime disk is a fresh 17 KiB COW child of that golden image. The
source and project chains and hashes are recorded in the backing-chain and
SHA-256 evidence files.

## Native checks

`native-probe.txt` records the OS, identity, services, ZFS datasets, and exact
tool versions. The bounded SSH executor compiled and ran independent Python,
C, and Rust probes successfully. It also demonstrated a guest process timeout
(exit 124), a 64 KiB hard output-file limit, host response truncation, and
remote workspace cleanup. Receipts are the `*-receipt.json` files.

As a separate unscored diagnostic, all sixteen frozen comparison responses
were rechecked. Grammar admission and semantic correctness remain separate.
For Lilith, six artifacts passed grammar admission: four passed all native
finite cases, one raised division by zero, and one produced a contract
mismatch; two were grammar-rejected and never executed. For Ptah, six were
grammar-admitted and passed all native cases; two were grammar-rejected and
never executed. This does not alter the official frozen evaluation scores.
See `native-model-diagnostic.json`.

This validates native Python behavior and the C/Rust toolchains. It does not
claim that the complete Akh runtime builds or passes on illumos yet.

## Operation

```bash
scripts/session-platform/lilith_vm.sh start
scripts/session-platform/lilith_vm.sh status
scripts/session-platform/lilith_vm.sh stop
```

Coordinator code imports `lilith_vm_executor.py` and calls
`local_executor()`. Its immutable `contract()` pins the source-rootfs hash,
executor and guest-supervisor source hashes, SSH host-key fingerprint and
known-hosts digest, staging protocol, bounds, cleanup policy, privilege policy,
and host exposure. `execute()` rejects
symlinks and escaping targets, limits snapshots to 32 MiB, runs commands in a
fresh guest process group with hard CPU/output/process bounds, distinguishes a
supervisor timeout from an ordinary exit 124, caps returned
stdout/stderr, and removes each unique remote staging directory.

The pretty-printed contract file SHA-256 is
`d2bd7611c23c9b308cdce72944ce671b5c02a7528aa36d980ccca381bf8392db`.
The coordinator pins the format-independent canonical JSON SHA-256
`ce73f2c4da203e537a4c3899453bccec0093b127ae47b989ee2d9b5881217b59`.
`real-timeout-receipt.json` records `timed_out: true`, while
`ordinary-124-receipt.json` records the same exit code with
`timed_out: false`; both cleanup attempts succeeded.

The C and Rust build/run probes were repeated after the supervisor and
contract were frozen. `final-c-*-receipt.json` and
`final-rust-*-receipt.json` all pass with `supervisor_status: exit`; the native
binaries measured 10,008 and 1,538,056 bytes respectively. The 64 KiB
`RLIMIT_FSIZE` stops a direct Python 1 MiB file write at exactly 65,536 bytes
(`final-artifact-limit-probe-receipt.json`), but the Rust linker can produce a
larger output. A child interpreter reports inherited soft and hard limits of
`(65536, 65536)` in `final-child-rlimit-receipt.json`; the linker behavior is
likely an illumos write-path distinction and was not broadened into an OS
investigation. The current executor therefore strictly caps captured output
and direct writes, while its artifact-size limit is not a general descendant
filesystem quota. The 32 MiB copy-in limit and unique-workspace cleanup still
bound retained inputs and remove build outputs after every attempt.

The VM can be stopped while Gemma needs host memory; the overlay is reusable.
