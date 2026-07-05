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

## OpenIndiana-under-KVM boot quirks

*(To be filled in once VMs actually boot — this document is being written
concurrently with the fleet coming up, per instructions to run long
operations detached and keep working rather than block on them.)*

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
