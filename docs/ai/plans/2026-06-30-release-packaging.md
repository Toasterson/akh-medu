# Plan: Release & packaging workflow (Linux images + illumos IPS)

- **Date:** 2026-06-30
- **Status:** Implemented (pending first tagged run + illumos runner registration)
- **ADR:** [079](../decisions/079-release-packaging-workflow.md)
- **Anima:** Akh-Medu project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (file a WI for
  the first-release validation)

## Goal

`git tag vX.Y.Z && git push origin vX.Y.Z` →
- Docker images `akh` / `akhomed` / `seshd` (`:vX.Y.Z` + `:latest`) on
  `code.aopc.cloud/toasterson`.
- `akh-medu-vX.Y.Z-x86_64-linux.tar.gz` (binaries + systemd + `install.sh`).
- `akh-medu-vX.Y.Z.p5p` IPS package (binaries + SMF) for illumos zones.
All three attached to the Forgejo Release for the tag.

## Deliverables (all landed)

| File | Purpose |
|------|---------|
| `.forgejo/workflows/release.yml` | tag-triggered 3-job pipeline (`images`, `linux-tarball`, `illumos-pkg`) |
| `.forgejo/scripts/resolve-tag.sh` | Solstice-robust tag resolution (companion to `resolve-branch.sh`) |
| `.forgejo/scripts/forgejo-release.sh` | jq-free get-or-create release + idempotent asset upload |
| `docker/akh.Dockerfile` | lean CLI image (default features, no ORT) |
| `contrib/smf/akhomed.xml`, `seshd.xml` | SMF service manifests (contract services) |
| `contrib/smf/akhomed.method`, `seshd.method` | SMF start methods (svcprop-driven) |
| `contrib/systemd/system/akhomed.service`, `seshd.service` | system-level units (bare-metal Linux) |
| `contrib/config/seshat.toml.sample` | seshd config template |
| `packaging/install.sh` | POSIX OS-detecting installer (Linux systemd / illumos SMF) |
| `packaging/build-linux-tarball.sh` | assemble Linux tarball from a bindir |
| `packaging/illumos/akh-medu.p5m.mog` | pkgmogrify transforms (fmri, SMF actuator, users, dirs) |
| `packaging/illumos/build-ips.sh` | proto → pkgsend/pkgdepend/pkglint → `.p5p` |
| `docs/ops/release.md` | runbook (cut a release; install on each OS) |

## Feature sets per artifact

| Binary | Linux image / tarball | illumos IPS |
|--------|----------------------|-------------|
| `akh` | default (`nlu-candle,acp`) | default |
| `akhomed` | `server,nlu-ml,nlu-candle,delegation,email,calendar` (Dockerfile) | `server` only (no ORT) |
| `seshd` | `embedding,synthesis` | `synthesis` only (no `embedding`/ORT) |

## Open items / follow-ups

1. **Register the illumos act_runner** with label `illumos` + pkg(5) toolchain +
   Rust. Confirm/adjust the `runs-on:` label in `release.yml`.
2. **Token scope**: ensure `REGISTRY_TOKEN` has `write:repository` (or add
   `RELEASE_TOKEN`) so asset uploads succeed.
3. **Confirm IPS publisher** (`akh-medu`) and **service UID/GIDs** (`911/912`).
4. **First-run validation on illumos**: `candle-core` CPU build, `akhomed-store`
   TLS stack (`ring`/sqlx), `pkglint` cleanliness. Capture results in a follow-up
   note + Anima WI.
5. **Later**: multi-arch (arm64) Linux images; onnxruntime-on-illumos so `seshd`
   can embed natively; sign the `.p5p` / images.

## Validation done in this session

- `sh -n` on every shell script — clean.
- `xmllint` well-formedness on both SMF manifests — clean.
- YAML parse of `release.yml` — clean.
- Static review of feature-gating (confirmed `--features server` pulls no ORT;
  `oxigraph` default has no RocksDB; the vendored `candle-core` patch does not
  hijack seshat's `candle 0.8`).
- Adversarial subagent review (Forgejo Actions semantics, POSIX sh, IPS/SMF,
  cross-file consistency). Fixes applied:
  - `REGISTRY` set **inline** in each `run:` block (Solstice doesn't reliably
    inject workflow-level `env:` — matches `ci.yml`).
  - dropped the akhomed SMF `refresh` method (`:kill -HUP` would kill the daemon;
    no SIGHUP handler) — config changes need `svcadm restart`.
  - `forgejo-release.sh` POST guarded against `set -e` abort + re-fetch on the
    concurrent-create 409.
  - `install.sh --prefix` no longer crashes under `set -u`; the illumos tarball
    fallback warns (and points at the `.p5p`) instead of silently skipping SMF;
    `build-linux-tarball.sh` now stages `smf/` so that fallback is functional.
