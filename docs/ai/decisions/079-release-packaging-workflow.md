# ADR 079 — Release & packaging workflow (Solstice CI → Linux images + illumos IPS)

> Date: 2026-06-30
> Status: Accepted (implemented; first tagged run pending)
> Related: `.forgejo/workflows/ci.yml` (existing per-crate CI + image/chart push),
>          `release.toml` (cargo-release: tag `v{{version}}`, allow-branch main),
>          `docker/{akhomed,seshd,akh}.Dockerfile`, ADR 049 (relocatable Akh /
>          Postgres truth), ADR 072 (akhomed-store), the SMF manifests under
>          `contrib/smf/`, the IPS recipe under `packaging/illumos/`

## Context

We ship three binaries and had no release pipeline that produced installable
artifacts:

- **`akh`** — CLI/TUI client (workspace root, default features `nlu-candle,acp`).
- **`akhomed`** — the reasoning daemon (root crate, `--bin akhomed`, requires the
  `server` feature = `daemon,acp,postgres`).
- **`seshd`** — Seshat's Archive MCP service (`crates/seshat`, `--bin seshd`,
  default feature `embedding`, optional `synthesis`).

Deployment targets differ by OS, per the operator's request:

- **Linux** runs as **container images** (already orchestrated via Flux/Helm; the
  existing CI pushes a `seshd` image + the Helm chart to the Forgejo OCI registry
  `code.aopc.cloud/toasterson`).
- **illumos** runs in **zones managed out-of-band** — CI only needs to *deliver
  binaries* plus SMF service definitions; the operator wires zones themselves.

Three constraints shaped the design:

1. **The Solstice (Forgejo Actions) Linux runner is memory-limited.** A full
   workspace release build OOM-kills rustc — existing CI works around it with
   `CARGO_BUILD_JOBS=2` and per-crate scoping. A release job that recompiles all
   three binaries on that runner is a non-starter.
2. **onnxruntime has no illumos build.** Both `akhomed` (`nlu-ml`) and `seshd`
   (`embedding`) dynamically load `libonnxruntime.so`. illumos builds must avoid
   those features.
3. **Solstice env quirks.** `github.*` expressions and `${{ secrets.* }}`
   substitution inside step `env:` blocks are unreliable; `$GITHUB_ENV` has
   crashed dash historically. Branch/tag/secret resolution must be done in-script
   from the ambient environment (the established `resolve-branch.sh` pattern).

A late discovery materially simplified the illumos side: in the feature sets that
matter, the heavy C/C++ deps **drop out**. `oxigraph` is `default-features =
false` (no RocksDB unless `sparql-persist`), and `akhomed --features server`
pulls **no** ORT — only the Dockerfile adds `nlu-ml`. So the native illumos build
is mostly pure-Rust + libc, not the RocksDB/ORT/C++ tangle the Dockerfiles
suggest.

## Decision

A tag-triggered `Release` workflow (`.forgejo/workflows/release.yml`, fires on
`push: tags: ['v*']`) with three independent jobs:

### 1. `images` (Linux, `ubuntu-22.04`)
Build and push `akh`, `akhomed`, `seshd` images tagged `:vX.Y.Z` + `:latest` to
the Forgejo registry, reusing the existing single-binary Dockerfiles (new lean
`docker/akh.Dockerfile` for the CLI). `DOCKER_BUILDKIT=1` so the akhomed
Dockerfile's cache mounts work without the buildx plugin. Reuses Solstice's
boot-time docker auto-auth.

### 2. `linux-tarball` (Linux, `ubuntu-22.04`, `needs: images`)
**Extract** the binaries (and `libonnxruntime.so`) from the just-built images via
`docker create`/`docker cp` — **no recompile**, so the OOM constraint never
bites — and assemble `akh-medu-vX.Y.Z-x86_64-linux.tar.gz` (binaries + system
`systemd` units + `install.sh` + sample config). Attach to the Forgejo Release.
This serves bare-metal Linux installs without contradicting "Linux uses images"
(images remain the primary deploy artifact).

### 3. `illumos-pkg` (illumos, native runner `runs-on: illumos`)
Build natively, **ORT-free**:
- `akh`: default features.
- `akhomed`: `--features server` (no `nlu-ml`).
- `seshd`: `--no-default-features --features synthesis` (drops `embedding`/ORT).

Then stage a proto area and build a **native IPS pkg(5) package**, emitted as a
self-contained `.p5p` archive (`pkg install -g akh-medu-X.Y.Z.p5p akh-medu`). The
package delivers the binaries to `/opt/akh-medu/bin`, the SMF manifests to
`/lib/svc/manifest/application/akh-medu/` with a `restart_fmri=svc:/system/
manifest-import:default` actuator (auto-import on install), method scripts to
`/lib/svc/method/`, a `preserve=true` live config, owned data/cache trees, and
the `akh`/`seshat` service accounts (`user`/`group` actions). Attach to the
Forgejo Release. Optional `--repo-uri` republishes to a remote pkg depot.

### Cross-cutting
- **SMF**: `svc:/application/akh-medu/{akhomed,seshd}` are contract services run by
  dedicated unprivileged users; XDG paths set via `method_environment`; network
  bind / config-file are svccfg-tunable `config/*` properties read by the method
  scripts. `akh` is a CLI — no service.
- **install.sh**: POSIX, OS-detecting — Linux → systemd, illumos-from-tarball →
  `svccfg import` (fallback to IPS). Creates users + dirs idempotently; never
  auto-starts.
- **Release uploads**: a dependency-free (no jq/python) `forgejo-release.sh`
  does get-or-create-by-tag + idempotent asset replace against the Forgejo REST
  API, so it runs on both runners. Tokens come from the ambient env
  (`FORGEJO_TOKEN`/`RELEASE_TOKEN`/`REGISTRY_TOKEN`), never `${{ secrets }}`
  templating.
- **Tag resolution**: `resolve-tag.sh`, companion to `resolve-branch.sh`.

## Consequences

**Positive**
- One `git tag vX.Y.Z && git push --tags` produces images, a Linux tarball, and an
  illumos IPS package, all attached to the Release.
- The illumos package is *native* (pkg/SMF), not a tarball-with-helper — upgrades,
  account creation, and service import are handled by `pkg`.
- The OOM constraint is sidestepped (images build one binary each; the tarball
  reuses them).

**Negative / caveats**
- **Requires a registered illumos act_runner** (label `illumos`) with the pkg(5)
  toolchain + a Rust toolchain. Without it the `illumos-pkg` job cannot run.
- **No in-process embedding on illumos.** `seshd` there is synthesis/Postgres
  capable but cannot embed (ORT). Point akhs at a Linux `seshd` for embedding, or
  port onnxruntime to illumos later.
- **First-run unknowns on illumos**: `candle-core` (CPU) and the `akhomed-store`
  TLS stack (`ring`/OpenSSL via sqlx) are unproven on illumos; the native runner
  will surface any gaps on the first tagged build.
- **Operator confirmations** (one-line edits): the runner label (`illumos`), the
  IPS publisher (`akh-medu`), and the service UID/GIDs (`911/912`). `REGISTRY_TOKEN`
  may need its scope broadened to `write:repository` (or add `RELEASE_TOKEN`) for
  asset uploads.

## Alternatives considered

- **Cross-compile illumos from Linux** (`x86_64-unknown-illumos`): rejected —
  fragile with the C deps and the memory-limited runner; native build is simpler
  now that the heavy deps drop out of the chosen feature sets.
- **illumos tarball + install.sh instead of IPS**: kept as a fallback path in
  `install.sh`, but IPS is the primary deliverable (operator chose native pkg).
- **Recompile the Linux tarball binaries**: rejected — OOM risk; extracting from
  the images is free and identical bits.
