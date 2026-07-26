# Release & install runbook

How to cut an akh-medu release and install it on Linux and illumos hosts.
See [ADR 079](../ai/decisions/079-release-packaging-workflow.md) for the design.

## Cutting a release

Releases are driven by a `vX.Y.Z` git tag (Solstice fires
`.forgejo/workflows/release.yml`).

```sh
# Bump the version + tag in one go (uses release.toml: tag v{{version}}, main only)
cargo release patch        # or: minor / major  — or tag by hand:
# git tag v0.7.1 && git push origin v0.7.1
```

On the tag, three jobs run:

| Job | Runner | Output |
|-----|--------|--------|
| `images` | `ubuntu-22.04` | `akh` / `akhomed` / `seshd` images `:vX.Y.Z` + `:latest` → `code.aopc.cloud/toasterson` |
| `linux-tarball` | `ubuntu-22.04` | `akh-medu-vX.Y.Z-x86_64-linux.tar.gz` → Forgejo Release |
| `illumos-pkg` | `illumos` (native) | `akh-medu-vX.Y.Z.p5p` → Forgejo Release |

Watch progress with the `solstice-mcp` tools (`list_runs` / `get_run` /
`tail_step_log` for `Toasterson/akh-medu`).

### Prerequisites (one-time)

- **illumos act_runner**: a Forgejo runner registered with label **`illumos`**,
  with the pkg(5) toolchain (`pkgsend`, `pkgmogrify`, `pkgdepend`, `pkgrepo`,
  `pkgrecv`, `pkgfmt`, `pkglint`) and a Rust toolchain (`~/.cargo/env`).
- **Token**: `REGISTRY_TOKEN` (Forgejo PAT) needs `write:package` for image
  pushes *and* `write:repository` for Release-asset uploads. If you keep
  `REGISTRY_TOKEN` package-only, add a `RELEASE_TOKEN` secret with
  `write:repository` (resolved automatically by `forgejo-release.sh`).

### Manual / re-run

Trigger `Release` via `workflow_dispatch` and set `RELEASE_TAG=vX.Y.Z` in the
runner environment (the dispatch path can't always read the tag ref). Re-runs are
idempotent — assets of the same name are replaced.

## Installing on Linux

**Containers (primary).** Pull from the registry:

```sh
docker run -d --name akhomed -p 8200:8200 -v akh-data:/data \
  code.aopc.cloud/toasterson/akhomed:vX.Y.Z
docker run -d --name seshd  -p 9090:9090 -v seshat-cfg:/config \
  code.aopc.cloud/toasterson/seshd:vX.Y.Z
# the CLI:
docker run --rm -it code.aopc.cloud/toasterson/akh:vX.Y.Z akh --help
```

(Or deploy via the Helm chart `oci://code.aopc.cloud/toasterson/akh-medu`.)

**Bare metal (systemd).** From the release tarball:

```sh
tar xzf akh-medu-vX.Y.Z-x86_64-linux.tar.gz
cd akh-medu-vX.Y.Z-x86_64-linux
sudo ./install.sh                       # binaries → /usr/local/bin, units → /etc/systemd/system
sudo $EDITOR /etc/akh-medu/seshat.toml  # set the Postgres URL
sudo systemctl enable --now akhomed
sudo systemctl enable --now seshd       # needs a reachable pgvector Postgres
```

Binaries: `akhomed` (daemon, env-tunable `AKH_SERVER_BIND`/`AKH_SERVER_PORT`),
`seshd` (Seshat MCP), `akh` (CLI). `libonnxruntime.so` is bundled under
`/usr/local/lib/akh-medu/`.

## Installing on illumos (zones)

The IPS package is the native path. Inside the zone (or its global zone for a
sparse install):

```sh
# from the .p5p archive attached to the release:
pkg install -g akh-medu-vX.Y.Z.p5p akh-medu

# the package created the akh/seshat users, imported the SMF manifests, and
# delivered binaries to /opt/akh-medu/bin (akh is symlinked to /usr/bin/akh).
svccfg -s svc:/application/akh-medu/akhomed:default listprop config   # bind/port knobs
$EDITOR /etc/akh-medu/seshat.toml                                     # set the Postgres URL
svcadm enable akhomed
svcadm enable seshd
svcs -xv akhomed seshd
```

To publish into a managed IPS depot instead of installing from the archive, the
build can push there too:

```sh
packaging/illumos/build-ips.sh --version X.Y.Z --bindir target/release \
  --repo-uri https://pkg.aopc.cloud/akh-medu
pkg set-publisher -g https://pkg.aopc.cloud/akh-medu akh-medu
pkg install application/akh-medu
```

**Caveat — embedding.** The illumos `seshd` is built without the `embedding`
feature (onnxruntime has no illumos build): `seshat_wish`/synthesis and the
Postgres archive work, but in-process embedding is inert. Run a Linux `seshd` for
embedding, or supply embeddings out of band.

### Tunables (SMF)

```sh
# change the akhomed bind/port:
svccfg -s svc:/application/akh-medu/akhomed:default setprop config/port = 8300
svcadm refresh akhomed; svcadm restart akhomed

# point seshd at a different config file:
svccfg -s svc:/application/akh-medu/seshd:default setprop config/file = /etc/akh-medu/other.toml
svcadm refresh seshd; svcadm restart seshd
```

## Installing the `akh` client (cargo-binstall)

`akh` is the lean ACP client (no candle/onnxruntime) you install on any machine
that needs to talk to an `akhomed` — your laptop, a jump host, a zone. It ships as
a prebuilt binary on each release, so [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall)
downloads it instead of compiling:

```sh
cargo binstall akh-medu          # pulls akh-<version>-<target>.tar.gz from the Forgejo release
akh --help
```

Only `akh` is installed (`akhomed`/`seshd` are daemons, excluded from the binstall
set). Prebuilts exist for `x86_64-unknown-linux-gnu` and `x86_64-unknown-illumos`;
other targets fall back to a source build with default features. To build the lean
client from source yourself:

```sh
cargo install --git https://code.aopc.cloud/toasterson/akh-medu \
  --no-default-features --features client --bin akh akh-medu
```

## Connecting to a remote akhomed

`akh` defaults to discovering a daemon on the **local** host (via its PID file).
To drive a **remote** `akhomed` over ACP/TCP, pass `--remote host:port` (or set
`AKH_REMOTE`). The remote daemon must serve ACP/TCP:

```sh
# On the daemon host — enable the ACP TCP listener:
AKH_ACP_TCP_BIND=0.0.0.0 AKH_ACP_TCP_PORT=8201 akhomed
#   (under SMF/systemd, set these in the service environment)

# From anywhere:
akh --remote akh-host.internal:8201 acp status
akh --remote akh-host.internal:8201 -w default chat
export AKH_REMOTE=akh-host.internal:8201   # or persist it in the environment
akh acp capabilities
```

ACP/TCP is **plaintext and unauthenticated** — expose it only on a trusted
network, or front it with an mTLS tunnel (see `docs/ops/akh-pg-remote-mtls.md` for
the pattern). See [ADR 080](../ai/decisions/080-lean-akh-client-binstall-remote.md).

## Building artifacts by hand

```sh
# Linux tarball from a local release build (needs a roomy machine):
cargo build --release --bin akh
cargo build --release --features server --bin akhomed
cargo build --release -p seshat --bin seshd --features embedding,synthesis
packaging/build-linux-tarball.sh --version X.Y.Z --bindir target/release \
  --libdir /usr/local/lib   # if you have onnxruntime to bundle

# illumos IPS (on an illumos host):
cargo build --release --bin akh
cargo build --release --features server --bin akhomed
cargo build --release -p seshat --bin seshd --no-default-features --features synthesis
packaging/illumos/build-ips.sh --version X.Y.Z --bindir target/release
```
