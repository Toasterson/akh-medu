#!/usr/bin/env bash
# Solstice CI job script — native AMQP dispatch path.
#
# WHY THIS FILE EXISTS: Anima's native dispatch falls back to
# `bash .solstice/job.sh` when a project has no explicit ci_steps. This repo
# did not have the file, so every native dispatch (8/8 as of 2026-07-25)
# failed instantly — and each failure fed the CI-red corrective loop, which
# parked the dispatched work item after 3 strikes ("auto-parked: CI-red
# failed dispatch cap reached"). The WI 1/2/3 session-concurrency series was
# parked exactly this way. A missing file must not read as "the code is red".
#
# Runner image: `rust` (solstice-rust.qcow2) — Ubuntu 22.04 + rustup Rust
# (minimal profile) + sccache, baked. The `admin` job user has passwordless
# sudo. Rust is on PATH; protoc and the ssl headers are NOT baked.
#
# SCOPE — deliberately identical to .forgejo/workflows/ci.yml `check-seshat`:
# the full akh-medu workspace release build OOM-kills rustc on the Solstice
# runner VMs, so CI checks the seshat crate only (default features: includes
# `embedding`, excludes the memory-heavy `synthesis` candle feature). Keep the
# two lanes at the SAME scope: a native verdict must never be red for work the
# Actions lane would call green, or the CI-red loop starts parking work items
# on phantom failures again. Widen both lanes together or neither.
set -euo pipefail

# The image bakes CARGO_HOME=/opt/rust/cargo (root-owned); jobs run as an
# unprivileged user whose registry-cache writes there fail with EACCES.
# Redirect CARGO_HOME into the job-writable checkout.
export CARGO_HOME="$PWD/.cargo-home"
mkdir -p "$CARGO_HOME"

echo "=== conflict-marker backstop ==="
# Begin/end markers only; the merge engine already guards this server-side.
if git grep -lnI -e '^<<<<<<< ' -e '^>>>>>>> ' -- . ':(exclude).forgejo/workflows/ci.yml' ':(exclude).solstice/job.sh' ':(exclude)vendor'; then
  echo "ERROR: unresolved git conflict markers found in the files listed above"
  exit 1
fi
echo "OK: no conflict markers"

echo "=== install build deps (Rust is pre-baked; protoc + headers are not) ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential pkg-config protobuf-compiler libprotobuf-dev libssl-dev
# Distro protoc (3.12) is sufficient here: this repo's protos use no proto3
# `optional` fields today, and the Actions lane has been green on the same
# distro protoc. If protoc ever starts failing on an `optional` field, copy
# the pinned-release install block from Anima's .solstice/job.sh rather than
# reaching for --experimental_allow_proto3_optional.

echo "=== toolchain ==="
command -v cargo
cargo --version
rustc --version

echo "=== cargo check -p seshat (default features) ==="
# CARGO_BUILD_JOBS=2 mirrors ci.yml — parallel rustc on the CI VM profile
# swaps/OOMs above this.
CARGO_BUILD_JOBS=2 cargo check -p seshat

echo "=== done: native lane green at Actions-lane scope ==="
