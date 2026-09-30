#!/bin/sh
set -eu
# Runner auto-auth owns docker config. Never replace it with docker login.
if ! docker buildx version >/dev/null 2>&1; then
  mkdir -p "$HOME/.docker/cli-plugins"
  case "$(uname -m)" in x86_64) buildx_arch=amd64 ;; aarch64|arm64) buildx_arch=arm64 ;; *) exit 2 ;; esac
  curl -fsSL "https://github.com/docker/buildx/releases/download/v0.25.0/buildx-v0.25.0.linux-$buildx_arch" -o "$HOME/.docker/cli-plugins/docker-buildx"
  chmod 755 "$HOME/.docker/cli-plugins/docker-buildx"
fi
# A native remote builder can be supplied by the enrolled runner configuration.
# Otherwise register both interpreters explicitly; builds can take several hours.
if [ -z "${AKH_BUILDX_BUILDER:-}" ]; then
  docker run --privileged --rm tonistiigi/binfmt:qemu-v9.2.0 --install arm64,amd64
  builder="akh-platform-${SOLSTICE_RUN_ID:-${GITHUB_RUN_ID:-$$}}"
  mkdir -p "$HOME/.config/akh-ci"
  cat > "$HOME/.config/akh-ci/buildkitd.toml" <<'TOML'
[worker.oci]
  max-parallelism = 1
TOML
  # Group jobs provide concurrency. Serialize architecture stages within each
  # builder so two Rust linkers do not exhaust one runner VM's memory.
  docker buildx create --name "$builder" --driver docker-container \
    --buildkitd-config "$HOME/.config/akh-ci/buildkitd.toml" --use
else
  docker buildx use "$AKH_BUILDX_BUILDER"
fi
docker buildx inspect --bootstrap
for arch in amd64 arm64; do
  docker run --rm --platform "linux/$arch" debian:bookworm-slim uname -m
done
