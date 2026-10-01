#!/bin/sh
set -eu
# Runner auto-auth owns docker config. Never replace it with docker login.
if ! docker buildx version >/dev/null 2>&1; then
  mkdir -p "$HOME/.docker/cli-plugins"
  case "$(uname -m)" in x86_64) buildx_arch=amd64 ;; aarch64|arm64) buildx_arch=arm64 ;; *) exit 2 ;; esac
  curl -fsSL "https://github.com/docker/buildx/releases/download/v0.25.0/buildx-v0.25.0.linux-$buildx_arch" -o "$HOME/.docker/cli-plugins/docker-buildx"
  chmod 755 "$HOME/.docker/cli-plugins/docker-buildx"
fi
# Compilation uses persistent native builders configured by run-platform-build.py.
# QEMU remains local to the ephemeral VM for short ARM64 smoke executions only.
docker run --privileged --rm tonistiigi/binfmt:qemu-v9.2.0 --install arm64
for arch in amd64 arm64; do
  docker run --rm --platform "linux/$arch" debian:bookworm-slim uname -m
done
