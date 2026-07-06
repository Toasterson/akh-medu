#!/usr/bin/env bash
# Environment setup for building/running `grug-train` with `--features gpu`
# (Phase 51 P1a, ADR 087, R3 findings).
#
# This box's ROCm 6.3.4 install (/opt/rocm-6.3.4, symlinked at /opt/rocm) is
# missing the `hipconfig` CLI that `cubecl-hip-sys`'s build.rs shells out to
# at BOTH build time (to discover the ROCm path + HIP patch version and emit
# the right `rustc-link-lib`/`rustc-link-search`) and, on some code paths,
# at runtime. A shim implementing the handful of subcommands actually used
# lives at `scripts/hipconfig-shim/hipconfig` (see that file's header comment
# for exactly why a real `hipconfig -R`/`-p`/`--version` behaves the way it
# does here). Source *this* script before any `--features gpu` cargo
# build/test/run so the shim is on PATH and ROCM_PATH/HIP_PATH are set.
#
# Usage:
#   source scripts/train-env.sh
#   cargo build -p grug-train --features gpu --release
#   cargo run -p grug-train --example train_p1a --features gpu --release -- --steps 20000
#
# Idempotent: sourcing this twice does not duplicate the PATH entry.

# Resolve this script's own directory so it works regardless of the caller's
# cwd (as long as it's sourced, not executed -- `source` is required because
# this script exports into the *calling* shell's environment).
_TRAIN_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_TRAIN_ENV_SHIM_DIR="${_TRAIN_ENV_SCRIPT_DIR}/hipconfig-shim"

if [ ! -x "${_TRAIN_ENV_SHIM_DIR}/hipconfig" ]; then
  echo "train-env.sh: expected hipconfig shim at ${_TRAIN_ENV_SHIM_DIR}/hipconfig (not found or not executable)" >&2
  return 1 2>/dev/null || exit 1
fi

case ":${PATH}:" in
  *":${_TRAIN_ENV_SHIM_DIR}:"*) ;; # already on PATH, don't duplicate
  *) export PATH="${_TRAIN_ENV_SHIM_DIR}:${PATH}" ;;
esac

# ROCm 6.3.4 is installed at /opt/rocm-6.3.4 with a /opt/rocm symlink to it
# on this box. Prefer an already-set ROCM_PATH/HIP_PATH (e.g. a differently
# configured box) over overwriting it.
if [ -z "${ROCM_PATH:-}" ]; then
  if [ -d /opt/rocm ]; then
    export ROCM_PATH=/opt/rocm
  elif [ -d /opt/rocm-6.3.4 ]; then
    export ROCM_PATH=/opt/rocm-6.3.4
  else
    echo "train-env.sh: warning -- neither /opt/rocm nor /opt/rocm-6.3.4 found; ROCM_PATH left unset" >&2
  fi
fi
export HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"

echo "train-env.sh: PATH prepended with ${_TRAIN_ENV_SHIM_DIR}"
echo "train-env.sh: ROCM_PATH=${ROCM_PATH:-<unset>} HIP_PATH=${HIP_PATH:-<unset>}"
if command -v hipconfig >/dev/null 2>&1; then
  echo "train-env.sh: hipconfig -> $(command -v hipconfig) (--version: $(hipconfig --version 2>/dev/null || echo '<failed>'))"
fi

unset _TRAIN_ENV_SCRIPT_DIR _TRAIN_ENV_SHIM_DIR
