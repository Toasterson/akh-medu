#!/usr/bin/env bash
# Build and install llama.cpp's llama-server for Lilith on hephaistos (ADR 121). Idempotent.
#
#   bash scripts/lilith/hephaistos/build_server.sh                 # HIP/ROCm build for gfx1100
#   bash scripts/lilith/hephaistos/build_server.sh --backend vulkan
#   bash scripts/lilith/hephaistos/build_server.sh --check         # verify only, change nothing
#   bash scripts/lilith/hephaistos/build_server.sh --install-deps  # apt-get the build tools first
#
# The source is the exact commit pinned in pins.json (llama.cpp v0.5.0), fetched by SHA and
# verified; the install goes to $PREFIX/llama.cpp/<commit12>-<backend>/ and $PREFIX/current
# points at it. A build for the same commit and backend that already passes its checks is
# reused. Refusals (fail closed): no ROCm/HIP toolchain, wrong GPU target, a build whose
# llama-server lacks a flag the service needs, unresolved shared libraries.
set -euo pipefail
umask 022
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND=hip
CHECK_ONLY=0
INSTALL_DEPS=0
JOBS="${LILITH_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
# Flags run_server.sh passes; a build without one of them would fail at service start.
REQUIRED_FLAGS=(--api-key-file --jinja --chat-template-file --reasoning-format --kv-unified-per-slot
  --parallel --no-slots --no-webui --alias --cache-ram --flash-attn)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    -h | --help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die lilith::heph::usage "unknown argument '$1'" "see --help" ;;
  esac
done
[[ "$BACKEND" == hip || "$BACKEND" == vulkan ]] || die lilith::heph::usage "backend '$BACKEND' is not hip or vulkan" \
  "use --backend hip (default) or --backend vulkan (fallback)"

COMMIT="$(pin llama_cpp.commit)"
GPU_TARGET="$(pin llama_cpp.gpu_target)"
DEST="$PREFIX/llama.cpp/${COMMIT:0:12}-$BACKEND"
SRC="$WORK/llama.cpp"
BUILD="$WORK/build-$BACKEND"

if [[ "$INSTALL_DEPS" == 1 ]]; then
  log "installing build tools with apt-get (cmake, ninja, git, python3-venv, build-essential)"
  as_root apt-get install -y cmake ninja-build git python3-venv build-essential pkg-config
  if [[ "$BACKEND" == vulkan ]]; then as_root apt-get install -y libvulkan-dev glslc mesa-vulkan-drivers; fi
fi

preflight() {
  need_cmd cmake "bash $0 --install-deps"
  need_cmd git "bash $0 --install-deps"
  need_cmd c++ "bash $0 --install-deps (build-essential)"
  if [[ "$BACKEND" == hip ]]; then
    need_cmd hipconfig "install ROCm (the training venv host already has it: ls /opt/rocm*); add /opt/rocm/bin to PATH"
    need_cmd rocminfo "install rocminfo with ROCm"
    rocminfo 2>/dev/null | grep "$GPU_TARGET" >/dev/null || die lilith::heph::gpu \
      "rocminfo does not list a $GPU_TARGET device" \
      "this build targets the RX 7900 XTX; check the driver (dmesg | grep amdgpu), or use --backend vulkan"
    [[ -f "$(hipconfig -R)/include/hipblas/hipblas.h" ]] || die lilith::heph::rocm \
      "hipBLAS headers are missing under $(hipconfig -R)" "sudo apt-get install -y hipblas-dev rocblas-dev"
  else
    need_cmd glslc "bash $0 --backend vulkan --install-deps"
  fi
}

check_install() { # check_install DIR: the installed build is complete and correct
  local dir="$1" server="$1/bin/llama-server" help missing=() flag unresolved
  [[ -x "$server" ]] || return 1
  [[ -f "$dir/BUILD.json" ]] || return 1
  python3 - "$dir/BUILD.json" "$COMMIT" "$BACKEND" <<'PY' || return 1
import json, sys
build = json.load(open(sys.argv[1]))
sys.exit(0 if build.get("commit") == sys.argv[2] and build.get("backend") == sys.argv[3] else 1)
PY
  export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if command -v ldd >/dev/null 2>&1; then
    unresolved="$(ldd "$server" 2>/dev/null | grep 'not found' || true)"
    [[ -z "$unresolved" ]] || die lilith::heph::libs "llama-server has unresolved libraries: $unresolved" \
      "add the ROCm library directory to the linker path: echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/rocm.conf && sudo ldconfig"
  fi
  help="$("$server" --help 2>&1 || true)"
  for flag in "${REQUIRED_FLAGS[@]}"; do
    grep -q -- "$flag" <<<"$help" || missing+=("$flag")
  done
  [[ ${#missing[@]} -eq 0 ]] || die lilith::heph::flags "the built llama-server lacks: ${missing[*]}" \
    "the pinned commit $COMMIT should have them; delete $dest_note and rebuild, or check pins.json"
  return 0
}
dest_note="$DEST"

if check_install "$DEST"; then
  log "llama-server ${COMMIT:0:12} ($BACKEND) is already installed at $DEST and passes its checks"
  if [[ "$CHECK_ONLY" == 0 ]]; then
    as_root install -d -m 755 "$PREFIX"
    as_root ln -sfn "$DEST" "$PREFIX/current"
    log "$PREFIX/current -> $DEST"
  fi
  exit 0
fi
[[ "$CHECK_ONLY" == 0 ]] || die lilith::heph::not_built "no complete build of ${COMMIT:0:12} ($BACKEND) at $DEST" \
  "run: bash $0 --backend $BACKEND"

preflight
need_free_gib "$WORK" 6 "the llama.cpp source and build"
ensure_llama_source "$SRC"

# A static, hermetic build: one llama-server binary (ggml and llama linked in), no prebuilt web UI
# fetched from Hugging Face at build time (the service runs --no-webui), no TLS stack (the LAN
# endpoint is plain HTTP by decision, ADR 121), no compiler cache. ROCm's own libraries stay
# dynamic (/opt/rocm/lib).
CMAKE_ARGS=(-S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
  -DBUILD_SHARED_LIBS=OFF -DGGML_CCACHE=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF)
if command -v ninja >/dev/null 2>&1; then CMAKE_ARGS+=(-G Ninja); fi
if [[ "$BACKEND" == hip ]]; then
  CMAKE_ARGS+=(-DGGML_HIP=ON "-DGPU_TARGETS=$GPU_TARGET")
  HIPCXX="$(hipconfig -l)/clang"
  HIP_PATH="$(hipconfig -R)"
  export HIPCXX HIP_PATH
else
  CMAKE_ARGS+=(-DGGML_VULKAN=ON)
fi
# Extra cmake arguments for a toolchain quirk (never for behaviour): LILITH_CMAKE_EXTRA="-DX=Y ...".
# shellcheck disable=SC2206
[[ -z "${LILITH_CMAKE_EXTRA:-}" ]] || CMAKE_ARGS+=(${LILITH_CMAKE_EXTRA})

log "configuring ($BACKEND, ${GPU_TARGET}); build dir $BUILD"
cmake "${CMAKE_ARGS[@]}" > "$WORK/cmake-$BACKEND.log" 2>&1 || die lilith::heph::cmake \
  "cmake configure failed" "see $WORK/cmake-$BACKEND.log (tail -n 40); a missing ROCm device library needs HIP_DEVICE_LIB_PATH (llama.cpp docs/build.md)"
log "building with $JOBS jobs (a HIP build takes 10-25 minutes)"
cmake --build "$BUILD" --config Release -j "$JOBS" --target llama-server llama-bench \
  > "$WORK/build-$BACKEND.log" 2>&1 || die lilith::heph::build "build failed" "see $WORK/build-$BACKEND.log (tail -n 60)"

STAGE="$WORK/stage-$BACKEND"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin"
for binary in llama-server llama-bench; do
  [[ -x "$BUILD/bin/$binary" ]] || die lilith::heph::build "$BUILD/bin/$binary was not produced" \
    "see $WORK/build-$BACKEND.log"
  cp "$BUILD/bin/$binary" "$STAGE/bin/$binary"
done
ROCM_VERSION="$(cat "$(hipconfig -R 2>/dev/null || echo /nonexistent)/.info/version" 2>/dev/null || echo none)"
python3 - "$STAGE/BUILD.json" "$COMMIT" "$(pin llama_cpp.tag)" "$BACKEND" "$GPU_TARGET" "$ROCM_VERSION" \
  "$(python3 "$HEPH_DIR/heph_tools.py" sha256 "$STAGE/bin/llama-server")" <<'PY'
import datetime, json, platform, sys
out, commit, tag, backend, gpu, rocm, sha = sys.argv[1:8]
with open(out, "w") as handle:
    json.dump({"schema": "lilith/llama-server-build@1", "commit": commit, "tag": tag, "backend": backend,
               "gpu_target": gpu if backend == "hip" else None, "rocm": rocm if backend == "hip" else None,
               "llama_server_sha256": sha, "host": platform.node(),
               "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
              handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

as_root install -d -m 755 "$PREFIX" "$PREFIX/llama.cpp"
as_root rm -rf "$DEST.partial"
as_root cp -a "$STAGE" "$DEST.partial"
as_root rm -rf "$DEST"
as_root mv "$DEST.partial" "$DEST"
check_install "$DEST" || die lilith::heph::verify "the installed build failed its own checks" "see the message above"
as_root ln -sfn "$DEST" "$PREFIX/current"
log "installed $DEST (llama.cpp $(pin llama_cpp.tag), $BACKEND); $PREFIX/current -> $DEST"
if [[ "$BACKEND" == hip ]]; then
  "$DEST/bin/llama-server" --list-devices 2>&1 | sed 's/^/  device: /' || true
fi
