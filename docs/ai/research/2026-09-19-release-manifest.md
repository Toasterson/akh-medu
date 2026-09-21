# Recording what a native Ptah build actually is

Goal item 8 asked for a reproducible release bundle with recorded hashes and
exact kernel-library paths. This is the identity half: a manifest that says
precisely what a build consists of. It builds nothing, installs nothing and
deploys nothing, and no production daemon exists.

## Why hashing the executables is not enough

The HIP and Vulkan kernels are separate shared objects produced by build
scripts and opened at runtime. A binary paired with a different kernel library
is a different runtime, so an executable hash alone does not identify one.

Listing every `.so` in the build tree does not identify one either, and this
tree proves it:

| Library name | Copies present |
| --- | ---: |
| `libakh_hip_kernels.so` | 6 |
| `libcandle_hip_kernels.so` | 4 |
| `libcandle_hip_quant.so` | 4 |

Six identically named candidates with different hashes name nothing. This
ambiguity already caused a real error today: a symbol check looked in
`libakh_hip_kernels.so` and found the newly added cast missing, because the
change belonged to vendored candle-core's library, not akh-medu's.

## What the manifest records instead

`scripts/session-platform/release-manifest.py` resolves the pairing per binary.
Kernel lookup prefers an explicit `AKH_HIP_KERNEL_DIR`, then `lib/` beside the
executable, then a compile-time path baked into the binary. Absent an override
or a bundle, the baked path is what loads, so the manifest reads those paths
back out of each executable and hashes exactly those libraries.

For commit `b715a629`:

| Binary | SHA-256 prefix | Kernel libraries |
| --- | --- | ---: |
| akhomed | `e1d3e93e0883` | 0 |
| akhrtd | `ba34aba1a6d6` | 2 |
| akhsessd | `b1a32a157885` | 2 |
| akhsbxd | `ecd12746d88a` | 2 |
| akhexecd | `50dda7a498ae` | 2 |

`akhomed` resolving zero is correct: it is built with `core-server` and no HIP.
The other four each pair with `libcandle_hip_kernels.so` and
`libcandle_hip_quant.so` from `candle-core-e13219964f178351`, which is the
directory independently confirmed to contain `akh_cast_u8_u32`. The manifest
also lists the same-named duplicates under `ambiguous_same_named_libraries`, so
a reader sees the ambiguity rather than tripping over it.

Pinned artifacts are recorded by digest alongside: the Q4_K_M model, the
tokenizer, and the `qwen3vl_merger` projector.

The script refuses to emit a manifest from a dirty tree unless told otherwise,
because a hash that cannot be tied to a commit is not evidence.

## The bundle itself

`scripts/session-platform/build-bundle.py` assembles one: it copies the
executables and, for each, the kernel libraries that executable actually
resolves, into a single directory with a `lib/` beside them. Kernel lookup
prefers `lib/` next to the executable ahead of the compile-time path, so the
bundle needs no `AKH_HIP_KERNEL_DIR` override. It refuses to overwrite an
existing bundle, and refuses to proceed if two binaries need different libraries
of the same name.

It records the toolchain from **files rather than commands on PATH**. The first
version shelled out to `hipcc` and `rocm-smi`, neither of which is on PATH here,
and recorded `null` for all three fields — worse than recording nothing, because
`null` looks like an answer. Reading `/opt/rocm/.info/version` and the amdgpu
module attributes instead gives ROCm `6.3.4-76`, kernel `6.8.0-47-generic` and
the module srcversion, and anything genuinely unavailable is listed under
`unrecorded` rather than silently nulled.

**The relocated bundle passes the five-daemon canary**: all seven checks, the
correct post-restart recall answer, running the daemons from the bundle
directory rather than the build tree.

That verification took two attempts, and the first was worthless. It set
`AKH_NATIVE_BIN_DIR` and `PATH`, but the smoke harness takes `--bin-dir` and
defaults to `target/release`, so it ran the originals and passed. The binary
hashes matched the bundle's, which proves nothing when the bundle is a copy.
Only the rerun with `--bin-dir` actually exercised the relocated executables.

## Not established

The canary running from the bundle does not prove the kernels loaded from the
bundle's `lib/`: the compile-time path still exists on this machine, so lookup
could have reached the same libraries by the fallback. Establishing the `lib/`
precedence needs a machine, container or mount where the build tree is absent.

Nor does this attempt a cutover. No production daemon is installed, and the
homelab canary remains untouched pending explicit direction.

Manifest: [20260919](../../../experiments/native-release/20260919/manifest.json).
