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

## Not established

This is identity, not deployment. It does not build a relocatable bundle, copy
libraries into a `lib/` directory beside the executables, or verify that a
relocated bundle runs — the earlier relocation canary did that for an older
build and its evidence stands separately. Nor does it record the ROCm or driver
versions the kernels were compiled against, which belong in a bundle intended to
move between machines.

Manifest: [20260919](../../../experiments/native-release/20260919/manifest.json).
