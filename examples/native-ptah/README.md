# Ptah native-runtime canary

These are explicit **hephaistos-specific, not yet deployed** canary inputs.
Build with `cargo build --release -p akh-session-services --no-default-features
--features hip,vulkan --bin akhrtd`. The executable owns Candle inference in-process;
the `binary` profile field is a compatibility field, not a subprocess launcher.

An explicit `vulkan` build/device is also available. Exact-model numerical,
resident-idle and isolated four-daemon tool/restart checks have passed, but
Vulkan throughput is now about 13 decode steps/s versus roughly 20 on HIP;
the staged deployment remains HIP. These are earlier fresh-prefill measurements,
not the new keyed-continuation throughput. Keyed Qwen HIP completions now scope
activation arithmetic to f32: Q8-decode versus fresh-prefill recurrence failed
the continuation quality gate. Q4_K_M weights remain compressed; the process-wide
Q8 setting is not changed. Neither backend spawns an inference server.

For isolated HTTP integration, build `akhsessd`, `akhsbxd`, and `akhexecd` with
the `session,executor,sandbox` features, then run
`python3 scripts/session-platform/candle-native-smoke.py --help`. Supply the
exact artifact paths, hashes, and device. It uses private loopback services and
tests real model tools, approvals, receipts and restarts. It does not contact the
homelab, and removes only its temporary test workspace on exit.
Use `--device hip --switch-device vulkan` to additionally test a same-process
device roundtrip and recovery of the original HIP conversation before restart.
Add `--require-prefix-reuse` to require a measured nonzero native token-prefix
cache hit across ordinary turns. The test does not infer cache reuse merely
from a correct answer. The public `candle_session_probe --continuation` also
checks exact-prefix replies, full-context limits, session eviction, cancellation
and output-limit recovery before its resident-idle measurement.

The canary deliberately caps physical context at 2048 tokens pending flash
attention and live KV compression. The model advertises 262144 tokens; that is
not evidence this implementation can safely allocate that window. Input plus
reserved output exceeding the configured cap fails without truncating history.

Provision distinct random incoming/outgoing credentials, mode 0600. The runtime
stores only a SHA-256 verifier for the session daemon's incoming credential. Its
outgoing credential must identify `runtime-ptah` on the canonical session daemon,
scoped to Ptah callbacks. Never reuse an operator or another agent's token.

HIP deployment needs a matching kernel bundle, not just the executable. Preserve
`libcandle_hip_kernels.so` and `libcandle_hip_quant.so` from Candle's build output,
plus `libakh_hip_kernels.so` from akh-medu's build output. Obtain those exact output
directories from the successful Cargo build's JSON `build-script-executed`
records, not a glob that can select an older feature build. Place them in `lib/`
beside the installed executable and set `AKH_HIP_KERNEL_DIR` to that absolute
directory in the service environment. Explicit missing/relative overrides fail
closed; without an override, development builds may still use `OUT_DIR`.
Keep the executable and libraries from one build together, record their hashes,
and test the relocated bundle before switching a service. ROCm runtime/rocBLAS
remain host dependencies; this is not a fully static or cross-GPU binary.
The staged unit names `~/.local/lib/akh-medu/ptah-runtime/canary-v1` as its
not-yet-installed bundle. Use a new immutable directory for each build and update
both unit paths together; do not overwrite libraries under a running daemon or
point its explicit kernel directory through a mutable `current` symlink.

Before starting the unit: validate the built config contract, install the bundle,
create the writable runtime data directory, copy the config/unit, and verify
cluster-to-hephaistos connectivity. Do not overwrite an existing deployment or
credentials without inspecting it. Model files remain in their old artifact
store temporarily; SHA-256 pins prevent substitution. File migration is a
separate cutover step, not permission to delete Ollama's entire model store.

Canonical session configuration needs an additive `runtime-ptah` endpoint and
Ptah profile bound to runtime `runtime-ptah`, profile `ptah`, revision
`ptah-native-canary-v1`, kind `internal`, model `ptah-qwen38`, execution mode
`brokered`. Its workspace, cognitive profile and tool grants must be explicitly
configured; do not repoint the existing `tecton`/Kimi runtime. Exercise actual
session message, tool, resume and cancellation before retiring legacy artifacts.

Source/verification ledger:
[native source checkpoint](../../docs/ai/research/2026-09-15-native-source-checkpoint.md)
and [five-daemon canary](../../docs/ai/research/2026-09-15-native-core-daemon-canary.md).
