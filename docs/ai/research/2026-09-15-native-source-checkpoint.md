# Native runtime source checkpoint

This checkpoint consolidates the previously uncommitted implementation used by
the [HIP/Vulkan five-daemon canary](2026-09-15-native-core-daemon-canary.md).
It is source closure, not completion of the deployment/performance project.
No inference algorithm or device default is changed by consolidation.

Included: explicit native provider configuration, resident Candle integration,
continuation/cancellation, bounded capture and recall callbacks, HIP kernel
lookup and arithmetic guards, Vulkan storage/operations/quantized kernels,
required shader sources, model probe helpers, tokenizer extraction tooling and
the isolated tool/restart smoke test. The root Rust llama.cpp dependencies are
removed; `nlu-llm` remains a compatibility alias for `candle-backend`. Historical
reference-test names are not runtime dependencies.

The small documentation/call-site changes in extraction, training, anima-ai and
the runtime bridge belong to that dependency/configuration migration. Research
drafts and `experiments/native-vulkan` prototypes are deliberately not included;
they must not be required to compile or run the native runtime.

## Clean-source verification

Verification uses a detached worktree containing only committed files. Cargo's
existing dependency cache may be reused; this is not a byte-for-byte reproducible
or hermetic-toolchain claim. Commands must use `--locked --offline`. The core
must build separately without GPU features, avoiding feature unification of
Vulkan into its `ServerState` type graph:

```bash
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release -p akh-medu --no-default-features --features core-server --bin akhomed
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release -p akh-session-services --no-default-features --features session,executor,sandbox,hip,vulkan,core-integration-tests --bin akhsessd --bin akhrtd --bin akhsbxd --bin akhexecd
CARGO_BUILD_JOBS=2 cargo test --locked --offline --release -p akh-session-services --no-default-features --features core-integration-tests,hip,vulkan --lib -- --test-threads=1
```

Ignored GPU/artifact tests remain explicit opt-ins. For relocation, package
the three HIP kernel libraries from the successful build's exact Cargo JSON
`build-script-executed` output directories, alongside the executables. Do not
select an arbitrary old `target/release/build/*` result. The bundle must be
tested with an authoritative absolute `AKH_HIP_KERNEL_DIR` before deployment.

Still open: held-out recall/instruction safety, current keyed-path performance,
live KV compression, remaining model features, immutable release-bundle
acceptance, canonical local/homelab cutover and scoped legacy cleanup.
