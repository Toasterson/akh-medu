# Bounded Vulkan pipeline residency

## Finding and change

The source-checkpoint review found a device-lifetime map retaining every
shape/offset-specialized shader and pipeline. This is a static application
retention defect, not a measured explanation for GPU idle power or driver OOM.

Dispatch now uses a private, CPU-testable cache with least-recently-used
eviction. Production limits are 2,048 entries and 16 MiB of source text.
These are provisional engineering limits, not full-model performance-tuned
values. Hits update recency in O(1); eviction scans the bounded map only on
misses. No dependency or runtime configuration change is required.

An oversized shader executes without being cached. Compilation errors do not
insert entries or evict valid ones. Eviction drops cache ownership, not handles
held by callers or pending GPU commands. The existing dispatch mutex scope is
preserved. The pinned local wgpu-core 29 implementation's compute `set_pipeline`
retains an Arc in the command resource tracker; actual pending-command eviction
is tested as well.

This bounds application-held cache entries and source length, **not** compiled
pipeline bytes, transient compilation memory, pending-command resources, or the
driver's internal cache. No immediate driver-memory release claim is made.

## Feedback loop

First extracted the dispatch cache while preserving unbounded insertion and
ran its CPU regression tests. Three tests failed: three entries retained with
a two-entry limit, five source bytes retained with a four-byte limit, and
retention despite zero capacity. After adding eviction, those tests passed.
This deliberately targets the review finding, not the original fan symptom.
Broad hypothesis search was skipped because the missing eviction path was
directly established and reproduced.

Final tests cover entry and byte limits, recency, 1,000 changing keys,
oversized bypass, compilation failure, zero capacity, ownership release,
multiple evictions, and recency-counter wrap.

Build/run the vendor tests with:

```sh
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target" cargo test \
  --manifest-path vendor/candle-core/Cargo.toml --offline --release \
  --no-default-features --features vulkan --lib \
  vulkan_backend::pipeline_cache:: -- --test-threads=1
```

On local RX 7900 XTX, the ignored GPU regression
`vulkan_backend::tests::pipeline_eviction_preserves_pending_commands` passed
with all three configurations:

- `AKH_VULKAN_BATCH_ENCODER=1 AKH_VULKAN_BATCH_PASS=1`
- `AKH_VULKAN_BATCH_ENCODER=1 AKH_VULKAN_BATCH_PASS=0`
- `AKH_VULKAN_BATCH_ENCODER=0 AKH_VULKAN_BATCH_PASS=0`

The test forces a two-entry cache and dispatches 130 distinct affine shaders
without intervening readback. The exact result is 8,515, followed by 8,516 after
recompiling an evicted specialization. The three existing shared-pass/encoder
correctness and failure-containment GPU tests also passed serially.

No shared services, HIP code, model artifacts, or external machines changed.

## Remaining acceptance

The full model has not been tested against this cache bound. Next measure cache
hits/misses/evictions, RSS/VRAM and resident idle power over varied long-lived
sessions, and compare keyed throughput before/after. A poor hot working-set fit
could cause recompilation and require parameterizing shaders or revising the
limit. Then rebuild the daemons and rerun exact-model five-daemon acceptance
before packaging or deployment. This change alone does not complete runtime
performance, memory, idle-power, or homelab cutover acceptance.
