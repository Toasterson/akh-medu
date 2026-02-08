# Decision Log — 2026-02-08

## Phase 1 Foundation Implementation

### D1: Dropped `hypervector` and `bitvector_simd` crates
**What:** Removed these two crate dependencies from the plan.
**Why:** `bitvector_simd` depends on `packed_simd_2` which fails to compile on stable Rust (Rust 1.93+, edition 2024) due to broken transmute operations on pointer SIMD types. `hypervector` only has v0.1.x (plan specified v0.2 which doesn't exist). Neither crate is needed — we implement our own VSA operations with hand-written SIMD kernels.
**Alternatives considered:** Pinning to an older Rust version (rejected — we target edition 2024). Forking `packed_simd_2` (too much maintenance burden for Phase 1).

### D2: Dropped `rten-simd`, `opendiskmap`, `leapfrog` crates
**What:** Removed from dependencies.
**Why:** `rten-simd` was planned for ISA detection but `is_x86_feature_detected!` from std is sufficient. `opendiskmap` was planned for mmap hashmap but a simpler `memmap2`-based append-only store with in-memory index serves the same purpose with less complexity. `leapfrog` was planned for lock-free hashmaps but DashMap provides the same concurrent access pattern with better ecosystem support.
**Alternatives considered:** Using all three (rejected — added complexity without proportional benefit for Phase 1).

### D3: Added `anndists` as explicit dependency
**What:** Added `anndists = "0.1"` to Cargo.toml.
**Why:** `hnsw_rs` 0.3.x moved distance functions (DistHamming, DistL2, etc.) to a separate `anndists` crate. The import path is `anndists::dist::DistHamming`, not `hnsw_rs::dist::DistHamming`.

### D4: HNSW lifetime parameter
**What:** HNSW struct requires `'static` lifetime in our usage.
**Why:** `Hnsw<'b, T, D>` has a lifetime parameter for supporting memory-mapped data. Since we own all data in `Vec<u32>`, we use `'static`. Manual `Send + Sync` impls on `ItemMemory` since HNSW uses internal atomic synchronization.

### D5: Edition 2024 unsafe-in-unsafe-fn
**What:** Added explicit `unsafe {}` blocks inside `unsafe fn` implementations.
**Why:** Rust 2024 edition changes the default: the body of an `unsafe fn` is no longer implicitly an unsafe context. Each unsafe operation needs its own `unsafe {}` block. This is a breaking change from edition 2021.

### D6: Bit indexing convention for permute
**What:** Permute uses MSB-first indexing within each byte.
**Why:** The generic kernel's permute treats bit position 0 as the MSB of byte 0 (standard network byte order). This is consistent with how binary VSA literature typically orders components. The HyperVec `get_bit`/`set_bit` API uses LSB-first for individual component access (more natural for indexing), so these are two different views of the same data.

### D7: rkyv without `validation` feature
**What:** Using `rkyv = "0.8"` without the `validation` feature.
**Why:** rkyv 0.8 removed the `validation` feature — validation is now handled by the `bytecheck` crate separately. We include rkyv for future zero-copy deserialization but don't use validation in Phase 1.

### D8: Oxigraph in-memory for non-persistent mode
**What:** SparqlStore supports both in-memory and on-disk modes.
**Why:** When no `data_dir` is configured, the engine runs purely in-memory with no SPARQL store overhead. When persistent, oxigraph stores its data in a subdirectory under `data_dir/oxigraph/`.
