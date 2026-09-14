# 52a report — Sokar artifact daemon foundation

Date: 2026-09-14. Branch: `sokar/52a`. Implementer: Codex gpt-6-astra.
Authority: [52a brief](52a-brief.md), [ADR 108](../../decisions/108-sokar-artifact-daemon.md), [Phase 52 plan](../../plans/2026-09-14-phase52-sokar-artifact-daemon.md).
Anima umbrella: `01a0a00a-d3b0-7713-90cc-99b6fdac941e`. No slice WI identifier was supplied in the brief; no tracker mutations were made.

Round 2 implementation and local verification are delivered. The reviewer accepted scoped formatting for 52a; the two crates pass formatting, warning-denying Clippy and the full test suite, including a fresh disposable PostgreSQL recovery and concurrency run. R1–R16 are addressed below. Nothing was pushed or deployed.

## What landed

- `crates/sokar`: library and `sokard` binary; separate authenticated API and unauthenticated admin listeners; `--check`, `--check-stores`, `--migrate`, and `reindex --store` commands; stderr tracing and OpenDAL Prometheus metrics labeled by store.
- `crates/sokar-client`: validated UUIDv7 resource IDs, store names, NFC relative artifact paths, SHA-256 digests, concrete MIME types, scopes, retention/quota records, versioned manifests, upload/page/problem records, and a streaming reqwest client mirroring every API endpoint. `default-features = false, features = ["types-only"]` excludes reqwest/TLS.
- S3, GCS, Azure Blob, filesystem and memory registry; retry, timeout, concurrency, tracing and Prometheus layers; effective capabilities; S3 ambient credential loading disabled by default; format-version checks and write/delete health probes. Unknown canonical backend option names fail with `sokar::storage` instead of being silently ignored.
- Streamed raw/multipart uploads, including file-before-path multipart order; body limits for known and unknown content lengths; incremental SHA-256; staged promotion and dedup; previous-version links; principal/collection-scoped 24-hour idempotency receipts with content-and-metadata fingerprint conflict checks.
- Collection UUID/scope lookup, artifact UUID/latest-path lookup, prefix/kind filtering, `(created_at,id)` keyset paging; streamed ranges (closed, open-ended, suffix), 206/304/416, ETags, If-Range, private immutable cache headers, MIME allow-list disposition and encoded attachment filenames. Metadata and filenames never choose object-store keys.
- File-reloaded bearer authentication with raw token or SHA-256 verifier, constant-time comparisons, exactly-one-match requirement, and denial if a credential source is unavailable. `authz::Authorizer` / `ServiceOnly` and `delivery::Delivery::{Redirect,Stream}` are the named follow-on seams. Only `Stream` is selected in 52a.
- `application/problem+json` with diagnostic codes and request IDs also covers authentication, extractors, router errors and body-limit rejections.
- Advisory-locked SQLx migrations and a transactional `Index` interface with Postgres and in-memory implementations; store-scoped reindex validates the entire inventory before replacing rows, recomputes refcounts, and preserves existing unexpired idempotency receipts for restored artifacts. Reindex after total database loss restores resource manifests, not credentials/receipts that never belonged in the bucket.

Only the two new crates, root workspace members, lockfile and this report changed.

## Storage and transaction invariants

Upload order is **staged bytes → final content-addressed blob → artifact manifest → blob/artifact/idempotency rows → commit**, followed by explicit staging cleanup on success or failure. The invariant remains **bytes → manifest → row**. Collection manifests likewise precede rows. PostgreSQL's microsecond timestamp resolution is applied before serializing manifests, preserving exact round trips and cursor ordering. Readers use the index.

Staging, promotion and artifact manifest writes run outside index mutation locks. Finalization reads the previous version before writing the manifest, then validates `latest()` under the store lock. If another finalizer won, it releases the transaction and rewrites the manifest with the new predecessor and monotonic timestamp before retrying. A live idempotency receipt wins only after checking its artifact; the unused new manifest is removed after releasing the lock, while content-addressed bytes remain.

Artifact finalization and deletion take the per-store advisory lock `hashtext('sokar.store:' || store)`. Collection creation and the reindex swap take the global `sokar.mutations` lock exclusively. Artifact transactions first take that global lock **shared**, then their store lock exclusively, so independent stores can mutate concurrently while global changes exclude all artifact mutations. Memory mirrors this with a global reader/writer lock, one mutex per store, and rollbackable snapshots merged only for the locked store. **Lock order: global before store; never acquire global while holding store.** Memory's short data mutex protects snapshot/commit operations, not the lifetime of a store transaction.

Deletion first rewrites the artifact manifest with optional `deleted_at`; only then does it take the store lock, tombstone the row, decrement refcount, purge every receipt pointing at the artifact, and commit. **Neither the manifest key nor blob bytes are deleted in the request path.** Zero-refcount blob index rows remain for tombstone foreign-key integrity. Reindex restores tombstones without adding references or requiring their blob bytes to exist. The brief's synchronous deletion instruction is superseded by ADR 108's reconcile-before-GC rule and the Round 2 review. Physical manifest/blob collection, staging sweeps and receipt expiry remain 52b janitor work.

Reindex lists, reads, checks schemas and validates live blob sizes before acquiring the global lock; only replacement and commit run under it. Duplicate scope manifests are resolved by ascending UUIDv7: the earliest survives in the index, the rest are retained in storage, counted as `orphans`, and warned about. Artifacts belonging to a losing collection are likewise retained as counted/warned orphans, without rewriting their portable identities. Counts report live artifacts and live distinct blobs; tombstone rows are restored but add no live counts.

Object stores and PostgreSQL cannot share an atomic transaction. A crash between a bucket write and SQL commit can leave an orphan manifest or an index row awaiting tombstone reconciliation; bytes remain available and explicit reindex repairs from manifests. The inventory scan is not an atomic snapshot of concurrent bucket writes. Staging errors preserve the original error while logging cleanup failures; cancellation schedules best-effort cleanup, and Drop without a runtime logs the staging key. Process death can leave staging keys for 52b. This slice does not claim crash-atomic distributed commits.

Non-conditional promotion retains the stat-then-copy TOCTOU window. It is safe only because a digest names byte-identical content. Index locks do not serialize bucket writes across independent indexes, and promotion deliberately runs outside those locks. Separate independent indexes must not manage the same prefix; the bucket owner marker remains deferred to 52b.

## Local run

From the workspace root, create a demo credential and filesystem root:

```sh
mkdir -p /tmp/sokar-local/objects
chmod 700 /tmp/sokar-local
openssl rand -hex 32 > /tmp/sokar-local/token
chmod 600 /tmp/sokar-local/token
export SOKAR_FS_ROOT=/tmp/sokar-local/objects
cargo run -p sokar --bin sokard -- --config crates/sokar/examples/local.toml --check
cargo run -p sokar --bin sokard -- --config crates/sokar/examples/local.toml
```

API: `127.0.0.1:9400`; admin: `127.0.0.1:9401`. The example omits `[database]`, so startup rebuilds an ephemeral index from the filesystem manifests. Idempotency receipts disappear when that process stops. Use a single daemon in this mode.

For PostgreSQL, copy the example config, enable `[database] url = "env:SOKAR_DATABASE_URL"`, and supply the URL through the environment. Startup migrates automatically; `--migrate` runs migrations only. Migration bookkeeping is in schema `sokar`, so dropping that schema also removes SQLx's ledger and allows a real rebuild. `--check` validates configuration, connects/migrates/pings the configured database, and reads the format marker once plus writes/deletes a probe on every store. `--check-stores` omits the database.

```sh
cargo run -p sokar --bin sokard -- --config /path/to/sokar.toml reindex --store local
```

The command prints a table of collections, live artifacts, live distinct blobs and orphans. In ephemeral mode the rebuilt index lasts for that command; normal server startup independently rebuilds it again.

Create a collection with `POST /v1/collections` (201 for creation; 200 for an existing compatible scope; 409 `ScopeExists` for a different explicitly requested store):

```json
{"scope":{"type":"named","name":"demo"}}
```

All API calls require `Authorization: Bearer <contents of token file>`. Upload raw bytes to `/v1/collections/<uuid>/artifacts` with `Sokar-Path`, `Sokar-Kind` (for example `document`) and `Content-Type`. `Sokar-Labels` is an optional JSON string map; `Idempotency-Key` is optional. Multipart uses a `file` part and a `path` field, plus `Sokar-Kind`. Scope lookup uses `/v1/collections/by-scope?scope_key=<URL-encoded JSON CollectionScope>`; the typed client handles encoding.

## Validation

Verified with rustc 1.97.1 on macOS; OpenDAL resolved to 0.59.1.

| Check | Result |
|---|---|
| `cargo fmt -p sokar -p sokar-client -- --check` | Pass (Round 2) |
| `cargo clippy -p sokar -p sokar-client --all-targets -- -D warnings` | Pass (Round 2) |
| `cargo test -p sokar -p sokar-client` with a disposable `SOKAR_TEST_DATABASE_URL` | 25 passed, 0 failed, 1 intentionally ignored MinIO test (Round 2) |
| `cargo test -p sokar-client --no-default-features --features types-only` | Pass; three wire-type tests (Round 2) |
| Types-only normal dependency tree | No reqwest, rustls or URL dependency (Round 1 evidence) |
| `cargo fmt --all --check` | Round 1 baseline failure; scoped gate explicitly accepted by review, not rerun or repaired in Round 2 |

The 100 MiB tests generate data in 64 KiB chunks, upload twice under different paths, verify `Deduplicated` and one physical filesystem blob, compare bytes 1000–1999, and test ETag/304. A separate test spawns the actual `sokard` executable, repeats the 100 MiB transfer/dedup/range workflow, kills/restarts it, and verifies reconstructed metadata.

The PostgreSQL test ran against a temporary, loopback-only `postgres:17` Docker container created for this slice, never against Anima. It exercised live reindex preserving idempotency, `DROP SCHEMA sokar CASCADE`, migrations, restoration of equal collection/artifact manifests, exact refcounts, repeated reindex and final-reference tombstoning with retained bytes and refcount zero. The fresh Round 2 run also exercised receipt purge/replay after deletion, stale-receipt fallback, tombstone recovery after byte removal, typed scope collisions, concurrent mutations of two stores, and global/store exclusion. The container was removed after verification. When `SOKAR_TEST_DATABASE_URL` is unset, the test explicitly prints a skip message and returns successfully; it must point at a **disposable** database.

Additional tests cover schema refusal before decoding newer fields, rollback on invalid inventory, retention of orphan-manifest blobs, duplicate-scope recovery, independent stores containing identical digests, conflicting imported UUIDs, concurrent idempotent uploads, body interruption and staging cleanup, chunked size limits, empty artifacts, multipart ordering/duplicates, Unicode paths, version chains, paging, bearer ambiguity/rotation/verifiers, admin separation and CLI diagnostics. The deliberately invalid store option test asserts exit code 2 and `sokar::storage` in miette stderr.

The MinIO S3 contract is `#[ignore]`, compiled but not executed. To run it, supply `SOKAR_MINIO_ENDPOINT`, `SOKAR_MINIO_BUCKET`, `SOKAR_MINIO_REGION`, `SOKAR_MINIO_ACCESS_KEY` and `SOKAR_MINIO_SECRET_KEY` for a local test bucket, then run `cargo test -p sokar --test recovery minio_s3_contract -- --ignored`. Cloud credentials are unnecessary for the ordinary suite. GCS/Azure and production deployment have build evidence only.

The Round 1 workspace-wide formatting failures were confined to these unchanged baseline files (the reviewer accepted scoped formatting and assigned baseline chores elsewhere):

- `crates/akh-session-services/src/server/sessions.rs`
- `crates/akh-session-services/src/session.rs`
- `examples/akh_causal_episode_bridge.rs`
- `examples/akh_symbolic_episode_bridge.rs`
- `examples/akh_temporal_incident_bridge.rs`
- `examples/akh_temporal_validity_probe.rs`
- `src/agent/event_calculus.rs`
- `src/engine.rs`

The user's explicit scope restriction forbids repairing those files in this slice. No broad formatter write was run.

## Deviations

1. **OpenDAL 0.59.1 APIs:** use `Operator::via_iter(service_name, options)` instead of the brief's generic builder syntax, `info().capability()` instead of removed `full_capability()`, and `reader_with(...).await?.into_bytes_stream(range)` instead of `reader_with(...).range(...)`. These are the pinned release's equivalent interfaces. Conditional copy is available and used where advertised. The memory backend has no copy operation, so it uses a chunked reader/writer relay outside the index lock. The brief's stat-then-copy fallback is used on copy-capable stores without conditional copy.
2. **Prometheus:** the pinned OpenDAL layer uses `prometheus-client` 0.25.1 and `PrometheusClientLayer::builder().register(...)`. Sokar matches that version. OpenDAL itself brings reqwest 0.13; the public client remains on the required reqwest 0.12.
3. **MSRV:** no root or existing workspace package declares `rust-version`. Both new crates explicitly use 1.91, OpenDAL 0.59.1's minimum, rather than inheriting a nonexistent workspace value. This run verified 1.97.1, not a separate 1.91 compiler.
4. **Manifest schema storage:** manifests use an owned `String` field with an associated `&'static str` schema constant. This permits deserialization of unknown tags and a typed `ManifestSchema` diagnostic; it avoids imposing a `'static` input lifetime on deserialized manifests. Unknown fields are tolerated, and schema checks run before typed decoding during reindex.
5. **Multi-store blob identity:** the brief's digest-only primary key cannot represent identical bytes independently stored in two configured stores. The implementation uses `(store,digest)` as the blob primary key and an artifact composite foreign key. Tombstones retain a zero-refcount blob index row; Round 2 retains physical bytes for the 52b janitor. A JSON manifest column preserves portable records exactly alongside the required indexed fields. These are schema refinements for review, not cloud/backend limitations.
6. **Formatting gate:** the Round 1 workspace-wide command failed on the untouched baseline files listed above. The reviewer explicitly accepted scoped formatting for this slice; Round 2 uses that accepted gate.

## Commits

- `7c26aa32` — `feat(sokar): add artifact daemon and typed client`
- `1190657d` — `test(sokar): cover streamed delivery and index recovery`
- `93addcbb` — `docs(sokar): record slice 52a verification and deviations` (Round 1)
- `fbbcc07a` — `fix(sokar): address round two lifecycle and storage review` (R1–R8, R10–R16, regression tests)
- Round 2 report commit: `docs(sokar): record round two fixes and verification` (R9).

Every commit ends with the requested Codex co-author trailer. No push was performed.

## Round 2

Authority: [52a-review.md, Round 2 required fixes](52a-review.md). No deferred 52b feature was implemented.

| Item | Change and evidence |
|---|---|
| R1 | Promotion and artifact manifest writes precede `index.begin`; optimistic latest-version validation retries manifest writes outside the lock. Receipt short-circuits remove the promoted-but-unused manifest after releasing the lock, preserving bytes. Existing version-chain and concurrent idempotency tests pass. |
| R2 | Optional `deleted_at` in artifact manifests; bucket tombstone before row/refcount/receipt commit. No manifest scan or physical delete in `delete_artifact`. Both backends rebuild tombstones without live blob references. Delete/recovery tests retain bytes at refcount zero and also recover when tombstone-only bytes are absent. |
| R3 | Inventory listing, decoding, validation and live blob stats run before acquiring the global reindex swap lock. No format recheck under the lock. Recovery and atomic invalid-inventory tests pass. |
| R4 | Exhaustive inherent `SokarError::code()` replaces the diagnostic `expect`. Unit coverage checks miette diagnostic code equality and problem codes for every variant, including each transparent client error. |
| R5 | Transparent `Client(#[from] sokar_client::Error)` preserves typed inner errors and diagnostic chaining. Ordinary client errors map to 400; inner schema errors remain 500. Transparent codes retain the client's `sokar_client::*` namespace. The client's exhaustive code helper is named `diagnostic_code` to avoid shadowing the miette transparent derive's trait call. |
| R6 | `UnknownStore` (500, `sokar::config`), `ScopeExists`, `IdempotencyMismatch`, and `StoreMismatch` (409) replace the corresponding string errors. Memory checks scope collisions explicitly; PostgreSQL maps the `collection_scope_key_key` unique violation. Both backend contracts verify typed scope collision. |
| R7 | `subtle::ConstantTimeEq` replaces the hand-written loop; exactly-one-match remains. Credential files use trim, minimum length 24 and no internal whitespace, matching session-services, plus the existing upper bound/verifier checks. Credential rotation, verifier, ambiguity and format tests pass. |
| R8 | `Store::check` reads the marker once. `read_json` no longer stats before reading; size rejection remains after read. CLI check/readiness tests pass. |
| R9 | Updated invariants, deletion guidance, review disposition, validation and this per-item account. Scoped gates are green; no out-of-scope formatting changes. |
| R10 | Stage/finalization error cleanup logs failures without replacing the original error; finalization explicitly cleans staging on every return, including mismatch/replay. Drop without a runtime warns with the staging key. Failed-stream, oversized-body, replay and mismatch cleanup tests pass. |
| R11 | Range handling and `Accept-Ranges` depend on the range-class capability. A fake capability with `read=true` and range support false proves valid, malformed and unsatisfiable Range headers yield a full 200 response with no range headers. Enabling the range flag yields 206 and the requested bytes. |
| R12 | `promote()` documents non-conditional stat/copy TOCTOU safety as byte-identical digest content, and explicitly states that independent indexes' bucket writes are not serialized. No owner marker added. |
| R13 | Collection creation is create-or-get with 201/200/409 HTTP tests. Reindex sorts UUIDv7 IDs and retains the earliest scope, warning/counting duplicate manifests as orphans instead of failing. Duplicate-scope reindex is tested twice for deterministic recovery. |
| R14 | Added `purge_idempotency_for`; deletion purges receipts in the row transaction. Replay validates the referenced artifact before fingerprint matching; missing/deleted artifacts fall through to fresh upload. Memory and real PostgreSQL test purge, replay after delete, stale receipts, and explicit cleanup. |
| R15 | `ManifestSchema { kind, major }` parses `sokar.<kind>.v<major>` while manifest fields remain owned strings. Equal major and kind are accepted; newer/unequal major, wrong kind and malformed tags return schema diagnostics. Unit tests cover artifact v1/v2 and collection v1, plus malformed inputs and tombstone serde compatibility. |
| R16 | Shared-global then exclusive-store locks for finish/delete; exclusive-global for create/reindex. Memory snapshots merge only the locked store to avoid losing concurrent commits. Tests on memory and PostgreSQL hold store A open while B mutates and commits, verify both artifacts survive, and verify same-store/global exclusion. |

Round 2 API adjustment for R11: pinned OpenDAL 0.59.1 has no `read_with_range` field. Its available range-class flag is `read_with_suffix`, which Sokar uses conservatively instead of `cap.read`. OpenDAL's default `SimulateLayer` exposes suffix availability for filesystem stores, so the 100 MiB ranged filesystem tests stay green. This matches the installed source and [OpenDAL's capability documentation](https://opendal.apache.org/docs/rust/opendal/struct.Capability.html). No dependency upgrade was needed. Other implementation choices above preserve the required semantics; no required fix was deferred.

The accepted `(store,digest)` key, retained zero-refcount rows, owned schema strings, per-request credential reload, validated portable `BlobRef.key`, timestamp/UUID ordering and delivery/authz seams remain. The local example now explicitly documents single-daemon ephemeral mode. Per-resource locks, bucket owner marker, janitor, OIDC, cookies, scoped-token issuance, grants, presigned delivery selection, sites, A2UI and existing-crate integration remain outside this slice.
