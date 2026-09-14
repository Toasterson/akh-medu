# 52a report — Sokar artifact daemon foundation

Date: 2026-09-14. Branch: `sokar/52a`. Implementer: Codex gpt-6-astra.
Authority: [52a brief](52a-brief.md), [ADR 108](../../decisions/108-sokar-artifact-daemon.md), [Phase 52 plan](../../plans/2026-09-14-phase52-sokar-artifact-daemon.md).
Anima umbrella: `01a0a00a-d3b0-7713-90cc-99b6fdac941e`. No slice WI identifier was supplied in the brief; no tracker mutations were made.

Implementation and local verification are delivered. **The workspace-wide formatting exit condition remains blocked by unchanged, out-of-scope baseline files.** The two new crates pass formatting and warning-denying Clippy; their full test suite passes, including a real disposable Postgres recovery run. Nothing was pushed or deployed.

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

Upload order is **staged bytes → final content-addressed blob → staging delete → artifact manifest → blob/artifact/idempotency rows → commit**. Collection manifests likewise precede rows. PostgreSQL's microsecond timestamp resolution is applied before serializing manifests, preserving exact round trips and cursor ordering. Readers use the index.

Staging runs outside the mutation lock. Finalization, latest-version selection, idempotency, deletion and reindex hold one global mutation lock: an advisory transaction lock in Postgres, or an owned mutex plus a rollbackable working copy in memory. This is deliberately conservative and coordinates multiple daemons sharing the same Postgres index. Separate independent indexes must not manage the same bucket prefix concurrently.

Deletion tombstones the row, removes the artifact manifest, decrements refcount, and synchronously deletes the blob when no references remain, while the index transaction is open. Before physical deletion it scans surviving manifests, so an orphan manifest from an earlier failed index commit protects its blob. Zero-refcount Postgres blob rows remain for tombstone foreign-key integrity. **52b replaces this synchronous scan/delete with the janitor.**

Object stores and Postgres cannot share an atomic transaction. A crash between bucket mutation and SQL commit can leave an orphan manifest or dangling row; explicit reindex repairs the index from surviving manifests. Staging errors clean up immediately, and cancellation schedules best-effort cleanup; process death can leave staging keys for the later janitor. This slice does not claim crash-atomic distributed commits.

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

For PostgreSQL, copy the example config, enable `[database] url = "env:SOKAR_DATABASE_URL"`, and supply the URL through the environment. Startup migrates automatically; `--migrate` runs migrations only. Migration bookkeeping is in schema `sokar`, so dropping that schema also removes SQLx's ledger and allows a real rebuild. `--check` validates configuration, connects/migrates/pings the configured database, and stats the format marker plus writes/deletes a probe on every store. `--check-stores` omits the database.

```sh
cargo run -p sokar --bin sokard -- --config /path/to/sokar.toml reindex --store local
```

The command prints a table of collections, artifacts and distinct blobs restored. In ephemeral mode the rebuilt index lasts for that command; normal server startup independently rebuilds it again.

Create a collection with `POST /v1/collections`:

```json
{"scope":{"type":"named","name":"demo"}}
```

All API calls require `Authorization: Bearer <contents of token file>`. Upload raw bytes to `/v1/collections/<uuid>/artifacts` with `Sokar-Path`, `Sokar-Kind` (for example `document`) and `Content-Type`. `Sokar-Labels` is an optional JSON string map; `Idempotency-Key` is optional. Multipart uses a `file` part and a `path` field, plus `Sokar-Kind`. Scope lookup uses `/v1/collections/by-scope?scope_key=<URL-encoded JSON CollectionScope>`; the typed client handles encoding.

## Validation

Verified with rustc 1.97.1 on macOS; OpenDAL resolved to 0.59.1.

| Check | Result |
|---|---|
| `cargo clippy -p sokar -p sokar-client --all-targets -- -D warnings` | Pass |
| `cargo test -p sokar -p sokar-client` with a disposable `SOKAR_TEST_DATABASE_URL` | 17 passed, 0 failed, 1 intentionally ignored MinIO test |
| `cargo fmt -p sokar -p sokar-client --check` | Pass |
| `cargo test -p sokar-client --no-default-features --features types-only` | Pass; two wire-type tests |
| Types-only normal dependency tree | No reqwest, rustls or URL dependency |
| `cargo fmt --all --check` | Exit 1: pre-existing formatting in eight unchanged files below |

The 100 MiB tests generate data in 64 KiB chunks, upload twice under different paths, verify `Deduplicated` and one physical filesystem blob, compare bytes 1000–1999, and test ETag/304. A separate test spawns the actual `sokard` executable, repeats the 100 MiB transfer/dedup/range workflow, kills/restarts it, and verifies reconstructed metadata.

The PostgreSQL test ran against a temporary, loopback-only `postgres:17` Docker container created for this slice, never against Anima. It exercised live reindex preserving idempotency, `DROP SCHEMA sokar CASCADE`, migrations, restoration of equal collection/artifact manifests, exact refcounts, repeated reindex and final-reference deletion. The container was removed after verification. When `SOKAR_TEST_DATABASE_URL` is unset, the test explicitly prints a skip message and returns successfully; it must point at a **disposable** database.

Additional tests cover schema refusal before decoding newer fields, rollback on invalid inventory, protection of orphan-manifest blobs, independent stores containing identical digests, conflicting imported UUIDs, concurrent idempotent uploads, body interruption and staging cleanup, chunked size limits, empty artifacts, multipart ordering/duplicates, Unicode paths, version chains, paging, bearer ambiguity/rotation/verifiers, admin separation and CLI diagnostics. The deliberately invalid store option test asserts exit code 2 and `sokar::storage` in miette stderr.

The MinIO S3 contract is `#[ignore]`, compiled but not executed. To run it, supply `SOKAR_MINIO_ENDPOINT`, `SOKAR_MINIO_BUCKET`, `SOKAR_MINIO_REGION`, `SOKAR_MINIO_ACCESS_KEY` and `SOKAR_MINIO_SECRET_KEY` for a local test bucket, then run `cargo test -p sokar --test recovery minio_s3_contract -- --ignored`. Cloud credentials are unnecessary for the ordinary suite. GCS/Azure and production deployment have build evidence only.

Workspace-wide formatting failures are confined to these unchanged baseline files:

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

1. **OpenDAL 0.59.1 APIs:** use `Operator::via_iter(service_name, options)` instead of the brief's generic builder syntax, `info().capability()` instead of removed `full_capability()`, and `reader_with(...).await?.into_bytes_stream(range)` instead of `reader_with(...).range(...)`. These are the pinned release's equivalent interfaces. Conditional copy is available and used where advertised. The memory backend has no copy operation, so it uses a chunked reader/writer relay under the same mutation lock. The brief's stat-then-copy fallback is used on copy-capable stores without conditional copy.
2. **Prometheus:** the pinned OpenDAL layer uses `prometheus-client` 0.25.1 and `PrometheusClientLayer::builder().register(...)`. Sokar matches that version. OpenDAL itself brings reqwest 0.13; the public client remains on the required reqwest 0.12.
3. **MSRV:** no root or existing workspace package declares `rust-version`. Both new crates explicitly use 1.91, OpenDAL 0.59.1's minimum, rather than inheriting a nonexistent workspace value. This run verified 1.97.1, not a separate 1.91 compiler.
4. **Manifest schema storage:** manifests use an owned `String` field with an associated `&'static str` schema constant. This permits deserialization of unknown tags and a typed `ManifestSchema` diagnostic; it avoids imposing a `'static` input lifetime on deserialized manifests. Unknown fields are tolerated, and schema checks run before typed decoding during reindex.
5. **Multi-store blob identity:** the brief's digest-only primary key cannot represent identical bytes independently stored in two configured stores. The implementation uses `(store,digest)` as the blob primary key and an artifact composite foreign key. Tombstones retain a zero-refcount blob index row after physical byte deletion. A JSON manifest column preserves portable records exactly alongside the required indexed fields. These are schema refinements for review, not cloud/backend limitations.
6. **Formatting gate:** the exact required workspace-wide command was run and fails on the untouched baseline files listed above. Scoped formatting passes; the full workspace exit condition cannot be made green within the authorized file scope.

## Commits

- `7c26aa32` — `feat(sokar): add artifact daemon and typed client`
- `1190657d` — `test(sokar): cover streamed delivery and index recovery`
- This report is committed separately as `docs(sokar): record slice 52a verification and deviations`.

Every commit ends with the requested Codex co-author trailer. No push was performed.

## Open questions for review

- Accept the scoped formatting evidence for this slice, or fix the baseline in a separate change before accepting the full exit condition?
- Confirm the `(store,digest)` key, retained zero-refcount rows for tombstones, and owned manifest schema tags as the intended foundation for 52b.
- Confirm the explicit local mode without `[database]`; deployment should configure Postgres for durable idempotency and coordination across replicas.
- The global mutation lock and manifest scan on final-reference deletion favor correctness over concurrent finalization throughput. Review these boundaries when adding the 52b janitor and production load tests.

No OIDC, cookies, scoped-token issuance, grants, presigned delivery selection, sites, A2UI, existing-crate integration, push or deployment is included.
