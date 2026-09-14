# 52a brief — Sokar crate skeleton, stores, blobs, manifests, proxy delivery

Implementer: Codex `gpt-6-astra` (local `codex exec`), branch `sokar/52a`. Reviewer: Fable architect session.
Authority: [ADR 108](../../decisions/108-sokar-artifact-daemon.md) (D1, D2, D5 proxy path, D9, Grilling) and [plan §52a](../../plans/2026-09-14-phase52-sokar-artifact-daemon.md). Anima WI: see plan header (umbrella) + the 52a slice WI.

## Deliverable

Two new workspace crates, building and tested, committed on `sokar/52a`:

- `crates/sokar` — library + binary `sokard`.
- `crates/sokar-client` — wire types (`types-only` feature) + `reqwest` client.

Add both to the root `Cargo.toml` `[workspace] members`. Do not touch Anima, `akhomed`, `akh-session-services` or any existing crate beyond the members list. No OIDC, cookies, grants, tokens, sites or A2UI in this slice (those are 52b–52e); leave clearly named seams (`authz::Authorizer` trait with a `ServiceOnly` impl, `delivery::Delivery` enum with only `Stream` used).

## Conventions (non-negotiable)

- Edition 2024, `rust-version` matching the workspace; `#![forbid(unsafe_code)]` in both crates.
- Errors: `SokarError` derives `Debug, thiserror::Error, miette::Diagnostic`; every variant has `#[diagnostic(code(sokar::<area>), help("..."))]`; OpenDAL errors wrapped as `Storage { store: StoreName, op: &'static str, kind: opendal::ErrorKind, #[source] source: opendal::Error }` with kind-specific help text; `SokarResult<T>`. HTTP mapping → `application/problem+json` (`type`, `title`, `status`, `detail`, `code` = the diagnostic code, `request_id`).
- Strong types: newtypes for `StoreName`, `CollectionId(Uuid v7)`, `ArtifactId(Uuid v7)`, `BlobDigest([u8;32])` (hex Display, serde as hex string), `ArtifactPath` (validated: relative, no `..`, no leading `/`, NFC), `MediaType` (parsed with `mime`). Typed outcomes: `UploadOutcome::{Stored, Deduplicated}`, `Delivery::{Redirect(Url), Stream(StreamBody)}`.
- No natural-language string constants for matching. Nothing language-specific.
- Config: TOML, `#[serde(deny_unknown_fields)]`, secrets via `env:NAME` / `file:/path` indirection only (a `SecretRef` type with `resolve()`); `sokard --check` validates config + DB + every store (`stat` of `_sokar/format.json`, write+delete of a probe key) and prints miette reports, exit non-zero on failure.
- Tracing via `tracing` + `tracing-subscriber` (env filter, stderr); Prometheus on the admin listener via `opendal::layers::PrometheusClientLayer` + `prometheus-client`.
- Tests must not need cloud credentials: OpenDAL `memory` and `fs` services for unit/integration tests; Postgres tests use `SOKAR_TEST_DATABASE_URL` and are skipped (not failed) when unset. An `s3` contract test against local MinIO is `#[ignore]`.

## Dependencies (pin majors)

`opendal = "0.59"` (features: `services-s3`, `services-fs`, `services-gcs`, `services-azblob`, `layers-prometheus-client`), `axum = "0.8"`, `tower-http = "0.6"` (trace, cors, request-id, limit, set-header), `tokio`, `sqlx = "0.8"` (postgres, runtime-tokio-rustls, uuid, time, json, migrate), `uuid` (v7, serde), `sha2`, `hex`, `serde`, `serde_json`, `toml`, `clap` (derive), `miette` (fancy), `thiserror = "2"`, `tracing`, `tracing-subscriber`, `mime`, `bytes`, `futures`, `http-body-util`, `prometheus-client`, `time`. Client: `reqwest = "0.12"` (rustls), `url`.

## Store registry

```rust
pub struct StoreConfig { name: StoreName, service: ServiceKind /* S3|Gcs|Azblob|Fs|Memory */, prefix: Option<String>, options: BTreeMap<String, SecretRef> }
pub struct StoreCapability { presign_read: bool, presign_write: bool, ranged_read: bool, conditional_write: bool }
pub struct Store { name, op: opendal::Operator, cap: StoreCapability }
pub struct StoreRegistry { stores: HashMap<StoreName, Arc<Store>> }
```
Build with `Operator::via_iter::<services::S3>(resolved_options)` etc., then `.layer(RetryLayer)`, `.layer(TimeoutLayer)`, `.layer(ConcurrentLimitLayer)`, `.layer(TracingLayer)`, `.layer(PrometheusClientLayer)`; capture `op.info().full_capability()` into `StoreCapability`. For S3 always set `disable_config_load=true` and `disable_ec2_metadata=true` unless explicitly overridden. On first use write `_sokar/format.json` = `{"schema":"sokar.layout.v1","created_at":...}` if absent; refuse to serve a store whose format schema is newer than known.

## Bucket layout and manifests (ADR 108 D1)

```
_sokar/format.json
blobs/sha256/<aa>/<bb>/<hex>
artifacts/<uuid>/manifest.json      sokar.artifact.v1
collections/<uuid>/manifest.json    sokar.collection.v1
```
Manifest structs live in `sokar-client` (`types-only`), each with `schema: &'static str` const tag, serde round-trip tests, and `#[serde(deny_unknown_fields)]` disabled (forward-compatible read) but an explicit `schema` check that refuses unknown *newer* major versions with `SokarError::ManifestSchema`. Manifests include `blob.key` (store-relative key) for portability.

Upload path: stream body → `sha2` while writing to `uploads/<uuid>` (staging key) → `copy_with(final_key).if_not_exists(true)` when the store supports conditional write (else `stat` then copy) → delete staging → write artifact manifest → insert Postgres rows in one transaction (`blob` upsert with refcount+1, `artifact` insert). `ConditionNotMatch`/`AlreadyExists` on the final copy is the `Deduplicated` outcome, not an error. Write order bytes → manifest → row is a documented invariant.

## Postgres (schema `sokar`, sqlx migrations, advisory-locked)

Tables: `blob(digest bytea pk, size bigint, media_type text, store text, refcount int, created_at)`, `collection(id uuid pk, scope jsonb, scope_key text unique, owner text, store text, retention jsonb, quota jsonb, created_at)`, `artifact(id uuid pk, collection uuid fk, path text, kind text, digest bytea fk, media_type text, labels jsonb, created_by text, created_at, expires_at, previous uuid, unique(collection, path, created_at))`. Migration lock: `pg_advisory_lock(hashtext('sokar.migrations'))` before `sqlx::migrate!`.

`sokard reindex --store <name>`: list `collections/` and `artifacts/` manifests, upsert rows, recompute `blob.refcount`; report counts as a miette-friendly table; must restore everything after `DROP SCHEMA sokar CASCADE` (integration test on `fs` + Postgres when available, on `memory` + an in-memory index fake otherwise — implement the index behind a small `Index` trait so tests can run without Postgres).

## HTTP (API listener; admin listener separate)

Auth in this slice: service bearer tokens from config `[[principal]] id, token_file (or token_sha256_file)`, constant-time compare, exactly one match required (copy the pattern from `crates/akh-session-services/src/http.rs`, do not depend on that crate). All `/v1/*` routes require it.

- `POST /v1/collections` `{scope, store?, retention?, quota?}` → 201 collection; `GET /v1/collections/{uuid}`; `GET /v1/collections/by-scope?…` (scope_key lookup).
- `POST /v1/collections/{uuid}/artifacts` — raw streamed body with headers `Sokar-Path`, `Content-Type`, `Sokar-Kind`, optional `Sokar-Labels` (JSON), optional `Idempotency-Key` (24 h table) → 201 `{artifact, outcome}`; also `multipart/form-data` with a `file` part + `path` field.
- `GET /v1/artifacts/{uuid}` (metadata), `GET /v1/artifacts/{uuid}/content` (proxy stream: `Range` support via `reader_with(..).range(..)`, `ETag: "sha256-<hex>"`, `If-None-Match`, `Accept-Ranges: bytes`, `Cache-Control: private, immutable, max-age=31536000`, `Content-Disposition: inline` only for an allow-listed media-type set from config (`[delivery] inline_media_types = [...]`), else `attachment; filename*=UTF-8''<encoded basename>`).
- `GET /v1/collections/{uuid}/artifacts?prefix=&kind=&cursor=&limit=` (keyset paging by `(created_at,id)`), `GET /v1/collections/{uuid}/artifacts/by-path/{path}` (latest).
- `DELETE /v1/artifacts/{uuid}` → tombstone row + manifest delete + `refcount-1`; blob bytes deleted only when refcount hits 0 (janitor is 52b; in 52a delete the blob synchronously when refcount reaches 0 inside the same transaction, documented as replaced later).
- Admin listener: `GET /healthz`, `GET /readyz` (DB ping + every store `stat` of format marker), `GET /metrics`.
- `DefaultBodyLimit` from `[limits] max_body_bytes` (default 256 MiB); request ids via `tower-http`; `TraceLayer`.

## Client crate

`sokar_client::Client::new(base_url, bearer)`, methods mirroring the endpoints, returning the same wire types; errors as `sokar_client::Error` (miette Diagnostic) with the `problem+json` code surfaced.

## Tests / exit condition (all must pass before you finish)

1. `cargo fmt --all --check`, `cargo clippy -p sokar -p sokar-client --all-targets -- -D warnings`, `cargo test -p sokar -p sokar-client`.
2. Integration test on `fs`: upload 100 MiB (generated), upload the same bytes under another path → `Deduplicated`, one blob on disk; ranged GET of bytes 1000–1999 matches; `If-None-Match` → 304.
3. Reindex test: after wiping the index, `reindex` restores collections and artifacts with equal manifests and correct refcounts.
4. `sokard --check` with a deliberately wrong store option prints a miette diagnostic with code `sokar::storage` and exits 2.
5. Write `docs/ai/phases/phase-52-sokar/52a-report.md`: what landed, how to run it locally (`fs` store + optional Postgres), open questions for review. Commit everything on `sokar/52a` with conventional messages; do not push.
