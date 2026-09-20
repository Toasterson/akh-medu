# Phase 52: Sokar — dedicated artifact daemon

- **Status**: Planned (grilling-ready)
- **Date**: 2026-09-14
- **ADR**: [108 — Sokar artifact daemon](../decisions/108-sokar-artifact-daemon.md)
- **Anima WI**: `01a0a00a-d3b0-7713-90cc-99b6fdac941e` (umbrella, draft; one WI per slice)
- **Governance**: every slice is one Anima WI with an exit condition (ADR 105). Implementer: Codex `gpt-6-astra` run locally with `codex exec` in a worktree branch `sokar/52<x>`; reviewer: the Fable architect session (brief → implement → `/code-review` + tests → commit). Anima is not touched in this phase; a replacement WI in the Anima project is filed once 52f lands.

## Goal

Move artifact storage out of Anima into `sokard`, an OpenDAL-backed daemon in `crates/sokar` that stores and maintains artifacts on S3-compatible and cloud object stores, offers every akh session an artifact space, lets agents manage access by scoped token, cookie session and OIDC claim (Barycenter, Keycloak), delivers downloads and websites by presigned URL where the store can and by proxy where it cannot, retains every feature Anima has today, and renders A2UI payloads so agents can draw UIs and clients can embed them as canvas webviews.

## Non-goals

- Replacing Seshat (documents/RAG) or Heka (memory). Sokar stores bytes and UI streams, not knowledge.
- Any UI logic in the daemon. Sokar transports and persists A2UI messages; agents decide what to draw.
- Provider-specific role code. Claim matching is generic; Keycloak/Barycenter knowledge is skill content.
- Anima's RBAC. Anima keeps `check_permission`; Sokar grants are the second gate, not a replacement.

## Slices

Each slice lands behind its own WI, builds and tests in CI with `--features` that do not require cloud credentials (the `memory` and `fs` OpenDAL services carry the unit and integration tests; an S3 contract test runs against a local MinIO/SeaweedFS in `scripts/sokar-s3-local.sh` and is `#[ignore]` by default).

### 52a — Crate skeleton, stores, blobs, proxy delivery (foundation)

- `crates/sokar` (lib + `sokard` bin, `clap` with `--config`, `--check`, `--check-stores`, `--migrate`) and `crates/sokar-client` (wire types + `reqwest` client, `types-only` feature so `akh-session-services` can depend on it without TLS/reqwest).
- `SokarError` (miette + thiserror, diagnostic codes `sokar::config`, `sokar::storage`, `sokar::database`, `sokar::authn`, `sokar::authz`, `sokar::delivery`, `sokar::surface`), `problem+json` mapping.
- Config: `[server] api_bind, sites_bind, admin_bind, public_api_url, public_sites_url`; `[database] url = "env:SOKAR_DATABASE_URL"`; `[[store]]` (name, service, options with `env:`/`file:` indirection); `[limits] max_body, presign_max_ttl, share_link_max_ttl`.
- Store registry: `StoreRegistry { HashMap<StoreName, Arc<Operator>> }` built with `via_iter`, layered (retry, timeout, concurrency, tracing, prometheus); `StoreCapability { presign_read, presign_write, ranges }` captured at boot from `info().capability()`.
- Bucket layout per ADR 108 D1 (`_sokar/format.json`, `blobs/sha256/…`, `artifacts/<uuid>/manifest.json`, `collections/<uuid>/manifest.json`), versioned `sokar.*.v1` manifest types with serde round-trip tests; write order bytes → manifest → Postgres row.
- Postgres schema `sokar` (sqlx migrations, advisory-locked) as index: `blob`, `collection`, `artifact`, `artifact_version`; `sokard reindex --store <name>` rebuilds it from manifests; UUIDv7 ids everywhere.
- Endpoints: `POST /v1/collections`, `GET /v1/collections/{id}`, `POST /v1/collections/{id}/artifacts` (stream + multipart, SHA-256 while streaming, staging key → `copy_with(..).if_not_exists(true)`, `UploadOutcome::{Stored, Deduplicated}`), `GET /v1/artifacts/{id}`, `GET /v1/artifacts/{id}/content` (proxy only in this slice: ranges, ETag, `Content-Disposition`), `DELETE /v1/artifacts/{id}` (refcounted blob delete), `GET /v1/collections/{id}/artifacts` (paged, by path prefix and kind).
- Auth in this slice: service bearer tokens only (file-mounted, constant-time, optional SHA-256 verifier), reusing the `akh-session-services` pattern verbatim.
- Exit condition: `cargo test -p sokar` green on `memory` + `fs`; an `fs`-backed `sokard` round-trips a 100 MiB upload with dedup and a range download; dropping the Postgres schema and running `sokard reindex` restores every artifact and collection from manifests; `sokard --check` reports config, DB and store health with diagnostics.

### 52b — Access model, scoped tokens, share links, presign-or-proxy, retention

- Tables `principal`, `token`, `grant`, `claim_rule`, `share_link`, `retention_job`; grants also written as `collections/<uuid>/grants/<uuid>.json` manifests and restored by `reindex`; tokens/sessions Postgres-only.
- `Principal`, `PrincipalMatcher`, `ClaimRule`, `ClaimOp`, `Permissions` exactly as in ADR 108 D3; `Authorizer::decide(principal, resource, permission) -> Decision { allowed, grant_id, reason }` with a `sokar::authz::decision` span.
- `POST /v1/tokens` (scoped opaque tokens, hashed at rest, embedded grant set, expiry; needs `MANAGE_ACCESS`), `DELETE /v1/tokens/{id}`, `POST /v1/grants`, `DELETE /v1/grants/{id}`, `GET /v1/{resource}/grants`.
- `POST /v1/artifacts/{id}/links` → HMAC-signed share link (key from `file:`, rotation with two active keys), `DELETE /v1/links/{id}`; proxy delivery honours `Link` principals.
- Delivery decision `Delivery::{Redirect(PresignedUrl), Stream}` per ADR 108 D5; `presign_read` with `response-content-disposition`; TTL = min(grant expiry, config max).
- Two-phase large uploads via `presign_write` with digest verification on commit.
- `RetentionPolicy` on collections, janitor task, quotas per collection. Janitor owns everything irreversible: blobs with refcount 0 deleted only after a manifest reconcile shows no live reference; tombstoned manifests (`deleted_at`) removed after a grace period; `uploads/` staging keys older than the upload timeout swept; expired idempotency receipts purged.
- Carried over from the 52a review (`docs/ai/phases/phase-52-sokar/52a-review.md`): per-resource serialization instead of the 52a advisory locks (per-`(store,digest)` for promotion, per-scope for collection creation, per-`(collection,path)` for version chains); a bucket owner marker `_sokar/owner.json` with a lease so two daemons with independent indexes cannot manage one prefix (ADR 049 split-brain guard pattern).
- Exit condition: property tests for the claim-rule evaluator (all ops, nested `All/Any/Not`, unknown pointer → deny); an `s3` contract test (MinIO) shows `302` presign, an `fs` store shows proxy for the same request; a revoked share link returns 403 within one request; janitor deletes an expired artifact and its orphaned blob.

### 52c — OIDC (Barycenter + Keycloak), cookie sessions, JWKS bearer, claim explain

- `[[oidc]]` providers, discovery via `openidconnect` 4.x, Authorization Code + PKCE, `/auth/{provider}/login|callback|logout`, server-side sessions, `PrivateCookieJar` cookie (`Secure`, `HttpOnly`, `SameSite=Lax`), CSRF state + nonce.
- Bearer JWT validation against cached JWKS (`jsonwebtoken` 11, refresh on unknown `kid`, clock skew config) and RFC 7662 introspection for opaque tokens; both produce a `ClaimSet` (merged ID token + userinfo).
- `POST /v1/claim-rules/validate` and `POST /v1/claim-rules/explain` (rule + sample claims → decision trace) for skills.
- `PrincipalMatcher::ExternalCheck` against Barycenter's `POST /v1/check` (`principal = "user/<sub>"`, `resource = "sokar/<resource>"`), with a bounded cache. Optional slice — land it if Barycenter's authz sidecar is deployed in the cluster by then, else record as open.
- Ops: register `sokar` as a client in Barycenter (`cloud.wegmueller.it`) and a Keycloak realm client; Keycloak needs a Group Membership mapper if group rules are wanted (document in `docs/ops/sokar-oidc.md`).
- Exit condition: an end-to-end test with a stub OIDC issuer (`oidc-mock` container or in-process `axum` issuer in `tests/`) logs in, receives a cookie, and is allowed by a `Claims` grant and denied after the grant is deleted; a Keycloak-shaped token with `realm_access.roles` passes a `Contains` rule; `explain` output names the failing sub-rule.

### 52d — Sites on an isolated origin

- Table `site`; `POST /v1/sites` (collection + root + index + CSP profile), `PUT /v1/sites/{id}/upload` (tar/zip or multi-file), `GET https://<sites>/s/{id}/{path}` served by proxy with correct media types, immutable caching by digest, strict CSP, `X-Frame-Options` disabled only for `/ui/surfaces` (52e) and sites explicitly marked `embeddable`.
- Sites origin never reads the API cookie; access is share link (query or path token) or a sites-origin scoped cookie minted from a share link.
- Exit condition: a generated static site (the Phase 50 leaderboard HTML from `benchmarks/harness-core/src/site.rs` is the fixture) publishes and serves with the right headers; a script in the site cannot reach `/v1/*` with the operator's cookie (CSP + origin test).

### 52e — A2UI surfaces and the canvas renderer

- Typed `sokar::a2ui::v0_9` message model + serde validation + catalog check; `v1_0` function calls behind `a2ui-v1` feature.
- Tables `surface`, `surface_event`; `POST /v1/surfaces`, `POST /v1/surfaces/{id}/messages`, `GET /v1/surfaces/{id}/stream` (SSE JSONL replay-then-live), `POST /v1/surfaces/{id}/actions`, `GET /v1/surfaces/{id}/actions/stream`; snapshot to `SurfaceLog` artifacts.
- Vendored `@a2ui/lit` renderer page at `GET /ui/surfaces/{id}` on the sites origin (pinned version, SRI, no CDN), resolves `sokar://artifact/{id}` to share links.
- Exit condition: an example (`examples/sokar_surface_demo.rs`) opens a surface, pushes `createSurface`/`updateComponents`/`updateDataModel`, the renderer page shows it in a headless browser test (playwright in CI is optional; a JSONL conformance test against the A2UI sample messages is mandatory), a click round-trips to the actions stream.

### 52f — akh integration: tool, ACP group, session credentials, skills

- `src/agent/tools/artifacts.rs` (`impl Tool`, sub-ops per ADR 108 D7), registered in `ToolRegistry`; `_akh/artifacts/*` ACP group + `akh artifacts` CLI (ADR 071 table updated).
- `akhsessd` issues per-session Sokar tokens (session_credentials pattern) and exports `SOKAR_URL`, `SOKAR_TOKEN_FILE`, `SOKAR_COLLECTION`; the Anima runner and `akh-kagent` forward them.
- Skillpacks: `skills/sokar-artifacts/` (procedures for put/link/site/surface), `skills/sokar-access-keycloak-team/`, `skills/sokar-access-barycenter-user/`, `skills/sokar-access-public-link/` — each with `procedures.md`, `actions.json`, `claim_rules.json` templates; `skill_author` documented as the way to add a situation.
- Exit condition: a chat turn "share the report with the SRE Keycloak group for a week" ends with a `Claims` grant whose rule was validated via `explain` before creation, all through the tool (no natural-language constants in code); a session publishes a transcript-shaped artifact without any Anima involvement.

### 52g — Deployment (chart + Flux) and the Anima hand-off

- `charts/akh-medu` gains a `sokard` component (Deployment, Service, HTTPRoutes for the API and sites hosts, ConfigMap, ExternalSecrets for DB URL, S3 identity, OIDC client secrets, HMAC key); Flux release in `talos` under `akh-medu`; a `sokar` identity in the SeaweedFS identities secret; `docs/ops/sokar.md` runbook.
- Rotate the plaintext `anima-transcripts` key found in the Anima HelmRelease values and move it into an ExternalSecret (talos repo, not this one).
- File the **Anima replacement WI** in the Anima project (ADR 108 D8: full replacement, import by writing manifests, remove old code in the same change, PWA iframe). That WI is outside Phase 52.
- Exit condition: `sokard` serves in the cluster against SeaweedFS with presign verified end to end; the Anima WI exists and links this plan.

## Sequencing and parallelism

52a → 52b → {52c, 52d, 52e in parallel} → 52f → 52g. 52f can start once 52b's client surface is stable; 52g can start as soon as 52b serves (chart work does not wait for A2UI). One Anima WI per slice; the Anima replacement is a separate WI filed at 52g.

## Risks

- **Browser-facing auth is new here.** Mitigation: separate origins, strict CSP, private cookies, a security review gate before the sites host is exposed through the gateway.
- **Proxy egress cost** when a non-presign store is default. Mitigation: presign is the default for S3-family stores; proxy metrics per store from day one.
- **A2UI spec churn** (v0.9 → v1.0). Mitigation: versioned module, feature flag, conformance test against the upstream sample corpus.
- **Barycenter has no group claim.** Mitigation: `sub`-based rules work today; group rules arrive as a skill when the claim exists; `ExternalCheck` bridges KDL policies.
- **Anima tracker flapped** during planning (server lost its Postgres route, HTTP 500); the umbrella WI landed once it recovered. Slice WIs are filed from this plan as each slice is grilled.
