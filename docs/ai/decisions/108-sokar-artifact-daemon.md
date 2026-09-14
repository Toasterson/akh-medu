# ADR 108: Sokar — a dedicated artifact daemon (OpenDAL storage, agent-managed access, A2UI surfaces)

- **Status**: Accepted (design), implementation planned as Phase 52
- **Date**: 2026-09-14
- **Anima WI**: `01a0a00a-d3b0-7713-90cc-99b6fdac941e` (Akh-Medu project, umbrella; slices get their own WIs per ADR 105)
- **Plan**: [2026-09-14-phase52-sokar-artifact-daemon.md](../plans/2026-09-14-phase52-sokar-artifact-daemon.md)
- **Related**: ADR 047 (Seshat's Archive — shared service pattern), ADR 061 (ACP-only surfaces), ADR 071 (`akh` CLI command tree), ADR 082 (executor backend seam), ADR 105 (run governance); Anima ADR 0007 (define-side artifacts), Anima ADR 0026 (dispatch-prompt persistence)

## Context

Artifact storage today is smeared across Anima and does not exist in akh-medu at all.
The 2026-09-14 survey of `~/ws/anima` found three unrelated artifact-shaped stores:

| Store | Where | Shape | Gaps |
|---|---|---|---|
| Design artifacts (pretotype HTML/CSS/SVG/images) | Postgres `artifacts` BYTEA table (`anima-db/migrations/038_artifacts.sql`), gRPC `ArtifactService`, `GET /artifacts/{id}` | closed `ArtifactKind` enum, whole blob in memory | raw HTTP route checks only "some valid MCP token", not project membership; no delete; no retention; 4 MiB tonic cap; no `Content-Disposition` |
| Session transcripts + dispatch prompts | SeaweedFS S3 via a hand-rolled SigV4 client in `anima-server/src/services/session.rs` | `transcripts/<wi>/<session>.jsonl`, `dispatch-prompts/<wi>/<session>.md` | no signed URLs, server proxies every read, no GC, credentials are all-or-nothing env vars, scheduler blocks dispatch on PUT failure (ADR 0026 — must be preserved) |
| Thread attachments | `thread_messages.blocks` JSONB with inline base64 | ACP `ContentBlock[]` | bounded only by the 32 MiB gRPC cap; bytes live in Postgres rows |

Anima has no OIDC integration (homegrown session tokens, PATs, MCP session-bound tokens, static service tokens) and no A2UI, canvas or webview concept — the ADR 0007 "render the design artifact in an iframe" intent has no live consumer in the PWA.
akh-medu has no object storage, no artifact subsystem, no OIDC or cookie auth (house style is file-mounted bearer tokens, constant-time compared, optional SHA-256 verifier), and its one prior web UI was deliberately deleted (ADR 061).
A2UI is referenced only as an optional follow-on in the wayfinder ACP/T3 plan.

Two identity providers are in play: **Barycenter** (`~/ws/barycenter`, CloudNebulaProject; our own Rust OIDC IdP at `cloud.wegmueller.it`, RS256 ID tokens, `/userinfo`, `/introspect`, device flow, plus a KDL ReBAC/ABAC authz sidecar exposing `POST /v1/check`; it issues **no** roles/groups claim today) and **Keycloak** (roles in `realm_access.roles` / `resource_access.<client>.roles`, groups only via a Group Membership mapper).

Library facts that constrain the design (verified 2026-09-14): `opendal` 0.59.1 (MSRV 1.91) presigns `read`/`write`/`stat` on `s3`, `gcs`, `azblob`, `oss`, `cos` and **not** on `fs`, `memory`, `webdav`, `sftp`; capability is queryable at runtime via `Operator::info().capability()`; SeaweedFS and MinIO go through the `s3` service; `write_with(..).if_not_exists(true)` gives create-if-absent for content-addressed dedup; `opendal::Error` is a plain `std::error::Error` and composes with `thiserror`. A2UI's stable line is v0.9.1 (`createSurface`, `updateComponents`, `updateDataModel`, `deleteSurface`, JSONL over HTTP/SSE, explicit `catalogId`), v1.0 RC adds `callRendererFunction`/`callAgentFunction`; official renderers are Lit, Angular, React (shared `@a2ui/web_core`) and Flutter; there is **no** Rust crate.

## Decision

Build **Sokar** (`crates/sokar`, binary `sokard`; client crate `crates/sokar-client`) in the akh-medu workspace, as a standalone service in the Seshat mould (own Postgres schema, own Helm component, bearer-authenticated ingress) that both Anima and every akh consume. Sokar owns four things and nothing else:

1. **Blobs and artifacts** — content-addressed bytes on OpenDAL-backed stores, with typed metadata in Postgres.
2. **Access** — principals, grants and claim rules that agents (akh or not) manage through the API; credentials are scoped tokens, cookie sessions and OIDC tokens from Barycenter or Keycloak.
3. **Delivery** — presigned URLs where the store supports them, a proxy endpoint otherwise; multi-file **sites** always through the proxy on an isolated origin.
4. **Surfaces** — persisted A2UI message streams plus a hosted renderer page, so an agent can draw a UI and any client can embed it as a canvas webview.

Anima keeps its RBAC and its work-item semantics; it becomes a Sokar client. The engine (`akhomed`) stays LLM/KG-only; Sokar is reachable from the agent through one tool and one ACP group.

### D1. Storage: OpenDAL `Operator` registry, content-addressed blobs, Postgres metadata

- Config declares named stores as data: `[[store]] name = "lab-s3"; service = "s3"; options = { bucket = "...", endpoint = "...", region = "us-east-1", access_key_id = "env:SOKAR_S3_KEY", secret_access_key = "file:/var/run/secrets/s3/secret", disable_config_load = "true", disable_ec2_metadata = "true" }` and builds each with `Operator::via_iter::<services::S3>` behind `RetryLayer`, `TimeoutLayer`, `ConcurrentLimitLayer`, `TracingLayer`, `PrometheusClientLayer`. Store choice is a Cargo feature set (`services-s3`, `services-gcs`, `services-azblob`, `services-fs`, `memory` always on for tests).
- Blob key layout: `blobs/sha256/<aa>/<bb>/<hex>`; upload computes SHA-256 while streaming to a staging key, then `copy_with(..).if_not_exists(true)` to the final key so identical content dedups and a `ConditionNotMatch` is the *normal* path, not an error. Digest is the ETag and the cache validator.
- Postgres tables (own schema `sokar`, sqlx migrations, advisory-locked like `akhomed_store::run_migrations_locked`): `blob`, `collection`, `artifact`, `artifact_version`, `site`, `principal`, `token`, `grant`, `claim_rule`, `share_link`, `surface`, `surface_event`, `retention_job`. Metadata never stores bytes; bytes never carry authorization.
- One `sokard` may serve several stores; a collection pins one store (`store` column) so relocation is explicit.

### D2. Domain types (express the model in the types)

```rust
pub struct Sha256Digest([u8; 32]);
pub struct BlobRef { digest: Sha256Digest, size: u64, media_type: MediaType }
pub enum CollectionScope {
    AnimaWorkItem { project: Uuid, work_item: Uuid },
    AnimaSession   { work_item: Option<Uuid>, session: Uuid },
    AkhSession     { akh: AkhName, session: SessionId },
    AkhWorkspace   { akh: AkhName, workspace: WorkspaceId },
    Named(CollectionName),
}
pub struct Collection { id: CollectionId, scope: CollectionScope, owner: PrincipalId, store: StoreName, retention: RetentionPolicy, quota: Quota }
pub enum ArtifactKind { Document, Image, Audio, Video, Archive, Transcript, DispatchPrompt, Site, SurfaceLog, Other }
pub struct Artifact { id: ArtifactId /* uuid v7 */, collection: CollectionId, path: ArtifactPath, kind: ArtifactKind, blob: BlobRef, labels: Labels, created_by: PrincipalId, created_at, expires_at: Option<..>, previous: Option<ArtifactId> }
pub enum RetentionPolicy { Forever, Ttl(Duration), UntilScopeClosed { grace: Duration }, KeepLast(u32) }
```

Anima's closed `ArtifactKind` (`Html | Css | Svg | Image | Other`) maps onto `Document`/`Image` plus a `media_type`; the old enum is preserved as a label during migration so nothing is lost.

### D3. Principals, credentials and the claim-match grant model

```rust
pub enum Principal {
    Service { name: ServiceName },                       // sokard-local bearer tokens (file-mounted, sha256-verifier optional)
    Agent   { akh: AkhName, session: Option<SessionId> },// per-session scoped tokens issued by akhsessd or by Sokar itself
    User    { issuer: IssuerUrl, subject: String, claims: ClaimSet }, // OIDC (cookie session or bearer JWT/introspected token)
    Link    { link: ShareLinkId },                       // bearer of a Sokar-signed share link
    Anonymous,
}
pub enum Credential { BearerToken(Secret), Cookie(SessionCookie), OidcBearer(Jwt), ShareLink(SignedLink) }

pub struct Grant { id, resource: ResourceSelector, matcher: PrincipalMatcher, permissions: Permissions, expires_at: Option<..>, issued_by: PrincipalId }
pub enum ResourceSelector { Collection(CollectionId), Artifact(ArtifactId), Site(SiteId), Surface(SurfaceId), Prefix { collection: CollectionId, prefix: ArtifactPath } }
pub enum PrincipalMatcher {
    Exact(PrincipalId),
    Token(TokenId),
    Claims(ClaimRule),
    ExternalCheck { authz: AuthzEndpoint, permission: String }, // Barycenter /v1/check, later
    Anyone,
}
pub enum ClaimRule {
    All(Vec<ClaimRule>), Any(Vec<ClaimRule>), Not(Box<ClaimRule>),
    Issuer(IssuerUrl),
    Claim { path: JsonPointer, op: ClaimOp },
}
pub enum ClaimOp { Exists, Equals(Value), OneOf(Vec<Value>), Contains(Value), MatchesRegex(BoundedRegex) }
bitflags! Permissions: READ | WRITE | DELETE | LIST | MANAGE_ACCESS | PRESENT (drive a surface) | ACT (send surface actions)
```

- The collection owner implicitly holds all permissions; `MANAGE_ACCESS` lets a principal create and revoke grants on that resource, which is how an agent "manages access by token, cookie and OIDC claim": it issues a scoped token (`POST /v1/tokens`, opaque, SHA-256-hashed at rest, with an embedded grant set and expiry), or posts a grant whose matcher is a `ClaimRule` (cookie/OIDC users), or mints a share link.
- Claim rules are pure data over the verified claim set, so Keycloak (`/realm_access/roles`, `/resource_access/<client>/roles`, `/groups`) and Barycenter (any userinfo/property claim, and a `groups` claim once Barycenter grows one) are handled by the same evaluator with zero provider-specific code. Provider knowledge lives in **skills**, not in the daemon (D6).
- Evaluation is deny-by-default, resource-then-prefix-then-collection, most-specific grant wins for `expires_at`, and every decision is traced with the grant id (`sokar::authz::decision` span) so agents can explain why a link works.
- No natural-language constants anywhere; matching is structural (JSON pointer + typed op).

### D4. Identity providers: OIDC relying party + JWKS bearer validation + cookie sessions

- `[[oidc]] name = "barycenter"; issuer = "https://cloud.wegmueller.it"; client_id; client_secret = "env:..."` and one for Keycloak. Discovery via `openidconnect` 4.x, Authorization Code + PKCE, login at `/auth/{provider}/login` → callback → server-side session row → `PrivateCookieJar` cookie (`axum-extra` 0.12), `SameSite=Lax`, `Secure`, `HttpOnly`. Session TTL is config; logout revokes the row.
- API callers may also present a provider access token as `Authorization: Bearer`; Sokar validates it against the cached JWKS (`jsonwebtoken` 11, `aws_lc_rs` backend) and, for opaque tokens, via RFC 7662 introspection. Claims from either path feed the same `ClaimSet`.
- Barycenter's authz sidecar (`POST /v1/check`, `principal`/`permission`/`resource`) is modelled as the `ExternalCheck` matcher so KDL policies can gate a Sokar resource without Sokar learning KDL. It is a later slice, not on the critical path.

### D5. Delivery: signed URL if possible, proxy if not; sites always proxied on their own origin

- `GET /v1/artifacts/{id}/content` authorises, then: if the collection's store reports `presign` capability **and** the artifact is not part of a site **and** the request's principal is not a share link bound to proxy delivery, respond `302` to `Operator::presign_read(key, ttl)` with `ttl = min(grant.expires_at, config.presign_max)` and a `Content-Disposition` override baked into the presigned query (`response-content-disposition`). Otherwise stream through OpenDAL `reader_with(..).range(..)` with `ETag`, `Accept-Ranges`, `Cache-Control: private, immutable`, and `Content-Disposition` = `inline` only for an allow-listed set of media types, `attachment` otherwise.
- **Share links** (`POST /v1/artifacts/{id}/links`) are Sokar-signed (HMAC-SHA256 over `artifact_id · expiry · optional principal binding · nonce`, key rotated from a file) and always served via proxy; they are the "signed URL" for `fs`/`webdav`/`sftp` stores and for sites. Revocable (row per link).
- **Sites** (`Site { id, collection, root: ArtifactPath, index: ArtifactPath, csp: ContentSecurityPolicy }`) are served at `https://<sites-origin>/s/{site_id}/{path}` on a **separate origin** from the API and cookie origin, with a strict CSP and no credentials, so agent-generated HTML can never read the operator's Sokar session. This is the "websites" delivery path and what A2UI surfaces embed for rich content.
- Uploads: `POST /v1/collections/{id}/artifacts` (streamed body or multipart, `DefaultBodyLimit` from config) and a two-phase large-upload path (`POST .../uploads` → presigned PUT from `presign_write` → `POST .../uploads/{id}/commit` with the client-computed digest, verified by `stat` + re-hash for non-presign stores).

### D6. A2UI surfaces and the canvas webview

- Sokar carries its own typed A2UI model (`sokar::a2ui::v0_9::{Message, CreateSurface, UpdateComponents, UpdateDataModel, DeleteSurface}` with `#[serde(tag = ..)]`, plus `v1_0` function-call messages behind a feature) validated on ingest; unknown component types are rejected against the surface's declared `catalog_id`.
- `POST /v1/surfaces` creates a surface inside a collection; `POST /v1/surfaces/{id}/messages` appends (needs `PRESENT`); `GET /v1/surfaces/{id}/stream` is SSE JSONL replay-then-live for renderers (needs `READ`); `POST /v1/surfaces/{id}/actions` accepts user events (needs `ACT`) and `GET /v1/surfaces/{id}/actions/stream` (SSE) delivers them to the presenting agent. Sokar is transport and storage, never UI logic.
- `GET /ui/surfaces/{id}` serves a vendored `@a2ui/lit` renderer page (pinned version, no CDN, subresource-integrity hashes committed) on the **sites origin**, taking auth from a share link or a cookie. That page is the "Canvas webview": Anima's PWA, the ratatui TUI's external-browser hand-off, T3 Code's preview and any other client embed it in an iframe.
- Every surface's event log is periodically snapshotted to an artifact of kind `SurfaceLog`, so surfaces are durable, replayable and subject to the same retention and access rules as any other artifact. Components reference artifacts by `sokar://artifact/{id}` URIs which the renderer page resolves to share links.

### D7. akh integration: one tool, one ACP group, per-session credentials, skills for provider knowledge

- Agent tool `artifacts` (`src/agent/tools/artifacts.rs`, `impl Tool`) with sub-ops `put`, `get_link`, `list`, `grant`, `token`, `site_publish`, `surface_open`, `surface_update`, `surface_actions`. Registered in `ToolRegistry`; conversation strategy unchanged.
- ACP group `_akh/artifacts/*` and CLI `akh artifacts {put,ls,link,grant,site,surface}` per ADR 071.
- `akhsessd` issues a per-session Sokar token scoped to that session's `AkhSession` collection using the existing `session_credentials` pattern (0o700 file, receipt without secret in the KV store); the runner exports `SOKAR_URL`, `SOKAR_TOKEN_FILE`, `SOKAR_COLLECTION` into the session.
- "Set up a claim match by skill": provider- and situation-specific knowledge is a skillpack, not daemon code. Convention: `skills/sokar-access-<situation>/` with `procedures.md` (when to use it), `actions.json` (the `artifacts.grant` action schema) and a `claim_rules.json` of `ClaimRule` templates with named placeholders (e.g. `keycloak-team-readonly`: `All[Issuer(keycloak), Claim{/realm_access/roles, Contains("${team}")}]`, `barycenter-user`: `All[Issuer(barycenter), Claim{/sub, Equals("${subject}")}]`). `skill_author` can author new ones from a description; the daemon exposes `POST /v1/claim-rules/validate` and `POST /v1/claim-rules/explain` so a skill can check its rule against a sample token before granting.

### D8. Anima migration (retain every feature, fix the gaps)

- `anima-server` gets `sokar-client` with a service token. New collections: `AnimaWorkItem` for design artifacts, `AnimaSession` for transcripts (kind `Transcript`) and dispatch prompts (kind `DispatchPrompt`). The scheduler keeps its ADR 0026 invariant: dispatch is blocked unless Sokar acknowledges the prompt write.
- `ArtifactService.{List,Get}Artifacts` and `GET /artifacts/{id}` become thin proxies that first run Anima's own `check_permission(WorkItemRead)` and then hand back a Sokar share link (fixing the per-project authorization gap). `RunDefinePhase` uploads the pretotype HTML instead of inserting BYTEA. MCP `get_session_transcript` / `get_dispatch_prompt` use Sokar range reads for tailing.
- One-shot migration binary copies `artifacts` rows and the SeaweedFS keys into Sokar (same bucket can be registered as a store, so transcripts do not even move bytes: Sokar adopts existing keys via `stat` + digest backfill).
- Thread attachments larger than a threshold are offloaded to Sokar with a `resource_link` block; small inline blocks stay as they are.
- The PWA renders design artifacts and A2UI surfaces in a sandboxed iframe pointing at the Sokar sites origin — the first live consumer of ADR 0007.
- Old code (`services/artifact.rs`, `handlers/artifact.rs`, the SigV4 client, migration 038 tables) is retired only after the cut-over WI verifies parity.

### D9. Errors, config, observability, conventions

- `SokarError` derives `Debug, Error, Diagnostic`; each variant carries `#[diagnostic(code(sokar::...), help(...))]`; OpenDAL errors are wrapped as `Storage { op, store, kind: opendal::ErrorKind, #[source] source }` with `ErrorKind`-specific help (e.g. `PermissionDenied` → "check the store's credentials and bucket policy; run `sokard --check-stores`"). HTTP mapping is `application/problem+json` carrying the diagnostic code. Consumers get `SokarResult<T>` and typed outcomes where a caller has a choice (`UploadOutcome::{Stored, Deduplicated}`, `Delivery::{Redirect(Url), Stream(..)}`).
- Config is TOML with `#[serde(deny_unknown_fields)]`, secrets referenced as `env:` or `file:` never inline, `sokard --check` validates config, stores (`stat` a probe key), Postgres and OIDC discovery without serving.
- `tracing` + Prometheus (`/metrics` on the admin listener), request ids, per-store latency histograms, GC counters. Retention janitor runs in-process (`RetentionPolicy` → `retention_job` rows) and deletes blobs only when their refcount hits zero.
- Three listeners: API (bearer/cookie, CORS to configured app origins), sites (no cookies from the API origin, strict CSP), admin (metrics, health, `/readyz` that checks Postgres and every store).

## Alternatives considered

- **Keep artifacts inside Anima and add OpenDAL there.** Rejected: akh sessions that are not Anima-dispatched (Mac lane, cluster akhs, T3 sessions) need the same storage and UI surface, and Anima's auth model has no OIDC and no per-resource grants. A shared service also lets Anima drop its hand-rolled SigV4 client.
- **Extend `akh-session-services` with an `akhartd` binary.** Rejected: that family is the per-akh session fabric with redb/KV store semantics and no browser-facing concerns. Sokar needs relational queries (listing, expiry, grants), cookie sessions and a public sites origin; it matches Seshat's shape, not `akhsessd`'s.
- **Presigned URLs only, no proxy.** Rejected: OpenDAL cannot presign on `fs`, `webdav`, `sftp`, sites need per-file serving with headers we control, and revocation of a presigned URL is impossible before expiry.
- **Provider-specific role plumbing (a `keycloak` and a `barycenter` module).** Rejected: a JSON-pointer claim rule covers both and every future provider; provider knowledge belongs in skills so the agent, not the daemon, adapts to a situation.
- **Run the A2UI renderer inside each client (PWA, TUI).** Rejected for now: one hosted, pinned renderer on an isolated origin gives every client the same canvas with one security review; clients may still embed `@a2ui/web_core` natively later.
- **Store metadata in `akh-service-store` (redb/KV).** Rejected: grant evaluation, listings and retention need indexes and transactions across rows; Postgres is already the fleet's truth store.

## Consequences

- New crates `sokar`, `sokar-client`; new Helm component `sokard` in `charts/akh-medu`; new Flux release in the `talos` repo (namespace `akh-medu`), Barycenter client registration, and an S3 identity for Sokar in the SeaweedFS identities secret.
- Anima loses ~1.2k lines (artifact service/handler, SigV4 client, BYTEA table) and gains a client; its raw artifact route stops leaking cross-project reads.
- akh sessions gain durable, shareable output and a way to draw UI that outlives the session.
- Operational: one more Postgres schema and one more public origin to protect. Browser-facing auth (cookies, OIDC) is a new pattern in this workspace and gets its own security review before the sites origin is exposed.
- Observation from the survey worth acting on outside this ADR: the SeaweedFS `anima-transcripts` secret key is committed in plaintext in the `talos` repo's Anima HelmRelease values; move it to an ExternalSecret and rotate it when Sokar's identity is created.

## Reopening triggers

- A2UI v1.0 goes stable → promote the `v1_0` feature to default and add function-call round-trips.
- Barycenter ships a groups/roles claim → add a `barycenter-group` claim-rule skill; no daemon change expected.
- A store that cannot presign becomes the production default → revisit proxy capacity (CPU/egress) and consider a CDN-signed path.
