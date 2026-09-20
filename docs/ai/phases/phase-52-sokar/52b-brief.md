# 52b brief — access model, scoped tokens, share links, presign-or-proxy, janitor

Implementer: Codex `gpt-6-astra` (local `codex exec`), branch `sokar/52b` from the merged 52a. Reviewer: Fable architect session.
Authority: [ADR 108](../../decisions/108-sokar-artifact-daemon.md) (D1 grants-in-bucket, D3, D5, Grilling) and [plan §52b](../../plans/2026-09-14-phase52-sokar-artifact-daemon.md); carry-overs from [52a-review.md](52a-review.md) "Deferred to 52b". Anima WI: to be filed (umbrella `01a0a00a-d3b0-7713-90cc-99b6fdac941e`).

## Deliverable

`crates/sokar` + `crates/sokar-client` extended, tests green, committed on `sokar/52b`. No OIDC, cookies, sites or A2UI (52c–52e). Anima and other crates untouched.

## 1. Principals and credentials

```rust
pub enum Principal { Service { name }, Agent { akh, session: Option<SessionId> }, User { issuer, subject, claims: ClaimSet }, Link { link: ShareLinkId }, Anonymous }
pub enum Credential { BearerToken(Secret), ShareLink(SignedLink) }   // Cookie/OidcBearer arrive in 52c; keep the variants but unimplemented → 401
```
- Service tokens as in 52a. **Scoped tokens**: `POST /v1/tokens { for: Principal (Agent|Service), grants: [GrantSpec], expires_at }` → `{ id, secret: "sk_<base32 of 32 random bytes>" }` returned once; stored as SHA-256; `DELETE /v1/tokens/{id}`; `GET /v1/tokens?issued_by=me`. Max lifetime 1 y (config). Tokens and sessions live only in Postgres (never in the bucket).
- Principal identity string for `created_by`/`issued_by` is stable and typed (`PrincipalId` Display: `service:<name>`, `agent:<akh>[/<session>]`, `user:<issuer-host>/<sub>`, `link:<id>`).

## 2. Grants and the authorizer

```rust
pub struct Grant { id: GrantId, resource: ResourceSelector, matcher: PrincipalMatcher, permissions: Permissions, expires_at: Option<..>, issued_by: PrincipalId, created_at }
pub enum ResourceSelector { Collection(CollectionId), Artifact(ArtifactId), Prefix { collection: CollectionId, prefix: ArtifactPath } }   // Site/Surface added later
pub enum PrincipalMatcher { Exact(PrincipalId), Token(TokenId), Claims(ClaimRule), Anyone }        // ExternalCheck is 52c
pub enum ClaimRule { All(Vec<ClaimRule>), Any(Vec<ClaimRule>), Not(Box<ClaimRule>), Issuer(IssuerUrl), Claim { path: JsonPointer, op: ClaimOp } }
pub enum ClaimOp { Exists, Equals(Value), OneOf(Vec<Value>), Contains(Value), MatchesRegex(BoundedRegex /* max 256 chars, size-limited, compiled once */) }
bitflags Permissions: READ | WRITE | DELETE | LIST | MANAGE_ACCESS   // PRESENT/ACT arrive with 52e
```
- `Authorizer::decide(&Principal, &ResourceRef, Permission) -> Decision { allowed: bool, grant: Option<GrantId>, reason: DenyReason }` replaces the 52a `ServiceOnly` seam; every route calls it; `sokar::authz::decision` tracing span with grant id.
- Order: owner implicit → artifact grants → prefix grants → collection grants; deny by default; expired grants ignored; unknown JSON pointer → rule evaluates false.
- `POST /v1/grants` / `DELETE /v1/grants/{id}` / `GET /v1/collections/{id}/grants` need `MANAGE_ACCESS` on the target (owner has it). Grants are written to the bucket as `collections/<uuid>/grants/<uuid>.json` (`sokar.grant.v1`) before the row, tombstoned on delete like artifacts, restored by `reindex`.
- `POST /v1/claim-rules/validate { rule }` (shape + regex limits) and `POST /v1/claim-rules/explain { rule, claims }` → decision trace naming the failing sub-rule. Pure functions over a `ClaimSet` (`serde_json::Value` object); no provider code, no natural-language constants.

## 3. Share links and delivery

- `POST /v1/artifacts/{id}/links { expires_at?, disposition?, bind_to?: PrincipalId }` (needs `MANAGE_ACCESS` or ownership) → `{ id, url }` where `url = <public_api_url>/l/<link_id>.<sig>`; `sig` = HMAC-SHA256 over `link_id·artifact_id·expiry·binding·nonce` with the key from `[links] key_file` (two keys accepted for rotation). Row per link; `DELETE /v1/links/{id}` revokes; `GET /l/{token}` verifies signature, checks the row, then delivers as `Principal::Link`. Defaults 7 d, max 90 d.
- `GET /v1/artifacts/{id}/content` decision: `Delivery::Redirect` (302 to `presign_read(key, ttl)` with `response-content-disposition`/`response-content-type` query overrides) when `store.cap.presign_read && !collection.force_proxy && principal is not Link`; else `Delivery::Stream` as in 52a. `ttl = min(grant expiry remaining, [delivery] presign_max_ttl (24 h), presign_default_ttl (15 min))`.
- Two-phase large upload: `POST /v1/collections/{id}/uploads { path, media_type, kind, size, sha256 }` → `{ upload_id, method, url, headers }` from `presign_write` (only when `cap.presign_write`; else 409 `PresignUnsupported` with help pointing at the streamed endpoint); `POST /v1/uploads/{id}/commit` → server `stat`s, verifies size, for non-presign stores re-hashes; then promotes exactly like the streamed path. Max 5 GiB (config).

## 4. Janitor (owns every irreversible operation)

- Background task with `[janitor] interval, grace_period, staging_ttl`; per store: reconcile manifests ↔ index (orphan manifests re-indexed, dangling rows tombstoned), delete blobs with refcount 0 **and** no live manifest, delete tombstoned manifests older than `grace_period`, sweep `uploads/` keys older than `staging_ttl`, purge expired idempotency receipts, expire artifacts per `RetentionPolicy` (`Ttl`, `KeepLast(n)`, `UntilScopeClosed` needs a `POST /v1/collections/{id}/close`). Each pass emits counters (Prometheus) and a `sokar::janitor::pass` span. `sokard janitor --once --store <name>` runs one pass for tests.
- Quotas: `Quota { max_bytes, max_artifacts }` enforced at upload with `QuotaExceeded` (413, typed).

## 5. Carry-overs from the 52a review

- Bucket owner marker `_sokar/owner.json { owner_id, lease_until }` written at boot and renewed; a store whose marker names a different live owner refuses to serve (`SokarError::StoreOwnedElsewhere`) unless `--take-over` is passed. ADR 049 pattern.
- Replace the 52a advisory locks with per-resource serialization: per-`(store,digest)` for promotion, per-scope for collection creation, per-`(collection,path)` for version chains; keep the global lock only for `reindex` swap. Deadlock-free order documented and tested.

## Tests / exit condition

1. Gates as in 52a (scoped fmt, clippy `-D warnings`, `cargo test -p sokar -p sokar-client`; Postgres tests skip when `SOKAR_TEST_DATABASE_URL` is unset).
2. Property tests (proptest) for the `ClaimRule` evaluator: all ops, nested `All/Any/Not`, unknown pointer → false, regex bound enforced; `explain` names the failing sub-rule.
3. MinIO contract test (`#[ignore]`): same request yields 302 on `s3` and 200 stream on `fs`; presigned PUT + commit round trip.
4. Share link: works, revoked → 403 on the next request, expired → 410; a link bound to a principal fails for another bearer.
5. Janitor: expired artifact and its orphaned blob disappear after one pass; a blob referenced by an orphan manifest survives and the manifest is re-indexed; staging keys older than TTL are swept.
6. Owner marker: a second daemon with a different owner id refuses to serve the store.
7. `docs/ai/phases/phase-52-sokar/52b-report.md` with deviations and open questions.
