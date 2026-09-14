# 52a review — round 1 findings and round 2 fix list

Reviewer: Fable architect session, 2026-09-14. Subject: branch `sokar/52a` at `93addcbb` (three Codex gpt-6-astra commits).
Verification by the reviewer: `cargo fmt -p sokar -p sokar-client --check` pass, `cargo clippy … -D warnings` pass, `cargo test -p sokar -p sokar-client` 17 passed / 1 ignored (MinIO). Scope respected: only the two crates, workspace members, lockfile and the report changed.

## Decisions on the implementer's open questions

1. **Workspace-wide `cargo fmt --all --check`**: accepted as scoped for this slice. The eight baseline files are outside 52a; a separate formatting-only commit on main fixes them (tracked below, not Astra's job).
2. **`(store, digest)` blob key, zero-refcount rows kept for tombstones, owned manifest `schema` string**: accepted as the 52b foundation.
3. **Ephemeral local mode without `[database]`**: accepted; document in `crates/sokar/examples/local.toml` that it is single-daemon only.
4. **Global mutation lock**: accepted as the 52a serialization model, **but** the lock must be held only around index mutations (see R1–R3). Per-resource locking and a bucket owner marker are 52b items (see "Deferred to 52b").

## Round 2 — required fixes (implementer: Codex gpt-6-astra, same branch)

- **R1 Lock window on upload** (`service.rs` `finish_upload`): move `staged.promote()` and the artifact manifest `write_json` *before* `index.begin()`; hold the lock only for the idempotency lookup, `latest()`, `put_artifact` and commit. Idempotency short-circuit must then clean up the promoted-but-unused manifest (delete the manifest key; the blob stays, it is content-addressed and refcounted). Keep the invariant "bytes → manifest → row" documented in the report.
- **R2 Deletion order and no synchronous blob deletion** (`service.rs` `delete_artifact`): today the manifest (and at refcount 0 the blob) is deleted from the bucket *before* `tx.commit()`, so a failed commit leaves a live row pointing at nothing — the unsafe direction. New order, keeping "bucket first, index second, nothing irreversible in the request path": (1) rewrite the artifact manifest with `deleted_at` set (a tombstone manifest, `sokar.artifact.v1` gains an optional `deleted_at`); (2) under the lock: tombstone the row, decrement refcount, commit. Do **not** delete the manifest key and do **not** delete blob bytes; both are 52b janitor work (tombstoned manifests older than a grace period, blobs with no live manifest after reconcile). `reindex` must treat a manifest with `deleted_at` as a tombstone row that holds no blob reference. Remove `artifact_manifests()` from the delete path. Update the report and note that the brief's "delete synchronously in 52a" is superseded by ADR 108.
- **R3 Reindex lock window** (`service.rs` `reindex`): list, read and validate collections, artifacts and blob sizes **outside** the lock; call `index.begin()` only immediately before `replace_store()` and commit. Re-check the format marker inside the lock is not needed.
- **R4 `problem()` must not panic** (`error.rs`): add an inherent `fn code(&self) -> &'static str` with an exhaustive `match` and use it in `problem()`; delete the `.expect(...)`. Keep the `#[diagnostic(code(...))]` attributes in sync (a unit test asserting `Diagnostic::code()` equals the inherent code for every variant is welcome).
- **R5 Chain client errors transparently** (`error.rs`): replace the `From<sokar_client::Error>` that flattens to `Invalid(String)` with a `#[error(transparent)] #[diagnostic(transparent)] Client(#[from] sokar_client::Error)` variant, mapping to 400 unless the inner is `ManifestSchema` (keep that as today).
- **R6 Typed caller-actionable errors**: add variants `UnknownStore(StoreName)` (500-class, `sokar::config`), `ScopeExists(String)` (409), `IdempotencyMismatch` (409), `StoreMismatch { expected, actual }` (409) with their own codes/help; use them at the current `Invalid`/`Conflict` string sites for those four cases. Both index backends must raise the same variant for scope collision (memory: explicit check; postgres: map the `scope_key` unique violation).
- **R7 Constant-time compare**: use `subtle::ConstantTimeEq` (already in the lock file) instead of the hand-rolled loop in `auth.rs`; keep the exactly-one-match rule. Enforce the same token-file rules as `akh-session-services::config::read_token` (≥ 24 chars, no whitespace) so operators get one credential format across daemons.
- **R8 `Store::check` round trips**: read the format marker once (`read` returns NotFound itself); drop the extra `stat` calls. `read_json` should not `stat` before `read`.
- **R10 Staging cleanup correctness** (`store.rs` `stage`, `service.rs` `finish_upload`): (a) in the error-recovery branch, run `cleanup()` and log its failure, but return the *original* error (never `?` on cleanup); (b) the idempotency-conflict branch must call `staged.cleanup().await` explicitly before returning `IdempotencyMismatch`; (c) when `StagedUpload::drop` runs without a runtime handle, emit `tracing::warn!` with the staging key so the leak is at least visible (janitor sweep of `uploads/` is 52b).
- **R11 `ranged_read` must mean something** (`store.rs`, `delivery.rs`): derive it from OpenDAL's `read_with_range`-class capability rather than `cap.read`; when false, ignore `Range` and answer 200 with the full body (and no `Accept-Ranges`). Add a unit test with a fake capability.
- **R12 Non-conditional copy TOCTOU**: keep the stat-then-copy fallback but document in `promote()` and the report that it is safe only because content is byte-identical by digest, and that the index lock does not serialize bucket writes across independent indexes (owner marker is 52b).
- **R13 Collection creation is create-or-get, and reindex tolerates duplicate scopes**: a crash after the collection manifest write but before commit, followed by a client retry, yields two manifests for one scope and makes `reindex` fail on `scope_key UNIQUE`. Fix both ends: `POST /v1/collections` returns `200` with the existing collection when the scope already exists and the request is compatible (same store or no store given), `409 ScopeExists` only on an incompatible request; `reindex` resolves duplicate scope manifests deterministically (keep the lowest UUIDv7 = earliest, count the rest as `orphans` in `ReindexCounts`, log them at warn) instead of aborting.
- **R14 Idempotency receipts follow the artifact**: `delete_artifact` purges receipts whose cached response points at the deleted artifact (add `IndexTransaction::purge_idempotency_for(artifact_id)`), and the replay branch treats a receipt whose artifact is gone as "no receipt" (fall through to a fresh upload) rather than erroring. Cleanup of the staged upload on every early return (see R10).
- **R15 Typed manifest schema**: replace raw string equality in `check_schema` with a `ManifestSchema { kind, major }` parser for `sokar.<kind>.v<major>`; accept equal major, refuse greater major with `ManifestSchema`, refuse wrong kind. Unit tests for `sokar.artifact.v1`, `sokar.artifact.v2`, `sokar.collection.v1`.
- **R16 Lock scope**: artifact mutations (finish, delete) take a per-store advisory lock (`hashtext('sokar.store:' || store)`), collection creation and the reindex swap take the global `sokar.mutations` lock (scope uniqueness is cross-store). Memory index mirrors this with one mutex per store plus a global one. Document the lock order (global before store) to keep it deadlock-free.
- **R9 Report**: update `52a-report.md` (invariants section, deletion paragraph, validation table) and add a "Round 2" section listing what changed. Run the same gates: scoped fmt, clippy `-D warnings`, `cargo test -p sokar -p sokar-client`; keep the 100 MiB and recovery tests green (adjust the delete test: blob bytes now survive with refcount 0).

## Accepted as-is (with notes)

- Per-request credential file reload (`auth.rs`): matches the house pattern (`seshat::auth::SharedSecretAuth` re-reads projected Secrets per request). Fine for the expected principal count.
- `BlobRef.key` stored alongside `digest`: intentional (ADR 108 D1 "manifests include `blob.key` for portability"); the `validate()` equality check stays.
- `created_at = max(now, previous + 1µs)`: acceptable with UUIDv7 as tiebreaker; revisit if keyset paging ever needs a sequence.
- Unused `StoreCapability.presign_*` / `Delivery::Redirect` / `Permission` taxonomy: requested seams for 52b; keep.
- Memory-backend reader/writer relay in `promote()`: test-only path; acceptable.

## Deferred to 52b (added to the plan)

- Per-resource serialization instead of one advisory lock: per-`(store,digest)` for promotion, per-scope for collection creation, per-`(collection,path)` for version chains.
- Bucket owner marker (`_sokar/owner.json` with a lease) so two daemons with independent indexes cannot manage one prefix — same idea as the ADR 049 split-brain guard.
- Janitor: zero-refcount blob GC after manifest reconcile, staging-key cleanup, idempotency receipt expiry.

## Baseline chores (not Astra, not this branch)

- Format the eight pre-existing files that fail `cargo fmt --all --check` (`crates/akh-session-services/src/{server/sessions.rs,session.rs}`, `examples/akh_{causal_episode,symbolic_episode,temporal_incident}_bridge.rs`, `examples/akh_temporal_validity_probe.rs`, `src/agent/event_calculus.rs`, `src/engine.rs`) in a formatting-only commit on main.
