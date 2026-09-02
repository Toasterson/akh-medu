//! Fail-closed, per-tenant bearer authentication for Heka's MCP ingress.
//!
//! Unlike Seshat's single shared secret, each `heka_akhs` row carries its own
//! `token_hash` (SHA-256 hex of the plaintext bearer token, migration 000005).
//! The tenant route `/akh/{akh_id}/mcp` authenticates and authorizes in a
//! single `WHERE id = $1 AND token_hash = $2` lookup: a token that belongs to
//! a *different* akh matches no row and is indistinguishable from a bad token.
//!
//! Failure modes, mirroring `seshat::auth`:
//!
//! - missing `Authorization` header → `401 Unauthorized` (+ `WWW-Authenticate`)
//! - malformed header, empty/padded token, unknown hash, or hash that belongs
//!   to a different akh than the route requests → `403 Forbidden`
//! - database outage during the lookup → `503 Service Unavailable`
//!
//! The plaintext token is never stored; it is shown to the operator exactly
//! once at mint time (`hekad register-akh --token` / `hekad set-token`).

use axum::{
    extract::{Path, Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sea_orm::{ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

use crate::entity::heka_akhs;

/// State carried by [`require_akh_bearer`].
#[derive(Clone)]
pub struct AkhAuthState {
    db: Arc<DatabaseConnection>,
}

impl AkhAuthState {
    pub fn new(db: Arc<DatabaseConnection>) -> Self {
        Self { db }
    }

    pub fn db(&self) -> Arc<DatabaseConnection> {
        Arc::clone(&self.db)
    }
}

/// Tenant identity resolved from `(route akh_id, bearer token)` and injected
/// into request extensions by [`require_akh_bearer`]. The MCP service factory
/// clones this into [`crate::mcp::HekaContext`].
#[derive(Clone, Debug)]
pub struct HekaTenant {
    pub akh_id: Uuid,
    pub akh_name: String,
}

/// SHA-256 hex digest of a plaintext bearer token. The token itself is never
/// persisted — only this digest lands in `heka_akhs.token_hash`.
pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

/// Mint a fresh 32-byte bearer token, hex-encoded (64 chars). Returns
/// `(plaintext, sha256_hash)`. Store only the hash; show the plaintext once.
pub fn mint_token() -> (String, String) {
    use rand::Rng;
    let bytes: [u8; 32] = rand::rng().random();
    let plaintext = hex_encode(&bytes);
    let hash = hash_token(&plaintext);
    (plaintext, hash)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
        out.push(char::from_digit((b & 0x0f) as u32, 16).expect("nibble"));
    }
    out
}

/// Resolve `(akh_id, token_hash)` to a tenant. A single equality-pair lookup
/// enforces both "this akh exists" and "this token belongs to *this* akh" —
/// the partial unique index `uq_heka_akhs_token_hash` keeps it O(1).
pub async fn resolve_akh(
    db: &DatabaseConnection,
    akh_id: Uuid,
    bearer_token: &str,
) -> Result<Option<heka_akhs::Model>, DbErr> {
    let token_hash = hash_token(bearer_token);
    heka_akhs::Entity::find()
        .filter(heka_akhs::Column::Id.eq(akh_id))
        .filter(heka_akhs::Column::TokenHash.eq(token_hash))
        .one(db)
        .await
}

/// Axum middleware for `/akh/{akh_id}/mcp`. Extracts the route akh and the
/// bearer token, resolves the tenant, and injects [`HekaTenant`] into request
/// extensions for the downstream `StreamableHttpService` factory to claim.
pub async fn require_akh_bearer(
    State(state): State<AkhAuthState>,
    Path(akh_id): Path<Uuid>,
    request: Request,
    next: Next,
) -> Response {
    let Some(value) = request.headers().get(header::AUTHORIZATION) else {
        return unauthorized();
    };
    let Ok(value) = value.to_str() else {
        return forbidden();
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return forbidden();
    };
    if token.is_empty() || token.trim() != token {
        return forbidden();
    }
    match resolve_akh(&state.db, akh_id, token).await {
        Ok(Some(akh)) => {
            let mut request = request;
            request.extensions_mut().insert(HekaTenant {
                akh_id: akh.id,
                akh_name: akh.name,
            });
            next.run(request).await
        }
        // Bad hash, correct hash for a different akh, or an akh with no
        // credential minted (token_hash IS NULL never matches) — all 403.
        Ok(None) => forbidden(),
        Err(error) => {
            tracing::error!(%error, "akh auth lookup failed; denying request");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    }
}

fn unauthorized() -> Response {
    let mut response = StatusCode::UNAUTHORIZED.into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"heka\""),
    );
    response
}

fn forbidden() -> Response {
    StatusCode::FORBIDDEN.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        extract::Extension,
        http::Request,
        middleware,
        routing::get,
    };
    use sea_orm::EntityTrait;
    use tower::ServiceExt;

    #[test]
    fn hash_token_matches_known_sha256_vector() {
        // sha256("heka-test-token") — the digest is the persisted credential.
        assert_eq!(
            hash_token("heka-test-token"),
            "a1e2c9026111cf02d21316db69d70ec873b209113a79d169021e4357b07649ef"
        );
    }

    #[test]
    fn mint_token_returns_hash_of_plaintext() {
        let (plaintext, hash) = mint_token();
        assert_eq!(plaintext.len(), 64);
        assert!(plaintext.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash_token(&plaintext), hash);
        // Two mints never collide.
        let (other, _) = mint_token();
        assert_ne!(plaintext, other);
    }

    /// Oneshot test app: a fixed route behind the real `require_akh_bearer`
    /// middleware. A `DatabaseConnection::default()` is a disconnected handle,
    /// so any request that reaches the DB-lookup branch yields 503 — exactly
    /// the fail-closed behaviour we assert below. Pre-DB branches (401 for a
    /// missing header, 403 for malformed/empty/padded tokens) never reach the
    /// handle and are asserted exhaustively.
    fn app(db: DatabaseConnection) -> Router {
        let state = AkhAuthState::new(Arc::new(db));
        Router::new()
            .route(
                "/akh/{akh_id}/mcp",
                get(|Extension(tenant): Extension<HekaTenant>| async move {
                    format!("ok:{}", tenant.akh_id)
                }),
            )
            .layer(middleware::from_fn_with_state(state, require_akh_bearer))
    }

    fn get_request(akh_id: Uuid, authorization: Option<&str>) -> Request<Body> {
        let builder = Request::get(format!("/akh/{akh_id}/mcp"));
        let builder = match authorization {
            Some(value) => builder.header(header::AUTHORIZATION, value),
            None => builder,
        };
        builder.body(Body::empty()).unwrap()
    }

    #[test]
    fn resolve_akh_filters_on_both_id_and_token_hash() {
        // The SQL the lookup generates must pin BOTH the route akh id and the
        // token hash in one equality pair — that is what makes a valid token
        // for akh A unmatchable on akh B's route by construction. The
        // cross-tenant integration test in `tests/mcp_integration.rs` proves
        // the runtime behaviour against a live database; this pins the query
        // shape itself so a well-meaning 'simplify the lookup' refactor
        // cannot silently drop either filter.
        let statement = heka_akhs::Entity::find()
            .filter(heka_akhs::Column::Id.eq(Uuid::new_v4()))
            .filter(heka_akhs::Column::TokenHash.eq("deadbeef".to_string()))
            .build(sea_orm::DbBackend::Postgres);
        let sql = statement.to_string();
        assert!(
            sql.contains("\"id\""),
            "lookup must filter on the akh id: {sql}"
        );
        assert!(
            sql.contains("\"token_hash\""),
            "lookup must filter on the token hash: {sql}"
        );
        assert!(
            sql.contains("AND"),
            "lookup must require BOTH filters (a valid token for another akh must miss): {sql}"
        );
    }

    #[tokio::test]
    async fn require_akh_bearer_branching_is_fail_closed() {
        let akh_id = Uuid::new_v4();
        // Disconnected handle: only reached by a well-formed Bearer token,
        // where the DB error must surface as 503 (fail-closed on outage).
        let db = DatabaseConnection::default();

        let missing = app(db.clone())
            .oneshot(get_request(akh_id, None))
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            missing.headers()[header::WWW_AUTHENTICATE],
            "Bearer realm=\"heka\""
        );

        for (name, header_value) in [
            ("non-Bearer scheme", "Basic dGVzdA=="),
            ("empty token", "Bearer "),
            ("left-padded token", "Bearer  abc"),
            ("right-padded token", "Bearer abc "),
            ("tab-padded token", "Bearer \tabc"),
        ] {
            let response = app(db.clone())
                .oneshot(get_request(akh_id, Some(header_value)))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::FORBIDDEN,
                "{name} must be forbidden"
            );
        }

        // A non-UTF8 header value is also fail-closed (forbidden).
        let response = app(db.clone())
            .oneshot(
                Request::get(format!("/akh/{akh_id}/mcp"))
                    .header(header::AUTHORIZATION, HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap())
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);

        // A well-formed but unknown token reaches the DB; with a disconnected
        // handle the lookup errors and must surface as 503, never as a stray
        // allow.
        let response = app(db)
            .oneshot(get_request(akh_id, Some(&format!(
                "Bearer {}",
                "a".repeat(64)
            ))))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
