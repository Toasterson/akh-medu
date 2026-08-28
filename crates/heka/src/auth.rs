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

    /// Tenant isolation at the SQL level: a cross-tenant token must constrain
    /// *both* the route akh id and the token hash, so "valid hash, wrong akh"
    /// matches zero rows exactly like a bad token does.
    #[tokio::test]
    async fn cross_tenant_lookup_filters_on_both_predicates() {
        let db = sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres)
            .append_query_results::<heka_akhs::Model>(Vec::<Vec<heka_akhs::Model>>::new())
            .into_connection();

        let akh_a = Uuid::new_v4();
        assert!(resolve_akh(&db, akh_a, "token-b").await.unwrap().is_none());

        let log = db.into_transaction_log();
        let sql = log[0].sql();
        assert!(sql.contains("\"id\" ="), "route akh predicate: {sql}");
        assert!(sql.contains("\"token_hash\" ="), "token predicate: {sql}");
    }

    /// Header-shape parsing is exercised over a live router (no DB needed):
    /// 401 only when the header is *absent*; malformed/blank/cross-tenant all
    /// 403 because no (id, hash) pair can match.
    mod http_shape {
        use super::*;
        use axum::{Router, body::Body, middleware, response::Json, routing::post};
        use tower::ServiceExt;

        fn app() -> Router {
            let db = sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres)
                .append_query_results::<heka_akhs::Model>(vec![
                    Vec::<heka_akhs::Model>::new(),
                    Vec::<heka_akhs::Model>::new(),
                    Vec::<heka_akhs::Model>::new(),
                ])
                .into_connection();
            let state = AkhAuthState::new(Arc::new(db));
            Router::new()
                .route(
                    "/akh/{akh_id}/mcp",
                    post(|Json(_body): Json<serde_json::Value>| async {
                        StatusCode::OK
                    }),
                )
                .route_layer(middleware::from_fn_with_state(state, require_akh_bearer))
        }

        #[tokio::test]
        async fn missing_header_is_unauthorized_with_realm() {
            let response = app()
                .oneshot(
                    Request::post(format!("/akh/{}/mcp", Uuid::new_v4()))
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(
                response.headers()[header::WWW_AUTHENTICATE],
                "Bearer realm=\"heka\""
            );
        }

        #[tokio::test]
        async fn malformed_and_unknown_bearer_are_forbidden() {
            for header_value in ["Token abc", "Bearer ", "Bearer  padded ", "Bearer unknown"] {
                let response = app()
                    .oneshot(
                        Request::post(format!("/akh/{}/mcp", Uuid::new_v4()))
                            .header(header::CONTENT_TYPE, "application/json")
                            .header(header::AUTHORIZATION, header_value)
                            .body(Body::from("{}"))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::FORBIDDEN,
                    "header: {header_value:?}"
                );
            }
        }
    }
}
