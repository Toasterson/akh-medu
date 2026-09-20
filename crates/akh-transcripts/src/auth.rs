//! Fail-closed bearer authentication for the `akh-transcripts-mcp` HTTP ingress.
//!
//! Ported from `crates/seshat/src/auth.rs` per
//! `docs/wayfinder/transcript-harvest/tickets/read-mcp-surface.md` §3: same
//! constant-time comparison, same `WWW-Authenticate` challenge, same
//! reloadable-file-or-inline shape. Only the environment variable names and
//! the realm differ. The read surface is deliberately **not** per-caller
//! scoped — the bearer token is the whole protection model for a
//! single-operator deployment (see the ticket's "No per-caller scoping"
//! section) — so there is exactly one axum middleware layer here, not a
//! per-tool authorization check.

use std::{path::PathBuf, sync::Arc};

use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Environment variable containing an inline token (primarily for local use).
pub const TOKEN_ENV: &str = "AKH_TRANSCRIPTS_BEARER_TOKEN";
/// Environment variable pointing at a projected Kubernetes Secret file.
pub const TOKEN_FILE_ENV: &str = "AKH_TRANSCRIPTS_BEARER_TOKEN_FILE";

#[derive(Clone)]
enum SecretSource {
    Inline(Arc<str>),
    File(Arc<PathBuf>),
}

/// Reloadable set of accepted bearer tokens.
///
/// A file is read for every request so a projected-Secret rotation takes
/// effect without restarting the server. Each non-empty line is one accepted
/// token, so old and new credentials can overlap during rotation.
#[derive(Clone)]
pub struct SharedSecretAuth {
    source: SecretSource,
}

impl SharedSecretAuth {
    /// Load the mandatory secret source from the process environment.
    ///
    /// **Fail-closed**: the contract requires the server to refuse to start
    /// without a token rather than come up unauthenticated, so this returns
    /// `Err` (never a silently-open server) when neither variable is set, or
    /// when the configured source is empty.
    pub fn from_env() -> Result<Self, String> {
        if let Some(path) = std::env::var_os(TOKEN_FILE_ENV) {
            return Self::from_file(path.into());
        }
        let value = std::env::var(TOKEN_ENV)
            .map_err(|_| format!("neither {TOKEN_FILE_ENV} nor {TOKEN_ENV} is set"))?;
        Self::from_inline(value)
    }

    pub fn from_file(path: PathBuf) -> Result<Self, String> {
        let auth = Self {
            source: SecretSource::File(Arc::new(path)),
        };
        auth.tokens()?;
        Ok(auth)
    }

    pub fn from_inline(value: impl Into<String>) -> Result<Self, String> {
        let auth = Self {
            source: SecretSource::Inline(Arc::from(value.into())),
        };
        auth.tokens()?;
        Ok(auth)
    }

    fn tokens(&self) -> Result<Vec<String>, String> {
        let contents = match &self.source {
            SecretSource::Inline(value) => value.to_string(),
            SecretSource::File(path) => std::fs::read_to_string(path.as_ref())
                .map_err(|e| format!("cannot read bearer-token file {}: {e}", path.display()))?,
        };
        let tokens: Vec<_> = contents
            .lines()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_owned)
            .collect();
        if tokens.is_empty() {
            Err("bearer-token source contains no non-empty tokens".into())
        } else {
            Ok(tokens)
        }
    }

    fn authorizes(&self, candidate: &str) -> Result<bool, String> {
        Ok(self
            .tokens()?
            .iter()
            .any(|expected| constant_time_eq(expected.as_bytes(), candidate.as_bytes())))
    }
}

fn constant_time_eq(expected: &[u8], candidate: &[u8]) -> bool {
    let mut difference = expected.len() ^ candidate.len();
    let width = expected.len().max(candidate.len());
    for index in 0..width {
        difference |= usize::from(
            expected.get(index).copied().unwrap_or(0) ^ candidate.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

/// Axum middleware enforcing `Authorization: Bearer <shared-secret>`.
///
/// Applied only to the `/mcp` route in the binary — `/healthz` and `/readyz`
/// stay open so a Kubernetes probe (which never carries a bearer token) can
/// reach them.
pub async fn require_bearer(
    State(auth): State<SharedSecretAuth>,
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
    match auth.authorizes(token) {
        Ok(true) => next.run(request).await,
        Ok(false) => forbidden(),
        Err(error) => {
            tracing::error!(%error, "bearer-token source unavailable; denying request");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    }
}

fn unauthorized() -> Response {
    let mut response = StatusCode::UNAUTHORIZED.into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"akh-transcripts\""),
    );
    response
}

fn forbidden() -> Response {
    StatusCode::FORBIDDEN.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, middleware, routing::get};
    use tower::ServiceExt;

    fn app(auth: SharedSecretAuth) -> Router {
        Router::new()
            .route("/mcp", get(|| async { StatusCode::OK }))
            .layer(middleware::from_fn_with_state(auth, require_bearer))
    }

    #[tokio::test]
    async fn missing_token_is_401_with_challenge() {
        let auth = SharedSecretAuth::from_inline("current-token").unwrap();
        let response = app(auth)
            .oneshot(Request::get("/mcp").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()[header::WWW_AUTHENTICATE],
            "Bearer realm=\"akh-transcripts\""
        );
    }

    #[tokio::test]
    async fn wrong_token_is_403() {
        let auth = SharedSecretAuth::from_inline("current-token").unwrap();
        let response = app(auth)
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn right_token_is_200() {
        let auth = SharedSecretAuth::from_inline("current-token").unwrap();
        let response = app(auth)
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer current-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn current_and_previous_credentials_are_both_accepted() {
        let auth = SharedSecretAuth::from_inline("current-token\nprevious-token\n").unwrap();
        for token in ["current-token", "previous-token"] {
            let response = app(auth.clone())
                .oneshot(
                    Request::get("/mcp")
                        .header(header::AUTHORIZATION, format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn projected_secret_file_is_reloaded_without_restart() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "old-token\n").unwrap();
        let auth = SharedSecretAuth::from_file(file.path().to_owned()).unwrap();

        let old = app(auth.clone())
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer old-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(old.status(), StatusCode::OK);

        std::fs::write(file.path(), "new-token\n").unwrap();
        let retired = app(auth)
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer old-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retired.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn empty_inline_token_is_rejected_fail_closed() {
        assert!(SharedSecretAuth::from_inline("").is_err());
        assert!(SharedSecretAuth::from_inline("   \n  \n").is_err());
    }

    #[test]
    fn from_env_fails_closed_when_unset() {
        // SAFETY-note (not `unsafe`): just env var manipulation local to this
        // test process; serial by virtue of `cargo test` isolating processes
        // per test binary run, but this test avoids asserting on global env
        // state left by sibling tests by only checking the *absence* case
        // with variables it does not otherwise touch.
        let had_token = std::env::var_os(TOKEN_ENV);
        let had_file = std::env::var_os(TOKEN_FILE_ENV);
        unsafe {
            std::env::remove_var(TOKEN_ENV);
            std::env::remove_var(TOKEN_FILE_ENV);
        }
        assert!(SharedSecretAuth::from_env().is_err());
        unsafe {
            if let Some(v) = had_token {
                std::env::set_var(TOKEN_ENV, v);
            }
            if let Some(v) = had_file {
                std::env::set_var(TOKEN_FILE_ENV, v);
            }
        }
    }
}
