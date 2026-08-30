//! Fail-closed bearer authentication for Seshat's HTTP ingress.

use std::{path::PathBuf, sync::Arc};

use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Environment variable containing an inline token (primarily for local use).
pub const SECRET_ENV: &str = "SESHAT_SHARED_SECRET";
/// Environment variable pointing at the projected Kubernetes Secret file.
pub const SECRET_FILE_ENV: &str = "SESHAT_SHARED_SECRET_FILE";

#[derive(Clone)]
enum SecretSource {
    Inline(Arc<str>),
    File(Arc<PathBuf>),
}

/// Reloadable set of accepted bearer tokens.
///
/// A file is read for every request so Kubernetes projected-Secret updates take
/// effect without restarting seshd. Each non-empty line is one accepted token,
/// allowing old and new credentials to overlap during rotation.
#[derive(Clone)]
pub struct SharedSecretAuth {
    source: SecretSource,
}

impl SharedSecretAuth {
    /// Load the mandatory secret source from the process environment.
    pub fn from_env() -> Result<Self, String> {
        if let Some(path) = std::env::var_os(SECRET_FILE_ENV) {
            return Self::from_file(path.into());
        }
        let value = std::env::var(SECRET_ENV)
            .map_err(|_| format!("neither {SECRET_FILE_ENV} nor {SECRET_ENV} is set"))?;
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
                .map_err(|e| format!("cannot read shared-secret file {}: {e}", path.display()))?,
        };
        let tokens: Vec<_> = contents
            .lines()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_owned)
            .collect();
        if tokens.is_empty() {
            Err("shared-secret source contains no non-empty tokens".into())
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
            tracing::error!(%error, "shared-secret source unavailable; denying request");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    }
}

fn unauthorized() -> Response {
    let mut response = StatusCode::UNAUTHORIZED.into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"seshat\""),
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
    async fn missing_and_invalid_credentials_are_rejected() {
        let auth = SharedSecretAuth::from_inline("current-token\nprevious-token\n").unwrap();

        let missing = app(auth.clone())
            .oneshot(Request::get("/mcp").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            missing.headers()[header::WWW_AUTHENTICATE],
            "Bearer realm=\"seshat\""
        );

        let invalid = app(auth)
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn current_and_previous_credentials_are_accepted() {
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
        let new = app(auth.clone())
            .oneshot(
                Request::get("/mcp")
                    .header(header::AUTHORIZATION, "Bearer new-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(new.status(), StatusCode::OK);

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
}
