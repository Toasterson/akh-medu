//! Shared authenticated transport. Tokens stay in mounted files and never enter wire records.
use crate::{
    config::{Config, Principal},
    types::*,
};
use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub struct ApiError(pub Fault);
impl From<Fault> for ApiError {
    fn from(fault: Fault) -> Self {
        Self(fault)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0.code.as_str() {
            "unauthorized" | "auth_unavailable" => StatusCode::UNAUTHORIZED,
            "forbidden" => StatusCode::FORBIDDEN,
            "not_found" => StatusCode::NOT_FOUND,
            "conflict" | "unknown_outcome" => StatusCode::CONFLICT,
            "busy" => StatusCode::TOO_MANY_REQUESTS,
            "invalid_request" | "expired" | "unsupported" => StatusCode::BAD_REQUEST,
            "transport" => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self.0)).into_response()
    }
}
pub type Api<T> = std::result::Result<Json<T>, ApiError>;

fn token_equal(a: &[u8], b: &[u8]) -> bool {
    let mut difference = a.len() ^ b.len();
    for index in 0..a.len().max(b.len()) {
        difference |=
            (a.get(index).copied().unwrap_or(0) ^ b.get(index).copied().unwrap_or(0)) as usize;
    }
    difference == 0
}

pub fn authenticate(config: &Config, headers: &HeaderMap) -> Result<Principal> {
    if let Some(origin) = headers.get("origin") {
        let origin = origin
            .to_str()
            .map_err(|_| Fault::new("forbidden", "invalid origin"))?;
        if !config.allowed_origins.iter().any(|o| o == origin) {
            return Err(Fault::new("forbidden", "origin is not allowed"));
        }
    }
    let supplied = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .filter(|s| s.len() <= 4096)
        .ok_or_else(|| Fault::new("unauthorized", "bearer authentication required"))?;
    // Do not stop at the first match; also detect accidentally shared credentials.
    let mut matches = vec![];
    for principal in &config.principals {
        if let Ok(token) = crate::config::read_token(&principal.token_file) {
            if token_equal(supplied.as_bytes(), token.as_bytes()) {
                matches.push(principal);
            }
        }
    }
    if matches.len() != 1 {
        return Err(Fault::new(
            "unauthorized",
            "credential is unavailable, invalid or ambiguous",
        ));
    }
    Ok(matches[0].clone())
}

pub async fn authorization(
    State(config): State<Arc<Config>>,
    mut request: Request,
    next: Next,
) -> Response {
    match authenticate(&config, request.headers()) {
        Ok(principal) => {
            request.extensions_mut().insert(principal);
            next.run(request).await
        }
        Err(error) => ApiError(error).into_response(),
    }
}

pub fn role(principal: &Principal, required: &str) -> Result<()> {
    if principal.has_role(required) {
        Ok(())
    } else {
        Err(Fault::new("forbidden", format!("{required} role required")))
    }
}

/// Percent encode a single path segment, never a complete path or a credential.
pub fn segment(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

#[derive(Clone)]
pub struct Client {
    config: Arc<Config>,
    clients: BTreeMap<String, reqwest::Client>,
    sandbox: Option<String>,
}
impl Client {
    pub fn new(config: Arc<Config>) -> Result<Self> {
        let clients = config
            .endpoints
            .iter()
            .map(|(name, endpoint)| Ok((name.clone(), endpoint.http_client()?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            config,
            clients,
            sandbox: None,
        })
    }
    pub fn in_sandbox(&self, id: &str) -> Result<Self> {
        if id.is_empty()
            || id.len() > 63
            || !id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(Fault::invalid("sandbox ID must be a DNS label"));
        }
        let mut client = self.clone();
        client.sandbox = Some(id.into());
        Ok(client)
    }
    pub async fn call<T: DeserializeOwned>(
        &self,
        service: &str,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<T> {
        self.call_with_headers(service, method, path, body)
            .await
            .map(|(value, _)| value)
    }
    pub async fn call_with_headers<T: DeserializeOwned>(
        &self,
        service: &str,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<(T, HeaderMap)> {
        let endpoint = self.config.endpoint(service)?;
        let route = endpoint.route(path);
        let route = if route.contains("{sandbox}") {
            route.replace(
                "{sandbox}",
                self.sandbox.as_deref().ok_or_else(|| {
                    Fault::invalid("sandbox-scoped endpoint requires an allocation")
                })?,
            )
        } else {
            route
        };
        let mut request = self
            .clients
            .get(service)
            .ok_or_else(|| Fault::invalid("service has no configured HTTP client"))?
            .request(method, route)
            .bearer_auth(endpoint.token()?);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let mut response = request.send().await.map_err(|_| {
            Fault::new(
                "transport",
                format!("{service} request failed; effects may have occurred"),
            )
        })?;
        let status = response.status();
        let headers = response.headers().clone();
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Fault::new("transport", "incomplete service response"))?
        {
            if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
                return Err(Fault::new("transport", "service response exceeds 16 MiB"));
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            let code = match status.as_u16() {
                404 => "not_found",
                409 => "conflict",
                401 => "unauthorized",
                403 => "forbidden",
                429 => "busy",
                _ => "transport",
            };
            return Err(serde_json::from_slice::<Fault>(&bytes).unwrap_or_else(|_| {
                Fault::new(code, format!("{service} returned HTTP {}", status.as_u16()))
            }));
        }
        serde_json::from_slice(&bytes)
            .map(|value| (value, headers))
            .map_err(|_| Fault::new("transport", "service returned an invalid response"))
    }
    pub async fn get<T: DeserializeOwned>(&self, service: &str, path: &str) -> Result<T> {
        self.call(service, reqwest::Method::GET, path, None).await
    }
    pub async fn post<T: DeserializeOwned>(
        &self,
        service: &str,
        path: &str,
        body: &impl Serialize,
    ) -> Result<T> {
        let body = serde_json::to_value(body).map_err(|e| Fault::invalid(e.to_string()))?;
        self.call(service, reqwest::Method::POST, path, Some(body))
            .await
    }
}
