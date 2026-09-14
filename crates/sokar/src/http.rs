use crate::{
    Sokar, SokarError, SokarResult,
    authz::{Permission, Principal},
};
use axum::{
    Extension, Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::{StreamExt, TryStreamExt};
use sokar_client::*;
use std::sync::Arc;
use tower_http::{
    limit::RequestBodyLimitLayer,
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};

pub fn api(state: Arc<Sokar>) -> Router {
    Router::new()
        .route("/v1/collections", post(create_collection))
        .route("/v1/collections/by-scope", get(by_scope))
        .route("/v1/collections/{id}", get(collection))
        .route("/v1/collections/{id}/artifacts", post(upload).get(list))
        .route(
            "/v1/collections/{id}/artifacts/by-path/{*path}",
            get(by_path),
        )
        .route("/v1/artifacts/{id}", get(artifact).delete(delete))
        .route("/v1/artifacts/{id}/content", get(content))
        .layer(DefaultBodyLimit::max(state.config.limits.max_body_bytes))
        .layer(RequestBodyLimitLayer::new(
            state.config.limits.max_body_bytes,
        ))
        .layer(middleware::from_fn_with_state(state.clone(), api_boundary))
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .with_state(state)
}
pub fn admin(state: Arc<Sokar>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { StatusCode::OK }))
        .route("/readyz", get(ready))
        .route("/metrics", get(metrics))
        .layer(middleware::from_fn(problem_boundary))
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .with_state(state)
}
impl IntoResponse for SokarError {
    fn into_response(self) -> Response {
        let status = self.status();
        let mut response = (status, Json(self.problem(String::new()))).into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/problem+json"),
        );
        if matches!(self, Self::Authn) {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"sokar\""),
            );
        }
        if let Self::Range { size } = self {
            response.headers_mut().insert(
                header::CONTENT_RANGE,
                format!("bytes */{size}").parse().expect("valid size"),
            );
        }
        response
    }
}
fn request_id(request: &Request) -> String {
    request
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_owned()
}
async fn api_boundary(
    State(state): State<Arc<Sokar>>,
    mut request: Request,
    next: Next,
) -> Response {
    let id = request_id(&request);
    let response = match crate::auth::authenticate(&state.config.principals, request.headers()) {
        Ok(principal) => {
            let permission = match *request.method() {
                axum::http::Method::POST => Permission::Write,
                axum::http::Method::DELETE => Permission::Delete,
                _ => Permission::Read,
            };
            match state
                .authorizer
                .authorize(&principal, request.uri().path(), permission)
            {
                Ok(()) => {
                    request.extensions_mut().insert(principal);
                    next.run(request).await
                }
                Err(e) => e.into_response(),
            }
        }
        Err(e) => e.into_response(),
    };
    normalize_problem(response, id).await
}
async fn problem_boundary(request: Request, next: Next) -> Response {
    let id = request_id(&request);
    normalize_problem(next.run(request).await, id).await
}
/// Extractor/routing/body-limit rejections use the same problem envelope as service errors.
async fn normalize_problem(response: Response, id: String) -> Response {
    if !response.status().is_client_error() && !response.status().is_server_error() {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 64 * 1024)
        .await
        .unwrap_or_default();
    let mut problem = serde_json::from_slice::<Problem>(&bytes).unwrap_or_else(|_| Problem {
        type_url: "urn:sokar:request".into(),
        title: parts.status.canonical_reason().unwrap_or("Error").into(),
        status: parts.status.as_u16(),
        detail: parts
            .status
            .canonical_reason()
            .unwrap_or("Request failed")
            .into(),
        code: if parts.status == StatusCode::PAYLOAD_TOO_LARGE {
            "sokar::body_limit"
        } else {
            "sokar::request"
        }
        .into(),
        request_id: String::new(),
    });
    problem.request_id = id;
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/problem+json"),
    );
    Response::from_parts(
        parts,
        Body::from(serde_json::to_vec(&problem).expect("problem serialization")),
    )
}
async fn create_collection(
    State(state): State<Arc<Sokar>>,
    Extension(p): Extension<Principal>,
    Json(input): Json<CreateCollection>,
) -> SokarResult<Response> {
    let (collection, created) = state.create_collection_outcome(&p.id, input).await?;
    Ok((
        if created {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        },
        Json(collection),
    )
        .into_response())
}
async fn collection(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<CollectionId>,
) -> SokarResult<Json<Collection>> {
    Ok(Json(state.index.collection(id).await?))
}
#[derive(serde::Deserialize)]
struct ScopeQuery {
    scope_key: String,
}
async fn by_scope(
    State(state): State<Arc<Sokar>>,
    Query(query): Query<ScopeQuery>,
) -> SokarResult<Json<Collection>> {
    // Accept canonical scope keys and normalize JSON object field order.
    let scope: CollectionScope = serde_json::from_str(&query.scope_key)
        .map_err(|_| SokarError::Invalid("scope_key must be a JSON CollectionScope".into()))?;
    Ok(Json(state.index.by_scope(&scope.key()).await?))
}
async fn artifact(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<ArtifactId>,
) -> SokarResult<Json<Artifact>> {
    Ok(Json(state.index.artifact(id).await?))
}
async fn list(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<CollectionId>,
    Query(query): Query<ListArtifacts>,
) -> SokarResult<Json<ArtifactPage>> {
    state.index.collection(id).await?;
    Ok(Json(state.index.list(id, &query).await?))
}
async fn by_path(
    State(state): State<Arc<Sokar>>,
    Path((id, path)): Path<(CollectionId, String)>,
) -> SokarResult<Json<Artifact>> {
    Ok(Json(state.index.latest(id, &path.parse()?).await?))
}
async fn delete(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<ArtifactId>,
) -> SokarResult<StatusCode> {
    state.delete_artifact(id).await?;
    Ok(StatusCode::NO_CONTENT)
}
async fn content(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<ArtifactId>,
    headers: HeaderMap,
) -> SokarResult<Response> {
    let a = state.index.artifact(id).await?;
    let c = state.index.collection(a.collection).await?;
    let store = state.stores.get(&c.store)?;
    Ok(
        crate::delivery::content(&store, &a, &headers, &state.config.delivery)
            .await?
            .into_response(),
    )
}
fn text_header<'a>(headers: &'a HeaderMap, name: &str) -> SokarResult<Option<&'a str>> {
    // HeaderValue::to_str excludes UTF-8; artifact paths and JSON labels permit it.
    headers
        .get(name)
        .map(|h| {
            std::str::from_utf8(h.as_bytes())
                .map_err(|_| SokarError::Invalid(format!("invalid {name} header")))
        })
        .transpose()
}
fn required_header<'a>(headers: &'a HeaderMap, name: &str) -> SokarResult<&'a str> {
    text_header(headers, name)?.ok_or_else(|| SokarError::Invalid(format!("missing {name} header")))
}
async fn upload(
    State(state): State<Arc<Sokar>>,
    Path(id): Path<CollectionId>,
    Extension(p): Extension<Principal>,
    request: Request,
) -> SokarResult<Response> {
    let collection = state.index.collection(id).await?;
    let store = state.stores.get(&collection.store)?;
    let headers = request.headers();
    let kind = required_header(headers, "sokar-kind")?.parse()?;
    let labels = text_header(headers, "sokar-labels")?
        .map(serde_json::from_str)
        .transpose()
        .map_err(|_| SokarError::Invalid("Sokar-Labels must be a JSON string map".into()))?
        .unwrap_or_default();
    let idempotency_key = text_header(headers, "idempotency-key")?.map(str::to_owned);
    let content_type = required_header(headers, "content-type")?
        .parse::<mime::Mime>()
        .map_err(|_| SokarError::Invalid("invalid Content-Type".into()))?;
    let (options, stage) =
        if content_type.type_() == mime::MULTIPART && content_type.subtype() == mime::FORM_DATA {
            let mut multipart = Multipart::from_request(request, &())
                .await
                .map_err(|e| SokarError::Invalid(e.to_string()))?;
            let mut path = None;
            let mut file = None;
            let mut media_type = None;
            while let Some(mut field) = multipart.next_field().await.map_err(multipart_error)? {
                match field.name() {
                    Some("path") if path.is_none() => {
                        let mut bytes = Vec::new();
                        while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
                            if bytes.len() + chunk.len() > 4096 {
                                return Err(SokarError::Invalid(
                                    "multipart path exceeds 4096 bytes".into(),
                                ));
                            }
                            bytes.extend_from_slice(&chunk);
                        }
                        path = Some(
                            String::from_utf8(bytes)
                                .map_err(|_| SokarError::Invalid("path must be UTF-8".into()))?
                                .parse()?,
                        );
                    }
                    Some("file") if file.is_none() => {
                        media_type = Some(
                            field
                                .content_type()
                                .unwrap_or("application/octet-stream")
                                .parse()?,
                        );
                        file = Some(
                            store
                                .stage(
                                    field.map_err(multipart_error),
                                    state.config.limits.max_body_bytes,
                                )
                                .await?,
                        );
                    }
                    _ => {
                        return Err(SokarError::Invalid(
                            "multipart requires exactly one path and one file".into(),
                        ));
                    }
                }
            }
            (
                UploadOptions {
                    path: path.ok_or_else(|| SokarError::Invalid("missing path field".into()))?,
                    media_type: media_type
                        .ok_or_else(|| SokarError::Invalid("missing file part".into()))?,
                    kind,
                    labels,
                    idempotency_key,
                },
                file.ok_or_else(|| SokarError::Invalid("missing file part".into()))?,
            )
        } else {
            let options = UploadOptions {
                path: required_header(headers, "sokar-path")?.parse()?,
                media_type: content_type.to_string().parse()?,
                kind,
                labels,
                idempotency_key,
            };
            let stream = request.into_body().into_data_stream().map(|r| {
                r.map_err(|e| {
                    use std::error::Error;
                    let mut source: &(dyn Error + 'static) = &e;
                    loop {
                        if source.is::<http_body_util::LengthLimitError>() {
                            return SokarError::BodyLimit;
                        }
                        match source.source() {
                            Some(next) => source = next,
                            None => break,
                        }
                    }
                    SokarError::Invalid("upload body interrupted".into())
                })
            });
            let stage = store
                .stage(stream, state.config.limits.max_body_bytes)
                .await?;
            (options, stage)
        };
    Ok((
        StatusCode::CREATED,
        Json(
            state
                .finish_upload(&p.id, &collection, options, stage)
                .await?,
        ),
    )
        .into_response())
}
fn multipart_error(error: axum::extract::multipart::MultipartError) -> SokarError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        SokarError::BodyLimit
    } else {
        SokarError::Invalid("invalid multipart body".into())
    }
}
async fn ready(State(state): State<Arc<Sokar>>) -> SokarResult<StatusCode> {
    state.index.ping().await?;
    state.stores.check(false).await?;
    Ok(StatusCode::OK)
}
async fn metrics(State(state): State<Arc<Sokar>>) -> SokarResult<Response> {
    let mut output = String::new();
    prometheus_client::encoding::text::encode(&mut output, &state.stores.metrics)
        .map_err(|_| SokarError::Invalid("metrics encoding failed".into()))?;
    Ok((
        [(
            header::CONTENT_TYPE,
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
        )],
        output,
    )
        .into_response())
}
