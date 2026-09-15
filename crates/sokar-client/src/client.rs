use crate::*;
use reqwest::{Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use url::Url;

#[derive(Clone)]
pub struct Client {
    base: Url,
    bearer: String,
    http: reqwest::Client,
}
impl Client {
    pub fn new(base_url: impl AsRef<str>, bearer: impl Into<String>) -> Result<Self, Error> {
        let base =
            Url::parse(base_url.as_ref()).map_err(|_| Error::Invalid("invalid base URL".into()))?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::Invalid(
                "expected HTTP(S) base URL without credentials, query or fragment".into(),
            ));
        }
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            base,
            bearer: bearer.into(),
            http,
        })
    }
    fn request(&self, method: Method, segments: &[&str]) -> RequestBuilder {
        let mut url = self.base.clone();
        url.path_segments_mut()
            .expect("HTTP URL")
            .pop_if_empty()
            .extend(segments);
        self.http.request(method, url).bearer_auth(&self.bearer)
    }
    async fn checked(request: RequestBuilder) -> Result<Response, Error> {
        let response = request.send().await?;
        if response.status().is_success() || response.status() == 304 {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let problem = response.json::<Problem>().await.unwrap_or(Problem {
            type_url: "about:blank".into(),
            title: "HTTP error".into(),
            status,
            detail: "server returned a non-problem error response".into(),
            code: "sokar_client::http".into(),
            request_id,
        });
        Err(Error::Remote(Box::new(problem)))
    }
    async fn json<T: DeserializeOwned>(request: RequestBuilder) -> Result<T, Error> {
        Ok(Self::checked(request).await?.json().await?)
    }
    pub async fn create_collection(&self, input: &CreateCollection) -> Result<Collection, Error> {
        Self::json(
            self.request(Method::POST, &["v1", "collections"])
                .json(input),
        )
        .await
    }
    pub async fn collection(&self, id: CollectionId) -> Result<Collection, Error> {
        Self::json(self.request(Method::GET, &["v1", "collections", &id.to_string()])).await
    }
    pub async fn collection_by_scope(&self, scope: &CollectionScope) -> Result<Collection, Error> {
        Self::json(
            self.request(Method::GET, &["v1", "collections", "by-scope"])
                .query(&[("scope_key", scope.key())]),
        )
        .await
    }
    pub async fn upload(
        &self,
        collection: CollectionId,
        options: &UploadOptions,
        body: impl Into<reqwest::Body>,
    ) -> Result<UploadResponse, Error> {
        let mut request = self
            .request(
                Method::POST,
                &["v1", "collections", &collection.to_string(), "artifacts"],
            )
            .header("Sokar-Path", options.path.as_str())
            .header("Content-Type", options.media_type.as_str())
            .header("Sokar-Kind", options.kind.to_string())
            .header("Sokar-Labels", serde_json::to_string(&options.labels)?)
            .body(body);
        if let Some(key) = &options.idempotency_key {
            request = request.header("Idempotency-Key", key);
        }
        Self::json(request).await
    }
    pub async fn upload_multipart(
        &self,
        collection: CollectionId,
        options: &UploadOptions,
        file: reqwest::multipart::Part,
    ) -> Result<UploadResponse, Error> {
        let form = reqwest::multipart::Form::new()
            .text("path", options.path.to_string())
            .part("file", file.mime_str(options.media_type.as_str())?);
        let mut request = self
            .request(
                Method::POST,
                &["v1", "collections", &collection.to_string(), "artifacts"],
            )
            .header("Sokar-Kind", options.kind.to_string())
            .header("Sokar-Labels", serde_json::to_string(&options.labels)?)
            .multipart(form);
        if let Some(key) = &options.idempotency_key {
            request = request.header("Idempotency-Key", key);
        }
        Self::json(request).await
    }
    pub async fn artifact(&self, id: ArtifactId) -> Result<Artifact, Error> {
        Self::json(self.request(Method::GET, &["v1", "artifacts", &id.to_string()])).await
    }
    /// Returns status, headers and a streaming body (including 206 and 304).
    pub async fn content(
        &self,
        id: ArtifactId,
        range: Option<&str>,
        if_none_match: Option<&str>,
    ) -> Result<Response, Error> {
        let mut request = self.request(
            Method::GET,
            &["v1", "artifacts", &id.to_string(), "content"],
        );
        if let Some(range) = range {
            request = request.header("Range", range);
        }
        if let Some(etag) = if_none_match {
            request = request.header("If-None-Match", etag);
        }
        Self::checked(request).await
    }
    pub async fn list_artifacts(
        &self,
        id: CollectionId,
        query: &ListArtifacts,
    ) -> Result<ArtifactPage, Error> {
        Self::json(
            self.request(
                Method::GET,
                &["v1", "collections", &id.to_string(), "artifacts"],
            )
            .query(query),
        )
        .await
    }
    pub async fn artifact_by_path(
        &self,
        id: CollectionId,
        path: &ArtifactPath,
    ) -> Result<Artifact, Error> {
        let id = id.to_string();
        let mut parts = vec!["v1", "collections", &id, "artifacts", "by-path"];
        parts.extend(path.as_str().split('/'));
        Self::json(self.request(Method::GET, &parts)).await
    }
    pub async fn delete_artifact(&self, id: ArtifactId) -> Result<(), Error> {
        Self::checked(self.request(Method::DELETE, &["v1", "artifacts", &id.to_string()])).await?;
        Ok(())
    }
}
