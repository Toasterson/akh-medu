use super::SokarMcp;
use crate::{SokarError, SokarResult, authz::Permission};
use base64::{Engine, engine::general_purpose::STANDARD};
use bytes::Bytes;
use miette::Diagnostic;
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    schemars::{self, JsonSchema},
    tool, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sokar_client::*;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct CollectionParams {
    /// Collection identity: named, akh_workspace, akh_session, anima_session or anima_work_item.
    scope: CollectionScope,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct PutTextParams {
    /// UUIDv7 returned by sokar_collection.
    collection_id: String,
    /// Relative NFC path within the collection, e.g. reports/result.txt. No dot or empty components.
    path: String,
    /// Complete UTF-8 text to store; limited by mcp.max_inline_bytes.
    text: String,
    /// Concrete MIME type; defaults to text/plain; charset=utf-8.
    #[serde(default = "text_media_type")]
    media_type: String,
    /// Artifact category; defaults to document.
    #[serde(default = "document_kind")]
    kind: ArtifactKind,
    /// Optional string metadata labels.
    labels: Option<Labels>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct PutBase64Params {
    /// UUIDv7 returned by sokar_collection.
    collection_id: String,
    /// Relative NFC path within the collection, e.g. images/chart.png. No dot or empty components.
    path: String,
    /// Standard padded base64, without whitespace or a data URL prefix. Decoded bytes are bounded.
    base64: String,
    /// Concrete MIME type of the decoded bytes, e.g. image/png.
    media_type: String,
    /// Artifact category; defaults to other.
    #[serde(default)]
    kind: ArtifactKind,
    /// Optional string metadata labels.
    labels: Option<Labels>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ListParams {
    /// Collection UUIDv7 to list.
    collection_id: String,
    /// Optional path prefix filter.
    prefix: Option<String>,
    /// Optional artifact category filter.
    kind: Option<ArtifactKind>,
    /// Page size; defaults to 50 (service maximum 1000).
    #[serde(default = "list_limit")]
    limit: u32,
    /// Opaque next_cursor from the preceding page, retaining the same filters.
    cursor: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ArtifactParams {
    /// Artifact version UUIDv7, returned by an upload or lookup tool.
    artifact_id: String,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct FindParams {
    /// Collection UUIDv7 to search.
    collection_id: String,
    /// Exact relative NFC path; returns its latest live version.
    path: String,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ReadTextParams {
    /// Artifact version UUIDv7 to read.
    artifact_id: String,
    /// Maximum full content size in bytes; defaults to 1 MiB and is capped by mcp.max_inline_bytes. Never truncates.
    #[serde(default = "text_limit")]
    max_bytes: usize,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ReadBase64Params {
    /// Artifact version UUIDv7 to read.
    artifact_id: String,
    /// Maximum decoded content size; defaults to 8 MiB and is capped by mcp.max_inline_bytes. Never truncates.
    #[serde(default = "binary_limit")]
    max_bytes: usize,
}
fn text_media_type() -> String {
    "text/plain; charset=utf-8".into()
}
fn document_kind() -> ArtifactKind {
    ArtifactKind::Document
}
fn list_limit() -> u32 {
    50
}
fn text_limit() -> usize {
    1_048_576
}
fn binary_limit() -> usize {
    8_388_608
}

fn reply<T: Serialize>(result: SokarResult<T>) -> Result<CallToolResult, rmcp::ErrorData> {
    Ok(
        match result.and_then(|value| serde_json::to_string(&value).map_err(Into::into)) {
            Ok(value) => CallToolResult::success(vec![Content::text(value)]),
            Err(error) => CallToolResult::error(vec![Content::text(
                json!({
                    "code": error.code(), "message": error.to_string(),
                    "help": Diagnostic::help(&error).map(|help| help.to_string()),
                })
                .to_string(),
            )]),
        },
    )
}

#[tool_router(vis = "pub(super)")]
impl SokarMcp {
    #[tool(
        description = "Create or get a collection by scope; returns the collection and its UUID."
    )]
    async fn sokar_collection(
        &self,
        Parameters(p): Parameters<CollectionParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                self.authorize("/v1/collections", Permission::Write)?;
                self.state
                    .create_collection(
                        &self.principal.id,
                        CreateCollection {
                            scope: p.scope,
                            store: None,
                            retention: RetentionPolicy::default(),
                            quota: Quota::default(),
                        },
                    )
                    .await
            }
            .await,
        )
    }
    #[tool(
        description = "Store UTF-8 text as a new immutable artifact version; returns artifact and stored/deduplicated outcome."
    )]
    async fn sokar_put_text(
        &self,
        Parameters(p): Parameters<PutTextParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                self.check_inline(p.text.len())?;
                self.put(
                    p.collection_id,
                    UploadOptions {
                        path: p.path.parse()?,
                        media_type: p.media_type.parse()?,
                        kind: p.kind,
                        labels: p.labels.unwrap_or_default(),
                        idempotency_key: None,
                    },
                    Bytes::from(p.text),
                )
                .await
            }
            .await,
        )
    }
    #[tool(
        description = "Store standard padded base64 as a new artifact version; decoded bytes must fit mcp.max_inline_bytes."
    )]
    async fn sokar_put_base64(
        &self,
        Parameters(p): Parameters<PutBase64Params>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(async {
            let limit = self.state.config.mcp.max_inline_bytes;
            // Bound allocation before decoding; exact length is checked after decoding.
            if p.base64.len() > limit.saturating_add(2).saturating_div(3).saturating_mul(4) {
                return Err(SokarError::InlineLimit { limit });
            }
            let bytes = STANDARD.decode(p.base64).map_err(|_| SokarError::Invalid(
                "base64 must use the standard alphabet and padding, without whitespace or a data URL prefix".into()))?;
            self.check_inline(bytes.len())?;
            self.put(p.collection_id, UploadOptions {
                path: p.path.parse()?, media_type: p.media_type.parse()?, kind: p.kind,
                labels: p.labels.unwrap_or_default(), idempotency_key: None,
            }, Bytes::from(bytes)).await
        }.await)
    }
    #[tool(
        description = "List live artifact versions with optional path prefix, category and cursor pagination."
    )]
    async fn sokar_list(
        &self,
        Parameters(p): Parameters<ListParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                let id = p.collection_id.parse()?;
                self.authorize(&format!("/v1/collections/{id}/artifacts"), Permission::List)?;
                self.state.index.collection(id).await?;
                self.state
                    .index
                    .list(
                        id,
                        &ListArtifacts {
                            prefix: p.prefix,
                            kind: p.kind,
                            limit: Some(p.limit),
                            cursor: p.cursor,
                        },
                    )
                    .await
            }
            .await,
        )
    }
    #[tool(
        description = "Get artifact version metadata and a content_url requiring the same bearer token."
    )]
    async fn sokar_get(
        &self,
        Parameters(p): Parameters<ArtifactParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                let id: ArtifactId = p.artifact_id.parse()?;
                self.authorize(&format!("/v1/artifacts/{id}"), Permission::Read)?;
                let artifact = self.state.index.artifact(id).await?;
                let path = format!("/v1/artifacts/{id}/content");
                let content_url = self
                    .state
                    .config
                    .server
                    .public_api_url
                    .as_ref()
                    .map_or_else(
                        || path.clone(),
                        |base| format!("{}{path}", base.as_str().trim_end_matches('/')),
                    );
                Ok(json!({ "artifact": artifact, "content_url": content_url }))
            }
            .await,
        )
    }
    #[tool(
        description = "Find the latest live artifact at a collection path. Missing paths return a tool error."
    )]
    async fn sokar_find(
        &self,
        Parameters(p): Parameters<FindParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                let id: CollectionId = p.collection_id.parse()?;
                let path: ArtifactPath = p.path.parse()?;
                self.authorize(
                    &format!("/v1/collections/{id}/artifacts/by-path/{path}"),
                    Permission::Read,
                )?;
                self.state.index.latest(id, &path).await
            }
            .await,
        )
    }
    #[tool(
        description = "Read an entire artifact as UTF-8 text. Oversized or non-UTF-8 content returns a tool error."
    )]
    async fn sokar_read_text(
        &self,
        Parameters(p): Parameters<ReadTextParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                let (_, bytes) = self.read(p.artifact_id, p.max_bytes).await?;
                String::from_utf8(bytes).map_err(|_| SokarError::NotUtf8)
            }
            .await,
        )
    }
    #[tool(
        description = "Read an entire artifact as standard base64 with media_type and decoded size. Oversized content returns a tool error."
    )]
    async fn sokar_read_base64(
        &self,
        Parameters(p): Parameters<ReadBase64Params>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(async {
            let (artifact, bytes) = self.read(p.artifact_id, p.max_bytes).await?;
            Ok(json!({ "base64": STANDARD.encode(&bytes), "media_type": artifact.blob.media_type, "size": bytes.len() }))
        }.await)
    }
    #[tool(
        description = "Tombstone an artifact version; subsequent lookups of this UUID fail. Returns tombstoned: true."
    )]
    async fn sokar_delete(
        &self,
        Parameters(p): Parameters<ArtifactParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        reply(
            async {
                let id: ArtifactId = p.artifact_id.parse()?;
                self.authorize(&format!("/v1/artifacts/{id}"), Permission::Delete)?;
                self.state.delete_artifact(id).await?;
                Ok(json!({ "tombstoned": true }))
            }
            .await,
        )
    }
}
impl SokarMcp {
    pub(super) fn new(
        state: std::sync::Arc<crate::Sokar>,
        principal: crate::authz::Principal,
    ) -> Self {
        Self {
            state,
            principal,
            tool_router: Self::tool_router(),
        }
    }
    fn authorize(&self, resource: &str, permission: Permission) -> SokarResult<()> {
        self.state
            .authorizer
            .authorize(&self.principal, resource, permission)
    }
    fn check_inline(&self, size: usize) -> SokarResult<()> {
        let limit = self.state.config.mcp.max_inline_bytes;
        if size > limit {
            Err(SokarError::InlineLimit { limit })
        } else {
            Ok(())
        }
    }
    async fn put(
        &self,
        collection_id: String,
        options: UploadOptions,
        bytes: Bytes,
    ) -> SokarResult<UploadResponse> {
        let id: CollectionId = collection_id.parse()?;
        self.authorize(
            &format!("/v1/collections/{id}/artifacts"),
            Permission::Write,
        )?;
        let collection = self.state.index.collection(id).await?;
        let store = self.state.stores.get(&collection.store)?;
        let staged = store
            .stage(
                futures::stream::once(async { Ok(bytes) }),
                self.state.config.limits.max_body_bytes,
            )
            .await?;
        self.state
            .finish_upload(&self.principal.id, &collection, options, staged)
            .await
    }
    async fn read(
        &self,
        artifact_id: String,
        max_bytes: usize,
    ) -> SokarResult<(Artifact, Vec<u8>)> {
        let id: ArtifactId = artifact_id.parse()?;
        self.authorize(&format!("/v1/artifacts/{id}/content"), Permission::Read)?;
        self.state
            .read_inline(id, max_bytes.min(self.state.config.mcp.max_inline_bytes))
            .await
    }
}
