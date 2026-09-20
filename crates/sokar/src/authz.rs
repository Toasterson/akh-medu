//! 52b replaces this service-only authorization seam with resource permissions.
use crate::{SokarError, SokarResult};
#[derive(Debug, Clone)]
pub struct Principal {
    pub id: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,
    Write,
    List,
    Delete,
}
pub trait Authorizer: Send + Sync {
    fn authorize(
        &self,
        principal: &Principal,
        resource: &str,
        permission: Permission,
    ) -> SokarResult<()>;
}
pub struct ServiceOnly;
impl Authorizer for ServiceOnly {
    fn authorize(
        &self,
        principal: &Principal,
        _resource: &str,
        _permission: Permission,
    ) -> SokarResult<()> {
        if principal.id.is_empty() {
            Err(SokarError::Authz)
        } else {
            Ok(())
        }
    }
}

/// Canonical resource paths shared by REST handlers and MCP tools. In
/// particular by-path resources use the REST /artifacts/by-path route and
/// percent-encode each path segment just as an HTTP client does.
#[derive(Debug, Clone, Copy)]
pub enum Resource<'a> {
    Collections,
    ByScope,
    Collection(sokar_client::CollectionId),
    Artifacts(sokar_client::CollectionId),
    ByPath(sokar_client::CollectionId, &'a sokar_client::ArtifactPath),
    Artifact(sokar_client::ArtifactId),
    Content(sokar_client::ArtifactId),
}
impl std::fmt::Display for Resource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collections => f.write_str("/v1/collections"),
            Self::ByScope => f.write_str("/v1/collections/by-scope"),
            Self::Collection(id) => write!(f, "/v1/collections/{id}"),
            Self::Artifacts(id) => write!(f, "/v1/collections/{id}/artifacts"),
            Self::ByPath(id, path) => {
                let mut url = url::Url::parse("http://resource.invalid").expect("literal URL");
                url.set_path(&format!("/v1/collections/{id}/artifacts/by-path/"));
                url.path_segments_mut()
                    .expect("hierarchical URL")
                    .pop_if_empty()
                    .extend(path.as_str().split('/'));
                f.write_str(url.path())
            }
            Self::Artifact(id) => write!(f, "/v1/artifacts/{id}"),
            Self::Content(id) => write!(f, "/v1/artifacts/{id}/content"),
        }
    }
}
