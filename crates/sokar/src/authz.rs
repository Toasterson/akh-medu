//! 52b replaces this service-only authorization seam with resource permissions.
use crate::{SokarError, SokarResult};
#[derive(Debug, Clone)]
pub struct Principal {
    pub id: String,
}
#[derive(Debug, Clone, Copy)]
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
