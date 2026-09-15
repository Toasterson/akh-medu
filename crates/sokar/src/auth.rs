//! File-reloaded bearer pattern copied from akh-session-services/src/http.rs.
use crate::{SokarError, SokarResult, authz::Principal, config::PrincipalConfig};
use axum::http::HeaderMap;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
pub fn authenticate(principals: &[PrincipalConfig], headers: &HeaderMap) -> SokarResult<Principal> {
    if headers.get_all("authorization").iter().count() != 1 {
        return Err(SokarError::Authn);
    }
    let supplied = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .filter(|s| !s.is_empty() && s.len() <= 4096 && s.trim() == *s)
        .ok_or(SokarError::Authn)?;
    let digest = hex::encode(Sha256::digest(supplied.as_bytes()));
    let mut matches = vec![];
    // Fail closed if any credential source is unavailable: ambiguity cannot be excluded.
    let mut unavailable = false;
    for principal in principals {
        match principal.credential() {
            Ok((token, hashed)) => {
                let candidate = if hashed { &digest } else { supplied };
                if bool::from(candidate.as_bytes().ct_eq(token.as_bytes())) {
                    matches.push(principal);
                }
            }
            Err(_) => unavailable = true,
        }
    }
    if unavailable || matches.len() != 1 {
        return Err(SokarError::Authn);
    }
    Ok(Principal {
        id: matches[0].id.clone(),
    })
}
