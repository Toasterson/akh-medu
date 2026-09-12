//! No inferred success or automatic retries when the previous owner disappeared.
use crate::{
    config::Principal,
    store::{Transaction, key},
    types::*,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Audit {
    principal: String,
    target: String,
    resolution: Reconciliation,
    at: u64,
}
pub fn fingerprint(value: &(impl Serialize + ?Sized)) -> Result<String> {
    let value =
        serde_json::to_value(value).map_err(|_| Fault::invalid("cannot fingerprint request"))?;
    fn sorted(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let ordered: std::collections::BTreeMap<_, _> = map.into_iter().collect();
                serde_json::Value::Object(
                    ordered
                        .into_iter()
                        .map(|(key, value)| (key, sorted(value)))
                        .collect(),
                )
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(sorted).collect())
            }
            other => other,
        }
    }
    Ok(crate::digest(&serde_json::to_vec(&sorted(value)).map_err(
        |_| Fault::invalid("cannot fingerprint request"),
    )?))
}
pub fn validate(
    p: &Principal,
    request: &impl Serialize,
    resolution: &Reconciliation,
) -> Result<()> {
    crate::http::role(p, "admin")?;
    validate_key(&resolution.request_id)?;
    if !resolution.quiescent
        || !resolution.status.terminal()
        || resolution.evidence.is_empty()
        || resolution
            .evidence
            .iter()
            .any(|v| v.trim().is_empty() || v.len() > 8192)
        || resolution.evidence.len() > 32
    {
        return Err(Fault::invalid(
            "reconciliation requires terminal status, quiescent owners, and bounded evidence references",
        ));
    }
    if resolution.expected_request_sha256 != fingerprint(request)? {
        return Err(Fault::conflict(
            "reconciliation does not identify this exact request",
        ));
    }
    Ok(())
}
/// Check before changing a record; a matching retry returns the existing result.
pub fn already(
    tx: &Transaction<'_>,
    p: &Principal,
    target: &str,
    r: &Reconciliation,
) -> Result<bool> {
    if let Some(old) = tx.get::<Audit>(&key("reconciliation", &[&r.request_id]))? {
        if old.principal != p.id || old.target != target || old.resolution != *r {
            return Err(Fault::conflict("reconciliation request ID was reused"));
        }
        return Ok(true);
    }
    Ok(false)
}
pub fn record(
    tx: &mut Transaction<'_>,
    p: &Principal,
    target: &str,
    r: &Reconciliation,
) -> Result<()> {
    tx.put(
        &key("reconciliation", &[&r.request_id]),
        &Audit {
            principal: p.id.clone(),
            target: target.into(),
            resolution: r.clone(),
            at: now_ms(),
        },
    )
}
