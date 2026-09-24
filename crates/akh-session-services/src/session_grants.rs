//! Per-session delegation grants (Phase 53e, ADR 113 decision 9).
//!
//! The akh mints a grant when it delegates to a native runtime and hands the
//! secret to `POST /v1/sessions` (`NewSession.grant`). The secret is kept the
//! way MCP session capabilities are: in an owner-only private slot file, never
//! in the `SessionRecord` and never in the store. The store only holds a
//! secret-free receipt (the SHA-256 of the secret). The broker reads the slot
//! when it forwards a cognitive tool call, so the core can attribute the call
//! to the delegate and check its tool list.
use crate::{
    digest,
    session::Sessions,
    store::{Transaction, key},
    types::*,
};
use serde::{Deserialize, Serialize};

/// The secret-free record of a session's grant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrantReceipt {
    pub schema_version: u16,
    pub session_id: String,
    pub grant_sha256: String,
    pub at: u64,
}

#[derive(Serialize, Deserialize)]
struct PrivateGrant {
    receipt: GrantReceipt,
    grant: GrantToken,
}

const GRANT_DIRECTORY: &str = "session-delegation-grants";

fn unavailable() -> Fault {
    Fault::new(
        "grant_unavailable",
        "private session delegation grant is unavailable",
    )
}

fn receipt_key(session: &str) -> String {
    key("delegation_grant", &[session])
}

fn valid_grant(grant: &GrantToken) -> bool {
    let secret = grant.expose();
    (24..=512).contains(&secret.len()) && secret.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

/// The fingerprint stored for create-request deduplication in place of the
/// secret: replays with the same grant match, a different grant conflicts,
/// and the secret itself never reaches the store.
pub(crate) fn grant_fingerprint(grant: &GrantToken) -> GrantToken {
    GrantToken::new(format!("sha256:{}", digest(grant.expose().as_bytes())))
}

impl Sessions {
    fn grant_path(&self, session: &str) -> Result<std::path::PathBuf> {
        self.private_slot_path(GRANT_DIRECTORY, &receipt_key(session))
    }

    /// Validate a presented grant before anything is written.
    pub(crate) fn validate_grant(grant: &GrantToken) -> Result<()> {
        if valid_grant(grant) {
            Ok(())
        } else {
            Err(Fault::invalid(
                "delegation grant must be 24..512 printable ASCII characters",
            ))
        }
    }

    /// Bind `grant` to a newly created session: private slot first, then the
    /// receipt, inside the creating transaction.
    pub(crate) fn store_grant(
        &self,
        tx: &mut Transaction<'_>,
        session_id: &str,
        grant: &GrantToken,
    ) -> Result<()> {
        Self::validate_grant(grant)?;
        let receipt = GrantReceipt {
            schema_version: SCHEMA_VERSION,
            session_id: session_id.into(),
            grant_sha256: digest(grant.expose().as_bytes()),
            at: now_ms(),
        };
        let bytes = serde_json::to_vec(&PrivateGrant {
            receipt: receipt.clone(),
            grant: grant.clone(),
        })
        .map_err(|_| unavailable())?;
        Self::write_private_slot(&self.grant_path(session_id)?, &bytes)?;
        tx.put(&receipt_key(session_id), &receipt)
    }

    /// A conversation that resolves to an existing session keeps the grant
    /// it was created with. Presenting another one is a conflict, not a swap.
    pub(crate) fn check_existing_grant(
        &self,
        tx: &Transaction<'_>,
        session_id: &str,
        grant: Option<&GrantToken>,
    ) -> Result<()> {
        let existing: Option<GrantReceipt> = tx.get(&receipt_key(session_id))?;
        let presented = grant.map(|grant| digest(grant.expose().as_bytes()));
        if existing.map(|receipt| receipt.grant_sha256) == presented {
            Ok(())
        } else {
            Err(Fault::conflict(
                "conversation is bound to a different delegation grant; create the \
                 delegated session without a conversation_key",
            ))
        }
    }

    /// The session's grant, if it was created with one. Resolved before a
    /// cognitive tool call is marked sent, like MCP capabilities.
    pub(crate) fn delegation_grant(&self, session_id: &str) -> Result<Option<GrantToken>> {
        let Some(receipt) = self.store.get::<GrantReceipt>(&receipt_key(session_id))? else {
            return Ok(None);
        };
        let bytes =
            Self::read_private_slot(&self.grant_path(session_id)?)?.ok_or_else(unavailable)?;
        let private: PrivateGrant = serde_json::from_slice(&bytes).map_err(|_| unavailable())?;
        if private.receipt != receipt
            || receipt.session_id != session_id
            || receipt.grant_sha256 != digest(private.grant.expose().as_bytes())
        {
            return Err(unavailable());
        }
        Ok(Some(private.grant))
    }

    /// Remove the grant slot of an archived session. The receipt stays as
    /// the audit trail (it holds no secret).
    pub(crate) fn remove_archived_grant(
        &self,
        tx: &Transaction<'_>,
        session_id: &str,
    ) -> Result<()> {
        if tx.get::<GrantReceipt>(&receipt_key(session_id))?.is_none() {
            return Ok(());
        }
        Self::remove_private_slot(&self.grant_path(session_id)?)
    }
}
