//! Immutable per-session MCP capabilities. Only secret-free receipts enter redb.
use crate::{
    config::Principal,
    digest,
    session::Sessions,
    store::{Transaction, key},
    types::*,
};
use credentials::{McpCredentialReceipt, ProvisionMcpCredential};
use serde::{Deserialize, Serialize};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Serialize, Deserialize)]
struct PrivateCredential {
    receipt: McpCredentialReceipt,
    token: String,
}

fn unavailable() -> Fault {
    Fault::new(
        "credential_unavailable",
        "private session MCP capability is unavailable",
    )
}
fn receipt_key(session: &str, service: &str) -> String {
    key("mcp_credential", &[session, service])
}
fn valid_token(token: &str) -> bool {
    (24..=4096).contains(&token.len()) && token.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

impl Sessions {
    /// Private, owner-only slot file under `data_dir/<directory>`. Shared by
    /// MCP session capabilities and (Phase 53e) delegation grants: secrets
    /// live here, only secret-free receipts enter the store.
    pub(crate) fn private_slot_path(&self, directory: &str, slot_key: &str) -> Result<PathBuf> {
        let directory = self.config.data_dir.join(directory);
        match fs::DirBuilder::new().mode(0o700).create(&directory) {
            Ok(()) => File::open(&self.config.data_dir)
                .and_then(|f| f.sync_all())
                .map_err(|_| unavailable())?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(unavailable()),
        }
        let metadata = fs::symlink_metadata(&directory).map_err(|_| unavailable())?;
        if !metadata.is_dir()
            || metadata.mode() & 0o077 != 0
            || metadata.uid() != unsafe { libc::geteuid() }
        {
            return Err(unavailable());
        }
        Ok(directory.join(digest(slot_key.as_bytes())))
    }

    fn credential_path(&self, session: &str, service: &str) -> Result<PathBuf> {
        self.private_slot_path("session-mcp-credentials", &receipt_key(session, service))
    }

    /// Read a private slot file; `None` when it does not exist. Anything that
    /// is not an owner-only regular file of sane size is refused.
    pub(crate) fn read_private_slot(path: &std::path::Path) -> Result<Option<Vec<u8>>> {
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(unavailable()),
        };
        let metadata = file.metadata().map_err(|_| unavailable())?;
        if !metadata.is_file()
            || metadata.mode() & 0o077 != 0
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.len() > 16_384
        {
            return Err(unavailable());
        }
        let mut bytes = Vec::new();
        file.take(16_385)
            .read_to_end(&mut bytes)
            .map_err(|_| unavailable())?;
        Ok(Some(bytes))
    }

    fn read_credential(&self, session: &str, service: &str) -> Result<Option<PrivateCredential>> {
        let path = self.credential_path(session, service)?;
        let Some(bytes) = Self::read_private_slot(&path)? else {
            return Ok(None);
        };
        let value: PrivateCredential = serde_json::from_slice(&bytes).map_err(|_| unavailable())?;
        if !valid_token(&value.token)
            || value.receipt.token_sha256 != digest(value.token.as_bytes())
            || value.receipt.session_id != session
            || value.receipt.service != service
        {
            return Err(unavailable());
        }
        Ok(Some(value))
    }

    /// Commit complete bytes before publishing the file. A crash before the
    /// store acknowledgement leaves this first intent available for exact replay.
    pub(crate) fn write_private_slot(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
        let temporary = path.with_extension(new_id());
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&temporary)
                .map_err(|_| unavailable())?;
            file.write_all(bytes)
                .and_then(|_| file.sync_all())
                .map_err(|_| unavailable())?;
            fs::hard_link(&temporary, path).map_err(|_| unavailable())?;
            File::open(path.parent().expect("private slot parent"))
                .and_then(|f| f.sync_all())
                .map_err(|_| unavailable())
        })();
        let _ = fs::remove_file(temporary);
        result
    }

    /// Remove a private slot file and any temporary link a crash left behind.
    pub(crate) fn remove_private_slot(path: &std::path::Path) -> Result<()> {
        let parent = path.parent().expect("private slot parent");
        // A crash before publishing a complete file may leave its private
        // temporary link. Only this slot's hashed prefix is ours.
        let prefix = format!(
            "{}.",
            path.file_name()
                .expect("private slot filename")
                .to_string_lossy()
        );
        for entry in fs::read_dir(parent).map_err(|_| unavailable())? {
            let entry = entry.map_err(|_| unavailable())?;
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                fs::remove_file(entry.path()).map_err(|_| unavailable())?;
            }
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(unavailable()),
        }
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|_| unavailable())
    }

    fn write_credential(&self, value: &PrivateCredential) -> Result<()> {
        let path = self.credential_path(&value.receipt.session_id, &value.receipt.service)?;
        let bytes = serde_json::to_vec(value).map_err(|_| unavailable())?;
        Self::write_private_slot(&path, &bytes)
    }

    fn endpoint_fingerprint(&self, service: &str) -> Result<String> {
        if !self
            .config
            .session_mcp_endpoints
            .iter()
            .any(|s| s == service)
        {
            return Err(Fault::conflict(
                "MCP destination is not configured for session capabilities",
            ));
        }
        let endpoint = self.config.endpoint(service)?;
        endpoint.validate()?;
        if endpoint.url.contains('{') {
            return Err(Fault::conflict(
                "session capability destination must be fixed",
            ));
        }
        Ok(digest(
            &serde_json::to_vec(endpoint).map_err(|_| unavailable())?,
        ))
    }

    pub(crate) fn required_mcp_credentials(
        &self,
        tools: &[ToolSpec],
    ) -> Result<BTreeMap<String, String>> {
        let mut required = BTreeMap::new();
        for tool in tools {
            if let ToolDestination::Mcp { service, .. } = &tool.destination {
                if self.config.session_mcp_endpoints.contains(service) {
                    required.insert(service.clone(), self.endpoint_fingerprint(service)?);
                }
            }
        }
        Ok(required)
    }

    pub fn provision_mcp_credential(
        &self,
        principal: &Principal,
        session_id: &str,
        service: &str,
        request: &ProvisionMcpCredential,
    ) -> Result<McpCredentialReceipt> {
        crate::http::role(principal, "session_credentials")?;
        validate_key(&request.request_id)?;
        if !valid_token(&request.token) {
            return Err(Fault::invalid("invalid session capability"));
        }
        self.store.transaction(|tx| {
            let mut session = crate::session::load_record(tx, session_id)?
                .ok_or_else(|| Fault::missing("session not found"))?;
            if session.principal != principal.id && !principal.has_role("admin") {
                return Err(Fault::new(
                    "forbidden",
                    "session belongs to another principal",
                ));
            }
            let endpoint_sha256 =
                session
                    .required_mcp_credentials
                    .get(service)
                    .ok_or_else(|| {
                        Fault::conflict("session has no capability slot for this MCP destination")
                    })?;
            let k = receipt_key(session_id, service);
            let committed: Option<McpCredentialReceipt> = tx.get(&k)?;
            let expected = McpCredentialReceipt {
                schema_version: SCHEMA_VERSION,
                session_id: session_id.into(),
                service: service.into(),
                endpoint_sha256: endpoint_sha256.clone(),
                request_id: request.request_id.clone(),
                token_sha256: digest(request.token.as_bytes()),
                principal: principal.id.clone(),
                at: 0,
            };
            let matches = |old: &McpCredentialReceipt| {
                let mut old = old.clone();
                old.at = 0;
                old == expected
            };
            if committed.as_ref().is_some_and(|old| !matches(old)) {
                return Err(Fault::conflict(
                    "session capability already provisioned with another intent",
                ));
            }
            if session.closed_at.is_some() || session.retirement.is_some() {
                return committed.ok_or_else(|| Fault::conflict("session is retiring or archived"));
            }
            if self.endpoint_fingerprint(service)? != *endpoint_sha256 {
                return Err(Fault::conflict(
                    "session MCP destination configuration changed",
                ));
            }
            let receipt = if let Some(private) = self.read_credential(session_id, service)? {
                if !matches(&private.receipt)
                    || committed
                        .as_ref()
                        .is_some_and(|old| old != &private.receipt)
                {
                    return Err(Fault::conflict(
                        "private capability differs from original intent",
                    ));
                }
                private.receipt
            } else {
                let receipt = committed.clone().unwrap_or(McpCredentialReceipt {
                    at: now_ms(),
                    ..expected
                });
                self.write_credential(&PrivateCredential {
                    receipt: receipt.clone(),
                    token: request.token.clone(),
                })?;
                receipt
            };
            if committed.is_none() {
                tx.put(&k, &receipt)?;
                crate::session::append_event(
                    tx,
                    &mut session,
                    None,
                    "mcp_credential_provisioned",
                    serde_json::to_value(&receipt).map_err(|_| unavailable())?,
                )?;
            }
            Ok(receipt)
        })
    }

    pub(crate) fn check_mcp_credentials(
        &self,
        tx: &Transaction<'_>,
        session: &SessionRecord,
    ) -> Result<()> {
        if self.required_mcp_credentials(&session.tools)? != session.required_mcp_credentials {
            return Err(Fault::conflict("session MCP credential policy changed"));
        }
        for service in session.required_mcp_credentials.keys() {
            self.scoped_token(tx, session, service)?;
        }
        Ok(())
    }

    fn scoped_token(
        &self,
        tx: &Transaction<'_>,
        session: &SessionRecord,
        service: &str,
    ) -> Result<String> {
        let expected = session
            .required_mcp_credentials
            .get(service)
            .ok_or_else(|| Fault::conflict("session has no pinned MCP capability slot"))?;
        if self.endpoint_fingerprint(service)? != *expected {
            return Err(Fault::conflict(
                "session MCP destination configuration changed",
            ));
        }
        let receipt: McpCredentialReceipt = tx
            .get(&receipt_key(&session.id, service))?
            .ok_or_else(unavailable)?;
        let private = self
            .read_credential(&session.id, service)?
            .ok_or_else(unavailable)?;
        if receipt != private.receipt || receipt.endpoint_sha256 != *expected {
            return Err(unavailable());
        }
        Ok(private.token)
    }

    /// Resolve locally before marking a tool request sent. No static fallback
    /// after a session requires a capability, even across config changes.
    pub(crate) fn mcp_token(&self, session: &SessionRecord, service: &str) -> Result<String> {
        self.store.transaction(|tx| {
            let current = crate::session::load_record(tx, &session.id)?.ok_or_else(unavailable)?;
            if current.closed_at.is_some() {
                return Err(Fault::conflict("session is archived"));
            }
            if current.required_mcp_credentials.contains_key(service)
                || self
                    .config
                    .session_mcp_endpoints
                    .iter()
                    .any(|s| s == service)
            {
                self.scoped_token(tx, &current, service)
            } else {
                self.config.endpoint(service)?.token()
            }
        })
    }

    pub fn remove_archived_mcp_credentials(&self, session_id: &str) -> Result<()> {
        // Closure is irreversible. A durable cleanup receipt avoids reopening a
        // write transaction and rereading a large transcript on every sweep.
        if self
            .store
            .get::<bool>(&key("mcp_credentials_removed", &[session_id]))?
            == Some(true)
        {
            return Ok(());
        }
        self.store.transaction(|tx| {
            let session = crate::session::load_record(tx, session_id)?.ok_or_else(unavailable)?;
            if session.closed_at.is_none() {
                return Err(Fault::conflict("session is still open"));
            }
            let removed = key("mcp_credentials_removed", &[session_id]);
            if tx.get::<bool>(&removed)? == Some(true) {
                return Ok(());
            }
            for service in session.required_mcp_credentials.keys() {
                Self::remove_private_slot(&self.credential_path(session_id, service)?)?;
            }
            // Phase 53e: the delegation grant slot closes with the session.
            self.remove_archived_grant(tx, session_id)?;
            tx.put(&removed, &true)
        })
    }
}
