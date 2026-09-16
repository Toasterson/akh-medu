//! Local SQLite records. Each callback runs in one immediate transaction;
//! unwinding or returning an error rolls back every write in that callback.
use crate::{Result, failure};
use rusqlite::{Connection, OptionalExtension};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    path::Path,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

pub(crate) struct SqliteInner(Mutex<Connection>);

impl SqliteInner {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(failure)?;
        }
        // Create privately before SQLite opens the file (including its journal).
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(path).map_err(failure)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(failure)?;
        }
        let db = Connection::open(path).map_err(failure)?;
        db.busy_timeout(Duration::from_secs(5)).map_err(failure)?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS records_v1 (key TEXT PRIMARY KEY COLLATE BINARY, value BLOB NOT NULL) WITHOUT ROWID;")
            .map_err(failure)?;
        Ok(Self(Mutex::new(db)))
    }
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, Connection>> {
        self.0.lock().map_err(|_| failure("SQLite mutex poisoned"))
    }
}

pub(crate) fn get<T: DeserializeOwned>(db: &Connection, key: &str) -> Result<Option<T>> {
    let bytes: Option<Vec<u8>> = db
        .query_row("SELECT value FROM records_v1 WHERE key=?1", [key], |r| {
            r.get(0)
        })
        .optional()
        .map_err(failure)?;
    bytes
        .map(|v| serde_json::from_slice(&v).map_err(failure))
        .transpose()
}
pub(crate) fn put<T: Serialize>(db: &Connection, key: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(failure)?;
    db.execute("INSERT INTO records_v1(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        rusqlite::params![key, bytes]).map_err(failure)?;
    Ok(())
}
pub(crate) fn delete(db: &Connection, key: &str) -> Result<()> {
    db.execute("DELETE FROM records_v1 WHERE key=?1", [key])
        .map_err(failure)?;
    Ok(())
}
pub(crate) fn scan<T: DeserializeOwned>(db: &Connection, prefix: &str) -> Result<Vec<T>> {
    let mut stmt = db
        .prepare("SELECT key,value FROM records_v1 WHERE key >= ?1 ORDER BY key COLLATE BINARY")
        .map_err(failure)?;
    let mut rows = stmt.query([prefix]).map_err(failure)?;
    let mut values = Vec::new();
    while let Some(row) = rows.next().map_err(failure)? {
        let key: String = row.get(0).map_err(failure)?;
        if !key.starts_with(prefix) {
            break;
        }
        let bytes: Vec<u8> = row.get(1).map_err(failure)?;
        values.push(serde_json::from_slice(&bytes).map_err(failure)?);
    }
    Ok(values)
}
